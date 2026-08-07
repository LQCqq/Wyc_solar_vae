import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import json as _json
from pathlib import Path as _Path

_TABLE_DIR = _Path(__file__).parent.parent / "pl_data"   
_RADII_TENSOR = None
_OXI_TENSOR   = None

def _get_radii_tensor(device):
    global _RADII_TENSOR
    if _RADII_TENSOR is None:
        table_path = _TABLE_DIR / "atomic_radii.json"
        with open(table_path) as f:
            raw = _json.load(f)
        t = torch.zeros(101)
        for z_str, info in raw.items():
            t[int(z_str)] = float(info["radius_A"])
        _RADII_TENSOR = t
    return _RADII_TENSOR.to(device)


def _get_oxi_tensor(device):
    global _OXI_TENSOR
    if _OXI_TENSOR is None:
        table_path = _TABLE_DIR / "oxidation_states.json"
        with open(table_path) as f:
            raw = _json.load(f)
        t = torch.zeros(101)
        for z_str, info in raw.items():
            t[int(z_str)] = float(info["common_oxi"])
        _OXI_TENSOR = t
    return _OXI_TENSOR.to(device)



# Overlap Penalty 对称缓存
_MAX_MULT = 8
_OVERLAP_OPS_CACHE = {}

# Wyckoff letter 表获取
_WYCKOFF_LETTERS = list(
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
)


def _get_ops_cpu(spg_num, letter_idx):
    key = (int(spg_num), int(letter_idx))
    if key in _OVERLAP_OPS_CACHE:
        return _OVERLAP_OPS_CACHE[key]
    try:
        from pyxtal.symmetry import Group
        letter_i = int(letter_idx)
        if letter_i < 0 or letter_i >= len(_WYCKOFF_LETTERS):
            raise ValueError(f"letter_idx {letter_i} 越界")
        letter = _WYCKOFF_LETTERS[letter_i]
        g = Group(int(spg_num))
        # g[letter] 在部分 pyxtal 版本里不支持字符串下标，改用遍历更稳健
        wp = None
        for w in g.Wyckoff_positions:
            if w.letter == letter:
                wp = w
                break
        if wp is None:
            raise KeyError(f"letter {letter} not in spg {spg_num}")
        ops = wp.ops
        m = min(len(ops), _MAX_MULT)
        R    = torch.zeros(_MAX_MULT, 3, 3)
        t    = torch.zeros(_MAX_MULT, 3)
        mask = torch.zeros(_MAX_MULT)
        for k in range(m):
            aff = ops[k].affine_matrix
            R[k]    = torch.tensor(aff[:3, :3], dtype=torch.float32)
            t[k]    = torch.tensor(aff[:3, 3],  dtype=torch.float32)
            mask[k] = 1.0
        R0 = torch.tensor(ops[0].affine_matrix[:3, :3], dtype=torch.float32)
        t0 = torch.tensor(ops[0].affine_matrix[:3, 3],  dtype=torch.float32)
    except Exception:
        R    = torch.zeros(_MAX_MULT, 3, 3)
        t    = torch.zeros(_MAX_MULT, 3)
        R0   = torch.eye(3)
        t0   = torch.zeros(3)
        mask = torch.zeros(_MAX_MULT)
    result = (R, t, R0, t0, mask)
    _OVERLAP_OPS_CACHE[key] = result
    return result


def _lattice_matrix_cpu(lattice_pred_row):
    lp = lattice_pred_row.cpu().float()
    if torch.isnan(lp).any() or torch.isinf(lp).any():
        return None  # sm_90 垃圾Nan
    a, b, c = lp[0].item(), lp[1].item(), lp[2].item()
    al, be, ga = lp[3].item(), lp[4].item(), lp[5].item()
    # 若看起来像弧度则转度
    if max(abs(al), abs(be), abs(ga)) < 2 * np.pi + 0.5:
        al = np.degrees(al)
        be = np.degrees(be)
        ga = np.degrees(ga)
    al, be, ga = np.clip([al, be, ga], 10, 170)
    a, b, c = max(a, 1.0), max(b, 1.0), max(c, 1.0)
    ca, cb, cg = np.cos(np.radians([al, be, ga]))
    sg = np.sin(np.radians(ga))
    M = torch.zeros(3, 3)
    M[0, 0] = a
    M[1, 0] = b * cg
    M[1, 1] = b * sg
    M[2, 0] = c * cb
    cy = (ca - cb * cg) / (sg + 1e-8)
    M[2, 1] = c * cy
    tmp = max(1.0 - cb ** 2 - cy ** 2, 1e-8)
    M[2, 2] = c * np.sqrt(tmp)
    return M   # (3, 3)



# Overlap Penalty
def _overlap_penalty(preds, targets, site_mask):
    
    orig_device = preds['free_params'].device
    # free_params 不 detach：轨道展开 pos=R@fp+t、% 1.0、@lat_mat、relu 全可导，
    # 梯度可回传到分数坐标，模型才学得到"别把原子放在对称元素上"。
    # lattice_pred 仍 detach：_lattice_matrix_cpu 全是 .item()/numpy，接不回梯度，
    # 晶胞过小导致的重叠由生成端 _fix_lattice 的动态长度下限兜底。
    free_params  = preds['free_params'].cpu().float()            # (B, S, 3) 带梯度
    # lattice_pred 是归一化量（loss 里比的是 target/lat_scale），必须先还原成
    # 真实的 Å 和度，否则 _lattice_matrix_cpu 会把 ~0.5 的边长 clamp 到 1.0 Å、
    # 把 ~1.0 的角度当成弧度 → 得到 1×1×1Å 的假晶胞 → 所有原子都被判为重叠
    _LAT_SCALE = torch.tensor([10., 10., 10., 90., 90., 90.])
    lattice_pred = preds['lattice_pred'].cpu().detach().float() * _LAT_SCALE  # (B, 6) 常数
    spg_target   = targets['spg_target'].cpu()                   # (B,)
    letter_target = targets['letter_target'].cpu()               # (B, S)
    elem_target  = targets['elem_target'].cpu()                  # (B, S)
    mask         = site_mask.cpu().float()                       # (B, S)

    B, S, _ = free_params.shape
    SM = S * _MAX_MULT

    total_penalty = torch.tensor(0.0, device=orig_device)
    n_valid = 0
    _eye_cache = {}  # N -> eye(N)，避免每个结构重复创建

    for b in range(B):
        spg_num = int(spg_target[b].item()) + 1   # 0-indexed → 1-indexed
        lat_mat = _lattice_matrix_cpu(lattice_pred[b])  # (3,3) CPU 或 None
        if lat_mat is None:
            continue  # sm_90 产出 NaN lattice_pred

        # 对每个有效位点展开轨道，收集分数坐标 + 元素 Z
        all_frac = []   # list of (M, 3) tensors
        all_Z    = []   # list of int (元素序号)
        all_site_idx = []  

        for s in range(S):
            if mask[b, s] < 0.5:
                continue
            letter_idx = int(letter_target[b, s].item())
            # elem_target 存的是类别(0-indexed)，radii_table 按原子序数 Z 索引，
            # Z = class + 1（与 decoder 里 argmax()+1 的约定一致），必须 +1
            elem_Z     = int(elem_target[b, s].item()) + 1
            fp         = free_params[b, s]   # (3,) CPU, no grad

            R, t, R0, t0, op_mask = _get_ops_cpu(spg_num, letter_idx)

            # ops[0] 投影（CPU）
            rep = R0 @ fp + t0
            rep = rep % 1.0

            # 轨道展开
            pos = torch.einsum('kij,j->ki', R, rep) + t  # (M, 3)
            pos = pos % 1.0

            m = int(op_mask.sum().item())
            if m == 0:
                continue

            all_frac.append(pos[:m])
            all_Z.extend([elem_Z] * m)
            all_site_idx.extend([s] * m)

        if len(all_frac) == 0:
            continue

        # 转笛卡尔coordinate
        frac_all = torch.cat(all_frac, dim=0)   # (N, 3)
        cart_all = frac_all @ lat_mat.T          # (N, 3)
        N = cart_all.shape[0]

        if N < 2:
            continue

        # 最小镜像约定：先在分数空间取周期最近邻，再转笛卡尔，
        # 否则分数 0.02 与 0.98（隔晶胞边界，实际仅差 0.04）会被算成 ~0.96 的假距离，
        # 漏判跨边界的重叠（<0.5Å 那批的主要来源）
        fdiff = frac_all.unsqueeze(1) - frac_all.unsqueeze(0)  # (N, N, 3)
        fdiff = fdiff - fdiff.round()                          # 映射到 [-0.5, 0.5)
        diff = fdiff @ lat_mat.T                               # (N, N, 3) 笛卡尔
        dist = (diff ** 2).sum(-1).clamp(min=1e-12).sqrt()    # (N, N)

        # distance from lemat
        radii_table = _get_radii_tensor('cpu')
        Z_tensor = torch.tensor(all_Z, dtype=torch.long)
        ri = radii_table[Z_tensor].unsqueeze(1).expand(N, N)  # (N, N)
        rj = radii_table[Z_tensor].unsqueeze(0).expand(N, N)  # (N, N)
        threshold = (0.7 + ri + rj) * 0.5                     # (N, N)

        # 非对角线 mask
        if N not in _eye_cache:
            _eye_cache[N] = torch.eye(N)
        eye = _eye_cache[N]
        pair_mask = 1.0 - eye   # (N, N)

        # hinge penalty：超过阈值的对贡献 0，重叠的对贡献正值
        violation = F.relu(threshold - dist) * pair_mask  # (N, N)

        # 除以「违反的对数」而非 N²：原先除 N² 会把 1 对真实重叠
        # 摊薄进几百对正常原子里（N=20 时信号缩水 ~400 倍），尾部重叠压不下去。
        # 现在只对真正违反的对取均值，重叠越多惩罚越大，信号不被稀释。
        n_viol = (violation > 0).float().sum().clamp(min=1)
        batch_penalty = violation.sum() / n_viol

        # batch_penalty 现在自带来自 free_params 的真实梯度，
        # 原先 "+ free_grad.sum()*0" 的桥接已无必要（且它把梯度也一并归零了）
        total_penalty = total_penalty + batch_penalty.to(orig_device)
        n_valid += 1

    if n_valid == 0:
        # nan_to_num 先清除 NaN，再 *0：nan*0=nan
        return torch.nan_to_num(preds['free_params'], nan=0.0).sum() * 0

    result = total_penalty / n_valid
    # 兜底：NaN 
    if torch.isnan(result) or torch.isinf(result):
        return torch.nan_to_num(preds['free_params'], nan=0.0).sum() * 0
    return result


# Charge Neutrality Penalty
def _charge_neutrality_penalty(preds, targets, site_mask):

    elem_logits  = preds['elem_logits']          # (B, S, 100)
    multiplicities = targets['multiplicities']    # (B, S) float
    mask = site_mask.float()                      # (B, S)
    device = elem_logits.device

    oxi_table = _get_oxi_tensor(device)          # (101,) 按原子序数 Z 索引
    # elem_logits 的类别 k 对应 Z = k+1（与 decoder 的 argmax()+1 一致），
    # 所以要取 [1:101] 而不是 [:100]，否则每个元素拿到的是 Z-1 那个元素的氧化态
    oxi_per_elem = oxi_table[1:101]              # (100,) index k → oxi(Z=k+1)

    # 软概率
    probs = F.softmax(elem_logits, dim=-1)        # (B, S, 100)

    expected_oxi = (probs * oxi_per_elem.view(1, 1, 100)).sum(-1)  # (B, S)

    total_charge = (expected_oxi * multiplicities * mask).sum(-1)  # (B,)

    penalty = (total_charge ** 2).mean()

    return penalty


class WyckoffReconLoss(nn.Module):
    def __init__(self,
                 w_spg=1.0, w_lattice=1.0, w_elem=1.0,
                 w_letter=1.0, w_free=1.0, w_nsites=0.5,
                 mask_weight=3.0,
                 w_overlap=0.0, overlap_threshold=None,  # threshold 已弃用，保留兼容
                 w_charge=0.0):
        super().__init__()
        self.w_spg     = w_spg
        self.w_lattice = w_lattice
        self.w_elem    = w_elem
        self.w_letter  = w_letter
        self.w_free    = w_free
        self.w_nsites  = w_nsites
        self.mask_weight = mask_weight
        self.w_overlap = w_overlap
        self.w_charge  = w_charge

    def forward(self, preds, targets, site_mask, masked_sites=None):

        # 重建 loss 
        loss_spg = F.cross_entropy(preds['spg_logits'], targets['spg_target'])
        loss_spg = torch.nan_to_num(loss_spg.abs(), nan=0.0)

        lat_scale = torch.tensor([10., 10., 10., 90., 90., 90.],
                                  device=preds['lattice_pred'].device)
        loss_lattice = F.mse_loss(
            preds['lattice_pred'],
            targets['lattice_target'] / lat_scale,
        )
        # sm_90 CUDA 产出垃圾值时 MSE 可能为负或 NaN：abs 修正符号，nan_to_num 兜底
        loss_lattice = loss_lattice.abs()
        loss_lattice = torch.nan_to_num(loss_lattice, nan=0.0, posinf=0.0, neginf=0.0)

        loss_nsites = F.cross_entropy(
            preds['num_sites_logits'], targets['num_sites_target']
        )
        loss_nsites = torch.nan_to_num(loss_nsites.abs(), nan=0.0)

        mask = site_mask.float()

        # elem loss（diffusion 加权）
        elem_loss = F.cross_entropy(
            preds['elem_logits'].view(-1, preds['elem_logits'].shape[-1]),
            targets['elem_target'].view(-1),
            reduction='none'
        ).view_as(site_mask)
        if masked_sites is not None:
            elem_weight = mask.clone()
            elem_weight[masked_sites & site_mask] = self.mask_weight
        else:
            elem_weight = mask
        loss_elem = (elem_loss * elem_weight).sum() / elem_weight.sum().clamp(min=1)
        loss_elem = torch.nan_to_num(loss_elem.abs(), nan=0.0)

        # letter loss（diffusion 加权）
        letter_loss = F.cross_entropy(
            preds['letter_logits'].view(-1, preds['letter_logits'].shape[-1]),
            targets['letter_target'].view(-1),
            reduction='none'
        ).view_as(site_mask)
        if masked_sites is not None:
            letter_weight = mask.clone()
            letter_weight[masked_sites & site_mask] = self.mask_weight
        else:
            letter_weight = mask
        loss_letter = (letter_loss * letter_weight).sum() / letter_weight.sum().clamp(min=1)
        loss_letter = torch.nan_to_num(loss_letter.abs(), nan=0.0)

        # free loss
        free_loss = F.mse_loss(
            preds['free_params'], targets['free_target'], reduction='none'
        ).mean(-1)
        loss_free = (free_loss * mask).sum() / mask.sum().clamp(min=1)
        loss_free = torch.nan_to_num(loss_free.abs(), nan=0.0)

        # Overlap Penalty 
        if self.w_overlap > 0:
            try:
                loss_overlap = _overlap_penalty(preds, targets, site_mask)
            except Exception as e:
                loss_overlap = preds['free_params'].sum() * 0
        else:
            loss_overlap = torch.tensor(0.0, device=preds['elem_logits'].device)

        # Charge Neutrality Penalty 
        if self.w_charge > 0:
            try:
                loss_charge = _charge_neutrality_penalty(preds, targets, site_mask)
                if torch.isnan(loss_charge) or torch.isinf(loss_charge):
                    loss_charge = preds['elem_logits'].sum() * 0
            except Exception as e:
                loss_charge = torch.tensor(0.0, device=preds['elem_logits'].device)
        else:
            loss_charge = torch.tensor(0.0, device=preds['elem_logits'].device)

        # ── 真值基线探针（仅当环境变量 WYCKOFF_GT_PROBE=1 时启用，每次训练打印一次）──
        # 用真值的元素/自由参数跑同样的惩罚，得到"地板值"
        import os as _os
        if _os.environ.get('WYCKOFF_GT_PROBE') == '1' and not getattr(self, '_gt_probe_done', False):
            with torch.no_grad():
                _V = preds['elem_logits'].shape[-1]
                _onehot = F.one_hot(targets['elem_target'].clamp(0, _V - 1), num_classes=_V).float()
                _gt_charge = _charge_neutrality_penalty(
                    {'elem_logits': (_onehot - 0.5) * 50.0}, targets, site_mask).item()
                _lat_norm = targets['lattice_target'] / lat_scale   # 还原成与 preds 同样的归一化尺度
                _gt_overlap = _overlap_penalty(
                    {'free_params': targets['free_target'], 'lattice_pred': _lat_norm},
                    targets, site_mask).item()
            print(f"[GT探针] 真值基线   charge={_gt_charge:12.4f}   overlap={_gt_overlap:.6f}", flush=True)
            print(f"[GT探针] 模型输出   charge={loss_charge.item():12.4f}   overlap={loss_overlap.item():.6f}", flush=True)
            self._gt_probe_done = True

        total = (
            self.w_spg     * loss_spg     +
            self.w_lattice * loss_lattice +
            self.w_nsites  * loss_nsites  +
            self.w_elem    * loss_elem    +
            self.w_letter  * loss_letter  +
            self.w_free    * loss_free    +
            self.w_overlap * loss_overlap +
            self.w_charge  * loss_charge
        )

        return total, {
            'loss_spg':     loss_spg.item(),
            'loss_lattice': loss_lattice.item(),
            'loss_nsites':  loss_nsites.item(),
            'loss_elem':    loss_elem.item(),
            'loss_letter':  loss_letter.item(),
            'loss_free':    loss_free.item(),
            'loss_overlap': loss_overlap.item(),
            'loss_charge':  loss_charge.item(),
        }