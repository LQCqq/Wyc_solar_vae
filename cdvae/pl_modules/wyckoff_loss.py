# models/wyckoff_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# ─── 轨道感知 overlap penalty：按需用pyxtal现算对称操作，按(spg,letter)缓存 ───
# 不使用预生成的.pt查找表：在本机(H200/sm_90)上，import时torch.load一个
# ~30MB的CPU张量表，会在与CUDA上下文初始化时机相关的情况下污染后续GPU
# kernel(如encoder的repeat_interleave)，报错与真实原因完全无关。改为按
# (spg,letter)增量缓存，每个key仅~7KB，缓存平缓增长，不会有突发大分配。
WYCKOFF_LETTERS = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
_MAX_MULT = 8  # 48->8: SM=S*MAX_MULT从576降到96，(B,SM,SM)张量缩小36倍，
               # 缓解CPU内存OOM(46GB)及相关数值异常。多数mp_20 Wyckoff
               # 轨道multiplicity<=8；个别高对称位点会被截断(惩罚略保守)。
_OVERLAP_OPS_CACHE = {}
_EYE_CACHE: dict = {}   # torch.eye(SM) 缓存：SM固定时只建一次
_ZERO_OPS = (
    torch.zeros(_MAX_MULT, 3, 3), torch.zeros(_MAX_MULT, 3),
    torch.eye(3), torch.zeros(3), torch.zeros(_MAX_MULT),
)


def _get_ops_cpu(spg_num, letter_idx):
    """按(spg_num 1-230, letter_idx 0-51)取对称操作(R,t,R0,t0,mask)，缓存。
    取不到则返回全0占位(mask全0)，该site不贡献任何原子，安全。"""
    key = (int(spg_num), int(letter_idx))
    cached = _OVERLAP_OPS_CACHE.get(key)
    if cached is not None:
        return cached
    result = _ZERO_OPS
    try:
        from pyxtal.symmetry import Group
        if 1 <= spg_num <= 230 and 0 <= letter_idx < len(WYCKOFF_LETTERS):
            g = Group(spg_num)
            letter = WYCKOFF_LETTERS[letter_idx]
            wp = None
            for w in g.Wyckoff_positions:
                if w.letter == letter:
                    wp = w
                    break
            if wp is not None:
                ops = wp.ops
                m = min(len(ops), _MAX_MULT)

                # ─── [NaN防护 / 修复1] ───────────────────────────────────
                # pyxtal对某些(spg,letter)组合可能返回含NaN/Inf的
                # affine_matrix，且不抛异常——外层except Exception抓不到
                # 这种"数值错误"。一旦这种坏值未被拦截，会被原样写入
                # R/t/R0/t0并永久缓存进_OVERLAP_OPS_CACHE，之后每次命中
                # 同一(spg,letter)组合都会复现NaN。
                #
                # 这里在写入前逐个检查 np.isfinite：
                #   - ops[0]非finite -> 整个轨道判定为_ZERO_OPS(该site不
                #     贡献任何原子，安全)。
                #   - ops[k] (k>0)非finite -> 仅跳过这一个对称操作副本，
                #     mask[k]保持0，不参与后续计算。
                am0 = np.asarray(ops[0].affine_matrix, dtype=np.float64)
                if not np.isfinite(am0).all():
                    result = _ZERO_OPS
                    _OVERLAP_OPS_CACHE[key] = result
                    return result

                R = np.zeros((_MAX_MULT, 3, 3), dtype=np.float32)
                t = np.zeros((_MAX_MULT, 3), dtype=np.float32)
                mask = np.zeros((_MAX_MULT,), dtype=np.float32)
                for k in range(m):
                    am = np.asarray(ops[k].affine_matrix, dtype=np.float64)
                    if not np.isfinite(am).all():
                        continue  # 该副本含NaN/Inf，跳过，mask[k]保持0
                    R[k] = am[:3, :3]
                    t[k] = am[:3, 3]
                    mask[k] = 1.0
                # ─────────────────────────────────────────────────────────

                R0 = am0[:3, :3].astype(np.float32)
                t0 = am0[:3, 3].astype(np.float32)
                result = (
                    torch.from_numpy(R), torch.from_numpy(t),
                    torch.from_numpy(R0), torch.from_numpy(t0),
                    torch.from_numpy(mask),
                )
    except Exception:
        result = _ZERO_OPS
    _OVERLAP_OPS_CACHE[key] = result
    return result


def _lattice_matrix_from_pred(lattice_pred):
    """由 lattice_pred 构造晶格矩阵 (B,3,3)。长度=Å，角度=弧度（与 _fix_lattice 解释一致）。
    建矩阵前做 abs+clip 兜底，防退化晶格导致 NaN。"""
    lengths = lattice_pred[:, :3].abs().clamp(2.0, 30.0)
    angles = lattice_pred[:, 3:].clamp(0.5, 2.7)
    a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]
    al, be, ga = angles[:, 0], angles[:, 1], angles[:, 2]
    cos_al, cos_be, cos_ga = al.cos(), be.cos(), ga.cos()
    sin_ga = ga.sin().clamp(min=1e-3)
    M = lattice_pred.new_zeros(lattice_pred.shape[0], 3, 3)
    M[:, 0, 0] = a
    M[:, 1, 0] = b * cos_ga
    M[:, 1, 1] = b * sin_ga
    M[:, 2, 0] = c * cos_be
    M[:, 2, 1] = c * (cos_al - cos_be * cos_ga) / sin_ga
    tmp = 1.0 - cos_be ** 2 - ((cos_al - cos_be * cos_ga) / sin_ga) ** 2
    M[:, 2, 2] = c * tmp.clamp(min=1e-4).sqrt()
    return M


def _overlap_penalty(free_params, lattice_pred, spg_target, letter_target,
                      site_mask, threshold=0.8):
    """轨道感知原子间距重叠惩罚，固定阈值0.8Å(不使用LeMat的半径缩放公式)。

    1. ops[0]投影:     rep   = R0 @ (free_params%1) + t0
    2. 轨道展开(封顶_MAX_MULT): pos_k = R_k @ rep + t_k
    3. 直接笛卡尔距离(不做min-image)
    4. hinge: relu(0.8 - dist)，按每结构有效对数归一化后batch平均

    对称操作按(spg,letter)用pyxtal现算+缓存(无.pt文件、无大内存分配)。

    [GPU实验已回退] 曾尝试让free_params/lattice_pred/site_mask全程留在GPU、
    仅小张量R/t/R0/t0/op_mask .to(device)——结果训练在batch0前向阶段(连
    [DEBUG step0]都未打印)即CPU OOM(46GB)，比之前(CPU版,MAX_MULT=48,无
    detach)能跑完整个epoch0更早崩，怀疑GPU路径下某处(可能是.to(device)
    与CUDA分配器交互、或autograd在GPU上为这些(B,SM,SM)中间张量保留的
    元数据)反而放大了CPU侧内存占用。改回全CPU计算+最后.to(orig_device)，
    与之前唯一已知能跑过step0-4的版本一致；本次仅保留MAX_MULT=8+eps-fix。

    注意: spg_target是0-indexed(spacegroup_num-1，见encode_wyckoff_tensors)，
    取pyxtal用的真实空间群号需 +1。
    """
    orig_device = free_params.device
    free_params = free_params.cpu()
    lattice_pred = lattice_pred.cpu()
    spg_target = spg_target.cpu()
    letter_target = letter_target.cpu()
    site_mask = site_mask.cpu()

    B, S, _ = free_params.shape
    M_MULT = _MAX_MULT

    R_list, t_list, R0_list, t0_list, mask_list = [], [], [], [], []
    for b in range(B):
        spg_num = int(spg_target[b].item()) + 1  # 0-indexed -> 1..230
        for s in range(S):
            letter_idx = int(letter_target[b, s].item())
            R_k, t_k, R0_k, t0_k, m_k = _get_ops_cpu(spg_num, letter_idx)
            R_list.append(R_k); t_list.append(t_k)
            R0_list.append(R0_k); t0_list.append(t0_k); mask_list.append(m_k)

    R_g  = torch.stack(R_list).view(B, S, M_MULT, 3, 3)
    t_g  = torch.stack(t_list).view(B, S, M_MULT, 3)
    R0_g = torch.stack(R0_list).view(B, S, 3, 3)
    t0_g = torch.stack(t0_list).view(B, S, 3)
    op_mask = torch.stack(mask_list).view(B, S, M_MULT)

    fp = free_params % 1.0
    rep = torch.einsum('bsij,bsj->bsi', R0_g, fp) + t0_g
    rep = rep % 1.0
    pos = torch.einsum('bsmij,bsj->bsmi', R_g, rep) + t_g
    pos = pos % 1.0

    Mlat = _lattice_matrix_from_pred(lattice_pred)
    cart = torch.einsum('bsmd,bde->bsme', pos, Mlat)

    atom_mask = site_mask.float().unsqueeze(-1) * op_mask
    SM = S * M_MULT
    cart_flat = cart.reshape(B, SM, 3)
    amask = atom_mask.reshape(B, SM)

    diff = cart_flat.unsqueeze(2) - cart_flat.unsqueeze(1)
    # 注意: 不用 diff.norm(dim=-1)。对角线(同一原子与自身, diff=0)在每次
    # 前向都必然出现; torch.norm在x=0处的梯度是0/0=nan, 即使前向值后续被
    # mask清零, 反向时 0(mask)*nan(d(norm)/d(diff)) 仍是 nan, 污染整个梯度。
    # sqrt前加eps可让0处梯度有限(1/(2*sqrt(eps)))，避免该问题。
    dist = (diff ** 2).sum(-1).clamp(min=1e-12).sqrt()

    # ─── [NaN防护 / 修复2 - 兜底] ──────────────────────────────────────────
    # clamp(min=...)不会修复NaN输入：torch.clamp(nan, min=x) 的结果仍是nan。
    # 万一上游(例如_get_ops_cpu修复前已缓存的坏值、或decoder早期训练阶段
    # 偶发的极端输出)仍有NaN/Inf流入dist，下面这行把异常距离视为
    # "远大于threshold"——经过relu(threshold - dist)后自然贡献0惩罚。
    #
    # 这一步是必要的，因为后面的 viol = relu(...) * pair_mask 这个乘法
    # 本身不是NaN-safe的：0 * nan = nan(不是0!)。如果dist里有nan，即使
    # pair_mask在该位置是0，viol仍会是nan，进而通过.sum()和.mean()扩散
    # 到整个batch的loss_overlap。
    dist = torch.nan_to_num(dist, nan=threshold + 1.0,
                             posinf=threshold + 1.0, neginf=0.0)
    # ─────────────────────────────────────────────────────────────────────

    pair_mask = amask.unsqueeze(2) * amask.unsqueeze(1)
    if SM not in _EYE_CACHE:
        _EYE_CACHE[SM] = torch.eye(SM).unsqueeze(0)
    eye = _EYE_CACHE[SM]
    pair_mask = pair_mask * (1.0 - eye)

    viol = torch.relu(threshold - dist) * pair_mask
    denom = pair_mask.sum(dim=(1, 2)).clamp(min=1.0)
    result = (viol.sum(dim=(1, 2)) / denom).mean()

    # ─── [NaN防护 / 修复2 - 最终保险] ──────────────────────────────────────
    # 即使以上防护都未命中，也不让单个batch里某个结构的NaN/Inf通过.mean()
    # 扩散到整个loss_overlap，进而污染total_loss和反向传播。这一行成本
    # 极低(单个标量)，作为最后一道防线。
    result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    # ─────────────────────────────────────────────────────────────────────

    return result.to(orig_device)


class WyckoffReconLoss(nn.Module):
    def __init__(self, 
                 w_spg=1.0, w_lattice=1.0, w_elem=1.0, 
                 w_letter=1.0, w_free=1.0, w_nsites=0.5,
                 mask_weight=3.0,
                 w_overlap=1.0, overlap_threshold=0.8):
        super().__init__()
        self.w_spg = w_spg
        self.w_lattice = w_lattice
        self.w_elem = w_elem
        self.w_letter = w_letter
        self.w_free = w_free
        self.w_nsites = w_nsites
        self.mask_weight = mask_weight
        self.w_overlap = w_overlap
        self.overlap_threshold = overlap_threshold

    def forward(self, preds, targets, site_mask, masked_sites=None):
        """
        preds: dict from WyckoffDecoder.forward()
        targets: dict with ground truth tensors
          - spg_target: (B,) long
          - lattice_target: (B, 6) float
          - num_sites_target: (B,) long
          - elem_target: (B, max_sites) long
          - letter_target: (B, max_sites) long
          - free_target: (B, max_sites, 3) float
        site_mask: (B, max_sites) bool，有效位点为True
        masked_sites: (B, max_sites) bool，本步diffusion中被mask的位点为True
        """

        # sp_loss
        loss_spg = F.cross_entropy(preds['spg_logits'], targets['spg_target'])
        
        
        lat_scale = torch.tensor([10., 10., 10., 90., 90., 90.],
                              device=preds['lattice_pred'].device)
        loss_lattice = F.mse_loss(
            preds['lattice_pred'] / lat_scale,
            targets['lattice_target'] / lat_scale,
        )
        # MSE 恒非负；若出现负值或 NaN，是 sm_90 CUDA kernel 对
        # NaN/Inf 输入产出垃圾值所致（与 repeats<0、numel overflow 同源）。
        # .abs() 修正符号，nan_to_num 兜底清除 NaN/Inf。
        loss_lattice = loss_lattice.abs()
        loss_lattice = torch.nan_to_num(loss_lattice, nan=0.0, posinf=0.0, neginf=0.0)
        
        
        loss_nsites = F.cross_entropy(
            preds['num_sites_logits'], targets['num_sites_target']
        )
        
        
        mask = site_mask.float()  # (B, max_sites)
        
        # elem_loss
        elem_loss = F.cross_entropy(
            preds['elem_logits'].view(-1, preds['elem_logits'].shape[-1]),
            targets['elem_target'].view(-1),
            reduction='none'
        ).view_as(site_mask)
        # masked位点给予更高权重，强化diffusion去噪信号
        if masked_sites is not None:
            elem_weight = mask.clone()
            elem_weight[masked_sites & site_mask] = self.mask_weight
        else:
            elem_weight = mask
        loss_elem = (elem_loss * elem_weight).sum() / elem_weight.sum().clamp(min=1)
        
        # wyckoff_letter_loss
        letter_loss = F.cross_entropy(
            preds['letter_logits'].view(-1, preds['letter_logits'].shape[-1]),
            targets['letter_target'].view(-1),
            reduction='none'
        ).view_as(site_mask)
        # masked位点给予更高权重（letter也参与diffusion）
        if masked_sites is not None:
            letter_weight = mask.clone()
            letter_weight[masked_sites & site_mask] = self.mask_weight
        else:
            letter_weight = mask
        loss_letter = (letter_loss * letter_weight).sum() / letter_weight.sum().clamp(min=1)
        
        # free_loss
        free_loss = F.mse_loss(
            preds['free_params'], targets['free_target'], reduction='none'
        ).mean(-1)  # (B, max_sites)
        loss_free = (free_loss * mask).sum() / mask.sum().clamp(min=1)
        
        # 轨道感知原子间距重叠惩罚，固定阈值0.8Å
        loss_overlap = _overlap_penalty(
            preds['free_params'], preds['lattice_pred'],
            targets['spg_target'], targets['letter_target'],
            site_mask, threshold=self.overlap_threshold,
        )
        
        total = (
            self.w_spg * loss_spg +
            self.w_lattice * loss_lattice +
            self.w_nsites * loss_nsites +
            self.w_elem * loss_elem +
            self.w_letter * loss_letter +
            self.w_free * loss_free +
            self.w_overlap * loss_overlap
        )
        
        return total, {
            'loss_spg': loss_spg.item(),
            'loss_lattice': loss_lattice.item(),
            'loss_nsites': loss_nsites.item(),
            'loss_elem': loss_elem.item(),
            'loss_letter': loss_letter.item(),
            'loss_free': loss_free.item(),
            'loss_overlap': loss_overlap.item(),
        }