import math
import numpy as _np

def _project_lattice_by_spg(lattice_pred, spg_nums):
    rows = []
    for i in range(lattice_pred.shape[0]):
        row = lattice_pred[i]
        one = row.new_tensor(1.0)
        spg = int(spg_nums[i].item())
        if 3 <= spg <= 15:
            row = torch.stack([row[0], row[1], row[2], one, row[4], one])
        elif 16 <= spg <= 74:
            row = torch.stack([row[0], row[1], row[2], one, one, one])
        elif 75 <= spg <= 142:
            ab = row[0:2].mean()
            row = torch.stack([ab, ab, row[2], one, one, one])
        elif 143 <= spg <= 194:
            ab = row[0:2].mean()
            row = torch.stack([ab, ab, row[2], one, one, row.new_tensor(4.0 / 3.0)])
        elif 195 <= spg <= 230:
            abc = row[0:3].mean()
            row = torch.stack([abc, abc, abc, one, one, one])
        rows.append(row)
    return torch.stack(rows, dim=0)


def _wyckoff_metadata(spg_num, num_letters, device):
    valid = torch.zeros(num_letters, dtype=torch.bool, device=device)
    multiplicities = torch.zeros(num_letters, dtype=torch.long, device=device)
    fixed = torch.zeros(num_letters, dtype=torch.bool, device=device)
    try:
        from pyxtal.symmetry import Group
        from cdvae.pl_data.wyckoff_utils import WYCKOFF_LETTERS
        for wp in Group(int(spg_num)).Wyckoff_positions:
            if wp.letter in WYCKOFF_LETTERS[:num_letters]:
                idx = WYCKOFF_LETTERS.index(wp.letter)
                valid[idx] = True
                multiplicities[idx] = int(wp.multiplicity)
                fixed[idx] = int(wp.get_dof()) == 0
    except Exception:
        valid[:] = True
        multiplicities[:] = 1
    return valid, multiplicities, fixed


def _fix_lattice(lp, n_atoms=1, spg_num=None):
    """晶格参数后处理：
    - 长度：根据原子数动态设定下限，防止晶胞过小导致密度过高
    - 角度：收紧到 [45, 135] 覆盖常见晶系
    - 体积正定性检查：确保晶格矩阵行列式 > 0
    """
    lp = lp.copy().astype(float)

    lengths = _np.abs(lp[:3])
    # 根据原子数动态最小长度：n个原子至少需要约 n^(1/3) × 1.8Å
    min_len = max(2.5, float(n_atoms) ** (1.0 / 3.0) * 1.8)
    lengths = _np.clip(lengths, min_len, 15.0)

    angles = _np.asarray(lp[3:], dtype=float)
    # 若处于弧度范围(|θ|<2π+裕度)则转角度
    if _np.all(_np.abs(angles) < 2.0 * _np.pi + 0.5):
        angles = _np.degrees(angles)
    angles = _np.abs(angles)
    angles = _np.clip(angles, 45.0, 135.0)  # 收紧：覆盖立方/四方/六方/单斜常见范围

    for _ in range(20):
        ca, cb, cg = _np.cos(_np.radians(angles))
        vol = 1.0 + 2.0 * ca * cb * cg - ca * ca - cb * cb - cg * cg
        if vol > 0.05:
            break
        angles = 90.0 + (angles - 90.0) * 0.8
    else:
        angles = _np.array([90.0, 90.0, 90.0])

    if spg_num is not None:
        spg_num = int(spg_num)
        if 3 <= spg_num <= 15:
            angles[[0, 2]] = 90.0
        elif 16 <= spg_num <= 74:
            angles[:] = 90.0
        elif 75 <= spg_num <= 142:
            lengths[0:2] = lengths[0:2].mean()
            angles[:] = 90.0
        elif 143 <= spg_num <= 194:
            lengths[0:2] = lengths[0:2].mean()
            angles[:] = [90.0, 90.0, 120.0]
        elif 195 <= spg_num <= 230:
            lengths[:] = lengths.mean()
            angles[:] = 90.0

    return _np.concatenate([lengths, angles])

import torch
import torch.nn as nn
import torch.nn.functional as F



class WyckoffDecoder(nn.Module):
    """
    从向量z解码:
     空间群 (230)
     wyckoff elements, letter, free
     lattice
    """
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        max_sites: int = 12,        # 每个晶体最多预测的Wyckoff位点数
        max_atoms: int = 20,
        num_spg: int = 230,
        num_wyckoff_letters: int = 27,
        num_elements: int = 100,
        num_stability_classes: int = 4,
        site_prior_logvar_min: float = -10.0,
        site_prior_logvar_max: float = 2.0,
    ):
        super().__init__()
        self.max_sites = max_sites
        self.max_atoms = max_atoms
        self.num_spg = num_spg
        self.num_letters = num_wyckoff_letters
        self.num_elements = num_elements
        self.num_stability_classes = num_stability_classes
        self.stability_null_class = num_stability_classes
        self.site_prior_logvar_min = site_prior_logvar_min
        self.site_prior_logvar_max = site_prior_logvar_max
        
        # 特征
        self.fc_shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # 空间群预测head
        self.spg_head = nn.Linear(hidden_dim, num_spg)
        
        # 晶格参数预测head
        self.lattice_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 6),
        )
        
        # 位点数量预测head
        self.num_sites_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, max_sites),  # softmax → 位点数
        )
        
        
        # 预测 max_sites 个位点
        self.site_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 位点预测head,并行 max_sites个
        #self.element_head = nn.Linear(hidden_dim, num_elements * max_sites)
        #self.letter_head = nn.Linear(hidden_dim, num_wyckoff_letters * max_sites)
        #self.free_param_head = nn.Linear(hidden_dim, 3 * max_sites)


        self.site_pos_emb = nn.Embedding(max_sites, hidden_dim)
        self.site_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.element_head =nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_elements),
        )
        self.letter_head = nn.Linear(hidden_dim, num_wyckoff_letters)
        # free_param_head 3层MLP
        self.free_param_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 3),
        )

        # diffusion时间步embedding
        self.T = 100
        self.time_emb = nn.Embedding(self.T + 1, hidden_dim)

        # index 0 = MASK token，index 1-100 = 实际元素
        self.noisy_elem_emb = nn.Embedding(num_elements + 1, hidden_dim // 4)
        self.noisy_elem_proj = nn.Linear(hidden_dim // 4, hidden_dim)

        # letter diffusion：decoder接收noisy letter作为条件输入
        # index 0 = MASK token，index 1-27 = 实际letter
        self.noisy_letter_emb = nn.Embedding(num_wyckoff_letters + 1, hidden_dim // 4)
        self.noisy_letter_proj = nn.Linear(hidden_dim // 4, hidden_dim)

        self.hidden_dim = hidden_dim
        # cross-attention：decoder site特征attend到per-site latent z
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.site_self_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.site_self_attn_norm = nn.LayerNorm(hidden_dim)
        self.site_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.site_ffn_norm = nn.LayerNorm(hidden_dim)

        # site_z_projector 从全局z派生per-site z
        self.site_z_projector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.site_prior = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.site_prior_log_var = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.site_prior_log_var.weight)
        nn.init.constant_(self.site_prior_log_var.bias, math.log(0.2 ** 2))

        # ── CFG 元素条件（阶段1）：结构级 multi-hot(100) → hidden_dim ──
        # 训练时注入"该结构含哪些元素"，生成时注入"想要哪些元素(如硫族)"。
        # null_elem_cond 是可学习的"空条件"，CFG 随机丢弃/无条件分支时使用。
        self.elem_cond_proj = nn.Sequential(
            nn.Linear(100, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.null_elem_cond = nn.Parameter(torch.zeros(hidden_dim))

        # Ehull 条件：0..num_stability_classes-1 为真实类别，最后一类为 null。
        # 条件同时进入 global heads、site heads 和 conditional site prior。
        self.stability_emb = nn.Embedding(
            num_stability_classes + 1, hidden_dim
        )
        self.stability_cond_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.site_prior_stability_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )


    def _stability_features(self, z, stability_cond=None, cfg_drop=None):
        """Return a (B, hidden_dim) stability feature, including null CFG rows."""
        B = z.shape[0]
        device = z.device
        if stability_cond is None:
            stability_ids = torch.full(
                (B,), self.stability_null_class,
                dtype=torch.long, device=device,
            )
        else:
            stability_ids = torch.as_tensor(
                stability_cond, dtype=torch.long, device=device
            ).view(-1)
            if stability_ids.numel() == 1 and B > 1:
                stability_ids = stability_ids.expand(B)
            if stability_ids.numel() != B:
                raise ValueError(
                    f'stability_cond has {stability_ids.numel()} rows; expected {B}.'
                )
            if ((stability_ids < 0) |
                    (stability_ids >= self.num_stability_classes)).any():
                raise ValueError(
                    'stability_cond must be in '
                    f'[0, {self.num_stability_classes - 1}].'
                )
            if cfg_drop is not None:
                drop = torch.as_tensor(
                    cfg_drop, dtype=torch.bool, device=device
                ).view(-1)
                if drop.numel() != B:
                    raise ValueError(f'cfg_drop has {drop.numel()} rows; expected {B}.')
                null_ids = torch.full_like(
                    stability_ids, self.stability_null_class
                )
                stability_ids = torch.where(drop, null_ids, stability_ids)
        return self.stability_cond_proj(self.stability_emb(stability_ids))

    def site_prior_stats(self, z, stability_cond=None, cfg_drop=None):
        B = z.shape[0]
        z_proj = self.site_z_projector(z)                           # (B, hidden_dim)
        stability_feat = self._stability_features(
            z, stability_cond=stability_cond, cfg_drop=cfg_drop
        )
        z_proj = z_proj + self.site_prior_stability_proj(stability_feat)
        z_proj = z_proj.unsqueeze(1).expand(B, self.max_sites, -1)  # (B, max_sites, D)
        pos = self.site_pos_emb.weight.unsqueeze(0).expand(B, -1, -1)
        prior_mu = self.site_prior(torch.cat([z_proj, pos], dim=-1))
        prior_log_var = self.site_prior_log_var(prior_mu)
        prior_log_var = torch.nan_to_num(
            prior_log_var,
            nan=self.site_prior_logvar_min,
            posinf=self.site_prior_logvar_max,
            neginf=self.site_prior_logvar_min,
        ).clamp(self.site_prior_logvar_min, self.site_prior_logvar_max)
        return prior_mu, prior_log_var

    def sample_site_prior(self, z, stability_cond=None, cfg_drop=None, eps=None):
        prior_mu, prior_log_var = self.site_prior_stats(
            z, stability_cond=stability_cond, cfg_drop=cfg_drop
        )
        if eps is None:
            eps = torch.randn_like(prior_mu)
        per_site_z = prior_mu + eps * torch.exp(0.5 * prior_log_var)
        return per_site_z, prior_mu, prior_log_var

    def build_site_z(self, z, stability_cond=None):
        prior_mu, _ = self.site_prior_stats(
            z, stability_cond=stability_cond
        )
        return prior_mu

    def global_predictions(
        self, z, lattice_spg=None, stability_cond=None, cfg_drop=None
    ):
        h_global = self.fc_shared(z)  # (B, D)
        h_global = h_global + self._stability_features(
            z, stability_cond=stability_cond, cfg_drop=cfg_drop
        )
        # 预测空间群
        spg_logits = self.spg_head(h_global)  # (B, 230)
        if lattice_spg is None:
            lattice_spg = spg_logits.argmax(-1) + 1
        # 预测晶格
        lattice_pred = _project_lattice_by_spg(self.lattice_head(h_global), lattice_spg)  # (B, 6)
        # 预测位点数量
        num_sites_logits = self.num_sites_head(h_global)  # (B, max_sites)
        return h_global, spg_logits, lattice_pred, num_sites_logits

    def _initial_num_sites(self, spg_nums, num_sites_logits):
        n_sites = num_sites_logits.argmax(-1) + 1
        for b in range(n_sites.shape[0]):
            valid, multiplicities, _ = _wyckoff_metadata(
                spg_nums[b].item(), self.num_letters, n_sites.device
            )
            feasible = multiplicities[valid & (multiplicities > 0) & (multiplicities <= self.max_atoms)]
            if feasible.numel() > 0:
                n_sites[b] = min(int(n_sites[b].item()), self.max_atoms // int(feasible.min().item()))
        return n_sites.clamp(min=1, max=self.max_sites)

    def _constrain_final_letters(self, letter_logits, letter_ids, spg_nums, n_sites):
        constrained = letter_ids.clone()
        constrained_n_sites = n_sites.clone()
        for b in range(letter_logits.shape[0]):
            valid, multiplicities, fixed = _wyckoff_metadata(
                spg_nums[b].item(), self.num_letters, letter_logits.device
            )
            valid_indices = torch.where(valid & (multiplicities > 0) & (multiplicities <= self.max_atoms))[0]
            if valid_indices.numel() == 0:
                continue
            min_mult = int(multiplicities[valid_indices].min().item())
            used_fixed = set()
            atom_count = 0
            kept = 0
            for s in range(int(n_sites[b].item())):
                current = int(letter_ids[b, s].item()) - 1
                order = torch.argsort(letter_logits[b, s], descending=True).tolist()
                candidates = ([current] if current >= 0 else []) + [idx for idx in order if idx != current]
                remaining_sites = int(n_sites[b].item()) - s - 1
                chosen = None
                for idx in candidates:
                    if idx < 0 or idx >= self.num_letters or not bool(valid[idx]):
                        continue
                    if bool(fixed[idx]) and idx in used_fixed:
                        continue
                    mult = int(multiplicities[idx].item())
                    if atom_count + mult + remaining_sites * min_mult > self.max_atoms:
                        continue
                    chosen = idx
                    break
                if chosen is None:
                    constrained_n_sites[b] = kept
                    break
                constrained[b, s] = chosen + 1
                atom_count += int(multiplicities[chosen].item())
                if bool(fixed[chosen]):
                    used_fixed.add(chosen)
                kept += 1
            if kept == 0:
                chosen = int(valid_indices[torch.argmin(multiplicities[valid_indices])].item())
                constrained[b, 0] = chosen + 1
                constrained_n_sites[b] = 1
        return constrained, constrained_n_sites

    def forward(
        self, z, t=None, noisy_elem_ids=None, noisy_letter_ids=None,
        per_site_z=None, enc_padding_mask=None, site_padding_mask=None,
        elem_cond=None, stability_cond=None, cfg_drop=None,
        lattice_spg=None, global_state=None,
    ):
        """
        z: (B, latent_dim)
        Returns dict of predictions (logits)
        """
        B = z.shape[0]
        if global_state is None:
            h_global, spg_logits, lattice_pred, num_sites_logits = self.global_predictions(
                z,
                lattice_spg=lattice_spg,
                stability_cond=stability_cond,
                cfg_drop=cfg_drop,
            )
        else:
            h_global, spg_logits, lattice_pred, num_sites_logits = global_state
        lattice_raw = self.lattice_head(h_global)
        h_site = h_global

        # 加入时间步条件
        if t is not None:
            h_site = h_site + self.time_emb(t)
        
        # 解码位点
        site_h = self.site_decoder(h_site)  # (B, D)
        
        # 展开为每个位点的预测
        #elem_logits = self.element_head(site_h).view(B, self.max_sites, self.num_elements)
        #letter_logits = self.letter_head(site_h).view(B, self.max_sites, self.num_letters)
        #free_params = self.free_param_head(site_h).view(B, self.max_sites, 3)
        #free_params = torch.sigmoid(free_params)  # 自由参数在 [0, 1)
                
        pos = self.site_pos_emb.weight
        site_feats = site_h.unsqueeze(1) + pos.unsqueeze(0)
        # 加入noisy elem
        if noisy_elem_ids is not None:
            noisy_feats = self.noisy_elem_emb(noisy_elem_ids)     # (B, max_sites, D/4)
            noisy_feats = self.noisy_elem_proj(noisy_feats)        # (B, max_sites, D)
            site_feats = site_feats + noisy_feats
        # 加入noisy letter
        if noisy_letter_ids is not None:
            noisy_letter_feats = self.noisy_letter_emb(noisy_letter_ids)    # (B, max_sites, D/4)
            noisy_letter_feats = self.noisy_letter_proj(noisy_letter_feats)  # (B, max_sites, D)
            site_feats = site_feats + noisy_letter_feats
        site_feats = self.site_fusion(site_feats)

        # ── CFG 元素条件注入（阶段1）：结构级条件广播到所有位点 ──
        # elem_cond: (B,100) multi-hot；None 时整批用 null（无条件分支）
        # cfg_drop: (B,) bool，True 的样本用 null（训练时随机丢弃）
        B_ = site_feats.size(0)
        null_feat = self.null_elem_cond.unsqueeze(0).expand(B_, -1)  # (B, D)
        if elem_cond is not None:
            cond_feat = self.elem_cond_proj(elem_cond)              # (B, D)
            if cfg_drop is not None:
                cond_feat = torch.where(cfg_drop.unsqueeze(1), null_feat, cond_feat)
        else:
            cond_feat = null_feat
        site_feats = site_feats + cond_feat.unsqueeze(1)           # 广播到 (B, max_sites, D)

        if site_padding_mask is None:
            site_padding_mask = enc_padding_mask
        _site_mask = site_padding_mask.clone() if site_padding_mask is not None else None
        _all_site_pad = _site_mask.all(dim=1) if _site_mask is not None else None
        if _all_site_pad is not None and _all_site_pad.any():
            _site_mask[_all_site_pad, 0] = False
        sa_out, _ = self.site_self_attn(
            site_feats, site_feats, site_feats,
            key_padding_mask=_site_mask
        )
        site_feats = self.site_self_attn_norm(site_feats + sa_out)
        site_feats = self.site_ffn_norm(site_feats + self.site_ffn(site_feats))

        # cross-attention：decoder site特征attend到per-site latent z（训练/生成均可用）
        if per_site_z is not None:
            # 全padding行会让softmax(全-inf)=NaN：临时放开首位，算完还原
            _enc_mask = enc_padding_mask.clone() if enc_padding_mask is not None else None
            _all_pad = _enc_mask.all(dim=1) if _enc_mask is not None else None
            if _all_pad is not None and _all_pad.any():
                _enc_mask[_all_pad, 0] = False
            ca_out, _ = self.cross_attn(
                site_feats, per_site_z, per_site_z,
                key_padding_mask=_enc_mask
            )
            site_feats = self.cross_attn_norm(site_feats + ca_out)

        elem_logits = self.element_head(site_feats)
        letter_logits = self.letter_head(site_feats)
        free_params = torch.sigmoid(self.free_param_head(site_feats))
        
        return {
            'spg_logits': spg_logits,           # (B, 230)
            'lattice_pred': lattice_pred,         # (B, 6)
            'lattice_raw': lattice_raw,           # (B, 6)
            'num_sites_logits': num_sites_logits, # (B, max_sites)
            'elem_logits': elem_logits,           # (B, max_sites, num_elem)
            'letter_logits': letter_logits,       # (B, max_sites, num_letters)
            'free_params': free_params,           # (B, max_sites, 3)
        }

    @torch.no_grad()
    def decode_to_wyckoff(
        self, z, temperature=0.5, elem_cond=None,
        stability_cond=None, cfg_w=0.0,
    ):
        """
        elem_cond: (B,100) 或 (100,) multi-hot 目标元素集合(如硫族)；None=无条件
        stability_cond: (B,) 或标量，0表示最低Ehull类别；None=无条件
        cfg_w:     classifier-free guidance 强度。0=普通条件(或无条件)，
                   >0 时对全部结构输出做联合CFG：
                   pred_uncond + (1+cfg_w)*(pred_cond-pred_uncond)
        """
        B = z.shape[0]
        device = z.device

        # 规整 elem_cond 到 (B,100)
        if elem_cond is not None:
            elem_cond = elem_cond.to(device).float().view(-1, 100)
            if elem_cond.size(0) == 1 and B > 1:
                elem_cond = elem_cond.expand(B, -1)

        if stability_cond is not None:
            stability_cond = torch.as_tensor(
                stability_cond, dtype=torch.long, device=device
            ).view(-1)
            if stability_cond.numel() == 1 and B > 1:
                stability_cond = stability_cond.expand(B)
            if stability_cond.numel() != B:
                raise ValueError(
                    f'stability_cond has {stability_cond.numel()} rows; expected {B}.'
                )

        use_cfg = (
            cfg_w > 0 and
            (elem_cond is not None or stability_cond is not None)
        )

        def _cfg_value(cond_value, uncond_value):
            if uncond_value is None or not use_cfg:
                return cond_value
            return uncond_value + (1.0 + cfg_w) * (
                cond_value - uncond_value
            )

        def _cfg_predictions(preds_c, preds_u):
            """Joint CFG for global, discrete-site and continuous-site outputs."""
            if preds_u is None or cfg_w <= 0:
                return preds_c
            guided = {
                key: _cfg_value(value, preds_u.get(key))
                for key, value in preds_c.items()
            }
            # Continuous fractional coordinates must remain inside the unit cell.
            guided['free_params'] = guided['free_params'].clamp(0.0, 1.0 - 1e-7)
            return guided

        # 条件/无条件 global branches。稳定性条件在这里直接影响 SPG、lattice、site count。
        h_cond, spg_cond, _, nsites_cond = self.global_predictions(
            z, stability_cond=stability_cond
        )
        if use_cfg:
            h_uncond, spg_uncond, _, nsites_uncond = self.global_predictions(
                z, stability_cond=None
            )
        else:
            h_uncond = spg_uncond = nsites_uncond = None

        spg_logits_init = _cfg_value(spg_cond, spg_uncond)
        num_sites_logits_init = _cfg_value(nsites_cond, nsites_uncond)
        spg_nums = spg_logits_init.argmax(-1) + 1  # (B,) 1-indexed
        lattice_cond = _project_lattice_by_spg(
            self.lattice_head(h_cond), spg_nums
        )
        lattice_uncond = None
        if use_cfg:
            lattice_uncond = _project_lattice_by_spg(
                self.lattice_head(h_uncond), spg_nums
            )
        lattice_pred_init = _cfg_value(lattice_cond, lattice_uncond)
        global_state_cond = (
            h_cond, spg_cond, lattice_cond, nsites_cond
        )
        global_state_uncond = None
        if use_cfg:
            global_state_uncond = (
                h_uncond, spg_uncond, lattice_uncond, nsites_uncond
            )
        n_sites = self._initial_num_sites(spg_nums, num_sites_logits_init)
        site_padding_mask = torch.arange(self.max_sites, device=device).unsqueeze(0) >= n_sites.unsqueeze(1)

        # 构建SPG-letter合法性mask（预缓存所有晶体）
        def build_spg_letter_mask(spg_num, num_letters, device):
            """返回该SPG合法的letter的0/1 mask"""
            try:
                from pyxtal.symmetry import Group
                from cdvae.pl_data.wyckoff_utils import WYCKOFF_LETTERS
                g = Group(int(spg_num))
                mask = torch.zeros(num_letters, device=device)
                # 按字母名对齐：pyxtal 的 Wyckoff_positions 按多重度排序，
                # 与模型 letter 类别的字母序不一致，必须按字母匹配而非位置，
                # 否则会随机允许非法 letter、禁止合法 letter
                letters_avail = {wp.letter for wp in g.Wyckoff_positions}
                for i, ch in enumerate(WYCKOFF_LETTERS[:num_letters]):
                    if ch in letters_avail:
                        mask[i] = 1.0
                return mask if mask.sum() > 0 else torch.ones(num_letters, device=device)
            except:
                return torch.ones(num_letters, device=device)

        letter_masks = torch.stack([
            build_spg_letter_mask(spg_nums[b].item(), self.num_letters, device)
            for b in range(B)
        ])  # (B, num_letters)

        
        # 条件 site prior；CFG 两个分支共用同一 eps，差异只来自条件。
        prior_mu_cond, prior_log_var_cond = self.site_prior_stats(
            z, stability_cond=stability_cond
        )
        site_eps = torch.randn_like(prior_mu_cond)
        per_site_z_cond = prior_mu_cond + site_eps * torch.exp(
            0.5 * prior_log_var_cond
        )
        per_site_z_uncond = None
        if use_cfg:
            prior_mu_uncond, prior_log_var_uncond = self.site_prior_stats(
                z, stability_cond=None
            )
            per_site_z_uncond = prior_mu_uncond + site_eps * torch.exp(
                0.5 * prior_log_var_uncond
            )

        noisy_elem_ids = torch.zeros(B, self.max_sites, dtype=torch.long, device=device)
        noisy_letter_ids = torch.zeros(B, self.max_sites, dtype=torch.long, device=device)

        def _predict_step(t_value, padding_mask):
            preds_cond = self.forward(
                z,
                t=t_value,
                noisy_elem_ids=noisy_elem_ids,
                noisy_letter_ids=noisy_letter_ids,
                per_site_z=per_site_z_cond,
                enc_padding_mask=padding_mask,
                site_padding_mask=padding_mask,
                elem_cond=elem_cond,
                stability_cond=stability_cond,
                lattice_spg=spg_nums,
                global_state=global_state_cond,
            )
            if not use_cfg:
                return preds_cond
            preds_uncond = self.forward(
                z,
                t=t_value,
                noisy_elem_ids=noisy_elem_ids,
                noisy_letter_ids=noisy_letter_ids,
                per_site_z=per_site_z_uncond,
                enc_padding_mask=padding_mask,
                site_padding_mask=padding_mask,
                elem_cond=None,
                stability_cond=None,
                lattice_spg=spg_nums,
                global_state=global_state_uncond,
            )
            return _cfg_predictions(preds_cond, preds_uncond)

        # 去噪：从T到1
        for step in range(self.T, 0, -1):
            t = torch.full((B,), step, dtype=torch.long, device=device)
            preds = _predict_step(t, site_padding_mask)

            # 对所有位点采样elem
            elem_probs = torch.softmax(preds['elem_logits'] / temperature, dim=-1)  # (B, S, 100)
            B_, S_, V_ = elem_probs.shape
            sampled_elems = torch.multinomial(
                elem_probs.view(-1, V_), 1
            ).view(B_, S_) + 1  # 1-indexed元素ID

            # 对所有位点采样letter（SPG约束）
            letter_probs = torch.softmax(preds['letter_logits'] / temperature, dim=-1)  # (B, S, 27)
            letter_probs = letter_probs * letter_masks.unsqueeze(1)  # (B, S, 27)
            letter_probs = letter_probs / letter_probs.sum(-1, keepdim=True).clamp(min=1e-8)
            B_, S_, L_ = letter_probs.shape
            sampled_letters = torch.multinomial(
                letter_probs.view(-1, L_), 1
            ).view(B_, S_) + 1  # 1-indexed letter ID

            unmask_prob = 1.0 / step
            should_unmask = torch.rand(B, self.max_sites, device=device) < unmask_prob

    
            still_masked = (noisy_elem_ids == 0)
            update = still_masked & should_unmask & ~site_padding_mask
            noisy_elem_ids = torch.where(update, sampled_elems, noisy_elem_ids)
            noisy_letter_ids = torch.where(update, sampled_letters, noisy_letter_ids)

        
        t_final = torch.ones(B, dtype=torch.long, device=device)
        preds = _predict_step(t_final, site_padding_mask)
        noisy_letter_ids, n_sites = self._constrain_final_letters(
            preds['letter_logits'], noisy_letter_ids, spg_nums, n_sites
        )
        site_padding_mask = torch.arange(self.max_sites, device=device).unsqueeze(0) >= n_sites.unsqueeze(1)
        preds = _predict_step(t_final, site_padding_mask)

        results = []
        for i in range(B):
            # 用与 letter mask 一致的 spg（spg_nums，初始预测, 避免 t=1 重新argmax得到不同spg，导致letter与空间群不匹配
            spg_num = int(spg_nums[i].item())
            n_sites_i = int(n_sites[i].item())

            elements = []
            letters = []
            free_params = []
            for s in range(n_sites_i):
        
                elem_z = noisy_elem_ids[i, s].item()
                if elem_z == 0:
                    elem_z = preds['elem_logits'][i, s].argmax().item() + 1
                # 用去噪后的letter ID
                letter_idx = noisy_letter_ids[i, s].item() - 1  # 转为0-indexed
                if noisy_letter_ids[i, s].item() == 0:
                    letter_idx = preds['letter_logits'][i, s].argmax().item()
                fp = preds['free_params'][i, s].cpu().numpy()

                from pymatgen.core import Element
                try:
                    elem = Element.from_Z(elem_z).symbol
                except:
                    elem = 'Si'

                from cdvae.pl_data.wyckoff_utils import WYCKOFF_LETTERS, wyckoff_to_structure
                letter = WYCKOFF_LETTERS[letter_idx]

                elements.append(elem)
                letters.append(letter)
                free_params.append(fp)

            # 计算展开后的总原子数，用于 _fix_lattice 的动态长度下限
            try:
                from pyxtal.symmetry import Group as _Group
                _g = _Group(spg_num)
                _valid_letters = {wp.letter for wp in _g.Wyckoff_positions}
                _n_atoms = sum(
                    _g[lt].multiplicity if lt in _valid_letters else 1
                    for lt in letters
                )
            except Exception:
                _n_atoms = n_sites_i * 2

            results.append({
                'spacegroup_num': spg_num,
                'site_elements': elements,
                'site_letters': letters,
                'site_free_params': free_params,
                'lattice_params': _fix_lattice(lattice_pred_init[i].cpu().numpy() * _np.array([10., 10., 10., 90., 90., 90.]), n_atoms=_n_atoms, spg_num=spg_num),  # ×lat_scale 还原归一化（训练时 target/lat_scale，生成必须乘回）
                'num_sites': n_sites_i,
            })
        return results
