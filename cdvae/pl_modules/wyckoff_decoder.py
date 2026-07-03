import numpy as _np

def _fix_lattice(lp):
    """归一化
    """
    lp = lp.copy().astype(float)

    lengths = _np.abs(lp[:3])
    lengths = _np.clip(lengths, 2.0, 30.0)

    angles = _np.asarray(lp[3:], dtype=float)
    # 若处于弧度范围(|θ|<2π+裕度)则转角度
    if _np.all(_np.abs(angles) < 2.0 * _np.pi + 0.5):
        angles = _np.degrees(angles)
    angles = _np.abs(angles)
    angles = _np.clip(angles, 30.0, 150.0)

    for _ in range(20):
        ca, cb, cg = _np.cos(_np.radians(angles))
        vol = 1.0 + 2.0 * ca * cb * cg - ca * ca - cb * cb - cg * cg
        if vol > 0.05:
            break
        angles = 90.0 + (angles - 90.0) * 0.8
    else:
        angles = _np.array([90.0, 90.0, 90.0])

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
        num_spg: int = 230,
        num_wyckoff_letters: int = 27,
        num_elements: int = 100,
    ):
        super().__init__()
        self.max_sites = max_sites
        self.num_spg = num_spg
        self.num_letters = num_wyckoff_letters
        self.num_elements = num_elements
        
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

        # site_z_projector 从全局z派生per-site z
        self.site_z_projector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )


    def forward(self, z, t=None, noisy_elem_ids=None, noisy_letter_ids=None, per_site_z=None, enc_padding_mask=None):
        """
        z: (B, latent_dim)
        Returns dict of predictions (logits)
        """
        B = z.shape[0]
        h = self.fc_shared(z)  # (B, D)

        # 加入时间步条件
        if t is not None:
            h = h + self.time_emb(t)

        # 预测空间群
        spg_logits = self.spg_head(h)  # (B, 230)
        
        # 预测晶格
        lattice_pred = self.lattice_head(h)  # (B, 6)
        
        # 预测位点数量
        num_sites_logits = self.num_sites_head(h)  # (B, max_sites)
        
        # 解码位点
        site_h = self.site_decoder(h)  # (B, D)
        
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

        # cross-attention：decoder site特征attend到per-site latent z（训练/生成均可用）
        if per_site_z is not None:
            ca_out, _ = self.cross_attn(
                site_feats, per_site_z, per_site_z,
                key_padding_mask=enc_padding_mask
            )
            site_feats = self.cross_attn_norm(site_feats + ca_out)

        elem_logits = self.element_head(site_feats)
        letter_logits = self.letter_head(site_feats)
        free_params = torch.sigmoid(self.free_param_head(site_feats))
        
        return {
            'spg_logits': spg_logits,           # (B, 230)
            'lattice_pred': lattice_pred,         # (B, 6)
            'num_sites_logits': num_sites_logits, # (B, max_sites)
            'elem_logits': elem_logits,           # (B, max_sites, num_elem)
            'letter_logits': letter_logits,       # (B, max_sites, num_letters)
            'free_params': free_params,           # (B, max_sites, 3)
        }

    @torch.no_grad()
    def decode_to_wyckoff(self, z, temperature=0.5):
        B = z.shape[0]
        device = z.device

        # 预测SPG
        h_init = self.fc_shared(z)
        spg_logits_init = self.spg_head(h_init)
        spg_nums = spg_logits_init.argmax(-1) + 1  # (B,) 1-indexed

        # 构建SPG-letter合法性mask（预缓存所有晶体）
        def build_spg_letter_mask(spg_num, num_letters, device):
            """返回该SPG合法的letter的0/1 mask"""
            try:
                from pyxtal.symmetry import Group
                g = Group(int(spg_num))
                mask = torch.zeros(num_letters, device=device)
                for i in range(len(g.Wyckoff_positions)):
                    if i < num_letters:
                        mask[i] = 1.0
                return mask if mask.sum() > 0 else torch.ones(num_letters, device=device)
            except:
                return torch.ones(num_letters, device=device)

        letter_masks = torch.stack([
            build_spg_letter_mask(spg_nums[b].item(), self.num_letters, device)
            for b in range(B)
        ])  # (B, num_letters)

        
        # site_z_projector(z)，携带全局晶体化学信息
        z_proj = self.site_z_projector(z)                           # (B, hidden_dim)
        z_proj = z_proj.unsqueeze(1).expand(B, self.max_sites, -1)  # (B, max_sites, D)
        per_site_z = z_proj + torch.randn_like(z_proj) * 0.2       # 小噪声保持多样性

        noisy_elem_ids = torch.zeros(B, self.max_sites, dtype=torch.long, device=device)
        noisy_letter_ids = torch.zeros(B, self.max_sites, dtype=torch.long, device=device)

        # 去噪：从T到1
        for step in range(self.T, 0, -1):
            t = torch.full((B,), step, dtype=torch.long, device=device)
            preds = self.forward(z, t=t, noisy_elem_ids=noisy_elem_ids, noisy_letter_ids=noisy_letter_ids, per_site_z=per_site_z)

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
            update = still_masked & should_unmask
            noisy_elem_ids = torch.where(update, sampled_elems, noisy_elem_ids)
            noisy_letter_ids = torch.where(update, sampled_letters, noisy_letter_ids)

    
        t_final = torch.ones(B, dtype=torch.long, device=device)
        preds = self.forward(z, t=t_final, noisy_elem_ids=noisy_elem_ids, noisy_letter_ids=noisy_letter_ids, per_site_z=per_site_z)

        results = []
        for i in range(B):
            # 用与 letter mask 一致的 spg（spg_nums，初始预测, 避免 t=1 重新argmax得到不同spg，导致letter与空间群不匹配
            spg_num = int(spg_nums[i].item())
            n_sites = preds['num_sites_logits'][i].argmax().item() + 1

            elements = []
            letters = []
            free_params = []
            for s in range(n_sites):
        
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

            results.append({
                'spacegroup_num': spg_num,
                'site_elements': elements,
                'site_letters': letters,
                'site_free_params': free_params,
                'lattice_params': _fix_lattice(preds['lattice_pred'][i].cpu().numpy()),
                'num_sites': n_sites,
            })
        return results