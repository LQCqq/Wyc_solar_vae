import torch
import torch.nn as nn
from torch_scatter import scatter_mean


class SetAttentionBlock(nn.Module):
    """SAB: sites之间的permutation equivariant self-attention"""
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ff(x))


class WyckoffEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        num_spg: int = 230,
        num_wyckoff_letters: int = 27,
        num_elements: int = 100,
        max_sites: int = 12,
    ):
        super().__init__()
        self.max_sites = max_sites
        self.hidden_dim = hidden_dim
        self.spg_emb = nn.Embedding(num_spg, hidden_dim // 4)
        self.letter_emb = nn.Embedding(num_wyckoff_letters, hidden_dim // 4)
        self.element_emb = nn.Embedding(num_elements, hidden_dim // 4)
        # 真正的 mask diffusion：可学习的 MASK token
        self.elem_mask_token = nn.Parameter(torch.zeros(hidden_dim // 4))
        nn.init.normal_(self.elem_mask_token)
        self.free_param_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
        )
        self.multi_emb = nn.Linear(1, hidden_dim // 4)
        self.site_mlp = nn.Sequential(
            nn.Linear(hidden_dim // 4 * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Set Transformer: SAB用于sites之间的inter-site attention
        self.sab = SetAttentionBlock(hidden_dim, num_heads=4)
        # Per-site latent：每个site有自己的mu和log_var（消除shortcut）
        self.per_site_mu_head = nn.Linear(hidden_dim, hidden_dim)
        self.per_site_log_var_head = nn.Linear(hidden_dim, hidden_dim)
        # 全局聚合：scatter_mean用于SPG/lattice预测（全局信息）
        self.crystal_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data, elem_mask=None):
        letter_feat = self.letter_emb(data.wyk_letters)
        elem_feat = self.element_emb(data.wyk_atom_types)
        if elem_mask is not None:
            mask_token = self.elem_mask_token.unsqueeze(0).expand_as(elem_feat)
            elem_feat = torch.where(
                elem_mask.unsqueeze(-1),
                mask_token,
                elem_feat
            )
        free_feat = self.free_param_mlp(data.wyk_free)
        multi_feat = self.multi_emb(data.wyk_multi.unsqueeze(-1).float())
        site_feat = torch.cat(
            [letter_feat, elem_feat, free_feat, multi_feat], dim=-1
        )
        site_feat = self.site_mlp(site_feat)
        B = data.num_wyk_sites.shape[0]
        device = data.wyk_atom_types.device
        # nan_to_num+clamp+.long()这条链在本机(H200/sm_90)GPU上不稳定，会产出垃圾值
        # (曾表现为负数或全0)，连带repeat_interleave"合法地"算错。整条链+repeat_interleave
        # 全部放CPU算(超小张量，几乎零成本)，结果再.to(device)。
        num_sites_safe = data.num_wyk_sites.cpu().view(-1).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 100).long()
        wyk_batch = torch.repeat_interleave(
            torch.arange(B, device='cpu'),
            num_sites_safe
        ).to(device)
        # 填充为(B, max_sites, D)，用于SAB和per-site latent
        padded = torch.zeros(B, self.max_sites, self.hidden_dim, device=device)
        enc_padding_mask = torch.ones(B, self.max_sites, dtype=torch.bool, device=device)  # True=padding
        offset = 0
        for i in range(B):
            n = min(int(num_sites_safe[i].item()), self.max_sites)
            padded[i, :n] = site_feat[offset:offset + n]
            enc_padding_mask[i, :n] = False
            offset += int(num_sites_safe[i].item())
        # SAB：sites之间self-attention，丰富per-site表示
        enriched = self.sab(padded, key_padding_mask=enc_padding_mask)  # (B, max_sites, D)
        # Per-site latent：每个site的mu和log_var（消除cross-attention shortcut）
        per_site_mu = self.per_site_mu_head(enriched)           # (B, max_sites, D)
        per_site_log_var = self.per_site_log_var_head(enriched)  # (B, max_sites, D)
        # 全局聚合：scatter_mean用于SPG/lattice（全局结构信息）
        crystal_feat = scatter_mean(site_feat, wyk_batch, dim=0, dim_size=B)
        spg_feat = self.spg_emb(data.spg_idx.squeeze(-1))
        crystal_feat = self.crystal_mlp(
            torch.cat([crystal_feat, spg_feat], dim=-1)
        )
        mu = self.fc_mu(crystal_feat)
        log_var = self.fc_var(crystal_feat)
        return mu, log_var, per_site_mu, per_site_log_var, enc_padding_mask