import torch
import torch.nn as nn
from torch_scatter import scatter_mean

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
        wyk_batch = torch.repeat_interleave(
            torch.arange(B, device=data.wyk_atom_types.device),
            data.num_wyk_sites.view(-1).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 100).long()
        )
        # 将flat site_feat填充为(B, max_sites, D)，供decoder cross-attention使用
        enc_site_feats = torch.zeros(B, self.max_sites, self.hidden_dim, device=site_feat.device)
        enc_padding_mask = torch.ones(B, self.max_sites, dtype=torch.bool, device=site_feat.device)
        num_sites_safe = data.num_wyk_sites.view(-1).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 100).long()
        offset = 0
        for i in range(B):
            n = min(int(num_sites_safe[i].item()), self.max_sites)
            enc_site_feats[i, :n] = site_feat[offset:offset + n]
            enc_padding_mask[i, :n] = False
            offset += int(num_sites_safe[i].item())

        crystal_feat = scatter_mean(site_feat, wyk_batch, dim=0, dim_size=B)
        spg_feat = self.spg_emb(data.spg_idx.squeeze(-1))
        crystal_feat = self.crystal_mlp(
            torch.cat([crystal_feat, spg_feat], dim=-1)
        )
        mu = self.fc_mu(crystal_feat)
        log_var = self.fc_var(crystal_feat)
        return mu, log_var, enc_site_feats, enc_padding_mask