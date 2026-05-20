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
    ):
        super().__init__()
        self.spg_emb = nn.Embedding(num_spg, hidden_dim // 4)
        self.letter_emb = nn.Embedding(num_wyckoff_letters, hidden_dim // 4)
        self.element_emb = nn.Embedding(num_elements, hidden_dim // 4)
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

    def forward(self, data):
        letter_feat = self.letter_emb(data.wyk_letters)
        elem_feat = self.element_emb(data.wyk_atom_types)
        free_feat = self.free_param_mlp(data.wyk_free)
        multi_feat = self.multi_emb(data.wyk_multi.unsqueeze(-1).float())
        site_feat = torch.cat([letter_feat, elem_feat, free_feat, multi_feat], dim=-1)
        site_feat = self.site_mlp(site_feat)

        # 不要用follow batch,直接拿sites
        B = data.num_wyk_sites.shape[0]
        wyk_batch = torch.repeat_interleave(
            torch.arange(B, device=data.wyk_atom_types.device),
            data.num_wyk_sites
        )
        crystal_feat = scatter_mean(site_feat, wyk_batch, dim=0, dim_size=B)
        spg_feat = self.spg_emb(data.spg_idx.squeeze(-1))
        crystal_feat = self.crystal_mlp(torch.cat([crystal_feat, spg_feat], dim=-1))

        mu = self.fc_mu(crystal_feat)
        log_var = self.fc_var(crystal_feat)
        return mu, log_var
