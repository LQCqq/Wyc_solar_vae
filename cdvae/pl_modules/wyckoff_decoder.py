# models/wyckoff_decoder.py
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
        max_sites: int = 8,        # 每个晶体最多预测的Wyckoff位点数
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
        
        
        # 并行解码：预测 max_sites 个位点，用 mask 截断
        self.site_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 位点预测head（并行 max_sites 个）
        self.element_head = nn.Linear(hidden_dim, num_elements * max_sites)
        self.letter_head = nn.Linear(hidden_dim, num_wyckoff_letters * max_sites)
        self.free_param_head = nn.Linear(hidden_dim, 3 * max_sites)

    def forward(self, z):
        """
        z: (B, latent_dim)
        Returns dict of predictions (logits)
        """
        B = z.shape[0]
        h = self.fc_shared(z)  # (B, D)
        
        # 预测空间群
        spg_logits = self.spg_head(h)  # (B, 230)
        
        # 预测晶格
        lattice_pred = self.lattice_head(h)  # (B, 6)
        
        # 预测位点数量
        num_sites_logits = self.num_sites_head(h)  # (B, max_sites)
        
        # 解码位点
        site_h = self.site_decoder(h)  # (B, D)
        
        # 展开为每个位点的预测
        elem_logits = self.element_head(site_h).view(B, self.max_sites, self.num_elements)
        letter_logits = self.letter_head(site_h).view(B, self.max_sites, self.num_letters)
        free_params = self.free_param_head(site_h).view(B, self.max_sites, 3)
        free_params = torch.sigmoid(free_params)  # 自由参数在 [0, 1)
        
        return {
            'spg_logits': spg_logits,           # (B, 230)
            'lattice_pred': lattice_pred,         # (B, 6)
            'num_sites_logits': num_sites_logits, # (B, max_sites)
            'elem_logits': elem_logits,           # (B, max_sites, num_elem)
            'letter_logits': letter_logits,       # (B, max_sites, num_letters)
            'free_params': free_params,           # (B, max_sites, 3)
        }

    @torch.no_grad()
    def decode_to_wyckoff(self, z, temperature=1.0):
        preds = self.forward(z)
        B = z.shape[0]
        
        results = []
        for i in range(B):
            spg_num = preds['spg_logits'][i].argmax().item() + 1
            n_sites = preds['num_sites_logits'][i].argmax().item() + 1
            
            elements = []
            letters = []
            free_params = []
            for s in range(n_sites):
                elem_z = preds['elem_logits'][i, s].argmax().item() + 1
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
                'lattice_params': preds['lattice_pred'][i].cpu().numpy(),
                'num_sites': n_sites,
            })
        return results