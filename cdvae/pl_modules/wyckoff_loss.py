# models/wyckoff_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class WyckoffReconLoss(nn.Module):
    def __init__(self, 
                 w_spg=1.0, w_lattice=1.0, w_elem=1.0, 
                 w_letter=1.0, w_free=1.0, w_nsites=0.5):
        super().__init__()
        self.w_spg = w_spg
        self.w_lattice = w_lattice
        self.w_elem = w_elem
        self.w_letter = w_letter
        self.w_free = w_free
        self.w_nsites = w_nsites

    def forward(self, preds, targets, site_mask):
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
        """

        # sp_loss
        loss_spg = F.cross_entropy(preds['spg_logits'], targets['spg_target'])
        
        
        lat_scale = torch.tensor([10., 10., 10., 90., 90., 90.],
                              device=preds['lattice_pred'].device)
        loss_lattice = F.mse_loss(
            preds['lattice_pred'] / lat_scale,
            targets['lattice_target'] / lat_scale,
        )
        
        
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
        loss_elem = (elem_loss * mask).sum() / mask.sum().clamp(min=1)
        
        # wyckoff_letter_loss
        letter_loss = F.cross_entropy(
            preds['letter_logits'].view(-1, preds['letter_logits'].shape[-1]),
            targets['letter_target'].view(-1),
            reduction='none'
        ).view_as(site_mask)
        loss_letter = (letter_loss * mask).sum() / mask.sum().clamp(min=1)
        
        # free_loss
        free_loss = F.mse_loss(
            preds['free_params'], targets['free_target'], reduction='none'
        ).mean(-1)  # (B, max_sites)
        loss_free = (free_loss * mask).sum() / mask.sum().clamp(min=1)
        
        total = (
            self.w_spg * loss_spg +
            self.w_lattice * loss_lattice +
            self.w_nsites * loss_nsites +
            self.w_elem * loss_elem +
            self.w_letter * loss_letter +
            self.w_free * loss_free
        )
        
        return total, {
            'loss_spg': loss_spg.item(),
            'loss_lattice': loss_lattice.item(),
            'loss_nsites': loss_nsites.item(),
            'loss_elem': loss_elem.item(),
            'loss_letter': loss_letter.item(),
            'loss_free': loss_free.item(),
        }