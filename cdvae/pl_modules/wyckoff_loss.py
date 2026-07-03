import torch
import torch.nn as nn
import torch.nn.functional as F


class WyckoffReconLoss(nn.Module):
    def __init__(self,
                 w_spg=1.0, w_lattice=1.0, w_elem=1.0,
                 w_letter=1.0, w_free=1.0, w_nsites=0.5,
                 mask_weight=3.0):
        super().__init__()
        self.w_spg = w_spg
        self.w_lattice = w_lattice
        self.w_elem = w_elem
        self.w_letter = w_letter
        self.w_free = w_free
        self.w_nsites = w_nsites
        self.mask_weight = mask_weight

    def forward(self, preds, targets, site_mask, masked_sites=None):

        # spg loss
        loss_spg = F.cross_entropy(preds['spg_logits'], targets['spg_target'])

        # lattice loss
        # 修复：只归一化 target，pred 直接学归一化后的值；
        # 与 _fix_lattice 的 ×scale 反归一化一致。
        # 旧代码 MSE(pred/scale, target/scale) 会把 pred 梯度 ÷ scale²，
        # 导致角度头(scale=90) 几乎收不到梯度、永远学不动。
        lat_scale = torch.tensor([10., 10., 10., 90., 90., 90.],
                                  device=preds['lattice_pred'].device)
        loss_lattice = F.mse_loss(
            preds['lattice_pred'],
            targets['lattice_target'] / lat_scale,
        )

        # nsites loss
        loss_nsites = F.cross_entropy(
            preds['num_sites_logits'], targets['num_sites_target']
        )

        mask = site_mask.float()  # (B, max_sites)

        # elem loss（masked位点给予更高权重，强化diffusion去噪信号）
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

        # letter loss（letter 也参与 diffusion 加权）
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

        # free loss
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