from typing import Any, Dict

import hydra
import numpy as np
import omegaconf
import random
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm

from cdvae.common.utils import PROJECT_ROOT
from cdvae.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume,
    frac_to_cart_coords, min_distance_sqr_pbc)
from cdvae.pl_modules.embeddings import MAX_ATOMIC_NUM
from cdvae.pl_modules.embeddings import KHOT_EMBEDDINGS

from cdvae.pl_modules.wyckoff_encoder import WyckoffEmbedding
from cdvae.pl_modules.wyckoff_decoder import WyckoffDecoder
from cdvae.pl_modules.wyckoff_loss import WyckoffReconLoss


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


def _linear_ramp(epoch, start_epoch, end_epoch, max_value):
    """Linearly ramp from zero to ``max_value`` over an epoch interval."""
    epoch = float(epoch)
    start_epoch = float(start_epoch)
    end_epoch = float(end_epoch)
    max_value = max(float(max_value), 0.0)
    if epoch <= start_epoch:
        return 0.0
    if end_epoch <= start_epoch:
        return max_value
    progress = min(max((epoch - start_epoch) / (end_epoch - start_epoch), 0.0), 1.0)
    return max_value * progress


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


class CrystGNN_Supervise(BaseModule):
    """
    GNN model for fitting the supervised objectives for crystals.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = hydra.utils.instantiate(self.hparams.encoder)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        preds = self.encoder(batch)  # shape (N, 1)
        return preds

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        preds = self(batch)

        loss = F.mse_loss(preds, batch.y)
        self.log_dict(
            {'train_loss': loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        preds = self(batch)

        log_dict, loss = self.compute_stats(batch, preds, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        preds = self(batch)

        log_dict, loss = self.compute_stats(batch, preds, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, batch, preds, prefix):
        loss = F.mse_loss(preds, batch.y)
        self.scaler.match_device(preds)
        scaled_preds = self.scaler.inverse_transform(preds)
        scaled_y = self.scaler.inverse_transform(batch.y)
        mae = torch.mean(torch.abs(scaled_preds - scaled_y))

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_mae': mae,
        }

        if self.hparams.data.prop == 'scaled_lattice':
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]
            if self.hparams.data.lattice_scale_method == 'scale_length':
                pred_lengths = pred_lengths * \
                    batch.num_atoms.view(-1, 1).float()**(1/3)
            lengths_mae = torch.mean(torch.abs(pred_lengths - batch.lengths))
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mard = mard(batch.angles, pred_angles)
            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(
                batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)
            log_dict.update({
                f'{prefix}_lengths_mae': lengths_mae,
                f'{prefix}_angles_mae': angles_mae,
                f'{prefix}_lengths_mard': lengths_mard,
                f'{prefix}_angles_mard': angles_mard,
                f'{prefix}_volumes_mard': volumes_mard,
            })
        return log_dict, loss

class WyckoffCDVAE(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = WyckoffEmbedding(
            hidden_dim=self.hparams.hidden_dim,
            latent_dim=self.hparams.latent_dim,
            max_sites=self.hparams.max_wyckoff_sites,
        )

        # decorder：WyckoffDecoder
        self.decoder = WyckoffDecoder(
            latent_dim=self.hparams.latent_dim,
            hidden_dim=self.hparams.hidden_dim,
            max_sites=self.hparams.max_wyckoff_sites,
            max_atoms=self.hparams.max_atoms,
            num_stability_classes=getattr(
                self.hparams, 'num_stability_classes', 4
            ),
            site_prior_logvar_min=getattr(self.hparams, 'site_prior_logvar_min', -10.0),
            site_prior_logvar_max=getattr(self.hparams, 'site_prior_logvar_max', 2.0),
            use_joint_site_planner=getattr(
                self.hparams, 'use_joint_site_planner', True
            ),
            site_planner_num_layers=getattr(
                self.hparams, 'site_planner_num_layers', 2
            ),
            site_planner_num_heads=getattr(
                self.hparams, 'site_planner_num_heads', 4
            ),
            site_planner_dropout=getattr(
                self.hparams, 'site_planner_dropout', 0.1
            ),
            iterative_site_refinement=getattr(
                self.hparams, 'iterative_site_refinement', True
            ),
            site_remask_power=getattr(
                self.hparams, 'site_remask_power', 1.0
            ),
        )

        # loss
        self.recon_loss = WyckoffReconLoss(
            w_spg=self.hparams.w_spg,
            w_lattice=self.hparams.w_lattice,
            w_elem=self.hparams.w_elem,
            w_letter=self.hparams.w_letter,
            w_free=self.hparams.w_free,
            w_nsites=getattr(self.hparams, 'w_nsites', 0.5),
            w_overlap=getattr(self.hparams, 'w_overlap', 0.0),
            w_charge=getattr(self.hparams, 'w_charge', 0.0),
            max_atoms=self.hparams.max_atoms,
            overlap_margin=getattr(self.hparams, 'overlap_margin', 0.05),
            overlap_tau=getattr(self.hparams, 'overlap_tau', 1.0),
            overlap_spg_topk=getattr(self.hparams, 'overlap_spg_topk', 3),
            overlap_elem_grad_scale=getattr(
                self.hparams, 'overlap_elem_grad_scale', 1.0
            ),
        )

        # diffusion步数
        self.T = 100

        # predict, 选择用
        if self.hparams.predict_property:
            self.fc_property = build_mlp(
                self.hparams.latent_dim,
                self.hparams.hidden_dim,
                self.hparams.fc_num_layers,
                1,
            )

    
    def reparameterize(self, mu, log_var):
        # 修复：防止log_var过大exp()溢出、NaN穿透clamp
        log_var = torch.nan_to_num(log_var, nan=-10.0, posinf=2.0, neginf=-10.0).clamp(-10.0, 2.0)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1e4, neginf=-1e4)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch):
        B = batch.num_wyk_sites.shape[0]
        device = batch.wyk_atom_types.device
        N = batch.wyk_atom_types.shape[0]  # 总位点数

        # 训练：随机 diffusion 时间步；validation/test：固定时间步，便于公平比较。
        if self.training:
            t = torch.randint(1, self.T + 1, (1,), device=device)
        else:
            eval_t = int(getattr(self.hparams, 'eval_diffusion_t', 50))
            eval_t = max(1, min(self.T, eval_t))
            t = torch.tensor([eval_t], device=device, dtype=torch.long)
        t_batch = t.expand(B)  # (B,) 同一个t广播给所有晶体
        mask_prob = t.float() / self.T  # 标量
        # Element 与 letter 使用独立 Bernoulli mask。两者仍共享同一个 t，
        # 但某个 site 的 element/letter 是否可见不再被强制绑定。
        elem_mask = torch.rand(N, device=device) < mask_prob  # (N,) flat mask
        letter_mask = torch.rand(N, device=device) < mask_prob  # (N,) flat mask

        # encoder：接收干净完整数据（真正的diffusion：噪声进decoder，不进encoder）
        mu, log_var, per_site_mu, per_site_log_var, enc_padding_mask = self.encoder(batch)
        z = self.reparameterize(mu, log_var)

        # 先获取targets（decoder需要noisy_elem_ids作为条件输入）
        targets, site_mask = self._prepare_targets(batch)

        # 将flat elem_mask转换为(B, max_sites)格式，用于loss加权和noisy输入
        S = self.decoder.max_sites
        masked_elem_sites = torch.zeros(B, S, dtype=torch.bool, device=device)
        masked_letter_sites = torch.zeros(B, S, dtype=torch.bool, device=device)
    

        num_sites_cpu = batch.num_wyk_sites.cpu()
        offset = 0
        for i in range(B):
            cnt = int(num_sites_cpu[i].item())
            n = min(cnt, S)
            if n > 0:
                masked_elem_sites[i, :n] = elem_mask[offset:offset + n]
                masked_letter_sites[i, :n] = letter_mask[offset:offset + n]
            offset += cnt

        # 构建noisy_elem_ids：masked位点置0（MASK token），其余保留原始元素ID
        # +1：0 保留给 MASK，元素占 1..100，与 noisy_elem_emb=Embedding(101)
        # 及 decode_to_wyckoff 生成端的 +1 对齐（否则训练/生成错位一格，
        # 且 elem_target=0 会与 MASK 撞车）
        noisy_elem_ids = targets['elem_target'] + 1  # (B, max_sites)
        noisy_elem_ids[masked_elem_sites] = 0  # 0 = MASK token

        # letter diffusion 使用自己的 mask，不再复用 element mask。
        noisy_letter_ids = targets['letter_target'] + 1  # (B, max_sites)
        noisy_letter_ids[masked_letter_sites] = 0  # 0 = MASK token

        # 对per-site latent reparameterize（仅valid sites）
        per_site_log_var = torch.nan_to_num(
            per_site_log_var, nan=-10.0, posinf=2.0, neginf=-10.0
        ).clamp(-10.0, 2.0)
        per_site_mu = torch.nan_to_num(per_site_mu, nan=0.0, posinf=1e4, neginf=-1e4)
        eps_site = torch.randn_like(per_site_mu)
        encoder_per_site_z = per_site_mu + eps_site * torch.exp(0.5 * per_site_log_var)

        # ── 联合CFG条件：元素意图 + Ehull稳定性类别 ──
        elem_cond = getattr(batch, 'elem_multihot', None)  # (B,100) 或 None
        if elem_cond is not None:
            elem_cond = elem_cond.view(-1, 100).to(device)

        stability_cond = None
        if getattr(self.hparams, 'use_stability_condition', False):
            if not hasattr(batch, 'stability_class'):
                raise AttributeError(
                    'use_stability_condition=true, but batch.stability_class is missing. '
                    'Use the modified dataset.py and an MP-20 CSV containing e_above_hull.'
                )
            stability_cond = batch.stability_class.view(-1).long().to(device)

        cfg_drop = None
        if self.training and (elem_cond is not None or stability_cond is not None):
            # 同一行同时丢弃元素和稳定性条件，训练真正的联合无条件分支。
            p_uncond = getattr(self.hparams, 'cfg_p_uncond', 0.15)
            cfg_drop = torch.rand(B, device=device) < p_uncond

        # 两条路径共享同一组 global predictions。
        target_spg_nums = targets['spg_target'] + 1
        target_n_sites = targets['num_sites_target'] + 1
        global_state = self.decoder.global_predictions(
            z,
            lattice_spg=target_spg_nums,
            stability_cond=stability_cond,
            cfg_drop=cfg_drop,
        )

        if self.decoder.use_joint_site_planner:
            predicted_spg_nums = global_state[1].detach().argmax(dim=-1) + 1
            predicted_n_sites = global_state[3].detach().argmax(dim=-1) + 1

            if self.training:
                planner_context_pred_prob = _linear_ramp(
                    getattr(self, 'current_epoch', 0),
                    getattr(self.hparams, 'planner_context_mix_start_epoch', 10),
                    getattr(self.hparams, 'planner_context_mix_end_epoch', 100),
                    getattr(self.hparams, 'planner_context_mix_max_prob', 0.5),
                )
                use_predicted_context = (
                    torch.rand(B, device=device) < planner_context_pred_prob
                )
            elif bool(getattr(
                self.hparams, 'planner_eval_use_predicted_context', True
            )):
                planner_context_pred_prob = 1.0
                use_predicted_context = torch.ones(
                    B, dtype=torch.bool, device=device
                )
            else:
                planner_context_pred_prob = 0.0
                use_predicted_context = torch.zeros(
                    B, dtype=torch.bool, device=device
                )

            planner_spg_nums = torch.where(
                use_predicted_context, predicted_spg_nums, target_spg_nums
            )
            planner_n_sites = torch.where(
                use_predicted_context, predicted_n_sites, target_n_sites
            ).clamp(1, S)
            planner_padding_mask = (
                torch.arange(S, device=device).unsqueeze(0)
                >= planner_n_sites.unsqueeze(1)
            )

            # Deterministic joint planner: learned site queries jointly attend to
            # global, element, stability, SPG and site-count context. During
            # training, predicted SPG/site count are introduced by a curriculum;
            # evaluation defaults to the generation-time predicted context.
            projector_per_site_z = self.decoder.plan_site_latents(
                z,
                elem_cond=elem_cond,
                stability_cond=stability_cond,
                cfg_drop=cfg_drop,
                spg_nums=planner_spg_nums,
                n_sites=planner_n_sites,
                site_padding_mask=planner_padding_mask,
                global_state=global_state,
            )
            site_prior_mu = None
            site_prior_log_var = None
        else:
            planner_context_pred_prob = 0.0
            use_predicted_context = torch.zeros(
                B, dtype=torch.bool, device=device
            )
            predicted_spg_nums = target_spg_nums
            predicted_n_sites = target_n_sites
            # Legacy factorized Gaussian prior, retained as an ablation switch.
            projector_per_site_z, site_prior_mu, site_prior_log_var = (
                self.decoder.sample_site_prior(
                    z,
                    stability_cond=stability_cond,
                    cfg_drop=cfg_drop,
                    detach_std_for_recon=(
                        self.training
                        and bool(getattr(
                            self.hparams,
                            'site_prior_detach_std_for_recon',
                            True,
                        ))
                    ),
                )
            )

        # decoder：noisy条件 + per-site latent z（双路径共享除per_site_z外的所有输入）
        preds_encoder = self.decoder(
            z, t=t_batch,
            noisy_elem_ids=noisy_elem_ids,
            noisy_letter_ids=noisy_letter_ids,
            per_site_z=encoder_per_site_z,
            enc_padding_mask=enc_padding_mask,
            site_padding_mask=enc_padding_mask,
            elem_cond=elem_cond,
            stability_cond=stability_cond,
            cfg_drop=cfg_drop,
            lattice_spg=target_spg_nums,
            global_state=global_state,
        )

        preds_projector = self.decoder(
            z, t=t_batch,
            noisy_elem_ids=noisy_elem_ids,
            noisy_letter_ids=noisy_letter_ids,
            per_site_z=projector_per_site_z,
            enc_padding_mask=enc_padding_mask,
            site_padding_mask=enc_padding_mask,
            elem_cond=elem_cond,
            stability_cond=stability_cond,
            cfg_drop=cfg_drop,
            lattice_spg=target_spg_nums,
            global_state=global_state,
        )

        # loss conduct
        recon_loss_encoder, loss_dict_encoder = self.recon_loss(
            preds_encoder,
            targets,
            site_mask,
            masked_elem_sites=masked_elem_sites,
            masked_letter_sites=masked_letter_sites,
        )
        recon_loss_projector, loss_dict_projector = self.recon_loss(
            preds_projector,
            targets,
            site_mask,
            masked_elem_sites=masked_elem_sites,
            masked_letter_sites=masked_letter_sites,
        )
        encoder_loss_weight = max(
            float(getattr(self.hparams, 'site_z_encoder_loss_weight', 0.3)), 0.0
        )
        projector_loss_weight = max(
            float(getattr(self.hparams, 'site_z_projector_loss_weight', 0.7)), 0.0
        )
        path_weight_sum = max(encoder_loss_weight + projector_loss_weight, 1e-8)
        encoder_loss_weight /= path_weight_sum
        projector_loss_weight /= path_weight_sum
        recon_loss = (
            encoder_loss_weight * recon_loss_encoder
            + projector_loss_weight * recon_loss_projector
        )
        loss_dict = {
            key: (
                encoder_loss_weight * loss_dict_encoder[key]
                + projector_loss_weight * loss_dict_projector[key]
            )
            for key in loss_dict_encoder
        }
        loss_dict.update({
            f'encoder_{key}': value for key, value in loss_dict_encoder.items()
        })
        loss_dict.update({
            f'projector_{key}': value for key, value in loss_dict_projector.items()
        })

        # Global KL
        kld_loss = self.kld_loss(mu, log_var)

        # Site alignment only uses valid sites. In planner mode the encoder mean
        # is a stop-gradient teacher; no Gaussian site KL is needed.
        site_valid = ~enc_padding_mask  # (B, max_sites), True=valid
        zero = mu.new_zeros(())
        if self.decoder.use_joint_site_planner:
            planner_alignment = self.site_planner_alignment(
                projector_per_site_z,
                per_site_mu,
                site_valid,
            )
            kld_site = zero
            site_prior_metrics = {}
        else:
            planner_alignment = {
                'mse': zero,
                'cosine_loss': zero,
                'scale_loss': zero,
                'planner_std': zero,
                'posterior_std': zero,
                'std_ratio': zero,
            }
            kld_site = self.conditional_site_kld(
                per_site_mu,
                per_site_log_var,
                site_prior_mu,
                site_prior_log_var,
                site_valid,
                detach_posterior=bool(getattr(
                    self.hparams, 'site_prior_detach_posterior', True
                )),
            )
            site_prior_metrics = self.site_prior_diagnostics(
                per_site_mu,
                per_site_log_var,
                site_prior_mu,
                site_prior_log_var,
                site_valid,
            )

        if self.hparams.predict_property and hasattr(batch, 'y'):
            property_loss = F.mse_loss(self.fc_property(z).squeeze(-1), batch.y)
        else:
            property_loss = torch.tensor(0., device=mu.device)

        # sm_90 修复：kld_loss/kld_site/property_loss 是 loss_dict 里仅剩的
        # 未经处理的原始 GPU tensor（仍带 grad_fn）。异步 CUDA 执行在 sm_90 上
        # 用 nan_to_num 立即处理：既强制同步，又清除真实的 NaN/Inf，
        # 且保留 grad_fn（不 detach），不影响反向传播。
        kld_loss = torch.nan_to_num(kld_loss, nan=0.0, posinf=0.0, neginf=0.0)
        kld_site = torch.nan_to_num(kld_site, nan=0.0, posinf=0.0, neginf=0.0)
        property_loss = torch.nan_to_num(property_loss, nan=0.0, posinf=0.0, neginf=0.0)

        if self.decoder.use_joint_site_planner:
            site_kl_scale = 0.0
            planner_mse_weight = max(float(getattr(
                self.hparams, 'site_planner_mse_weight', 0.1
            )), 0.0)
            planner_cosine_weight = max(float(getattr(
                self.hparams, 'site_planner_cosine_weight', 0.1
            )), 0.0)
            planner_scale_weight = max(float(getattr(
                self.hparams, 'site_planner_scale_weight', 0.05
            )), 0.0)
            site_regularization = (
                planner_mse_weight * planner_alignment['mse']
                + planner_cosine_weight * planner_alignment['cosine_loss']
                + planner_scale_weight * planner_alignment['scale_loss']
            )
        else:
            # Legacy conditional site-KL warm-up, used only for ablation.
            site_kl_scale_start = float(getattr(
                self.hparams,
                'site_prior_kl_scale_start',
                getattr(self.hparams, 'site_prior_kl_scale', 0.1),
            ))
            site_kl_scale_max = float(getattr(
                self.hparams,
                'site_prior_kl_scale_max',
                getattr(self.hparams, 'site_prior_kl_scale', 0.1),
            ))
            site_kl_warmup_epochs = max(int(getattr(
                self.hparams, 'site_prior_kl_warmup_epochs', 0
            )), 0)
            if site_kl_warmup_epochs > 0:
                site_kl_progress = min(
                    max(float(getattr(self, 'current_epoch', 0)), 0.0)
                    / float(site_kl_warmup_epochs),
                    1.0,
                )
                site_kl_scale = (
                    site_kl_scale_start
                    + (site_kl_scale_max - site_kl_scale_start) * site_kl_progress
                )
            else:
                site_kl_scale = site_kl_scale_max
            planner_mse_weight = 0.0
            planner_cosine_weight = 0.0
            planner_scale_weight = 0.0
            site_regularization = self.hparams.beta * site_kl_scale * kld_site

        # total loss
        regularization_loss = (
            self.hparams.beta * kld_loss
            + site_regularization
            + self.hparams.cost_property * property_loss
        )
        total_loss = (
            recon_loss
            + regularization_loss
        )
        encoder_total_loss = (
            recon_loss_encoder
            + regularization_loss
        )
        projector_total_loss = (
            recon_loss_projector
            + self.hparams.beta * kld_loss
            + site_regularization
            + self.hparams.cost_property * property_loss
        )

        loss_dict.update({
            'encoder_recon_loss': recon_loss_encoder,
            'projector_recon_loss': recon_loss_projector,
            'combined_recon_loss': recon_loss,
            'encoder_total_loss': encoder_total_loss,
            'projector_total_loss': projector_total_loss,
            'kld_loss': kld_loss,
            'kld_site': kld_site,
            'kld_site_conditional': kld_site,
            'site_prior_kl_scale': site_kl_scale,
            'site_prior_kl_weight': float(self.hparams.beta) * site_kl_scale,
            'site_planner_mse': planner_alignment['mse'],
            'site_planner_cosine_loss': planner_alignment['cosine_loss'],
            'site_planner_cosine_similarity': (
                1.0 - planner_alignment['cosine_loss']
            ),
            'site_planner_std': planner_alignment['planner_std'],
            'site_planner_posterior_std': planner_alignment['posterior_std'],
            'site_planner_std_ratio': planner_alignment['std_ratio'],
            'site_planner_scale_loss': planner_alignment['scale_loss'],
            'site_planner_mse_weight': planner_mse_weight,
            'site_planner_cosine_weight': planner_cosine_weight,
            'site_planner_scale_weight': planner_scale_weight,
            'site_planner_regularization': site_regularization,
            'planner_context_pred_prob': planner_context_pred_prob,
            'planner_context_pred_fraction': (
                use_predicted_context.float().mean()
            ),
            'planner_context_spg_accuracy': (
                predicted_spg_nums == target_spg_nums
            ).float().mean(),
            'planner_context_nsites_accuracy': (
                predicted_n_sites == target_n_sites
            ).float().mean(),
            'property_loss': property_loss,
        })
        loss_dict.update(site_prior_metrics)
        return total_loss, loss_dict

    def _prepare_targets(self, batch):
        B = batch.num_wyk_sites.shape[0]
        S = self.decoder.max_sites
        device = batch.num_wyk_sites.device

        num_wyk_sites_cpu = (
            batch.num_wyk_sites.cpu().view(-1)
            .nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            .clamp(0, 100).long()
        )
        batch.num_wyk_sites = num_wyk_sites_cpu.to(device)

        # spg_idx: 0-indexed整数(0..229)，NaN/越界→0
        spg_idx_cpu = (
            batch.spg_idx.cpu().view(-1)
            .nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            .clamp(0, 229).long()
        )
        batch.spg_idx = spg_idx_cpu.to(device)

        # wyk_lattice: (N_total, 6) 或 (B, 6) 浮点，直接nan_to_num后reshape
        # 这是 test_loss_lattice=nan 的直接根因：H200/sm_90 上 batch.wyk_lattice
        # 未经保护直接 .view(B,6) 传入 F.mse_loss，NaN原样污染 loss_lattice。
        lattice_cpu = (
            batch.wyk_lattice.cpu()
            .nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            .view(B, 6)
        )

        # wyk_atom_types / wyk_letters / wyk_free: flat张量，CPU清洗后在循环里切片
        atom_types_cpu = (
            batch.wyk_atom_types.cpu()
            .nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            .clamp(0, 1000).long()          # 元素ID上界宽松，clamp防越界即可
        )
        letters_cpu = (
            batch.wyk_letters.cpu()
            .nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            .clamp(0, 51).long()            # Wyckoff letter: 0..51(a..Z)
        )
        free_cpu = (
            batch.wyk_free.cpu()
            .nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        )
        # ──────────────────────────────────────────────────────────────────────

        elem_target   = torch.zeros(B, S, dtype=torch.long, device=device)
        letter_target = torch.zeros(B, S, dtype=torch.long, device=device)
        free_target   = torch.zeros(B, S, 3,                device=device)
        multi_target  = torch.ones(B,  S,                   device=device)  # 默认1（不加权）
        site_mask     = torch.zeros(B, S, dtype=torch.bool, device=device)

        # multiplicity：尝试从 batch.wyk_multi 读取，不存在则保持默认值1
        # （wyk_multi 由 dataset.py 从 encode_wyckoff_tensors 的 multiplicities 字段存入）
        multi_cpu = None
        for attr in ['wyk_multi', 'multiplicities']:
            if hasattr(batch, attr):
                try:
                    multi_cpu = getattr(batch, attr).cpu().float()
                    multi_cpu = multi_cpu.nan_to_num(nan=1.0, posinf=1.0, neginf=1.0)
                    break
                except Exception:
                    pass

        # site_batch_idx(repeat_interleave产出的单调不减序列)在flat张量中每个
        # 结构i对应连续一段——用offset连续切片(与wyckoff_encoder.py一致，已验证
        # 可用)，避免tensor[bool_mask][:n]在训练模式下触发"Expected !is_symbolic()"。
        offset = 0
        for i in range(B):
            cnt = int(num_wyk_sites_cpu[i].item())
            n   = min(cnt, S)
            if n > 0:
                elem_target[i,   :n]    = (atom_types_cpu[offset:offset + n] - 1).to(device)
                letter_target[i, :n]    = letters_cpu[offset:offset + n].to(device)
                free_target[i,   :n, :] = free_cpu[offset:offset + n].to(device)
                if multi_cpu is not None:
                    multi_target[i, :n] = multi_cpu[offset:offset + n].to(device)
                site_mask[i,     :n]    = True
            offset += cnt

        # 同上：用CPU上的num_wyk_sites_cpu算，避免GPU上.clamp()链路问题，再.to(device)。
        num_sites_target = (num_wyk_sites_cpu - 1).clamp(0, S - 1).to(device)

        targets = {
            'spg_target':       spg_idx_cpu.to(device),
            'lattice_target':   lattice_cpu.to(device),
            'num_sites_target': num_sites_target,
            'elem_target':      elem_target,
            'letter_target':    letter_target,
            'free_target':      free_target,
            'multiplicities':   multi_target,   # charge penalty 使用
        }
        return targets, site_mask


    def kld_loss(self, mu, log_var):
        # clamp 防止 exp() 溢出：log_var > 88 时 exp(log_var) > float32 上限
        # → +inf → inf - inf = nan → val_loss=nan → EarlyStopping 误杀训练
        log_var = torch.nan_to_num(log_var, nan=-10.0, posinf=2.0, neginf=-10.0).clamp(-10.0, 2.0)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1e4, neginf=-1e4)
        return torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)
        )

    @staticmethod
    def site_planner_alignment(planned_site_z, posterior_mu, site_valid):
        """Match the generation-time planner to encoder site means.

        The posterior is a stop-gradient teacher. Both objectives are averaged
        over valid sites only, so structures with fewer sites are not penalized
        by padded slots.
        """
        if planned_site_z.shape != posterior_mu.shape:
            raise ValueError(
                'Joint Site Planner output and encoder per-site mean must have '
                f'the same shape, got {tuple(planned_site_z.shape)} and '
                f'{tuple(posterior_mu.shape)}. Set latent_dim == hidden_dim.'
            )
        planned_site_z = torch.nan_to_num(
            planned_site_z, nan=0.0, posinf=1e4, neginf=-1e4
        )
        posterior_mu = torch.nan_to_num(
            posterior_mu.detach(), nan=0.0, posinf=1e4, neginf=-1e4
        )
        mask = site_valid.unsqueeze(-1).to(planned_site_z.dtype)
        site_count = mask.sum().clamp(min=1.0)
        dim_count = site_count * float(planned_site_z.shape[-1])

        mse = ((planned_site_z - posterior_mu).pow(2) * mask).sum() / dim_count
        cosine_distance = 1.0 - F.cosine_similarity(
            planned_site_z, posterior_mu, dim=-1, eps=1e-8
        )
        cosine_loss = (
            cosine_distance * site_valid.to(cosine_distance.dtype)
        ).sum() / site_count

        # Align relative latent scale per crystal. A log-standard-deviation
        # objective penalizes ratio mismatch more usefully than absolute error.
        dims_per_crystal = (
            mask.sum(dim=(1, 2)) * float(planned_site_z.shape[-1])
        ).clamp(min=1.0)
        planner_mean = (
            planned_site_z * mask
        ).sum(dim=(1, 2)) / dims_per_crystal
        posterior_mean = (
            posterior_mu * mask
        ).sum(dim=(1, 2)) / dims_per_crystal
        planner_var = (
            (planned_site_z - planner_mean[:, None, None]).pow(2) * mask
        ).sum(dim=(1, 2)) / dims_per_crystal
        posterior_var = (
            (posterior_mu - posterior_mean[:, None, None]).pow(2) * mask
        ).sum(dim=(1, 2)) / dims_per_crystal
        planner_std_per_crystal = planner_var.clamp(min=1e-8).sqrt()
        posterior_std_per_crystal = posterior_var.clamp(min=1e-8).sqrt()
        valid_crystal = site_valid.any(dim=1).to(planned_site_z.dtype)
        valid_crystal_count = valid_crystal.sum().clamp(min=1.0)
        log_std_delta = (
            torch.log(planner_std_per_crystal.clamp(min=1e-4))
            - torch.log(posterior_std_per_crystal.clamp(min=1e-4))
        )
        scale_loss = (
            log_std_delta.pow(2) * valid_crystal
        ).sum() / valid_crystal_count
        planner_std = (
            planner_std_per_crystal * valid_crystal
        ).sum() / valid_crystal_count
        posterior_std = (
            posterior_std_per_crystal * valid_crystal
        ).sum() / valid_crystal_count
        std_ratio = (
            planner_std_per_crystal
            / posterior_std_per_crystal.clamp(min=1e-4)
            * valid_crystal
        ).sum() / valid_crystal_count

        return {
            'mse': mse,
            'cosine_loss': cosine_loss,
            'scale_loss': scale_loss,
            'planner_std': planner_std,
            'posterior_std': posterior_std,
            'std_ratio': std_ratio,
        }

    @staticmethod
    def conditional_site_kld(
        posterior_mu,
        posterior_log_var,
        prior_mu,
        prior_log_var,
        site_valid,
        detach_posterior=True,
    ):
        # Encoder posterior is the teacher. Prevent a weak prior from pulling
        # the better reconstruction path toward itself; gradients still update
        # the prior and the global latent feeding it.
        if detach_posterior:
            posterior_mu = posterior_mu.detach()
            posterior_log_var = posterior_log_var.detach()
        posterior_mu = torch.nan_to_num(
            posterior_mu, nan=0.0, posinf=1e4, neginf=-1e4
        )
        prior_mu = torch.nan_to_num(
            prior_mu, nan=0.0, posinf=1e4, neginf=-1e4
        )
        posterior_log_var = torch.nan_to_num(
            posterior_log_var, nan=-10.0, posinf=2.0, neginf=-10.0
        ).clamp(-10.0, 2.0)
        prior_log_var = torch.nan_to_num(
            prior_log_var, nan=-10.0, posinf=2.0, neginf=-10.0
        ).clamp(-10.0, 2.0)
        kld = 0.5 * (
            prior_log_var
            - posterior_log_var
            + (
                posterior_log_var.exp()
                + (posterior_mu - prior_mu).pow(2)
            ) / prior_log_var.exp().clamp(min=1e-8)
            - 1.0
        )
        site_valid = site_valid.unsqueeze(-1).to(kld.dtype)
        return (kld * site_valid).sum() / site_valid.sum().clamp(min=1.0)

    @staticmethod
    def site_prior_diagnostics(
        posterior_mu,
        posterior_log_var,
        prior_mu,
        prior_log_var,
        site_valid,
    ):
        """Return valid-site-only posterior/prior alignment diagnostics."""
        with torch.no_grad():
            posterior_log_var = torch.nan_to_num(
                posterior_log_var, nan=-10.0, posinf=2.0, neginf=-10.0
            ).clamp(-10.0, 2.0)
            prior_log_var = torch.nan_to_num(
                prior_log_var, nan=-10.0, posinf=2.0, neginf=-10.0
            ).clamp(-10.0, 2.0)
            posterior_mu = torch.nan_to_num(
                posterior_mu, nan=0.0, posinf=1e4, neginf=-1e4
            )
            prior_mu = torch.nan_to_num(
                prior_mu, nan=0.0, posinf=1e4, neginf=-1e4
            )

            mask = site_valid.unsqueeze(-1).to(prior_mu.dtype)
            site_count = mask.sum().clamp(min=1.0)
            dim_count = site_count * float(prior_mu.shape[-1])
            posterior_std_tensor = torch.exp(0.5 * posterior_log_var)
            prior_std_tensor = torch.exp(0.5 * prior_log_var)
            posterior_std = (posterior_std_tensor * mask).sum() / dim_count
            prior_std = (prior_std_tensor * mask).sum() / dim_count

            mean_sq = (posterior_mu - prior_mu).pow(2)
            prior_var = prior_log_var.exp().clamp(min=1e-8)
            variance_term = 0.5 * (
                prior_log_var
                - posterior_log_var
                + posterior_log_var.exp() / prior_var
                - 1.0
            )
            mean_term = 0.5 * mean_sq / prior_var

            return {
                'site_posterior_std': posterior_std,
                'site_prior_std': prior_std,
                'site_prior_posterior_std_ratio': (
                    prior_std / posterior_std.clamp(min=1e-8)
                ),
                'site_prior_mu_mse': (mean_sq * mask).sum() / dim_count,
                'site_prior_logvar_mae': (
                    (posterior_log_var - prior_log_var).abs() * mask
                ).sum() / dim_count,
                # Per-site sums over latent dimensions; together they should
                # approximately equal kld_site_conditional.
                'site_kl_mean_component': (
                    mean_term * mask
                ).sum() / site_count,
                'site_kl_variance_component': (
                    variance_term * mask
                ).sum() / site_count,
            }

   
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        total_loss, loss_dict = self(batch)
        # ── [CONFIG] 打印真正生效的权重（读self.recon_loss内部实际存的值，
        # 不是self.hparams表面值，避免getattr默认值/Hydra读取问题被掩盖）──
        if batch_idx == 0:
            print(f"[CONFIG] w_spg={self.recon_loss.w_spg} "
                  f"w_lattice={self.recon_loss.w_lattice} "
                  f"w_elem={self.recon_loss.w_elem} "
                  f"w_letter={self.recon_loss.w_letter} "
                  f"w_free={self.recon_loss.w_free} "
                  f"w_nsites={self.recon_loss.w_nsites} "
                  f"w_overlap={self.recon_loss.w_overlap} "
                  f"w_charge={self.recon_loss.w_charge} "
                  f"beta={self.hparams.beta}", flush=True)
        # ── [临时诊断] 打印前5个batch的完整loss_dict，定位nan第一次出现 ──
        # 注意: loss_dict里的大部分项已经是.item()过的Python float
        # (见WyckoffReconLoss.forward的return)，但kld_loss/kld_site/
        # property_loss是forward()第274-278行后加进去的、带grad_fn的
        # GPU tensor。直接对它们做repr()(f-string会调用)在本机
        # H200/sm_90上会触发torch._tensor_str的masked_select整数溢出
        # bug(RuntimeError: numel: integer multiplication overflow)，
        # 与loss本身是否nan无关。这里统一转成Python标量再打印。
        if batch_idx < 5:
            debug_dict = {k: (v.item() if torch.is_tensor(v) else v) for k, v in loss_dict.items()}
            print(f"[DEBUG step{batch_idx}] total_loss={total_loss.item()} "
                  f"loss_dict={debug_dict}", flush=True)
        # 最终保障：NaN/Inf 进入 total_loss 会用 NaN 梯度污染所有权重，
        # 且会导致 val_loss 长期失真（被后续环节的 nan_to_num 压成 0），
        # EarlyStopping 误判"已收敛"而静默提前停止（之前踩过这个坑，
        # 但该保护此前未真正部署到 training_step，这次补上）。
        # nan_to_num 保留 grad_fn（不 detach），NaN/Inf 处梯度自动为 0，
        # PL 仍能正常调用 loss.backward()。
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            # 打印是哪个 loss 项导致的，方便定位根因（不受 batch_idx<5 限制，
            # 只要出现 NaN 就打印，这样能抓到训练中期偶发的 NaN）
            bad_keys = {k: v for k, v in loss_dict.items()
                        if (torch.is_tensor(v) and (torch.isnan(v).any() or torch.isinf(v).any()))
                        or (isinstance(v, float) and (v != v or abs(v) == float('inf')))}
            full_dict = {k: (v.item() if torch.is_tensor(v) else v) for k, v in loss_dict.items()}
            print(f"[NaN警告][train step{batch_idx}] total_loss={total_loss.item()} "
                  f"→ 已被nan_to_num替换为0。异常来源字段: {list(bad_keys.keys())}", flush=True)
            print(f"  完整loss_dict: {full_dict}", flush=True)
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
        # detach带grad_fn的tensor条目(kld_loss/kld_site/property_loss)，
        # 否则 on_epoch=True 的聚合会保留整张计算图，
        # 大CPU中间张量)直到epoch结束，导致CPU内存累积OOM。
        log_dict = {f'train_{k}': (v.detach() if torch.is_tensor(v) else v) for k, v in loss_dict.items()}
        log_dict['train_loss'] = total_loss.detach()
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        total_loss, loss_dict = self(batch)
        projector_total_loss = loss_dict.get('projector_total_loss', total_loss)
        # 同样加保障 + 诊断打印：val_loss 若为 NaN 会导致 EarlyStopping 行为异常
        if (torch.isnan(total_loss) or torch.isinf(total_loss)
                or torch.isnan(projector_total_loss) or torch.isinf(projector_total_loss)):
            bad_keys = {k: v for k, v in loss_dict.items()
                        if (torch.is_tensor(v) and (torch.isnan(v).any() or torch.isinf(v).any()))
                        or (isinstance(v, float) and (v != v or abs(v) == float('inf')))}
            print(f"[NaN警告][val step{batch_idx}] total_loss={total_loss.item()} "
                  f"projector_total_loss={projector_total_loss.item()} "
                  f"→ 已被nan_to_num替换为0。异常来源字段: {list(bad_keys.keys())}", flush=True)
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
        projector_total_loss = torch.nan_to_num(
            projector_total_loss, nan=0.0, posinf=0.0, neginf=0.0
        )
        log_dict = {f'val_{k}': (v.detach() if torch.is_tensor(v) else v) for k, v in loss_dict.items()}
        log_dict['val_combined_loss'] = total_loss.detach()
        log_dict['val_loss'] = projector_total_loss.detach()
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)
        return projector_total_loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        total_loss, loss_dict = self(batch)
        projector_total_loss = loss_dict.get('projector_total_loss', total_loss)
        # 与 training_step / validation_step 保持一致：total_loss 也做 NaN 防护
        # 之前缺了这一层，导致 checkpoint 权重退化时 test 结果直接暴露裸 nan
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
        projector_total_loss = torch.nan_to_num(
            projector_total_loss, nan=0.0, posinf=0.0, neginf=0.0
        )
        log_dict = {f'test_{k}': (v.item() if torch.is_tensor(v) else v)
                    for k, v in loss_dict.items()}
        log_dict['test_combined_loss'] = total_loss.item()
        log_dict['test_loss'] = projector_total_loss.item()
        self.log_dict(log_dict)
        return projector_total_loss

    # Generate
   
    @torch.no_grad()
    def generate(
        self, num_samples: int = 10, elem_cond=None,
        stability_cond=None, cfg_w=0.0,
    ):
        z = torch.randn(
            num_samples, self.hparams.latent_dim, device=self.device
        )
        wyckoff_list = self.decoder.decode_to_wyckoff(
            z,
            elem_cond=elem_cond,
            stability_cond=stability_cond,
            cfg_w=cfg_w,
        )

        from cdvae.pl_data.wyckoff_utils import wyckoff_to_structure
        structures = []
        for w in wyckoff_list:
            try:
                struct = wyckoff_to_structure(
                    w['spacegroup_num'],
                    w['site_elements'],
                    w['site_letters'],
                    w['site_free_params'],
                    w['lattice_params'],
                    max_atoms=self.hparams.max_atoms,
                    distance_margin=self.recon_loss.overlap_margin,
                )
                structures.append(struct)
            except Exception as e:
                print(f'Structure generation failed: {e}')
                structures.append(None)
        return structures

class CDVAE(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = hydra.utils.instantiate(
            self.hparams.encoder, num_targets=self.hparams.latent_dim)
        self.decoder = hydra.utils.instantiate(self.hparams.decoder)

        self.fc_mu = nn.Linear(self.hparams.latent_dim,
                               self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.latent_dim,
                                self.hparams.latent_dim)

        self.fc_num_atoms = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                      self.hparams.fc_num_layers, self.hparams.max_atoms+1)
        self.fc_lattice = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                    self.hparams.fc_num_layers, 6)
        self.fc_composition = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                        self.hparams.fc_num_layers, MAX_ATOMIC_NUM)
        # for property prediction.
        if self.hparams.predict_property:
            self.fc_property = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                         self.hparams.fc_num_layers, 1)

        sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.sigma_begin),
            np.log(self.hparams.sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

        type_sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.type_sigma_begin),
            np.log(self.hparams.type_sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.type_sigmas = nn.Parameter(type_sigmas, requires_grad=False)

        self.embedding = torch.zeros(100, 92)
        for i in range(100):
            self.embedding[i] = torch.tensor(KHOT_EMBEDDINGS[i + 1])

        # obtain from datamodule.
        self.lattice_scaler = None
        self.scaler = None

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, batch):
        """
        encode crystal structures to latents.
        """
        hidden = self.encoder(batch)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z

    def decode_stats(self, z, gt_num_atoms=None, gt_lengths=None, gt_angles=None,
                     teacher_forcing=False):
        """
        decode key stats from latent embeddings.
        batch is input during training for teach-forcing.
        """
        if gt_num_atoms is not None:
            num_atoms = self.predict_num_atoms(z)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, gt_num_atoms))
            composition_per_atom = self.predict_composition(z, gt_num_atoms)
            if self.hparams.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
        else:
            num_atoms = self.predict_num_atoms(z).argmax(dim=-1)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, num_atoms))
            composition_per_atom = self.predict_composition(z, num_atoms)
        return num_atoms, lengths_and_angles, lengths, angles, composition_per_atom

    @torch.no_grad()
    def langevin_dynamics(self, z, ld_kwargs, gt_num_atoms=None, gt_atom_types=None):
        """
        decode crystral structure from latent embeddings.
        ld_kwargs: args for doing annealed langevin dynamics sampling:
            n_step_each:  number of steps for each sigma level.
            step_lr:      step size param.
            min_sigma:    minimum sigma to use in annealed langevin dynamics.
            save_traj:    if <True>, save the entire LD trajectory.
            disable_bar:  disable the progress bar of langevin dynamics.
        gt_num_atoms: if not <None>, use the ground truth number of atoms.
        gt_atom_types: if not <None>, use the ground truth atom types.
        """
        if ld_kwargs.save_traj:
            all_frac_coords = []
            all_pred_cart_coord_diff = []
            all_noise_cart = []
            all_atom_types = []

        # obtain key stats.
        num_atoms, _, lengths, angles, composition_per_atom = self.decode_stats(
            z, gt_num_atoms)
        if gt_num_atoms is not None:
            num_atoms = gt_num_atoms

        # obtain atom types.
        composition_per_atom = F.softmax(composition_per_atom, dim=-1)
        if gt_atom_types is None:
            cur_atom_types = self.sample_composition(
                composition_per_atom, num_atoms)
        else:
            cur_atom_types = gt_atom_types

        # init coords.
        cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=z.device)

        # annealed langevin dynamics.
        for sigma in tqdm(self.sigmas, total=self.sigmas.size(0), disable=ld_kwargs.disable_bar):
            if sigma < ld_kwargs.min_sigma:
                break
            step_size = ld_kwargs.step_lr * (sigma / self.sigmas[-1]) ** 2

            for step in range(ld_kwargs.n_step_each):
                noise_cart = torch.randn_like(
                    cur_frac_coords) * torch.sqrt(step_size * 2)
                pred_cart_coord_diff, pred_atom_types = self.decoder(
                    z, cur_frac_coords, cur_atom_types, num_atoms, lengths, angles)
                cur_cart_coords = frac_to_cart_coords(
                    cur_frac_coords, lengths, angles, num_atoms)
                pred_cart_coord_diff = pred_cart_coord_diff / sigma
                cur_cart_coords = cur_cart_coords + step_size * pred_cart_coord_diff + noise_cart
                cur_frac_coords = cart_to_frac_coords(
                    cur_cart_coords, lengths, angles, num_atoms)

                if gt_atom_types is None:
                    cur_atom_types = torch.argmax(pred_atom_types, dim=1) + 1

                if ld_kwargs.save_traj:
                    all_frac_coords.append(cur_frac_coords)
                    all_pred_cart_coord_diff.append(
                        step_size * pred_cart_coord_diff)
                    all_noise_cart.append(noise_cart)
                    all_atom_types.append(cur_atom_types)

        output_dict = {'num_atoms': num_atoms, 'lengths': lengths, 'angles': angles,
                       'frac_coords': cur_frac_coords, 'atom_types': cur_atom_types,
                       'is_traj': False}

        if ld_kwargs.save_traj:
            output_dict.update(dict(
                all_frac_coords=torch.stack(all_frac_coords, dim=0),
                all_atom_types=torch.stack(all_atom_types, dim=0),
                all_pred_cart_coord_diff=torch.stack(
                    all_pred_cart_coord_diff, dim=0),
                all_noise_cart=torch.stack(all_noise_cart, dim=0),
                is_traj=True))

        return output_dict

    def sample(self, num_samples, ld_kwargs):
        z = torch.randn(num_samples, self.hparams.hidden_dim,
                        device=self.device)
        samples = self.langevin_dynamics(z, ld_kwargs)
        return samples

    def forward(self, batch, teacher_forcing, training):
        # hacky way to resolve the NaN issue. Will need more careful debugging later.
        mu, log_var, z = self.encode(batch)

        (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles,
         pred_composition_per_atom) = self.decode_stats(
            z, batch.num_atoms, batch.lengths, batch.angles, teacher_forcing)

        # sample noise levels.
        noise_level = torch.randint(0, self.sigmas.size(0),
                                    (batch.num_atoms.size(0),),
                                    device=self.device)
        used_sigmas_per_atom = self.sigmas[noise_level].repeat_interleave(
            batch.num_atoms, dim=0)

        type_noise_level = torch.randint(0, self.type_sigmas.size(0),
                                         (batch.num_atoms.size(0),),
                                         device=self.device)
        used_type_sigmas_per_atom = (
            self.type_sigmas[type_noise_level].repeat_interleave(
                batch.num_atoms, dim=0))

        # add noise to atom types and sample atom types.
        pred_composition_probs = F.softmax(
            pred_composition_per_atom.detach(), dim=-1)
        atom_type_probs = (
            F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM) +
            pred_composition_probs * used_type_sigmas_per_atom[:, None])
        rand_atom_types = torch.multinomial(
            atom_type_probs, num_samples=1).squeeze(1) + 1

        # add noise to the cart coords
        cart_noises_per_atom = (
            torch.randn_like(batch.frac_coords) *
            used_sigmas_per_atom[:, None])
        cart_coords = frac_to_cart_coords(
            batch.frac_coords, pred_lengths, pred_angles, batch.num_atoms)
        cart_coords = cart_coords + cart_noises_per_atom
        noisy_frac_coords = cart_to_frac_coords(
            cart_coords, pred_lengths, pred_angles, batch.num_atoms)

        pred_cart_coord_diff, pred_atom_types = self.decoder(
            z, noisy_frac_coords, rand_atom_types, batch.num_atoms, pred_lengths, pred_angles)

        # compute loss.
        num_atom_loss = self.num_atom_loss(pred_num_atoms, batch)
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        composition_loss = self.composition_loss(
            pred_composition_per_atom, batch.atom_types, batch)
        coord_loss = self.coord_loss(
            pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch)
        type_loss = self.type_loss(pred_atom_types, batch.atom_types,
                                   used_type_sigmas_per_atom, batch)

        kld_loss = self.kld_loss(mu, log_var)

        if self.hparams.predict_property:
            property_loss = self.property_loss(z, batch)
        else:
            property_loss = 0.

        return {
            'num_atom_loss': num_atom_loss,
            'lattice_loss': lattice_loss,
            'composition_loss': composition_loss,
            'coord_loss': coord_loss,
            'type_loss': type_loss,
            'kld_loss': kld_loss,
            'property_loss': property_loss,
            'pred_num_atoms': pred_num_atoms,
            'pred_lengths_and_angles': pred_lengths_and_angles,
            'pred_lengths': pred_lengths,
            'pred_angles': pred_angles,
            'pred_cart_coord_diff': pred_cart_coord_diff,
            'pred_atom_types': pred_atom_types,
            'pred_composition_per_atom': pred_composition_per_atom,
            'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,
            'rand_frac_coords': noisy_frac_coords,
            'rand_atom_types': rand_atom_types,
            'z': z,
        }

    def generate_rand_init(self, pred_composition_per_atom, pred_lengths,
                           pred_angles, num_atoms, batch):
        rand_frac_coords = torch.rand(num_atoms.sum(), 3,
                                      device=num_atoms.device)
        pred_composition_per_atom = F.softmax(pred_composition_per_atom,
                                              dim=-1)
        rand_atom_types = self.sample_composition(
            pred_composition_per_atom, num_atoms)
        return rand_frac_coords, rand_atom_types

    def sample_composition(self, composition_prob, num_atoms):
        """
        Samples composition such that it exactly satisfies composition_prob
        """
        batch = torch.arange(
            len(num_atoms), device=num_atoms.device).repeat_interleave(num_atoms)
        assert composition_prob.size(0) == num_atoms.sum() == batch.size(0)
        composition_prob = scatter(
            composition_prob, index=batch, dim=0, reduce='mean')

        all_sampled_comp = []

        for comp_prob, num_atom in zip(list(composition_prob), list(num_atoms)):
            comp_num = torch.round(comp_prob * num_atom)
            atom_type = torch.nonzero(comp_num, as_tuple=True)[0] + 1
            atom_num = comp_num[atom_type - 1].long()

            sampled_comp = atom_type.repeat_interleave(atom_num, dim=0)

            # if the rounded composition gives less atoms, sample the rest
            if sampled_comp.size(0) < num_atom:
                left_atom_num = num_atom - sampled_comp.size(0)

                left_comp_prob = comp_prob - comp_num.float() / num_atom

                left_comp_prob[left_comp_prob < 0.] = 0.
                left_comp = torch.multinomial(
                    left_comp_prob, num_samples=left_atom_num, replacement=True)
                # convert to atomic number
                left_comp = left_comp + 1
                sampled_comp = torch.cat([sampled_comp, left_comp], dim=0)

            sampled_comp = sampled_comp[torch.randperm(sampled_comp.size(0))]
            sampled_comp = sampled_comp[:num_atom]
            all_sampled_comp.append(sampled_comp)

        all_sampled_comp = torch.cat(all_sampled_comp, dim=0)
        assert all_sampled_comp.size(0) == num_atoms.sum()
        return all_sampled_comp

    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)

    def predict_property(self, z):
        self.scaler.match_device(z)
        return self.scaler.inverse_transform(self.fc_property(z))

    def predict_lattice(self, z, num_atoms):
        self.lattice_scaler.match_device(z)
        pred_lengths_and_angles = self.fc_lattice(z)  # (N, 6)
        scaled_preds = self.lattice_scaler.inverse_transform(
            pred_lengths_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        if self.hparams.data.lattice_scale_method == 'scale_length':
            pred_lengths = pred_lengths * num_atoms.view(-1, 1).float()**(1/3)
        # <pred_lengths_and_angles> is scaled.
        return pred_lengths_and_angles, pred_lengths, pred_angles

    def predict_composition(self, z, num_atoms):
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom

    def num_atom_loss(self, pred_num_atoms, batch):
        return F.cross_entropy(pred_num_atoms, batch.num_atoms)

    def property_loss(self, z, batch):
        return F.mse_loss(self.fc_property(z), batch.y)

    def lattice_loss(self, pred_lengths_and_angles, batch):
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        if self.hparams.data.lattice_scale_method == 'scale_length':
            target_lengths = batch.lengths / \
                batch.num_atoms.view(-1, 1).float()**(1/3)
        target_lengths_and_angles = torch.cat(
            [target_lengths, batch.angles], dim=-1)
        target_lengths_and_angles = self.lattice_scaler.transform(
            target_lengths_and_angles)
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_composition_per_atom,
                               target_atom_types, reduction='none')
        return scatter(loss, batch.batch, reduce='mean').mean()

    def coord_loss(self, pred_cart_coord_diff, noisy_frac_coords,
                   used_sigmas_per_atom, batch):
        noisy_cart_coords = frac_to_cart_coords(
            noisy_frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        target_cart_coords = frac_to_cart_coords(
            batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        _, target_cart_coord_diff = min_distance_sqr_pbc(
            target_cart_coords, noisy_cart_coords, batch.lengths, batch.angles,
            batch.num_atoms, self.device, return_vector=True)

        target_cart_coord_diff = target_cart_coord_diff / \
            used_sigmas_per_atom[:, None]**2
        pred_cart_coord_diff = pred_cart_coord_diff / \
            used_sigmas_per_atom[:, None]

        loss_per_atom = torch.sum(
            (target_cart_coord_diff - pred_cart_coord_diff)**2, dim=1)

        loss_per_atom = 0.5 * loss_per_atom * used_sigmas_per_atom**2
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()

    def type_loss(self, pred_atom_types, target_atom_types,
                  used_type_sigmas_per_atom, batch):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(
            pred_atom_types, target_atom_types, reduction='none')
        # rescale loss according to noise
        loss = loss / used_type_sigmas_per_atom
        return scatter(loss, batch.batch, reduce='mean').mean()

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        return kld_loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        teacher_forcing = (
            self.current_epoch <= self.hparams.teacher_forcing_max_epoch)
        outputs = self(batch, teacher_forcing, training=True)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, batch, outputs, prefix):
        num_atom_loss = outputs['num_atom_loss']
        lattice_loss = outputs['lattice_loss']
        coord_loss = outputs['coord_loss']
        type_loss = outputs['type_loss']
        kld_loss = outputs['kld_loss']
        composition_loss = outputs['composition_loss']
        property_loss = outputs['property_loss']

        loss = (
            self.hparams.cost_natom * num_atom_loss +
            self.hparams.cost_lattice * lattice_loss +
            self.hparams.cost_coord * coord_loss +
            self.hparams.cost_type * type_loss +
            self.hparams.beta * kld_loss +
            self.hparams.cost_composition * composition_loss +
            self.hparams.cost_property * property_loss)

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_natom_loss': num_atom_loss,
            f'{prefix}_lattice_loss': lattice_loss,
            f'{prefix}_coord_loss': coord_loss,
            f'{prefix}_type_loss': type_loss,
            f'{prefix}_kld_loss': kld_loss,
            f'{prefix}_composition_loss': composition_loss,
        }

        if prefix != 'train':
            # validation/test loss only has coord and type
            loss = (
                self.hparams.cost_coord * coord_loss +
                self.hparams.cost_type * type_loss)

            # evaluate num_atom prediction.
            pred_num_atoms = outputs['pred_num_atoms'].argmax(dim=-1)
            num_atom_accuracy = (
                pred_num_atoms == batch.num_atoms).sum() / batch.num_graphs

            # evalute lattice prediction.
            pred_lengths_and_angles = outputs['pred_lengths_and_angles']
            scaled_preds = self.lattice_scaler.inverse_transform(
                pred_lengths_and_angles)
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]

            if self.hparams.data.lattice_scale_method == 'scale_length':
                pred_lengths = pred_lengths * \
                    batch.num_atoms.view(-1, 1).float()**(1/3)
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))

            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(
                batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)

            # evaluate atom type prediction.
            pred_atom_types = outputs['pred_atom_types']
            target_atom_types = outputs['target_atom_types']
            type_accuracy = pred_atom_types.argmax(
                dim=-1) == (target_atom_types - 1)
            type_accuracy = scatter(type_accuracy.float(
            ), batch.batch, dim=0, reduce='mean').mean()

            log_dict.update({
                f'{prefix}_loss': loss,
                f'{prefix}_property_loss': property_loss,
                f'{prefix}_natom_accuracy': num_atom_accuracy,
                f'{prefix}_lengths_mard': lengths_mard,
                f'{prefix}_angles_mae': angles_mae,
                f'{prefix}_volumes_mard': volumes_mard,
                f'{prefix}_type_accuracy': type_accuracy,
            })

        return log_dict, loss


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )
    return model


if __name__ == "__main__":
    main()
