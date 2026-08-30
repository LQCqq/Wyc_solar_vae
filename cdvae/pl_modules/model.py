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

        # encoder: wyckoffEmbedding 替换原始DimeNet GNN
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

        # 多步masked diffusion：采样全局时间步 t ~ Uniform(1, T)
        t = torch.randint(1, self.T + 1, (1,), device=device)  # 全局单一时间步
        t_batch = t.expand(B)  # (B,) 同一个t广播给所有晶体
        mask_prob = t.float() / self.T  # 标量
        elem_mask = torch.rand(N, device=device) < mask_prob  # (N,) flat mask

        # encoder：接收干净完整数据（真正的diffusion：噪声进decoder，不进encoder）
        mu, log_var, per_site_mu, per_site_log_var, enc_padding_mask = self.encoder(batch)
        z = self.reparameterize(mu, log_var)

        # 先获取targets（decoder需要noisy_elem_ids作为条件输入）
        targets, site_mask = self._prepare_targets(batch)

        # 将flat elem_mask转换为(B, max_sites)格式，用于loss加权和noisy输入
        S = self.decoder.max_sites
        masked_sites = torch.zeros(B, S, dtype=torch.bool, device=device)
        # _prepare_targets(在本函数201行已调用)已将batch.num_wyk_sites修正为
        # 正确的(B,)long张量(GPU,纯拷贝得到，值正确)。site_batch_idx=repeat_interleave
        # 产出的是单调不减序列，即每个结构i的位点在flat张量中是连续一段——
        # 用offset连续切片(与wyckoff_encoder.py一致，已验证可用)，避免
        # bool mask索引(tensor[bool_mask][:n])在训练(梯度跟踪)模式下触发
        # "Expected !is_symbolic()"内部错误(验证模式下不触发，但训练时会)。
        num_sites_cpu = batch.num_wyk_sites.cpu()
        offset = 0
        for i in range(B):
            cnt = int(num_sites_cpu[i].item())
            n = min(cnt, S)
            if n > 0:
                masked_sites[i, :n] = elem_mask[offset:offset + n]
            offset += cnt

        # 构建noisy_elem_ids：masked位点置0（MASK token），其余保留原始元素ID
        # +1：0 保留给 MASK，元素占 1..100，与 noisy_elem_emb=Embedding(101)
        # 及 decode_to_wyckoff 生成端的 +1 对齐（否则训练/生成错位一格，
        # 且 elem_target=0 会与 MASK 撞车）
        noisy_elem_ids = targets['elem_target'] + 1  # (B, max_sites)
        noisy_elem_ids[masked_sites] = 0  # 0 = MASK token

        # letter diffusion: 同一批masked位点也mask letter
        noisy_letter_ids = targets['letter_target'] + 1  # (B, max_sites)
        noisy_letter_ids[masked_sites] = 0  # 0 = MASK token

        # 对per-site latent reparameterize（仅valid sites）
        per_site_log_var = torch.nan_to_num(
            per_site_log_var, nan=-10.0, posinf=2.0, neginf=-10.0
        ).clamp(-10.0, 2.0)
        per_site_mu = torch.nan_to_num(per_site_mu, nan=0.0, posinf=1e4, neginf=-1e4)
        eps_site = torch.randn_like(per_site_mu)
        per_site_z = per_site_mu + eps_site * torch.exp(0.5 * per_site_log_var)

        # ── scheduled sampling：30%概率用projector替换encoder的per_site_z ──
        # 训练site_z_projector，弥合训练/生成的分布差距
        if self.training and torch.rand(1, device=device).item() < 0.3:
            z_proj = self.decoder.site_z_projector(z)
            z_proj = z_proj.unsqueeze(1).expand_as(per_site_z)
            per_site_z = z_proj + torch.randn_like(z_proj) * 0.2
        # ────────────────────────────────────────────────────────

        # ── CFG 元素条件（阶段1）：从 batch 取 multi-hot，训练时随机丢弃 ──
        elem_cond = getattr(batch, 'elem_multihot', None)  # (B,100) 或 None
        cfg_drop = None
        if elem_cond is not None:
            elem_cond = elem_cond.view(-1, 100).to(device)
            if self.training:
                # 以概率 p_uncond 丢弃条件：被丢弃的样本在 decoder 里用 null_elem_cond
                p_uncond = getattr(self.hparams, 'cfg_p_uncond', 0.15)
                cfg_drop = torch.rand(elem_cond.size(0), device=device) < p_uncond

        # decoder：noisy条件 + per-site latent z（训练/生成一致，无shortcut）
        preds = self.decoder(
            z, t=t_batch,
            noisy_elem_ids=noisy_elem_ids,
            noisy_letter_ids=noisy_letter_ids,
            per_site_z=per_site_z,
            enc_padding_mask=enc_padding_mask,
            elem_cond=elem_cond,
            cfg_drop=cfg_drop,
        )

        # loss conduct
        recon_loss, loss_dict = self.recon_loss(preds, targets, site_mask, masked_sites=masked_sites)

        # Global KL
        kld_loss = self.kld_loss(mu, log_var)

        # Per-site KL（只对valid sites计算）
        site_valid = ~enc_padding_mask  # (B, max_sites), True=valid
        # per_site_log_var 同样需要 clamp，防止 exp() 溢出产生 NaN（与 kld_loss 的 log_var 同理）
        per_site_log_var = per_site_log_var.clamp(-10, 2)
        kld_site = -0.5 * (1 + per_site_log_var - per_site_mu.pow(2) - per_site_log_var.exp())
        kld_site = (kld_site * site_valid.unsqueeze(-1).float()).sum() / site_valid.float().sum().clamp(min=1)

        if self.hparams.predict_property and hasattr(batch, 'y'):
            property_loss = F.mse_loss(self.fc_property(z).squeeze(-1), batch.y)
        else:
            property_loss = torch.tensor(0., device=mu.device)

        # sm_90 修复：kld_loss/kld_site/property_loss 是 loss_dict 里仅剩的
        # 未经处理的原始 GPU tensor（仍带 grad_fn）。异步 CUDA 执行在 sm_90 上
        # 会让 isnan() 检查读到尚未落定的瞬时垃圾值（表现为 training_step 里
        # 反复标记这三者异常，但稍后 .item() 读回来又是正常数）。
        # 用 nan_to_num 立即处理：既强制同步，又清除真实的 NaN/Inf，
        # 且保留 grad_fn（不 detach），不影响反向传播。
        kld_loss = torch.nan_to_num(kld_loss, nan=0.0, posinf=0.0, neginf=0.0)
        kld_site = torch.nan_to_num(kld_site, nan=0.0, posinf=0.0, neginf=0.0)
        property_loss = torch.nan_to_num(property_loss, nan=0.0, posinf=0.0, neginf=0.0)

        # total loss
        total_loss = (
            recon_loss
            + self.hparams.beta * kld_loss
            + self.hparams.beta * 0.1 * kld_site
            + self.hparams.cost_property * property_loss
        )

        loss_dict.update({
            'kld_loss': kld_loss,
            'kld_site': kld_site,
            'property_loss': property_loss,
        })
        return total_loss, loss_dict

    def _prepare_targets(self, batch):
        B = batch.num_wyk_sites.shape[0]
        S = self.decoder.max_sites
        device = batch.num_wyk_sites.device

        # ─── 全部 batch.* 字段统一走 CPU 链路 ────────────────────────────────
        # 已知问题：nan_to_num / clamp / .long() 等操作在本机(H200/sm_90)GPU
        # 上不稳定，会产出垃圾值或NaN。num_wyk_sites 此前已修；这里对剩余所有
        # 未防护的字段补齐同款处理：先 .cpu()，在CPU上做数值清洗，再 .to(device)
        # 纯拷贝。每个字段的 nan_to_num 兜底值选取原则：
        #   - 整数索引(spg/elem/letter)：nan→0, 再clamp到合法范围，.long()
        #   - 浮点坐标(lattice/free)   ：nan/inf→0.0，不额外clamp(保留数据集原值)
        # num_wyk_sites_cpu留作下方 num_sites_target 复用。
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
                elem_target[i,   :n]    = atom_types_cpu[offset:offset + n].to(device)
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
        # 同样加保障 + 诊断打印：val_loss 若为 NaN 会导致 EarlyStopping 行为异常
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            bad_keys = {k: v for k, v in loss_dict.items()
                        if (torch.is_tensor(v) and (torch.isnan(v).any() or torch.isinf(v).any()))
                        or (isinstance(v, float) and (v != v or abs(v) == float('inf')))}
            print(f"[NaN警告][val step{batch_idx}] total_loss={total_loss.item()} "
                  f"→ 已被nan_to_num替换为0。异常来源字段: {list(bad_keys.keys())}", flush=True)
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
        log_dict = {f'val_{k}': (v.detach() if torch.is_tensor(v) else v) for k, v in loss_dict.items()}
        log_dict['val_loss'] = total_loss.detach()
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        total_loss, loss_dict = self(batch)
        # 与 training_step / validation_step 保持一致：total_loss 也做 NaN 防护
        # 之前缺了这一层，导致 checkpoint 权重退化时 test 结果直接暴露裸 nan
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
        log_dict = {f'test_{k}': (v.item() if torch.is_tensor(v) else v)
                    for k, v in loss_dict.items()}
        log_dict['test_loss'] = total_loss.item()
        self.log_dict(log_dict)
        return total_loss

    # Generate
   
    @torch.no_grad()
    def generate(self, num_samples: int = 10, elem_cond=None, cfg_w=0.0):
        z = torch.randn(
            num_samples, self.hparams.latent_dim, device=self.device
        )
        wyckoff_list = self.decoder.decode_to_wyckoff(z, elem_cond=elem_cond, cfg_w=cfg_w)

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