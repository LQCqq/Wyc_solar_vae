from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from cdvae.common.utils import log_hyperparameters, PROJECT_ROOT


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if "lr_monitor" in cfg.logging and cfg.logging.get("wandb", {}).get("mode", "online") != "disabled":
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(HydraConfig.get().run.dir),
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
            )
        )

    return callbacks


def run(cfg: DictConfig) -> None:
    """
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /conf
    """
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run=}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0

        # Switch wandb mode to offline to prevent online logging
        cfg.logging.wandb.mode = "offline"

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    # Pass scaler from datamodule to model
    hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.scaler}>")
    model.lattice_scaler = datamodule.lattice_scaler.copy()
    model.scaler = datamodule.scaler.copy()
    torch.save(datamodule.lattice_scaler, hydra_dir / 'lattice_scaler.pt')
    torch.save(datamodule.scaler, hydra_dir / 'prop_scaler.pt')
    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # Logger instantiation/configuration
    #wandb_logger = None
    #if "wandb" in cfg.logging:
    #    hydra.utils.log.info("Instantiating <WandbLogger>")
    #    wandb_config = cfg.logging.wandb
    #    wandb_logger = WandbLogger(
    #        **wandb_config,
    #        tags=cfg.core.tags,
    #    )
    #    hydra.utils.log.info("W&B is now watching <{cfg.logging.wandb_watch.log}>!")
    #    wandb_logger.watch(
    #        model,
    #        log=cfg.logging.wandb_watch.log,
    #        log_freq=cfg.logging.wandb_watch.log_freq,
    #    )

    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    # Load checkpoint (if exist)
    ckpts = list(hydra_dir.glob('*.ckpt'))
    if len(ckpts) > 0:
        ckpt_epochs = np.array([int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
        ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        hydra.utils.log.info(f"found checkpoint: {ckpt}")
    else:
        ckpt = None

    # ── 热启动开关 ──────────────────────────────────────────────────────────
    # 背景：resume_from_checkpoint 会把 Adam 优化器的动量/二阶矩估计也一起恢复。
    # 这个统计量是按"旧loss权重(比如w_overlap=0.1)"训练时的梯度量级校准的。
    # 如果这次训练改了 w_overlap/w_charge 这类loss权重，loss landscape会突变，
    # 但Adam还在用旧统计量算自适应学习率 → 第一步更新就可能把权重推向极端值
    # (实测表现为 property_loss 变成 -1.75e32 这种不可能的数字，随后全面NaN)。
    # 修改了loss权重、只想复用旧权重当初始点时，用这个模式：只加载模型权重，
    # 不带旧优化器状态，Adam从零状态重新校准，避免上述发散。
    # 用法：命令行加 train.weights_only_resume=true，或在 conf/train/default.yaml
    # 里加 weights_only_resume: true。默认false，不改变原有的完整resume行为。
    weights_only = bool(getattr(cfg.train, 'weights_only_resume', False))
    resume_ckpt_path = None

    if ckpt is not None and weights_only:
        hydra.utils.log.info(f"[热启动] 仅加载模型权重，不恢复优化器状态: {ckpt}")
        state = torch.load(ckpt, map_location='cpu')
        missing, unexpected = model.load_state_dict(state['state_dict'], strict=False)
        if missing:
            hydra.utils.log.info(f"  未匹配到checkpoint的权重(将保持随机初始化): {missing}")
        if unexpected:
            hydra.utils.log.info(f"  checkpoint中多出的权重(已忽略): {unexpected}")
        resume_ckpt_path = None  # 关键：不传给Trainer，优化器/调度器/epoch计数全部从零开始
    elif ckpt is not None:
        hydra.utils.log.info(f"[完整resume] 恢复权重+优化器状态+epoch计数: {ckpt}")
        resume_ckpt_path = ckpt
    else:
        hydra.utils.log.info("未找到checkpoint，从头开始训练")

    hydra.utils.log.info("Instantiating the Trainer")
    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        #logger=wandb_logger,
                logger = False,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        #progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
        resume_from_checkpoint=resume_ckpt_path,
        **cfg.train.pl_trainer,
    )
    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    hydra.utils.log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    hydra.utils.log.info("Starting testing!")
    trainer.test(datamodule=datamodule)

    # Logger closing to release resources/avoid multi-run conflicts
    # if wandb_logger is not None:
    #     wandb_logger.experiment.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()