from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra
import torch
from omegaconf import DictConfig

from topoanchor.data.manifest import assert_disjoint_sample_ids
from topoanchor.utils.seed import seed_everything


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    from lightning import Trainer
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    from topoanchor.data.datamodule import ManifestDataModule
    from topoanchor.training.lightning_module import TopologyAnchoredLightningModule

    assert_disjoint_sample_ids([cfg.data.train_manifest, cfg.data.val_manifest, cfg.data.test_manifest])
    seed_everything(int(cfg.seed), deterministic=bool(cfg.train.deterministic))
    torch.set_float32_matmul_precision(str(cfg.train.matmul_precision))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.log_dir).mkdir(parents=True, exist_ok=True)

    datamodule = ManifestDataModule(cfg)
    model = TopologyAnchoredLightningModule(cfg)
    if bool(cfg.train.compile.enabled):
        model.model = torch.compile(model.model, mode=str(cfg.train.compile.mode))
    checkpoint = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor="val/dice",
        mode="max",
        save_top_k=3,
        save_last=True,
        filename="epoch={epoch:03d}",
        auto_insert_metric_name=False,
    )
    trainer = Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=str(cfg.trainer.accelerator),
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        fast_dev_run=bool(cfg.trainer.fast_dev_run),
        deterministic=bool(cfg.trainer.deterministic),
        gradient_clip_val=float(cfg.train.gradient_clip_val),
        accumulate_grad_batches=int(cfg.train.accumulate_grad_batches),
        log_every_n_steps=int(cfg.train.log_every_n_steps),
        val_check_interval=cfg.trainer.val_check_interval,
        num_sanity_val_steps=int(cfg.trainer.num_sanity_val_steps),
        logger=CSVLogger(save_dir=str(cfg.paths.log_dir), name=str(cfg.project_name)),
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
