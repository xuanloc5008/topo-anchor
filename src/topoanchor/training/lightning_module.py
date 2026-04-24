from __future__ import annotations

import torch

from topoanchor.evaluation.calibration_metrics import calibrated_confidence, raw_segmentation_confidence
from topoanchor.evaluation.segmentation_metrics import multiclass_dice_from_logits
from topoanchor.losses import TopologyAnchoredLoss
from topoanchor.models.topology_anchored_model import TopologyAnchoredSegmentationModel
from topoanchor.utils.imports import require_package


class TopologyAnchoredLightningModule(
    require_package("lightning", "pip install lightning").LightningModule
):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = TopologyAnchoredSegmentationModel(cfg)
        self.loss_fn = TopologyAnchoredLoss(cfg)
        self.save_hyperparameters(ignore=["cfg"])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(x)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        output = self(batch["image"])
        losses = self.loss_fn(output, batch)
        dice = multiclass_dice_from_logits(
            output["logits"],
            batch["mask"].to(output["logits"].device),
            include_background=bool(self.cfg.loss.seg.include_background),
        )
        raw_conf = raw_segmentation_confidence(output["prob"]).mean()
        cal_conf = calibrated_confidence(
            raw_segmentation_confidence(output["prob"]),
            output["mahalanobis"],
            gamma=float(self.cfg.eval.calibration_gamma),
        ).mean()
        log_values = {
            f"{stage}/loss": losses["loss"],
            f"{stage}/loss_seg": losses["loss_seg"],
            f"{stage}/loss_metric": losses["loss_metric"],
            f"{stage}/loss_dist": losses["loss_dist"],
            f"{stage}/dice": dice,
            f"{stage}/mahalanobis": output["mahalanobis"].mean(),
            f"{stage}/raw_confidence": raw_conf,
            f"{stage}/calibrated_confidence": cal_conf,
        }
        self.log_dict(log_values, prog_bar=stage != "train", on_step=stage == "train", on_epoch=True)
        return losses["loss"]

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer_name = str(self.cfg.train.optimizer.name).lower()
        if optimizer_name != "adamw":
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.cfg.train.optimizer.lr),
            weight_decay=float(self.cfg.train.optimizer.weight_decay),
        )
        scheduler_name = str(self.cfg.train.scheduler.name).lower()
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(int(self.cfg.trainer.max_epochs), 1),
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer
