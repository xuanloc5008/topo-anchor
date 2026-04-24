from __future__ import annotations

import torch
import torch.nn as nn

from topoanchor.losses.anchor_distribution_loss import AnchorDistributionLoss
from topoanchor.losses.metric_loss import SupervisedContrastiveTopologyLoss
from topoanchor.losses.segmentation_losses import DiceCrossEntropyLoss
from topoanchor.topology.pair_builder import build_pair_masks_from_vectors


class TopologyAnchoredLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.lambda_metric = float(cfg.loss.lambda_metric)
        self.lambda_dist = float(cfg.loss.lambda_dist)
        self.seg = DiceCrossEntropyLoss(
            dice_weight=float(cfg.loss.seg.dice_weight),
            ce_weight=float(cfg.loss.seg.ce_weight),
            include_background=bool(cfg.loss.seg.include_background),
        )
        self.metric = SupervisedContrastiveTopologyLoss(
            temperature=float(cfg.loss.metric.temperature)
        )
        self.dist = AnchorDistributionLoss(epsilon=float(cfg.loss.dist.epsilon))
        self.positive_distance = float(cfg.loss.metric.positive_distance)
        self.negative_distance = float(cfg.loss.metric.negative_distance)

    def forward(self, model_output: dict[str, torch.Tensor], batch: dict) -> dict[str, torch.Tensor]:
        logits = model_output["logits"]
        target = batch["mask"].to(device=logits.device)
        seg_loss = self.seg(logits, target)
        dist_loss = self.dist(model_output["z_topo"], model_output["mu"], model_output["var"])

        if self.lambda_metric > 0:
            topology_vectors = batch["topology_vector"].to(device=logits.device)
            positive_mask, _ = build_pair_masks_from_vectors(
                topology_vectors,
                positive_distance=self.positive_distance,
                negative_distance=self.negative_distance,
            )
            metric_loss = self.metric(model_output["z_topo"], positive_mask)
        else:
            metric_loss = model_output["z_topo"].sum() * 0.0
        total = seg_loss + self.lambda_metric * metric_loss + self.lambda_dist * dist_loss
        return {
            "loss": total,
            "loss_seg": seg_loss.detach(),
            "loss_metric": metric_loss.detach(),
            "loss_dist": dist_loss.detach(),
        }
