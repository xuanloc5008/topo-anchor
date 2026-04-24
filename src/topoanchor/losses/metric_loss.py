from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveTopologyLoss(nn.Module):
    def __init__(self, *, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = float(temperature)

    def forward(
        self,
        z_topo: torch.Tensor,
        positive_mask: torch.Tensor,
        valid_anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_topo.ndim != 2:
            raise ValueError(f"z_topo must be [B, D], got {tuple(z_topo.shape)}.")
        if positive_mask.shape != (z_topo.shape[0], z_topo.shape[0]):
            raise ValueError("positive_mask must have shape [B, B].")

        batch_size = z_topo.shape[0]
        if batch_size < 2:
            return z_topo.sum() * 0.0
        positive_mask = positive_mask.to(device=z_topo.device, dtype=torch.bool)
        logits = torch.matmul(F.normalize(z_topo, dim=1), F.normalize(z_topo, dim=1).T)
        logits = logits / self.temperature
        self_mask = torch.eye(batch_size, device=z_topo.device, dtype=torch.bool)
        logits = logits.masked_fill(self_mask, float("-inf"))

        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positives_per_anchor = positive_mask.sum(dim=1)
        anchors_with_positive = positives_per_anchor > 0
        if valid_anchor_mask is not None:
            anchors_with_positive = anchors_with_positive & valid_anchor_mask.to(
                device=z_topo.device, dtype=torch.bool
            )
        if not anchors_with_positive.any():
            return z_topo.sum() * 0.0

        positive_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob)).sum(
            dim=1
        ) / positives_per_anchor.clamp_min(1)
        return -positive_log_prob[anchors_with_positive].mean()
