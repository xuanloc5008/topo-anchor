from __future__ import annotations

import torch
import torch.nn as nn


class AnchorDistributionLoss(nn.Module):
    def __init__(self, *, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, z_topo: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        if z_topo.shape != mu.shape or z_topo.shape != var.shape:
            raise ValueError(
                f"z_topo, mu, and var must share shape; got {z_topo.shape}, {mu.shape}, {var.shape}."
            )
        stable_var = var.clamp_min(self.epsilon)
        per_dim = ((z_topo - mu).pow(2) / stable_var) + torch.log(stable_var)
        return 0.5 * per_dim.sum(dim=1).mean()


def mahalanobis_distance(z_topo: torch.Tensor, mu: torch.Tensor, var: torch.Tensor, *, epsilon: float) -> torch.Tensor:
    stable_var = var.clamp_min(epsilon)
    return ((z_topo - mu).pow(2) / stable_var).sum(dim=1)
