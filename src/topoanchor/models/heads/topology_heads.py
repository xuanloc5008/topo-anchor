from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PooledProjectionHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool3d(h, output_size=1).flatten(1)
        max_ = F.adaptive_max_pool3d(h, output_size=1).flatten(1)
        return self.mlp(torch.cat([avg, max_], dim=1))
