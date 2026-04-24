from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorDistributionHead(nn.Module):
    def __init__(
        self,
        *,
        topo_dim: int,
        app_dim: int,
        hidden_dim: int,
        eps_sigma: float,
    ) -> None:
        super().__init__()
        self.eps_sigma = float(eps_sigma)
        self.mu_head = nn.Sequential(
            nn.Linear(topo_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, topo_dim),
        )
        self.var_head = nn.Sequential(
            nn.Linear(topo_dim + app_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, topo_dim),
        )

    def forward(self, z_topo: torch.Tensor, c_app: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_detached = z_topo.detach()
        mu = self.mu_head(z_detached)
        rho = self.var_head(torch.cat([z_detached, c_app], dim=1))
        var = F.softplus(rho) + self.eps_sigma
        return mu, var
