from __future__ import annotations

import torch
import torch.nn as nn


class AnchorTokenGenerator(nn.Module):
    def __init__(self, *, topo_dim: int, token_dim: int) -> None:
        super().__init__()
        self.topo_token = nn.Linear(topo_dim, token_dim)
        self.mu_token = nn.Linear(topo_dim, token_dim)
        self.var_token = nn.Linear(topo_dim, token_dim)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, z_topo: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack(
            [
                self.topo_token(z_topo),
                self.mu_token(mu),
                self.var_token(torch.log(var.clamp_min(1e-8))),
            ],
            dim=1,
        )
        return self.norm(tokens)
