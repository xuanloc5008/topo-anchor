from __future__ import annotations

import torch
import torch.nn as nn

from topoanchor.models.modules.conv_blocks import DownBlock3D, ResConvBlock3D
from topoanchor.models.modules.mamba_block import build_sequence_block_3d


class NNUNetResMambaEncoder3D(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], mamba_cfg: dict) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("Expected exactly four channel stages for E1-E4.")
        c1, c2, c3, c4 = channels
        self.e1 = ResConvBlock3D(in_channels, c1)
        self.e2 = DownBlock3D(c1, c2)
        self.e3 = DownBlock3D(c2, c3)
        self.e4 = DownBlock3D(c3, c4)
        self.bottleneck_mamba = build_sequence_block_3d(c4, **mamba_cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        h = self.bottleneck_mamba(e4)
        return h, [e1, e2, e3]
