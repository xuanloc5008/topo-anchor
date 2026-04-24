from __future__ import annotations

import torch
import torch.nn as nn

from topoanchor.models.decoder.cross_attention import D3AnchorCrossAttention
from topoanchor.models.modules.conv_blocks import UpBlock3D
from topoanchor.models.modules.mamba_block import build_sequence_block_3d


class AnchorConditionedDecoder3D(nn.Module):
    def __init__(
        self,
        *,
        channels: list[int],
        num_classes: int,
        token_dim: int,
        attention_heads: int,
        mamba_cfg: dict,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.up3 = UpBlock3D(c4, c3, c3)
        self.d3_mamba = build_sequence_block_3d(c3, **mamba_cfg)
        self.d3_attention = D3AnchorCrossAttention(
            feature_channels=c3,
            token_dim=token_dim,
            num_heads=attention_heads,
        )
        self.up2 = UpBlock3D(c3, c2, c2)
        self.up1 = UpBlock3D(c2, c1, c1)
        self.out = nn.Conv3d(c1, num_classes, kernel_size=1)

    def forward(
        self,
        h: torch.Tensor,
        skips: list[torch.Tensor],
        anchor_tokens: torch.Tensor,
    ) -> torch.Tensor:
        e1, e2, e3 = skips
        d3 = self.up3(h, e3)
        d3 = self.d3_mamba(d3)
        d3 = self.d3_attention(d3, anchor_tokens)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.out(d1)
