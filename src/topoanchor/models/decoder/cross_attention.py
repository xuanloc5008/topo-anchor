from __future__ import annotations

import torch
import torch.nn as nn


class D3AnchorCrossAttention(nn.Module):
    def __init__(self, *, feature_channels: int, token_dim: int, num_heads: int) -> None:
        super().__init__()
        self.token_proj = nn.Linear(token_dim, feature_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feature_channels)

    def forward(self, feature: torch.Tensor, anchor_tokens: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = feature.shape
        query = feature.flatten(2).transpose(1, 2).contiguous()
        query_norm = self.norm(query)
        key_value = self.token_proj(anchor_tokens)
        attended, _ = self.attn(query=query_norm, key=key_value, value=key_value, need_weights=False)
        out = query + attended
        return out.transpose(1, 2).reshape(batch, channels, depth, height, width)
