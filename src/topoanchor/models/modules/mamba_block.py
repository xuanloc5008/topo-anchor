from __future__ import annotations

import torch
import torch.nn as nn

from topoanchor.models.modules.conv_blocks import ResConvBlock3D
from topoanchor.utils.imports import MissingDependencyError, require_package


def _load_mamba_class():
    module = require_package("mamba_ssm", "pip install mamba-ssm")
    if hasattr(module, "Mamba"):
        return module.Mamba
    simple = require_package("mamba_ssm.modules.mamba_simple", "pip install mamba-ssm")
    if hasattr(simple, "Mamba"):
        return simple.Mamba
    raise MissingDependencyError("Installed mamba_ssm does not expose a Mamba class.")


class ResMambaBlock3D(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        mamba_class = _load_mamba_class()
        self.norm = nn.LayerNorm(channels)
        self.mamba = mamba_class(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.post = ResConvBlock3D(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        tokens = self.mamba(self.norm(tokens))
        y = tokens.transpose(1, 2).reshape(batch, channels, depth, height, width)
        return self.post(x + y)


class ConvSequenceFallbackBlock3D(nn.Module):
    """Explicit local fallback for CPU/macOS smoke tests, not the paper model."""

    def __init__(self, channels: int, **_: int) -> None:
        super().__init__()
        self.block = ResConvBlock3D(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def build_sequence_block_3d(
    channels: int,
    *,
    backend: str,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
) -> nn.Module:
    backend = str(backend).lower()
    if backend == "mamba_ssm":
        return ResMambaBlock3D(channels, d_state=d_state, d_conv=d_conv, expand=expand)
    if backend in {"conv", "conv_fallback"}:
        return ConvSequenceFallbackBlock3D(channels)
    raise ValueError(f"Unsupported sequence backend `{backend}`. Use `mamba_ssm` or `conv_fallback`.")
