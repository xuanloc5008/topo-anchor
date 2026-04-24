from __future__ import annotations

import torch
import torch.nn as nn

from topoanchor.losses.anchor_distribution_loss import mahalanobis_distance
from topoanchor.models.anchors.anchor_distribution import AnchorDistributionHead
from topoanchor.models.anchors.token_generator import AnchorTokenGenerator
from topoanchor.models.backbones.nnunet_resmamba import NNUNetResMambaEncoder3D
from topoanchor.models.decoder.nnunet_decoder import AnchorConditionedDecoder3D
from topoanchor.models.heads.topology_heads import PooledProjectionHead


class TopologyAnchoredSegmentationModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        model_cfg = cfg.model
        channels = [int(value) for value in model_cfg.channels]
        mamba_cfg = {
            "backend": str(model_cfg.mamba.backend),
            "d_state": int(model_cfg.mamba.d_state),
            "d_conv": int(model_cfg.mamba.d_conv),
            "expand": int(model_cfg.mamba.expand),
        }
        bottleneck_channels = channels[-1]
        self.encoder = NNUNetResMambaEncoder3D(
            in_channels=int(model_cfg.in_channels),
            channels=channels,
            mamba_cfg=mamba_cfg,
        )
        self.topology_head = PooledProjectionHead(
            bottleneck_channels,
            hidden_dim=bottleneck_channels,
            out_dim=int(model_cfg.topo_dim),
            dropout=float(model_cfg.dropout),
        )
        self.context_head = PooledProjectionHead(
            bottleneck_channels,
            hidden_dim=bottleneck_channels,
            out_dim=int(model_cfg.app_dim),
            dropout=float(model_cfg.dropout),
        )
        self.anchor_distribution = AnchorDistributionHead(
            topo_dim=int(model_cfg.topo_dim),
            app_dim=int(model_cfg.app_dim),
            hidden_dim=bottleneck_channels,
            eps_sigma=float(model_cfg.anchor.eps_sigma),
        )
        self.token_generator = AnchorTokenGenerator(
            topo_dim=int(model_cfg.topo_dim),
            token_dim=int(model_cfg.token_dim),
        )
        self.decoder = AnchorConditionedDecoder3D(
            channels=channels,
            num_classes=int(model_cfg.num_classes),
            token_dim=int(model_cfg.token_dim),
            attention_heads=int(model_cfg.attention_heads),
            mamba_cfg=mamba_cfg,
        )
        self.eps_distance = float(model_cfg.anchor.eps_distance)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, skips = self.encoder(x)
        z_topo = self.topology_head(h)
        c_app = self.context_head(h)
        mu, var = self.anchor_distribution(z_topo, c_app)
        anchor_tokens = self.token_generator(z_topo, mu, var)
        logits = self.decoder(h, skips, anchor_tokens)
        prob = torch.softmax(logits, dim=1)
        mah = mahalanobis_distance(z_topo, mu, var, epsilon=self.eps_distance)
        return {
            "logits": logits,
            "prob": prob,
            "z_topo": z_topo,
            "c_app": c_app,
            "mu": mu,
            "var": var,
            "anchor_tokens": anchor_tokens,
            "mahalanobis": mah,
        }
