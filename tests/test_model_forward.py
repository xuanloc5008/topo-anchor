import pytest

torch = pytest.importorskip("torch")


def test_model_forward_contract() -> None:
    from omegaconf import OmegaConf

    from topoanchor.models.topology_anchored_model import TopologyAnchoredSegmentationModel

    cfg = OmegaConf.create(
        {
            "model": {
                "in_channels": 1,
                "num_classes": 3,
                "channels": [4, 8, 16, 32],
                "topo_dim": 12,
                "app_dim": 6,
                "token_dim": 12,
                "dropout": 0.0,
                "attention_heads": 2,
                "mamba": {"backend": "conv_fallback", "d_state": 4, "d_conv": 2, "expand": 1},
                "anchor": {"eps_sigma": 1e-5, "eps_distance": 1e-6},
            }
        }
    )
    model = TopologyAnchoredSegmentationModel(cfg)
    output = model(torch.randn(1, 1, 16, 16, 16))
    assert output["logits"].shape == (1, 3, 16, 16, 16)
    assert output["z_topo"].shape == (1, 12)
    assert output["anchor_tokens"].shape == (1, 3, 12)
