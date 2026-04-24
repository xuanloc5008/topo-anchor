from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_hydra_model_backend_override_is_valid() -> None:
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=["model.mamba.backend=conv_fallback", "trainer.fast_dev_run=true"],
        )
    assert cfg.model.mamba.backend == "conv_fallback"
    assert cfg.trainer.fast_dev_run is True
    assert "mamba" not in cfg.model.get("model", {})


def test_l40_hardware_preset_composes() -> None:
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["+hardware=l40"])
    assert cfg.model.mamba.backend == "mamba_ssm"
    assert cfg.trainer.accelerator == "gpu"
    assert cfg.data.patch_size == [192, 192, 32]
    assert cfg.data.loader.prefetch_factor == 4
