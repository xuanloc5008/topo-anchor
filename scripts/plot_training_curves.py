from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra
import pandas as pd
from omegaconf import DictConfig

from topoanchor.visualize.curves import plot_metric_curves


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    metrics_files = list(Path(cfg.paths.log_dir).rglob("metrics.csv"))
    if not metrics_files:
        raise FileNotFoundError(f"No Lightning metrics.csv found under {cfg.paths.log_dir}.")
    frame = pd.concat([pd.read_csv(path) for path in metrics_files], ignore_index=True)
    figure_dir = Path(cfg.paths.output_dir) / "figures"
    outputs = plot_metric_curves(
        frame,
        output_dir=figure_dir,
        metrics=["train/loss_epoch", "val/loss", "val/dice", "val/mahalanobis"],
    )
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
