from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra
import pandas as pd
from omegaconf import DictConfig

from topoanchor.visualize.calibration import plot_topology_calibration_scatter


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    metrics_path = Path(cfg.paths.output_dir) / "tables" / "evaluation_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Run scripts/evaluate.py first: {metrics_path}")
    frame = pd.read_csv(metrics_path)
    figure_dir = Path(cfg.paths.output_dir) / "figures"
    out = plot_topology_calibration_scatter(
        frame,
        output_path=figure_dir / "topology_calibration_scatter.png",
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
