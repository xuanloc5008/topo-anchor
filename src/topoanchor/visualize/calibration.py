from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_topology_calibration_scatter(
    metrics_frame: pd.DataFrame,
    *,
    output_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    required = {"mahalanobis", "dice_mean", "calibrated_confidence"}
    missing = required - set(metrics_frame.columns)
    if missing:
        raise ValueError(f"Calibration scatter is missing required columns: {sorted(missing)}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        metrics_frame["mahalanobis"],
        metrics_frame["dice_mean"],
        c=metrics_frame["calibrated_confidence"],
    )
    ax.set_xlabel("Mahalanobis topology-anchor distance")
    ax.set_ylabel("Mean Dice")
    fig.colorbar(scatter, ax=ax, label="Calibrated confidence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
