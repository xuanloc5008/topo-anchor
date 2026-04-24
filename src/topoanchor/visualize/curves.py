from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_metric_curves(
    metrics_frame: pd.DataFrame,
    *,
    output_dir: str | Path,
    metrics: list[str],
) -> list[Path]:
    import matplotlib.pyplot as plt

    if "epoch" not in metrics_frame.columns:
        raise ValueError("Lightning metrics frame must include an `epoch` column.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for metric in metrics:
        if metric not in metrics_frame.columns:
            continue
        series = metrics_frame[["epoch", metric]].dropna()
        if series.empty:
            continue
        fig, ax = plt.subplots()
        ax.plot(series["epoch"], series[metric])
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        fig.tight_layout()
        path = output_dir / f"{metric.replace('/', '_')}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        outputs.append(path)
    return outputs
