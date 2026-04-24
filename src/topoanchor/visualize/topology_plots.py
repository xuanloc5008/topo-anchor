from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_descriptor_histograms(
    descriptor_frame: pd.DataFrame,
    *,
    output_dir: str | Path,
    columns: list[str] | None = None,
) -> list[Path]:
    import matplotlib.pyplot as plt

    numeric_columns = list(descriptor_frame.select_dtypes(include="number").columns)
    columns = columns or numeric_columns
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for column in columns:
        if column not in descriptor_frame.columns or column not in numeric_columns:
            continue
        fig, ax = plt.subplots()
        descriptor_frame[column].dropna().hist(ax=ax, bins=30)
        ax.set_xlabel(column)
        ax.set_ylabel("count")
        fig.tight_layout()
        path = output_dir / f"{column}_hist.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        outputs.append(path)
    return outputs
