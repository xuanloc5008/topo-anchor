from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def _numeric_columns(frame: pd.DataFrame, exclude: Iterable[str]) -> list[str]:
    excluded = set(exclude)
    return [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in excluded
    ]


def correlate_descriptor_table(
    descriptor_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    on: str = "sample_id",
    method: str = "spearman",
    descriptor_columns: list[str] | None = None,
    target_columns: list[str] | None = None,
) -> pd.DataFrame:
    if on not in descriptor_frame.columns or on not in target_frame.columns:
        raise ValueError(f"Both frames must contain merge key `{on}`.")
    merged = descriptor_frame.merge(target_frame, on=on, suffixes=("_descriptor", "_target"))
    if merged.empty:
        return pd.DataFrame(columns=["descriptor", "target", "correlation", "count"])

    descriptor_columns = descriptor_columns or _numeric_columns(
        descriptor_frame, exclude=[on, "num_classes", "vector_length"]
    )
    target_columns = target_columns or _numeric_columns(target_frame, exclude=[on])
    rows = []
    for descriptor in descriptor_columns:
        for target in target_columns:
            values = merged[[descriptor, target]].dropna()
            if len(values) < 3 or values[descriptor].nunique() < 2 or values[target].nunique() < 2:
                continue
            rows.append(
                {
                    "descriptor": descriptor,
                    "target": target,
                    "correlation": float(values[descriptor].corr(values[target], method=method)),
                    "count": int(len(values)),
                }
            )
    return pd.DataFrame(rows).sort_values("correlation", key=lambda col: col.abs(), ascending=False)
