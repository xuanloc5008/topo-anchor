from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METRIC_COLUMNS = [
    "dice_mean",
    "iou_mean",
    "raw_confidence",
    "calibrated_confidence",
    "mahalanobis",
    "ece",
    "brier",
    "topology_vector_distance",
]


def summarize_by_domain(
    frame: pd.DataFrame,
    *,
    domain_columns: list[str],
    metric_columns: list[str] | None = None,
) -> pd.DataFrame:
    metric_columns = metric_columns or DEFAULT_METRIC_COLUMNS
    available_domains = [column for column in domain_columns if column in frame.columns]
    available_metrics = [column for column in metric_columns if column in frame.columns]
    if not available_domains:
        available_domains = ["split"] if "split" in frame.columns else []
    if not available_metrics:
        raise ValueError("No metric columns available for robustness summary.")

    rows = []
    groupers = available_domains if available_domains else [lambda _: "all"]
    grouped = frame.groupby(groupers, dropna=False) if available_domains else [("all", frame)]
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: value for column, value in zip(available_domains or ["group"], key)}
        row["n"] = int(len(group))
        for metric in available_metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_local_vs_shift(
    frame: pd.DataFrame,
    *,
    local_splits: list[str] | None = None,
    shifted_splits: list[str] | None = None,
    metric_columns: list[str] | None = None,
) -> pd.DataFrame:
    if "split" not in frame.columns:
        raise ValueError("Evaluation frame must include a `split` column.")
    local_splits = local_splits or ["test"]
    shifted_splits = shifted_splits or ["ood"]
    metric_columns = metric_columns or DEFAULT_METRIC_COLUMNS
    available_metrics = [column for column in metric_columns if column in frame.columns]
    rows = []
    for group_name, splits in [("local", local_splits), ("distribution_shift", shifted_splits)]:
        group = frame[frame["split"].isin(splits)]
        row = {"domain_group": group_name, "splits": ",".join(splits), "n": int(len(group))}
        for metric in available_metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    summary = pd.DataFrame(rows)
    local = summary[summary["domain_group"] == "local"]
    shifted = summary[summary["domain_group"] == "distribution_shift"]
    if not local.empty and not shifted.empty:
        delta = {"domain_group": "shift_delta", "splits": "distribution_shift-local", "n": int(shifted.iloc[0]["n"])}
        for metric in available_metrics:
            delta[f"{metric}_mean"] = shifted.iloc[0][f"{metric}_mean"] - local.iloc[0][f"{metric}_mean"]
            delta[f"{metric}_std"] = np.nan
        summary = pd.concat([summary, pd.DataFrame([delta])], ignore_index=True)
    return summary


def write_robustness_tables(
    evaluation_csv: str | Path,
    *,
    output_dir: str | Path,
    domain_columns: list[str],
) -> tuple[Path, Path]:
    frame = pd.read_csv(evaluation_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    domain_summary = summarize_by_domain(frame, domain_columns=domain_columns)
    local_shift_summary = summarize_local_vs_shift(frame)
    domain_path = output_dir / "robustness_by_domain.csv"
    local_shift_path = output_dir / "robustness_local_vs_shift.csv"
    domain_summary.to_csv(domain_path, index=False)
    local_shift_summary.to_csv(local_shift_path, index=False)
    return domain_path, local_shift_path
