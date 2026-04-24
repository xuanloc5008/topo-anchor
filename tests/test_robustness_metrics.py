from __future__ import annotations

import pandas as pd

from topoanchor.evaluation.robustness_metrics import (
    summarize_by_domain,
    summarize_local_vs_shift,
)


def test_local_vs_shift_summary_reports_delta() -> None:
    frame = pd.DataFrame(
        {
            "split": ["test", "test", "ood", "ood"],
            "dice_mean": [0.9, 0.8, 0.75, 0.65],
            "ece": [0.05, 0.07, 0.12, 0.14],
        }
    )
    summary = summarize_local_vs_shift(frame, metric_columns=["dice_mean", "ece"])
    assert set(summary["domain_group"]) == {"local", "distribution_shift", "shift_delta"}
    delta = summary[summary["domain_group"] == "shift_delta"].iloc[0]
    assert round(delta["dice_mean_mean"], 6) == -0.15
    assert round(delta["ece_mean"], 6) == 0.07


def test_domain_summary_groups_by_vendor() -> None:
    frame = pd.DataFrame(
        {
            "vendor": ["A", "A", "B"],
            "split": ["test", "test", "ood"],
            "dice_mean": [0.9, 0.8, 0.7],
        }
    )
    summary = summarize_by_domain(frame, domain_columns=["vendor"], metric_columns=["dice_mean"])
    assert set(summary["vendor"]) == {"A", "B"}
    assert int(summary[summary["vendor"] == "A"].iloc[0]["n"]) == 2
