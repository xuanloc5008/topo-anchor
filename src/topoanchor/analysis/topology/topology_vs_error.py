from __future__ import annotations

import pandas as pd

from topoanchor.analysis.topology.descriptor_correlation import correlate_descriptor_table


ERROR_COLUMNS = [
    "dice_mean",
    "iou_mean",
    "hausdorff_mean_finite",
    "assd_mean_finite",
    "topology_vector_distance",
]


def topology_error_correlations(
    descriptor_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
) -> pd.DataFrame:
    available_targets = [column for column in ERROR_COLUMNS if column in evaluation_frame.columns]
    if not available_targets:
        return pd.DataFrame(columns=["descriptor", "target", "correlation", "count"])
    return correlate_descriptor_table(
        descriptor_frame,
        evaluation_frame[["sample_id", *available_targets]],
        target_columns=available_targets,
    )
