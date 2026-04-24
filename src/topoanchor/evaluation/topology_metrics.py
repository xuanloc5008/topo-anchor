from __future__ import annotations

import numpy as np

from topoanchor.topology.skimage_descriptors import compute_skimage_topology_descriptor


def topology_vector_distance(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    sample_id: str,
    num_classes: int,
    descriptor_version: str,
) -> float:
    pred_desc = compute_skimage_topology_descriptor(
        pred_mask,
        sample_id=f"{sample_id}:pred",
        num_classes=num_classes,
        descriptor_version=descriptor_version,
    )
    target_desc = compute_skimage_topology_descriptor(
        target_mask,
        sample_id=f"{sample_id}:target",
        num_classes=num_classes,
        descriptor_version=descriptor_version,
    )
    pred_vec = np.log1p(np.clip(pred_desc.vector(), a_min=0.0, a_max=None))
    target_vec = np.log1p(np.clip(target_desc.vector(), a_min=0.0, a_max=None))
    return float(np.linalg.norm(pred_vec - target_vec))
