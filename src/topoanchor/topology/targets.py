from __future__ import annotations

import numpy as np

from topoanchor.topology.gtda_vectorization import append_gtda_vectorization
from topoanchor.topology.gudhi_persistence import binary_mask_persistence_descriptor
from topoanchor.topology.schema import TopologyCacheRecord
from topoanchor.topology.skimage_descriptors import compute_skimage_topology_descriptor


def build_topology_cache_record(
    mask: np.ndarray,
    *,
    sample_id: str,
    mask_path: str,
    num_classes: int,
    descriptor_version: str,
    include_aggregate_foreground: bool = True,
    use_gudhi: bool = True,
    use_gtda: bool = True,
    max_persistence_dimension: int = 2,
    top_k_lifetimes: int = 16,
) -> TopologyCacheRecord:
    descriptor = compute_skimage_topology_descriptor(
        mask,
        sample_id=sample_id,
        num_classes=num_classes,
        descriptor_version=descriptor_version,
        include_aggregate_foreground=include_aggregate_foreground,
    )
    if use_gudhi:
        persistence = binary_mask_persistence_descriptor(
            np.asarray(mask) > 0,
            max_dimension=max_persistence_dimension,
            top_k_lifetimes=top_k_lifetimes,
        )
        if use_gtda:
            persistence = append_gtda_vectorization(persistence)
        descriptor.persistence = persistence

    vector = descriptor.vector().astype(float).tolist()
    return TopologyCacheRecord(
        sample_id=sample_id,
        descriptor_version=descriptor_version,
        mask_path=mask_path,
        num_classes=num_classes,
        descriptor=descriptor,
        vector=vector,
        signature=descriptor.signature(),
    )
