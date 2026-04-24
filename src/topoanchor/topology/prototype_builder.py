from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TopologyPrototype:
    prototype_id: str
    sample_ids: list[str]
    center: np.ndarray


def build_signature_prototypes(
    sample_ids: list[str],
    signatures: list[str],
    vectors: np.ndarray,
) -> list[TopologyPrototype]:
    if len(sample_ids) != len(signatures) or len(sample_ids) != len(vectors):
        raise ValueError("sample_ids, signatures, and vectors must have the same length.")
    groups: dict[str, list[int]] = {}
    for idx, signature in enumerate(signatures):
        groups.setdefault(signature, []).append(idx)
    prototypes = []
    for signature, indices in sorted(groups.items()):
        group_vectors = vectors[np.asarray(indices)]
        prototypes.append(
            TopologyPrototype(
                prototype_id=signature,
                sample_ids=[sample_ids[idx] for idx in indices],
                center=group_vectors.mean(axis=0),
            )
        )
    return prototypes
