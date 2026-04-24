from __future__ import annotations

import numpy as np

from topoanchor.topology.schema import PersistenceDescriptor
from topoanchor.utils.imports import require_package


def persistence_entropy_vector(descriptor: PersistenceDescriptor) -> list[float]:
    require_package("gtda.diagrams", "pip install giotto-tda")
    values: list[float] = []
    for dim in sorted(descriptor.top_lifetimes_by_dim):
        lifetimes = np.asarray(descriptor.top_lifetimes_by_dim[dim], dtype=np.float64)
        lifetimes = lifetimes[lifetimes > 0]
        if lifetimes.size == 0:
            values.append(0.0)
            continue
        probs = lifetimes / lifetimes.sum()
        values.append(float(-(probs * np.log(probs + 1e-12)).sum()))
    return values


def append_gtda_vectorization(descriptor: PersistenceDescriptor) -> PersistenceDescriptor:
    entropy = persistence_entropy_vector(descriptor)
    descriptor.vector = list(descriptor.vector) + entropy
    return descriptor
