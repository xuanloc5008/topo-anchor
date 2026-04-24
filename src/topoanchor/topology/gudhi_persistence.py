from __future__ import annotations

import math

import numpy as np

from topoanchor.topology.schema import PersistenceDescriptor
from topoanchor.utils.imports import require_package


def _finite_lifetime(birth: float, death: float) -> float:
    if math.isinf(death) or math.isnan(death):
        return 0.0
    return max(float(death - birth), 0.0)


def compute_cubical_persistence_descriptor(
    scalar_field: np.ndarray,
    *,
    max_dimension: int = 2,
    top_k_lifetimes: int = 16,
) -> PersistenceDescriptor:
    gudhi = require_package("gudhi", "pip install gudhi")
    field = np.asarray(scalar_field, dtype=np.float64)
    if field.ndim not in (2, 3):
        raise ValueError(f"GUDHI cubical persistence expects 2D or 3D data, got {field.shape}.")

    complex_ = gudhi.CubicalComplex(dimensions=field.shape, top_dimensional_cells=field.ravel())
    intervals = complex_.persistence()
    intervals_by_dim: dict[int, list[tuple[float, float]]] = {dim: [] for dim in range(max_dimension + 1)}
    top_lifetimes_by_dim: dict[int, list[float]] = {}
    vector: list[float] = []

    for dim, pair in intervals:
        if dim > max_dimension:
            continue
        birth, death = float(pair[0]), float(pair[1])
        if math.isinf(death) or math.isnan(death):
            death = birth
        intervals_by_dim.setdefault(dim, []).append((birth, death))

    for dim in range(max_dimension + 1):
        lifetimes = sorted(
            (_finite_lifetime(birth, death) for birth, death in intervals_by_dim.get(dim, [])),
            reverse=True,
        )
        padded = lifetimes[:top_k_lifetimes] + [0.0] * max(top_k_lifetimes - len(lifetimes), 0)
        top_lifetimes_by_dim[dim] = padded
        vector.extend(padded)

    return PersistenceDescriptor(
        intervals_by_dim=intervals_by_dim,
        top_lifetimes_by_dim=top_lifetimes_by_dim,
        vector=vector,
    )


def binary_mask_persistence_descriptor(
    mask: np.ndarray,
    *,
    max_dimension: int = 2,
    top_k_lifetimes: int = 16,
) -> PersistenceDescriptor:
    mask = np.asarray(mask).astype(bool)
    # Foreground cells are born before background cells in this lower-star filtration.
    field = np.where(mask, 0.0, 1.0)
    return compute_cubical_persistence_descriptor(
        field,
        max_dimension=max_dimension,
        top_k_lifetimes=top_k_lifetimes,
    )
