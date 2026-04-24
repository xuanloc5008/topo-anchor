from __future__ import annotations

import math
from pathlib import Path

from topoanchor.topology.cache import read_cache_record


def validate_cache_record(path: str | Path, *, expected_version: str | None = None) -> None:
    record = read_cache_record(path)
    if expected_version and record.descriptor_version != expected_version:
        raise ValueError(
            f"{path} has descriptor version {record.descriptor_version}, expected {expected_version}."
        )
    if not record.vector:
        raise ValueError(f"{path} contains an empty topology vector.")
    if any(math.isnan(float(value)) or math.isinf(float(value)) for value in record.vector):
        raise ValueError(f"{path} contains NaN or infinite topology vector values.")
