from __future__ import annotations

from pathlib import Path

import pandas as pd


def update_manifest_topology_paths(
    manifest_path: str | Path,
    *,
    sample_to_cache_path: dict[str, str],
    descriptor_version: str,
) -> Path:
    manifest_path = Path(manifest_path)
    frame = pd.read_csv(manifest_path)
    if "sample_id" not in frame.columns:
        raise ValueError(f"{manifest_path} is missing required column `sample_id`.")
    frame["topology_cache_path"] = frame["sample_id"].map(sample_to_cache_path).fillna(
        frame.get("topology_cache_path", "")
    )
    frame["topology_descriptor_version"] = descriptor_version
    frame.to_csv(manifest_path, index=False)
    return manifest_path
