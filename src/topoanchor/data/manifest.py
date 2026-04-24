from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_MANIFEST_COLUMNS = [
    "sample_id",
    "image_path",
    "mask_path",
    "split",
    "dataset_name",
    "spacing",
    "shape",
    "orientation",
    "roi_bbox",
    "topology_cache_path",
    "topology_descriptor_version",
]


@dataclass(slots=True)
class ManifestRecord:
    sample_id: str
    image_path: Path
    mask_path: Path | None
    split: str
    dataset_name: str
    topology_cache_path: Path | None
    topology_descriptor_version: str
    metadata: dict[str, str]


def resolve_manifest_path(path_value: str | float | None, *, manifest_path: Path) -> Path | None:
    if path_value is None or pd.isna(path_value) or str(path_value).strip() == "":
        return None
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def read_manifest_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def validate_manifest_columns(frame: pd.DataFrame, *, path: str | Path) -> None:
    missing = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required manifest columns: {missing}")
    if frame["sample_id"].isna().any():
        raise ValueError(f"{path} contains rows with empty sample_id.")
    duplicated = frame["sample_id"][frame["sample_id"].duplicated()].tolist()
    if duplicated:
        raise ValueError(f"{path} contains duplicated sample_id values: {duplicated[:10]}")


def validate_manifest_paths(
    path: str | Path,
    *,
    require_masks: bool = True,
    require_topology_cache: bool = False,
    expected_topology_version: str | None = None,
) -> pd.DataFrame:
    manifest_path = Path(path)
    frame = read_manifest_frame(manifest_path)
    validate_manifest_columns(frame, path=manifest_path)
    for row in frame.itertuples(index=False):
        image_path = resolve_manifest_path(getattr(row, "image_path"), manifest_path=manifest_path)
        if image_path is None or not image_path.exists():
            raise FileNotFoundError(f"Missing image for sample {row.sample_id}: {image_path}")
        mask_path = resolve_manifest_path(getattr(row, "mask_path"), manifest_path=manifest_path)
        if require_masks and (mask_path is None or not mask_path.exists()):
            raise FileNotFoundError(f"Missing mask for sample {row.sample_id}: {mask_path}")
        cache_path = resolve_manifest_path(
            getattr(row, "topology_cache_path"), manifest_path=manifest_path
        )
        if require_topology_cache and (cache_path is None or not cache_path.exists()):
            raise FileNotFoundError(
                f"Missing topology cache for sample {row.sample_id}: {cache_path}. "
                "Run scripts/precompute_topology.py first."
            )
        if expected_topology_version:
            version = str(getattr(row, "topology_descriptor_version"))
            if version != expected_topology_version:
                raise ValueError(
                    f"Sample {row.sample_id} has topology_descriptor_version={version}, "
                    f"expected {expected_topology_version}."
                )
    return frame


def load_manifest_records(
    path: str | Path,
    *,
    require_masks: bool = True,
    require_topology_cache: bool = False,
    expected_topology_version: str | None = None,
) -> list[ManifestRecord]:
    manifest_path = Path(path)
    frame = validate_manifest_paths(
        manifest_path,
        require_masks=require_masks,
        require_topology_cache=require_topology_cache,
        expected_topology_version=expected_topology_version,
    )
    records: list[ManifestRecord] = []
    for row in frame.itertuples(index=False):
        image_path = resolve_manifest_path(getattr(row, "image_path"), manifest_path=manifest_path)
        mask_path = resolve_manifest_path(getattr(row, "mask_path"), manifest_path=manifest_path)
        cache_path = resolve_manifest_path(
            getattr(row, "topology_cache_path"), manifest_path=manifest_path
        )
        metadata = {
            column: "" if pd.isna(getattr(row, column)) else str(getattr(row, column))
            for column in frame.columns
            if column
            not in {
                "sample_id",
                "image_path",
                "mask_path",
                "split",
                "dataset_name",
                "topology_cache_path",
                "topology_descriptor_version",
            }
        }
        records.append(
            ManifestRecord(
                sample_id=str(getattr(row, "sample_id")),
                image_path=image_path or Path(),
                mask_path=mask_path,
                split=str(getattr(row, "split")),
                dataset_name=str(getattr(row, "dataset_name")),
                topology_cache_path=cache_path,
                topology_descriptor_version=str(getattr(row, "topology_descriptor_version")),
                metadata=metadata,
            )
        )
    return records


def assert_disjoint_sample_ids(manifest_paths: Iterable[str | Path]) -> None:
    seen: dict[str, Path] = {}
    for path in manifest_paths:
        path = Path(path)
        if not path.exists():
            continue
        frame = read_manifest_frame(path)
        validate_manifest_columns(frame, path=path)
        for sample_id in frame["sample_id"].astype(str):
            if sample_id in seen:
                raise ValueError(f"sample_id {sample_id} appears in both {seen[sample_id]} and {path}.")
            seen[sample_id] = path
