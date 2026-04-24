from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from topoanchor.topology.cache import read_cache_record
from topoanchor.topology.schema import TopologyCacheRecord


def load_topology_records(cache_dir: str | Path) -> list[TopologyCacheRecord]:
    cache_dir = Path(cache_dir)
    records = [read_cache_record(path) for path in sorted(cache_dir.glob("*.json"))]
    if not records:
        raise FileNotFoundError(f"No topology cache JSON files found in {cache_dir}.")
    return records


def records_to_descriptor_frame(records: list[TopologyCacheRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {
            "sample_id": record.sample_id,
            "signature": record.signature,
            "descriptor_version": record.descriptor_version,
            "num_classes": record.num_classes,
            "vector_length": len(record.vector),
        }
        aggregate = record.descriptor.aggregate_descriptor
        if aggregate is not None:
            row.update(
                {
                    "foreground_voxel_fraction": aggregate.voxel_fraction,
                    "foreground_components": aggregate.component_count,
                    "foreground_euler": aggregate.euler_number,
                    "foreground_hole_proxy": aggregate.hole_count_proxy,
                    "foreground_mean_component_size": aggregate.mean_component_size,
                    "foreground_max_component_size": aggregate.max_component_size,
                }
            )
        for desc in record.descriptor.class_descriptors:
            prefix = f"class_{desc.label}"
            row.update(
                {
                    f"{prefix}_voxel_fraction": desc.voxel_fraction,
                    f"{prefix}_components": desc.component_count,
                    f"{prefix}_euler": desc.euler_number,
                    f"{prefix}_hole_proxy": desc.hole_count_proxy,
                    f"{prefix}_mean_component_size": desc.mean_component_size,
                    f"{prefix}_max_component_size": desc.max_component_size,
                    f"{prefix}_extent": desc.mean_extent,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def descriptor_summary(frame: pd.DataFrame) -> dict:
    numeric = frame.select_dtypes(include="number")
    summary = {
        "num_samples": int(len(frame)),
        "num_unique_signatures": int(frame["signature"].nunique()) if "signature" in frame else 0,
        "numeric_describe": numeric.describe().to_dict() if not numeric.empty else {},
    }
    if "signature" in frame:
        summary["signature_counts"] = frame["signature"].value_counts().to_dict()
    return summary


def write_descriptor_summary(
    cache_dir: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    records = load_topology_records(cache_dir)
    frame = records_to_descriptor_frame(records)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "topology_descriptor_summary.csv"
    json_path = output_dir / "topology_descriptor_summary.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(descriptor_summary(frame), indent=2), encoding="utf-8")
    return csv_path, json_path
