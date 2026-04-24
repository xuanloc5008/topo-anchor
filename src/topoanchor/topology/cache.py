from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from topoanchor.topology.schema import TopologyCacheRecord


def write_cache_record(record: TopologyCacheRecord, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    return path


def read_cache_record(path: str | Path) -> TopologyCacheRecord:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return TopologyCacheRecord.from_dict(data)


def write_jsonl(records: Iterable[TopologyCacheRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict()) + "\n")
    return path


def write_vector_npz(records: Iterable[TopologyCacheRecord], path: str | Path) -> Path:
    records = list(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray([record.vector for record in records], dtype=np.float32)
    sample_ids = np.asarray([record.sample_id for record in records])
    np.savez_compressed(path, sample_id=sample_ids, vector=vectors)
    return path
