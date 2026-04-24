from __future__ import annotations

import json
import random
from pathlib import Path


def split_sample_ids(
    sample_ids: list[str],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[str]]:
    if not sample_ids:
        raise ValueError("Cannot split an empty sample list.")
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be non-negative and sum to < 1.")
    ordered = sorted(sample_ids)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    total = len(ordered)
    test_count = int(round(total * test_fraction))
    val_count = int(round(total * val_fraction))
    if total >= 3:
        if test_fraction > 0:
            test_count = max(test_count, 1)
        if val_fraction > 0:
            val_count = max(val_count, 1)
    if val_count + test_count >= total:
        overflow = val_count + test_count - total + 1
        test_count = max(test_count - overflow, 0)
    test_ids = ordered[:test_count]
    val_ids = ordered[test_count : test_count + val_count]
    train_ids = ordered[test_count + val_count :]
    if not train_ids:
        raise ValueError("Split settings leave no training samples.")
    return {"train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)}


def write_split_json(splits: dict[str, list[str]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return path
