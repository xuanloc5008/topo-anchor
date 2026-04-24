from __future__ import annotations

import pandas as pd
import pytest

from topoanchor.data.manifest import REQUIRED_MANIFEST_COLUMNS, validate_manifest_columns


def test_manifest_requires_full_contract() -> None:
    frame = pd.DataFrame({"sample_id": ["a"], "image_path": ["image.nii.gz"]})
    with pytest.raises(ValueError, match="missing required manifest columns"):
        validate_manifest_columns(frame, path="manifest.csv")


def test_manifest_rejects_duplicate_sample_ids() -> None:
    frame = pd.DataFrame([{column: "" for column in REQUIRED_MANIFEST_COLUMNS} for _ in range(2)])
    frame["sample_id"] = ["same", "same"]
    with pytest.raises(ValueError, match="duplicated"):
        validate_manifest_columns(frame, path="manifest.csv")
