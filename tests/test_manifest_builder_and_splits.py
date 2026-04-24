from __future__ import annotations

from topoanchor.data.manifests.builder import nifti_stem, sample_id_from_mask
from topoanchor.data.splits import split_sample_ids


def test_nifti_stem_handles_nii_gz() -> None:
    assert nifti_stem("case001.nii.gz") == "case001"
    assert nifti_stem("case001.nii") == "case001"


def test_sample_id_from_mask_strips_known_suffix() -> None:
    assert sample_id_from_mask("case001_mask.nii.gz", ["_mask"]) == "case001"
    assert sample_id_from_mask("case001-seg.nii.gz", ["_mask", "-seg"]) == "case001"


def test_split_sample_ids_is_deterministic_and_complete() -> None:
    sample_ids = [f"case{i:03d}" for i in range(10)]
    first = split_sample_ids(sample_ids, val_fraction=0.2, test_fraction=0.2, seed=7)
    second = split_sample_ids(sample_ids, val_fraction=0.2, test_fraction=0.2, seed=7)
    assert first == second
    recovered = sorted(first["train"] + first["val"] + first["test"])
    assert recovered == sorted(sample_ids)
    assert first["train"]
