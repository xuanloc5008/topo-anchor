from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from topoanchor.data.datasets.manifest_dataset import ManifestSegmentationDataset
from topoanchor.data.manifests.cardiac_challenges import ACDC_TO_TARGET, _remap_labels


def test_acdc_label_map_standardizes_to_mnms_order() -> None:
    source = np.array([0, 1, 2, 3], dtype=np.int16)
    remapped = _remap_labels(source, ACDC_TO_TARGET)
    np.testing.assert_array_equal(remapped, np.array([0, 3, 2, 1], dtype=np.int16))


def test_manifest_dataset_label_remap_helper() -> None:
    mask = torch.tensor([0, 1, 2, 3])
    remapped = ManifestSegmentationDataset._remap_mask(mask, "1:3,2:2,3:1")
    assert remapped.tolist() == [0, 3, 2, 1]
