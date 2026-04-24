from __future__ import annotations

import numpy as np

from topoanchor.visualize.latent_space import project_latents_2d
from topoanchor.visualize.overlay import middle_slice


def test_middle_slice_extracts_requested_axis() -> None:
    volume = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = middle_slice(volume, axis=2)
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result, volume[:, :, 2])


def test_project_latents_pca_returns_two_dimensions() -> None:
    latents = np.arange(20, dtype=np.float32).reshape(5, 4)
    projected = project_latents_2d(latents, method="pca")
    assert projected.shape == (5, 2)
