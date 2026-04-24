from __future__ import annotations

from pathlib import Path

import numpy as np

from topoanchor.utils.imports import require_package


def load_nifti_array(path: str | Path) -> np.ndarray:
    nib = require_package("nibabel", "pip install nibabel")
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata())
    return data


def load_nifti_shape(path: str | Path) -> tuple[int, ...]:
    nib = require_package("nibabel", "pip install nibabel")
    image = nib.load(str(path))
    return tuple(int(dim) for dim in image.shape)


def get_nifti_metadata(path: str | Path) -> dict:
    nib = require_package("nibabel", "pip install nibabel")
    image = nib.load(str(path))
    zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    shape = tuple(int(dim) for dim in image.shape[:3])
    orientation = "".join(nib.aff2axcodes(image.affine))
    return {
        "spacing": zooms,
        "spatial_shape": shape,
        "orientation": orientation,
    }
