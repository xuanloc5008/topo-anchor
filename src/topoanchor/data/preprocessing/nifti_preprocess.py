from __future__ import annotations

from pathlib import Path

import numpy as np

from topoanchor.data.manifests.builder import (
    PairedNiftiSample,
    nifti_stem,
    pair_image_mask_files,
)
from topoanchor.utils.imports import require_package


def _as_canonical(image):
    nib = require_package("nibabel", "pip install nibabel")
    return nib.as_closest_canonical(image)


def _resample_array(
    data: np.ndarray,
    *,
    source_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    order: int,
) -> np.ndarray:
    ndimage = require_package("scipy.ndimage", "pip install scipy")
    zoom = tuple(source / target for source, target in zip(source_spacing, target_spacing))
    return ndimage.zoom(data, zoom=zoom, order=order)


def _normalize_nonzero(data: np.ndarray) -> np.ndarray:
    data = data.astype(np.float32, copy=False)
    mask = data != 0
    if not mask.any():
        return data
    mean = float(data[mask].mean())
    std = float(data[mask].std())
    if std < 1e-8:
        std = 1.0
    out = data.copy()
    out[mask] = (out[mask] - mean) / std
    return out


def preprocess_pair(
    sample: PairedNiftiSample,
    *,
    output_image_dir: str | Path,
    output_mask_dir: str | Path,
    canonical_orientation: bool,
    target_spacing: tuple[float, float, float] | None,
    normalize_nonzero: bool,
) -> PairedNiftiSample:
    nib = require_package("nibabel", "pip install nibabel")
    image_nii = nib.load(str(sample.image_path))
    mask_nii = nib.load(str(sample.mask_path))
    if canonical_orientation:
        image_nii = _as_canonical(image_nii)
        mask_nii = _as_canonical(mask_nii)

    image = np.asarray(image_nii.get_fdata(), dtype=np.float32)
    mask = np.rint(np.asarray(mask_nii.get_fdata())).astype(np.int16)
    source_spacing = tuple(float(value) for value in image_nii.header.get_zooms()[:3])
    affine = image_nii.affine

    if target_spacing is not None:
        image = _resample_array(
            image,
            source_spacing=source_spacing,
            target_spacing=target_spacing,
            order=1,
        ).astype(np.float32)
        mask = _resample_array(
            mask,
            source_spacing=source_spacing,
            target_spacing=target_spacing,
            order=0,
        ).astype(np.int16)
        scale = np.diag(
            [
                target_spacing[0] / source_spacing[0],
                target_spacing[1] / source_spacing[1],
                target_spacing[2] / source_spacing[2],
                1.0,
            ]
        )
        affine = affine @ scale

    if image.shape[:3] != mask.shape[:3]:
        raise ValueError(
            f"Preprocessed image/mask shape mismatch for {sample.sample_id}: {image.shape} vs {mask.shape}"
        )
    if normalize_nonzero:
        image = _normalize_nonzero(image)

    output_image_dir = Path(output_image_dir)
    output_mask_dir = Path(output_mask_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_image_dir / f"{sample.sample_id}.nii.gz"
    mask_path = output_mask_dir / f"{sample.sample_id}_mask.nii.gz"
    nib.save(nib.Nifti1Image(image.astype(np.float32), affine), str(image_path))
    nib.save(nib.Nifti1Image(mask.astype(np.int16), affine), str(mask_path))
    return PairedNiftiSample(sample.sample_id, image_path.resolve(), mask_path.resolve())


def preprocess_paired_nifti_dataset(
    *,
    source_image_dir: str | Path,
    source_mask_dir: str | Path,
    output_image_dir: str | Path,
    output_mask_dir: str | Path,
    image_globs: list[str],
    mask_globs: list[str],
    mask_suffixes: list[str],
    canonical_orientation: bool,
    target_spacing: tuple[float, float, float] | None,
    normalize_nonzero: bool,
) -> list[PairedNiftiSample]:
    samples = pair_image_mask_files(
        source_image_dir,
        source_mask_dir,
        image_globs=image_globs,
        mask_globs=mask_globs,
        mask_suffixes=mask_suffixes,
    )
    outputs = []
    for sample in samples:
        outputs.append(
            preprocess_pair(
                sample,
                output_image_dir=output_image_dir,
                output_mask_dir=output_mask_dir,
                canonical_orientation=canonical_orientation,
                target_spacing=target_spacing,
                normalize_nonzero=normalize_nonzero,
            )
        )
    return outputs
