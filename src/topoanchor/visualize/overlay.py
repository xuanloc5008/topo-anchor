from __future__ import annotations

from pathlib import Path

import numpy as np


def middle_slice(volume: np.ndarray, *, axis: int = 2) -> np.ndarray:
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}.")
    index = volume.shape[axis] // 2
    return np.take(volume, index, axis=axis)


def save_middle_slice_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    output_path: str | Path,
    axis: int = 2,
    alpha: float = 0.35,
) -> Path:
    import matplotlib.pyplot as plt

    image_slice = middle_slice(image, axis=axis)
    mask_slice = middle_slice(mask, axis=axis)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.imshow(np.rot90(image_slice), cmap="gray")
    masked = np.ma.masked_where(mask_slice <= 0, mask_slice)
    ax.imshow(np.rot90(masked), cmap="tab20", alpha=alpha, interpolation="nearest")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path
