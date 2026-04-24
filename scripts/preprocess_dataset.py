from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra
from omegaconf import DictConfig

from topoanchor.data.preprocessing import preprocess_paired_nifti_dataset


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    preprocessing = cfg.data.preprocessing
    target_spacing = (
        None
        if preprocessing.target_spacing is None
        else tuple(float(value) for value in preprocessing.target_spacing)
    )
    samples = preprocess_paired_nifti_dataset(
        source_image_dir=preprocessing.source_image_dir,
        source_mask_dir=preprocessing.source_mask_dir,
        output_image_dir=preprocessing.output_image_dir,
        output_mask_dir=preprocessing.output_mask_dir,
        image_globs=list(preprocessing.image_globs),
        mask_globs=list(preprocessing.mask_globs),
        mask_suffixes=list(preprocessing.mask_suffixes),
        canonical_orientation=bool(preprocessing.canonical_orientation),
        target_spacing=target_spacing,
        normalize_nonzero=bool(preprocessing.normalize_nonzero),
    )
    print(f"Preprocessed {len(samples)} paired NIfTI samples.")
    print(f"Images: {preprocessing.output_image_dir}")
    print(f"Masks: {preprocessing.output_mask_dir}")


if __name__ == "__main__":
    main()
