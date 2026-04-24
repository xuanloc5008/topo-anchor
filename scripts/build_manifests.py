from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra
from omegaconf import DictConfig

from topoanchor.data.manifests.builder import write_split_manifests


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    builder = cfg.data.manifest_builder
    outputs = write_split_manifests(
        image_dir=builder.image_dir,
        mask_dir=builder.mask_dir,
        output_dir=cfg.paths.manifest_dir,
        image_globs=list(builder.image_globs),
        mask_globs=list(builder.mask_globs),
        mask_suffixes=list(builder.mask_suffixes),
        dataset_name=str(builder.dataset_name),
        descriptor_version=str(cfg.data.topology_descriptor_version),
        val_fraction=float(builder.val_fraction),
        test_fraction=float(builder.test_fraction),
        seed=int(builder.split_seed),
        split_output_path=Path(cfg.paths.split_dir) / "splits.json",
        metadata_csv=builder.metadata_csv,
        metadata_sample_id_column=str(builder.metadata_sample_id_column),
        vendor_column=str(builder.vendor_column),
        vendor_holdout=list(builder.vendor_holdout),
        ood_split_name=str(builder.ood_split_name),
    )
    for name, path in outputs.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
