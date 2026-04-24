from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra
from omegaconf import DictConfig

from topoanchor.data.manifest import assert_disjoint_sample_ids, load_manifest_records
from topoanchor.data.nifti import load_nifti_shape
from topoanchor.topology.sanity import validate_cache_record


def _shapes_compatible(image_shape: tuple[int, ...], mask_shape: tuple[int, ...]) -> bool:
    if image_shape == mask_shape:
        return True
    if len(image_shape) == len(mask_shape) + 1 and image_shape[-len(mask_shape) :] == mask_shape:
        return True
    if len(image_shape) == len(mask_shape) + 1 and image_shape[: len(mask_shape)] == mask_shape:
        return True
    return False


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    manifests = [Path(cfg.data.train_manifest), Path(cfg.data.val_manifest), Path(cfg.data.test_manifest)]
    ood_manifest = Path(cfg.paths.ood_manifest)
    if ood_manifest.exists() and ood_manifest not in manifests:
        manifests.append(ood_manifest)
    assert_disjoint_sample_ids(manifests)

    total = 0
    for manifest in manifests:
        records = load_manifest_records(
            manifest,
            require_masks=True,
            require_topology_cache=bool(cfg.data.require_topology_cache),
            expected_topology_version=str(cfg.data.topology_descriptor_version)
            if bool(cfg.data.require_topology_cache)
            else None,
        )
        for record in records:
            image_shape = load_nifti_shape(record.image_path)
            mask_shape = load_nifti_shape(record.mask_path)
            if not _shapes_compatible(image_shape, mask_shape):
                raise ValueError(
                    f"Image/mask shape mismatch for {record.sample_id}: "
                    f"image={image_shape}, mask={mask_shape}"
                )
            if bool(cfg.data.require_topology_cache):
                validate_cache_record(
                    record.topology_cache_path,
                    expected_version=str(cfg.data.topology_descriptor_version),
                )
            total += 1
    print(f"Data verification passed for {total} samples.")


if __name__ == "__main__":
    main()
