from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra
import pandas as pd
from omegaconf import DictConfig

from topoanchor.data.manifest import read_manifest_frame, resolve_manifest_path, validate_manifest_columns
from topoanchor.data.nifti import load_nifti_array
from topoanchor.topology.cache import read_cache_record, write_cache_record, write_jsonl, write_vector_npz
from topoanchor.topology.manifest_writer import update_manifest_topology_paths
from topoanchor.topology.targets import build_topology_cache_record


def _mask_to_labels(mask):
    import numpy as np

    mask = np.asarray(mask)
    mask = np.squeeze(mask)
    return np.rint(mask).astype("int64")


def _read_reusable_cache(cache_path: Path, *, sample_id: str, descriptor_version: str):
    if not cache_path.exists():
        return None
    try:
        record = read_cache_record(cache_path)
    except Exception:
        return None
    if record.sample_id != sample_id or record.descriptor_version != descriptor_version:
        return None
    if not record.vector:
        return None
    return record


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cache_dir = Path(cfg.paths.topology_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    descriptor_version = str(cfg.topology.descriptor_version)
    reuse_existing = bool(cfg.topology.cache.get("reuse_existing", True))

    manifests = [Path(cfg.data.train_manifest), Path(cfg.data.val_manifest), Path(cfg.data.test_manifest)]
    ood_manifest = Path(cfg.paths.ood_manifest)
    if ood_manifest.exists() and ood_manifest not in manifests:
        manifests.append(ood_manifest)

    for manifest in manifests:
        manifest_path = Path(manifest)
        frame = read_manifest_frame(manifest_path)
        validate_manifest_columns(frame, path=manifest_path)
        records = []
        sample_to_cache_path: dict[str, str] = {}
        reused = 0
        computed = 0

        for row in frame.itertuples(index=False):
            sample_id = str(getattr(row, "sample_id"))
            mask_path = resolve_manifest_path(getattr(row, "mask_path"), manifest_path=manifest_path)
            if mask_path is None or not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for sample {sample_id}: {mask_path}")
            cache_path = cache_dir / f"{sample_id}.json"
            record = (
                _read_reusable_cache(
                    cache_path,
                    sample_id=sample_id,
                    descriptor_version=descriptor_version,
                )
                if reuse_existing
                else None
            )
            if record is None:
                mask = _mask_to_labels(load_nifti_array(mask_path))
                record = build_topology_cache_record(
                    mask,
                    sample_id=sample_id,
                    mask_path=str(mask_path),
                    num_classes=int(cfg.data.num_classes),
                    descriptor_version=descriptor_version,
                    include_aggregate_foreground=bool(cfg.topology.include_aggregate_foreground),
                    use_gudhi=bool(cfg.topology.use_gudhi),
                    use_gtda=bool(cfg.topology.use_gtda) and bool(cfg.topology.use_gudhi),
                    max_persistence_dimension=int(cfg.topology.persistence.max_dimension),
                    top_k_lifetimes=int(cfg.topology.persistence.top_k_lifetimes),
                )
                write_cache_record(record, cache_path)
                computed += 1
            else:
                reused += 1
            sample_to_cache_path[sample_id] = str(cache_path)
            records.append(record)

        write_jsonl(records, cache_dir / f"{manifest_path.stem}.jsonl")
        write_vector_npz(records, cache_dir / f"{manifest_path.stem}.npz")
        if bool(cfg.topology.cache.update_manifests):
            update_manifest_topology_paths(
                manifest_path,
                sample_to_cache_path=sample_to_cache_path,
                descriptor_version=descriptor_version,
            )
        print(
            f"Wrote topology cache index for {len(records)} samples from {manifest_path} "
            f"({computed} computed, {reused} reused)."
        )


if __name__ == "__main__":
    main()
