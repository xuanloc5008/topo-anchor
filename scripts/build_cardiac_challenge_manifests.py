from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra
from omegaconf import DictConfig

from topoanchor.data.manifests.cardiac_challenges import write_cardiac_challenge_manifests


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cardiac = cfg.data.cardiac_challenges
    outputs = write_cardiac_challenge_manifests(
        acdc_root=cardiac.acdc_root,
        mnms1_root=cardiac.mnms1_root,
        mnms2_root=cardiac.mnms2_root,
        output_dir=cardiac.output_dir,
        manifest_dir=cfg.paths.manifest_dir,
        split_output_path=Path(cfg.paths.split_dir) / "cardiac_challenge_splits.json",
        descriptor_version=str(cfg.data.topology_descriptor_version),
        mnms1_metadata_csv=cardiac.mnms1_metadata_csv,
        mnms2_metadata_csv=cardiac.mnms2_metadata_csv,
        metadata_sample_id_columns=list(cardiac.metadata_sample_id_columns),
        metadata_vendor_columns=list(cardiac.metadata_vendor_columns),
        metadata_site_columns=list(cardiac.metadata_site_columns),
        metadata_protocol_columns=list(cardiac.metadata_protocol_columns),
        mnms1_vendor_holdout=list(cardiac.mnms1_vendor_holdout),
        train_datasets=list(cardiac.train_datasets),
        external_ood_datasets=list(cardiac.external_ood_datasets),
        val_fraction=float(cardiac.val_fraction),
        test_fraction=float(cardiac.test_fraction),
        seed=int(cardiac.split_seed),
        include_acdc_testing=bool(cardiac.include_acdc_testing),
        include_mnms1_unlabelled=bool(cardiac.include_mnms1_unlabelled),
        include_mnms2_la=bool(cardiac.include_mnms2_la),
    )
    for name, path in outputs.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
