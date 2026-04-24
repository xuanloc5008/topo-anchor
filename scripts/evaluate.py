from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from topoanchor.evaluation.calibration_metrics import (
    brier_score,
    calibrated_confidence,
    expected_calibration_error,
    raw_segmentation_confidence,
    voxel_confidence_correctness,
)
from topoanchor.data.manifest import read_manifest_frame
from topoanchor.data.datasets.manifest_dataset import ManifestSegmentationDataset
from topoanchor.evaluation.segmentation_metrics import dice_iou_per_class
from topoanchor.evaluation.surface_metrics import surface_metrics_per_class
from topoanchor.evaluation.topology_metrics import topology_vector_distance


def _load_module(cfg: DictConfig, device: torch.device):
    from topoanchor.training.lightning_module import TopologyAnchoredLightningModule

    if cfg.paths.checkpoint_path is None:
        raise ValueError("Set paths.checkpoint_path to evaluate a trained checkpoint.")
    module = TopologyAnchoredLightningModule(cfg)
    checkpoint = torch.load(cfg.paths.checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    module.load_state_dict(state_dict)
    module.to(device)
    module.eval()
    return module


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = _load_module(cfg, device)
    manifests = [Path(cfg.data.test_manifest)]
    ood_manifest = Path(cfg.paths.ood_manifest)
    if ood_manifest.exists() and ood_manifest.resolve() != manifests[0].resolve():
        manifests.append(ood_manifest)
    rows = []
    prediction_dir = Path(cfg.paths.prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for manifest in manifests:
            manifest_frame = read_manifest_frame(manifest)
            metadata_by_sample_id = {
                str(row.sample_id): row._asdict() for row in manifest_frame.itertuples(index=False)
            }
            dataset = ManifestSegmentationDataset(
                manifest,
                split="test",
                patch_size=list(cfg.data.patch_size),
                spacing=None if cfg.data.spacing is None else list(cfg.data.spacing),
                require_masks=True,
                require_topology_cache=bool(cfg.data.require_topology_cache),
                topology_descriptor_version=str(cfg.data.topology_descriptor_version),
                normalize_nonzero=bool(cfg.data.intensity.normalize_nonzero),
                spatial_prob=0.0,
                intensity_prob=0.0,
            )
            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=int(cfg.data.num_workers),
                pin_memory=True,
            )
            for batch in loader:
                image = batch["image"].to(device)
                mask = batch["mask"].cpu().numpy()
                if mask.ndim == 5 and mask.shape[1] == 1:
                    mask = mask[:, 0]
                output = module(image)
                prob = output["prob"].detach().cpu()
                pred = prob.argmax(dim=1).numpy()
                prob_np = prob.numpy()
                raw_conf = raw_segmentation_confidence(prob).numpy()
                cal_conf = calibrated_confidence(
                    raw_segmentation_confidence(prob),
                    output["mahalanobis"].detach().cpu(),
                    gamma=float(cfg.eval.calibration_gamma),
                ).numpy()
                for idx, sample_id in enumerate(batch["sample_id"]):
                    metrics = dice_iou_per_class(
                        pred[idx],
                        mask[idx],
                        num_classes=int(cfg.data.num_classes),
                        include_background=bool(cfg.loss.seg.include_background),
                    )
                    if bool(cfg.eval.compute_surface_metrics):
                        metrics.update(
                            surface_metrics_per_class(
                                pred[idx],
                                mask[idx],
                                num_classes=int(cfg.data.num_classes),
                            )
                        )
                    if bool(cfg.eval.topology_compare_predictions):
                        metrics["topology_vector_distance"] = topology_vector_distance(
                            pred[idx],
                            mask[idx],
                            sample_id=str(sample_id),
                            num_classes=int(cfg.data.num_classes),
                            descriptor_version=str(cfg.data.topology_descriptor_version),
                        )
                    voxel_conf, voxel_correct = voxel_confidence_correctness(prob_np[idx], mask[idx])
                    metrics["ece"] = expected_calibration_error(
                        voxel_conf,
                        voxel_correct,
                        num_bins=int(cfg.eval.ece_bins),
                    )
                    metrics["brier"] = brier_score(
                        prob_np[idx],
                        mask[idx],
                        num_classes=int(cfg.data.num_classes),
                    )
                    metadata = metadata_by_sample_id.get(str(sample_id), {})
                    for column in cfg.eval.domain_columns:
                        if column in metadata:
                            metrics[str(column)] = metadata[column]
                    metrics.update(
                        {
                            "sample_id": str(sample_id),
                            "manifest": str(manifest),
                            "raw_confidence": float(raw_conf[idx]),
                            "calibrated_confidence": float(cal_conf[idx]),
                            "mahalanobis": float(output["mahalanobis"][idx].detach().cpu()),
                        }
                    )
                    rows.append(metrics)

    output_dir = Path(cfg.paths.output_dir) / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    csv_path = output_dir / "evaluation_metrics.csv"
    json_path = output_dir / "evaluation_summary.json"
    frame.to_csv(csv_path, index=False)
    summary = frame.select_dtypes(include=[np.number]).mean(numeric_only=True).to_dict()
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
