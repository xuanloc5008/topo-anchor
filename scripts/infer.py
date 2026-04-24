from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def _save_prediction(prediction: np.ndarray, image_path: str, output_path: Path) -> None:
    import nibabel as nib

    source = nib.load(image_path)
    affine = source.affine if tuple(source.shape[: prediction.ndim]) == tuple(prediction.shape) else np.eye(4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(prediction.astype(np.int16), affine), str(output_path))


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    from topoanchor.data.datasets.inference_dataset import ManifestInferenceDataset
    from topoanchor.training.lightning_module import TopologyAnchoredLightningModule

    if cfg.paths.infer_manifest is None:
        raise ValueError("Set paths.infer_manifest to a real NIfTI manifest for inference.")
    if cfg.paths.checkpoint_path is None:
        raise ValueError("Set paths.checkpoint_path to a trained checkpoint for inference.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = TopologyAnchoredLightningModule(cfg)
    checkpoint = torch.load(cfg.paths.checkpoint_path, map_location=device)
    module.load_state_dict(checkpoint.get("state_dict", checkpoint))
    module.to(device)
    module.eval()

    dataset = ManifestInferenceDataset(
        cfg.paths.infer_manifest,
        spacing=None if cfg.data.spacing is None else list(cfg.data.spacing),
        normalize_nonzero=bool(cfg.data.intensity.normalize_nonzero),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(cfg.data.num_workers))
    output_dir = Path(cfg.paths.prediction_dir)
    with torch.no_grad():
        for batch in loader:
            output = module(batch["image"].to(device))
            pred = output["prob"].argmax(dim=1).detach().cpu().numpy()[0]
            sample_id = str(batch["sample_id"][0])
            path = output_dir / f"{sample_id}_prediction.nii.gz"
            _save_prediction(pred, batch["image_path"][0], path)
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
