from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    prediction_dir = Path(cfg.paths.prediction_dir)
    export_dir = Path(cfg.paths.output_dir) / "exported_predictions"
    export_dir.mkdir(parents=True, exist_ok=True)
    predictions = list(prediction_dir.glob("*.nii*"))
    if not predictions:
        raise FileNotFoundError(f"No NIfTI predictions found in {prediction_dir}.")
    for path in predictions:
        target = export_dir / path.name
        shutil.copy2(path, target)
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
