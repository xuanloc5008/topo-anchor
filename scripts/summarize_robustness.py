from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra
from omegaconf import DictConfig

from topoanchor.evaluation.robustness_metrics import write_robustness_tables


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    evaluation_csv = Path(cfg.paths.output_dir) / "tables" / "evaluation_metrics.csv"
    if not evaluation_csv.exists():
        raise FileNotFoundError(f"Run scripts/evaluate.py first: {evaluation_csv}")
    domain_path, local_shift_path = write_robustness_tables(
        evaluation_csv,
        output_dir=Path(cfg.paths.output_dir) / "tables",
        domain_columns=[str(column) for column in cfg.eval.domain_columns],
    )
    print(f"Wrote {domain_path}")
    print(f"Wrote {local_shift_path}")


if __name__ == "__main__":
    main()
