from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra
import pandas as pd
from omegaconf import DictConfig

from topoanchor.analysis.topology.descriptor_stats import (
    load_topology_records,
    records_to_descriptor_frame,
    write_descriptor_summary,
)
from topoanchor.analysis.topology.topology_vs_confidence import topology_confidence_correlations
from topoanchor.analysis.topology.topology_vs_error import topology_error_correlations


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cache_dir = Path(cfg.paths.topology_cache_dir)
    output_dir = Path(cfg.paths.output_dir) / "topology_analysis"
    csv_path, json_path = write_descriptor_summary(cache_dir, output_dir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")

    evaluation_path = Path(cfg.paths.output_dir) / "tables" / "evaluation_metrics.csv"
    if evaluation_path.exists():
        records = load_topology_records(cache_dir)
        descriptor_frame = records_to_descriptor_frame(records)
        evaluation_frame = pd.read_csv(evaluation_path)
        error_path = output_dir / "topology_vs_error_correlations.csv"
        confidence_path = output_dir / "topology_vs_confidence_correlations.csv"
        topology_error_correlations(descriptor_frame, evaluation_frame).to_csv(error_path, index=False)
        topology_confidence_correlations(descriptor_frame, evaluation_frame).to_csv(
            confidence_path, index=False
        )
        print(f"Wrote {error_path}")
        print(f"Wrote {confidence_path}")


if __name__ == "__main__":
    main()
