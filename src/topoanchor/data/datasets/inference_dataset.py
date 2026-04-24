from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from topoanchor.data.manifest import load_manifest_records
from topoanchor.data.transforms.monai_transforms import build_monai_image_transforms


class ManifestInferenceDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        spacing: list[float] | tuple[float, float, float] | None,
        normalize_nonzero: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = load_manifest_records(
            self.manifest_path,
            require_masks=False,
            require_topology_cache=False,
        )
        if not self.records:
            raise ValueError(f"Manifest contains no samples: {self.manifest_path}")
        self.transforms = build_monai_image_transforms(
            spacing=spacing,
            normalize_nonzero=normalize_nonzero,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        transformed = self.transforms({"image": str(record.image_path)})
        return {
            "image": transformed["image"].float(),
            "sample_id": record.sample_id,
            "image_path": str(record.image_path),
        }
