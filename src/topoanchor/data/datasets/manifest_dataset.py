from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from topoanchor.data.manifest import ManifestRecord, load_manifest_records
from topoanchor.data.transforms.monai_transforms import build_monai_transforms
from topoanchor.topology.cache import read_cache_record


class ManifestSegmentationDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        patch_size: list[int] | tuple[int, int, int],
        spacing: list[float] | tuple[float, float, float] | None,
        require_masks: bool = True,
        require_topology_cache: bool = False,
        topology_descriptor_version: str | None = None,
        normalize_nonzero: bool = True,
        spatial_prob: float = 0.2,
        intensity_prob: float = 0.15,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.records: list[ManifestRecord] = load_manifest_records(
            self.manifest_path,
            require_masks=require_masks,
            require_topology_cache=require_topology_cache,
            expected_topology_version=topology_descriptor_version if require_topology_cache else None,
        )
        if not self.records:
            raise ValueError(f"Manifest contains no samples: {self.manifest_path}")
        self.transforms = build_monai_transforms(
            split=split,
            patch_size=patch_size,
            spacing=spacing,
            normalize_nonzero=normalize_nonzero,
            spatial_prob=spatial_prob,
            intensity_prob=intensity_prob,
        )

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _remap_mask(mask: torch.Tensor, label_map: str) -> torch.Tensor:
        label_map = str(label_map or "").strip()
        if label_map in {"", "identity", "none", "None"}:
            return mask
        output = mask.clone()
        source = mask.clone()
        for item in label_map.split(","):
            if not item.strip():
                continue
            left, right = item.split(":")
            output[source == int(left.strip())] = int(right.strip())
        return output

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        item = {"image": str(record.image_path), "mask": str(record.mask_path)}
        transformed = self.transforms(item)
        if isinstance(transformed, list):
            transformed = transformed[0]

        topology_vector = torch.empty(0, dtype=torch.float32)
        topology_signature = ""
        if record.topology_cache_path is not None and record.topology_cache_path.exists():
            cache_record = read_cache_record(record.topology_cache_path)
            topology_vector = torch.tensor(cache_record.vector, dtype=torch.float32)
            topology_signature = cache_record.signature

        return {
            "image": transformed["image"].float(),
            "mask": self._remap_mask(
                transformed["mask"].long(),
                record.metadata.get("label_map", "identity"),
            ),
            "sample_id": record.sample_id,
            "image_path": str(record.image_path),
            "mask_path": str(record.mask_path) if record.mask_path else "",
            "topology_vector": topology_vector,
            "topology_signature": topology_signature,
        }
