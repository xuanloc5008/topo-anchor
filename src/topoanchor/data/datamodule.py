from __future__ import annotations

from torch.utils.data import DataLoader

from topoanchor.data.datasets.manifest_dataset import ManifestSegmentationDataset
from topoanchor.data.samplers.topology_sampler import TopologyBalancedBatchSampler
from topoanchor.topology.cache import read_cache_record
from topoanchor.utils.imports import require_package


class ManifestDataModule(require_package("lightning", "pip install lightning").LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        data = self.cfg.data
        common = dict(
            patch_size=list(data.patch_size),
            spacing=None if data.spacing is None else list(data.spacing),
            require_masks=True,
            require_topology_cache=bool(data.require_topology_cache),
            topology_descriptor_version=str(data.topology_descriptor_version),
            normalize_nonzero=bool(data.intensity.normalize_nonzero),
            spatial_prob=float(data.augmentation.spatial_prob),
            intensity_prob=float(data.augmentation.intensity_prob),
        )
        if stage in (None, "fit"):
            self.train_dataset = ManifestSegmentationDataset(
                data.train_manifest,
                split="train",
                **common,
            )
            self.val_dataset = ManifestSegmentationDataset(
                data.val_manifest,
                split="val",
                **common,
            )
        if stage in (None, "test"):
            self.test_dataset = ManifestSegmentationDataset(
                data.test_manifest,
                split="test",
                **common,
            )

    def _loader_kwargs(self, *, shuffle: bool = False, drop_last: bool = False) -> dict:
        data = self.cfg.data
        num_workers = int(data.num_workers)
        kwargs = {
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": bool(data.loader.pin_memory),
            "drop_last": drop_last,
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = bool(data.loader.persistent_workers)
            kwargs["prefetch_factor"] = int(data.loader.prefetch_factor)
        return kwargs

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup("fit")
        data = self.cfg.data
        if bool(data.sampler.topology_balanced):
            signatures = []
            for record in self.train_dataset.records:
                if record.topology_cache_path is not None and record.topology_cache_path.exists():
                    signatures.append(read_cache_record(record.topology_cache_path).signature)
                else:
                    signatures.append(record.sample_id)
            sampler = TopologyBalancedBatchSampler(
                signatures,
                batch_size=int(data.batch_size),
                samples_per_topology=int(data.sampler.samples_per_topology),
                seed=int(self.cfg.seed),
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                num_workers=int(data.num_workers),
                pin_memory=bool(data.loader.pin_memory),
                persistent_workers=bool(data.loader.persistent_workers)
                if int(data.num_workers) > 0
                else False,
                prefetch_factor=int(data.loader.prefetch_factor)
                if int(data.num_workers) > 0
                else None,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=int(data.batch_size),
            **self._loader_kwargs(shuffle=True, drop_last=bool(data.loader.drop_last)),
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            **self._loader_kwargs(shuffle=False, drop_last=False),
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            self.setup("test")
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            **self._loader_kwargs(shuffle=False, drop_last=False),
        )
