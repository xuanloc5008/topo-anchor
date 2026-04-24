from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator

from torch.utils.data import Sampler


class TopologyBalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        signatures: list[str],
        *,
        batch_size: int,
        samples_per_topology: int = 1,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if samples_per_topology < 1:
            raise ValueError("samples_per_topology must be positive.")
        self.signatures = signatures
        self.batch_size = batch_size
        self.samples_per_topology = samples_per_topology
        self.seed = seed
        self.groups: dict[str, list[int]] = defaultdict(list)
        for index, signature in enumerate(signatures):
            self.groups[signature or f"sample-{index}"].append(index)

    def __len__(self) -> int:
        return max(len(self.signatures) // self.batch_size, 1)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        signatures = list(self.groups)
        for _ in range(len(self)):
            rng.shuffle(signatures)
            batch: list[int] = []
            for signature in signatures:
                candidates = self.groups[signature]
                picks = rng.sample(candidates, k=min(self.samples_per_topology, len(candidates)))
                batch.extend(picks)
                if len(batch) >= self.batch_size:
                    break
            if len(batch) < self.batch_size:
                remaining = list(range(len(self.signatures)))
                rng.shuffle(remaining)
                batch.extend(remaining[: self.batch_size - len(batch)])
            yield batch[: self.batch_size]
