from __future__ import annotations

import torch
import torch.nn.functional as F


def build_pair_masks_from_vectors(
    topology_vectors: torch.Tensor,
    *,
    positive_distance: float,
    negative_distance: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if topology_vectors.ndim != 2:
        raise ValueError(f"Expected topology vectors [B, F], got {tuple(topology_vectors.shape)}.")
    batch_size = topology_vectors.shape[0]
    if batch_size < 2:
        empty = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=topology_vectors.device)
        return empty, empty

    if topology_vectors.shape[1] == 0:
        raise ValueError("Topology vectors are empty; precompute topology descriptors before training.")

    vectors = topology_vectors.float()
    vectors = torch.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    vectors = torch.sign(vectors) * torch.log1p(torch.abs(vectors))
    vectors = F.normalize(vectors, p=2, dim=1)
    distances = torch.cdist(vectors, vectors, p=2)
    not_self = ~torch.eye(batch_size, dtype=torch.bool, device=topology_vectors.device)
    positive_mask = (distances <= positive_distance) & not_self
    if negative_distance is None:
        negative_mask = (~positive_mask) & not_self
    else:
        negative_mask = (distances >= negative_distance) & not_self
    return positive_mask, negative_mask
