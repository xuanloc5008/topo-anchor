from __future__ import annotations

import torch

from topoanchor.topology.pair_builder import build_pair_masks_from_vectors
from topoanchor.topology.schema import ClassTopologyDescriptor, TopologyDescriptor


def test_topology_descriptor_vector_includes_aggregate_and_classes() -> None:
    aggregate = ClassTopologyDescriptor("foreground", 10, 0.5, 1, 1, 0, 10.0, 10.0, 0.8)
    class_one = ClassTopologyDescriptor(1, 5, 0.25, 1, 1, 0, 5.0, 5.0, 0.6)
    descriptor = TopologyDescriptor(
        sample_id="s1",
        descriptor_version="v1",
        num_classes=2,
        class_descriptors=[class_one],
        aggregate_descriptor=aggregate,
    )
    vector = descriptor.vector()
    assert vector.shape == (14,)
    assert descriptor.signature().startswith("foreground:cc=1")


def test_pair_builder_constructs_positive_mask() -> None:
    vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.01, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    positive, negative = build_pair_masks_from_vectors(
        vectors,
        positive_distance=0.1,
        negative_distance=0.5,
    )
    assert positive[0, 1]
    assert positive[1, 0]
    assert not positive[0, 0]
    assert negative[0, 2]
