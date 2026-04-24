from __future__ import annotations

from topoanchor.analysis.topology.descriptor_stats import records_to_descriptor_frame
from topoanchor.topology.schema import (
    ClassTopologyDescriptor,
    TopologyCacheRecord,
    TopologyDescriptor,
)


def test_descriptor_stats_frame_contains_foreground_and_class_columns() -> None:
    aggregate = ClassTopologyDescriptor("foreground", 10, 0.5, 1, 1, 0, 10.0, 10.0, 0.8)
    class_one = ClassTopologyDescriptor(1, 5, 0.25, 1, 1, 0, 5.0, 5.0, 0.6)
    descriptor = TopologyDescriptor(
        sample_id="case001",
        descriptor_version="v1",
        num_classes=2,
        class_descriptors=[class_one],
        aggregate_descriptor=aggregate,
    )
    record = TopologyCacheRecord(
        sample_id="case001",
        descriptor_version="v1",
        mask_path="/tmp/mask.nii.gz",
        num_classes=2,
        descriptor=descriptor,
        vector=descriptor.vector().tolist(),
        signature=descriptor.signature(),
    )
    frame = records_to_descriptor_frame([record])
    assert frame.loc[0, "sample_id"] == "case001"
    assert frame.loc[0, "foreground_components"] == 1
    assert frame.loc[0, "class_1_components"] == 1
