from __future__ import annotations

import numpy as np

from topoanchor.utils.imports import require_package
from topoanchor.topology.schema import ClassTopologyDescriptor, TopologyDescriptor


def _class_descriptor(mask: np.ndarray, label_value: int | str, binary: np.ndarray) -> ClassTopologyDescriptor:
    measure = require_package("skimage.measure", "pip install scikit-image")
    binary = np.asarray(binary).astype(bool)
    total_voxels = int(binary.size)
    voxel_count = int(binary.sum())
    if voxel_count == 0:
        return ClassTopologyDescriptor(
            label=label_value,
            voxel_count=0,
            voxel_fraction=0.0,
            component_count=0,
            euler_number=0,
            hole_count_proxy=0,
            mean_component_size=0.0,
            max_component_size=0.0,
            mean_extent=0.0,
        )

    connectivity = binary.ndim
    labeled = measure.label(binary, connectivity=connectivity)
    component_count = int(labeled.max())
    euler = int(measure.euler_number(binary, connectivity=connectivity))
    props = measure.regionprops(labeled)
    component_sizes = [float(prop.area) for prop in props]
    extents = [float(getattr(prop, "extent", 0.0)) for prop in props]
    mean_component_size = float(np.mean(component_sizes)) if component_sizes else 0.0
    max_component_size = float(np.max(component_sizes)) if component_sizes else 0.0
    mean_extent = float(np.mean(extents)) if extents else 0.0
    if binary.ndim == 2:
        hole_count_proxy = max(component_count - euler, 0)
    else:
        hole_count_proxy = max(abs(euler - component_count), 0)

    return ClassTopologyDescriptor(
        label=label_value,
        voxel_count=voxel_count,
        voxel_fraction=float(voxel_count / max(total_voxels, 1)),
        component_count=component_count,
        euler_number=euler,
        hole_count_proxy=int(hole_count_proxy),
        mean_component_size=mean_component_size,
        max_component_size=max_component_size,
        mean_extent=mean_extent,
    )


def compute_skimage_topology_descriptor(
    mask: np.ndarray,
    *,
    sample_id: str,
    num_classes: int,
    descriptor_version: str,
    include_aggregate_foreground: bool = True,
) -> TopologyDescriptor:
    mask = np.asarray(mask)
    if mask.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D mask, got shape {mask.shape}.")
    if num_classes < 2:
        raise ValueError("Multiclass topology descriptors require num_classes >= 2.")

    class_descriptors = [
        _class_descriptor(mask, label_value=class_id, binary=(mask == class_id))
        for class_id in range(1, num_classes)
    ]
    aggregate_descriptor = None
    if include_aggregate_foreground:
        aggregate_descriptor = _class_descriptor(mask, label_value="foreground", binary=(mask > 0))

    return TopologyDescriptor(
        sample_id=sample_id,
        descriptor_version=descriptor_version,
        num_classes=num_classes,
        class_descriptors=class_descriptors,
        aggregate_descriptor=aggregate_descriptor,
    )
