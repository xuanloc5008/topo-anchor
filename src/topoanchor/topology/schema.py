from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ClassTopologyDescriptor:
    label: int | str
    voxel_count: int
    voxel_fraction: float
    component_count: int
    euler_number: int
    hole_count_proxy: int
    mean_component_size: float
    max_component_size: float
    mean_extent: float

    def vector(self) -> list[float]:
        return [
            float(self.voxel_fraction),
            float(self.component_count),
            float(self.euler_number),
            float(self.hole_count_proxy),
            float(self.mean_component_size),
            float(self.max_component_size),
            float(self.mean_extent),
        ]

    def signature(self) -> str:
        return f"{self.label}:cc={self.component_count}:euler={self.euler_number}"


@dataclass(slots=True)
class PersistenceDescriptor:
    intervals_by_dim: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    top_lifetimes_by_dim: dict[int, list[float]] = field(default_factory=dict)
    vector: list[float] = field(default_factory=list)


@dataclass(slots=True)
class TopologyDescriptor:
    sample_id: str
    descriptor_version: str
    num_classes: int
    class_descriptors: list[ClassTopologyDescriptor]
    aggregate_descriptor: ClassTopologyDescriptor | None = None
    persistence: PersistenceDescriptor | None = None

    def vector(self) -> np.ndarray:
        values: list[float] = []
        if self.aggregate_descriptor is not None:
            values.extend(self.aggregate_descriptor.vector())
        for desc in sorted(
            self.class_descriptors,
            key=lambda item: int(item.label) if isinstance(item.label, int) else str(item.label),
        ):
            values.extend(desc.vector())
        if self.persistence is not None:
            values.extend(self.persistence.vector)
        return np.asarray(values, dtype=np.float32)

    def signature(self) -> str:
        parts = [desc.signature() for desc in self.class_descriptors]
        if self.aggregate_descriptor is not None:
            parts.insert(0, self.aggregate_descriptor.signature())
        return "|".join(parts)


@dataclass(slots=True)
class TopologyCacheRecord:
    sample_id: str
    descriptor_version: str
    mask_path: str
    num_classes: int
    descriptor: TopologyDescriptor
    vector: list[float]
    signature: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopologyCacheRecord":
        descriptor_data = data["descriptor"]
        class_descriptors = [
            ClassTopologyDescriptor(**item) for item in descriptor_data.get("class_descriptors", [])
        ]
        aggregate = descriptor_data.get("aggregate_descriptor")
        aggregate_descriptor = (
            ClassTopologyDescriptor(**aggregate) if isinstance(aggregate, dict) else None
        )
        persistence_data = descriptor_data.get("persistence")
        persistence = PersistenceDescriptor(**persistence_data) if persistence_data else None
        descriptor = TopologyDescriptor(
            sample_id=descriptor_data["sample_id"],
            descriptor_version=descriptor_data["descriptor_version"],
            num_classes=int(descriptor_data["num_classes"]),
            class_descriptors=class_descriptors,
            aggregate_descriptor=aggregate_descriptor,
            persistence=persistence,
        )
        return cls(
            sample_id=data["sample_id"],
            descriptor_version=data["descriptor_version"],
            mask_path=data["mask_path"],
            num_classes=int(data["num_classes"]),
            descriptor=descriptor,
            vector=[float(value) for value in data.get("vector", [])],
            signature=data.get("signature", descriptor.signature()),
            metadata=data.get("metadata", {}),
        )
