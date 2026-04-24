from __future__ import annotations

from collections.abc import Sequence

import torch

from topoanchor.utils.imports import require_package


class EnsureMaskLongd:
    def __init__(self, key: str = "mask") -> None:
        self.key = key

    def __call__(self, data):
        mask = data[self.key]
        if isinstance(mask, torch.Tensor):
            if mask.ndim >= 4 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            data[self.key] = mask.long()
        return data


def build_monai_transforms(
    *,
    split: str,
    patch_size: Sequence[int],
    spacing: Sequence[float] | None = None,
    normalize_nonzero: bool = True,
    spatial_prob: float = 0.2,
    intensity_prob: float = 0.15,
):
    transforms = require_package("monai.transforms", "pip install monai")

    keys = ["image", "mask"]
    pipeline = [
        transforms.LoadImaged(keys=keys, image_only=True),
        transforms.EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        transforms.Orientationd(keys=keys, axcodes="RAS"),
    ]
    if spacing is not None:
        pipeline.append(
            transforms.Spacingd(
                keys=keys,
                pixdim=tuple(float(value) for value in spacing),
                mode=("bilinear", "nearest"),
            )
        )
    pipeline.append(transforms.NormalizeIntensityd(keys=["image"], nonzero=normalize_nonzero, channel_wise=True))

    if split == "train":
        pipeline.extend(
            [
                transforms.RandCropByPosNegLabeld(
                    keys=keys,
                    label_key="mask",
                    spatial_size=tuple(int(value) for value in patch_size),
                    pos=1,
                    neg=1,
                    num_samples=1,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandFlipd(keys=keys, prob=spatial_prob, spatial_axis=0),
                transforms.RandFlipd(keys=keys, prob=spatial_prob, spatial_axis=1),
                transforms.RandFlipd(keys=keys, prob=spatial_prob, spatial_axis=2),
                transforms.RandScaleIntensityd(keys=["image"], factors=0.1, prob=intensity_prob),
                transforms.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=intensity_prob),
            ]
        )

    pipeline.extend([transforms.EnsureTyped(keys=keys), EnsureMaskLongd("mask")])
    return transforms.Compose(pipeline)


def build_monai_image_transforms(
    *,
    spacing: Sequence[float] | None = None,
    normalize_nonzero: bool = True,
):
    transforms = require_package("monai.transforms", "pip install monai")
    pipeline = [
        transforms.LoadImaged(keys=["image"], image_only=True),
        transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        transforms.Orientationd(keys=["image"], axcodes="RAS"),
    ]
    if spacing is not None:
        pipeline.append(
            transforms.Spacingd(
                keys=["image"],
                pixdim=tuple(float(value) for value in spacing),
                mode=("bilinear",),
            )
        )
    pipeline.extend(
        [
            transforms.NormalizeIntensityd(keys=["image"], nonzero=normalize_nonzero, channel_wise=True),
            transforms.EnsureTyped(keys=["image"]),
        ]
    )
    return transforms.Compose(pipeline)
