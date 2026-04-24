from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from topoanchor.data.manifest import REQUIRED_MANIFEST_COLUMNS
from topoanchor.data.nifti import get_nifti_metadata
from topoanchor.data.splits import split_sample_ids, write_split_json


NIFTI_EXTENSIONS = (".nii.gz", ".nii")


@dataclass(slots=True)
class PairedNiftiSample:
    sample_id: str
    image_path: Path
    mask_path: Path


def nifti_stem(path: str | Path) -> str:
    name = Path(path).name
    for suffix in NIFTI_EXTENSIONS:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def sample_id_from_mask(path: str | Path, mask_suffixes: list[str]) -> str:
    stem = nifti_stem(path)
    for suffix in mask_suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def discover_nifti_files(root: str | Path, globs: list[str]) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"NIfTI directory does not exist: {root}")
    files: list[Path] = []
    for pattern in globs:
        files.extend(path for path in root.rglob(pattern) if path.is_file())
    unique = sorted(set(files))
    if not unique:
        raise FileNotFoundError(f"No NIfTI files found in {root} with patterns {globs}.")
    return unique


def pair_image_mask_files(
    image_dir: str | Path,
    mask_dir: str | Path,
    *,
    image_globs: list[str],
    mask_globs: list[str],
    mask_suffixes: list[str],
) -> list[PairedNiftiSample]:
    images = discover_nifti_files(image_dir, image_globs)
    masks = discover_nifti_files(mask_dir, mask_globs)
    image_by_id = {nifti_stem(path): path.resolve() for path in images}
    mask_by_id = {sample_id_from_mask(path, mask_suffixes): path.resolve() for path in masks}
    common_ids = sorted(set(image_by_id) & set(mask_by_id))
    if not common_ids:
        raise ValueError(
            "No image/mask pairs found. Check image names, mask names, and data.manifest_builder.mask_suffixes."
        )
    return [
        PairedNiftiSample(
            sample_id=sample_id,
            image_path=image_by_id[sample_id],
            mask_path=mask_by_id[sample_id],
        )
        for sample_id in common_ids
    ]


def build_manifest_rows(
    samples: list[PairedNiftiSample],
    *,
    dataset_name: str,
    split_by_sample_id: dict[str, str],
    descriptor_version: str,
    metadata_by_sample_id: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    rows = []
    for sample in samples:
        image_meta = get_nifti_metadata(sample.image_path)
        mask_meta = get_nifti_metadata(sample.mask_path)
        if image_meta["spatial_shape"] != mask_meta["spatial_shape"]:
            raise ValueError(
                f"Image/mask shape mismatch for {sample.sample_id}: "
                f"{image_meta['spatial_shape']} vs {mask_meta['spatial_shape']}"
            )
        row = {
            "sample_id": sample.sample_id,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
            "split": split_by_sample_id[sample.sample_id],
            "dataset_name": dataset_name,
            "spacing": " ".join(str(value) for value in image_meta["spacing"]),
            "shape": " ".join(str(value) for value in image_meta["spatial_shape"]),
            "orientation": image_meta["orientation"],
            "roi_bbox": "",
            "topology_cache_path": "",
            "topology_descriptor_version": descriptor_version,
        }
        if metadata_by_sample_id is not None:
            row.update(metadata_by_sample_id.get(sample.sample_id, {}))
        rows.append(row)
    frame = pd.DataFrame(rows)
    ordered_columns = REQUIRED_MANIFEST_COLUMNS + [
        column for column in frame.columns if column not in REQUIRED_MANIFEST_COLUMNS
    ]
    return frame[ordered_columns]


def load_manifest_metadata(
    metadata_csv: str | Path | None,
    *,
    sample_id_column: str,
) -> dict[str, dict[str, str]]:
    if metadata_csv is None or str(metadata_csv).strip() in {"", "None", "null"}:
        return {}
    metadata_path = Path(metadata_csv).expanduser()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV does not exist: {metadata_path}")
    frame = pd.read_csv(metadata_path)
    if sample_id_column not in frame.columns:
        raise ValueError(f"{metadata_path} is missing sample id column `{sample_id_column}`.")
    metadata: dict[str, dict[str, str]] = {}
    for row in frame.itertuples(index=False):
        sample_id = str(getattr(row, sample_id_column))
        values = {}
        for column in frame.columns:
            if column == sample_id_column:
                continue
            value = getattr(row, column)
            values[column] = "" if pd.isna(value) else str(value)
        metadata[sample_id] = values
    return metadata


def write_split_manifests(
    *,
    image_dir: str | Path,
    mask_dir: str | Path,
    output_dir: str | Path,
    image_globs: list[str],
    mask_globs: list[str],
    mask_suffixes: list[str],
    dataset_name: str,
    descriptor_version: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    split_output_path: str | Path | None = None,
    metadata_csv: str | Path | None = None,
    metadata_sample_id_column: str = "sample_id",
    vendor_column: str = "vendor",
    vendor_holdout: list[str] | None = None,
    ood_split_name: str = "ood",
) -> dict[str, Path]:
    samples = pair_image_mask_files(
        image_dir,
        mask_dir,
        image_globs=image_globs,
        mask_globs=mask_globs,
        mask_suffixes=mask_suffixes,
    )
    splits = split_sample_ids(
        [sample.sample_id for sample in samples],
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    metadata_by_sample_id = load_manifest_metadata(
        metadata_csv,
        sample_id_column=metadata_sample_id_column,
    )
    holdout_vendors = {str(vendor) for vendor in (vendor_holdout or [])}
    if holdout_vendors:
        ood_ids = []
        for sample in samples:
            vendor = metadata_by_sample_id.get(sample.sample_id, {}).get(vendor_column, "")
            if vendor in holdout_vendors:
                ood_ids.append(sample.sample_id)
        if not ood_ids:
            raise ValueError(
                f"No samples matched vendor_holdout={sorted(holdout_vendors)} using column `{vendor_column}`."
            )
        for split_name in ["train", "val", "test"]:
            splits[split_name] = [sample_id for sample_id in splits[split_name] if sample_id not in ood_ids]
        splits[ood_split_name] = sorted(ood_ids)
        if not splits["train"]:
            raise ValueError("Vendor holdout removed all training samples.")
    outputs = {}
    if split_output_path is not None:
        outputs["splits"] = write_split_json(splits, split_output_path)
    split_by_sample_id = {
        sample_id: split_name for split_name, ids in splits.items() for sample_id in ids
    }
    frame = build_manifest_rows(
        samples,
        dataset_name=dataset_name,
        split_by_sample_id=split_by_sample_id,
        descriptor_version=descriptor_version,
        metadata_by_sample_id=metadata_by_sample_id,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test", ood_split_name]:
        if split_name not in splits:
            continue
        split_frame = frame[frame["split"] == split_name]
        path = output_dir / f"{split_name}.csv"
        split_frame.to_csv(path, index=False)
        outputs[split_name] = path
    return outputs
