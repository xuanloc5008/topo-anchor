from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from topoanchor.data.manifest import REQUIRED_MANIFEST_COLUMNS
from topoanchor.data.nifti import get_nifti_metadata
from topoanchor.data.splits import split_sample_ids, write_split_json
from topoanchor.utils.imports import require_package


TARGET_LABEL_SCHEMA = "0=background,1=LV,2=MYO,3=RV"
IDENTITY_LABEL_MAP = "identity"
ACDC_TO_TARGET = {1: 3, 2: 2, 3: 1}


@dataclass(slots=True)
class CardiacChallengeSample:
    sample_id: str
    patient_id: str
    image_path: Path
    mask_path: Path
    source_dataset: str
    source_split: str
    phase: str
    view: str
    vendor: str
    site_id: str
    protocol: str
    label_map: str


def _read_info_cfg(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def _metadata_lookup(
    path: str | Path | None,
    *,
    sample_id_columns: Iterable[str],
) -> dict[str, dict[str, str]]:
    if path is None or str(path).strip() in {"", "None", "null"}:
        return {}
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV does not exist: {path}")
    frame = pd.read_csv(path)
    sample_column = next((column for column in sample_id_columns if column in frame.columns), None)
    if sample_column is None:
        raise ValueError(
            f"{path} must contain one of these sample id columns: {list(sample_id_columns)}"
        )
    lookup: dict[str, dict[str, str]] = {}
    for _, row in frame.iterrows():
        sample_id = str(row[sample_column])
        values = {}
        for column in frame.columns:
            value = row[column]
            values[column] = "" if pd.isna(value) else str(value)
        lookup[sample_id] = values
    return lookup


def _first_metadata_value(
    metadata: dict[str, str],
    columns: Iterable[str],
    *,
    default: str,
) -> str:
    for column in columns:
        value = metadata.get(column)
        if value not in (None, ""):
            return str(value)
    return default


def _remap_labels(mask: np.ndarray, label_map: dict[int, int] | None) -> np.ndarray:
    mask = np.rint(mask).astype(np.int16)
    if not label_map:
        return mask
    output = np.zeros_like(mask, dtype=np.int16)
    output[mask == 0] = 0
    for source, target in label_map.items():
        output[mask == int(source)] = int(target)
    unknown = (mask != 0) & ~np.isin(mask, list(label_map))
    if unknown.any():
        output[unknown] = mask[unknown]
    return output


def _save_3d_pair(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    affine: np.ndarray,
    output_dir: Path,
    sample_id: str,
    label_map: dict[int, int] | None,
) -> tuple[Path, Path]:
    nib = require_package("nibabel", "pip install nibabel")
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{sample_id}.nii.gz"
    mask_path = mask_dir / f"{sample_id}_gt.nii.gz"
    remapped = _remap_labels(mask, label_map)
    nib.save(nib.Nifti1Image(np.asarray(image, dtype=np.float32), affine), str(image_path))
    nib.save(nib.Nifti1Image(remapped.astype(np.int16), affine), str(mask_path))
    return image_path.resolve(), mask_path.resolve()


def _load_nifti(path: Path):
    nib = require_package("nibabel", "pip install nibabel")
    nii = nib.load(str(path))
    return nii, np.asarray(nii.get_fdata())


def collect_acdc_samples(
    root: str | Path | None,
    *,
    output_dir: str | Path,
    include_testing: bool,
) -> list[CardiacChallengeSample]:
    if root is None or str(root).strip() in {"", "None", "null"}:
        return []
    root = Path(root).expanduser()
    database = root / "database" if (root / "database").exists() else root
    split_names = ["training"] + (["testing"] if include_testing else [])
    samples: list[CardiacChallengeSample] = []
    for source_split in split_names:
        split_dir = database / source_split
        if not split_dir.exists():
            continue
        for patient_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            patient_id = patient_dir.name
            info = _read_info_cfg(patient_dir / "Info.cfg")
            frame_to_phase = {}
            if "ED" in info:
                frame_to_phase[f"{int(info['ED']):02d}"] = "ED"
            if "ES" in info:
                frame_to_phase[f"{int(info['ES']):02d}"] = "ES"
            gt_files = sorted(patient_dir.glob(f"{patient_id}_frame*_gt.nii.gz"))
            for gt_path in gt_files:
                frame_token = gt_path.name.split("_frame", 1)[1].split("_gt", 1)[0]
                image_path = patient_dir / f"{patient_id}_frame{frame_token}.nii.gz"
                if not image_path.exists():
                    continue
                phase = frame_to_phase.get(frame_token, f"frame{frame_token}")
                sample_id = f"acdc_{patient_id}_{phase.lower()}"
                image_nii, image = _load_nifti(image_path)
                _, mask = _load_nifti(gt_path)
                out_image, out_mask = _save_3d_pair(
                    image,
                    mask,
                    affine=image_nii.affine,
                    output_dir=Path(output_dir) / "acdc",
                    sample_id=sample_id,
                    label_map=ACDC_TO_TARGET,
                )
                samples.append(
                    CardiacChallengeSample(
                        sample_id=sample_id,
                        patient_id=patient_id,
                        image_path=out_image,
                        mask_path=out_mask,
                        source_dataset="acdc",
                        source_split=source_split,
                        phase=phase,
                        view="SA",
                        vendor="ACDC_single_vendor",
                        site_id="ACDC",
                        protocol="ACDC_SA",
                        label_map=IDENTITY_LABEL_MAP,
                    )
                )
    return samples


def _iter_mnms1_case_dirs(root: Path, *, include_unlabelled: bool):
    for split_name in ["Training", "Validation", "Testing"]:
        split_dir = root / split_name
        if not split_dir.exists():
            continue
        child_dirs = []
        labelled = split_dir / "Labelled"
        if labelled.exists():
            child_dirs.extend(path for path in labelled.iterdir() if path.is_dir())
        if include_unlabelled:
            unlabelled = split_dir / "Unlabelled"
            if unlabelled.exists():
                child_dirs.extend(path for path in unlabelled.iterdir() if path.is_dir())
        child_dirs.extend(path for path in split_dir.iterdir() if path.is_dir() and path.name not in {"Labelled", "Unlabelled"})
        for case_dir in sorted(set(child_dirs)):
            yield split_name, case_dir


def collect_mnms1_samples(
    root: str | Path | None,
    *,
    output_dir: str | Path,
    metadata_csv: str | Path | None,
    metadata_sample_id_columns: Iterable[str],
    metadata_vendor_columns: Iterable[str],
    metadata_site_columns: Iterable[str],
    metadata_protocol_columns: Iterable[str],
    include_unlabelled: bool,
) -> list[CardiacChallengeSample]:
    if root is None or str(root).strip() in {"", "None", "null"}:
        return []
    root = Path(root).expanduser()
    metadata_lookup = _metadata_lookup(metadata_csv, sample_id_columns=metadata_sample_id_columns)
    samples: list[CardiacChallengeSample] = []
    for source_split, case_dir in _iter_mnms1_case_dirs(root, include_unlabelled=include_unlabelled):
        case_id = case_dir.name
        image_path = case_dir / f"{case_id}_sa.nii.gz"
        gt_path = case_dir / f"{case_id}_sa_gt.nii.gz"
        if not image_path.exists() or not gt_path.exists():
            continue
        image_nii, image = _load_nifti(image_path)
        _, mask = _load_nifti(gt_path)
        metadata = metadata_lookup.get(case_id, {})
        vendor = _first_metadata_value(metadata, metadata_vendor_columns, default=f"MMS1_prefix_{case_id[:1]}")
        site_id = _first_metadata_value(metadata, metadata_site_columns, default="MMS1_unknown_site")
        protocol = _first_metadata_value(metadata, metadata_protocol_columns, default="MMS1_SA")
        if image.ndim == 3:
            frame_indices = [0]
        elif image.ndim == 4:
            if mask.ndim == 4:
                frame_indices = [idx for idx in range(mask.shape[3]) if np.any(mask[..., idx] > 0)]
            else:
                frame_indices = [0]
        else:
            raise ValueError(f"Unsupported M&Ms1 image shape for {case_id}: {image.shape}")
        if not frame_indices:
            continue
        for frame_order, frame_idx in enumerate(frame_indices):
            phase = "ED" if frame_order == 0 else "ES" if frame_order == 1 else f"frame{frame_idx:02d}"
            sample_id = f"mnms1_{case_id}_{phase.lower()}"
            image_3d = image if image.ndim == 3 else image[..., frame_idx]
            mask_3d = mask if mask.ndim == 3 else mask[..., frame_idx]
            out_image, out_mask = _save_3d_pair(
                image_3d,
                mask_3d,
                affine=image_nii.affine,
                output_dir=Path(output_dir) / "mnms1",
                sample_id=sample_id,
                label_map=None,
            )
            samples.append(
                CardiacChallengeSample(
                    sample_id=sample_id,
                    patient_id=case_id,
                    image_path=out_image,
                    mask_path=out_mask,
                    source_dataset="mnms1",
                    source_split=source_split,
                    phase=phase,
                    view="SA",
                    vendor=vendor,
                    site_id=site_id,
                    protocol=protocol,
                    label_map=IDENTITY_LABEL_MAP,
                )
            )
    return samples


def collect_mnms2_samples(
    root: str | Path | None,
    *,
    output_dir: str | Path,
    metadata_csv: str | Path | None,
    metadata_sample_id_columns: Iterable[str],
    metadata_vendor_columns: Iterable[str],
    metadata_site_columns: Iterable[str],
    metadata_protocol_columns: Iterable[str],
    include_la: bool,
) -> list[CardiacChallengeSample]:
    if root is None or str(root).strip() in {"", "None", "null"}:
        return []
    root = Path(root).expanduser()
    dataset_root = root / "dataset" if (root / "dataset").exists() else root
    metadata_lookup = _metadata_lookup(metadata_csv, sample_id_columns=metadata_sample_id_columns)
    samples: list[CardiacChallengeSample] = []
    for patient_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        patient_id = patient_dir.name
        metadata = metadata_lookup.get(patient_id, {})
        vendor = _first_metadata_value(metadata, metadata_vendor_columns, default="MMS2_unknown_vendor")
        site_id = _first_metadata_value(metadata, metadata_site_columns, default="MMS2_unknown_site")
        protocol = _first_metadata_value(metadata, metadata_protocol_columns, default="MMS2_SA")
        views = ["SA"] + (["LA"] if include_la else [])
        for view in views:
            for phase in ["ED", "ES"]:
                image_path = patient_dir / f"{patient_id}_{view}_{phase}.nii.gz"
                gt_path = patient_dir / f"{patient_id}_{view}_{phase}_gt.nii.gz"
                if not image_path.exists() or not gt_path.exists():
                    continue
                image_nii, image = _load_nifti(image_path)
                _, mask = _load_nifti(gt_path)
                sample_id = f"mnms2_{patient_id}_{view.lower()}_{phase.lower()}"
                out_image, out_mask = _save_3d_pair(
                    image,
                    mask,
                    affine=image_nii.affine,
                    output_dir=Path(output_dir) / "mnms2",
                    sample_id=sample_id,
                    label_map=None,
                )
                samples.append(
                    CardiacChallengeSample(
                        sample_id=sample_id,
                        patient_id=patient_id,
                        image_path=out_image,
                        mask_path=out_mask,
                        source_dataset="mnms2",
                        source_split="external",
                        phase=phase,
                        view=view,
                        vendor=vendor,
                        site_id=site_id,
                        protocol=protocol,
                        label_map=IDENTITY_LABEL_MAP,
                    )
                )
    return samples


def _split_mnms1_samples(
    samples: list[CardiacChallengeSample],
    *,
    train_datasets: set[str],
    external_ood_datasets: set[str],
    mnms1_vendor_holdout: set[str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    split_by_sample_id: dict[str, str] = {}
    train_candidate_patients = sorted(
        {
            sample.patient_id
            for sample in samples
            if sample.source_dataset in train_datasets
            and sample.source_dataset == "mnms1"
            and sample.vendor not in mnms1_vendor_holdout
        }
    )
    if train_candidate_patients:
        patient_splits = split_sample_ids(
            train_candidate_patients,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
        patient_to_split = {
            patient_id: split_name
            for split_name, patient_ids in patient_splits.items()
            for patient_id in patient_ids
        }
    else:
        patient_to_split = {}

    for sample in samples:
        if sample.source_dataset == "mnms1" and sample.vendor in mnms1_vendor_holdout:
            split_by_sample_id[sample.sample_id] = "ood"
        elif sample.source_dataset in external_ood_datasets:
            split_by_sample_id[sample.sample_id] = "ood"
        elif sample.source_dataset in train_datasets and sample.patient_id in patient_to_split:
            split_by_sample_id[sample.sample_id] = patient_to_split[sample.patient_id]
        else:
            split_by_sample_id[sample.sample_id] = "unused"
    return split_by_sample_id


def cardiac_samples_to_manifest(
    samples: list[CardiacChallengeSample],
    *,
    split_by_sample_id: dict[str, str],
    descriptor_version: str,
) -> pd.DataFrame:
    rows = []
    for sample in samples:
        if split_by_sample_id.get(sample.sample_id) == "unused":
            continue
        image_meta = get_nifti_metadata(sample.image_path)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path),
                "mask_path": str(sample.mask_path),
                "split": split_by_sample_id[sample.sample_id],
                "dataset_name": sample.source_dataset,
                "spacing": " ".join(str(value) for value in image_meta["spacing"]),
                "shape": " ".join(str(value) for value in image_meta["spatial_shape"]),
                "orientation": image_meta["orientation"],
                "roi_bbox": "",
                "topology_cache_path": "",
                "topology_descriptor_version": descriptor_version,
                "source_dataset": sample.source_dataset,
                "source_split": sample.source_split,
                "patient_id": sample.patient_id,
                "phase": sample.phase,
                "view": sample.view,
                "vendor": sample.vendor,
                "site_id": sample.site_id,
                "protocol": sample.protocol,
                "label_map": sample.label_map,
                "label_schema": TARGET_LABEL_SCHEMA,
            }
        )
    frame = pd.DataFrame(rows)
    ordered_columns = REQUIRED_MANIFEST_COLUMNS + [
        column for column in frame.columns if column not in REQUIRED_MANIFEST_COLUMNS
    ]
    return frame[ordered_columns]


def write_cardiac_challenge_manifests(
    *,
    acdc_root: str | Path | None,
    mnms1_root: str | Path | None,
    mnms2_root: str | Path | None,
    output_dir: str | Path,
    manifest_dir: str | Path,
    split_output_path: str | Path,
    descriptor_version: str,
    mnms1_metadata_csv: str | Path | None,
    mnms2_metadata_csv: str | Path | None,
    metadata_sample_id_columns: Iterable[str],
    metadata_vendor_columns: Iterable[str],
    metadata_site_columns: Iterable[str],
    metadata_protocol_columns: Iterable[str],
    mnms1_vendor_holdout: Iterable[str],
    train_datasets: Iterable[str],
    external_ood_datasets: Iterable[str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
    include_acdc_testing: bool,
    include_mnms1_unlabelled: bool,
    include_mnms2_la: bool,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    samples = []
    samples.extend(collect_acdc_samples(acdc_root, output_dir=output_dir, include_testing=include_acdc_testing))
    samples.extend(
        collect_mnms1_samples(
            mnms1_root,
            output_dir=output_dir,
            metadata_csv=mnms1_metadata_csv,
            metadata_sample_id_columns=metadata_sample_id_columns,
            metadata_vendor_columns=metadata_vendor_columns,
            metadata_site_columns=metadata_site_columns,
            metadata_protocol_columns=metadata_protocol_columns,
            include_unlabelled=include_mnms1_unlabelled,
        )
    )
    samples.extend(
        collect_mnms2_samples(
            mnms2_root,
            output_dir=output_dir,
            metadata_csv=mnms2_metadata_csv,
            metadata_sample_id_columns=metadata_sample_id_columns,
            metadata_vendor_columns=metadata_vendor_columns,
            metadata_site_columns=metadata_site_columns,
            metadata_protocol_columns=metadata_protocol_columns,
            include_la=include_mnms2_la,
        )
    )
    if not samples:
        raise ValueError("No labelled cardiac challenge samples were found.")
    split_by_sample_id = _split_mnms1_samples(
        samples,
        train_datasets={str(item) for item in train_datasets},
        external_ood_datasets={str(item) for item in external_ood_datasets},
        mnms1_vendor_holdout={str(item) for item in mnms1_vendor_holdout},
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    frame = cardiac_samples_to_manifest(
        samples,
        split_by_sample_id=split_by_sample_id,
        descriptor_version=descriptor_version,
    )
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    splits = {}
    for split_name in ["train", "val", "test", "ood"]:
        split_frame = frame[frame["split"] == split_name]
        path = manifest_dir / f"{split_name}.csv"
        split_frame.to_csv(path, index=False)
        outputs[split_name] = path
        splits[split_name] = split_frame["sample_id"].tolist()
    outputs["splits"] = write_split_json(splits, split_output_path)
    return outputs
