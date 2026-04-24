# TopoAnchor

Topology-anchored, context-modulated 3D MRI segmentation training code.

This repository implements the blueprint in
`README_topology_anchored_full_math_render_fixed.md` as real training
infrastructure. Training is manifest-driven and refuses to run without real
NIfTI image and mask files.

## Required Data Contract

Default manifests live at:

- `data/manifests/train.csv`
- `data/manifests/val.csv`
- `data/manifests/test.csv`

Each row must include:

```text
sample_id,image_path,mask_path,split,dataset_name,spacing,shape,orientation,roi_bbox,topology_cache_path,topology_descriptor_version
```

Masks are expected to be integer multiclass labels in `[0, num_classes - 1]`.

## Typical Workflow

```bash
python scripts/verify_environment.py
python scripts/preprocess_dataset.py --config-name config
python scripts/build_manifests.py --config-name config
python scripts/verify_data.py --config-name config
python scripts/precompute_topology.py --config-name config
python scripts/train.py --config-name config
python scripts/evaluate.py --config-name config
python scripts/summarize_robustness.py --config-name config
python scripts/infer.py --config-name config paths.infer_manifest=/path/to/infer.csv
```

The paper model uses `model.mamba.backend=mamba_ssm`, which requires a Linux
NVIDIA/CUDA environment with NVCC. On macOS/CPU, use
`model.mamba.backend=conv_fallback` only for data-pipeline and smoke-test runs;
do not report it as the Res-Mamba model.

```bash
python scripts/train.py --config-name config model.mamba.backend=conv_fallback trainer.fast_dev_run=true
```

## NVIDIA L40/L40S Training Preset

For a single ~45-48GB L40-class GPU, use the hardware preset:

```bash
python scripts/train.py --config-name config +hardware=l40
```

The preset uses the real `mamba_ssm` backend, GPU mixed precision, larger cardiac patches, gradient accumulation, and pinned persistent DataLoader workers:

```text
model.mamba.backend=mamba_ssm
trainer.accelerator=gpu
trainer.precision=16-mixed
data.patch_size=[192,192,32]
data.batch_size=2
train.accumulate_grad_batches=2
```

If memory is tight, reduce:

```bash
data.batch_size=1 data.patch_size='[160,160,24]'
```

If there is spare memory, try:

```bash
data.batch_size=3 train.accumulate_grad_batches=1
```

## Vendor-Held-Out Robustness

For a multi-vendor claim, provide metadata with at least `sample_id` and
`vendor`, then build manifests with a held-out vendor:

```bash
python scripts/build_manifests.py --config-name config \
  data.manifest_builder.metadata_csv=/path/to/metadata.csv \
  data.manifest_builder.vendor_holdout='[VendorD]'
```

This writes `train.csv`, `val.csv`, `test.csv`, and `ood.csv`. Evaluation reads
`test.csv` and, when present, `ood.csv`, then `summarize_robustness.py` writes
local-vs-distribution-shift Dice and calibration tables.

## ACDC, M&Ms1, and M&Ms2 Workflow

The cardiac challenge builder understands the observed layouts:

- ACDC: `database/training|testing/patientXXX/patientXXX_frameYY(.nii.gz|_gt.nii.gz)`
- M&Ms1: `Training|Validation|Testing/Labelled|Unlabelled/CASE/CASE_sa(.nii.gz|_gt.nii.gz)`
- M&Ms2: `dataset/ID/ID_SA_ED(.nii.gz|_gt.nii.gz)` and `ID_SA_ES(.nii.gz|_gt.nii.gz)`

It writes 3D ED/ES samples into `data/processed/cardiac_challenges`, standardizing masks to:

```text
0=background, 1=LV, 2=MYO, 3=RV
```

ACDC masks are remapped from `1=RV,2=MYO,3=LV` into this target order.

Example:

```bash
python scripts/build_cardiac_challenge_manifests.py --config-name config \
  data.cardiac_challenges.acdc_root=/Volumes/Transcend/ACDC-dataset \
  data.cardiac_challenges.mnms1_root=/Volumes/Transcend/M\&M1 \
  data.cardiac_challenges.mnms2_root=/Volumes/Transcend/MnM2 \
  data.cardiac_challenges.mnms1_metadata_csv=/Volumes/Transcend/M\&M1/211230_M\&Ms_Dataset.csv \
  data.cardiac_challenges.mnms1_vendor_holdout='[VendorD]'
```

Default strategy:

- Train/validate/local-test on M&Ms1 vendors except the held-out vendor.
- Put the held-out M&Ms1 vendor plus ACDC and M&Ms2 into `ood.csv`.
- Use `summarize_robustness.py` to report local-vs-shift Dice, ECE, Brier, confidence, and topology-anchor distance.
