#!/usr/bin/env bash
set -euo pipefail

# Dataset roots on the L40/L40S machine. Override these env vars if paths change:
#   ACDC_ROOT=/new/path MNMS1_ROOT=/new/path MNMS2_ROOT=/new/path bash test.sh
ACDC_ROOT=${ACDC_ROOT:-"/workspace/dataset/acdcdata/ACDC"}
MNMS1_ROOT=${MNMS1_ROOT:-"/workspace/dataset/mm1-data/M&M1"}
MNMS2_ROOT=${MNMS2_ROOT:-"/workspace/dataset/m-and-m-candy/MnM2"}
MNMS1_METADATA_CSV=${MNMS1_METADATA_CSV:-"${MNMS1_ROOT}/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"}

# Use FOLDS="D" for one fold, or FOLDS="A B C D" for the full paper experiment.
FOLDS=${FOLDS:-"A B C D"}

# Set RUN_TRAIN=0 to only build/verify manifests and topology cache.
RUN_TRAIN=${RUN_TRAIN:-1}
RUN_EVAL=${RUN_EVAL:-1}
SKIP_GLOBAL=${SKIP_GLOBAL:-0}
GLOBAL_ONLY=${GLOBAL_ONLY:-0}

COMMON_DATA_OVERRIDES=(
  "data.cardiac_challenges.acdc_root=\"${ACDC_ROOT}\""
  "data.cardiac_challenges.mnms1_root=\"${MNMS1_ROOT}\""
  "data.cardiac_challenges.mnms2_root=\"${MNMS2_ROOT}\""
  "data.cardiac_challenges.mnms1_metadata_csv=\"${MNMS1_METADATA_CSV}\""
)

run_global_cache_stage() {
  local manifest_dir="data/global/manifests"
  local split_dir="data/global/splits"

  echo "==> Global preprocessing/cache manifest stage"
  python scripts/build_cardiac_challenge_manifests.py --config-name config \
    "paths.manifest_dir=\"${manifest_dir}\"" \
    "paths.split_dir=\"${split_dir}\"" \
    "${COMMON_DATA_OVERRIDES[@]}" \
    "data.cardiac_challenges.mnms1_vendor_holdout=[]"

  python scripts/verify_data.py --config-name config \
    "paths.manifest_dir=\"${manifest_dir}\"" \
    "paths.split_dir=\"${split_dir}\""

  python scripts/precompute_topology.py --config-name config \
    "paths.manifest_dir=\"${manifest_dir}\"" \
    "paths.split_dir=\"${split_dir}\""

  python scripts/verify_data.py --config-name config \
    "paths.manifest_dir=\"${manifest_dir}\"" \
    "paths.split_dir=\"${split_dir}\"" \
    "data.verification.require_topology_cache=true"
}

run_fold() {
  local vendor="$1"
  local fold_name="vendor_${vendor}"
  local manifest_dir="data/folds/${fold_name}/manifests"
  local split_dir="data/folds/${fold_name}/splits"
  local output_dir="reports/folds/${fold_name}"
  local checkpoint_path="${output_dir}/checkpoints/last.ckpt"

  echo "==> Fold ${fold_name}"
  python scripts/build_cardiac_challenge_manifests.py --config-name config \
    "paths.manifest_dir=\"${manifest_dir}\"" \
    "paths.split_dir=\"${split_dir}\"" \
    "${COMMON_DATA_OVERRIDES[@]}" \
    "data.cardiac_challenges.mnms1_vendor_holdout=[${vendor}]"

  python scripts/verify_data.py --config-name config \
    "paths.manifest_dir=\"${manifest_dir}\"" \
    "paths.split_dir=\"${split_dir}\""

  # This reuses global JSON cache files and only updates this fold's manifest paths.
  python scripts/precompute_topology.py --config-name config \
    "paths.manifest_dir=\"${manifest_dir}\"" \
    "paths.split_dir=\"${split_dir}\""

  python scripts/verify_data.py --config-name config \
    "paths.manifest_dir=\"${manifest_dir}\"" \
    "paths.split_dir=\"${split_dir}\"" \
    "data.verification.require_topology_cache=true"

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    python scripts/train.py --config-name config +hardware=l40 \
      "paths.manifest_dir=\"${manifest_dir}\"" \
      "paths.split_dir=\"${split_dir}\"" \
      "paths.output_dir=\"${output_dir}\"" \
      "project_name=\"topoanchor_${fold_name}\""
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    python scripts/evaluate.py --config-name config +hardware=l40 \
      "paths.manifest_dir=\"${manifest_dir}\"" \
      "paths.split_dir=\"${split_dir}\"" \
      "paths.output_dir=\"${output_dir}\"" \
      "paths.checkpoint_path=\"${checkpoint_path}\""

    python scripts/summarize_robustness.py --config-name config \
      "paths.manifest_dir=\"${manifest_dir}\"" \
      "paths.split_dir=\"${split_dir}\"" \
      "paths.output_dir=\"${output_dir}\""
  fi
}

if [[ "${SKIP_GLOBAL}" != "1" ]]; then
  run_global_cache_stage
fi

if [[ "${GLOBAL_ONLY}" == "1" ]]; then
  exit 0
fi

for vendor in ${FOLDS}; do
  run_fold "${vendor}"
done
