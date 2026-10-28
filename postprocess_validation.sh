#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./postprocess_validation.sh

Set VALIDATION_DIR in this file to the fold's validation directory.

Applies the custom BraTS component filtering and morphological closing to the
validation segmentations already saved in the directory. It does not run model
inference or nnU-Net's selected connected-component postprocessing.
Writes <run-name>.out and <run-name>.err in postprocessinglogs/ next to
this script, while also streaming output to the terminal.

Uses nnUNet_preprocessed if set; otherwise uses this cluster’s nnU-Net preprocessed data path.
USAGE
}

# Edit this path to process another run or fold.
VALIDATION_DIR="/gpfs/scratch1/shared/apiaghiardelli/SNNMamba_k128_temporal_analog/Dataset001_BraTS24GLI/nnUNetTrainerSNNFPTT__SNNPlans__3d_fullres/fold_0/validation"

# Minimum connected-component sizes in voxels (ET, TC, WT).
MIN_COMPONENT_ET=300
MIN_COMPONENT_TC=300
MIN_COMPONENT_WT=50
CLOSING_RADIUS=1

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -ne 0 ]]; then
    usage >&2
    exit 2
fi

if [[ ! -d "$VALIDATION_DIR" ]]; then
    printf 'Error: validation directory not found: %s\n' "$VALIDATION_DIR" >&2
    exit 2
fi

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
VALIDATION_DIR="$(cd -- "$VALIDATION_DIR" && pwd -P)"
FOLD_DIR="$(dirname -- "$VALIDATION_DIR")"
RESULTS_CONFIG_DIR="$(dirname -- "$FOLD_DIR")"
DATASET_DIR="$(dirname -- "$RESULTS_CONFIG_DIR")"
RUN_DIR="$(dirname -- "$DATASET_DIR")"
RUN_NAME="$(basename -- "$RUN_DIR")"
LOG_DIR="$REPO_DIR/postprocessinglogs"
mkdir -p "$LOG_DIR"
LOG_OUT="$LOG_DIR/$RUN_NAME.out"
LOG_ERR="$LOG_DIR/$RUN_NAME.err"
exec > >(tee "$LOG_OUT") 2> >(tee "$LOG_ERR" >&2)

PREPROCESSED_ROOT="${nnUNet_preprocessed:-/gpfs/scratch1/shared/apiaghiardelli/nnUNet_preprocessed}"
if [[ ! -d "$PREPROCESSED_ROOT" ]]; then
    printf 'Error: nnU-Net preprocessed data root not found: %s\n' "$PREPROCESSED_ROOT" >&2
    printf 'Set nnUNet_preprocessed to the correct directory and retry.\n' >&2
    exit 2
fi
DATASET_NAME="$(basename -- "$DATASET_DIR")"
REFERENCE_DIR="$PREPROCESSED_ROOT/$DATASET_NAME/gt_segmentations"
PLANS_JSON="$RESULTS_CONFIG_DIR/plans.json"
DATASET_JSON="$RESULTS_CONFIG_DIR/dataset.json"

for required_file in \
    "$VALIDATION_DIR/summary.json" \
    "$PLANS_JSON" \
    "$DATASET_JSON"; do
    if [[ ! -f "$required_file" ]]; then
        printf 'Error: required file not found: %s\n' "$required_file" >&2
        exit 2
    fi
done

if [[ ! -d "$REFERENCE_DIR" ]]; then
    printf 'Error: reference masks directory not found: %s\n' "$REFERENCE_DIR" >&2
    printf 'Check nnUNet_preprocessed and the validation directory structure.\n' >&2
    exit 2
fi

shopt -s nullglob
PREDICTION_MASKS=("$VALIDATION_DIR"/*.nii.gz)
if [[ ${#PREDICTION_MASKS[@]} -eq 0 ]]; then
    printf 'Error: no .nii.gz prediction masks found in: %s\n' "$VALIDATION_DIR" >&2
    exit 2
fi

# Avoid each spawned nnU-Net worker creating a large OpenBLAS thread pool.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export nnUNet_extTrainer="$REPO_DIR/snn_nnunet"
export nnUNet_compile=false

printf 'Validation masks: %s (%s cases)\n' "$VALIDATION_DIR" "${#PREDICTION_MASKS[@]}"
printf 'Raw Dice summary: %s/summary.json\n' "$VALIDATION_DIR"
POSTPROCESSED_DIR="$VALIDATION_DIR/postprocessed_custom"
printf 'Custom postprocessed outputs: %s/\n' "$POSTPROCESSED_DIR"
printf 'Log files: %s and %s\n' "$LOG_OUT" "$LOG_ERR"
printf 'Component thresholds ET/TC/WT: %s/%s/%s; closing radius: %s\n' \
    "$MIN_COMPONENT_ET" "$MIN_COMPONENT_TC" "$MIN_COMPONENT_WT" "$CLOSING_RADIUS"
printf 'Running custom postprocessing; inference is not run.\n\n'

cd "$REPO_DIR"
uv run --python 3.10 python -m snn_nnunet.postprocess_saved_volumes \
    --input-dir "$VALIDATION_DIR" \
    --reference-dir "$REFERENCE_DIR" \
    --output-dir "$POSTPROCESSED_DIR" \
    --plans-json "$PLANS_JSON" \
    --dataset-json "$DATASET_JSON" \
    --min-component-sizes "$MIN_COMPONENT_ET" "$MIN_COMPONENT_TC" "$MIN_COMPONENT_WT" \
    --closing-radius "$CLOSING_RADIUS"
