#!/bin/bash
#SBATCH --job-name=arch2_brats24
#SBATCH --output=arch2_%j.out
#SBATCH --error=arch2_%j.err
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G

set -euo pipefail
# Submit from the checkout containing Arch2, or export REPO_DIR explicitly.
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/gpfs/scratch1/shared/apiaghiardelli}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/arch2_config.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
SRUN_BIN="${SRUN_BIN:-srun}"
cd "$REPO_DIR"
export nnUNet_raw="${nnUNet_raw:-$SCRATCH_ROOT/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$SCRATCH_ROOT/nnUNet_preprocessed}"
export nnUNet_extTrainer="$REPO_DIR/snn_nnunet"
export nnUNet_compile=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$SRUN_BIN" "$PYTHON_BIN" -m snn_nnunet.run_model --config "$CONFIG_PATH"
