#!/bin/bash --login
# ==============================================================================
# SCRIPT 02: Granular Detection Criteria Grid Sweep
# ==============================================================================
# Purpose:
#   Executes a systematic sweep across a granular grid of detection criteria
#   (SNR thresholds, minimum detections, qualifying timespan, time separation,
#   fade rate, and galactic latitude) to investigate the parametric response of
#   ZTF kilonova detection efficiencies.
#
# Execution Modes:
#   1. Slurm Job Array (MSI Cluster):
#      Iterates over array indices 0 to N_COMBINATIONS-1.
#   2. Local 1D Parameter Slice:
#      Pass SLICE_PARAM (e.g. SLICE_PARAM="snr_threshold") to quickly evaluate
#      marginal sensitivity along a single axis.
# TODO: add number of detections as a detection criteria parameter to sweep
# ==============================================================================
# Resource allocations aligned with MSI msismall partition (3900 MB/core):
#SBATCH -p msismall
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32g
#SBATCH --tmp=10g
#SBATCH --array=0-2591%2592
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --job-name=kn_criteria_sweep
#SBATCH --output=logs/02_granular_criteria_sweep/%x_%A_%a.log
#SBATCH --error=logs/02_granular_criteria_sweep/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/02_granular_criteria_sweep results/02_granular_criteria_sweep

# Activate Conda environment
if command -v module &>/dev/null; then
    module load conda 2>/dev/null || true
    module load hdf5 2>/dev/null || true
fi

eval "$(conda shell.bash hook 2>/dev/null || true)"

if conda env list | grep -q "kn_rates_dev"; then
    conda activate kn_rates_dev
elif conda env list | grep -q "simsurvey_no_gpu"; then
    conda activate simsurvey_no_gpu
elif [ -d "$HOME/.conda/envs/simsurvey_no_gpu" ]; then
    conda activate "$HOME/.conda/envs/simsurvey_no_gpu"
elif [ -d "${PROJECT_DIR}/.conda" ]; then
    conda activate "${PROJECT_DIR}/.conda"
fi

echo "=============================================================================="
echo "Job ID:           ${SLURM_JOB_ID:-LOCAL}"
echo "Array Task ID:    ${SLURM_ARRAY_TASK_ID:-NONE}"
echo "Date/Time:        $(date)"
echo "=============================================================================="

MODEL="${MODEL:-Bu2026Fixed}"
N_TRANSIENTS="${N_TRANSIENTS:-100000}"
SLICE_PARAM="${SLICE_PARAM:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/02_granular_criteria_sweep}"

ARGS=(
    --output-root "${OUTPUT_ROOT}"
    --model "${MODEL}"
    --n-transients "${N_TRANSIENTS}"
)

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    ARGS+=(--array-index "${SLURM_ARRAY_TASK_ID}")
elif [ -n "${SLICE_PARAM}" ]; then
    ARGS+=(--slice-param "${SLICE_PARAM}")
fi

python analysis_suites/02_granular_criteria_sweep/run_granular_sweep.py "${ARGS[@]}"

echo "Criteria sweep execution finished at $(date)."
