#!/bin/bash --login
# ==============================================================================
# SCRIPT 02: Granular Detection Criteria Grid Sweep (MSI HPC Compliant)
# ==============================================================================
# Purpose:
#   Executes a systematic sweep across a broad, granular grid covering ALL 10
#   detection criteria (Primary SNR, Secondary SNR, Minimum Total Detections,
#   Minimum Primary SNR Detections, Qualifying Timespan Window, Minimum First-Last
#   Separation, Rise Rate, Fade Rate, Absolute Galactic Latitude, and Fast-Transient
#   Requirement) to investigate the parametric response of ZTF kilonova search volume.
#
# MSI HPC Scheduling Alignment:
#   - Hard cluster limit: At most 5,000 jobs submitted at one time per user.
#   - Scheduling throughput: MSI schedules and runs tasks in chunks of 20 to 50.
#   - Job Array Configuration:
#       #SBATCH --array=0-4031%50
#     Submits 4,032 tasks (<= 5,000) throttled to 50 concurrent tasks (%50).
#   - Automatic Chunking:
#     For ultra-fine or fine grids (>5,000 points), points are automatically
#     batched into chunks per task so submitted jobs NEVER exceed 5,000.
# ==============================================================================
# Resource allocations aligned with MSI msismall partition (3900 MB/core):
#SBATCH -p msismall
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32g
#SBATCH --tmp=10g
#SBATCH --array=0-4031%50
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
GRID_PRESET="${GRID_PRESET:-broad}"
CHUNK_SIZE="${CHUNK_SIZE:-}"
MAX_TASKS="${MAX_TASKS:-5000}"
FORCE_RERUN="${FORCE_RERUN:-0}"

ARGS=(
    --output-root "${OUTPUT_ROOT}"
    --model "${MODEL}"
    --grid-preset "${GRID_PRESET}"
    --n-transients "${N_TRANSIENTS}"
    --max-tasks "${MAX_TASKS}"
)

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    ARGS+=(--array-index "${SLURM_ARRAY_TASK_ID}")
elif [ -n "${SLICE_PARAM}" ]; then
    ARGS+=(--slice-param "${SLICE_PARAM}")
fi

if [ -n "${CHUNK_SIZE}" ]; then
    ARGS+=(--chunk-size "${CHUNK_SIZE}")
fi

if [ "${FORCE_RERUN}" = "1" ]; then
    ARGS+=(--force-rerun)
fi

python analysis_suites/02_granular_criteria_sweep/run_granular_sweep.py "${ARGS[@]}"

echo "Criteria sweep execution finished at $(date)."
