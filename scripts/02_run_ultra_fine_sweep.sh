#!/bin/bash --login
# ==============================================================================
# SCRIPT 02 (ULTRA-FINE): 158,460-Point Detection Criteria Sweep
# ==============================================================================
# Purpose:
#   Executes the maximum granular criteria sweep (158,460 physically valid points)
#   across all 10 detection criteria with multi-point task chunking compliant
#   with MSI's 5,000-job per-user cluster submission limit.
#
# Storage Protection:
#   - Automatically writes to MSI high-capacity scratch (/scratch.global/$USER or $TMPDIR)
#     to safeguard home directory quotas.
#   - Excludes non-detections (reducing raw output from 1.05 TB to ~14 GB).
#   - Synchronizes final summary CSVs back to results/02_granular_criteria_sweep/.
#
# MSI HPC Scheduling Notice:
#   - RETIRED: The legacy 4,997-task array has been disabled to prevent
#     exhausting MSI FairShare priorities.
#   - REPLACEMENT: Use `scripts/submit_fast_criteria_sweep.sh` (3 tasks total).
# ==============================================================================
#SBATCH -p msismall
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g
#SBATCH --job-name=kn_ultrafine_deprecated
#SBATCH --output=logs/02_granular_criteria_sweep/%x_%j.log
#SBATCH --error=logs/02_granular_criteria_sweep/%x_%j.err

set -euo pipefail

# FairShare Protection Guard:
echo "=============================================================================="
echo "[DEPRECATED] 02_run_ultra_fine_sweep.sh has been disabled."
echo "Submitting 4,997 tasks requesting 256 GB for 71 hours destroys your MSI FairShare score."
echo "Please use the high-performance 'Simulate-Once, Filter-Many' pipeline instead (3 tasks total):"
echo "  bash scripts/submit_fast_criteria_sweep.sh --fiducial"
echo "=============================================================================="
exit 1

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
echo "Grid Preset:      ultra_fine (158,460 valid points)"
echo "Date/Time:        $(date)"
echo "=============================================================================="

MODEL="${MODEL:-Bu2026Fixed}"
N_TRANSIENTS="${N_TRANSIENTS:-1000000}"
SLICE_PARAM="${SLICE_PARAM:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/02_granular_criteria_sweep}"
GRID_PRESET="ultra_fine"
CHUNK_SIZE="${CHUNK_SIZE:-428}"
MAX_TASKS="${MAX_TASKS:-5000}"
FORCE_RERUN="${FORCE_RERUN:-0}"
EXCLUDE_NON_DETECTIONS="${EXCLUDE_NON_DETECTIONS:-1}"
COMPACT_SUMMARY_ONLY="${COMPACT_SUMMARY_ONLY:-0}"

# Auto-detect MSI high-capacity scratch storage to safeguard home directory quota:
SCRATCH_DIR="${SCRATCH_DIR:-}"
if [ -z "${SCRATCH_DIR}" ]; then
    if [ -d "/scratch.global/${USER:-}" ]; then
        SCRATCH_DIR="/scratch.global/${USER:-}/kn-rates/02_granular_criteria_sweep"
    elif [ -n "${TMPDIR:-}" ] && [ -d "${TMPDIR}" ]; then
        SCRATCH_DIR="${TMPDIR}/02_granular_criteria_sweep"
    fi
fi

ARGS=(
    --output-root "${OUTPUT_ROOT}"
    --model "${MODEL}"
    --grid-preset "${GRID_PRESET}"
    --n-transients "${N_TRANSIENTS}"
    --max-tasks "${MAX_TASKS}"
)

if [ -n "${SCRATCH_DIR}" ]; then
    mkdir -p "${SCRATCH_DIR}"
    ARGS+=(--scratch-dir "${SCRATCH_DIR}")
fi

if [ "${EXCLUDE_NON_DETECTIONS}" = "1" ]; then
    ARGS+=(--exclude-non-detections)
fi

if [ "${COMPACT_SUMMARY_ONLY}" = "1" ]; then
    ARGS+=(--compact-summary-only)
fi

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

echo "Ultra-fine criteria sweep execution finished at $(date)."
