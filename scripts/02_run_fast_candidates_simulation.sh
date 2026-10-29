#!/bin/bash --login
# ==============================================================================
# SCRIPT 02: Candidate Generation for Fast Criteria Sweep (MSI HPC Compliant)
# ==============================================================================
# Purpose:
#   Simulates candidate kilonova detections under the minimally restrictive
#   loose bounding criteria envelope across all 3 models (Bu2026Fixed, Bu2026Vary, Metzger).
#
# Execution Modes:
#   - 'fiducial' (default): 3 tasks (tasks 0, 1, 2 = Metzger, Bu2026Fixed, Bu2026Vary)
#   - 'ensemble': Multi-sample ensemble (e.g. 3 models x 20 samples = 60 tasks,
#     tasks 00-19 = Metzger, 20-39 = Bu2026Fixed, 40-59 = Bu2026Vary)
#
# MSI msismall Resource Calibration:
#   - Memory: 64 GB per task (Standard Tier, median queue wait ~0 min, plenty for ~45 GB peak RSS)
#   - Walltime: 03:00:00 (~1.5 hours per 1M run, ample safety cushion)
#   - CPU: 4 cores per task
# ==============================================================================
#SBATCH -p msismall
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --tmp=30g
#SBATCH --array=0-2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --job-name=kn_crit_cand
#SBATCH --output=logs/02_fast_criteria_sweep/%x_%A_%a.log
#SBATCH --error=logs/02_fast_criteria_sweep/%x_%A_%a.err

set -euo pipefail

# Suppress CUPTI initialization warnings on CPU compute nodes
export JAX_PLATFORMS=cpu

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/02_fast_criteria_sweep results/02_granular_criteria_sweep/candidates

# Activate Conda environment
if command -v module &>/dev/null; then
    module load conda 2>/dev/null || true
    module load hdf5 2>/dev/null || true
    module load cuda/12.0 2>/dev/null || true
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
echo "Array Task ID:    ${SLURM_ARRAY_TASK_ID:-0}"
echo "Hostname:         $(hostname -f 2>/dev/null || hostname)"
echo "Date/Time:        $(date)"
echo "=============================================================================="

MODE="${MODE:-fiducial}"
N_SAMPLES="${N_SAMPLES:-20}"
N_TRANSIENTS="${N_TRANSIENTS:-1000000}"
OUTPUT_DIR="${OUTPUT_DIR:-results/02_granular_criteria_sweep/candidates}"
RANDOMIZE_PARAMS="${RANDOMIZE_PARAMS:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

ARGS=(
    --output-dir "${OUTPUT_DIR}"
    --mode "${MODE}"
    --n-samples "${N_SAMPLES}"
    --n-transients "${N_TRANSIENTS}"
    --n-processes 4
)

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    if [ "${MODE}" = "ensemble" ]; then
        ARGS+=(--task-id "${SLURM_ARRAY_TASK_ID}")
    else
        MODELS=("Metzger" "Bu2026Fixed" "Bu2026Vary")
        M_IDX=$((SLURM_ARRAY_TASK_ID % 3))
        ARGS+=(--model "${MODELS[$M_IDX]}")
    fi
fi

if [ "${RANDOMIZE_PARAMS}" = "1" ]; then
    ARGS+=(--randomize-params)
fi

if [ "${FORCE_RERUN}" = "1" ]; then
    ARGS+=(--force-rerun)
fi

python analysis_suites/02_granular_criteria_sweep/simulate_candidates.py "${ARGS[@]}"

echo "Candidate generation completed at $(date)."
