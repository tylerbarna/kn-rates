#!/bin/bash --login
# ==============================================================================
# SCRIPT 02: Fast Vectorized Grid Filter & Aggregator (MSI HPC Compliant)
# ==============================================================================
# Purpose:
#   Ingests candidate lightcurve pools from simulate_candidates.py across all 3
#   models (Bu2026Fixed, Bu2026Vary, Metzger) and filters them across all 2,138,460
#   ultra-fine detection criteria combinations in parallel, generating unified metrics CSVs.
#
# MSI msismall Resource Calibration:
#   - Memory: 64 GB (in-memory vectorized candidate pool filtering)
#   - Walltime: 06:00:00 (ample safety cushion for 100 samples across 2.14M points)
#   - CPU: 64 cores (evaluates ~50,000-80,000 points/sec)
# ==============================================================================
#SBATCH -p msismall
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --job-name=kn_crit_filter
#SBATCH --output=logs/02_fast_criteria_sweep/%x_%A_%a.log
#SBATCH --error=logs/02_fast_criteria_sweep/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/02_fast_criteria_sweep results/02_granular_criteria_sweep

# Activate Conda environment
if command -v module &>/dev/null; then
    module load conda 2>/dev/null || true
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

CANDIDATES_DIR="${CANDIDATES_DIR:-results/02_granular_criteria_sweep/candidates}"
OUTPUT_DIR="${OUTPUT_DIR:-results/02_granular_criteria_sweep}"
GRID_PRESET="${GRID_PRESET:-ultra_fine}"
N_WORKERS="${SLURM_CPUS_PER_TASK:-${N_WORKERS:-64}}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

MODEL="${MODEL:-all}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    MODELS=("Bu2026Fixed" "Bu2026Vary" "Metzger")
    M_IDX=$((SLURM_ARRAY_TASK_ID % 3))
    MODEL="${MODELS[$M_IDX]}"
    echo "[INFO] Slurm Array Task ${SLURM_ARRAY_TASK_ID} mapped to Model: ${MODEL}"
fi

echo "=============================================================================="
echo "LAUNCHING FAST CRITERIA FILTER & AGGREGATOR"
echo "Candidates Dir:   ${CANDIDATES_DIR}"
echo "Output Dir:       ${OUTPUT_DIR}"
echo "Grid Preset:      ${GRID_PRESET}"
echo "Model:            ${MODEL}"
echo "Workers:          ${N_WORKERS}"
echo "Date/Time:        $(date)"
echo "=============================================================================="

ARGS=(
    --candidates-dir "${CANDIDATES_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --grid-preset "${GRID_PRESET}"
    --model "${MODEL}"
    --n-workers "${N_WORKERS}"
)

if [ -n "${MAX_SAMPLES}" ]; then
    ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

python analysis_suites/02_granular_criteria_sweep/fast_criteria_filter.py "${ARGS[@]}"

echo "Criteria filtering and multi-model aggregation finished at $(date)."
