#!/bin/bash --login
# ==============================================================================
# SCRIPT 01: Large-Scale ZTF 8-Year Kilonova Rate Limits Ensemble Simulation
# ==============================================================================
# Purpose:
#   Executes a high-statistics Monte Carlo ensemble of 100 independent 1,000,000-transient
#   simulation samples for each of the three primary kilonova models:
#     - Tasks 000-099: Metzger (analytic 1-component model) - 100 x 1M samples
#     - Tasks 100-199: Bu2026Fixed (2-component Fiesta, iota=0.45 rad) - 100 x 1M samples
#     - Tasks 200-299: Bu2026Vary (2-component Fiesta, isotropic cos iota) - 100 x 1M samples
#
# Total Ensemble Scale:
#   - 300 array tasks x 1,000,000 transients = 300,000,000 total simulated transients
#   - 100,000,000 transients per model family (10x statistics of monolithic 10M run)
#
# MSI msismall Resource Calibration:
#   - Memory: 128 GB per task (massive headroom over 25-35 GB peak RSS, completely eliminating OOMs)
#   - Walltime: 04:00:00 (~1.5-2.0 hours per 1M run, 2x safety cushion)
#   - Scheduling: Runs concurrently across standard 248 GB msismall nodes (no waiting for 755 GB nodes)
# ==============================================================================
#SBATCH -p msismall
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --tmp=30g
#SBATCH --array=0-299
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --job-name=kn_rate_limits_1M_ensemble
#SBATCH --output=logs/01_large_scale_rate_limits/%x_%A_%a.log
#SBATCH --error=logs/01_large_scale_rate_limits/%x_%A_%a.err

set -euo pipefail

# Determine project root dynamically
PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

# Ensure log and output directories exist
mkdir -p logs/01_large_scale_rate_limits results/01_large_scale_rate_limits

# Activate appropriate Conda environment
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
else
    echo "Warning: No standard conda environment matched; using currently active environment: $(which python)"
fi

echo "=============================================================================="
echo "Job ID:           ${SLURM_JOB_ID:-LOCAL}"
echo "Array Task ID:    ${SLURM_ARRAY_TASK_ID:-0}"
echo "Hostname:         $(hostname -f)"
echo "Python Executable: $(which python)"
echo "Date/Time:        $(date)"
echo "=============================================================================="

# Define model array (3 models x 100 samples each = 300 array tasks)
MODELS=("Metzger" "Bu2026Fixed" "Bu2026Vary")
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_IDX=$((TASK_ID / 100))
SAMPLE_ID=$((TASK_ID % 100))
MODEL="${MODELS[$MODEL_IDX]}"

# Configure simulation parameters (default: 1,000,000 sources per sample)
N_TRANSIENTS="${N_TRANSIENTS:-1000000}"
N_RUNS="${N_RUNS:-1}"
N_PROCESSES="${N_PROCESSES:-8}"
BASE_SEED="${BASE_SEED:-20261020}"
TASK_SEED=$((BASE_SEED + TASK_ID * 10007))
OUTPUT_DIR="${OUTPUT_DIR:-results/01_large_scale_rate_limits}"

echo "Model:            ${MODEL} (Sample #${SAMPLE_ID} / 100)"
echo "Transients:       ${N_TRANSIENTS}"
echo "Runs:             ${N_RUNS}"
echo "Processes:        ${N_PROCESSES}"
echo "Seed:             ${TASK_SEED}"
echo "Output Directory: ${OUTPUT_DIR}/${MODEL}"
echo "=============================================================================="

python analysis_suites/01_large_scale_rate_limits/run_large_scale_simulation.py \
    --model "${MODEL}" \
    --n-transients "${N_TRANSIENTS}" \
    --n-runs "${N_RUNS}" \
    --n-processes "${N_PROCESSES}" \
    --seed "${TASK_SEED}" \
    --output-dir "${OUTPUT_DIR}"

echo "Task ${TASK_ID} (${MODEL} sample ${SAMPLE_ID}) completed successfully at $(date)."
