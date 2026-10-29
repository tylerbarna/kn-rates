#!/bin/bash --login
# ==============================================================================
# SCRIPT 07: Large-Scale ZTF 8-Year GW170817 Fixed-Parameter Rate Limits Ensemble
# ==============================================================================
# Purpose:
#   Executes a high-statistics Monte Carlo ensemble of 100 independent 1,000,000-transient
#   simulations (1M per task) with physical parameters pinned to commonly accepted values
#   for the landmark binary neutron star merger GW170817 (AT2017gfo):
#     - Tasks 000-099: GW170817_Bu2026_FixedView (iota=0.45 rad / ~26 deg line-of-sight)
#     - Tasks 100-199: GW170817_Bu2026_Isotropic (uniform cos iota angle-averaged)
#     - Tasks 200-299: GW170817_Metzger_Blue (lanthanide-free fast blue peak: Mej=0.02, vej=0.25, kappa=0.5)
#     - Tasks 300-399: GW170817_Metzger_Mean (total mass-weighted ejecta: Mej=0.04, vej=0.15, kappa=1.0)
#
# Total Ensemble Scale:
#   - 400 array tasks x 1,000,000 transients = 400,000,000 total simulated transients
#   - 100,000,000 transients per archetype (matching Suite 01 statistical power)
#
# MSI msismall Resource Allocation:
#   - Memory: 128 GB per task (robust headroom, zero OOM risk)
#   - Walltime: 04:00:00 (~1.5-2 hours per 1M run, 2x safety cushion)
# ==============================================================================
#SBATCH -p msismall
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --tmp=30g
#SBATCH --array=0-399
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --job-name=kn_gw170817_fixed_ensemble
#SBATCH --output=logs/07_gw170817_rate_limits/%x_%A_%a.log
#SBATCH --error=logs/07_gw170817_rate_limits/%x_%A_%a.err

set -euo pipefail

# Suppress CUPTI initialization warnings on CPU compute nodes
export JAX_PLATFORMS=cpu

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/07_gw170817_rate_limits results/07_gw170817_rate_limits

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
echo "Hostname:         $(hostname -f 2>/dev/null || hostname)"
echo "Python Executable: $(which python)"
echo "Date/Time:        $(date)"
echo "=============================================================================="

# Define model array (4 models x 100 samples each = 400 array tasks)
MODELS=(
    "GW170817_Bu2026_FixedView"
    "GW170817_Bu2026_Isotropic"
    "GW170817_Metzger_Blue"
    "GW170817_Metzger_Mean"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SAMPLES_PER_MODEL="${SAMPLES_PER_MODEL:-100}"
MODEL_IDX=$((TASK_ID / SAMPLES_PER_MODEL))
SAMPLE_ID=$((TASK_ID % SAMPLES_PER_MODEL))

if [ "${MODEL_IDX}" -ge "${#MODELS[@]}" ]; then
    echo "Task ID ${TASK_ID} exceeds configured model bounds (max index $((${#MODELS[@]} - 1))). Exiting cleanly."
    exit 0
fi

MODEL="${MODELS[$MODEL_IDX]}"

# Default to 1,000,000 transients per run
N_TRANSIENTS="${N_TRANSIENTS:-1000000}"
N_RUNS="${N_RUNS:-1}"
N_PROCESSES="${N_PROCESSES:-8}"
BASE_SEED="${BASE_SEED:-20261028}"
TASK_SEED=$((BASE_SEED + TASK_ID * 10007))
OUTPUT_DIR="${OUTPUT_DIR:-results/07_gw170817_rate_limits}"

echo "Model Archetype:  ${MODEL} (Sample #${SAMPLE_ID} / ${SAMPLES_PER_MODEL})"
echo "Transients:       ${N_TRANSIENTS}"
echo "Runs:             ${N_RUNS}"
echo "Processes:        ${N_PROCESSES}"
echo "Task Seed:        ${TASK_SEED}"
echo "Output Directory: ${OUTPUT_DIR}/${MODEL}"
echo "=============================================================================="

python analysis_suites/07_gw170817_rate_limits/run_gw170817_simulation.py \
    --model "${MODEL}" \
    --n-transients "${N_TRANSIENTS}" \
    --n-runs "${N_RUNS}" \
    --n-processes "${N_PROCESSES}" \
    --seed "${TASK_SEED}" \
    --task-id "${TASK_ID}" \
    --output-dir "${OUTPUT_DIR}"

echo "Task ${TASK_ID} (${MODEL} sample ${SAMPLE_ID}) completed successfully at $(date)."
