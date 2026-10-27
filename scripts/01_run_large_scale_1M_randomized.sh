#!/bin/bash --login
# ==============================================================================
# SCRIPT: Large-Scale 1M Randomized Parameter Simulation Suite
# ==============================================================================
# Purpose:
#   Executes large-scale Monte Carlo simulations with 1,000,000 transients per
#   run and randomized astrophysical ejecta parameters across the 3 canonical
#   kilonova model families:
#     - Metzger:      M_ej in [0.0005, 0.0050] M_sun, v_ej in [0.20, 0.80]c, kappa in [50, 1000] cm^2/g
#     - Bu2026Fixed:  Log10 M_dyn, Log10 M_wind, v_dyn, v_wind, Y_e randomized, fixed polar iota=26 deg
#     - Bu2026Vary:   All ejecta parameters + isotropic viewing angle (flat in cos iota) randomized
#
# HPC Calibration & Partition Alignment:
#   - 1M Transients Profile (calibrated against MSI Agate Job 14237734):
#       * Peak Memory RSS: ~25-100+ GB depending on detection volume / data structures
#       * Wall-clock Time: ~40 minutes per realization (with 8-16 threads)
#   - Allocation:
#       * #SBATCH -p msismall
#       * #SBATCH --mem=128g (expanded to 128 GB to prevent OOM kills)
#       * #SBATCH --cpus-per-task=8
#       * #SBATCH --time=02:00:00 (3x cushion for ~40m runtime)
#       * #SBATCH --array=0-29 (overridden dynamically via submit_1M_randomized_suite01.sh)
# ==============================================================================
#SBATCH -p msismall
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256g
#SBATCH --tmp=20g
#SBATCH --array=0-29
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --job-name=kn_sim_1M_random
#SBATCH --output=logs/01_large_scale_rate_limits/%x_%A_%a.log
#SBATCH --error=logs/01_large_scale_rate_limits/%x_%A_%a.err

set -euo pipefail

# Determine project root dynamically
PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

# Ensure output and log directories exist
mkdir -p logs/01_large_scale_rate_limits results/01_large_scale_rate_limits

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
else
    echo "Warning: No standard conda environment matched; using currently active environment: $(which python)"
fi

# Model mapping from array task ID
MODELS=("Metzger" "Bu2026Fixed" "Bu2026Vary")
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${MODELS[$((TASK_ID % 3))]}"

# Configuration
N_TRANSIENTS="${N_TRANSIENTS:-1000000}"
N_PROCESSES="${N_PROCESSES:-8}"
BASE_SEED="${BASE_SEED:-20261021}"
TASK_SEED=$((BASE_SEED + TASK_ID * 10007))
OUTPUT_DIR="${OUTPUT_DIR:-results/01_large_scale_rate_limits}"

echo "=============================================================================="
echo "Job ID:            ${SLURM_JOB_ID:-LOCAL}"
echo "Array Task ID:     ${TASK_ID}"
echo "Hostname:          $(hostname -f)"
echo "Python Executable: $(which python)"
echo "Date/Time:         $(date)"
echo "Model:             ${MODEL}"
echo "Transients:        ${N_TRANSIENTS}"
echo "Processes:         ${N_PROCESSES}"
echo "Randomize Params:  YES (--randomize-params)"
echo "Vary Rate:         YES (--vary-rate)"
echo "Task Seed:         ${TASK_SEED}"
echo "Output Directory:  ${OUTPUT_DIR}/${MODEL}"
echo "=============================================================================="

python analysis_suites/01_large_scale_rate_limits/run_large_scale_simulation.py \
    --model "${MODEL}" \
    --n-transients "${N_TRANSIENTS}" \
    --n-runs 1 \
    --n-processes "${N_PROCESSES}" \
    --randomize-params \
    --vary-rate \
    --seed "${TASK_SEED}" \
    --output-dir "${OUTPUT_DIR}"

echo "=============================================================================="
echo "Task ${TASK_ID} (${MODEL}, 1M random params) completed successfully at $(date)."
echo "=============================================================================="
