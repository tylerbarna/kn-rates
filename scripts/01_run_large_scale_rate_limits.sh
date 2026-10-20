#!/bin/bash --login
# ==============================================================================
# SCRIPT 01: Large-Scale ZTF 8-Year Kilonova Rate Limits Simulation
# ==============================================================================
# Purpose:
#   Executes large-scale Monte Carlo simulations (default 10,000,000 transients)
#   for the three primary kilonova models:
#     - Task 0: Metzger (analytic 1-component model)
#     - Task 1: Bu2026Fixed (2-component Fiesta surrogate, fixed inclination 0.45 rad)
#     - Task 2: Bu2026Vary (2-component Fiesta surrogate, isotropic cos iota)
#
# MSI msismall Resource Calibration & Empirical Benchmark:
#   - Verified Empirical Benchmark (MSI Agate Job 14237734, 10M transients):
#       * Memory Utilized: 649.67 GB (86.62% of 750.00 GB allocation)
#       * Job Wall-clock:  20:47:00 (~21 hours)
#       * Cores Utilized:  10 cores (28.91% CPU efficiency)
#   - Partition Specifications (msismall):
#       * Max Cores:       128 cores/node
#       * Memory:          248 - 755 GB RAM per node (3,900 MB/core base ratio)
#       * Max Walltime:    96:00:00
#       * Local Scratch:   850 GB
#   - Resource Configuration:
#       * --mem=750g: Automatically targets the 755 GB high-memory msismall nodes,
#                     guaranteeing sufficient headroom for the ~650 GB peak footprint.
#       * --time=36:00:00: Comfortably accommodates the ~21 hr walltime with 70% cushion.
#       * --cpus-per-task=16: Provides 16 worker threads for multi-threaded Rust/BLAS.
# ==============================================================================
#SBATCH -p msismall
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=750g
#SBATCH --tmp=100g
#SBATCH --array=0-2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --job-name=kn_rate_limits_10M
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

# Define model array
MODELS=("Metzger" "Bu2026Fixed" "Bu2026Vary")
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${MODELS[$((TASK_ID % 3))]}"

# Configure simulation parameters (default: 10,000,000 sources)
N_TRANSIENTS="${N_TRANSIENTS:-10000000}"
N_RUNS="${N_RUNS:-1}"
N_PROCESSES="${N_PROCESSES:-16}"
BASE_SEED="${BASE_SEED:-20261020}"
TASK_SEED=$((BASE_SEED + TASK_ID * 10007))
OUTPUT_DIR="${OUTPUT_DIR:-results/01_large_scale_rate_limits}"

echo "Model:            ${MODEL}"
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

echo "Task ${TASK_ID} (${MODEL}) completed successfully at $(date)."
