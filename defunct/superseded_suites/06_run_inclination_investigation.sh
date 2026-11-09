#!/bin/bash --login
# ==============================================================================
# SCRIPT 06: Bu2026 Fixed vs. Varying Inclination Angle Discrepancy Investigation
# ==============================================================================
# Purpose:
#   Investigates whether the ~3.56x difference in estimated volumetric rate limits
#   between Bu2026Fixed (R_95 ~ 1,679) and Bu2026Vary (R_95 ~ 5,973) is legitimate
#   astrophysics or an error in the simulation approach.
#
# Experiments Evaluated:
#   1. Discrete Inclination Slicing (0° to 90° in 15° steps)
#   2. Cosmological z_max Sensitivity (0.10 to 0.30)
#   3. Radiative Transfer & Lanthanide Curtaining Flux Profiles
#   4. Cadence & Selection Cut Survival Rates
#   5. Numerical Angle Quadrature vs. Monte Carlo Integration
#
# MSI Allocation Profile:
#   - Partition:   msismall (3,900 MB/core base ratio)
#   - Resources:   8 cores, 128 GB RAM, 2 hours walltime
# ==============================================================================
#SBATCH -p msismall
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --tmp=30g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --job-name=kn_inclination_study
#SBATCH --output=logs/06_inclination_effect/%x_%j.log
#SBATCH --error=logs/06_inclination_effect/%x_%j.err

set -euo pipefail

# Suppress CUPTI initialization warnings on CPU compute nodes
export JAX_PLATFORMS=cpu

# Determine project directory dynamically
PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/06_inclination_effect results/06_inclination_effect

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
else
    echo "Using default python environment: $(which python)"
fi

echo "=============================================================================="
echo "Job ID:           ${SLURM_JOB_ID:-LOCAL}"
echo "Hostname:         $(hostname -f)"
echo "Python Executable: $(which python)"
echo "Date/Time:        $(date)"
echo "=============================================================================="

N_TRANSIENTS="${N_TRANSIENTS:-1000000}"
SEED="${SEED:-20261026}"
OUTPUT_DIR="${OUTPUT_DIR:-results/06_inclination_effect}"

echo "Starting Inclination Discrepancy Investigation Suite..."
echo "Transients:       ${N_TRANSIENTS}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "=============================================================================="

python analysis_suites/06_inclination_effect/run_inclination_investigation.py \
    --n-transients "${N_TRANSIENTS}" \
    --seed "${SEED}" \
    --output-dir "${OUTPUT_DIR}" \
    --simulate-live

echo "Inclination investigation completed successfully at $(date)."
