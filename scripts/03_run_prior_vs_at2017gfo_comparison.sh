#!/bin/bash --login
# ==============================================================================
# SCRIPT 03: Broad Swath Priors vs. AT2017gfo-Like Kilonovae Comparison
# ==============================================================================
# Purpose:
#   Directly contrasts the performance of detection criteria against kilonovae
#   drawn from broad parameter priors vs. an AT2017gfo-like archetype.
#
# Scientific Focus:
#   - Isolates viewing angle effects (polar iota=26 deg vs. isotropic cos iota).
#   - Quantifies the selection bias toward high-mass, blue wind components.
#   - Evaluates the divergence in effective search volume <VT> and rate limits R_95%.
# ==============================================================================
# Resource allocations aligned with MSI msismall partition (3900 MB/core):
#SBATCH -p msismall
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15g
#SBATCH --tmp=10g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --job-name=kn_prior_vs_at2017gfo
#SBATCH --output=logs/03_prior_vs_at2017gfo/%x_%j.log
#SBATCH --error=logs/03_prior_vs_at2017gfo/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/03_prior_vs_at2017gfo results/03_prior_vs_at2017gfo

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
echo "Hostname:         $(hostname -f)"
echo "Date/Time:        $(date)"
echo "=============================================================================="

N_TRANSIENTS="${N_TRANSIENTS:-100000}"
N_RUNS="${N_RUNS:-3}"
SEED="${SEED:-20261020}"
OUTPUT_DIR="${OUTPUT_DIR:-results/03_prior_vs_at2017gfo}"

python analysis_suites/03_prior_vs_at2017gfo/compare_priors_at2017gfo.py \
    --n-transients "${N_TRANSIENTS}" \
    --n-runs "${N_RUNS}" \
    --seed "${SEED}" \
    --output-dir "${OUTPUT_DIR}"

echo "Comparison suite completed successfully at $(date)."
