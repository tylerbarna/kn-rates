#!/bin/bash --login
# ==============================================================================
# SCRIPT 07: Aggregate Suite 07 Rate Limits & Execute Suite 01 Comparison
# ==============================================================================
# Purpose:
#   Parses all serialized JSON outputs from Suite 07 (GW170817 fixed-parameter ensemble),
#   calculates Poisson rate limits (R_90, R_95, R_99), performs a direct pairwise
#   comparison against Suite 01 (population/fiducial models), generates markdown
#   and tabular comparison summaries, and renders Notebook 07.
# ==============================================================================
#SBATCH -p msismall
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16g
#SBATCH --tmp=10g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --job-name=kn_agg_suite07
#SBATCH --output=logs/07_gw170817_rate_limits/%x_%j.log
#SBATCH --error=logs/07_gw170817_rate_limits/%x_%j.err

set -euo pipefail

# Suppress CUPTI initialization warnings on CPU compute nodes
export JAX_PLATFORMS=cpu

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/07_gw170817_rate_limits results/07_gw170817_rate_limits

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
else
    echo "Warning: No standard conda environment matched; using currently active environment: $(which python)"
fi

SUITE07_DIR="${1:-results/07_gw170817_rate_limits}"
SUITE01_CSV="${2:-results/01_large_scale_rate_limits/rate_limit_summary.csv}"
SUITE01_FALLBACK="results/rate_limit_benchmark.csv"

echo "=============================================================================="
echo "AGGREGATING SUITE 07 & EXECUTING COMPARISON WITH SUITE 01"
echo "Project Directory:  ${PROJECT_DIR}"
echo "Suite 07 Directory: ${SUITE07_DIR}"
echo "Suite 01 CSV:       ${SUITE01_CSV}"
echo "Date/Time:          $(date)"
echo "=============================================================================="

python analysis_suites/07_gw170817_rate_limits/aggregate_and_compare_suite01.py \
    --suite07-dir "${SUITE07_DIR}" \
    --suite01-csv "${SUITE01_CSV}" \
    --suite01-fallback-csv "${SUITE01_FALLBACK}" \
    --output-dir "${SUITE07_DIR}"

echo "Building and rendering Notebook 07..."
python build_notebooks.py --notebooks 07 2>/dev/null || python build_notebooks.py

echo "Suite 07 aggregation and Suite 01 comparative analysis completed successfully."
