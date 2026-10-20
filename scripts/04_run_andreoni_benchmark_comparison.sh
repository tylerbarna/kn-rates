#!/bin/bash --login
# ==============================================================================
# SCRIPT 04: Historical Comparison against Andreoni et al. (2020)
# ==============================================================================
# Purpose:
#   Performs an automated quantitative comparison between the 8-year ZTF kilonova
#   rate limits and the historical 23-month limits published by Andreoni et al. (2020),
#   arXiv:2008.00008.
#
# Generates:
#   - results/04_historical_andreoni_comparison/andreoni_comparison_metrics.csv
#   - Detailed terminal summary of raw rate limits, duration scaling, and critical analysis.
# ==============================================================================
# Resource allocations aligned with MSI msismall partition (3900 MB/core):
#SBATCH -p msismall
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3900M
#SBATCH --tmp=5g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --job-name=kn_andreoni_comp
#SBATCH --output=logs/04_historical_andreoni_comparison/%x_%j.log
#SBATCH --error=logs/04_historical_andreoni_comparison/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/04_historical_andreoni_comparison results/04_historical_andreoni_comparison

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

RATE_LIMIT_CSV="${RATE_LIMIT_CSV:-results/rate_limit_benchmark.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-results/04_historical_andreoni_comparison}"

python analysis_suites/04_historical_andreoni_comparison/compare_with_andreoni2020.py \
    --rate-limit-csv "${RATE_LIMIT_CSV}" \
    --output-dir "${OUTPUT_DIR}"

echo "Andreoni comparison completed successfully at $(date)."
