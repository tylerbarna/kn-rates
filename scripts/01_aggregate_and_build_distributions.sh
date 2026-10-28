#!/bin/bash --login
# ==============================================================================
# SCRIPT: Aggregate 1M Randomized Results & Update Distribution Notebooks
# ==============================================================================
# Purpose:
#   Parses all serialized JSON outputs from 01_run_large_scale_1M_randomized.sh,
#   computes Poisson rate limits (R_90, R_95, R_99) and duration scalings, writes
#   the consolidated CSV, and automatically executes Notebook 01 to generate
#   populated distribution boxplots, ECDFs, and search volume envelopes.
# ==============================================================================
#SBATCH -p msismall
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16g
#SBATCH --tmp=10g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --job-name=kn_agg_1M_dist
#SBATCH --output=logs/01_large_scale_rate_limits/%x_%j.log
#SBATCH --error=logs/01_large_scale_rate_limits/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/01_large_scale_rate_limits results/01_large_scale_rate_limits

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

TARGET_DIR="${1:-results/01_large_scale_rate_limits}"
OUTPUT_CSV="${TARGET_DIR}/rate_limit_summary.csv"

echo "=============================================================================="
echo "AGGREGATING KILONOVA RATE LIMIT SIMULATION RUNS"
echo "Project Directory: ${PROJECT_DIR}"
echo "Target Directory:  ${TARGET_DIR}"
echo "Output CSV:        ${OUTPUT_CSV}"
echo "Date/Time:         $(date)"
echo "=============================================================================="

# 1. Aggregate simulation JSONs into rate_limit_summary.csv
echo "[1/3] Ingesting serialized JSON outputs from ${TARGET_DIR}..."
python analysis_suites/01_large_scale_rate_limits/aggregate_rate_limits.py \
    --glob "${TARGET_DIR}/*/*.json" \
    --output-csv "${OUTPUT_CSV}"

# Only sync canonical fiducial suite to benchmark CSV if not running randomized target
if [ "${TARGET_DIR}" = "results/01_large_scale_rate_limits" ]; then
    cp "${OUTPUT_CSV}" results/rate_limit_benchmark.csv 2>/dev/null || true
fi

# 2. Update historical comparison with Andreoni (2020)
echo "[2/3] Updating comparative analysis with Andreoni et al. (2020)..."
python analysis_suites/04_historical_andreoni_comparison/compare_with_andreoni2020.py \
    --rate-limit-csv "${OUTPUT_CSV}" \
    --output-dir "results/04_historical_andreoni_comparison"

# 3. Execute Notebook 01 to populate rich boxplots, ECDFs, and scatter plots
echo "[3/3] Re-executing Notebook 01 to generate populated distribution plots..."
python -c "
from pathlib import Path
import execute_notebooks
execute_notebooks.execute_notebook(Path('notebooks/01_large_scale_rate_limits_analysis.ipynb'))
"

echo "=============================================================================="
echo "1M RANDOMIZED AGGREGATION & DISTRIBUTION RE-EXECUTION COMPLETE"
echo "Results available at: results/01_large_scale_rate_limits/rate_limit_summary.csv"
echo "Notebook updated at:  notebooks/01_large_scale_rate_limits_analysis.ipynb"
echo "Finished at:          $(date)"
echo "=============================================================================="
