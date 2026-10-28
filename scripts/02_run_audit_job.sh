#!/bin/bash --login
# ==============================================================================
# SCRIPT 02 (AUDIT): Consolidated Audit & Elasticity Sensitivity Ranking
# ==============================================================================
# Purpose:
#   Aggregates completed sweep outputs from all array tasks, audits selection
#   rule compliance across all 10 criteria, calculates empirical elasticities,
#   and outputs consolidated CSVs for the analysis notebook.
#
# Execution:
#   Designed to run automatically via Slurm dependency:
#     --dependency=afterok:<SWEEP_JOB_ID>
# ==============================================================================
#SBATCH -p msismall
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16g
#SBATCH --job-name=kn_crit_audit
#SBATCH --output=logs/02_granular_criteria_sweep/%x_%j.log
#SBATCH --error=logs/02_granular_criteria_sweep/%x_%j.err

set -euo pipefail

# Suppress CUPTI initialization warnings on CPU compute nodes
export JAX_PLATFORMS=cpu

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/02_granular_criteria_sweep results/02_granular_criteria_sweep

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

SWEEP_ROOT="${1:-}"
if [ -z "${SWEEP_ROOT}" ]; then
    if [ -d "/scratch.global/${USER:-}/kn-rates/02_granular_criteria_sweep" ]; then
        SWEEP_ROOT="/scratch.global/${USER:-}/kn-rates/02_granular_criteria_sweep"
    else
        SWEEP_ROOT="results/02_granular_criteria_sweep"
    fi
fi

echo "=============================================================================="
echo "RUNNING DETECTION CRITERIA SWEEP AUDIT & SENSITIVITY AGGREGATION"
echo "Job ID:           ${SLURM_JOB_ID:-LOCAL}"
echo "Sweep Source:     ${SWEEP_ROOT}"
echo "Date/Time:        $(date)"
echo "=============================================================================="

python analysis_suites/02_granular_criteria_sweep/audit_granular_sweep.py \
    --sweep-root "${SWEEP_ROOT}" \
    --output-csv results/02_granular_criteria_sweep/granular_sweep_metrics.csv \
    --ranking-csv results/02_granular_criteria_sweep/criteria_sensitivity_ranking.csv

echo "Updating Job Telemetry & Resource Rightsizing Registry..."
python -m telemetry update || true

echo "Audit, aggregation, and telemetry tracking completed successfully at $(date)."
