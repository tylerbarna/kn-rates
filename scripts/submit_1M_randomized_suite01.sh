#!/bin/bash --login
# ==============================================================================
# SUBMITTER: 1M Randomized Parameter Large-Scale Rate Limit Suite
# ==============================================================================
# Purpose:
#   Submits the 1M randomized simulation array to MSI's `msismall` partition,
#   then automatically chains the aggregation and notebook execution job to run
#   as soon as all array tasks complete.
#
# Usage:
#   bash scripts/submit_1M_randomized_suite01.sh [RUNS_PER_MODEL]
# ==============================================================================
#SBATCH -p msismall
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --job-name=kn_sub_1M_rand
#SBATCH --output=logs/01_large_scale_rate_limits/submit_1M_randomized_%j.log
#SBATCH --error=logs/01_large_scale_rate_limits/submit_1M_randomized_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

RUNS_PER_MODEL="${1:-10}"

# Validate input
if ! [[ "${RUNS_PER_MODEL}" =~ ^[0-9]+$ ]] || [ "${RUNS_PER_MODEL}" -le 0 ]; then
    echo "Error: RUNS_PER_MODEL must be a positive integer, got '${RUNS_PER_MODEL}'" >&2
    exit 1
fi

NUM_MODELS=3
TOTAL_TASKS=$(( RUNS_PER_MODEL * NUM_MODELS ))
ARRAY_RANGE="0-$(( TOTAL_TASKS - 1 ))"

mkdir -p logs/01_large_scale_rate_limits results/01_large_scale_rate_limits

# If running interactively outside Slurm batch, log to timestamped file in logs/
if [ -z "${SLURM_JOB_ID:-}" ] && [ -t 1 ]; then
    INTERACTIVE_LOG="logs/01_large_scale_rate_limits/submit_1M_randomized_$(date +%Y%m%d_%H%M%S).log"
    echo "[INFO] Running interactively; teeing execution log to ${INTERACTIVE_LOG}"
    exec > >(tee -a "${INTERACTIVE_LOG}") 2>&1
fi

echo "=============================================================================="
echo "SUBMITTING 1M RANDOMIZED KILONOVA RATE LIMIT PIPELINE"
echo "Project Directory: ${PROJECT_DIR}"
echo "Models:            Metzger, Bu2026Fixed, Bu2026Vary (${NUM_MODELS} models)"
echo "Runs per Model:    ${RUNS_PER_MODEL}"
echo "Total Array Tasks: ${TOTAL_TASKS} (array range: ${ARRAY_RANGE})"
echo "Memory per Task:   256 GB"
echo "Date/Time:         $(date)"
echo "Execution Host:    $(hostname -f 2>/dev/null || hostname)"
echo "Slurm Job ID:      ${SLURM_JOB_ID:-INTERACTIVE_SHELL}"
echo "=============================================================================="

# 1. Submit simulation array
echo "--> Submitting 1M randomized simulation array (tasks ${ARRAY_RANGE})..."
sim_job_id=$(sbatch --parsable --array="${ARRAY_RANGE}" scripts/01_run_large_scale_1M_randomized.sh)
echo "    Simulation Job ID: ${sim_job_id}"

# 2. Submit aggregation and notebook rendering job (dependent on array completion)
echo "--> Submitting post-processing aggregator & notebook generator (afterany:${sim_job_id})..."
agg_job_id=$(sbatch --parsable --dependency="afterany:${sim_job_id}" scripts/01_aggregate_and_build_distributions.sh)
echo "    Aggregator Job ID: ${agg_job_id}"

echo "=============================================================================="
echo "PIPELINE SUBMITTED SUCCESSFULLY"
echo "=============================================================================="
echo "  1. Simulation Array: ${sim_job_id} (${ARRAY_RANGE})"
echo "  2. Distribution Aggregator: ${agg_job_id} (queued on ${sim_job_id})"
echo ""
echo "Monitor status with:"
echo "  squeue -u ${USER:-\$USER}"
echo ""
echo "Logs will be written to:"
echo "  logs/01_large_scale_rate_limits/"
echo "=============================================================================="
