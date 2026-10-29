#!/bin/bash --login
# ==============================================================================
# SUBMITTER: Suite 01 - Large-Scale 8-Year Kilonova Rate Limits Ensemble (300 Tasks)
# ==============================================================================
# Purpose:
#   Submits the canonical 300-task fiducial rate limit ensemble (100 samples x 1M
#   transients each for Metzger, Bu2026Fixed, and Bu2026Vary) to MSI's `msismall`,
#   then automatically chains the aggregation and notebook execution job to run
#   as soon as all array tasks complete.
#
# Usage:
#   bash scripts/submit_suite01_rate_limits.sh
# ==============================================================================
#SBATCH -p msismall
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --job-name=kn_sub_suite01
#SBATCH --output=logs/01_large_scale_rate_limits/submit_suite01_%j.log
#SBATCH --error=logs/01_large_scale_rate_limits/submit_suite01_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/01_large_scale_rate_limits results/01_large_scale_rate_limits

# If running interactively outside Slurm batch, log to timestamped file in logs/
if [ -z "${SLURM_JOB_ID:-}" ] && [ -t 1 ]; then
    INTERACTIVE_LOG="logs/01_large_scale_rate_limits/submit_suite01_$(date +%Y%m%d_%H%M%S).log"
    echo "[INFO] Running interactively; teeing execution log to ${INTERACTIVE_LOG}"
    exec > >(tee -a "${INTERACTIVE_LOG}") 2>&1
fi

echo "=============================================================================="
echo "SUBMITTING SUITE 01: 300-TASK FIDUCIAL RATE LIMIT ENSEMBLE"
echo "Project Directory: ${PROJECT_DIR}"
echo "Models:            Metzger, Bu2026Fixed, Bu2026Vary (100 runs each = 300 tasks)"
echo "Transients/Task:   1,000,000"
echo "Memory per Task:   128 GB"
echo "Output Directory:  results/01_large_scale_rate_limits"
echo "Date/Time:         $(date)"
echo "Execution Host:    $(hostname -f 2>/dev/null || hostname)"
echo "Slurm Job ID:      ${SLURM_JOB_ID:-INTERACTIVE_SHELL}"
echo "=============================================================================="

# 1. Submit simulation array (300 tasks)
echo "--> Submitting Suite 01 simulation array (tasks 0-299)..."
sim_job_id=$(sbatch --parsable scripts/01_run_large_scale_rate_limits.sh)
echo "    Simulation Job ID: ${sim_job_id}"

# 2. Submit aggregation and notebook rendering job (dependent on array completion)
echo "--> Submitting post-processing aggregator & notebook generator (afterany:${sim_job_id})..."
agg_job_id=$(sbatch --parsable --dependency="afterany:${sim_job_id}" scripts/01_aggregate_and_build_distributions.sh)
echo "    Aggregator Job ID: ${agg_job_id}"

echo "=============================================================================="
echo "SUITE 01 PIPELINE SUBMITTED SUCCESSFULLY"
echo "=============================================================================="
echo "  1. Simulation Array:        ${sim_job_id} (0-299)"
echo "  2. Distribution Aggregator: ${agg_job_id} (queued on ${sim_job_id})"
echo ""
echo "Monitor status with:"
echo "  squeue -u \${USER:-\$USER}"
echo ""
echo "Logs will be written to:"
echo "  logs/01_large_scale_rate_limits/"
echo "=============================================================================="
