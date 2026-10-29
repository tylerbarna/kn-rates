#!/bin/bash --login
# ==============================================================================
# SUBMITTER: Suite 07 - GW170817 Fixed-Parameter Rate Limits Ensemble (400 Tasks)
# ==============================================================================
# Purpose:
#   Submits the 400-task GW170817 fixed-parameter rate limits ensemble (100 samples
#   x 1M transients each for GW170817_Bu2026_FixedView, GW170817_Bu2026_Isotropic,
#   GW170817_Metzger_Blue, and GW170817_Metzger_Mean) to MSI's `msismall` partition,
#   then automatically chains post-processing aggregation, direct Suite 01 comparison,
#   and Notebook 07 execution as soon as all array tasks finish.
#
# Usage:
#   bash scripts/submit_suite07_rate_limits.sh
# ==============================================================================
#SBATCH -p msismall
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --job-name=kn_sub_suite07
#SBATCH --output=logs/07_gw170817_rate_limits/submit_suite07_%j.log
#SBATCH --error=logs/07_gw170817_rate_limits/submit_suite07_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/07_gw170817_rate_limits results/07_gw170817_rate_limits

# If running interactively outside Slurm batch, log to timestamped file in logs/
if [ -z "${SLURM_JOB_ID:-}" ] && [ -t 1 ]; then
    INTERACTIVE_LOG="logs/07_gw170817_rate_limits/submit_suite07_$(date +%Y%m%d_%H%M%S).log"
    echo "[INFO] Running interactively; teeing execution log to ${INTERACTIVE_LOG}"
    exec > >(tee -a "${INTERACTIVE_LOG}") 2>&1
fi

echo "=============================================================================="
echo "SUBMITTING SUITE 07: 400-TASK GW170817 FIXED-PARAMETER RATE LIMITS ENSEMBLE"
echo "Project Directory: ${PROJECT_DIR}"
echo "Archetypes:        GW170817_Bu2026_FixedView, GW170817_Bu2026_Isotropic,"
echo "                   GW170817_Metzger_Blue, GW170817_Metzger_Mean"
echo "Ensemble Size:     4 models x 100 samples = 400 tasks (1M transients each)"
echo "Memory per Task:   128 GB"
echo "Output Directory:  results/07_gw170817_rate_limits"
echo "Date/Time:         $(date)"
echo "Execution Host:    $(hostname -f 2>/dev/null || hostname)"
echo "Slurm Job ID:      ${SLURM_JOB_ID:-INTERACTIVE_SHELL}"
echo "=============================================================================="

# 1. Submit simulation array (400 tasks)
echo "--> Submitting Suite 07 simulation array (tasks 0-399)..."
sim_job_id=$(sbatch --parsable scripts/07_run_gw170817_rate_limits.sh)
echo "    Simulation Job ID: ${sim_job_id}"

# 2. Submit aggregation, comparison, and notebook rendering job (dependent on array completion)
echo "--> Submitting aggregator & Suite 01 comparative analyzer (afterany:${sim_job_id})..."
agg_job_id=$(sbatch --parsable --dependency="afterany:${sim_job_id}" scripts/07_aggregate_and_compare.sh)
echo "    Aggregator Job ID: ${agg_job_id}"

echo "=============================================================================="
echo "SUITE 07 PIPELINE SUBMITTED SUCCESSFULLY"
echo "=============================================================================="
echo "  1. Simulation Array:             ${sim_job_id} (0-399)"
echo "  2. Distribution & Comparison:    ${agg_job_id} (queued on ${sim_job_id})"
echo ""
echo "Monitor status with:"
echo "  squeue -u \${USER:-\$USER}"
echo ""
echo "Logs will be written to:"
echo "  logs/07_gw170817_rate_limits/"
echo "=============================================================================="
