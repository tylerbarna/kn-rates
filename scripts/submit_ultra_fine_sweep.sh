#!/bin/bash --login
# ==============================================================================
# AUTOMATED SLURM SUBMISSION: Ultra-Fine Sweep + Chained Audit Job
# ==============================================================================
# Usage:
#   ./scripts/submit_ultra_fine_sweep.sh
#
# Execution Architecture:
#   1. Submits 02_run_ultra_fine_sweep.sh (4,997 parallel array tasks on msismall)
#   2. Captures Array Job ID
#   3. Automatically submits 02_run_audit_job.sh with dependency:
#        --dependency=afterany:<array_job_id>
#
# NOTE: For the new Simulate-Once, Filter-Many pipeline (3 tasks total),
#       use `bash scripts/submit_fast_criteria_sweep.sh` instead.
# ==============================================================================
#SBATCH -p msismall
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --job-name=kn_sub_ultrafine
#SBATCH --output=logs/02_granular_criteria_sweep/submit_ultrafine_%j.log
#SBATCH --error=logs/02_granular_criteria_sweep/submit_ultrafine_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/02_granular_criteria_sweep results/02_granular_criteria_sweep

# If running interactively outside Slurm batch, log to timestamped file in logs/
if [ -z "${SLURM_JOB_ID:-}" ] && [ -t 1 ]; then
    INTERACTIVE_LOG="logs/02_granular_criteria_sweep/submit_ultrafine_$(date +%Y%m%d_%H%M%S).log"
    echo "[INFO] Running interactively; teeing execution log to ${INTERACTIVE_LOG}"
    exec > >(tee -a "${INTERACTIVE_LOG}") 2>&1
fi

echo "=============================================================================="
echo "LAUNCHING ULTRA-FINE CRITERIA SWEEP PIPELINE (2,138,460 POINTS)"
echo "Project Directory: ${PROJECT_DIR}"
echo "Current Date/Time: $(date)"
echo "Execution Host:    $(hostname -f 2>/dev/null || hostname)"
echo "Slurm Job ID:      ${SLURM_JOB_ID:-INTERACTIVE_SHELL}"
echo "=============================================================================="

# 1. Forward directly to fast pipeline to protect MSI FairShare scores
echo "[INFO] Forwarding to fast criteria sweep pipeline (3 tasks total to protect MSI FairShare)..."
exec bash scripts/submit_fast_criteria_sweep.sh --fiducial --grid-preset ultra_fine "$@"
echo "and run all cells."
echo "=============================================================================="
