#!/bin/bash
# ==============================================================================
# AUTOMATED SLURM SUBMISSION: Ultra-Fine Sweep + Chained Audit Job
# ==============================================================================
# Usage:
#   ./scripts/submit_ultra_fine_sweep.sh
#
# Execution Architecture:
#   1. Submits 02_run_ultra_fine_sweep.sh (4,952 parallel array tasks on msismall)
#   2. Captures Array Job ID
#   3. Automatically submits 02_run_audit_job.sh with dependency:
#        --dependency=afterok:<array_job_id>
# ==============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/02_granular_criteria_sweep results/02_granular_criteria_sweep

echo "=============================================================================="
echo "LAUNCHING ULTRA-FINE CRITERIA SWEEP PIPELINE (158,460 POINTS)"
echo "Project Directory: ${PROJECT_DIR}"
echo "Current Date/Time: $(date)"
echo "=============================================================================="

# 1. Submit the array sweep job
echo "[1/2] Submitting Ultra-Fine Sweep Array (4,952 tasks, 32 pts/task)..."
array_job_id=$(sbatch --parsable scripts/02_run_ultra_fine_sweep.sh)
echo "  --> Array Job ID: ${array_job_id}"

# 2. Submit the dependent audit job
echo "[2/2] Submitting Automated Audit & Aggregation (afterok:${array_job_id})..."
audit_job_id=$(sbatch --parsable --dependency="afterok:${array_job_id}" scripts/02_run_audit_job.sh)
echo "  --> Audit Job ID: ${audit_job_id} (queued on ${array_job_id})"

echo "=============================================================================="
echo "SUBMISSION COMPLETE"
echo "=============================================================================="
echo "  1. Array Tasks:   ${array_job_id} [0-4951] (actively computing on msismall)"
echo "  2. Audit Job:     ${audit_job_id} (held until task 4951 succeeds)"
echo "------------------------------------------------------------------------------"
echo "Once the audit completes, open notebooks/02_granular_criteria_sweep_analysis.ipynb"
echo "and run all cells."
echo "=============================================================================="
