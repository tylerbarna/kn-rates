#!/bin/bash
# ==============================================================================
# ATTACH AUDIT DEPENDENCY TO AN ACTIVE SLURM SWEEP JOB
# ==============================================================================
# Usage:
#   ./scripts/attach_audit_to_job.sh [JOB_ID]
#
# If JOB_ID is omitted, it auto-detects your active sweep job array on MSI.
# ==============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

JOB_ID="${1:-}"

if [ -z "${JOB_ID}" ]; then
    echo "Querying active criteria sweep jobs for user '${USER:-}'..."
    ACTIVE_JOB=$(squeue -u "${USER:-}" -o "%A %j" -h | grep -E "kn_crit" | head -n 1 | awk '{print $1}' || true)
    if [ -n "${ACTIVE_JOB}" ]; then
        JOB_ID="${ACTIVE_JOB}"
        echo "Auto-detected running job array: ${JOB_ID}"
    else
        echo "Error: No active 'kn_crit' job array found. Please supply the Job ID manually:"
        echo "  ./scripts/attach_audit_to_job.sh <JOB_ID>"
        exit 1
    fi
fi

echo "=============================================================================="
echo "ATTACHING AUDIT JOB TO RUNNING SWEEP: ${JOB_ID}"
echo "=============================================================================="

audit_job_id=$(sbatch --parsable --dependency="afterok:${JOB_ID}" scripts/02_run_audit_job.sh)
echo "  [+] Dependent Audit Job Submitted: ${audit_job_id}"
echo "  --> State: PENDING (held by Slurm until sweep job ${JOB_ID} completes)"
echo "=============================================================================="
