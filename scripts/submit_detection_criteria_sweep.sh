#!/bin/bash --login
# ==============================================================================
# SUBMITTER: Detection Criteria Sweep (v2) Pipeline
# ==============================================================================
#SBATCH -p msismall
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --job-name=kn_sub_crit_v2
#SBATCH --output=logs/detection_criteria_sweep_v2/submit_criteria_sweep_%j.log
#SBATCH --error=logs/detection_criteria_sweep_v2/submit_criteria_sweep_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/detection_criteria_sweep_v2 results/detection_criteria_sweep_v2

# If running interactively outside Slurm batch, log to timestamped file in logs/
if [ -z "${SLURM_JOB_ID:-}" ] && [ -t 1 ]; then
    INTERACTIVE_LOG="logs/detection_criteria_sweep_v2/submit_criteria_sweep_$(date +%Y%m%d_%H%M%S).log"
    echo "[INFO] Running interactively; teeing execution log to ${INTERACTIVE_LOG}"
    exec > >(tee -a "${INTERACTIVE_LOG}") 2>&1
fi

echo "=============================================================================="
echo "LAUNCHING DETECTION CRITERIA SWEEP (V2) PIPELINE"
echo "Project Directory: ${PROJECT_DIR}"
echo "Current Date/Time: $(date)"
echo "Execution Host:    $(hostname -f 2>/dev/null || hostname)"
echo "Slurm Job ID:      ${SLURM_JOB_ID:-INTERACTIVE_SHELL}"
echo "=============================================================================="

array_job_id=$(sbatch --parsable scripts/run_detection_criteria_sweep.sh)
quality_job_id=$(sbatch --parsable \
  --dependency="afterany:${array_job_id}" \
  scripts/run_detection_criteria_quality.sh)
rate_limit_job_id=$(sbatch --parsable \
  --dependency="afterany:${array_job_id}" \
  scripts/run_rate_limit.sh)

printf 'Submitted detection array: %s\n' "$array_job_id"
printf 'Submitted quality audit: %s (afterany:%s)\n' "$quality_job_id" "$array_job_id"
printf 'Submitted rate limit: %s (afterany:%s)\n' "$rate_limit_job_id" "$array_job_id"
echo "Logs will be written to logs/detection_criteria_sweep_v2/"
echo "=============================================================================="
