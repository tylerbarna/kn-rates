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

# Forward to fast criteria sweep pipeline to protect FairShare
echo "[INFO] Forwarding to fast criteria sweep pipeline (3 tasks total to protect MSI FairShare)..."
exec bash scripts/submit_fast_criteria_sweep.sh "$@"
