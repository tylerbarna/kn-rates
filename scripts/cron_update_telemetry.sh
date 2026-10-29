#!/bin/bash
# ==============================================================================
# PERIODIC CRON EXECUTION WRAPPER FOR JOB TELEMETRY TRACKER
# ==============================================================================
# Purpose:
#   Designed to be executed hourly via crontab (or a background runner).
#   Harvests Slurm sacct accounting and cluster task logs, updates
#   telemetry/job_resource_registry.csv and telemetry/scaling_models.json,
#   and optionally auto-commits changes to git.
# ==============================================================================
#SBATCH --job-name=telemetry_cron
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs telemetry

LOG_FILE="${PROJECT_DIR}/logs/telemetry_cron.log"

# Log rotation: if log file exceeds 5MB, keep the last 5,000 lines
if [ -f "${LOG_FILE}" ] && [ $(wc -c < "${LOG_FILE}") -gt 5242880 ]; then
    tail -n 5000 "${LOG_FILE}" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "${LOG_FILE}"
fi

exec >> "${LOG_FILE}" 2>&1

echo "=============================================================================="
echo "STARTING TELEMETRY CRON HARVEST RUN"
echo "Date/Time:        $(date)"
echo "Working Dir:      ${PROJECT_DIR}"
echo "User:             ${USER:-$(whoami)}"
echo "Host:             $(hostname -f 2>/dev/null || hostname)"
echo "=============================================================================="

# Ensure standard binaries and Slurm paths are accessible in cron environment
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/common/software/install/slurm/bin:${PATH:-}"

# Load modules on HPC clusters if available
if command -v module &>/dev/null; then
    module load conda 2>/dev/null || true
fi

eval "$(conda shell.bash hook 2>/dev/null || true)"
for env_target in kn_rates_dev simsurvey_no_gpu "$HOME/.conda/envs/simsurvey_no_gpu" "${PROJECT_DIR}/.conda"; do
    if conda env list 2>/dev/null | grep -q "$env_target" || [ -d "$env_target" ]; then
        conda activate "$env_target" 2>/dev/null || true
        break
    fi
done

PYTHON_BIN=""
for candidate in python3 python /opt/homebrew/bin/python3 /usr/bin/python3; do
    if command -v "$candidate" &>/dev/null && "$candidate" -c "import sys" &>/dev/null; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "Error: No operational Python interpreter found." >&2
    exit 1
fi

echo "Using Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)' 2>/dev/null || which ${PYTHON_BIN})"

# Run telemetry update
${PYTHON_BIN} -m telemetry update

echo "Telemetry registry updated successfully at $(date)."

# Optional Git Auto-Commit: Set AUTO_COMMIT=1 (or pass --commit) to version control changes
COMMIT_FLAG="${AUTO_COMMIT:-0}"
for arg in "$@"; do
    if [ "$arg" = "--commit" ]; then
        COMMIT_FLAG=1
    fi
done

if [ "${COMMIT_FLAG}" = "1" ] && command -v git &>/dev/null; then
    if ! git diff --quiet telemetry/job_resource_registry.csv telemetry/scaling_models.json 2>/dev/null; then
        echo "Detected changes in telemetry files. Committing to version control..."
        git add telemetry/job_resource_registry.csv telemetry/scaling_models.json
        git commit -m "chore(telemetry): auto-update job resource registry [skip ci]" || true
        echo "Committed updated registry at $(date)."
    else
        echo "No changes in telemetry files to commit."
    fi
fi

echo "=============================================================================="
