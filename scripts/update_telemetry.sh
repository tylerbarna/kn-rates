#!/bin/bash
# ==============================================================================
# UPDATE JOB RESOURCE USAGE & QUEUE TELEMETRY REGISTRY
# ==============================================================================
# Harvests kernel metrics from Slurm sacct, parses task logs, updates
# telemetry/job_resource_registry.csv and telemetry/scaling_models.json,
# and outputs an updated queue and rightsizing summary.
# ==============================================================================
#SBATCH --job-name=telemetry

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

# Activate Conda environment if available
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

echo "=============================================================================="
echo "UPDATING JOB RESOURCE USAGE & QUEUE TELEMETRY REGISTRY"
echo "Date/Time:        $(date)"
echo "Host:             $(hostname -f 2>/dev/null || hostname)"
echo "Python:           $(${PYTHON_BIN} -c 'import sys; print(sys.executable)' 2>/dev/null || which ${PYTHON_BIN})"
echo "=============================================================================="

${PYTHON_BIN} -m telemetry update "$@"
echo ""
${PYTHON_BIN} -m telemetry report

echo ""
echo "Telemetry registry updated: telemetry/job_resource_registry.csv"
echo "Scaling model updated:     telemetry/scaling_models.json"
echo "=============================================================================="
