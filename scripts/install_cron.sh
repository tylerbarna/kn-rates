#!/bin/bash
# ==============================================================================
# INSTALL / UNINSTALL / MANAGE TELEMETRY SCRONTAB JOB
# ==============================================================================
# Usage:
#   ./scripts/install_cron.sh             # Installs hourly scrontab job with auto-commit
#   ./scripts/install_cron.sh --install   # Same as above
#   ./scripts/install_cron.sh --uninstall # Removes telemetry job from scrontab
#   ./scripts/install_cron.sh --status    # Checks active cron status and recent log
# ==============================================================================
#SBATCH --job-name=telemetry_cron
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CRON_SCRIPT="${PROJECT_DIR}/scripts/cron_update_telemetry.sh"
CRON_TAG="# KN_RATES_TELEMETRY_TRACKER_CRON"
LOG_FILE="${PROJECT_DIR}/logs/telemetry_cron.log"

ACTION="${1:---install}"

case "${ACTION}" in
    --install)
        echo "=============================================================================="
        echo "INSTALLING HOURLY TELEMETRY TRACKER SCRONTAB JOB"
        echo "Script:    ${CRON_SCRIPT}"
        echo "Cadence:   Hourly at minute 0 (0 * * * *)"
        echo "=============================================================================="

        chmod +x "${CRON_SCRIPT}"

        # Scrontab block with explicit Slurm directives so jobs are cleanly named in squeue
        CRON_BLOCK="#SCRON --job-name=kn_telemetry_cron ${CRON_TAG}
#SCRON --time=00:15:00 ${CRON_TAG}
#SCRON --mem=2G ${CRON_TAG}
#SCRON --cpus-per-task=1 ${CRON_TAG}
0 * * * * ${CRON_SCRIPT} --commit ${CRON_TAG}"

        # Get existing scrontab excluding previous kn_rates telemetry entries
        EXISTING_CRON=$(scrontab -l 2>/dev/null | grep -v "${CRON_TAG}" || true)

        if [ -n "${EXISTING_CRON}" ]; then
            NEW_CRONTAB="${EXISTING_CRON}"$'\n'"${CRON_BLOCK}"
        else
            NEW_CRONTAB="${CRON_BLOCK}"
        fi

        echo "${NEW_CRONTAB}" | scrontab -

        echo "[+] Successfully installed scrontab entry:"
        echo "${CRON_BLOCK}"
        echo ""
        echo "Telemetry registry will update at the top of every hour under job name 'kn_telemetry_cron'."
        echo "To monitor cron activity:"
        echo "  tail -f ${LOG_FILE}"
        echo "=============================================================================="
        ;;

    --uninstall)
        echo "=============================================================================="
        echo "UNINSTALLING TELEMETRY TRACKER SCRONTAB JOB"
        echo "=============================================================================="

        EXISTING_CRON=$(scrontab -l 2>/dev/null | grep -v "${CRON_TAG}" || true)

        if [ -n "${EXISTING_CRON}" ]; then
            echo "${EXISTING_CRON}" | scrontab -
        else
            scrontab -r 2>/dev/null || true
        fi

        echo "[+] Removed telemetry tracker from crontab."
        echo "=============================================================================="
        ;;

    --status)
        echo "=============================================================================="
        echo "TELEMETRY TRACKER CRON STATUS"
        echo "=============================================================================="

        ACTIVE_ENTRY=$(scrontab -l 2>/dev/null | grep "${CRON_TAG}" || true)

        if [ -n "${ACTIVE_ENTRY}" ]; then
            echo "[*] Status: ACTIVE (Installed in scrontab)"
            echo "    Entry:  ${ACTIVE_ENTRY}"
        else
            echo "[*] Status: NOT INSTALLED in user scrontab"
        fi

        echo ""
        if [ -f "${LOG_FILE}" ]; then
            echo "--- Recent Execution Log (last 15 lines from ${LOG_FILE}) ---"
            tail -n 15 "${LOG_FILE}"
        else
            echo "--- No execution log found yet at ${LOG_FILE} ---"
        fi
        echo "=============================================================================="
        ;;

    *)
        echo "Usage: $0 [--install | --uninstall | --status]"
        exit 1
        ;;
esac
