#!/bin/bash
# ==============================================================================
# INSTALL / UNINSTALL / MANAGE TELEMETRY CRON JOB
# ==============================================================================
# Usage:
#   ./scripts/install_cron.sh             # Installs hourly cron job with auto-commit
#   ./scripts/install_cron.sh --install   # Same as above
#   ./scripts/install_cron.sh --uninstall # Removes telemetry cron job from crontab
#   ./scripts/install_cron.sh --status    # Checks active cron status and recent log
# ==============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CRON_SCRIPT="${PROJECT_DIR}/scripts/cron_update_telemetry.sh"
CRON_TAG="# KN_RATES_TELEMETRY_TRACKER_CRON"
LOG_FILE="${PROJECT_DIR}/logs/telemetry_cron.log"

ACTION="${1:---install}"

case "${ACTION}" in
    --install)
        echo "=============================================================================="
        echo "INSTALLING HOURLY TELEMETRY TRACKER CRON JOB"
        echo "Script:    ${CRON_SCRIPT}"
        echo "Cadence:   Hourly at minute 0 (0 * * * *)"
        echo "=============================================================================="

        chmod +x "${CRON_SCRIPT}"

        CRON_CMD="0 * * * * ${CRON_SCRIPT} --commit ${CRON_TAG}"

        # Get existing crontab excluding previous kn_rates telemetry entries
        EXISTING_CRON=$(crontab -l 2>/dev/null | grep -v "${CRON_TAG}" || true)

        if [ -n "${EXISTING_CRON}" ]; then
            NEW_CRONTAB="${EXISTING_CRON}"$'\n'"${CRON_CMD}"
        else
            NEW_CRONTAB="${CRON_CMD}"
        fi

        echo "${NEW_CRONTAB}" | crontab -

        echo "[+] Successfully installed crontab entry:"
        echo "    ${CRON_CMD}"
        echo ""
        echo "Telemetry registry will update at the top of every hour."
        echo "To monitor cron activity:"
        echo "  tail -f ${LOG_FILE}"
        echo "=============================================================================="
        ;;

    --uninstall)
        echo "=============================================================================="
        echo "UNINSTALLING TELEMETRY TRACKER CRON JOB"
        echo "=============================================================================="

        EXISTING_CRON=$(crontab -l 2>/dev/null | grep -v "${CRON_TAG}" || true)

        if [ -n "${EXISTING_CRON}" ]; then
            echo "${EXISTING_CRON}" | crontab -
        else
            crontab -r 2>/dev/null || true
        fi

        echo "[+] Removed telemetry tracker from crontab."
        echo "=============================================================================="
        ;;

    --status)
        echo "=============================================================================="
        echo "TELEMETRY TRACKER CRON STATUS"
        echo "=============================================================================="

        ACTIVE_ENTRY=$(crontab -l 2>/dev/null | grep "${CRON_TAG}" || true)

        if [ -n "${ACTIVE_ENTRY}" ]; then
            echo "[*] Status: ACTIVE (Installed in crontab)"
            echo "    Entry:  ${ACTIVE_ENTRY}"
        else
            echo "[*] Status: NOT INSTALLED in user crontab"
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
