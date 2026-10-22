#!/bin/bash
# ==============================================================================
# STANDALONE TELEMETRY BACKGROUND DAEMON RUNNER
# ==============================================================================
# Purpose:
#   Runs an hourly telemetry update loop in the background. Useful when
#   system crontab is restricted (e.g. HPC nodes, container sandboxes) or
#   when user prefers a persistent background process.
#
# Usage:
#   ./scripts/run_telemetry_daemon.sh start     # Starts daemon in background
#   ./scripts/run_telemetry_daemon.sh stop      # Stops running daemon
#   ./scripts/run_telemetry_daemon.sh status    # Checks daemon PID and status
#   ./scripts/run_telemetry_daemon.sh run       # Runs in foreground (Ctrl+C to stop)
# ==============================================================================

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PID_FILE="${PROJECT_DIR}/logs/telemetry_daemon.pid"
LOG_FILE="${PROJECT_DIR}/logs/telemetry_cron.log"
CRON_SCRIPT="${PROJECT_DIR}/scripts/cron_update_telemetry.sh"
INTERVAL_SECONDS="${TELEMETRY_INTERVAL_SEC:-3600}"

mkdir -p "${PROJECT_DIR}/logs"

ACTION="${1:-status}"

case "${ACTION}" in
    start)
        if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
            echo "[*] Telemetry daemon is already running (PID: $(cat "${PID_FILE}"))."
            exit 0
        fi

        echo "Starting telemetry daemon (interval: ${INTERVAL_SECONDS}s / 1 hour)..."
        nohup "${CRON_SCRIPT}" --commit >> "${LOG_FILE}" 2>&1
        (
            while true; do
                sleep "${INTERVAL_SECONDS}"
                "${CRON_SCRIPT}" --commit >> "${LOG_FILE}" 2>&1
            done
        ) >/dev/null 2>&1 &

        DAEMON_PID=$!
        echo "${DAEMON_PID}" > "${PID_FILE}"
        echo "[+] Telemetry daemon started successfully with PID: ${DAEMON_PID}"
        echo "    Logging to: ${LOG_FILE}"
        ;;

    stop)
        if [ -f "${PID_FILE}" ]; then
            PID=$(cat "${PID_FILE}")
            if kill -0 "${PID}" 2>/dev/null; then
                kill "${PID}" && rm -f "${PID_FILE}"
                echo "[+] Stopped telemetry daemon (PID: ${PID})."
            else
                rm -f "${PID_FILE}"
                echo "[*] Removed stale PID file (process ${PID} was not running)."
            fi
        else
            echo "[*] No telemetry daemon PID file found."
        fi
        ;;

    status)
        if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
            echo "[*] Telemetry daemon: RUNNING (PID: $(cat "${PID_FILE}") | Interval: ${INTERVAL_SECONDS}s)"
            echo "--- Recent Log (from ${LOG_FILE}) ---"
            tail -n 10 "${LOG_FILE}" 2>/dev/null || true
        else
            echo "[*] Telemetry daemon: STOPPED"
        fi
        ;;

    run)
        echo "Running telemetry loop in foreground (Interval: ${INTERVAL_SECONDS}s, Press Ctrl+C to stop)..."
        while true; do
            "${CRON_SCRIPT}" --commit
            echo "Sleeping for ${INTERVAL_SECONDS} seconds until next harvest..."
            sleep "${INTERVAL_SECONDS}"
        done
        ;;

    *)
        echo "Usage: $0 {start|stop|status|run}"
        exit 1
        ;;
esac
