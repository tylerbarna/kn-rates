#!/bin/bash
# ==============================================================================
# Update the version-controlled results snapshot (snapshots/knrates_results_latest.zip)
# ==============================================================================
# Called at the end of every aggregation/filter job. It re-reads only the suites
# whose result files changed, validates every table and replaces the snapshot
# atomically. If validation fails the previous snapshot is kept and this script
# exits non-zero (callers use `|| true` so a snapshot problem never fails the
# science job itself; check the log).
#
# Usage:
#   bash scripts/update_results_snapshot.sh                # update 'latest'
#   bash scripts/update_results_snapshot.sh --timestamped  # also keep a dated copy
#   SNAPSHOT_TIMESTAMPED=1 sbatch ...                      # same, via environment
# Extra arguments are passed to `python -m knrates.collect`.
# ==============================================================================
set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}" || exit 1
ARGS=("$@")
if [ "${SNAPSHOT_TIMESTAMPED:-0}" = "1" ]; then
    ARGS+=(--timestamped --also-latest)
fi
echo "[snapshot] updating results snapshot at $(date) (args: ${ARGS[*]:-none})"
if python -m knrates.collect "${ARGS[@]}"; then
    echo "[snapshot] snapshot updated. Commit snapshots/ to version-control the new results."
else
    status=$?
    echo "[snapshot] WARNING: snapshot NOT updated (exit ${status}); previous snapshot left untouched." >&2
    exit "${status}"
fi
