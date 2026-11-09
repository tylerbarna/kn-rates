#!/bin/bash --login
# ==============================================================================
# MASTER SLURM SUBMISSION WRAPPER: Concurrent Execution Across Suites
# ==============================================================================
# Purpose:
#   Submits all simulation and analysis suites to MSI's `msismall` partition
#   in a single command, leveraging concurrent cluster execution limits.
#
# Execution Architecture:
#   - Suite 01 (Rate Limits 100x1M Ensemble):  300 tasks (100 x 1M for Metzger, Bu2026Fixed, Bu2026Vary) [CONCURRENT, 128 GB/task]
#   - Suite 02 (Fast Criteria Sweep, 3 models):    3 tasks (Simulate-Once 1M sources for all 3 models) [CONCURRENT, 64 GB/task]
#   - Suite 03 (Prior vs AT2017gfo, 1M):         1 task  [CONCURRENT, 256 GB]
#   -----------------------------------------------------------------------
#   Total Initial Concurrent Tasks:            306 tasks (far below 5,000 submission limit)
#   Total Walltime to Complete:              ~2-4 hours (all tasks execute concurrently)
#   -----------------------------------------------------------------------
#   - Suite 02 Fast Filter & Aggregation:        1 task  [QUEUED: afterany:Suite 02, filters 2.14M ultra-fine grid]
# ==============================================================================
# Slurm resource directives to route output cleanly into logs/ (preventing root pollution):
#SBATCH -p msismall
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16g
#SBATCH --job-name=kn_submit_all
#SBATCH --output=logs/submit_all_%j.log
#SBATCH --error=logs/submit_all_%j.err

set -euo pipefail

# Error trap for informative diagnostic logging on failure
trap 'echo "[ERROR] Pipeline submission aborted unexpectedly at line ${LINENO} in command: ${BASH_COMMAND}" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

# Ensure standard log and output directories exist prior to any job dispatch
mkdir -p logs \
         logs/01_large_scale_rate_limits \
         logs/02_granular_criteria_sweep \
         logs/03_prior_vs_at2017gfo \
         logs/04_historical_andreoni_comparison \
         logs/05_sample_convergence \
         logs/06_inclination_effect \
         logs/07_gw170817_rate_limits \
         results/01_large_scale_rate_limits \
         results/02_granular_criteria_sweep \
         results/03_prior_vs_at2017gfo \
         results/04_historical_andreoni_comparison \
         results/05_sample_convergence \
         results/06_inclination_effect \
         results/07_gw170817_rate_limits

# If invoked interactively (outside Slurm batch), duplicate output into a timestamped log
if [ -z "${SLURM_JOB_ID:-}" ] && [ -t 1 ]; then
    INTERACTIVE_LOG="logs/submit_all_$(date +%Y%m%d_%H%M%S).log"
    echo "[INFO] Running interactively; teeing execution log to ${INTERACTIVE_LOG}"
    exec > >(tee -a "${INTERACTIVE_LOG}") 2>&1
fi

echo "=============================================================================="
echo "LAUNCHING COMPLETE ZTF 8-YEAR KILONOVA SIMULATION & ANALYSIS PIPELINE"
echo "Project Directory: ${PROJECT_DIR}"
echo "Current Date/Time: $(date)"
echo "Execution Host:    $(hostname -f 2>/dev/null || hostname)"
echo "Executing User:    ${USER:-unknown}"
echo "Slurm Job ID:      ${SLURM_JOB_ID:-INTERACTIVE_SHELL}"
echo "Git Branch/Commit: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown') / $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "=============================================================================="

# Environment diagnostics
PYTHON_BIN="$(which python3 2>/dev/null || which python 2>/dev/null || echo 'not found')"
echo "[ENV] Python binary:     ${PYTHON_BIN}"
if [ "${PYTHON_BIN}" != "not found" ]; then
    echo "[ENV] Python version:    $(${PYTHON_BIN} --version 2>&1)"
fi
echo "[ENV] Active Conda Env:  ${CONDA_DEFAULT_ENV:-none}"
if ! command -v sbatch &>/dev/null; then
    echo "[ERROR] 'sbatch' command not found in PATH. Are you on an HPC login or compute node?" >&2
    exit 1
fi
echo "[ENV] Slurm version:     $(sbatch --version 2>&1)"
echo "------------------------------------------------------------------------------"

# NOTE (2026-11): Suites 04 (Andreoni comparison), 05 (sample convergence) and 06 (inclination) are no
# longer submitted. Their former scripts did not run simulations (hard-coded efficiencies / typed-in tables)
# and now live in defunct/superseded_suites/. The corresponding analyses are computed from the Suite 01/07
# realisations in notebooks 04-06 via the results snapshot (python -m knrates.collect).

# 1. Submit Suite 01: Rate Limits Ensemble (300 array tasks: 100 x 1M for Metzger, Bu2026Fixed, Bu2026Vary, 128 GB each)
echo "[1/4] Submitting Suite 01 (100 x 1M Ensemble Rate Limits, 300 array tasks, 128 GB/task)..."
job1_id=$(sbatch --parsable scripts/01_run_large_scale_rate_limits.sh)
echo "  --> Suite 01 Job ID: ${job1_id} (Array 0-299)"

# 2. Submit Suite 02: Fast Granular Detection Criteria Sweep (3 array tasks: Bu2026Fixed, Bu2026Vary, Metzger)
echo "[2/4] Submitting Suite 02 (Fast Simulate-Once 3-Model Candidate Sweep, 3 array tasks, 64 GB/task)..."
job2_id=$(sbatch --parsable --array=0-2 --export=ALL,MODE=fiducial scripts/02_run_fast_candidates_simulation.sh)
echo "  --> Suite 02 Candidate Simulation Job ID: ${job2_id} (Array 0-2: Bu2026Fixed, Bu2026Vary, Metzger)"

# 3. Submit Suite 03: Broad Priors vs. AT2017gfo Archetype Comparison (1 task, 1M transients, 256 GB)
echo "[3/4] Submitting Suite 03 (Broad Priors vs AT2017gfo Comparison, 1 task, 256 GB)..."
job3_id=$(sbatch --parsable scripts/03_run_prior_vs_at2017gfo_comparison.sh)
echo "  --> Suite 03 Job ID: ${job3_id}"

# 5. Submit Suite 07: GW170817 Fixed-Parameter Rate Limits Ensemble (400 tasks: 100 x 1M for 4 archetypes, 128 GB each)
echo "[4/5] Submitting Suite 07 (GW170817 Fixed Ensemble, 400 array tasks, 128 GB/task)..."
job7_id=$(sbatch --parsable scripts/07_run_gw170817_rate_limits.sh)
echo "  --> Suite 07 Job ID: ${job7_id} (Array 0-399)"

# Submit Suite 01 Aggregation & Notebook Generation (depends on Suite 01 completion, 16 GB)
echo "[+] Submitting Suite 01 Aggregation (afterany:${job1_id}, 16 GB)..."
agg1_id=$(sbatch --parsable --dependency="afterany:${job1_id}" scripts/01_aggregate_and_build_distributions.sh)
echo "  --> Suite 01 Aggregation Job ID: ${agg1_id} (queued on Suite 01 termination)"

# Submit Suite 07 Aggregation & Suite 01 Comparison (depends on both Suite 07 array and Suite 01 Aggregation completion)
echo "[+] Submitting Suite 07 Aggregation & Suite 01 Comparison (afterany:${job7_id},afterany:${agg1_id}, 16 GB)..."
agg7_id=$(sbatch --parsable --dependency="afterany:${job7_id},afterany:${agg1_id}" scripts/07_aggregate_and_compare.sh)
echo "  --> Suite 07 Aggregation & Comparison Job ID: ${agg7_id}"

# 7. Submit Suite 02 Fast Filter & Aggregation (depends on Suite 02 candidate simulation, allocated 32 GB with ultra_fine 2.14M grid)
echo "[+] Submitting Suite 02 Fast Filter & Aggregation (afterany:${job2_id}, 32 GB, ultra_fine)..."
audit_job_id=$(sbatch --parsable --dependency="afterany:${job2_id}" --export=ALL,GRID_PRESET=ultra_fine scripts/02_run_fast_filter.sh)
echo "  --> Suite 02 Filter & Aggregation Job ID: ${audit_job_id} (queued on Suite 02 termination)"

echo "=============================================================================="
echo "PIPELINE SUBMISSION SUMMARY"
echo "=============================================================================="
printf "%-12s | %-10s | %-14s | %-10s | %-28s\n" "Suite" "Job ID" "Tasks" "Memory" "Dependency"
echo "------------------------------------------------------------------------------"
printf "%-12s | %-10s | %-14s | %-10s | %-28s\n" "Suite 01" "${job1_id}" "300 tasks" "128 GB" "None (Immediate)"
printf "%-12s | %-10s | %-14s | %-10s | %-28s\n" "Suite 01 Agg" "${agg1_id}" "1 task" "16 GB" "afterany:${job1_id}"
printf "%-12s | %-10s | %-14s | %-10s | %-28s\n" "Suite 02" "${job2_id}" "3 tasks" "64 GB" "None (Immediate)"
printf "%-12s | %-10s | %-14s | %-10s | %-28s\n" "Suite 03" "${job3_id}" "1 task" "256 GB" "None (Immediate)"
printf "%-12s | %-10s | %-14s | %-10s | %-28s\n" "Suite 07" "${job7_id}" "400 tasks" "128 GB" "None (Immediate)"
printf "%-12s | %-10s | %-14s | %-10s | %-28s\n" "Suite 07 Agg" "${agg7_id}" "1 task" "16 GB" "afterany:${job7_id},${agg1_id}"
printf "%-12s | %-10s | %-14s | %-10s | %-28s\n" "Suite 02 Filter" "${audit_job_id}" "1 task" "32 GB" "afterany:${job2_id}"
echo "------------------------------------------------------------------------------"
echo "Total Active Concurrent Slots:            306 / 5,000 max queue limit"
echo "Estimated Time to Completion:             ~2 to 4 hours total wall-clock time"
echo "OOM Protection:                           All memory allocations boosted to 128-256 GB"
echo "Log Locations:                            logs/submit_all_*.log and logs/*/*.log"
echo "------------------------------------------------------------------------------"
echo "To monitor status across the cluster:"
echo "  squeue -u ${USER:-\$USER}"
echo "=============================================================================="

# Register newly submitted jobs in the version-controlled telemetry tracker
echo "[TELEMETRY] Registering newly submitted pipeline jobs in resource tracker..."
if [ "${PYTHON_BIN}" != "not found" ]; then
    "${PYTHON_BIN}" -m telemetry update || echo "[WARNING] Telemetry update encountered non-critical issue." >&2
else
    echo "[WARNING] Python not available; skipping telemetry update." >&2
fi

echo "[DONE] Pipeline launch process completed successfully at $(date)."
