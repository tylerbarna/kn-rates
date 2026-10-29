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
#   - Suite 02 (Granular Criteria Sweep):    4,560 tasks (68,400 points, 15 pts/task, 1M sources) [CONCURRENT, 128 GB/task]
#   - Suite 03 (Prior vs AT2017gfo, 1M):         1 task  [CONCURRENT, 256 GB]
#   - Suite 06 (Inclination Investigation):      1 task  [CONCURRENT, 128 GB]
#   -----------------------------------------------------------------------
#   Total Initial Concurrent Tasks:          4,862 tasks (< 5,000 submission limit)
#   Total Walltime to Complete:              ~2-4 hours (all tasks execute concurrently)
#   -----------------------------------------------------------------------
#   - Suite 04 (Historical Andreoni Comp):       1 task  [QUEUED: afterany:Suite 01]
#   - Sweep Audit & Aggregation:                 1 task  [QUEUED: afterany:Suite 02]
#   - Suite 05 (Sample Size Convergence):        1 task  [QUEUED: afterany:Suite 04]
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

# 1. Submit Suite 01: Rate Limits Ensemble (300 array tasks: 100 x 1M for Metzger, Bu2026Fixed, Bu2026Vary, 128 GB each)
echo "[1/6] Submitting Suite 01 (100 x 1M Ensemble Rate Limits, 300 array tasks, 128 GB/task)..."
job1_id=$(sbatch --parsable scripts/01_run_large_scale_rate_limits.sh)
echo "  --> Suite 01 Job ID: ${job1_id} (Array 0-299)"

# 2. Submit Suite 02: Granular Detection Criteria Sweep (4,560 array tasks, 15 points/task, 1M sources, 128 GB each)
echo "[2/6] Submitting Suite 02 (Granular Criteria Sweep, 4,560 array tasks, 128 GB/task)..."
job2_id=$(sbatch --parsable scripts/02_run_granular_criteria_sweep.sh)
echo "  --> Suite 02 Job ID: ${job2_id} (Array 0-4559)"

# 3. Submit Suite 03: Broad Priors vs. AT2017gfo Archetype Comparison (1 task, 1M transients, 256 GB)
echo "[3/6] Submitting Suite 03 (Broad Priors vs AT2017gfo Comparison, 1 task, 256 GB)..."
job3_id=$(sbatch --parsable scripts/03_run_prior_vs_at2017gfo_comparison.sh)
echo "  --> Suite 03 Job ID: ${job3_id}"

# 4. Submit Suite 06: Inclination Discrepancy Investigation (1 task, 128 GB)
echo "[4/7] Submitting Suite 06 (Inclination Discrepancy Investigation, 1 task, 128 GB)..."
job6_id=$(sbatch --parsable scripts/06_run_inclination_investigation.sh)
echo "  --> Suite 06 Job ID: ${job6_id}"

# 5. Submit Suite 07: GW170817 Fixed-Parameter Rate Limits Ensemble (400 tasks: 100 x 1M for 4 archetypes, 128 GB each)
echo "[5/7] Submitting Suite 07 (GW170817 Fixed Ensemble, 400 array tasks, 128 GB/task)..."
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

# 6. Submit Suite 04: Historical Andreoni Comparison (depends on Suite 01 Aggregation completion)
echo "[6/7] Submitting Suite 04 (Historical Andreoni Comparison, afterany:${agg1_id})..."
job4_id=$(sbatch --parsable --dependency="afterany:${agg1_id}" scripts/04_run_andreoni_benchmark_comparison.sh)
echo "  --> Suite 04 Job ID: ${job4_id} (queued on Suite 01 Aggregation termination)"

# 7. Submit Sweep Audit & Sensitivity Analysis (depends on Suite 02 completion, allocated 16 GB with full conda environment)
echo "[+] Submitting Sweep Audit & Sensitivity Analysis (afterany:${job2_id}, 16 GB)..."
audit_job_id=$(sbatch --parsable --dependency="afterany:${job2_id}" scripts/02_run_audit_job.sh)
echo "  --> Sweep Audit Job ID: ${audit_job_id} (queued on Suite 02 termination)"

# 8. Submit Suite 05: Sample Size Convergence (10k vs 100k vs 1M vs 10M, runs after Suite 04)
echo "[7/7] Submitting Suite 05 (Sample Size Convergence Grid, afterany:${job4_id})..."
job5_id=$(sbatch --parsable --dependency="afterany:${job4_id}" scripts/05_run_sample_convergence.sh)
echo "  --> Suite 05 Job ID: ${job5_id} (queued on Suite 04 termination)"

echo "=============================================================================="
echo "PIPELINE SUBMISSION SUMMARY"
echo "=============================================================================="
printf "%-12s | %-10s | %-14s | %-10s | %-28s\n" "Suite" "Job ID" "Tasks" "Memory" "Dependency"
echo "------------------------------------------------------------------------------"
printf "%-12s | %-10s | %-14s | %-10s | %-28s\n" "Suite 01" "${job1_id}" "300 tasks" "128 GB" "None (Immediate)"
printf "%-12s | %-10s | %-14s | %-10s | %-28s\n" "Suite 01 Agg" "${agg1_id}" "1 task" "16 GB" "afterany:${job1_id}"
printf "%-12s | %-10s | %-14s | %-10s | %-28s\n" "Suite 02" "${job2_id}" "4,560 tasks" "128 GB" "None (Immediate)"
printf "%-12s | %-10s | %-14s | %-10s | %-28s\n" "Suite 03" "${job3_id}" "1 task" "256 GB" "None (Immediate)"
printf "%-12s | %-10s | %-14s | %-10s | %-28s\n" "Suite 06" "${job6_id}" "1 task" "128 GB" "None (Immediate)"
printf "%-12s | %-10s | %-14s | %-10s | %-28s\n" "Suite 07" "${job7_id}" "400 tasks" "128 GB" "None (Immediate)"
printf "%-12s | %-10s | %-14s | %-10s | %-28s\n" "Suite 07 Agg" "${agg7_id}" "1 task" "16 GB" "afterany:${job7_id},${agg1_id}"
printf "%-12s | %-10s | %-14s | %-10s | %-28s\n" "Suite 04" "${job4_id}" "1 task" "256 GB" "afterany:${agg1_id}"
printf "%-12s | %-10s | %-14s | %-10s | %-28s\n" "Sweep Audit" "${audit_job_id}" "1 task" "16 GB" "afterany:${job2_id}"
printf "%-12s | %-10s | %-14s | %-10s | %-28s\n" "Suite 05" "${job5_id}" "1 task" "750 GB" "afterany:${job4_id}"
echo "------------------------------------------------------------------------------"
echo "Total Active Concurrent Slots:            4,862 / 5,000 max queue limit"
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
