#!/bin/bash
# ==============================================================================
# MASTER SLURM SUBMISSION WRAPPER: Concurrent Execution Across Suites
# ==============================================================================
# Purpose:
#   Submits all 4 simulation and analysis suites to MSI's `msismall` partition
#   in a single command, leveraging your 3,000 concurrent job limit.
#
# Execution Architecture:
#   - Suite 01 (Large-Scale Rate Limits):     3 tasks   [RUNS CONCURRENTLY]
#   - Suite 02 (Granular Criteria Sweep): 4,248 tasks (42,480 points, 10 pts/task, 1M sources, <= 5,000 limit)
#   - Suite 03 (Prior vs AT2017gfo):           1 task    [RUNS CONCURRENTLY]
#   -----------------------------------------------------------------------
#   Total Concurrent Tasks:                 4,252 tasks (< 5,000 submission limit)
#   Total Walltime to Finish All 3:         ~8 hours (parallelized)
#   -----------------------------------------------------------------------
#   - Suite 04 (Historical Andreoni Comp):     1 task    [QUEUED: afterok:Suite 01]
#   - Sweep Audit & Aggregation:               1 task    [QUEUED: afterok:Suite 02]
# ==============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/01_large_scale_rate_limits \
         logs/02_granular_criteria_sweep \
         logs/03_prior_vs_at2017gfo \
         logs/04_historical_andreoni_comparison

echo "=============================================================================="
echo "LAUNCHING COMPLETE ZTF 8-YEAR KILONOVA SIMULATION & ANALYSIS PIPELINE"
echo "Project Directory: ${PROJECT_DIR}"
echo "Current Date/Time: $(date)"
echo "=============================================================================="

# 1. Submit Suite 01: Large-Scale Rate Limits (3 array tasks: Metzger, Bu2026Fixed, Bu2026Vary)
echo "[1/4] Submitting Suite 01 (Large-Scale Rate Limits, 3 array tasks)..."
job1_id=$(sbatch --parsable scripts/01_run_large_scale_rate_limits.sh)
echo "  --> Job ID: ${job1_id}"

# 2. Submit Suite 02: Granular Detection Criteria Sweep (4,248 array tasks, 10 points/task, 1M sources)
echo "[2/4] Submitting Suite 02 (Granular Criteria Sweep, 4,248 array tasks, 1M sources)..."
job2_id=$(sbatch --parsable scripts/02_run_granular_criteria_sweep.sh)
echo "  --> Job ID: ${job2_id}"

# 3. Submit Suite 03: Broad Priors vs. AT2017gfo Archetype Comparison (1 task)
echo "[3/4] Submitting Suite 03 (Broad Priors vs AT2017gfo Comparison, 1 task)..."
job3_id=$(sbatch --parsable scripts/03_run_prior_vs_at2017gfo_comparison.sh)
echo "  --> Job ID: ${job3_id}"

# 4. Submit Suite 04: Historical Andreoni Comparison (depends on Suite 01 completion)
echo "[4/5] Submitting Suite 04 (Historical Andreoni Comparison, afterok:${job1_id})..."
job4_id=$(sbatch --parsable --dependency="afterok:${job1_id}" scripts/04_run_andreoni_benchmark_comparison.sh)
echo "  --> Job ID: ${job4_id} (queued on Suite 01)"

# 5. Submit Sweep Audit (depends on Suite 02 completion)
echo "[+] Submitting Sweep Audit & Sensitivity Analysis (afterok:${job2_id})..."
audit_job_id=$(sbatch --parsable \
  --dependency="afterok:${job2_id}" \
  --partition=msismall \
  --time=00:30:00 \
  --cpus-per-task=1 \
  --mem=3900M \
  --output=logs/02_granular_criteria_sweep/audit_%j.log \
  --error=logs/02_granular_criteria_sweep/audit_%j.err \
  --wrap="python analysis_suites/02_granular_criteria_sweep/audit_granular_sweep.py")
echo "  --> Job ID: ${audit_job_id} (queued on Suite 02)"

# 6. Submit Suite 05: Sample Size Convergence (10k vs 100k vs 1M vs 10M, runs at end)
echo "[5/5] Submitting Suite 05 (Sample Size Convergence Grid, afterok:${job4_id})..."
job5_id=$(sbatch --parsable --dependency="afterok:${job4_id}" scripts/05_run_sample_convergence.sh)
echo "  --> Job ID: ${job5_id} (queued on Suite 04 to run at pipeline conclusion)"

echo "=============================================================================="
echo "PIPELINE SUBMISSION SUMMARY"
echo "=============================================================================="
echo "  Suite 01 (Rate Limits 10M 8-Yr):    ${job1_id} (3 tasks)"
echo "  Suite 02 (Criteria Sweep 2,592):    ${job2_id} (2,592 tasks)"
echo "  Suite 03 (Prior vs AT2017gfo):      ${job3_id} (1 task)"
echo "  Suite 04 (Andreoni Comparison):     ${job4_id} (dependent on ${job1_id})"
echo "  Suite 05 (Sample Convergence Grid): ${job5_id} (dependent on ${job4_id})"
echo "  Audit    (Criteria Sweep Audit):    ${audit_job_id} (dependent on ${job2_id})"
echo "------------------------------------------------------------------------------"
echo "Total Active Concurrent Slots:       2,596 / 3,000 max"
echo "Estimated Time to Completion:        ~2 hours for sweep, ~21-36 hours for 10M rate limits"
echo "------------------------------------------------------------------------------"
echo "To monitor status across the cluster:"
echo "  squeue -u \$USER"
echo "=============================================================================="
