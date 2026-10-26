#!/bin/bash
# ==============================================================================
# MASTER SLURM SUBMISSION WRAPPER: Concurrent Execution Across Suites
# ==============================================================================
# Purpose:
#   Submits all 4 simulation and analysis suites to MSI's `msismall` partition
#   in a single command, leveraging your 3,000 concurrent job limit.
#
# Execution Architecture:
#   - Suite 01 (Rate Limits 100x1M Ensemble):  300 tasks (100 x 1M for Metzger, Bu2026Fixed, Bu2026Vary) [CONCURRENT, 128 GB/task]
#   - Suite 02 (Granular Criteria Sweep):    4,560 tasks (68,400 points, 15 pts/task, 1M sources) [CONCURRENT, 128 GB/task]
#   - Suite 03 (Prior vs AT2017gfo, 1M):         1 task  [CONCURRENT, 128 GB]
#   - Suite 06 (Inclination Investigation):      1 task  [CONCURRENT, 128 GB]
#   -----------------------------------------------------------------------
#   Total Initial Concurrent Tasks:          4,862 tasks (< 5,000 submission limit)
#   Total Walltime to Complete:              ~2-4 hours (all tasks execute concurrently)
#   -----------------------------------------------------------------------
#   - Suite 04 (Historical Andreoni Comp):       1 task  [QUEUED: afterok:Suite 01]
#   - Sweep Audit & Aggregation:                 1 task  [QUEUED: afterok:Suite 02]
#   - Suite 05 (Sample Size Convergence):        1 task  [QUEUED: afterok:Suite 04]
# ==============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/01_large_scale_rate_limits \
         logs/02_granular_criteria_sweep \
         logs/03_prior_vs_at2017gfo \
         logs/04_historical_andreoni_comparison \
         logs/06_inclination_effect

echo "=============================================================================="
echo "LAUNCHING COMPLETE ZTF 8-YEAR KILONOVA SIMULATION & ANALYSIS PIPELINE"
echo "Project Directory: ${PROJECT_DIR}"
echo "Current Date/Time: $(date)"
echo "=============================================================================="

# 1. Submit Suite 01: Rate Limits Ensemble (300 array tasks: 100 x 1M for Metzger, Bu2026Fixed, Bu2026Vary, 128 GB each)
echo "[1/6] Submitting Suite 01 (100 x 1M Ensemble Rate Limits, 300 array tasks, 128GB/task)..."
job1_id=$(sbatch --parsable scripts/01_run_large_scale_rate_limits.sh)
echo "  --> Job ID: ${job1_id}"

# 2. Submit Suite 02: Granular Detection Criteria Sweep (4,560 array tasks, 15 points/task, 1M sources, 128 GB each)
echo "[2/6] Submitting Suite 02 (Granular Criteria Sweep, 4,560 array tasks, 128GB/task)..."
job2_id=$(sbatch --parsable scripts/02_run_granular_criteria_sweep.sh)
echo "  --> Job ID: ${job2_id}"

# 3. Submit Suite 03: Broad Priors vs. AT2017gfo Archetype Comparison (1 task, 1M transients, 128 GB)
echo "[3/6] Submitting Suite 03 (Broad Priors vs AT2017gfo Comparison, 1 task, 1M transients, 128GB)..."
job3_id=$(sbatch --parsable scripts/03_run_prior_vs_at2017gfo_comparison.sh)
echo "  --> Job ID: ${job3_id}"

# 4. Submit Suite 06: Inclination Discrepancy Investigation (1 task, 128 GB)
echo "[4/6] Submitting Suite 06 (Inclination Discrepancy Investigation, 1 task, 128GB)..."
job6_id=$(sbatch --parsable scripts/06_run_inclination_investigation.sh)
echo "  --> Job ID: ${job6_id}"

# 5. Submit Suite 04: Historical Andreoni Comparison (depends on Suite 01 completion)
echo "[5/6] Submitting Suite 04 (Historical Andreoni Comparison, afterok:${job1_id})..."
job4_id=$(sbatch --parsable --dependency="afterok:${job1_id}" scripts/04_run_andreoni_benchmark_comparison.sh)
echo "  --> Job ID: ${job4_id} (queued on Suite 01)"

# 6. Submit Sweep Audit (depends on Suite 02 completion)
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

# 7. Submit Suite 05: Sample Size Convergence (10k vs 100k vs 1M vs 10M, runs at end)
echo "[6/6] Submitting Suite 05 (Sample Size Convergence Grid, afterok:${job4_id})..."
job5_id=$(sbatch --parsable --dependency="afterok:${job4_id}" scripts/05_run_sample_convergence.sh)
echo "  --> Job ID: ${job5_id} (queued on Suite 04 to run at pipeline conclusion)"

echo "=============================================================================="
echo "PIPELINE SUBMISSION SUMMARY"
echo "=============================================================================="
echo "  Suite 01 (100x1M Ensemble Rate Limits): ${job1_id} (300 tasks @ 128 GB)"
echo "  Suite 02 (Criteria Sweep 4,560 tasks):  ${job2_id} (4,560 tasks @ 128 GB)"
echo "  Suite 03 (Prior vs AT2017gfo 1M):       ${job3_id} (1 task @ 128 GB)"
echo "  Suite 04 (Andreoni Comparison):         ${job4_id} (dependent on ${job1_id})"
echo "  Suite 05 (Sample Convergence Grid):     ${job5_id} (dependent on ${job4_id})"
echo "  Suite 06 (Inclination Study):           ${job6_id} (1 task @ 128 GB)"
echo "  Audit    (Criteria Sweep Audit):        ${audit_job_id} (dependent on ${job2_id})"
echo "------------------------------------------------------------------------------"
echo "Total Active Concurrent Slots:            4,862 / 5,000 max queue limit"
echo "Estimated Time to Completion:             ~2 to 4 hours total wall-clock time"
echo "OOM Protection:                           All memory allocations boosted to 128 GB"
echo "------------------------------------------------------------------------------"
echo "To monitor status across the cluster:"
echo "  squeue -u \$USER"
echo "=============================================================================="
