#!/bin/bash
# ==============================================================================
# SUBMITTER: 1M Randomized Parameter Large-Scale Rate Limit Suite
# ==============================================================================
# Purpose:
#   Submits the 1M randomized simulation array to MSI's `msismall` partition,
#   then automatically chains the aggregation and notebook execution job to run
#   as soon as all array tasks complete.
#
# Usage:
#   bash scripts/submit_1M_randomized_suite01.sh [ARRAY_RANGE]
#
# Examples:
#   bash scripts/submit_1M_randomized_suite01.sh           # Default: 30 tasks (0-29: 10 per model)
#   bash scripts/submit_1M_randomized_suite01.sh 0-59      # 60 tasks (20 per model)
#   bash scripts/submit_1M_randomized_suite01.sh 0-8       # 9 quick test tasks (3 per model)
# ==============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

ARRAY_RANGE="${1:-0-29}"

mkdir -p logs/01_large_scale_rate_limits results/01_large_scale_rate_limits

echo "=============================================================================="
echo "SUBMITTING 1M RANDOMIZED KILONOVA RATE LIMIT PIPELINE"
echo "Project Directory: ${PROJECT_DIR}"
echo "Array Range:       ${ARRAY_RANGE}"
echo "Date/Time:         $(date)"
echo "=============================================================================="

# 1. Submit simulation array
echo "--> Submitting 1M randomized simulation array (tasks ${ARRAY_RANGE})..."
sim_job_id=$(sbatch --parsable --array="${ARRAY_RANGE}" scripts/01_run_large_scale_1M_randomized.sh)
echo "    Simulation Job ID: ${sim_job_id}"

# 2. Submit aggregation and notebook rendering job (dependent on array completion)
echo "--> Submitting post-processing aggregator & notebook generator (afterok:${sim_job_id})..."
agg_job_id=$(sbatch --parsable --dependency="afterok:${sim_job_id}" scripts/01_aggregate_and_build_distributions.sh)
echo "    Aggregator Job ID: ${agg_job_id}"

echo "=============================================================================="
echo "PIPELINE SUBMITTED SUCCESSFULLY"
echo "=============================================================================="
echo "  1. Simulation Array: ${sim_job_id} (${ARRAY_RANGE})"
echo "  2. Distribution Aggregator: ${agg_job_id} (queued on ${sim_job_id})"
echo ""
echo "Monitor status with:"
echo "  squeue -u \$USER"
echo ""
echo "Logs will be written to:"
echo "  logs/01_large_scale_rate_limits/"
echo "=============================================================================="
