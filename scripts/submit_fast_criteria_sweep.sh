#!/bin/bash --login
# ==============================================================================
# SUBMITTER: Fast 3-Model Criteria Sweep Pipeline (Simulate-Once, Filter-Many)
# ==============================================================================
# Usage:
#   bash scripts/submit_fast_criteria_sweep.sh [--fiducial | --ensemble] [--samples 20] [--grid-preset ultra_fine]
#
# Examples:
#   # Run fast fiducial benchmark across all 3 models with ultra_fine grid (2,138,460 points, 3 array tasks total):
#   bash scripts/submit_fast_criteria_sweep.sh --fiducial
#
#   # Run 20-sample 1M ensemble across all 3 models (60 array tasks total):
#   bash scripts/submit_fast_criteria_sweep.sh --ensemble --samples 20
#
#   # Run 100-sample 1M ensemble (matching Suite 01 scale, 300 array tasks total):
#   bash scripts/submit_fast_criteria_sweep.sh --ensemble --samples 100
# ==============================================================================
#SBATCH -p msismall
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --job-name=kn_sub_crit_fast
#SBATCH --output=logs/02_fast_criteria_sweep/submit_%j.log
#SBATCH --error=logs/02_fast_criteria_sweep/submit_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/02_fast_criteria_sweep results/02_granular_criteria_sweep/candidates

# Defaults
MODE="fiducial"
N_SAMPLES=20
RANDOMIZE=0
GRID_PRESET="ultra_fine"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fiducial)
            MODE="fiducial"
            shift
            ;;
        --ensemble)
            MODE="ensemble"
            shift
            ;;
        --samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --randomize)
            RANDOMIZE=1
            shift
            ;;
        --grid-preset)
            GRID_PRESET="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--fiducial | --ensemble] [--samples N] [--randomize] [--grid-preset PRESET]"
            exit 1
            ;;
    esac
done

if [ "${MODE}" = "fiducial" ]; then
    ARRAY_RANGE="0-2"
    TOTAL_TASKS=3
    echo "[INFO] Running in FIDUCIAL mode across all 3 models (3 array tasks)."
else
    TOTAL_TASKS=$(( 3 * N_SAMPLES ))
    ARRAY_RANGE="0-$(( TOTAL_TASKS - 1 ))"
    echo "[INFO] Running in ENSEMBLE mode with ${N_SAMPLES} samples per model (${TOTAL_TASKS} array tasks total)."
fi

# Submit candidate simulation array
cand_job_id=$(sbatch --parsable \
    --array="${ARRAY_RANGE}" \
    --export=ALL,MODE="${MODE}",N_SAMPLES="${N_SAMPLES}",RANDOMIZE_PARAMS="${RANDOMIZE}" \
    scripts/02_run_fast_candidates_simulation.sh)

# Chain fast filter job to run after candidate array completes
if [ "${MODE}" = "ensemble" ]; then
    # 3-model parallel filter array (Tasks 0, 1, 2)
    filter_job_id=$(sbatch --parsable \
        --array="0-2" \
        --dependency="afterany:${cand_job_id}" \
        --export=ALL,GRID_PRESET="${GRID_PRESET}" \
        scripts/02_run_fast_filter.sh)

    # Chained instant merge job to unify per-model CSVs into granular_sweep_metrics.csv
    merge_job_id=$(sbatch --parsable \
        --dependency="afterok:${filter_job_id}" \
        -p msismall --time=00:10:00 --cpus-per-task=1 --mem=8g \
        --job-name=kn_crit_merge \
        --output=logs/02_fast_criteria_sweep/merge_%j.log \
        --wrap="
            eval \"\$(conda shell.bash hook 2>/dev/null)\" && conda activate kn_rates_dev || conda activate simsurvey_no_gpu
            python analysis_suites/02_granular_criteria_sweep/fast_criteria_filter.py --merge-only
        ")
else
    filter_job_id=$(sbatch --parsable \
        --dependency="afterany:${cand_job_id}" \
        --export=ALL,GRID_PRESET="${GRID_PRESET}" \
        scripts/02_run_fast_filter.sh)
    merge_job_id="N/A (single task)"
fi

echo "=============================================================================="
echo "FAST CRITERIA SWEEP PIPELINE SUBMITTED"
echo "  Mode:               ${MODE} (Models: Bu2026Fixed, Bu2026Vary, Metzger)"
echo "  Grid Preset:        ${GRID_PRESET} (2,138,460 points)"
echo "  Candidate Array ID: ${cand_job_id} (${TOTAL_TASKS} tasks, array: ${ARRAY_RANGE})"
echo "  Filter Job ID:      ${filter_job_id} (chained afterany:${cand_job_id})"
if [ "${MODE}" = "ensemble" ]; then
    echo "  Merge Job ID:       ${merge_job_id} (chained afterok:${filter_job_id})"
fi
echo "  Logs:               logs/02_fast_criteria_sweep/"
echo "  Results:            results/02_granular_criteria_sweep/"
echo "=============================================================================="
