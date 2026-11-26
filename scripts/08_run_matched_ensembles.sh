#!/bin/bash --login
# ==============================================================================
# SCRIPT 08: Matched-Exposure ZTF 8-Year Kilonova Ensemble Simulation (WP-5)
# ==============================================================================
# Purpose:
#   Executes Paper I's 5 matched-exposure kilonova simulation tiers against the
#   exact monthly ZTF HDF5 exposure logs exported by real-ztf-search
#   (knsearch.exposure_export), enforcing pure fade-only selection parity
#   (min_rise_rate = 99.0) with zero edits to frozen survey-sim:
#     - Tasks 000-199: Bu2026_Polar                 (200 x 500k = 10^8)
#     - Tasks 200-399: Bu2026_Isotropic             (200 x 500k = 10^8)
#     - Tasks 400-599: AT2017gfo_Posterior_Polar    (200 posterior draws x 500k = 10^8)
#     - Tasks 600-799: AT2017gfo_Posterior_Isotropic(200 posterior draws x 500k = 10^8)
#     - Tasks 800-999: BNS_NR_Population            (200 stratified NR draws x 500k = 10^8)
#
# FairShare-Rightsized MSI Agate Resources (per 500k-injection task):
#   - 4 CPUs / 32 GB RAM (~8 GB/core, matching msismall ~3.9-8 GB/core ratio)
#   - Walltime: 02:00:00 (~45-60 min expected runtime with 2x cushion)
#
# Usage:
#   EXPOSURE_DIR=/path/to/ztf_boom_searched sbatch scripts/08_run_matched_ensembles.sh
# ==============================================================================
#SBATCH -p msismall
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --tmp=16g
#SBATCH --array=0-999%100
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --job-name=kn_s08_matched
#SBATCH --output=logs/08_matched_ensembles/%x_%A_%a.log
#SBATCH --error=logs/08_matched_ensembles/%x_%A_%a.err

set -euo pipefail

export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/08_matched_ensembles results/08_matched_ensembles

if command -v module &>/dev/null; then
    module load conda 2>/dev/null || true
    module load hdf5 2>/dev/null || true
fi

eval "$(conda shell.bash hook 2>/dev/null || true)"
if conda env list | grep -q "kn_rates_dev"; then
    conda activate kn_rates_dev
elif conda env list | grep -q "simsurvey_no_gpu"; then
    conda activate simsurvey_no_gpu
elif [ -d "$HOME/.conda/envs/simsurvey_no_gpu" ]; then
    conda activate "$HOME/.conda/envs/simsurvey_no_gpu"
elif [ -d "${PROJECT_DIR}/.conda" ]; then
    conda activate "${PROJECT_DIR}/.conda"
fi

TIERS=(
    "Bu2026_Polar"
    "Bu2026_Isotropic"
    "AT2017gfo_Posterior_Polar"
    "AT2017gfo_Posterior_Isotropic"
    "BNS_NR_Population"
)

N_TASKS_PER_TIER="${N_TASKS_PER_TIER:-200}"
RAW_ID="${SLURM_ARRAY_TASK_ID:-0}"
TIER_IDX=$((RAW_ID / N_TASKS_PER_TIER))
DRAW_IDX=$((RAW_ID % N_TASKS_PER_TIER))
TIER="${TIERS[$TIER_IDX]}"

N_TRANSIENTS="${N_TRANSIENTS:-500000}"
ENSEMBLE_SEED="${ENSEMBLE_SEED:-20261125}"
OUTPUT_DIR="${OUTPUT_DIR:-results/08_matched_ensembles}"
EXPOSURE_DIR="${EXPOSURE_DIR:-${PROJECT_DIR}/../real-ztf-search/results/ztf_boom_searched}"

EXTRA_ARGS=()
if [ -d "${EXPOSURE_DIR}" ]; then
    EXTRA_ARGS+=(--exposure-dir "${EXPOSURE_DIR}")
else
    echo "[ERROR] Searched exposure directory ${EXPOSURE_DIR} not found. Export monthly HDF5s via knsearch.exposure_export first." >&2
    exit 2
fi

if [ -n "${POSTERIOR_CHAIN:-}" ] && [ -f "${POSTERIOR_CHAIN}" ]; then
    EXTRA_ARGS+=(--posterior-chain "${POSTERIOR_CHAIN}")
fi

echo "=============================================================================="
echo "Suite 08 Matched Ensemble | Tier=${TIER} | Draw=${DRAW_IDX}/${N_TASKS_PER_TIER}"
echo "Transients=${N_TRANSIENTS} | ExposureDir=${EXPOSURE_DIR}"
echo "=============================================================================="

python analysis_suites/08_matched_ensembles/run_matched_ensemble.py \
    --tier "${TIER}" \
    --task-index "${DRAW_IDX}" \
    --n-tasks "${N_TASKS_PER_TIER}" \
    --n-transients "${N_TRANSIENTS}" \
    --ensemble-seed "${ENSEMBLE_SEED}" \
    --output-dir "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}"
