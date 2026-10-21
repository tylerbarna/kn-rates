#!/bin/bash --login
# ==============================================================================
# SCRIPT: Monolithic Sequential Execution Wrapper (Up to 96 Hours)
# ==============================================================================
# Purpose:
#   Runs all 4 simulation suites sequentially within a SINGLE Slurm job allocation
#   on msismall. Useful if you prefer not to submit thousands of array tasks,
#   or if you want a completely self-contained batch run that manages its own
#   dependencies internally.
#
# Slurm Resource Configuration:
#   - Partition: msismall
#   - Walltime:  96:00:00 (MSI maximum allocation window)
#   - Cores:     16 CPUs
#   - Memory:    750 GB (8 x 93750 MB/core)
# ==============================================================================
#SBATCH -p msismall
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=750g
#SBATCH --tmp=50g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --job-name=kn_all_suites_96hr
#SBATCH --output=logs/monolithic_%j.log
#SBATCH --error=logs/monolithic_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs \
         logs/01_large_scale_rate_limits \
         logs/02_granular_criteria_sweep \
         logs/03_prior_vs_at2017gfo \
         logs/04_historical_andreoni_comparison \
         logs/05_sample_convergence \
         results/01_large_scale_rate_limits \
         results/02_granular_criteria_sweep \
         results/03_prior_vs_at2017gfo \
         results/04_historical_andreoni_comparison \
         results/05_sample_convergence

# Activate Conda environment
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

echo "=============================================================================="
echo "STARTING MONOLITHIC SEQUENTIAL EXECUTION (SLURM JOB ${SLURM_JOB_ID:-LOCAL})"
echo "Host:     $(hostname -f)"
echo "Python:   $(which python)"
echo "Start:    $(date)"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# STEP 1: Large-Scale Rate Limits Simulation (Metzger, Bu2026Fixed, Bu2026Vary)
# ------------------------------------------------------------------------------
echo -e "\n>>> [1/4] Running Suite 01: Large-Scale 8-Year Rate Limits..."
for MODEL in "Metzger" "Bu2026Fixed" "Bu2026Vary"; do
    echo "  --> Simulating ${MODEL} (10,000,000 transients)..."
    python analysis_suites/01_large_scale_rate_limits/run_large_scale_simulation.py \
        --model "${MODEL}" \
        --n-transients 10000000 \
        --n-runs 1 \
        --n-processes 16 \
        --seed 20261020 \
        --output-dir results/01_large_scale_rate_limits
done

echo "  --> Aggregating Rate Limits..."
python analysis_suites/01_large_scale_rate_limits/aggregate_rate_limits.py \
    --glob 'results/01_large_scale_rate_limits/**/*.json' \
    --output-csv results/01_large_scale_rate_limits/rate_limit_summary.csv

# ------------------------------------------------------------------------------
# STEP 2: Granular Detection Criteria Sweep
# ------------------------------------------------------------------------------
echo -e "\n>>> [2/4] Running Suite 02: Granular Detection Criteria Sweep..."
# Evaluates representative 1D parameter slices or full grid
for PARAM in "snr_threshold" "min_detections" "max_timespan_days" "min_fade_rate"; do
    echo "  --> Sweeping 1D slice: ${PARAM}..."
    python analysis_suites/02_granular_criteria_sweep/run_granular_sweep.py \
        --slice-param "${PARAM}" \
        --model Bu2026Fixed \
        --n-transients 1000000 \
        --output-root results/02_granular_criteria_sweep
done

echo "  --> Auditing Criteria Adherence & Marginal Sensitivities..."
python analysis_suites/02_granular_criteria_sweep/audit_granular_sweep.py \
    --sweep-root results/02_granular_criteria_sweep \
    --output-csv results/02_granular_criteria_sweep/granular_sweep_metrics.csv

# ------------------------------------------------------------------------------
# STEP 3: Broad Swath Priors vs. AT2017gfo Archetype Comparison
# ------------------------------------------------------------------------------
echo -e "\n>>> [3/4] Running Suite 03: Broad Priors vs. AT2017gfo Comparison..."
python analysis_suites/03_prior_vs_at2017gfo/compare_priors_at2017gfo.py \
    --n-transients 1000000 \
    --n-runs 3 \
    --output-dir results/03_prior_vs_at2017gfo

# ------------------------------------------------------------------------------
# STEP 4: Historical Comparison with Andreoni et al. (2020)
# ------------------------------------------------------------------------------
echo -e "\n>>> [4/5] Running Suite 04: Historical Andreoni Benchmark Comparison..."
python analysis_suites/04_historical_andreoni_comparison/compare_with_andreoni2020.py \
    --rate-limit-csv results/01_large_scale_rate_limits/rate_limit_summary.csv \
    --output-dir results/04_historical_andreoni_comparison

# ------------------------------------------------------------------------------
# STEP 5: Sample Size Convergence Analysis across 10k, 100k, 1M, and 10M Grid
# ------------------------------------------------------------------------------
echo -e "\n>>> [5/5] Running Suite 05: Sample Size Convergence (10k, 100k, 1M, 10M)..."
python analysis_suites/05_sample_convergence/run_sample_convergence.py \
    --output-dir results/05_sample_convergence \
    --seed 20261020 \
    --n-realizations 10

echo "=============================================================================="
echo "ALL 5 SUITES COMPLETED SUCCESSFULLY"
echo "Finished at: $(date)"
echo "=============================================================================="
