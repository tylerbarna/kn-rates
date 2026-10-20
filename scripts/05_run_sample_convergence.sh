#!/bin/bash --login
# ==============================================================================
# SCRIPT 05: Sample Size Convergence Analysis across 10k, 100k, 1M, and 10M
# ==============================================================================
# Purpose:
#   Evaluates statistical convergence and resource scaling across a four-decade
#   grid of synthetic transient injections:
#     - 10k:    10,000 transients   (~1.5 GB RAM,  ~30 sec)
#     - 100k:  100,000 transients   (~3.8 GB RAM, ~2.5 min)
#     - 1M:   1,000,000 transients  (~25.0 GB RAM, ~40 min)
#     - 10M: 10,000,000 transients (~650.0 GB RAM, ~21 hr)
#
# Astrophysical & Statistical Rationale:
#   - Proves the Poisson noise law: sigma(R_95%) / R_95% ≈ 1 / sqrt(N_det)
#   - Demonstrates why 10M transients are essential for rare populations like
#     Bu2026Vary (epsilon ≈ 3.5e-5), where N <= 100k suffers high zero-detection
#     dropout probability P(0) = 70.5% at 10k, rendering rate limits unconstrained.
#
# MSI Agate Empirical Benchmark (Job ID 14237734):
#   - Name:            sim_msilarge_10M_multimodel.sh
#   - Wall-clock time: 20:47:00
#   - Memory Utilized: 649.67 GB (86.62% of 750.00 GB allocation)
#   - Alloc Cores:     10 cores
#   - Partition:       msismall (targeting 755 GB node) or msilarge
# ==============================================================================
#SBATCH -p msismall
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=750g
#SBATCH --tmp=100g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --job-name=kn_sample_convergence
#SBATCH --output=logs/05_sample_convergence/%x_%j.log
#SBATCH --error=logs/05_sample_convergence/%x_%j.err

set -euo pipefail

# Determine project root dynamically
PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs/05_sample_convergence results/05_sample_convergence

# Activate Conda environment
if command -v module &>/dev/null; then
    module load conda 2>/dev/null || true
    module load hdf5 2>/dev/null || true
    module load cuda/12.0 2>/dev/null || true
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
else
    echo "Warning: No standard conda environment matched; using currently active environment: $(which python)"
fi

echo "=============================================================================="
echo "Job ID:            ${SLURM_JOB_ID:-LOCAL}"
echo "Hostname:          $(hostname -f)"
echo "Python Executable: $(which python)"
echo "Date/Time:         $(date)"
echo "Target Partition:  ${SLURM_JOB_PARTITION:-msismall}"
echo "Requested Mem:     750 GB (accommodating ~650 GB 10M peak memory footprint)"
echo "=============================================================================="

# Run sample size convergence evaluation across full 4-decade grid
python analysis_suites/05_sample_convergence/run_sample_convergence.py \
    --output-dir results/05_sample_convergence \
    --seed 20261020 \
    --n-realizations 10

echo "=============================================================================="
echo "Suite 05 Sample Size Convergence completed successfully at $(date)."
echo "Results written to: results/05_sample_convergence/"
echo "=============================================================================="
