#!/bin/bash --login
#SBATCH -p msismall
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g
#SBATCH --job-name=kn_crit_v2_deprecated
#SBATCH --output=logs/detection_criteria_sweep_v2/%x_%j.log
#SBATCH --error=logs/detection_criteria_sweep_v2/%x_%j.err

set -euo pipefail

# FairShare Protection Guard:
echo "=============================================================================="
echo "[DEPRECATED] run_detection_criteria_sweep.sh has been disabled."
echo "Submitting 2,592 monolithic simulation tasks severely degrades MSI FairShare scores."
echo "Please use the high-performance 'Simulate-Once, Filter-Many' pipeline instead (3 tasks total):"
echo "  bash scripts/submit_fast_criteria_sweep.sh --fiducial"
echo "=============================================================================="
exit 1

module load conda
module load hdf5
module load cuda/12.0
module load texlive

eval "$(conda shell.bash hook)"
conda activate ~/.conda/envs/simsurvey_no_gpu/

python detection_criteria_sweep.py \
  --output-root results/detection_criteria_sweep_v2 \
  --n-transients 100000 \
  --n-runs 1 \
  --n-processes 1 \
  --seed 20261008 \
  --array-index "$SLURM_ARRAY_TASK_ID"
