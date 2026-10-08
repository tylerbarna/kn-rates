#!/bin/bash --login
#SBATCH -p msismall
#SBATCH --time=11:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2591%2592
#SBATCH --mem=100g
#SBATCH --tmp=10g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --output=logs/detection_criteria_sweep/%x_%A.log
#SBATCH --error=logs/detection_criteria_sweep/%x_%A.err

set -euo pipefail
cd /users/8/barna314/kn-rates/

module load conda
module load hdf5
module load cuda/12.0
module load texlive

eval "$(conda shell.bash hook)"
conda activate ~/.conda/envs/simsurvey_no_gpu/

python detection_criteria_sweep.py \
  --output-root results/detection_criteria_sweep \
  --n-transients 100000 \
  --n-runs 1 \
  --n-processes 1 \
  --seed 20261008 \
  --array-index "$SLURM_ARRAY_TASK_ID"
