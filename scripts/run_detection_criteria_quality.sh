#!/bin/bash --login
#SBATCH -p msismall
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16g
#SBATCH --tmp=10g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --output=logs/detection_criteria_quality/%x_%A.log
#SBATCH --error=logs/detection_criteria_quality/%x_%A.err

set -euo pipefail
cd /users/8/barna314/kn-rates/

module load conda
module load hdf5
module load cuda/12.0
module load texlive

eval "$(conda shell.bash hook)"
conda activate ~/.conda/envs/simsurvey_no_gpu/

python lightcurve_quality_analysis.py \
  --glob 'results/detection_criteria_sweep/**/*.json' \
  --output results/detection_criteria_sweep/lightcurve_quality.csv
