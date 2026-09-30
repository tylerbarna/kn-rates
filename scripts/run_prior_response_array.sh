#!/bin/bash --login
#SBATCH -p msismall
#SBATCH --time=71:59:00
#SBATCH --ntasks=4
#SBATCH --mem=250g
#SBATCH --tmp=100g
#SBATCH --array=0-2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --output=logs/prior_response/%x_%A_%a.log
#SBATCH --error=logs/prior_response/%x_%A_%a.err

set -euo pipefail
cd /users/8/barna314/kn-rates/

module load conda
module load hdf5
module load cuda/12.0
module load texlive

eval "$(conda shell.bash hook)"
conda init bash
conda activate ~/.conda/envs/simsurvey_no_gpu/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/common/software/install/spack/linux-centos7-ivybridge/gcc-7.2.0/cuda-11.8.0-xqzqlf2v77opht3bv4onsqt7uuiomqec/extras/CUPTI/lib64

declare -a MODELS=("Bu2026Fixed" "Bu2026Vary" "Metzger")
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
BASE_SEED="${SEED:-20260930}"
MODEL_SEED=$((BASE_SEED + SLURM_ARRAY_TASK_ID))
OUTPUT_DIR="${OUTPUT_DIR:-results/prior_response/${MODEL}}"

mkdir -p "${OUTPUT_DIR}"
echo "Running response task ${SLURM_ARRAY_TASK_ID}: ${MODEL}"

python prior_sensitivity_analysis.py \
  --model "${MODEL}" \
  --n-samples "${N_SAMPLES:-10000}" \
  --n-simulations "${N_SIMULATIONS:-100}" \
  --n-transients "${N_TRANSIENTS:-100000}" \
  --run-simulations \
  --seed "${MODEL_SEED}" \
  --rate "${RATE:-1000.0}" \
  --output-dir "${OUTPUT_DIR}"
