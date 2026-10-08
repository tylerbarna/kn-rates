#!/bin/bash --login
#SBATCH -p msismall
#SBATCH --time=11:59:00
#SBATCH --ntasks=4
#SBATCH --mem=100g
#SBATCH --tmp=10g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --output=logs/rate_limit_benchmark/%x_%A_%a.log
#SBATCH --error=logs/rate_limit_benchmark/%x_%A_%a.err

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


python rate_limit_benchmark.py
