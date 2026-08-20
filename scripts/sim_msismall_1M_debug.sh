#!/bin/bash --login
#SBATCH -p msismall      
#SBATCH --time=11:59:00
#SBATCH --ntasks=4
#SBATCH --mem=250g
#SBATCH --tmp=10g
#SBATCH --mail-type=BEGIN,END,FAIL  
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --output=logs/testing/%x_%A_%a.log
#SBATCH --error=logs/testing/%x_%A_%a.err

cd /users/8/barna314/kn-rates/

if [ ! -d "logs/testing" ]; then
    mkdir -p logs/testing
fi

module load conda
module load hdf5
module load cuda/12.0
module load texlive


eval "$(conda shell.bash hook)"
conda init bash

which conda
which python

echo "Running task number $SLURM_ARRAY_TASK_ID"
conda activate ~/.conda/envs/simsurvey-dev/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/common/software/install/spack/linux-centos7-ivybridge/gcc-7.2.0/cuda-11.8.0-xqzqlf2v77opht3bv4onsqt7uuiomqec/extras/CUPTI/lib64

python ZTF_sim.py --output-base results/testing/Bu2026Fixed/half_day_cadence --n-transients 1000000 --n-runs 1 --n-processes 4 --model Bu2026Fixed --debug-cadence

