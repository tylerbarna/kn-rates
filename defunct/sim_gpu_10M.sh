#!/bin/bash --login
#SBATCH -p msigpu       
#SBATCH --time=23:59:00
#SBATCH --ntasks=10
#SBATCH --mem=1000g
#SBATCH --tmp=10g
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL  
#SBATCH --mail-user=barna314@umn.edu
#SBATCH --output=logs/output_%x_%j.log
#SBATCH --error=logs/error_%x_%j.err

cd /users/8/barna314/kn-rates/

module load conda
module load hdf5
module load cuda/12.0
module load texlive

eval "$(conda shell.bash hook)"
conda init bash

conda activate /users/8/barna314/anaconda3/envs/simsurvey
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/common/software/install/spack/linux-centos7-ivybridge/gcc-7.2.0/cuda-11.8.0-xqzqlf2v77opht3bv4onsqt7uuiomqec/extras/CUPTI/lib64

python ZTF_sim.py --output-base results/ztf_10M_msigpu_result --n-transients 10000000 --n-runs 10 --n-processes 2
# > logs/output_${SLURM_JOB_ID}_$i.log 2> logs/error_${SLURM_JOB_ID}_$i.err & done