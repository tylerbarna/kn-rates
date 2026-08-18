#!/bin/bash --login
#SBATCH -p msismall      
#SBATCH --time=1:59:00
#SBATCH --ntasks=2
#SBATCH --mem=50g
#SBATCH --tmp=10g
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

conda activate ~/.conda/envs/simsurvey_no_gpu/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/common/software/install/spack/linux-centos7-ivybridge/gcc-7.2.0/cuda-11.8.0-xqzqlf2v77opht3bv4onsqt7uuiomqec/extras/CUPTI/lib64

python ZTF_sim.py --output-base results/test/Metzger/ztf_1M_msismall_result --n-transients 100000 --n-runs 1 --n-processes 1 --model Metzger

python ZTF_sim.py --output-base results/test/Bu2026Fixed/ztf_1M_msismall_result --n-transients 100000 --n-runs 1 --n-processes 1 --model Bu2026Fixed 

python ZTF_sim.py --output-base results/test/Bu2026Vary/ztf_1M_msismall_result --n-transients 100000 --n-runs 1 --n-processes 1 --model Bu2026Vary