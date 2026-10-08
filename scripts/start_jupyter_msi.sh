#!/bin/bash --login

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/users/8/barna314/kn-rates}"
PARTITION="${PARTITION:-msismall}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
CPUS="${CPUS:-4}"
MEMORY="${MEMORY:-16G}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
CONDA_ENV="${CONDA_ENV:-$HOME/.conda/envs/simsurvey_no_gpu}"
MSI_LOGIN_HOST="${MSI_LOGIN_HOST:-$(hostname -f)}"

export PROJECT_DIR JUPYTER_PORT CONDA_ENV MSI_LOGIN_HOST

module load conda
module load hdf5

echo "Requesting an interactive MSI allocation..."
echo "Partition: ${PARTITION}; time: ${TIME_LIMIT}; CPUs: ${CPUS}; memory: ${MEMORY}"

salloc \
    --job-name=kn-rates-jupyter \
    --partition="${PARTITION}" \
    --time="${TIME_LIMIT}" \
    --ntasks=1 \
    --cpus-per-task="${CPUS}" \
    --mem="${MEMORY}" \
    bash -lc '
        set -euo pipefail
        module load conda
        module load hdf5
        eval "$(conda shell.bash hook)"
        conda activate "${CONDA_ENV}"
        cd "${PROJECT_DIR}"

        COMPUTE_NODE=$(hostname -f)
        echo
        echo "Jupyter will run on ${COMPUTE_NODE}:${JUPYTER_PORT}"
        echo "From your local machine, create the tunnel with:"
        echo "  ssh -N -L ${JUPYTER_PORT}:${COMPUTE_NODE}:${JUPYTER_PORT} ${USER}@${MSI_LOGIN_HOST}"
        echo "Then open the URL printed by Jupyter Lab."
        echo

        exec jupyter lab \
            --no-browser \
            --ip=0.0.0.0 \
            --port="${JUPYTER_PORT}"
    '