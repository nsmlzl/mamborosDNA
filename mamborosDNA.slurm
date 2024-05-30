#!/bin/bash
#SBATCH -A bif148
#SBATCH -J frontier_lightning_test
#SBATCH -o logs/frontier_lightning_test-%j.o
#SBATCH -e logs/frontier_lightning_test-%j.e
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 4

# Only necessary if submitting like: sbatch --export=NONE ... (recommended)
# Do NOT include this line when submitting without --export=NONE
unset SLURM_EXPORT_ENV

# Load modules
module --ignore_cache load PrgEnv-gnu/8.3.3
module --ignore_cache load amd-mixed/6.0.0
module --ignore_cache load craype-accel-amd-gfx90a
module --ignore_cache load miniforge3/23.11.0

export ROCM_PATH="/ccs/proj/bif148/mamboros/rocm-6.0.0"
export TRITON_HIP_LLD_PATH="/ccs/proj/bif148/mamboros/rocm-6.0.0/llvm/bin/ld.lld"

# Activate your environment
source activate /ccs/proj/bif148/mamboros/mamboros_env

# Needed to bypass MIOpen, Disk I/O Errors
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# Prevent I/O Errors
export TRITON_CACHE_DIR="/tmp/my-triton-cache"
rm -rf ${TRITON_CACHE_DIR}
mkdir -p ${TRITON_CACHE_DIR}

# Run script
srun -N4 --ntasks-per-node=8 -c7 --gres=gpu:8 python3 -W ignore -u ./mamborosDNA.py train --dataset yeast
