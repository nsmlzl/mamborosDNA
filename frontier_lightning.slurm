#!/bin/bash
#SBATCH -A bif148
#SBATCH -J frontier_lightning_test
#SBATCH -o logs/frontier_lightning_test-%j.o
#SBATCH -e logs/frontier_lightning_test-%j.e
#SBATCH -t 00:03:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2

# Only necessary if submitting like: sbatch --export=NONE ... (recommended)
# Do NOT include this line when submitting without --export=NONE
unset SLURM_EXPORT_ENV

# Load modules
module load PrgEnv-gnu/8.3.3
module load amd-mixed/6.0.0
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0

# Activate your environment
source activate /ccs/proj/bif148/mamboros/mamboros_env

# Needed to bypass MIOpen, Disk I/O Errors
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# Run script
srun -N2 --ntasks-per-node=8 -c7 --gres=gpu:8 python3 -W ignore -u ./frontier_lightning.py
