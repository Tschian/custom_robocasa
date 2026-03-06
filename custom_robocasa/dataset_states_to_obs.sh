#!/bin/bash
#SBATCH -A hk-project-p0024023
#SBATCH -p accelerated
#SBATCH -J dataset_process # Cluster Settings
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1 # Define the paths for storing output and error files
#SBATCH --output=/home/hk-project-p0024023/mn4777/depth_vla/slurm_jobs/robocasa/%x_%j.out
#SBATCH --error=/home/hk-project-p0024023/mn4777/depth_vla/slurm_jobs/robocasa/%x_%j.err

source ~/.bashrc
conda activate robocasa

export LD_LIBRARY_PATH=$HOME/miniconda3/lib:$LD_LIBRARY_PATH
export CC=/opt/gcc/11/bin/gcc
export CXX=/opt/gcc/11/bin/g++
export CXXFLAGS="-O2 -march=core-avx2"
export CFLAGS="-O2 -march=core-avx2"

export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun python custom_robocasa/robocasa/scripts/dataset_states_to_obs.py