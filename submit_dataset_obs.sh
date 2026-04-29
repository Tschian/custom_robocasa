#!/bin/bash
#SBATCH -A hk-project-p0024023
#SBATCH -p cpuonly
#SBATCH -t 18:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --mem=239400mb
#SBATCH -J dataset_obs
#SBATCH -o /hkfs/work/workspace/scratch/mn4777-qian_space/logs/dataset_obs_%j.out
#SBATCH -e /hkfs/work/workspace/scratch/mn4777-qian_space/logs/dataset_obs_%j.err

# Software rendering — no display or GPU needed
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
# Prevent MuJoCo/OpenBLAS from spawning extra threads per worker
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# TODO: adjust the path below to your conda installation on HoreKa
source ~/.bashrc
# TODO: replace <your_env> with your actual conda environment name
conda activate pil

WORKSPACE=/hkfs/work/workspace/scratch/mn4777-qian_space

mkdir -p $WORKSPACE/logs

cd $WORKSPACE/custom_robocasa

# 76 cores: 60 workers + 1 writer + 1 main + headroom for MuJoCo internals
python custom_robocasa/robocasa/scripts/dataset_states_to_obs.py \
    --num_procs 60 \
    --use_local_sphere \
    --num_virtual_cameras 3 \
    --point_ray_dtype float16
