#!/bin/bash -l

# Script to run GMG test on Perlmutter

#SBATCH -t 00:05:00
#SBATCH -N 8
#SBATCH -J brick
#SBATCH -A mXXXX_g 
#SBATCH -C gpu
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --gpus-per-node=1
#SBATCH -o brick.o%j
#SBATCH -e brick.e%j
#SBATCH --gpu-bind=closest
#SBATCH --exclusive

module load cmake cudatoolkit nvidia

# GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1
export CRAY_ACCEL_TARGET=nvidia80

# pin to closest NIC to GPU
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_OFI_NIC_VERBOSE=2
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0

srun ./examples/gmg/nvidia_cuda/cuda -s 512,512,512 -I 10 -l 6 -n 20 > output.txt
