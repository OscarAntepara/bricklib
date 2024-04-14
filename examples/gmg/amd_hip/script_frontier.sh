#!/bin/bash
#SBATCH -A XXXXXX
#SBATCH -J test
#SBATCH -o %x-%j.out
#SBATCH -t 00:05:00
#SBATCH -N 8
#SBATCH -q debug

module load craype-accel-amd-gfx90a
module load rocm

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_OFI_NIC_VERBOSE=2
export PE_MPICH_GTL_DIR_amd_gfx90a="-L${CRAY_MPICH_ROOTDIR}/gtl/lib"
export PE_MPICH_GTL_LIBS_amd_gfx90a="-lmpi_gtl_hsa"
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RX_MATCH_MODE=hardware


export HIP_VISIBLE_DEVICES=7
srun \
--ntasks-per-node 1 \
--gpus-per-node 1 \
--gpu-bind=closest \
-N 8 \
./examples/gmg/amd_hip/hip -s 512,512,512 -I 10 -l 6 -n 20 > output.txt
