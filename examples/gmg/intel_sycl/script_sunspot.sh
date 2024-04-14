#!/bin/bash -l
#PBS -N BRICKS_test
#PBS -l select=8
#PBS -l walltime=0:05:00
#PBS -q workq
#PBS -A XXXXXX

module load cmake
module load oneapi/release/2023.12.15.001

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=1 # Number of MPI ranks to spawn per node
NDEPTH=1 # Number of hardware threads per rank (i.e. spacing between MPI ranks)
NTHREADS=1 # Number of software threads per rank to launch (i.e. OMP_NUM_THREADS)

NTOTRANKS=$(( NNODES * NRANKS ))

echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS} THREADS_PER_RANK= ${NTHREADS}"

export OMP_NUM_THREADS=1

export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1

export ZE_AFFINITY_MASK=0.0
mpiexec -n ${NTOTRANKS} -ppn ${NRANKS} -d 1 --cpu-bind=list:0 -envall bricklib/build/examples/gmg/intel_sycl/sycl -s 512,512,512 -I 10 -l 6 -n 20
