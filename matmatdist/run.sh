#/usr/bin/bash

cd "$(dirname "$0")"
rm -f ./matmatdist.out
mpicc -fopenmp -O3 matmatdist.c -o matmatdist.out 
export OMP_NUM_THREADS=12
mpirun --oversubscribe -np 1 ./matmatdist.out 4096 1 1 256 1 1