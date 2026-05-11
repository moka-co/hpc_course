#/usr/bin/bash

cd "$(dirname "$0")"
rm -f ./matmatdist.out
mpicc -fopenmp -O3 matmatdist.c -o matmatdist.out 
export OMP_NUM_THREADS=8
mpirun --oversubscribe -np 8 ./matmatdist.out 256 4 2 256 2 4