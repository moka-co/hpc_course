#/usr/bin/bash

cd "$(dirname "$0")"
rm -f ./matmatdist.out
mpicc -fopenmp -O3 matmatdist.c -o matmatdist.out 
export OMP_NUM_THREADS=4
mpirun -np 4 ./matmatdist.out 4096 3 4 256