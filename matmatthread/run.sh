#/usr/bin/bash

cd "$(dirname "$0")"
gcc -fopenmp -O3 matmatthread.c -o matmatthread.out && ./matmatthread.out 2048 3 4 256