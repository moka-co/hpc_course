#!/usr/bin/env python3
import subprocess
import os

tests = [
    # N=2048, (PROW, PCOL), [(NTROW, NTCOL), ...]
    (2048, (1, 1), [(1, 1), (2, 1), (2, 2), (4, 2)]),
    (2048, (1, 2), [(1, 1), (2, 1), (2, 2), (4, 2)]),
    (2048, (2, 2), [(1, 1), (2, 1), (2, 2), (4, 2)]),
    (2048, (2, 4), [(1, 1), (2, 1), (2, 2), (4, 2)]),
    (4096, (1, 1), [(1, 1), (2, 1), (2, 2), (4, 2)]),
    (4096, (1, 2), [(1, 1), (2, 1), (2, 2), (4, 2)]),
    (4096, (2, 2), [(1, 1), (2, 1), (2, 2), (4, 2)]),
    (4096, (2, 4), [(1, 1), (2, 1), (2, 2), (4, 2)]),
    (6144, (2, 2), [(1, 1), (2, 1), (2, 2), (4, 2)]),
    (6144, (2, 4), [(1, 1), (2, 1), (2, 2), (4, 2)]),
]

os.chdir(os.path.dirname(os.path.abspath(__file__)))
subprocess.run("rm -f ./matmatdist.out", shell=True)
subprocess.run("mpicc -fopenmp -O3 matmatdist.c -o matmatdist.out", shell=True)
os.environ['OMP_NUM_THREADS'] = '12'

for N, (PROW, PCOL), thread_configs in tests:
    for NTROW, NTCOL in thread_configs:
        np = PROW * PCOL
        print(f"\nRunning: N={N}, Grid=({PROW},{PCOL}), Threads=({NTROW},{NTCOL}), np={np}")
        cmd = f"mpirun --oversubscribe -np {np} ./matmatdist.out {N} {NTROW} {NTCOL} 256 {PROW} {PCOL}"
        subprocess.run(cmd, shell=True)