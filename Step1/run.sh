#!/bin/bash
mpirun -np 1 python ./GeometryWarp_smooth.py 2>&1 | tee hist.dat
# mpirun -np 4 python opt.py --procs 4 2>&1 | tee hist.dat

