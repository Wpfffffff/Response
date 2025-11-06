#!/bin/bash
mpirun -np 28 python ./RAE2822Tran_Grassmannian.py 2>&1 | tee hist.dat
# mpirun -np 4 python opt.py --procs 4 2>&1 | tee hist.dat

