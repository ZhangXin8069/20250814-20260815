#!/bin/bash

#SBATCH --job-name=matching
#SBATCH --partition=cpu6248R,cpueicc,i72c512g
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=8
##SBATCH --exclusive 




exe=/public/group/imp/zengch/LQCD/renorma/matching_MPI.py
mpirun -np 8 python -u $exe  > log/output.log 2>&1