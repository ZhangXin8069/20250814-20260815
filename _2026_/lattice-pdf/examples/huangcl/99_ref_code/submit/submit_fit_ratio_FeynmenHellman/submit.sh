#!/bin/bash
#SBATCH --job-name=fit_ratio
#SBATCH --partition=cpu6248R,cpueicc,i72c512g,cpu-short
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
# 总任务数 24
#SBATCH --ntasks=24

# 确保日志目录存在
mkdir -p log

exe=/public/group/imp/zengch/LQCD/renorma/fit_ratio_FeynmenHellman_new_MPI.py
# 不再统一重定向日志，日志由Python内部每个任务单独输出
mpirun -np $SLURM_NTASKS python -u $exe