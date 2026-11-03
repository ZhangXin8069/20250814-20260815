#!/bin/bash

# gpu partition: nv100-ins,nv100-sug,dgx2,na100-ins,na100-sug,gpu-debug,na100-40g,na800-sug,na800-pcie,h20-nettr
# cpu partition: cpu6248R,cpueicc,i72c512g
# if using gpu partition, you need to set how many gpu to be use in a task. Otherwise, commenting it.

#SBATCH --job-name=ratio
#SBATCH --partition=cpu6248R,cpueicc,i72c512g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# 限制单线程，匹配 --cpus-per-task=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exe="code.py"
log=".log"

. /public/home/huangcl/act_venv.sh

current_dir=$(pwd)
result_dir="$current_dir/result"
if [ ! -d "$result_dir" ]; then
    mkdir -p "$result_dir"
fi

echo "job starts at $(date)" >$log
python -u "$exe" >>$log 2>&1
echo "job ends at $(date)" >>$log
