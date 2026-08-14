#!/bin/bash

#SBATCH --job-name=Ope_7400
#SBATCH --partition=cpu6248R,cpueicc,i72c512g
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH -n 3
#SBATCH --cpus-per-task=1

##SBATCH --time=2:00:00
##SBATCH --nodelist=

ulimit -s unlimited
ulimit -l unlimited

. /public/home/huangcl/act_venv.sh

input_dir="/public/home/huangcl/04_gluon_unpolarized_PDF/00_contract/01_submit/01_create_task/../../02_input/L24x72/7400"
log_dir="/public/home/huangcl/04_gluon_unpolarized_PDF/00_contract/01_submit/01_create_task/../../03_log/L24x72/7400"
code_dir="$input_dir/../../../00_code"

exe="$code_dir/Calc_ope_unpol.py"

echo "7400 job starts at $(date)" >"$log_dir/output_7400_rank0.log"
mpirun -n 3 bash -c "python -u \"$exe\" \"$input_dir/input_7400\" >> \"$log_dir/output_7400_rank\${OMPI_COMM_WORLD_RANK}.log\" 2>&1"
echo "7400 job ends at $(date)" >>"$log_dir/output_7400_rank0.log"
