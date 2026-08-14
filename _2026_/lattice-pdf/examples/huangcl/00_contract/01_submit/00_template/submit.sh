#!/bin/bash

#SBATCH --job-name=Ope_=CONF=
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

input_dir="=INPUT_DIR="
log_dir="=LOG_DIR="
code_dir=$(realpath -m "$input_dir/../../../00_code")

exe="$code_dir/Calc_ope_unpol.py"

echo "=CONF= job starts at $(date)" >"$log_dir/output_=CONF=_rank0.log"
mpirun -n 3 bash -c "python -u \"$exe\" \"$input_dir/input_=CONF=\" >> \"$log_dir/output_=CONF=_rank\${OMPI_COMM_WORLD_RANK}.log\" 2>&1"
echo "=CONF= job ends at $(date)" >>"$log_dir/output_=CONF=_rank0.log"
