#!/bin/bash
#SBATCH --job-name=PlaAll
#SBATCH --partition=i72c512g,cpueicc,cpu6248R
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/step_%A_%a.out
#SBATCH --error=logs/step_%A_%a.err
##SBATCH --array=0-3           # 改提交文件的数量
#SBATCH --array=0-1        # 改提交文件的数量

input_dir=/public/group/imp/zengch/LQCD/renorma/submit/submit_fit_ratio/input_for_ratio
exe=/public/group/imp/zengch/LQCD/renorma/submit/submit_fit_ratio/fit_ratio_for_submit.py

mkdir -p logs
mapfile -t INPUTS < <(ls -v "$input_dir"/input_file_*)
N=${#INPUTS[@]}

this_input=${INPUTS[$SLURM_ARRAY_TASK_ID]}      # 关键：用下标取
base_name=$(basename "$this_input" .txt)        # 想去掉后缀可再改

python -u "$exe" "$this_input" > "logs/output_${base_name}.log" 2>&1
