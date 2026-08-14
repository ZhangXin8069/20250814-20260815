#!/bin/bash
#SBATCH --job-name=fit_hR_big_lambda
#SBATCH --partition=cpu6248R,cpueicc,i72c512g
#SBATCH --output=logs/out_%a.log
#SBATCH --error=logs/err_%a.log
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --exclusive

exe=/public/group/imp/zengch/LQCD/renorma/fit_hR_big_lambda.py

# 定义输入文件数组
input_files=(
    "input_C24P29.json"
    "input_C32P23.json"
    "input_C32P29.json"
    "input_C48P14.json"
    "input_E32P29.json"
    "input_F32P30.json"
    "input_G36P29.json"
    "input_H48P32.json"
)

# 根据数组索引选择输入文件
input_file="${input_files[$SLURM_ARRAY_TASK_ID]}"


# 运行对应的输入文件，将所有输出重定向到日志文件
#echo "Processing $input_file with SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" > "logs/${input_file}.log" 2>&1
python "$exe" "$input_file" > "logs/${input_file}.log" 2>&1