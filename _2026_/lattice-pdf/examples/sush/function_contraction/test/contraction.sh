#!/bin/bash

#SBATCH --job-name=Pi_ENV
#SBATCH --array=0
#SBATCH --nodes=1

#SBATCH --error=test_pion_3pt_error
#SBATCH --output=test_pion_3pt_output

##SBATCH --partition=i72c512g,cpueicc,cpu6248R
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

##SBATCH --partition=na800-pcie
##SBATCH --nodes=1
##SBATCH --gpus=1
##SBATCH --cpus-per-task=2
##SBATCH --gres=gpu:1

##SBATCH --begin=now+7hour
##SBATCH --time="2-00:00:00"
#SBATCH --time="30:00"

##SBATCH -w gpu048

##SBATCH -x cpu052
##SBATCH --exclusive

# 禁用不需要的传输方式
export UCX_TLS=rc,shm  # 仅使用InfiniBand和共享内存

source /public/home/sush/envs/sm80

conf_stare=10000
gap=100

conf=$[${conf_stare}+${gap}*${SLURM_ARRAY_TASK_ID}]

run_dir=/public/home/sush/distillation/function_contraction/test

exe=${run_dir}/contraction.py
dir_file=${run_dir}/file/contraction.${conf}

output_file=${run_dir}/output/contraction.${conf}.log

if [ ! -e "${run_dir}/output/" ];then
    mkdir ${run_dir}/output/
fi

if [ ! -e "${run_dir}/file" ];then
    mkdir ${run_dir}/file
fi

echo "${conf} job starts at" `date` > ${output_file}
mkdir ${dir_file}

if [ -d ${dir_file} ]; then
    mpirun -n 1 --bind-to core --map-by core python -u $exe ${conf} >> ${output_file} 2>&1 

    if [ $? -eq 0 ]; then
        echo "Python 程序执行成功" >> ${output_file} 2>&1
        rm -r ${dir_file}
    else
        echo "Python 程序执行失败" >> ${output_file} 2>&1
    fi

else
    echo "calculate complete" >> ${output_file} 2>&1
fi

echo "$conf job ends at" `date` >> ${output_file}
