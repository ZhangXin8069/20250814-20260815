#!/bin/bash

#SBATCH --job-name=pp_2pt
#SBATCH --array=0-39
#SBATCH --nodes=1

#SBATCH --error=pp_2pt_error
#SBATCH --output=pp_2pt_output

#SBATCH --partition=i72c512g,cpueicc,cpu6248R
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

#SBATCH --time="2-00:00:00"

#SBATCH -x cpu057,cpu060

# 禁用不需要的传输方式
export UCX_TLS=rc,shm  # 仅使用InfiniBand和共享内存

source /public/home/sush/envs/sm80

conf_start=1000
gap=400
gap_2=40

for i in {0..9..1}
do
    conf=$(( ${conf_start} + ${gap} * ${SLURM_ARRAY_TASK_ID} + ${i} * ${gap_2} ))

    run_dir=/public/home/sush/distillation/0v2b

    exe=${run_dir}/contraction.pp.2pt.numpy.py
    dir_file=${run_dir}/file/contraction.pp.2pt.numpy.${conf}

    output_file=${run_dir}/output/contraction.pp.2pt.numpy.${conf}.log

    if [ ! -e "${run_dir}/output/" ];then
        mkdir "${run_dir}/output/"
    fi

    if [ ! -e "${run_dir}/file" ];then
        mkdir "${run_dir}/file"
    fi

    echo "${conf} job starts at $(date)" > "${output_file}"
    mkdir ${dir_file}

    if [ -d ${dir_file} ]; then
        mpirun -n 16 --bind-to core --map-by core python -u "${exe}" "${conf}" >> "${output_file}" 2>&1
        ret=$?

        if [ $ret -eq 0 ]; then
            echo "Python 程序执行成功" >> "${output_file}" 2>&1
            rm -r "${dir_file}"
        else
            echo "Python 程序执行失败 (exit=$ret)" >> "${output_file}" 2>&1
            exit $ret
        fi

    else
        echo "calculate complete" >> "${output_file}" 2>&1
    fi

    echo "${conf} job ends at $(date)" >> "${output_file}"
done