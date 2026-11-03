#!/bin/bash

#SBATCH --job-name=Npi_I15_2pt
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=nv100-ins,nv100-sug
##SBATCH --time="2-00:00:00"

#SBATCH --error=Npi_I15_2pt_error
#SBATCH --output=Npi_I15_2pt_output

# 禁用不需要的传输方式
export UCX_TLS=rc,shm  # 仅使用InfiniBand和共享内存

source /public/home/sush/envs/sm80

conf_start=1000
gap=320
gap_2=40

for i in {5..7..1}
do
    conf=$(( ${conf_start} + ${gap} * ${SLURM_ARRAY_TASK_ID} + ${i} * ${gap_2} ))

    run_dir=/public/home/sush/distillation/0v2b

    exe=${run_dir}/contraction.Np.I1.5.2pt.cupy.py
    dir_file=${run_dir}/file/contraction.Np.I1.5.2pt.cupy.${conf}

    output_file=${run_dir}/output/contraction.Np.I1.5.2pt.cupy.${conf}.log

    if [ ! -e "${run_dir}/output/" ];then
        mkdir "${run_dir}/output/"
    fi

    if [ ! -e "${run_dir}/file" ];then
        mkdir "${run_dir}/file"
    fi

    echo "${conf} job starts at $(date)" > "${output_file}"
    mkdir ${dir_file}

    if [ -d ${dir_file} ]; then
        # GPU 可用性检查
        python -c "import cupy; assert cupy.cuda.runtime.getDeviceCount() > 0, 'No GPU'" || {
            echo "GPU 不可用，放弃本任务" >> "${output_file}" 2>&1
            exit 1
        }

        mpirun -n 1 --bind-to core --map-by core python -u "${exe}" "${conf}" >> "${output_file}" 2>&1
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