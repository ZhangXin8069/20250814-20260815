#!/bin/bash
#SBATCH --partition=cpu6248R,cpueicc,i72c512g
#SBATCH --output=create_one_task.log
#SBATCH --error=create_one_task.log

conf=4050               # e.g. 4050
lattice_size="24x72"  # e.g. "24x72"


if [ "$conf" == 0 ] || [ "$lattice_size" == "###" ]; then
  echo "Error: do not input all required parameters"
  exit 1
fi

if [[ $lattice_size =~ ([0-9]+)x([0-9]+) ]]; then
    Nx=${BASH_REMATCH[1]}
    Nt=${BASH_REMATCH[2]}
fi

case "${Nt}" in
  72)
    conf_name="beta6.20_mu-0.2770_ms-0.2400_L24x72"
  ;;
  96)
    conf_name="beta6.41_mu-0.2295_ms-0.2050_L32x96"
  ;;
  *)
    echo "not this config"
    exit 1
  ;;
esac

current_dir=$(pwd)
work_dir="$current_dir/.."

task_dir="$work_dir/task/$lattice_size/$conf"
if [ ! -d "$task_dir" ]; then
  mkdir -p "$task_dir"
fi

result_dir="$work_dir/result/$lattice_size/$conf"
if [ ! -d "$result_dir" ]; then
  mkdir -p "$result_dir"
fi

sed "s/=CONF=/$conf/g; s/=LATTICE_SIZE=/$lattice_size/g" "slurm_text.sh" >"$task_dir/submit_${conf}.sh"
chmod +x "$task_dir/submit_${conf}.sh"

./input_text.sh "$Nt" "$Nx" "$conf" "$conf_name" "$result_dir" >"$task_dir/input_${conf}"

sbatch "$task_dir/submit_${conf}.sh"