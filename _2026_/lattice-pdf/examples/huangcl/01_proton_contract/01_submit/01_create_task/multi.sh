#!/bin/bash
#SBATCH --partition=cpu6248R,cpueicc,i72c512g
#SBATCH --output=1_multi.log
#SBATCH --error=1_multi.log

start_conf=6600
N_conf=8
interval=200
conf_short="L24x72"

if [[ $conf_short =~ L([0-9]+)x([0-9]+) ]]; then
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

base_dir=$(pwd)
submit_dir="$base_dir/.."

for ((i = 0; i < N_conf; i += 1)); do
  conf=$((start_conf + i * interval))

  input_dir=$(realpath -m "$submit_dir/../02_input/$conf_short/$conf")
  if [ ! -d "$input_dir" ]; then
    mkdir -p "$input_dir"
  fi

  log_dir=$(realpath -m "$submit_dir/../03_log/$conf_short/$conf")
  if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
  fi

  result_dir=$(realpath -m "$submit_dir/../04_test_result/$conf_short/$conf")
  if [ ! -d "$result_dir" ]; then
    mkdir -p "$result_dir"
  fi

  sed "s/=CONF=/$conf/g; s/=LATTICE_SIZE=/$conf_short/g; s#=LOG_DIR=#$log_dir#g; s#=INPUT_DIR=#$input_dir#g" "$submit_dir/00_template/submit.sh" >"$input_dir/submit_${conf}.sh"
  chmod +x "$input_dir/submit_${conf}.sh"

  sed "s/=NT=/$Nt/g; s/=NX=/$Nx/g; s/=CONF=/$conf/g; s/=CONF_NAME=/$conf_name/g; s#=RESULT_DIR=#$result_dir#g" "$submit_dir/00_template/input" >"$input_dir/input_${conf}"

  sbatch "$input_dir/submit_${conf}.sh"
done
