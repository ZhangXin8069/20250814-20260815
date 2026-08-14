# AGENTS.md — examples/huangcl/00_contract/01_submit/00_template

流水线步骤 0（OPE）占位模板。`multi.sh` 替换模板变量生成 `../02_input/` 下的逐组态 `input_<conf>` 参数文件与 `submit_<conf>.sh` Slurm 脚本。

## 文件

| 文件 | 用途 |
|---|---|
| `input` | `Calc_ope_unpol.py` 的参数模板（stdin 键值格式）。占位符：`=NT=`、`=NX=`、`=CONF=`、`=CONF_NAME=`、`=RESULT_DIR=`；固定参数 `delta_z 24`、`link_dir z`、`conf_file` 规范场路径 |
| `submit.sh` | Slurm 模板。占位符 `=CONF=`、`=INPUT_DIR=`、`=LOG_DIR=`；source `/public/home/huangcl/act_venv.sh`，`mpirun -n 3` 跑 `Calc_ope_unpol.py`，逐 rank 日志 |

`=NT=`/`=NX=` 由 `multi.sh` 从 `L<Lx>x<Lt>` 解析。
