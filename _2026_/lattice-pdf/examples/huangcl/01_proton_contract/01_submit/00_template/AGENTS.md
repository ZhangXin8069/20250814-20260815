# AGENTS.md — examples/huangcl/01_proton_contract/01_submit/00_template

流水线步骤 1（质子 2pt）占位模板。与步骤 0 模板（`../00_contract/01_submit/00_template/`）**逐字节相同**；`multi.sh` 替换变量生成逐组态输入。

## 文件

| 文件 | 用途 |
|---|---|
| `input` | 参数模板（stdin 键值格式）。占位符：`=NT=`、`=NX=`、`=CONF=`、`=CONF_NAME=`、`=RESULT_DIR=`；固定 `delta_z 24`、`link_dir z`、`conf_file` 规范场路径 |
| `submit.sh` | Slurm 模板。占位符 `=CONF=`、`=INPUT_DIR=`、`=LOG_DIR=`；source `/public/home/huangcl/act_venv.sh`，`mpirun -n 3` + 逐 rank 日志 |

## 注意

`submit.sh` 执行 `Calc_ope_unpol.py`（步骤 0 的 `00_code/`），但本步骤 `00_code/` 只有质子 2pt 脚本——模板是从步骤 0 复制的，对步骤 1 非权威（见父 `../CLAUDE.md`，已归档）。
