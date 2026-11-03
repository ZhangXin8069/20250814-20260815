# AGENTS.md — examples/huangcl/01_proton_contract/01_submit

流水线步骤 1（质子 2pt）的 Slurm 提交脚手架。镜像步骤 0（`../00_contract/01_submit/`）：占位模板 + 写逐组态输入目录的任务生成脚本。

## 文件

| 文件 / 目录 | 用途 |
|---|---|
| `00_template/input` | 参数模板（`=NT=`、`=NX=`、`=CONF=`、`=CONF_NAME=`、`=RESULT_DIR=`）——与步骤 0 相同 |
| `00_template/submit.sh` | Slurm 模板——与步骤 0 相同（仍调用 `Calc_ope_unpol.py`，见注意） |
| `01_create_task/multi.sh` | 批量生成器（默认 start_conf=6600、N_conf=8、L24x72） |
| `01_create_task/test.sh` | 单组态生成（引用缺失的 `slurm_text.sh`/`input_text.sh`） |

## 注意

`submit.sh` 模板执行的是步骤 0 `00_code/` 的 `Calc_ope_unpol.py`，而本步骤 `00_code/` 只有质子 2pt 脚本（2pt 脚本从 `sys.argv[1]` 读 conf_id，与模板的 stdin `input_<conf>` 调用不匹配）——模板视为从步骤 0 复制，而非本步骤的权威。
