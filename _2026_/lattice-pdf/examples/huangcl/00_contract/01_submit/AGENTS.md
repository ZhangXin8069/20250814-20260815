# AGENTS.md — examples/huangcl/00_contract/01_submit

流水线步骤 0（OPE + 质子 2pt）的 Slurm 提交脚手架。含占位模板（`00_template/`）与任务生成脚本（`01_create_task/`），后者替换模板变量并把逐组态输入目录写到 `../02_input/`。

```
01_submit/
├── 00_template/       # 占位 input + submit.sh 模板
└── 01_create_task/    # multi.sh（批量任务生成）+ test.sh（单组态）
```

`00_template/input` 占位符 `=NT=`、`=NX=`、`=CONF=`、`=CONF_NAME=`、`=RESULT_DIR=`；`00_template/submit.sh` 占位符 `=CONF=`、`=INPUT_DIR=`、`=LOG_DIR=`（`mpirun -n 3` 跑 `Calc_ope_unpol.py`）。
