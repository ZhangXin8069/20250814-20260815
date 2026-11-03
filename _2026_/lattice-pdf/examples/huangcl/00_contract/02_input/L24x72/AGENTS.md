# AGENTS.md — examples/huangcl/00_contract/02_input/L24x72

流水线步骤 0（OPE）L24x72 系综的逐组态任务输入。每个子目录（`6200`…`8600`）由 `01_submit/01_create_task/multi.sh` 从 `00_template/` 占位符生成。**勿手改。**

```
L24x72/
└── <conf_id>/
    ├── input_<conf_id>         # stdin 格式参数文件（Calc_ope_unpol.py）
    └── submit_<conf_id>.sh     # Slurm 脚本
```

视为构建产物——用 `multi.sh` 重新生成。路径指向集群存储（`/public/...`），仅集群可解析。
