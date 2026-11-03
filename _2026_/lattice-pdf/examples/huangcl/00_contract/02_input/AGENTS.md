# AGENTS.md — examples/huangcl/00_contract/02_input

流水线步骤 0（OPE）的**生成任务输入**。`01_submit/01_create_task/multi.sh` 在这里为每个组态写一个子目录，各含替换后的参数文件与 Slurm 脚本。**勿手改数字组态子目录**——用 `multi.sh` 重新生成。

```
02_input/
└── L24x72/
    └── <conf_id>/        # 如 6200, 6400, ...
        ├── input_<conf_id>        # 替换后的参数文件（Calc_ope_unpol.py stdin 格式）
        └── submit_<conf_id>.sh    # 替换后的 Slurm 脚本
```

`multi.sh` 引用的 `03_log/`、`04_test_result/` 在集群运行时创建，不属于本树。所有路径指向集群存储，仅集群可解析。
