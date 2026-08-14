# AGENTS.md — examples/huangcl/00_contract/01_submit/01_create_task

流水线步骤 0（OPE）任务生成脚本：把 `00_template/` 占位文件替换进逐组态输入目录并提交 Slurm 作业。

## 文件

| 文件 | 用途 |
|---|---|
| `multi.sh` | 批量生成器。读 `start_conf`、`N_conf`、`interval`、`conf_short`（默认 L24x72，组态 8400、8600）；创建 `../02_input/<lattice>/<conf>/`（及 `../03_log/`、`../04_test_result/`）；sed 替换 `submit.sh` 与 `input` 模板；逐个 sbatch |
| `test.sh` | 单组态生成器（手动 conf + lattice_size）。引用**本树中不存在**的 `slurm_text.sh`/`input_text.sh`——遗留变体 |

## 使用

```bash
bash multi.sh    # 生成 ../02_input 下逐组态任务并提交
```

模板在 `../00_template/`；规范场路径指向集群存储（`/public/group/lqcd/donghx/Hpysmear_*`），仅集群可解析。
