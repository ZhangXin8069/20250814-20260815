# AGENTS.md — examples/huangcl/01_proton_contract/01_submit/01_create_task

流水线步骤 1（质子 2pt）任务生成脚本。机制与步骤 0 的 `01_create_task/` 相同：替换 `00_template/` 占位符到逐组态输入目录并提交 Slurm 作业。

## 文件

`multi.sh`（批量生成器。默认 L24x72、组态 6600..7600 步 200；创建 `../02_input/<lattice>/<conf>/` 及 `../03_log/`、`../04_test_result/`；sed 替换并 sbatch）、`test.sh`（单组态，引用不存在的 `slurm_text.sh`/`input_text.sh`——遗留变体）。

## 使用

```bash
bash multi.sh
```

`multi.sh` 是步骤 0 的近副本，仅默认组态范围不同（这里 start_conf=6600、N_conf=8；步骤 0 是 8400、2）。模板在 `../00_template/`；规范场路径指向集群存储，仅集群可解析。
