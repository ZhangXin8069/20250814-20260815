# AGENTS.md — agent/snsc

**历史目录——已被取代。** 曾用于跨 LQCD_Master + lamet-agent 编排完整质子 2pt 流水线的桥接目录；AI agent 阶段已移除，精简 CPU 流水线移到 `../snsc-v20260726/`，现行 GPU 流水线在 `../docker-v20260802/`。

仅有 `runs/` 子目录含一次历史验证运行（`validate_20260725_101924/`，5 阶段：01_lqcd_master → 02_sample_data → 03_core_computation → 04_huangcl_analysis → 05_lamet_agent）。

**不要修改或新增。** 新验证运行用 docker-v* GPU 流水线或 `../../snsc/main.py`。
