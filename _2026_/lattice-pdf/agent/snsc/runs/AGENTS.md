# AGENTS.md — agent/snsc/runs

联合验证流水线（`agent/snsc/run_pipeline.py`，编排 LQCD_Master + lamet-agent 端到端）的输出。每个子目录为一次带时间戳的 5 阶段验证运行。**视为生成输出，非源码，勿手改。**

```
runs/validate_YYYYMMDD_HHMMSS/
├── run_pipeline.py / run_validate.sh / run_config.json / final_report.md / run.log
├── 01_lqcd_master/       # 阶段1：LQCD_Master 计划
├── 02_sample_data/       # 阶段2：样本 2pt .npy + OPE .npz
├── 03_core_computation/  # 阶段3：2pt 蒸馏、OPE 算符、γ 矩阵
├── 04_huangcl_analysis/  # 阶段4：huangcl 式比值 R(z) + meff
└── 05_lamet_agent/       # 阶段5：lamet-agent manifest + HDF5 桥接
```

5 阶段流水线复用 `snsc/main.py` 做蒸馏，2pt 经桥接转 lamet-agent HDF5 格式，误差分析用 jackknife（`resample_mode: "jk"`）。
