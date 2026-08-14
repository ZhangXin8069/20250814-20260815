# AGENTS.md — agent/docker-v20260727

**历史版本——已被 `../docker-v20260802/` 取代。** 断连胶子 PDF 流水线首次 GPU 移植。

| 特性 | 值 |
|---|---|
| 计算引擎 | **GPU 优先**（CuPy/CUDA） |
| 精度 | 单精度（complex64） |
| 特征向量 | 共享 `.npy`，逐时间片流式传 GPU（防 OOM） |
| 关键新增 | `check_env.py`、`regenerate_plots.py` |

首次将 VVV 重子块、Wick 收缩、Clover F_{μν}（独立 CuPy 实现）、Wilson 线 + OPE 收缩全部跑在 GPU。**不要用于新工作。**
