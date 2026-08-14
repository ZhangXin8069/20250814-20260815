# AGENTS.md — agent/docker-v20260726

**历史版本——已被 `../docker-v20260802/` 取代。** 断连胶子 PDF GPU 验证流水线的原始 CPU 基线。

| 特性 | 值 |
|---|---|
| 计算引擎 | **仅 CPU**（NumPy） |
| 精度 | 双精度 |
| 特征向量 | 共享 `.npy`（`cfg_48000.eigenvector.npy`） |
| 有效质量 | 朴素 arccosh |
| OPE 算法 | 基础 F（无对偶 F̃） |

后续版本线：v27 首次 GPU 移植 → v28 CPU meff 修复 → v29 GPU 修正 → v30 重大 bug 修复（对偶 F̃ 等）→ v31 双精度 GPU → **v20260802** 当前标准版。

**不要用于新工作。** 运行：`python run_pipeline.py`（纯 CPU）。
