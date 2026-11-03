---
name: docker-v20260726
description: 运行 docker-v20260726 历史 CPU 基线管线 — 原始 NumPy 版本 (历史, 已被 v20260802+ 取代)
---

# docker-v20260726 — Historical CPU Baseline Pipeline

在 /root/lattice-pdf/agent/docker-v20260726 运行 disconnected 胶子 PDF 验证管线（CPU NumPy）。

**⚠️ 历史版本** — 系列中最原始的 CPU 基线（2026-07-26）；已被 `../docker-v20260802/` 取代。

## 本版本特点

| 特性 | 值 |
|------|-----|
| 计算引擎 | **仅 CPU** (NumPy, 无 CuPy) |
| 精度 | 双精度 (NumPy 原生) |
| 特征向量 | 共享 `.npy` (`cfg_48000.eigenvector.npy`) |
| 数据路径 | `sunpeng/` 集群路径 |
| 有效质量 | naive arccosh |
| Eigvec smearing | 隐式开启 |
| OPE 算法 | 基本 F (无对偶 F̃) |

## 运行

```bash
cd /root/lattice-pdf/agent/docker-v20260726
python run_pipeline.py    # CPU only
```

**不要用于新工作**，使用 `../docker-v20260802/` 或更新版本。
