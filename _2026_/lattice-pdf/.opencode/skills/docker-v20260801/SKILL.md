---
name: docker-v20260801
description: 运行 docker-v20260801 历史 GPU 管线 — 自动平台有效质量 + Pz=0 校准 (历史版本, 已被 v20260802+ 取代)
---

# docker-v20260801 — Historical GPU Pipeline

在 /root/lattice-pdf/agent/docker-v20260801 运行 disconnected 胶子 PDF 验证管线（CuPy/CUDA，L24x72）。

**⚠️ 历史版本** — 已被 `../docker-v20260802/`（及更新版本）取代，新工作请勿使用。

## 本版本特点

- **自动平台有效质量** (`exp_forward`)：自动探测平台，拒绝激发态污染
- **Pz=0 + Pz=-2 校准**：先校准静止质量，再验证 E = √(m² + P²)
- 最小文件集（7 个），无 README/check_env/diagnose

## 固定配置

```
系综:   beta6.20_mu-0.2770_ms-0.2400_L24x72
格点:   72×24³, a=0.1053 fm, β=6.20
精度:   complex64 (单精度)
特征向量: per-config 二进制
数据路径: 标准 (eigensystem/perambulators/gauge)
OPE 算法: 对偶 F̃ (donghx)
```

## 运行

```bash
cd /root/lattice-pdf/agent/docker-v20260801
python run_pipeline.py    # GPU 单精度, 自动平台 meff + Pz=0 校准
```

**不要用于新工作**，使用 `../docker-v20260802/` 或更新版本。
