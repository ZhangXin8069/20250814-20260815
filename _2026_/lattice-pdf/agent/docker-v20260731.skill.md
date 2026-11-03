---
name: docker-v20260731
description: 运行 docker-v20260731 历史 GPU 管线 — 首个双精度 (complex128) 版本 (历史, 已被 v20260802+ 取代)
---

# docker-v20260731 — Historical GPU Pipeline (First Double Precision)

在 /root/lattice-pdf/agent/docker-v20260731 运行 disconnected 胶子 PDF 验证管线（CuPy/CUDA，L24x72）。

**⚠️ 历史版本** — 系列中首个双精度版本；已被 `../docker-v20260802/` 取代。

## 本版本特点

- **双精度 (complex128)**：全程双精度（输入、计算、输出），内存约为单精度的 2 倍
- 读取 per-config、per-time-slice 二进制特征向量（little-endian float64 对）
- OPE 算法沿用 v20260730 的对偶 F̃

## 固定配置

```
系综:   beta6.20_mu-0.2770_ms-0.2400_L24x72
格点:   72×24³, a=0.1053 fm, β=6.20
精度:   complex128 (默认)
特征向量: per-config 二进制 (每 config 72 个时间片, ~4.5 GB)
```

## 运行

```bash
cd /root/lattice-pdf/agent/docker-v20260731
python run_pipeline.py --conf-id 6250                 # 单组态双精度
python run_pipeline.py --precision complex64 --conf-id 6250   # 切换单精度
```

**不要用于新工作**，使用 `../docker-v20260802/` 或更新版本。
