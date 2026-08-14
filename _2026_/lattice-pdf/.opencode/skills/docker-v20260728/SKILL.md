---
name: docker-v20260728
description: 运行 docker-v20260728 历史 CPU 管线 — 改进有效质量提取 (前向平均 + log 方法) (历史, 已被 v20260802+ 取代)
---

# docker-v20260728 — Historical CPU Pipeline (Improved Effective Mass)

在 /root/lattice-pdf/agent/docker-v20260728 运行 disconnected 胶子 PDF 验证管线（CPU NumPy）。

**⚠️ 历史版本** — 改进有效质量提取的 CPU 版本；已被 `../docker-v20260802/` 取代。

## 本版本创新

修复有效质量计算：
- naive arccosh → **前向平均**（避免边界反射污染）
- 新增 **log 有效质量** 方法（对噪声重子更稳健）
- 增强诊断：`C2pt_1d_fwd`, `meff_log`, `meff_cosh`

该修复沿用至 v20260729 及之后。

## 固定配置

```
系综:   beta6.20_mu-0.2770_ms-0.2400_L24x72
计算:   CPU (NumPy), 双精度
特征向量: 共享 .npy
数据路径: sunpeng/
OPE 算法: 基本 F (无对偶 F̃)
```

## 运行

```bash
cd /root/lattice-pdf/agent/docker-v20260728
python run_pipeline.py    # CPU only
bash download_data.sh --yes --skip-existing   # 从集群下载数据
```

**不要用于新工作**，使用 `../docker-v20260802/` 或更新版本。
