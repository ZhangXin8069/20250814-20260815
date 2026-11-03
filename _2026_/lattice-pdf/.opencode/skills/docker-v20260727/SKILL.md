---
name: docker-v20260727
description: 运行 docker-v20260727 历史 GPU 管线 — 首个 GPU 移植 (CuPy, 单精度), OPE 从头计算 (历史, 已被 v20260802+ 取代)
---

# docker-v20260727 — Historical GPU Pipeline (First GPU Port)

在 /root/lattice-pdf/agent/docker-v20260727 运行 disconnected 胶子 PDF 验证管线（CuPy/CUDA，L24x72）。

**⚠️ 历史版本** — 系列中首个 GPU 移植版本；已被 `../docker-v20260802/` 取代。

## 本版本特点

- **GPU 加速 (CuPy)**：VVV 重子块、Wick 收缩、F_{μν}、Wilson 线、OPE 缩并全部在 GPU
- **大规模数据传输**：4.5 GB 特征向量分时间片传 GPU，避免 OOM
- **OPE 从头计算**：gauge config (.lime) → Clover plaquette → F_{μν} → Wilson 线 → 非定域 OPE → .npz
- 完整日志系统 + 自动 bug 修复与优化循环

> 注：本目录自带 `.claude/skills/run-pipeline.md`（skill 名 `run-pipeline-gpu`），内容与此 skill 等价。

## 固定配置

```
系综:   beta6.20_mu-0.2770_ms-0.2400_L24x72
格点:   72×24³, a=0.1053 fm, β=6.20
Nev:    100 (eigvec cfg_48000 复用)
组态:   6250, 6450, 6650
动量:   P=(0, 0, -2), mom_smear=-2
算符:   _Cg5g4 (Cγ₅γ₄)
精度:   complex64
```

## 数据路径

| 数据 | 路径 |
|------|------|
| Eigenvectors | `/public/group/lqcd/sunpeng/eigen_vector/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_48000.eigenvector.npy` |
| Perambulators | `/public/group/lqcd/sunpeng/mom_smear_perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/output_dir_data/mz2_my0_mx0/{conf_id}/` |
| Gauge configs | `/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{conf_id}.lime` |

## 运行

```bash
cd /root/lattice-pdf/agent/docker-v20260727
python run_pipeline.py                 # 完整 GPU 管线 (3 组态)
python run_pipeline.py --conf-id 6250  # 单组态 (~3 min)
```

**不要用于新工作**，使用 `../docker-v20260802/` 或更新版本。
