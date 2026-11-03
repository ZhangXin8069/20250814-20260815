---
name: snsc-v20260726
description: 运行 snsc-v20260726 CPU 集群管线 — 剥离 AI agent 阶段的 NumPy 验证管线 (已被 GPU 管线取代)
---

# snsc-v20260726 — CPU Cluster Validation Pipeline

在 /root/lattice-pdf/agent/snsc-v20260726 运行剥离了 AI agent 阶段（LQCD_Master + lamet-agent）的 CPU (NumPy) disconnected 胶子 PDF 验证管线。

**⚠️ 已被 GPU 管线 `../docker-v20260802/`（及更新版本）取代。** 无 AI agent，纯物理计算。

## 固定配置

```
系综:   beta6.20_mu-0.2770_ms-0.2400_L24x72
格点:   72×24³, a=0.1053 fm, β=6.20
Nev:    100
组态:   6250, 6450, 6650
动量:   P = (0, 0, -2)
算符:   Cγ₅γ₄
```

## 运行

```bash
cd /root/lattice-pdf/agent/snsc-v20260726

# 直接运行
python run_pipeline.py

# 选择性步骤
python run_pipeline.py --skip-2pt --skip-ope --skip-analysis --skip-report

# Slurm 集群提交
sbatch sbatch.sh

# 先从集群下载数据
bash download_data.sh --yes --skip-existing
```

## 管线步骤

```
0 环境检查 → 1 质子 2pt 蒸馏 (CPU) → 2 OPE 从头计算或加载预计算 donghx 数据
→ 3 huangcl 比率分析 (R(z) = C3_disc/C2, jackknife) → 4 Markdown 报告
```

## 关键文件

| 文件 | 用途 |
|------|------|
| `run_pipeline.py` | 4 步主编排器 (计时/内存跟踪) |
| `compute_2pt.py` | 质子 2pt 蒸馏 (CPU) |
| `compute_ope.py` | OPE 算符 (ILDG .lime → Clover F_{μν}) |
| `analyze_ratio.py` | huangcl 比率分析 + jackknife |
| `gamma_matrix.py` / `utils.py` | γ 矩阵 / 共享工具 |
| `download_data.sh` / `sbatch.sh` | 数据下载 / Slurm 提交 |

## 依赖

依赖 `../../snsc/main.py` 的 plaquette 构造。数据从 SNSC 集群经 `download_data.sh` (rsync/scp) 下载。
