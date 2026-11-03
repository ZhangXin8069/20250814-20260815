---
name: run-snsc-pipeline
description: 运行 SNSC 验证管线 — 聚合入口; 集群 CPU 版 snsc-v20260726 与 GPU 版 docker-vYYYYMMDD 各有独立完整技能
---

# run-snsc-pipeline — SNSC 验证管线（聚合入口）

运行整合版 disconnected 胶子 PDF 验证管线，使用真实 L24x72 系综格点数据。

**核心算法**：质子 2pt 蒸馏 + OPE 非定域胶子算符 + huangcl 比率分析。

## 版本选择

| 管线 | 类型 | 技能 | 说明 |
|------|------|------|------|
| `snsc-v20260726` | 集群 CPU (NumPy) | `snsc-v20260726` | 剥离 AI 阶段，Slurm 提交，OPE 从预计算 donghx 目录加载 |
| `snsc-v20260725` | 历史 | `snsc-v20260725` | 最早版本，无源码（仅历史运行输出） |
| `docker-v20260805` | **★当前 GPU** | `docker-v20260805` | 全量蒸馏工具包（自包含 lib/ + 10 组态 + OPE + code_1.py 统计 + LaTeX 报告） |
| `docker-v20260804` | GPU | `docker-v20260804` | 完整蒸馏工具包（顶点 + 多强子 Wick + 统计） |
| `docker-v20260803` | GPU | `docker-v20260803` | 集中 config.py + sush lib/ + LaTeX 报告 |
| `docker-v20260802` | GPU 经典 | `docker-v20260802` | donghx 正确 OPE + complex128 |
| `docker-v20260731` | GPU 历史 | `docker-v20260731` | 首个双精度 |
| `docker-v20260726` | GPU/CPU 历史 | `docker-v20260726` | CPU 基线（本 skill 旧文主推） |

**每个版本的详细命令、固定配置、数据路径见对应的 `docker-vYYYYMMDD` / `snsc-vYYYYMMDD` 技能文件**（在 `/root/lattice-pdf/agent/.claude/skills/`）。

## 快速使用

```bash
# 集群 CPU 版
cd /root/lattice-pdf/agent/snsc-v20260726
python run_pipeline.py          # 直接运行
sbatch sbatch.sh                # Slurm 提交
bash download_data.sh --yes     # 从 SNSC 集群下载数据

# 当前 GPU 版（本地）
cd /root/lattice-pdf/agent/docker-v20260804
python check_env.py
python run_pipeline.py --conf-id 6250 --skip-4pt
```

## 管线步骤（各版本通用）

```
0 环境检查 → 1 质子 2pt 蒸馏 (VVV + Wick + 宇称投影) → 2 OPE 算符（GPU 从头计算 或 加载 donghx 预计算）
→ 3 huangcl 比率分析 (R(z) = C3_disc/C2, Jackknife) → 4 Markdown 报告
```

## 数据路径

- **GPU 版（标准路径）**：eigenvectors `/public/group/lqcd/eigensystem/...`，perambulators `/public/group/lqcd/perambulators/.../light/`，gauge `/public/group/lqcd/configurations/CLOVER/...`，OPE 从头计算。
- **snsc 版**：OPE 从 donghx 预计算目录 `/public/group/lqcd/donghx/Ope_Gluon/.../L24x72/zdir/` 加载。
- **早期 docker-v20260726/27/28**：`sunpeng/` 旧路径，共享 `cfg_48000.eigenvector.npy`。

## 依赖

- Python 3.8+、numpy/scipy/matplotlib、opt_einsum（可选）、h5py（可选）
- `../../snsc/main.py` 的 `plaquette_clover` 函数（OPE 计算用）
- 数据经 SSH rsync 从 SNSC 集群同步（见各版本 `download_data.sh`）

## 关键设计决策（历史回顾）

1. **OPE 从头计算 vs 预计算**：docker 版本从 gauge config 出发完整计算 OPE；snsc 版本加载 donghx 预计算目录。
2. **不调用 LQCD_Master/lamet-agent**：直接运行物理计算，不做 AI agent 编排（这是与 `../snsc/` 的区别）。
3. **所有中间变量保存**：VVV 块、F_{μν}、Wick 收缩、Wilson 线等存为 .npy/.npz/.json，支持增量重跑与调试。
