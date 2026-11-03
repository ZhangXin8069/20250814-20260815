# docker-v20260805 最终报告

**日期**: 2026-08-02
**工作目录**: `/root/lattice-pdf/agent/docker-v20260805`
**运行目录**: `output/output_20260802_120104`

## 概述

docker-v20260805 是在 GPU (CUDA, RTX 4060 8GB) 上实现的全量蒸馏计算管线，以
`examples/sush/lqcddb` 为蓝本（照抄而不 import），参考 `agent/docker-v20260803`
的集中式配置与 LaTeX 报告架构。实现了：

1. **顶点函数** — VdV (介子) / VVV (重子)，GPU 按时间片流式计算，VVV 采用
   x-slice 分解（比单 einsum 快 ~20×）。
2. **Wick 收缩分析 + 动态收缩** — sush lqcddb 引擎（注册表 + einsum 计划缓存，
   热路径加速 ~56×）。
3. **关联函数** — 两点 (pp / pn / pion)、OPE (donghx 胶子算符)、三点 (PJN)、
   四点 (PJNNJNp)。
4. **统计分析** — Jackknife / 有效质量 / ratio_3p，输出形式与
   `examples/huangcl/02_ratio/code_1.py` 一致（fit report + 汇总表 + 图）。
5. **pion & proton** 在 P=(0,0,0) 与 P=(0,0,2)。
6. **10 个组态**: 6250, 6450, 6650, 6850, 7050, 7250, 7450, 7650, 7850, 8050。
7. **单精度 (complex64)** 计算，数据读写间有明确的精度转换（磁盘 complex128 →
   GPU complex64）。

## 物理结果（Jackknife，10 组态）

| 粒子 | 动量 | E0 (GeV) | 期望 (GeV) | 说明 |
|------|------|----------|------------|------|
| proton | P=(0,0,0) | **1.118 ± 0.008** | 1.0 | 该系综夸克质量较重 (m_π≈0.3)，质子偏重（v20260803 得 1.053） |
| proton | P=(0,0,2) | **1.559 ± 0.018** | 1.49 (√(m0²+p²)) | 色散关系基本一致 |
| pion | P=(0,0,0) | **0.2863 ± 0.0020** | 0.30 | 干净平台，0.7% 精度 |
| pion | P=(0,0,2) | **1.18 ± 0.19** | 1.02 | 噪声较大（高动量） |

连通三点/两点比值 R(τ)（γ₃ 分量）:
- **proton P0**: R(10..12) ≈ **+0.134 → 0.140**（小正值，稳定）
- **pion P0**: R(10..11) ≈ **-0.959**（0.15% 精度，清晰平台）
- proton P2 / pion P2: 高动量道噪声较大

OPE（不相连胶子算符）: 与 v20260802 参考完全一致（相关系数 1.000000）。
不相连比值按 code_1.py 算法计算；由于仅 10 组态（code_1.py 用 200），
jackknife 交叉项被小分母放大，拟合参数约束较弱——已如实记录。

## 与 docker-v20260803 的对照

| 项目 | v20260803 | v20260805 |
|------|-----------|-----------|
| 组态数 | 3 | **10** |
| Nev | 50 | **100** |
| 2pt | pp + pion | **pp + pn + pion** |
| OPE | 未计算 | **donghx 胶子算符**（与参考一致） |
| 3pt / 4pt | 3pt (ratio_analysis) | **3pt PJN + 4pt PJNNJNp** |
| 不相连比值 | 无 | **code_1.py 形式（fit report + 图）** |
| 统计输出 | 自定义 | **code_1.py 形式一致** |
| 报告 | physics_report.tex | **physics_report.tex（自动填数值）** |

## 计算耗时（RTX 4060, 10 组态）

| 步骤 | 耗时 |
|------|------|
| 顶点函数 (VdV/VVV) | ~18 min |
| 2pt (pp/pn/pion, P0/P2) | ~63 min |
| OPE (3 分量 × 10) | ~86 min |
| 3pt PJN (P0/P2) | ~104 min |
| 4pt PJNNJNp (简化范围) | ~37 min |
| 分析 + 绘图 + 报告 | ~0.3 min |
| **合计** | **~5.2 h** |

4pt 采用简化范围（Nev1=60, t_sep=6, P0, src_step=2）以控制在 8GB GPU 上的开销；
全范围（Nev=100, 全动量）约需 4.6 h/组态，不可行。

## 输出文件

- 每个组态: `data/conf{id}/` 下的 VdV/VVV、corr_{pp|pn|pion}_{P0|P2}、ops_mu*_nu*、
  ope_combined、{proton|pion}_{P0|P2}_3pt、pjnnjnp_4pt
- 分析: `data/analysis/` meff/corr/ratio mean±err
- 不相连比值: `analysis/disconnected/` (ratio_*.npy, 0_fit_data.npz, 1_fit_report.txt, 图)
- 图: `plots/` meff_all_channels.png, correlators_all_channels.png, ratio_3pt_all_channels.png
- 报告: `physics_report.pdf`（复制到 `/root/lattice-pdf/agent/logs/`）
- 日志: `/root/lattice-pdf/agent/logs/docker-v20260805-*.log`
