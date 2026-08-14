---
name: docker-v20260802
description: 运行 docker-v20260802 经典 GPU 胶子 PDF 验证管线 — 正确的 donghx OPE 算法 (对偶 F̃ + Wilson 线) + complex128 双精度 + 30 项修复
---

# docker-v20260802 — Classic Gluon PDF Validation Pipeline (GPU)

在 /root/lattice-pdf/agent/docker-v20260802 运行 disconnected 胶子 PDF 验证管线（CuPy/CUDA）。这是 **donghx 正确 OPE 算法的参考实现**：v20260730 算法 + v20260731 双精度 + 30 项 bug 修复。

## 版本身份

**v20260802** = v20260730 正确 OPE + v20260731 双精度 + 30 项修复
- OPE 算法：donghx（对偶 F̃ = 0.5·ε·F + roll 型 Wilson 线）
- 精度：complex128（可切 complex64）
- 特征向量：per-config 二进制，自动检测

## 固定配置

```
系综:   beta6.20_mu-0.2770_ms-0.2400_L24x72 (L24x72)
格点:   72×24³, a=0.1053 fm, β=6.20
Nev/Nev1: 100/100
组态:   6250, 6450, 6650
动量:   P=(0, 0, -2)
算符:   _Cg5g4 (Cγ₅γ₄)
精度:   complex128 (默认)
```

## 数据路径

| 数据 | 路径 |
|------|------|
| Eigenvectors | `/public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/` |
| Perambulators | `/public/group/lqcd/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/` |
| Gauge configs | `/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/` |

## 运行

```bash
cd /root/lattice-pdf/agent/docker-v20260802

python check_env.py                                    # 1. 环境检查
python run_pipeline.py --conf-id 6250 --skip-ope --skip-analysis -v   # 2. 快速 2pt 测试 (~8 min)
python run_pipeline.py --conf-id 6250                  # 3. 单组态完整 (~20 min 双精度)
python run_pipeline.py                                 # 4. 3 组态完整运行
python run_pipeline.py --precision complex64 --conf-id 6250   # 5. 单精度
python run_pipeline.py --smear --conf-id 6250          # 6. 启用特征向量 smearing
python diagnose_2pt.py --run-dir output_YYYYMMDD_HHMMSS --conf-id 6250   # 7. 后诊断
```

## 管线步骤

```
0 环境检查 → 1 质子 2pt 蒸馏 (GPU: VVV 重子块 + Wick 收缩 + 宇称投影 + 4 种 meff)
→ 2 OPE 从头计算 (GPU: Clover F_{μν} → 对偶 F̃ → roll Wilson 线 → trace + 空间求和)
→ 3 huangcl 比率分析 (Jackknife, R(z) = C3_disc / C2)
→ 4 Markdown 报告
```

## 关键文件

| 文件 | 用途 |
|------|------|
| `run_pipeline.py` | 4 步主编排器 (计时/内存跟踪) |
| `compute_2pt_gpu.py` | GPU 质子 2pt 蒸馏 (VVV + Wick) |
| `compute_ope_gpu.py` | GPU OPE (donghx 对偶 F̃ 算法) |
| `analyze_ratio.py` | Jackknife 比率分析 + 绘图 |
| `gamma_matrix_gpu.py` | DeGrand-Rossi γ 矩阵 (GPU) |
| `check_env.py` / `diagnose_2pt.py` | 环境检查 / 2pt 诊断 |
| `utils.py` / `run_config.json` | 工具 / 默认配置 |

## 依赖

Python 3.8+、CuPy (CUDA 12.x)、numpy/scipy/matplotlib、opt_einsum（可选）。依赖 `snsc/main.py` 的 `plaquette_clover` 函数。

## 版本历史

| 版本 | 关键变化 |
|------|----------|
| v20260726 | 初始 CPU 基线 |
| v20260729 | 单精度 GPU 基线 |
| v20260730 | donghx OPE 修复 (对偶 F̃, per-config eigvecs) |
| v20260731 | 双精度 (complex128) |
| **v20260802** | **合并 v60730 算法 + v60731 精度 + 30 修复** |
