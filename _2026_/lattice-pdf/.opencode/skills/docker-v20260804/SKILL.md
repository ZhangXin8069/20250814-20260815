---
name: docker-v20260804
description: 运行 docker-v20260804 完整蒸馏工具包 GPU 管线 — VdV/VVV 顶点 + 多强子 Wick 收缩 (π/p/n 2pt, OPE, PJN 3pt, PJNNJNp 4pt) + Jackknife 统计 (历史版, 当前最新为 docker-v20260805)
---

# docker-v20260804 — Full Distillation Toolkit (GPU)

在 /root/lattice-pdf/agent/docker-v20260804 运行完整蒸馏工具包 GPU 管线（CuPy/CUDA，L24x72 真实格点数据）。**上一代最新版（当前最新为 `docker-v20260805`）**。综合了 LQCD_Master、lamet-agent、`examples/sush/lqcddb`、docker-v20260802 的模式。

## 固定配置

```
系综:   beta6.20_mu-0.2770_ms-0.2400_L24x72 (L24x72)
格点:   72 × 24³, a=0.1053 fm, β=6.20
组态:   6250, 6450, 6650
Nev:    100;  Nev1: 100 (VVV/收缩截断)
算符:   _Cg5g4 (Cγ₅γ₄); 可选 _Cg5g3 / _Cg5
精度:   complex64 (默认)
```

## 数据路径

| 数据 | 路径 |
|------|------|
| Eigenvectors | `/public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/` |
| Perambulators | `/public/group/lqcd/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/` |
| Gauge configs | `/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/` |

## 运行

```bash
cd /root/lattice-pdf/agent/docker-v20260804

python check_env.py                               # 1. 环境检查
python run_pipeline.py --conf-id 6250 --skip-4pt  # 2. 单组态测试 (~15 min)
python run_pipeline.py                            # 3. 完整运行 (全部组态)
python run_pipeline.py --precision complex128     # 4. 双精度
python run_pipeline.py --element _Cg5g4 --meff-method fit_cosh   # 5. 选算符/拟合方法
python run_pipeline.py --skip-2pt --skip-ope --skip-3pt --skip-4pt   # 6. 仅分析
```

CLI 参数：`--conf-id`、`--conf-ids`（逗号分隔）、`--precision`、`--element`、`--meff-method`（fit_cosh/cosh/fit_exp）、`--Nev1`、`--verbose`。

## 管线步骤

```
0 环境检查 → 1 数据加载 → 2 顶点计算 (VdV+VVV, GPU) → 3 Wick 收缩 (2pt/OPE/3pt/4pt)
→ 4 统计分析 (Jackknife, meff, ratio_3pt) → 5 绘图 → 6 Markdown 报告
```

## 验证目标

| 粒子 | P=(0,0,0) | P=(0,0,2) |
|------|-----------|-----------|
| Pion | ~0.14 GeV | ~0.52 GeV |
| Proton | ~1.0 GeV | ~1.4 GeV |

## 关键文件

| 文件 | 用途 |
|------|------|
| `run_pipeline.py` | 6 步主编排器 |
| `compute_vertex.py` | VdV/VVV 动量投影顶点 |
| `compute_contraction.py` | 各类关联函数 Wick 收缩 |
| `analyze.py` | Jackknife / meff / ratio / 绘图 |
| `data_io.py` | 特征向量/传播子/gauge 读取 |
| `gamma_matrix_gpu.py` | DeGrand-Rossi γ 矩阵 (GPU) |
| `check_env.py` / `utils.py` | 环境检查 / 共享工具 |
| `README.md` | 完整英文文档 |

## 输出

`output/output_YYYYMMDD_HHMMSS/` → `data/conf{id}/*.npy`、`analysis/summary.json`、`plots/*.png`、`REPORT.md`。日志同时写入 `/root/lattice-pdf/agent/logs/`。
