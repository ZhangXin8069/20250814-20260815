---
name: docker-v20260803
description: 运行 docker-v20260803 生产版 GPU 蒸馏管线 — 集中 config.py + sush/lqcddb 风格 lib/ 库 + 介子/重子 2pt + 3pt/2pt 比率 + LaTeX 物理报告
---

# docker-v20260803 — Production Distillation Pipeline (GPU)

在 /root/lattice-pdf/agent/docker-v20260803 运行生产版 GPU 蒸馏管线（CuPy/CUDA，L24x72）。将配置集中到 `config.py`，分析代码抽取为 `lib/` 库（改编自 `examples/sush/lqcddb`），并产出 LaTeX 物理报告。

## 固定配置

```
系综:   beta6.20_mu-0.2770_ms-0.2400_L24x72 (L24x72)
格点:   24³×72, a=0.1053 fm, a⁻¹≈1.874 GeV, β=6.20
组态:   6250, 6450, 6650
NEV:    50 (GPU 测试默认; 生产 HPC 设为 100)
动量:   VdV/VVV sink (0,0,0), (0,0,2); 分析 P=(0,0,0), (0,0,2)
精度:   complex64 (默认)
```

所有参数在 `config.py` 中集中定义（`CONF_IDS`、`NEV`、数据路径 `BASE_*_DIR` 等）。

## 运行

```bash
cd /root/lattice-pdf/agent/docker-v20260803

python run_pipeline.py                          # 完整管线 (全部组态)
python run_pipeline.py --conf-ids 6250          # 单组态
python run_pipeline.py --steps vertex,corr      # 指定步骤
python run_pipeline.py --skip-vertex --skip-correlators   # 仅分析/绘图
python run_pipeline.py --precision complex128   # 双精度
python run_single_config.py --conf-id 6250      # 单组态端到端
```

CLI 参数：`--conf-ids`（空格分隔）、`--precision`、`--steps`（all|vertex|corr|...）、`--skip-vertex`、`--skip-correlators`、`--skip-analysis`、`--skip-plots`。

## 管线步骤

```
0 环境检查 → 1 顶点计算 (VdV/VVV, GPU) → 2 Wick 收缩 (介子/重子 2pt, 动态收缩)
→ 3 Jackknife 分析 (meff, 3pt/2pt 比率) → 4 绘图 → 5 LaTeX 物理报告
```

预期物理量：质子 E(P=0) ≈ 1.0 GeV，π 介子 m ≈ 0.3 GeV。

## 关键文件

| 文件 | 用途 |
|------|------|
| `config.py` | 集中配置 (系综/格点/组态/路径/动量/精度) |
| `run_pipeline.py` | 6 步主编排器 |
| `run_single_config.py` | 单组态端到端运行 |
| `full_analysis.py` | 全组态+多动量 Jackknife 分析 |
| `ratio_analysis.py` | 3pt/2pt 比率 R(τ) + Jackknife |
| `lib/` | 改编自 sush/lqcddb 的分析库 (vertex/autowick/dynamic/analyse...) |
| `physics_report.tex` | LaTeX 物理报告 (→ PDF) |
| `utils.py` | 日志 / Timer / GPU 内存 / 精度 |

## 说明

- VVV 计算量 ~Nev³，快速测试请只用 P=0。
- `lib/` 是 `examples/sush/lqcddb` 的快照，权威版本见该目录。
