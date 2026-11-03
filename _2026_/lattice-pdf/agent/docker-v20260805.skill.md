---
name: docker-v20260805
description: 运行 docker-v20260805 全量蒸馏 GPU 管线 — 自包含 lqcddb lib/ + 集中 config.py + VdV/VVV 顶点 + Wick/动态收缩 + 全关联函数集 (2pt pp/pn/pion、OPE、3pt PJN、4pt PJNNJNp) + code_1.py 统计 + LaTeX 报告 (10 组态, 当前最新版)
---

# docker-v20260805 — Full Distillation Pipeline (GPU, 10 configs)

在 /root/lattice-pdf/agent/docker-v20260805 运行生产级 GPU (CuPy/CUDA) 蒸馏管线。
以 `examples/sush/lqcddb` 为蓝本（照抄不 import 的自包含 `lib/`），参考 `docker-v20260803`。

**★ 当前最新版**。版本标识：v20260805 = 自包含 lib + 全关联函数集 + code_1.py 统计形式 + 10 组态。

## 固定配置

```
系综:   beta6.20_mu-0.2770_ms-0.2400_L24x72 (L24x72)
格点:   72 × 24³, a=0.1053 fm, a⁻¹≈1.874 GeV, β=6.20
组态:   10 个 — 6250, 6450, 6650, 6850, 7050, 7250, 7450, 7650, 7850, 8050
Nev:    100;  Nev1: 100 (VVV/收缩截断);  4pt 用 FOURPT_NEV1=60
2pt:    pp + pn + pion @ P0/P2;  3pt: PJN;  4pt: PJNNJNp (简化范围)
OPE:    donghx 胶子算符 (Clover F̃ + Wilson 线), Δz=24, (μ,ν)=(0,1),(3,0),(3,1)
精度:   complex64 (默认, 可 --precision complex128)
```

## 数据路径

| 数据 | 路径 |
|------|------|
| Eigenvectors | `/public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/` |
| Perambulators | `/public/group/lqcd/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/` |
| Gauge configs | `/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/` |

## 运行

```bash
cd /root/lattice-pdf/agent/docker-v20260805

python run_pipeline.py --conf-id 6250 --skip-3pt --skip-4pt --skip-report   # 1. 冒烟测试 (~15 min)
python run_pipeline.py --conf-id 6250 --Nev1 60                             # 2. 单组态快速 (截断 VVV)
python run_pipeline.py                                                      # 3. 完整运行 (10 组态, ~5.2h)
python run_pipeline.py --precision complex128                               # 4. 双精度
python run_pipeline.py --channels pp,pion                                   # 5. 选 2pt 道
python run_pipeline.py --run-dir output/output_XXX --steps analysis,plots,report   # 6. 续跑 (仅分析)
python report.py --run-dir output/output_XXX --out /root/lattice-pdf/agent/logs    # 7. 单独生成 LaTeX 报告
```

CLI 参数：`--conf-id` / `--conf-ids`（逗号分隔）、`--precision`、`--Nev1`、`--steps`（env,vertex,2pt,ope,3pt,4pt,analysis,plots,report）、
`--skip-2pt/--skip-ope/--skip-3pt/--skip-4pt/--skip-analysis/--skip-plots/--skip-report`、`--channels`（pp,pn,pion）、
`--fourpt-nev1`、`--fourpt-tsep`、`--run-dir`、`--verbose`。

## 管线步骤

```
0 env → 1 vertex (VdV+VVV, GPU, x-slice 分解) → 2 2pt (pp/pn/pion) → 3 OPE (胶子算符)
→ 4 3pt (PJN) → 5 4pt (PJNNJNp) → 6 统计 (Jackknife/meff/ratio_3p, code_1.py 形式)
→ 7 绘图 → 8 LaTeX 报告 (physics_report.tex→PDF)
```

## 验证目标

| 粒子 | P=(0,0,0) | P=(0,0,2) |
|------|-----------|-----------|
| Pion | ~0.30 GeV | ~0.98 GeV |
| Proton | ~1.12 GeV (该系综实测) | ~1.48 GeV |

## 关键文件

| 文件 | 用途 |
|------|------|
| `config.py` | 集中配置（系综、组态、动量、算符、OPE、路径） |
| `lib/` | 自包含蒸馏收缩框架（lqcddb 快照，照抄不 import） |
| `compute_vertex.py` | VdV/VVV 顶点函数（GPU, x-slice 分解, 比单 einsum 快 20×） |
| `compute_contraction.py` | Wick + 动态收缩：2pt/3pt/4pt |
| `compute_ope.py` | donghx 胶子算符（Clover F̃ + Wilson 线） |
| `analyze.py` | Jackknife / meff / ratio_3p + code_1.py 拟合与绘图 |
| `run_pipeline.py` | 主调度器（9 步，支持 `--run-dir` 续跑） |
| `report.py` | 自动生成并编译 LaTeX 报告 |
| `utils.py` | 日志/计时/显存/数组 I/O |
| `README.md` | 完整中文文档 |

## 关键物理发现（勿重复调试）

- **pn 2pt = 0**: 质子 (uud) ↔ 中子 (udd) 味不守恒，Wick 无有效图（KeyError→0）。物理正确。
- **质子质量 ~1.12 GeV**（非 1.0）：该系综夸克重 (m_π≈0.286)，v20260803 得 1.053。平台窗 [6,12] 避免早期激发态污染。
- **OPE 已验证**：与 v20260802 相关系数 1.0。.lime 文件有 136 字节 trailer → 数据偏移 = `file_size - expected_bytes - 136`（代码内 ±16KB 扫描自动处理）。
- **不相连比值（code_1.py 形式）在 10 组态下噪声大**（code_1.py 用 200 组态）；连通 3pt/2pt 比值 R(τ) 是干净结果（pion P0 ≈ -0.96）。
- **4pt 全范围不可行**（230s/t_src）；已用简化范围（Nev1=60, t_sep=6, P0, src_step=2）。
- **内存约束 15GB**: 不得同时持有所有组态的 VVV（~1.15GB/组态），按组态从磁盘加载。

## 输出

`output/output_YYYYMMDD_HHMMSS/` → `data/conf{id}/*.npy`（VdV/VVV、corr_{pp|pn|pi}_{P0|P2}、3pt、4pt、OPE npz）、
`data/analysis/`（meff/ratio 均值与误差）、`analysis/disconnected/`（code_1.py 不相连比值拟合）、`plots/*.png`、
`physics_report.tex/.pdf`、`run_config.json`、`analysis_summary.json`。日志同时写入 `/root/lattice-pdf/agent/logs/`。
