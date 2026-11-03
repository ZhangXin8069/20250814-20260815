---
name: run-pipeline
description: 运行 docker-v20260726 本地验证管线 — OPE 从头计算，所有中间结果/日志/图表保存
---

# run-pipeline — docker-v20260726 本地验证管线

在当前环境直接运行完整的 disconnected 胶子 PDF 验证管线。

**核心特征**：
- **OPE 从头计算**：gauge config (.lime) → Clover plaquette → F_{μν} → Wilson 线 → 非定域 OPE → .npz
- **所有中间变量保存**：VVV 块、F_{μν} 张量、Wick 收缩、Wilson 线、有效质量等
- **完善日志系统**：主日志 (run.log) + 每步耗时内存记录 (timing.jsonl) + 每模块独立日志 + 每配置摘要 JSON
- **自动 bug 修复与优化循环**
- **所有过程输出为日志，所有中间/最终结果和数据图表均保存**

## 固定配置

```
系综:   beta6.20_mu-0.2770_ms-0.2400_L24x72 (L24x72)
格点:   72×24³, a=0.1053 fm, β=6.20
Nev:    100 (eigvec cfg_48000 复用)
组态:   6250, 6450, 6650 (Nconf=3)
动量:   P=(0, 0, -2), mom_smear=-2
算符:   _Cg5g4 (Cγ₅γ₄)
```

## 数据路径

| 数据 | 路径 |
|------|------|
| Eigenvectors | `/public/group/lqcd/sunpeng/eigen_vector/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_48000.eigenvector.npy` |
| Eigenvalues | `/public/group/lqcd/sunpeng/eigen_vector/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_48000.eigenvalue.npy` |
| Perambulators | `/public/group/lqcd/sunpeng/mom_smear_perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/output_dir_data/mz2_my0_mx0/{conf_id}/` |
| Gauge configs | `/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{conf_id}.lime` |
| OPE | **从头计算**：gauge config → Clover plaquette → F_{μν} → Wilson 线 → 非定域 OPE → .npz |

## 管线步骤

| 步骤 | 描述 | 保存内容 |
|------|------|----------|
| 0 | 环境检查 | Python 模块、数据路径可访问性 |
| 1 | 质子 2pt 蒸馏 | VVV 块、原始 Wick 收缩、宇称投影 (PP/PM)、有效质量 |
| 2 | OPE 从头计算 | F_{μν} 场强张量、Gauge 验证诊断、OPE 算符 .npz |
| 3 | huangcl 比率分析 | 比率 R(z)、诊断图、有效质量图、场强诊断图 |
| 4 | 最终报告 | Markdown 综合报告 |

## 输出目录结构

```
output_YYYYMMDD_HHMMSS/
├── run.log                      # 主日志 (全部 stdout/stderr, 含详细耗时)
├── run_config.json              # 运行时配置快照
├── run_config_snapshot.json     # 配置快照
├── final_report.md              # Markdown 报告
├── timing.jsonl                 # 每步耗时与内存记录
├── data/
│   ├── eigenvalues_Nev100.npy                   # 特征值 (中间结果)
│   ├── conf_{id}/
│   │   ├── gauge_validation_conf{id}.json       # Gauge 验证诊断
│   │   ├── Fmunu_mu{n}_nu{n}.npz                # 场强张量 (中间结果)
│   │   ├── ops_mu{n}_nu{n}_dz{delta_z}_conf{id}.npz  # OPE 算符
│   │   ├── VVV_Nev1{n}_Px{n}Py{n}Pz{n}_conf{id}.npy  # VVV 重子块 (中间结果)
│   │   ├── twopt_slice_pp_*_contract_*.npy      # 原始 Wick 收缩
│   │   ├── twopt_slice_pp_*_nopol_ss_*.npy      # 质子 2pt (PP)
│   │   ├── twopt_slice_pm_*_nopol_ss_*.npy      # 质子 2pt (PM)
│   │   ├── meff_Pz{n}_conf{id}.npz              # 有效质量
│   │   ├── compute_2pt_summary.json             # 2pt 计算摘要
│   │   └── compute_ope_summary_conf{id}.json    # OPE 计算摘要
│   ├── ...
└── plots/
    ├── ratio.png                        # R(z) 比率
    ├── ratio_diagnostics.png            # 多 z 值 Re/Im 诊断
    ├── effective_mass.png               # 有效质量 + 平台拟合
    ├── field_strength_diagnostics.png   # OPE 场强质量诊断
    ├── ratio_results.npz                # 数值结果
    ├── ope_combined.npz                 # 组合 OPE (-F_tx-F_ty+2*F_xy)
    ├── correlators_rel_time.npz         # 相对时间关联函数
    └── analysis_summary.json            # 分析摘要
```

## 子命令

### `run-pipeline run` — 运行完整管线

在当前环境直接运行，OPE 从头计算，保存所有中间结果。

```bash
cd /root/lattice-pdf/agent/docker-v20260726

# 完整运行 (3 组态, 约需数十分钟)
python run_pipeline.py

# 单组态快速测试 (推荐先跑)
python run_pipeline.py --conf-id 6250

# 跳过某些步骤 (使用已有数据)
python run_pipeline.py --skip-2pt --skip-ope

# 仅计算不分析
python run_pipeline.py --skip-analysis --skip-report

# 自定义输出目录
python run_pipeline.py --output-dir /path/to/output

# 详细调试模式
python run_pipeline.py --verbose
```

### `run-pipeline status` — 查看最新运行状态

```bash
latest=$(ls -dt /root/lattice-pdf/agent/docker-v20260726/output_*/ 2>/dev/null | head -1)
[ -n "$latest" ] && echo "=== $latest ===" && tail -50 "$latest/run.log"
```

### `run-pipeline check` — 环境检查

```bash
cd /root/lattice-pdf/agent/docker-v20260726 && python -c "
import sys, os
print(f'Python: {sys.version}')

# 核心依赖
for mod in ['numpy', 'scipy', 'matplotlib']:
    try:
        m = __import__(mod)
        print(f'  ✓ {mod}: {getattr(m, \"__version__\", \"?\")}')
    except ImportError:
        print(f'  ✗ {mod}: MISSING')

# 可选依赖
for mod in ['opt_einsum', 'h5py']:
    try:
        m = __import__(mod)
        print(f'  ○ {mod}: {getattr(m, \"__version__\", \"?\")}')
    except ImportError:
        print(f'  ○ {mod}: not installed (optional)')

# 数据路径
for name, path in [
    ('eigenvector', '/public/group/lqcd/sunpeng/eigen_vector/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_48000.eigenvector.npy'),
    ('eigenvalue', '/public/group/lqcd/sunpeng/eigen_vector/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_48000.eigenvalue.npy'),
    ('gauge_6250', '/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_6250.lime'),
]:
    ok = os.path.exists(path)
    print(f'  {\"✓\" if ok else \"✗\"} {name}: {path}')
if ok else print(f'  ✗ {name}: NOT FOUND')
"
```

### `run-pipeline download` — 数据下载说明

数据已在本地可访问（直接文件路径），无需下载。如果在新环境，使用 SSH rsync：

```bash
cd /root/lattice-pdf/agent/docker-v20260726
bash download_data.sh --yes
```

### `run-pipeline clean` — 清理输出

```bash
rm -rf /root/lattice-pdf/agent/docker-v20260726/output_*/
```

### `run-pipeline module <module>` — 单独运行某模块

```bash
cd /root/lattice-pdf/agent/docker-v20260726

# 仅计算 2pt
python compute_2pt.py --run-dir ./output_test

# 仅计算 OPE
python compute_ope.py --run-dir ./output_test --conf-id 6250

# 仅运行分析 (需要已有 2pt + OPE 数据)
python analyze_ratio.py --run-dir ./output_test --data-dir ./output_test/data --output-dir ./output_test/plots
```

## 关键设计决策

1. **OPE 从头计算**：从 gauge config (.lime ILDG 格式) 出发，通过 Clover plaquette → F_{μν} → Wilson 线 → 非定域 OPE。不依赖预计算 OPE 数据。
2. **所有中间变量保存**：VVV 块、F_{μν} 张量、Wick 收缩、Wilson 线 — 全部保存为 .npy/.npz/.json，支持增量重跑。
3. **完善日志**：主日志 (run.log) + timing.jsonl (每步耗时+内存) + 每模块独立日志 + 每配置摘要 JSON。
4. **Eigenvector cfg_48000 复用**：蒸馏标准做法，同一套特征向量用于所有组态。
5. **无 LQCD_Master/lamet-agent**：直接运行物理计算，不做 AI agent 编排。
6. **自动 bug 修复**：遇到错误自动诊断并循环修复。

## 依赖

- Python 3.8+
- numpy, scipy, matplotlib
- opt_einsum (可选，加速张量缩并)
- h5py (可选)
- 本项目 `snsc/main.py` (plaquette_clover)
- 本项目 `agent/docker-v20260726/gamma_matrix.py` (DeGrand-Rossi γ 矩阵)

## 不明确问题 (已确认)

1. **Eigenvector cfg 复用**: ✓ cfg_48000 同一套特征向量用于所有组态 (蒸馏标准做法)。
2. **Perambulator 格式**: ✓ 每个 conf_id 目录含 288 个 perams.{conf_id}.{d_src}.{t_src} 文件。
3. **Gauge config 格式**: ✓ .lime ILDG 格式, big-endian float64, 含 XML header。
4. **OPE 算符**: ✓ Unpolarized gluon OPE 组合 = -F_{tx} - F_{ty} + 2*F_{xy}。
5. **动量**: ✓ 仅 Pz=-2（对应 mz2_my0_mx0 目录）。
