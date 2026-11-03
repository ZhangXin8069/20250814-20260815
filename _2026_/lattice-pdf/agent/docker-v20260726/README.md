# Docker/本地环境直接运行版 — v20260726

整合版验证管线，使用真实格点数据复现 huangcl 的 disconnected 胶子 PDF 比率分析。

**与 snsc-v20260726 的关键区别**：
1. **OPE 从头计算**（不依赖预计算数据）：gauge config → Clover plaquette → F_{μν} → 非定域 OPE → .npz
2. **所有中间变量保存**（VVV, F_{μν}, Wilson线, Wick收缩等）
3. **完善日志系统**（主日志 + 每模块独立日志）
4. **本地环境直接运行**（无需 Slurm 提交）
5. **自动 bug 修复与优化循环**

## 快速开始

```bash
# 1. 直接运行 (当前环境)
cd /root/lattice-pdf/agent/docker-v20260726
python run_pipeline.py

# 2. 单组态测试 (快速验证)
python run_pipeline.py --conf-id 6250

# 3. 只运行 OPE 计算 + 分析 (已有 2pt 数据)
python run_pipeline.py --skip-2pt

# 4. 只计算 2pt + OPE (不分析)
python run_pipeline.py --skip-analysis --skip-report

# 5. 启用详细调试输出
python run_pipeline.py --verbose
```

## 管线步骤

```
Step 0: 环境检查 (Python 模块, 数据路径可访问性)
Step 1: 质子 2pt 蒸馏计算
        - 读取 eigvecs (.npy), perambulators (二进制)
        - 计算 VVV baryon block (6-term ε 收缩)
        - Wick 收缩 (Direct - Exchange)
        - 宇称投影 (P+ / P-)
        - 保存: conf_{id}/twopt_slice_pp_*.npy, VVV_*.npy, meff_*.npz
Step 2: OPE 从头计算 ★
        - 读取规范组态 (.lime ILDG 格式)
        - 验证规范组态 (unitarity, trace, plaquette)
        - 计算场强张量 F_{μν} (Clover plaquette) → 保存为中间结果
        - 构造非定域 OPE 算符: Tr[F(z)·W(z→0)·F(0)·W(0→z)]
        - 保存为统一格式: ops_mu{mu}_nu{nu}_dz{delta_z}_conf{id}.npz
Step 3: huangcl 比率分析
        - 加载 2pt + OPE → 构建 3pt disconnected
        - Jackknife 重采样
        - 计算 R(z) = C3_disc / C2
        - 画图: ratio.png, diagnostics.png, effective_mass.png, field_strength_diagnostics.png
Step 4: 最终报告 (Markdown)
```

## 固定参数

| 参数 | 值 |
|------|-----|
| 系综 | beta6.20_mu-0.2770_ms-0.2400_L24x72 |
| 格点 | 72×24³ |
| 格距 a | 0.1053 fm |
| Nev | 100 |
| 组态 | 6250, 6450, 6650 (Nconf=3) |
| 动量 | P = (0, 0, -2) |
| 插值算符 | Cγ₅γ₄ (_Cg5g4) |
| 动量涂抹 | mom_smear=-2, phase=2 |

## 数据源

| 数据 | 路径 |
|------|------|
| Eigenvectors | `/public/group/lqcd/sunpeng/eigen_vector/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_48000.eigenvector.npy` |
| Eigenvalues | `/public/group/lqcd/sunpeng/eigen_vector/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_48000.eigenvalue.npy` |
| Perambulators | `/public/group/lqcd/sunpeng/mom_smear_perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/output_dir_data/mz2_my0_mx0/{conf_id}/` |
| Gauge configs | `/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{conf_id}.lime` |
| OPE | **从头计算**：gauge config → Clover plaquette → F_{μν} → 非定域 OPE → .npz |

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
│   ├── conf_6250/
│   │   ├── gauge_validation_conf6250.json       # Gauge 验证诊断
│   │   ├── Fmunu_mu0_nu1.npz                    # F_{xy} 场强张量 (中间结果)
│   │   ├── Fmunu_mu3_nu0.npz                    # F_{tx} 场强张量 (中间结果)
│   │   ├── Fmunu_mu3_nu1.npz                    # F_{ty} 场强张量 (中间结果)
│   │   ├── ops_mu0_nu1_dz24_conf6250.npz        # OPE F_xy
│   │   ├── ops_mu3_nu0_dz24_conf6250.npz        # OPE F_tx
│   │   ├── ops_mu3_nu1_dz24_conf6250.npz        # OPE F_ty
│   │   ├── VVV_Nev1100_Px0Py0Pz-2_conf6250.npy   # VVV 缓存 (中间结果)
│   │   ├── twopt_slice_pp_..._contract_conf6250.npy   # 原始 Wick 收缩
│   │   ├── twopt_slice_pp_..._nopol_ss_conf6250.npy   # 质子 2pt (PP)
│   │   ├── twopt_slice_pm_..._nopol_ss_conf6250.npy   # 质子 2pt (PM)
│   │   ├── meff_Pz-2_conf6250.npz                    # 有效质量
│   │   ├── compute_2pt_summary.json              # 2pt 计算摘要
│   │   └── compute_ope_summary_conf6250.json     # OPE 计算摘要
│   ├── conf_6450/...
│   └── conf_6650/...
└── plots/
    ├── ratio.png                        # 比率 R(z), Pz=-2, 多 tsep
    ├── ratio_diagnostics.png            # 诊断: 多 z 值的 Re/Im
    ├── effective_mass.png               # 有效质量 (3-panel)
    ├── field_strength_diagnostics.png   # OPE 质量诊断 (4-panel)
    ├── ratio_results.npz                # 数值结果
    ├── analysis_summary.json            # 分析摘要
    ├── ope_combined.npz                 # 组合 OPE 数据
    └── correlators_rel_time.npz         # 相对时间关联函数
```

## 中间结果说明

| 文件 | 内容 | 用途 |
|------|------|------|
| `Fmunu_mu{n}_nu{n}.npz` | Clover 场强张量 F_{μν}(x) | 可重用，跳过重复计算 |
| `VVV_*.npy` | VVV 重子块 | 缓存，同组态多动量时复用 |
| `twopt_slice_pp_*_contract_*.npy` | 原始 Wick 收缩矩阵 | 调试宇称投影 |
| `gauge_validation_*.json` | Gauge 组态验证诊断 | 质量检查 |
| `correlators_rel_time.npz` | 相对时间关联函数 | 中间分析数据 |
| `ope_combined.npz` | 组合 OPE (-F_tx - F_ty + 2*F_xy) | 中间分析数据 |

## 依赖

- Python 3.8+
- numpy, scipy, matplotlib
- opt_einsum (可选，加速张量缩并)
- h5py (可选)
- snsc/main.py (本项目，用于 plaquette_clover)

## 复用关系

| 源 | 复用内容 |
|----|---------|
| `snsc/main.py` | plaquette_clover (Clover 场强张量) |
| `agent/snsc-v20260726/gamma_matrix.py` | DeGrand-Rossi γ 矩阵 |
| `agent/snsc-v20260726/utils.py` | 工具函数基类 |
| `examples/huangcl/code.py` | 原始 huangcl 分析代码 |
| `examples/donghx/` | 数据命名约定与文件格式 |
