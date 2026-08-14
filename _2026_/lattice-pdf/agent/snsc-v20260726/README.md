# SNSC Validation Pipeline — v20260726

整合版验证管线，使用真实格点数据复现 huangcl 的 disconnected 胶子 PDF 比率分析。

**与 agent/snsc 的关键区别**：不调用 LQCD_Master（AI planner）和 lamet-agent（agentic 分析），直接运行物理计算。

## 快速开始

```bash
# 1. 直接运行 (交互式节点)
cd /root/lattice-pdf/agent/snsc-v20260726
python run_pipeline.py

# 2. Slurm 提交 (集群)
cd /root/lattice-pdf/agent/snsc-v20260726
sbatch sbatch.sh

# 3. 只运行分析 (已有 2pt + OPE 数据)
python run_pipeline.py --skip-2pt --skip-ope

# 4. 只计算 2pt + OPE (不画图)
python run_pipeline.py --skip-analysis --skip-report
```

## 数据下载

管线运行前需从 SNSC 集群下载所需数据：

```bash
cd /root/lattice-pdf/agent/snsc-v20260726

# 预览要下载的内容（不实际传输）
bash download_data.sh --dry-run

# 交互式下载（逐个文件确认）
bash download_data.sh

# 一键下载全部（跳过确认）
bash download_data.sh --yes

# 查看帮助
bash download_data.sh --help
```

**下载内容**：

| 步骤 | 数据 | 组态 |
|------|------|------|
| 1 | Eigenvectors + Eigenvalues (`cfg_48000`, 所有组态共用) | — |
| 2 | Perambulators 目录 (`mom_smear=-2`, `mz2_my0_mx0`) | 6250, 6450, 6650 |
| 3 | Gauge 组态 `.lime` 文件 | 6250, 6450, 6650 |

**传输方式**：`rsync -avP` over SSH（支持断点续传、进度显示、增量同步）。本地保存路径与原文件路径一致（`/public/group/lqcd/...`），无需修改 `run_config.json`。

**SSH 连接**：`ssh 222.200.137.16 -p 10023 -l zhangxin`

## 管线步骤

```
Step 0: 环境检查 (Python 模块, 数据路径可访问性)
Step 1: 质子 2pt 蒸馏计算
        - 读取 eigvecs (.npy), perambulators (二进制)
        - 计算 VVV baryon block (6-term ε 收缩)
        - Wick 收缩 (Direct - Exchange)
        - 宇称投影 (P+ / P-)
        - 保存: conf_{id}/twopt_slice_pp_*.npy
Step 2: OPE 从头计算并保存
        - 读取规范组态 (.lime ILDG 格式)
        - 计算场强张量 F_{μν} (Clover plaquette)
        - 构造非定域 OPE 算符: Tr[F(z)·W(z→0)·F(0)·W(0→z)]
        - 保存为统一格式: ops_mu{mu}_nu{nu}_dz{delta_z}_conf{id}.npz
Step 3: huangcl 比率分析
        - 加载 2pt + OPE → 构建 3pt disconnected
        - Jackknife 重采样
        - 计算 R(z) = C3_disc / C2
        - 画图: ratio.png, diagnostics.png, effective_mass.png
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
| OPE (从头计算) | 从 gauge configs 计算 F_{μν} → 非定域 OPE 算符 → 保存为 .npz |

## 不明确问题（已确认）

1. **Eigenvector cfg 编号不匹配**：✓ 确认复用。特征向量 cfg_48000 用于所有组态（蒸馏标准做法）。
2. **Perambulator 完整格式**：✓ 确认。每个 conf_id 目录下有完整的 `perams.{conf_id}.{d_src}.{t_src}` 文件 (d_src=0..3, t_src=0..71)。
3. **Gauge 组态路径**：已指定 `.lime` 格式文件。**OPE 从头计算**：读取 gauge config → Clover plaquette → F_{μν} → 非定域 OPE 算符 (含 Wilson 线) → 保存为 .npz。若 gauge config 缺失则跳过该组态。
4. **动量 Pz**：仅计算 Pz=-2。
5. **输出目录**：时间戳格式 `output_YYYYMMDD_HHMMSS/`。

## 输出目录结构

```
output_YYYYMMDD_HHMMSS/
├── run.log                   # 主日志 (全部 stdout/stderr, 含详细耗时)
├── run_config.json           # 运行时配置快照
├── final_report.md           # Markdown 报告
├── timing.jsonl              # 每步耗时与内存记录
├── data/
│   ├── conf_6250/
│   │   ├── VVV_Nev1100_Px0Py0Pz-2_conf6250.npy     # VVV 缓存
│   │   ├── twopt_slice_pp_..._conf6250.npy           # 质子 2pt (PP)
│   │   ├── twopt_slice_pm_..._conf6250.npy           # 质子 2pt (PM)
│   │   ├── ops_mu0_nu1_dz24_conf6250.npz             # OPE F_xy
│   │   ├── ops_mu3_nu0_dz24_conf6250.npz             # OPE F_tx
│   │   ├── ops_mu3_nu1_dz24_conf6250.npz             # OPE F_ty
│   │   └── meff_Pz-2_conf6250.npz                    # 有效质量
│   ├── conf_6450/...
│   └── conf_6650/...
└── plots/
    ├── ratio.png             # 比率 R(z), Pz=-2, z=2, 多 tsep
    ├── ratio_diagnostics.png # 诊断: 多 z 值的 Re/Im
    └── effective_mass.png    # 有效质量 (cosh) + 平台拟合
```

## 依赖

- Python 3.8+
- numpy, scipy, matplotlib
- h5py (可选, HDF5 输出)
- conda env: `zhangxin-snsc` (集群) 或系统 Python

## 复用关系

| 源 | 复用内容 |
|----|---------|
| `agent/snsc/runs/.../analyze_ratio.py` | huangcl 比率分析算法 (Jackknife, R(z)) |
| `agent/snsc/runs/.../gamma_matrix.py` | DeGrand-Rossi γ 矩阵 (完整复制) |
| `agent/snsc/runs/.../run_pipeline.py` | 主编排框架 (run_step, log, Timer) |
| `snsc/main.py` | 算法参考 (VVV, Wick, meff 实现) |
| `examples/huangcl/code.py` | 原始 huangcl 分析代码 |
| `examples/donghx/` | 数据命名约定与文件格式 |
