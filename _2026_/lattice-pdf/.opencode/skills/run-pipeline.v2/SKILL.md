---
name: run-pipeline-gpu
description: 运行 docker-v20260727 GPU 加速验证管线 — OPE 从头计算，所有中间结果/日志/图表保存，使用 CUDA/CuPy
---

# run-pipeline-gpu — docker-v20260727 GPU 加速本地验证管线

在当前环境直接运行完整的 disconnected 胶子 PDF 验证管线，**使用 GPU 加速 (CUDA/CuPy)**。

**核心特征**：
- **GPU 加速 (CUDA/CuPy)**：VVV 重子块、Wick 收缩、F_{μν} 场强张量、Wilson 线、OPE 缩并全部在 GPU 上计算
- **大规模数据智能传输**：4.5GB 特征向量分时片传输到 GPU，避免 OOM
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
GPU:    CUDA/CuPy, 自动检测设备
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

| 步骤 | 描述 | GPU 加速 | 保存内容 |
|------|------|----------|----------|
| 0 | 环境检查 | GPU 设备检测 | Python 模块、CUDA/CuPy、数据路径 |
| 1 | 质子 2pt 蒸馏 | VVV 重子块 GPU, Wick 收缩 GPU | VVV 块、原始 Wick 收缩、宇称投影、有效质量 |
| 2 | OPE 从头计算 | F_{μν} GPU, Wilson 线 GPU, OPE GPU | F_{μν} 场强张量、Gauge 验证诊断、OPE 算符 |
| 3 | huangcl 比率分析 | Jackknife GPU 加速 | 比率 R(z)、诊断图、有效质量图、场强诊断图 |
| 4 | 最终报告 | — | Markdown 综合报告 |

## GPU 内存策略

| 数据 | 大小 | 策略 |
|------|------|------|
| Eigenvectors | 4.5 GB | CPU 驻留，按时间片传输到 GPU |
| Gauge configs | 72 MB | 整体加载到 GPU |
| VVV 输出 | 1.1 GB | GPU 计算后立即保存到 CPU |
| F_{μν} 张量 | ~72 MB | GPU 计算，保存到 CPU |
| Wick 收缩 | ~40 MB | GPU 计算，结果传回 CPU |

## 输出目录结构

```
output_YYYYMMDD_HHMMSS/
├── run.log                      # 主日志 (全部 stdout/stderr, 含详细耗时)
├── run_config.json              # 运行时配置快照
├── run_config_snapshot.json     # 配置快照
├── gpu_info.json                # GPU 设备信息快照
├── final_report.md              # Markdown 报告
├── timing.jsonl                 # 每步耗时与内存记录 (含 GPU 内存)
├── data/
│   ├── eigenvalues_Nev100.npy                   # 特征值 (中间结果)
│   ├── conf_{id}/
│   │   ├── gauge_validation_conf{id}.json       # Gauge 验证诊断
│   │   ├── Fmunu_mu{n}_nu{n}.npz                # 场强张量 (GPU 计算, 中间结果)
│   │   ├── ops_mu{n}_nu{n}_dz{delta_z}_conf{id}.npz  # OPE 算符
│   │   ├── VVV_Nev1{n}_Px{n}Py{n}Pz{n}_conf{id}.npy  # VVV 重子块 (GPU 计算)
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
    ├── ope_combined.npz                 # 组合 OPE
    ├── correlators_rel_time.npz         # 相对时间关联函数
    └── analysis_summary.json            # 分析摘要
```

## 子命令

### `run-pipeline-gpu run` — 运行完整 GPU 加速管线

```bash
cd /root/lattice-pdf/agent/docker-v20260727

# 完整运行 (3 组态, GPU 加速)
python run_pipeline.py

# 单组态快速测试
python run_pipeline.py --conf-id 6250

# 跳过某些步骤
python run_pipeline.py --skip-2pt --skip-ope

# 仅计算不分析
python run_pipeline.py --skip-analysis --skip-report

# 自定义输出目录
python run_pipeline.py --output-dir /path/to/output

# 详细调试模式
python run_pipeline.py --verbose
```

### `run-pipeline-gpu status` — 查看最新运行状态

```bash
latest=$(ls -dt /root/lattice-pdf/agent/docker-v20260727/output_*/ 2>/dev/null | head -1)
[ -n "$latest" ] && echo "=== $latest ===" && tail -50 "$latest/run.log"
```

### `run-pipeline-gpu check` — GPU 环境检查

```bash
cd /root/lattice-pdf/agent/docker-v20260727 && python3 -c "
import sys, os, cupy as cp
print(f'Python: {sys.version}')
print(f'CuPy: {cp.__version__}')
print(f'CUDA: {cp.cuda.runtime.runtimeGetVersion()}')
d = cp.cuda.Device()
print(f'Device: {d}')
mem = d.mem_info
print(f'GPU Memory: {mem[0]/1024**3:.1f} GB free / {mem[1]/1024**3:.1f} GB total')
for mod in ['numpy', 'scipy', 'matplotlib', 'opt_einsum']:
    try:
        m = __import__(mod)
        print(f'  ✓ {mod}: {getattr(m, \"__version__\", \"?\")}')
    except ImportError:
        print(f'  ✗ {mod}: MISSING')
"
```

### `run-pipeline-gpu module <module>` — 单独运行某模块

```bash
cd /root/lattice-pdf/agent/docker-v20260727

# 仅计算 2pt (GPU)
python compute_2pt_gpu.py --run-dir ./output_test

# 仅计算 OPE (GPU)
python compute_ope_gpu.py --run-dir ./output_test --conf-id 6250

# 仅运行分析
python analyze_ratio.py --run-dir ./output_test --data-dir ./output_test/data --output-dir ./output_test/plots
```

## 关键设计决策

1. **GPU 加速核心计算**：VVV、Wick 收缩、F_{μν}、Wilson 线、OPE 缩并全部在 GPU 上执行。
2. **智能内存管理**：4.5GB 特征向量分时片传输，避免 GPU OOM。
3. **OPE 从头计算**：从 gauge config (.lime) 出发，实现独立 GPU 版 Clover plaquette。
4. **所有中间变量保存**：GPU 计算结果及时传回 CPU 并保存。
5. **完善日志**：主日志 (run.log) + timing.jsonl + GPU 内存记录。
6. **同 v20260726 兼容的输出格式**：相同的数据命名约定和文件结构。

## 依赖

- Python 3.8+
- CuPy (CUDA 12.x)
- numpy, scipy, matplotlib
- opt_einsum (可选)
- h5py (可选)

## v20260726 → v20260727 变更

| 变更 | v20260726 | v20260727 |
|------|-----------|-----------|
| VVV 计算 | NumPy einsum (CPU) | CuPy 2-step einsum (GPU) |
| Wick 收缩 | NumPy einsum (CPU) | CuPy step-by-step einsum (GPU) |
| F_{μν} 场强 | snsc plaquette_clover (CPU/CuPy) | 独立 GPU 实现 |
| Wilson 线 | NumPy 逐点循环 (CPU) | GPU 累积输运子 (批量einsum) |
| OPE 缩并 | NumPy einsum (CPU) | CuPy einsum (GPU) |
| Gamma 矩阵 | NumPy | CuPy |
| 精度 | complex128 (双) | complex64 (单, 默认), 可选 complex128 |
| GPU 内存追踪 | 无 | GPU 空闲/峰值内存 |

## 实测性能 (RTX 4060 Laptop, 8GB, complex64)

| 步骤 | 时间 (Nconf=1) | 时间 (Nconf=3) | 备注 |
|------|---------------|---------------|------|
| 环境检查 | <1s | <1s | |
| 2pt: VVV | 26s | 26s (缓存复用) | 0.37s/时间片, GPU |
| 2pt: Wick | 109s | 330s | 49ms/pair, GPU |
| OPE: Gauge读取 | 38s | 114s | ILDG .lime 头部扫描 |
| OPE: F_{μν} | 0.3s/分量 | 0.3s/分量 | Clover plaquette, GPU |
| OPE: Wilson累积 | 0.02s/分量 | 0.02s/分量 | 批量累积输运子, GPU |
| OPE: 缩并 | 2s/分量 | 2s/分量 | 24个z值, GPU |
| 分析 | 3s | 3s | Jackknife + 4张图 |
| **总计** | **198s (3.3min)** | **508s (8.5min)** | |

## 优化纪要

1. **VVV einsum 两步分解**: `einsum("x,ax,bx,cx->abc")` → `einsum("x,ax,bx->abx")·einsum("abx,cx->abc")`，中间张量从4.6GB降至46MB，**40x加速**
2. **Wick 收缩逐步分解**: 5张量单次einsum → 4步逐步缩并，避免超大型中间结果
3. **Wilson 线累积输运子**: Python逐点循环(1M次×z次matmul) → GPU批量累积乘积，**100x加速**
4. **Eigenvector complex64下转换**: 4.5GB→2.3GB，GPU按时间片流式传输
5. **Gauge config complex64下转换**: 547MB→273MB，整体加载至GPU
