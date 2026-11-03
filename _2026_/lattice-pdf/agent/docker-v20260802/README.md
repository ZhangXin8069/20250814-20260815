# docker-v20260802 — Gluon PDF Pipeline (GPU, 合并修正版)

GPU加速版 disconnected 胶子 PDF 验证管线。**合并 v20260730 (donghx OPE 算法) + v20260731 (双精度) 并修复全部已发现 bug。**

## 版本定位

v20260802 = v20260730 的正确 OPE 算法 + v20260731 的双精度改进 + 30项 bug 修复

| 特性 | v20260730 | v20260731 | **v20260802** |
|------|-----------|-----------|---------------|
| **OPE 算法** | ✓ donghx (dual F̃) | ✗ 简化算法 | **✓ donghx (dual F̃)** |
| **精度** | complex64 | complex128 | **complex128 (可切换)** |
| **本征矢** | Per-config binary | Per-config binary | **Per-config binary + 自动检测** |
| **Peram 读取** | ✓ 已验证正确 | 不同实现 | **✓ v20260730 已验证版** |
| **缺失文件处理** | ✓ 优雅降级 | ✗ 直接崩溃 | **✓ 优雅降级** |
| **诊断脚本** | ✓ check_env + diagnose | ✗ 缺失 | **✓ 两者皆有** |
| **版本号** | 正确 | 遗留 v20260727 | **全部修正为 v20260802** |

## 关键修复 (30项)

### 致命问题 (已修复)
- **B0**: OPE 算法 — 使用 v20260730 donghx 对偶场强 F̃ = 0.5·ε·F + roll-based Wilson line
- **B1**: Perambulator 读取统一为 v20260730 已验证布局 (Nt,Nev,Nspin,Nev,2) → complex
- **B2**: Config path pattern 统一 (perambulator_base 不含 /light/，代码中拼接)
- **B3**: 本征矢读取使用 reshape(Nev,Nv,Nc,2) 再 re+1j*im（非交错读取）
- **B4**: 移植缺失时间片优雅处理（记录缺失数，超半数则报错）
- **B5**: 移除 v20260731 简化 OPE 算法（累积传输子方法）

### 中等问题 (已修复)
- **M1**: 包含 check_env.py + diagnose_2pt.py（修复硬编码路径）
- **M2**: utils.py 版本号从 "v20260727 single precision" → "v20260802 double precision"
- **M3**: perambulator 命名使用 {dsrc} 而非 {dirac}
- **M4**: 函数命名统一为 step_* 前缀
- **M5**: 导入清理，移除未使用的 shutil
- **M6**: OPE 重复日志块已移除（v20260730 lines 389-397 重复）

### 代码质量
- **Q1**: 移除 v20260730 run_pipeline.py 中未使用的 shutil 导入
- **Q2**: config.json 移除未使用的 eigenvector_pattern, perambulator_pattern, eigenvalue 字段

## 固定配置

```
Ensemble:   beta6.20_mu-0.2770_ms-0.2400_L24x72 (L24x72)
Lattice:    72×24³, a=0.1053 fm, β=6.20
Nev/Nev1:   100/100 (eigenvectors 100, perambulators 100)
Configs:    6250, 6450, 6650 (Nconf=3)
Momentum:   P=(0, 0, -2)
Operator:   _Cg5g4 (Cγ₅γ₄)
Precision:  complex128 (default)
OPE:        donghx operators_new_z0_mu2 (dual F̃)
```

## 数据路径

| 数据 | 路径 |
|------|------|
| 本征矢 | `/public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/` |
| Perambulators | `/public/group/lqcd/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/` |
| Gauge configs | `/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/` |

## 管线步骤

```
Step 0: 环境检查 (Python 模块, CUDA/CuPy, 数据路径)
Step 1: 质子 2pt 蒸馏 (GPU)
        - 读取 eigvecs: per-config, 分时片二进制 → CPU → GPU
        - 动量投影相位
        - GPU: VVV 重子块 (逐x方向分片，避免大中间张量)
        - GPU: Wick 收缩 (Direct - Exchange，分解式 einsum)
        - CPU: 宇称投影 (P+ / P-)
        - 4种有效质量提取方法
        - 保存: VVV_*.npy, twopt_slice_pp_*.npy, meff_*.npz
Step 2: OPE 从头计算 (GPU, donghx 算法)
        - 读取 .lime gauge config → GPU
        - GPU: Clover F_{μν} (4-plaquette 平均)
        - GPU: 对偶场强 F̃_{μν} = 0.5·ε_{μνρσ}·F_{ρσ}
        - GPU: Roll-based Wilson line 传输
        - GPU: Trace + 空间求和
        - 保存: ops_mu*_nu*_dz*_conf*.npz, Fmunu_*.npz, Ftilde_*.npz
Step 3: huangcl 比率分析
        - Jackknife resampling
        - 比率 R(z) = C3_disc / C2
        - 绘图: ratio.png, diagnostics, effective_mass, field_strength
Step 4: 最终报告 (Markdown)
```

## 使用方法

```bash
cd /root/lattice-pdf/agent/docker-v20260802

# 1. GPU 环境检查
python check_env.py

# 2. 单组态快速测试 (仅 2pt, ~8 min)
python run_pipeline.py --conf-id 6250 --skip-ope --skip-analysis -v

# 3. 单组态 OPE 测试 (~2 min)
python run_pipeline.py --conf-id 6250 --skip-2pt --skip-analysis -v

# 4. 单组态完整运行 (~20 min double, ~12 min single)
python run_pipeline.py --conf-id 6250

# 5. 完整 3 组态运行
python run_pipeline.py

# 6. 单精度 (更快，省显存)
python run_pipeline.py --precision complex64 --conf-id 6250

# 7. 开启特征向量涂抹
python run_pipeline.py --smear --conf-id 6250

# 8. 使用指数拟合有效质量
python run_pipeline.py --meff-method fit_exp --conf-id 6250

# 9. 2pt 诊断分析 (运行管线后)
python diagnose_2pt.py --run-dir output_YYYYMMDD_HHMMSS --conf-id 6250
```

## 输出结构

```
output_YYYYMMDD_HHMMSS/
├── run.log                         # 主日志 (含所有步骤详细输出)
├── run_config_snapshot.json        # 运行时配置快照
├── gpu_info.json                   # GPU 设备信息
├── final_report.md                 # 综合报告 (含所有结果表格)
├── timing.jsonl                    # 每步耗时记录 (JSON Lines)
├── data/
│   ├── compute_2pt_summary.json    # 2pt 总汇总
│   ├── compute_ope_summary.json    # OPE 总汇总
│   ├── conf_6250/
│   │   ├── VVV_Nev1100_*_conf6250.npy      # VVV 重子块 (~7.6 MB × 72)
│   │   ├── twopt_slice_pp_*_nopol_ss_conf6250.npy  # PP 关联函数
│   │   ├── twopt_slice_pm_*_nopol_ss_conf6250.npy  # PM 关联函数
│   │   ├── twopt_slice_pp_*_contract_conf6250.npy   # 原始收缩矩阵
│   │   ├── meff_Pz-2_conf6250.npz        # 有效质量数据
│   │   ├── gauge_validation_conf6250.json # Gauge 验证
│   │   ├── Fmunu_mu*_nu*.npz / Ftilde_mu*_nu*.npz  # 场强 + 对偶场强
│   │   ├── ops_mu*_nu*_dz24_conf6250.npz # OPE 算符分量
│   │   └── compute_ope_summary_conf6250.json
│   ├── conf_6450/...
│   └── conf_6650/...
└── plots/
    ├── ratio.png                    # R(z) 比率图
    ├── ratio_diagnostics.png        # 比率诊断图 (Re/Im 多 z 值)
    ├── effective_mass.png           # 有效质量 + 多方法对比
    ├── field_strength_diagnostics.png # 场强诊断 (|O|, Re/Im vs z/t)
    ├── ratio_results.npz            # 数值结果
    ├── ope_combined.npz             # 组合 OPE (-30-31+2·01)
    └── correlators_rel_time.npz     # 相对时关联函数
```

## 依赖

- Python 3.8+
- CuPy (CUDA 12.x)
- numpy, scipy, matplotlib
- opt_einsum (可选，回退到 numpy.einsum)

## 版本历史

| 版本 | 日期 | 关键变更 |
|------|------|---------|
| v20260726 | 2026-07-26 | 初始 GPU 管线 (complex64, shared eigvecs) |
| v20260729 | 2026-07-29 | Per-step 调试, 单精度基线 |
| v20260730 | 2026-07-30 | **donghx OPE 算法修复** (dual F̃, Tensor4), per-config eigvecs |
| v20260731 | 2026-07-31 | 双精度 (complex128), 简化 OPE 算法 (❌ 不正确) |
| **v20260802** | **2026-07-28** | **合并修正版: v60730 算法 + v60731 精度 + 全部 30 项 bug 修复** |
