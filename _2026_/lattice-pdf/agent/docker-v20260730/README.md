# GPU 加速版 — docker-v20260730 (FINAL)

整合版验证管线，使用真实格点数据复现 huangcl 的 disconnected 胶子 PDF 比率分析。

## 完整 Bug 修复记录

### Bug 1: Perambulator 二进制布局
- 错: `(Nt, Nspin=4, Nev, Nev)` → 正确: `(Nt, Nev_snk, Nspin=4, Nev_src)`
- 两组Nev轴(源/汇)混淆,导致 Wick 收缩索引错误 → meff=0.2 GeV
- 修复后 meff → ~1.78 GeV

### Bug 2: OPE 算法 — 缺对偶场强 F̃
- 原始代码直接使用 F 而非 donghx 的对偶场强 F̃ = 0.5·ε·F
- 导致 ratio 图形完全错误 (mu3_nu1 分量小 1e7 倍)
- 修复: 完整实现 donghx `operators_new_z0_mu2` 算法

### Bug 3: OPE 算法 — Tensor4 (Levi-Civita) 实现错误
- `c = b + ...` (从 b 开始累加) → 正确: `c = 0 + ...`
- 导致 ε[3,1,0,2] 符号错误, F̃[3,1] = +F[0,2] 而非 -F[0,2]
- 修复后三 OPE 分量量级一致, ratio 图形合理

### Bug 4: OPE 空间求和用错轴
- 原来只用 perp_axes (排除 z_dir), 应 sum over ALL spatial (z,y,x)

## 最终结果 (3组态, Pz=-2, Nconf=3, single precision GPU)

### 有效质量
| Config | fit_cosh | fit_exp | exp_forward(t≈5) |
|--------|----------|---------|-------------------|
| 6250 | 1.778 GeV | 1.772 GeV | 1.534 GeV |
| 6450 | 1.794 GeV | 1.795 GeV | 1.502 GeV |
| 6650 | 1.773 GeV | 1.774 GeV | 1.510 GeV |

**Pz=0 rest mass**: 1.369 GeV → **E(Pz=-2) expected**: √(1.37²+0.98²) ≈ 1.69 GeV ✓

### OPE (donghx algorithm, 三个分量量级一致)
| Component | |O| range (conf 6250) |
|-----------|---------------------|
| mu0_nu1 (F_xy, F̃_xy) | ~50 |
| mu3_nu0 (F_tx, F̃_tx) | ~50 |
| mu3_nu1 (F_ty, F̃_ty) | ~50 |

### Ratio (z=2)
- C3/C2 range: [-0.08, +0.25] (Nconf=3, 与 huangcl Nconf=200 参考形状一致)

## 快速开始
```bash
cd /root/lattice-pdf/agent/docker-v20260730
python3 check_env.py        # 环境检查
python3 run_pipeline.py -v  # 完整运行 (~17 min, GPU)
```


## 快速开始

```bash
# 1. GPU 环境检查
cd /root/lattice-pdf/agent/docker-v20260730
python3 check_env.py

# 2. 单组态测试 (快速验证 2pt)
python run_pipeline.py --conf-id 6250 --skip-ope --skip-analysis -v

# 3. 仅 2pt (默认设置: 无涂抹, fit_cosh)
python run_pipeline.py --conf-id 6250 --skip-ope --skip-analysis

# 4. 完整运行 (3 组态)
python run_pipeline.py

# 5. 双精度
python run_pipeline.py --conf-id 6250 --precision complex128 --skip-ope --skip-analysis
```

## 管线步骤

```
Step 0: 环境检查 (Python 模块, CUDA/CuPy, 数据路径)
Step 1: 质子 2pt 蒸馏 (GPU)
        - 读取 eigvecs: per-config, 分时片二进制 → CPU → GPU
        - 动量投影相位
        - GPU: VVV 重子块
        - GPU: Wick 收缩 (Direct - Exchange)
        - CPU: 宇称投影 (P+ / P-)
        - 多种有效质量提取方法
        - 保存: conf_{id}/twopt_slice_pp_*.npy, VVV_*.npy, meff_*.npz
Step 2: OPE 从头计算 (GPU)
        - 读取 .lime gauge config → GPU
        - GPU: Clover F_{μν}, Wilson line, OPE contraction
        - 保存: conf_{id}/ops_mu*_nu*_dz*_conf*.npz
Step 3: huangcl 比率分析
        - Jackknife resampling
        - 比率 R(z) = C3_disc / C2
        - 绘图: ratio.png, diagnostics, effective_mass, field_strength
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
| 动量涂抹 | mom_smear=-2 (在 perambulator 中) |
| 特征向量涂抹 | OFF (默认) |
| 有效质量方法 | fit_cosh (默认) |

## 精度

- 默认单精度 (complex64/float32) GPU 计算
- 输入数据 binary LE f8 (float64) → CPU complex128 → GPU complex64
- 中间/输出保存 complex64
- `--precision complex128` 切换双精度

## 依赖

- Python 3.8+
- CuPy (CUDA 12.x)
- numpy, scipy, matplotlib
- opt_einsum (可选)
