# GPU 加速版 (CORRECTED) — docker-tv20260729

整合版验证管线，使用真实格点数据复现 huangcl 的 disconnected 胶子 PDF 比率分析。

**与 docker-v20260727 的关键区别 (FIXES)**：

1. **特征向量涂抹现在是可选的** (默认: OFF)。
   当 perambulator 已经编码了动量涂抹 (如 `mz2_my0_mx0`)，
   VVV 中的特征向量**不应**额外涂抹。添加了 `--smear` 选项来启用。

2. **有效质量使用拟合方法** (fit_cosh/fit_exp) 而非朴素的 arccosh。
   arccosh 公式在振荡/噪声关联函数上会灾难性地失败。
   支持多种方法: `fit_cosh`, `fit_exp`, `exp_forward`, `cosh`。

3. **多种 meff 方法同时计算并对比报告**，用于诊断。

4. **基于物理的动量投影**: VVV 相位使用 `P_proj = P_phys`，
   因为 perambulator 的涂抹在 Wick 缩并中与 VVV 涂抹相互抵消。

## 快速开始

```bash
# 1. GPU 环境检查
cd /root/lattice-pdf/agent/docker-tv20260729
python3 check_env.py

# 2. 单组态测试 (快速验证)
python run_pipeline.py --conf-id 6250 --skip-ope --skip-analysis -v

# 3. 仅 2pt (默认设置: 无涂抹, fit_cosh)
python run_pipeline.py --conf-id 6250 --skip-ope --skip-analysis

# 4. 启用特征向量涂抹 (匹配 v20260727)
python run_pipeline.py --conf-id 6250 --smear --skip-ope --skip-analysis

# 5. 使用不同有效质量方法
python run_pipeline.py --conf-id 6250 --meff-method fit_exp --skip-ope --skip-analysis

# 6. 完整运行 (3 组态)
python run_pipeline.py

# 7. 双精度
python run_pipeline.py --conf-id 6250 --precision complex128 --skip-ope --skip-analysis
```

## 管线步骤

```
Step 0: 环境检查 (Python 模块, CUDA/CuPy, 数据路径)
Step 1: 质子 2pt 蒸馏 (GPU, CORRECTED)
        - 读取 eigvecs (CPU) → 分时片传 GPU
        - 可选特征向量涂抹 (默认 OFF)
        - GPU: 动量投影相位, VVV 重子块
        - GPU: Wick 收缩 (Direct - Exchange)
        - CPU: 宇称投影 (P+ / P-)
        - 多种有效质量提取方法
        - 保存: conf_{id}/twopt_slice_pp_*.npy, VVV_*.npy, meff_*.npz
Step 2: OPE 从头计算 (GPU)
        - (同 v20260727)
Step 3: huangcl 比率分析
        - (同 v20260727, 改进的有效质量图)
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
| **特征向量涂抹** | **OFF (默认, corrected)** |
| **有效质量方法** | **fit_cosh (默认, corrected)** |

## 有效质量方法对比

| 方法 | 描述 | 适用场景 |
|------|------|---------|
| `fit_cosh` | 对源平均 |C(Δt)| 进行 cosh 拟合 | 最鲁棒, 处理噪声关联函数 |
| `fit_exp` | 对 forward-only |C(Δt)| 进行单指数拟合 | 适合干净的前向传播信号 |
| `exp_forward` | log(C(t)/C(t+1)) 逐点提取 | 需要干净的指数衰减, 用于诊断 |
| `cosh` | 标准 arccosh 公式 (v20260727 的方法) | 仅适合无振荡的干净信号 |

## 数据源

| 数据 | 路径 |
|------|------|
| Eigenvectors | `/public/group/lqcd/sunpeng/eigen_vector/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_48000.eigenvector.npy` |
| Eigenvalues | `/public/group/lqcd/sunpeng/eigen_vector/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_48000.eigenvalue.npy` |
| Perambulators | `/public/group/lqcd/sunpeng/mom_smear_perambulators/.../mz2_my0_mx0/{conf_id}/` |
| Gauge configs | `/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/` |

## 依赖

- Python 3.8+
- CuPy (CUDA 12.x)
- numpy, scipy, matplotlib
- opt_einsum (可选)

## 预期结果

对于 Pz=-2 的质子 (动量为 2 单位 2π/L):
- 有效质量应约为 **1.2-1.4 GeV** (基态核子能量 E = √(m² + p²))
- 前向关联函数应呈指数衰减
