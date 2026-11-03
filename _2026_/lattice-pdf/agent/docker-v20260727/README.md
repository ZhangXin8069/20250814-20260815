# GPU 加速版 — docker-v20260727

整合版验证管线，使用真实格点数据复现 huangcl 的 disconnected 胶子 PDF 比率分析。

**与 docker-v20260726 的关键区别**：
1. **GPU 加速 (CUDA/CuPy)**：VVV、Wick 收缩、F_{μν}、Wilson 线、OPE 缩并全部在 GPU 上计算
2. **独立 GPU 版 plaquette_clover**：不依赖 snsc/main.py，自主实现 CuPy 版本
3. **智能 CPU↔GPU 数据传输**：4.5GB 特征向量按时间片传输，避免 GPU OOM
4. **GPU 内存追踪**：实时记录 GPU 空闲/峰值内存
5. **所有中间变量保存**（同 v20260726）
6. **完善日志系统**（同 v20260726）
7. **同 v20260726 兼容的输出格式**

## 快速开始

```bash
# 1. GPU 环境检查
cd /root/lattice-pdf/agent/docker-v20260727
python3 -c "
import cupy as cp
d = cp.cuda.Device()
mem = d.mem_info
print(f'GPU: {d}')
print(f'Memory: {mem[0]/1024**3:.1f} GB free / {mem[1]/1024**3:.1f} GB total')
"

# 2. 单组态测试 (快速验证 GPU 管线)
python run_pipeline.py --conf-id 6250

# 3. 完整运行 (3 组态)
python run_pipeline.py

# 4. 跳过某些步骤
python run_pipeline.py --skip-2pt --skip-ope --skip-analysis
```

## GPU 加速详情

| 计算步骤 | CPU (v20260726) | GPU (v20260727) | 预计加速比 |
|----------|-----------------|-----------------|-----------|
| VVV 重子块 | NumPy einsum over Nx³=13824 sites | CuPy einsum on GPU | 10-50x |
| Wick 收缩 | NumPy einsum over Nev³=10⁶ | CuPy einsum on GPU | 20-100x |
| Clover F_{μν} | snsc plaquette_clover | 独立 CuPy 实现 | 5-20x |
| Wilson 线 | NumPy 逐点循环 | CuPy 批量矩阵乘法 | 10-30x |
| OPE 缩并 | NumPy einsum | CuPy einsum | 5-15x |

## 管线步骤

```
Step 0: 环境检查 (Python 模块, CUDA/CuPy, 数据路径)
Step 1: 质子 2pt 蒸馏 (GPU)
        - 读取 eigvecs (CPU) → 分时片传 GPU
        - GPU: 动量涂抹相位, VVV 重子块
        - GPU: Wick 收缩 (Direct - Exchange)
        - CPU: 宇称投影 (P+ / P-)
        - 保存: conf_{id}/twopt_slice_pp_*.npy, VVV_*.npy, meff_*.npz
Step 2: OPE 从头计算 (GPU) ★
        - 读取规范组态 (CPU) → 传 GPU
        - GPU: 验证规范组态
        - GPU: Clover F_{μν} → 保存为中间结果
        - GPU: Wilson 线构造
        - GPU: 非定域 OPE 算符缩并
        - 保存: ops_mu{mu}_nu{nu}_dz{delta_z}_conf{id}.npz
Step 3: huangcl 比率分析
        - 加载 2pt + OPE → 构建 3pt disconnected
        - Jackknife 重采样
        - 计算 R(z) = C3_disc / C2
        - 画图
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
| Perambulators | `/public/group/lqcd/sunpeng/mom_smear_perambulators/.../mz2_my0_mx0/{conf_id}/` |
| Gauge configs | `/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/` |

## 依赖

- Python 3.8+
- CuPy (CUDA 12.x)
- numpy, scipy, matplotlib
- opt_einsum (可选，GPU 上使用 CuPy einsum)

## 复用关系

| 源 | 复用内容 |
|----|---------|
| `agent/docker-v20260726/` | 管线架构、分析代码、数据命名约定 |
| `examples/donghx/gamma_matrix_cupy_DR.py` | DeGrand-Rossi γ 矩阵 (转为独立 CuPy 版) |
| `snsc/main.py` | plaquette_clover 算法 (独立 GPU 实现) |
| `examples/huangcl/code.py` | huangcl 分析逻辑 |
