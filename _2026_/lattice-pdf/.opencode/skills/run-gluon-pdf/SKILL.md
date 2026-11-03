---
name: run-gluon-pdf
description: 运行 gluon_pdf_full_workflow.py — 从格点规范组态到光锥 PDF 的全流程计算
---

# run-gluon-pdf — 胶子 PDF 全流程计算

运行 `examples/zhangxin/gluon_pdf_full_workflow.py`，实现从格点规范组态到光锥胶子 PDF 的完整 10 步计算管线。

## 理论框架

**LaMET (Large Momentum Effective Theory)** — 通过准 PDF 和微扰匹配连接 Euclidean 格点关联函数到光锥 PDF。

胶子 PDF 涉及 **非连通图 (disconnected diagrams)**。三点关联函数分解为：
1. **质子 2pt 关联函数** — 动量涂抹蒸馏
2. **OPE 部分** — 非定域胶子算符

计算的矩阵元对应内部笔记 Eq(20)；OPE 分解对应 Eq(25)。

## 10 步计算管线

| 步骤 | 描述 | 核心操作 |
|------|------|----------|
| 1 | 读取规范组态 | 加载规范链接 $U_\mu(x)$ |
| 2 | 构造场强张量 | Clover plaquette → $F_{\mu\nu}$ |
| 3 | 构造对偶场强张量 | Levi-Civita 缩并 → $\tilde{F}_{\mu\nu}$ |
| 4 | 构造非定域胶子 OPE 算符 | $O_{\mu\nu}(z)$ 含 Wilson 线 |
| 5 | 蒸馏框架 | 本征矢 + perambulator + VVV |
| 6 | 质子 2pt 关联函数 | 动量涂抹 |
| 7 | 矩阵元提取 | $h(z, P_z)$ |
| 8 | 傅里叶变换 | → 准 PDF $\tilde{g}(x, P_z)$ |
| 9 | 微扰匹配 | → 光锥 PDF $g(x, \mu)$ |
| 10 | Jackknife 误差分析 | 统计不确定性 |

## 子命令

### `run-gluon-pdf all`
**运行完整 10 步管线**（使用默认参数）

```bash
cd /root/lattice-pdf && python examples/zhangxin/gluon_pdf_full_workflow.py
```

### `run-gluon-pdf <步骤名>`
**从指定步骤开始运行**

支持的步骤名：
- `field-strength` — 从步骤 2 开始（场强张量构造）
- `ope` — 从步骤 4 开始（OPE 算符构造）
- `distillation` — 从步骤 5 开始（蒸馏框架）
- `correlator` — 从步骤 6 开始（关联函数计算）
- `matrix-element` — 从步骤 7 开始（矩阵元提取）
- `fourier` — 从步骤 8 开始（傅里叶变换）
- `matching` — 从步骤 9 开始（微扰匹配）
- `jackknife` — 仅步骤 10（误差分析）

```bash
cd /root/lattice-pdf && python examples/zhangxin/gluon_pdf_full_workflow.py --start <step_name>
```

### `run-gluon-pdf custom`
**使用自定义参数运行**

常用参数：

```bash
cd /root/lattice-pdf && python examples/zhangxin/gluon_pdf_full_workflow.py \
  --conf 20000 \           # 规范组态数量
  --Pz 6 \                 # 质子动量 (2π/L 单位)
  --delta_z 15 \           # Wilson 线长度
  --gauge_file <file> \    # 规范组态文件路径
  --eig_dir <dir> \        # 本征矢目录
  --peram_dir <dir> \      # perambulator 目录
  --output_dir <dir>       # 输出目录
```

### `run-gluon-pdf check`
**检查环境和依赖**

```bash
cd /root/lattice-pdf && python -c "
import sys
print(f'Python: {sys.version}')

# 检查 numpy
try:
    import numpy as np
    print(f'numpy: {np.__version__}')
except ImportError:
    print('numpy: MISSING')

# 检查 cupy (GPU)
try:
    import cupy as cp
    print(f'cupy: {cp.__version__}  (GPU count: {cp.cuda.runtime.getDeviceCount()})')
except ImportError:
    print('cupy: NOT AVAILABLE (CPU numpy fallback)')

# 检查 opt_einsum
try:
    import opt_einsum
    print(f'opt_einsum: available')
except ImportError:
    print('opt_einsum: NOT AVAILABLE (numpy.einsum fallback)')

# 检查代码文件
import os
wf = 'examples/zhangxin/gluon_pdf_full_workflow.py'
if os.path.exists(wf):
    size = os.path.getsize(wf)
    print(f'{wf}: {size:,} bytes')
else:
    print(f'{wf}: MISSING')
"
```

### `run-gluon-pdf help`
**查看代码支持的完整参数列表**

```bash
cd /root/lattice-pdf && python examples/zhangxin/gluon_pdf_full_workflow.py --help 2>&1 || head -50 examples/zhangxin/gluon_pdf_full_workflow.py
```

## 相关代码

| 文件 | 说明 |
|------|------|
| `examples/zhangxin/gluon_pdf_full_workflow.py` | 主工作流（~1900 行） |
| `examples/2pt_proton_Cg5gmu_L32x64_mom2_xdir_gpu.py` | 原始质子 2pt 代码（donghx） |
| `examples/Calc_ope_unpol.py` | 原始 OPE 代码（MPI 并行） |
| `examples/Operator.py` | CuPy 算符工具（plaquette, Fμν） |
| `examples/gamma_matrix_cupy_DR.py` | DeGrand-Rossi 基 γ 矩阵 |
| `examples/input_output_4_cupy.py` | 输入输出工具 |

## 输出文件

运行后在 `--output_dir`（默认 `./results/`）生成：

| 文件 | 内容 |
|------|------|
| `quasi_pdf_*.npy` | 准 PDF $\tilde{g}(x, P_z)$ |
| `lightcone_pdf_*.npy` | 光锥 PDF $g(x, \mu)$ |
| `matrix_element_*.npy` | 矩阵元 $h(z, P_z)$ |
| `jackknife_errors_*.npy` | Jackknife 误差估计 |

## 环境要求

- Python 3.8+
- numpy
- cupy (GPU 加速，可选；无 GPU 时自动回退到 numpy)
- opt_einsum (张量缩并优化，可选；无安装时回退到 numpy.einsum)
- HPC 集群环境（数据文件在集群存储上）

## 使用示例

```
/run-gluon-pdf                        # 查看选项
/run-gluon-pdf all                    # 运行完整管线
/run-gluon-pdf field-strength         # 从场强张量开始
/run-gluon-pdf custom --Pz 8 --conf 50000  # 自定义参数
/run-gluon-pdf check                  # 检查环境
/run-gluon-pdf help                   # 查看参数帮助
```
