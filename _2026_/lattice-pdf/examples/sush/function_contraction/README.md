# function_contraction 包使用说明

格点 QCD 蒸馏（distillation）方法下的关联函数计算与统计分析工具包。

## 导入方式

```python
from function_contraction import *
```

共 14 个模块，以下按功能分类说明。

---

## 1. Wick 收缩（`corr_wick`）

### `wick_contraction` — 自动 Wick 收缩

输入 sink、source、current 三组算符（以 `'|'` 分隔强子），自动穷举所有可能的 Wick 缩并图，返回每个图的缩并指标、权重符号、peram 条目等。

**示例 — π⁺ 介子 2pt 函数：**

```python
from function_contraction import wick_contraction

result = wick_contraction(
    sink_operators   = ["|", "u^d", "gamma_5", "d", "|"],
    source_operators = ["|", "u^d", "gamma_5", "d", "|"],
    curr_operators   = [],
    Cpt='2pt'
)
```

**返回结果说明：**
- `result['peram']` — 每个缩并图的 peram 条目列表，每项为 `[夸克位置, 反夸克位置, 味组合, 缩并标签, 时间标签]`
- `result['result_sign']` — 每个图的 Fermi 符号（+1 或 −1）
- `result['result_indx']` — 缩并指标字符串，形如 `"ab,cd->..."`（大写字母为 VVV/VDV/gamma 指标，小写为自由夸克指标）
- `result['V']` — VVV/VDV 顶点信息
- `result['gamma_pos']` — γ 矩阵插入位置

**支持通配符 `q`（六味）、`l`（轻味 u/d）：**

```python
# 同时计算 π⁺(ūd) 和 K⁺(ūs)
result = wick_contraction(
    sink_operators   = ["|", "q^d", "gamma_5", "q", "|"],
    source_operators = ["|", "q^d", "gamma_5", "q", "|"],
    curr_operators   = [],
    Cpt='2pt'
)
# 返回 list，长度等于合法味替换数
```

**示例 — 核子 2pt 函数：**

```python
result = wick_contraction(
    sink_operators   = ["|", "u", "C*gamma_5", "d", "u", "|"],
    source_operators = ["|", "u", "C*gamma_5", "d", "u", "|"],
    curr_operators   = [],
    Cpt='2pt'
)
```

**示例 — K→π 3pt 函数：**

```python
result = wick_contraction(
    sink_operators   = ["|", "u^d", "gamma_5", "s", "|"],
    source_operators = ["|", "u^d", "gamma_5", "d", "|"],
    curr_operators   = ["|", "s^d", "gamma_mu", "d", "|"],
    Cpt='3pt'
)
```

---

### `plot_figure_wick` — 绘制 Wick 缩并图

将 `wick_contraction` 的结果可视化为夸克线缩并图。

```python
from function_contraction import wick_contraction, plot_figure_wick

result = wick_contraction(
    sink_operators   = ["|", "u^d", "gamma_5", "d", "|"],
    source_operators = ["|", "u^d", "gamma_5", "d", "|"],
    curr_operators   = [],
    Cpt='2pt'
)

fig, ax = plot_figure_wick(result, diagram_index=0, Cpt='2pt', plot_text=True)
fig.savefig("wick_pion_2pt.pdf")
```

图像包含：夸克节点（带味颜色）、传播子箭头、γ 矩阵虚线、VVV/VDV 顶点标签、时间方向箭头、缩并指标与符号信息。

---

### `identify_equivalent_diagrams` — 识别等价缩并图

输入多个 `wick_contraction` 返回的 dict，通过并查集找出由 γ 矩阵位置交换连通的等价图。

```python
from function_contraction import identify_equivalent_diagrams

groups = identify_equivalent_diagrams(result1, result2, result3)
# 返回: [[0, 2], [1]]  → 第0和第2个 dict 的图等价，第1个独立
```

---

### `contraction_index` — 生成缩并字母指标

为张量缩并生成类似爱因斯坦求和约定的字母指标。用于追踪多数组缩并时的维度对应关系。

```python
from function_contraction import contraction_index

i1, i2, combined, new_shapes = contraction_index(
    contraction_same_indx = [[0], [0]],
    dims  = [2, 2],
    shapes = [(4, 5), (5, 6)],
    name  = [["spin", "color"], ["color", "flavor"]]
)
# i1 = "AB", i2 = "AC", combined = "ABC"
# new_shapes = [4, 5, 6]
```

---

## 2. Peram 变换（`corr_contraction`）

### `seq_peram` — 时序传播子变换

对 perambulator 施加 γ₅ 时序变换：τ → γ₅ · τ† · γ₅

```python
from function_contraction import seq_peram
import numpy as np

# peram 形状: (t_sink, d_sink, d_src, Nev_sink, Nev_src)
peram = np.random.randn(64, 4, 4, 32, 32) + 1j * np.random.randn(64, 4, 4, 32, 32)
peram_seq = seq_peram(peram)
# 输出形状: (t_sink, d_src, d_sink, Nev_src, Nev_sink)
```

---

## 3. 本征矢运算（`corr_eigvecs` 类）

### 初始化

```python
from function_contraction import corr_eigvecs

ev = corr_eigvecs(Nx=32, backend='numpy', Nc=3)
```

`Nx` 为空间格点数，`backend` 可选 `'numpy'` 或 `'cupy'`。

---

### `check` — 检验本征矢正交归一性

```python
eigvecs = np.load("eigvecs.npy")  # 形状 (Nev, Nz, Ny, Nx, Nc)
status = ev.check(eigvecs, dtype='print', tol=1e-10)
# 输出: "orth in the tol: 1e-10" 或具体非正交位置
```

---

### `normal` — 归一化本征矢

```python
eigvecs_normalized = ev.normal(eigvecs)
# 对最后4维展平后做 |v|→1 的归一化
```

---

### `phase_exp_2pt` / `phase_exp_3pt` — 动量相位因子

```python
# 零动量
phase0 = ev.phase_exp_2pt(Mom=[0, 0, 0])  # shape: (Nx, Nx, Nx, Nc)

# 非零动量 p = (1, 0, 0)
phase_p = ev.phase_exp_2pt(Mom=[1, 0, 0])
# 每点为 exp(-i·p·x)，x 为格点坐标
# 注意 Mom = [pz, py, px]
```

---

### `Mom_VdV_sink_t` — 动量投影的 VDV 矩阵元

```python
phase_exp = ev.phase_exp_2pt(Mom=[0, 0, 0])
VdV = ev.Mom_VdV_sink_t(phase_exp=phase_exp, eigvecs=eigvecs)
# VdV 形状: (num_Mom, Nev, Nev)
# 计算了 v† · diag(phase) · v 的缩并
```

---

### `Mom_VVV_sink_t` — 动量投影的 VVV 矩阵元

```python
VVV = ev.Mom_VVV_sink_t(phase_exp=phase_exp, eigvecs=eigvecs)
# VVV 形状: (num_Mom, Nev, Nev, Nev)
# 三本征矢动量投影的完全反对称组合
```

---

### `VdV_sink_t_link` — 带规范链的 VDV

```python
VDV_link = ev.VdV_sink_t_link(
    eigvecs    = eigvecs,
    link_dir   = 'Z',        # 规范链方向: 'T'/'Z'/'Y'/'X'/'all'
    link_max   = 5,          # 最大链长度
    phase_exp  = phase_exp,
    gauge_link = gauge,      # 规范场 (Nd, Nt, Nz, Ny, Nx, Nc, Nc)
    t          = 0           # 时间片
)
# VDV_link 形状: (num_Mom, 2*link_max+1, Nev, Nev)
```

---

### `create_omega_accelerate` — 随机蒸馏权重张量

构建 Ω 张量（用于方差缩减的加权方案）：

```python
omega = ev.create_omega_accelerate(
    exact     = 10,          # 精确本征矢数
    N_eigen   = [100],       # block 压缩前本征矢数
    N_sum     = [40],        # block 压缩后数量
    N_extract = [20],        # 每个子 block 提取数
    noise     = 5,           # 噪声矢量数
    dim       = 4            # 输出维度
)
# omega 形状: (Nev, Nev, Nev, Nev)，类型为复数
# Nev = exact + N_sum + noise
```

---

## 4. Gamma 矩阵（`gamma_matrix`）

### `gamma` — 获取 γ 矩阵

Dirac-Pauli (DR) 基下的 4×4 矩阵：

```python
from function_contraction import gamma

g0 = gamma(0)   # 单位阵 I
g5 = gamma(5)   # γ₅ = diag(1, 1, -1, -1)
g4 = gamma(4)   # γ₄（时间方向）
g1 = gamma(1)   # γ₁

# 常用组合:
gamma(6)   # γ₂γ₃
gamma(9)   # γ₁γ₄
gamma(12)  # γ₁γ₅
gamma(15)  # γ₄γ₅
```

完整索引表（i = 0~17）。

---

### `tran_indx_to_gamma` — 索引批量转 γ 矩阵

```python
from function_contraction import tran_indx_to_gamma

gammas = tran_indx_to_gamma([1, 2, 3])
# gammas.shape = (3, 4, 4) → [γ₁, γ₂, γ₃]
```

---

### `PFF_Mom_to_gamma_new` — 动量到 γ 结构映射

通过 Levi-Civita 张量将动量分量映射为 γ 矩阵指标对。

```python
from function_contraction import PFF_Mom_to_gamma_new

idx_matrix, gamma_mat, idx_all, gamma_all = PFF_Mom_to_gamma_new(
    Mom=[[1, 0, 0], [0, 0, 1]],  # Mom = [pz, py, px]
    allow_t=False                   # 纯空间（3D LC 张量）
)
# idx_matrix: 每个动量的非零 γ 指标组合
# gamma_mat:  对应的 4×4 γ 矩阵堆叠
# idx_all:    合并后的 γ 指标
# gamma_all:  合并后的 γ 矩阵
```

---

## 5. Pauli 矩阵（`sigma_matrix`）

### `sigma` — 获取 Pauli 矩阵

```python
from function_contraction import sigma

s0 = sigma(0)  # 单位阵
s1 = sigma(1)  # σₓ
s2 = sigma(2)  # σᵧ
s3 = sigma(3)  # σ_z
```

---

### `Mom_times_sigma` — 动量点乘 Pauli 矢量

```python
from function_contraction import Mom_times_sigma

# p · σ = pz*σz + py*σy + px*σx
result = Mom_times_sigma(Mom=[1, 2, 0], upto4dim=True)
# upto4dim=True: 嵌入 4×4 Dirac 空间 [[S,0],[0,S]]
# Mom 顺序: [pz, py, px]
```

---

## 6. Clebsch-Gordan 系数（`corr_cg`）

### `SU2combine` — 角动量耦合

```python
from function_contraction import SU2combine

# 耦合两个自旋 1/2 粒子
states = [(0.5, 0.5), (0.5, -0.5)]
result = SU2combine(states)
# {(J=1, M=0, int_Js=()): sqrt(0.5),
#  (J=0, M=0, int_Js=()): sqrt(0.5)}
```

### `SU2decompose` — 角动量分解

```python
from function_contraction import SU2decompose

# 将 (J=1, M=1) 分解为两个自旋 1/2 粒子
result = SU2decompose(
    j_list = [0.5, 0.5],
    target = [1, 1]
)
# {(0.5, 0.5): 1.0} → 两个粒子必须都是 m=+1/2
```

---

## 7. 算符共轭（`baroperator`）

### `conjugate_operator` — 算符厄米共轭

```python
from function_contraction import conjugate_operator

# π⁺ = ū γ₅ d 的厄米共轭 → π⁻
op     = ["|", "u^d", "gamma_5", "d", "|"]
op_bar = conjugate_operator(op)
# 返回: [sign, "|", "d^d", "gamma_5", "u", "|"], sign = 1
```

内部自动判断介子/重子结构，正确计算 γ 矩阵链的 H/T/C 变换。

---

## 8. 统计分析（`analyse`）

### `loop_tsrc` — 源位置平均

```python
from function_contraction import loop_tsrc
import numpy as np

# 2pt 函数: data 形状 (Nconf, Nt, Nt, ...)
data_avg = loop_tsrc(
    data   = np.random.randn(100, 64, 64),  # (Nconf, tsink, tsrc)
    indx   = [-2, -3],                       # tsink, tsrc 轴
    Ctype  = '2pt'
)
# 输出形状: (Nconf, Nt, 1, ...) — tsrc 轴被平均并压缩
```

对于 3pt，需额外提供 `t_sep`；反周期边界条件设 `Boundary_Conditions='Antiperiodic'`。

---

### `Jackknife` — Jackknife 重采样

```python
from function_contraction import Jackknife

data = np.random.randn(100, 64)  # (Nconf, Nt)
info = Jackknife(data, Nconf_axes=0)
# info['data_sample']  — JK 样本, shape (100, 64)
# info['data_mean']    — 均值,    shape (64,)
# info['data_err']     — 误差,    shape (64,)
# info['data_cov']     — 协方差,  shape (64, 64)
```

Jackknife 样本构造：去掉第 i 个组态，用剩余 N−1 个组态的平均作为第 i 个样本。

---

### `meff` — 有效质量

```python
from function_contraction import meff, Jackknife

jk = Jackknife(data_real)  # data_real 需为实数（取实部或虚部）

eff = meff(
    data_sample = jk['data_sample'],
    alttc       = 0.12,          # 格距 (fm)
    Nconf_axes  = 0,
    Nt_axes     = 1,
    meff_type   = 'log'          # 'log' / 'cosh' / 'GEVP'
)
# eff['data_mean'] — 有效质量平均值 (GeV)
# eff['data_err']  — 误差 (GeV)
```

- `'log'`:  m_eff(t) = ln[C(t)/C(t+1)] / a
- `'cosh'`: 解 cosh 方程 C(t+2)+C(t) / 2C(t+1)
- `'GEVP'`: 同 log（用于 GEVP 特征值）

---

### `solve_gevp` — 广义本征值问题

```python
from function_contraction import solve_gevp

C = np.random.randn(3, 3, 64)  # (N_op, N_op, Nt) — 关联矩阵
eigvals, eigvecs = solve_gevp(C, t0=2)
# eigvals: (3, 64) — 排序后的广义本征值
#   t < t0 → 升序; t >= t0 → 降序（λ 最大 ≡ 基态）
# eigvecs: (3, 3, 64) — 对应的本征矢
```

---

### `PDF` — 3pt/2pt 比值

用于抽取部分子分布函数（PDF）的矩阵元比值：

```python
from function_contraction import PDF

pdf = PDF(
    data_3pt_sample = data_3pt,   # 3pt 关联函数
    data_2pt_sample = data_2pt,   # 2pt 关联函数
    t_sep   = 8,                  # 源-汇间隔
    link_fold = True              # 折叠 ±z 链方向
)
# pdf['data_mean'] — 比值矩阵元 R(t_sink, z)
```

---

### `Mom2GeV` — 动量转 GeV

```python
from function_contraction import Mom2GeV

E = Mom2GeV(Nx=32, alttc=0.12, Mom=[1, 0, 0], M0=0.5)
# E ≈ 0.586 GeV
# E = √((2π/Nx · fm2GeV/alttc)² · p² + M0²)

# M0 为列表时对应多粒子系统
E = Mom2GeV(Nx=32, alttc=0.12, Mom=[1, 0, 0], M0=[0.5, 0.5])
# E ≈ 0.586 + 0.586 = 1.172 GeV (两粒子总能量)
```

---

### `sum_over_array_of_list` — 分组求和

```python
from function_contraction import sum_over_array_of_list
import numpy as np

arr = np.arange(24).reshape(2, 3, 4)  # (2, 3, 4)
result = sum_over_array_of_list(
    arr, axes=(1, 2),
    groupings=([(0, 2), (1,)], [(0, 3), (1, 2)])
)
# result.shape = (2, 2, 2)
# axis1: 3→2组, axis2: 4→2组
```

---

## 9. 基础工具（`corr_base_functions`）

### `levi_civita_tensor` — Levi-Civita 张量

```python
from function_contraction import levi_civita_tensor

eps = levi_civita_tensor(3)
# eps[i,j,k] = sign(permutation), shape (3,3,3)
# eps[0,1,2] = +1, eps[1,0,2] = -1
```

---

### `creat_mom_list` — 动量组合生成

```python
from function_contraction import creat_mom_list

# 从参考动量 [1,1,0] 生成所有等价动量
mom_list = creat_mom_list(Mom=[1, 1, 0], fix_Q2=True, only_g0=False)
# 返回所有 Q²=2 的动量组合，含正负号变体
# [[1,1,0], [1,-1,0], [-1,1,0], [-1,-1,0], [1,0,1], ...]
```

---

### `ArraySlicer` — 高级数组切片

```python
from function_contraction import ArraySlicer
import numpy as np

arr = np.zeros((4, 5, 6))
slicer = ArraySlicer(arr)
slicer.assign(dims=[0, 1], indices=[[1, 3], [0, 2]], values=[[1, 2], [3, 4]])
# arr[1,0]=1, arr[1,2]=2, arr[3,0]=3, arr[3,2]=4
```

---

## 10. 文件 I/O（`corr_io`）

```python
from function_contraction import load, write_data_ascii, check_dir_path

# 读取本征矢
eigvecs = load(Nx=32, path="eigvecs.dat", dtype='readin', vec_or_peram='vector')

# 读取 perambulator
peram = load(Nx=32, path="peram.npy", dtype='numpy', vec_or_peram='peram')

# 写入 ASCII（L. Liu 格式）
write_data_ascii(data=correlator, T=64, L=32, filename="output/corr.dat", complex=True)

# 自动创建输出目录
check_dir_path("output/result/")
```

---

## 11. MPI 并行（`mpi_init`）

```python
from function_contraction import mpinit, get_mpi_data, get_mpi_tlist, getMPIRank

# 初始化 MPI 网格 + CUDA 设备
mpinit(grid_size=[1, 1, 1, 4], latt_size=[32, 32, 32, 64], backend='cupy')

# 时间片到 MPI rank 的映射
t_rank, t_idx = get_mpi_tlist(Nt=64, t=10, gtype='find')
# t_rank: t=10 所在的 rank; t_idx: 该 rank 上的局部时间索引

# MPI 数据收发
data_gathered = get_mpi_data(data, mdtype='Gather', root=0)
# 将各 rank 的数据 gather 到 rank 0
```

支持的 `mdtype`: `'Send'`, `'Gather'`, `'TGather'`, `'Allgather'`, `'Bcast'`, `'Scatter'`, `'TScatter'`, `'Transport'`。

---

## 12. Gauge Smearing（`smear_gauge`）

### `stout_smear_ndarray` — 规范场 Stout Smearing

```python
from function_contraction import stout_smear_ndarray
import numpy as np

# gauge: (Nd-1, Nt, Nz, Ny, Nx, Nc, Nc)
gauge = np.zeros((3, 64, 32, 32, 32, 3, 3), dtype=complex)
# ... 填充 gauge links ...

gauge_smeared = stout_smear_ndarray(gauge, nstep=36, rho=0.1)
# 原地修改 gauge，返回 smearing 后的规范场
```

---

## 13. 后端切换（`backend`）

```python
from function_contraction import set_backend, get_backend

set_backend('cupy')   # 切换到 GPU
set_backend('numpy')  # 切换到 CPU

backend = get_backend()
x = backend.ones((10, 10))  # 使用当前后端的数组操作
```

---

## 14. 常量（`constant`）

```python
Nc = 3         # 颜色数
Ns = 4         # 自旋分量数
Nd = 4         # Dirac 分量数
fm2GeV = 0.197 # 单位转换: 1 fm = 0.197 GeV⁻¹
```

---

## 实用绘图常量（`analyse`）

```python
plot_analyse_marker = ['s','*','+','x','p','h','v','X','D','P','H','o']
plot_analyse_color  = ['#3498DB','#ff7f0e','#2ECC71','#E74C3C',
                       '#9467bd','#8c564b','#CB4335','#e377c2',
                       '#7f7f7f','#F1C40F','#17becf','#2ca02c']
```

---

## 典型工作流示例

### 介子 2pt 关联函数完整流程

```python
from function_contraction import *

# 1. 读取数据
eigvecs = load(Nx=32, path="eigvecs.dat", dtype='readin', vec_or_peram='vector')
peram   = load(Nx=32, path="peram.npy", dtype='numpy', vec_or_peram='peram')

# 2. 初始化本征矢工具
ev = corr_eigvecs(Nx=32, backend='numpy')

# 3. Wick 缩并
result = wick_contraction(
    sink_operators   = ["|", "u^d", "gamma_5", "d", "|"],
    source_operators = ["|", "u^d", "gamma_5", "d", "|"],
    curr_operators   = [],
    Cpt='2pt'
)

# 4. 查看缩并图
fig, ax = plot_figure_wick(result, diagram_index=0, Cpt='2pt')
fig.savefig("wick.pdf")

# 5. 构建动量投影（零动量为例）
phase = ev.phase_exp_2pt(Mom=[0, 0, 0])
VdV   = ev.Mom_VdV_sink_t(phase_exp=phase, eigvecs=eigvecs)

# 6. 缩并计算关联函数（由 result['peram'], result['V'], result['gamma_pos'] 驱动）
#    ...缩并逻辑依赖于具体缩并框架...

# 7. 统计分析
corr_avg = loop_tsrc(corr_data, indx=[-2, -3], Ctype='2pt')
jk = Jackknife(corr_avg, Nconf_axes=0)
eff_mass = meff(jk['data_sample'].real, alttc=0.12, meff_type='log')

# 8. 画图
import matplotlib.pyplot as plt
t = np.arange(len(eff_mass['data_mean']))
plt.errorbar(t, eff_mass['data_mean'], yerr=eff_mass['data_err'])
plt.xlabel('t')
plt.ylabel('m_eff (GeV)')
plt.savefig("meff.pdf")
```

### 重子 GEVP 分析

```python
# 假设已有多个算符的关联矩阵 C (Nop, Nop, Nt, Nconf)
C = np.load("corr_matrix.npy")
C_jk = Jackknife(C, Nconf_axes=-1)['data_sample']  # JK 样本

eigvals, eigvecs = solve_gevp(C_jk.mean(axis=-2), t0=2)

eff = meff(eigvals[0], alttc=0.12, meff_type='GEVP')  # 基态有效质量
```
