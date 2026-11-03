# lqcddb —— 格点 QCD 蒸馏（Distillation）关联函数计算与统计分析工具包

基于蒸馏（distillation / blending）方法的格点 QCD 关联函数计算、Wick 收缩与统计分析工具包。支持 numpy（CPU）和 cupy（GPU）两种后端，内置 MPI 并行。

## 快速开始

```python
from lqcddb import *
set_backend('numpy')    # 或 'cupy' 切换 GPU
backend = get_backend()
```

### 物理常量

```python
Nc = 3           # 颜色数
Ns = 4           # 自旋分量数
Nd = 4           # Dirac 分量数
fm2GeV = 0.197   # 单位转换: 1 fm = 0.197 GeV⁻¹
```

---

## 包结构

```
lqcddb/
├── __init__.py              # 顶层入口，懒加载全部公开 API
├── physics.md               # 强子算符库、流算符、规范组态参数表
├── README.md                # 本文件
├── skill/                   # 项目级 Skill（供 AI 助手使用）
├── base/                    # 基础工具层
│   ├── backend.py           # numpy / cupy 全局后端切换
│   ├── base_functions.py    # Levi-Civita 张量、动量列表生成、ArraySlicer、cached_contract
│   ├── cg_coeff.py          # SU(2) Clebsch-Gordan 系数（耦合与分解）
│   ├── mpi_init.py          # MPI 初始化、网格划分、时间片分配、数据收发
│   └── smear_gauge.py       # Stout 规范场 smearing
├── constant/                # 物理常量与矩阵
│   ├── constant.py          # Nc, Ns, Nd, fm2GeV
│   ├── gamma_matrix.py      # Dirac γ 矩阵（DR 基，共 18 种）及动量→γ 映射
│   └── sigma_matrix.py      # Pauli σ 矩阵及 p·σ 缩并
├── contraction/             # Wick 收缩核心
│   ├── autowick.py          # 自动 Wick 收缩、缩并图可视化、等价图识别
│   ├── baroperator.py       # 强子算符厄米共轭（自动判断介子/重子结构）
│   ├── seqperam.py          # Perambulator 的 γ₅ 时序变换
│   └── contractadviser.py   # 带宽感知的张量缩并分析器（Roofline 模型）
├── eigvectors/              # 本征矢（Distillation Eigenvectors）运算
│   ├── vector.py            # 本征矢代数：内积、正交归一化、随机噪声、压缩
│   └── vertex.py            # 顶点函数：相位因子、VdV、VVV、带规范链 VdV、Ω 权重
├── analyse/                 # 统计分析
│   └── analyse.py           # Jackknife、Bootstrap、有效质量、GEVP、PDF、源平均
├── io/                      # 输入输出
│   └── write_date.py        # 二进制/ASCII 格式读写本征矢和 perambulator
└── test/                    # 测试脚本与示例
```

---

## 1. Wick 收缩（`contraction/autowick.py`）

### `wick_contraction` — 自动 Wick 收缩

输入 sink、source、current 三组算符（以 `'|'` 分隔强子），自动穷举所有可能的 Wick 缩并图，返回每个图的缩并指标、Fermi 符号、peram 条目、VVV/VDV 顶点和 γ 矩阵插入位置。

**算符格式约定：**
- 味标记：`'u'`, `'d'`, `'s'`, `'c'`, `'b'`, `'t'` 及对应的反夸克 `'u^d'`, `'d^d'` 等
- γ 矩阵插入：`'gamma_5'`, `'gamma_mu'`, `'gamma_4'` 等，或自定义标签如 `'gamma_C1'`
- 强子分隔符：`'|'`（每个强子以一对 `'|'` 包裹）
- 通配符：`'q'`/`'q^d'`（匹配全部六味），`'l'`/`'l^d'`（仅匹配轻味 u, d）
- 数值系数：可在算符列表中直接放入 `int`/`float`/`complex` 作为全局系数

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `sink_operators` | `List[str]` | sink 端强子算符列表 |
| `source_operators` | `List[str]` | source 端强子算符列表 |
| `curr_operators` | `List[str]` | 流算符列表，2pt 时传 `[]` |
| `Cpt` | `'bubble' \| '2pt' \| '3pt' \| '4pt'` | 关联函数类型 |
| `Pindex` | `list`，可选 | peram 对象的前缀字母，不传则自动生成大写字母 |
| `Vindex` | `list`，可选 | VVV/VDV 对象的前缀字母 |
| `Gindex` | `list`，可选 | γ 插入的前缀字母 |

**返回值（单 dict，使用通配符时返回 `list[dict]`）：**

| 键 | 类型 | 说明 |
|----|------|------|
| `result_indx` | `list[list[str]]` | 每个缩并图的指标字符串，形如 `"ab,cd,ef->gh"` |
| `result_name` | `list[list[str]]` | 每个缩并图的组件名称，如 `"peram_d, gamma_5, VDV_0"` |
| `result_sign` | `list[float]` | 每个缩并图的 Fermi 符号（+1 或 −1），已包含全局系数 |
| `operators` | `list` | 拼接后的完整算符序列（含分隔符） |
| `sink_operators` | `list` | sink 部分算符 |
| `source_operators` | `list` | source 部分算符 |
| `curr_operators` | `list` | 流部分算符 |
| `quark_pos` | `list[tuple]` | `(位置, 味标记, 缩并标签)`，缩并标签如 `'ab'`, `'cd'` |
| `sep_pos` | `list[int]` | 所有 `'\|'` 分隔符的索引位置 |
| `gamma_pos` | `list[tuple]` | `(位置, γ名称, 组合指标, 时间标签)` |
| `V` | `list[tuple]` | `[(区间), 类型名, 指标串, 时间标签]`，VVV 或 VDV |
| `peram` | `list[list[tuple]]` | 每个缩并图的 peram 条目，每项为 `[夸克位置, 反夸克位置, 组合类型, 标签, [时间1, 时间2]]` |

**示例 — π⁺ 介子 2pt 函数：**

```python
from lqcddb import wick_contraction

result = wick_contraction(
    sink_operators   = ["|", "u^d", "gamma_5", "d", "|"],
    source_operators = ["|", "u^d", "gamma_5", "d", "|"],
    curr_operators   = [],
    Cpt='2pt'
)
# 返回单个 dict，包含 1 个缩并图
```

**示例 — 使用通配符同时计算 π⁺ 和 K⁺：**

```python
# 'q' 会展开为 u, d, s, c, b, t 六味
result = wick_contraction(
    sink_operators   = ["|", "q^d", "gamma_5", "q", "|"],
    source_operators = ["|", "q^d", "gamma_5", "q", "|"],
    curr_operators   = [],
    Cpt='2pt'
)
# 返回 list[dict]，每个合法味替换一个 dict
# 例如 "u^d ... d"（π⁺）和 "u^d ... s"（K⁺）
```

**示例 — 质子 2pt 函数：**

```python
result = wick_contraction(
    sink_operators   = ["|", "u", "C*gamma_5", "d", "u", "|"],
    source_operators = ["|", "u", "C*gamma_5", "d", "u", "|"],
    curr_operators   = [],
    Cpt='2pt'
)
# 质子 → 质子，2 个缩并图（direct + exchange）
```

**示例 — K→π 半轻子衰变 3pt 函数：**

```python
result = wick_contraction(
    sink_operators   = ["|", "u^d", "gamma_5", "s", "|"],       # K⁺
    source_operators = ["|", "u^d", "gamma_5", "d", "|"],       # π⁺
    curr_operators   = ["|", "s^d", "gamma_mu", "d", "|"],      # 矢量流
    Cpt='3pt'
)
```

**示例 — 质子→中子 3pt 函数（isovector 矢量流）：**

```python
wick_diag = wick_contraction(
    sink_operators   = ['|', 'u',  'u',  'gamma_7', 'd',  '|'],          # 质子
    source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|'],       # 反中子
    Cpt='3pt',
    curr_operators   = ['|', 'u^d', 'gamma_C2', 'd', '|'],               # isovector 流
)
# 返回 4 个缩并图（2 对变体）
```

---

### `plot_figure_wick` — 绘制 Wick 缩并图

将 `wick_contraction` 的结果可视化为夸克线 Feynman 图。图中包含：夸克节点（按味着色）、传播子箭头（带曲率）、γ 矩阵虚线、VVV/VDV 顶点标签、强子间分隔虚线、时间方向箭头、图例和缩并信息。

```python
from lqcddb import wick_contraction, plot_figure_wick

result = wick_contraction(
    sink_operators   = ["|", "u^d", "gamma_5", "d", "|"],
    source_operators = ["|", "u^d", "gamma_5", "d", "|"],
    curr_operators   = [],
    Cpt='2pt'
)

fig, ax = plot_figure_wick(result, diagram_index=0, Cpt='2pt', plot_text=True)
fig.savefig("wick_pion_2pt.pdf", dpi=300, bbox_inches='tight')
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `result_dict` | `dict` | `wick_contraction` 返回的结果字典 |
| `diagram_index` | `int` | 要绘制的缩并图编号（从 0 开始） |
| `Cpt` | `'2pt' \| '3pt' \| '4pt'` | 关联函数类型，决定时间分区布局 |
| `plot_text` | `bool` | 是否绘制 γ 矩阵和 VVV/VDV 的文字标签 |

---

### `identify_equivalent_diagrams` — 识别等价缩并图

利用并查集算法，找出由 γ 矩阵插入位置交换而连通的多组等价缩并图。输入多个 `wick_contraction` 返回的 dict，输出等价类的分组。

```python
from lqcddb import identify_equivalent_diagrams

# 输入 3 个 dict（如来自不同味替换）
groups = identify_equivalent_diagrams(result1, result2, result3)
# 返回: [[0, 2], [1]]
# 含义：第 0 和第 2 个 dict 的缩并图等价，第 1 个独立
```

---

## 2. Peram 变换（`contraction/seqperam.py`）

### `seq_peram` — γ₅ 时序传播子变换

对 perambulator 施加 γ₅ 厄米共轭变换，用于将夸克传播子转为反夸克传播子：

$$\tau \rightarrow \gamma_5 \cdot \tau^\dagger \cdot \gamma_5$$

```python
from lqcddb import seq_peram
import numpy as np

# 输入形状: (..., d_sink, d_src, Nev_sink, Nev_src)
peram = np.random.randn(64, 4, 4, 32, 32) + 1j * np.random.randn(64, 4, 4, 32, 32)
peram_seq = seq_peram(peram)
# 输出形状: (..., d_src, d_sink, Nev_src, Nev_sink)
# Dirac 指标和本征矢指标的 sink↔src 已交换
```

典型用法：
- `seq_peram(peram_u_src[t_curr])` —— 把 source→current 传播子转为 current→source
- `seq_peram(peram_u_sink_seq)` —— 把 sink→current 传播子链调转方向

**注意**：该函数做的是共轭（conjugate）而非共轭转置（conjugate-transpose）。Peram 的 5D 单文件形状为 `(t_sink, d_sink, d_src, ev_sink, ev_src)`。

---

## 3. 本征矢运算

旧包中的 `corr_eigvecs` 大类已按职责拆分为两个独立的类：

| 类 | 模块 | 职责 |
|----|------|------|
| `vector_creator` | `eigvectors/vector.py` | 本征矢代数：内积、正交归一化检验、Gram-Schmidt 正交化、随机噪声矢生成、本征矢压缩 |
| `vertex_creator` | `eigvectors/vertex.py` | 顶点函数：动量相位因子、VdV/VVV 动量投影、带规范链接的 VdV、Ω 蒸馏权重张量 |

---

### `vector_creator` — 本征矢代数运算

```python
from lqcddb import vector_creator

vec = vector_creator()
```

#### `check(eigvecs, dtype='find', tol=1e-10, check_normal=True)`

检验本征矢集的正交归一性：验证 $V^\dagger V \approx I$。

- `dtype='find'`：静默检查，返回 `True`/`False`
- `dtype='print'`：打印具体哪个位置不正交、不正交的程度
- `check_normal=True`：同时检查每个矢量的模是否为 1

```python
eigvecs = np.load("eigvecs.npy")  # 形状 (Nev, Nz, Ny, Nx, Nc)
is_orth = vec.check(eigvecs, dtype='print', tol=1e-10)
```

#### `normal(vectors)`

归一化：$v \rightarrow v / |v|$。对最后 4 维（空间×颜色）展平后归一化。

```python
eigvecs_norm = vec.normal(eigvecs)
```

#### `inner_product(init_vector, test_vector, dtype='')`

计算两组矢量的内积矩阵。若 `dtype='abs'`，返回内积的模平方。

```python
C = vec.inner_product(eigvecs_A, eigvecs_B)          # 复内积矩阵
C_abs = vec.inner_product(eigvecs_A, eigvecs_B, dtype='abs')  # |内积|²
```

#### `orthnormal(vectors_init, vector)`

对一个新的矢量做 Gram-Schmidt 正交化（相对于已有矢量集），然后归一化。返回追加了新矢量后的完整集合。

```python
vectors = vec.orthnormal(existing_set, new_vector)
```

#### `creat_noise(vectors_init, N, dtype='complex')`

生成 N 个与已有矢量集正交的随机噪声矢。`dtype='complex'` 或 `'float'`。

```python
# 在 100 个精确本征矢基础上生成 50 个正交噪声矢
noise_augmented = vec.creat_noise(exact_eigvecs, N=50, dtype='complex')
```

#### `compress_matrix_V1(eigenvectors, N_eigen, N_sum, Ctype='I')`

**求和压缩**（第一版）。将本征矢按指定方式分组，组内求和并归一化。

三种压缩类型：

| Ctype | 含义 | 适用场景 |
|-------|------|---------|
| `'I'`（interlace） | 交错分组，均匀穿插 | 单一块，均匀混合 |
| `'B'`（block） | 分块分组 | 多个独立块，每块内部分组 |
| `'BI'`（block-interlace） | 第一维分块，第二维交错 | 混合策略 |

```python
# 将 100 个本征矢交错压缩为 40 个
compressed = vec.compress_matrix_V1(
    eigenvectors, N_eigen=[100], N_sum=[40], Ctype='I'
)
```

#### `compress_matrix_V2(eigenvectors, N_eigen, N_sum, N_extract, Ctype='I')`

**随机提取压缩**（第二版）。从每组中随机选取 `N_extract` 个本征矢，而非求和。

```python
# 从每组 25 个本征矢中随机选 5 个
compressed = vec.compress_matrix_V2(
    eigenvectors, N_eigen=[100], N_sum=[40], N_extract=[5], Ctype='B'
)
```

#### `compress_matrix_V3(eigenvectors, N_eigen, N_sum, N_extract, Ctype='I', adjcent=False)`

**随机投影压缩**（第三版）。生成随机正交矢量，投影到本征子空间。

- `adjcent=True`：使用相邻本征矢做投影
- `adjcent=False`：使用交错本征矢做投影

```python
compressed = vec.compress_matrix_V3(
    eigenvectors, N_eigen=[100], N_sum=[40], N_extract=[10],
    Ctype='I', adjcent=False
)
```

#### `compress_matrix_V4(eigenvectors, N_eigen, N_sum, N_extract, Ctype='I', adjcent=False, random_type='orthnormal')`

**可配置随机投影压缩**（第四版）。V3 的扩展，支持选择随机矢量生成方式：

- `random_type='orthnormal'`：连续正交随机矢量（默认，自动检验正交性）
- `random_type='Z_N'`：Z_N 离散噪声（如 `'Z_4'` 表示 {±1, ±i}）

```python
compressed = vec.compress_matrix_V4(
    eigenvectors, N_eigen=[200], N_sum=[80], N_extract=[20],
    Ctype='BI', random_type='Z_4'
)
```

---

### `vertex_creator` — 顶点函数

```python
from lqcddb import vertex_creator

vtx = vertex_creator(Nx=32)  # Nx 为空间格点数
```

#### `check(eigvecs, dtype='find', tol=1e-10, check_normal=True)`

与 `vector_creator.check()` 功能相同，检验本征矢的正交归一性。

#### `normal(vectors)`

对最后 4 维 `(Nz, Ny, Nx, Nc)` 做归一化。

#### `phase_exp_2pt(Mom=[0, 0, 0])`

计算 2pt 函数的动量相位因子 $e^{-i\mathbf{p}\cdot\mathbf{x}}$，sink 和 source 共用。

```python
# Mom 顺序: [pz, py, px]
phase_zero = vtx.phase_exp_2pt(Mom=[0, 0, 0])    # 形状 (Nx, Nx, Nx, Nc)
phase_p    = vtx.phase_exp_2pt(Mom=[1, 0, 0])    # p = (0, 0, 1)，即 pz=1
```

#### `phase_exp_3pt(Mom=[0, 0, 0])`

计算 3pt 函数的动量相位因子（用于 sink 端投影）。返回形状 `(Nx, Nx, Nx)`（无颜色维）。

#### `Mom_VdV_sink_t(phase_exp, eigvecs)`

计算介子 sink/source 的动量投影 VDV 矩阵元：

$$V_{ij}(p) = \sum_x e^{-ip\cdot x}\, \phi_i^\dagger(x)\, \phi_j(x)$$

- `eigvecs` 形状：`(Nev, Nz, Ny, Nx, Nc)`
- 返回形状：`(num_Mom, Nev, Nev)`

```python
phase = vtx.phase_exp_2pt(Mom=[0, 0, 0])
VdV = vtx.Mom_VdV_sink_t(phase_exp=phase, eigvecs=eigvecs)
```

#### `Mom_VVV_sink_t(phase_exp, eigvecs)`

计算重子 sink/source 的动量投影 VVV 矩阵元（含 Levi-Civita 颜色缩并）：

$$V_{ijk}(p) = \sum_x e^{-ip\cdot x}\, \varepsilon_{abc}\, \phi_i^a(x)\, \phi_j^b(x)\, \phi_k^c(x)$$

计算全部 6 种颜色排列（3 偶排列 + 3 奇排列带负号）。返回形状 `(num_Mom, Nev, Nev, Nev)`。

```python
VVV = vtx.Mom_VVV_sink_t(phase_exp=phase_3pt, eigvecs=eigvecs)
```

#### `VdV_sink_t_link` — 带规范链接的 VDV

计算带规范场平行输运的 VDV 矩阵元，用于流插入：

$$V_{mn}(p, \Delta x) = \sum_x e^{-ip\cdot x}\, \phi_m^\dagger(x)\, U(x, x+\Delta x)\, \phi_n(x+\Delta x)$$

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `eigvecs` | `ndarray` | 本征矢，形状 `(Nev, Nz, Ny, Nx, 3)` |
| `link_dir` | `str` | 链接方向：`'0'`（无链接）、`'T'`、`'Z'`、`'Y'`、`'X'`、`'all'` |
| `link_max` | `int` | 空间链接的最大位移（仅空间方向有效） |
| `phase_exp` | `ndarray` | 动量相位因子 |
| `gauge_link` | `ndarray` 或 `bool` | 规范场，形状 `(Nd, Nt, Nz, Ny, Nx, 3, 3)`。传 `False` 表示无规范链接 |
| `t` | `int` | 时间片索引 |
| `eigvecs_min` | `ndarray` 或 `None` | 第二组本征矢（仅守恒流/时间方向需要） |
| `conserved` | `bool` | 是否为守恒流 |

**三种计算情形：**

| 情形 | 条件 | 返回形状 | 说明 |
|------|------|---------|------|
| 无规范链接 | `gauge_link=False` 或 `link_dir='0'` | `(num_Mom, 1, Nev, Nev)` | 直接 V†·V |
| 守恒流/时间 | `conserved=True` 或 `link_dir='T'` | `(2, Nev, Nev)` | 前向+后向，需 `eigvecs_min` |
| 空间方向 | `link_dir='X'/'Y'/'Z'/'all'` | `(num_Mom, 2*link_max+1, Nev, Nev)` | 沿空间方向构建规范路径 |

```python
# 零动量，Z 方向，最大位移 ±5
VDV_link = vtx.VdV_sink_t_link(
    eigvecs=eigvecs,
    link_dir='Z',
    link_max=5,
    phase_exp=phase_2pt,
    gauge_link=gauge,
    t=0
)
# 形状: (1, 11, Nev, Nev)

# 守恒流
VDV_cons = vtx.VdV_sink_t_link(
    eigvecs=eigvecs,
    link_dir='T',
    link_max=0,
    phase_exp=phase_2pt,
    gauge_link=gauge,
    t=0,
    eigvecs_min=eigvecs_min,
    conserved=True
)
# 形状: (2, Nev, Nev)
```

#### `create_omega_accelerate` — Ω 蒸馏权重张量

构建用于方差缩减的 Ω 权重张量。支持任意维度（2D、3D、4D）。

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `exact` | `int` | `0` | 精确（未 smearing）本征矢数量 |
| `N_eigen` | `list[int]` | `[0]` | 各 block 中压缩前的本征矢数量 |
| `N_sum` | `list[int]` | `[0]` | 各 block 中压缩后的矢数量 |
| `N_extract` | `list[int]` | `[0]` | 各 block 中每个子块提取的数量 |
| `noise` | `int` | `0` | 噪声矢数量 |
| `conserved` | `bool` | `False` | 是否守恒模式（维度固定为 2） |
| `normal` | `bool` | `False` | 是否对 Ω 做归一化 |
| `fixed_first_pos` | `list[int]` | `[-1]` | 固定第一维的位置（用于特殊加权方案） |
| `dim` | `int` | `2` | 输出张量的维度（2、3 或 4） |

```python
# 2D Ω 权重（介子）
omega_2d = vtx.create_omega_accelerate(
    exact=50, N_eigen=[], N_sum=[], N_extract=[],
    noise=150, dim=2
)  # 形状: (Nev, Nev)

# 4D Ω 权重（重子-介子散射等）
omega_4d = vtx.create_omega_accelerate(
    exact=10, N_eigen=[100], N_sum=[40], N_extract=[20],
    noise=5, dim=4
)  # 形状: (Nev, Nev, Nev, Nev)
```

返回复数张量。使用时直接与 VDV/VVV 相乘即可施加蒸馏权重。

#### `perm_comb(N, M=1, dtype='perm', renormal=False)`

计算排列数 $P(N,M)$ 或组合数 $C(N,M)$。

```python
n_perm = vtx.perm_comb(10, 3, dtype='perm')    # P(10,3) = 720
n_comb = vtx.perm_comb(10, 3, dtype='comb')    # C(10,3) = 120
```

#### `src_sink_MPI_tran(src_sink, mpi_size, trtype='forward')`

MPI 时间维转置：将数据的时间维拆分到多个 rank（`'forward'`），或从多个 rank 恢复（`'backward'`）。

---

## 4. Gamma 矩阵（`constant/gamma_matrix.py`）

Dirac 矩阵使用 DeGrand-Rossi（DR）基，全部为 4×4 复矩阵。由 `get_backend()` 自动适配 numpy 或 cupy。

### `gamma(i)` — 获取 γ 矩阵

| i | 矩阵 | 说明 | i | 矩阵 | 说明 |
|---|------|------|---|------|------|
| 0 | $I_4$ | 单位阵 | 9 | $\gamma_1\gamma_4$ | |
| 1 | $\gamma_1$ | | 10 | $\gamma_2\gamma_4$ | |
| 2 | $\gamma_2$ | | 11 | $\gamma_3\gamma_4$ | |
| 3 | $\gamma_3$ | | 12 | $\gamma_1\gamma_5$ | |
| 4 | $\gamma_4$ | 时间方向 | 13 | $\gamma_2\gamma_5$ | |
| 5 | $\gamma_5$ | $\text{diag}(1,1,-1,-1)$ | 14 | $\gamma_3\gamma_5$ | |
| 6 | $\gamma_2\gamma_3$ | | 15 | $\gamma_4\gamma_5$ | |
| 7 | $\gamma_3\gamma_1$ | 即 $C\gamma_5$ | 16 | $(\gamma_3\gamma_1)(1+\gamma_4)/2$ | 正宇称投影 |
| 8 | $\gamma_1\gamma_2$ | | 17 | $(\gamma_3\gamma_1)(1-\gamma_4)/2$ | 负宇称投影 |

```python
from lqcddb import gamma

g0 = gamma(0)   # 单位阵
g5 = gamma(5)   # γ₅
g4 = gamma(4)   # γ₄（时间方向）
g7 = gamma(7)   # γ₃γ₁（即 C·γ₅，常用于重子 diquark）
```

### `tran_indx_to_gamma(indx)`

将 γ 矩阵索引（整数或数组）批量转换为 4×4 矩阵。

```python
from lqcddb import tran_indx_to_gamma

# 单个索引 → (4, 4)
g = tran_indx_to_gamma(5)

# 数组索引 → (N, 4, 4)
gammas = tran_indx_to_gamma([1, 2, 3, 4])  # 形状: (4, 4, 4)

# 多维索引 → (..., 4, 4)
gammas = tran_indx_to_gamma(np.array([[1, 2], [3, 4]]))  # 形状: (2, 2, 4, 4)
```

### `PFF_Mom_to_gamma_new(Mom, allow_t=False)`

通过 Levi-Civita 张量将动量分量映射为 γ 矩阵指标对，用于投影形状因子（Projected Form Factors）。

- `Mom`：动量列表，格式 `[[pz, py, px], ...]`
- `allow_t=False`：仅使用 3D Levi-Civita 张量（纯空间）
- `allow_t=True`：使用 4D Levi-Civita 张量（含时间分量）

返回 4 个值：
- `gamma_indx_matrix`：每个动量的非零 γ 指标组合
- `gamma_matrix`：对应的 4×4 γ 矩阵
- `gamma_indx_all`：合并去重后的 γ 指标
- `gamma_matrix_all`：合并后的 γ 矩阵

```python
from lqcddb import PFF_Mom_to_gamma_new

idx_mat, g_mat, idx_all, g_all = PFF_Mom_to_gamma_new(
    Mom=[[1, 0, 0], [0, 0, 1]],
    allow_t=False
)
```

---

## 5. Pauli 矩阵（`constant/sigma_matrix.py`）

2×2 Pauli 矩阵，用于自旋空间的运算。

### `sigma(i)` — 获取 Pauli 矩阵

| i | 矩阵 | 符号 |
|---|------|------|
| 0 | 单位阵 | $I$ |
| 1 | $\sigma_x$ | $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ |
| 2 | $\sigma_y$ | $\begin{pmatrix}0&-i\\i&0\end{pmatrix}$ |
| 3 | $\sigma_z$ | $\begin{pmatrix}1&0\\0&-1\end{pmatrix}$ |

```python
from lqcddb import sigma

s0 = sigma(0)  # 单位阵
s1 = sigma(1)  # σₓ
s2 = sigma(2)  # σ_y
s3 = sigma(3)  # σ_z
```

### `Mom_times_sigma(Mom, upto4dim=False)`

计算动量与 Pauli 矢量的点乘：$\mathbf{p}\cdot\boldsymbol{\sigma} = p_z\sigma_z + p_y\sigma_y + p_x\sigma_x$。

- `Mom`：动量列表 `[pz, py, px]`（注意顺序！）
- `upto4dim=False`：返回 2×2 矩阵
- `upto4dim=True`：嵌入 4×4 Dirac 空间，$\begin{pmatrix} \mathbf{p}\cdot\boldsymbol{\sigma} & 0 \\ 0 & \mathbf{p}\cdot\boldsymbol{\sigma} \end{pmatrix}$

```python
from lqcddb import Mom_times_sigma

# 2×2 结果
result_2x2 = Mom_times_sigma(Mom=[1, 2, 0])

# 4×4 嵌入
result_4x4 = Mom_times_sigma(Mom=[1, 2, 0], upto4dim=True)
```

---

## 6. SU(2) Clebsch-Gordan 系数（`base/cg_coeff.py`）

基于 SymPy 的精确 CG 系数计算，用于自旋和同位旋的角动量耦合与分解。

### `SU2combine(states)` — 角动量耦合

将多个 SU(2) 态耦合为总角动量态。输入为 `[(j1, m1), (j2, m2), ...]`，返回 `{(J_total, M_total, intermediate_Js): coefficient}`。

对于 N>2 个粒子，中间耦合的 J 值会被记录在 `intermediate_Js` 中以区分简并态。

```python
from lqcddb import SU2combine

# 耦合两个自旋 1/2 粒子
states = [(0.5, 0.5), (0.5, -0.5)]
result = SU2combine(states)
# {(J=1, M=0, int_Js=()): sqrt(0.5),
#  (J=0, M=0, int_Js=()): sqrt(0.5)}

# 耦合三个粒子
states = [(0.5, 0.5), (0.5, 0.5), (1, -1)]
result = SU2combine(states)
# 键中包含 intermediate_Js 元组以区分正交态
```

### `SU2decompose(j_list, target, intermediate_Js=None)` — 角动量分解

将总角动量态分解为各粒子的 $m$ 分量。返回 `{(m1, m2, ..., mN): coefficient}`。

- `j_list`：各粒子的总角动量 `[j1, j2, ..., jN]`
- `target`：目标态 `[J_total, M_total]`
- `intermediate_Js`：N>2 时必须提供中间耦合路径以消除歧义

```python
from lqcddb import SU2decompose

# 分解 (J=1, M=1) → 两个自旋 1/2 粒子
result = SU2decompose(j_list=[0.5, 0.5], target=[1, 1])
# {(0.5, 0.5): 1.0} → 两个粒子必须都是 m=+1/2

# 分解 (J=1/2, M=1/2) → 三个自旋 1/2 粒子（需指定中间 J）
result = SU2decompose(
    j_list=[0.5, 0.5, 0.5],
    target=[0.5, 0.5],
    intermediate_Js=[1]     # 前两个粒子先耦合成 J=1
)
```

---

## 7. 算符共轭（`contraction/baroperator.py`）

### `conjugate_operator(tokens)` — 强子算符厄米共轭

对强子算符做厄米共轭变换。自动识别介子（2 夸克）、重子（3 夸克）和一般多夸克结构，正确计算 γ 矩阵链的 H/T/C 变换和 Fermi 符号。

**支持的 γ 结构**：`"1"`, `"gamma_5"`, `"gamma_0"` ~ `"gamma_3"`, `"gamma_mu"`, `"gamma_5 * gamma_mu"`, `"sigma_mu_nu"`, `"C"`, `"C * gamma_5"`, `"C * gamma_mu"`, `"C * gamma_5 * gamma_mu"`, `"C * sigma_mu_nu"`

```python
from lqcddb import conjugate_operator

# π⁺ = ū γ₅ d 的厄米共轭 → π⁻ = d̄ γ₅ u
op_pi_plus  = ["|", "u^d", "gamma_5", "d", "|"]
op_pi_minus = conjugate_operator(op_pi_plus)
# 返回: [1, "|", "d^d", "gamma_5", "u", "|"]

# 质子的厄米共轭
proton     = ["|", "u", "u", "C * gamma_5", "d", "|"]
proton_bar = conjugate_operator(proton)
# 返回包含正确符号和反夸克标记的列表
```

**返回值格式**：`[sign, token1, token2, ...]`，其中 `sign` 为 ±1（已包含 Fermi 符号和 diquark 转置对称性符号）。

**辅助函数**（可按需从模块直接导入）：

| 函数 | 用途 |
|------|------|
| `is_separator(x)` | 判断是否 `'\|'` 分隔符 |
| `is_gamma(x)` | 判断是否已知 γ 结构 |
| `is_quark(x)` | 判断是否夸克场（非分隔符、非 γ） |
| `dagger_quark(q)` | 切换 `^d` 标记：`'u'` ↔ `'u^d'` |
| `split_hadrons(tokens)` | 按 `'\|'` 分隔符拆分为独立强子 |
| `classify_structure(body)` | 按夸克数分类：`'meson'`（2q）、`'baryon'`（3q）、`'generic'` |
| `hermitian_gamma_chain(gammas)` | 计算 γ 链的厄米共轭 |
| `transpose_gamma(gamma_expr)` | γ 结构的转置符号 |
| `charge_conjugation_gamma(gamma_expr)` | γ 结构的电荷共轭符号 |
| `diquark_symmetry(gamma_expr)` | 返回 diquark 转置对称性符号 $\eta$：$(C\Gamma)^T = \eta (C\Gamma)$ |

---

## 8. 统计分析（`analyse/analyse.py`）

所有统计函数均通过 `get_backend()` 支持 numpy/cupy 双后端。

### `Jackknife` — Jackknife 重采样

单点消除 Jackknife。第 $k$ 个样本去掉第 $k$ 个组态：

$$\text{sample}_k = -\frac{\sum_{\text{all}} - \text{data}_k}{N_{\text{conf}} - 1}$$

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | `ndarray` | | 输入数据，至少包含组态轴 |
| `Nconf_axes` | `int` | `0` | 组态所在的轴编号 |
| `only_sample` | `bool` | `False` | 若 `True`，仅返回样本不计算统计量 |
| `cov_axes` | `int` 或 `tuple` 或 `None` | `None` | 需要构建协方差矩阵的轴 |

**返回值：**

| 键 | 形状 | 说明 |
|----|------|------|
| `data_sample` | 与输入相同 | Nconf 个 Jackknife 样本 |
| `data_mean` | 去掉 Nconf_axes | $\text{mean}(\text{data}, \text{axis}=\text{Nconf\_axes})$ |
| `data_err` | 同 data_mean | $\sqrt{N_{\text{conf}}-1} \times \text{std}(\text{samples})$ |
| `data_cov` | `shape_other + shape_cov + shape_cov` | 仅当 `cov_axes` 不为 `None` |

```python
from lqcddb import Jackknife

# 基本用法
jk = Jackknife(corr_data, Nconf_axes=0)
mean, err = jk['data_mean'], jk['data_err']

# 沿指定轴构建协方差矩阵
jk = Jackknife(data, Nconf_axes=0, cov_axes=1)
cov = jk['data_cov']  # 形状: (other_dims..., dim_cov, dim_cov)
```

---

### `Bootstrap` — Bootstrap 重采样

有放回随机重采样。从 $N_{\text{conf}}$ 个组态中有放回地抽取 $M$ 个，重复 $N$ 次。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `M` | `int` | `max(Nconf-5, 1)` | 每个 Bootstrap 样本的组态数 |
| `N` | `int` | `Nconf * 4` | Bootstrap 样本总数 |
| 其余参数 | | | 同 `Jackknife` |

返回值键与 `Jackknife` 相同。`data_sample` 形状为 `(N, *sample_shape)`（第一个轴为 Bootstrap 样本）。

```python
from lqcddb import Bootstrap

boot = Bootstrap(data, Nconf_axes=0, N=500, cov_axes=1)
```

---

### `loop_tsrc` — 源位置平均

对所有 $t_{\text{src}}$ 求和，将 $(t_{\text{src}}, t_{\text{sink}})$ 映射为 $\tau = (t_{\text{sink}} - t_{\text{src}}) \bmod N_t$。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | `ndarray` | | 输入数据 |
| `indx` | `list` | `[-2, -3]` | `[tsink轴, tsrc轴]`，两轴大小必须相等 |
| `Boundary_Conditions` | `'Periodic' \| 'Antiperiodic'` | `'Periodic'` | 边界条件，反周期在 $t_{\text{sink}} < t_{\text{src}}$ 时翻转符号 |
| `Ctype` | `'2pt' \| '3pt'` | `'2pt'` | 关联函数类型 |
| `t_sep` | `int` | `0` | 源-汇间隔（仅 3pt 使用） |

```python
from lqcddb import loop_tsrc

# 2pt 函数源平均
corr_avg = loop_tsrc(corr_data, indx=[-2, -3], Ctype='2pt')
# 输入: (Nconf, Nt, Nt) → 输出: (Nconf, Nt, 1)

# 3pt 函数源平均
corr_avg_3pt = loop_tsrc(corr_3pt, indx=[-2, -3], Ctype='3pt', t_sep=8)
```

---

### `meff` — 有效质量

在 Jackknife/Bootstrap 样本上计算有效质量。

| `meff_type` | 公式 | 有效范围 |
|-------------|------|---------|
| `'log'` | $\frac{1}{a}\ln\frac{C(t)}{C(t+1)}$ | $t \in [0, N_t-2)$ |
| `'cosh'` | $\frac{1}{a}\text{arccosh}\frac{C(t+2)+C(t)}{2C(t+1)}$ | $t \in [0, N_t-3)$ |
| `'GEVP'` | 同 log，作用于 GEVP 特征值 | $t \in [0, N_t-2)$ |

**参数：**

| 参数 | 说明 |
|------|------|
| `data_sample` | Jackknife/Bootstrap 样本数组 |
| `alttc` | 格距（fm） |
| `Nconf_axes` | 组态轴编号（默认 0） |
| `Nt_axes` | 时间轴编号（默认 1） |
| `meff_type` | `'log'`、`'cosh'` 或 `'GEVP'` |

返回 `{'data_sample': ..., 'data_mean': ..., 'data_err': ...}`。

```python
from lqcddb import meff

jk = Jackknife(corr_real, Nconf_axes=0)
eff = meff(jk['data_sample'], alttc=0.12, Nt_axes=1, meff_type='log')
# eff['data_mean'][t] — 有效质量平均值 (GeV)
# eff['data_err'][t]  — 误差 (GeV)
```

**注意**：`data_sample` 的 `dtype` 必须为 `float`（不能是 complex）。对复关联函数，先取实部或虚部再传入。

---

### `solve_gevp` — 广义本征值问题

求解 $C(t) v_n = \lambda_n(t, t_0) C(t_0) v_n$，使用 `scipy.linalg.eigh`。

**参数：**
- `C`：关联矩阵，形状 `(N_op, N_op, Nt)`
- `t0`：参考时间片

**返回：**
- `eigenvalues`：`(N_op, Nt)`，已排序
  - $t < t_0$：升序（$\lambda > 1$）
  - $t \ge t_0$：降序（$\lambda \le 1$，最大特征值对应基态）
- `eigenvectors`：`(N_op, N_op, Nt)`（复数），相位归一化 $v^\dagger v = 1$

矩阵 $C$ 在求解前会被对称化：$(C + C^\dagger)/2$。

```python
from lqcddb import solve_gevp

C = np.load("corr_matrix.npy")  # (3, 3, 64)
eigvals, eigvecs = solve_gevp(C, t0=2)
# eigvals[0, :] → 基态特征值
# eigvals[1, :] → 第一激发态特征值
```

---

### `PDF` — 3pt/2pt 比值（部分子分布函数）

计算比值 $R(t_{\text{src}}, t_{\text{curr}}) = C_3(t_{\text{src}}, t_{\text{curr}}) / C_2(t_{\text{src}}, t_{\text{src}}+t_{\text{sep}})$，然后对所有 $t_{\text{src}}$ 取平均。

- `link_fold=True`：折叠 link 方向，将 $[+\text{link\_max}, -\text{link\_max}]$ 对折为 $[0, \text{link\_max}]$
- 返回 `{'data_sample': ..., 'data_mean': ..., 'data_err': ...}`

```python
from lqcddb import PDF

pdf = PDF(data_3pt_sample, data_2pt_sample, t_sep=8, link_fold=True)
```

---

### `Mom2GeV` — 动量→能量转换

将格点动量转换为 GeV 单位的能量：

$$E = \sum_i \sqrt{\left(\frac{2\pi}{N_x} \cdot \frac{\text{fm2GeV}}{a}\right)^2 \cdot \mathbf{p}^2 + M_{0,i}^2}$$

- `Mom`：标量 $p^2$、列表 `[pz, py, px]`、或二维动量列表
- `M0`：标量质量或质量列表（多质量时能量求和）

```python
from lqcddb import Mom2GeV

# 单粒子能量
E = Mom2GeV(Nx=32, alttc=0.12, Mom=[0, 0, 1], M0=0.5)

# 多粒子系统总能量
E_total = Mom2GeV(Nx=32, alttc=0.12, Mom=[0, 0, 1], M0=[0.5, 0.5, 0.3])
```

---

### `dis_connect` — 非连通图贡献

计算 bubble 图对 2pt 关联函数的非连通贡献。

- `dtype='PDF'`：仅计算 $C_2^{\text{bubble}}(t_{\text{sep}}) \times C_2^{\mu\nu}$
- `dtype='PFF'`：同时计算两个交叉项

```python
from lqcddb import dis_connect

result = dis_connect(data_2pt_sample, data_bubble_sample,
                     Nconf_axes=0, t_src_axes=1, t_sink_axes=2,
                     tsep=8, dtype='PDF')
```

---

### `sum_over_array_of_list` — 分组求和

按显式指定的索引分组，对数组沿指定轴求和。每个组的索引被求和为一个输出索引。

```python
from lqcddb import sum_over_array_of_list

# axis=1: 大小3 → 2组（{0,2}→0, {1}→1）
# axis=2: 大小4 → 2组（{0,3}→0, {1,2}→1）
result = sum_over_array_of_list(
    arr, axes=(1, 2),
    groupings=[[[0, 2], [1]], [[0, 3], [1, 2]]]
)
# arr.shape: (a, 3, 4, b) → result.shape: (a, 2, 2, b)
```

函数会验证：无重复索引、所有索引均被覆盖。

---

### `get_data_info` — 数据维度信息

根据数据类型（real/complex）和时间类型（`'t_src'`/`'all_t'`）返回组态数、时间长度和组态轴位置。

```python
from lqcddb import get_data_info

Nconf, Nt, Nconf_axes = get_data_info(data, dtype=complex, ttype='t_src')
```

### 绘图辅助常量

```python
from lqcddb import plot_analyse_marker, plot_analyse_color

# 12 种标记和颜色，循环使用
for i in range(n_states):
    plt.errorbar(t, mean[i], yerr=err[i],
                 marker=plot_analyse_marker[i % 12],
                 color=plot_analyse_color[i % 12])
```

---

## 9. 基础工具（`base/base_functions.py`）

### `cached_contract` — 带缓存的张量缩并

对 `opt_einsum.contract` 的封装，加入路径缓存。首次遇到某个表达式+形状+优化策略的组合时，编译缩并路径并缓存；后续相同签名的调用直接命中缓存，跳过编译开销。**推荐在所有缩并代码中使用此函数替代裸 `contract()`。**

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `einsum_str` | `str` | 缩并字符串，如 `'ab,bc->ac'` |
| `*tensors` | `array_like` | 参与缩并的张量 |
| `optimize` | `str \| bool \| list[str]` | 路径优化策略 |

**`optimize` 参数的三种用法：**

| 值 | 行为 |
|----|------|
| `'auto'`（默认） | 使用 opt_einsum 的 `'auto'` 策略 |
| `True` | 自动尝试 `'auto'`, `'greedy'`, `'optimal'`, `'dp'` 四种策略，比较 FLOPs 和最大中间张量大小，选最优的缓存 |
| `list[str]` | 手动指定候选策略列表，如 `['auto', 'greedy']` |

**缓存键**：`(einsum_str, shapes_tuple, opt_key)`。编译开销仅在首次调用时产生。

```python
from lqcddb import cached_contract, clear_cache

# 基本用法（替代 opt_einsum.contract）
result = cached_contract('ab,bc,cd->ad', A, B, C)

# 自动选择最优策略
result = cached_contract('ijk,kl,lj->il', A, B, C, optimize=True)

# 手动指定候选策略
result = cached_contract('ab,bc->ac', A, B, optimize=['auto', 'greedy'])

# 清空全局缓存（释放内存）
clear_cache()
```

---

### `levi_civita_tensor(n=3)`

生成 $n$ 维 Levi-Civita 完全反对称张量。形状 `(n,) * n`。

```python
from lqcddb import levi_civita_tensor

eps3 = levi_civita_tensor(3)  # (3, 3, 3)
eps3[0, 1, 2]  # = +1
eps3[1, 0, 2]  # = -1

eps4 = levi_civita_tensor(4)  # (4, 4, 4, 4)
```

---

### `creat_mom_list(Mom, fix_Q2=False, only_g0=False)`

从参考动量生成所有等价动量组合（含正负号和置换）。

- `Mom`：参考动量 `[pz_min, py_min, px_min]` 或范围列表 `[pz_min, pz_max]`
- `fix_Q2=True`：仅保留与参考动量具有相同 $Q^2$ 的组合
- `only_g0=True`：仅保留所有分量为非负的组合

```python
from lqcddb import creat_mom_list

# 生成 Q²=2 的所有动量
mom_list = creat_mom_list(Mom=[1, 1, 0], fix_Q2=True)
# [[-1,-1,0], [-1,0,-1], [-1,0,1], [-1,1,0], [0,-1,-1], ...]

# 仅非负分量
mom_list_pos = creat_mom_list(Mom=[1, 1, 0], fix_Q2=True, only_g0=True)
```

---

### `ArraySlicer` — 高级多维数组切片

基于 `np.ix_` 的高级切片和赋值工具。

```python
from lqcddb import ArraySlicer

arr = np.zeros((4, 5, 6))
slicer = ArraySlicer(arr)

# 读取切片
sub = slicer.slice(dims=[0, 2], indices=[[1, 3], [0, 2]])

# 赋值
slicer.assign(dims=[1], indices=[[0, 2]], values=[[1, 2], [3, 4]])

# 带 keep_dims 的赋值（保持指定维度的形状）
slicer.assign(dims=[1], indices=[[0, 2]], values=new_vals, keep_dims=[0])

# 查询切片后的形状
shape = slicer.get_slice_shape(dims=[0, 1], indices=[[1, 3], [0, 2]])

# 获取数组信息
info = slicer.get_info()  # {'shape': (4,5,6), 'ndim': 3, 'dtype': ...}
```

---

## 10. 带宽分析（`contraction/contractadviser.py`）

使用 Roofline 模型分析张量缩并操作的带宽瓶颈。判断收缩是计算瓶颈还是带宽瓶颈，并给出切分自由指标的建议。

### `analyze_bandwidth` — 完整带宽瓶颈分析

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `subscript` | `str` | | opt_einsum 风格下标，如 `'Mabc,Nabc->MN'` |
| `shapes` | `list[tuple]` | | 输入张量形状列表 |
| `hardware` | `HardwareSpec \| str` | `'A100_80GB'` | 硬件规格（预设名或对象） |
| `dtype` | `str` | `'complex128'` | 数据类型 |
| `optimize` | `str` | `'auto'` | 缩并路径优化策略 |
| `verbose` | `bool` | `True` | 是否打印详细分析结果 |

**返回值** `BandwidthAnalysis` 包含：

| 属性 | 说明 |
|------|------|
| `is_bandwidth_bound` | 是否为带宽瓶颈 |
| `bottleneck_severity` | `'none'` / `'mild'` / `'significant'` / `'severe'` |
| `suggestions` | `list[SlicingSuggestion]`，切分建议列表 |
| `upsizing_suggestions` | `list[UpsizingSuggestion]`，增大维度建议列表 |
| `estimated_compute_time` | 预计纯计算时间（秒） |
| `estimated_data_time` | 预计数据搬运时间（秒） |
| `estimated_total_time` | 预计总耗时（秒） |

每条 `SlicingSuggestion` 包含：

| 属性 | 说明 |
|------|------|
| `index` | 建议切分的指标名 |
| `size` | 该指标的总大小 |
| `suggested_chunk_size` | 推荐的每块大小 |
| `n_chunks` | 切分块数 |
| `working_set_per_chunk_bytes` | 每块的工作集大小 |
| `peak_memory_without_bytes` | 不切分时的峰值内存 |
| `peak_memory_with_bytes` | 切分后的峰值内存 |
| `reduction_in_reads` | 数据读取减少的倍数 |
| `rationale` | 切分理由（人类可读） |

```python
from lqcddb import analyze_bandwidth

# 分析一个缩并操作
result = analyze_bandwidth(
    'Mabc,Nabc->MN',
    [(2, 1000, 500, 64), (2, 1000, 500, 64)],
    hardware='A100_80GB',
    dtype='complex128'
)

# 遍历切分建议
for s in result.suggestions:
    print(f"切分指标 '{s.index}': 总大小={s.size}, "
          f"建议块大小={s.suggested_chunk_size}, "
          f"切分块数={s.n_chunks}")
    print(f"  峰值内存: {s.peak_memory_without_bytes/1e9:.1f} GB → "
          f"{s.peak_memory_with_bytes/1e9:.1f} GB")
    print(f"  理由: {s.rationale}")
```

---

### `quick_check` — 快速检查

`analyze_bandwidth` 的静默版本（`verbose=False`）。

```python
from lqcddb import quick_check

result = quick_check('ab,bc->ac', [(100, 500), (500, 1000)],
                     hardware='V100', dtype='complex128')
if result.is_bandwidth_bound:
    print("带宽瓶颈！")
```

---

### `printGPUinfo` — 查看可用 GPU 预设

```python
from lqcddb import printGPUinfo
printGPUinfo()
```

**可用硬件预设：**

| 预设名 | GPU | FP64 (TFLOPS) | 带宽 (GB/s) | 显存 |
|--------|-----|---------------|-------------|------|
| `V100` | NVIDIA V100 32GB | 7.8 | 900 | 32 GB |
| `A100_40GB` | NVIDIA A100 40GB | 9.7 | 1555 | 40 GB |
| `A100_80GB` / `A100` | NVIDIA A100 80GB | 9.7 | 2039 | 80 GB |
| `A800` | NVIDIA A800 80GB | 9.7 | 2039 | 80 GB |
| `H20` | NVIDIA H20 96GB | 1.0 | 4000 | 96 GB |
| `CPU_I72C512G` | 通用服务器 CPU | ~0.04 | ~2.8 | ~7 GB/核 |
| `CPU_CPU6248R` | Intel Xeon 6248R | ~0.06 | ~5.8 | ~8 GB/核 |

---

## 11. 文件 I/O（`io/write_date.py`）

以下两个读取函数**未通过顶层 `__init__.py` 导出**，需要从模块直接导入：

```python
from lqcddb.io.write_date import readin_eigvecs, readin_peram
```

### `readin_eigvecs(file_path, Nx)`

读取二进制格式的本征矢文件。内部存储为 `float64`，自动重组为复数。

- 输入：二进制文件路径 + 空间格点数 `Nx`
- 输出：`(Nev, Nx³, 3)` 复数数组

```python
eigvecs = readin_eigvecs("eigvecs.dat", Nx=32)
```

### `readin_peram(peram_dir, conf_id, Nt, Nev1)`

读取二进制格式的 perambulator 目录（每个 `t_source` 一个文件）。

- 输出：`(Nt, Nt, 4, 4, Nev1, Nev1)` 复数数组
- 维度顺序：`(t_source, t_sink, d_sink, d_source, ev_sink, ev_source)`

```python
peram = readin_peram("peram_dir/", conf_id="1000", Nt=64, Nev1=32)
```

### `write_data_ascii(data, T, L, filename, complex=True, verbose=False)`

以 L. Liu 格式写入 ASCII 文件。文件头包含 `nsamples T complex L 1`。

```python
from lqcddb import write_data_ascii

write_data_ascii(correlator, T=64, L=32, filename="output/corr.dat", complex=True)
```

### `check_dir_path(save_path)`

自动创建输出目录（等效 `mkdir -p`）。

```python
from lqcddb import check_dir_path

check_dir_path("output/result/")
```

---

## 12. MPI 并行（`base/mpi_init.py`）

### `mpinit` — 初始化 MPI 环境

同时初始化 MPI 网格和 CUDA 设备。

```python
from lqcddb import mpinit

mpinit(
    grid_size=[1, 1, 1, 4],       # MPI 网格 [Gx, Gy, Gz, Gt]
    latt_size=[32, 32, 32, 64],   # 格点尺寸 [Lx, Ly, Lz, Lt]
    backend='cupy',               # 'numpy' / 'cupy' / 'torch'
    device=-1,                    # GPU 设备编号，-1 为自动分配
    enable_mps=False              # 是否启用 MPS（多进程共享 GPU）
)
```

### `get_mpi_tlist(Nt, t, gtype='find')`

时间片到 MPI rank 的映射。分发模式为轮询：全局时间 $t \to$ rank $t \bmod \text{size}$。

- `gtype='find'`：查询时间 `t` 所在的 rank 和局部索引
- `gtype='TScatter'`：将时间列表分配到各 rank，返回本 rank 拥有的局部数据

```python
from lqcddb import get_mpi_tlist

# 查询 t=10 在哪个 rank
t_rank, t_local_idx = get_mpi_tlist(Nt=64, t=10, gtype='find')

# 获取本 rank 拥有的时间片
t_list, rank_list, t_local_indices = get_mpi_tlist(Nt=64, t=range(64), gtype='TScatter')
```

**全局↔局部映射**：$\text{local} = (\text{global\_t} - \text{rank}) / \text{size}$（含模 $N_t$ 回绕）。

### `get_mpi_data` — MPI 数据收发

自动处理 numpy/cupy 转换的 MPI 数据传输。

**支持的 `mdtype`：**

| mdtype | 行为 |
|--------|------|
| `'Send'` | 点对点发送（需指定 `recv_rank`） |
| `'Gather'` | 汇聚到 root，新轴插入到 `axis=0` |
| `'TGather'` | 转置汇聚：先汇聚再合并相邻维度 |
| `'Allgather'` | 全局收集，所有 rank 获取完整数据 |
| `'Bcast'` | 从 root 广播到所有 rank |
| `'Scatter'` | 从 root 均匀分散到各 rank |
| `'TScatter'` | 转置分散：处理不能均分的情况 |
| `'Transport'` | Gather → Scatter 转置传输 |

```python
from lqcddb import get_mpi_data, getMPIRank

rank = getMPIRank()

# 广播：从 rank 0 广播到所有 rank
data_all = get_mpi_data(data_local, mdtype='Bcast', root=0)

# 汇聚：收集各 rank 数据到 rank 0
data_gathered = get_mpi_data(data_local, mdtype='Gather', root=0)

# 转置分散：加载 peram 时常用（rank 0 读取，分散到各 rank）
if rank == 0:
    peram = load_peram(...)
else:
    peram = None
peram_local = get_mpi_data(peram, mdtype='TScatter', root=0, axis=0)
```

### 查询函数

```python
from lqcddb import (getMPIComm, getMPIRank, getMPISize,
                    getGridSize, getGridCoord,
                    getCUDABackend, getCUDADevice, isHIP)

rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()
```

---

## 13. Gauge Smearing（`base/smear_gauge.py`）

### `stout_smear_ndarray(gauge, nstep, rho)`

Stout 链 smearing：对空间方向的规范链做迭代指数映射。原地修改 `gauge` 数组。

- `gauge`：形状 `(Nd, Nt, Nz, Ny, Nx, Nc, Nc)`（注意 `Nd` 包含时间分量）
- `nstep`：smearing 迭代步数
- `rho`：smearing 参数（典型值 0.1）

```python
from lqcddb import stout_smear_ndarray

# 36 步 stout smearing
gauge = stout_smear_ndarray(gauge, nstep=36, rho=0.1)
```

---

## 14. 后端切换（`base/backend.py`）

```python
from lqcddb import set_backend, get_backend

set_backend('cupy')   # 切换到 GPU
set_backend('numpy')  # 切换到 CPU

xp = get_backend()
x = xp.ones((10, 10))  # 使用当前后端的数组操作
```

---

## 15. 完整工作流示例

### 介子 2pt 关联函数（从缩并到有效质量）

```python
from lqcddb import *
import numpy as np
import matplotlib.pyplot as plt

set_backend('numpy')

# 1. Wick 缩并
result = wick_contraction(
    sink_operators   = ["|", "u^d", "gamma_5", "d", "|"],
    source_operators = ["|", "u^d", "gamma_5", "d", "|"],
    curr_operators   = [],
    Cpt='2pt'
)

# 2. 查看缩并图（可选）
fig, ax = plot_figure_wick(result, diagram_index=0, Cpt='2pt')
fig.savefig("wick_pion.pdf")

# 3. 构建动量投影
vtx = vertex_creator(Nx=32)
phase = vtx.phase_exp_2pt(Mom=[0, 0, 0])
VdV = vtx.Mom_VdV_sink_t(phase_exp=phase, eigvecs=eigvecs)

# 4. 缩并计算关联函数
#    由 result['peram'], result['V'], result['gamma_pos'] 驱动
#    corr = contract(..., VdV, peram, ...)

# 5. 源位置平均
corr_avg = loop_tsrc(corr_data, indx=[-2, -3], Ctype='2pt')

# 6. Jackknife 重采样
jk = Jackknife(corr_avg.real, Nconf_axes=0)

# 7. 有效质量
eff = meff(jk['data_sample'], alttc=0.12, Nt_axes=1, meff_type='log')

# 8. 绘图
t = np.arange(len(eff['data_mean']))
plt.errorbar(t, eff['data_mean'], yerr=eff['data_err'],
             marker=plot_analyse_marker[0], color=plot_analyse_color[0],
             capsize=3, ls='none')
plt.xlabel('t')
plt.ylabel('$m_{\\rm eff}$ (GeV)')
plt.savefig("meff.pdf")
```

### 重子 GEVP 分析

```python
# 假设已有多个算符的关联矩阵 C (Nop, Nop, Nt, Nconf)
C = np.load("corr_matrix.npy")

# Jackknife 样本
C_jk = Jackknife(C, Nconf_axes=-1)['data_sample']

# 求解 GEVP
eigvals, eigvecs = solve_gevp(C_jk.mean(axis=-1), t0=2)

# 基态有效质量
eff_ground = meff(eigvals[0], alttc=0.12, meff_type='GEVP')

# 第一激发态
eff_excited = meff(eigvals[1], alttc=0.12, meff_type='GEVP')
```

### 质子→中子 3pt 函数（带宽分析 + 缩并）

```python
# 先分析缩并是否带宽瓶颈
from lqcddb import quick_check

# 典型缩并维度
result = quick_check(
    'eifj,ambn,cgdh,okpl,ce,Gmo,gi,Mbdf,Lnp,Nhjl,Pak->MNL',
    [(4, 4, 100, 100), (4, 4, 100, 400), (4, 4, 100, 100),
     (4, 4, 400, 100), (4, 4), (4, 4, 4), (4, 4),
     (3, 100, 100, 100), (5, 400, 400), (3, 100, 100, 100), (1, 4, 4)],
    hardware='A100_80GB', dtype='complex128'
)

if result.is_bandwidth_bound:
    print(f"瓶颈严重程度: {result.bottleneck_severity}")
    for s in result.suggestions:
        print(f"建议切分: {s.index}, 块大小={s.suggested_chunk_size}")
```

---

## 16. 强子算符与流算符

详见包内 `physics.md`，包含：

- **介子算符**：π⁺, π⁻, π⁰ 的蒸馏算符字符串
- **重子算符**：质子、中子、Λ 的蒸馏算符字符串
- **流算符**：标量、矢量、轴矢、张量流的插入形式及重子投影子
- **相互作用流**：电磁流 `['q^d', 'gamma_mu', 'q']`、弱流 `['u^d', 'gamma_w', 'd']`
- **规范组态参数表**：19 个组态的格距、体积、$m_\pi$、$m_\pi L$、$m_{\eta_s}$

常用重子投影子：
- 正宇称：$(\gamma_0 + \gamma_4)/2$
- 负宇称：$(\gamma_0 - \gamma_4)/2$

矢量流 4 分量构造：
```python
gamma_curr = (gamma(0) - gamma(5)) @ backend.asarray([gamma(1), gamma(2), gamma(3), gamma(4)])
# 形状: (4, 4, 4) — (component, spin, spin)
```
