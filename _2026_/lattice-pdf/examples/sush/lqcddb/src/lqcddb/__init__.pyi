"""
lqcddb 类型存根 — IDE 自动补全与类型检查。

参照 numpy 做法，本文件将所有公开 API 的完整签名内联定义，
不依赖从子包的 re-export（Pylance 无法跟踪经过 ``__getattr__`` 的 re-export 链）。

- 修改函数签名时需同步更新本文件。
- 运行时仍由 ``__init__.py`` 的 ``__getattr__`` 按模块级懒加载。
- 子包 ``__init__.pyi`` 保持不变，供 ``from lqcddb.analyse import ...`` 等直接导入使用。

用法::

    from lqcddb import *
    set_backend('numpy')    # 或 'cupy'
    backend = get_backend()
"""
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
from matplotlib.pyplot import Figure, Axes

# 尝试导入 cupy 以提供更精确的后端类型（安装 cupy 时 IDE 可区分 numpy/cupy）
try:
    import cupy as cp
except ImportError:
    cp = np  # type: ignore[assignment]

# 版本号
__version__: str


# ═══════════════════════════════════════════════════════════════════════════════
# base — 后端管理
# ═══════════════════════════════════════════════════════════════════════════════

def get_backend() -> np:
    """返回当前计算后端模块 (numpy 或 cupy)。"""
    ...
def set_backend(backend: Literal["numpy", "cupy"]) -> None:
    """切换全局计算后端。

    参数:
        backend: 目标后端，``"numpy"`` 或 ``"cupy"``。
    """
    ...

# ═══════════════════════════════════════════════════════════════════════════════
# base — 张量工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def levi_civita_tensor(n: int = 3) -> np.ndarray:
    """生成 n 维全反对称 Levi-Civita 张量，形状 ``(n,)*n``。

    参数:
        n: 张量维度，默认为 3。
    """
    ...
def creat_mom_list(
    Mom: Union[List[int], List[List[int]]] = [0, 0, 0],
    fix_Q2: bool = False,
    only_g0: bool = False,
) -> List[List[int]]:
    """从各分量取值范围生成动量三元组列表。

    参数:
        Mom: 动量范围，如 ``[0, 0, [0, 1]]`` 表示 pz∈{0}、py∈{0}、px∈{0,±1}。
        fix_Q2: 仅保留 Q² 相同的三元组。
        only_g0: 仅保留非负分量。
    """
    ...

class ArraySlicer:
    """高级多维数组切片器，通过 ``np.ix_`` 网格实现读写操作。

    参数:
        array: 要包装的数组。
    """
    array: np.ndarray
    ndim: int
    shape: Tuple[int, ...]

    def __init__(self, array: np.ndarray) -> None:
        """初始化切片器。

        参数:
            array: 要包装的数组。
        """
        ...
    def get_slices(
        self, dims: List[int], indices: List[Any]
    ) -> Tuple[np.ndarray, ...]:
        """返回 ``np.ix_`` 风格的索引网格。"""
        ...
    def slice(self, dims: List[int], indices: List[Any]) -> np.ndarray:
        """按指定维度和索引读取子数组。"""
        ...
    def assign(
        self,
        dims: List[int],
        indices: List[Any],
        values: Union[int, np.ndarray],
        keep_dims: List[int] = [],
    ) -> np.ndarray:
        """按指定维度和索引写入值并返回修改后数组。

        参数:
            dims: 要写入的维度。
            indices: 每个维度的索引列表。
            values: 写入的值 (标量或数组)。
            keep_dims: 写入后保留的维度列表。
        """
        ...
    def get_slice_shape(
        self, dims: List[int], indices: List[Any]
    ) -> Tuple[int, ...]:
        """返回切片后的形状。"""
        ...
    def get_info(self) -> Dict[str, Any]:
        """返回 ``{shape, ndim, dtype}`` 字典。"""
        ...

def cached_contract(
    einsum_str: str,
    *tensors: np.ndarray,
    optimize: Union[str, bool, List[str]] = "auto",
) -> Any:
    """带缓存的 opt_einsum 张量收缩。

    首次调用时编译收缩路径并缓存，后续调用直接使用缓存的表达式对象跳过编译。

    参数:
        einsum_str: opt_einsum 风格收缩字符串，如 ``'ab,bc->ac'``。
        *tensors: 参与收缩的张量。
        optimize: 路径优化策略。

            - ``str``: 使用指定策略 (``'auto'``, ``'greedy'``, ``'optimal'``, ``'dp'``)。
            - ``True``: 自动尝试所有策略并缓存最优的。
            - ``List[str]``: 仅尝试指定的一组策略。

    返回:
        收缩结果张量。
    """
    ...
def clear_cache() -> None:
    """清空 ``cached_contract`` 的全局路径缓存。"""
    ...
def get_cache_keys() -> List[Tuple[str, Tuple[Tuple[int, ...], ...], Any]]:
    """返回当前 ``cached_contract`` 缓存中所有键的列表。

    每个键为 ``(einsum_str, shapes, opt_key)`` 三元组：

    - ``einsum_str``: 收缩字符串，如 ``'ab,bc->ac'``
    - ``shapes``: 参与收缩张量的形状元组
    - ``opt_key``: 优化策略标识（字符串或策略元组）

    返回:
        缓存键列表，按插入顺序排列。
    """
    ...

# ═══════════════════════════════════════════════════════════════════════════════
# base — SU(2) Clebsch-Gordan
# ═══════════════════════════════════════════════════════════════════════════════

def SU2combine(
    states: List[Tuple[Any, Any]],
) -> Dict[Tuple[Any, ...], Any]:
    """通过 Clebsch-Gordan 系数组合多个 SU(2) 态。

    参数:
        states: ``[(J1, M1), (J2, M2), ...]`` 形式的态列表。

    返回:
        ``{(J_total, M_total, intermediate_Js): coefficient}`` 字典。
    """
    ...
def SU2decompose(
    j_list: List[Any],
    target: List[Any],
    intermediate_Js: Optional[List[Any]] = None,
) -> Dict[Tuple[Any, ...], Any]:
    """将总 J 态分解为各个 m 分量。

    参数:
        j_list: 各粒子的 J 值列表。
        target: 目标 ``[J_total, M_total]``。
        intermediate_Js: N>2 时需要的中间 J 值，用于消除简并度。

    返回:
        ``{(m1, m2, ...): coefficient}`` 字典。
    """
    ...

# ═══════════════════════════════════════════════════════════════════════════════
# base — MPI 通信（需要 mpi4py）
# ═══════════════════════════════════════════════════════════════════════════════

def get_mpi_data(
    data: Any,
    mdtype: Literal[
        "Send", "Gather", "TGather", "Allgather",
        "Bcast", "Scatter", "TScatter", "Transport",
    ] = "Gather",
    root: int = 0,
    recv_rank: int = 0,
    recv_buff: Optional[np.ndarray] = None,
    axis: int = 0,
) -> Optional[Any]:
    """MPI 数据搬运，自动处理 numpy/cupy 格式。

    参数:
        data: 要传输的数据。
        mdtype: 传输模式。

            - ``"Send"``: 点到点发送
            - ``"Gather"``: 收集到根进程
            - ``"TGather"``: 转置收集
            - ``"Allgather"``: 全局收集
            - ``"Bcast"``: 广播
            - ``"Scatter"``: 分发
            - ``"TScatter"``: 转置分发
            - ``"Transport"``: 通用转置传输

        root: 根进程编号。
        recv_rank: 接收进程编号 (仅 ``"Send"`` 模式)。
        recv_buff: 接收缓冲区。
        axis: 分发/收集的轴。

    返回:
        传输后的数据，非接收方可能返回 ``None``。
    """
    ...
def get_mpi_tlist(
    Nt: int,
    t: Union[int, float, List[int], List[float], np.ndarray],
    gtype: Literal["find", "TScatter"] = "find",
) -> Any:
    """将时间片映射到 MPI 进程。

    参数:
        Nt: 时间方向总格点数。
        t: 要查询的时间 (全局索引)。
        gtype: 查询类型。

            - ``"find"``: 返回 ``(rank, local_index)``。
            - ``"TScatter"``: 返回 ``(t_list_local, rank_list, local_indices)``。

    返回:
        (rank, local_index) 或完整的 TScatter 映射。
    """
    ...
def mpinit(
    grid_size: List[int],
    latt_size: Optional[List[int]] = None,
    backend: Optional[Literal["numpy", "cupy", "torch"]] = None,
    device: int = -1,
    enable_mps: bool = False,
    cuda_device_count: Optional[int] = None,
) -> None:
    """初始化 MPI 网格和 CUDA 设备。

    参数:
        grid_size: MPI 网格，如 ``[1, 1, 1, 4]``。
        latt_size: 格子大小，如 ``[32, 32, 32, 64]``。
        backend: 计算后端。
        device: CUDA 设备编号，``-1`` 表示自动。
        enable_mps: 是否启用 MPS (Multi-Process Service)。
        cuda_device_count: CUDA 设备总数。
    """
    ...
def getMPIComm() -> Any:
    """返回 MPI 通信器。"""
    ...
def getMPIRank() -> int:
    """返回本进程 MPI rank。"""
    ...
def getMPISize() -> int:
    """返回 MPI 总进程数。"""
    ...

# ═══════════════════════════════════════════════════════════════════════════════
# base — 规范场平滑（需要 opt_einsum）
# ═══════════════════════════════════════════════════════════════════════════════

def stout_smear_ndarray(
    gauge: np.ndarray, nstep: int, rho: float
) -> np.ndarray:
    """Stout 规范场平滑。

    参数:
        gauge: 规范链接，形状 ``(Nd, Nz, Ny, Nx, Nc, Nc)``。
        nstep: 平滑迭代步数。
        rho: 平滑参数 ρ。

    返回:
        平滑后的规范场，形状相同。
    """
    ...


# ═══════════════════════════════════════════════════════════════════════════════
# constant — 物理常数
# ═══════════════════════════════════════════════════════════════════════════════

Nc: int  # 颜色数 (3)
Ns: int  # 自旋分量数 (4)
Nd: int  # 时空维数 (4)
fm2GeV: float  # 转换因子: 1 fm⁻¹ = 0.197 GeV

# ═══════════════════════════════════════════════════════════════════════════════
# constant — Dirac gamma 矩阵
# ═══════════════════════════════════════════════════════════════════════════════

def gamma(i: int) -> np.ndarray:
    """返回第 i 号 Dirac gamma 矩阵 (DeGrand-Rossi 基)，形状 ``(4, 4)``。

    = ======  ==========
    i 矩阵    说明
    = ======  ==========
    0 I₄      单位矩阵
    1 γ₁      gamma_1
    2 γ₂      gamma_2
    3 γ₃      gamma_3
    4 γ₄      gamma_4
    5 γ₅      diag(1,1,-1,-1)
    6 γ₂γ₃
    7 γ₃γ₁    C*γ₅ (= gamma_7)
    8 γ₁γ₂
    9 γ₁γ₄
    10 γ₂γ₄
    11 γ₃γ₄
    12-15 γ_{1-4}γ₅
    16 (γ₃γ₁)(1+γ₄)/2
    17 (γ₃γ₁)(1-γ₄)/2
    = ======  ==========
    """
    ...
def tran_indx_to_gamma(
    indx: Union[int, List[int], np.ndarray],
) -> np.ndarray:
    """将 gamma 指标数组转换为 gamma 矩阵张量。

    参数:
        indx: gamma 指标，如 ``[5, 4]``。
    返回:
        形状 ``(len(indx), 4, 4)`` 的 gamma 矩阵堆叠。
    """
    ...
def PFF_Mom_to_gamma_new(
    Mom: List[List[int]], allow_t: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """将动量映射为 Levi-Civita 张量缩并后的 gamma 指标组合 (投影形状因子)。

    参数:
        Mom: 动量三元组列表。
        allow_t: 是否允许时间方向动量。

    返回:
        ``(gamma_indx_matrix, gamma_matrix, gamma_indx_all, gamma_matrix_all)``。
    """
    ...

# ═══════════════════════════════════════════════════════════════════════════════
# constant — Pauli sigma 矩阵
# ═══════════════════════════════════════════════════════════════════════════════

def sigma(i: int) -> np.ndarray:
    """返回第 i 号 Pauli 矩阵，形状 ``(2, 2)``。

    = ======  ==========
    i 矩阵    说明
    = ======  ==========
    0 I₂      单位矩阵
    1 σ_x     Pauli X
    2 σ_y     Pauli Y
    3 σ_z     Pauli Z
    = ======  ==========
    """
    ...
def Mom_times_sigma(
    Mom: List[int] = [0, 0, 0],
    upto4dim: bool = False,
) -> np.ndarray:
    """计算 **p·σ** = p_x σ_x + p_y σ_y + p_z σ_z。

    参数:
        Mom: 动量 ``[pz, py, px]``。
        upto4dim: 是否嵌入为 4×4 块对角矩阵。
    """
    ...
def Mom_cross_sigma(
    Mom: List[int] = [0, 0, 0],
    upto4dim: bool = False,
) -> np.ndarray:
    """计算 **p×σ** = ε_{ijk} p_j σ_k，即动量与 Pauli 矩阵的叉积。

    参数:
        Mom: 动量 ``[pz, py, px]``。
        upto4dim: 是否嵌入为 4×4 块对角矩阵。

    返回:
        ``p×σ`` 张量，形状 ``(3, 2, 2)``，若 ``upto4dim`` 则为 ``(3, 4, 4)``。
        分量轴 (大小 3) 对应 Z, Y, X。
    """
    ...


# ═══════════════════════════════════════════════════════════════════════════════
# contraction — Wick 收缩
# ═══════════════════════════════════════════════════════════════════════════════

def conjugate_operator(operator: str) -> str:
    """返回给定强子算符的厄米共轭形式。

    自动处理介子 (2q)、重子 (3q) 和通用多夸克算符。

    参数:
        operator: 算符表达式的 token 列表字符串表示。
    """
    ...

def wick_contraction(
    sink_operators: List[str],
    source_operators: List[str],
    curr_operators: List[str] = [],
    Cpt: Literal["bubble", "2pt", "3pt", "4pt"] = "2pt",
    Pindex: Optional[List[Any]] = None,
    Vindex: Optional[List[Any]] = None,
    Gindex: Optional[List[Any]] = None,
) -> Union[Dict, List[Dict]]:
    """自动 Wick 收缩生成器。

    对 N 粒子的算符 (用 ``'|'`` 分隔) 自动生成所有合法的费米子配对图，
    并附带置换符号。

    参数:
        sink_operators: sink 端算符 token 列表。
        source_operators: source 端算符 token 列表。
        curr_operators: 流插入算符 token 列表 (2pt 时为空列表)。
        Cpt: 关联函数类型。

            - ``"bubble"``: 单粒子 bubble 图。
            - ``"2pt"``: 两点函数。
            - ``"3pt"``: 三点函数。
            - ``"4pt"``: 四点函数。

        Pindex: peram 标签前缀列表。``None`` 或 ``[]`` 均视为空字符串。
        Vindex: VVV/VDV 标签前缀列表。
        Gindex: gamma 标签前缀列表。

    返回:
        不含通配符时返回 ``dict``，含通配符 (``'q'``/``'l'``) 时返回 ``list[dict]``。
        每个 dict 包含:

        ==================  ============================================
        Key                  说明
        ==================  ============================================
        ``result_indx``     ``['contraction_string->free_indices']``
        ``result_name``     ``['component_name_1, ...']``
        ``result_sign``     每个图的符号 (含费米子反对称 × 整体系数)
        ``operators``       解析后的完整算符
        ``quark_pos``       ``(位置, 味, 收缩标签)`` 列表
        ``sep_pos``         ``'|'`` 分隔符位置
        ``gamma_pos``       ``(位置, gamma名, 组合指标, 时间标签)``
        ``V``               VVV/VDV 顶点条目
        ``peram``           每个图的 peram 配对列表
        ==================  ============================================
    """
    ...

def plot_figure_wick(
    result_dict: Dict,
    diagram_index: int = 0,
    Cpt: Literal["2pt", "3pt", "4pt"] = "2pt",
    plot_text: bool = True,
) -> Tuple[Figure, Axes]:
    """绘制 Wick 收缩图。

    自动布局，包含彩色夸克节点 (按味道)、传播子弧线、gamma 矩阵虚线、
    VVV/VDV 顶点标签、粒子分隔虚线和时间方向箭头。

    参数:
        result_dict: ``wick_contraction`` 返回的 dict。
        diagram_index: 要绘制的图序号。
        Cpt: 关联函数类型。
        plot_text: 是否显示文本标签。

    返回:
        ``(matplotlib Figure, Axes)`` 元组，可用于 ``fig.savefig("wick.pdf")``。
    """
    ...

def identify_equivalent_diagrams(*dicts: Dict) -> List[List[int]]:
    """识别并分组等价 Wick 收缩图。

    使用 Union-Find 比较排列后的 peram 配对，考虑 gamma 插入的夸克交换。

    参数:
        *dicts: ``wick_contraction`` 返回的多个 dict。

    返回:
        等价的图索引分组，如 ``[[0, 2], [1]]`` 表示第 0 和第 2 个图等价。
    """
    ...

# ═══════════════════════════════════════════════════════════════════════════════
# contraction — 动态收缩框架
# ═══════════════════════════════════════════════════════════════════════════════

class PeramRegistry:
    """传播子 (perambulator) 注册表。

    按夸克味和时间标签存储 peram 数组引用，不做复制。
    """

    def __init__(self) -> None: ...
    def register(
        self, flavor: str, time_labels: Tuple[str, str], data: np.ndarray
    ) -> None:
        """注册一个 peram 数组。同名键重复注册会覆盖旧值。

        参数:
            flavor: 夸克味，``'u'``, ``'d'``, ``'s'`` 或 ``'light'``。
            time_labels: ``(t_quark, t_antiquark)``。夸克时间在前，反夸克时间在后。
            data: peram 数组，至少 4 维。仅存引用不复制。
        """
        ...
    def resolve(
        self, combined_type: str, time_labels: List[str]
    ) -> np.ndarray:
        """按 Wick 条目的味和时间标签查找 peram（内部使用）。

        参数:
            combined_type: Wick 输出的组合味字符串。
            time_labels: 时间标签对 ``[t_q, t_aq]``。

        返回:
            匹配的 peram 数组引用。
        """
        ...

class VRegistry:
    """V 顶点张量注册表。

    按 Wick V 结构名称和时间端存储 V 数组引用，不做复制。
    """

    def __init__(self) -> None: ...
    def register(
        self, v_name: str, time_label: str, data: np.ndarray
    ) -> None:
        """注册一个 V 张量。

        参数:
            v_name: V 结构名称，如 ``'VVV_0'``, ``'VDV_0'``。
            time_label: 时间端，``'tsink'``, ``'tsrc'``, ``'tcur0'`` 等。
            data: V 张量数组，第一维为动量。仅存引用不复制。
        """
        ...
    def resolve(self, v_name: str, time_label: str) -> np.ndarray:
        """按名称和时间端查找 V 张量（内部使用）。"""
        ...

class GammaRegistry:
    """Gamma 矩阵注册表。将算符中的 gamma 名称映射到复数数组，形状不限。"""

    def __init__(self) -> None: ...
    def register(self, name: str, data: np.ndarray) -> None:
        """注册一个 gamma 矩阵。

        参数:
            name: 算符中出现的 gamma 名称，如 ``'gamma_7'``。
            data: 复数数组，形状不限。
        """
        ...
    def resolve(self, name: str) -> np.ndarray:
        """按名称查找 gamma 矩阵（内部使用）。"""
        ...

class dynamic_contraction:
    """动态 Wick 收缩计算器。

    初始化时自动完成 Wick 分析、等价图检测、纠错和注册校验，
    收缩计算在后续调用 :meth:`calculate` 或 :meth:`calculate_all` 时按需执行。
    """

    plan: List[Any]
    """同 :func:`run_wick_analysis` 的返回值。"""
    missing: List[Any]
    """校验缺失项列表，空表示全部通过。"""

    def __init__(
        self,
        operator_groups: List[Tuple],
        *,
        peram_registry: PeramRegistry,
        v_registry: VRegistry,
        gamma_registry: GammaRegistry,
        Cpt: str = "2pt",
        Pindex: Optional[List[str]] = None,
        Vindex: Optional[List[str]] = None,
        Gindex: Optional[List[str]] = None,
        use_equivalence: bool = False,
        ignore_dis: bool = True,
        verbose: bool = True,
        max_detail: int = -1,
        plot: str = "",
        Projection: bool = False,
        optimize: Union[str, bool, List[str]] = "auto",
    ) -> None:
        """初始化收缩计算器并运行 Wick 分析。

        参数:
            operator_groups: 算符组列表。2pt: ``[(sink_op, src_op), ...]``. 3pt: ``[(sink_op, src_op, curr_op), ...]``
            peram_registry: 已注册 peram 数据的注册表。
            v_registry: 已注册 V 张量数据的注册表。
            gamma_registry: 已注册 gamma 矩阵数据的注册表。
            Cpt: 关联函数类型 (``'2pt'``, ``'3pt'``, ``'4pt'`` 等)。
            Pindex: peram 指标前缀。
            Vindex: V 结构指标前缀。
            Gindex: gamma 指标前缀。
            use_equivalence: 是否调用等价图检测。
            ignore_dis: 是否忽略 disconnected 图。
            verbose: 是否打印分析信息。
            max_detail: 每组显示单图详情的数量，-1 表示全部。
            plot: 若为非空，将全部 Wick 收缩图输出为多页 PDF。
                强制 ``.pdf`` 后缀。默认 ``''`` (不绘图)。
            Projection: 若为 ``True``，将输出自旋指标通过 ``'Projector'`` 收缩。
            optimize: 传递给 :func:`cached_contract` 的优化策略。
        """
        ...
    def calculate(
        self, index: int,
    ) -> Any:
        """计算 plan 中第 ``index`` 个条目的收缩。

        参数:
            index: plan 中的条目索引。
        """
        ...
    def calculate_all(
        self,
    ) -> Any:
        """计算所有收缩并求和，返回总关联函数。"""
        ...
    def __len__(self) -> int:
        """返回 plan 条目数。"""
        ...
    def __getitem__(self, index: int) -> List[Any]:
        """返回 plan 第 ``index`` 个条目。"""
        ...

def clear_plan_cache() -> None:
    """清空分析缓存，下次调用 :func:`run_wick_analysis` 将重新分析并输出。"""
    ...

def run_wick_analysis(
    operator_groups: List[Tuple],
    *,
    Cpt: str = "2pt",
    Pindex: Optional[List[str]] = None,
    Vindex: Optional[List[str]] = None,
    Gindex: Optional[List[str]] = None,
    use_equivalence: bool = False,
    ignore_dis: bool = True,
    verbose: bool = True,
    max_detail: int = -1,
    plot: str = "",
    peram_registry: Optional[PeramRegistry] = None,
    v_registry: Optional[VRegistry] = None,
    gamma_registry: Optional[GammaRegistry] = None,
    optimize: Union[str, bool, List[str]] = "auto",
) -> List[List[Any]]:
    """运行 Wick 收缩分析，整合等价图检测，返回统一格式的收缩计划。

    对 ``operator_groups`` 中的每一组算符调用 :func:`wick_contraction`，
    收集所有 Wick 收缩结果，可选调用 :func:`identify_equivalent_diagrams`
    消除等价冗余图，最终输出可直接供 :func:`calculate_contraction` 使用的多维列表。

    参数:
        operator_groups: 算符组列表。
            2pt: ``[(sink_op, src_op), ...]``
            3pt: ``[(sink_op, src_op, curr_op), ...]``
        Cpt: 关联函数类型 (``'2pt'``, ``'3pt'``, ``'4pt'`` 等)。
        Pindex: peram 指标前缀。
        Vindex: V 结构指标前缀。
        Gindex: gamma 指标前缀。
        use_equivalence: 是否做等价图归并。
        ignore_dis: 是否忽略 disconnected 图。
        verbose: 是否打印分析信息。
        max_detail: 每组显示单图详情的数量。
        plot: 若为非空，将全部 Wick 收缩图输出为多页 PDF。
            强制 ``.pdf`` 后缀。默认 ``''`` (不绘图)。
        peram_registry: 已注册 peram 数据的注册表，提供后输出收缩路径分析。
        v_registry: 已注册 V 张量数据的注册表。
        gamma_registry: 已注册 gamma 矩阵数据的注册表。
        optimize: 收缩路径优化策略，仅当三个 registry 均非 None 时生效。

    返回:
        每行为 ``[equiv_list, contraction_idx, wick_dict, diag_idx]`` 的列表。
    """
    ...

def calculate_contraction(
    entry: List[Any],
    *,
    peram_registry: PeramRegistry,
    v_registry: VRegistry,
    gamma_registry: GammaRegistry,
    optimize: Union[str, bool, List[str]] = "auto",
    Projection: bool = False,
) -> Any:
    """将 :func:`run_wick_analysis` 输出的一行解析为张量并执行收缩。

    先检查 ``entry[0]`` 中所有等价图的系数之和：若为 0 则直接返回 ``0``；
    否则调用 :func:`cached_contract` 执行收缩并将结果乘以总系数。

    参数:
        entry: :func:`run_wick_analysis` 返回的 plan 中的一行。
        peram_registry: 已注册 peram 数据的注册表。
        v_registry: 已注册 V 张量数据的注册表。
        gamma_registry: 已注册 gamma 矩阵数据的注册表。
        optimize: 传递给 :func:`cached_contract` 的优化策略。
        Projection: 若为 ``True``，将输出自旋指标通过 ``'Projector'`` 收缩。

    返回:
        收缩结果乘以总系数后的张量，或 ``0``。
    """
    ...

# ═══════════════════════════════════════════════════════════════════════════════
# contraction — 顺序传播子
# ═══════════════════════════════════════════════════════════════════════════════

def seq_peram(peram: np.ndarray) -> np.ndarray:
    """γ₅-厄米共轭: ``γ₅ · P* · γ₅`` (共轭，非共轭转置)。

    将夸克 peram 转换为反夸克 peram。
    **注意**: 变换后本征矢量指标顺序交换 (ev_src, ev_sink)。

    参数:
        peram: perambulator 数组 (6D 或 5D)。

    返回:
        γ₅ 共轭后的 perambulator，同形状。
    """
    ...

# ═══════════════════════════════════════════════════════════════════════════════
# contraction — 带宽分析（需要 opt_einsum）
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_bandwidth(
    subscript: str,
    shapes: List[Tuple[int, ...]],
    hardware: Any = "A100_80GB",
    dtype: str = "complex128",
    optimize: str = "auto",
    verbose: bool = True,
) -> Any:
    """分析张量收缩的带宽瓶颈，给出切分建议 (Roofline 模型)。

    参数:
        subscript: opt_einsum 风格下标，如 ``'Mabc,Nabc->MN'``。
        shapes: 每个输入张量的形状，如 ``[(100, 64, 64, 64), (200, 64, 64, 64)]``。
        hardware: 硬件规格或预设名称。

            预设名称: ``V100``, ``A100_40GB``, ``A100_80GB``, ``A800``, ``H20``,
            ``I72C512G``, ``CPU6248R``, ``CPUEICC``。

        dtype: 数据类型。

            ``'float16'``, ``'float32'``, ``'float64'``, ``'complex64'``, ``'complex128'``。

        optimize: opt_einsum 路径优化策略。

            ``'optimal'``, ``'greedy'``, ``'auto'``, ``'dp'``, ``'branch-2'``, ``'branch-all'``。

        verbose: 是否打印详细分析过程。

    返回:
        ``BandwidthAnalysis`` 包含瓶颈判断、切分建议、增维建议和时间估算。

    示例::

        result = analyze_bandwidth('Mabc,Nabc->MN', [(2, 1000, 500, 64), (2, 1000, 500, 64)])
        for s in result.suggestions:
            print(f"建议切分 '{s.index}': 每块 {s.suggested_chunk_size}")
    """
    ...

def printGPUinfo() -> None:
    """打印可用的 GPU 硬件预设及其规格列表。"""
    ...


# ═══════════════════════════════════════════════════════════════════════════════
# analyse — 动量转换
# ═══════════════════════════════════════════════════════════════════════════════

def Mom2GeV(
    Nx: int,
    alttc: float,
    Mom: Union[float, List[float], List[List[float]]],
    M0: Union[float, List[float]],
) -> Union[float, List[float]]:
    """将格点动量转换为真实能量 (GeV)。

    公式: ``E = Σᵢ √((2π/Nx · fm2GeV/alttc)² · p² + M0ᵢ²)``

    参数:
        Nx: 格子空间方向长度。
        alttc: 格距 (fm)。
        Mom: 动量输入。

            - 标量: 直接用作动量模平方。
            - ``[px, py, pz]``: 计算 ``sum(pᵢ²)``。
            - ``[[...], ...]``: 对每个子列表计算模平方，返回结果列表。

        M0: 质量项。

            - 标量: ``E = √(single_Q2²·p² + M0²)``。
            - 列表: ``E = Σᵢ √(single_Q2²·p² + M0ᵢ²)``。

    返回:
        转换后的能量 (GeV)。类型取决于 Mom 和 M0 的组合。
    """
    ...

# ═══════════════════════════════════════════════════════════════════════════════
# analyse — 源时间循环
# ═══════════════════════════════════════════════════════════════════════════════

def loop_tsrc(
    data: np.ndarray,
    indx: List[int] = [-2, -3],
    Boundary_Conditions: Literal["Periodic", "Antiperiodic"] = "Periodic",
    Ctype: Literal["2pt", "3pt"] = "2pt",
    t_sep: int = 0,
) -> np.ndarray:
    """对关联函数在 t_src 上进行循环平移累加，将 (t_src, t_sink) 映射为 τ = t_sink - t_src。

    参数:
        data: 输入数据数组，至少包含 t_src 和 t_sink 两个轴。
        indx: ``[t_src_axis, t_sink_axis]``，长度必须为 2。
        Boundary_Conditions: 边界条件。

            - ``"Periodic"``: 周期边界。
            - ``"Antiperiodic"``: 反周期边界，t_sink < t_src 时翻转符号。

        Ctype: 关联函数类型 (``"2pt"`` 或 ``"3pt"``)。
        t_sep: 3pt 函数中 source-sink 的时间间隔。

    返回:
        循环累加后的数据数组，自动保持输入类型 (numpy/cupy)。
    """
    ...

# ═══════════════════════════════════════════════════════════════════════════════
# analyse — 分组聚合
# ═══════════════════════════════════════════════════════════════════════════════

def sum_over_array_of_list(
    arr: np.ndarray,
    axes: Union[Tuple[int, ...], List[int]],
    groupings: List[List[List[int]]],
) -> np.ndarray:
    """按指标分组对指定轴进行求和聚合。

    每个 ``groupings`` 中的子列表定义一组要加和的原始指标。
    聚合后轴大小等于分组数。

    参数:
        arr: 输入数组 (任意形状)。
        axes: 要聚合的轴 (0-based)。
        groupings: 每个轴的分组列表，每个分组是原始指标索引的列表。
            所有指标必须恰好覆盖一次。

    返回:
        聚合后的数组。被聚合的轴替换为对应的分组数。

    示例::

        a = backend.arange(24).reshape(2, 3, 4)
        axes = (1, 2)
        groupings = ([[0, 2], [1]], [[0, 3], [1, 2]])
        sum_over_array_of_list(a, axes, groupings).shape  # → (2, 2, 2)
    """
    ...

def mean_over_array_of_list(
    arr: np.ndarray,
    axes: Union[Tuple[int, ...], List[int]],
    groupings: List[List[List[int]]],
) -> np.ndarray:
    """按指标分组对指定轴求均值聚合。与 ``sum_over_array_of_list`` 接口一致，
    但使用均值替代求和。

    参数:
        arr: 输入数组 (任意形状)。
        axes: 要聚合的轴 (0-based)。
        groupings: 每个轴的分组列表，每个分组是原始指标索引的列表。
            所有指标必须恰好覆盖一次。

    返回:
        聚合后的数组。被聚合的轴替换为对应的分组数。

    示例::

        a = backend.arange(24).reshape(2, 3, 4)
        axes = (1, 2)
        groupings = ([[0, 2], [1]], [[0, 3], [1, 2]])
        mean_over_array_of_list(a, axes, groupings).shape  # → (2, 2, 2)
    """
    ...

# ═══════════════════════════════════════════════════════════════════════════════
# analyse — 重采样 (Jackknife / Bootstrap)
# ═══════════════════════════════════════════════════════════════════════════════

def Jackknife(
    data: np.ndarray,
    Nconf_axes: int = 0,
    only_sample: bool = False,
    cov_axes: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Dict[str, np.ndarray]:
    """单消除 Jackknife 重采样。

    第 k 个样本省略第 k 个组态: ``sample_k = -(Σ_all - data_k) / (Nconf - 1)``

    参数:
        data: 输入数据数组，至少包含组态轴。
        Nconf_axes: 组态所在轴编号 (默认 0)。
        only_sample: 若 ``True``，仅返回 ``{'data_sample'}``。
        cov_axes: 构建协方差矩阵的轴 (单个 int 或 tuple)。``None`` 时不计算协方差。

    返回:
        ==============  ==================================================
        Key              说明
        ==============  ==================================================
        ``data_sample`` Jackknife 样本，形状与输入相同
        ``data_mean``   数据均值 (去掉组态轴)
        ``data_err``    标准误差 ``√(Nconf-1) × std(samples)``
        ``data_cov``    协方差矩阵 (仅当 ``cov_axes is not None``)
        ==============  ==================================================
    """
    ...

def Bootstrap(
    data: np.ndarray,
    Nconf_axes: int = 0,
    only_sample: bool = False,
    cov_axes: Optional[Union[int, Tuple[int, ...]]] = None,
    M: int = 0,
    N: int = 0,
) -> Dict[str, np.ndarray]:
    """有放回 Bootstrap 重采样。

    第 0 个样本为全部 Nconf 个组态的无放回抽取 (即原始数据均值)，
    其余 N-1 个样本为有放回随机抽取 M 个组态的均值。

    参数:
        data: 输入数据数组，至少包含组态轴。
        Nconf_axes: 组态所在轴编号 (默认 0)。
        only_sample: 若 ``True``，仅返回 ``{'data_sample'}``。
        cov_axes: 构建协方差矩阵的轴。``None`` 时不计算协方差。
        M: 每个 Bootstrap 样本 (i>=1) 抽取的组态数 (默认 ``max(Nconf - 5, 1)``)。
        N: Bootstrap 样本总数 (默认 ``Nconf × 4``)。

    返回:
        ==============  ==================================================
        Key              说明
        ==============  ==================================================
        ``data_sample`` Bootstrap 样本，形状 ``(N, ...)``。
                        ``data_sample[0]`` 为全组态均值，
                        ``data_sample[1:]`` 为有放回重采样均值。
        ``data_mean``   样本均值
        ``data_err``    样本标准差
        ``data_cov``    协方差矩阵 (仅当 ``cov_axes is not None``)
        ==============  ==================================================

    示例::

        boot = Bootstrap(data, Nconf_axes=0, N=500)
        mean, err = boot['data_mean'], boot['data_err']
        # boot['data_sample'][0]  == 原始数据均值
        # boot['data_sample'][1:] == N-1 个 Bootstrap 重采样均值
    """
    ...

# ═══════════════════════════════════════════════════════════════════════════════
# analyse — 有效质量
# ═══════════════════════════════════════════════════════════════════════════════

def meff(
    data_sample: np.ndarray,
    alttc: float,
    Nconf_axes: int = 0,
    Nt_axes: int = 1,
    meff_type: Literal["log", "cosh", "GEVP"] = "log",
) -> Dict[str, np.ndarray]:
    """从关联函数提取有效质量。

    参数:
        data_sample: 关联函数样本数据 (dtype 必须为 ``float``)。
        alttc: 格距 (fm)。
        Nconf_axes: 组态轴编号。
        Nt_axes: 时间轴编号。
        meff_type: 提取方法。

            =========== ==========================================  ================
            类型         公式                                        有效范围
            =========== ==========================================  ================
            ``"log"``   ``ln(C(t)/C(t+1)) × fm2GeV/alttc``         t ∈ [0, Nt-2)
            ``"cosh"``  ``arccosh((C(t+2)+C(t))/(2C(t+1))) × ...`` t ∈ [0, Nt-3)
            ``"GEVP"``  同 log，作用于 GEVP 特征值                   t ∈ [0, Nt-2)
            =========== ==========================================  ================

    返回:
        ``{'data_sample', 'data_mean', 'data_err'}``。
    """
    ...

# ═══════════════════════════════════════════════════════════════════════════════
# analyse — ratio 比值 / GEVP / Disconnected
# ═══════════════════════════════════════════════════════════════════════════════

def ratio_3pt(
    data_3pt_sample: np.ndarray,
    data_2ptI_sample: np.ndarray,
    data_2ptF_sample: Optional[np.ndarray] = None,
    t_sep: int = 12,
    Nconf_axes: int = 0,
    tau_axes: int = -1,
    t_sink_axes: int = -1,
    t_src_axes: Optional[int] = None,
    link_axes: Optional[int] = None,
    link_fold: bool = False,
) -> Dict[str, np.ndarray]:
    """计算三点函数与两点函数的比值。
        R = C₃ / C₂^F(t_sep) × √[C₂^I(t_sep-τ) C₂^F(τ) C₂^F(t_sep) / (C₂^F(t_sep-τ) C₂^I(τ) C₂^I(t_sep))]

    支持一维模式（t_src_axes=None）和二维模式（t_src_axes 不为 None）。
    初末态粒子相同时自动退化，sqrt 修正项恒为 1。

    参数:
        data_3pt_sample:  三点 Jackknife 样本，C₃。
        data_2ptI_sample: 初态两点 Jackknife 样本，C₂^I。
        data_2ptF_sample: 末态两点 Jackknife 样本，C₂^F。若为 None 则用 data_2ptI_sample。
        t_sep: 固定的源-汇时间间隔。
        Nconf_axes: Jackknife 样本所在的轴。
        tau_axes: data_3pt_sample 中算子插入时间 τ 所在的轴。
        t_sink_axes: 两点数据中汇时间所在的轴。
        t_src_axes: 源时间轴。提供时启用二维模式。
        link_axes: link 插入方向轴，用于折叠。
        link_fold: 是否在计算比值前对 link 轴做折叠。

    返回:
        ``{'data_sample', 'data_mean', 'data_err'}``。

    示例::

        # 一维模式
        r = ratio_3pt(C3, C2I, data_2ptF_sample=C2F, t_sep=10)
        # 二维模式 + link 折叠
        r = ratio_3pt(C3, C2I, data_2ptF_sample=C2F, t_sep=10,
                    tau_axes=3, t_sink_axes=3, t_src_axes=2,
                    link_axes=1, link_fold=True)
    """
    ...

def solve_gevp(
    C: np.ndarray,
    t0: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """求解广义特征值问题 (GEVP): ``C(t) v_n = λ_n(t,t₀) C(t₀) v_n``。

    使用 ``scipy.linalg.eigh`` 求解。对称化输入矩阵以确保厄米性。

    参数:
        C: 关联函数矩阵，形状 ``(N, N, Nt)``，N 为插值场数目。
        t0: 参考时间切片。

    返回:
        ``(eigenvalues, eigenvectors)`` 元组。

        - eigenvalues: 形状 ``(N, Nt)``。
          - t < t₀: 升序排列
          - t ≥ t₀: 降序排列 (最大特征值对应基态)
        - eigenvectors: 形状 ``(N, N, Nt)``
    """
    ...

def dis_connect(
    data_2pt_sample: np.ndarray,
    data_bubble_sample: np.ndarray,
    Nconf_axes: int,
    t_src_axes: int,
    t_sink_axes: int,
    tsep: int,
    dtype: Literal["PFF", "PDF"] = "PDF",
) -> np.ndarray:
    """计算 bubble 图对 2pt 关联函数的 disconnected 贡献。

    参数:
        data_2pt_sample: 2pt 关联函数样本。
        data_bubble_sample: Bubble 图样本。
        Nconf_axes: 组态轴编号。
        t_src_axes: t_src 轴编号。
        t_sink_axes: t_sink 轴编号。
        tsep: 时间间隔。
        dtype: 扣除类型。

            - ``"PDF"``: 仅一项扣除。
            - ``"PFF"``: 两项扣除 (前向+后向)。

    返回:
        disconnected 贡献数组。
    """
    ...

# ═══════════════════════════════════════════════════════════════════════════════
# analyse — 绘图辅助常量
# ═══════════════════════════════════════════════════════════════════════════════

plot_analyse_marker: List[str]
"""12 种 matplotlib 标记: ``['s','*','+','x','p','h','v','X','D','P','H','o']``。"""

plot_analyse_color: List[str]
"""12 种十六进制颜色码，用于绘图时区分不同数据组。"""


# ═══════════════════════════════════════════════════════════════════════════════
# eigvectors — 本征矢量操作
# ═══════════════════════════════════════════════════════════════════════════════

class vector_creator:
    """本征矢量创建、归一化、正交化和压缩工具。

    处理形状 ``(Nev, Nz, Ny, Nx, Nc)`` 的本征矢量，支持 4 种压缩方案 (V1-V4)。
    """

    def __init__(self) -> None:
        """初始化，获取当前计算后端。"""
        ...

    def inner_product(
        self,
        init_vector: np.ndarray,
        test_vector: np.ndarray,
        dtype: Literal["", "abs"] = "",
    ) -> np.ndarray:
        """计算两个本征矢量集合的内积矩阵。

        参数:
            init_vector: 初始矢量集，形状 ``(N₁, V)``。
            test_vector: 测试矢量集，形状 ``(N₂, V)``。
            dtype: ``""`` 返回复数内积，``"abs"`` 返回模平方。

        返回:
            内积矩阵，形状 ``(N₁, N₂)``。
        """
        ...

    def check(
        self,
        eigvecs: np.ndarray,
        dtype: Literal["find", "print"] = "find",
        tol: float = 1e-10,
        check_normal: bool = True,
    ) -> bool:
        """验证本征矢量的归一性和正交性 (V†V ≈ I)。

        参数:
            eigvecs: 本征矢量数组。
            dtype: ``"find"`` 返回 ``True/False``；``"print"`` 打印详细信息。
            tol: 容差阈值。
            check_normal: 是否同时检查归一性 (对角线=1)。

        返回:
            通过检查返回 ``True``，否则 ``False``。
        """
        ...

    def normal(self, vectors: np.ndarray) -> np.ndarray:
        """归一化本征矢量: ``v / |v|``。

        参数:
            vectors: 输入矢量数组。

        返回:
            归一化后的矢量数组。
        """
        ...

    def orthnormal(
        self,
        vectors_init: np.ndarray,
        vector: np.ndarray,
    ) -> np.ndarray:
        """将新矢量 Gram-Schmidt 正交归一化后追加到现有矢量集。

        参数:
            vectors_init: 现有的正交归一矢量集。
            vector: 要正交化并追加的新矢量。

        返回:
            拼接后的矢量数组，第一维增加 1。
        """
        ...

    def creat_noise(
        self,
        vectors_init: np.ndarray,
        N: int,
        dtype: Literal["complex", "float"] = "complex",
    ) -> np.ndarray:
        """生成 N 个随机噪声矢量，正交归一化后追加到现有集合。

        参数:
            vectors_init: 现有的正交归一矢量集。
            N: 要生成的噪声矢量数量。
            dtype: ``"complex"`` 生成复噪声，``"float"`` 生成实噪声。

        返回:
            拼接后的矢量数组，第一维增加 N。
        """
        ...

    def compress_matrix_V1(
        self,
        eigenvectors: np.ndarray,
        N_eigen: List[int] = [0],
        N_sum: List[int] = [0],
        Ctype: str = "I",
    ) -> np.ndarray:
        """压缩矩阵 V1: 求和基压缩。

        参数:
            eigenvectors: 输入本征矢量。
            N_eigen: 各组压缩前的本征矢数量列表。
            N_sum: 各组压缩后的矢量数量列表。
            Ctype: 压缩模式。

                - ``"I"`` (interlace): 均匀交错分组后求平均。
                - ``"B"`` (block): 块状分组。
                - ``"BI"`` (block-interlace): 第一维 block，第二维 interlace。

        返回:
            压缩后的本征矢量。
        """
        ...

    def compress_matrix_V2(
        self,
        eigenvectors: np.ndarray,
        N_eigen: List[int] = [0],
        N_sum: List[int] = [0],
        N_extract: List[int] = [0],
        Ctype: str = "I",
    ) -> np.ndarray:
        """压缩矩阵 V2: 随机抽取基压缩。

        每组中随机抽取 ``N_extract`` 个本征矢量，每次抽取不重复。

        参数:
            eigenvectors: 输入本征矢量。
            N_eigen: 各组压缩前的本征矢数量列表。
            N_sum: 各组压缩后的矢量数量列表。
            N_extract: 每组中每组随机抽取个数。
            Ctype: 压缩模式 (``"I"``, ``"B"``, ``"BI"``)。

        返回:
            压缩后的本征矢量。
        """
        ...

    def compress_matrix_V3(
        self,
        eigenvectors: np.ndarray,
        N_eigen: List[int] = [0],
        N_sum: List[int] = [0],
        N_extract: List[int] = [1],
        Ctype: str = "I",
        adjcent: bool = False,
    ) -> np.ndarray:
        """压缩矩阵 V3: 正交随机投影压缩。

        生成随机正交矢量，投影到本征子空间。

        参数:
            eigenvectors: 输入本征矢量。
            N_eigen: 各组压缩前的本征矢数量列表。
            N_sum: 各组压缩后的矢量数量列表。
            N_extract: 每组提取的随机矢量数。
            Ctype: 压缩模式。
            adjcent: 是否在邻接分组中采样。

        返回:
            压缩后的本征矢量。
        """
        ...

    def compress_matrix_V4(
        self,
        eigenvectors: np.ndarray,
        N_eigen: List[int] = [0],
        N_sum: List[int] = [0],
        N_extract: List[int] = [1],
        Ctype: Literal["I", "B", "BI"] = "I",
        adjcent: bool = False,
        random_type: Literal["orthnormal", "Z_N"] = "orthnormal",
    ) -> np.ndarray:
        """压缩矩阵 V4: V3 的扩展版本，支持可配置的随机矢量生成。

        参数:
            eigenvectors: 输入本征矢量。
            N_eigen: 各组压缩前的本征矢数量列表。
            N_sum: 各组压缩后的矢量数量列表。
            N_extract: 每组提取的随机矢量数。
            Ctype: 压缩模式。
            adjcent: 是否在邻接分组中采样。
            random_type: 随机矢量类型。

                - ``"orthnormal"``: 正交归一随机矢量。
                - ``"Z_N"``: Z_N 噪声矢量 (N 由字符串后数字指定，如 ``"Z_4"``)。

        返回:
            压缩后的本征矢量。
        """
        ...

# ═══════════════════════════════════════════════════════════════════════════════
# eigvectors — 顶点函数工具
# ═══════════════════════════════════════════════════════════════════════════════

class vertex_creator:
    """顶点函数创建工具。

    处理动量投影 (VVV/VdV)、相位因子、规范链接 VdV
    和 omega 加速权重等操作。
    """

    def __init__(self, Nx: int) -> None:
        """初始化 vertex_creator。

        参数:
            Nx: 格子空间方向长度 (假设各向同性 Nx=Ny=Nz)。
        """
        ...

    def check(
        self,
        eigvecs: np.ndarray,
        dtype: str = "find",
        tol: float = 1e-10,
        check_normal: bool = True,
    ) -> str:
        """检查本征矢量的归一性和正交性。

        参数:
            eigvecs: 本征矢量数组。
            dtype: ``"find"`` 或 ``"print"``。
            tol: 容差阈值。
            check_normal: 是否检查归一性。

        返回:
            检查结果字符串 (``"orth"`` 或 ``"don't orth"``)。
        """
        ...

    def normal(self, vectors: np.ndarray) -> np.ndarray:
        """归一化本征矢量 (最后 4 维: Nz, Ny, Nx, Nc)。

        参数:
            vectors: 输入矢量数组。

        返回:
            归一化后的矢量数组。
        """
        ...

    def src_sink_MPI_tran(
        self,
        src_sink: np.ndarray,
        mpi_size: int,
        trtype: Literal["forward", "backward"] = "forward",
    ) -> np.ndarray:
        """MPI 转置: 在 MPI 进程间分离或重组时间维。

        参数:
            src_sink: 源/汇数据数组。
            mpi_size: MPI 进程数。
            trtype: ``"forward"`` 将时间维切分到各进程；``"backward"`` 重组。

        返回:
            转置后的数组。
        """
        ...

    def perm_comb(
        self,
        N: float,
        M: int = 1,
        dtype: Literal["perm", "comb"] = "perm",
        renormal: bool = False,
    ) -> float:
        """计算排列数 P(N,M) 或组合数 C(N,M)。

        参数:
            N: 总数。
            M: 选取数。
            dtype: ``"perm"`` 排列数，``"comb"`` 组合数。
            renormal: 归一化模式。

        返回:
            排列数或组合数。
        """
        ...

    def create_omega_accelerate(
        self,
        exact: int = 0,
        N_eigen: List[int] = [0],
        N_sum: List[int] = [0],
        N_extract: List[int] = [0],
        noise: int = 0,
        conserved: bool = False,
        normal: bool = False,
        fixed_first_pos: List[int] = [-1],
        dim: int = 2,
    ) -> np.ndarray:
        """创建任意维度 (2D/3D/4D) 的 Ω 稀释加速权重张量。

        参数:
            exact: 精确 (未压缩) 本征矢的数量。
            N_eigen: 各 block 压缩前的本征矢数量列表。
            N_sum: 各 block 压缩后的本征矢数量列表。
            N_extract: 各 block 中提取的本征矢数量。
            noise: 噪声矢量数量。
            conserved: 是否守恒模式 (dim 固定为 2)。
            normal: 是否归一化权重矩阵。
            fixed_first_pos: 固定第一个指标的位置。
            dim: 输出张量维度 (2/3/4)。

        返回:
            Ω 权重张量 (complex)，形状取决于 dim 和 fixed_first_pos。
        """
        ...

    def phase_exp_2pt(
        self,
        Mom: List[int] = [0, 0, 0],
    ) -> np.ndarray:
        """生成 2pt 函数的动量相因子 ``exp(-i p·x)``。

        sink 和 source 使用相同相因子。广播到颜色维。
        Mom 顺序: ``[pz, py, px]``。

        参数:
            Mom: 动量三元组。

        返回:
            相因子数组，形状 ``(Nx, Nx, Nx, Nc)``。
        """
        ...

    def phase_exp_3pt(
        self,
        Mom: List[int] = [0, 0, 0],
    ) -> np.ndarray:
        """生成 3pt 函数的动量相因子 (仅 sink 投影)。

        展平为一维: ``exp(-i p·x)``，x 遍历所有格点。Mom 顺序: ``[pz, py, px]``。

        参数:
            Mom: 动量三元组。

        返回:
            相因子数组，形状 ``(Nx*Nx*Nx,)`` 展平。
        """
        ...

    def VdV_sink_t_link(
        self,
        eigvecs: np.ndarray,
        link_dir: str,
        link_max: int,
        phase_exp: np.ndarray,
        gauge_link: Union[np.ndarray, bool],
        t: int = 0,
        eigvecs_min: Optional[np.ndarray] = None,
        conserved: bool = False,
    ) -> np.ndarray:
        """计算 sink 端带规范链接的 **V†·D·V** 关联函数。

        参数:
            eigvecs: 本征矢量，形状 ``(Nev, Nz, Ny, Nx, 3)``。
            link_dir: 链接方向。``'0'`` 无链接，``'T'`` 时间方向，
                ``'X'``/``'Y'``/``'Z'`` 空间单方向，``'all'`` 所有空间方向求和。
            link_max: 空间链接的最大位移长度。
            phase_exp: 动量相因子 ``exp(-i p·x)``。
            gauge_link: 规范链接数组或 ``False``。
            t: 时间片。
            eigvecs_min: 第二组本征矢量 (仅守恒流需要)。
            conserved: 是否守恒流计算模式。

        返回:
            VDV 关联函数矩阵。
        """
        ...

    def Mom_VdV_sink_t(
        self,
        phase_exp: np.ndarray,
        eigvecs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """计算 **V†·diag(phase)·V** (介子 sink/source 元)。

        ``VdV[mom, i, j] = Σ_x V†_i(x) · exp(-ipx) · V_j(x)``

        参数:
            phase_exp: 动量相因子，形状 ``(num_Mom, V_full)``。
            eigvecs: 本征矢量，形状 ``(Nev, Nz, Ny, Nx, Nc)``。

        返回:
            VdV 数组，形状 ``(num_Mom, Nev, Nev)``。
        """
        ...

    def Mom_VVV_sink_t(
        self,
        phase_exp: np.ndarray,
        eigvecs: np.ndarray,
    ) -> np.ndarray:
        """计算 **ε_abc V_a V_b V_c · exp(-ipx)** (重子 sink/source 元)。

        包含 6 种颜色置换 (3 偶 + 3 奇带负号) 的 Levi-Civita 收缩。

        参数:
            phase_exp: 动量相因子，形状 ``(num_Mom, Nz, Ny, Nx)``。
            eigvecs: 本征矢量，形状 ``(Nev, Nz, Ny, Nx, Nc)``。

        返回:
            VVV 数组，形状 ``(num_Mom, Nev, Nev, Nev)``。
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# io — 数据读写
# ═══════════════════════════════════════════════════════════════════════════════

def write_data_ascii(
    data: np.ndarray,
    T: int,
    L: int,
    filename: str,
    complex: bool = True,
    verbose: bool = False,
) -> None:
    """以 L. Liu 格式写入 ASCII 数据文件。

    首行为 ``nsamples T complex L 1`` 头部，后跟数据行。
    复数时实部和虚部分列输出。

    参数:
        data: 输入数据数组，第一维为样本。
        T: 时间方向格点数。
        L: 半时间格点数。
        filename: 输出文件路径。
        complex: ``True`` 时写入复数 (实部+虚部)，``False`` 时仅实数。
        verbose: 是否打印详细写入信息。
    """
    ...
def check_dir_path(path: str) -> str:
    """检查并创建目录路径 (``mkdir -p``)。

    参数:
        path: 目录路径字符串。
    """
    ...
def safe_save(
    file: Union[str, "pathlib.Path"],
    arr: np.ndarray,
    allow_pickle: bool = True,
    fix_imports: bool = True,
    fallback_dirs: Optional[List[str]] = None,
) -> str:
    """将数组保存为 NumPy ``.npy`` 二进制文件，出错时自动回退到备用目录。

    用法与 ``numpy.save`` 一致。先尝试保存到主路径；若失败（如磁盘满、
    权限不足），依次尝试 ``fallback_dirs``（用户指定）；若仍然失败，
    自动构建 ``/nexdata/project/lqcd/${USER}/result/`` 下的路径再试。

    参数:
        file: 目标文件路径。若不以 ``.npy`` 结尾，自动追加后缀。
        arr: 待保存的数组数据。
        allow_pickle: 是否允许使用 pickle 保存对象数组。
        fix_imports: 仅用于 Python 2 兼容。
        fallback_dirs: 用户指定的备用目录列表，优先级高于自动回退。

    返回:
        实际保存成功的文件路径。

    异常:
        OSError: 所有路径（主 + 用户备用 + 自动回退）都保存失败时抛出。
    """
    ...
