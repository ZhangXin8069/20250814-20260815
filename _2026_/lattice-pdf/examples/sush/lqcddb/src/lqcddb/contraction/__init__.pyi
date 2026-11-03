"""
Type stub for lqcddb.contraction subpackage.

提供 Wick 收缩自动生成、算符共轭、带宽瓶颈分析和顺序传播子等功能。
"""
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════
# 内部 dataclass stubs — 仅用于类型标注，不导出到运行时
# ═══════════════════════════════════════════════════════════════════════════

class HardwareSpec:
    """硬件规格：存储各精度算力峰值，根据 dtype 自动选用。"""
    name: str
    peak_fp16: float
    peak_fp32: float
    peak_fp64: float
    memory_bandwidth_gbs: float
    total_memory_bytes: int
    l2_cache_bytes: int
    compute_efficiency: float
    memory_efficiency: float
    complex_compute_eff: float

class ParsedContraction:
    """解析后的收缩操作信息。"""
    subscript: str
    input_subs: List[str]
    output_subs: str
    shapes: List[Tuple[int, ...]]
    index_size: Dict[str, int]
    free_indices: Dict[str, int]
    contracted_indices: Dict[str, int]
    batch_indices: Dict[str, int]

class ContractionCost:
    """收缩操作的成本估算。"""
    total_flops: int
    total_read_bytes: int
    total_write_bytes: int
    arithmetic_intensity: float
    ideal_arithmetic_intensity: float
    output_size: int
    cache_penalty: float
    intermediate_write_bytes: int
    peak_working_bytes: int

class SlicingSuggestion:
    """对某个自由指标进行切分的建议。"""
    index: str
    size: int
    suggested_chunk_size: int
    n_chunks: int
    working_set_per_chunk_bytes: int
    peak_memory_without_bytes: int
    peak_memory_with_bytes: int
    reduction_in_reads: float
    rationale: str

class UpsizingSuggestion:
    """增加某个指标维度的建议。"""
    index: str
    index_type: str
    current_size: int
    target_size: int
    scale_factor: float
    current_ai: float
    target_ai: float
    data_increase_bytes: int
    rationale: str

class BandwidthAnalysis:
    """带宽瓶颈分析的完整结果。"""
    parsed: ParsedContraction
    cost: ContractionCost
    hardware: HardwareSpec
    is_bandwidth_bound: bool
    bottleneck_severity: str  # 'none' | 'mild' | 'significant' | 'severe'
    suggestions: List[SlicingSuggestion]
    upsizing_suggestions: List[UpsizingSuggestion]
    summary: str
    estimated_compute_time: float
    estimated_data_time: float
    estimated_total_time: float

# ═══════════════════════════════════════════════════════════════════════════
# 公共 API
# ═══════════════════════════════════════════════════════════════════════════

# --- baroperator.py ---

def conjugate_operator(operator: str) -> str:
    """返回给定强子算符的厄米共轭形式。

    自动处理介子 (2q)、重子 (3q) 和通用多夸克算符。

    参数:
        operator: 算符表达式的 token 列表字符串表示。
    """
    ...

# --- autowick.py ---

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
) -> Tuple[plt.Figure, plt.Axes]:
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

# --- seqperam.py ---

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

# --- contractadviser.py ---

def analyze_bandwidth(
    subscript: str,
    shapes: List[Tuple[int, ...]],
    hardware: Union[HardwareSpec, str] = "A100_80GB",
    dtype: str = "complex128",
    optimize: str = "auto",
    verbose: bool = True,
) -> BandwidthAnalysis:
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

# --- dynamic.py ---

class PeramRegistry:
    """传播子 (perambulator) 注册表。

    按夸克味和时间标签存储 peram 数组引用，不做复制。
    """

    def __init__(self) -> None:
        """初始化注册表。"""
        ...
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
        """按 Wick 条目的味和时间标签查找 peram。

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

    def __init__(self) -> None:
        """初始化注册表。"""
        ...
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
        """按名称和时间端查找 V 张量。

        参数:
            v_name: V 结构名称。
            time_label: 时间端标签。

        返回:
            匹配的 V 张量数组引用。
        """
        ...

class GammaRegistry:
    """Gamma 矩阵注册表。将算符中的 gamma 名称映射到复数数组，形状不限。"""

    def __init__(self) -> None:
        """初始化注册表。"""
        ...
    def register(self, name: str, data: np.ndarray) -> None:
        """注册一个 gamma 矩阵。

        参数:
            name: 算符中出现的 gamma 名称，如 ``'gamma_7'``。
            data: 复数数组，形状不限。
        """
        ...
    def resolve(self, name: str) -> np.ndarray:
        """按名称查找 gamma 矩阵。

        参数:
            name: gamma 矩阵名称。

        返回:
            gamma 矩阵数组引用。
        """
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
            operator_groups: 算符组列表，2pt 为 ``[(sink, src), ...]``，3pt 为 ``[(sink, src, curr), ...]``。
            peram_registry: 已注册 peram 数据的注册表。
            v_registry: 已注册 V 张量数据的注册表。
            gamma_registry: 已注册 gamma 矩阵数据的注册表。
            Cpt: 关联函数类型，``'2pt'``, ``'3pt'``, ``'4pt'`` 等。
            Pindex: peram 指标前缀。
            Vindex: V 结构指标前缀。
            Gindex: gamma 指标前缀。
            use_equivalence: 是否调用等价图检测。
            ignore_dis: 是否忽略 disconnected 图。
            verbose: 是否打印分析信息。
            max_detail: 每组显示单图详情的数量，-1 表示全部。
            plot: 若为非空，将全部 Wick 收缩图输出为多页 PDF。
                目录路径使用默认名 ``wick_contraction_fig.pdf``；
                否则视为文件路径（强制 ``.pdf`` 后缀）。
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

        返回:
            收缩结果（已乘总系数），``total_coeff==0`` 时返回 ``0``。
        """
        ...
    def calculate_all(
        self,
    ) -> Any:
        """计算所有收缩并求和，返回总关联函数。

        返回:
            所有收缩结果的加权和。
        """
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
    """运行 Wick 收缩分析，整合等价图检测，返回收缩计划。

    对 ``operator_groups`` 中的每组算符调用 :func:`wick_contraction`，
    可选调用 :func:`identify_equivalent_diagrams` 消除等价冗余图。

    参数:
        operator_groups: 算符组列表，2pt 为 ``[(sink, src), ...]``，3pt 为 ``[(sink, src, curr), ...]``。
        Cpt: 关联函数类型 (``'2pt'``, ``'3pt'``, ``'4pt'`` 等)。
        Pindex: peram 指标前缀。
        Vindex: V 结构指标前缀。
        Gindex: gamma 指标前缀。
        use_equivalence: 是否做等价图归并。
        ignore_dis: 是否忽略 disconnected 图。
        verbose: 是否打印分析信息。
        max_detail: 每组显示单图详情的数量，-1 表示全部。
        plot: 若为非空，将全部 Wick 收缩图输出为多页 PDF。
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

    若 ``entry[0]`` 中等价图系数之和为 0 则直接返回 ``0``；
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
