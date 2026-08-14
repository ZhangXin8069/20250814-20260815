"""
格点 QCD 动态收缩计算
=================================

可复用的注册表类和收缩函数，用户按 Wick 输出名称严格对应注册数据。

核心流程 (推荐使用 :class:`dynamic_contraction`):

1. 注册数据 — ``PeramRegistry``, ``VRegistry``, ``GammaRegistry``
2. ``dynamic_contraction(operator_groups, registries...)`` — 自动完成分析、纠错、校验
3. ``dc.calculate(i)`` 或 ``dc.calculate_all()`` — 按需执行收缩

内存
----
建议传入视图（切片）而非副本以减少显存占用。
"""

from ..base import cached_contract, getMPIRank, getMPIComm
from ..contraction.autowick import wick_contraction, identify_equivalent_diagrams


def _log(*args, **kwargs):
    """MPI-aware print: only rank 0 outputs, always flushed."""
    if getMPIRank() == 0:
        kwargs.setdefault('flush', True)
        print(*args, **kwargs)


def _abort(msg=''):
    """Print error on ALL ranks (prefixed with rank id), flush, then abort."""
    import sys
    rank = getMPIRank()
    if msg:
        print(f'⛔ FATAL [rank={rank}]: {msg}', flush=True)
    sys.stderr.flush()
    sys.stdout.flush()
    getMPIComm().Abort(1)

# ═══════════════════════════════════════════════════════════════
# 纠错工具
# ═══════════════════════════════════════════════════════════════

def _check_not_none(data, what):
    """检查数据不为 None，否则抛出 ValueError。

    Parameters
    ----------
    data : array_like or None
        待检查的数据。
    what : str
        错误消息中描述该数据的名称。

    Raises
    ------
    ValueError
        如果 data 为 None。
    """
    if data is None:
        raise ValueError(f'{what} 不能为 None。请检查是否忘记注册或数组未加载。')

def _check_min_ndim(data, min_ndim, what):
    """检查数组最小维度，不足时抛出 ValueError。

    Parameters
    ----------
    data : ndarray
        待检查的数组。
    min_ndim : int
        要求的最小维度数。
    what : str
        错误消息中描述该数据的名称。

    Raises
    ------
    ValueError
        如果 data.ndim < min_ndim。
    """
    if data.ndim < min_ndim:
        raise ValueError(
            f'{what} 维度不足: 至少需要 {min_ndim} 维, '
            f'实际 shape={data.shape}。')


# ═══════════════════════════════════════════════════════════════
# 数据注册表: Peram
# ═══════════════════════════════════════════════════════════════

class PeramRegistry:
    """传播子 (perambulator) 注册表。

    按夸克味和时间标签存储 peram 数组引用，不做复制。
    同名键重复注册会覆盖旧值。用户负责管理数组生命周期。

    时间标签命名
    ------------
    ``'tsink'``  汇端时间片
    ``'tsrc'``   源端时间片
    ``'tcur0'``  第 0 个流插入点 (3pt)；多点函数依次为 ``'tcur1'``, ``'tcur2'``, ...
    格式 ``(t_quark, t_antiquark)``，夸克时间在前，反夸克时间在后。

    Parameters
    ----------
    无。初始化后通过 ``register`` 方法添加数据。

    Examples
    --------
    >>> reg = PeramRegistry()
    >>> reg.register('light', ('tsink', 'tsrc'), peram)       # 2pt 正向
    >>> reg.register('light', ('tsrc', 'tsink'), peram_rev)   # 2pt 反向
    >>> reg.register('light', ('tsink', 'tcur0'), peram_sep)  # 3pt 汇端→流
    >>> reg.register('light', ('tcur0', 'tsrc'),  seq_peram)  # 3pt 流→源端
    """

    def __init__(self):
        self._entries = {}  # {(flavor, (t_q, t_aq)): data}

    def register(self, flavor, time_labels, data):
        """注册一个 peram 数组。同名键重复注册会覆盖旧值。

        Parameters
        ----------
        flavor : str
            夸克味。 ``'u'``, ``'d'``, ``'s'`` 或 ``'light'``。
            ``'light'`` 同时匹配 ``u`` 和 ``d``。
        time_labels : tuple of str
            ``(t_quark, t_antiquark)``。夸克时间在前，反夸克时间在后。
            合法值: ``'tsink'`` (汇), ``'tsrc'`` (源), ``'tcur0'`` / ``'tcur1'`` / ... (流插入点)。
        data : ndarray
            peram 数组，至少 ``(Ns, Ns, Nev, Nev)`` 维。仅存引用不复制。

        Raises
        ------
        ValueError
            如果 data 为 None 或维度不足 4。
        """
        _check_not_none(data, f'peram({flavor},{time_labels})')
        _check_min_ndim(data, 4, f'peram({flavor},{time_labels})')
        self._entries[(flavor, tuple(time_labels))] = data

    def resolve(self, combined_type, time_labels):
        """按 Wick 条目的味和时间标签查找 peram（内部使用）。

        Parameters
        ----------
        combined_type : str
            Wick 输出的组合味字符串，如 ``'uu^d'``，取其首字符为夸克味。
        time_labels : list of str
            时间标签对 ``[t_q, t_aq]``。

        Returns
        -------
        ndarray
            匹配的 peram 数组引用。

        Raises
        ------
        KeyError
            如果未找到匹配的注册条目。
        """
        qf = combined_type[0]
        tk = tuple(time_labels)
        # 精确匹配 flavor
        key = (qf, tk)
        if key in self._entries:
            return self._entries[key]
        # 'light' 通配
        key_light = ('light', tk)
        if key_light in self._entries:
            return self._entries[key_light]
        raise KeyError(
            f'Peram 未注册: 味={qf}, 时间={tk}。'
            f'已注册: {list(self._entries.keys())}')


# ═══════════════════════════════════════════════════════════════
# 数据注册表: V
# ═══════════════════════════════════════════════════════════════

class VRegistry:
    """V 顶点张量注册表。

    按 Wick V 结构名称和时间端存储 V 数组引用，不做复制。
    每个时间端内 ``VVV`` 和 ``VDV`` 各自从 0 开始独立编号。

    V 名称
    ------
    ``VVV_0``, ``VVV_1``, ...  三夸克顶点 (重子)
    ``VDV_0``, ``VDV_1``, ...  夸克-反夸克顶点 (介子)
    名称和编号由 ``wick_contraction`` 输出中的 ``_per_region_v_names`` 确定，
    直接按 verbose 输出注册即可。

    时间标签
    --------
    ``'tsink'``  汇端,  ``'tsrc'``  源端,  ``'tcur0'``  流插入点 (3pt), ...

    Parameters
    ----------
    无。初始化后通过 ``register`` 方法添加数据。

    Examples
    --------
    >>> reg = VRegistry()
    >>> reg.register('VVV_0', 'tsink', snkVVV)     # 汇端重子
    >>> reg.register('VVV_0', 'tsrc',  srcVVV)     # 源端重子
    >>> reg.register('VDV_0', 'tcur0', currVDV)    # 3pt 流插入点介子
    """

    def __init__(self):
        self._entries = {}  # (v_name, time_label) → data

    def register(self, v_name, time_label, data):
        """注册一个 V 张量。

        Parameters
        ----------
        v_name : str
            V 结构名称，如 ``'VVV_0'`` (三夸克顶点), ``'VDV_0'`` (介子顶点)。
            编号按时间端独立，由 verbose 输出的 Wick 分析结果决定，直接照抄即可。
        time_label : str
            该 V 结构所在的时间端。``'tsink'`` (汇), ``'tsrc'`` (源),
            ``'tcur0'`` / ``'tcur1'`` / ... (流插入点)。
        data : ndarray
            V 张量数组，第一维为动量。仅存引用不复制。

        Raises
        ------
        ValueError
            如果 data 为 None。
        """
        _check_not_none(data, f'V({v_name}@{time_label})')
        self._entries[(v_name, time_label)] = data

    def resolve(self, v_name, time_label):
        """按名称和时间端查找 V 张量（内部使用）。

        Parameters
        ----------
        v_name : str
            V 结构名称。
        time_label : str
            时间端标签。

        Returns
        -------
        ndarray
            匹配的 V 张量数组引用。

        Raises
        ------
        KeyError
            如果未找到匹配的注册条目。
        """
        k = (v_name, time_label)
        if k in self._entries:
            return self._entries[k]
        raise KeyError(
            f'V 未注册: {v_name}@{time_label}。'
            f'已注册: {list(self._entries.keys())}')


# ═══════════════════════════════════════════════════════════════
# 数据注册表: Gamma
# ═══════════════════════════════════════════════════════════════

class GammaRegistry:
    """Gamma 矩阵注册表。

    将算符中的 gamma 名称映射到具体复数数组，形状不限。
    名称不限标准格式 — 可注册 ``'gamma_7'`` 或自定义 ``'gamma_mu*gamma_5'``。

    Parameters
    ----------
    无。初始化后通过 ``register`` 方法添加数据。

    Examples
    --------
    >>> reg = GammaRegistry()
    >>> reg.register('gamma_7', backend.asarray(gamma(7)))
    >>> reg.register('gamma_w', gamma_0 - gamma_11)   # 自定义组合
    """

    def __init__(self):
        self._entries = {}

    def register(self, name, data):
        """注册一个 gamma 矩阵。

        Parameters
        ----------
        name : str
            算符中出现的 gamma 名称，如 ``'gamma_7'``。
        data : ndarray
            复数数组，任意形状。

        Raises
        ------
        ValueError
            如果 data 为 None。
        """
        _check_not_none(data, f'gamma({name})')
        self._entries[name] = data

    def resolve(self, name):
        """按名称查找 gamma 矩阵（内部使用）。

        Parameters
        ----------
        name : str
            gamma 矩阵名称。

        Returns
        -------
        ndarray
            gamma 矩阵数组引用。

        Raises
        ------
        KeyError
            如果未找到匹配的注册条目。
        """
        if name in self._entries:
            return self._entries[name]
        raise KeyError(
            f'Gamma 未注册: {name}。已注册: {list(self._entries.keys())}')


# ═══════════════════════════════════════════════════════════════
# 工具: V 结构按时间端重新编号
# ═══════════════════════════════════════════════════════════════

def _per_region_v_names(v_info):
    """将 Wick 全局 V 编号转为每个时间端独立编号（内部使用）。

    Wick 输出使用跨时间端递增的全局编号（如 ``VVV_0, VDV_1, VVV_2, ...``），
    本函数转为按时间端独立编号，同一时间端内 ``VVV`` 和 ``VDV`` 各自从 0 开始。

    Parameters
    ----------
    v_info : list of tuple
        ``wick['V']`` 的输出，每个元素为 ``[(low, high), V_name, index_string, time_label]``。

    Returns
    -------
    list of (str, str)
        ``[(per_region_name, time_label), ...]``，与 ``v_info`` 同顺序。
        ``per_region_name`` 如 ``'VVV_0'``, ``'VDV_0'``。
    """
    counters = {}  # (time_label, type_prefix) -> next_index
    result = []
    for v in v_info:
        vtype = 'VVV' if 'VVV' in v[1] else 'VDV'
        vtime = v[3]
        key = (vtime, vtype)
        idx = counters.get(key, 0)
        counters[key] = idx + 1
        result.append((f'{vtype}_{idx}', vtime))
    return result


# ═══════════════════════════════════════════════════════════════
# 工具: 双侧投影 einsum 构建
# ═══════════════════════════════════════════════════════════════

def _build_projection_einsum(ein, proj_label, output_labels,
                             oindex_given=False):
    """构建双侧投影 einsum（内部使用）。

    将输出的两个自旋指标分别与 ``Projector[0]`` (汇端自旋, 第一个前导小写)
    和 ``Projector[1]`` (源端自旋, 第二个) 收缩。

    - ``oindex_given=False``: 引入两个新的自由大写指标 X, Y
      （自动选取不与 einsum 中任何已有字母冲突者），
      输出为 ``XY + proj_label + output_labels``。
    - ``oindex_given=True``: 输出严格等于 ``output_labels`` (即 Oindex)。
      两个投影指标取同一字母互相收缩掉；``proj_label`` 若不在
      ``output_labels`` 中则被求和。

    Parameters
    ----------
    ein : str
        已完成 Oindex 覆盖的 einsum，RHS 形如 ``'{spin_out}{output_labels}'``。
    proj_label : str
        Projector 的额外维度标签（``Gindex`` 第 n_gammas+1 项），``''`` 表示无。
        两个投影算符共享该维度，且位于旋量指标之外（领先轴）。
    output_labels : str
        输出大写指标（``Oindex`` 或默认提取的大写指标）。
    oindex_given : bool, optional
        ``output_labels`` 是否来自用户显式给定的 ``Oindex``，默认 ``False``。

    Returns
    -------
    str
        ``oindex_given=False``:
        ``'{lhs},{L}aX,{L}sY->XY{L}{output_labels}'``；
        ``oindex_given=True``:
        ``'{lhs},{L}aZ,{L}sZ->{output_labels}'``
        （``a``/``s`` 为汇端/源端自旋指标, ``L`` 为 proj_label）。

    Raises
    ------
    ValueError
        如果 RHS 前导小写自旋指标不是恰好 2 个。
    """
    lhs, rhs = ein.split('->')
    n_lower = 0
    for ch in rhs:
        if ch.islower():
            n_lower += 1
        else:
            break
    spin_out = rhs[:n_lower]
    if len(spin_out) != 2:
        raise ValueError(
            f'双侧投影要求恰好 2 个输出自旋指标, '
            f'实际 spin_out={spin_out!r} (ein={ein!r})')
    used = set(ein) | set(proj_label) | set(output_labels)
    a, s = spin_out[0], spin_out[1]
    if oindex_given:
        # 投影指标取同一字母互相收缩; 输出严格按 Oindex
        z = [c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if c not in used][0]
        return (f'{lhs},{proj_label}{a}{z},{proj_label}{s}{z}'
                f'->{output_labels}')
    x, y = [c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if c not in used][:2]
    return (f'{lhs},{proj_label}{a}{x},{proj_label}{s}{y}'
            f'->{x}{y}{proj_label}{output_labels}')


# ═══════════════════════════════════════════════════════════════
# Wick 收缩分析 (含打印 + 纠错)
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# 分析结果缓存  (operator 配置 → plan)
# ═══════════════════════════════════════════════════════════════

_plan_cache = {}


def _make_hashable(obj):
    """递归将 list 转为 tuple，使对象可哈希。"""
    if isinstance(obj, list):
        return tuple(_make_hashable(x) for x in obj)
    if isinstance(obj, tuple):
        return tuple(_make_hashable(x) for x in obj)
    return obj


def _plan_cache_key(operator_groups, Cpt, Pindex, Vindex, Gindex,
                    use_equivalence, ignore_dis,
                    Projection=False, Oindex=None):
    """生成分析缓存的键。将所有 list/tuple 递归转为可哈希形式。"""
    groups_hashable = _make_hashable(operator_groups)
    return (groups_hashable, Cpt,
            _make_hashable(Pindex), _make_hashable(Vindex),
            _make_hashable(Gindex),
            use_equivalence, ignore_dis,
            Projection, _make_hashable(Oindex))


def clear_plan_cache():
    """清空分析缓存，下次调用 :func:`run_wick_analysis` 将重新分析并输出。"""
    _plan_cache.clear()


# ═══════════════════════════════════════════════════════════════
# 收缩路径分析辅助函数
# ═══════════════════════════════════════════════════════════════

def _format_cost(n):
    """Format a cost number (MACs) into human-readable string."""
    from decimal import Decimal
    n = int(n)
    if n >= 1e15:
        return f"{format(Decimal(n), '.3e')}"
    elif n >= 1e12:
        return f'{n / 1e12:.2f}T'
    elif n >= 1e9:
        return f'{n / 1e9:.2f}G'
    elif n >= 1e6:
        return f'{n / 1e6:.2f}M'
    elif n >= 1e3:
        return f'{n / 1e3:.2f}K'
    else:
        return f'{n:.0f}'


def _analyze_contraction_path(einsum_str, shapes, optimize='auto'):
    """Compute naive/optimized FLOP counts and largest intermediate.

    Uses ``opt_einsum.contract_path`` with placeholder tensors (no actual
    data allocation).  The naive path is sequential left-to-right; the
    optimized path follows the same optimizer-selection logic as
    :func:`cached_contract`.

    Parameters
    ----------
    einsum_str : str
        opt_einsum-style contraction string, e.g. ``'ab,bc->ac'``.
    shapes : list of tuple
        Shapes of the input tensors.
    optimize : str, bool, or list of str
        Same semantics as :func:`cached_contract`:
        - ``str``: use that single strategy
        - ``True``: try all built-in strategies and pick best
        - ``list``: try each in the list and pick best

    Returns
    -------
    tuple (naive_flops, opt_flops, speedup, largest_intermediate_bytes, opt_name)
        naive_flops, opt_flops: in MAC units (opt_einsum convention).
        largest_intermediate_bytes: complex128 bytes (elements × 16).
        opt_name: name of the winning optimizer (e.g. ``'auto'``, ``'greedy'``).
        Returns ``(0, 0, 1.0, 0, 'N/A')`` if path analysis fails.
    """
    from opt_einsum import contract_path
    import numpy as np

    n = len(shapes)
    if n < 2:
        return 0, 0, 1.0, 0, 'N/A'

    # 零内存占位符
    placeholders = [
        np.broadcast_to(np.empty((), dtype=np.float64), s)
        for s in shapes
    ]

    # ── Naive: sequential left-to-right ──
    try:
        _, naive_info = contract_path(
            einsum_str, *placeholders,
            optimize=[(0, 1)] * (n - 1))
        naive_flops = int(naive_info.opt_cost)
    except Exception as _e:
        _log(f'    [FLOP naive path 失败] {type(_e).__name__}: {_e}')
        naive_flops = 0

    # ── Optimized: replicate cached_contract's optimizer selection ──
    if isinstance(optimize, str):
        candidate_opts = [optimize]
    elif optimize is True:
        candidate_opts = ['auto', 'greedy', 'optimal', 'dp']
    elif isinstance(optimize, list):
        candidate_opts = optimize
    else:
        candidate_opts = ['auto']

    best_path_info = None
    best_cost = (float('inf'), float('inf'))
    best_opt_name = 'N/A'

    for opt in candidate_opts:
        try:
            _, path_info = contract_path(
                einsum_str, *placeholders, optimize=opt)
            cost = (int(path_info.opt_cost), int(path_info.largest_intermediate))
            if cost < best_cost:
                best_cost = cost
                best_path_info = path_info
                best_opt_name = opt
        except Exception as _e:
            _log(f'    [FLOP optimize={opt} 失败] {type(_e).__name__}: {_e}')
            continue

    if best_path_info is None:
        return naive_flops, 0, 1.0, 0, 'N/A'

    opt_flops = int(best_path_info.opt_cost)
    largest_intermediate = int(best_path_info.largest_intermediate)
    speedup = naive_flops / max(opt_flops, 1)
    li_bytes = largest_intermediate * 16  # complex128

    return naive_flops, opt_flops, speedup, li_bytes, best_opt_name


def run_wick_analysis(operator_groups, *,
                      Cpt='2pt',
                      Pindex=None, Vindex=None, Gindex=None,
                      use_equivalence=False,
                      ignore_dis=True,
                      verbose=True, max_detail=-1,
                      plot='',
                      peram_registry=None, v_registry=None, gamma_registry=None,
                      optimize='auto',
                      Projection=False,
                      Oindex=None):
    """运行 Wick 收缩分析，整合等价图检测，返回统一格式的收缩计划。

    对 ``operator_groups`` 中的每一组算符调用 :func:`wick_contraction`，
    收集所有 Wick 收缩结果，可选调用 :func:`identify_equivalent_diagrams`
    消除等价冗余图，最终输出可直接供 :func:`calculate_contraction` 使用的多维列表。

    Parameters
    ----------
    operator_groups : list of tuple
        - 2pt: ``[(sink_operators, source_operators), ...]``
        - 3pt: ``[(sink_operators, source_operators, curr_operators), ...]``

        算符为字符串列表，用 ``'|'`` 分隔粒子，``gamma_*`` 表示 gamma 矩阵，
        ``u/d/s/u^d`` 等表示夸克/反夸克味，支持数值系数前缀（如 ``-1``）。
        所有条目视为同一个过程，无需提供标签。
    Cpt : str, optional
        关联函数类型，``'2pt'``, ``'3pt'``, ``'4pt'`` 等，默认 ``'2pt'``。
    Pindex : list of str, optional
        peram 指标前缀，``None`` 使用默认大写字母。
    Vindex : list of str, optional
        V 结构指标前缀，``None`` 使用默认大写字母。
    Gindex : list of str, optional
        gamma 指标前缀，``None`` 使用默认大写字母。
    use_equivalence : bool, optional
        是否调用 :func:`identify_equivalent_diagrams` 做等价图归并，默认 ``False``。
    ignore_dis : bool, optional
        是否忽略 disconnected 图（任意 peram 的 ``t_q == t_aq``），默认 ``True``。
        忽略的图不会出现在输出 plan 中，但会在 verbose 模式打印 warning。
    verbose : bool, optional
        是否打印结构分析与注册指引，默认 ``True``。
    max_detail : int, optional
        每组显示单图详情的数量。``-1`` 表示全部输出，``0`` 表示不输出，
        ``>0`` 则显示前 ``max_detail`` 个，默认 ``3``。
    plot : str, optional
        若为非空字符串，将所有 Wick 收缩图输出为多页 PDF。
        若 ``plot`` 为目录路径，则在该目录下使用默认文件名
        ``wick_contraction_fig.pdf``；否则视为文件路径，强制使用 ``.pdf`` 后缀。
        默认 ``''`` (不绘图)。
    peram_registry : PeramRegistry, optional
        已注册 peram 数据的注册表。提供后，若同时提供 ``v_registry`` 与
        ``gamma_registry``，则 verbose 模式会在每张图后附加收缩路径分析
        (朴素 FLOP / 优化 FLOP / 加速比 / 中间最大张量数据 / 最优 optimize)。
        默认 ``None`` (不输出路径分析)。
    v_registry : VRegistry, optional
        已注册 V 张量数据的注册表，默认 ``None``。
    gamma_registry : GammaRegistry, optional
        已注册 gamma 矩阵数据的注册表，默认 ``None``。
    optimize : str, bool, or list of str, optional
        收缩路径优化策略，同 :func:`cached_contract`。仅当三个 registry
        均非 ``None`` 时生效，默认 ``'auto'``。
    Projection : bool, optional
        是否启用双侧投影的校验与显示。``True`` 时要求 ``gamma_registry``
        中的 ``'Projector'`` 注册为两个投影算符的序列
        ``[proj_sink, proj_src]``；额外维度标签取 ``Gindex`` 第
        n_gammas+1 项，两个算符共享该维度（领先轴）。默认 ``False``。
    Oindex : str, optional
        显式指定输出大写指标，``None`` 使用默认（RHS 提取的大写指标；
        Projection 时前面还会加上投影指标 X, Y 与额外标签）。
        给定时输出**严格等于** ``Oindex``：只能使用默认指标
        （及 Projection 额外标签）的子集，缺失的指标被求和；
        Projection 的投影指标 X, Y 取同一字母互相收缩掉。
        默认 ``None``。

    Returns
    -------
    list of list
        每行为 ``[equiv_list, contraction_idx, wick_dict, diag_idx,
        proj_label, output_labels, oindex_given]``：

        - ``equiv_list`` : ``list`` of ``(group_idx, diag_idx, coeff)``，统一为列表格式
            | ``use_equivalence=False`` → ``[(gidx, di, 1.0)]``
            | ``use_equivalence=True``  → ``[(gidx, di, coeff), ...]`` 等价类全部成员及相对符号
        - ``contraction_idx`` : ``int``，代表图所属 group 在 ``operator_groups`` 中的索引
        - ``wick_dict`` : ``dict``，:func:`wick_contraction` 返回的 dict 引用
        - ``diag_idx`` : ``int``，代表图在 ``wick_dict`` 内的图索引
        - ``proj_label`` : ``str``，Projector 的额外维度标签 (``''`` 表示无)
        - ``output_labels`` : ``str``，输出大写指标 (``Oindex`` 覆盖后的)
        - ``oindex_given`` : ``bool``，``Oindex`` 是否显式给定

        :func:`calculate_contraction` 可直接用 ``entry[2]`` + ``entry[3]``。

    Examples
    --------
    >>> # 2pt 单组 (无等价图检测)
    >>> plan = run_wick_analysis([
    ...     (['|', 'u', 'u', 'gamma_7', 'd', '|'],
    ...      ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|']),
    ... ], Cpt='2pt')
    >>> for entry in plan:
    ...     eq_list, grp_idx, w, di, *_ = entry
    ...     gidx, d, coeff = eq_list[0]
    ...     print(f'图 (group{gidx}, 图{d}), coeff={coeff:+.0f}')

    >>> # 2pt 多组 + 等价图检测
    >>> plan = run_wick_analysis([
    ...     (sink_op1, src_op1),
    ...     (sink_op2, src_op2),
    ... ], Cpt='2pt', use_equivalence=True)
    >>> for entry in plan:
    ...     eq_list, grp_idx, w, di, *_ = entry
    ...     print(f'等价类 {eq_list}，代表: group{grp_idx} 图{di}')
    """
    # ── 缓存: 相同算符配置跳过分析 ──
    cache_key = _plan_cache_key(operator_groups, Cpt,
                                Pindex, Vindex, Gindex,
                                use_equivalence, ignore_dis,
                                Projection, Oindex)
    if cache_key in _plan_cache:
        return _plan_cache[cache_key]

    wick_results = []
    all_peram_types = set()
    all_gamma_names = set()
    all_v_names = set()
    errors = []
    disconnected = set()  # {(group_idx, diag_idx), ...} if ignore_dis

    # ── 确定 gamma 数量 + Projection 预处理 ──
    # 从第一组算符统计 gamma 出现次数
    _first_item = operator_groups[0]
    if len(_first_item) == 2:
        _s0, _s1, _s2 = _first_item[0], _first_item[1], []
    else:
        _s0, _s1, _s2 = _first_item
    _all_ops = _s0 + _s1 + _s2
    n_gammas = sum(1 for x in _all_ops if isinstance(x, str) and x.startswith('gamma_'))

    _gindex_w = list(Gindex) if Gindex else []
    # 补齐 Gindex 至 n_gammas (不足用 '')
    if len(_gindex_w) < n_gammas:
        _gindex_w += [''] * (n_gammas - len(_gindex_w))
    _proj_label = ''
    if Projection:
        # projector 使用 Gindex 的第 n_gammas 项 (若存在)
        if len(Gindex) > n_gammas if Gindex else False:
            _proj_label = Gindex[n_gammas]
        # 纠错: Projection=True 但 Projector 未注册或结构不对
        if gamma_registry is not None:
            try:
                _proj_check = gamma_registry.resolve('Projector')
                try:
                    _n_proj = len(_proj_check)
                except TypeError:
                    _n_proj = -1
                if _n_proj != 2:
                    errors.append(
                        "Projector 必须为两个投影算符组成的序列 "
                        "[proj_sink, proj_src]。")
            except KeyError:
                errors.append(
                    'Projection=True 但 gamma_registry 中未注册 '
                    "'Projector'。请注册投影算符。")
        else:
            errors.append(
                'Projection=True 但未提供 gamma_registry，'
                '无法校验 Projector。')

    # ── 输出外部指标 ──
    # 从第一个 wick 结果提取默认大写指标，Oindex 非 None 时覆盖
    _output_labels = None  # 最终使用的输出大写指标
    def _get_output_labels(einsum_rhs):
        """从 RHS 提取大写字母，若 Oindex 非 None 则覆盖。"""
        _upper = ''.join(c for c in einsum_rhs if c.isupper())
        if Oindex is not None:
            # Projection 时 Oindex 还允许包含投影额外标签 (proj_label)
            _allowed = _upper + (_proj_label if Projection else '')
            _unknown = [c for c in Oindex if c not in _allowed]
            if _unknown:
                errors.append(
                    f'Oindex={Oindex!r} 包含未知指标 {_unknown}，'
                    f'默认输出指标为 {_upper!r}。'
                    '请只使用默认指标的子集。')
            return Oindex
        return _upper

    # ── Phase 1: 收集所有 Wick 结果与元数据（不打印）──
    group_info = []  # 缓存每组数据供 Phase 3 打印
    for idx, item in enumerate(operator_groups):
        if len(item) == 2:
            sink_op, src_op = item
            curr_op = []
        else:
            sink_op, src_op, curr_op = item
        w = wick_contraction(
            sink_operators=sink_op, source_operators=src_op,
            Cpt=Cpt, curr_operators=curr_op,
            Pindex=Pindex, Vindex=Vindex, Gindex=_gindex_w)
        wick_results.append(w)

        if _output_labels is None:
            _rhs = w['result_indx'][0][0].split('->')[1]
            _output_labels = _get_output_labels(_rhs)

        nd = len(w['result_indx'])
        sink_seps = sum(1 for x in w['sink_operators'] if x == '|')
        src_seps = sum(1 for x in w['source_operators'] if x == '|')

        # 收集 peram 类型
        ptypes = set()
        for di in range(nd):
            for p in w['peram'][di]:
                pt = (p[2][0], tuple(p[4]))
                ptypes.add(pt)
                all_peram_types.add(pt)

        # 收集 gamma 和 V 名称 (V 使用按时间端独立编号)
        gnames = [(g[1], g[3]) for g in w['gamma_pos']]
        vnames_per_region = _per_region_v_names(w['V'])
        for gn, _ in gnames:
            all_gamma_names.add(gn)
        for vn, vt in vnames_per_region:
            all_v_names.add((vn, vt))

        # ── 纠错: 检查是否有 sign=0 的图 ──
        for di in range(nd):
            if abs(w['result_sign'][di]) < 1e-12:
                errors.append(f'[group{idx}] 图{di} sign≈0, 可能为无效收缩。')

        # ── 纠错: 检查 peram 时间标签是否合法 ──
        # 根据 Cpt 动态构建有效时间标签集合:
        #   2pt → {tsink, tsrc}
        #   3pt → +tcur0
        #   4pt → +tcur0, tcur1, ...
        _base_labels = {'tsink', 'tsrc'}
        if Cpt == '3pt':
            _base_labels.add('tcur0')
        elif Cpt not in ('2pt',):
            # 4pt 及以上: 按区间数推断 current 数量
            n_cur = int(Cpt[0]) - 2 if Cpt[0].isdigit() else 0
            for _ci in range(n_cur):
                _base_labels.add(f'tcur{_ci}')
        valid = {(a, b) for a in _base_labels for b in _base_labels}
        for pt in ptypes:
            t = pt[1]
            if t not in valid:
                errors.append(f'[group{idx}] 未知 peram 时间标签: {t}')

        # ── 检测 disconnected 图 (任意 peram 的 t_q == t_aq) ──
        dis_info = []  # [(di, [(peram_label, t_q, t_aq), ...]), ...]
        for di in range(nd):
            dis_perams = []
            for p in w['peram'][di]:
                if p[4][0] == p[4][1]:
                    dis_perams.append((p[3], p[4][0], p[4][1]))
            if dis_perams:
                dis_info.append((di, dis_perams))
                if ignore_dis:
                    disconnected.add((idx, di))

        n_dis = len(dis_info)
        n_con = nd - n_dis

        group_info.append((idx, w, nd, sink_seps, src_seps, ptypes, gnames,
                          vnames_per_region, dis_info, n_dis, n_con))

    # ── Phase 2: 等价图检测，建立 equiv_lookup ──
    # equiv_lookup: (gidx, di) -> (rep_gidx, rep_di)  非代表元指向代表元
    equiv_lookup = {}
    eqs = None
    if use_equivalence:
        eqs = identify_equivalent_diagrams(*wick_results)
        for eq in eqs:
            equiv_list = [(d, di, coeff) for d, di, coeff in eq]
            rd, rdi, _ = eq[0]
            # 若忽略 disconnected 且代表元是 disconnected，跳过整个等价类
            if ignore_dis and (rd, rdi) in disconnected:
                continue
            for gidx, di, _ in equiv_list:
                if ignore_dis and (gidx, di) in disconnected:
                    continue
                if (gidx, di) != (rd, rdi):
                    equiv_lookup[(gidx, di)] = (rd, rdi)

    # ── Phase 3: 打印每组详情（含内联等价标注）──
    if verbose:
        for (idx, w, nd, sink_seps, src_seps, ptypes, gnames,
             vnames_per_region, dis_info, n_dis, n_con) in group_info:
            _log(f'\n{"─"*60}')
            if ignore_dis and n_dis:
                dis_str = f'  (disconnected: {n_dis}, 已忽略)'
            elif n_dis:
                dis_str = f'  (disconnected: {n_dis})'
            else:
                dis_str = ''
            shown_total = nd if not ignore_dis else n_con
            _log(f'组 {idx:2d}  汇|={sink_seps} 源|={src_seps}  图数={nd}  '
                  f'有效图={shown_total}{dis_str}')
            _log(f'  需注册 peram (味,时间): {sorted(ptypes)}')
            _log(f'  需注册 gamma: {gnames}')
            _log(f'  需注册 V (tsink/src 独立编号): {vnames_per_region}')

            # 输出 disconnected warning
            if ignore_dis and n_dis and max_detail != 0:
                _log(f'  ⚠ 忽略 {n_dis} 个 disconnected 图:')
                for di, dps in dis_info:
                    dps_str = ', '.join(
                        f'{lb}({tq},{taq})' for lb, tq, taq in dps)
                    _log(f'    图{di}: {dps_str}')

            if max_detail != 0:
                total_show = nd if not ignore_dis else n_con
                limit = total_show if max_detail == -1 else min(total_show, max_detail)
                shown = 0
                for di in range(nd):
                    if ignore_dis and any(d == di for d, _ in dis_info):
                        continue
                    if shown >= limit:
                        break
                    s = w['result_sign'][di]
                    s_fmt = f'{s.real:+.3f}{s.imag:+.3f}j'
                    e = w['result_indx'][di][0]
                    rhs = e.split('->')[1]
                    free = ''.join(c for c in rhs if c.islower())
                    pl = []
                    for p in w['peram'][di]:
                        pl.append(f'{p[3]}({p[2][0]},{p[4][0]},{p[4][1]})')
                    # 等价标注：非代表元指向代表元
                    eq_note = ''
                    if use_equivalence:
                        eq_key = (idx, di)
                        if eq_key in equiv_lookup:
                            rg, rd = equiv_lookup[eq_key]
                            eq_note = f'  等价于 组{rg} 图{rd}'
                    # 收缩路径分析（仅当三个 registry 均提供时）
                    cost_note = ''
                    _has_reg = (peram_registry is not None
                                and v_registry is not None
                                and gamma_registry is not None)
                    # 构造显示用 einsum（含 Projection + Oindex）
                    _e_display = e
                    if Projection:
                        _lhs, _rhs = e.split('->')
                        _n_lower = 0
                        for _ch in _rhs:
                            if _ch.islower():
                                _n_lower += 1
                            else:
                                break
                        _spin_out = _rhs[:_n_lower]
                        _e_base = f'{_lhs}->{_spin_out}{_output_labels}'
                        try:
                            _e_display = _build_projection_einsum(
                                _e_base, _proj_label, _output_labels,
                                oindex_given=Oindex is not None)
                        except ValueError as _pe:
                            _log(f'    [Projection einsum 构造失败] {_pe}')
                            _e_display = _e_base
                    else:
                        _e_display = f'{e.split("->")[0]}->{free}{_output_labels}'
                    if _has_reg:
                        try:
                            _p_shapes = [
                                peram_registry.resolve(p[2], p[4]).shape
                                for p in w['peram'][di]]
                            _v_names = _per_region_v_names(w['V'])
                            _v_shapes = [
                                v_registry.resolve(vn, vt).shape
                                for vn, vt in _v_names]
                            _g_shapes = [
                                gamma_registry.resolve(g[1]).shape
                                for g in w['gamma_pos']]
                            _shapes = _p_shapes + _g_shapes + _v_shapes
                            if Projection:
                                _proj = gamma_registry.resolve('Projector')
                                _shapes = _shapes + [_proj[0].shape,
                                                     _proj[1].shape]
                            nf, of, sp, li, opt_name = \
                                _analyze_contraction_path(
                                    _e_display, _shapes, optimize)
                            cost_note = (
                                f'  朴素 FLOP={_format_cost(nf)}'
                                f'  优化 FLOP={_format_cost(of)}'
                                f'  加速比={_format_cost(sp)}x'
                                f'  中间最大张量数据={li / 1e9:.2f}GB(cpx)'
                                f'  最优optimize={opt_name}')
                        except Exception as _e:
                            _log(f'    [FLOP分析失败] {type(_e).__name__}: {_e}')
                    _log(f'  图{di}: sign={s_fmt} 自由={free}{eq_note}{cost_note}')
                    _log(f'    peram: {", ".join(pl)}')
                    _log(f'    einsum: {_e_display}')
                    shown += 1
                if max_detail > 0 and total_show > max_detail:
                    label = '图' if not ignore_dis else 'connected 图'
                    _log(f'  ... (共 {total_show} 个 {label}, 显示前 {max_detail})')

    if verbose and errors:
        _log(f'\n⚠ 纠错检测到 {len(errors)} 个问题:')
        for e in errors:
            _log(f'  - {e}')

    # ── 输出 peram/gamma/V 注册指引 (verbose 模式) ──
    if verbose:
        _log(f'\n{"─"*60}')
        _log(f'全局需注册:')
        _log(f'  peram (味,时间): {sorted(all_peram_types)}')
        _log(f'  gamma: {sorted(all_gamma_names)}')
        _log(f'  V (名称,时间端): {sorted(all_v_names)}')

    # ── 全局 disconnected warning ──
    if verbose and disconnected:
        _log(f'\n⚠ Warning: {len(disconnected)} 个 disconnected 图已忽略 '
              f'(ignore_dis=True)。')
        _log(f'  判断条件: peram 的 t_q == t_aq (传播子两端时间相同)。')
        _log(f'  被忽略的 (group, diagram): {sorted(disconnected)}')

    # ── Phase 4: 构建 plan ──
    plan = []
    total_raw = 0

    if use_equivalence:
        for eq in eqs:
            equiv_list = [(d, di, coeff) for d, di, coeff in eq]
            rd, rdi, _ = eq[0]
            if ignore_dis and (rd, rdi) in disconnected:
                continue
            plan.append([
                equiv_list,        # [0] 等价图列表: [(group_idx, diag_idx, coeff), ...]
                rd,                # [1] 收缩的指标: 代表图所在 group
                wick_results[rd],  # [2] wick_dict
                rdi,               # [3] diag_idx
                _proj_label,       # [4] Projector 的 Gindex 标签 ('' 表示无投影或 2D)
                _output_labels,    # [5] 输出大写指标 (Oindex 覆盖后的)
                Oindex is not None,  # [6] Oindex 是否显式给定
            ])
            total_raw += len(equiv_list)

        if verbose:
            nu = len(plan)
            nr = total_raw
            pct = 100 - nu * 100 // nr if nr else 0
            n_skip = len(disconnected)
            skip_str = f' (另有 {n_skip} 个 disconnected 图已跳过)' if n_skip else ''
            _log(f'\n等价图优化: {nr}图→{nu}唯一 ({pct}%减){skip_str}')
    else:
        for gidx, w in enumerate(wick_results):
            for di in range(len(w['result_indx'])):
                if ignore_dis and (gidx, di) in disconnected:
                    continue
                plan.append([
                    [(gidx, di, w['result_sign'][di])],  # [0] 等价图列表: 单元素, coeff=result_sign
                    gidx,                # [1] 收缩的指标: group 索引
                    w,                   # [2] wick_dict
                    di,                  # [3] diag_idx
                    _proj_label,         # [4] Projector 的 Gindex 标签
                    _output_labels,      # [5] 输出大写指标
                    Oindex is not None,  # [6] Oindex 是否显式给定
                ])
                total_raw += 1

    _plan_cache[cache_key] = plan

    # ── Phase 5: 可选多页 PDF 输出 ──
    if plot and getMPIRank() == 0:
        import os
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from ..contraction.autowick import plot_figure_wick

        if os.path.isdir(plot):
            filename = os.path.join(plot, 'wick_contraction_fig.pdf')
        else:
            base, _ = os.path.splitext(plot)
            filename = base + '.pdf'
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

        with PdfPages(filename) as pdf:
            for entry in plan:
                eq_list, group_idx, wick_dict, diag_idx, *_ = entry
                fig, ax = plot_figure_wick(
                    wick_dict, diagram_index=diag_idx,
                    Cpt=Cpt, plot_text=True)
                sign = wick_dict['result_sign'][diag_idx]
                sign_fmt = f'{sign.real:+.3f}{sign.imag:+.3f}j'
                ax.set_title(
                    f'Group {group_idx}: Wick Contraction Diagram #{diag_idx}  '
                    f'({Cpt}, sign = {sign_fmt})',
                    fontsize=ax.title.get_fontsize(), fontweight='bold')
                pdf.savefig(fig, dpi=250, bbox_inches='tight')
                plt.close(fig)
        if verbose:
            _log(f'\n✓ Wick 收缩图已保存至: {filename}')

    return plan


# ═══════════════════════════════════════════════════════════════
# 动态收缩构建
# ═══════════════════════════════════════════════════════════════

def calculate_contraction(entry, *,
                          peram_registry, v_registry, gamma_registry,
                          optimize='auto',
                          Projection=False):
    """将 :func:`run_wick_analysis` 输出的一行解析为张量并执行收缩。

    先检查 ``entry[0]`` 中所有等价图的系数之和 ``total_coeff``：
    若为 0 则直接返回 ``0``，跳过计算；否则调用 :func:`cached_contract`
    执行收缩并将结果乘以 ``total_coeff``。

    Parameters
    ----------
    entry : list
        :func:`run_wick_analysis` 返回的 ``plan`` 中的一行，
        格式 ``[equiv_list, contraction_idx, wick_dict, diag_idx,
        proj_label]``。``proj_label`` 为可选的第 5 项。
    peram_registry : PeramRegistry
        已注册 peram 数据的注册表。
    v_registry : VRegistry
        已注册 V 张量数据的注册表。
    gamma_registry : GammaRegistry
        已注册 gamma 矩阵数据的注册表。
    optimize : str, bool, or list of str, optional
        传递给 :func:`cached_contract` 的优化策略，默认 ``'auto'``。
    Projection : bool or str, optional
        若为非 ``False``: 使用双侧投影。``gamma_registry`` 中的 ``'Projector'``
        必须注册为两个投影算符组成的序列 ``[proj_sink, proj_src]``，
        ``Projector[0]`` 收缩汇端自旋，``Projector[1]`` 收缩源端自旋。
        额外维度标签（字符串直接作为标签；``True`` 时自动从 ``entry[4]`` 读取）
        为两个投影算符共享的领先轴（位于旋量指标之外）。
        输出指标顺序为 ``XY + 额外标签 + 原大写输出指标``，
        X, Y 为投影引入的两个新自由指标。
        若 plan 由显式 ``Oindex`` 生成（``entry[6]`` 为 ``True``），
        则输出严格等于 ``Oindex``：投影指标取同一字母互相收缩掉，
        额外标签不在 ``Oindex`` 中时被求和。
        默认 ``False`` (不投影)。

    Returns
    -------
    ndarray or int
        收缩结果乘以 ``total_coeff`` 后的张量。
        ``total_coeff == 0`` 时返回 ``0``。

    Raises
    ------
    RuntimeError
        如果张量数与 einsum 逗号段数不一致。
    KeyError
        如果 peram / V / gamma 名称在注册表中未找到。

    Examples
    --------
    >>> plan = run_wick_analysis([...], Cpt='2pt')
    >>> for entry in plan:
    ...     result = calculate_contraction(
    ...         entry, peram_registry=peram_reg,
    ...         v_registry=v_reg, gamma_registry=gamma_reg)
    ...     accumulator += result
    """
    # 检查总系数，为 0 则跳过
    equiv_list = entry[0]
    total_coeff = sum(coeff for _, _, coeff in equiv_list)
    if abs(total_coeff) < 1e-12:
        return complex(0)

    _, _, wick, diag_idx, *_ = entry
    _output_labels = entry[5] if len(entry) > 5 else ''
    _oindex_given = entry[6] if len(entry) > 6 else False
    ein = wick['result_indx'][diag_idx][0]
    perams = wick['peram'][diag_idx]
    v_info = wick['V']
    g_info = wick['gamma_pos']

    p_vars = [peram_registry.resolve(p[2], p[4]) for p in perams]
    v_names = _per_region_v_names(v_info)  # 转为按时间端独立编号
    v_vars = [v_registry.resolve(vn, vt) for vn, vt in v_names]
    g_vars = [gamma_registry.resolve(g[1]) for g in g_info]

    tensors = p_vars + g_vars + v_vars

    # ── Oindex 覆盖输出大写指标 (entry[5] 存在即重写, 允许 Oindex='') ──
    if len(entry) > 5:
        lhs, rhs = ein.split('->')
        n_lower = 0
        for ch in rhs:
            if ch.islower(): n_lower += 1
            else: break
        spin_out = rhs[:n_lower]
        ein = f'{lhs}->{spin_out}{_output_labels}'

    # ── Projection: 双侧投影, Projector[0]↔汇端自旋, [1]↔源端自旋 ──
    if Projection is not False:
        # 确定标签: True→从 entry 读取, str→直接使用
        if Projection is True:
            _proj_label = entry[4] if len(entry) > 4 else ''
        else:
            _proj_label = str(Projection)
        proj = gamma_registry.resolve('Projector')
        try:
            n_proj = len(proj)
        except TypeError:
            n_proj = -1
        if n_proj != 2:
            raise ValueError(
                "Projector 必须为两个投影算符组成的序列 "
                "[proj_sink, proj_src]。")
        ein = _build_projection_einsum(ein, _proj_label, _output_labels,
                                       oindex_given=_oindex_given)
        tensors = tensors + [proj[0], proj[1]]

    # 纠错: 张量数与 einsum 的逗号段数一致
    n_parts = len(ein.split('->')[0].split(','))
    if len(tensors) != n_parts:
        raise RuntimeError(
            f'张量数({len(tensors)}) ≠ einsum段数({n_parts}): '
            f'perams={len(perams)} gammas={len(g_info)} V={len(v_info)}')

    
    return total_coeff * cached_contract(ein, *tensors, optimize=optimize)


# ═══════════════════════════════════════════════════════════════
# 工具: 校验与便捷计算
# ═══════════════════════════════════════════════════════════════

def validate_plan(plan, *,
                  peram_registry, v_registry, gamma_registry):
    """提前校验 plan 中所有条目，一次性列出所有缺失的注册项。

    在实际计算前批量检查，避免运行到一半才因 KeyError 中断。

    Parameters
    ----------
    plan : list
        :func:`run_wick_analysis` 的输出。
    peram_registry : PeramRegistry
    v_registry : VRegistry
    gamma_registry : GammaRegistry

    Returns
    -------
    list of (int, str)
        ``[(entry_index, error_message), ...]``，空列表表示全部通过校验。

    Examples
    --------
    >>> missing = validate_plan(plan, peram_registry=pr, v_registry=vr, gamma_registry=gr)
    >>> if missing:
    ...     for idx, msg in missing:
    ...         print(f'plan[{idx}]: {msg}')
    ... else:
    ...     print('全部校验通过')
    """
    missing = []
    for i, entry in enumerate(plan):
        try:
            # 仅做张量解析校验，不实际收缩
            _, _, wick, diag_idx, *_ = entry
            perams = wick['peram'][diag_idx]
            v_info = wick['V']
            g_info = wick['gamma_pos']
            for p in perams:
                peram_registry.resolve(p[2], p[4])
            for vn, vt in _per_region_v_names(v_info):
                v_registry.resolve(vn, vt)
            for g in g_info:
                gamma_registry.resolve(g[1])
        except KeyError as e:
            missing.append((i, str(e)))
    return missing


# ═══════════════════════════════════════════════════════════════
# 动态收缩计算器
# ═══════════════════════════════════════════════════════════════

class dynamic_contraction:
    """动态 Wick 收缩计算器。

    初始化时自动完成 Wick 分析、等价图检测、纠错和注册校验，
    收缩计算在后续调用 :meth:`calculate` 或 :meth:`calculate_all` 时按需执行。

    Parameters
    ----------
    operator_groups : list of tuple
        同 :func:`run_wick_analysis`。
    peram_registry : PeramRegistry
        已注册 peram 数据的注册表。
    v_registry : VRegistry
        已注册 V 张量数据的注册表。
    gamma_registry : GammaRegistry
        已注册 gamma 矩阵数据的注册表。
    Cpt : str, optional
    use_equivalence : bool, optional
    Pindex, Vindex, Gindex : list of str, optional
    verbose : bool, optional
    max_detail : int, optional
        以上参数同 :func:`run_wick_analysis`。
    plot : str, optional
        若为非空，将全部 Wick 收缩图输出为多页 PDF。
        默认 ``''`` (不绘图)。同 :func:`run_wick_analysis`。

    Attributes
    ----------
    plan : list
        同 :func:`run_wick_analysis` 的返回值。
    missing : list
        校验缺失项列表，空表示全部通过。

    输出指标顺序
    ------------
    收缩结果是 Wiсk 原始 einsum 的自由指标加上 ``Vindex`` / ``Gindex``
    添加的动量/流指标。指标按 ``小写→大写`` 排列：

    - 小写字母 (a, b, c, ...): 自旋 (Dirac) 指标，``Ns=4``
    - 大写字母 (M, N, L, G, ...): 由 ``Vindex`` (给 V 张量加前缀) /
      ``Gindex`` (给 gamma 矩阵加前缀) 引入的外部指标。
      不加前缀的 V/gamma 指标只参与内部收缩，不会出现在输出中。
    - ``M`` 汇动量, ``N`` 源动量, ``L`` link, ``G`` 流 gamma — 字母由用户定义
    - 指标顺序即 einsum 中 ``->`` 右侧的顺序，对应输出的各维度
    - ``Projection=True`` 时输出为 ``XY + 投影额外标签 + 上述大写指标``：
      X, Y 为两个投影算符 (``Projector[0]``↔汇端, ``[1]``↔源端) 引入的
      新自由指标，自旋指标已被投影收缩。
    - ``Oindex`` 显式给定时输出严格等于 ``Oindex``（缺失的指标被求和；
      Projection 的投影指标 X, Y 互相收缩掉，不出现在输出中）。

    Examples
    --------
    >>> dc = dynamic_contraction(
    ...     operator_groups=[(sink_op, src_op)],
    ...     peram_registry=peram_reg, v_registry=v_reg,
    ...     gamma_registry=gamma_reg,
    ...     Cpt='2pt', use_equivalence=True)
    >>> if not dc.missing:
    ...     for i in range(len(dc)):
    ...         result = dc.calculate(i)
    """
    def __init__(
        self, operator_groups, *,
        peram_registry, v_registry, gamma_registry,
        Cpt='2pt',
        Pindex=None, Vindex=None, Gindex=None,
        use_equivalence=False,
        ignore_dis=True,
        verbose=True,
        max_detail=-1,
        plot='',
        Projection=False,
        optimize='auto',
        Oindex=None,
        ):

        self._peram_registry = peram_registry
        self._v_registry = v_registry
        self._gamma_registry = gamma_registry
        self._projection = Projection
        self._optimize = optimize

        # 检查缓存: 仅首次 (cache miss) 打印校验成功信息
        _cache_key = _plan_cache_key(operator_groups, Cpt,
                                     Pindex, Vindex, Gindex,
                                     use_equivalence, ignore_dis,
                                     Projection, Oindex)
        _is_cached = _cache_key in _plan_cache

        self.plan = run_wick_analysis(
            operator_groups,
            Cpt=Cpt,
            Pindex=Pindex, Vindex=Vindex, Gindex=Gindex,
            use_equivalence=use_equivalence,
            ignore_dis=ignore_dis,
            verbose=verbose, max_detail=max_detail,
            plot=plot,
            peram_registry=peram_registry,
            v_registry=v_registry,
            gamma_registry=gamma_registry,
            optimize=optimize,
            Projection=Projection,
            Oindex=Oindex,
            )

        self.missing = validate_plan(
            self.plan,
            peram_registry=peram_registry,
            v_registry=v_registry,
            gamma_registry=gamma_registry
            )
        if self.missing:
            msg = '\n'.join(f'  plan[{i}]: {m}' for i, m in self.missing)
            _abort(f'注册校验失败 ({len(self.missing)} 项缺失):\n{msg}')
        elif verbose and not _is_cached:
            _log('✓ 注册校验全部通过')

    def calculate(self, index):
        """计算 plan 中第 ``index`` 个条目的收缩。

        优化策略和投影设置继承自初始化参数。

        Parameters
        ----------
        index : int
            plan 中的条目索引。

        Returns
        -------
        ndarray or int
            收缩结果（已乘 ``total_coeff``），``total_coeff==0`` 时返回 ``0``。
            形状由 einsum 输出指标决定，见类文档中"输出指标顺序"。

        Examples
        --------
        >>> # 2pt, Vindex=['M','M'], 每个图的输出 shape: (4, 4, n_mom)
        >>> for i in range(len(dc)):
        ...     diagram_result = dc.calculate(i)
        >>> total = sum(dc.calculate(i) for i in range(len(dc)))

        Raises
        ------
        IndexError
            如果 index 超出范围。
        KeyError
            如果注册表缺失所需数据。
        """
        if index < 0 or index >= len(self.plan):
            raise IndexError(
                f'index={index} 超出范围，plan 共 {len(self.plan)} 个条目。')
        try:
            return calculate_contraction(
                self.plan[index],
                peram_registry=self._peram_registry,
                v_registry=self._v_registry,
                gamma_registry=self._gamma_registry,
                optimize=self._optimize,
                Projection=self._projection)
        except KeyError as e:
            _abort(f'plan[{index}] 收缩失败: {self.plan[index][0]}\n  {e}')

    def calculate_all(self):
        """计算所有收缩并求和，返回总关联函数。

        对 plan 中每个条目调用 :meth:`calculate`，将结果累加。
        ``total_coeff==0`` 的条目返回 ``0``，不影响求和。
        优化策略和投影设置继承自初始化参数。

        Returns
        -------
        ndarray or int
            所有收缩结果的加权和，即完整关联函数。
            形状由 einsum 输出指标决定，见类文档中"输出指标顺序"。
            若全部条目均为 ``0`` 则返回 ``0``。

        Examples
        --------
        >>> # 2pt, Vindex=['M','M']
        >>> # 输出 einsum: ...->agM  →  shape: (4, 4, n_mom)
        >>> #   4=汇自旋, 4=源自旋, n_mom=动量
        >>> corr = dc.calculate_all()
        >>>
        >>> # 3pt, Vindex=['M','L','M'], Gindex=['','G','']
        >>> # 给第 2 个 gamma (流 gamma_4) 加前缀 'G'，使之成为外部指标
        >>> # 输出 einsum: ...->GML  →  shape: (n_gamma_curr, n_mom, n_link)
        >>> corr_3pt = dc_3pt.calculate_all()
        """
        total = 0
        for i in range(len(self.plan)):
            try:
                total += self.calculate(i)
            except KeyError as e:
                _abort(f'calculate_all: plan[{i}] 收缩失败, '
                       f'等价类={self.plan[i][0]}\n  {e}')

        # # 释放 CuPy 内存池中未使用的块, 减少多轮计算中的显存碎片化
        # _backend = get_backend()
        # if hasattr(_backend, 'get_default_memory_pool'):
        #     _backend.get_default_memory_pool().free_all_blocks()

        return total

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        return self.plan[index]