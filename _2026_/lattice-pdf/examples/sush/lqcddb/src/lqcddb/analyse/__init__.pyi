"""
Type stub for lqcddb.analyse subpackage.

提供动量转换、源时间循环、Jackknife/Bootstrap 重采样、
有效质量提取、GEVP 求解、ratio_3pt 比值计算和 disconnected 扣除等统计分析功能。
"""
from typing import Any, Dict, List, Literal, Tuple, Union, Optional
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Mom2GeV — 动量转能量
# ═══════════════════════════════════════════════════════════════════════════

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

    示例::

        E = Mom2GeV(32, 0.12, [1, 0, 0], 0.5)        # 单个动量
        E_list = Mom2GeV(32, 0.12, [[1,0,0], [0,0,0]], [0.5, 0.8])
    """
    ...
# ═══════════════════════════════════════════════════════════════════════════

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

    示例::

        corr_avg = loop_tsrc(corr_2pt, indx=[-2, -3], Ctype='2pt')
    """
    ...

# ═══════════════════════════════════════════════════════════════════════════
# sum_over_array_of_list — 分组求和
# ═══════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════
# mean_over_array_of_list — 分组求均值
# ═══════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════
# Jackknife — 刀切法重采样
# ═══════════════════════════════════════════════════════════════════════════

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

    示例::

        jk = Jackknife(corr_2pt, Nconf_axes=0)
        mean, err = jk['data_mean'], jk['data_err']
    """
    ...
# ═══════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════
# meff — 有效质量提取
# ═══════════════════════════════════════════════════════════════════════════

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

    示例::

        eff = meff(jk['data_sample'], alttc=0.12, meff_type='log')
        m_eff, m_err = eff['data_mean'], eff['data_err']
    """
    ...

# ═══════════════════════════════════════════════════════════════════════════
# ratio_3pt — 3pt/2pt 比值
# ═══════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════
# solve_gevp — 广义特征值问题
# ═══════════════════════════════════════════════════════════════════════════

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

    示例::

        eigvals, eigvecs = solve_gevp(C_jk, t0=2)
        eff = meff(eigvals[0], alttc=0.12, meff_type='GEVP')
    """
    ...

# ═══════════════════════════════════════════════════════════════════════════
# dis_connect — Disconnected 贡献扣除
# ═══════════════════════════════════════════════════════════════════════════

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

    示例::

        disc = dis_connect(C2_jk, bubble_jk, Nconf_axes=0,
                           t_src_axes=1, t_sink_axes=2, tsep=12, dtype='PDF')
    """
    ...

# ═══════════════════════════════════════════════════════════════════════════
# 绘图辅助常量
# ═══════════════════════════════════════════════════════════════════════════

plot_analyse_marker: List[str]
"""12 种 matplotlib 标记: ``['s','*','+','x','p','h','v','X','D','P','H','o']``。"""

plot_analyse_color: List[str]
"""12 种十六进制颜色码，用于绘图时区分不同数据组。"""
