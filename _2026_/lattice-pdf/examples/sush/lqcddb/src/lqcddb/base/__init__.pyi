"""
Type stub for lqcddb.base subpackage.

提供后端管理、张量工具函数、SU(2) Clebsch-Gordan 代数、
MPI 通信基础设施和规范场平滑等功能。
"""
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np

# 尝试导入 cupy 以提供更精确的后端类型（安装 cupy 时 IDE 可区分 numpy/cupy）
try:
    import cupy as cp
except ImportError:
    cp = np  # type: ignore[assignment]  # cupy 不可用时退化为 numpy 类型

# ═══════════════════════════════════════════════════════════════════════════
# backend.py — 计算后端切换
# ═══════════════════════════════════════════════════════════════════════════

def get_backend() -> np:
    """返回当前计算后端模块 (numpy 或 cupy)。"""
    ...
def set_backend(backend: Literal["numpy", "cupy"]) -> None:
    """切换全局计算后端。

    参数:
        backend: 目标后端，``"numpy"`` 或 ``"cupy"``。
    """
    ...

# ═══════════════════════════════════════════════════════════════════════════
# base_functions.py — 基础张量工具
# ═══════════════════════════════════════════════════════════════════════════

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
        """返回 ``np.ix_`` 风格的索引网格。

        参数:
            dims: 要切片的维度列表。
            indices: 每个维度的索引列表。

        返回:
            可用于高级索引的网格元组。
        """
        ...
    def slice(self, dims: List[int], indices: List[Any]) -> np.ndarray:
        """按指定维度和索引读取子数组。

        参数:
            dims: 要切片的维度列表。
            indices: 每个维度的索引列表。

        返回:
            切片后的子数组。
        """
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

        返回:
            修改后的数组。
        """
        ...
    def get_slice_shape(
        self, dims: List[int], indices: List[Any]
    ) -> Tuple[int, ...]:
        """返回切片后的形状。

        参数:
            dims: 要切片的维度列表。
            indices: 每个维度的索引列表。

        返回:
            切片后的形状元组。
        """
        ...
    def get_info(self) -> Dict[str, Any]:
        """返回 ``{shape, ndim, dtype}`` 字典。

        返回:
            ``{'shape': ..., 'ndim': ..., 'dtype': ...}``。
        """
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

# ═══════════════════════════════════════════════════════════════════════════
# cg_coeff.py — SU(2) Clebsch-Gordan 系数
# ═══════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════
# mpi_init.py — MPI 基础设施
# ═══════════════════════════════════════════════════════════════════════════

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

from mpi4py import MPI

def getMPIComm() -> MPI.Comm:
    """返回 MPI 通信器。"""
    ...
def getMPIRank() -> int:
    """返回本进程 MPI rank。"""
    ...
def getMPISize() -> int:
    """返回 MPI 总进程数。"""
    ...

# ═══════════════════════════════════════════════════════════════════════════
# smear_gauge.py — 规范场平滑
# ═══════════════════════════════════════════════════════════════════════════

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
