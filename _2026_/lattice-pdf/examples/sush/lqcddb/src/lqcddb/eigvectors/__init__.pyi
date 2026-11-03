"""
Type stub for lqcddb.eigvectors subpackage.

提供本征矢量压缩工具 (vector_creator) 和顶点函数工具 (vertex_creator)，
用于 distillation 框架中的 meson/baryon 关联函数构建。
"""
from typing import List, Literal, Optional
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# vector_creator — 本征矢量操作与压缩
# ═══════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════
# vertex_creator — 顶点函数工具
# ═══════════════════════════════════════════════════════════════════════════

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
        phase_exp: Optional[np.ndarray] = None,
        link_dir: str = "0",
        link_max: int = 0,
        gauge_link: Union[np.ndarray, bool, None] = None,
        eigvecs_max: Optional[np.ndarray] = None,
        conserved: bool = False,
    ) -> np.ndarray:
        """计算 sink 端带规范链接的 **V†·D·V** 关联函数。

        :math:`V_{mn}(p, \\Delta x) = \\sum_x e^{-ip\\cdot x} \\phi_m^\\dagger(x) U(x, x+\\Delta x) \\phi_n(x+\\Delta x)`

        根据输入参数分为三种情形：

        1. **无规范链接** (``gauge_link`` 为 bool/None 或 ``link_dir='0'``):
           计算 :math:`V_{mn}(p) = \\sum_x e^{-ipx} \\phi_m^\\dagger(x) \\phi_n(x)`
           → 形状 ``(N_mom, 1, Nev, Nev)``

        2. **守恒流/时间方向** (``conserved=True`` 或 ``link_dir='T'``):
           需提供 ``eigvecs_max`` (另一组本征矢量，时间片为 eigvecs+1)。
           使用时间方向规范链接 (Nd-index 3) 计算正向+反向输运。
           → 形状 ``(2, Nev, Nev)``

           - ``[0]`` = eigvecs† @ U_t @ eigvecs_max
           - ``[1]`` = eigvecs_max† @ U_t† @ eigvecs

        3. **空间方向** (``link_dir in {'X','Y','Z','all'}``, 且非守恒):
           沿空间方向构建规范输运路径，逐动量计算投影。
           ``link_max`` 控制输运路径的最大位移步数。
           → 形状 ``(N_mom, 2*link_max+1, Nev, Nev)``

        参数:
            eigvecs: 本征矢量，形状 ``(Nev, Nz, Ny, Nx, 3)``。
            phase_exp: 动量相因子 ``exp(-i p·x)``，形状会被展平为 ``(N_mom, V_full)``。
            link_dir: 链接方向。``'0'`` 无链接，``'T'`` 时间方向，
                ``'X'``/``'Y'``/``'Z'`` 空间单方向，``'all'`` 所有空间方向求和。
            link_max: 空间链接的最大位移长度 (仅情形 3)。
            gauge_link: 规范链接，应为已切片到特定时间片的数组
                ``(Nd, Nx, Nx, Nx, 3, 3)``。传入 bool/None 则跳过链接 (情形 1)。
            eigvecs_max: 第二组本征矢量 (仅情形 2 需要)，形状同 eigvecs。
            conserved: 是否守恒流计算模式。

        返回:
            VDV 关联函数矩阵。形状取决于情形：
            - 情形 1: ``(N_mom, 1, Nev, Nev)``
            - 情形 2: ``(2, Nev, Nev)``
            - 情形 3: ``(N_mom, 2*link_max+1, Nev, Nev)``
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
