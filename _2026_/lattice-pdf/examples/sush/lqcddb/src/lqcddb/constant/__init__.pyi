"""
Type stub for lqcddb.constant subpackage.

提供格点 QCD 物理常数、Dirac gamma 矩阵和 Pauli sigma 矩阵。
"""
from typing import List, Tuple, Union
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# constant.py — 物理常数
# ═══════════════════════════════════════════════════════════════════════════

Nc: int
"""颜色数，取值 3。"""
Ns: int
"""自旋分量数，取值 4。"""
Nd: int
"""时空维数，取值 4。"""
fm2GeV: float
"""转换因子，1 fm⁻¹ = 0.197 GeV。"""

# ═══════════════════════════════════════════════════════════════════════════
# gamma_matrix.py — Dirac gamma 矩阵
# ═══════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════
# sigma_matrix.py — Pauli sigma 矩阵
# ═══════════════════════════════════════════════════════════════════════════

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

    返回:
        ``p·σ`` 矩阵，形状 ``(2, 2)``，若 ``upto4dim`` 则为 ``(4, 4)``。
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
