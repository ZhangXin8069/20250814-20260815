#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
格点 QCD 计算质子中非极化胶子 PDF 的完整工作流
================================================================================

理论框架: 大动量有效理论 (LaMET)
方法: Distillation + 动量涂抹 + Disconnected 图分解

该模块实现了从格点规范组态到光锥胶子 PDF g(x,μ) 的完整计算链路:

  1. 读取规范组态 (gauge links)
  2. 构造场强张量 F_{μν} (Clover plaquette)
  3. 构造对偶场强张量 F̃_{μν} (Levi-Civita 收缩)
  4. 构造 nonlocal 胶子 OPE 算符 O_{μν}(z) (包含 Wilson 线)
  5. 实现 Distillation 框架 (本征矢量 + Perambulator + VVV)
  6. 动量涂抹质子两点关联函数
  7. 矩阵元提取 h(z, P_z)
  8. Fourier 变换 → quasi-PDF g̃(x, P_z)
  9. 微扰匹配 → 光锥 PDF g(x, μ)
  10. Jackknife 统计误差分析

参考文献:
  - Note of gluon PDFs (内部笔记, Eq(20) & Eq(25))
  - Unpolarized gluon PDF of the nucleon from lattice QCD in the continuum limit
  - First Nucleon Gluon PDF from Large Momentum Effective Theory

作者: 基于 donghx 原始代码的系统化重构
日期: 2026-07-09
================================================================================
"""

import numpy as np
import os
import sys
import time
import json
from typing import Dict, Tuple, List, Optional, Union
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor

# 尝试导入 GPU 加速库 (可选)
try:
    import cupy as cp
    HAS_CUPY = True
    print("[INFO] CuPy GPU acceleration enabled.")
except ImportError:
    cp = np  # fallback to numpy
    HAS_CUPY = False
    print("[INFO] CuPy not available, using NumPy (CPU-only mode).")

# 尝试导入优化的张量缩并库
try:
    from opt_einsum import contract
    print("[INFO] opt_einsum enabled for optimized tensor contractions.")
except ImportError:
    contract = np.einsum  # fallback
    print("[INFO] opt_einsum not available, using numpy.einsum.")

# ============================================================================
# 第一部分: 全局配置与数据类型
# ============================================================================

@dataclass
class LatticeConfig:
    """
    格点几何与物理参数配置

    Attributes
    ----------
    Nt : int
        时间方向格点数
    Nx, Ny, Nz : int
        空间方向格点数
    Nc : int
        颜色数 (QCD: Nc=3)
    Nd : int
        Dirac 旋量维数 (Nd=4)
    """

    Nt: int = 64
    Nx: int = 32
    Ny: int = 32
    Nz: int = 32
    Nc: int = 3
    Nd: int = 4

    @property
    def lattice_shape(self) -> Tuple[int, ...]:
        """返回格点的标准形状: (Nt, Nz, Ny, Nx)."""
        return (self.Nt, self.Nz, self.Ny, self.Nx)

    @property
    def spatial_volume(self) -> int:
        """空间体积 V = Nx * Ny * Nz."""
        return self.Nx * self.Ny * self.Nz


@dataclass
class CalcParams:
    """
    计算参数配置

    Attributes
    ----------
    Nev : int
        Laplace 算符的本征矢量总数
    Nev1 : int
        实际用于收缩的本征矢量数 (Nev1 ≤ Nev)
    Px, Py, Pz : int
        动量分量 (以 2π/L 为单位)
    mom_smear : int
        动量涂抹参数
    delta_z : int
        Wilson 线最大长度 (格距单位)
    z_dir : int
        Wilson 线方向: 0(x), 1(y), 2(z)
    conf_id : str
        规范组态编号
    """

    Nev: int = 100
    Nev1: int = 100
    Px: int = 0
    Py: int = 0
    Pz: int = 0
    mom_smear: int = 3
    delta_z: int = 15
    z_dir: int = 2  # 默认 z-方向
    conf_id: str = "20000"


# ============================================================================
# 第二部分: Dirac 矩阵 (DeGrand-Rossi 基)
# ============================================================================

def build_gamma_matrices_degrand_rossi() -> Dict[int, np.ndarray]:
    """
    构造 DeGrand-Rossi (DR) 基下的所有 Dirac 矩阵。

    在 DR 基 (又称手征基的变体) 下，gamma 矩阵定义为:

        γ₀ = ID (4×4)
        γ₁ = i * anti-diag(1, 1, -1, -1)  [γ_x]
        γ₂ = anti-diag(-1, 1, 1, -1)       [γ_y]
        γ₃ = i * anti-diag(1, -1, -1, 1)   [γ_z]
        γ₄ = anti-diag(1, 1, 1, 1)         [γ_t]
        γ₅ = diag(1, 1, -1, -1)

    Returns
    -------
    gamma_dict : dict
        字典，键为索引，值为 4×4 复数矩阵
        索引含义:
            0  = I (单位矩阵)
            1  = γ₁ (γ_x)
            2  = γ₂ (γ_y)
            3  = γ₃ (γ_z)
            4  = γ₄ (γ_t)
            5  = γ₅
            6  = γ₂γ₃ = -γ₁γ₄γ₅
            7  = γ₃γ₁ = -γ₂γ₄γ₅
            8  = γ₁γ₂ = -γ₃γ₄γ₅
            9  = γ₁γ₄
            10 = γ₂γ₄
            11 = γ₃γ₄
            12 = γ₁γ₅
            13 = γ₂γ₅
            14 = γ₃γ₅
            15 = γ₄γ₅
            16 = γ₃γ₁ * P₊  [正宇称投影]
            17 = γ₃γ₁ * P₋  [负宇称投影]
    """
    # 初始化 4×4 零矩阵
    g0 = np.zeros((4, 4), dtype=complex)  # 单位矩阵
    g1 = np.zeros((4, 4), dtype=complex)  # γ_x
    g2 = np.zeros((4, 4), dtype=complex)  # γ_y
    g3 = np.zeros((4, 4), dtype=complex)  # γ_z
    g4 = np.zeros((4, 4), dtype=complex)  # γ_t
    g5 = np.zeros((4, 4), dtype=complex)  # γ₅

    # --- 单位矩阵 γ₀ = I₄ ---
    for i in range(4):
        g0[i, i] = 1.0 + 0.0j

    # --- γ₁ (γ_x): 反对角线, i * (1,1,-1,-1) ---
    g1[0, 3] = 0.0 + 1.0j
    g1[1, 2] = 0.0 + 1.0j
    g1[2, 1] = 0.0 - 1.0j
    g1[3, 0] = 0.0 - 1.0j

    # --- γ₂ (γ_y): 反对角线, (-1,1,1,-1) ---
    g2[0, 3] = -1.0 + 0.0j
    g2[1, 2] = 1.0 + 0.0j
    g2[2, 1] = 1.0 + 0.0j
    g2[3, 0] = -1.0 + 0.0j

    # --- γ₃ (γ_z): 反对角线, i * (1,-1,-1,1) ---
    g3[0, 2] = 0.0 + 1.0j
    g3[1, 3] = 0.0 - 1.0j
    g3[2, 0] = 0.0 - 1.0j
    g3[3, 1] = 0.0 + 1.0j

    # --- γ₄ (γ_t): 反对角线, (1,1,1,1) ---
    g4[0, 2] = 1.0 + 0.0j
    g4[1, 3] = 1.0 + 0.0j
    g4[2, 0] = 1.0 + 0.0j
    g4[3, 1] = 1.0 + 0.0j

    # --- γ₅ = γ₁γ₂γ₃γ₄ = diag(1,1,-1,-1) ---
    g5[0, 0] = 1.0 + 0.0j
    g5[1, 1] = 1.0 + 0.0j
    g5[2, 2] = -1.0 + 0.0j
    g5[3, 3] = -1.0 + 0.0j

    return {
        0: g0,
        1: g1, 2: g2, 3: g3, 4: g4, 5: g5,
        6: g2 @ g3,                          # -γ₁γ₄γ₅
        7: g3 @ g1,                          # -γ₂γ₄γ₅
        8: g1 @ g2,                          # -γ₃γ₄γ₅
        9: g1 @ g4,                          # γ₁γ₄
        10: g2 @ g4,                         # γ₂γ₄
        11: g3 @ g4,                         # γ₃γ₄
        12: g1 @ g5,                         # γ₁γ₅
        13: g2 @ g5,                         # γ₂γ₅
        14: g3 @ g5,                         # γ₃γ₅
        15: g4 @ g5,                         # γ₄γ₅
        16: (g3 @ g1) @ (0.5 * (g0 + g4)),   # 正宇称投影
        17: (g3 @ g1) @ (0.5 * (g0 - g4)),   # 负宇称投影
    }


# ============================================================================
# 第三部分: Levi-Civita 张量与场强张量构造
# ============================================================================

def build_levi_civita_tensor() -> np.ndarray:
    """
    构造四维 Levi-Civita 符号 ε_{μνρσ} / 2。

    ε_{μνρσ} 定义为:
        +1  如果 (μ,ν,ρ,σ) 是 (0,1,2,3) 的偶排列
        -1  如果是奇排列
         0  如果有重复指标

    Returns
    -------
    epsilon4 : np.ndarray, shape (4, 4, 4, 4)
        ε_{μνρσ} / 2，缩并 F^{ρσ} 得到 F̃_{μν}
    """
    epsilon4 = np.zeros((4, 4, 4, 4), dtype=float)

    for mu in range(4):
        for nu in range(4):
            # 计算比 mu 小的指标数 (用于确定排列的奇偶性)
            a = 1.0 if mu > nu else 0.0
            for rho in range(4):
                b = 0.0
                if mu > rho:
                    b += 1.0
                if nu > rho:
                    b += 1.0
                for sigma in range(4):
                    c = 0.0
                    if mu > sigma:
                        c += 1.0
                    if nu > sigma:
                        c += 1.0
                    if rho > sigma:
                        c += 1.0

                    # 确定排列的奇偶性
                    parity = (a + b + c)
                    if parity % 2 == 0:
                        epsilon4[mu, nu, rho, sigma] = 1.0
                    else:
                        epsilon4[mu, nu, rho, sigma] = -1.0

                    # 如果有重复指标 → 0
                    if len({mu, nu, rho, sigma}) != 4:
                        epsilon4[mu, nu, rho, sigma] = 0.0

    # 因子 1/2 使得 F̃_{μν} = ε_{μνρσ} * F^{ρσ} / 2
    epsilon4 *= 0.5
    return epsilon4


def plaquette(gauge: np.ndarray, mu: int, nu: int,
              Nt: int, Nx: int) -> np.ndarray:
    """
    计算 μ-ν 平面上的标准 plaquette:

        P_{μν}(x) = U_μ(x) · U_ν(x+μ̂) · U_μ^†(x+ν̂) · U_ν^†(x)

    Parameters
    ----------
    gauge : np.ndarray, shape (Nt, Nz, Ny, Nx, 4, 3, 3)
        SU(3) 规范链接变量
    mu, nu : int
        Lorentz 指标 (0=x, 1=y, 2=z, 3=t)
    Nt, Nx : int
        格点维度

    Returns
    -------
    pla : np.ndarray, shape (Nt, Nz, Ny, Nx, 3, 3)
        P_{μν}(x)
    """
    # Step 1: U_μ(x) · U_ν(x+μ̂)
    pla = contract(
        "tzyxab, tzyxbc -> tzyxac",
        gauge[:, :, :, :, mu, :, :],
        np.roll(gauge, -1, axis=3 - mu)[:, :, :, :, nu, :, :],
    )
    # Step 2: · U_μ^†(x+ν̂)
    pla = contract(
        "tzyxab, tzyxcb -> tzyxac",
        pla,
        np.roll(gauge, -1, axis=3 - nu)[:, :, :, :, mu, :, :].conj(),
    )
    # Step 3: · U_ν^†(x)
    pla = contract(
        "tzyxab, tzyxcb -> tzyxac",
        pla,
        gauge[:, :, :, :, nu, :, :].conj(),
    )
    return pla


def plaquette_clover(gauge: np.ndarray, mu: int, nu: int) -> np.ndarray:
    """
    计算 Clover 型的场强张量 F_{μν}(x)。

    Clover 叶由围绕点 x 的四个 plaquette 组成:

        Q_{μν}(x) = P_{μν}(x) + P_{ν,-μ}(x) + P_{-μ,-ν}(x) + P_{-ν,μ}(x)

    则场强张量为:

        F_{μν}(x) = -i/(8) · (Q_{μν} - Q_{μν}^†)  [在 a=1, g₀=1 单位下]

    注意: 这里忽略了因子 1/(a²g₀)，因为在准 PDF 算符中它们会消去。

    四个 plaquette 的几何排列:
    ```
              μ →
          +---(1)---+
          |         |
        ν ↑  (2)  (4)  ↓ (反方向)
          |         |
          +---(3)---+
              ← (反方向)
    ```
    (1) P_{μν}:   标准 1×1 Wilson loop
    (2) P_{ν,-μ}: 从 x 先走 ν 再走 -μ 方向
    (3) P_{-μ,-ν}: 从 x 先走 -μ 再走 -ν 方向
    (4) P_{-ν,μ}: 从 x 先走 -ν 再走 μ 方向

    Parameters
    ----------
    gauge : np.ndarray, shape (Nt, Nz, Ny, Nx, 4, 3, 3)
        SU(3) 规范链接
    mu, nu : int
        Lorentz 指标

    Returns
    -------
    F_munu : np.ndarray, shape (Nt, Nz, Ny, Nx, 3, 3)
        场强张量 F_{μν}(x)
    """
    # --- 准备平移后的规范场 (用于 4 个 plaquette) ---
    gauge_rightup = gauge                                    # x
    gauge_leftup = np.roll(gauge, 1, axis=3 - mu)           # x - μ̂
    gauge_rightdown = np.roll(gauge, 1, axis=3 - nu)        # x - ν̂
    gauge_leftdown = np.roll(gauge_leftup, 1, axis=3 - nu)  # x - μ̂ - ν̂

    # --- Plaquette (1): P_{μν} ---
    # U_μ(x) · U_ν(x+μ̂) · U_μ^†(x+ν̂) · U_ν^†(x)
    pla_ru = contract(
        "tzyxab, tzyxbc -> tzyxac",
        gauge_rightup[:, :, :, :, mu, :, :],
        np.roll(gauge_rightup, -1, axis=3 - mu)[:, :, :, :, nu, :, :],
    )
    pla_ru = contract(
        "tzyxab, tzyxcb -> tzyxac",
        pla_ru,
        np.roll(gauge_rightup, -1, axis=3 - nu)[:, :, :, :, mu, :, :].conj(),
    )
    pla_ru = contract(
        "tzyxab, tzyxcb -> tzyxac",
        pla_ru,
        gauge_rightup[:, :, :, :, nu, :, :].conj(),
    )

    # --- Plaquette (2): P_{ν,-μ} ---
    # U_ν(x-μ̂) · U_{-μ}(x-μ̂+ν̂) · U_ν^†(x-μ̂-ν̂) · U_{-μ}^†(x-μ̂)
    # 等价于:
    # U_ν(x-μ̂) · U_μ^†(x-μ̂+ν̂-μ̂) ... → 使用 U_μ^† = U_μ(x-μ̂)†
    pla_lu = contract(
        "tzyxab, tzyxcb -> tzyxac",
        np.roll(gauge_leftup, -1, axis=3 - mu)[:, :, :, :, nu, :, :],
        np.roll(gauge_leftup, -1, axis=3 - nu)[:, :, :, :, mu, :, :].conj(),
    )
    pla_lu = contract(
        "tzyxab, tzyxcb -> tzyxac",
        pla_lu,
        gauge_leftup[:, :, :, :, nu, :, :].conj(),
    )
    pla_lu = contract(
        "tzyxab, tzyxbc -> tzyxac",
        pla_lu,
        gauge_leftup[:, :, :, :, mu, :, :],
    )

    # --- Plaquette (3): P_{-μ,-ν} ---
    # U_{-μ}(x-μ̂-ν̂) · U_{-ν}(x-μ̂-ν̂-μ̂) ...
    pla_ld = contract(
        "tzyxba, tzyxcb -> tzyxac",
        np.roll(gauge_leftdown, -1, axis=3 - nu)[:, :, :, :, mu, :, :].conj(),
        gauge_leftdown[:, :, :, :, nu, :, :].conj(),
    )
    pla_ld = contract(
        "tzyxab, tzyxbc -> tzyxac",
        pla_ld,
        gauge_leftdown[:, :, :, :, mu, :, :],
    )
    pla_ld = contract(
        "tzyxab, tzyxbc -> tzyxac",
        pla_ld,
        np.roll(gauge_leftdown, -1, axis=3 - mu)[:, :, :, :, nu, :, :],
    )

    # --- Plaquette (4): P_{-ν,μ} ---
    # U_{-ν}(x-ν̂) · U_μ(x-ν̂-ν̂) ... → 类似构造
    pla_rd = contract(
        "tzyxba, tzyxbc -> tzyxac",
        gauge_rightdown[:, :, :, :, nu, :, :].conj(),
        gauge_rightdown[:, :, :, :, mu, :, :],
    )
    pla_rd = contract(
        "tzyxab, tzyxbc -> tzyxac",
        pla_rd,
        np.roll(gauge_rightdown, -1, axis=3 - mu)[:, :, :, :, nu, :, :],
    )
    pla_rd = contract(
        "tzyxab, tzyxcb -> tzyxac",
        pla_rd,
        np.roll(gauge_rightdown, -1, axis=3 - nu)[:, :, :, :, mu, :, :].conj(),
    )

    # --- 构造反厄米部分: Q - Q^† ---
    # F_{μν} ∝ -i * (Q - Q^†)
    ans = (
        pla_ru - pla_ru.conj().transpose(0, 1, 2, 3, 5, 4)
        + pla_lu - pla_lu.conj().transpose(0, 1, 2, 3, 5, 4)
        + pla_ld - pla_ld.conj().transpose(0, 1, 2, 3, 5, 4)
        + pla_rd - pla_rd.conj().transpose(0, 1, 2, 3, 5, 4)
    )
    return -1j * ans / 8.0


def compute_field_strength_all(gauge: np.ndarray,
                                Nt: int, Nx: int) -> np.ndarray:
    """
    计算所有独立 (μ, ν) 对的 Clover 场强张量。

    Parameters
    ----------
    gauge : np.ndarray
        SU(3) 规范链接
    Nt, Nx : int
        格点维度

    Returns
    -------
    F_all : np.ndarray, shape (4, 4, Nt, Nz, Ny, Nx, 3, 3)
        F_all[μ, ν] = F_{μν}(x)，且 F_{μμ} = 0
    """
    F_all = np.zeros((4, 4, Nt, Nx, Nx, Nx, 3, 3), dtype=complex)
    for mu in range(4):
        for nu in range(4):
            if mu != nu:
                F_all[mu, nu] = plaquette_clover(gauge, mu, nu)
            # mu == nu 的情况恒为零
    return F_all


def compute_dual_field_strength(F_all: np.ndarray,
                                 epsilon: np.ndarray) -> np.ndarray:
    """
    通过对偶变换从 F_{μν} 计算 F̃_{μν}:

        F̃_{μν} = (1/2) ε_{μνρσ} F^{ρσ}

    Parameters
    ----------
    F_all : np.ndarray, shape (4, 4, Nt, Nz, Ny, Nx, 3, 3)
        所有 (μ,ν) 分量的场强张量
    epsilon : np.ndarray, shape (4, 4, 4, 4)
        ε_{μνρσ} / 2 张量

    Returns
    -------
    F_tilde_all : np.ndarray, shape (4, 4, Nt, Nz, Ny, Nx, 3, 3)
        F̃_{μν}(x)
    """
    F_tilde_all = contract(
        "opmn, mntzyxab -> optzyxab",
        epsilon,
        F_all,
    )
    return F_tilde_all


# ============================================================================
# 第四部分: Nonlocal 胶子 OPE 算符的构造
# ============================================================================

def gluon_ope_operator_z0_mu2(
    gauge: np.ndarray,
    F_all: np.ndarray,
    F_tilde_all: np.ndarray,
    delta_z: int,
    z_dir: int,
    mu: int,
    nu: int,
    mu2: int,
    nu2: int,
) -> np.ndarray:
    """
    构造非定域胶子 OPE 算符（z=0 处插入 F̃，z 处插入 F）:

        O(z) = Tr_c[ F_{zμ}(z) · W(z→0) · F̃_{ν}^{z}(0) · W(0→z) ]

    这是 Eq(25) 的实现。

    几何结构 (以 z_dir=2 为例，Wilson 线沿 z 方向):

        x=0 (z=0):  F̃_{μ2,ν2}(0)
                        │
                  W(0→z) ↑  (U_z 链接)
                        │
        x=z (z=dz): F_{μ,ν}(z)

    其中 F̃ 是 F 的对偶场强。

    Parameters
    ----------
    gauge : np.ndarray
        SU(3) 规范链接
    F_all : np.ndarray, shape (4, 4, Nt, Nz, Ny, Nx, 3, 3)
        场强张量
    F_tilde_all : np.ndarray, shape (4, 4, Nt, Nz, Ny, Nx, 3, 3)
        对偶场强张量
    delta_z : int
        Wilson 线长度 (格距单位), z = delta_z
    z_dir : int
        Wilson 线方向 (0=x, 1=y, 2=z)
    mu, nu : int
        z 处 F 的 Lorentz 指标
    mu2, nu2 : int
        0 处 F̃ 的 Lorentz 指标

    Returns
    -------
    op_trace : np.ndarray, shape (Nt,)
        每个时间片的 OPE 算符迹 (已对空间求和)
    """
    Nt = gauge.shape[0]

    # Step 1: 将 F_{μν} 平移到 z = delta_z 处
    ope = np.roll(F_all[mu, nu], -delta_z, axis=3 - z_dir)

    # Step 2: 从 z 到 0 的 Wilson 线 (即 U_z 的共轭)
    #   W(z→0) = Π_{k=dz-1}^{0} U_z^†(k)
    for dz_step in range(delta_z):
        shift_amount = -(delta_z - 1 - dz_step)
        ope = contract(
            "tzyxab, tzyxcb -> tzyxac",
            ope,
            np.roll(gauge, shift_amount, axis=3 - z_dir)[
                :, :, :, :, z_dir, :, :
            ].conj(),
        )

    # Step 3: 在 z=0 处插入 F̃_{μ2, ν2}(0)
    ope = contract(
        "tzyxab, tzyxbc -> tzyxac",
        ope,
        F_tilde_all[mu2, nu2],
    )

    # Step 4: 从 0 到 z 的 Wilson 线 (U_z 正向)
    #   W(0→z) = Π_{k=0}^{dz-1} U_z(k)
    for dz_step in range(delta_z):
        shift_amount = -dz_step
        ope = contract(
            "tzyxab, tzyxbc -> tzyxac",
            ope,
            np.roll(gauge, shift_amount, axis=3 - z_dir)[
                :, :, :, :, z_dir, :, :
            ],
        )

    # Step 5: 对颜色空间取迹 Tr_c[...] → 标量
    trace_color = np.trace(ope, axis1=4, axis2=5)

    # Step 6: 对空间求和 Σ_{x,y,z}
    op_trace = np.sum(trace_color, axis=(1, 2, 3))

    return op_trace


def compute_ope_all_z(
    gauge: np.ndarray,
    F_all: np.ndarray,
    F_tilde_all: np.ndarray,
    delta_z: int,
    z_dir: int,
    mu: int,
    nu: int,
    mu2: int,
    nu2: int,
    Nt: int,
) -> np.ndarray:
    """
    为所有 z ∈ [0, delta_z-1] 计算 OPE 算符。

    这是 Calc_ope_unpol.py 的核心计算逻辑。

    Parameters
    ----------
    (同上，但循环遍历所有 delta_z)

    Returns
    -------
    ops : np.ndarray, shape (Nt, delta_z)
        ops[t, dz] = O_{μν, μ2ν2}(z=dz) 在时间 t 的值
    """
    ops = np.zeros((Nt, delta_z), dtype=complex)

    for dz in range(delta_z):
        print(f"  [OPE] Computing z = {dz}/{delta_z-1} "
              f"for mu={mu}, nu={nu}, mu2={mu2}, nu2={nu2}")
        ops[:, dz] = gluon_ope_operator_z0_mu2(
            gauge, F_all, F_tilde_all, dz,
            z_dir, mu, nu, mu2, nu2
        )

    return ops


# ============================================================================
# 第五部分: Distillation 框架
# ============================================================================

def compute_phase_factor(momentum: np.ndarray,
                          Nx: int) -> np.ndarray:
    """
    计算动量涂抹的相位因子:

        φ_P(x) = exp(-i * 2π * P·x / L)

    Parameters
    ----------
    momentum : np.ndarray, shape (3,)
        动量矢量 P = (Pz, Py, Px)，以 2π/L 为单位
    Nx : int
        空间方向的格点数

    Returns
    -------
    phase : np.ndarray, shape (Nx^3,)
        exp(-i * 2π * P·x / L) 在每个空间点的值
    """
    V = Nx * Nx * Nx
    phase = np.zeros(V, dtype=complex)
    idx = 0
    for z in range(Nx):
        for y in range(Nx):
            for x in range(Nx):
                pos = np.array([z, y, x])  # 注意: (z, y, x) 顺序
                phase[idx] = np.exp(
                    -np.dot(momentum, pos) * 2.0j * np.pi / Nx
                )
                idx += 1
    return phase


def compute_vvv_baryon_block(
    eigvecs: np.ndarray,
    phase_factor: np.ndarray,
    Nev1: int,
    Nx: int,
) -> np.ndarray:
    """
    计算 VVV (Baryon Block) 张量:

        Φ_{abc}(P) = Σ_x e^{-iP·x} ε_{ijk} v_i^a(x) v_j^b(x) v_k^c(x)

    其中 i,j,k ∈ {0,1,2} 是色指标，a,b,c 是本征矢量指标。

    VVV 是 distillation 框架中夸克三体波函数的动量投影，
    用于质子插值算符的构造。

    Parameters
    ----------
    eigvecs : np.ndarray, shape (Nev, Nx^3, 3)
        Laplace 本征矢量 v_i^a(x)
    phase_factor : np.ndarray, shape (Nx^3,)
        动量相位因子
    Nev1 : int
        截断的本征矢量数
    Nx : int
        空间格点数

    Returns
    -------
    VVV : np.ndarray, shape (Nev1, Nev1, Nev1)
        色收缩后的三夸克张量 (ε_{ijk} 已收缩)

    注意: 为了处理显存限制，对大格点需要分片计算（逐 x-层）。
    """
    VVV = np.zeros((Nev1, Nev1, Nev1), dtype=complex)
    layer_size = Nx * Nx  # 每层包含 Nx*Nx 个空间点

    for xi in range(Nx):
        # 取出第 xi 层的本征矢量和相位因子
        start = xi * layer_size
        end = (xi + 1) * layer_size

        eig_slice = eigvecs[:Nev1, start:end, :]      # (Nev1, Nx², 3)
        phase_slice = phase_factor[start:end]          # (Nx²,)

        # ε_{ijk} 收缩: 六个置换，符号由排列奇偶性决定
        # 三项正号 (偶排列: 012, 120, 201):
        VVV += contract(
            "x, ax, bx, cx -> abc",          # ε_{012} = +1
            phase_slice,
            eig_slice[:, :, 0],              # v^a_0
            eig_slice[:, :, 1],              # v^b_1
            eig_slice[:, :, 2],              # v^c_2
        )
        VVV += contract(
            "x, ax, bx, cx -> abc",          # ε_{120} = +1
            phase_slice,
            eig_slice[:, :, 1],
            eig_slice[:, :, 2],
            eig_slice[:, :, 0],
        )
        VVV += contract(
            "x, ax, bx, cx -> abc",          # ε_{201} = +1
            phase_slice,
            eig_slice[:, :, 2],
            eig_slice[:, :, 0],
            eig_slice[:, :, 1],
        )

        # 三项负号 (奇排列: 021, 102, 210):
        VVV -= contract(
            "x, ax, bx, cx -> abc",          # ε_{021} = -1
            phase_slice,
            eig_slice[:, :, 0],
            eig_slice[:, :, 2],
            eig_slice[:, :, 1],
        )
        VVV -= contract(
            "x, ax, bx, cx -> abc",          # ε_{102} = -1
            phase_slice,
            eig_slice[:, :, 1],
            eig_slice[:, :, 0],
            eig_slice[:, :, 2],
        )
        VVV -= contract(
            "x, ax, bx, cx -> abc",          # ε_{210} = -1
            phase_slice,
            eig_slice[:, :, 2],
            eig_slice[:, :, 1],
            eig_slice[:, :, 0],
        )

    return VVV


def contract_proton_2pt_single_tsrc(
    VVV_sink_t: np.ndarray,
    peram_u: np.ndarray,
    CG5peram_uCG5: np.ndarray,
    VVV_source_t: np.ndarray,
) -> np.ndarray:
    """
    对单个 (t_sink, t_source) 组合执行质子两点函数的 Wick 收缩。

    收缩公式 (两个费曼图):
        C₂ = Tr[ Φ(t_sink) · τ₁ · (Γτ₂Γ) · τ₃ · Φ(t_src)* ]  [直接项]
           - Tr[ Φ(t_sink) · τ₁ · (Γτ₂Γ) · τ₃ · Φ(t_src)* ]  [交换项]

    其中:
        Φ  = VVV baryon block
        τ  = perambulator (夸克传播子)
        Γ  = 插值算符的 Dirac 结构 (例如 Cγ₅γ_μ)

    直接项和交换项的差异在于夸克线中指标的不同配对方式。

    Parameters
    ----------
    VVV_sink_t : np.ndarray, shape (Nev1, Nev1, Nev1)
        sink 时间片的 VVV
    peram_u : np.ndarray, shape (4, 4, Nev1, Nev1)
        sink 时间片的 perambulator τ(t_sink, t_src)
    CG5peram_uCG5 : np.ndarray, shape (4, 4, Nev1, Nev1)
        插入 Dirac 矩阵后的 perambulator: Γ · τ · Γ
        其中 Γ = Cγ₅γ_μ
    VVV_source_t : np.ndarray, shape (Nev1, Nev1, Nev1)
        source 时间片的 VVV 复共轭

    Returns
    -------
    C2_matrix : np.ndarray, shape (4, 4)
        C2_matrix[i, j] = 质子两点函数 (Dirac 指标 i, j)
    """
    # 直接项:
    # Φ_{abc} · τ_{gi}^{ad} · (ΓτΓ)_{gj}^{be} · τ_{il}^{cf} · Φ_{def}^*
    direct = contract(
        "abc, gjad, gjbe, ilcf, def -> il",
        VVV_sink_t,
        peram_u,
        CG5peram_uCG5,
        peram_u,
        VVV_source_t,
    )

    # 交换项:
    # Φ_{abc} · τ_{gl}^{af} · (ΓτΓ)_{gj}^{be} · τ_{ij}^{cd} · Φ_{def}^*
    exchange = contract(
        "abc, glaf, gjbe, ijcd, def -> il",
        VVV_sink_t,
        peram_u,
        CG5peram_uCG5,
        peram_u,
        VVV_source_t,
    )

    return direct - exchange


# ============================================================================
# 第六部分: 数据 I/O 函数
# ============================================================================

def read_gauge_config(filename: str, Nt: int, Nx: int) -> np.ndarray:
    """
    从 ILDG 格式的二进制文件读取 SU(3) 规范组态。

    文件格式:
        - 大端序 (big-endian) float64
        - 维度: (Nt, Nx, Nx, Nx, 4, 3, 3, 2)
        - 最后维度 2 对应复数的实部和虚部

    Parameters
    ----------
    filename : str
        规范组态文件路径
    Nt, Nx : int
        格点维度

    Returns
    -------
    gauge : np.ndarray, shape (Nt, Nx, Nx, Nx, 4, 3, 3)
        规范链接 U_μ(t, z, y, x) (3×3 复矩阵)
    """
    with open(filename, "rb") as f:
        raw = np.fromfile(f, dtype=">f8")  # 大端序 float64

    # 重构为复数矩阵
    raw = raw.reshape(Nt, Nx, Nx, Nx, 4, 3, 3, 2)
    gauge = raw[..., 0] + 1j * raw[..., 1]
    return gauge


def read_eigenvectors(eig_dir: str, t: int, Nev1: int,
                       conf_id: str, Nx: int) -> np.ndarray:
    """
    读取单个时间片 t 的 Laplace 本征矢量。

    文件格式:
        - 64 字节头部, 然后 float64 数据
        - 形状: (Nev, Nx, Nx, Nx, 3, 2)
        - 最后维度为复数的实部/虚部

    Parameters
    ----------
    eig_dir : str
        本征矢量目录
    t : int
        时间片索引
    Nev1 : int
        截断本征矢量数
    conf_id : str
        组态编号
    Nx : int
        空间格点数

    Returns
    -------
    eigvecs : np.ndarray, shape (Nev1, Nx^3, 3)
        本征矢量 v_i^a(x)
    """
    filename = f"{eig_dir}/eigvecs_t{t:03d}_{conf_id}"
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype="f8").reshape(-1, Nx, Nx, Nx, 3, 2)

    eigvecs = data[..., 0] + 1j * data[..., 1]
    eigvecs = eigvecs[:Nev1]
    eigvecs = eigvecs.reshape(Nev1, Nx * Nx * Nx, 3)
    return eigvecs


def read_perambulator(peram_dir: str, conf_id: str, Nt: int,
                       Nev1: int, t_source: int) -> np.ndarray:
    """
    读取单个 source 时间片的 perambulator（夸克传播子的子空间投影）。

    Perambulator τ(t_sink; t_src) 的维度:
        (Nt, 4, 4, Nev1, Nev1)，即
        (t_sink, d_sink, d_source, ev_sink, ev_source)

    文件包括 4 个 Dirac 分量 (d_source = 0,1,2,3)。

    Parameters
    ----------
    peram_dir : str
        Perambulator 目录
    conf_id : str
        组态编号
    Nt : int
        时间格点数
    Nev1 : int
        截断本征矢量数
    t_source : int
        Source 时间片

    Returns
    -------
    peram : np.ndarray, shape (Nt, 4, 4, Nev1, Nev1)
        Perambulator 数组
    """
    # 读取 4 个 Dirac 分量
    peram_list = []
    for d_source in range(4):
        fname = f"{peram_dir}/perams.{conf_id}.{d_source}.{t_source}"
        with open(fname, "rb") as f:
            peram_list.append(np.fromfile(f, dtype="f8"))

    peram = np.concatenate(peram_list)

    # 从文件大小反推 Nev
    peram_size = peram.size
    Nev_full = int(np.sqrt(peram_size / (4 * 4 * Nt * 2)))

    # 重构: (d_source, t_sink, ev_source, d_sink, ev_sink, complex)
    peram = peram.reshape(4, Nt, Nev_full, 4, Nev_full, 2)
    # 转置: → (t_sink, d_sink, d_source, ev_sink, ev_source, complex)
    peram = peram.transpose(1, 3, 0, 4, 2, 5)
    # 合并复数的实部/虚部
    peram = peram[..., 0] + 1j * peram[..., 1]
    # 截断到 Nev1
    peram = peram[:, :, :, :Nev1, :Nev1]

    return peram


# ============================================================================
# 第七部分: 矩阵元提取与 quasi-PDF 重构
# ============================================================================

def extract_matrix_element_from_ratio(
    C3: np.ndarray,
    C2: np.ndarray,
    tsep_min: int,
    tsep_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    通过关联函数比提取基态矩阵元:

        h(z, P_z, Δt) = C₃(Δt, z) / C₂(Δt)

    在大 Δt 极限下:

        h(z, P_z) = lim_{Δt→∞} h(z, P_z, Δt)

    实际中通过 plateau 拟合提取:

        h(z, P_z) = average_{Δt ∈ [tsep_min, tsep_max]} h(z, P_z, Δt)

    Parameters
    ----------
    C3 : np.ndarray, shape (Nt, Nz)
        三点关联函数 (或 OPE 期望值)
    C2 : np.ndarray, shape (Nt,)
        两点关联函数 (质子传播子)
    tsep_min, tsep_max : int
        plateau 区域的时间分离范围

    Returns
    -------
    h_matrix : np.ndarray, shape (Nz,)
        提取的矩阵元 h(z, P_z)
    h_errors : np.ndarray, shape (Nz,)
        Jackknife 误差估计
    """
    Nz = C3.shape[1]
    h_matrix = np.zeros(Nz, dtype=complex)
    h_errors = np.zeros(Nz, dtype=float)

    for z in range(Nz):
        # 计算每个 time separation 的有效矩阵元
        h_eff = np.zeros(tsep_max - tsep_min, dtype=complex)

        for i, dt in enumerate(range(tsep_min, tsep_max)):
            # 注意: 在 disconnect 近似下 C₃/C₂ → ⟨O(z)⟩
            # ⟨O(z)⟩ 本身不依赖于 Δt (已在 OPE 计算中平均)
            # 这里展示标准的关联函数比形式
            h_eff[i] = C3[dt, z] / C2[dt] if abs(C2[dt]) > 1e-15 else 0.0

        h_matrix[z] = np.mean(h_eff)
        h_errors[z] = np.std(h_eff) / np.sqrt(len(h_eff))

    return h_matrix, h_errors


def extract_matrix_element_from_ope_average(
    ope_data: np.ndarray,
    n_conf: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从多个组态的 OPE 期望值中提取矩阵元。

    在 disconnect 近似下:
        h(z) = ⟨O(z)⟩ = (1/N_conf) Σ_{conf} ⟨O(z)⟩_{conf}

    Parameters
    ----------
    ope_data : np.ndarray, shape (N_conf, Nt, Nz)
        每个组态在每个时间片的 OPE 算符值
    n_conf : int
        组态数目

    Returns
    -------
    h_z : np.ndarray, shape (Nz,)
        对时间和组态平均后的矩阵元 h(z)
    h_z_err : np.ndarray, shape (Nz,)
        Jackknife 误差
    """
    Nz = ope_data.shape[2]
    Nt = ope_data.shape[1]

    # Step 1: 先对时间平均
    ope_tavg = np.mean(ope_data, axis=1)  # (N_conf, Nz)

    # Step 2: 对组态平均
    h_z = np.mean(ope_tavg, axis=0)  # (Nz,)

    # Step 3: Jackknife 误差估计
    h_z_jk = np.zeros((n_conf, Nz), dtype=complex)
    for i in range(n_conf):
        # 删除第 i 个组态后的平均
        mask = np.ones(n_conf, dtype=bool)
        mask[i] = False
        h_z_jk[i] = np.mean(ope_tavg[mask], axis=0)

    # Jackknife 误差: σ_JK = sqrt((N-1)/N * Σ (h_i - h̄)²)
    h_z_jk_mean = np.mean(h_z_jk, axis=0)
    h_z_err = np.sqrt(
        (n_conf - 1) / n_conf * np.sum(
            np.abs(h_z_jk - h_z_jk_mean) ** 2, axis=0
        )
    )

    return h_z, h_z_err


def fourier_transform_to_quasi_pdf(
    h_z: np.ndarray,
    z_values: np.ndarray,
    Pz: float,
    x_values: np.ndarray,
) -> np.ndarray:
    """
    通过 Fourier 变换将坐标空间矩阵元转换为 quasi-PDF:

        g̃(x, P_z) = ∫ dz / (2π x P_z) · e^{i x P_z z} · h(z, P_z)

    对于反对称部分（非极化胶子），使用 sin 变换:

        g̃(x, P_z) = (2P_z / x) · ∫_0^{z_max} dz · h(z, P_z) · sin(x P_z z)

    注意: 这里忽略了重整化因子。实际计算中需要先对 h(z) 进行重整化。

    Parameters
    ----------
    h_z : np.ndarray, shape (Nz,)
        坐标空间矩阵元 h(z, P_z)
    z_values : np.ndarray, shape (Nz,)
        z 值数组 (格距单位)
    Pz : float
        boost 动量 (格点单位)
    x_values : np.ndarray, shape (Nx,)
        Bjorken-x 值数组

    Returns
    -------
    quasi_pdf : np.ndarray, shape (Nx,)
        quasi-PDF g̃(x, P_z) 在各 x 点的值
    """
    Nx = len(x_values)
    quasi_pdf = np.zeros(Nx)

    for i, x in enumerate(x_values):
        if abs(x) < 1e-15:
            quasi_pdf[i] = 0.0
            continue

        # 数值积分: ∫ dz h(z) sin(x P_z z)
        integrand = h_z.real * np.sin(x * Pz * z_values)
        dz = z_values[1] - z_values[0] if len(z_values) > 1 else 1.0

        # 梯形法则积分
        integral = np.trapz(integrand, z_values)

        # 前因子: 2P_z / x
        quasi_pdf[i] = (2.0 * Pz / x) * integral

    return quasi_pdf


# ============================================================================
# 第八部分: 微扰匹配 (Matching)
# ============================================================================

def matching_kernel_gg_lo(
    xi: float,
    mu_over_Pz: float = 1.0,
) -> float:
    """
    胶子 quasi-PDF 到光锥 PDF 的 LO 匹配核。

    在 LO (leading order):
        C_{gg}(ξ, μ/P_z) = δ(1 - ξ) + O(α_s)

    此处返回 identity 部分的值。
    完整的 NLO 匹配核需要更多微扰计算，此处为占位符。

    Parameters
    ----------
    xi : float
        动量分数比值 ξ = x/y
    mu_over_Pz : float
        重整化标度与 boost 动量的比值 μ/P_z

    Returns
    -------
    kernel_value : float
        匹配核 C_{gg}(ξ) 在 ξ 处的值
    """
    # LO: 恒等变换
    if abs(xi - 1.0) < 1e-10:
        return 1.0
    else:
        return 0.0


def apply_matching(
    quasi_pdf: np.ndarray,
    x_values: np.ndarray,
    alpha_s: float = 0.2,
    mu_over_Pz: float = 1.0,
) -> np.ndarray:
    """
    将 quasi-PDF 通过匹配核转换为光锥 PDF:

        g(x, μ) = g̃(x, P_z) - (α_s/2π) ∫_x^1 (dy/y) C_{gg}(x/y) g̃(y, P_z)

    在 LO 近似下: g(x, μ) ≈ g̃(x, P_z)

    Parameters
    ----------
    quasi_pdf : np.ndarray, shape (Nx,)
        准 PDF g̃(x, P_z)
    x_values : np.ndarray, shape (Nx,)
        x 值 (升序排列)
    alpha_s : float
        强耦合常数 α_s(μ)
    mu_over_Pz : float
        μ/P_z

    Returns
    -------
    lightcone_pdf : np.ndarray, shape (Nx,)
        光锥 PDF g(x, μ)
    """
    Nx = len(x_values)
    lightcone_pdf = quasi_pdf.copy()

    # NLO 修正项 (占位符: α_s → 0 时恢复 LO)
    if alpha_s > 1e-10:
        for i, x in enumerate(x_values):
            correction = 0.0
            for j, y in enumerate(x_values):
                if y >= x and y > 1e-10:
                    xi = x / y
                    # 匹配核 * g̃(y) 的积分
                    correction += (
                        matching_kernel_gg_lo(xi, mu_over_Pz)
                        * quasi_pdf[j]
                        * (x_values[1] - x_values[0]) / y
                    )
            lightcone_pdf[i] -= (alpha_s / (2.0 * np.pi)) * correction

    return lightcone_pdf


# ============================================================================
# 第九部分: 统计误差分析 (Jackknife)
# ============================================================================

def jackknife_analysis(
    data: np.ndarray,
    axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Jackknife 重采样误差分析。

    Jackknife 方法:
        对 N 个样本，创建 N 个子样本 (每个删除一个原始样本)，
        然后计算子样本统计量的方差。

        σ_JK² = (N-1)/N · Σ_i (θ_i - θ̄)²

    其中 θ_i 是删除第 i 个样本后的估计量，
    θ̄ = (1/N) Σ_i θ_i 是 Jackknife 估计量。

    Parameters
    ----------
    data : np.ndarray, shape (N_samples, ...)
        沿 axis 方向排列的独立样本
    axis : int
        样本所在的轴 (默认 0)

    Returns
    -------
    mean_jk : np.ndarray, shape (...,)
        Jackknife 平均值
    error_jk : np.ndarray, shape (...,)
        Jackknife 标准误差
    """
    N = data.shape[axis]
    if N < 2:
        return np.mean(data, axis=axis), np.zeros_like(np.mean(data, axis=axis))

    # 完整样本的平均值
    mean_full = np.mean(data, axis=axis)

    # Jackknife 子样本平均值
    means_jk = []
    for i in range(N):
        # 创建掩码: 删除第 i 个样本
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        # 沿指定轴取子样本
        data_sub = np.compress(mask, data, axis=axis)
        means_jk.append(np.mean(data_sub, axis=axis))

    means_jk = np.array(means_jk)
    mean_jk = np.mean(means_jk, axis=0)

    # Jackknife 方差
    variance_jk = (N - 1) / N * np.sum(
        np.abs(means_jk - mean_jk) ** 2, axis=0
    )
    error_jk = np.sqrt(variance_jk)

    return mean_jk, error_jk


# ============================================================================
# 第十部分: 完整工作流编排
# ============================================================================

class GluonPDFWorkflow:
    """
    胶子 PDF 计算的完整工作流协调器。

    将各个模块整合为一个完整的计算管道:

        1. 读取规范组态 → 2. 构造 OPE 算符 → 3. 计算质子 2pt
        → 4. 提取矩阵元 → 5. Fourier 变换 → 6. 匹配到光锥 PDF

    Examples
    --------
    >>> lattice = LatticeConfig(Nt=64, Nx=32)
    >>> params = CalcParams(Pz=6, delta_z=15)
    >>> wf = GluonPDFWorkflow(lattice, params)
    >>> # wf.run_ope_calculation("gauge_conf.dat")
    >>> # wf.run_proton_2pt()
    >>> # h_z, h_err = wf.extract_matrix_element()
    >>> # g_pdf = wf.reconstruct_lightcone_pdf(h_z)
    """

    def __init__(self, lattice: LatticeConfig, params: CalcParams):
        """
        初始化工作流。

        Parameters
        ----------
        lattice : LatticeConfig
            格点几何配置
        params : CalcParams
            计算参数
        """
        self.lattice = lattice
        self.params = params

        # Dirac 矩阵
        self.gamma = build_gamma_matrices_degrand_rossi()

        # Levi-Civita 张量 (用于对偶变换)
        self.epsilon = build_levi_civita_tensor()

        # 宇称投影算符
        self.P_plus = 0.5 * (self.gamma[0] + self.gamma[4])   # P₊ = (γ₀ + γ₄)/2
        self.P_minus = 0.5 * (self.gamma[0] - self.gamma[4])  # P₋ = (γ₀ - γ₄)/2

        # 中间结果存储
        self.F_all: Optional[np.ndarray] = None
        self.F_tilde_all: Optional[np.ndarray] = None
        self.ope_results: Dict = {}
        self.proton_2pt: Optional[np.ndarray] = None
        self.matrix_elements: Optional[np.ndarray] = None
        self.quasi_pdf: Optional[np.ndarray] = None
        self.lightcone_pdf: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Step 1: OPE 计算
    # ------------------------------------------------------------------

    def compute_field_strengths(self, gauge: np.ndarray) -> None:
        """
        从规范链接计算场强张量 F_{μν} 和对偶场强 F̃_{μν}。

        Parameters
        ----------
        gauge : np.ndarray
            SU(3) 规范链接 U_μ(x)
        """
        print("[Workflow] Computing field strength tensors...")
        t0 = time.time()

        Nt, Nx = self.lattice.Nt, self.lattice.Nx
        self.F_all = compute_field_strength_all(gauge, Nt, Nx)
        self.F_tilde_all = compute_dual_field_strength(self.F_all, self.epsilon)

        t1 = time.time()
        print(f"[Workflow] Field strengths computed in {t1 - t0:.1f}s.")

    def compute_ope_operators(self, gauge: np.ndarray) -> Dict[str, np.ndarray]:
        """
        为 unpolarized 胶子 quasi-PDF 计算所有需要的 OPE 算符分量。

        对于 z_dir = z (Wilson 线沿 z 方向)，非极化的算符分量包括:
            (μ,ν,μ₂,ν₂) = (t,x, t,x), (t,y, t,y), (x,y, x,y)

        对应不同的 F_{zμ} F̃^{μz} 组合。

        Parameters
        ----------
        gauge : np.ndarray
            SU(3) 规范链接

        Returns
        -------
        ope_results : dict
            key: "mu{μ}_nu{ν}_mu2{μ2}_nu2{ν2}"
            value: np.ndarray, shape (Nt, delta_z)
        """
        if self.F_all is None or self.F_tilde_all is None:
            self.compute_field_strengths(gauge)

        Nt = self.lattice.Nt
        z_dir = self.params.z_dir
        delta_z = self.params.delta_z

        # 定义需要的 Lorentz 指标组合
        # 对于 unpolarized gluon, 需要:
        #   F_{zμ}(z) W(z,0) F̃^{μz}(0)  对 μ ≠ z 求和
        lorentz_pairs = [
            # (mu, nu, mu2, nu2) — F_{zμ} 和 F̃^{μz}
            # 注意: F_{zμ} 对应 (z_dir, transverse_dir)
            #       F̃^{μz} 对应 (transverse_dir, z_dir)
            # 但对于 clover 型的对偶，使用:
            (3, (z_dir + 1) % 3, 3, (z_dir + 1) % 3),   # F_{zt} F̃^{tz}
            (3, (z_dir + 2) % 3, 3, (z_dir + 2) % 3),   # F_{zt} F̃^{tz} (交叉)
            ((z_dir + 1) % 3, (z_dir + 2) % 3,
             (z_dir + 1) % 3, (z_dir + 2) % 3),          # F_{zx} F̃^{xz}
        ]

        ope_results = {}
        for mu, nu, mu2, nu2 in lorentz_pairs:
            key = f"mu{mu}_nu{nu}_mu2{mu2}_nu2{nu2}"
            print(f"[Workflow] Computing OPE for {key}...")
            t0 = time.time()

            ops = compute_ope_all_z(
                gauge, self.F_all, self.F_tilde_all,
                delta_z, z_dir, mu, nu, mu2, nu2, Nt
            )
            ope_results[key] = ops

            t1 = time.time()
            print(f"[Workflow] OPE {key} done in {t1 - t0:.1f}s.")

        self.ope_results = ope_results
        return ope_results

    # ------------------------------------------------------------------
    # Step 2: 质子两点函数
    # ------------------------------------------------------------------

    def compute_proton_2pt_distillation(
        self,
        eig_dir: str,
        peram_dir: str,
    ) -> np.ndarray:
        """
        使用 Distillation 方法 + 动量涂抹计算质子两点关联函数。

        完整流程:
            1. 对每个时间片 t 读取本征矢量，计算 VVV (动量涂抹版本)
            2. 对每个 t_src 读取 perambulator
            3. 执行 Wick 收缩
            4. 宇称投影

        Parameters
        ----------
        eig_dir : str
            本征矢量目录路径
        peram_dir : str
            Perambulator 目录路径

        Returns
        -------
        C2_pp : np.ndarray, shape (Nt, Nt)
            正宇称 (P₊) 质子两点函数 C₂(t_sink, t_src)
        """
        Nt = self.lattice.Nt
        Nx = self.lattice.Nx
        Nev1 = self.params.Nev1
        conf_id = self.params.conf_id

        Mom = np.array([self.params.Pz, self.params.Py, self.params.Px])

        print(f"[Workflow] Computing proton 2pt, Mom = {Mom}")

        # --- 计算 VVV (所有 t) ---
        print("[Workflow] Computing VVV baryon blocks...")
        VVV_all = np.zeros((Nt, Nev1, Nev1, Nev1), dtype=complex)
        phase_factor = compute_phase_factor(Mom, Nx)

        for t in range(Nt):
            eigvecs = read_eigenvectors(eig_dir, t, Nev1, conf_id, Nx)
            VVV_all[t] = compute_vvv_baryon_block(
                eigvecs, phase_factor, Nev1, Nx
            )

        # --- 质子两点函数收缩 ---
        print("[Workflow] Performing Wick contractions...")
        # 插值算符: Cγ₅γ₄ (index 15 = γ₄γ₅)
        interp = self.gamma[15]  # Cγ₅γ₄

        C2_matrix = np.zeros((Nt, Nt, 4, 4), dtype=complex)

        for t_src in range(Nt):
            # 读取 perambulator
            peram = read_perambulator(peram_dir, conf_id, Nt, Nev1, t_src)

            # Dirac 矩阵插入: Γ · τ · Γ
            CG5_tau_CG5 = contract(
                "gh, thkbe, jk -> tgjbe",
                interp, peram, interp,
            )

            VVV_source = VVV_all[t_src].conj()

            for t_snk in range(Nt):
                delta_t = (t_snk - t_src + Nt) % Nt
                # 选择合适的时间分离范围
                if 2 <= delta_t <= 30:
                    C2_matrix[t_snk, t_src] = contract_proton_2pt_single_tsrc(
                        VVV_all[t_snk],
                        peram[t_snk],
                        CG5_tau_CG5[t_snk],
                        VVV_source,
                    )

        # --- 宇称投影 ---
        C2_pp = contract("li, yxil -> yx", self.P_plus, C2_matrix)
        C2_pm = contract("li, yxil -> yx", self.P_minus, C2_matrix)

        # --- 边界条件符号修正 ---
        # 由于质子插值算符交换反对称性，对 t_sink < t_src 取负号
        for t_src in range(Nt):
            for t_snk in range(Nt):
                if t_snk < t_src:
                    C2_pp[t_snk, t_src] *= -1.0
                if t_snk > t_src:
                    C2_pm[t_snk, t_src] *= -1.0

        self.proton_2pt = C2_pp
        return C2_pp

    # ------------------------------------------------------------------
    # Step 3: 矩阵元提取
    # ------------------------------------------------------------------

    def extract_matrix_elements(
        self,
        ope_data: np.ndarray,
        n_conf: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 OPE 数据中提取胶子 quasi-PDF 矩阵元 h(z, P_z)。

        Parameters
        ----------
        ope_data : np.ndarray
            OPE 算符数据，形状取决于输入:
            - 单组态: (Nt, Nz)
            - 多组态: (N_conf, Nt, Nz)
        n_conf : int
            组态数目

        Returns
        -------
        h_z : np.ndarray, shape (Nz,)
            矩阵元 h(z)
        h_z_err : np.ndarray, shape (Nz,)
            Jackknife 误差
        """
        Nz = self.params.delta_z

        if ope_data.ndim == 2:
            # 单组态: 对时间平均
            h_z = np.mean(ope_data, axis=0)
            h_z_err = np.std(ope_data, axis=0) / np.sqrt(ope_data.shape[0])
        elif ope_data.ndim == 3:
            # 多组态: 先对时间平均，再对组态做 Jackknife
            h_z, h_z_err = extract_matrix_element_from_ope_average(
                ope_data, n_conf
            )
        else:
            raise ValueError(f"Unexpected ope_data shape: {ope_data.shape}")

        self.matrix_elements = (h_z, h_z_err)
        return h_z, h_z_err

    # ------------------------------------------------------------------
    # Step 4: quasi-PDF 重构
    # ------------------------------------------------------------------

    def reconstruct_quasi_pdf(
        self,
        h_z: np.ndarray,
        x_values: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        通过 Fourier 变换将坐标空间矩阵元转换为 quasi-PDF。

        Parameters
        ----------
        h_z : np.ndarray, shape (Nz,)
            坐标空间矩阵元 h(z)
        x_values : np.ndarray, optional
            Bjorken-x 值，默认在 [0.05, 0.95] 范围内等距采样

        Returns
        -------
        quasi_pdf : np.ndarray, shape (Nx,)
            quasi-PDF g̃(x, P_z)
        """
        Nz = len(h_z)
        Nx_lattice = self.lattice.Nx

        # z 值: z = 0, 1, 2, ..., Nz-1 (格距单位)
        z_values = np.arange(Nz)

        # P_z 动量 (格点单位: 2π * n_z / L)
        Pz = 2.0 * np.pi * self.params.Pz / Nx_lattice

        # x 值 (默认范围)
        if x_values is None:
            x_values = np.linspace(0.05, 0.95, 50)

        quasi_pdf = fourier_transform_to_quasi_pdf(
            h_z, z_values, Pz, x_values
        )
        self.quasi_pdf = quasi_pdf
        self.x_values = x_values

        return quasi_pdf

    # ------------------------------------------------------------------
    # Step 5: 匹配到光锥 PDF
    # ------------------------------------------------------------------

    def reconstruct_lightcone_pdf(
        self,
        quasi_pdf: np.ndarray,
        alpha_s: float = 0.2,
        mu_over_Pz: float = 1.0,
    ) -> np.ndarray:
        """
        通过微扰匹配将 quasi-PDF 转换为光锥 PDF。

        Parameters
        ----------
        quasi_pdf : np.ndarray
            quasi-PDF g̃(x, P_z)
        alpha_s : float
            强耦合常数
        mu_over_Pz : float
            重整化标度比值

        Returns
        -------
        lightcone_pdf : np.ndarray
            光锥 PDF g(x, μ)
        """
        if not hasattr(self, 'x_values'):
            self.x_values = np.linspace(0.05, 0.95, len(quasi_pdf))

        lightcone_pdf = apply_matching(
            quasi_pdf, self.x_values, alpha_s, mu_over_Pz
        )
        self.lightcone_pdf = lightcone_pdf

        return lightcone_pdf

    # ------------------------------------------------------------------
    # 完整工作流 (一站式)
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        gauge_file: str,
        eig_dir: str,
        peram_dir: str,
        n_conf: int = 1,
    ) -> Dict[str, np.ndarray]:
        """
        执行完整的胶子 PDF 计算管道。

        Parameters
        ----------
        gauge_file : str
            规范组态文件路径
        eig_dir : str
            本征矢量目录
        peram_dir : str
            Perambulator 目录
        n_conf : int
            组态数目

        Returns
        -------
        results : dict
            包含所有中间和最终结果:
            - "ope": OPE 算符
            - "C2": 质子两点函数
            - "h_z": 矩阵元
            - "h_z_err": 矩阵元误差
            - "quasi_pdf": quasi-PDF
            - "lightcone_pdf": 光锥 PDF
            - "x_values": x 值数组
        """
        results = {}

        # Step 1: 读取规范组态
        print("[Pipeline] Step 1/5: Reading gauge configuration...")
        t0 = time.time()
        gauge = read_gauge_config(
            gauge_file, self.lattice.Nt, self.lattice.Nx
        )
        t1 = time.time()
        print(f"[Pipeline] Gauge config read in {t1 - t0:.1f}s.")

        # Step 2: OPE 算符计算
        print("[Pipeline] Step 2/5: Computing OPE operators...")
        self.compute_field_strengths(gauge)
        ope_results = self.compute_ope_operators(gauge)
        results["ope"] = ope_results

        # Step 3: 质子两点函数
        print("[Pipeline] Step 3/5: Computing proton 2pt function...")
        C2 = self.compute_proton_2pt_distillation(eig_dir, peram_dir)
        results["C2"] = C2

        # Step 4: 矩阵元提取
        print("[Pipeline] Step 4/5: Extracting matrix elements...")
        # 对所有 OPE 分量求和/平均 (unpolarized)
        ope_sum = np.zeros((self.lattice.Nt, self.params.delta_z),
                            dtype=complex)
        for key, ops in ope_results.items():
            ope_sum += ops
        ope_sum /= len(ope_results)

        h_z, h_z_err = self.extract_matrix_elements(ope_sum, n_conf)
        results["h_z"] = h_z
        results["h_z_err"] = h_z_err

        # Step 5: quasi-PDF + 匹配
        print("[Pipeline] Step 5/5: Reconstructing PDF...")
        quasi_pdf = self.reconstruct_quasi_pdf(h_z)
        results["quasi_pdf"] = quasi_pdf

        lightcone_pdf = self.reconstruct_lightcone_pdf(quasi_pdf)
        results["lightcone_pdf"] = lightcone_pdf
        results["x_values"] = self.x_values

        t2 = time.time()
        print(f"[Pipeline] Full pipeline completed in {t2 - t0:.1f}s.")

        return results


# ============================================================================
# 第十一部分: 结果保存与绘图
# ============================================================================

def save_results(results: Dict[str, np.ndarray],
                  output_dir: str,
                  prefix: str = "gluon_pdf") -> None:
    """
    保存计算结果到文件。

    Parameters
    ----------
    results : dict
        GluonPDFWorkflow.run_full_pipeline() 的输出
    output_dir : str
        输出目录
    prefix : str
        文件名前缀
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存为 .npy 格式
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            fname = f"{output_dir}/{prefix}_{key}.npy"
            np.save(fname, value)
            print(f"[Save] {fname} saved.")

    # 保存为可读文本格式
    if "lightcone_pdf" in results and "x_values" in results:
        txt_file = f"{output_dir}/{prefix}_lightcone_pdf.dat"
        data = np.column_stack([
            results["x_values"],
            results["lightcone_pdf"].real,
        ])
        header = "# x    g(x, mu)"
        np.savetxt(txt_file, data, header=header)
        print(f"[Save] {txt_file} saved.")


def plot_results(results: Dict[str, np.ndarray],
                  output_dir: str = "./plots") -> None:
    """
    绘制 PDF 结果（需要 matplotlib）。

    Parameters
    ----------
    results : dict
        计算结果
    output_dir : str
        图片输出目录
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not available, skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 图 1: 矩阵元 h(z) vs z
    if "h_z" in results:
        fig, ax = plt.subplots(figsize=(8, 5))
        z = np.arange(len(results["h_z"]))
        ax.errorbar(
            z, results["h_z"].real,
            yerr=results.get("h_z_err", None),
            fmt='o-', capsize=3, label="Re[h(z)]"
        )
        ax.set_xlabel("z / a")
        ax.set_ylabel("h(z, P_z)")
        ax.set_title("Coordinate-space matrix element")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/matrix_element.png", dpi=150)
        plt.close(fig)
        print(f"[Plot] {output_dir}/matrix_element.png saved.")

    # 图 2: 光锥 PDF g(x)
    if "lightcone_pdf" in results and "x_values" in results:
        fig, ax = plt.subplots(figsize=(8, 6))
        x = results["x_values"]
        g = results["lightcone_pdf"].real
        ax.plot(x, x * g, 'b-', linewidth=2, label=r"$x g(x, \mu)$ (This work)")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$x g(x, \mu)$")
        ax.set_title("Gluon PDF of the Proton")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/gluon_pdf.png", dpi=150)
        plt.close(fig)
        print(f"[Plot] {output_dir}/gluon_pdf.png saved.")

    # 图 3: quasi-PDF vs light-cone PDF 比较
    if "quasi_pdf" in results and "lightcone_pdf" in results:
        fig, ax = plt.subplots(figsize=(8, 6))
        x = results["x_values"]
        ax.plot(x, results["quasi_pdf"].real, 'r--', linewidth=1.5,
                alpha=0.7, label="quasi-PDF")
        ax.plot(x, results["lightcone_pdf"].real, 'b-', linewidth=2,
                label="Light-cone PDF (matched)")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$g(x)$")
        ax.set_title("quasi-PDF vs Light-cone PDF")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/quasi_vs_lightcone.png", dpi=150)
        plt.close(fig)
        print(f"[Plot] {output_dir}/quasi_vs_lightcone.png saved.")


# ============================================================================
# 第十二部分: 命令行接口与示例
# ============================================================================

def main():
    """
    命令行入口点。

    用法:
        python gluon_pdf_full_workflow.py --conf 20000 --Pz 6 --delta_z 15
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="格点 QCD 胶子 PDF 完整计算工作流"
    )
    parser.add_argument("--Nt", type=int, default=64,
                        help="时间方向格点数")
    parser.add_argument("--Nx", type=int, default=32,
                        help="空间方向格点数")
    parser.add_argument("--Nev", type=int, default=100,
                        help="本征矢量数")
    parser.add_argument("--Nev1", type=int, default=100,
                        help="收缩用本征矢量数")
    parser.add_argument("--Pz", type=int, default=6,
                        help="z-方向动量 (2π/L 单位)")
    parser.add_argument("--delta_z", type=int, default=15,
                        help="Wilson 线最大长度")
    parser.add_argument("--z_dir", type=int, default=2,
                        help="Wilson 线方向 (0=x, 1=y, 2=z)")
    parser.add_argument("--conf", type=str, default="20000",
                        help="组态编号")
    parser.add_argument("--gauge_file", type=str,
                        default="gauge_config.dat",
                        help="规范组态文件路径")
    parser.add_argument("--eig_dir", type=str,
                        default="./eigvecs/",
                        help="本征矢量目录")
    parser.add_argument("--peram_dir", type=str,
                        default="./perams/",
                        help="Perambulator 目录")
    parser.add_argument("--output_dir", type=str,
                        default="./results/",
                        help="输出目录")
    parser.add_argument("--alpha_s", type=float, default=0.2,
                        help="强耦合常数 α_s")

    args = parser.parse_args()

    # 初始化
    lattice = LatticeConfig(Nt=args.Nt, Nx=args.Nx)
    params = CalcParams(
        Nev=args.Nev, Nev1=args.Nev1,
        Pz=args.Pz, delta_z=args.delta_z,
        z_dir=args.z_dir, conf_id=args.conf,
    )

    wf = GluonPDFWorkflow(lattice, params)

    # 执行完整管道
    results = wf.run_full_pipeline(
        gauge_file=args.gauge_file,
        eig_dir=args.eig_dir,
        peram_dir=args.peram_dir,
        n_conf=1,
    )

    # 保存结果
    save_results(results, args.output_dir)

    # 绘图
    plot_results(results, f"{args.output_dir}/plots")

    print("\n" + "=" * 60)
    print("胶子 PDF 计算完成!")
    print(f"结果保存在: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
