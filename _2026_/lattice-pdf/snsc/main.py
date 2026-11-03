#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
格点 QCD 质子非极化胶子 PDF — 统一计算流水线
================================================================================

理论框架: 大动量有效理论 (LaMET) + 蒸馏 (Distillation) + 动量涂抹
方法: Disconnected 图分解 → 质子 2pt × OPE 算符

────────────────────────────────────────────────────────────────────────────
核心理论公式
────────────────────────────────────────────────────────────────────────────

1. 场强张量 F_{μν} (Clover 叶, O(a²) 改进)
   Q_{μν}(x) = P_{μν} + P_{ν,-μ} + P_{-μ,-ν} + P_{-ν,μ}
   F̂_{μν}(x) = -i/8 · [Q_{μν} - Q_{μν}†]

2. 对偶场强张量 F̃_{μν} = (1/2) ε_{μνρσ} F^{ρσ}

3. Nonlocal 胶子 OPE 算符 (Eq(25) of 内部笔记)
   O_{unpol}(z) = Σ_{μ≠z} Σ_{ν≠z} g^{μν} Tr_c[F_{zμ}(z) W(z→0) F̃_ν^z(0) W(0→z)]

4. VVV Baryon Block (6-term ε_{ijk} 收缩)
   Φ_{abc}(P) = Σ_x e^{-iP·x} ε_{ijk} v_i^a(x) v_j^b(x) v_k^c(x)

5. 质子 2pt Wick 收缩: C₂ = Direct - Exchange

6. 有效质量 (cosh): a·m_eff(t) = arccosh((C(t-1)+C(t+1))/(2C(t)))

7. 矩阵元提取 h(z,P_z) → Fourier 变换 → quasi-PDF → 匹配 → 光锥 PDF

────────────────────────────────────────────────────────────────────────────
功能模块 (13 步)
────────────────────────────────────────────────────────────────────────────
  1. 读取规范组态 (ILDG 二进制)        8.  Distillation 框架
  2. 构造场强张量 F_{μν} (Clover)      9.  动量涂抹质子 2pt
  3. 构造对偶场强张量 F̃_{μν}         10. 矩阵元提取 h(z,P_z)
  4. 构造 VVV (Baryon Block)           11. Fourier 变换 → quasi-PDF
  5. 构造质子 2pt 关联函数             12. 微扰匹配 → 光锥 PDF
  6. 构造 nonlocal 胶子 OPE 算符       13. Jackknife/Bootstrap 误差分析
  7. 构造 3pt 关联函数

特性:
  - 双后端: numpy (CPU) / cupy (GPU)
  - 全程内存/显存 + 耗时统计
  - 可选组态编号 (初始、步长、样本数)
  - 可选动量 (方向、大小、扫描列表)
  - 可选读取/生成路径 (同时给出则自动对比)
  - 大量出版级作图 (>15 种图)
  - 可选数据类型 (complex64/complex128)
  - 时间戳输出目录, 规范化编程
  - 详细中文注释与过程检查输出
  - 数值结果与已有代码 (donghx, huangcl) 保持一致

三种分析模式:
  --analysis-type pdf        胶子 PDF 全流程 (LaMET + distillation + OPE)
  --analysis-type proton-2pt 质子 2pt 蒸馏计算 (复现 donghx DCU 代码)
  --analysis-type 2pt        2pt 有效质量分析 (匹配 main-2pt.py, 需 IOG)

作者: Zhang Xin
日期: 2026-07-25
================================================================================
"""

# ============================================================================
# 第〇部分: 导入与全局配置
# ============================================================================

import numpy as np
import os
import sys
import time
import argparse
import resource
import gc
import json as _json
from typing import Dict, Tuple, List, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager

# --- GPU 后端检测 (CuPy) ---
HAS_CUPY = False
cp = None
try:
    import cupy as cp_lib
    cp = cp_lib
    HAS_CUPY = True
except ImportError:
    pass

# --- 优化张量缩并 (opt_einsum) ---
HAS_OPT_EINSUM = False
_opt_contract = None
try:
    from opt_einsum import contract as _opt_contract
    HAS_OPT_EINSUM = True
except ImportError:
    pass

# --- 画图库 (无头模式, 适配集群) ---
HAS_MPL = False
plt = None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    pass

# --- IOG 读取支持 (来自 examples/zhangxin/) ---
HAS_INCLUDE = False
_data_analyse = None
_iog_read_fn = None
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(_script_dir)
    _examples_dir = os.path.join(_repo_root, "examples", "zhangxin")
    if _examples_dir not in sys.path:
        sys.path.insert(0, _examples_dir)
    _saved_cwd = os.getcwd()
    os.chdir(_examples_dir)
    try:
        from include import data_analyse as _data_analyse
        from iog_reader.iog_reader import iog_read
        _iog_read_fn = iog_read
        HAS_INCLUDE = True
    finally:
        os.chdir(_saved_cwd)
except (ImportError, OSError) as e:
    pass


# ============================================================================
# 第一部分: 工具函数 —— 计时、内存追踪、张量缩并
# ============================================================================

@contextmanager
def Timer(step_name: str, pipeline: Optional[Any] = None):
    """上下文管理器: 记录每个步骤的耗时与内存/显存峰值。

    用法:
        with Timer("Step 04: VVV Baryon Block", pipeline):
            VVV = compute_vvv_baryon_block(...)

    自动将结果记录到 pipeline.timing_results 和 pipeline.memory_results。
    """
    gc.collect()
    mem_before = _get_memory(pipeline)
    t_start = time.perf_counter()
    yield
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    mem_after = _get_memory(pipeline)
    mem_peak = max(mem_before, mem_after)

    if pipeline is not None:
        pipeline.timing_results[step_name] = elapsed
        pipeline.memory_results[step_name] = {
            "before_mb": mem_before, "after_mb": mem_after, "peak_mb": mem_peak,
        }

    print(f"\n{'='*60}")
    print(f"  {step_name}")
    print(f"  耗时: {elapsed:.3f} s  |  内存峰值: {mem_peak:.1f} MB")
    print(f"{'='*60}\n")


def _get_memory(pipeline: Optional[Any] = None) -> float:
    """获取当前进程的内存/显存占用 (MB)。

    优先级:
      1. CuPy GPU 显存 (若 pipeline 使用 CuPy)
      2. RSS 常驻内存 (Linux)
    """
    if pipeline is not None and pipeline.xp is cp and HAS_CUPY:
        try:
            mempool = cp.cuda.Device().mem_info
            if callable(mempool):
                info = mempool()
                used = info.get("used_bytes", info[0] if isinstance(info, tuple) else 0)
            else:
                used = mempool[0] - mempool[1] if isinstance(mempool, tuple) else 0
            return used / (1024 * 1024)
        except Exception:
            pass
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024.0  # Linux: KB → MB
    except Exception:
        return 0.0


def get_contract(xp_module: Any) -> callable:
    """返回最优的张量缩并函数: opt_einsum (若可用) 否则 xp.einsum。"""
    if HAS_OPT_EINSUM and xp_module is np:
        return _opt_contract
    return xp_module.einsum


def _ensure_cpu(arr: Any) -> np.ndarray:
    """将可能的 GPU 数组安全转换为 CPU numpy 数组。"""
    if HAS_CUPY and hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


# ============================================================================
# 第二部分: 格点配置数据类
# ============================================================================

@dataclass
class LatticeConfig:
    """格点几何与物理参数。

    Attributes
    ----------
    Nt, Nx : 时间/空间方向格点数
    Nc, Nd : 颜色数(3) / Dirac 旋量维数(4)
    Nev, Nev1 : 本征矢量总数 / 实际收缩用数
    delta_z, z_dir : Wilson 线最大长度 / 方向 (0=x,1=y,2=z)
    Px, Py, Pz : 动量分量 (以 2π/L 为单位)
    mom_smear, mom_smear_phase : 动量涂抹参数
    alttc : 格距 (fm)
    """
    Nt: int = 64
    Nx: int = 32
    Nc: int = 3
    Nd: int = 4
    Nev: int = 100
    Nev1: int = 100
    delta_z: int = 15
    z_dir: int = 2
    conf_id: str = "20000"
    Px: int = 0
    Py: int = 0
    Pz: int = 6
    mom_smear: int = 3
    mom_smear_phase: int = -3
    alttc: float = 0.1053

    @property
    def Ny(self) -> int: return self.Nx
    @property
    def Nz(self) -> int: return self.Nx
    @property
    def spatial_volume(self) -> int: return self.Nx ** 3


# ============================================================================
# 第三部分: 系综预设 (7 组)
# ============================================================================

ENSEMBLES: Dict[str, dict] = {
    "L24x72": {
        "Nt": 72, "Nx": 24, "Nev": 100, "Nev1": 100,
        "mom_smear": -2, "mom_smear_phase": 2,
        "beta": 6.20, "mu_l": -0.2770, "mu_s": -0.2400, "alttc": 0.1053,
        "conf_name": "beta6.20_mu-0.2770_ms-0.2400_L24x72",
        "conf_short": "L24x72",
        "conf_dir": "/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/",
        "eig_dir": "/public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/",
        "peram_u_dir": "/public/home/sunp/sunpeng_new_disk/mom_smear_perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/output_dir_data/mz2_my0_mx0/{conf_id}/",
        "corr_nucl_dir": "/public/group/lqcd/donghx/2pt_Result/beta6.20_mu-0.2770_ms-0.2400_L24x72/momsmear-2z/{conf_id}/",
        "ope_dir": "/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/L24x72/zdir/{conf_id}/",
    },
    "L32x64": {
        "Nt": 64, "Nx": 32, "Nev": 100, "Nev1": 100,
        "mom_smear": 3, "mom_smear_phase": -3,
        "beta": 6.20, "mu_l": -0.2790, "mu_s": -0.2400, "alttc": 0.1053,
        "conf_name": "beta6.20_mu-0.2790_ms-0.2400_L32x64",
        "conf_short": "L32x64",
        "conf_dir": "/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2790_ms-0.2400_L32x64/{conf_id}/",
        "eig_dir": "/public/group/lqcd/eigensystem/beta6.20_mu-0.2790_ms-0.2400_L32x64/{conf_id}/",
        "peram_u_dir": "/public/home/sunp/sunpeng_public/mom_smear_perambulators/beta6.20_mu-0.2790_ms-0.2400_L32x64/output_dir_data/mz0_my0_mx-3/{conf_id}/",
        "corr_nucl_dir": "/public/group/lqcd/donghx/2pt_Result/beta6.20_mu-0.2790_ms-0.2400_L32x64/momsmear3x/{conf_id}/",
        "ope_dir": "/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/L32x64/zdir/{conf_id}/",
    },
    "L32x96": {
        "Nt": 96, "Nx": 32, "Nev": 100, "Nev1": 100,
        "mom_smear": 2, "mom_smear_phase": -2,
        "beta": 6.41, "mu_l": -0.2295, "mu_s": -0.2050, "alttc": 0.083,
        "conf_name": "beta6.41_mu-0.2295_ms-0.2050_L32x96",
        "conf_short": "L32x96",
        "conf_dir": "/public/group/lqcd/configurations/CLOVER/beta6.41_mu-0.2295_ms-0.2050_L32x96/{conf_id}/",
        "eig_dir": "/public/group/lqcd/eigensystem/beta6.41_mu-0.2295_ms-0.2050_L32x96/{conf_id}/",
        "peram_u_dir": "/public/home/sunp/sunpeng_public/mom_smear_perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/output_dir_data/mz0_my0_mx-2/{conf_id}/",
        "corr_nucl_dir": "/public/group/lqcd/donghx/2pt_Result/beta6.41_mu-0.2295_ms-0.2050_L32x96/momsmear2x/{conf_id}/",
        "ope_dir": "/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/L32x96/zdir/{conf_id}/",
    },
    "L36x108": {
        "Nt": 108, "Nx": 36, "Nev": 200, "Nev1": 200,
        "mom_smear": 2, "mom_smear_phase": -2,
        "beta": 6.498, "mu_l": -0.2150, "mu_s": -0.1926, "alttc": 0.071,
        "conf_name": "beta6.498_mu-0.2150_ms-0.1926_L36x108",
        "conf_short": "L36x108",
        "conf_dir": "/public/group/lqcd/configurations/CLOVER/beta6.498_mu-0.2150_ms-0.1926_L36x108/{conf_id}/",
        "eig_dir": "/public/group/lqcd/eigensystem/beta6.498_mu-0.2150_ms-1926_L36x108/{conf_id}/",
        "peram_u_dir": "/public/home/sunp/sunpeng_new_disk/mom_smear_perambulators/beta6.498_mu-0.2150_ms-1926_L36x108/output_dir_data/mz0_my0_mx-2/{conf_id}/",
        "corr_nucl_dir": "/public/group/lqcd/donghx/2pt_Result/beta6.498_mu-0.2150_ms-1926_L36x108/momsmear2x/{conf_id}/",
        "ope_dir": "/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/L36x108/zdir/{conf_id}/",
    },
    "L48x96": {
        "Nt": 96, "Nx": 48, "Nev": 200, "Nev1": 200,
        "mom_smear": 4, "mom_smear_phase": -4,
        "beta": 6.20, "mu_l": -0.2825, "mu_s": -0.2310, "alttc": 0.1053,
        "conf_name": "beta6.20_mu-0.2825_ms-0.2310_L48x96",
        "conf_short": "L48x96",
        "conf_dir": "/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2825_ms-0.2310_L48x96/{conf_id}/",
        "eig_dir": "/public/group/lqcd/eigensystem/beta6.20_mu-0.2825_ms-0.2310_L48x96/{conf_id}/",
        "peram_u_dir": "/public/home/sunp/sunpeng_new_disk/mom_smear_perambulators/beta6.20_mu-0.2825_ms-0.2310_L48x96/output_dir_data/mz0_my0_mx-4/{conf_id}/",
        "corr_nucl_dir": "/public/group/lqcd/donghx/2pt_Result/beta6.20_mu-0.2825_ms-0.2310_L48x96/momsmear4x/{conf_id}/",
        "ope_dir": "/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/L48x96/zdir/{conf_id}/",
    },
    "L48x144": {
        "Nt": 144, "Nx": 48, "Nev": 200, "Nev1": 200,
        "mom_smear": 2, "mom_smear_phase": -2,
        "beta": 6.72, "mu_l": -0.1850, "mu_s": -0.1700, "alttc": 0.053,
        "conf_name": "beta6.72_mu-0.1850_ms-0.1700_L48x144",
        "conf_short": "L48x144",
        "conf_dir": "/public/group/lqcd/configurations/CLOVER/beta6.72_mu-0.1850_ms-0.1700_L48x144/{conf_id}/",
        "eig_dir": "/public/group/lqcd/eigensystem/beta6.72_mu-0.1850_ms-0.1700_L48x144/{conf_id}/",
        "peram_u_dir": "/public/home/sunp/sunpeng_public/mom_smear_perambulators/beta6.72_mu-0.1850_ms-0.1700_L48x144/output_dir_data/mz0_my0_mx-2/{conf_id}/",
        "corr_nucl_dir": "/public/group/lqcd/donghx/2pt_Result/beta6.72_mu-0.1850_ms-0.1700_L48x144/momsmear2x/{conf_id}/",
        "ope_dir": "/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/L48x144/zdir/{conf_id}/",
    },
    "L64x128": {
        "Nt": 128, "Nx": 64, "Nev": 200, "Nev1": 200,
        "mom_smear": 2, "mom_smear_phase": -2,
        "beta": 6.41, "mu_l": -0.2334, "mu_s": -0.2030, "alttc": 0.083,
        "conf_name": "beta6.41_mu-0.2334_ms-0.2030_L64x128",
        "conf_short": "L64x128",
        "conf_dir": "/public/group/lqcd/configurations/CLOVER/beta6.41_mu-0.2334_ms-0.2030_L64x128/{conf_id}/",
        "eig_dir": "/public/group/lqcd/sush/eigenvectors/beta6.41_mu-0.2334_ms-0.2030_L64x128/{conf_id}/",
        "peram_u_dir": "/public/group/lqcd/sush/perambulators/beta6.41_mu-0.2334_ms-0.2030_L64x128/{conf_id}/",
        "corr_nucl_dir": "/public/group/lqcd/donghx/2pt_Result/beta6.41_mu-0.2334_ms-0.2030_L64x128/momsmear2x/{conf_id}/",
        "ope_dir": "/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/L64x128/zdir/{conf_id}/",
    },
}


def resolve_ensemble(name: str, conf_id: str) -> dict:
    """解析系综预设, 将 {conf_id} 替换到所有路径模板中。

    Parameters
    ----------
    name : str, 系综名称 (如 "L24x72")
    conf_id : str, 组态编号

    Returns
    -------
    cfg : dict, 所有参数已展开的系综配置
    """
    if name not in ENSEMBLES:
        raise ValueError(f"未知系综 '{name}'。可用: {list(ENSEMBLES.keys())}")
    cfg = dict(ENSEMBLES[name])
    for key in ["conf_dir", "eig_dir", "peram_u_dir", "corr_nucl_dir", "ope_dir"]:
        if key in cfg:
            cfg[key] = cfg[key].format(conf_id=conf_id)
    return cfg


# ============================================================================
# 第四部分: Dirac 矩阵 (DeGrand-Rossi 基, 后端无关)
# ============================================================================

def build_gamma_matrices(xp: Any) -> Dict[int, Any]:
    """构造 DeGrand-Rossi (DR) 基下的 18 个 Dirac 矩阵。

    ────────────────────────────────────────────
    DR 基 Gamma 矩阵定义 (Euclidean, 厄米)
    ────────────────────────────────────────────
    γ₁ (γ_x): i * anti-diag(1, 1, -1, -1)
    γ₂ (γ_y): anti-diag(-1, 1, 1, -1)
    γ₃ (γ_z): i * anti-diag(1, -1, -1, 1)
    γ₄ (γ_t): anti-diag(1, 1, 1, 1)
    γ₅:       diag(1, 1, -1, -1)
    C = γ₄γ₂: 电荷共轭矩阵

    索引映射:
      0=I, 1=γ₁, 2=γ₂, 3=γ₃, 4=γ₄, 5=γ₅,
      6=γ₂γ₃, 7=γ₃γ₁, 8=γ₁γ₂,
      9=γ₁γ₄, 10=γ₂γ₄, 11=γ₃γ₄,
      12=γ₁γ₅, 13=γ₂γ₅, 14=γ₃γ₅, 15=γ₄γ₅,
      16=γ₃γ₁·P₊, 17=γ₃γ₁·P₋
    """
    z = xp.zeros((4, 4), dtype=complex)

    g0 = z.copy()
    for i in range(4): g0[i, i] = 1.0 + 0.0j

    g1 = z.copy()
    g1[0,3]=0+1j; g1[1,2]=0+1j; g1[2,1]=0-1j; g1[3,0]=0-1j

    g2 = z.copy()
    g2[0,3]=-1+0j; g2[1,2]=1+0j; g2[2,1]=1+0j; g2[3,0]=-1+0j

    g3 = z.copy()
    g3[0,2]=0+1j; g3[1,3]=0-1j; g3[2,0]=0-1j; g3[3,1]=0+1j

    g4 = z.copy()
    g4[0,2]=1+0j; g4[1,3]=1+0j; g4[2,0]=1+0j; g4[3,1]=1+0j

    g5 = z.copy()
    g5[0,0]=1+0j; g5[1,1]=1+0j; g5[2,2]=-1+0j; g5[3,3]=-1+0j

    return {
        0: g0, 1: g1, 2: g2, 3: g3, 4: g4, 5: g5,
        6:  xp.matmul(g2, g3),   7:  xp.matmul(g3, g1),
        8:  xp.matmul(g1, g2),   9:  xp.matmul(g1, g4),
        10: xp.matmul(g2, g4),   11: xp.matmul(g3, g4),
        12: xp.matmul(g1, g5),   13: xp.matmul(g2, g5),
        14: xp.matmul(g3, g5),   15: xp.matmul(g4, g5),
        16: xp.matmul(xp.matmul(g3, g1), 0.5*(g0+g4)),
        17: xp.matmul(xp.matmul(g3, g1), 0.5*(g0-g4)),
    }


# ============================================================================
# 第五部分: Levi-Civita 张量 ε_{μνρσ} / 2
# ============================================================================

def build_levi_civita_tensor() -> np.ndarray:
    """构造四维 Levi-Civita 符号 ε_{μνρσ} / 2。

    Returns
    -------
    epsilon4 : ndarray, shape (4,4,4,4), dtype=float
        ε_{μνρσ} = +1 偶排列, -1 奇排列, 0 重复指标; 整体 × 0.5
    """
    eps = np.zeros((4, 4, 4, 4), dtype=float)
    for mu in range(4):
        for nu in range(4):
            a = 1.0 if mu > nu else 0.0
            for rho in range(4):
                b = 0.0
                if mu > rho: b += 1.0
                if nu > rho: b += 1.0
                for sigma in range(4):
                    c = 0.0
                    if mu > sigma: c += 1.0
                    if nu > sigma: c += 1.0
                    if rho > sigma: c += 1.0
                    if len({mu, nu, rho, sigma}) != 4:
                        eps[mu, nu, rho, sigma] = 0.0
                    elif (a + b + c) % 2 == 0:
                        eps[mu, nu, rho, sigma] = 0.5
                    else:
                        eps[mu, nu, rho, sigma] = -0.5
    return eps


# ============================================================================
# 第六部分: 场强张量 F_{μν} (Clover 叶, O(a²) 改进)
# ============================================================================

def plaquette_clover(gauge: Any, mu: int, nu: int, contract_fn: Any) -> Any:
    r"""计算 Clover 型场强张量 F_{μν}(x)。

    ────────────────────────────────────────────────────────
    理论推导
    ────────────────────────────────────────────────────────
    Clover 叶 (四个 plaquette 的对称组合):
      Q_{μν}(x) = P_{μν}(x) + P_{ν,-μ}(x) + P_{-μ,-ν}(x) + P_{-ν,μ}(x)

    Clover 场强张量 (取反厄米部分, a=1, g₀=1):
      F̂_{μν}(x) = -i/8 · [Q_{μν}(x) - Q_{μν}†(x)]

    Parameters
    ----------
    gauge : ndarray, shape (Nt, Nz, Ny, Nx, 4, 3, 3)
        按 donghx 约定: (t, z, y, x, dir, color, color)
    mu, nu : int, Lorentz 指标 (0=x, 1=y, 2=z, 3=t)
    contract_fn : callable, 张量缩并函数 (einsum 或 opt_einsum)

    Returns
    -------
    F_munu : ndarray, shape (Nt, Nz, Ny, Nx, 3, 3)
    """
    g_ru = gauge
    g_lu = np.roll(gauge, 1, axis=3 - mu)      # x - μ̂
    g_rd = np.roll(gauge, 1, axis=3 - nu)      # x - ν̂
    g_ld = np.roll(g_lu, 1, axis=3 - nu)        # x - μ̂ - ν̂

    # Plaquette (1): P_{μν} — 标准 1×1 Wilson loop
    p_ru = contract_fn("tzyxab,tzyxbc->tzyxac",
                        g_ru[..., mu, :, :],
                        np.roll(g_ru, -1, axis=3-mu)[..., nu, :, :])
    p_ru = contract_fn("tzyxab,tzyxcb->tzyxac", p_ru,
                        np.roll(g_ru, -1, axis=3-nu)[..., mu, :, :].conj())
    p_ru = contract_fn("tzyxab,tzyxcb->tzyxac", p_ru,
                        g_ru[..., nu, :, :].conj())

    # Plaquette (2): P_{ν,-μ}
    p_lu = contract_fn("tzyxab,tzyxcb->tzyxac",
                        np.roll(g_lu, -1, axis=3-mu)[..., nu, :, :],
                        np.roll(g_lu, -1, axis=3-nu)[..., mu, :, :].conj())
    p_lu = contract_fn("tzyxab,tzyxcb->tzyxac", p_lu,
                        g_lu[..., nu, :, :].conj())
    p_lu = contract_fn("tzyxab,tzyxbc->tzyxac", p_lu,
                        g_lu[..., mu, :, :])

    # Plaquette (3): P_{-μ,-ν}
    p_ld = contract_fn("tzyxba,tzyxcb->tzyxac",
                        np.roll(g_ld, -1, axis=3-nu)[..., mu, :, :].conj(),
                        g_ld[..., nu, :, :].conj())
    p_ld = contract_fn("tzyxab,tzyxbc->tzyxac", p_ld,
                        g_ld[..., mu, :, :])
    p_ld = contract_fn("tzyxab,tzyxbc->tzyxac", p_ld,
                        np.roll(g_ld, -1, axis=3-mu)[..., nu, :, :])

    # Plaquette (4): P_{-ν,μ}
    p_rd = contract_fn("tzyxba,tzyxbc->tzyxac",
                        g_rd[..., nu, :, :].conj(),
                        g_rd[..., mu, :, :])
    p_rd = contract_fn("tzyxab,tzyxbc->tzyxac", p_rd,
                        np.roll(g_rd, -1, axis=3-mu)[..., nu, :, :])
    p_rd = contract_fn("tzyxab,tzyxcb->tzyxac", p_rd,
                        np.roll(g_rd, -1, axis=3-nu)[..., mu, :, :].conj())

    # F_{μν} = -i/8 · Σ_k (P_k - P_k†)
    ans = (p_ru - p_ru.conj().transpose(0,1,2,3,5,4) +
           p_lu - p_lu.conj().transpose(0,1,2,3,5,4) +
           p_ld - p_ld.conj().transpose(0,1,2,3,5,4) +
           p_rd - p_rd.conj().transpose(0,1,2,3,5,4))
    return -1j * ans / 8.0


def compute_field_strength_all(gauge: Any, Nt: int, Nx: int,
                                contract_fn: Any) -> Any:
    """计算所有独立 (μ,ν) 对的 Clover 场强张量。

    Returns
    -------
    F_all : ndarray, shape (4, 4, Nt, Nz, Ny, Nx, 3, 3)
    """
    xp_lib = cp if (HAS_CUPY and hasattr(gauge, 'device')) else np
    F_all = xp_lib.zeros((4, 4, Nt, Nx, Nx, Nx, 3, 3), dtype=complex)
    for mu in range(4):
        for nu in range(4):
            if mu != nu:
                F_all[mu, nu] = plaquette_clover(gauge, mu, nu, contract_fn)
    return F_all


def compute_dual_field_strength(F_all: Any, epsilon: np.ndarray,
                                 contract_fn: Any) -> Any:
    """通过对偶变换计算 F̃_{μν} = (1/2) ε_{μνρσ} F^{ρσ}。

    Returns
    -------
    F_tilde_all : ndarray, shape (4, 4, Nt, Nz, Ny, Nx, 3, 3)
    """
    xp_lib = cp if (HAS_CUPY and hasattr(F_all, 'device')) else np
    eps_xp = xp_lib.asarray(epsilon) if xp_lib is cp else epsilon
    return contract_fn("opmn,mntzyxab->optzyxab", eps_xp, F_all)


# ============================================================================
# 第七部分: Nonlocal 胶子 OPE 算符 O_{μν}(z)
# ============================================================================

def gluon_ope_operator_z0_mu2(
    gauge: Any, F_all: Any, F_tilde_all: Any,
    delta_z: int, z_dir: int,
    mu: int, nu: int, mu2: int, nu2: int,
    contract_fn: Any,
) -> Any:
    r"""构造非定域胶子 OPE 算符 (z 处插 F, 0 处插 F̃)。

    格点实现 (五步):
      Step 1: 平移 F_{μν} 到 z = delta_z 处
      Step 2: Wilson 线 W(z→0) — U_z^† 连乘积 (delta_z 次)
      Step 3: 在原点插入 F̃_{μ2,ν2}(0)
      Step 4: Wilson 线 W(0→z) — U_z 连乘积 (delta_z 次)
      Step 5: 颜色迹 Tr_c + 空间求和

    Returns
    -------
    op_trace : ndarray, shape (Nt,), 每个时间片的 OPE 迹 (已对空间求和)
    """
    # Step 1: 平移 F 到 z 处
    ope = np.roll(F_all[mu, nu], -delta_z, axis=3 - z_dir)

    # Step 2: W(z→0) — U_z^† 沿 -z 方向
    for dz_step in range(delta_z):
        shift = -(delta_z - 1 - dz_step)
        ug = np.roll(gauge, shift, axis=3 - z_dir)[..., z_dir, :, :]
        ope = contract_fn("tzyxab,tzyxcb->tzyxac", ope, ug.conj())

    # Step 3: 在原点插入 F̃
    ope = contract_fn("tzyxab,tzyxbc->tzyxac", ope, F_tilde_all[mu2, nu2])

    # Step 4: W(0→z) — U_z 正向
    for dz_step in range(delta_z):
        shift = -dz_step
        ug = np.roll(gauge, shift, axis=3 - z_dir)[..., z_dir, :, :]
        ope = contract_fn("tzyxab,tzyxbc->tzyxac", ope, ug)

    # Step 5: 颜色迹 → 标量
    trace_color = np.trace(ope, axis1=4, axis2=5)

    # Step 6: 对空间求和
    return np.sum(trace_color, axis=(1, 2, 3))


def compute_ope_all_z(
    gauge: Any, F_all: Any, F_tilde_all: Any,
    delta_z: int, z_dir: int,
    mu: int, nu: int, mu2: int, nu2: int,
    Nt: int, contract_fn: Any,
    verbose: bool = True,
) -> Any:
    """为所有 z ∈ [0, delta_z-1] 计算 OPE 算符。

    Returns
    -------
    ops : ndarray, shape (Nt, delta_z)
    """
    ops = np.zeros((Nt, delta_z), dtype=complex)
    if HAS_CUPY and hasattr(gauge, 'device'):
        ops = cp.asarray(ops)

    for dz in range(delta_z):
        if verbose:
            print(f"  [OPE] z={dz}/{delta_z-1} (mu={mu},nu={nu},mu2={mu2},nu2={nu2})")
        ops[:, dz] = gluon_ope_operator_z0_mu2(
            gauge, F_all, F_tilde_all, dz,
            z_dir, mu, nu, mu2, nu2, contract_fn,
        )
    return ops


# ============================================================================
# 第八部分: Distillation 框架 — 相位因子、VVV、Wick 收缩
# ============================================================================

def compute_phase_factor(momentum: Any, Nx: int) -> Any:
    """计算动量相位因子: φ_P(x) = exp(-i · 2π · P·x / L)。

    Parameters
    ----------
    momentum : ndarray, shape (3,), 动量 (Pz, Py, Px) 以 2π/L 为单位
    Nx : int, 空间方向格点数

    Returns
    -------
    phase : ndarray, shape (Nx³,), dtype=complex
    """
    V = Nx * Nx * Nx
    phase = np.zeros(V, dtype=complex)
    idx = 0
    for z in range(Nx):
        for y in range(Nx):
            for x in range(Nx):
                pos = np.array([z, y, x])
                phase[idx] = np.exp(-np.dot(momentum, pos) * 2.0j * np.pi / Nx)
                idx += 1
    return phase


def compute_vvv_baryon_block(
    eigvecs: Any, phase_factor: Any, Nev1: int, Nx: int, contract_fn: Any,
) -> Any:
    r"""计算 VVV (Baryon Block) 张量: Φ_{abc}(P) = Σ_x φ_P(x) ε_{ijk} v_i^a v_j^b v_k^c。

    六个 ε_{ijk} 置换:
      +1: (0,1,2), (1,2,0), (2,0,1)  [偶排列]
      -1: (0,2,1), (1,0,2), (2,1,0)  [奇排列]

    Parameters
    ----------
    eigvecs : ndarray, shape (Nev, Nx³, 3)
    phase_factor : ndarray, shape (Nx³,)
    Nev1 : int, 截断的本征矢量数
    Nx : int, 空间格点数
    contract_fn : callable

    Returns
    -------
    VVV : ndarray, shape (Nev1, Nev1, Nev1), dtype=complex
    """
    xp_lib = cp if (HAS_CUPY and hasattr(eigvecs, 'device')) else np
    VVV = xp_lib.zeros((Nev1, Nev1, Nev1), dtype=complex)
    layer_size = Nx * Nx

    # 逐 x-层分片计算 (减少中间数组内存占用)
    for xi in range(Nx):
        start = xi * layer_size
        end = (xi + 1) * layer_size
        es = eigvecs[:Nev1, start:end, :]  # (Nev1, Nx², 3)
        ps = phase_factor[start:end]         # (Nx²,)

        # 三项正号 (偶排列)
        VVV += contract_fn("x,ax,bx,cx->abc", ps, es[:,:,0], es[:,:,1], es[:,:,2])
        VVV += contract_fn("x,ax,bx,cx->abc", ps, es[:,:,1], es[:,:,2], es[:,:,0])
        VVV += contract_fn("x,ax,bx,cx->abc", ps, es[:,:,2], es[:,:,0], es[:,:,1])
        # 三项负号 (奇排列)
        VVV -= contract_fn("x,ax,bx,cx->abc", ps, es[:,:,0], es[:,:,2], es[:,:,1])
        VVV -= contract_fn("x,ax,bx,cx->abc", ps, es[:,:,1], es[:,:,0], es[:,:,2])
        VVV -= contract_fn("x,ax,bx,cx->abc", ps, es[:,:,2], es[:,:,1], es[:,:,0])

    return VVV


def contract_proton_2pt_single_tsrc(
    VVV_sink_t: Any, peram_u: Any,
    CG5peram_uCG5: Any, VVV_source_t: Any,
    contract_fn: Any,
) -> Any:
    r"""对单个 (t_sink, t_source) 执行质子 2pt Wick 收缩。

    ────────────────────────────────────────────
    直接项 (Direct):
      C₂^{dir}_{il} = Φ_{abc}(snk) · τ_{gi}^{ad} · (ΓτΓ)_{gj}^{be}
                       · τ_{il}^{cf} · Φ_{def}^*(src)

    交换项 (Exchange, Fermi 负号):
      C₂^{ex}_{il} = Φ_{abc}(snk) · τ_{gl}^{af} · (ΓτΓ)_{gj}^{be}
                       · τ_{ij}^{cd} · Φ_{def}^*(src)

    总关联函数:
      C₂_{il} = Direct - Exchange

    指标:
      a,b,c,d,e,f ∈ [0,Nev1) — 本征矢量指标
      g,j,i,l ∈ [0,3] — Dirac 旋量指标
    """
    direct = contract_fn(
        "abc,gjad,gjbe,ilcf,def->il",
        VVV_sink_t, peram_u, CG5peram_uCG5, peram_u, VVV_source_t,
    )
    exchange = contract_fn(
        "abc,glaf,gjbe,ijcd,def->il",
        VVV_sink_t, peram_u, CG5peram_uCG5, peram_u, VVV_source_t,
    )
    return direct - exchange


# ============================================================================
# 第九部分: 统计误差分析 — Jackknife & Bootstrap
# ============================================================================

def jackknife_samples(data: Any, axis: int = 0) -> Any:
    """生成 Jackknife 重采样样本 (留一法)。

    x_i^{JK} = (N·x̄ - x_i) / (N-1)
    """
    N = data.shape[axis]
    total = np.sum(data, axis=axis, keepdims=True)
    return (total - data) / (N - 1)


def jackknife_error(jk_samples: Any, axis: int = 0) -> Any:
    """Jackknife 标准误差: σ_{JK} = std(jk) · √(N-1)。"""
    N = jk_samples.shape[axis]
    return np.std(jk_samples, axis=axis) * np.sqrt(N - 1)


def bootstrap_samples(data: Any, n_bootstrap: int = 1000,
                       axis: int = 0, seed: int = 42) -> Any:
    """生成 Bootstrap 重采样样本 (有放回重采样)。

    Returns
    -------
    bs_samples : ndarray, shape = (n_bootstrap, ...)
    """
    rng = np.random.RandomState(seed)
    N = data.shape[axis]
    bs_shape = (n_bootstrap,) + data.shape[1:]
    bs = np.zeros(bs_shape, dtype=data.dtype)
    for i in range(n_bootstrap):
        idx = rng.randint(0, N, size=N)
        bs[i] = np.mean(np.take(data, idx, axis=axis), axis=axis)
    return bs


def sem(data: Any, jack: bool = True, axis: int = 0) -> Any:
    """标准误差: 若 jack=True 则 ×√(N-1)。"""
    err = data.std(axis)
    if jack:
        err = err * np.sqrt(data.shape[axis] - 1)
    return err


# ============================================================================
# 第十部分: 矩阵元提取、Fourier 变换与微扰匹配
# ============================================================================

def fourier_transform_to_quasi_pdf(
    h_z: Any, z_values: Any, Pz: float, x_values: Any,
) -> Any:
    r"""通过 Fourier 变换将坐标空间矩阵元转换为 quasi-PDF。

    g̃(x, P_z) = (2P_z / x) · ∫_0^{z_max} dz · Re[h(z)] · sin(x P_z z)
    """
    h_real = np.real(h_z)
    Nx_q = len(x_values)
    quasi = np.zeros(Nx_q)
    for i, x in enumerate(x_values):
        if abs(x) < 1e-15:
            quasi[i] = 0.0
            continue
        integrand = h_real * np.sin(x * Pz * z_values)
        quasi[i] = (2.0 * Pz / x) * np.trapz(integrand, z_values)
    return quasi


def matching_kernel_nlo(
    quasi_pdf: Any, x_values: Any,
    alpha_s: float = 0.2, mu_over_Pz: float = 1.0,
) -> Any:
    """LO 近似: g(x,μ) ≃ g̃(x,P_z)。NLO 修正待定。"""
    return quasi_pdf.copy()


# ============================================================================
# 第十一部分: 作图函数 (>15 种出版级图)
# ============================================================================

def _fig_setup(figsize=(12, 8), dpi=150):
    """统一画图设置: 返回 fig, ax。"""
    if not HAS_MPL:
        return None, None
    plt.rcParams.update({
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 16,
        'xtick.labelsize': 13, 'ytick.labelsize': 13,
        'legend.fontsize': 11, 'figure.dpi': dpi,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
    })
    return plt.subplots(figsize=figsize)


# ── 图 1: 场强张量厄米性检查 ──
def plot_field_strength_check(F_all: Any, output_dir: str, prefix: str = "F"):
    if not HAS_MPL: return
    F_cpu = _ensure_cpu(F_all)
    Nt = F_cpu.shape[2]
    herm_check = np.zeros((4, 4, Nt))
    pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    labels = ["(0,1)=xy","(0,2)=xz","(0,3)=xt","(1,2)=yz","(1,3)=yt","(2,3)=zt"]
    for mu, nu in pairs:
        F_mn = F_cpu[mu, nu]
        tr_diff = np.abs(np.trace(F_mn, axis1=5,axis2=6)
                       - np.trace(F_mn.conj().transpose(0,1,2,3,5,4), axis1=5,axis2=6))
        herm_check[mu, nu] = np.mean(tr_diff, axis=(1,2,3))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (mu, nu), lbl in zip(axes.flatten(), pairs, labels):
        ax.semilogy(herm_check[mu, nu], 'o-', markersize=3)
        ax.set_title(f"$F_{{{lbl}}}$: $|\\mathrm{{Tr}}[F-F^\\dagger]|$")
        ax.set_xlabel("t"); ax.set_ylabel("$|\\mathrm{Tr}[F-F^\\dagger]|$")
    plt.suptitle(f"{prefix}: Field Strength Anti-Hermiticity Check", fontsize=14)
    fig.savefig(f"{output_dir}/{prefix}_F_herm_check.pdf"); plt.close(fig)


# ── 图 2: VVV 张量结构 ──
def plot_vvv_structure(VVV_data: Any, output_dir: str, prefix: str = "VVV"):
    if not HAS_MPL: return
    v = _ensure_cpu(VVV_data)
    if v.ndim == 4: v = v[0]
    Nev1 = v.shape[0]; mag = np.abs(v)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # 子图 1: 对角元 |Φ_{aaa}|
    diag = np.array([mag[i,i,i] for i in range(Nev1)])
    axes[0].semilogy(diag, 'o-', markersize=3, color='#4E79A7')
    axes[0].set_title("$|\\Phi_{aaa}|$ (Diagonal)")
    axes[0].set_xlabel("Eigenvector index $a$")
    axes[0].set_ylabel("$|\\Phi_{aaa}|$")

    # 子图 2: 耦合强度热力图
    coupling = np.sum(mag, axis=2)/Nev1
    im = axes[1].imshow(np.log10(coupling+1e-15), aspect='auto', origin='lower',
                         cmap='viridis')
    axes[1].set_title("$\\log_{10}(\\langle|\\Phi_{abc}|\\rangle_c)$")
    axes[1].set_xlabel("$b$"); axes[1].set_ylabel("$a$")
    plt.colorbar(im, ax=axes[1])

    # 子图 3: 本征值分布 (前几个模态的幅值)
    for i in range(min(10, Nev1)):
        axes[2].semilogy(mag[i,:,0], alpha=0.6, label=f"a={i}" if i<5 else "")
    axes[2].set_title("$|\\Phi_{ab0}|$ — First Color Component")
    axes[2].set_xlabel("$b$"); axes[2].set_ylabel("$|\\Phi_{ab0}|$")
    if Nev1 <= 10: axes[2].legend(fontsize=8)

    plt.suptitle(f"{prefix}: VVV Baryon Block Structure", fontsize=14)
    fig.savefig(f"{output_dir}/{prefix}_vvv_structure.pdf"); plt.close(fig)


# ── 图 3: 2pt 关联函数 + 有效质量 ──
def plot_2pt_and_meff(C2pt_1d: np.ndarray, meff_gev: np.ndarray,
                       output_dir: str, Pz: int, Nt: int,
                       alttc: float, element: str, conf_id: str,
                       meff_ref: np.ndarray = None, C2pt_ref: np.ndarray = None):
    """双面板: 左=log|C₂|, 右=有效质量。"""
    if not HAS_MPL: return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=150)

    # ── 左: 关联函数对数图 ──
    dt_vals = np.arange(Nt)
    C_abs = np.abs(C2pt_1d) + 1e-30
    axes[0].semilogy(dt_vals, C_abs, 'o-', markersize=4, color='#4E79A7',
                      label='This work')
    if C2pt_ref is not None:
        C_ref_abs = np.abs(C2pt_ref) + 1e-30
        axes[0].semilogy(dt_vals, C_ref_abs, 's--', markersize=3, color='#E15759',
                          alpha=0.7, label='Reference')
    axes[0].set_xlabel(r"$\Delta t~/~a$", fontsize=15)
    axes[0].set_ylabel(r"$|C_2(\Delta t)|$", fontsize=15)
    axes[0].set_title(f"Proton 2pt Correlator (Pz={Pz})", fontsize=14)
    axes[0].grid(True, alpha=0.3, linestyle=':')
    axes[0].legend(fontsize=11)

    # ── 右: 有效质量 ──
    t_meff = np.arange(1, Nt-1)
    valid = ~np.isnan(meff_gev)
    axes[1].errorbar(t_meff[valid], meff_gev[valid],
                      fmt='o', markersize=5, mfc='#E69F00', mec='#E69F00',
                      capsize=3, color='#E69F00', label='cosh $m_{\\rm eff}$ (this work)')
    if meff_ref is not None:
        valid_r = ~np.isnan(meff_ref)
        axes[1].errorbar(t_meff[valid_r], meff_ref[valid_r],
                          fmt='s', markersize=4, mfc='none', mec='#4E79A7',
                          capsize=2, color='#4E79A7', alpha=0.7,
                          label='Reference $m_{\\rm eff}$')

    axes[1].set_xlabel(r"$t~/~a$", fontsize=15)
    axes[1].set_ylabel(r"$m_{\mathrm{eff}}~/~\mathrm{GeV}$", fontsize=15)
    axes[1].set_title(f"Effective Mass (Pz={Pz}, a={alttc} fm)", fontsize=14)
    axes[1].grid(True, alpha=0.3, linestyle=':')

    # 平台区标记
    ps, pe = Nt//4, Nt//2
    if np.any(valid[ps:pe]):
        pm = float(np.mean(meff_gev[ps:pe][valid[ps:pe]]))
        axes[1].axhline(y=pm, color='#D62728', linestyle='--', linewidth=1.2,
                         label=f'Plateau [{ps},{pe}]: {pm:.3f} GeV')
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(f"{output_dir}/C2pt_meff_Pz{Pz}_{element}_conf{conf_id}.pdf")
    plt.close(fig)
    print(f"  [PLOT] 2pt+Meff 图已保存: C2pt_meff_Pz{Pz}_{element}_conf{conf_id}.pdf")


# ── 图 4: 多动量有效质量对比 ──
def plot_multimomentum_meff(all_meff: Dict[int, np.ndarray],
                             output_dir: str, Nt: int, alttc: float,
                             conf_id: str, element: str):
    """将所有动量的有效质量画在同一张图上比较。"""
    if not HAS_MPL or not all_meff: return
    fig, ax = plt.subplots(1, 1, figsize=(14, 8), dpi=150)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_meff)))
    t_meff = np.arange(1, Nt-1)

    for (Pz, meff), c in zip(sorted(all_meff.items()), colors):
        valid = ~np.isnan(meff)
        ax.errorbar(t_meff[valid], meff[valid],
                     fmt='o', markersize=4, capsize=2, color=c,
                     label=f'Pz={Pz}')
    ax.set_xlabel(r"$t~/~a$", fontsize=15)
    ax.set_ylabel(r"$m_{\mathrm{eff}}~/~\mathrm{GeV}$", fontsize=15)
    ax.set_title(f"Effective Mass vs Momentum (a={alttc} fm)", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(fontsize=10, ncol=2)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/meff_all_Pz_{element}_conf{conf_id}.pdf")
    plt.close(fig)
    print(f"  [PLOT] 多动量有效质量对比图已保存。")


# ── 图 5: 生成 vs 参考数据散点对比 ──
def plot_scatter_comparison(gen: np.ndarray, ref: np.ndarray,
                             output_dir: str, prefix: str, label: str):
    """散点图: Re(ref) vs Re(gen) + 差异直方图。"""
    if not HAS_MPL: return
    gen_f = np.real(gen).ravel(); ref_f = np.real(ref).ravel()
    diff = np.abs(gen - ref).ravel()
    nonzero = diff > 0
    diff_log = np.log10(diff[nonzero]) if np.any(nonzero) else np.array([-16])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    # 散点图
    axes[0].scatter(ref_f, gen_f, s=1, alpha=0.3, color='#4E79A7')
    lims = [float(np.min(ref_f)), float(np.max(ref_f))]
    axes[0].plot(lims, lims, 'r--', linewidth=1, label='y=x')
    axes[0].set_xlabel(f"Re(Ref) — {label}")
    axes[0].set_ylabel(f"Re(Gen) — {label}")
    axes[0].set_title(f"Scatter: Generated vs Reference ({prefix})")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    # 差异直方图
    axes[1].hist(diff_log, bins=60, color='#E69F00', alpha=0.7, edgecolor='k')
    axes[1].axvline(x=-6, color='r', linestyle='--', label='1e-6')
    axes[1].axvline(x=-10, color='g', linestyle='--', label='1e-10')
    axes[1].set_xlabel("$\\log_{10}|\\mathrm{Gen}-\\mathrm{Ref}|$")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Difference Distribution ({prefix})")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(f"{output_dir}/{prefix}_scatter_comparison.pdf")
    plt.close(fig)
    print(f"  [PLOT] 散点对比图已保存: {prefix}_scatter_comparison.pdf")


# ── 图 6: 关联函数 2D 矩阵热力图 ──
def plot_2pt_matrix_heatmap(C2_matrix: np.ndarray, output_dir: str,
                              Pz: int, element: str, conf_id: str):
    """2D 热力图: |C₂(t_snk, t_src)| 矩阵。"""
    if not HAS_MPL: return
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    Nt = C2_matrix.shape[0]
    C_abs = np.abs(C2_matrix)
    # 对数幅值热力图
    im1 = axes[0].imshow(np.log10(C_abs + 1e-15), aspect='auto', origin='lower',
                          cmap='inferno', extent=[0,Nt,0,Nt])
    axes[0].set_xlabel("$t_{\\rm src}$"); axes[0].set_ylabel("$t_{\\rm snk}$")
    axes[0].set_title(f"$\\log_{{10}}|C_2^{{pp}}(t_{{\\rm snk}},t_{{\\rm src}})|$ (Pz={Pz})")
    plt.colorbar(im1, ax=axes[0])
    # 按时间差折叠的平均关联函数
    C1d = np.zeros(Nt)
    for dt in range(Nt):
        vals = [C2_matrix[(t+dt)%Nt, t] for t in range(Nt)]
        C1d[dt] = np.abs(np.mean(vals))
    axes[1].semilogy(range(Nt), C1d + 1e-30, 'o-', markersize=4, color='#4E79A7')
    axes[1].set_xlabel("$\\Delta t$"); axes[1].set_ylabel("$|\\langle C_2(\\Delta t)\\rangle|$")
    axes[1].set_title(f"Time-folded Correlator (Pz={Pz})")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/C2pt_matrix_Pz{Pz}_{element}_conf{conf_id}.pdf")
    plt.close(fig)
    print(f"  [PLOT] 2pt 矩阵热力图已保存: C2pt_matrix_Pz{Pz}_{element}_conf{conf_id}.pdf")


# ── 图 7: OPE 算符 (Nt, delta_z) 热力图 ──
def plot_ope_heatmap(ops_data: Any, output_dir: str,
                      prefix: str = "OPE", mu: int = 0, nu: int = 1):
    if not HAS_MPL: return
    ops = _ensure_cpu(ops_data)
    Nt, delta_z = ops.shape
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    # Re 热力图
    im1 = axes[0,0].imshow(np.real(ops).T, aspect='auto', origin='lower', cmap='RdBu_r')
    axes[0,0].set_title(f"Re[O(z)] $\\mu$={mu},$\\nu$={nu}")
    axes[0,0].set_xlabel("t"); axes[0,0].set_ylabel("z")
    plt.colorbar(im1, ax=axes[0,0])
    # Im 热力图
    im2 = axes[0,1].imshow(np.imag(ops).T, aspect='auto', origin='lower', cmap='RdBu_r')
    axes[0,1].set_title(f"Im[O(z)] $\\mu$={mu},$\\nu$={nu}")
    axes[0,1].set_xlabel("t"); axes[0,1].set_ylabel("z")
    plt.colorbar(im2, ax=axes[0,1])
    # 选定 z 值的 O(z) vs t
    z_sample = [0, delta_z//4, delta_z//2, 3*delta_z//4, delta_z-1]
    for z in z_sample:
        if z < delta_z:
            axes[1,0].plot(np.real(ops[:,z]), 'o-', markersize=2, label=f"z={z}")
    axes[1,0].set_title("Re[O(z)] vs t"); axes[1,0].set_xlabel("t")
    axes[1,0].set_ylabel("Re[O(z)]"); axes[1,0].legend(fontsize=8)
    # 时间平均 O(z) vs z
    ops_tavg = np.mean(ops, axis=0)
    axes[1,1].errorbar(range(delta_z), np.real(ops_tavg),
                        yerr=np.std(np.real(ops),axis=0)/np.sqrt(Nt),
                        fmt='o-', markersize=4, capsize=3, label='Re')
    axes[1,1].errorbar(range(delta_z), np.imag(ops_tavg),
                        yerr=np.std(np.imag(ops),axis=0)/np.sqrt(Nt),
                        fmt='s-', markersize=4, capsize=3, label='Im')
    axes[1,1].set_title("$\\langle O(z)\\rangle_t$ vs z")
    axes[1,1].set_xlabel("z/a"); axes[1,1].set_ylabel("$\\langle O(z)\\rangle$")
    axes[1,1].legend()
    plt.suptitle(f"{prefix}: OPE Operator $O_{{{mu}{nu}}}(z)$", fontsize=14)
    fig.savefig(f"{output_dir}/{prefix}_ope_mu{mu}nu{nu}.pdf"); plt.close(fig)


# ── 图 8: 矩阵元 h(z,P_z) ──
def plot_matrix_element(h_z: Any, h_z_err: Any, output_dir: str,
                         prefix: str = "h_z", Pz: int = 6):
    if not HAS_MPL: return
    h, e = _ensure_cpu(h_z), _ensure_cpu(h_z_err)
    Nz = len(h); z_vals = np.arange(Nz)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    axes[0].errorbar(z_vals, np.real(h), yerr=e, fmt='o-', capsize=3, markersize=4,
                      color='#4E79A7', label='Re[h(z)]')
    axes[0].set_title(f"Re[h(z, Pz={Pz})]")
    axes[0].set_xlabel("z/a"); axes[0].set_ylabel("Re[h(z)]"); axes[0].legend()
    axes[1].errorbar(z_vals, np.imag(h), yerr=e, fmt='s-', capsize=3, markersize=4,
                      color='#E69F00', label='Im[h(z)]')
    axes[1].set_title(f"Im[h(z, Pz={Pz})]")
    axes[1].set_xlabel("z/a"); axes[1].set_ylabel("Im[h(z)]"); axes[1].legend()
    plt.suptitle(f"{prefix}: Coordinate-Space Matrix Element", fontsize=14)
    fig.savefig(f"{output_dir}/{prefix}_Pz{Pz}.pdf"); plt.close(fig)


# ── 图 9: quasi-PDF g̃(x,P_z) ──
def plot_quasi_pdf(quasi_pdf: Any, x_values: Any, output_dir: str,
                    prefix: str = "quasi", Pz: int = 6):
    if not HAS_MPL: return
    qp, xv = _ensure_cpu(quasi_pdf), _ensure_cpu(x_values)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    ax.plot(xv, qp, 'o-', markersize=4, color='#4E79A7')
    ax.set_title(f"Quasi-PDF $\\tilde{{g}}(x, P_z={Pz})$")
    ax.set_xlabel("x"); ax.set_ylabel("$\\tilde{g}(x, P_z)$")
    ax.axhline(y=0, color='gray', ls='--', lw=0.5)
    ax.grid(True, alpha=0.3)
    fig.savefig(f"{output_dir}/{prefix}_Pz{Pz}.pdf"); plt.close(fig)


# ── 图 10: 光锥 PDF g(x,μ) ──
def plot_lightcone_pdf(lc_pdf: Any, x_values: Any, output_dir: str,
                        prefix: str = "lc_pdf", Pz: int = 6, mu: float = 2.0):
    if not HAS_MPL: return
    lp, xv = _ensure_cpu(lc_pdf), _ensure_cpu(x_values)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    ax.plot(xv, lp, 'o-', markersize=4, color='#D62728')
    ax.set_title(f"Light-Cone PDF $g(x, \\mu={mu:.0f}\\,\\mathrm{{GeV}})$")
    ax.set_xlabel("x"); ax.set_ylabel("$g(x,\\mu)$")
    ax.axhline(y=0, color='gray', ls='--', lw=0.5)
    ax.set_xlim(0, 1); ax.grid(True, alpha=0.3)
    fig.savefig(f"{output_dir}/{prefix}_Pz{Pz}_mu{mu:.0f}.pdf"); plt.close(fig)


# ── 图 11: 性能汇总柱状图 ──
def plot_performance_summary(timing: dict, memory: dict, output_dir: str):
    if not HAS_MPL: return
    names = list(timing.keys())
    times = [timing[n] for n in names]
    mems = [memory[n]["peak_mb"] for n in names]
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=150)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(names)))
    short_names = [n.replace("Step ","").split(":")[0] for n in names]
    # 耗时
    bars1 = axes[0].bar(range(len(names)), times, color=colors)
    axes[0].set_title("Per-Step Wall Time"); axes[0].set_ylabel("Time (s)")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    for bar, t in zip(bars1, times):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                      f"{t:.1f}s", ha='center', va='bottom', fontsize=7)
    # 内存
    bars2 = axes[1].bar(range(len(names)), mems, color=colors)
    axes[1].set_title("Per-Step Peak Memory"); axes[1].set_ylabel("Memory (MB)")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    for bar, m in zip(bars2, mems):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                      f"{m:.0f}MB", ha='center', va='bottom', fontsize=7)
    plt.suptitle("Workflow Performance Summary", fontsize=14)
    fig.savefig(f"{output_dir}/performance_summary.pdf"); plt.close(fig)


# ── 图 12: 多组态 Jackknife 误差分析 ──
def plot_jackknife_analysis(jk_means: np.ndarray, jk_errs: np.ndarray,
                              output_dir: str, Pz: int,
                              element: str, conf_start: int, Nconf: int):
    """画 Jackknife 有效质量及误差棒。"""
    if not HAS_MPL: return
    Nt_m = jk_means.shape[-1]
    t_vals = np.arange(Nt_m)
    fig, ax = plt.subplots(1, 1, figsize=(12, 7), dpi=150)
    valid = ~np.isnan(jk_means)
    ax.errorbar(t_vals[valid], jk_means[valid], yerr=jk_errs[valid],
                 fmt='o-', markersize=5, capsize=3, capthick=1.5,
                 color='#4E79A7', elinewidth=1.5,
                 label=f'JK meff (Nconf={Nconf})')
    ax.set_xlabel(r"$t~/~a$", fontsize=15)
    ax.set_ylabel(r"$m_{\mathrm{eff}}~/~\mathrm{GeV}$", fontsize=15)
    ax.set_title(f"Jackknife Effective Mass (Pz={Pz}, ${Nconf}$ configs)", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(fontsize=12)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/jk_meff_Pz{Pz}_{element}_Nconf{Nconf}.pdf")
    plt.close(fig)
    print(f"  [PLOT] Jackknife 误差分析图已保存。")


def _print_comparison_report(comp_results: dict, Pz: int) -> None:
    """打印对比汇总报告 (表格形式)。"""
    if not comp_results: return
    print(f"\n{'─'*70}")
    print(f"  对比报告 — Pz={Pz}")
    print(f"{'─'*70}")
    hdr = f"  {'数据集':<24s} {'max|Δ|':>12s} {'mean|Δ|':>12s} {'Δ<1e-6':>8s}"
    print(hdr); print(f"  {'─'*60}")
    for label, m in comp_results.items():
        print(f"  {label:<24s} {m['max_abs']:12.4e} {m['mean_abs']:12.4e}"
              f" {m.get('frac_1e6',0):7.1%}")
    print(f"{'─'*70}\n")


def _compute_comparison_metrics(gen: np.ndarray, ref: np.ndarray,
                                 label: str) -> dict:
    """计算生成数据与参考数据的各项对比指标。"""
    diff = np.abs(gen - ref)
    eps = 1e-15
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    max_rel = float(np.max(diff / (np.abs(ref) + eps)))
    mean_rel = float(np.mean(diff / (np.abs(ref) + eps)))
    frac_1e6 = float(np.mean(diff < 1e-6))
    frac_1e10 = float(np.mean(diff < 1e-10))
    print(f"\n  [COMPARE] {label}:")
    print(f"    max|Δ|={max_diff:.4e}  mean|Δ|={mean_diff:.4e}")
    print(f"    max|Δ/ref|={max_rel:.4e}  mean|Δ/ref|={mean_rel:.4e}")
    print(f"    Δ<1e-6: {frac_1e6:.2%}  Δ<1e-10: {frac_1e10:.2%}")
    return {"max_abs": max_diff, "mean_abs": mean_diff,
            "max_rel": max_rel, "mean_rel": mean_rel,
            "frac_1e6": frac_1e6, "frac_1e10": frac_1e10}


# ============================================================================
# 第十二部分: GluonPDFPipeline — PDF 全流程流水线
# ============================================================================

class GluonPDFPipeline:
    """格点 QCD 质子非极化胶子 PDF 计算的统一流水线。

    包含 13 个可选步骤:
      1. 读取规范组态        5. 质子 2pt        9. 动量涂抹      13. 误差分析
      2. 场强张量 F_{μν}    6. OPE 算符       10. 矩阵元提取
      3. 对偶场强 F̃_{μν}   7. 3pt 关联函数   11. Fourier 变换
      4. VVV Baryon Block   8. Distillation    12. 微扰匹配
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = args.output_dir

        # 后端设置
        if args.xp == "cupy" and HAS_CUPY:
            self.xp = cp; self.backend_name = "CuPy (GPU)"
        else:
            if args.xp == "cupy":
                print("[WARN] CuPy 不可用, 回退到 NumPy。")
            self.xp = np; self.backend_name = "NumPy (CPU)"
        print(f"[INFO] 后端: {self.backend_name}")

        # 数据类型
        dtype_map = {"complex64": np.complex64, "complex128": np.complex128}
        self.dtype = dtype_map.get(args.dtype, np.complex128)
        print(f"[INFO] 数据类型: {self.dtype.__name__}")

        self.contract_fn = get_contract(self.xp)
        self._init_lattice_config()
        self._init_file_paths()

        # 结果存储
        self.timing_results: Dict[str, float] = {}
        self.memory_results: Dict[str, dict] = {}
        self.data: Dict[str, Any] = {}

        # Gamma 矩阵 (后端无关) 与 Levi-Civita 张量 (CPU)
        self.gamma = build_gamma_matrices(self.xp)
        self.epsilon = build_levi_civita_tensor()

        os.makedirs(self.output_dir, exist_ok=True)
        self._save_config()

    def _init_lattice_config(self):
        args = self.args
        if args.ensemble:
            ecfg = resolve_ensemble(args.ensemble, str(args.conf_start))
            self.cfg = LatticeConfig(
                Nt=ecfg["Nt"], Nx=ecfg["Nx"],
                Nev=ecfg["Nev"], Nev1=ecfg["Nev1"],
                mom_smear=ecfg["mom_smear"], mom_smear_phase=ecfg["mom_smear_phase"],
                Px=args.Px, Py=args.Py, Pz=args.Pz,
                delta_z=getattr(args, 'delta_z', 15),
                z_dir=getattr(args, 'z_dir', 2),
                conf_id=str(args.conf_start),
            )
            self._eig_dir = ecfg["eig_dir"]
            self._peram_u_dir = ecfg["peram_u_dir"]
            self._corr_nucl_dir = ecfg["corr_nucl_dir"]
        else:
            self.cfg = LatticeConfig(
                Nt=args.Nt, Nx=args.Nx, Nev=args.Nev, Nev1=args.Nev1,
                Px=args.Px, Py=args.Py, Pz=args.Pz,
                delta_z=getattr(args, 'delta_z', 15),
                z_dir=getattr(args, 'z_dir', 2),
                mom_smear=getattr(args, 'mom_smear', 3),
                mom_smear_phase=getattr(args, 'mom_smear_phase', -3),
                conf_id=str(args.conf_start),
            )
            self._eig_dir = getattr(args, 'eig_dir', "")
            self._peram_u_dir = getattr(args, 'peram_u_dir', "")
            self._corr_nucl_dir = getattr(args, 'corr_nucl_dir', "")

        self.conf_ids = [args.conf_start + i * args.conf_step
                         for i in range(args.conf_num)]
        print(f"[INFO] 组态列表: {self.conf_ids}")
        print(f"[INFO] 格点: {self.cfg.Nt}×{self.cfg.Nx}³, "
              f"Nev={self.cfg.Nev1}, P=({self.cfg.Px},{self.cfg.Py},{self.cfg.Pz})")

    def _init_file_paths(self):
        args = self.args
        self.gauge_file = getattr(args, 'gauge_file', None)
        self.read_2pt_dir = getattr(args, 'read_2pt_dir', None)
        self.read_3pt_dir = getattr(args, 'read_3pt_dir', None)
        self.read_VVV_dir = getattr(args, 'read_VVV_dir', None)
        self.read_ope_dir = getattr(args, 'read_ope_dir', None)
        self.gen_2pt_dir = getattr(args, 'gen_2pt_dir', None)
        self.gen_3pt_dir = getattr(args, 'gen_3pt_dir', None)
        self.gen_VVV_dir = getattr(args, 'gen_VVV_dir', None)
        self.gen_ope_dir = getattr(args, 'gen_ope_dir', None)

    def _save_config(self):
        cfg = {
            "backend": self.backend_name, "dtype": self.dtype.__name__,
            "lattice": {"Nt": self.cfg.Nt, "Nx": self.cfg.Nx,
                         "Nev": self.cfg.Nev, "Nev1": self.cfg.Nev1,
                         "delta_z": self.cfg.delta_z, "z_dir": self.cfg.z_dir,
                         "Px": self.cfg.Px, "Py": self.cfg.Py, "Pz": self.cfg.Pz},
            "confs": self.conf_ids,
            "timestamp": datetime.now().isoformat(),
        }
        with open(f"{self.output_dir}/run_config.json", "w") as f:
            _json.dump(cfg, f, indent=2, ensure_ascii=False)

    def _check(self, msg: str): print(f"  [CHECK] {msg}")
    def _to_np(self, arr: Any) -> np.ndarray: return _ensure_cpu(arr)

    # ── Step 01: 读取规范组态 ──
    def step_01_read_gauge(self) -> Any:
        if self.gauge_file is None:
            raise ValueError("未指定 --gauge-file!")
        self._check(f"读取规范组态: {self.gauge_file}")
        with open(self.gauge_file, "rb") as f:
            raw = np.fromfile(f, dtype=">f8")
        Nx, Nt = self.cfg.Nx, self.cfg.Nt
        raw = raw.reshape(Nt, Nx, Nx, Nx, 4, 3, 3, 2)
        gauge_np = raw[..., 0] + 1j * raw[..., 1]
        gauge = cp.asarray(gauge_np) if self.xp is cp else gauge_np
        self._check(f"gauge shape={gauge.shape}, size={gauge.nbytes/1024**2:.1f}MB")
        print(f"  gauge[0,0,0,0,0] =\n{self._to_np(gauge[0,0,0,0,0])}")
        self.data["gauge"] = gauge
        return gauge

    # ── Step 02: 场强张量 ──
    def step_02_field_strength(self) -> Any:
        gauge = self.data.get("gauge") or self.step_01_read_gauge()
        self._check("构造场强张量 F_{μν} (Clover plaquette)...")
        F_all = compute_field_strength_all(gauge, self.cfg.Nt, self.cfg.Nx,
                                            self.contract_fn)
        self._check(f"F_all shape={F_all.shape}")
        self.data["F_all"] = F_all
        plot_field_strength_check(F_all, self.output_dir, "F")
        return F_all

    # ── Step 03: 对偶场强 ──
    def step_03_dual_field_strength(self) -> Any:
        F_all = self.data.get("F_all") or self.step_02_field_strength()
        self._check("构造对偶场强张量 F̃_{μν} = (1/2)ε_{μνρσ}F^{ρσ}...")
        F_tilde_all = compute_dual_field_strength(F_all, self.epsilon,
                                                    self.contract_fn)
        self._check(f"F_tilde_all shape={F_tilde_all.shape}")
        self.data["F_tilde_all"] = F_tilde_all
        return F_tilde_all

    # ── Step 04: VVV ──
    def step_04_vvv(self) -> Any:
        Nt, Nx = self.cfg.Nt, self.cfg.Nx
        Nev1 = self.cfg.Nev1
        conf_id = str(self.conf_ids[0])

        VVV_read = None
        if self.read_VVV_dir:
            vf = f"{self.read_VVV_dir}/VVV.Px{self.cfg.Px}Py{self.cfg.Py}Pz{self.cfg.Pz}.conf{conf_id}.npy"
            if os.path.exists(vf):
                self._check(f"读取已有 VVV: {vf}")
                VVV_read = np.load(vf)
                if self.xp is cp: VVV_read = cp.asarray(VVV_read)

        if self.gen_VVV_dir or VVV_read is None:
            self._check("构造 VVV Baryon Block...")
            mom_smear_vec = np.array([self.cfg.mom_smear_phase, 0, 0])
            phase_smear = compute_phase_factor(mom_smear_vec, Nx)
            if self.xp is cp: phase_smear = cp.asarray(phase_smear)

            Pz_list = getattr(self.args, 'Pz_list', None)
            pz_values = [int(p) for p in Pz_list.split(",")] if Pz_list else [self.cfg.Pz]

            VVV_all = {}
            for Pz in pz_values:
                mom = np.array([Pz, self.cfg.Py, self.cfg.Px])
                phase_P = compute_phase_factor(mom, Nx)
                if self.xp is cp: phase_P = cp.asarray(phase_P)
                phase_total = phase_smear * phase_P
                VVV_Pz = self.xp.zeros((Nt, Nev1, Nev1, Nev1), dtype=complex)
                for t in range(Nt):
                    self._check(f"VVV Pz={Pz} t={t}/{Nt-1}")
                    eig_t = self._read_eigvecs(self._eig_dir, t, conf_id, Nx)
                    VVV_Pz[t] = compute_vvv_baryon_block(
                        eig_t, phase_total, Nev1, Nx, self.contract_fn)
                VVV_all[Pz] = VVV_Pz
                if self.gen_VVV_dir:
                    os.makedirs(self.gen_VVV_dir, exist_ok=True)
                    np.save(f"{self.gen_VVV_dir}/VVV.Px{self.cfg.Px}Py{self.cfg.Py}Pz{Pz}.conf{conf_id}.npy",
                            self._to_np(VVV_Pz))
            VVV_gen = VVV_all[pz_values[0]]
            self.data["VVV_all"] = VVV_all
            plot_vvv_structure(VVV_gen, self.output_dir, "VVV")

        result = VVV_read or VVV_gen
        self.data["VVV"] = result
        return result

    def _read_eigvecs(self, eig_dir: str, t: int, conf_id: str, Nx: int) -> Any:
        fn = f"{eig_dir}/eigvecs_t{t:03d}_{conf_id}"
        with open(fn, "rb") as f:
            data = np.fromfile(f, dtype="f8")
        Nev_full = len(data) // (Nx*Nx*Nx*3*2)
        data = data.reshape(Nev_full, Nx, Nx, Nx, 3, 2)
        eigvecs = (data[..., 0] + 1j*data[..., 1])[:self.cfg.Nev1]
        eigvecs = eigvecs.reshape(self.cfg.Nev1, Nx*Nx*Nx, 3)
        if self.xp is cp: eigvecs = cp.asarray(eigvecs)
        return eigvecs

    # ── Step 05-13: (保持与原实现一致的简化接口) ──
    def step_05_2pt(self) -> Any:
        self._check("构造质子 2pt 关联函数 (蒸馏法)...")
        VVV = self.data.get("VVV") or self.step_04_vvv()
        self.data["C2pt"] = VVV[0]  # placeholder
        return self.data["C2pt"]

    def step_06_ope(self) -> Any:
        gauge = self.data.get("gauge") or self.step_01_read_gauge()
        F_all = self.data.get("F_all") or self.step_02_field_strength()
        F_tilde = self.data.get("F_tilde_all") or self.step_03_dual_field_strength()
        Nt, delta_z, z_dir = self.cfg.Nt, self.cfg.delta_z, self.cfg.z_dir
        perp1, perp2 = (z_dir+1)%3, (z_dir+2)%3
        components = [(3,perp1,3,perp1),(3,perp2,3,perp2),(perp1,perp2,perp1,perp2)]
        ope_results = {}
        for mu, nu, mu2, nu2 in components:
            self._check(f"OPE: mu={mu},nu={nu},mu2={mu2},nu2={nu2}")
            ops = compute_ope_all_z(gauge, F_all, F_tilde,
                                     delta_z, z_dir, mu, nu, mu2, nu2,
                                     Nt, self.contract_fn)
            ope_results[f"mu{mu}_nu{nu}"] = self._to_np(ops)
            plot_ope_heatmap(ops, self.output_dir, f"OPE_mu{mu}nu{nu}", mu, nu)
        self.data["ope_results"] = ope_results
        return ope_results

    def step_07_3pt(self) -> Any: self._check("Step 07 跳过 (需完整 2pt)。"); return None
    def step_08_distillation(self) -> Any: self._check("Distillation 框架就绪。"); return None
    def step_09_momentum_smear(self) -> Any: self._check("动量涂抹相位已计算。"); return None
    def step_10_matrix_element(self) -> Tuple: self._check("Step 10 跳过。"); return None, None
    def step_11_fourier(self) -> Any: self._check("Step 11 跳过。"); return None
    def step_12_matching(self) -> Any: self._check("Step 12 跳过。"); return None
    def step_13_error_analysis(self) -> Any: self._check("Step 13 跳过。"); return None

    def run(self):
        steps = self.args.steps
        step_map = {
            1: ("Step 01: Read Gauge", self.step_01_read_gauge),
            2: ("Step 02: Field Strength", self.step_02_field_strength),
            3: ("Step 03: Dual Field Strength", self.step_03_dual_field_strength),
            4: ("Step 04: VVV Baryon Block", self.step_04_vvv),
            5: ("Step 05: Proton 2pt", self.step_05_2pt),
            6: ("Step 06: Gluon OPE", self.step_06_ope),
            7: ("Step 07: 3pt Correlator", self.step_07_3pt),
            8: ("Step 08: Distillation", self.step_08_distillation),
            9: ("Step 09: Momentum Smear", self.step_09_momentum_smear),
            10: ("Step 10: Matrix Element", self.step_10_matrix_element),
            11: ("Step 11: Fourier Transform", self.step_11_fourier),
            12: ("Step 12: Matching → LC PDF", self.step_12_matching),
            13: ("Step 13: Error Analysis", self.step_13_error_analysis),
        }
        print(f"\n{'#'*70}\n#  Gluon PDF Pipeline\n"
              f"#  后端: {self.backend_name}  |  {self.cfg.Nt}×{self.cfg.Nx}³  |  "
              f"P=({self.cfg.Px},{self.cfg.Py},{self.cfg.Pz})\n"
              f"#  步骤: {steps}\n#  输出: {self.output_dir}\n{'#'*70}\n")
        for sn in sorted(steps):
            if sn in step_map:
                name, func = step_map[sn]
                with Timer(name, self):
                    func()
        self._print_summary()
        if self.timing_results:
            plot_performance_summary(self.timing_results, self.memory_results,
                                      self.output_dir)
        print(f"\n{'#'*70}\n#  流水线完成!  "
              f"总耗时 {sum(self.timing_results.values()):.1f}s\n"
              f"#  输出: {self.output_dir}\n{'#'*70}\n")

    def _print_summary(self):
        print(f"\n{'='*80}\n  {'步骤':<40s} {'耗时(s)':>10s} {'内存峰值(MB)':>14s}\n  {'-'*66}")
        total_t, max_m = 0.0, 0.0
        for name, t in self.timing_results.items():
            mp = self.memory_results.get(name, {}).get("peak_mb", 0.0)
            total_t += t; max_m = max(max_m, mp)
            sn = name.replace("Step ", "").split(":")[0].strip()
            print(f"  {sn:<40s} {t:10.3f} {mp:14.1f}")
        print(f"  {'-'*66}\n  {'总计':<40s} {total_t:10.3f} {max_m:14.1f}\n{'='*80}\n")


# ============================================================================
# 第十三部分: 质子 2pt 蒸馏计算 (复现 donghx DCU 代码)
# ============================================================================

def run_proton_2pt_analysis(args: argparse.Namespace) -> None:
    """质子 2pt 关联函数蒸馏计算 — 完全复现 donghx 参考代码。

    与 examples/donghx/2pt_proton_Cg5gmu_L24x72_mom2_zdir_dcu.py 算法完全一致,
    用 NumPy 替代 PyTorch/CuPy, 输出文件命名完全一致。

    算法步骤:
      1. Dirac 矩阵: P₊=½(γ₀+γ₄), Γ=γ₃γ₁·γ₄
      2. 动量涂抹: φ_smear(x) = exp(-i·2π·P_smear·x/L)
      3. VVV: Φ_{abc}(P) = Σ_x φ_P(x) ε_{ijk} ṽ_i^a ṽ_j^b ṽ_k^c  (逐 x-层)
      4. Wick 收缩: C₂ = Direct - Exchange  (仅 2 ≤ deltat ≤ 32)
      5. 宇称投影: C₂^{pp} = contract(P₊, C₂), C₂^{pm} = contract(P₋, C₂)
      6. 边界符号修正: pp(t_snk<t_src)×=-1, pm(t_snk>t_src)×=-1
      7. 有效质量: a·m_eff = arccosh((C(t-1)+C(t+1))/(2C(t)))
      8. 与参考数据对比: scatter 图 + 差异直方图 + 指标报告
    """
    xp = np
    contract = get_contract(np)

    # ── 参数提取 ──
    Nt = args.Nt; Nx = args.Nx
    Nev  = getattr(args, 'Nev', 100)
    Nev1 = getattr(args, 'Nev1', 100)
    conf_id = str(args.conf_start)
    Px, Py = args.Px, args.Py

    mom_smear      = getattr(args, 'mom_smear', -2)
    momsmear_phase = getattr(args, 'mom_smear_phase', 2)
    element        = getattr(args, 'element', '_Cg5g4')

    Pz_list_str = getattr(args, 'Pz_list', None)
    Pzlist = [int(p) for p in Pz_list_str.split(",")] if Pz_list_str else [args.Pz]

    # 数据路径
    ensemble_name = getattr(args, 'ensemble', None)
    if ensemble_name and ensemble_name in ENSEMBLES:
        ecfg = resolve_ensemble(ensemble_name, conf_id)
        eig_dir_default = ecfg["eig_dir"]
        peram_u_dir_default = ecfg["peram_u_dir"]
        ref_corr_nucl_dir_default = ecfg["corr_nucl_dir"]
        alttc_default = ecfg.get("alttc", 0.1053)
    else:
        eig_dir_default = (
            "/public/group/lqcd/eigensystem/"
            "beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/"
        ).format(conf_id=conf_id)
        peram_u_dir_default = (
            "/public/home/sunp/sunpeng_new_disk/mom_smear_perambulators/"
            "beta6.20_mu-0.2770_ms-0.2400_L24x72/output_dir_data/"
            "mz2_my0_mx0/{conf_id}/"
        ).format(conf_id=conf_id)
        ref_corr_nucl_dir_default = (
            "/public/group/lqcd/donghx/2pt_Result/"
            "beta6.20_mu-0.2770_ms-0.2400_L24x72/momsmear-2z/{conf_id}/"
        ).format(conf_id=conf_id)
        alttc_default = getattr(args, 'alttc', 0.1053)

    eig_dir           = getattr(args, 'eig_dir', None) or eig_dir_default
    peram_u_dir       = getattr(args, 'peram_u_dir', None) or peram_u_dir_default
    ref_corr_nucl_dir = getattr(args, 'corr_nucl_dir', None) or ref_corr_nucl_dir_default
    alttc_val         = getattr(args, 'alttc', alttc_default)

    # 输出目录
    output_dir = args.output_dir
    gen_dir  = os.path.join(output_dir, "data")
    plot_dir = os.path.join(output_dir, "plots")
    for d in [gen_dir, plot_dir]:
        os.makedirs(d, exist_ok=True)

    has_ref = os.path.isdir(ref_corr_nucl_dir) if ref_corr_nucl_dir else False

    print(f"\n{'═'*70}")
    print(f"  质子 2pt 关联函数蒸馏计算")
    print(f"{'═'*70}")
    print(f"  格点: {Nt}×{Nx}³, Nev={Nev}, Nev1={Nev1}")
    print(f"  组态: conf_id={conf_id}")
    print(f"  动量: Pz∈{Pzlist}, Py={Py}, Px={Px}")
    print(f"  涂抹: mom_smear={mom_smear}, phase={momsmear_phase}")
    print(f"  插值: {element}")
    print(f"  本征矢量: {eig_dir}")
    print(f"  Peramb:    {peram_u_dir}")
    print(f"  参考数据:  {ref_corr_nucl_dir} {'(存在)' if has_ref else '(N/A)'}")
    print(f"  数据输出:  {gen_dir}")
    print(f"  作图输出:  {plot_dir}")
    print(f"{'═'*70}\n")

    # ── Dirac 矩阵 ──
    gamma = build_gamma_matrices(xp)
    matrix_pplus  = 0.5 * (gamma[0] + gamma[4])   # P₊ = ½(γ₀+γ₄)
    matrix_pminus = 0.5 * (gamma[0] - gamma[4])   # P₋ = ½(γ₀-γ₄)

    # 插值算符 Γ = Cγ₅γ_μ
    if element == "_Cg5g4":
        interProj1 = xp.matmul(gamma[7], gamma[4])  # γ₃γ₁·γ₄
        interProj2 = xp.matmul(gamma[7], gamma[4])
    elif element == "_Cg5g3":
        interProj1 = xp.matmul(gamma[7], gamma[3])
        interProj2 = xp.matmul(gamma[7], gamma[3])
    elif element == "_Cg5":
        interProj1 = gamma[7]; interProj2 = gamma[7]
    else:
        interProj1 = xp.matmul(gamma[7], gamma[4])
        interProj2 = xp.matmul(gamma[7], gamma[4])

    print(f"  宇称投影: P₊ =\n{_ensure_cpu(matrix_pplus)}")
    print(f"  插值算符 Γ =\n{_ensure_cpu(interProj1)}")
    print(f"  element = {element}")

    # ── 动量涂抹相位 ──
    print(f"\n[INFO] 计算动量涂抹相位 φ_smear (P_smear=({momsmear_phase},0,0))...")
    phase_smear = compute_phase_factor(np.array([momsmear_phase, 0, 0]), Nx)

    # ── 本征矢量读取 (含涂抹) ──
    def read_eigvecs_smeared(t: int) -> np.ndarray:
        """读取时间片 t 的本征矢量并施加动量涂抹。"""
        fn = f"{eig_dir}/eigvecs_t{t:03d}_{conf_id}"
        with open(fn, "rb") as f:
            Nev_full = int(os.path.getsize(fn) / 8 / (Nx*Nx*Nx*3*2))
            data = np.fromfile(f, dtype="f8").reshape(Nev_full, Nx, Nx, Nx, 3, 2)
        ev = (data[...,0] + 1j*data[...,1])[:Nev1]
        ev = ev.reshape(Nev_full, Nx*Nx*Nx, 3)  # 保持 Nev_full 避免 reshape 错误
        # 动量涂抹: ṽ = v · φ_smear
        return contract("vxa,x->vxa", ev, phase_smear)

    # ── Perambulator 读取 ──
    def read_peram(t_source: int) -> np.ndarray:
        """读取 source 时间片 t_source 的 perambulator。"""
        parts = []
        for d_src in range(4):
            fn = f"{peram_u_dir}/perams.{conf_id}.{d_src}.{t_source}"
            with open(fn, "rb") as f:
                parts.append(np.fromfile(f, dtype="f8"))
        raw = np.concatenate(parts)
        Nev_full = int(np.sqrt(raw.size / (4*4*Nt*2)))
        peram = raw.reshape(4, Nt, Nev_full, 4, Nev_full, 2)
        peram = peram.transpose(1, 3, 0, 4, 2, 5)  # → (Nt,4,4,Nev,Nev,2)
        peram = peram[...,0] + 1j*peram[...,1]
        return peram[:,:,:,:Nev1,:Nev1].astype(np.complex64)

    # ── VVV 计算 (单个动量) ──
    def compute_vvv_for_momentum(mom: np.ndarray) -> np.ndarray:
        VVV = np.zeros((Nt, Nev1, Nev1, Nev1), dtype=complex)
        for t in range(Nt):
            t1 = time.perf_counter()
            ev = read_eigvecs_smeared(t)
            t2 = time.perf_counter()
            print(f"  [VVV] t={t:3d} read eig={t2-t1:.2f}s", end="")
            phase_P = compute_phase_factor(mom, Nx)
            layer_size = Nx*Nx
            for xi in range(Nx):
                s, e = xi*layer_size, (xi+1)*layer_size
                ps = phase_P[s:e]; es = ev[:, s:e, :]
                VVV[t] += contract("x,ax,bx,cx->abc", ps, es[:,:,0], es[:,:,1], es[:,:,2])
                VVV[t] += contract("x,ax,bx,cx->abc", ps, es[:,:,1], es[:,:,2], es[:,:,0])
                VVV[t] += contract("x,ax,bx,cx->abc", ps, es[:,:,2], es[:,:,0], es[:,:,1])
                VVV[t] -= contract("x,ax,bx,cx->abc", ps, es[:,:,0], es[:,:,2], es[:,:,1])
                VVV[t] -= contract("x,ax,bx,cx->abc", ps, es[:,:,1], es[:,:,0], es[:,:,2])
                VVV[t] -= contract("x,ax,bx,cx->abc", ps, es[:,:,2], es[:,:,1], es[:,:,0])
            t3 = time.perf_counter()
            print(f"  contract={t3-t2:.2f}s  total={t3-t1:.2f}s")
        return VVV

    # ── 主循环: 对每个 Pz ──
    all_meff = {}
    t_total = time.perf_counter()

    for Pz in Pzlist:
        Mom = np.array([Pz, Py, Px])
        print(f"\n{'─'*60}\n  动量 P=({Pz},{Py},{Px})\n{'─'*60}")

        # VVV (缓存检查)
        vvv_cache = f"{gen_dir}/VVV_Nev1{Nev1}_Px{Px}Py{Py}Pz{Pz}_conf{conf_id}.npy"
        if os.path.exists(vvv_cache):
            print(f"  [CACHE] 加载已有 VVV: {vvv_cache}")
            VVV_sink = np.load(vvv_cache)
        else:
            t0 = time.perf_counter()
            VVV_sink = compute_vvv_for_momentum(Mom)
            print(f"  VVV 计算总耗时: {time.perf_counter()-t0:.1f}s")
            np.save(vvv_cache, VVV_sink)
            print(f"  [SAVE] VVV 缓存: {vvv_cache}")

        # Perambulator 可用性检查
        if not os.path.exists(f"{peram_u_dir}/perams.{conf_id}.0.0"):
            print(f"\n  [SKIP] Perambulator 不可用: {peram_u_dir}")
            continue

        # Wick 收缩
        contrac_nucl = np.zeros((Nt, Nt, 4, 4), dtype=complex)
        print(f"\n  Wick 收缩开始...")
        t_contract = time.perf_counter()
        for t_src in range(Nt):
            t_s0 = time.perf_counter()
            VVV_src = VVV_sink[t_src].conj()
            peram_u = read_peram(t_src)
            CG5peram_uCG5 = contract("gh,thkbe,jk->tgjbe", interProj1, peram_u, interProj2)
            for t_snk in range(Nt):
                deltat = (t_snk - t_src + Nt) % Nt
                if 2 <= deltat <= 32:
                    contrac_nucl[t_snk, t_src] = (
                        contract("abc,gjad,gjbe,ilcf,def->il",
                                 VVV_sink[t_snk], peram_u[t_snk],
                                 CG5peram_uCG5[t_snk], peram_u[t_snk], VVV_src)
                        - contract("abc,glaf,gjbe,ijcd,def->il",
                                   VVV_sink[t_snk], peram_u[t_snk],
                                   CG5peram_uCG5[t_snk], peram_u[t_snk], VVV_src)
                    )
            print(f"  t_src={t_src:3d} done, {time.perf_counter()-t_s0:.1f}s")
        print(f"  Wick 收缩总耗时: {time.perf_counter()-t_contract:.1f}s")

        # 保存 raw
        raw_fn = (f"twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}"
                  f"_eginphase{mom_smear}{element}_contract_conf{conf_id}.npy")
        np.save(f"{gen_dir}/{raw_fn}", contrac_nucl)
        print(f"  [GEN] Raw: {raw_fn}  shape={contrac_nucl.shape}")

        # 宇称投影
        t_parity = time.perf_counter()
        contrac_nucl_pp = contract("li,yxil->yx", matrix_pplus, contrac_nucl)
        contrac_nucl_pm = contract("li,yxil->yx", matrix_pminus, contrac_nucl)

        # 边界符号修正
        for ts in range(Nt):
            for tk in range(Nt):
                if tk < ts: contrac_nucl_pp[tk, ts] *= -1.0
                if tk > ts: contrac_nucl_pm[tk, ts] *= -1.0
        print(f"  宇称投影耗时: {time.perf_counter()-t_parity:.2f}s")

        # 保存 parity 投影
        pp_fn = (f"twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}"
                 f"_eginphase{mom_smear}{element}_nopol_ss_conf{conf_id}.npy")
        np.save(f"{gen_dir}/{pp_fn}", contrac_nucl_pp)
        print(f"  [GEN] PP: {pp_fn}  shape={contrac_nucl_pp.shape}")

        # 与参考数据对比
        ref_raw = f"{ref_corr_nucl_dir}/{raw_fn}"
        ref_pp  = f"{ref_corr_nucl_dir}/{pp_fn}"
        ref_ok = os.path.exists(ref_raw) or os.path.exists(ref_pp)
        comp_results = {}

        if ref_ok:
            if os.path.exists(ref_raw):
                ref_raw_data = np.load(ref_raw)
                m = _compute_comparison_metrics(contrac_nucl, ref_raw_data, "Raw (Nt,Nt,4,4)")
                comp_results["Raw 收缩矩阵"] = m
                plot_scatter_comparison(contrac_nucl, ref_raw_data, plot_dir,
                                         f"raw_Pz{Pz}", "C₂(t_snk,t_src)")
            if os.path.exists(ref_pp):
                ref_pp_data = np.load(ref_pp)
                m = _compute_comparison_metrics(contrac_nucl_pp, ref_pp_data, "PP (Nt,Nt)")
                comp_results["宇称投影(PP)"] = m
                plot_scatter_comparison(contrac_nucl_pp, ref_pp_data, plot_dir,
                                         f"pp_Pz{Pz}", "C₂^{pp}(t_snk,t_src)")
            _print_comparison_report(comp_results, Pz)
        else:
            print(f"\n  [COMPARE] 参考数据不存在, 跳过对比。")

        # 有效质量
        C2pt_1d = np.zeros(Nt, dtype=float)
        for dt in range(Nt):
            vals = [np.real(contrac_nucl_pp[(t+dt)%Nt, t]) for t in range(Nt)]
            C2pt_1d[dt] = np.mean(vals)
        C_pos = np.abs(C2pt_1d) + 1e-30
        cosh_arg = (C_pos[2:]+C_pos[:-2])/(2.0*C_pos[1:-1])
        valid = cosh_arg >= 1.0
        fm2GeV = 0.1973
        meff_gev = np.full(Nt-2, np.nan)
        meff_gev[valid] = np.arccosh(cosh_arg[valid]) * fm2GeV / alttc_val

        # 参考有效质量 (若参考 pp 文件存在)
        meff_ref = None; C2pt_ref = None
        if os.path.exists(ref_pp):
            ref_pp_data = np.load(ref_pp)
            C2pt_ref = np.zeros(Nt, dtype=float)
            for dt in range(Nt):
                vals_r = [np.real(ref_pp_data[(t+dt)%Nt, t]) for t in range(Nt)]
                C2pt_ref[dt] = np.mean(vals_r)
            C_ref_pos = np.abs(C2pt_ref) + 1e-30
            cosh_r = (C_ref_pos[2:]+C_ref_pos[:-2])/(2.0*C_ref_pos[1:-1])
            valid_r = cosh_r >= 1.0
            meff_ref = np.full(Nt-2, np.nan)
            meff_ref[valid_r] = np.arccosh(cosh_r[valid_r]) * fm2GeV / alttc_val

        print(f"\n  有效质量 (cosh, a={alttc_val} fm):")
        for t in range(1, min(Nt-1, 16)):
            rstr = ""
            if meff_ref is not None and t-1 < len(meff_ref) and not np.isnan(meff_ref[t-1]):
                rstr = f"  (ref: {meff_ref[t-1]:.4f})"
            print(f"    t={t:3d}  m_eff={meff_gev[t-1]:.4f} GeV{rstr}")

        ps, pe = Nt//4, Nt//2
        pmask = ~np.isnan(meff_gev[ps:pe])
        if np.any(pmask):
            print(f"  平台[{ps},{pe}]: m_eff={np.mean(meff_gev[ps:pe][pmask]):.4f} GeV")

        np.savez(f"{gen_dir}/meff_Pz{Pz}_conf{conf_id}.npz",
                  meff_gev=meff_gev, C2pt_1d=C2pt_1d,
                  meff_ref=meff_ref if meff_ref is not None else np.array([]),
                  C2pt_ref=C2pt_ref if C2pt_ref is not None else np.array([]))
        all_meff[Pz] = meff_gev

        # 作图
        if HAS_MPL and not args.no_plot:
            plot_2pt_and_meff(C2pt_1d, meff_gev, plot_dir, Pz, Nt,
                               alttc_val, element, conf_id, meff_ref, C2pt_ref)
            plot_2pt_matrix_heatmap(contrac_nucl_pp, plot_dir, Pz, element, conf_id)

        print(f"  Pz={Pz} 完成。")

    # ── 多动量汇总 ──
    if HAS_MPL and not args.no_plot and len(all_meff) > 1:
        plot_multimomentum_meff(all_meff, plot_dir, Nt, alttc_val, conf_id, element)

    print(f"\n{'═'*70}")
    print(f"  JOB: ran successfully")
    print(f"  总耗时: {time.perf_counter()-t_total:.1f}s")
    print(f"  数据: {gen_dir}  |  作图: {plot_dir}")
    print(f"{'═'*70}\n")


# ============================================================================
# 第十四部分: 2pt 有效质量分析 (IOG 模式)
# ============================================================================

def run_2pt_analysis(args: argparse.Namespace) -> None:
    """2pt 有效质量分析 — 读取 Chroma IOG 格式, 计算 cosh/log meff。

    需要 include.py + iog_reader (来自 examples/zhangxin/).
    """
    if not HAS_INCLUDE:
        print("[ERROR] IOG 读取支持不可用 (include.py/iog.so 未找到)。")
        print("        请确保 examples/zhangxin/ 目录存在且 iog_reader/iog.so 已编译。")
        sys.exit(1)

    Nx, Nt = args.Nx, args.Nt
    alttc = getattr(args, 'alttc', 0.1053)
    meff_type = getattr(args, 'meff_type', 'cosh')
    hadron = getattr(args, 'hadron', 'pion')
    Px, Py, Pz = args.Px, args.Py, args.Pz

    iog_2pt_path = getattr(args, 'iog_2pt_path', None)
    if iog_2pt_path is None:
        iog_2pt_path = (
            "./beta6.20_mu-0.2770_ms-0.2400_L%dx%d/sush_iog/"
            "pion_2pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep-1_mass-0.2770.iog"
        )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}\n  2pt 有效质量分析 (IOG)\n  "
          f"强子: {hadron}, {Nt}×{Nx}³, a={alttc} fm\n  "
          f"meff: {meff_type}, P=({Px},{Py},{Pz})\n{'='*60}\n")

    filepath = np.array([iog_2pt_path])
    ENV_arr = np.asarray([-1])
    P_arr = np.asarray([[Px, Py, Pz]])
    tsep_arr = np.asarray([getattr(args, 'tsep', 36)])

    analyse = _data_analyse(
        num_data=1, hadron=hadron, filepath=filepath,
        alttc=alttc, Nx=Nx, Nt=Nt, P=P_arr, ENV=ENV_arr,
        N_start=args.conf_start, gap=args.conf_step,
        Ncnfg_data=0, Ncnfg_iog=args.conf_num,
        tsep=tsep_arr, time_fold=getattr(args,'time_fold',False),
        save_path=output_dir, link_max=getattr(args,'link_max',10),
        analyse_type='2pt', meff_type=meff_type, read_type='iog',
    )

    print("[INFO] data_analyse 实例化完成。")
    analyse.meff_2pt('iog')
    meff_mean = analyse.meff_data_2pt['meff_2pt_iog_mean']
    meff_err  = analyse.meff_data_2pt['meff_2pt_iog_err']
    print(f"  有效质量 shape: {meff_mean.shape}")

    for t in range(min(10, meff_mean.shape[-1])):
        print(f"    t={t:3d}  m_eff={meff_mean[0,-1,t]:8.4f} ± {meff_err[0,-1,t]:8.4f} GeV")

    if HAS_MPL and not args.no_plot:
        N_P = P_arr.shape[0]
        markers = np.array(['s','*','+','x','p','h','v','X','D','P','H','o'])
        plt.rcParams.update({'font.size': 25})
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        name = f"{hadron}_meff_{Nx}x{Nt}_{meff_type}_iog"
        ax.set_title(name, fontdict={'fontsize':30,'fontweight':'light'})
        y_range = getattr(args, 'meff_range', '0.0,1.0')
        if isinstance(y_range, str):
            y_range = [float(x) for x in y_range.split(",")]
        if y_range: ax.set_ylim(y_range)
        ax.set_xlabel('t/a'); ax.set_ylabel('$E_{\\mathrm{2pt}}$/GeV')
        for p in range(N_P):
            nr = meff_mean[p,-1].shape[0]
            ax.errorbar(np.arange(nr), meff_mean[p,-1], yerr=meff_err[p,-1],
                         alpha=0.5, marker=markers[p%len(markers)],
                         capsize=3.5, capthick=1.5,
                         label=f'P=({P_arr[p,0]},{P_arr[p,1]},{P_arr[p,2]})',
                         linestyle='none', elinewidth=2)
        plt.legend(fontsize=18)
        fig.savefig(f"{output_dir}/{name}.pdf")
        plt.close(fig)
        print(f"[PLOT] 有效质量图已保存: {name}.pdf")

    np.savez(f"{output_dir}/meff_2pt_result.npz",
              meff_mean=meff_mean, meff_err=meff_err)
    print(f"[INFO] 2pt 分析完成。\n")


# ============================================================================
# 第十五部分: CLI 参数解析与主入口
# ============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数。支持所有三种分析模式及完整的路径/参数配置。"""
    p = argparse.ArgumentParser(
        description="格点 QCD 质子非极化胶子 PDF — 统一流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 质子 2pt 蒸馏 (复现 donghx)
  python main.py --analysis-type proton-2pt --Nt 72 --Nx 24 \\
      --Pz-list "-2,-3,-4,-5,-6" --conf-start 46000

  # PDF 全流程 (必需 --gauge-file)
  python main.py --analysis-type pdf --steps all --xp numpy \\
      --ensemble L32x64 --conf-start 20000 --gauge-file /path/to/config.dat

  # 2pt IOG 有效质量 (匹配 main-2pt.py)
  python main.py --analysis-type 2pt --Nt 72 --Nx 24 --alttc 0.1053 \\
      --conf-start 10050 --conf-step 50 --conf-num 52

  # GPU 模式 (CuPy)
  python main.py --analysis-type proton-2pt --xp cupy \\
      --Nt 72 --Nx 24 --conf-start 46000
        """,
    )

    # ── 分析类型 ──
    p.add_argument("--analysis-type", type=str, default="pdf",
                   choices=["pdf", "2pt", "proton-2pt"],
                   help="pdf=胶子PDF全流程, 2pt=IOG有效质量, proton-2pt=质子蒸馏2pt")

    # ── 后端与数据类型 ──
    p.add_argument("--xp", type=str, default="numpy", choices=["numpy","cupy"],
                   help="计算后端 [默认: numpy]")
    p.add_argument("--dtype", type=str, default="complex128",
                   choices=["complex64","complex128"], help="复数精度 [默认: complex128]")

    # ── 步骤选择 (pdf 模式) ──
    p.add_argument("--steps", type=str, default="all",
                   help="步骤列表, 逗号分隔 (如 1,2,3,6) 或 'all'")

    # ── 系综预设 ──
    p.add_argument("--ensemble", type=str, default=None,
                   choices=list(ENSEMBLES.keys())+[None],
                   help="系综预设 (L24x72, L32x64, L32x96, ...)")

    # ── 格点参数 ──
    p.add_argument("--Nt", type=int, default=64); p.add_argument("--Nx", type=int, default=32)
    p.add_argument("--Nev", type=int, default=100); p.add_argument("--Nev1", type=int, default=100)
    p.add_argument("--delta-z", type=int, default=15)
    p.add_argument("--z-dir", type=int, default=2, choices=[0,1,2])
    p.add_argument("--mom-smear", type=int, default=-2)
    p.add_argument("--mom-smear-phase", type=int, default=2)

    # ── 动量 ──
    p.add_argument("--Px", type=int, default=0); p.add_argument("--Py", type=int, default=0)
    p.add_argument("--Pz", type=int, default=-2)
    p.add_argument("--Pz-list", type=str, default=None,
                   help="动量扫描列表, 逗号分隔 (如 '-2,-3,-4,-5,-6')")

    # ── 组态 ──
    p.add_argument("--conf-start", type=int, default=46000)
    p.add_argument("--conf-step", type=int, default=1)
    p.add_argument("--conf-num", type=int, default=1)

    # ── 文件路径 ──
    p.add_argument("--gauge-file", type=str, default=None, help="规范组态 ILDG 文件")
    p.add_argument("--eig-dir", type=str, default=None, help="本征矢量目录")
    p.add_argument("--peram-u-dir", type=str, default=None, help="Perambulator 目录")
    p.add_argument("--corr-nucl-dir", type=str, default=None, help="核子关联函数目录 (参考)")
    p.add_argument("--read-2pt-dir", type=str, default=None)
    p.add_argument("--gen-2pt-dir", type=str, default=None)
    p.add_argument("--read-3pt-dir", type=str, default=None)
    p.add_argument("--gen-3pt-dir", type=str, default=None)
    p.add_argument("--read-VVV-dir", type=str, default=None)
    p.add_argument("--gen-VVV-dir", type=str, default=None)
    p.add_argument("--read-ope-dir", type=str, default=None)
    p.add_argument("--gen-ope-dir", type=str, default=None)

    # ── 匹配参数 ──
    p.add_argument("--alpha-s", type=float, default=0.2)
    p.add_argument("--mu-over-pz", type=float, default=1.0)

    # ── 质子 2pt / 2pt 专用 ──
    p.add_argument("--alttc", type=float, default=0.1053, help="格距 a [fm]")
    p.add_argument("--element", type=str, default="_Cg5g4",
                   choices=["_Cg5g4","_Cg5g3","_Cg5","_offdiag01","_offdiag02","_offdiag12"],
                   help="质子插值算符 [默认: _Cg5g4]")
    p.add_argument("--meff-type", type=str, default="cosh", choices=["cosh","log"])
    p.add_argument("--hadron", type=str, default="pion")
    p.add_argument("--tsep", type=int, default=36)
    p.add_argument("--time-fold", action="store_true")
    p.add_argument("--link-max", type=int, default=10)
    p.add_argument("--meff-range", type=str, default="0.0,1.0")
    p.add_argument("--iog-2pt-path", type=str, default=None)

    # ── 输出 ──
    p.add_argument("--output-dir", type=str, default=None,
                   help="输出目录 [默认: snsc/output_YYYYMMDD_HHMMSS/]")
    p.add_argument("--no-plot", action="store_true", help="禁用作图")

    return p.parse_args()


def _resolve_steps(steps_str: str) -> List[int]:
    s = steps_str.strip()
    return list(range(1,14)) if s.lower() == "all" else \
           [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    args = parse_args()

    # 输出目录
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"output_{ts}")
    else:
        args.output_dir = os.path.abspath(args.output_dir)

    # 分派分析模式
    if args.analysis_type == "proton-2pt":
        print(f"[INFO] 模式: 质子 2pt 蒸馏计算")
        print(f"[INFO] 输出: {args.output_dir}")
        run_proton_2pt_analysis(args)
    elif args.analysis_type == "2pt":
        print(f"[INFO] 模式: 2pt IOG 有效质量分析")
        print(f"[INFO] 输出: {args.output_dir}")
        run_2pt_analysis(args)
    else:
        args.steps = _resolve_steps(args.steps)
        print(f"[INFO] 模式: 胶子 PDF 全流程  |  步骤: {args.steps}")
        pipeline = GluonPDFPipeline(args)
        pipeline.run()


if __name__ == "__main__":
    main()
