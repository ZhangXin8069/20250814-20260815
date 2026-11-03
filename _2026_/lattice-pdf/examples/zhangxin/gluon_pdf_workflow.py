#!/usr/bin/env python3
"""
Gluon PDF Full Workflow — consolidated from examples/

Computes the unpolarized gluon PDF in the proton using LaMET + distillation.
The disconnected three-point correlation function is factorized into:
  1. Proton 2pt correlator (distillation with momentum smearing)
  2. OPE part (nonlocal gluon operator from field-strength tensor)

Usage:
  python code/gluon_pdf_workflow.py 2pt --ensemble L32x64 --conf-id 20000
  python code/gluon_pdf_workflow.py ope --ensemble L32x64 --conf-id 20000 --gauge-file config.dat
  python code/gluon_pdf_workflow.py pla --ensemble L32x64 --conf-id 20000 --gauge-file config.dat
  python code/gluon_pdf_workflow.py full --ensemble L32x64 --conf-id 20000 --gauge-file config.dat

Backward-compatible stdin params:
  python code/gluon_pdf_workflow.py 2pt --params-file params.txt
  python code/gluon_pdf_workflow.py ope --params-file params.txt

Author: Consolidated from donghx's original code in examples/
"""

# ==============================================================================
# Section 1: Header & Imports
# ==============================================================================

import numpy as np
import os
import sys
import time
import argparse
import fileinput
import math

try:
    from opt_einsum import contract
except ImportError:
    contract = np.einsum
    print("WARNING: opt_einsum not available, falling back to numpy.einsum")

# ==============================================================================
# Section 2: Backend Detection & Setup
# ==============================================================================

HAS_CUPY = False
cp = None
try:
    import cupy as cp
    HAS_CUPY = True
    print("CuPy backend available — using GPU")
except ImportError:
    print("CuPy not available — using NumPy (CPU)")

HAS_MPI = False
comm = None
rank = 0
size = 1
try:
    from mpi4py import MPI as _MPI
    comm = _MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    HAS_MPI = True
except ImportError:
    pass


def get_xp():
    """Return the active array module (cupy or numpy)."""
    return cp if HAS_CUPY else np


def asarray(data, dtype=None):
    """Transfer data to GPU if available, else keep as numpy."""
    if HAS_CUPY:
        if dtype is not None:
            return cp.asarray(data, dtype=dtype)
        return cp.asarray(data)
    if dtype is not None:
        return np.asarray(data, dtype=dtype)
    return np.asarray(data)


def to_numpy(data):
    """Transfer data back to CPU if on GPU."""
    if HAS_CUPY and hasattr(data, 'get'):
        return data.get()
    return np.asarray(data)


def synchronize():
    """Synchronize GPU device if using CuPy."""
    if HAS_CUPY:
        cp.cuda.Device().synchronize()


def free_memory_pool():
    """Free GPU memory pool if using CuPy."""
    if HAS_CUPY:
        cp._default_memory_pool.free_all_blocks()


# ==============================================================================
# Section 3: Ensemble Presets
# ==============================================================================

ENSEMBLES = {
    "L24x72": {
        "Nt": 72, "Nx": 24, "Nev": 100, "Nev1": 100,
        "mom_smear": -2, "momsmear_phase": 2,
        "eig_dir": "/public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/",
        "peram_u_dir": "/public/home/sunp/sunpeng_public/mom_smear_perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/output_dir_data/mz0_my0_mx2/{conf_id}/",
        "corr_nucl_dir": "/public/group/lqcd/donghx/2pt_Result/beta6.20_mu-0.2770_ms-0.2400_L24x72/momsmear-2x/{conf_id}/",
    },
    "L32x64": {
        "Nt": 64, "Nx": 32, "Nev": 100, "Nev1": 100,
        "mom_smear": 3, "momsmear_phase": -3,
        "eig_dir": "/public/group/lqcd/eigensystem/beta6.20_mu-0.2790_ms-0.2400_L32x64/{conf_id}/",
        "peram_u_dir": "/public/home/sunp/sunpeng_public/mom_smear_perambulators/beta6.20_mu-0.2790_ms-0.2400_L32x64/output_dir_data/mz0_my0_mx-3/{conf_id}/",
        "corr_nucl_dir": "/public/group/lqcd/donghx/2pt_Result/beta6.20_mu-0.2790_ms-0.2400_L32x64/momsmear3x/{conf_id}/",
        "Pzlist": [3, 4, 5, 6, 7, 8],
    },
    "L32x96": {
        "Nt": 96, "Nx": 32, "Nev": 100, "Nev1": 100,
        "mom_smear": 2, "momsmear_phase": -2,
        "eig_dir": "/public/group/lqcd/eigensystem/beta6.41_mu-0.2295_ms-0.2050_L32x96/{conf_id}/",
        "peram_u_dir": "/public/home/sunp/sunpeng_public/mom_smear_perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/output_dir_data/mz0_my0_mx-2/{conf_id}/",
        "corr_nucl_dir": "/public/group/lqcd/donghx/2pt_Result/beta6.41_mu-0.2295_ms-0.2050_L32x96/momsmear2x/{conf_id}/",
    },
    "L36x108": {
        "Nt": 108, "Nx": 36, "Nev": 200, "Nev1": 200,
        "mom_smear": 2, "momsmear_phase": -2,
        "eig_dir": "/public/group/lqcd/eigensystem/beta6.498_mu-0.2150_ms-1926_L36x108/{conf_id}/",
        "peram_u_dir": "/public/home/sunp/sunpeng_new_disk/mom_smear_perambulators/beta6.498_mu-0.2150_ms-1926_L36x108/output_dir_data/mz0_my0_mx-2/{conf_id}/",
        "corr_nucl_dir": "/public/group/lqcd/donghx/2pt_Result/beta6.498_mu-0.2150_ms-1926_L36x108/momsmear2x/{conf_id}/",
    },
    "L48x96": {
        "Nt": 96, "Nx": 48, "Nev": 200, "Nev1": 200,
        "mom_smear": 4, "momsmear_phase": -4,
        "eig_dir": "/public/group/lqcd/eigensystem/beta6.20_mu-0.2825_ms-0.2310_L48x96/{conf_id}/",
        "peram_u_dir": "/public/home/sunp/sunpeng_new_disk/mom_smear_perambulators/beta6.20_mu-0.2825_ms-0.2310_L48x96/output_dir_data/mz0_my0_mx-4/{conf_id}/",
        "corr_nucl_dir": "/public/group/lqcd/donghx/2pt_Result/beta6.20_mu-0.2825_ms-0.2310_L48x96/momsmear4x/{conf_id}/",
    },
    "L48x144": {
        "Nt": 144, "Nx": 48, "Nev": 200, "Nev1": 200,
        "mom_smear": 2, "momsmear_phase": -2,
        "eig_dir": "/public/group/lqcd/eigensystem/beta6.72_mu-0.1850_ms-0.1700_L48x144/{conf_id}/",
        "peram_u_dir": "/public/home/sunp/sunpeng_public/mom_smear_perambulators/beta6.72_mu-0.1850_ms-0.1700_L48x144/output_dir_data/mz0_my0_mx-2/{conf_id}/",
        "corr_nucl_dir": "/public/group/lqcd/donghx/2pt_Result/beta6.72_mu-0.1850_ms-0.1700_L48x144/momsmear2x/{conf_id}/",
    },
}


def resolve_ensemble(ensemble_name, conf_id):
    """Resolve ensemble preset with conf_id substituted into paths."""
    if ensemble_name not in ENSEMBLES:
        raise ValueError(f"Unknown ensemble '{ensemble_name}'. "
                         f"Available: {list(ENSEMBLES.keys())}")
    cfg = dict(ENSEMBLES[ensemble_name])
    # Substitute {conf_id} in path templates
    for key in ["eig_dir", "peram_u_dir", "corr_nucl_dir"]:
        if key in cfg:
            cfg[key] = cfg[key].format(conf_id=conf_id)
    return cfg


# ==============================================================================
# Section 4: Gamma Matrices (DeGrand-Rossi basis)
# ==============================================================================
# EXACT copy from gamma_matrix_cupy_DR.py

_g0 = None
_g1 = None
_g2 = None
_g3 = None
_g4 = None
_g5 = None
_gamma_cache = {}


def _init_gamma():
    """Initialize gamma matrices in DeGrand-Rossi basis."""
    global _g0, _g1, _g2, _g3, _g4, _g5, _gamma_cache
    if _g0 is not None:
        return

    xp = get_xp()

    # identity
    _g0 = xp.zeros((4, 4), dtype=complex)
    _g0[0, 0] = 1.0 + 0.0j
    _g0[1, 1] = 1.0 + 0.0j
    _g0[2, 2] = 1.0 + 0.0j
    _g0[3, 3] = 1.0 + 0.0j

    # gamma1
    _g1 = xp.zeros((4, 4), dtype=complex)
    _g1[0, 3] = 0.0 + 1.0j
    _g1[1, 2] = 0.0 + 1.0j
    _g1[2, 1] = 0.0 - 1.0j
    _g1[3, 0] = 0.0 - 1.0j

    # gamma2
    _g2 = xp.zeros((4, 4), dtype=complex)
    _g2[0, 3] = -1.0 + 0.0j
    _g2[1, 2] = 1.0 + 0.0j
    _g2[2, 1] = 1.0 + 0.0j
    _g2[3, 0] = -1.0 + 0.0j

    # gamma3
    _g3 = xp.zeros((4, 4), dtype=complex)
    _g3[0, 2] = 0.0 + 1.0j
    _g3[1, 3] = 0.0 - 1.0j
    _g3[2, 0] = 0.0 - 1.0j
    _g3[3, 1] = 0.0 + 1.0j

    # gamma4
    _g4 = xp.zeros((4, 4), dtype=complex)
    _g4[0, 2] = 1.0 + 0.0j
    _g4[1, 3] = 1.0 + 0.0j
    _g4[2, 0] = 1.0 + 0.0j
    _g4[3, 1] = 1.0 + 0.0j

    # gamma5
    _g5 = xp.zeros((4, 4), dtype=complex)
    _g5[0, 0] = 1.0 + 0.0j
    _g5[1, 1] = 1.0 + 0.0j
    _g5[2, 2] = -1.0 + 0.0j
    _g5[3, 3] = -1.0 + 0.0j


def gamma(i):
    """Return gamma matrix by index (0-17) in DR basis.

    Index mapping (identical to gamma_matrix_cupy_DR.py):
      0: identity (gamma_0)
      1: gamma_x, 2: gamma_y, 3: gamma_z, 4: gamma_t
      5: gamma_5
      6: gamma_2 @ gamma_3  (= -gamma_1 gamma_4 gamma_5)
      7: gamma_3 @ gamma_1  (= -gamma_2 gamma_4 gamma_5)
      8: gamma_1 @ gamma_2  (= -gamma_3 gamma_4 gamma_5)
      9: gamma_1 @ gamma_4
      10: gamma_2 @ gamma_4
      11: gamma_3 @ gamma_4
      12: gamma_1 @ gamma_5
      13: gamma_2 @ gamma_5
      14: gamma_3 @ gamma_5
      15: gamma_4 @ gamma_5
      16: gamma(7) @ (1+gamma_4)/2  [not used in production]
      17: gamma(7) @ (1-gamma_4)/2  [not used in production]
    """
    global _gamma_cache
    _init_gamma()

    i = int(i)
    if i in _gamma_cache:
        return _gamma_cache[i]

    xp = get_xp()

    if i == 0:
        result = _g0
    elif i == 1:
        result = _g1
    elif i == 2:
        result = _g2
    elif i == 3:
        result = _g3
    elif i == 4:
        result = _g4
    elif i == 5:
        result = _g5
    elif i == 6:
        result = xp.matmul(_g2, _g3)
    elif i == 7:
        result = xp.matmul(_g3, _g1)
    elif i == 8:
        result = xp.matmul(_g1, _g2)
    elif i == 9:
        result = xp.matmul(_g1, _g4)
    elif i == 10:
        result = xp.matmul(_g2, _g4)
    elif i == 11:
        result = xp.matmul(_g3, _g4)
    elif i == 12:
        result = xp.matmul(_g1, _g5)
    elif i == 13:
        result = xp.matmul(_g2, _g5)
    elif i == 14:
        result = xp.matmul(_g3, _g5)
    elif i == 15:
        result = xp.matmul(_g4, _g5)
    elif i == 16:
        m1 = xp.matmul(_g3, _g1)
        m2 = 0.5 * (_g0 + _g4)
        result = xp.matmul(m1, m2)
    elif i == 17:
        m1 = xp.matmul(_g3, _g1)
        m2 = 0.5 * (_g0 - _g4)
        result = xp.matmul(m1, m2)
    else:
        raise ValueError(f"Unknown gamma index: {i}")

    _gamma_cache[i] = result
    return result


def get_interpolator_matrices(element):
    """Return (interProject1, interProject2) for a given element type.

    EXACT copy from the original 2pt code (line ~122-139).
    interProject1 and interProject2 are gamma matrix products
    used to project the perambulator.
    """
    if element == "_offdiag01":
        # gamma(7) = gamma3 @ gamma1
        interProject1 = gamma(7) @ gamma(3)
        interProject2 = gamma(7)
    elif element == "_offdiag02":
        interProject1 = gamma(7) @ gamma(4)
        interProject2 = gamma(7)
    elif element == "_offdiag12":
        interProject1 = gamma(7) @ gamma(3)
        interProject2 = gamma(7) @ gamma(4)
    elif element == "_Cg5g3":
        interProject1 = gamma(7) @ gamma(3)
        interProject2 = gamma(7) @ gamma(3)
    elif element == "_Cg5g4":
        # PRODUCTION DEFAULT: C gamma5 gamma4
        interProject1 = gamma(7) @ gamma(4)
        interProject2 = gamma(7) @ gamma(4)
    else:
        # "_Cg5" or empty: just gamma(7) = gamma3 @ gamma1
        interProject1 = gamma(7)
        interProject2 = gamma(7)
    return interProject1, interProject2


# ==============================================================================
# Section 5: I/O Functions
# ==============================================================================
# EXACT copies from input_output_4_cupy.py and original scripts


def read_eigenvectors(eig_dir, t, Nev1, conf_id, Nx):
    """Read eigenvectors for timeslice t, apply momentum smearing phase.

    EXACT copy from 2pt code line ~172-186.
    File format: float64 binary, shape (Nev, Nx, Nx, Nx, 3, 2)
                 last dim: [real, imag]
    Returns: GPU array of shape (Nev_full, Nx*Nx*Nx, 3)
    """
    filename = "%s/eigvecs_t%03d_%s" % (eig_dir, t, conf_id)
    with open(filename, "rb") as f:
        file_size = os.path.getsize(filename)
        element_size = 8  # float64
        Nev = int(file_size / element_size / (Nx * Nx * Nx * 3 * 2))
        data = np.fromfile(f, dtype="f8").reshape(Nev, Nx, Nx, Nx, 3, 2)
    Eigvec = data[..., 0] + 1j * data[..., 1]
    Eigvec = Eigvec[0:Nev1]
    eigvecs_cupy = asarray(Eigvec)
    eigvecs_cupy = eigvecs_cupy.reshape(Nev, Nx * Nx * Nx, 3)
    return eigvecs_cupy


def read_eigenvectors_no_smear(eig_dir, t, Nev1, conf_id, Nx):
    """Read eigenvectors WITHOUT momentum smearing phase.

    Used by the mom0/cupy_mom0_Liu workflow variants.
    EXACT copy from Calc_VVV.py line ~45-59 and mom0 variants.
    """
    filename = "%s/eigvecs_t%03d_%s" % (eig_dir, t, conf_id)
    with open(filename, "rb") as f:
        file_size = os.path.getsize(filename)
        element_size = 8
        Nev = int(file_size / element_size / (Nx * Nx * Nx * 3 * 2))
        data = np.fromfile(f, dtype="f8").reshape(Nev, Nx, Nx, Nx, 3, 2)
    Eigvec = data[..., 0] + 1j * data[..., 1]
    Eigvec = Eigvec[0:Nev1]
    eigvecs_cupy = asarray(Eigvec)
    eigvecs_cupy = eigvecs_cupy.reshape(Nev, Nx * Nx * Nx, 3)
    return eigvecs_cupy


def read_perambulator(peram_dir, conf_id, Nt, Nev1, t_source):
    """Read perambulator for one source timeslice.

    EXACT copy from 2pt code line ~205-236.
    File format: 4 float64 binary files (d_source=0,1,2,3)
    Each: (4, Nt, Nev, 4, Nev, 2) → transposed to (t_sink, d_sink, d_source, ev_sink, ev_source)
    Returns: GPU array of shape (Nt, 4, 4, Nev1, Nev1)
    """
    f = open("%s/perams.%s.0.%i" % (peram_dir, conf_id, t_source), "rb")
    peram = np.fromfile(f, dtype="f8")
    f.close()

    for d_source in range(1, 4):
        f = open("%s/perams.%s.%i.%i" % (peram_dir, conf_id, d_source, t_source), "rb")
        temp = np.fromfile(f, dtype="f8")
        peram = np.append(peram, temp)
        temp = None
        f.close()
    peram_size = peram.size
    Nev = int(np.sqrt(peram_size / (4 * 4 * Nt * 2)))
    peram = peram.reshape(
        4, Nt, Nev, 4, Nev, 2
    )  # d_source, t_sink, ev_source, d_sink, ev_sink, complex
    peram = peram.transpose(
        1, 3, 0, 4, 2, 5
    )  # t_sink, d_sink, d_source, ev_sink, ev_souce, complex
    peram = peram[..., 0] + peram[..., 1] * 1j
    peram = peram[:, :, :, 0:Nev1, 0:Nev1]
    peram_cupy = asarray(peram)
    return peram_cupy


def read_perambulator_all_cpu(peram_dir, conf_id, Nt, Nev1):
    """Read ALL perambulators for all t_source into CPU memory.

    EXACT copy from input_output_4_cupy.py readin_peram_all_cpu.
    Returns: numpy array of shape (Nt, Nt, 4, 4, Nev1, Nev1)
    """
    peram_cpu_all = np.zeros(
        (Nt, Nt, 4, 4, Nev1, Nev1), dtype=complex
    )
    for t_source in range(0, Nt):
        f = open("%s/perams.%s.0.%i" % (peram_dir, conf_id, t_source), "rb")
        peram = np.fromfile(f, dtype="f8")
        f.close()
        for d_source in range(1, 4):
            f = open(
                "%s/perams.%s.%i.%i" % (peram_dir, conf_id, d_source, t_source), "rb"
            )
            temp = np.fromfile(f, dtype="f8")
            peram = np.append(peram, temp)
            temp = None
            f.close()
        peram_size = peram.size
        Nev = int(np.sqrt(peram_size / (4 * 4 * Nt * 2)))
        peram = peram.reshape(4, Nt, Nev, 4, Nev, 2)
        peram = peram.transpose(1, 3, 0, 4, 2, 5)
        peram = peram[..., 0] + peram[..., 1] * 1j
        peram_cpu_all[t_source] = peram[:, :, :, 0:Nev1, 0:Nev1]
    return peram_cpu_all


def read_VVV(VVV_dir, t, Nev1, conf_id, Px, Py, Pz):
    """Read pre-computed VVV tensor for one timeslice.

    EXACT copy from input_output_4_cupy.py readin_VVV_device.
    File format: float64 binary, shape (Nev, Nev, Nev, 2) → real+imag → complex
    Returns: numpy array of shape (Nev1, Nev1, Nev1)
    """
    VVV = np.zeros((Nev1, Nev1, Nev1), dtype=complex)
    f = open(
        "%s/VVV.t%03i.Px%iPy%iPz%i.conf%s" % (VVV_dir, t, Px, Py, Pz, conf_id), "rb"
    )
    temp = np.fromfile(f, dtype="f8")
    Nev = int(np.cbrt(temp.size / 2))
    temp = temp.reshape(Nev, Nev, Nev, 2)
    temp = temp[..., 0] + temp[..., 1] * 1j
    temp = temp[0:Nev1, 0:Nev1, 0:Nev1]
    VVV = np.copy(temp)
    f.close()
    return VVV


def read_gauge_config(filename, Nt, Nx):
    """Read ILDG gauge configuration (big-endian float64 binary).

    EXACT copy from Calc_ope_unpol.py line ~98-105.
    Returns: numpy array of shape (Nt, Nx, Nx, Nx, 4, 3, 3)
    """
    f = open("%s" % filename, "rb")
    gauge = np.fromfile(f, dtype=">f8")
    gauge = np.array(gauge)
    gauge = gauge.reshape(Nt, Nx, Nx, Nx, 4, 3, 3, 2)
    gauge = gauge[..., 0] + gauge[..., 1] * 1j
    f.close()
    return gauge


# ==============================================================================
# Section 6: Field Strength Tensor & Nonlocal Operators
# ==============================================================================
# EXACT copies from Operator.py
# Tensor conventions: dir_mu,nv: 0(x), 1(y), 2(z), 3(t)


def build_levi_civita_tensor():
    """Build 1/2 * epsilon_{mu,nu,rho,sigma} antisymmetric tensor.

    EXACT copy from Operator.py Tensor4 (line ~9-38).
    Returns: numpy array of shape (4, 4, 4, 4)
    """
    Tensor4 = np.zeros((4, 4, 4, 4), dtype=complex)
    for i in range(0, 4):
        for j in range(0, 4):
            a = 0
            if i > j:
                a += 1.0
            for k in range(0, 4):
                b = 0
                if i > k:
                    b += 1.0
                if j > k:
                    b += 1.0
                for l in range(0, 4):
                    c = 0
                    if i > l:
                        c += 1.0
                    if j > l:
                        c += 1.0
                    if k > l:
                        c += 1.0
                    if (a + b + c) % 2 == 0:
                        Tensor4[i, j, k, l] = 1
                    elif (a + b + c) % 2 == 1:
                        Tensor4[i, j, k, l] = -1
                    lst = [i, j, k, l]
                    set_lst = set(lst)
                    if len(set_lst) != len(lst):
                        Tensor4[i, j, k, l] = 0
    Tensor4 = 0.5 * Tensor4
    return Tensor4


def plaquette_clover_single(gauge, mu, nu):
    """Compute clover plaquette F_{mu,nu} for one (mu,nu) pair.

    EXACT copy from Operator.py plaquette_clover_new (line ~77-161).
    Uses clover (4-plaquette average) to construct the field strength tensor.
    Returns: numpy array of shape (Nt, Nx, Nx, Nx, 3, 3)
    """
    gauge_rightup = gauge
    gauge_leftup = np.roll(gauge, 1, 3 - mu)
    gauge_rightdown = np.roll(gauge, 1, 3 - nu)
    gauge_leftdown = np.roll(gauge_leftup, 1, 3 - nu)

    # P_{mu,nu} — right-up plaquette
    pla_rightup = np.einsum(
        "tzyxab,tzyxbc->tzyxac",
        gauge_rightup[:, :, :, :, mu, :, :],
        np.roll(gauge_rightup, -1, 3 - mu)[:, :, :, :, nu, :, :],
    )
    pla_rightup = np.einsum(
        "tzyxab,tzyxcb->tzyxac",
        pla_rightup,
        np.roll(gauge_rightup, -1, 3 - nu)[:, :, :, :, mu, :, :].conj(),
    )
    pla_rightup = np.einsum(
        "tzyxab,tzyxcb->tzyxac",
        pla_rightup,
        gauge_rightup[:, :, :, :, nu, :, :].conj(),
    )

    # P_{nu,-mu} — left-up plaquette
    pla_leftup = np.einsum(
        "tzyxab,tzyxcb->tzyxac",
        np.roll(gauge_leftup, -1, 3 - mu)[:, :, :, :, nu, :, :],
        np.roll(gauge_leftup, -1, 3 - nu)[:, :, :, :, mu, :, :].conj(),
    )
    pla_leftup = np.einsum(
        "tzyxab,tzyxcb->tzyxac",
        pla_leftup,
        gauge_leftup[:, :, :, :, nu, :, :].conj(),
    )
    pla_leftup = np.einsum(
        "tzyxab,tzyxbc->tzyxac",
        pla_leftup,
        gauge_leftup[:, :, :, :, mu, :, :],
    )

    # P_{-mu,-nu} — left-down plaquette
    pla_leftdown = np.einsum(
        "tzyxba,tzyxcb->tzyxac",
        np.roll(gauge_leftdown, -1, 3 - nu)[:, :, :, :, mu, :, :].conj(),
        gauge_leftdown[:, :, :, :, nu, :, :].conj(),
    )
    pla_leftdown = np.einsum(
        "tzyxab,tzyxbc->tzyxac",
        pla_leftdown,
        gauge_leftdown[:, :, :, :, mu, :, :],
    )
    pla_leftdown = np.einsum(
        "tzyxab,tzyxbc->tzyxac",
        pla_leftdown,
        np.roll(gauge_leftdown, -1, 3 - mu)[:, :, :, :, nu, :, :],
    )

    # P_{-nu,mu} — right-down plaquette
    pla_rightdown = np.einsum(
        "tzyxba,tzyxbc->tzyxac",
        gauge_rightdown[:, :, :, :, nu, :, :].conj(),
        gauge_rightdown[:, :, :, :, mu, :, :],
    )
    pla_rightdown = np.einsum(
        "tzyxab,tzyxbc->tzyxac",
        pla_rightdown,
        np.roll(gauge_rightdown, -1, 3 - mu)[:, :, :, :, nu, :, :],
    )
    pla_rightdown = np.einsum(
        "tzyxab,tzyxcb->tzyxac",
        pla_rightdown,
        np.roll(gauge_rightdown, -1, 3 - nu)[:, :, :, :, mu, :, :].conj(),
    )

    # Clover average: F = -i/8 * sum of (P - P^dagger) for 4 orientations
    ans = (
        pla_rightup
        - pla_rightup.conj().transpose(0, 1, 2, 3, 5, 4)
        + pla_leftup
        - pla_leftup.conj().transpose(0, 1, 2, 3, 5, 4)
        + pla_leftdown
        - pla_leftdown.conj().transpose(0, 1, 2, 3, 5, 4)
        + pla_rightdown
        - pla_rightdown.conj().transpose(0, 1, 2, 3, 5, 4)
    )
    return -1j * ans / 8.0


def plaquette_clover_all(gauge, Nt, Nx):
    """Compute clover plaquettes for all 4x4 (mu,nu) pairs.

    EXACT copy from Operator.py plaquette_clover_all_new (line ~164-174).
    Returns: numpy array of shape (4, 4, Nt, Nx, Nx, Nx, 3, 3)
    """
    pla = np.zeros((4, 4, Nt, Nx, Nx, Nx, 3, 3), dtype=complex)
    Temp_zeros = np.zeros((Nt, Nx, Nx, Nx, 3, 3), dtype=complex)
    for _mu in range(4):
        for _nu in range(4):
            if _mu == _nu:
                pla[_mu, _nu] = Temp_zeros
            else:
                pla[_mu, _nu] = plaquette_clover_single(gauge, _mu, _nu)
    return pla


def plaquette_clover_all_tilde(pla_all, Nt, Nx):
    """Compute dual field strength F_tilde from Levi-Civita contraction.

    EXACT copy from Operator.py plaquette_clover_all_tilde (line ~177-184).
    Returns: numpy array of shape (4, 4, Nt, Nx, Nx, Nx, 3, 3)
    """
    Tensor4 = build_levi_civita_tensor()
    pla_tilde = contract(
        "opmn,mntzyxab->optzyxab",
        Tensor4,
        pla_all,
    )
    return pla_tilde


# ---------------------------------------------------------------------------
# Nonlocal gluon OPE operators (from Operator.py)
# ---------------------------------------------------------------------------


def operators_new_z0_mu2(gauge, zdir, pla, pla_tilde, delta_z, mu, nu, mu2, nu2, Nt):
    """Construct nonlocal gluon OPE operator with Wilson lines.

    EXACT copy from Operator.py operators_new_z0_mu2 (line ~339-368).
    Structure: F_{mu,nu}(z) * W(z←0) * F_tilde_{mu2,nu2}(0) * W(0←z)
    where W is the Wilson line in the zdir direction.

    Returns: numpy array of shape (Nt,)
    """
    op = np.zeros(Nt, dtype=complex)
    z_dir = zdir

    # Step 1: Shift F to position z
    ope = np.roll(pla[mu, nu], -delta_z, 3 - z_dir)

    # Step 2: Wilson line from z to 0 (contract with U_z^dagger)
    for _dz in range(delta_z):
        ope = np.einsum(
            "tzyxab,tzyxcb->tzyxac",
            ope,
            np.roll(gauge, -(delta_z - 1 - _dz), 3 - z_dir)[
                :, :, :, :, z_dir, :, :
            ].conj(),
        )

    # Step 3: Insert F_tilde at z=0
    ope = np.einsum(
        "tzyxab,tzyxbc->tzyxac",
        ope,
        pla_tilde[mu2, nu2],
    )

    # Step 4: Wilson line from 0 to z (contract with U_z)
    for _dz in range(delta_z):
        ope = np.einsum(
            "tzyxab,tzyxbc->tzyxac",
            ope,
            np.roll(gauge, -_dz, 3 - z_dir)[:, :, :, :, z_dir, :, :],
        )

    # Step 5: Color trace + spatial sum
    ans = np.trace(ope, axis1=4, axis2=5)
    op = np.sum(ans, axis=(1, 2, 3))
    return op


def operators_new_z0_mu2_xy(
    gauge, zdir, pla, pla_tilde, delta_z, mu, nu, mu2, nu2, Nt, Nx
):
    """Nonlocal OPE operator, preserving transverse spatial dims (x,y).

    EXACT copy from Operator.py operators_new_z0_mu2_xy (line ~402-433).
    Same as operators_new_z0_mu2 but sums only over the Wilson line direction,
    preserving the two transverse spatial dimensions.

    Returns: numpy array of shape (Nt, Nx, Nx)
    """
    op = np.zeros((Nt, Nx, Nx), dtype=complex)
    z_dir = zdir

    ope = np.roll(pla[mu, nu], -delta_z, 3 - z_dir)
    for _dz in range(delta_z):
        ope = np.einsum(
            "tzyxab,tzyxcb->tzyxac",
            ope,
            np.roll(gauge, -(delta_z - 1 - _dz), 3 - z_dir)[
                :, :, :, :, z_dir, :, :
            ].conj(),
        )
    ope = np.einsum(
        "tzyxab,tzyxbc->tzyxac",
        ope,
        pla_tilde[mu2, nu2],
    )
    for _dz in range(delta_z):
        ope = np.einsum(
            "tzyxab,tzyxbc->tzyxac",
            ope,
            np.roll(gauge, -_dz, 3 - z_dir)[:, :, :, :, z_dir, :, :],
        )

    ans = np.trace(ope, axis1=4, axis2=5)
    # Sum only along the zdir direction, keep transverse dims
    op = np.sum(ans, axis=3 - z_dir)
    return op


def operators_new_z0_mz_mu2(
    gauge, zdir, pla, pla_tilde, delta_z, mu, nu, mu2, nu2, Nt
):
    """Nonlocal OPE operator with NEGATIVE z-direction Wilson line.

    EXACT copy from Operator.py operators_new_z0_mz_mu2 (line ~371-398).
    Same structure but delta_z → -delta_z.

    Returns: numpy array of shape (Nt,)
    """
    op = np.zeros(Nt, dtype=complex)
    z_dir = zdir
    delta_z_neg = -1 * delta_z

    ope = np.roll(pla[mu, nu], -delta_z_neg, 3 - z_dir)
    for _dz in range(0, delta_z_neg, -1):
        ope = np.einsum(
            "tzyxab,tzyxbc->tzyxac",
            ope,
            np.roll(gauge, -(delta_z_neg - _dz), 3 - z_dir)[
                :, :, :, :, z_dir, :, :
            ],
        )
    ope = np.einsum(
        "tzyxab,tzyxbc->tzyxac",
        ope,
        pla_tilde[mu2, nu2],
    )
    for _dz in range(0, delta_z_neg, -1):
        ope = np.einsum(
            "tzyxab,tzyxcb->tzyxac",
            ope,
            np.roll(gauge, -_dz + 1, 3 - z_dir)[:, :, :, :, z_dir, :, :].conj(),
        )

    ans = np.trace(ope, axis1=4, axis2=5)
    op = np.sum(ans, axis=(1, 2, 3))
    return op


def operators_FF_z0(gauge, pla, pla_tilde, delta_z, mu, nu, mu2, nu2):
    """Gauge-fixed OPE operator (NO Wilson lines).

    EXACT copy from Operator.py operators_FF_z0 (line ~187-198).
    Simply correlates F_{mu,nu}(z) with F_tilde_{mu2,nu2}(0) without Wilson lines.
    Used when gauge is fixed (Coulomb/Landau).

    Returns: numpy array of shape (Nt,) — summed over spatial dims
    """
    op = 0.0 + 0.0 * 1j
    z_dir = 2  # hardcoded z-direction
    ope = np.roll(pla[mu, nu], -delta_z, 3 - z_dir)
    ope = np.einsum("tzyxab,tzyxbc->tzyxac", ope, pla_tilde[mu2, nu2])
    ans = np.trace(ope, axis1=4, axis2=5)
    op = np.sum(ans, axis=(1, 2, 3))
    return op


def operators_FF_z0_mz(gauge, pla, pla_tilde, delta_z, mu, nu, mu2, nu2):
    """Gauge-fixed OPE operator, negative z-direction.

    EXACT copy from Operator.py operators_FF_z0_mz (line ~200-211).
    """
    op = 0.0 + 0.0 * 1j
    z_dir = 2
    delta_z_neg = -1 * delta_z
    ope = np.roll(pla[mu, nu], -delta_z_neg, 3 - z_dir)
    ope = np.einsum("tzyxab,tzyxbc->tzyxac", ope, pla_tilde[mu2, nu2])
    ans = np.trace(ope, axis1=4, axis2=5)
    op = np.sum(ans, axis=(1, 2, 3))
    return op


# ==============================================================================
# Section 7: Proton 2pt Correlator (Distillation)
# ==============================================================================
# EXACT copies from the 2pt correlator scripts


def phase_calc(Mom, Nx):
    """Compute momentum phase factor exp(-i P·x).

    EXACT copy from 2pt code line ~155-165.
    Coordinate order: (z, y, x) for position array.

    Args:
        Mom: 3-vector momentum [Pz, Py, Px]
        Nx: spatial lattice extent

    Returns:
        phase_factor on GPU, shape (Nx³,)
    """
    phase_factor = np.zeros(Nx * Nx * Nx, dtype=complex)
    for z in range(0, Nx):
        for y in range(0, Nx):
            for x in range(0, Nx):
                Pos = np.array([z, y, x])
                phase_factor[z * Nx * Nx + y * Nx + x] = np.exp(
                    -np.dot(Mom, Pos) * 2 * np.pi * 1j / Nx
                )
    phase_factor_cupy = asarray(phase_factor)
    return phase_factor_cupy


def compute_VVV(eig_dir, Nt, Nev1, conf_id, Nx, Mom, phase_factor):
    """Compute VVV baryon block (3-quark vertex) with momentum projection.

    EXACT copy from 2pt code line ~244-318.
    Includes:
    - Read eigenvectors for each timeslice t
    - Compute momentum phase factor
    - Loop over x-slices to manage GPU memory
    - 6-term Levi-Civita contraction of color indices

    The 6 terms correspond to the 6 permutations of color indices
    in the baryon (epsilon_{abc} q_a q_b q_c) with the phase factor:
      + epsilon_{abc} * e_a * e_b * e_c [cyclic: 012, 120, 201]
      - epsilon_{abc} * e_a * e_c * e_b [anticyclic: 021, 102, 210]
    where e_a, e_b, e_c are color components (0,1,2) of the eigenvector.

    Returns:
        VVV_sink on GPU, shape (Nt, Nev1, Nev1, Nev1)
    """
    xp = get_xp()
    VVV_sink = xp.zeros((Nt, Nev1, Nev1, Nev1), dtype=complex)

    for t in range(0, Nt):
        st1 = time.time()
        # Read eigenvectors with momentum smearing
        eigvecs_cupy = read_eigenvectors(eig_dir, t, Nev1, conf_id, Nx)
        # Apply momentum phase: eigvecs_mom2 = contract("vxa,x->vxa", eigvecs, phase_factor)
        eigvecs_mom2 = contract("vxa,x->vxa", eigvecs_cupy, phase_factor)
        ed1 = time.time()
        print("Read-in eigenvector done , time used: %.3f s" % (ed1 - st1))

        st2 = time.time()
        phase_factor_cupy = phase_calc(Mom, Nx)
        # Loop over x-slices to manage memory
        for xi in range(0, Nx):
            VVV_sink[t] = (
                VVV_sink[t]
                + contract(
                    "x,ax,bx,cx->abc",
                    phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                    eigvecs_mom2[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                    eigvecs_mom2[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                    eigvecs_mom2[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                )
                + contract(
                    "x,ax,bx,cx->abc",
                    phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                    eigvecs_mom2[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                    eigvecs_mom2[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                    eigvecs_mom2[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                )
                + contract(
                    "x,ax,bx,cx->abc",
                    phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                    eigvecs_mom2[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                    eigvecs_mom2[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                    eigvecs_mom2[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                )
                - contract(
                    "x,ax,bx,cx->abc",
                    phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                    eigvecs_mom2[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                    eigvecs_mom2[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                    eigvecs_mom2[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                )
                - contract(
                    "x,ax,bx,cx->abc",
                    phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                    eigvecs_mom2[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                    eigvecs_mom2[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                    eigvecs_mom2[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                )
                - contract(
                    "x,ax,bx,cx->abc",
                    phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                    eigvecs_mom2[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                    eigvecs_mom2[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                    eigvecs_mom2[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                )
            )
        ed2 = time.time()
        print("t %d VVV Contraction done , time used: %.3f s" % (t, ed2 - st2))

    return VVV_sink


def apply_parity_and_boundary(contrac_nucl_matrix, Nt):
    """Apply parity projection and boundary sign conventions.

    EXACT copy from 2pt code line ~408-433.

    Parity projectors:
      P_plus  = 0.5 * (gamma_0 + gamma_4)
      P_minus = 0.5 * (gamma_0 - gamma_4)

    Sign conventions (anti-periodic BC in time):
      t_sink < t_source: flip positive parity sign
      t_sink > t_source: flip negative parity sign

    Returns:
        contrac_nucl_pp, contrac_nucl_pm on GPU, each shape (Nt, Nt)
    """
    xp = get_xp()
    matrix_pplus = 0.5 * (gamma(0) + gamma(4))
    matrix_pminus = 0.5 * (gamma(0) - gamma(4))

    contrac_nucl_pp = contract("li,yxil->yx", matrix_pplus, contrac_nucl_matrix)
    contrac_nucl_pm = contract("li,yxil->yx", matrix_pminus, contrac_nucl_matrix)

    for t_source in range(0, Nt):
        for t_sink in range(0, Nt):
            if t_sink < t_source:
                contrac_nucl_pp[t_sink, t_source] = (
                    -1.0 * contrac_nucl_pp[t_sink, t_source]
                )
            if t_sink > t_source:
                contrac_nucl_pm[t_sink, t_source] = (
                    -1.0 * contrac_nucl_pm[t_sink, t_source]
                )

    return contrac_nucl_pp, contrac_nucl_pm


# ==============================================================================
# Section 8: OPE Lorentz Pair Assignments (MPI-aware)
# ==============================================================================


def get_ope_lorentz_pairs(zdir, mode="unpol"):
    """Return list of (mu, nu, mu2, nu2) Lorentz index pairs for OPE.

    EXACT copy from the MPI rank assignments in the original OPE code.

    For unpol mode (3 pairs, 3 MPI ranks):
      Rank 0: mu=3, nu=(zdir+1)%3  — time with one spatial direction
      Rank 1: mu=3, nu=(zdir+2)%3  — time with other spatial direction
      Rank 2: mu=(zdir+1)%3, nu=(zdir+2)%3 — two spatial directions

    For helicity mode (same 3 pairs but uses pla_tilde):
      Same indices as unpol.

    For gauge_fix mode (hardcoded z=2, 3 or 4 pairs):
      See the original Calc_ope_gauge_fix_*.py files.

    Args:
        zdir: Wilson line direction (0=x, 1=y, 2=z)
        mode: "unpol", "helicity", or "gauge_fix"

    Returns:
        List of (mu, nu, mu2, nu2) tuples
    """
    if mode in ("unpol", "helicity"):
        # EXACT copy from Calc_ope_unpol.py and Calc_ope_helicity.py
        pairs = [
            (3, (zdir + 1) % 3, 3, (zdir + 1) % 3),           # rank 0
            (3, (zdir + 2) % 3, 3, (zdir + 2) % 3),           # rank 1
            ((zdir + 1) % 3, (zdir + 2) % 3, (zdir + 1) % 3, (zdir + 2) % 3),  # rank 2
        ]
    elif mode == "gauge_fix_unpol":
        # EXACT copy from Calc_ope_gauge_fix_unpol.py
        pairs = [
            (3, 0, 3, 0),   # F_{t,x}
            (3, 1, 3, 1),   # F_{t,y}
            (0, 1, 0, 1),   # F_{x,y}
        ]
    elif mode == "gauge_fix_helicity":
        # EXACT copy from Calc_ope_gauge_fix_helicity.py
        pairs = [
            (3, 0, 2, 1),
            (3, 1, 0, 2),
            (3, 2, 0, 1),
            (0, 1, 3, 2),
        ]
    else:
        raise ValueError(f"Unknown OPE mode: {mode}")
    return pairs


# ==============================================================================
# Section 9: Main Workflow Functions
# ==============================================================================


def run_proton_2pt(config):
    """Run complete proton 2pt correlator computation.

    This is the EXACT equivalent of the original 2pt_proton_Cg5gmu_*.py scripts.
    Pipeline:
      1. For each momentum in Pzlist:
         a. Compute VVV for all timeslices
         b. For each t_source: read perambulator, perform Wick contraction
         c. Apply parity projection and boundary signs
         d. Save results with original naming convention

    Args:
        config: dict with keys:
            Nt, Nx, Nev, Nev1, Px, Py, Pz, mom_smear, momsmear_phase,
            conf_id, eig_dir, peram_u_dir, corr_nucl_dir,
            element (default "_Cg5g4"),
            Pzlist (optional list of momenta)
    """
    Nt = config["Nt"]
    Nx = config["Nx"]
    Nev1 = config.get("Nev1", config["Nev"])
    conf_id = config["conf_id"]
    eig_dir = config["eig_dir"]
    peram_u_dir = config["peram_u_dir"]
    corr_nucl_dir = config["corr_nucl_dir"]
    Px = config.get("Px", 0)
    Py = config.get("Py", 0)
    Pz = config.get("Pz", 0)
    mom_smear = config.get("mom_smear", 3)
    momsmear_phase = config.get("momsmear_phase", -3)
    element = config.get("element", "_Cg5g4")
    Pzlist = config.get("Pzlist", [3, 4, 5, 6, 7, 8])

    print("=" * 70)
    print("Proton 2pt Correlator — Distillation + Momentum Smearing")
    print(f"  Ensemble Nt={Nt}, Nx={Nx}, Nev1={Nev1}")
    print(f"  conf_id = {conf_id}")
    print(f"  momentum smear: mom_smear={mom_smear}, phase={momsmear_phase}")
    print(f"  element = {element}")
    print(f"  Mom list: {Pzlist}")
    print(f"  eig_dir = {eig_dir}")
    print(f"  peram_u_dir = {peram_u_dir}")
    print(f"  corr_nucl_dir = {corr_nucl_dir}")
    print("=" * 70)

    # Get interpolator matrices
    interProject1, interProject2 = get_interpolator_matrices(element)
    print("interProject1 =", interProject1)
    print("interProject2 =", interProject2)

    st00 = time.time()

    for Px_val in Pzlist:
        Mom = np.array([Pz, Py, Px_val])
        print("Mom:", Mom)

        # Step 1: Compute VVV
        VVV_sink = compute_VVV(eig_dir, Nt, Nev1, conf_id, Nx, Mom,
                               phase_calc(np.array([0, 0, momsmear_phase], dtype=complex), Nx))

        # Step 2: Contract perambulators for each t_source
        xp = get_xp()
        interProject1_cupy = asarray(interProject1)
        interProject2_cupy = asarray(interProject2)
        contrac_nucl_matrix = xp.zeros((Nt, Nt, 4, 4), dtype=complex)

        st = time.time()
        print("Contract start")

        for t_source in range(0, Nt):
            st0 = time.time()
            peram_u = read_perambulator(peram_u_dir, conf_id, Nt, Nev1, t_source)

            VVV_source = xp.conj(VVV_sink[t_source])

            CG5peram_uCG5 = contract(
                "gh,thkbe,jk->tgjbe",
                interProject1_cupy,
                peram_u,
                interProject2_cupy,
            )

            for t_sink in range(0, Nt):
                deltat = (t_sink - t_source + Nt) % Nt
                if 2 <= deltat <= 30:
                    contrac_nucl_matrix[t_sink, t_source] = contract(
                        "abc,gjad,gjbe,ilcf,def->il",
                        VVV_sink[t_sink],
                        peram_u[t_sink],
                        CG5peram_uCG5[t_sink],
                        peram_u[t_sink],
                        VVV_source,
                    ) - contract(
                        "abc,glaf,gjbe,ijcd,def->il",
                        VVV_sink[t_sink],
                        peram_u[t_sink],
                        CG5peram_uCG5[t_sink],
                        peram_u[t_sink],
                        VVV_source,
                    )

            free_memory_pool()
            synchronize()

            ed0 = time.time()
            print(
                "****************all complete for t_source %d, time used: %.3f s****************"
                % (t_source, ed0 - st0)
            )

        # Step 3: Save raw contraction matrix
        output_contract = "%s/twopt_slice_pp_Px%sPy%sPz%s_eginphase%s%s_contract_conf%s.npy" % (
            corr_nucl_dir, Px_val, Py, Pz, mom_smear, element, conf_id
        )
        np.save(output_contract, to_numpy(contrac_nucl_matrix))
        print("Saved raw contraction to:", output_contract)

        # Step 4: Apply parity projection and boundary signs
        contrac_nucl_pp, contrac_nucl_pm = apply_parity_and_boundary(
            contrac_nucl_matrix, Nt
        )

        # Step 5: Save parity-projected correlators
        output_pp = "%s/twopt_slice_pp_Px%sPy%sPz%s_eginphase%s%s_nopol_ss_conf%s.npy" % (
            corr_nucl_dir, Px_val, Py, Pz, mom_smear, element, conf_id
        )
        if HAS_CUPY:
            cp.save(output_pp, contrac_nucl_pp)
        else:
            np.save(output_pp, contrac_nucl_pp)
        print("Saved parity-projected correlator to:", output_pp)

        # Clean up
        del VVV_sink
        del VVV_source
        print(Px_val, "succeed")

    ed00 = time.time()
    print("JOB: ran successfully. Total time: %.3f s" % (ed00 - st00))


def run_ope(config):
    """Run complete OPE (Operator Product Expansion) calculation.

    This is the EXACT equivalent of Calc_ope_unpol.py / Calc_ope_helicity.py.
    Pipeline:
      1. Read gauge configuration
      2. Compute or load clover plaquettes F_{mu,nu}
      3. Optionally compute dual F_tilde_{mu,nu}
      4. For each Lorentz index pair (distributed via MPI or sequential):
         a. Loop over dz ∈ [0, delta_z)
         b. Construct nonlocal operator with Wilson lines
         c. Save operator data

    Args:
        config: dict with keys:
            Nt, Nx, conf_id, conf_file, output_dir,
            delta_z, link_dir (x/y/z), mode ("unpol" or "helicity"),
            pla_dir (optional, for loading precomputed plaquettes),
            use_xy (output shape with transverse dims),
            compute_mz (also compute negative-z operators)
    """
    Nt = config["Nt"]
    Nx = config["Nx"]
    conf_id = config["conf_id"]
    conf_file = config["conf_file"]
    output_dir = config["output_dir"]
    delta_z = config["delta_z"]
    link_dir = config.get("link_dir", "z")
    mode = config.get("mode", "unpol")
    pla_dir = config.get("pla_dir", None)
    use_xy = config.get("use_xy", False)
    compute_mz = config.get("compute_mz", False)

    # Determine zdir from link_dir
    if link_dir == "x":
        zdir = 0
        output_dir = output_dir + "/xdir/%s" % (conf_id)
    elif link_dir == "y":
        zdir = 1
        output_dir = output_dir + "/ydir/%s" % (conf_id)
    elif link_dir == "z":
        zdir = 2
        output_dir = output_dir + "/zdir/%s" % (conf_id)
    else:
        raise ValueError(f"Invalid link_dir: {link_dir}")

    print(f"link_dir: {link_dir}, zdir: {zdir}, output_dir: {output_dir}")

    # Determine sysm suffix and whether to use pla_tilde
    if mode == "helicity":
        sysm = "_AS"
        use_pla_tilde = True
    else:
        sysm = ""
        use_pla_tilde = False

    # Get Lorentz pairs
    pairs = get_ope_lorentz_pairs(zdir, mode)

    # MPI distribution
    if HAS_MPI and config.get("use_mpi", True):
        my_pairs = []
        for i, pair in enumerate(pairs):
            if i % size == rank:
                my_pairs.append((i, pair))
    else:
        my_pairs = list(enumerate(pairs))

    if len(my_pairs) == 0:
        print(f"Rank {rank}: no pairs assigned, exiting.")
        return

    # Step 1: Read gauge configuration
    st = time.time()
    print(f"Rank {rank}: Reading gauge config from {conf_file}")
    gauge = read_gauge_config(conf_file, Nt, Nx)
    print(f"Rank {rank}: gauge shape = {gauge.shape}")
    ed = time.time()
    print(f"Rank {rank}: read gauge done, time used: %.3f s" % (ed - st))

    # Step 2: Compute or load plaquettes
    if pla_dir is not None:
        pla = np.load("%s/ops_pla_conf%s.npz" % (pla_dir, conf_id))["pla"]
        print(f"Rank {rank}: loaded plaquettes from file")
    else:
        st = time.time()
        pla = plaquette_clover_all(gauge, Nt, Nx)
        ed = time.time()
        print(f"Rank {rank}: calculate plaquette done, time used: %.3f s" % (ed - st))

    # Step 3: Optionally compute pla_tilde (Hodge dual)
    if use_pla_tilde:
        pla_tilde = plaquette_clover_all_tilde(pla, Nt, Nx)
        print(f"Rank {rank}: computed pla_tilde (Hodge dual)")
    else:
        pla_tilde = pla  # use same plaquette for unpol

    # Step 4: Compute OPE operators for each assigned Lorentz pair
    for pair_idx, (mu, nu, mu2, nu2) in my_pairs:
        print(f"Rank {rank}: computing pair {pair_idx}: mu={mu}, nu={nu}, mu2={mu2}, nu2={nu2}")

        if use_xy:
            ops = np.zeros((Nt, Nx, Nx, delta_z), dtype=complex)
            if compute_mz:
                ops_mz = np.zeros((Nt, Nx, Nx, delta_z), dtype=complex)
        else:
            ops = np.zeros((Nt, delta_z), dtype=complex)
            if compute_mz:
                ops_mz = np.zeros((Nt, delta_z), dtype=complex)

        st = time.time()
        for _dz in range(delta_z):
            print(f"  dz = {_dz}")
            if use_xy:
                ops[:, :, :, _dz] = operators_new_z0_mu2_xy(
                    gauge, zdir, pla, pla_tilde, _dz, mu, nu, mu2, nu2, Nt, Nx
                )
                if compute_mz:
                    ops_mz[:, :, :, _dz] = operators_new_z0_mz_mu2(
                        gauge, zdir, pla, pla_tilde, _dz, mu, nu, mu2, nu2, Nt
                    )
            else:
                ops[:, _dz] = operators_new_z0_mu2(
                    gauge, zdir, pla, pla_tilde, _dz, mu, nu, mu2, nu2, Nt
                )
                if compute_mz:
                    ops_mz[:, _dz] = operators_new_z0_mz_mu2(
                        gauge, zdir, pla, pla_tilde, _dz, mu, nu, mu2, nu2, Nt
                    )

        ed = time.time()
        print(f"Rank {rank}: pair {pair_idx} done, time used: %.3f s" % (ed - st))

        # Save results
        outfile = "%s/ops_mu%d_nu%d_dz%d_conf%s%s.npy" % (
            output_dir, mu, nu, delta_z, conf_id, sysm
        )
        np.save(outfile, ops)
        print(f"Rank {rank}: saved to {outfile}")

        if compute_mz:
            outfile_mz = "%s/ops_minus_mu%d_nu%d_dz%d_conf%s%s.npy" % (
                output_dir, mu, nu, delta_z, conf_id, sysm
            )
            np.save(outfile_mz, ops_mz)
            print(f"Rank {rank}: saved minus-z to {outfile_mz}")

    print(f"Rank {rank}: OPE calculation complete.")


def run_plaquette_only(config):
    """Compute and save clover plaquettes only (no OPE operators).

    EXACT copy of Calc_pla.py.
    """
    Nt = config["Nt"]
    Nx = config["Nx"]
    conf_id = config["conf_id"]
    conf_file = config["conf_file"]
    output_dir = config["output_dir"]

    st = time.time()
    print(f"Reading gauge config from {conf_file}")
    gauge = read_gauge_config(conf_file, Nt, Nx)
    print(f"gauge shape = {gauge.shape}")
    ed = time.time()
    print(f"read gauge done, time used: %.3f s" % (ed - st))

    st = time.time()
    pla = plaquette_clover_all(gauge, Nt, Nx)
    ed = time.time()
    print(f"calculate plaquette done, time used: %.3f s" % (ed - st))

    outfile = "%s/ops_pla_conf%s.npz" % (output_dir, conf_id)
    np.savez(outfile, pla=pla)
    print(f"Saved plaquettes to {outfile}")


# ==============================================================================
# Section 10: Parameter Parsing (argparse + backward-compat stdin)
# ==============================================================================


def parse_stdin_params(params_file=None):
    """Parse parameters from stdin or a params file.

    Backward-compatible with the original fileinput.input() format.
    Format: key value (one per line)
    Supported keys: Nt, Nx, Nev, Nev1, Px, Py, Pz, mom_smear, delta_z,
                    conf_id, conf_file, link_dir, pla_dir, output_dir,
                    eig_dir, peram_u_dir, corr_nucl_dir, element, Pzlist
    """
    params = {}
    if params_file:
        lines = open(params_file, "r").readlines()
    else:
        lines = sys.stdin.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tmp = line.split()
        if len(tmp) < 2:
            continue
        key = tmp[0]
        # Try to parse as int, fall back to string
        try:
            value = int(tmp[1])
        except ValueError:
            try:
                value = float(tmp[1])
            except ValueError:
                value = tmp[1]
        params[key] = value
    return params


def build_config_from_args(args, params_file=None):
    """Build unified config dict from argparse args and optional params file."""
    # Start with params file if provided
    config = {}
    if params_file:
        config.update(parse_stdin_params(params_file))

    # Override with CLI args
    if args.ensemble:
        config.update(resolve_ensemble(args.ensemble, args.conf_id))

    # Manual overrides
    for key in ["Nt", "Nx", "Nev", "Nev1", "Px", "Py", "Pz",
                "mom_smear", "momsmear_phase", "delta_z"]:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    for key in ["conf_id", "conf_file", "link_dir", "output_dir",
                "eig_dir", "peram_u_dir", "corr_nucl_dir",
                "pla_dir", "element", "mode"]:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    # Parse Pzlist if provided
    if hasattr(args, "pzlist") and args.pzlist:
        config["Pzlist"] = [int(x) for x in args.pzlist.split(",")]

    # Compute derived paths
    if "conf_id" not in config:
        raise ValueError("conf_id is required (--conf-id or in params file)")

    # Ensure output_dir exists
    if "output_dir" in config:
        os.makedirs(config["output_dir"], exist_ok=True)

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Gluon PDF Workflow — unified proton 2pt + OPE calculation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 2pt --ensemble L32x64 --conf-id 20000
  %(prog)s ope --ensemble L32x64 --conf-id 20000 --gauge-file config.dat --output-dir ./results/
  %(prog)s pla --ensemble L32x64 --conf-id 20000 --gauge-file config.dat --output-dir ./results/
  %(prog)s 2pt --params-file params.txt
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Workflow step to run")

    # Common arguments for all subcommands
    def add_common_args(p):
        p.add_argument("--ensemble", default=None,
                       choices=list(ENSEMBLES.keys()),
                       help="Ensemble preset name")
        p.add_argument("--conf-id", type=str, default=None,
                       help="Configuration ID")
        p.add_argument("--Nt", type=int, default=None, help="Temporal extent")
        p.add_argument("--Nx", type=int, default=None, help="Spatial extent")
        p.add_argument("--Nev", type=int, default=None,
                       help="Number of eigenvectors")
        p.add_argument("--Nev1", type=int, default=None,
                       help="Number of eigenvectors in contraction")
        p.add_argument("--params-file", type=str, default=None,
                       help="Read parameters from file (backward compat)")
        p.add_argument("--no-mpi", action="store_true",
                       help="Disable MPI even if available")

    # ---- 2pt subcommand ----
    p2pt = subparsers.add_parser("2pt", help="Proton 2pt correlator")
    add_common_args(p2pt)
    p2pt.add_argument("--Px", type=int, default=0, help="Momentum x")
    p2pt.add_argument("--Py", type=int, default=0, help="Momentum y")
    p2pt.add_argument("--Pz", type=int, default=0, help="Momentum z")
    p2pt.add_argument("--mom-smear", type=int, default=None,
                       help="Momentum smearing parameter")
    p2pt.add_argument("--momsmear-phase", type=int, default=None,
                       help="Momentum smearing phase")
    p2pt.add_argument("--element", type=str, default="_Cg5g4",
                       choices=["_Cg5g4", "_Cg5g3", "_Cg5",
                                "_offdiag01", "_offdiag02", "_offdiag12"],
                       help="Interpolator element")
    p2pt.add_argument("--eig-dir", type=str, default=None,
                       help="Eigenvector directory")
    p2pt.add_argument("--peram-u-dir", type=str, default=None,
                       help="Perambulator directory")
    p2pt.add_argument("--corr-nucl-dir", type=str, default=None,
                       help="Output directory for correlator")
    p2pt.add_argument("--pzlist", type=str, default=None,
                       help="Comma-separated momentum list (e.g. 3,4,5,6,7,8)")

    # ---- ope subcommand ----
    pope = subparsers.add_parser("ope", help="OPE calculation")
    add_common_args(pope)
    pope.add_argument("--conf-file", type=str, default=None,
                       help="Gauge configuration file path")
    pope.add_argument("--link-dir", type=str, default="z",
                       choices=["x", "y", "z"],
                       help="Wilson line direction")
    pope.add_argument("--delta-z", type=int, default=15,
                       help="Maximum Wilson line length")
    pope.add_argument("--output-dir", type=str, default="./ope_output",
                       help="Output directory")
    pope.add_argument("--pla-dir", type=str, default=None,
                       help="Directory with precomputed plaquettes")
    pope.add_argument("--mode", type=str, default="unpol",
                       choices=["unpol", "helicity",
                                "gauge_fix_unpol", "gauge_fix_helicity"],
                       help="OPE mode (unpol=no tilde, helicity=with tilde)")
    pope.add_argument("--use-xy", action="store_true",
                       help="Preserve transverse spatial dimensions")
    pope.add_argument("--compute-mz", action="store_true",
                       help="Also compute negative-z operators")

    # ---- pla subcommand ----
    ppla = subparsers.add_parser("pla", help="Compute plaquettes only")
    add_common_args(ppla)
    ppla.add_argument("--conf-file", type=str, required=True,
                       help="Gauge configuration file path")
    ppla.add_argument("--output-dir", type=str, default="./pla_output",
                       help="Output directory for plaquettes")

    # ---- full subcommand (combined) ----
    pfull = subparsers.add_parser("full", help="Full workflow (2pt + OPE)")
    add_common_args(pfull)
    pfull.add_argument("--conf-file", type=str, default=None,
                        help="Gauge configuration file path")
    pfull.add_argument("--link-dir", type=str, default="z")
    pfull.add_argument("--delta-z", type=int, default=15)
    pfull.add_argument("--output-dir", type=str, default="./results")
    pfull.add_argument("--mode", type=str, default="unpol")
    pfull.add_argument("--element", type=str, default="_Cg5g4")
    pfull.add_argument("--pzlist", type=str, default=None)
    pfull.add_argument("--eig-dir", type=str, default=None)
    pfull.add_argument("--peram-u-dir", type=str, default=None)
    pfull.add_argument("--corr-nucl-dir", type=str, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Build config
    config = build_config_from_args(args, params_file=args.params_file)

    # Dispatch
    if args.command == "2pt":
        run_proton_2pt(config)
    elif args.command == "ope":
        config.setdefault("use_mpi", not args.no_mpi)
        config.setdefault("output_dir", "./ope_output")
        config.setdefault("link_dir", "z")
        if not config.get("conf_file"):
            parser.error("--conf-file is required for OPE calculation")
        run_ope(config)
    elif args.command == "pla":
        run_plaquette_only(config)
    elif args.command == "full":
        print("=" * 70)
        print("Running full gluon PDF workflow")
        print("=" * 70)
        # Step 1: OPE
        ope_config = dict(config)
        ope_config.setdefault("use_mpi", not args.no_mpi)
        ope_config.setdefault("output_dir", f"{args.output_dir}/ope")
        run_ope(ope_config)
        # Step 2: Proton 2pt
        pt2_config = dict(config)
        pt2_config.setdefault("corr_nucl_dir", f"{args.output_dir}/2pt")
        run_proton_2pt(pt2_config)

    # MPI finalize
    if HAS_MPI:
        comm.Barrier()


# ==============================================================================
# Section 11: Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    main()
