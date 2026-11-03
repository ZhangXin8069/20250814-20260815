#!/usr/bin/env python3
"""
Proton 2pt distillation — GPU (CuPy) accelerated, docker-v20260802.

MERGE NOTE (v20260802):
  - Uses v20260730's verified perambulator reading logic (shape (Nt,Nev,Nspin,Nev,2)).
  - Uses v20260730's eigenvector reshape (reshape to (Nev,Nv,Nc,2) then re+1j*im).
  - Adds v20260731's eigenvector format auto-detection (Nev_full from file size).
  - Ports missing time-slice graceful handling from v20260730.
  - Double precision (complex128) by default, configurable.
  - All version strings updated to v20260802.

Precision flow:
  Input (disk): eigenvector binary [LE f8] → CPU complex128 → GPU compute dtype
                perambulator binary [LE f8] → CPU complex128 → GPU compute dtype
  GPU compute:  complex128 (default) or complex64
  Output (disk): compute dtype .npy/.npz

Usage:
    python compute_2pt_gpu.py --run-dir /path/to/output
"""

from __future__ import annotations

import argparse, gc, json, os, sys, time
from pathlib import Path
from typing import Optional

import numpy as np
import cupy as cp

_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))
from utils import (
    Timer, print_banner, format_size, save_intermediate,
    validate_array, log_exception, get_current_memory_mb,
    get_gpu_memory_mb, to_cpu, to_gpu, gpu_sync, log_gpu_status,
    get_compute_dtype, get_compute_dtype_real,
)
from gamma_matrix_gpu import (
    get_gamma_cached, get_P_plus_cached, get_P_minus_cached,
    clear_cache as clear_gamma_cache,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Momentum phase factor
# ═══════════════════════════════════════════════════════════════════════════════

def compute_phase_factor_gpu(momentum: np.ndarray, Nx: int) -> cp.ndarray:
    """GPU phase factor: phi_P(x) = exp(-i * 2π * P·x / L).

    Args:
        momentum: 3-vector [Pz, Py, Px] in units of 2π/L.
        Nx: Spatial lattice extent (cubic: L = Nx).

    Returns:
        Flattened (Nx^3,) phase array on GPU, dtype=compute_dtype.
    """
    rtype = get_compute_dtype_real()
    ctype = get_compute_dtype()
    coords = cp.arange(Nx, dtype=rtype)
    Z, Y, X = cp.meshgrid(coords, coords, coords, indexing='ij')
    mom = cp.asarray(momentum, dtype=rtype)
    phase = cp.exp(cp.array(-1j, dtype=ctype) * cp.array(2.0 * cp.pi, dtype=rtype)
                   * (mom[0] * Z + mom[1] * Y + mom[2] * X) / cp.array(Nx, dtype=rtype))
    return phase.ravel()


# ═══════════════════════════════════════════════════════════════════════════════
# Eigenvector reader — per-config, per-time-slice BINARY files
# ═══════════════════════════════════════════════════════════════════════════════

def _read_eigenvector_slice(filepath: str, Nev: int, Nx: int) -> np.ndarray:
    """Read one eigenvector time-slice binary file.

    Binary format: little-endian float64, shape (Nev, Nx*Ny*Nz, Nc, 2)
    where the last dimension stores (real, imag) pairs.

    IMPLEMENTATION (v20260730 verified):
      reshape to (Nev, Nv, Nc, 2) then take [...,0] + 1j*[...,1].
      This is CORRECT for data stored as (re0, im0, re1, im1, ...) interleaved pairs.

    Args:
        filepath: Path to binary eigenvector file.
        Nev: Number of eigenvectors (first Nev are used).
        Nx: Spatial lattice extent.

    Returns:
        Complex numpy array of shape (Nev, Nx^3, 3), dtype=complex128.
    """
    raw = np.fromfile(filepath, dtype='<f8')
    Nc = 3
    Nv = Nx * Nx * Nx
    expected = Nev * Nv * Nc * 2
    if raw.size != expected:
        raise ValueError(f"Eigenvector {filepath}: expected {expected} floats, got {raw.size}")
    raw = raw.reshape(Nev, Nv, Nc, 2)  # (Nev, Nv, Nc, 2) — re/im interleaved in last dim
    return raw[..., 0] + 1j * raw[..., 1]  # → (Nev, Nv, Nc) complex128


def load_eigenvectors_per_config(
    eigvec_base: str, conf_id: int, Nev: int, Nt: int, Nx: int, logger
) -> np.ndarray:
    """Load eigenvectors for ONE config from per-time-slice binary files.

    Reads all Nt time slices from: {eigvec_base}/{conf_id}/eigvecs_t{tsrc:03d}_{conf_id}

    FEATURES:
      - Auto-detects Nev_full from first file size (v20260731 feature).
      - Graceful handling of missing time slices (v20260730 feature).
      - Reports |eig| range for sanity checking.

    Args:
        eigvec_base: Base path containing {conf_id}/ subdirectories.
        conf_id: Configuration ID.
        Nev: Number of eigenvectors to use (clamped to Nev_full).
        Nt, Nx: Lattice dimensions.
        logger: Logger instance.

    Returns:
        (Nt, Nev_use, Nx^3, 3) numpy array, dtype=compute_dtype.

    Raises:
        FileNotFoundError: If eigenvector directory is missing or >50% time slices missing.
    """
    eigvec_dir = os.path.join(eigvec_base, str(conf_id))
    compute_dt = get_compute_dtype()
    Nv = Nx * Nx * Nx

    logger.info(f"Loading eigenvectors for conf={conf_id} from {eigvec_dir}")
    t0 = time.perf_counter()

    if not os.path.isdir(eigvec_dir):
        raise FileNotFoundError(f"Eigenvector directory not found: {eigvec_dir}")

    # Auto-detect Nev_full from first file (v20260731 feature)
    first_file = os.path.join(eigvec_dir, f"eigvecs_t000_{conf_id}")
    if os.path.exists(first_file):
        raw = np.fromfile(first_file, dtype='<f8')
        Nev_full = (raw.size // 2) // (Nv * 3)  # 3 = Nc
        Nev_use = min(Nev, Nev_full)
        file_size_mb = os.path.getsize(first_file) / 1024**2
        total_est = file_size_mb * Nt * (Nev_use / Nev_full)
        logger.info(f"  Auto-detected: Nev_full={Nev_full}, using Nev={Nev_use}")
        logger.info(f"  First file: {file_size_mb:.1f} MB, est. total: ~{total_est:.1f} MB")
    else:
        raise FileNotFoundError(f"No eigenvector files found in {eigvec_dir}")

    # Pre-allocate and load time slice by time slice
    eigvecs = np.zeros((Nt, Nev_use, Nv, 3), dtype=compute_dt)
    missing = []

    for tsrc in range(Nt):
        fname = f"eigvecs_t{tsrc:03d}_{conf_id}"
        fpath = os.path.join(eigvec_dir, fname)
        if not os.path.exists(fpath):
            missing.append(tsrc)
            continue
        ev_slice = _read_eigenvector_slice(fpath, Nev_use, Nx)
        eigvecs[tsrc] = ev_slice.astype(compute_dt, copy=False)

    # Graceful missing-file handling (v20260730 feature)
    if missing:
        n_missing = len(missing)
        logger.warning(f"  Missing {n_missing}/{Nt} time slices: t={missing[:5]}...")
        if n_missing > Nt // 2:
            raise FileNotFoundError(
                f"Too many missing eigenvector files ({n_missing}/{Nt}) for conf={conf_id}"
            )

    elapsed = time.perf_counter() - t0
    mem_mb = eigvecs.nbytes / 1024**2
    logger.info(f"  Loaded: shape={eigvecs.shape}, dtype={eigvecs.dtype}, "
                f"mem={mem_mb:.1f} MB, time={elapsed:.1f}s")
    logger.info(f"  |eig| range: [{np.abs(eigvecs).min():.2e}, {np.abs(eigvecs).max():.2e}]")
    assert np.all(np.isfinite(eigvecs)), f"Eigenvectors conf={conf_id} contain NaN/inf"
    return eigvecs


# ═══════════════════════════════════════════════════════════════════════════════
# Perambulator reader — v20260730 CORRECTED binary format
# ═══════════════════════════════════════════════════════════════════════════════

def _read_perambulator_file(filepath: str, Nev: int, Nt: int) -> np.ndarray:
    """Read one perambulator file (for one (tsrc, dsrc) pair).

    CORRECTED binary format (v20260730, matches snsc/main.py convention):
      LE float64, shape (Nt_snk, Nev_snk, Nspin_snk=4, Nev_src, 2) → complex (Nt, Nev, 4, Nev).

    Verified: at t_snk=0 the matrix (Nev_snk, Nev_src) is diagonal-dominant,
    and at t_snk=36 it decays, confirming correct axis assignment.

    Args:
        filepath: Path to the perambulator binary file.
        Nev: Number of eigenvectors.
        Nt: Number of time slices.

    Returns:
        Complex numpy array of shape (Nt, Nev, 4, Nev), dtype=complex128.
    """
    raw = np.fromfile(filepath, dtype='<f8')
    Nspin = 4
    expected = Nt * Nev * Nspin * Nev * 2
    if raw.size != expected:
        raise ValueError(f"Perambulator {filepath}: expected {expected} floats, got {raw.size}")
    raw = raw.reshape(Nt, Nev, Nspin, Nev, 2)
    return raw[..., 0] + 1j * raw[..., 1]


def read_perambulator_single_t(
    peram_dir: str, conf_id: int, t_source: int,
    Nev: int, Nt: int, logger,
) -> np.ndarray:
    """Read perambulator for single t_src — CORRECTED to match snsc/main.py.

    For each t_source, reads 4 d_src files:
      perams.{conf_id}.{d_src}.{t_source}

    Each file: (Nt_snk, Nev_snk, Nspin_snk=4, Nev_src) complex128.
    Stacked: (Ndsrc=4, Nt, Nev_snk, Nspin=4, Nev_src).
    Transposed: (Nt, Nspin=4, Ndsrc=4, Nev_src, Nev_snk) — MATCHES snsc/main.py.

    Args:
        peram_dir: Directory containing perambulator files.
        conf_id: Configuration ID.
        t_source: Source time slice.
        Nev: Number of eigenvectors (all loaded, then truncated to Nev1).
        Nt: Number of time slices.
        logger: Logger instance.

    Returns:
        (Nt, 4, 4, Nev1, Nev1) numpy array, dtype=compute_dtype.
    """
    compute_dt = get_compute_dtype()
    Nev1 = Nev  # Use all eigenvectors (truncation done at VVV stage)

    parts = []
    for d_src in range(4):
        fn = os.path.join(peram_dir, f"perams.{conf_id}.{d_src}.{t_source}")
        if not os.path.exists(fn):
            raise FileNotFoundError(f"Peram not found: {fn}")
        peram_ds = _read_perambulator_file(fn, Nev, Nt)
        # peram_ds: (Nt, Nev_snk, Nspin=4, Nev_src) complex128
        parts.append(peram_ds)

    # Stack on d_src axis: (Ndsrc=4, Nt, Nev_snk, Nspin=4, Nev_src)
    # Transpose to: (Nt, Nspin=4, Ndsrc=4, Nev_src, Nev_snk)  — MATCHES snsc/main.py
    peram = np.stack(parts, axis=0)  # (4, Nt, Nev, 4, Nev)
    peram = peram.transpose(1, 3, 0, 4, 2)  # → (Nt, 4_spin, 4_dsrc, Nev_src, Nev_snk)
    peram = peram.astype(compute_dt, copy=False)
    peram = peram[:, :, :, :Nev1, :Nev1]
    assert np.all(np.isfinite(peram)), f"Peram (t={t_source}) contains NaN/inf"
    return peram


# ═══════════════════════════════════════════════════════════════════════════════
# VVV Baryon Block — GPU
# ═══════════════════════════════════════════════════════════════════════════════

def compute_vvv_single_t_gpu(
    eigvecs_t_gpu: cp.ndarray,      # (Nev, Nx^3, 3) — single time slice on GPU
    phase_factor_gpu: cp.ndarray,   # (Nx^3,) — momentum projection phase on GPU
    Nx: int, Nev1: int,
) -> cp.ndarray:
    """VVV baryon block on GPU, explicit two-step contraction.

    VVV_{abc} = Σ_x φ_a(x)·φ_b(x)·φ_c(x) · phase(x)

    Uses two-step einsum to avoid the large (Nev,Nev,Nev,Nx^3) intermediate
    from a naive single einsum call.

    Args:
        eigvecs_t_gpu: Eigvecs for one time slice, shape (Nev, Nx^3, 3), on GPU.
        phase_factor_gpu: Momentum phase, shape (Nx^3,), on GPU.
        Nx: Spatial lattice extent.
        Nev1: Number of eigenvectors to use in VVV (≤ Nev).

    Returns:
        VVV tensor, shape (Nev1, Nev1, Nev1), dtype=compute_dtype, on GPU.
    """
    VVV = cp.zeros((Nev1, Nev1, Nev1), dtype=get_compute_dtype())
    L = Nx * Nx  # Number of sites per x-slice

    # Loop over x-direction to limit intermediate tensor size to (Nev1,Nev1,Nx^2)
    for xi in range(Nx):
        s, e = xi * L, (xi + 1) * L
        es = eigvecs_t_gpu[:Nev1, s:e, :]  # (Nev1, Nx^2, 3)
        ps = phase_factor_gpu[s:e]          # (Nx^2,)

        # Even permutations (cyclic): ε_{abc} with a,b,c → b,c,a and c,a,b
        T = cp.einsum("x,ax,bx->abx", ps, es[..., 0], es[..., 1]); VVV += cp.einsum("abx,cx->abc", T, es[..., 2])
        T = cp.einsum("x,ax,bx->abx", ps, es[..., 1], es[..., 2]); VVV += cp.einsum("abx,cx->abc", T, es[..., 0])
        T = cp.einsum("x,ax,bx->abx", ps, es[..., 2], es[..., 0]); VVV += cp.einsum("abx,cx->abc", T, es[..., 1])

        # Odd permutations (anti-cyclic): swaps for antisymmetric contribution
        T = cp.einsum("x,ax,bx->abx", ps, es[..., 0], es[..., 2]); VVV -= cp.einsum("abx,cx->abc", T, es[..., 1])
        T = cp.einsum("x,ax,bx->abx", ps, es[..., 1], es[..., 0]); VVV -= cp.einsum("abx,cx->abc", T, es[..., 2])
        T = cp.einsum("x,ax,bx->abx", ps, es[..., 2], es[..., 1]); VVV -= cp.einsum("abx,cx->abc", T, es[..., 0])
    return VVV


def compute_vvv_all_t_gpu(
    eigvecs: np.ndarray,            # (Nt, Nev, Nx^3, 3) — all time slices on CPU
    phase_gpu: cp.ndarray,          # (Nx^3,) — TOTAL phase (projection) on GPU
    Nt: int, Nx: int, Nev1: int,
    logger,
) -> np.ndarray:
    """VVV for all time slices, streaming CPU→GPU per slice.

    Each time slice's eigenvectors are transferred to GPU, VVV is computed,
    and the result is transferred back to CPU before the next slice.

    Args:
        eigvecs: All eigenvectors, shape (Nt, Nev_use, Nx^3, 3), on CPU.
        phase_gpu: Momentum projection phase, shape (Nx^3,), on GPU.
        Nt, Nx: Lattice dimensions.
        Nev1: Number of eigenvectors used in VVV.
        logger: Logger instance.

    Returns:
        (Nt, Nev1, Nev1, Nev1) numpy array on CPU, dtype=compute_dtype.
    """
    logger.info(f"VVV (GPU, {get_compute_dtype()}): Nt={Nt}, Nev1={Nev1}, Nx={Nx}")
    t_start = time.perf_counter()
    gpu_mem = get_gpu_memory_mb()
    logger.info(f"  GPU free before: {gpu_mem['free_mb']:.0f} MB")

    compute_dt = get_compute_dtype()
    VVV_all = np.zeros((Nt, Nev1, Nev1, Nev1), dtype=compute_dt)

    for t in range(Nt):
        t1 = time.perf_counter()
        ev_t_gpu = cp.asarray(eigvecs[t])  # CPU → GPU
        vvv_gpu = compute_vvv_single_t_gpu(ev_t_gpu, phase_gpu, Nx, Nev1)
        VVV_all[t] = cp.asnumpy(vvv_gpu)   # GPU → CPU
        del ev_t_gpu, vvv_gpu

        if t % 12 == 0 or t == Nt - 1:
            gpu_mem = get_gpu_memory_mb()
            logger.debug(f"  VVV t={t:3d}/{Nt}  time={time.perf_counter()-t1:.2f}s  "
                         f"|VVV|_max={np.abs(VVV_all[t]).max():.4e}  GPU free={gpu_mem['free_mb']:.0f}MB")

    gpu_sync()
    elapsed = time.perf_counter() - t_start
    logger.info(f"VVV done (GPU) in {elapsed:.1f}s ({elapsed/Nt:.2f}s/slice)")
    logger.info(f"  |VVV| range: [{np.abs(VVV_all).min():.2e}, {np.abs(VVV_all).max():.2e}]")
    logger.info(f"  Memory: {VVV_all.nbytes/1024**2:.1f} MB ({VVV_all.dtype})")
    assert np.all(np.isfinite(VVV_all)), "VVV contains NaN/inf"
    return VVV_all


# ═══════════════════════════════════════════════════════════════════════════════
# Wick contraction + Parity projection — GPU
# ═══════════════════════════════════════════════════════════════════════════════

def compute_wick_and_project_gpu(
    VVV: np.ndarray,            # (Nt, Nev1, Nev1, Nev1) — on CPU
    peram_dir: str, conf_id: int,
    Nev: int, Nev1: int,
    Nt: int, element: str,
    logger,
) -> dict:
    """Wick contraction (Direct - Exchange) + parity projection on GPU.

    The disconnected proton 2pt function factorizes as:
      C_2pt = Tr[Γ_snk · S(x_snk; x_src) · Γ_src · S(x_src; x_snk)]
    where S is the perambulator and Γ are interpolation operators.

    For each (t_src, t_snk) pair with 2 ≤ dt ≤ 32:
      - Direct diagram: VVV(t_snk) · peram(t_snk) · cg5p(t_snk) · peram(t_snk) · VVV*(t_src)
      - Exchange diagram: similar with crossing
      - Raw correlator = Direct - Exchange

    Args:
        VVV: Baryon blocks, shape (Nt, Nev1, Nev1, Nev1), on CPU.
        peram_dir: Directory containing perambulator files.
        conf_id: Configuration ID.
        Nev: Total eigenvector count (for peram reading).
        Nev1: Truncated eigenvector count used in contractions.
        Nt: Number of time slices.
        element: Interpolation operator name (_Cg5g4, _Cg5g3, or _Cg5).
        logger: Logger instance.

    Returns:
        dict with keys: corr_raw (Nt,Nt,4,4), corr_pp (Nt,Nt), corr_pm (Nt,Nt).
    """
    logger.info(f"Wick contraction (GPU, {get_compute_dtype()}): "
                f"Nt={Nt}, Nev1={Nev1}, element={element}")

    compute_dt = get_compute_dtype()

    # Build interpolation operators on GPU
    G7 = get_gamma_cached(7)  # γ₇ = γ₃γ₁
    G4 = get_gamma_cached(4)  # γ₄

    if element == "_Cg5g4":
        ip1 = ip2 = G7 @ G4  # Cγ₅γ₄ interpolation operator
    elif element == "_Cg5g3":
        G3 = get_gamma_cached(3)
        ip1 = ip2 = G7 @ G3
    elif element == "_Cg5":
        ip1 = ip2 = G7
    else:
        ip1 = ip2 = G7 @ G4  # default

    # Parity projectors
    Pp = get_P_plus_cached()   # P₊ = (γ₀+γ₄)/2
    Pm = get_P_minus_cached()  # P₋ = (γ₀-γ₄)/2

    # Transfer VVV to GPU (stays there for entire contraction loop)
    VVV_gpu = cp.asarray(VVV)
    logger.info(f"  VVV on GPU: {VVV_gpu.nbytes/1024**2:.1f} MB")

    corr_raw = np.zeros((Nt, Nt, 4, 4), dtype=compute_dt)
    t0 = time.perf_counter()
    n_pairs = 0

    for t_src in range(Nt):
        t_s0 = time.perf_counter()
        VVV_src = VVV_gpu[t_src].conj()  # VVV* at source

        # Read perambulator for this source time slice
        peram_u = read_perambulator_single_t(peram_dir, conf_id, t_src, Nev, Nt, logger)
        peram_u_gpu = cp.asarray(peram_u)  # (Nt, 4, 4, Nev1, Nev1) on GPU

        # Pre-compute cg5p = ip1 · peram · ip2
        cg5p = cp.einsum("gh,thkbe->tgkbe", ip1, peram_u_gpu)
        cg5p = cp.einsum("tgkbe,jk->tgjbe", cg5p, ip2)

        for t_snk in range(Nt):
            dt = (t_snk - t_src + Nt) % Nt
            # Only compute pairs with 2 ≤ dt ≤ 32 (physical time separation)
            if not (2 <= dt <= 32):
                continue
            n_pairs += 1

            # Direct diagram: decomposed contraction chain
            T1 = cp.einsum("abc,gjad->gjbcd", VVV_gpu[t_snk], peram_u_gpu[t_snk])
            T2 = cp.einsum("gjbcd,gjbe->cde", T1, cg5p[t_snk])
            T3 = cp.einsum("cde,ilcf->ildef", T2, peram_u_gpu[t_snk])
            direct = cp.einsum("ildef,def->il", T3, VVV_src)

            # Exchange diagram: crossing in perambulator indices
            T1x = cp.einsum("abc,glaf->glbcf", VVV_gpu[t_snk], peram_u_gpu[t_snk])
            T2x = cp.einsum("glbcf,gjbe->ljcef", T1x, cg5p[t_snk])
            T3x = cp.einsum("ljcef,ijcd->ildef", T2x, peram_u_gpu[t_snk])
            exchange = cp.einsum("ildef,def->il", T3x, VVV_src)

            # Raw Wick contraction: Direct - Exchange
            corr_raw[t_snk, t_src] = cp.asnumpy(direct - exchange)

        del peram_u_gpu, cg5p
        if t_src % 10 == 0:
            gpu_mem = get_gpu_memory_mb()
            logger.debug(f"  Wick t_src={t_src:3d}/{Nt} done, {time.perf_counter()-t_s0:.1f}s, "
                         f"n_pairs={n_pairs}, GPU free={gpu_mem['free_mb']:.0f}MB")

    del VVV_gpu
    cp.get_default_memory_pool().free_all_blocks()

    elapsed = time.perf_counter() - t0
    logger.info(f"Wick done (GPU) in {elapsed:.1f}s, n_pairs={n_pairs}")
    logger.info(f"  Avg: {elapsed/n_pairs*1000:.1f} ms/pair")
    assert np.all(np.isfinite(corr_raw)), "Raw correlator contains NaN/inf"

    # Parity projection on GPU
    t_par = time.perf_counter()
    cr = cp.asarray(corr_raw)  # CPU → GPU
    corr_pp = cp.asnumpy(cp.einsum("li,yxil->yx", Pp, cr))  # P₊ projection
    corr_pm = cp.asnumpy(cp.einsum("li,yxil->yx", Pm, cr))  # P₋ projection
    del cr

    # Anti-periodic boundary condition fix for fermions
    # When t_snk wraps around the lattice (t_snk < t_src), the correlator
    # picks up a minus sign from anti-periodic BC in the time direction.
    for ts in range(Nt):
        for tk in range(Nt):
            if tk < ts:
                corr_pp[tk, ts] *= -1.0
            if tk > ts:
                corr_pm[tk, ts] *= -1.0

    logger.info(f"Parity projection done in {time.perf_counter()-t_par:.2f}s")
    logger.info(f"  PP range: [{corr_pp.real.min():.4e}, {corr_pp.real.max():.4e}]")
    logger.info(f"  PM range: [{corr_pm.real.min():.4e}, {corr_pm.real.max():.4e}]")
    return {"corr_raw": corr_raw, "corr_pp": corr_pp, "corr_pm": corr_pm}


# ═══════════════════════════════════════════════════════════════════════════════
# Effective mass — MULTIPLE METHODS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_effective_mass(
    corr_pp: np.ndarray,
    Nt: int,
    alttc: float = 0.1053,
    method: str = "fit_cosh",
    logger=None,
) -> dict:
    """Compute effective mass using multiple methods.

    Methods:
      - "cosh": Standard arccosh on source-averaged C(t).
      - "exp_forward": log(C(t)/C(t+1)) using forward-only time pairs.
      - "fit_cosh": Non-linear fit to A·cosh(m·(t-Nt/2)) → most robust.
      - "fit_exp": Single-exponential fit to forward time range.

    Args:
        corr_pp: Parity-projected correlator, shape (Nt, Nt), complex.
        Nt: Number of time slices.
        alttc: Lattice spacing in fm.
        method: Effective mass extraction method.
        logger: Logger instance.

    Returns:
        dict with keys: meff_gev (Nt-2,), C2pt_1d (Nt,), C2pt_forward (Nt,),
        meff_plateau_gev (float), meff_plateau_range [ps, pe], method.
    """
    fm2GeV = 0.1973  # conversion: fm⁻¹ → GeV

    # Source-averaged 1D correlator
    C2pt_1d = np.zeros(Nt, dtype=np.float64)
    C2pt_forward = np.zeros(Nt, dtype=np.float64)

    for dt in range(Nt):
        vals = []
        vals_fwd = []
        for t in range(Nt):
            t_snk = (t + dt) % Nt
            v = np.real(corr_pp[t_snk, t])
            if abs(v) > 1e-30:
                vals.append(v)
            if t_snk > t:
                vals_fwd.append(np.real(corr_pp[t_snk, t]))
        if vals:
            C2pt_1d[dt] = np.mean(vals)
        if vals_fwd:
            C2pt_forward[dt] = np.mean(vals_fwd)

    if logger:
        logger.info(f"C2pt_1d (source-avg) range: [{C2pt_1d[2:33].min():.4e}, {C2pt_1d[2:33].max():.4e}]")
        logger.info(f"C2pt_forward (fwd-only) range: [{C2pt_forward[2:33].min():.4e}, {C2pt_forward[2:33].max():.4e}]")

    meff_gev = np.full(Nt - 2, np.nan)

    # ── Method: cosh ──────────────────────────────────────────────────────
    if method == "cosh":
        C_pos = np.abs(C2pt_1d) + 1e-30
        cosh_arg = (C_pos[2:] + C_pos[:-2]) / (2.0 * C_pos[1:-1])
        valid = cosh_arg >= 1.0
        meff_gev[valid] = np.arccosh(np.minimum(cosh_arg[valid], 1e15)) * fm2GeV / alttc

    # ── Method: exp_forward ───────────────────────────────────────────────
    elif method == "exp_forward":
        for t in range(1, Nt - 1):
            if C2pt_forward[t] != 0 and C2pt_forward[t+1] != 0:
                ratio = abs(C2pt_forward[t] / (C2pt_forward[t+1] + 1e-30))
                if ratio > 1.0:
                    meff_gev[t-1] = np.log(ratio) * fm2GeV / alttc

    # ── Method: fit_cosh ──────────────────────────────────────────────────
    elif method == "fit_cosh":
        from scipy.optimize import curve_fit

        t_valid = np.arange(2, min(33, Nt))
        c_valid = C2pt_1d[2:min(33, Nt)]

        if len(t_valid) < 5:
            if logger:
                logger.warning("Too few valid time slices for fit_cosh")

        mask = np.abs(c_valid) > 1e-25
        t_fit = t_valid[mask]
        c_fit = np.abs(c_valid[mask])

        if len(t_fit) >= 5:
            try:
                def cosh_model(t, A, m):
                    return A * np.cosh(m * (t - Nt / 2.0))

                p0 = [np.abs(c_fit).max(), 0.5]
                popt, pcov = curve_fit(cosh_model, t_fit, c_fit, p0=p0,
                                       maxfev=10000, bounds=([1e-30, 0.01], [1e10, 5.0]))
                m_lattice = popt[1]
                m_phys = m_lattice * fm2GeV / alttc

                if logger:
                    logger.info(f"fit_cosh: A={popt[0]:.4e}, m_latt={m_lattice:.4f}, "
                               f"m_phys={m_phys:.3f} GeV")

                meff_gev[:] = m_phys
            except Exception as e:
                if logger:
                    logger.warning(f"fit_cosh failed: {e}, falling back to exp_forward")
                return compute_effective_mass(corr_pp, Nt, alttc, method="exp_forward", logger=logger)
        else:
            if logger:
                logger.warning("Too few valid points, falling back to exp_forward")
            return compute_effective_mass(corr_pp, Nt, alttc, method="exp_forward", logger=logger)

    # ── Method: fit_exp ───────────────────────────────────────────────────
    elif method == "fit_exp":
        from scipy.optimize import curve_fit

        t_fwd = np.arange(2, min(33, Nt))
        c_fwd = np.abs(C2pt_forward[2:min(33, Nt)])

        mask = c_fwd > 1e-25
        t_fit = t_fwd[mask]
        c_fit = c_fwd[mask]

        if len(t_fit) >= 4:
            try:
                def exp_model(t, A, m):
                    return A * np.exp(-m * t)

                p0 = [c_fit[0], 0.5]
                popt, pcov = curve_fit(exp_model, t_fit, c_fit, p0=p0, maxfev=10000)
                m_lattice = popt[1]
                m_phys = m_lattice * fm2GeV / alttc

                if logger:
                    logger.info(f"fit_exp: A={popt[0]:.4e}, m_latt={m_lattice:.4f}, "
                               f"m_phys={m_phys:.3f} GeV")

                meff_gev[:] = m_phys
            except Exception as e:
                if logger:
                    logger.warning(f"fit_exp failed: {e}")
                return compute_effective_mass(corr_pp, Nt, alttc, method="exp_forward", logger=logger)

    # ── Plateau estimate ──────────────────────────────────────────────────
    ps, pe = Nt // 4, Nt // 2  # plateau region: [t=Nt/4, t=Nt/2]
    pmask = ~np.isnan(meff_gev[ps:pe])
    meff_plateau = float(np.mean(meff_gev[ps:pe][pmask])) if np.any(pmask) else np.nan

    if method in ("fit_cosh", "fit_exp") and not np.isnan(meff_gev[0]):
        meff_plateau = float(meff_gev[0])

    if logger:
        logger.info(f"Effective mass (method={method}, a={alttc} fm):")
        for t in range(1, min(Nt - 1, 16)):
            logger.info(f"  t={t:3d}  m_eff={meff_gev[t-1]:.6f} GeV  "
                       f"C_fwd={C2pt_forward[t]:.4e}  C_avg={C2pt_1d[t]:.4e}")
        logger.info(f"  Plateau [{ps},{pe}]: m_eff={meff_plateau:.4f} GeV")

    return {"meff_gev": meff_gev, "C2pt_1d": C2pt_1d,
            "C2pt_forward": C2pt_forward,
            "meff_plateau_gev": meff_plateau,
            "meff_plateau_range": [ps, pe],
            "method": method}


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_2pt_computation_gpu(config: dict, output_dir: Path, logger) -> dict:
    """Full 2pt distillation with GPU acceleration (v20260802 merged version).

    Args:
        config: Full pipeline config dict.
        output_dir: Base data output directory.
        logger: Logger instance.

    Returns:
        dict mapping conf_id → results with status and per-Pz details.
    """
    params = config["parameters"]
    paths = config["data_paths"]
    ensemble = config["ensemble"]

    Nt, Nx = ensemble["Nt"], ensemble["Nx"]
    Nev, Nev1 = params["Nev"], params["Nev1"]
    Pz_list = params.get("Pz_list", [params["Pz"]])
    Px, Py = params["Px"], params["Py"]
    element = params["element"]
    conf_ids = params["conf_ids"]
    alttc = ensemble["alttc"]

    # Smearing control
    apply_smear = params.get("apply_eigenvec_smearing", False)
    mom_smear_phase = params.get("mom_smear_phase", 0) if apply_smear else 0
    meff_method = params.get("meff_method", "fit_cosh")

    eigvec_base = paths["eigenvector_base"]
    # perambulator_base does NOT include /light/ — code appends it
    peram_base = paths["perambulator_base"]

    print_banner("Step 01: Proton 2pt Distillation (GPU, v20260802)", logger)
    log_gpu_status(logger, "  ")
    logger.info(f"  Compute dtype: {get_compute_dtype()}")
    logger.info(f"  Ensemble: {ensemble['full_name']} | {Nt}x{Nx}^3 | a={alttc} fm")
    logger.info(f"  Nev={Nev}, Nev1={Nev1}, element={element}")
    logger.info(f"  Momentum: P=({Px},{Py}), Pz in {Pz_list}")
    logger.info(f"  Configs: {conf_ids}")
    logger.info(f"  Eigenvector base: {eigvec_base} (per-config, per-time-slice)")
    logger.info(f"  Perambulator base: {peram_base} (light/{{conf_id}}/)")
    logger.info(f"  Eigenvector smearing: {apply_smear} (phase={mom_smear_phase})")
    logger.info(f"  Effective mass method: {meff_method}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for iconf, conf_id in enumerate(conf_ids):
        conf_dir = output_dir / f"conf_{conf_id}"
        conf_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*60}")
        logger.info(f"  Config {iconf+1}/{len(conf_ids)}: conf_id={conf_id}")
        logger.info(f"{'='*60}")

        # ── Load eigenvectors for THIS config ──────────────────────────────
        with Timer(f"load_eigvecs_conf{conf_id}", logger, output_dir.parent,
                  extra={"conf_id": conf_id}):
            eigvecs = load_eigenvectors_per_config(
                eigvec_base, conf_id, Nev, Nt, Nx, logger)
            logger.info(f"  Eigvecs: shape={eigvecs.shape}, dtype={eigvecs.dtype}, "
                       f"mem={eigvecs.nbytes/1024**2:.1f}MB")

        # ── Pre-smear eigenvectors if requested ────────────────────────────
        eigvecs = np.array(eigvecs, copy=True)  # Ensure writable
        if apply_smear and mom_smear_phase != 0:
            smear_mom = np.array([mom_smear_phase, 0, 0])
            phase_smear_gpu = compute_phase_factor_gpu(smear_mom, Nx)
            logger.info(f"Pre-smearing eigenvectors: P_smear=({mom_smear_phase},0,0)")
            t_smear = time.perf_counter()
            for t in range(Nt):
                ev_gpu = cp.asarray(eigvecs[t])
                ev_smeared = cp.einsum("vxa,x->vxa", ev_gpu, phase_smear_gpu)
                eigvecs[t] = cp.asnumpy(ev_smeared)
                del ev_gpu, ev_smeared
            cp.get_default_memory_pool().free_all_blocks()
            logger.info(f"Eigenvector smearing done in {time.perf_counter()-t_smear:.1f}s")
        else:
            logger.info("Eigenvector smearing DISABLED (perambulator encodes smearing)")

        # ── Check perambulator directory ───────────────────────────────────
        peram_dir = os.path.join(peram_base, "light", str(conf_id))
        first_peram = os.path.join(peram_dir, f"perams.{conf_id}.0.0")
        if not os.path.exists(first_peram):
            logger.error(f"  [SKIP] conf={conf_id}: peram not found at {first_peram}")
            all_results[conf_id] = {"status": "skipped", "reason": "peram not found"}
            continue

        n_perams = len([f for f in os.listdir(peram_dir)
                       if f.startswith(f"perams.{conf_id}")])
        logger.info(f"  Peram dir: {peram_dir} ({n_perams} files)")

        conf_results = {}

        for Pz in Pz_list:
            Mom = np.array([Pz, Py, Px])
            logger.info(f"\n  --- P=({Px},{Py},{Pz}) ---")

            # ── Compute VVV phase ─────────────────────────────────────────
            phase_P_gpu = compute_phase_factor_gpu(Mom, Nx)

            # VVV cache: skip recomputation if already computed
            smear_tag = f"_smear{mom_smear_phase}" if apply_smear else "_nosmear"
            vvv_cache = conf_dir / f"VVV_Nev1{Nev1}_Px{Px}Py{Py}Pz{Pz}{smear_tag}_conf{conf_id}.npy"

            if vvv_cache.exists():
                logger.info(f"  [CACHE] Loading VVV from {vvv_cache.name}")
                VVV = np.load(vvv_cache)
                logger.info(f"  VVV shape: {VVV.shape}, dtype: {VVV.dtype}, "
                           f"|VVV| max: {np.abs(VVV).max():.4e}")
            else:
                with Timer(f"VVV_GPU_Pz{Pz}_conf{conf_id}", logger, output_dir.parent,
                          extra={"conf_id": conf_id, "Pz": Pz, "device": "gpu",
                                 "dtype": str(get_compute_dtype())}):
                    VVV = compute_vvv_all_t_gpu(
                        eigvecs, phase_P_gpu, Nt, Nx, Nev1, logger)
                np.save(vvv_cache, VVV)
                logger.info(f"  [SAVE] VVV: {vvv_cache.name} "
                           f"({vvv_cache.stat().st_size/1024**2:.1f} MB, {VVV.dtype})")

            # ── Wick contraction ──────────────────────────────────────────
            with Timer(f"Wick_GPU_Pz{Pz}_conf{conf_id}", logger, output_dir.parent,
                      extra={"conf_id": conf_id, "Pz": Pz, "device": "gpu",
                             "dtype": str(get_compute_dtype())}):
                corr_dict = compute_wick_and_project_gpu(
                    VVV, peram_dir, conf_id, Nev, Nev1, Nt, element, logger)

            # ── Save results ──────────────────────────────────────────────
            base = f"Px{Px}Py{Py}Pz{Pz}_eginphase{abs(params.get('mom_smear', 0))}{element}"

            raw_fn = f"twopt_slice_pp_{base}_contract_conf{conf_id}.npy"
            np.save(conf_dir / raw_fn, corr_dict["corr_raw"])
            logger.info(f"  [SAVE] Raw: {raw_fn} ({corr_dict['corr_raw'].dtype})")

            pp_fn = f"twopt_slice_pp_{base}_nopol_ss_conf{conf_id}.npy"
            np.save(conf_dir / pp_fn, corr_dict["corr_pp"])
            logger.info(f"  [SAVE] PP: {pp_fn} ({corr_dict['corr_pp'].dtype})")

            pm_fn = f"twopt_slice_pm_{base}_nopol_ss_conf{conf_id}.npy"
            np.save(conf_dir / pm_fn, corr_dict["corr_pm"])
            logger.info(f"  [SAVE] PM: {pm_fn} ({corr_dict['corr_pm'].dtype})")

            # ── Effective mass ────────────────────────────────────────────
            with Timer(f"meff_Pz{Pz}_conf{conf_id}", logger, output_dir.parent,
                      extra={"conf_id": conf_id, "Pz": Pz}):
                meff_results = {}
                for method_name in ["fit_cosh", "fit_exp", "exp_forward", "cosh"]:
                    try:
                        meff_results[method_name] = compute_effective_mass(
                            corr_dict["corr_pp"], Nt, alttc, method=method_name, logger=logger)
                    except Exception as e:
                        logger.warning(f"  meff method {method_name} failed: {e}")

                meff_result = meff_results.get(meff_method, list(meff_results.values())[0])

            np.savez(conf_dir / f"meff_Pz{Pz}_conf{conf_id}.npz",
                     meff_gev=meff_result["meff_gev"],
                     C2pt_1d=meff_result["C2pt_1d"],
                     C2pt_forward=meff_result.get("C2pt_forward", np.zeros(Nt)),
                     meff_plateau_gev=meff_result["meff_plateau_gev"],
                     meff_plateau_range=np.array(meff_result["meff_plateau_range"]),
                     method=np.array(meff_method))

            logger.info(f"  Effective mass results (a={alttc} fm):")
            for method_name, res in meff_results.items():
                logger.info(f"    {method_name:15s}: plateau={res['meff_plateau_gev']:.4f} GeV")

            conf_results[Pz] = {
                "corr_pp_shape": list(corr_dict["corr_pp"].shape),
                "corr_pp_range_re": [float(corr_dict["corr_pp"].real.min()),
                                     float(corr_dict["corr_pp"].real.max())],
                "corr_pp_range_im": [float(corr_dict["corr_pp"].imag.min()),
                                     float(corr_dict["corr_pp"].imag.max())],
                "meff_plateau_gev": meff_result["meff_plateau_gev"],
                "meff_plateau_range": meff_result["meff_plateau_range"],
                "meff_method": meff_method,
                "dtype": str(corr_dict["corr_pp"].dtype),
                "all_meff_methods": {m: r["meff_plateau_gev"]
                                     for m, r in meff_results.items()
                                     if not np.isnan(r["meff_plateau_gev"])},
            }

        save_intermediate({"conf_id": conf_id, "status": "ok", "Pz_list": Pz_list,
                          "results": {str(Pz): v for Pz, v in conf_results.items()}},
                         conf_dir, "compute_2pt_summary.json", logger)

        all_results[conf_id] = {"status": "ok", "results": conf_results,
                                "n_peram_files": n_perams}
        logger.info(f"  [PROGRESS] 2pt: {iconf+1}/{len(conf_ids)} done")

        # Free memory between configs
        del eigvecs
        if 'VVV' in dir():
            del VVV
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info(f"\n{'═'*60}")
    logger.info(f"2pt (GPU, {get_compute_dtype()}) Summary:")
    for conf_id, result in all_results.items():
        if result.get("status") == "ok":
            for Pz, r in result["results"].items():
                logger.info(f"  conf={conf_id} Pz={Pz}: PP range={r['corr_pp_range_re']}, "
                           f"meff={r['meff_plateau_gev']:.4f} GeV (method={r.get('meff_method','?')})")
                if 'all_meff_methods' in r:
                    logger.info(f"    All methods: {r['all_meff_methods']}")
        else:
            logger.info(f"  conf={conf_id}: {result.get('status','?')}")
    logger.info(f"{'═'*60}")
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Proton 2pt distillation (GPU, v20260802)")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    config_path = args.config or (run_dir / "run_config.json")
    with open(config_path) as f:
        config = json.load(f)

    output_dir = run_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    from utils import setup_logging
    logger = setup_logging(run_dir / "run.log", "compute_2pt_gpu")
    results = run_2pt_computation_gpu(config, output_dir, logger)

    summary_path = output_dir / "compute_2pt_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Summary saved to {summary_path}")

    all_ok = all(r["status"] == "ok" for r in results.values())
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
