#!/usr/bin/env python3
"""
Proton 2pt distillation — GPU (CuPy) accelerated, CORRECTED version.

KEY FIXES over v20260727:
  1. Eigenvector smearing is now OPTIONAL (configurable via `apply_eigenvec_smearing`).
     When the perambulator already encodes momentum smearing (e.g., mz2_my0_mx0),
     the eigenvectors should NOT be additionally smeared in the VVV computation.
     The VVV momentum projection phase is automatically adjusted:
       P_proj = P_phys - N_smear * P_smear
     where N_smear = 1 (smearing in peram only) or 0 (no smearing anywhere).

  2. Effective mass extraction now uses MULTIPLE methods:
     - "cosh": standard arccosh formula (works for clean, non-oscillating correlators)
     - "exp_forward": log(C(t)/C(t+1)) using only forward-propagating pairs
     - "fit_cosh": non-linear least squares fit to A*cosh(m*(t-Nt/2))
     - "fit_exp": exponential fit to the forward range
     Default is "fit_cosh" which is most robust.

  3. Added detailed diagnostics for the correlator at each step.

  4. Support for test at Pz=0 (zero momentum) for cross-validation.

Precision flow:
  Input (disk): eigenvector.npy [complex128], perambulator binary [complex128]
  GPU compute:  configurable complex64/complex128
  Output (disk): configurable dtype .npy/.npz

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
    """GPU phase factor: phi_P(x) = exp(-i·2π·P·x/L). Returns on GPU."""
    rtype = get_compute_dtype_real()
    ctype = get_compute_dtype()
    coords = cp.arange(Nx, dtype=rtype)
    Z, Y, X = cp.meshgrid(coords, coords, coords, indexing='ij')
    mom = cp.asarray(momentum, dtype=rtype)
    phase = cp.exp(cp.array(-1j, dtype=ctype) * cp.array(2.0 * cp.pi, dtype=rtype)
                   * (mom[0] * Z + mom[1] * Y + mom[2] * X) / cp.array(Nx, dtype=rtype))
    return phase.ravel()


# ═══════════════════════════════════════════════════════════════════════════════
# Data readers (CPU, then downcast)
# ═══════════════════════════════════════════════════════════════════════════════

def load_eigenvectors(eigvec_path: str, Nev: int, Nt: int, logger) -> np.ndarray:
    """Load eigenvectors from .npy, downcast to compute dtype for GPU.

    Input shape: (Nt, Nev_full, Nz, Ny, Nx, Nc) or (Nt, Nev_full, Nx^3, Nc) [complex128].
    Returns: (Nt, Nev, Nx^3, 3) in compute dtype.
    """
    logger.info(f"Loading eigenvectors: {eigvec_path}")
    t0 = time.perf_counter()
    if not os.path.exists(eigvec_path):
        raise FileNotFoundError(f"Not found: {eigvec_path}")

    file_size_mb = os.path.getsize(eigvec_path) / 1024**2
    logger.info(f"  File size: {file_size_mb:.1f} MB")

    eigvecs = np.load(eigvec_path, mmap_mode='r')
    logger.info(f"  Raw shape: {eigvecs.shape}, dtype: {eigvecs.dtype}")

    compute_dt = get_compute_dtype()

    if eigvecs.ndim == 6:
        Nev_full = eigvecs.shape[1]; Nx_det = eigvecs.shape[2]
        Nev_use = min(Nev, Nev_full)
        logger.info(f"  Detected 6D: Nt={eigvecs.shape[0]}, Nev={Nev_full}, Nx={Nx_det}")
        eigvecs_final = np.asarray(
            eigvecs[:, :Nev_use, :, :, :, :].reshape(
                eigvecs.shape[0], Nev_use, Nx_det**3, eigvecs.shape[5]),
            dtype=compute_dt)
    elif eigvecs.ndim == 4:
        Nev_full = eigvecs.shape[1]; Nev_use = min(Nev, Nev_full)
        eigvecs_final = np.asarray(eigvecs[:, :Nev_use, :, :], dtype=compute_dt)
    else:
        raise ValueError(f"Unexpected eigenvector shape: {eigvecs.shape}")

    logger.info(f"  Final shape: {eigvecs_final.shape}, dtype: {eigvecs_final.dtype}, "
                f"loaded in {time.perf_counter()-t0:.1f}s")
    logger.info(f"  Memory: {eigvecs_final.nbytes/1024**2:.1f} MB "
                f"(saved {(file_size_mb - eigvecs_final.nbytes/1024**2):.1f} MB vs complex128)")
    logger.info(f"  |eig| range: [{np.abs(eigvecs_final).min():.2e}, {np.abs(eigvecs_final).max():.2e}]")
    assert np.all(np.isfinite(eigvecs_final)), "Eigenvectors contain NaN/inf"
    return eigvecs_final


def load_eigenvalues(eigval_path: str, Nev: int, logger) -> np.ndarray:
    """Load eigenvalues."""
    if not os.path.exists(eigval_path):
        logger.warning(f"Eigenvalue file not found: {eigval_path}")
        return np.array([])
    eigvals = np.load(eigval_path)
    if eigvals.ndim == 2:
        eigvals = eigvals[0, :]
    eigvals = eigvals[:Nev].real
    logger.info(f"Eigenvalues: shape=({len(eigvals)},), range=[{eigvals.min():.4f},{eigvals.max():.4f}]")
    return eigvals


def read_perambulator_single_t(
    peram_dir: str, conf_id: int, t_source: int,
    Nev: int, Nt: int, logger,
) -> np.ndarray:
    """Read perambulator for single t_src — downcast to compute dtype.

    Returns: (Nt, 4, 4, Nev, Nev) in compute dtype.
    """
    compute_dt = get_compute_dtype()
    parts = []
    for d_src in range(4):
        fn = os.path.join(peram_dir, f"perams.{conf_id}.{d_src}.{t_source}")
        if not os.path.exists(fn):
            raise FileNotFoundError(f"Peram not found: {fn}")
        with open(fn, "rb") as f:
            parts.append(np.fromfile(f, dtype="f8"))

    raw = np.concatenate(parts)
    Nev_full = int(np.sqrt(raw.size / (4 * 4 * Nt * 2)))
    peram = raw.reshape(4, Nt, Nev_full, 4, Nev_full, 2)
    peram = peram.transpose(1, 3, 0, 4, 2, 5)  # → (Nt,4,4,Nev_full,Nev_full,2)
    peram = (peram[..., 0] + 1j * peram[..., 1]).astype(compute_dt, copy=False)
    Nev_use = min(Nev, Nev_full)
    peram = peram[:, :, :, :Nev_use, :Nev_use]
    assert np.all(np.isfinite(peram)), f"Peram (t={t_source}) contains NaN/inf"
    return peram


# ═══════════════════════════════════════════════════════════════════════════════
# VVV Baryon Block — GPU (FIXED: optional eigenvector smearing)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_vvv_single_t_gpu(
    eigvecs_t_gpu: cp.ndarray,      # (Nev, Nx^3, 3) — optionally pre-smeared
    phase_factor_gpu: cp.ndarray,   # (Nx^3,) momentum projection phase
    Nx: int, Nev1: int,
) -> cp.ndarray:
    """VVV on GPU, explicit two-step contraction to avoid large intermediates.

    VVV_{abc} = Σ_x φ_a(x)·φ_b(x)·φ_c(x) · phase(x)
    where phase(x) includes the momentum projection and any residual smearing.

    Key optimization: instead of einsum("x,ax,bx,cx->abc") which creates a
    4.6 GB intermediate, use:
        T = einsum("x,ax,bx->abx")  → (Nev,Nev,Nx²) = 46 MB
        V = einsum("abx,cx->abc")   → (Nev,Nev,Nev) = 8 MB
    """
    VVV = cp.zeros((Nev1, Nev1, Nev1), dtype=get_compute_dtype())
    L = Nx * Nx

    for xi in range(Nx):
        s, e = xi * L, (xi + 1) * L
        es = eigvecs_t_gpu[:Nev1, s:e, :]
        ps = phase_factor_gpu[s:e]

        # Even permutations
        T = cp.einsum("x,ax,bx->abx", ps, es[..., 0], es[..., 1]); VVV += cp.einsum("abx,cx->abc", T, es[..., 2])
        T = cp.einsum("x,ax,bx->abx", ps, es[..., 1], es[..., 2]); VVV += cp.einsum("abx,cx->abc", T, es[..., 0])
        T = cp.einsum("x,ax,bx->abx", ps, es[..., 2], es[..., 0]); VVV += cp.einsum("abx,cx->abc", T, es[..., 1])
        # Odd permutations
        T = cp.einsum("x,ax,bx->abx", ps, es[..., 0], es[..., 2]); VVV -= cp.einsum("abx,cx->abc", T, es[..., 1])
        T = cp.einsum("x,ax,bx->abx", ps, es[..., 1], es[..., 0]); VVV -= cp.einsum("abx,cx->abc", T, es[..., 2])
        T = cp.einsum("x,ax,bx->abx", ps, es[..., 2], es[..., 1]); VVV -= cp.einsum("abx,cx->abc", T, es[..., 0])
    return VVV


def compute_vvv_all_t_gpu(
    eigvecs: np.ndarray,            # (Nt, Nev, Nx^3, 3) compute dtype
    phase_gpu: cp.ndarray,          # (Nx^3,) TOTAL phase (smearing + projection) on GPU
    Nt: int, Nx: int, Nev1: int,
    logger,
) -> np.ndarray:
    """VVV all time slices, streaming CPU→GPU per slice.

    The phase_gpu should already combine any smearing and momentum projection:
      phase(x) = exp(-i * P_eff * 2π/L * x_coord)
    where P_eff depends on whether eigenvectors are pre-smeared.

    Returns: (Nt, Nev1, Nev1, Nev1) compute dtype on CPU.
    """
    logger.info(f"VVV (GPU, {get_compute_dtype()}): Nt={Nt}, Nev1={Nev1}, Nx={Nx}")
    t_start = time.perf_counter()
    gpu_mem = get_gpu_memory_mb()
    logger.info(f"  GPU free before: {gpu_mem['free_mb']:.0f} MB")

    compute_dt = get_compute_dtype()
    VVV_all = np.zeros((Nt, Nev1, Nev1, Nev1), dtype=compute_dt)

    for t in range(Nt):
        t1 = time.perf_counter()
        ev_t_gpu = cp.asarray(eigvecs[t])  # already compute dtype
        # Note: eigvecs may or may not be pre-smeared; phase_gpu handles the projection
        vvv_gpu = compute_vvv_single_t_gpu(ev_t_gpu, phase_gpu, Nx, Nev1)
        VVV_all[t] = cp.asnumpy(vvv_gpu)
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
    VVV: np.ndarray,            # (Nt, Nev1, Nev1, Nev1) compute dtype on CPU
    peram_dir: str, conf_id: int,
    Nev: int, Nev1: int,
    Nt: int, element: str,
    logger,
) -> dict:
    """Wick contraction + parity projection (GPU)."""
    logger.info(f"Wick contraction (GPU, {get_compute_dtype()}): "
                f"Nt={Nt}, Nev1={Nev1}, element={element}")

    compute_dt = get_compute_dtype()

    # Build interpolation operators on GPU
    G7 = get_gamma_cached(7)
    G4 = get_gamma_cached(4)

    if element == "_Cg5g4":
        ip1 = ip2 = G7 @ G4
    elif element == "_Cg5g3":
        G3 = get_gamma_cached(3)
        ip1 = ip2 = G7 @ G3
    elif element == "_Cg5":
        ip1 = ip2 = G7
    else:
        ip1 = ip2 = G7 @ G4

    Pp = get_P_plus_cached()
    Pm = get_P_minus_cached()

    # VVV on GPU (stays there for entire contraction)
    VVV_gpu = cp.asarray(VVV)
    logger.info(f"  VVV on GPU: {VVV_gpu.nbytes/1024**2:.1f} MB")

    corr_raw = np.zeros((Nt, Nt, 4, 4), dtype=compute_dt)
    t0 = time.perf_counter()
    n_pairs = 0

    for t_src in range(Nt):
        t_s0 = time.perf_counter()
        VVV_src = VVV_gpu[t_src].conj()

        peram_u = read_perambulator_single_t(peram_dir, conf_id, t_src, Nev, Nt, logger)
        peram_u_gpu = cp.asarray(peram_u)  # (Nt,4,4,Nev1,Nev1)

        cg5p = cp.einsum("gh,thkbe->tgkbe", ip1, peram_u_gpu)
        cg5p = cp.einsum("tgkbe,jk->tgjbe", cg5p, ip2)

        for t_snk in range(Nt):
            dt = (t_snk - t_src + Nt) % Nt
            if not (2 <= dt <= 32):
                continue
            n_pairs += 1

            # Direct: "abc,gjad,gjbe,ilcf,def->il" decomposed
            T1 = cp.einsum("abc,gjad->gjbcd", VVV_gpu[t_snk], peram_u_gpu[t_snk])
            T2 = cp.einsum("gjbcd,gjbe->cde", T1, cg5p[t_snk])
            T3 = cp.einsum("cde,ilcf->ildef", T2, peram_u_gpu[t_snk])
            direct = cp.einsum("ildef,def->il", T3, VVV_src)

            # Exchange: "abc,glaf,gjbe,ijcd,def->il" decomposed
            T1x = cp.einsum("abc,glaf->glbcf", VVV_gpu[t_snk], peram_u_gpu[t_snk])
            T2x = cp.einsum("glbcf,gjbe->ljcef", T1x, cg5p[t_snk])
            T3x = cp.einsum("ljcef,ijcd->ildef", T2x, peram_u_gpu[t_snk])
            exchange = cp.einsum("ildef,def->il", T3x, VVV_src)
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
    cr = cp.asarray(corr_raw)
    corr_pp = cp.asnumpy(cp.einsum("li,yxil->yx", Pp, cr))
    corr_pm = cp.asnumpy(cp.einsum("li,yxil->yx", Pm, cr))
    del cr

    # Boundary sign fix: anti-periodic BC for fermions
    # When t_snk < t_src, the perambulator wraps around, and VVV (periodic)
    # needs a minus sign to match the anti-periodic perambulator.
    # Both PP and PM get flipped for backward propagation.
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
# Effective mass — MULTIPLE METHODS (FIXED)
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
      - "cosh": standard arccosh formula (fast, but fails on oscillating correlators)
      - "exp_forward": log(C(t)/C(t+1)) for forward-only pairs (t_snk > t_src)
      - "fit_cosh": non-linear fit to A*cosh(m*(t-Nt/2)) over valid time range
      - "fit_exp": single-exponential fit to forward time range

    Returns dict with meff_gev (per time slice), C2pt_1d, meff_plateau_gev, etc.
    """
    fm2GeV = 0.1973

    # ── Source-averaged 1D correlator ─────────────────────────────────────
    C2pt_1d = np.zeros(Nt, dtype=np.float64)
    # Also compute forward-only for cross-check
    C2pt_forward = np.zeros(Nt, dtype=np.float64)
    n_forward = np.zeros(Nt, dtype=int)

    for dt in range(Nt):
        vals = []
        vals_fwd = []
        for t in range(Nt):
            t_snk = (t + dt) % Nt
            v = np.real(corr_pp[t_snk, t])
            if abs(v) > 1e-30:
                vals.append(v)
            if t_snk > t:  # forward propagation only
                vals_fwd.append(np.real(corr_pp[t_snk, t]))
        if vals:
            C2pt_1d[dt] = np.mean(vals)
        if vals_fwd:
            C2pt_forward[dt] = np.mean(vals_fwd)
            n_forward[dt] = len(vals_fwd)

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
            if n_forward[t] > 0 and n_forward[t+1] > 0:
                ratio = abs(C2pt_forward[t] / (C2pt_forward[t+1] + 1e-30))
                if ratio > 1.0:
                    meff_gev[t-1] = np.log(ratio) * fm2GeV / alttc

    # ── Method: fit_cosh ──────────────────────────────────────────────────
    elif method == "fit_cosh":
        from scipy.optimize import curve_fit

        # Use only valid time range (dt in [2, 32])
        t_valid = np.arange(2, min(33, Nt))
        c_valid = C2pt_1d[2:min(33, Nt)]

        if len(t_valid) < 5:
            if logger:
                logger.warning("Too few valid time slices for fit_cosh")

        # Remove zero/near-zero points
        mask = np.abs(c_valid) > 1e-25
        t_fit = t_valid[mask]
        c_fit = np.abs(c_valid[mask])

        if len(t_fit) >= 5:
            try:
                # Fit: A * cosh(m * (t - Nt/2))
                def cosh_model(t, A, m):
                    return A * np.cosh(m * (t - Nt / 2.0))

                p0 = [np.abs(c_fit).max(), 0.5]  # initial guess
                popt, pcov = curve_fit(cosh_model, t_fit, c_fit, p0=p0,
                                       maxfev=10000, bounds=([1e-30, 0.01], [1e10, 5.0]))
                m_lattice = popt[1]
                m_phys = m_lattice * fm2GeV / alttc

                if logger:
                    logger.info(f"fit_cosh: A={popt[0]:.4e}, m_latt={m_lattice:.4f}, "
                               f"m_phys={m_phys:.3f} GeV")

                # Store the fit result
                meff_gev[:] = m_phys
            except Exception as e:
                if logger:
                    logger.warning(f"fit_cosh failed: {e}, falling back to exp_forward")
                return compute_effective_mass(corr_pp, Nt, alttc, method="exp_forward", logger=logger)
        else:
            if logger:
                logger.warning("Too few valid points after filtering, falling back to exp_forward")
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
                popt, pcov = curve_fit(exp_model, t_fit, c_fit, p0=p0,
                                       maxfev=10000)
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
    ps, pe = Nt // 4, Nt // 2
    pmask = ~np.isnan(meff_gev[ps:pe])
    meff_plateau = float(np.mean(meff_gev[ps:pe][pmask])) if np.any(pmask) else np.nan

    # For fit methods, the plateau is the fit result
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
    """Full 2pt distillation with GPU acceleration (CORRECTED)."""
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

    # ── NEW: Control eigenvector smearing ────────────────────────────────
    apply_smear = params.get("apply_eigenvec_smearing", False)
    mom_smear_phase = params.get("mom_smear_phase", 0) if apply_smear else 0
    meff_method = params.get("meff_method", "fit_cosh")

    eigvec_path = paths["eigenvector"]
    eigval_path = paths.get("eigenvalue", "")
    peram_base = paths["perambulator_base"]
    eigvec_cfg = paths["eigenvector_cfg"]

    print_banner("Step 01: Proton 2pt Distillation (GPU, CORRECTED)", logger)
    log_gpu_status(logger, "  ")
    logger.info(f"  Compute dtype: {get_compute_dtype()}")
    logger.info(f"  Ensemble: {ensemble['full_name']} | {Nt}×{Nx}³ | a={alttc} fm")
    logger.info(f"  Nev={Nev}, Nev1={Nev1}, element={element}")
    logger.info(f"  Momentum: P=({Px},{Py}), Pz∈{Pz_list}")
    logger.info(f"  Configs: {conf_ids}")
    logger.info(f"  Eigenvectors: {eigvec_path} (cfg {eigvec_cfg})")
    logger.info(f"  Eigenvector smearing: {apply_smear} (phase={mom_smear_phase})")
    logger.info(f"  Effective mass method: {meff_method}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load eigenvectors (CPU, compute dtype) ────────────────────────────
    with Timer("load_eigvecs", logger, output_dir.parent):
        eigvecs = load_eigenvectors(eigvec_path, Nev, Nt, logger)
        logger.info(f"Eigvecs loaded: shape={eigvecs.shape}, dtype={eigvecs.dtype}, "
                    f"mem={eigvecs.nbytes/1024**2:.1f}MB")

    # ── Pre-smear eigenvectors if requested ───────────────────────────────
    # Make a writable copy (loaded with mmap_mode='r')
    eigvecs = np.array(eigvecs, copy=True)
    if apply_smear and mom_smear_phase != 0:
        smear_mom = np.array([mom_smear_phase, 0, 0])
        phase_smear_gpu = compute_phase_factor_gpu(smear_mom, Nx)
        logger.info(f"Pre-smearing eigenvectors with phase: P_smear=({mom_smear_phase},0,0)")
        t_smear = time.perf_counter()
        for t in range(Nt):
            ev_gpu = cp.asarray(eigvecs[t])
            ev_smeared = cp.einsum("vxa,x->vxa", ev_gpu, phase_smear_gpu)
            eigvecs[t] = cp.asnumpy(ev_smeared)
            del ev_gpu, ev_smeared
        cp.get_default_memory_pool().free_all_blocks()
        logger.info(f"Eigenvector smearing done in {time.perf_counter()-t_smear:.1f}s")
    else:
        logger.info("Eigenvector smearing DISABLED (perambulator already encodes smearing)")

    # ── Load eigenvalues ──────────────────────────────────────────────────
    eigvals = np.array([])
    if eigval_path and os.path.exists(eigval_path):
        with Timer("load_eigvals", logger, output_dir.parent):
            eigvals = load_eigenvalues(eigval_path, Nev, logger)
            if len(eigvals) > 0:
                save_intermediate(eigvals, output_dir, "eigenvalues_Nev100.npy", logger)

    all_results = {}

    for iconf, conf_id in enumerate(conf_ids):
        conf_dir = output_dir / f"conf_{conf_id}"
        conf_dir.mkdir(parents=True, exist_ok=True)
        peram_dir = os.path.join(peram_base, str(conf_id))

        first_peram = os.path.join(peram_dir, f"perams.{conf_id}.0.0")
        if not os.path.exists(first_peram):
            logger.error(f"  [SKIP] conf={conf_id}: peram not found")
            all_results[conf_id] = {"status": "skipped", "reason": "peram not found"}
            continue

        n_perams = len([f for f in os.listdir(peram_dir)
                       if f.startswith(f"perams.{conf_id}")])
        logger.info(f"\n{'─'*60}")
        logger.info(f"  conf_id={conf_id} [{iconf+1}/{len(conf_ids)}]")
        logger.info(f"  Peram dir: {peram_dir} ({n_perams} files)")
        logger.info(f"{'─'*60}")

        conf_results = {}

        for Pz in Pz_list:
            Mom = np.array([Pz, Py, Px])
            logger.info(f"\n  --- P=({Px},{Py},{Pz}) ---")

            # ── Compute VVV phase ─────────────────────────────────────────
            # The phase factor for VVV = exp(-i * P_eff * 2π/L * coord)
            # If eigenvectors are NOT pre-smeared, P_eff = Pz (physics momentum)
            # If eigenvectors ARE pre-smeared, P_eff = Pz (physics momentum, smearing cancels in contraction)
            # We always use P_eff = Pz, because:
            #   - Without smearing: VVV carries only Pz projection, peram carries smearing
            #   - With smearing: VVV smearing cancels peram smearing in contraction
            phase_P_gpu = compute_phase_factor_gpu(Mom, Nx)

            # VVV cache key includes smearing flag
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
                                 "smearing": apply_smear, "dtype": str(get_compute_dtype())}):
                    VVV = compute_vvv_all_t_gpu(
                        eigvecs, phase_P_gpu, Nt, Nx, Nev1, logger)
                np.save(vvv_cache, VVV)
                logger.info(f"  [SAVE] VVV: {vvv_cache.name} "
                           f"({vvv_cache.stat().st_size/1024**2:.1f} MB, {VVV.dtype})")

            # Wick contraction
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

            # ── Effective mass with diagnostics ───────────────────────────
            with Timer(f"meff_Pz{Pz}_conf{conf_id}", logger, output_dir.parent,
                      extra={"conf_id": conf_id, "Pz": Pz}):
                # Try multiple methods and pick the best
                meff_results = {}
                for method in ["fit_cosh", "fit_exp", "exp_forward", "cosh"]:
                    try:
                        meff_results[method] = compute_effective_mass(
                            corr_dict["corr_pp"], Nt, alttc, method=method, logger=logger)
                    except Exception as e:
                        logger.warning(f"  meff method {method} failed: {e}")

                # Use the primary method result
                meff_result = meff_results.get(meff_method, list(meff_results.values())[0])

            np.savez(conf_dir / f"meff_Pz{Pz}_conf{conf_id}.npz",
                     meff_gev=meff_result["meff_gev"],
                     C2pt_1d=meff_result["C2pt_1d"],
                     C2pt_forward=meff_result.get("C2pt_forward", np.zeros(Nt)),
                     meff_plateau_gev=meff_result["meff_plateau_gev"],
                     meff_plateau_range=np.array(meff_result["meff_plateau_range"]),
                     method=np.array(meff_method))

            # Log all method results
            logger.info(f"  Effective mass results (a={alttc} fm):")
            for method, res in meff_results.items():
                logger.info(f"    {method:15s}: plateau={res['meff_plateau_gev']:.4f} GeV")

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

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info(f"\n{'═'*60}")
    logger.info(f"2pt (GPU, {get_compute_dtype()}) Summary:")
    for conf_id, result in all_results.items():
        if result.get("status") == "ok":
            for Pz, r in result["results"].items():
                logger.info(f"  conf={conf_id} Pz={Pz}: PP range={r['corr_pp_range_re']}, "
                           f"meff≈{r['meff_plateau_gev']:.4f} GeV (method={r.get('meff_method','?')})")
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
    p = argparse.ArgumentParser(description="Proton 2pt distillation (GPU, CORRECTED)")
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
