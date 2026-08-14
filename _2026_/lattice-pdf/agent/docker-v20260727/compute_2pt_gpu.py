#!/usr/bin/env python3
"""
Proton 2pt distillation — GPU (CuPy) accelerated, single precision (complex64).

Precision flow:
  Input (disk): eigenvector.npy [complex128], perambulator binary [complex128]
  GPU compute:  complex64 via to_gpu() downcast
  Output (disk): complex64 .npy/.npz (half disk vs complex128)

Memory strategy:
  - Eigenvectors (4.5 GB complex128 / 2.25 GB complex64): CPU then per-slice GPU
  - VVV output: GPU compute → CPU .npy (complex64)
  - Wick contraction: GPU compute → CPU result (complex64)

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
    """GPU phase factor: phi_P(x) = exp(-i·2π·P·x/L). Returns complex64 on GPU."""
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
    Returns: (Nt, Nev, Nx^3, 3) in compute dtype (complex64).
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
        # Copy needed slice + downcast
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

    Returns: (Nt, 4, 4, Nev, Nev) in compute dtype (complex64).
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
# VVV Baryon Block — GPU
# ═══════════════════════════════════════════════════════════════════════════════

def compute_vvv_single_t_gpu(
    eigvecs_t_gpu: cp.ndarray,      # (Nev, Nx^3, 3) complex64
    phase_factor_gpu: cp.ndarray,   # (Nx^3,) complex64
    Nx: int, Nev1: int,
) -> cp.ndarray:
    """VVV on GPU, explicit two-step contraction to avoid large intermediates.

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

        # Even permutations (two-step: T=ps·V_i·V_j, then VVV+=T·V_k)
        T = cp.einsum("x,ax,bx->abx", ps, es[..., 0], es[..., 1]); VVV += cp.einsum("abx,cx->abc", T, es[..., 2])
        T = cp.einsum("x,ax,bx->abx", ps, es[..., 1], es[..., 2]); VVV += cp.einsum("abx,cx->abc", T, es[..., 0])
        T = cp.einsum("x,ax,bx->abx", ps, es[..., 2], es[..., 0]); VVV += cp.einsum("abx,cx->abc", T, es[..., 1])
        # Odd permutations
        T = cp.einsum("x,ax,bx->abx", ps, es[..., 0], es[..., 2]); VVV -= cp.einsum("abx,cx->abc", T, es[..., 1])
        T = cp.einsum("x,ax,bx->abx", ps, es[..., 1], es[..., 0]); VVV -= cp.einsum("abx,cx->abc", T, es[..., 2])
        T = cp.einsum("x,ax,bx->abx", ps, es[..., 2], es[..., 1]); VVV -= cp.einsum("abx,cx->abc", T, es[..., 0])
    return VVV


def compute_vvv_all_t_gpu(
    eigvecs: np.ndarray,           # (Nt, Nev, Nx^3, 3) compute dtype (complex64)
    phase_smear_gpu: cp.ndarray,   # (Nx^3,) complex64 on GPU
    phase_P_gpu: cp.ndarray,        # (Nx^3,) complex64 on GPU
    Nt: int, Nx: int, Nev1: int,
    logger,
) -> np.ndarray:
    """VVV all time slices, streaming CPU→GPU per slice.

    Returns: (Nt, Nev1, Nev1, Nev1) compute dtype (complex64) on CPU.
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
        ev_sm = cp.einsum("vxa,x->vxa", ev_t_gpu, phase_smear_gpu)
        vvv_gpu = compute_vvv_single_t_gpu(ev_sm, phase_P_gpu, Nx, Nev1)
        VVV_all[t] = cp.asnumpy(vvv_gpu)
        del ev_t_gpu, ev_sm, vvv_gpu

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
            # Step 1: VVV_snk(a,b,c)·peram(g,j,a,d)→T1[g,j,b,c,d] (contract a)
            T1 = cp.einsum("abc,gjad->gjbcd", VVV_gpu[t_snk], peram_u_gpu[t_snk])
            # Step 2: T1·CG5p(g,j,b,e)→T2[c,d,e] (contract g,j,b)
            T2 = cp.einsum("gjbcd,gjbe->cde", T1, cg5p[t_snk])
            # Step 3: T2·peram(i,l,c,f)→T3[i,l,d,e,f] (contract c)
            T3 = cp.einsum("cde,ilcf->ildef", T2, peram_u_gpu[t_snk])
            # Step 4: T3·VVV_src(d,e,f)→result[i,l] (contract d,e,f)
            direct = cp.einsum("ildef,def->il", T3, VVV_src)

            # Exchange: "abc,glaf,gjbe,ijcd,def->il" decomposed
            # Step 1: VVV_snk(a,b,c)·peram(g,l,a,f)→T1x[g,l,b,c,f] (contract a)
            T1x = cp.einsum("abc,glaf->glbcf", VVV_gpu[t_snk], peram_u_gpu[t_snk])
            # Step 2: T1x·CG5p(g,j,b,e)→T2x[l,j,c,e,f] (contract g,b)
            T2x = cp.einsum("glbcf,gjbe->ljcef", T1x, cg5p[t_snk])
            # Step 3: T2x·peram(i,j,c,d)→T3x[i,l,d,e,f] (contract j,c)
            T3x = cp.einsum("ljcef,ijcd->ildef", T2x, peram_u_gpu[t_snk])
            # Step 4: T3x·VVV_src(d,e,f)→result[i,l] (contract d,e,f)
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

    # Boundary sign fix
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
# Effective mass (CPU)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_effective_mass(corr_pp: np.ndarray, Nt: int, alttc: float = 0.1053,
                           logger=None) -> dict:
    """Robust effective mass: cosh formula + exponential fit.

    Dual-method with SNR diagnostics.
    """
    fm2GeV = 0.1973

    C2pt_1d = np.zeros(Nt, dtype=np.float64)
    for dt in range(Nt):
        vals = [np.real(corr_pp[(t + dt) % Nt, t]) for t in range(Nt)]
        C2pt_1d[dt] = np.mean(vals)

    # ── Method 1: Cosh ──
    C_abs = np.abs(C2pt_1d) + 1e-30
    cosh_arg = (C_abs[2:] + C_abs[:-2]) / (2.0 * C_abs[1:-1])
    valid = cosh_arg >= 1.0
    meff_gev = np.full(Nt - 2, np.nan)
    meff_gev[valid] = np.arccosh(cosh_arg[valid]) * fm2GeV / alttc

    ps, pe = Nt // 4, Nt // 2
    pmask = ~np.isnan(meff_gev[ps:pe])
    meff_plateau_cosh = float(np.mean(meff_gev[ps:pe][pmask])) if np.any(pmask) else np.nan

    # ── Method 2: Exponential fit to |C(t)| ──
    t_min = max(3, int(np.argmax(C_abs[:Nt//2])))
    t_max = min(Nt // 2 - 1, 18)
    t_fit = np.arange(t_min, t_max + 1)
    C_fit = C_abs[t_fit]
    meff_exp, meff_exp_err, snr_avg = np.nan, np.nan, 0.0

    if len(t_fit) >= 3 and np.all(C_fit > 1e-25):
        logC = np.log(C_fit)
        A = np.vstack([np.ones_like(t_fit), t_fit]).T
        try:
            coeff, residuals, _, _ = np.linalg.lstsq(A, logC, rcond=None)
            meff_exp = -coeff[1] * fm2GeV / alttc
            if len(residuals) > 0:
                meff_exp_err = np.sqrt(residuals[0]/max(1,len(t_fit)-2)) * fm2GeV / alttc * 2.0
        except np.linalg.LinAlgError:
            pass
        snr_avg = float(np.mean(C_abs[t_min:t_max+1] / (np.std(C2pt_1d[t_min:t_max+1]) + 1e-30)))

    meff_best = meff_exp if np.isfinite(meff_exp) else meff_plateau_cosh
    meff_best_err = meff_exp_err if np.isfinite(meff_exp_err) else np.nan

    if logger:
        logger.info(f"C2pt_1d range: [{C2pt_1d.min():.4e}, {C2pt_1d.max():.4e}]")
        logger.info(f"Effective mass (a={alttc} fm, fm2GeV={fm2GeV}):")
        for t in range(1, min(Nt - 1, 12)):
            logger.info(f"  t={t:3d}  |C|={C_abs[t]:.4e}  m_cosh={meff_gev[t-1]:.4f} GeV")
        logger.info(f"  Cosh plateau [{ps},{pe}]: {meff_plateau_cosh:.3f} GeV")
        logger.info(f"  Exp fit [{t_min},{t_max}]: meff={meff_exp:.3f} ± {meff_exp_err:.2f} GeV  SNR={snr_avg:.1f}")
        if snr_avg < 2.0:
            logger.warning(f"  ⚠ SNR={snr_avg:.1f}<2 — meff UNRELIABLE (Nconf too small for disconnected)")
        logger.info(f"  Expected: proton m≈0.94 GeV (rest), E≈1.36 GeV (Pz=-2), E≈2.17 GeV (net P_eff=4)")
        logger.info(f"  Net effective Pz = Pz + 3*smear_z = -2 + 3*2 = 4")

    return {"meff_gev": meff_gev, "C2pt_1d": C2pt_1d,
            "meff_plateau_gev": meff_best,
            "meff_plateau_range": [ps, pe],
            "meff_cosh_plateau": meff_plateau_cosh,
            "meff_exp_fit": meff_exp, "meff_exp_err": meff_exp_err,
            "snr": snr_avg}


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_2pt_computation_gpu(config: dict, output_dir: Path, logger) -> dict:
    """Full 2pt distillation with GPU acceleration."""
    params = config["parameters"]
    paths = config["data_paths"]
    ensemble = config["ensemble"]

    Nt, Nx = ensemble["Nt"], ensemble["Nx"]
    Nev, Nev1 = params["Nev"], params["Nev1"]
    Pz_list = params["Pz_list"]
    Px, Py = params["Px"], params["Py"]
    mom_smear_phase = params["mom_smear_phase"]
    element = params["element"]
    conf_ids = params["conf_ids"]
    alttc = ensemble["alttc"]

    eigvec_path = paths["eigenvector"]
    eigval_path = paths.get("eigenvalue", "")
    peram_base = paths["perambulator_base"]
    eigvec_cfg = paths["eigenvector_cfg"]

    print_banner("Step 01: Proton 2pt Distillation (GPU, complex64)", logger)
    log_gpu_status(logger, "  ")
    logger.info(f"  Compute dtype: {get_compute_dtype()}")
    logger.info(f"  Ensemble: {ensemble['full_name']} | {Nt}×{Nx}³ | a={alttc} fm")
    logger.info(f"  Nev={Nev}, Nev1={Nev1}, element={element}")
    logger.info(f"  Momentum: P=({Px},{Py}), Pz∈{Pz_list}")
    logger.info(f"  Configs: {conf_ids}")
    logger.info(f"  Eigenvectors: {eigvec_path} (cfg {eigvec_cfg})")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load eigenvectors (CPU, compute dtype) ────────────────────────────
    with Timer("load_eigvecs", logger, output_dir.parent):
        eigvecs = load_eigenvectors(eigvec_path, Nev, Nt, logger)
        logger.info(f"Eigvecs loaded: shape={eigvecs.shape}, dtype={eigvecs.dtype}, "
                    f"mem={eigvecs.nbytes/1024**2:.1f}MB")

    # ── Load eigenvalues ──────────────────────────────────────────────────
    eigvals = np.array([])
    if eigval_path and os.path.exists(eigval_path):
        with Timer("load_eigvals", logger, output_dir.parent):
            eigvals = load_eigenvalues(eigval_path, Nev, logger)
            if len(eigvals) > 0:
                save_intermediate(eigvals, output_dir, "eigenvalues_Nev100.npy", logger)

    # ── Momentum smearing phase (GPU) ─────────────────────────────────────
    smear_mom = np.array([mom_smear_phase, 0, 0])
    phase_smear_gpu = compute_phase_factor_gpu(smear_mom, Nx)
    logger.info(f"Momentum smearing phase (GPU): P_smear=({mom_smear_phase},0,0)")

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

            # VVV cache check
            vvv_cache = conf_dir / f"VVV_Nev1{Nev1}_Px{Px}Py{Py}Pz{Pz}_conf{conf_id}.npy"
            if vvv_cache.exists():
                logger.info(f"  [CACHE] Loading VVV from {vvv_cache.name}")
                VVV = np.load(vvv_cache)
                logger.info(f"  VVV shape: {VVV.shape}, dtype: {VVV.dtype}, "
                           f"|VVV| max: {np.abs(VVV).max():.4e}")
            else:
                phase_P_gpu = compute_phase_factor_gpu(Mom, Nx)
                with Timer(f"VVV_GPU_Pz{Pz}_conf{conf_id}", logger, output_dir.parent,
                          extra={"conf_id": conf_id, "Pz": Pz, "device": "gpu",
                                 "dtype": str(get_compute_dtype())}):
                    VVV = compute_vvv_all_t_gpu(
                        eigvecs, phase_smear_gpu, phase_P_gpu,
                        Nt, Nx, Nev1, logger)
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
            base = f"Px{Px}Py{Py}Pz{Pz}_eginphase{abs(params['mom_smear'])}{element}"

            raw_fn = f"twopt_slice_pp_{base}_contract_conf{conf_id}.npy"
            np.save(conf_dir / raw_fn, corr_dict["corr_raw"])
            logger.info(f"  [SAVE] Raw: {raw_fn} ({corr_dict['corr_raw'].dtype})")

            pp_fn = f"twopt_slice_pp_{base}_nopol_ss_conf{conf_id}.npy"
            np.save(conf_dir / pp_fn, corr_dict["corr_pp"])
            logger.info(f"  [SAVE] PP: {pp_fn} ({corr_dict['corr_pp'].dtype})")

            pm_fn = f"twopt_slice_pm_{base}_nopol_ss_conf{conf_id}.npy"
            np.save(conf_dir / pm_fn, corr_dict["corr_pm"])
            logger.info(f"  [SAVE] PM: {pm_fn} ({corr_dict['corr_pm'].dtype})")

            # Effective mass
            with Timer(f"meff_Pz{Pz}_conf{conf_id}", logger, output_dir.parent,
                      extra={"conf_id": conf_id, "Pz": Pz}):
                meff_result = compute_effective_mass(corr_dict["corr_pp"], Nt, alttc, logger)
            np.savez(conf_dir / f"meff_Pz{Pz}_conf{conf_id}.npz",
                     meff_gev=meff_result["meff_gev"],
                     C2pt_1d=meff_result["C2pt_1d"],
                     meff_plateau_gev=meff_result["meff_plateau_gev"],
                     meff_plateau_range=np.array(meff_result["meff_plateau_range"]))

            conf_results[Pz] = {
                "corr_pp_shape": list(corr_dict["corr_pp"].shape),
                "corr_pp_range_re": [float(corr_dict["corr_pp"].real.min()),
                                     float(corr_dict["corr_pp"].real.max())],
                "corr_pp_range_im": [float(corr_dict["corr_pp"].imag.min()),
                                     float(corr_dict["corr_pp"].imag.max())],
                "meff_plateau_gev": meff_result["meff_plateau_gev"],
                "meff_plateau_range": meff_result["meff_plateau_range"],
                "dtype": str(corr_dict["corr_pp"].dtype),
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
                           f"meff≈{r['meff_plateau_gev']:.4f} GeV (dtype={r.get('dtype','?')})")
        else:
            logger.info(f"  conf={conf_id}: {result.get('status','?')}")
    logger.info(f"{'═'*60}")
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Proton 2pt distillation (GPU, complex64)")
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
