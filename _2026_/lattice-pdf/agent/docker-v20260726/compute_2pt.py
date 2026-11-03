#!/usr/bin/env python3
"""
Proton 2pt distillation computation using real lattice data.

Enhanced version (docker-v20260726):
  - Comprehensive module-level logging
  - All intermediate results saved (VVV blocks, raw peram data, etc.)
  - Step-by-step validation
  - Per-config error recovery
  - Progress bars for long operations
  - Memory tracking per step

Reads eigenvectors (.npy format) and perambulators (binary format) from
local data paths, computes VVV baryon blocks + Wick contraction + parity
projection, and saves results.

Usage (standalone):
    python compute_2pt.py --run-dir /path/to/output

Usage (imported):
    from compute_2pt import run_2pt_computation
    results = run_2pt_computation(config, output_dir, logger)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ─── Import shared utilities ─────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))
from utils import (
    Timer, print_banner, format_size, save_intermediate,
    validate_array, progress_bar, log_exception,
    get_current_memory_mb,
)
from gamma_matrix import (
    gamma_0, gamma_4, gamma_7,
    Cg5g4, Cg5g3, P_plus, P_minus,
)

# ─── Try import opt_einsum ───────────────────────────────────────────────────
try:
    from opt_einsum import contract as _opt_contract
    from opt_einsum import contract_expression as _contract_expr
    HAS_OPT_EINSUM = True
except ImportError:
    HAS_OPT_EINSUM = False
    _opt_contract = None
    _contract_expr = None


def get_contract():
    """Return optimal einsum contraction function.

    Uses opt_einsum.contract with default optimize='auto' which:
      - Handles 5-tensor contractions correctly (~1-3s/pair)
      - Has built-in path caching for repeated small contractions
    Falls back to numpy.einsum if opt_einsum unavailable.
    """
    if HAS_OPT_EINSUM:
        return _opt_contract
    return np.einsum


# ═══════════════════════════════════════════════════════════════════════════════
# Momentum phase factor
# ═══════════════════════════════════════════════════════════════════════════════

def compute_phase_factor(momentum: np.ndarray, Nx: int) -> np.ndarray:
    """Compute momentum phase factor: phi_P(x) = exp(-i * 2*pi * P·x / L).

    Args:
        momentum: shape (3,), momentum in units of 2π/L, order (Pz, Py, Px).
        Nx: spatial lattice extent.

    Returns:
        phase: shape (Nx³,), dtype complex128.
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


# ═══════════════════════════════════════════════════════════════════════════════
# Data readers
# ═══════════════════════════════════════════════════════════════════════════════

def load_eigenvectors(eigvec_path: str, Nev: int, Nt: int, logger) -> np.ndarray:
    """Load eigenvectors from .npy file using memory-mapped mode.

    Expected shape: (Nt, Nev, Nx^3, 3) complex128.

    Uses mmap_mode='r' to avoid loading the full 4.5GB file into RAM.
    Each time-slice access reads only ~66MB from disk on demand.

    Args:
        eigvec_path: Path to .npy file.
        Nev: Number of eigenvectors to keep.
        Nt: Temporal extent (for validation).
        logger: Logger instance.

    Returns:
        eigvecs: shape (Nt, Nev, Nx^3, 3) complex128 (memmap).
    """
    logger.info(f"Loading eigenvectors (mmap) from: {eigvec_path}")
    t0 = time.perf_counter()

    if not os.path.exists(eigvec_path):
        raise FileNotFoundError(f"Eigenvector file not found: {eigvec_path}")

    file_size_mb = os.path.getsize(eigvec_path) / 1024**2
    logger.info(f"  File size: {file_size_mb:.1f} MB")

    eigvecs_raw = np.load(eigvec_path, mmap_mode='r')
    logger.info(f"  Raw shape: {eigvecs_raw.shape}, dtype: {eigvecs_raw.dtype}")
    logger.info(f"  |eig| range: [{np.abs(eigvecs_raw).min():.2e}, {np.abs(eigvecs_raw).max():.2e}]")

    # Detect and reshape: raw data may be 6D (Nt, Nev, Nx, Nx, Nx, Nc)
    # or already 4D (Nt, Nev, Nx^3, Nc). Reshape to (Nt, Nev, Nx^3, Nc).
    if eigvecs_raw.ndim == 6:
        Nev_full = eigvecs_raw.shape[1]
        Nx_detected = eigvecs_raw.shape[2]
        Nc_detected = eigvecs_raw.shape[5]
        logger.info(f"  Detected 6D: Nt={eigvecs_raw.shape[0]}, Nev={Nev_full}, "
                    f"Nx={Nx_detected}, Nc={Nc_detected}")
        eigvecs = eigvecs_raw.reshape(eigvecs_raw.shape[0], Nev_full,
                                      Nx_detected * Nx_detected * Nx_detected,
                                      Nc_detected)
    elif eigvecs_raw.ndim == 4:
        Nev_full = eigvecs_raw.shape[1]
        eigvecs = eigvecs_raw
    else:
        raise ValueError(f"Unexpected eigenvector shape: {eigvecs_raw.shape}")

    # Validate
    if eigvecs.shape[0] != Nt:
        logger.warning(f"  Expected Nt={Nt}, got {eigvecs.shape[0]}")
    Nev = min(Nev, Nev_full)

    logger.info(f"  Using Nev={Nev}/{Nev_full}, shape={eigvecs.shape}, loaded in {time.perf_counter()-t0:.1f}s")
    logger.info(f"  mmap mode — slices read on demand (~66MB per time slice)")

    # Quick validation check on first slice
    first_slice = np.asarray(eigvecs[0, :Nev, :, :])
    assert np.all(np.isfinite(first_slice)), "Eigenvectors contain NaN/inf"
    assert np.abs(first_slice).max() > 0, "Eigenvectors are all zeros"

    return eigvecs, Nev


def load_eigenvalues(eigval_path: str, Nev: int, logger) -> np.ndarray:
    """Load eigenvalues from .npy file.

    Args:
        eigval_path: Path to eigenvalue file.
        Nev: Number of eigenvalues to keep.
        logger: Logger instance.

    Returns:
        eigenvalues: shape (Nev,), float64.
    """
    if not os.path.exists(eigval_path):
        logger.warning(f"Eigenvalue file not found: {eigval_path}")
        return np.array([])

    eigvals = np.load(eigval_path)
    raw_shape = eigvals.shape
    # May be 2D complex (Nt, Nev) or 1D real (Nev,)
    if eigvals.ndim == 2:
        eigvals = eigvals[0, :]  # Take first time slice
    eigvals = eigvals[:Nev].real  # Keep real part
    logger.info(f"Eigenvalues loaded: raw_shape={raw_shape}, "
                f"use_shape=({len(eigvals)},), "
                f"range=[{eigvals.min():.4f}, {eigvals.max():.4f}]")
    return eigvals


def read_perambulator_single_t(
    peram_dir: str, conf_id: int, t_source: int,
    Nev: int, Nt: int, logger,
) -> np.ndarray:
    """Read perambulator for a single source time slice.

    File format (matching donghx/snsc convention):
        perams.{conf_id}.{d_src}.{t_source}
        Each file is f8 (float64) binary.
        Total size = 4 * Nt * Nev_full * 4 * Nev_full * 2 doubles.

    Args:
        peram_dir: Directory containing perams.* files.
        conf_id: Configuration ID.
        t_source: Source time slice.
        Nev: Number of eigenvectors (used for truncation).
        Nt: Temporal extent.
        logger: Logger instance.

    Returns:
        peram: shape (Nt, 4, 4, Nev, Nev) complex128.
    """
    parts = []
    for d_src in range(4):
        fn = os.path.join(peram_dir, f"perams.{conf_id}.{d_src}.{t_source}")
        if not os.path.exists(fn):
            raise FileNotFoundError(
                f"Perambulator file not found: {fn}\n"
                f"  (conf_id={conf_id}, d_src={d_src}, t_src={t_source})"
            )
        with open(fn, "rb") as f:
            parts.append(np.fromfile(f, dtype="f8"))

    raw = np.concatenate(parts)
    Nev_full = int(np.sqrt(raw.size / (4 * 4 * Nt * 2)))

    # Reshape: (4, Nt, Nev_full, 4, Nev_full, 2)
    peram = raw.reshape(4, Nt, Nev_full, 4, Nev_full, 2)
    # Transpose → (Nt, 4, 4, Nev_full, Nev_full, 2)
    peram = peram.transpose(1, 3, 0, 4, 2, 5)
    # Complex → (Nt, 4, 4, Nev_full, Nev_full)
    peram = peram[..., 0] + 1j * peram[..., 1]

    # Truncate to Nev
    Nev_use = min(Nev, Nev_full)
    peram = peram[:, :, :, :Nev_use, :Nev_use].astype(np.complex128)

    # Validate
    assert np.all(np.isfinite(peram)), f"Peram (t={t_source}) contains NaN/inf"

    return peram


# ═══════════════════════════════════════════════════════════════════════════════
# VVV Baryon Block
# ═══════════════════════════════════════════════════════════════════════════════

def compute_vvv_single_t(
    eigvecs_t: np.ndarray,        # (Nev, Nx^3, 3)
    phase_factor: np.ndarray,     # (Nx^3,)
    contract_fn,
    Nx: int,
    Nev1: int,
) -> np.ndarray:
    """Compute VVV baryon block for a single time slice.

    VVV_{abc}(P) = sum_x phi_P(x) * epsilon_{ijk} * v_i^a * v_j^b * v_k^c

    Six epsilon_{ijk} permutations (3 even + 3 odd).

    Args:
        eigvecs_t: Eigenvectors at time t, shape (Nev, Nx^3, 3).
        phase_factor: Momentum phase, shape (Nx^3,).
        contract_fn: Einsum function.
        Nx: Spatial extent.
        Nev1: Number of eigenvectors to use.

    Returns:
        VVV: shape (Nev1, Nev1, Nev1) complex.
    """
    VVV = np.zeros((Nev1, Nev1, Nev1), dtype=complex)
    layer_size = Nx * Nx

    # Process in x-layers to limit memory
    for xi in range(Nx):
        s, e = xi * layer_size, (xi + 1) * layer_size
        es = eigvecs_t[:Nev1, s:e, :]  # (Nev1, Nx^2, 3)
        ps = phase_factor[s:e]           # (Nx^2,)

        # +1: even permutations
        VVV += contract_fn("x,ax,bx,cx->abc", ps, es[:, :, 0], es[:, :, 1], es[:, :, 2])
        VVV += contract_fn("x,ax,bx,cx->abc", ps, es[:, :, 1], es[:, :, 2], es[:, :, 0])
        VVV += contract_fn("x,ax,bx,cx->abc", ps, es[:, :, 2], es[:, :, 0], es[:, :, 1])
        # -1: odd permutations
        VVV -= contract_fn("x,ax,bx,cx->abc", ps, es[:, :, 0], es[:, :, 2], es[:, :, 1])
        VVV -= contract_fn("x,ax,bx,cx->abc", ps, es[:, :, 1], es[:, :, 0], es[:, :, 2])
        VVV -= contract_fn("x,ax,bx,cx->abc", ps, es[:, :, 2], es[:, :, 1], es[:, :, 0])

    return VVV


def compute_vvv_all_t(
    eigvecs: np.ndarray,        # (Nt, Nev, Nx^3, 3)
    phase_smear: np.ndarray,    # (Nx^3,) momentum smearing phase
    phase_P: np.ndarray,        # (Nx^3,) physics momentum phase
    Nt: int, Nx: int, Nev1: int,
    contract_fn,
    logger,
) -> np.ndarray:
    """Compute VVV for all time slices with momentum smearing.

    Args:
        eigvecs: Eigenvectors, shape (Nt, Nev, Nx^3, 3).
        phase_smear: Momentum smearing phase factor.
        phase_P: Physics momentum phase factor.
        Nt, Nx: Lattice extents.
        Nev1: Number of eigenvectors to use.
        contract_fn: Einsum function.
        logger: Logger instance.

    Returns:
        VVV: shape (Nt, Nev1, Nev1, Nev1) complex.
    """
    logger.info(f"Computing VVV for all Nt={Nt} time slices (Nev1={Nev1}, Nx={Nx})")
    t_start = time.perf_counter()

    VVV_all = np.zeros((Nt, Nev1, Nev1, Nev1), dtype=complex)

    for t in range(Nt):
        t1 = time.perf_counter()
        # Apply momentum smearing to eigenvectors
        ev_smeared = np.einsum("vxa,x->vxa", eigvecs[t, :, :, :], phase_smear)
        # Compute VVV
        VVV_all[t] = compute_vvv_single_t(ev_smeared, phase_P, contract_fn, Nx, Nev1)
        t2 = time.perf_counter()

        if t % 12 == 0 or t == Nt - 1:
            logger.debug(
                f"  VVV t={t:3d}/{Nt}  time={t2-t1:.2f}s  "
                f"|VVV|_max={np.abs(VVV_all[t]).max():.4e}"
            )

    elapsed = time.perf_counter() - t_start
    logger.info(f"VVV complete in {elapsed:.1f}s ({elapsed/Nt:.2f}s per time slice)")
    logger.info(f"  |VVV| range: [{np.abs(VVV_all).min():.2e}, {np.abs(VVV_all).max():.2e}]")
    logger.info(f"  Memory: {VVV_all.nbytes / 1024**2:.1f} MB")

    # Validate
    assert np.all(np.isfinite(VVV_all)), "VVV contains NaN/inf"

    return VVV_all


# ═══════════════════════════════════════════════════════════════════════════════
# Wick contraction + Parity projection
# ═══════════════════════════════════════════════════════════════════════════════

def compute_wick_and_project(
    VVV: np.ndarray,            # (Nt, Nev1, Nev1, Nev1)
    peram_dir: str,
    conf_id: int,
    Nev: int, Nev1: int,
    Nt: int,
    element: str,
    contract_fn,
    logger,
) -> dict:
    """Perform Wick contraction and parity projection.

    PERFORMANCE: Uses opt_einsum.contract_expression for 5-tensor Wick contractions
    (~1.6s per pair with Nev1=100 on CPU, ~60 min total per config).

    Algorithm (matching donghx DCU code exactly):
      1. Build interpolation operators: Gamma1, Gamma2
      2. Pre-compile einsum expressions for direct and exchange
      3. For each t_src:
         a. Read perambulator at t_src
         b. CG5peramCG5 = Gamma1 @ peram @ Gamma2
         c. For each t_snk (with 2 <= deltat <= 32):
            Direct  = expr_dir(VVV_snk, peram, cg5p, peram, VVV_src)
            Exchange= expr_ex(VVV_snk, peram, cg5p, peram, VVV_src)
            C = Direct - Exchange  → (4,4) matrix
      4. Parity projection
      5. Boundary sign fix
    """
    logger.info(f"Wick contraction: Nt={Nt}, Nev1={Nev1}, element={element}")

    # Build interpolation operators
    G7 = gamma_7()
    G4 = gamma_4()

    if element == "_Cg5g4":
        interProj1 = G7 @ G4
        interProj2 = G7 @ G4
    elif element == "_Cg5g3":
        G3 = gamma_3()
        interProj1 = G7 @ G3
        interProj2 = G7 @ G3
    elif element == "_Cg5":
        interProj1 = G7
        interProj2 = G7
    else:
        interProj1 = G7 @ G4
        interProj2 = G7 @ G4

    logger.info(f"  Interpolation operator: {element}")

    # Parity projectors
    Pplus = P_plus()
    Pminus = P_minus()

    # Raw correlator (before parity projection)
    corr_raw = np.zeros((Nt, Nt, 4, 4), dtype=complex)

    t_contract = time.perf_counter()
    n_pairs = 0

    for t_src in range(Nt):
        t_s0 = time.perf_counter()

        # Source VVV (complex conjugated)
        VVV_src = VVV[t_src].conj()

        # Read perambulator at this t_src
        peram_u = read_perambulator_single_t(
            peram_dir, conf_id, t_src, Nev, Nt, logger
        )  # → (Nt, 4, 4, Nev1, Nev1)

        # CG5peramCG5 = Gamma1 @ peram @ Gamma2
        CG5peramCG5 = contract_fn(
            "gh,thkbe,jk->tgjbe", interProj1, peram_u, interProj2
        )

        for t_snk in range(Nt):
            deltat = (t_snk - t_src + Nt) % Nt
            if not (2 <= deltat <= 32):
                continue

            n_pairs += 1

            # Direct and exchange terms
            direct = contract_fn(
                "abc,gjad,gjbe,ilcf,def->il",
                VVV[t_snk], peram_u[t_snk],
                CG5peramCG5[t_snk], peram_u[t_snk], VVV_src,
            )
            exchange = contract_fn(
                "abc,glaf,gjbe,ijcd,def->il",
                VVV[t_snk], peram_u[t_snk],
                CG5peramCG5[t_snk], peram_u[t_snk], VVV_src,
            )
            corr_raw[t_snk, t_src] = direct - exchange

        if t_src % 10 == 0 or t_src == Nt - 1:
            t_s1 = time.perf_counter()
            elapsed_t = t_s1 - t_s0
            est_total = elapsed_t * Nt / max(t_src + 1, 1) / 60
            logger.info(
                f"  Wick t_src={t_src:3d}/{Nt} done, {elapsed_t:.1f}s, "
                f"n_pairs={n_pairs}, mem={get_current_memory_mb():.0f}MB, "
                f"est_total~{est_total:.0f}min"
            )

    elapsed = time.perf_counter() - t_contract
    logger.info(f"Wick contraction complete in {elapsed:.1f}s ({elapsed/60:.1f} min), n_pairs={n_pairs}")
    logger.info(f"  Average: {elapsed/max(n_pairs,1)*1000:.1f} ms per (t_snk, t_src) pair")

    # Validate raw correlator
    assert np.all(np.isfinite(corr_raw)), "Raw correlator contains NaN/inf"

    # Parity projection
    t_parity = time.perf_counter()

    corr_pp = np.einsum("li,yxil->yx", Pplus, corr_raw)
    corr_pm = np.einsum("li,yxil->yx", Pminus, corr_raw)

    # Boundary sign fix (matching donghx code exactly)
    for ts in range(Nt):
        for tk in range(Nt):
            if tk < ts:
                corr_pp[tk, ts] *= -1.0
            if tk > ts:
                corr_pm[tk, ts] *= -1.0

    logger.info(f"Parity projection complete in {time.perf_counter()-t_parity:.2f}s")
    logger.info(f"  PP shape: {corr_pp.shape}")
    logger.info(f"  PP range: [{corr_pp.real.min():.4e}, {corr_pp.real.max():.4e}]")
    logger.info(f"  PM range: [{corr_pm.real.min():.4e}, {corr_pm.real.max():.4e}]")

    return {
        "corr_raw": corr_raw,
        "corr_pp": corr_pp,
        "corr_pm": corr_pm,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Effective mass
# ═══════════════════════════════════════════════════════════════════════════════

def compute_effective_mass(
    corr_pp: np.ndarray,  # (Nt, Nt)
    Nt: int,
    alttc: float = 0.1053,  # fm
    logger=None,
) -> dict:
    """Compute cosh effective mass from pp-projected 2pt correlator.

    a * m_eff(t) = arccosh((C(t-1) + C(t+1)) / (2 * C(t)))
    m_eff [GeV] = a*m_eff / a * hbar*c = a*m_eff * 0.1973 / a_fm

    Args:
        corr_pp: Positive-parity 2pt correlator, shape (Nt, Nt).
        Nt: Temporal extent.
        alttc: Lattice spacing in fm.
        logger: Logger instance.

    Returns:
        dict with keys: 'meff_gev', 'C2pt_1d', 'meff_plateau_gev', 'meff_plateau_range'.
    """
    # Average over source time slices to get 1D correlator
    C2pt_1d = np.zeros(Nt, dtype=float)
    for dt in range(Nt):
        vals = [np.real(corr_pp[(t + dt) % Nt, t]) for t in range(Nt)]
        C2pt_1d[dt] = np.mean(vals)

    if logger:
        logger.info(f"C2pt_1d range: [{C2pt_1d.min():.4e}, {C2pt_1d.max():.4e}]")

    # Cosh effective mass
    C_pos = np.abs(C2pt_1d) + 1e-30
    cosh_arg = (C_pos[2:] + C_pos[:-2]) / (2.0 * C_pos[1:-1])
    valid = cosh_arg >= 1.0

    fm2GeV = 0.1973  # hbar*c in GeV*fm
    meff_gev = np.full(Nt - 2, np.nan)
    meff_gev[valid] = np.arccosh(cosh_arg[valid]) * fm2GeV / alttc

    # Plateau estimate (always computed, regardless of logger)
    ps, pe = Nt // 4, Nt // 2
    pmask = ~np.isnan(meff_gev[ps:pe])
    meff_plateau = float(np.mean(meff_gev[ps:pe][pmask])) if np.any(pmask) else np.nan

    if logger:
        logger.info(f"Effective mass (cosh, a={alttc} fm):")
        for t in range(1, min(Nt - 1, 16)):
            logger.info(f"  t={t:3d}  m_eff={meff_gev[t-1]:.6f} GeV")
        if np.any(pmask):
            logger.info(f"  Plateau [{ps},{pe}]: m_eff={meff_plateau:.4f} GeV")

    return {
        "meff_gev": meff_gev,
        "C2pt_1d": C2pt_1d,
        "meff_plateau_gev": meff_plateau,
        "meff_plateau_range": [ps, pe],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main computation entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_2pt_computation(
    config: dict,
    output_dir: Path,
    logger,
) -> dict:
    """Run the full 2pt distillation computation for all configurations.

    Saves all intermediate results:
      - VVV blocks: VVV_Nev1{Nev1}_Px{Px}Py{Py}Pz{Pz}_conf{id}.npy
      - Raw contraction: twopt_slice_pp_..._contract_conf{id}.npy
      - PP projected: twopt_slice_pp_..._nopol_ss_conf{id}.npy
      - PM projected: twopt_slice_pm_..._nopol_ss_conf{id}.npy
      - Effective mass: meff_Pz{Pz}_conf{id}.npz
      - Per-config summary: conf_{id}/compute_2pt_summary.json

    Args:
        config: Full configuration dict (from run_config.json).
        output_dir: Output directory for data.
        logger: Logger instance.

    Returns:
        dict mapping conf_id → result dict.
    """
    params = config["parameters"]
    paths = config["data_paths"]
    ensemble = config["ensemble"]

    Nt = ensemble["Nt"]
    Nx = ensemble["Nx"]
    Nev = params["Nev"]
    Nev1 = params["Nev1"]
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

    contract_fn = get_contract()

    print_banner("Step 01: Proton 2pt Distillation", logger)
    logger.info(f"  Ensemble: {ensemble['full_name']} ({ensemble['name']})")
    logger.info(f"  Lattice: {Nt}×{Nx}³, β={ensemble['beta']}, a={alttc} fm")
    logger.info(f"  Nev={Nev}, Nev1={Nev1}, element={element}")
    logger.info(f"  Momentum: P=({Px},{Py}), Pz∈{Pz_list}")
    logger.info(f"  mom_smear_phase={mom_smear_phase}")
    logger.info(f"  Configs: {conf_ids} (Nconf={len(conf_ids)})")
    logger.info(f"  Eigenvectors: {eigvec_path} (cfg {eigvec_cfg})")
    logger.info(f"  Perambulators: {peram_base}/{{conf_id}}/")
    logger.info(f"  Contract function: {'opt_einsum' if HAS_OPT_EINSUM else 'numpy.einsum'}")
    logger.info(f"  Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load eigenvectors (mmap, once, reused for all configs) ──────────────
    with Timer("load_eigvecs", logger, output_dir.parent):
        eigvecs, Nev = load_eigenvectors(eigvec_path, Nev, Nt, logger)
        Nev1 = min(Nev1, Nev)
        logger.info(f"Eigenvectors loaded: shape={eigvecs.shape}, Nev={Nev}, Nev1={Nev1}")
        # Shape: (Nt, Nev, Nx^3, 3)

    # ── Load eigenvalues (optional) ─────────────────────────────────────────
    eigvals = np.array([])
    if eigval_path and os.path.exists(eigval_path):
        with Timer("load_eigvals", logger, output_dir.parent):
            eigvals = load_eigenvalues(eigval_path, Nev, logger)
            if len(eigvals) > 0:
                save_intermediate(eigvals, output_dir, "eigenvalues_Nev100.npy", logger)

    # ── Momentum smearing phase (same for all configs) ──────────────────────
    smear_mom = np.array([mom_smear_phase, 0, 0])
    phase_smear = compute_phase_factor(smear_mom, Nx)
    logger.info(f"Momentum smearing phase: P_smear=({mom_smear_phase},0,0), shape={phase_smear.shape}")

    all_results = {}

    # ── Loop over configurations ───────────────────────────────────────────
    for iconf, conf_id in enumerate(conf_ids):
        conf_dir = output_dir / f"conf_{conf_id}"
        conf_dir.mkdir(parents=True, exist_ok=True)

        peram_dir = os.path.join(peram_base, str(conf_id))

        # Check perambulator availability
        first_peram = os.path.join(peram_dir, f"perams.{conf_id}.0.0")
        if not os.path.exists(first_peram):
            logger.error(f"  [SKIP] conf_id={conf_id}: perambulator not found at {first_peram}")
            all_results[conf_id] = {
                "status": "skipped",
                "reason": "perambulator not found",
                "peram_dir": peram_dir,
            }
            continue

        # Count perambulator files
        n_perams = len([f for f in os.listdir(peram_dir) if f.startswith(f"perams.{conf_id}")])
        logger.info(f"\n{'─'*60}")
        logger.info(f"  Processing conf_id={conf_id} [{iconf+1}/{len(conf_ids)}]")
        logger.info(f"  Peram dir: {peram_dir} ({n_perams} files)")
        logger.info(f"{'─'*60}")

        conf_results = {}

        for Pz in Pz_list:
            Mom = np.array([Pz, Py, Px])
            logger.info(f"\n  --- P=({Px},{Py},{Pz}) ---")

            # Check VVV cache
            vvv_cache = conf_dir / f"VVV_Nev1{Nev1}_Px{Px}Py{Py}Pz{Pz}_conf{conf_id}.npy"
            if vvv_cache.exists():
                logger.info(f"  [CACHE] Loading VVV from {vvv_cache.name}")
                VVV = np.load(vvv_cache)
                logger.info(f"  VVV shape: {VVV.shape}, |VVV| max: {np.abs(VVV).max():.4e}")
            else:
                # Compute physics momentum phase
                phase_P = compute_phase_factor(Mom, Nx)
                logger.debug(f"  Phase factor shape: {phase_P.shape}")

                # Compute VVV
                with Timer(f"VVV_Pz{Pz}_conf{conf_id}", logger, output_dir.parent,
                          extra={"conf_id": conf_id, "Pz": Pz, "Nev1": Nev1}):
                    VVV = compute_vvv_all_t(
                        eigvecs, phase_smear, phase_P,
                        Nt, Nx, Nev1, contract_fn, logger,
                    )

                # Save VVV as intermediate result
                np.save(vvv_cache, VVV)
                size_mb = vvv_cache.stat().st_size / 1024**2
                logger.info(f"  [SAVE] VVV cache: {vvv_cache.name} ({size_mb:.1f} MB)")

            # Wick contraction + parity projection
            with Timer(f"Wick_Pz{Pz}_conf{conf_id}", logger, output_dir.parent,
                      extra={"conf_id": conf_id, "Pz": Pz, "Nev1": Nev1}):
                corr_dict = compute_wick_and_project(
                    VVV, peram_dir, conf_id, Nev, Nev1, Nt,
                    element, contract_fn, logger,
                )

            # ── Save all intermediate results ───────────────────────────────

            # 1. Raw contraction
            raw_fn = (
                f"twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}"
                f"_eginphase{abs(params['mom_smear'])}{element}"
                f"_contract_conf{conf_id}.npy"
            )
            np.save(conf_dir / raw_fn, corr_dict["corr_raw"])
            logger.info(f"  [SAVE] Raw contraction: {raw_fn} "
                       f"({corr_dict['corr_raw'].nbytes/1024**2:.1f}MB)")

            # 2. PP projected
            pp_fn = (
                f"twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}"
                f"_eginphase{abs(params['mom_smear'])}{element}"
                f"_nopol_ss_conf{conf_id}.npy"
            )
            np.save(conf_dir / pp_fn, corr_dict["corr_pp"])
            logger.info(f"  [SAVE] PP projected: {pp_fn} "
                       f"shape={corr_dict['corr_pp'].shape}")

            # 3. PM projected
            pm_fn = (
                f"twopt_slice_pm_Px{Px}Py{Py}Pz{Pz}"
                f"_eginphase{abs(params['mom_smear'])}{element}"
                f"_nopol_ss_conf{conf_id}.npy"
            )
            np.save(conf_dir / pm_fn, corr_dict["corr_pm"])
            logger.info(f"  [SAVE] PM projected: {pm_fn}")

            # 4. Effective mass
            with Timer(f"meff_Pz{Pz}_conf{conf_id}", logger, output_dir.parent,
                      extra={"conf_id": conf_id, "Pz": Pz}):
                meff_result = compute_effective_mass(
                    corr_dict["corr_pp"], Nt, alttc, logger,
                )
            np.savez(
                conf_dir / f"meff_Pz{Pz}_conf{conf_id}.npz",
                meff_gev=meff_result["meff_gev"],
                C2pt_1d=meff_result["C2pt_1d"],
                meff_plateau_gev=meff_result["meff_plateau_gev"],
                meff_plateau_range=np.array(meff_result["meff_plateau_range"]),
            )
            logger.info(f"  [SAVE] Effective mass: meff_Pz{Pz}_conf{conf_id}.npz")

            # Collect results
            conf_results[Pz] = {
                "corr_pp_shape": list(corr_dict["corr_pp"].shape),
                "corr_pp_range_re": [
                    float(corr_dict["corr_pp"].real.min()),
                    float(corr_dict["corr_pp"].real.max()),
                ],
                "corr_pp_range_im": [
                    float(corr_dict["corr_pp"].imag.min()),
                    float(corr_dict["corr_pp"].imag.max()),
                ],
                "meff_plateau_gev": meff_result["meff_plateau_gev"],
                "meff_plateau_range": meff_result["meff_plateau_range"],
            }

        # Save per-config summary
        conf_summary = {
            "conf_id": conf_id,
            "status": "ok",
            "Pz_list": Pz_list,
            "results": {str(Pz): v for Pz, v in conf_results.items()},
        }
        save_intermediate(conf_summary, conf_dir, "compute_2pt_summary.json", logger)

        all_results[conf_id] = {
            "status": "ok",
            "results": conf_results,
            "n_peram_files": n_perams,
        }

        # Log progress
        logger.info(f"  [PROGRESS] 2pt: {iconf+1}/{len(conf_ids)} configs done")

    # ── Summary ─────────────────────────────────────────────────────────────
    logger.info(f"\n{'═'*60}")
    logger.info("2pt Computation Summary:")
    for conf_id, result in all_results.items():
        if result["status"] == "ok":
            for Pz, r in result["results"].items():
                logger.info(
                    f"  conf={conf_id} Pz={Pz}: "
                    f"PP range={r['corr_pp_range_re']}, "
                    f"meff_plateau≈{r['meff_plateau_gev']:.4f} GeV"
                )
        else:
            logger.info(f"  conf={conf_id}: {result.get('status','?')} - {result.get('reason','')}")
    logger.info(f"{'═'*60}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Proton 2pt distillation (real data)")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory")
    parser.add_argument("--config", type=str, default=None, help="Path to run_config.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = args.config or (run_dir / "run_config.json")
    with open(config_path) as f:
        config = json.load(f)

    output_dir = run_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    from utils import setup_logging
    logger = setup_logging(run_dir / "run.log", "compute_2pt")

    results = run_2pt_computation(config, output_dir, logger)

    summary_path = output_dir / "compute_2pt_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Overall summary saved to {summary_path}")

    all_ok = all(r["status"] == "ok" for r in results.values())
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
