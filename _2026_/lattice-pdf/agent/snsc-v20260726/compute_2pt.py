#!/usr/bin/env python3
"""
Proton 2pt distillation computation using real lattice data.

Reads eigenvectors (.npy format) and perambulators (binary format) from
cluster paths, computes VVV baryon blocks + Wick contraction + parity
projection, and saves results in donghx's naming convention.

Usage (standalone):
    python compute_2pt.py --run-dir /path/to/output

Usage (imported):
    from compute_2pt import run_2pt_computation
    results = run_2pt_computation(config, output_dir, logger)
"""

from __future__ import annotations

import argparse
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
from utils import Timer, print_banner
from gamma_matrix import (
    gamma_0, gamma_4, gamma_7,  # gamma_7 = gamma_3 @ gamma_1
    Cg5g4, Cg5g3, P_plus, P_minus,
)

# ─── Try import opt_einsum ───────────────────────────────────────────────────
try:
    from opt_einsum import contract as _opt_contract
    HAS_OPT_EINSUM = True
except ImportError:
    HAS_OPT_EINSUM = False
    _opt_contract = None


def get_contract():
    """Return optimal einsum contraction function."""
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
    """Load eigenvectors from .npy file.

    Expected shape: (Nt, Nev, Nx^3, 3) complex128.

    Args:
        eigvec_path: Path to .npy file.
        Nev: Number of eigenvectors to keep.
        Nt: Temporal extent (for validation).
        logger: Logger instance.

    Returns:
        eigvecs: shape (Nt, Nev, Nx^3, 3) complex128.
    """
    logger.info(f"Loading eigenvectors from: {eigvec_path}")
    t0 = time.perf_counter()

    eigvecs = np.load(eigvec_path)

    logger.info(f"  Raw shape: {eigvecs.shape}, dtype: {eigvecs.dtype}")
    logger.info(f"  |eig| range: [{np.abs(eigvecs).min():.2e}, {np.abs(eigvecs).max():.2e}]")

    # Validate and truncate
    Nev_full = eigvecs.shape[1]
    if eigvecs.shape[0] != Nt:
        logger.warning(f"  Expected Nt={Nt}, got {eigvecs.shape[0]}")
    if Nev > Nev_full:
        logger.warning(f"  Requested Nev={Nev} > available {Nev_full}, truncating to {Nev_full}")
        Nev = Nev_full

    eigvecs = eigvecs[:, :Nev, :, :].astype(np.complex128)
    logger.info(f"  Truncated shape: {eigvecs.shape}, loaded in {time.perf_counter()-t0:.1f}s")

    return eigvecs


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
    logger.debug(f"  read_peram(t={t_source}): Nev_full={Nev_full}, total_doubles={raw.size}")

    # Reshape: (4, Nt, Nev_full, 4, Nev_full, 2)
    peram = raw.reshape(4, Nt, Nev_full, 4, Nev_full, 2)
    # Transpose → (Nt, 4, 4, Nev_full, Nev_full, 2)
    peram = peram.transpose(1, 3, 0, 4, 2, 5)
    # Complex → (Nt, 4, 4, Nev_full, Nev_full)
    peram = peram[..., 0] + 1j * peram[..., 1]

    # Truncate to Nev
    Nev_use = min(Nev, Nev_full)
    return peram[:, :, :, :Nev_use, :Nev_use].astype(np.complex128)


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
        ev_smeared = contract_fn("vxa,x->vxa", eigvecs[t, :, :, :], phase_smear)
        # Compute VVV
        VVV_all[t] = compute_vvv_single_t(ev_smeared, phase_P, contract_fn, Nx, Nev1)
        t2 = time.perf_counter()

        if t % 10 == 0:
            logger.debug(
                f"  VVV t={t:3d}/{Nt}  time={t2-t1:.2f}s  "
                f"|VVV|_max={np.abs(VVV_all[t]).max():.4e}"
            )

    elapsed = time.perf_counter() - t_start
    logger.info(f"VVV complete in {elapsed:.1f}s")
    logger.info(f"  |VVV| range: [{np.abs(VVV_all).min():.2e}, {np.abs(VVV_all).max():.2e}]")

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

    Algorithm (matching donghx DCU code exactly):
      1. Build interpolation operators: Gamma1, Gamma2
         - _Cg5g4: Gamma = gamma_7 @ gamma_4
         - _Cg5g3: Gamma = gamma_7 @ gamma_3
      2. For each t_src:
         a. Read perambulator at t_src
         b. CG5peramCG5 = Gamma1 @ peram @ Gamma2
         c. For each t_snk (with 2 <= deltat <= 32):
            Direct  = VVV_snk(abc) * p_snk(giad) * cg5p_snk(gjbe) * p_snk(ilcf) * VVV_src(def)*
            Exchange= VVV_snk(abc) * p_snk(glaf) * cg5p_snk(gjbe) * p_snk(ijcd) * VVV_src(def)*
            C = Direct - Exchange  → (4,4) matrix
      3. Parity projection:
         C_pp = contract(P_plus, C)  → (Nt, Nt)
         C_pm = contract(P_minus, C) → (Nt, Nt)
      4. Boundary sign fix:
         pp(t_snk < t_src) *= -1
         pm(t_snk > t_src) *= -1

    Args:
        VVV: Baryon block, shape (Nt, Nev1, Nev1, Nev1).
        peram_dir: Directory containing perams.* files.
        conf_id: Configuration ID.
        Nev, Nev1: Number of eigenvectors.
        Nt: Temporal extent.
        element: Interpolation operator name.
        contract_fn: Einsum function.
        logger: Logger instance.

    Returns:
        dict with keys 'corr_raw', 'corr_pp', 'corr_pm'.
    """
    logger.info(f"Wick contraction: Nt={Nt}, Nev1={Nev1}, element={element}")

    # Build interpolation operators
    G7 = gamma_7()
    G4 = gamma_4()
    G3 = gamma_3()

    if element == "_Cg5g4":
        interProj1 = G7 @ G4  # gamma_3 @ gamma_1 @ gamma_4
        interProj2 = G7 @ G4
    elif element == "_Cg5g3":
        interProj1 = G7 @ G3
        interProj2 = G7 @ G3
    elif element == "_Cg5":
        interProj1 = G7
        interProj2 = G7
    else:
        # Default to Cg5g4
        interProj1 = G7 @ G4
        interProj2 = G7 @ G4

    logger.info(f"  Interpolation operator: {element}")
    logger.debug(f"  Gamma1 =\n{interProj1}")

    # Parity projectors
    Pplus = P_plus()   # (1 + gamma_4) / 2
    Pminus = P_minus()  # (1 - gamma_4) / 2

    # Raw correlator (before parity projection)
    corr_raw = np.zeros((Nt, Nt, 4, 4), dtype=complex)

    t_contract = time.perf_counter()

    for t_src in range(Nt):
        t_s0 = time.perf_counter()

        # Source VVV (complex conjugated)
        VVV_src = VVV[t_src].conj()

        # Read perambulator at this t_src
        peram_u = read_perambulator_single_t(
            peram_dir, conf_id, t_src, Nev, Nt, logger
        )  # → (Nt, 4, 4, Nev1, Nev1)

        # CG5peramCG5 = Gamma1 @ peram @ Gamma2
        # peram_u shape: (Nt, dirac_src, dirac_snk, Nev, Nev)
        # interProj1 shape: (4, 4)
        # CG5peramCG5 shape: (Nt, 4, 4, Nev, Nev)
        CG5peramCG5 = contract_fn(
            "gh,thkbe,jk->tgjbe", interProj1, peram_u, interProj2
        )

        for t_snk in range(Nt):
            deltat = (t_snk - t_src + Nt) % Nt
            if not (2 <= deltat <= 32):
                continue

            # Direct term
            direct = contract_fn(
                "abc,gjad,gjbe,ilcf,def->il",
                VVV[t_snk], peram_u[t_snk],
                CG5peramCG5[t_snk], peram_u[t_snk], VVV_src,
            )
            # Exchange term
            exchange = contract_fn(
                "abc,glaf,gjbe,ijcd,def->il",
                VVV[t_snk], peram_u[t_snk],
                CG5peramCG5[t_snk], peram_u[t_snk], VVV_src,
            )
            corr_raw[t_snk, t_src] = direct - exchange

        t_s1 = time.perf_counter()
        if t_src % 10 == 0:
            logger.debug(f"  Wick t_src={t_src:3d}/{Nt} done, {t_s1-t_s0:.1f}s")

    logger.info(f"Wick contraction complete in {time.perf_counter()-t_contract:.1f}s")

    # Parity projection
    t_parity = time.perf_counter()

    # C_pp = P_plus @ C @ P_plus (trace over Dirac indices)
    corr_pp = contract_fn("li,yxil->yx", Pplus, corr_raw)
    corr_pm = contract_fn("li,yxil->yx", Pminus, corr_raw)

    # Boundary sign fix (matching donghx code exactly)
    for ts in range(Nt):
        for tk in range(Nt):
            if tk < ts:
                corr_pp[tk, ts] *= -1.0
            if tk > ts:
                corr_pm[tk, ts] *= -1.0

    logger.info(f"Parity projection complete in {time.perf_counter()-t_parity:.2f}s")
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
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cosh effective mass from pp-projected 2pt correlator.

    a * m_eff(t) = arccosh((C(t-1) + C(t+1)) / (2 * C(t)))
    m_eff [GeV] = a*m_eff / a * hbar*c = a*m_eff * 0.1973 / a_fm

    Args:
        corr_pp: Positive-parity 2pt correlator, shape (Nt, Nt).
        Nt: Temporal extent.
        alttc: Lattice spacing in fm.
        logger: Logger instance.

    Returns:
        (meff_gev, C2pt_1d): effective mass in GeV and 1D correlator.
    """
    # Average over source time slices to get 1D correlator
    C2pt_1d = np.zeros(Nt, dtype=float)
    for dt in range(Nt):
        vals = [np.real(corr_pp[(t + dt) % Nt, t]) for t in range(Nt)]
        C2pt_1d[dt] = np.mean(vals)

    # Cosh effective mass
    C_pos = np.abs(C2pt_1d) + 1e-30
    cosh_arg = (C_pos[2:] + C_pos[:-2]) / (2.0 * C_pos[1:-1])
    valid = cosh_arg >= 1.0

    fm2GeV = 0.1973  # hbar*c in GeV*fm
    meff_gev = np.full(Nt - 2, np.nan)
    meff_gev[valid] = np.arccosh(cosh_arg[valid]) * fm2GeV / alttc

    if logger:
        logger.info(f"Effective mass (cosh, a={alttc} fm):")
        for t in range(1, min(Nt - 1, 16)):
            logger.info(f"  t={t:3d}  m_eff={meff_gev[t-1]:.6f} GeV")

        # Plateau estimate
        ps, pe = Nt // 4, Nt // 2
        pmask = ~np.isnan(meff_gev[ps:pe])
        if np.any(pmask):
            logger.info(f"  Plateau [{ps},{pe}]: m_eff={np.mean(meff_gev[ps:pe][pmask]):.4f} GeV")

    return meff_gev, C2pt_1d


# ═══════════════════════════════════════════════════════════════════════════════
# Main computation entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_2pt_computation(
    config: dict,
    output_dir: Path,
    logger,
) -> dict:
    """Run the full 2pt distillation computation for all configurations.

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
    peram_base = paths["perambulator_base"]
    eigvec_cfg = paths["eigenvector_cfg"]

    contract_fn = get_contract()

    print_banner("Step 01: Proton 2pt Distillation", logger)
    logger.info(f"  Lattice: {Nt}x{Nx}^3, Nev={Nev}, Nev1={Nev1}")
    logger.info(f"  Momentum: Pz∈{Pz_list}, Py={Py}, Px={Px}")
    logger.info(f"  Configs: {conf_ids}")
    logger.info(f"  Element: {element}")
    logger.info(f"  Eigenvectors: {eigvec_path}")
    logger.info(f"  Perambulators: {peram_base}/{{conf_id}}/")
    logger.info(f"  Note: eigvec cfg={eigvec_cfg} reused for all configs (standard practice)")
    logger.info(f"  Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load eigenvectors (once, reused for all configs) ────────────────────
    with Timer("load_eigvecs", logger, output_dir):
        eigvecs = load_eigenvectors(eigvec_path, Nev, Nt, logger)
        # Shape: (Nt, Nev, Nx^3, 3)

    # ── Momentum smearing phase (same for all configs) ──────────────────────
    smear_mom = np.array([mom_smear_phase, 0, 0])
    phase_smear = compute_phase_factor(smear_mom, Nx)
    logger.info(f"Momentum smearing phase: P_smear=({mom_smear_phase},0,0)")

    all_results = {}

    # ── Loop over configurations ───────────────────────────────────────────
    for conf_id in conf_ids:
        conf_dir = output_dir / f"conf_{conf_id}"
        conf_dir.mkdir(parents=True, exist_ok=True)

        peram_dir = os.path.join(peram_base, str(conf_id))

        # Check perambulator availability
        first_peram = os.path.join(peram_dir, f"perams.{conf_id}.0.0")
        if not os.path.exists(first_peram):
            logger.error(f"  [SKIP] conf_id={conf_id}: perambulator not found at {first_peram}")
            all_results[conf_id] = {"status": "skipped", "reason": "perambulator not found"}
            continue

        logger.info(f"\n{'─'*60}")
        logger.info(f"  Processing conf_id={conf_id}")
        logger.info(f"  Peram dir: {peram_dir}")
        logger.info(f"{'─'*60}")

        conf_results = {}

        for Pz in Pz_list:
            Mom = np.array([Pz, Py, Px])
            logger.info(f"\n  --- Pz={Pz} ---")

            # Check VVV cache
            vvv_cache = conf_dir / f"VVV_Nev1{Nev1}_Px{Px}Py{Py}Pz{Pz}_conf{conf_id}.npy"
            if vvv_cache.exists():
                logger.info(f"  [CACHE] Loading VVV from {vvv_cache.name}")
                VVV = np.load(vvv_cache)
            else:
                # Compute physics momentum phase
                with Timer(f"VVV_Pz{Pz}_conf{conf_id}", logger, output_dir):
                    phase_P = compute_phase_factor(Mom, Nx)
                    VVV = compute_vvv_all_t(
                        eigvecs, phase_smear, phase_P,
                        Nt, Nx, Nev1, contract_fn, logger,
                    )
                np.save(vvv_cache, VVV)
                logger.info(f"  [SAVE] VVV cache: {vvv_cache}")

            # Wick contraction + parity projection
            with Timer(f"Wick_Pz{Pz}_conf{conf_id}", logger, output_dir):
                corr_dict = compute_wick_and_project(
                    VVV, peram_dir, conf_id, Nev, Nev1, Nt,
                    element, contract_fn, logger,
                )

            # Save raw contraction
            raw_fn = (
                f"twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}"
                f"_eginphase{abs(params['mom_smear'])}{element}"
                f"_contract_conf{conf_id}.npy"
            )
            np.save(conf_dir / raw_fn, corr_dict["corr_raw"])
            logger.info(f"  [SAVE] Raw contraction: {raw_fn}")

            # Save parity-projected correlator (donghx naming convention)
            pp_fn = (
                f"twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}"
                f"_eginphase{abs(params['mom_smear'])}{element}"
                f"_nopol_ss_conf{conf_id}.npy"
            )
            np.save(conf_dir / pp_fn, corr_dict["corr_pp"])
            logger.info(f"  [SAVE] PP: {pp_fn}  shape={corr_dict['corr_pp'].shape}")

            pm_fn = (
                f"twopt_slice_pm_Px{Px}Py{Py}Pz{Pz}"
                f"_eginphase{abs(params['mom_smear'])}{element}"
                f"_nopol_ss_conf{conf_id}.npy"
            )
            np.save(conf_dir / pm_fn, corr_dict["corr_pm"])
            logger.info(f"  [SAVE] PM: {pm_fn}")

            # Effective mass
            with Timer(f"meff_Pz{Pz}_conf{conf_id}", logger, output_dir):
                meff_gev, C2pt_1d = compute_effective_mass(
                    corr_dict["corr_pp"], Nt, alttc, logger,
                )
            np.savez(
                conf_dir / f"meff_Pz{Pz}_conf{conf_id}.npz",
                meff_gev=meff_gev,
                C2pt_1d=C2pt_1d,
            )

            conf_results[Pz] = {
                "corr_pp_shape": list(corr_dict["corr_pp"].shape),
                "corr_pp_range_re": [
                    float(corr_dict["corr_pp"].real.min()),
                    float(corr_dict["corr_pp"].real.max()),
                ],
                "meff_plateau_gev": float(np.nanmean(meff_gev[Nt//4:Nt//2])),
            }

        all_results[conf_id] = {
            "status": "ok",
            "results": conf_results,
        }

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
            logger.info(f"  conf={conf_id}: {result['status']} - {result.get('reason','')}")
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

    # Setup logging
    from utils import setup_logging
    logger = setup_logging(run_dir / "run.log", "compute_2pt")

    results = run_2pt_computation(config, output_dir, logger)

    # Write summary
    import json as _json
    summary_path = output_dir / "compute_2pt_summary.json"
    with open(summary_path, "w") as f:
        _json.dump(results, f, indent=2, default=str)
    logger.info(f"Summary saved to {summary_path}")

    all_ok = all(r["status"] == "ok" for r in results.values())
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
