#!/usr/bin/env python3
"""
OPE computation from gauge configurations — GPU (CuPy) accelerated.
CORRECTED (v20260802): donghx Operator.py algorithm (dual F̃, proper Wilson line).

ALGORITHM: operators_new_z0_mu2 — the STANDARD donghx algorithm from v20260730.
  O(z) = Σ_{x⊥} Tr[ F_{μν}(x+z) · W^dag(z→0) · F̃_{μν}(x) · W(0→z) ]
  where F̃_{μν} = 0.5 * ε_{μνρσ} * F_{ρσ} is the DUAL field strength.

KEY FIXES in v20260802:
  1. Uses Tensor4 (0.5 * ε_{μνρσ}) for dual field strength — CRITICAL for correctness.
  2. Roll-based Wilson line transport matches donghx EXACTLY.
  3. Fixed: duplicate debug log at lines 389-397 removed.
  4. Fixed: spatial sum over ALL axes (1,2,3), not just perpendicular axes.
  5. Double precision (complex128) by default.

Precision flow:
  Input (disk): gauge .lime [complex128 BE] → CPU complex128 → GPU compute dtype
  GPU compute:  complex64 or complex128
  Output (disk): same as compute dtype

Usage:
    python compute_ope_gpu.py --run-dir /path/to/output
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
    get_compute_dtype,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Tensor4 (0.5 * ε_{μνρσ}) — Levi-Civita on CPU for broadcasting
# ═══════════════════════════════════════════════════════════════════════════════

def build_tensor4() -> np.ndarray:
    """Build Tensor4 = 0.5 * ε_{μνρσ} matching donghx Operator.py EXACTLY.

    FIXED: c was incorrectly initialized from b instead of 0.
    Each non-zero element is 0.5 * sign(permutation) for the Levi-Civita symbol.

    Returns:
        (4,4,4,4) float64 array where T4[μ,ν,ρ,σ] = 0.5 * ε_{μνρσ}.
    """
    T = np.zeros((4, 4, 4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            a = 1.0 if i > j else 0.0
            for k in range(4):
                b = (1.0 if i > k else 0.0) + (1.0 if j > k else 0.0)
                for l in range(4):
                    # FIXED: c starts from 0, NOT from b (was the bug)
                    c = (1.0 if i > l else 0.0) + (1.0 if j > l else 0.0) + (1.0 if k > l else 0.0)
                    if len({i, j, k, l}) == 4:  # all indices distinct
                        # Even permutation → +1, odd permutation → -1
                        T[i, j, k, l] = 1.0 if int(a + b + c) % 2 == 0 else -1.0
    return 0.5 * T


# ═══════════════════════════════════════════════════════════════════════════════
# GPU plaquette_clover (independent CuPy implementation, matches donghx)
# ═══════════════════════════════════════════════════════════════════════════════

def plaquette_clover_gpu(gauge_gpu: cp.ndarray, mu: int, nu: int) -> cp.ndarray:
    r"""Clover field strength F_{μν}(x) on GPU. Matches donghx plaquette_clover_new.

    F_{μν} = -i/8 * Σ_k (P_k - P_k†)
    with 4 clover leaves using the gauge convention:
    P1 = P_{μ,ν},  P2 = P_{ν,-μ},  P3 = P_{-μ,-ν},  P4 = P_{-ν,μ}

    Each plaquette P is the product of 4 SU(3) link matrices around a 1×1 Wilson loop.

    Args:
        gauge_gpu: Gauge field on GPU, shape (Nt,Nz,Ny,Nx,4,3,3), dtype=compute_dtype.
        mu: First Lorentz index (0=t, 1=z, 2=y, 3=x).
        nu: Second Lorentz index (0=t, 1=z, 2=y, 3=x). Must differ from mu.

    Returns:
        Field strength tensor F_{μν}, shape (Nt,Nz,Ny,Nx,3,3), dtype=compute_dtype.
    """
    e = cp.einsum
    g = gauge_gpu
    # Map Lorentz index to spatial axis in (t, z, y, x)
    a_mu = 3 - mu
    a_nu = 3 - nu

    # Shifted gauge fields for clover leaves
    g_lu = cp.roll(g, 1, axis=a_mu)          # shifted up by mu
    g_rd = cp.roll(g, 1, axis=a_nu)          # shifted right by nu
    g_ld = cp.roll(g_lu, 1, axis=a_nu)       # shifted left-down (up by mu, right by nu)

    # Plaquette 1: P_{μν} — forward in μ, forward in ν
    p1 = e("tzyxab,tzyxbc->tzyxac", g[..., mu, :, :],
           cp.roll(g, -1, axis=a_mu)[..., nu, :, :])
    p1 = e("tzyxab,tzyxcb->tzyxac", p1,
           cp.roll(g, -1, axis=a_nu)[..., mu, :, :].conj())
    p1 = e("tzyxab,tzyxcb->tzyxac", p1, g[..., nu, :, :].conj())

    # Plaquette 2: P_{ν,-μ} — forward in ν, backward in μ
    p2 = e("tzyxab,tzyxcb->tzyxac",
           cp.roll(g_lu, -1, axis=a_mu)[..., nu, :, :],
           cp.roll(g_lu, -1, axis=a_nu)[..., mu, :, :].conj())
    p2 = e("tzyxab,tzyxcb->tzyxac", p2, g_lu[..., nu, :, :].conj())
    p2 = e("tzyxab,tzyxbc->tzyxac", p2, g_lu[..., mu, :, :])

    # Plaquette 3: P_{-μ,-ν} — backward in μ, backward in ν
    p3 = e("tzyxba,tzyxcb->tzyxac",
           cp.roll(g_ld, -1, axis=a_nu)[..., mu, :, :].conj(),
           g_ld[..., nu, :, :].conj())
    p3 = e("tzyxab,tzyxbc->tzyxac", p3, g_ld[..., mu, :, :])
    p3 = e("tzyxab,tzyxbc->tzyxac", p3, cp.roll(g_ld, -1, axis=a_mu)[..., nu, :, :])

    # Plaquette 4: P_{-ν,μ} — backward in ν, forward in μ
    p4 = e("tzyxba,tzyxbc->tzyxac", g_rd[..., nu, :, :].conj(), g_rd[..., mu, :, :])
    p4 = e("tzyxab,tzyxbc->tzyxac", p4, cp.roll(g_rd, -1, axis=a_mu)[..., nu, :, :])
    p4 = e("tzyxab,tzyxcb->tzyxac", p4,
           cp.roll(g_rd, -1, axis=a_nu)[..., mu, :, :].conj())

    # F = -i/8 * Σ(P_k - P_k†)  — matches donghx line 150-160 exactly
    tr = (0, 1, 2, 3, 5, 4)  # transpose indices for conjugate: (col,row) swap
    ans = (p1 - p1.conj().transpose(*tr)
         + p2 - p2.conj().transpose(*tr)
         + p3 - p3.conj().transpose(*tr)
         + p4 - p4.conj().transpose(*tr))
    return cp.array(-1j, dtype=get_compute_dtype()) * ans / cp.array(8.0, dtype=ans.real.dtype)


# ═══════════════════════════════════════════════════════════════════════════════
# Dual field strength: F̃_{μν} = 0.5 * ε_{μνρσ} * F_{ρσ}
# ═══════════════════════════════════════════════════════════════════════════════

_TENSOR4_NP = build_tensor4()  # CPU copy of Levi-Civita tensor for coefficient lookup


def compute_dual_field_strength_munu(F_dict: dict, mu: int, nu: int) -> cp.ndarray:
    """Compute F̃_{μν} = 0.5 * Σ_{ρσ} ε_{μνρσ} * F_{ρσ} on GPU.

    Uses precomputed F components from F_dict; sums over non-zero ε contributions.
    This is the DUAL field strength that mixes electric and magnetic components
    — the key physics input to the OPE operator.

    Args:
        F_dict: Dictionary {(rho, sigma): F_{ρσ}_gpu} of precomputed field strengths.
        mu, nu: Indices of the dual field strength to compute.

    Returns:
        F̃_{μν}, shape (Nt,Nz,Ny,Nx,3,3), dtype=compute_dtype, on GPU.
    """
    compute_dt = get_compute_dtype()
    result = cp.zeros_like(list(F_dict.values())[0])
    for rho in range(4):
        for sigma in range(4):
            coeff = _TENSOR4_NP[mu, nu, rho, sigma]
            if abs(coeff) < 1e-10 or rho == sigma:
                continue
            F_rs = F_dict.get((rho, sigma))
            if F_rs is None:
                continue
            result = result + cp.array(coeff, dtype=compute_dt) * F_rs
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Gauge reader (CPU complex128 → GPU compute dtype)
# ═══════════════════════════════════════════════════════════════════════════════

def read_gauge_lime(filepath: str, Nt: int, Nx: int, Nc: int = 3) -> np.ndarray:
    """Read .lime gauge config — try tail offset first, fall back to header scan.

    ILDG .lime format: big-endian float64, shape (Nt,Nx,Nx,Nx,4,Nc,Nc) complex.

    Args:
        filepath: Path to .lime gauge configuration file.
        Nt, Nx: Lattice dimensions.
        Nc: Number of colors (3 for SU(3)).

    Returns:
        Gauge field as numpy array, shape (Nt,Nx,Nx,Nx,4,Nc,Nc), dtype=complex128.
    """
    expected_elems = Nt * Nx * Nx * Nx * 4 * Nc * Nc * 2
    expected_bytes = expected_elems * 8
    file_size = os.path.getsize(filepath)
    data_offset = file_size - expected_bytes

    # Fast path: try tail read (most configs have fixed-size XML header)
    if 0 <= data_offset < file_size:
        with open(filepath, "rb") as f:
            f.seek(data_offset)
            raw = np.fromfile(f, dtype=">f8", count=expected_elems)
        if raw.size == expected_elems:
            tg = (raw.reshape(Nt, Nx, Nx, Nx, 4, Nc, Nc, 2)[..., 0]
                  + 1j * raw.reshape(Nt, Nx, Nx, Nx, 4, Nc, Nc, 2)[..., 1])
            # Quick unitarity check on first link
            U = tg[0, 0, 0, 0, 0]
            if np.abs(U @ U.conj().T - np.eye(Nc)).max() < 1e-3:
                return tg.astype(np.complex128, copy=False)

    # Fallback: scan for valid gauge data near expected offset
    return _scan_gauge_data(filepath, file_size, data_offset,
                            expected_bytes, expected_elems, Nt, Nx, Nc)


def _scan_gauge_data(filepath, file_size, approx_hdr, expected_bytes, expected_elems,
                     Nt, Nx, Nc) -> np.ndarray:
    """Scan for valid gauge data near the expected offset when tail-read fails.

    Tests candidate byte offsets within ±4KB of the expected position,
    plus a coarse scan of the first 256KB. Validates each candidate by
    checking unitarity (U†U ≈ I) on random SU(3) links.
    """
    # Build candidate offsets: fine scan ±4KB + coarse scan first 256KB
    candidates = []
    for delta in range(-4096, 4097, 8):
        off = approx_hdr + delta
        if 0 <= off <= file_size - expected_bytes:
            candidates.append(off)
    for off in range(0, min(file_size - expected_bytes + 1, 262144), 512):
        if off not in candidates:
            candidates.append(off)
    # Sort by proximity to expected offset (closest first)
    candidates = sorted(set(candidates), key=lambda x: abs(x - approx_hdr))

    with open(filepath, "rb") as f:
        raw_bytes = f.read()

    best_dev, best_offset, best_gauge = float('inf'), None, None
    rng = np.random.default_rng(42)

    for off_bytes in candidates:
        if off_bytes + expected_bytes > file_size:
            continue
        chunk = raw_bytes[off_bytes:off_bytes + expected_bytes]
        test_raw = np.frombuffer(chunk, dtype=">f8", count=expected_elems)
        if test_raw.size != expected_elems:
            continue

        test = test_raw.reshape(Nt, Nx, Nx, Nx, 4, Nc, Nc, 2)
        test_gauge = test[..., 0] + 1j * test[..., 1]

        # Reject clearly invalid data
        abs_max = np.abs(test_gauge).max()
        if not np.isfinite(abs_max) or abs_max > 100.0:
            continue

        # Unitarity check on 20 random links
        devs = []
        try:
            for _ in range(20):
                t_i, z_i = rng.integers(0, Nt), rng.integers(0, Nx)
                y_i, x_i = rng.integers(0, Nx), rng.integers(0, Nx)
                d_i = rng.integers(0, 4)
                U = test_gauge[t_i, z_i, y_i, x_i, d_i]
                devs.append(np.abs(U @ U.conj().T - np.eye(Nc)).max())
        except Exception:
            continue

        if not devs:
            continue
        max_dev, mean_dev = max(devs), sum(devs) / len(devs)
        if mean_dev < best_dev and max_dev < 1e-3:
            best_dev, best_offset, best_gauge = mean_dev, off_bytes, test_gauge
        # Early exit: perfect match near expected offset
        if max_dev < 1e-6 and mean_dev < 1e-6 and abs(off_bytes - approx_hdr) < 4096:
            return test_gauge.astype(np.complex128, copy=False)

    if best_gauge is not None:
        return best_gauge.astype(np.complex128, copy=False)
    raise ValueError(f"No valid gauge data in {filepath}")


def validate_gauge(gauge: np.ndarray, tag: str = "", logger=None) -> dict:
    """Validate gauge config on CPU: unitarity, trace, plaquette trace.

    Args:
        gauge: Gauge field, shape (Nt,Nz,Ny,Nx,4,Nc,Nc), complex128.
        tag: Label for log messages.
        logger: Logger instance.

    Returns:
        dict with validation metrics: unitary_dev_max/mean, trace_mean_re/im,
        plaq_trace_mean_re/im, shape.
    """
    Nt, Nz, Ny, Nx, Nd, Nc, _ = gauge.shape
    results = {"shape": list(gauge.shape), "tag": tag}
    rng = np.random.default_rng(42)
    n_check = min(200, Nt * Nz * Ny * Nx)

    # Unitarity check
    devs = []
    for _ in range(n_check):
        t_i, z_i = rng.integers(0, Nt), rng.integers(0, Nz)
        y_i, x_i = rng.integers(0, Ny), rng.integers(0, Nx)
        d_i = rng.integers(0, Nd)
        U = gauge[t_i, z_i, y_i, x_i, d_i]
        devs.append(np.abs(U @ U.conj().T - np.eye(Nc)).max())
    results["unitary_dev_max"] = float(np.max(devs))
    results["unitary_dev_mean"] = float(np.mean(devs))

    # Link trace statistics
    traces = [np.trace(gauge[rng.integers(0,Nt), rng.integers(0,Nz),
                            rng.integers(0,Ny), rng.integers(0,Nx), rng.integers(0,Nd)])
              for _ in range(n_check)]
    results["trace_mean_re"] = float(np.real(np.mean(traces)))
    results["trace_mean_im"] = float(np.imag(np.mean(traces)))

    # Plaquette trace (50 random plaquettes)
    plaq_traces = []
    for _ in range(50):
        ti, zi, yi, xi = rng.integers(0,Nt), rng.integers(0,Nz), rng.integers(0,Ny), rng.integers(0,Nx)
        for mu in range(3):
            for nu in range(mu+1, 4):
                U1 = gauge[ti, zi, yi, xi, mu]
                idx2 = [ti, zi, yi, xi]; idx2[3-nu] = (idx2[3-nu]+1) % Nx
                U2 = gauge[tuple(idx2+[nu])]
                idx3 = [ti, zi, yi, xi]; idx3[3-mu] = (idx3[3-mu]+1) % Nx
                U3 = gauge[tuple(idx3+[mu])].conj().T
                U4 = gauge[ti, zi, yi, xi, nu].conj().T
                plaq_traces.append(np.trace(U1 @ U2 @ U3 @ U4))
    results["plaq_trace_mean_re"] = float(np.real(np.mean(plaq_traces)))

    if logger:
        logger.info(f"  Gauge [{tag}]: unitarity={results['unitary_dev_max']:.2e}, "
                    f"trace_re={results['trace_mean_re']:.4f}, plaq_re={results['plaq_trace_mean_re']:.6f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# OPE computation — donghx operators_new_z0_mu2 (the CORRECT algorithm)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ope_donghx_gpu(
    gauge_gpu: cp.ndarray,        # (Nt,Nz,Ny,Nx,4,3,3) on GPU
    F_dict: dict,                 # {(mu,nu): F_mu_nu_gpu} or None for lazy compute
    mu: int, nu: int,
    z_dir: int, delta_z: int,
    Nt: int, Nx: int,
    logger,
    save_fmunu: bool = False,
    fmunu_output_dir: Optional[Path] = None,
) -> np.ndarray:
    r"""OPE operator matching donghx operators_new_z0_mu2.

    Algorithm (roll-based, exact match to donghx):
      1. Compute F and F̃ (dual) for needed components
      2. For each z displacement:
         a. Roll F forward to position z
         b. Transport backward along Wilson line: F(z)·U^dag(z-1)·...·U^dag(0)
         c. Multiply by F̃ at origin
         d. Transport forward along Wilson line: ...→ F(z)·W^dag·F̃(0)·W
         e. Color trace + sum over ALL spatial axes

    The dual field strength F̃ mixes electric and magnetic components —
    this is the PHYSICALLY CORRECT definition per donghx's formulation.

    Returns: (delta_z, Nt) complex array (CPU, in compute dtype).
    """
    compute_dt = get_compute_dtype()
    logger.info(f"  OPE (donghx, GPU, {compute_dt}): mu={mu}, nu={nu}, "
                f"z_dir={z_dir}, dz={delta_z}")

    if mu == nu:
        return np.zeros((delta_z, Nt), dtype=compute_dt)

    t0 = time.perf_counter()
    z_axis = 3 - z_dir  # Map z_dir (0=t,1=z,2=y,3=x) to spatial axis in (t,z,y,x)

    # ── Step 1: Compute F and F̃ lazily (GPU memory efficient) ───────────────
    # Determine which F components are needed for F̃ using Tensor4
    need_pairs = {(mu, nu)}
    for rho in range(4):
        for sigma in range(4):
            if abs(_TENSOR4_NP[mu, nu, rho, sigma]) > 1e-10 and rho != sigma:
                need_pairs.add((rho, sigma))

    # Compute or fetch from cache
    local_F = {}
    if F_dict is not None:
        # Use precomputed dict (all 12 components available)
        for pair in need_pairs:
            local_F[pair] = F_dict[pair]
    else:
        # Compute on-the-fly (lazy, saves GPU memory)
        for pair in need_pairs:
            local_F[pair] = plaquette_clover_gpu(gauge_gpu, pair[0], pair[1])

    F = local_F[(mu, nu)]

    with Timer(f"Ftilde_mu{mu}_nu{nu}", logger):
        F_tilde = compute_dual_field_strength_munu(local_F, mu, nu)
    logger.info(f"    Ftilde_{mu}_{nu} (GPU): shape={F_tilde.shape}")

    # Optionally save F and F̃ to disk for diagnostics
    if save_fmunu and fmunu_output_dir:
        fmunu_output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(fmunu_output_dir / f"Fmunu_mu{mu}_nu{nu}.npz", F_munu=cp.asnumpy(F))
        np.savez(fmunu_output_dir / f"Ftilde_mu{mu}_nu{nu}.npz", Ftilde=cp.asnumpy(F_tilde))
        logger.info(f"    [SAVE] Fmunu + Ftilde for mu{mu}_nu{nu}")

    # Free local F dict — no longer needed
    del local_F

    # ── Step 2: OPE matching donghx's roll-based algorithm ─────────────────
    # U_z = gauge link along Wilson line direction, for all positions
    U_z = gauge_gpu[..., z_dir, :, :]  # (Nt,Nz,Ny,Nx,3,3)

    # Sum over ALL spatial axes (z,y,x) — matches donghx np.sum(ans, axis=(1,2,3))
    spatial_axes = (1, 2, 3)
    ope = np.zeros((delta_z, Nt), dtype=compute_dt)

    ope_start = time.perf_counter()
    for zi in range(delta_z):
        if zi == 0:
            # z=0: Tr[F·F̃] summed over space
            ope_tensor = cp.einsum("tzyxab,tzyxba->tzyx", F, F_tilde)
            ope[0, :] = cp.asnumpy(cp.sum(ope_tensor, axis=spatial_axes))
            continue

        # --- ope_tensor = F(z), rolled to origin along z_dir ---
        ope_tensor = cp.roll(F, -zi, axis=z_axis)

        # --- Backward Wilson line: ope = ope · U^dag(z-1) · ... · U^dag(0) ---
        for step in range(zi):
            shift = zi - 1 - step
            U_conj = cp.roll(U_z, -shift, axis=z_axis).conj()
            # "...ab,...cb->...ac" = ope_tensor · U_conj^T
            # (ope_tensor right-multiplied by U^dag)
            ope_tensor = cp.einsum("...ab,...cb->...ac", ope_tensor, U_conj)

        # --- Multiply by F_tilde at origin ---
        ope_tensor = cp.einsum("...ab,...bc->...ac", ope_tensor, F_tilde)

        # --- Forward Wilson line: ope = ope · U(0) · ... · U(z-1) ---
        for step in range(zi):
            U_fwd = cp.roll(U_z, -step, axis=z_axis)
            # "...ab,...bc->...ac" = ope_tensor · U_fwd (right multiply)
            ope_tensor = cp.einsum("...ab,...bc->...ac", ope_tensor, U_fwd)

        # --- Color trace (Tr over 3×3 color) + spatial sum ---
        trace = cp.einsum("...aa->...", ope_tensor)
        ope[zi, :] = cp.asnumpy(cp.sum(trace, axis=spatial_axes))

        # FIXED (v20260802): removed duplicate log block from v20260730 lines 389-397
        if zi % 4 == 0 or zi == delta_z - 1:
            gpu_mem = get_gpu_memory_mb()
            logger.debug(f"    z={zi:3d}/{delta_z}  |O|_max={np.abs(ope[zi]).max():.4e}  "
                         f"{time.perf_counter()-ope_start:.1f}s  GPU free={gpu_mem['free_mb']:.0f}MB")

    elapsed = time.perf_counter() - t0
    logger.info(f"    OPE (donghx GPU) mu={mu},nu={nu} done in {elapsed:.1f}s  "
                f"|O| range: [{np.abs(ope).min():.2e},{np.abs(ope).max():.2e}]")
    assert np.all(np.isfinite(ope)), f"OPE mu={mu},nu={nu} has NaN/inf"
    return ope


def save_ope_data(ope, output_dir, mu, nu, conf_id, delta_z, logger) -> Path:
    """Save OPE result as .npz with metadata.

    Args:
        ope: OPE array, shape (delta_z, Nt), complex.
        output_dir: Directory to save to.
        mu, nu: Lorentz indices.
        conf_id: Configuration ID.
        delta_z: Maximum z displacement.
        logger: Logger instance.

    Returns:
        Path to the saved file.
    """
    fn = f"ops_mu{mu}_nu{nu}_dz{delta_z}_conf{conf_id}.npz"
    p = output_dir / fn
    np.savez(p, ops=ope, mu=np.array(mu), nu=np.array(nu),
             delta_z=np.array(delta_z), conf_id=np.array(conf_id),
             shape=np.array(ope.shape))
    logger.info(f"    Saved {fn}: shape={ope.shape}, {p.stat().st_size/1024:.1f} KB, "
                f"|O|∈[{np.abs(ope).min():.2e},{np.abs(ope).max():.2e}]")
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ope_all_configs_gpu(config: dict, output_dir: Path, logger) -> dict:
    """Compute OPE for all configs using the CORRECT donghx algorithm (dual F̃).

    Workflow per config:
      1. Read .lime gauge config → CPU complex128 → GPU compute dtype
      2. Validate gauge (unitarity, trace)
      3. For each (mu,nu) component: compute F, F̃, and OPE operator
      4. Save results as .npz files

    Args:
        config: Full pipeline config dict.
        output_dir: Base data output directory.
        logger: Logger instance.

    Returns:
        dict mapping conf_id → results with status and component details.
    """
    params = config["parameters"]
    paths = config["data_paths"]
    ensemble = config["ensemble"]

    Nt, Nx, Nc = ensemble["Nt"], ensemble["Nx"], ensemble["Nc"]
    conf_ids = params["conf_ids"]
    delta_z, z_dir = params["delta_z"], params["z_dir"]
    gauge_base = paths["gauge_config_base"]
    gauge_pattern = paths["gauge_config_pattern"]
    save_fmunu = params.get("save_intermediate_fmunu", True)

    # OPE components matching Calc_ope_unpol.py with zdir=2:
    # Rank 0: (3,0), Rank 1: (3,1), Rank 2: (0,1)
    components = [(0, 1), (3, 0), (3, 1)]

    print_banner("Step 02: Compute OPE from Gauge Configs (GPU, donghx algorithm)", logger)
    log_gpu_status(logger, "  ")
    logger.info(f"  Compute dtype: {get_compute_dtype()}")
    logger.info(f"  Ensemble: {ensemble['full_name']} | {Nt}x{Nx}^3 | Nc={Nc}")
    logger.info(f"  Components: {components} | dz={delta_z} | z_dir={z_dir}")
    logger.info(f"  Configs: {conf_ids}")
    logger.info(f"  Algorithm: operators_new_z0_mu2 (F(z)·W^dag·F̃(0)·W) — donghx标准算法")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for iconf, conf_id in enumerate(conf_ids):
        conf_out = output_dir / f"conf_{conf_id}"
        conf_out.mkdir(parents=True, exist_ok=True)

        gauge_file = os.path.join(gauge_base, gauge_pattern.format(conf_id=conf_id))
        logger.info(f"\n{'─'*60}")
        logger.info(f"  conf_id={conf_id} [{iconf+1}/{len(conf_ids)}]")
        logger.info(f"  Gauge: {gauge_file}")

        if not os.path.exists(gauge_file):
            logger.error(f"  [ERROR] Gauge file not found")
            all_results[conf_id] = {"status": "missing", "reason": f"not found: {gauge_file}"}
            continue

        # ── Read gauge: CPU → GPU ──────────────────────────────────────────
        try:
            with Timer(f"read_gauge_conf{conf_id}", logger, output_dir.parent,
                      extra={"conf_id": conf_id}):
                gauge_cpu = read_gauge_lime(gauge_file, Nt, Nx, Nc)
                compute_dt = get_compute_dtype()
                gauge_gpu = cp.asarray(gauge_cpu.astype(compute_dt, copy=False))
                del gauge_cpu
                gpu_sync()
                gpu_mem = get_gpu_memory_mb()
                logger.info(f"  Gauge GPU: dtype={gauge_gpu.dtype}, "
                           f"mem={gauge_gpu.nbytes/1024**2:.1f} MB, GPU free={gpu_mem['free_mb']:.0f} MB")
        except Exception as e:
            logger.error(f"  [ERROR] Gauge read failed: {e}")
            log_exception(logger, e, f"Reading gauge {conf_id}")
            all_results[conf_id] = {"status": "error", "reason": str(e)}
            continue

        # ── Validate gauge ─────────────────────────────────────────────────
        gauge_cpu_val = cp.asnumpy(gauge_gpu)
        with Timer(f"validate_gauge_conf{conf_id}", logger, output_dir.parent,
                  extra={"conf_id": conf_id}):
            val = validate_gauge(gauge_cpu_val, f"conf_{conf_id}", logger)
        save_intermediate(val, conf_out, f"gauge_validation_conf{conf_id}.json", logger)
        del gauge_cpu_val

        # ── Compute OPE components (lazy F computation for GPU memory) ─────
        conf_results = {"status": "ok", "components": {}, "validation": val}
        all_ok = True

        for mu, nu in components:
            key = f"mu{mu}_nu{nu}"
            try:
                with Timer(f"ope_GPU_{key}_conf{conf_id}", logger, output_dir.parent,
                          extra={"conf_id": conf_id, "mu": mu, "nu": nu, "device": "gpu",
                                 "dtype": str(get_compute_dtype()), "algorithm": "donghx"}):
                    ope = compute_ope_donghx_gpu(
                        gauge_gpu, None, mu, nu, z_dir, delta_z,
                        Nt, Nx, logger,
                        save_fmunu=save_fmunu, fmunu_output_dir=conf_out)

                path = save_ope_data(ope, conf_out, mu, nu, conf_id, delta_z, logger)
                conf_results["components"][key] = {
                    "status": "ok", "output": str(path),
                    "shape": list(ope.shape),
                    "re_range": [float(ope.real.min()), float(ope.real.max())],
                    "im_range": [float(ope.imag.min()), float(ope.imag.max())],
                    "re_mean": float(ope.real.mean()),
                    "dtype": str(ope.dtype),
                }
            except Exception as e:
                logger.error(f"  [ERROR] {key}: {e}")
                log_exception(logger, e, f"OPE {key} conf {conf_id}")
                conf_results["components"][key] = {"status": "error", "message": str(e)}
                all_ok = False

        if not all_ok:
            conf_results["status"] = "partial"
        all_results[conf_id] = conf_results
        save_intermediate(conf_results, conf_out, f"compute_ope_summary_conf{conf_id}.json", logger)

        # Free GPU memory for next config
        del gauge_gpu
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

        gpu_mem = get_gpu_memory_mb()
        logger.info(f"  [PROGRESS] OPE: {iconf+1}/{len(conf_ids)} done, GPU free={gpu_mem['free_mb']:.0f}MB")

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info(f"\n{'═'*60}")
    logger.info(f"OPE (donghx GPU, {get_compute_dtype()}) Summary:")
    for conf_id, result in all_results.items():
        cs = []
        for k in sorted(result.get("components", {}).keys()):
            c = result["components"][k]
            if c.get("status") == "ok":
                cs.append(f"{k}:|O|∈[{c['re_range'][0]:.2e},{c['re_range'][1]:.2e}]")
        s = "✓" if result["status"] == "ok" else "⚠"
        logger.info(f"  {s} conf={conf_id}: {'; '.join(cs)}")
    logger.info(f"{'═'*60}")
    return all_results


def main():
    p = argparse.ArgumentParser(description="Compute OPE (donghx algorithm, GPU)")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--conf-id", type=int, default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    config_path = args.config or (run_dir / "run_config.json")
    with open(config_path) as f:
        config = json.load(f)

    if args.conf_id is not None:
        config["parameters"]["conf_ids"] = [args.conf_id]
        config["parameters"]["Nconf"] = 1

    output_dir = run_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    from utils import setup_logging
    logger = setup_logging(run_dir / "run.log", "compute_ope_gpu")
    results = compute_ope_all_configs_gpu(config, output_dir, logger)
    return 0 if all(r["status"] == "ok" for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
