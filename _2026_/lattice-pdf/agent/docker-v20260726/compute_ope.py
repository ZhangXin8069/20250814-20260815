#!/usr/bin/env python3
"""
OPE (Operator Product Expansion) computation from gauge configurations.

Enhanced version (docker-v20260726):
  - OPE computed FROM SCRATCH: gauge config → F_{μν} → nonlocal OPE → .npz
  - All intermediate results saved:
    * F_{μν} field strength tensors for each (mu,nu) component
    * OPE operator arrays for each (mu,nu) component
    * Gauge validation diagnostics
  - Per-component error recovery
  - Comprehensive timing and memory tracking
  - Detailed logging to module-specific log file

Algorithm:
    1. Read gauge configuration (.lime ILDG format, big-endian float64)
    2. Validate gauge (unitarity, trace)
    3. Compute F_{mu,nu}(x) via clover plaquette → Save as intermediate
    4. For each z ∈ [0, delta_z-1]:
       a. Shift F to position z along z_dir
       b. Build Wilson line W(z→0) as product of U_z^dag
       c. Contract: O(z) = Σ_{x_perp} Tr[F(z) * W^dag * F(0) * W]
    5. Save OPE as .npz

Usage (standalone):
    python compute_ope.py --run-dir /path/to/output

Usage (imported):
    from compute_ope import compute_ope_all_configs
    results = compute_ope_all_configs(config, output_dir, logger)
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

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent  # /root/lattice-pdf
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from utils import (
    Timer, print_banner, format_size, save_intermediate,
    validate_array, progress_bar, log_exception,
    get_current_memory_mb,
)

# ─── Import from snsc/main.py ────────────────────────────────────────────────
from snsc.main import plaquette_clover


# ═══════════════════════════════════════════════════════════════════════════════
# Gauge configuration reader
# ═══════════════════════════════════════════════════════════════════════════════

def read_gauge_lime(filepath: str, Nt: int, Nx: int, Nc: int = 3) -> np.ndarray:
    """Read a gauge configuration in ILDG/LIME big-endian float64 format.

    The .lime file stores the gauge field as a flat array of
    big-endian float64 values, reshaped to:
        (Nt, Nx, Nx, Nx, 4, Nc, Nc, 2)  → complex (real, imag)

    donghx convention: gauge[t, z, y, x, dir, color, color]
    Dir mapping: 0=x, 1=y, 2=z, 3=t in the convention of donghx.
    BUT .lime stores as: (t, z, y, x, dir, color, color) with dir: 0=x, 1=y, 2=z, 3=t

    Args:
        filepath: Path to .lime file.
        Nt: Temporal extent.
        Nx: Spatial extent.
        Nc: Number of colors (default 3).

    Returns:
        gauge: shape (Nt, Nx, Nx, Nx, 4, Nc, Nc), dtype complex128.
    """
    file_size_mb = os.path.getsize(filepath) / 1024**2
    with open(filepath, "rb") as f:
        raw = np.fromfile(f, dtype=">f8")  # big-endian float64

    expected = Nt * Nx * Nx * Nx * 4 * Nc * Nc * 2
    if raw.size != expected:
        raise ValueError(
            f"Gauge file size mismatch: expected {expected} doubles (= {expected*8/1024**2:.1f} MB), "
            f"got {raw.size} doubles (= {raw.size*8/1024**2:.1f} MB)\n"
            f"  File: {filepath}\n"
            f"  Expected shape: ({Nt},{Nx},{Nx},{Nx},4,{Nc},{Nc},2)"
        )

    raw = raw.reshape(Nt, Nx, Nx, Nx, 4, Nc, Nc, 2)
    gauge = raw[..., 0] + 1j * raw[..., 1]
    return gauge.astype(np.complex128)


def read_gauge_lime_with_header_check(filepath: str, Nt: int, Nx: int, Nc: int, logger) -> np.ndarray:
    """Read gauge config with diagnostic header inspection.

    The ILDG/LIME format has a header section with XML metadata,
    followed by binary float64 gauge data.

    Strategy: the binary gauge data occupies the last expected_data_bytes
    of the file. We compute the header size from (file_size - expected_data_bytes)
    and scan around that offset for valid SU(3) data.
    """
    file_size = os.path.getsize(filepath)
    expected_elements = Nt * Nx * Nx * Nx * 4 * Nc * Nc * 2
    expected_data_bytes = expected_elements * 8

    logger.info(f"  Gauge file size: {file_size / 1024**2:.1f} MB ({file_size:,} bytes)")
    logger.info(f"  Expected data: {expected_data_bytes / 1024**2:.1f} MB "
                f"({expected_elements:,} doubles)")
    logger.info(f"  Header approx: {(file_size - expected_data_bytes) / 1024:.1f} KB "
                f"({file_size - expected_data_bytes:,} bytes)")

    # Read only the tail of the file: the gauge data is at the end
    approx_header = file_size - expected_data_bytes
    logger.info(f"  approx_header = {approx_header} bytes")

    # Scan backwards from approx_header (±2KB in 8-byte steps), then forward if no match
    candidates = []
    # Primary scan: around approx_header
    for delta in range(-4096, 4097, 8):
        off = approx_header + delta
        if 0 <= off <= file_size - expected_data_bytes:
            candidates.append(off)
    # Fallback: scan forward from 0 in larger steps
    for off in range(0, min(file_size - expected_data_bytes + 1, 262144), 512):
        if off not in candidates:
            candidates.append(off)

    # Deduplicate and sort by proximity to approx_header (closest first)
    candidates = sorted(set(candidates), key=lambda x: abs(x - approx_header))

    logger.info(f"  Scanning {len(candidates)} candidate offsets "
                f"(prioritizing near approx_header={approx_header})...")

    with open(filepath, "rb") as f:
        raw_bytes = f.read()

    best_dev = float('inf')
    best_offset = None
    best_gauge = None

    n_checked = 0
    for off_bytes in candidates:
        if off_bytes + expected_data_bytes > file_size:
            continue

        chunk = raw_bytes[off_bytes:off_bytes + expected_data_bytes]
        test_raw = np.frombuffer(chunk, dtype=">f8", count=expected_elements)

        if test_raw.size != expected_elements:
            continue

        test = test_raw.reshape(Nt, Nx, Nx, Nx, 4, Nc, Nc, 2)
        test_gauge = test[..., 0] + 1j * test[..., 1]

        # Check if values are in reasonable range for SU(3) gauge links
        # (no overflow, no inf, reasonable magnitude)
        abs_max = np.abs(test_gauge).max()
        if not np.isfinite(abs_max) or abs_max > 100.0:
            n_checked += 1
            continue

        # Quick unitarity check on a handful of random links
        rng = np.random.default_rng(42 + off_bytes % 256)
        devs = []
        try:
            for _ in range(20):
                t = rng.integers(0, Nt)
                z = rng.integers(0, Nx)
                y = rng.integers(0, Nx)
                xx = rng.integers(0, Nx)
                d = rng.integers(0, 4)
                U = test_gauge[t, z, y, xx, d]
                dev = np.max(np.abs(U @ U.conj().T - np.eye(Nc)))
                if np.isfinite(dev):
                    devs.append(dev)
        except Exception:
            continue

        if not devs:
            continue

        max_dev = max(devs)
        mean_dev = sum(devs) / len(devs)

        # Track the best candidate (lowest mean unitarity deviation)
        if mean_dev < best_dev and max_dev < 1e-3:
            best_dev = mean_dev
            best_offset = off_bytes
            best_gauge = test_gauge

        n_checked += 1
        if n_checked <= 3 or n_checked % 200 == 0:
            logger.debug(f"    offset={off_bytes}: abs_max={abs_max:.2e}, "
                        f"max_dev={max_dev:.2e}, mean_dev={mean_dev:.2e}")

        # Early exit: if unitarity is excellent and we're near approx_header
        if max_dev < 1e-6 and mean_dev < 1e-6 and abs(off_bytes - approx_header) < 4096:
            logger.info(f"  Found valid gauge data at offset {off_bytes} bytes "
                       f"(unitarity: max_dev={max_dev:.2e}, mean_dev={mean_dev:.2e})")
            return test_gauge.astype(np.complex128)

    if best_gauge is not None:
        logger.info(f"  Using best gauge data at offset {best_offset} bytes "
                   f"(unitarity: mean_dev={best_dev:.2e})")
        return best_gauge.astype(np.complex128)

    raise ValueError(
        f"Could not find valid gauge data in {filepath}. "
        f"Scanned {len(candidates)} offsets, none produced unitary SU(3) data. "
        f"File size: {file_size}, expected data: {expected_data_bytes}"
    )


def validate_gauge(gauge: np.ndarray, tag: str = "", logger=None) -> dict:
    """Comprehensive validation of a gauge configuration.

    Returns dict with keys: unitary_deviation, trace_mean, plaq_trace_mean, shape, tag.
    """
    Nt, Nz, Ny, Nx, Nd, Nc, _ = gauge.shape
    results = {"shape": list(gauge.shape), "tag": tag}

    # Check unitarity: U @ U^dag ≈ I on ~200 random sites
    rng = np.random.default_rng(42)
    n_check = min(200, Nt * Nz * Ny * Nx)
    devs = []
    for _ in range(n_check):
        t = rng.integers(0, Nt)
        z = rng.integers(0, Nz)
        y = rng.integers(0, Ny)
        x = rng.integers(0, Nx)
        d = rng.integers(0, Nd)
        U = gauge[t, z, y, x, d]
        dev = np.max(np.abs(U @ U.conj().T - np.eye(Nc)))
        devs.append(dev)

    results["unitary_dev_max"] = float(np.max(devs))
    results["unitary_dev_mean"] = float(np.mean(devs))
    results["unitary_dev_median"] = float(np.median(devs))

    # Trace mean
    traces = []
    for _ in range(n_check):
        t = rng.integers(0, Nt)
        z = rng.integers(0, Nz)
        y = rng.integers(0, Ny)
        x = rng.integers(0, Nx)
        d = rng.integers(0, Nd)
        traces.append(np.trace(gauge[t, z, y, x, d]))
    results["trace_mean_re"] = float(np.real(np.mean(traces)))
    results["trace_mean_im"] = float(np.imag(np.mean(traces)))
    results["det_mean"] = float(np.mean([np.linalg.det(gauge[t, z, y, x, d])
                                         for t, z, y, x, d in
                                         zip(rng.integers(0, Nt, 20),
                                             rng.integers(0, Nz, 20),
                                             rng.integers(0, Ny, 20),
                                             rng.integers(0, Nx, 20),
                                             rng.integers(0, Nd, 20))]))

    # Plaquette trace (1x1 Wilson loop)
    plaq_traces = []
    for _ in range(50):
        t = rng.integers(0, Nt)
        z = rng.integers(0, Nz)
        y = rng.integers(0, Ny)
        x = rng.integers(0, Nx)
        for mu in range(3):  # spatial directions
            for nu in range(mu + 1, 4):
                U1 = gauge[t, z, y, x, mu]
                # neighbor in nu direction
                idx2 = [t, z, y, x]
                idx2[3 - nu] = (idx2[3 - nu] + 1) % Nx
                U2 = gauge[tuple(idx2 + [nu])]
                # neighbor in mu direction from x+nu
                idx3 = [t, z, y, x]
                idx3[3 - mu] = (idx3[3 - mu] + 1) % Nx
                U3 = gauge[tuple(idx3 + [mu])].conj().T
                U4 = gauge[t, z, y, x, nu].conj().T
                plaq = np.trace(U1 @ U2 @ U3 @ U4)
                plaq_traces.append(plaq)

    results["plaq_trace_mean_re"] = float(np.real(np.mean(plaq_traces)))
    results["plaq_trace_mean_im"] = float(np.imag(np.mean(plaq_traces)))

    if logger:
        logger.info(f"  Gauge validation [{tag}]:")
        logger.info(f"    Unitarity: max_dev={results['unitary_dev_max']:.2e}, "
                    f"mean={results['unitary_dev_mean']:.2e}")
        logger.info(f"    Trace: re={results['trace_mean_re']:.4f}, "
                    f"im={results['trace_mean_im']:.4f}")
        logger.info(f"    Det mean: {results['det_mean']:.6f}")
        logger.info(f"    Plaq trace: re={results['plaq_trace_mean_re']:.6f}, "
                    f"im={results['plaq_trace_mean_im']:.6f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# OPE computation per component
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ope_per_component(
    gauge: np.ndarray,       # (Nt, Nz, Ny, Nx, 4, 3, 3)
    mu: int,                  # First Lorentz index of F
    nu: int,                  # Second Lorentz index of F
    z_dir: int,               # Wilson line direction (0=x, 1=y, 2=z)
    delta_z: int,             # Max z separation
    Nt: int,
    Nx: int,
    contract_fn,
    logger,
    save_fmunu: bool = False,
    fmunu_output_dir: Optional[Path] = None,
) -> np.ndarray:
    r"""Compute OPE operator for a single (mu, nu) component.

    Algorithm:
      1. Compute F_{mu,nu}(x) via clover plaquette → shape (Nt,Nz,Ny,Nx,3,3)
      2. For each z ∈ [0, delta_z-1]:
         a. Shift F to position z along z_dir
         b. Build Wilson line W(z→0) as product of U_z^dag
         c. Contract: O(z) = Σ_{x_perp} Tr[F(z) * W^dag * F(0) * W]
      3. Return ope of shape (Nx, Nt)

    Args:
        gauge: Gauge links, shape (Nt, Nz, Ny, Nx, 4, 3, 3), donghx convention.
        mu, nu: Lorentz indices of F_{mu,nu}.
        z_dir: Wilson line direction (0=x, 1=y, 2=z).
        delta_z: Maximum Wilson line length in lattice units.
        Nt, Nx: Lattice extents.
        contract_fn: Einsum-like contraction function.
        logger: Logger instance.
        save_fmunu: If True, save F_{mu,nu} as intermediate result.
        fmunu_output_dir: Directory to save F_{mu,nu} if save_fmunu=True.

    Returns:
        ope: shape (Nx, Nt), complex128.
    """
    logger.info(f"  Computing OPE: mu={mu}, nu={nu}, z_dir={z_dir}, delta_z={delta_z}")

    if mu == nu:
        logger.warning(f"  mu=nu={mu}, F is identically zero. Returning zeros.")
        return np.zeros((Nx, Nt), dtype=complex)

    t_start = time.perf_counter()

    # ═══ Step 1: Field strength tensor F_{mu,nu} via clover plaquette ═══════
    with Timer(f"Fmunu_mu{mu}_nu{nu}", logger):
        F_munu = plaquette_clover(gauge, mu, nu, contract_fn)
        # Shape: (Nt, Nz, Ny, Nx, 3, 3)

    F_shape = F_munu.shape
    F_mem_mb = F_munu.nbytes / 1024**2
    logger.info(f"    F_{{{mu},{nu}}} shape: {F_shape}, mem: {F_mem_mb:.1f} MB")
    logger.info(f"    |F| range: [{np.abs(F_munu).min():.2e}, {np.abs(F_munu).max():.2e}]")

    # Validate F
    assert np.all(np.isfinite(F_munu)), f"F_{mu},{nu} contains NaN/inf"

    # Save F_{mu,nu} as intermediate result
    if save_fmunu and fmunu_output_dir is not None:
        fmunu_output_dir.mkdir(parents=True, exist_ok=True)
        fmunu_path = fmunu_output_dir / f"Fmunu_mu{mu}_nu{nu}.npz"
        np.savez(fmunu_path, F_munu=F_munu)
        logger.info(f"    [SAVE] F_{mu},{nu}: {fmunu_path.name}")

    # ═══ Step 2: Compute OPE for each z ══════════════════════════════════════
    # The 4 axes in donghx convention are: (t=0, z=1, y=2, x=3)
    # z_dir maps: 0→x (axis 3), 1→y (axis 2), 2→z (axis 1)
    z_axis = 3 - z_dir  # Axis in the (t,z,y,x) ordering

    # Identify perpendicular axes (the two spatial axes NOT z_dir)
    spatial_axes = [1, 2, 3]  # (z, y, x) in donghx ordering
    perp_axes = [a for a in spatial_axes if a != z_axis]
    logger.debug(f"    z_dir={z_dir}, z_axis={z_axis}, perp_axes={perp_axes}")

    # ── Precompute cumulative products of U^dag along z_dir ──────────────────
    # U_dir = gauge links in z_dir direction: shape (Nt, Nz, Ny, Nx, 3, 3)
    U_dir = gauge[..., z_dir, :, :]
    # C_cum[z] = ∏_{k=0}^{z-1} U^dag(k) for z=0..Nx
    # C_cum[0] = I (identity), C_cum[z] = U^dag(z-1) @ C_cum[z-1]
    C_cum = np.zeros((Nx, Nt, Nx, Nx, 3, 3), dtype=complex)
    eye_batch = np.tile(np.eye(3, dtype=complex), (Nt, Nx, Nx, 1, 1))
    C_cum[0] = eye_batch.copy()

    logger.debug(f"    Precomputing cumulative U^dag products along z_dir (Nx={Nx})...")
    t_cum = time.perf_counter()
    for z in range(1, Nx):
        # U^dag at position z-1: shape (Nt, Ny, Nx, 3, 3)
        # Need to access U_dir[:, z-1, :, :, :, :] and conjugate-transpose the color indices
        Udag = U_dir[:, z-1, :, :, :, :].conj()
        Udag = Udag.transpose(0, 1, 2, 4, 3)  # (Nt, Ny, Nx, 3, 3) with color axes swapped
        # Batched matmul: C_cum[z] = Udag @ C_cum[z-1]
        C_cum[z] = np.einsum('tyxab,tyxbc->tyxac', Udag, C_cum[z-1])
    logger.debug(f"    Cumulative products done in {time.perf_counter()-t_cum:.1f}s")

    # ── Also precompute inverses of C_cum (3x3 LU per site, very fast) ───────
    # C_inv[z] = C_cum[z]^{-1}, shape (Nx, Nt, Nx, Nx, 3, 3)
    C_inv = np.zeros_like(C_cum)
    t_inv = time.perf_counter()
    for z in range(Nx):
        # (Nt, Nx, Nx) independent 3x3 matrices — vectorized via np.linalg.inv
        C_inv[z] = np.linalg.inv(C_cum[z])
    logger.debug(f"    C_cum inverses done in {time.perf_counter()-t_inv:.1f}s")

    ope = np.zeros((delta_z, Nt), dtype=complex)

    for zi in range(delta_z):
        t_z_start = time.perf_counter()

        if zi == 0:
            # z=0: Tr[F * F] at origin (no Wilson line needed)
            F0_sq = np.einsum("tzyxab,tzyxba->tzyx", F_munu, F_munu)
            ope[zi, :] = np.sum(F0_sq, axis=tuple(perp_axes))[:, 0]
            continue

        # Step 2a: Shift F to position z = zi along z_dir
        F_at_z = np.roll(F_munu, -zi, axis=z_axis)

        # Step 2b: Build W_dag using precomputed cumulative products
        # W_dag(z) = C_cum[(z+Nx-zi)%Nx]^{-1} @ C_cum[z]  (going backward zi steps)
        # For z >= zi: W_dag(z) = C_inv[z-zi] @ C_cum[z]
        # For z < zi:  W_dag(z) = C_inv[z+Nx-zi] @ C_cum[z]  (wrap around)
        W_dag = np.zeros_like(F_at_z, dtype=complex)

        # Build W_dag efficiently using the precomputed C_cum and C_inv
        for z in range(Nx):
            z_src = (z - zi) % Nx  # source index after going back zi steps
            # W_dag at this z: C_inv[z_src] @ C_cum[z]
            # Broadcast: (Nt, Ny, Nx,) 3x3 matmul
            W_dag_z = np.einsum(
                'tyxab,tyxbc->tyxac',
                C_inv[z_src], C_cum[z]
            )
            # Place in the z_dir axis
            slices = [slice(None)] * 4  # (Nt, Nz, Ny, Nx)
            slices[z_axis] = z
            W_dag[tuple(slices)] = W_dag_z

        # W_fwd = W = (W_dag)^dag
        W_fwd = W_dag.conj().transpose(0, 1, 2, 3, 5, 4)

        # Step 2c: O(z) = Tr[F(z) * W^dag * F(0) * W]
        Fz_Wdag = contract_fn("tzyxab,tzyxbc->tzyxac", F_at_z, W_dag)
        Fz_Wdag_F0 = contract_fn("tzyxab,tzyxbc->tzyxac", Fz_Wdag, F_munu)
        Fz_Wdag_F0_W = contract_fn("tzyxab,tzyxba->tzyx", Fz_Wdag_F0, W_fwd)

        # Step 2d: Sum over perpendicular directions → (Nt, Nz_zdir)
        summed = np.sum(Fz_Wdag_F0_W, axis=tuple(perp_axes))
        ope[zi, :] = summed[:, 0] if summed.ndim == 2 else summed

        t_z_elapsed = time.perf_counter() - t_z_start
        if zi % 8 == 0 or zi == delta_z - 1:
            logger.debug(
                f"    z={zi:3d}/{delta_z}  |O|_max={np.abs(ope[zi]).max():.4e}  "
                f"time={t_z_elapsed:.1f}s"
            )

    elapsed = time.perf_counter() - t_start
    logger.info(
        f"    OPE(mu={mu},nu={nu}) complete in {elapsed:.1f}s  "
        f"|O| range: [{np.abs(ope).min():.2e}, {np.abs(ope).max():.2e}]"
    )

    # Validate
    assert np.all(np.isfinite(ope)), f"OPE mu={mu},nu={nu} contains NaN/inf"
    assert ope.shape == (delta_z, Nt), f"OPE shape mismatch: expected ({delta_z},{Nt}), got {ope.shape}"

    return ope


# ═══════════════════════════════════════════════════════════════════════════════
# Save OPE data
# ═══════════════════════════════════════════════════════════════════════════════

def save_ope_data(
    ope: np.ndarray,
    output_dir: Path,
    mu: int,
    nu: int,
    conf_id: int,
    delta_z: int,
    logger,
) -> Path:
    """Save OPE data in standard naming convention.

    Format: ops_mu{mu}_nu{nu}_dz{delta_z}_conf{conf_id}.npz
    With keys: 'ops' (complex128 array), 'mu', 'nu', 'delta_z', 'conf_id', 'shape'
    """
    fname = f"ops_mu{mu}_nu{nu}_dz{delta_z}_conf{conf_id}.npz"
    path = output_dir / fname
    np.savez(
        path,
        ops=ope,
        mu=np.array(mu),
        nu=np.array(nu),
        delta_z=np.array(delta_z),
        conf_id=np.array(conf_id),
        shape=np.array(ope.shape),
    )
    size_kb = path.stat().st_size / 1024
    logger.info(f"    Saved {fname}: shape={ope.shape}, {size_kb:.1f} KB, "
                f"|O| range [{np.abs(ope).min():.2e}, {np.abs(ope).max():.2e}]")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Main computation entry point
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ope_all_configs(
    config: dict,
    output_dir: Path,
    logger,
) -> dict:
    """Compute OPE for all configurations FROM SCRATCH using gauge configs.

    For each configuration:
      1. Read gauge config (.lime ILDG format)
      2. Validate gauge (unitarity, trace, plaquette)
      3. Compute F_{mu,nu} via clover plaquette for each component
      4. Save F_{mu,nu} as intermediate result
      5. Compute nonlocal OPE operator for each (mu,nu)
      6. Save OPE as .npz

    Args:
        config: Full configuration dict (from run_config.json).
        output_dir: Output directory (data saved under conf_{id}/).
        logger: Logger instance.

    Returns:
        dict: conf_id → {status, components: {mu_nu: {path, shape, ...}}, validation: {...}}
    """
    params = config["parameters"]
    paths = config["data_paths"]
    ensemble = config["ensemble"]

    Nt = ensemble["Nt"]
    Nx = ensemble["Nx"]
    Nc = ensemble["Nc"]
    conf_ids = params["conf_ids"]
    delta_z = params["delta_z"]
    z_dir = params["z_dir"]
    gauge_base = paths["gauge_config_base"]
    gauge_pattern = paths["gauge_config_pattern"]
    save_fmunu = params.get("save_intermediate_fmunu", True)
    save_wline = params.get("save_intermediate_wilson_line", False)

    # OPE tensor components needed for unpolarized gluon:
    # (0,1) = F_xy (spatial-spatial, dominant)
    # (3,0) = F_tx (time-spatial)
    # (3,1) = F_ty (time-spatial)
    components = [(0, 1), (3, 0), (3, 1)]

    # Use opt_einsum if available, numpy.einsum as fallback
    try:
        from opt_einsum import contract as _opt_contract
        contract_fn = _opt_contract
    except ImportError:
        contract_fn = np.einsum

    print_banner("Step 02: Compute OPE from Gauge Configs (FROM SCRATCH)", logger)
    logger.info(f"  Ensemble: {ensemble['full_name']} ({ensemble['name']})")
    logger.info(f"  Lattice: {Nt}×{Nx}³, Nc={Nc}")
    logger.info(f"  Components: mu,nu ∈ {components}")
    logger.info(f"  delta_z = {delta_z}, z_dir = {z_dir} (z_dir=0:x, 1:y, 2:z)")
    logger.info(f"  Configs: {conf_ids} (Nconf={len(conf_ids)})")
    logger.info(f"  Gauge base: {gauge_base}")
    logger.info(f"  Gauge pattern: {gauge_pattern}")
    logger.info(f"  Save F_munu: {save_fmunu}")
    logger.info(f"  Contract function: {'opt_einsum' if 'opt_einsum' in str(contract_fn.__module__) else 'numpy.einsum'}")
    logger.info(f"  Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for iconf, conf_id in enumerate(conf_ids):
        conf_out_dir = output_dir / f"conf_{conf_id}"
        conf_out_dir.mkdir(parents=True, exist_ok=True)

        # ── Read gauge configuration ───────────────────────────────────────
        gauge_file = os.path.join(gauge_base, gauge_pattern.format(conf_id=conf_id))
        logger.info(f"\n{'─'*60}")
        logger.info(f"  conf_id={conf_id} [{iconf+1}/{len(conf_ids)}]")
        logger.info(f"  Gauge file: {gauge_file}")

        if not os.path.exists(gauge_file):
            logger.error(f"  [ERROR] Gauge file not found: {gauge_file}")
            all_results[conf_id] = {
                "status": "missing",
                "reason": f"gauge file not found: {gauge_file}",
            }
            continue

        # Read and validate gauge
        try:
            with Timer(f"read_gauge_conf{conf_id}", logger, output_dir.parent,
                      extra={"conf_id": conf_id}):
                try:
                    # First try simple read
                    gauge = read_gauge_lime(gauge_file, Nt, Nx, Nc)
                except Exception as e0:
                    logger.warning(f"  Simple gauge read failed ({e0}), "
                                 f"trying with header detection...")
                    gauge = read_gauge_lime_with_header_check(
                        gauge_file, Nt, Nx, Nc, logger
                    )

                logger.info(f"  Gauge shape: {gauge.shape}, dtype: {gauge.dtype}")
                logger.info(f"  Gauge memory: {gauge.nbytes / 1024**2:.1f} MB")
        except Exception as e:
            logger.error(f"  [ERROR] Failed to read gauge: {e}")
            log_exception(logger, e, f"Reading gauge config {conf_id}")
            all_results[conf_id] = {"status": "error", "reason": str(e)}
            continue

        # Validate gauge
        with Timer(f"validate_gauge_conf{conf_id}", logger, output_dir.parent,
                  extra={"conf_id": conf_id}):
            val = validate_gauge(gauge, f"conf_{conf_id}", logger)

        # Save gauge validation as intermediate result
        save_intermediate(val, conf_out_dir, f"gauge_validation_conf{conf_id}.json", logger)

        # ── Compute OPE for each component ─────────────────────────────────
        conf_results = {
            "status": "ok",
            "components": {},
            "validation": val,
        }
        all_ok = True

        for mu, nu in components:
            key = f"mu{mu}_nu{nu}"

            try:
                with Timer(f"ope_{key}_conf{conf_id}", logger, output_dir.parent,
                          extra={"conf_id": conf_id, "mu": mu, "nu": nu}):
                    ope = compute_ope_per_component(
                        gauge, mu, nu, z_dir, delta_z,
                        Nt, Nx, contract_fn, logger,
                        save_fmunu=save_fmunu,
                        fmunu_output_dir=conf_out_dir if save_fmunu else None,
                    )

                # Validate
                assert np.all(np.isfinite(ope)), f"OPE {key} contains NaN/inf"
                assert ope.shape == (delta_z, Nt), \
                    f"OPE shape mismatch: expected ({delta_z},{Nt}), got {ope.shape}"

                # Save
                path = save_ope_data(ope, conf_out_dir, mu, nu, conf_id, delta_z, logger)

                conf_results["components"][key] = {
                    "status": "ok",
                    "output": str(path),
                    "shape": list(ope.shape),
                    "re_range": [float(ope.real.min()), float(ope.real.max())],
                    "im_range": [float(ope.imag.min()), float(ope.imag.max())],
                    "re_mean": float(ope.real.mean()),
                    "im_mean": float(ope.imag.mean()),
                    "nonzero_fraction": float(np.count_nonzero(ope) / ope.size),
                }

            except Exception as e:
                logger.error(f"  [ERROR] {key}: {e}")
                log_exception(logger, e, f"Computing OPE {key} for conf {conf_id}")
                conf_results["components"][key] = {
                    "status": "error",
                    "message": str(e),
                }
                all_ok = False

        if not all_ok:
            conf_results["status"] = "partial"

        all_results[conf_id] = conf_results

        # Save per-config summary
        save_intermediate(conf_results, conf_out_dir, f"compute_ope_summary_conf{conf_id}.json", logger)

        # Free gauge memory
        del gauge
        gc.collect()

        # Log progress
        logger.info(f"  [PROGRESS] OPE: {iconf+1}/{len(conf_ids)} configs done, "
                   f"mem={get_current_memory_mb():.0f}MB")

    # ── Summary ─────────────────────────────────────────────────────────────
    logger.info(f"\n{'═'*60}")
    logger.info("OPE Computation Summary:")
    required = {f"mu{mu}_nu{nu}" for mu, nu in components}
    for conf_id, result in all_results.items():
        found_keys = {
            k for k, v in result.get("components", {}).items()
            if v.get("status") == "ok"
        }
        missing = required - found_keys
        status = "✓" if result["status"] == "ok" else "⚠"
        info_parts = []
        for k in sorted(found_keys):
            comp = result["components"][k]
            re_r = comp.get("re_range", [0, 0])
            info_parts.append(f"{k}: |O|∈[{re_r[0]:.2e},{re_r[1]:.2e}]")
        logger.info(f"  {status} conf={conf_id}: {'; '.join(info_parts)}")
        if missing:
            logger.warning(f"    Missing: {sorted(missing)}")
    logger.info(f"{'═'*60}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Compute OPE from gauge configurations FROM SCRATCH")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory")
    parser.add_argument("--config", type=str, default=None, help="Path to run_config.json")
    parser.add_argument("--conf-id", type=int, default=None,
                       help="Compute for a single config (overrides config list)")
    args = parser.parse_args()

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
    logger = setup_logging(run_dir / "run.log", "compute_ope")

    results = compute_ope_all_configs(config, output_dir, logger)

    summary_path = output_dir / "compute_ope_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Overall summary saved to {summary_path}")

    all_ok = all(r["status"] == "ok" for r in results.values())
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
