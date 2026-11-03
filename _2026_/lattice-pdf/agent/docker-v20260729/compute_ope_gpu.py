#!/usr/bin/env python3
"""
OPE computation from gauge configurations — GPU (CuPy) accelerated, single precision.

Precision flow:
  Input (disk): gauge .lime [complex128 big-endian] → CPU complex128 → GPU complex64
  GPU compute:  Clover F_{μν}, Wilson line, OPE contract — all complex64
  Output (disk): F_{μν} .npz, OPE .npz — complex64

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
# GPU plaquette_clover (independent CuPy implementation, complex64)
# ═══════════════════════════════════════════════════════════════════════════════

def plaquette_clover_gpu(gauge_gpu: cp.ndarray, mu: int, nu: int, _einsum) -> cp.ndarray:
    r"""Clover field strength F_{μν}(x) on GPU.

    F̂_{μν} = -i/8 · Σ_k (P_k - P_k†)

    gauge_gpu: (Nt,Nz,Ny,Nx,4,3,3) complex64 on GPU.
    Returns:   (Nt,Nz,Ny,Nx,3,3) complex64 on GPU.
    """
    e = _einsum  # cp.einsum
    g = gauge_gpu
    a_mu = 3 - mu  # spatial axis for mu
    a_nu = 3 - nu  # spatial axis for nu

    g_lu = cp.roll(g, 1, axis=a_mu)
    g_rd = cp.roll(g, 1, axis=a_nu)
    g_ld = cp.roll(g_lu, 1, axis=a_nu)

    # Plaquette 1: P_{μν}
    p1 = e("tzyxab,tzyxbc->tzyxac", g[..., mu, :, :],
           cp.roll(g, -1, axis=a_mu)[..., nu, :, :])
    p1 = e("tzyxab,tzyxcb->tzyxac", p1,
           cp.roll(g, -1, axis=a_nu)[..., mu, :, :].conj())
    p1 = e("tzyxab,tzyxcb->tzyxac", p1, g[..., nu, :, :].conj())

    # Plaquette 2: P_{ν,-μ}
    p2 = e("tzyxab,tzyxcb->tzyxac",
           cp.roll(g_lu, -1, axis=a_mu)[..., nu, :, :],
           cp.roll(g_lu, -1, axis=a_nu)[..., mu, :, :].conj())
    p2 = e("tzyxab,tzyxcb->tzyxac", p2, g_lu[..., nu, :, :].conj())
    p2 = e("tzyxab,tzyxbc->tzyxac", p2, g_lu[..., mu, :, :])

    # Plaquette 3: P_{-μ,-ν}
    p3 = e("tzyxba,tzyxcb->tzyxac",
           cp.roll(g_ld, -1, axis=a_nu)[..., mu, :, :].conj(),
           g_ld[..., nu, :, :].conj())
    p3 = e("tzyxab,tzyxbc->tzyxac", p3, g_ld[..., mu, :, :])
    p3 = e("tzyxab,tzyxbc->tzyxac", p3, cp.roll(g_ld, -1, axis=a_mu)[..., nu, :, :])

    # Plaquette 4: P_{-ν,μ}
    p4 = e("tzyxba,tzyxbc->tzyxac", g_rd[..., nu, :, :].conj(), g_rd[..., mu, :, :])
    p4 = e("tzyxab,tzyxbc->tzyxac", p4, cp.roll(g_rd, -1, axis=a_mu)[..., nu, :, :])
    p4 = e("tzyxab,tzyxcb->tzyxac", p4,
           cp.roll(g_rd, -1, axis=a_nu)[..., mu, :, :].conj())

    # F = -i/8 * Σ(P_k - P_k†)
    tr = (0, 1, 2, 3, 5, 4)  # transpose for conjugate
    ans = (p1 - p1.conj().transpose(*tr)
         + p2 - p2.conj().transpose(*tr)
         + p3 - p3.conj().transpose(*tr)
         + p4 - p4.conj().transpose(*tr))
    return cp.array(-1j, dtype=get_compute_dtype()) * ans / cp.array(8.0, dtype=ans.real.dtype)


# ═══════════════════════════════════════════════════════════════════════════════
# Gauge reader (CPU complex128 → GPU complex64)
# ═══════════════════════════════════════════════════════════════════════════════

def read_gauge_lime(filepath: str, Nt: int, Nx: int, Nc: int = 3) -> np.ndarray:
    """Read .lime gauge config — try tail offset, fall back to header scan."""
    expected_elems = Nt * Nx * Nx * Nx * 4 * Nc * Nc * 2
    expected_bytes = expected_elems * 8
    file_size = os.path.getsize(filepath)
    data_offset = file_size - expected_bytes

    # Quick check: try tail read first
    if 0 <= data_offset < file_size:
        with open(filepath, "rb") as f:
            f.seek(data_offset)
            raw = np.fromfile(f, dtype=">f8", count=expected_elems)
        if raw.size == expected_elems:
            tg = (raw.reshape(Nt, Nx, Nx, Nx, 4, Nc, Nc, 2)[..., 0]
                  + 1j * raw.reshape(Nt, Nx, Nx, Nx, 4, Nc, Nc, 2)[..., 1])
            U = tg[0, 0, 0, 0, 0]
            dev = np.max(np.abs(U @ U.conj().T - np.eye(Nc)))
            if np.isfinite(dev) and dev < 1e-3:
                return tg.astype(np.complex128, copy=False)

    # Full scan (proven algorithm from v20260726)
    return _scan_gauge_data(filepath, file_size, data_offset,
                            expected_bytes, expected_elems, Nt, Nx, Nc)


def _scan_gauge_data(filepath, file_size, approx_hdr, expected_bytes, expected_elems,
                     Nt, Nx, Nc) -> np.ndarray:
    """Scan for valid gauge data near the expected offset (v20260726 algorithm)."""
    # Build candidates: dense around approx_hdr, sparse elsewhere
    candidates = []
    for delta in range(-4096, 4097, 8):
        off = approx_hdr + delta
        if 0 <= off <= file_size - expected_bytes:
            candidates.append(off)
    for off in range(0, min(file_size - expected_bytes + 1, 262144), 512):
        if off not in candidates:
            candidates.append(off)
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

        abs_max = np.abs(test_gauge).max()
        if not np.isfinite(abs_max) or abs_max > 100.0:
            continue

        # Unitarity check on random links
        devs = []
        try:
            for _ in range(20):
                t_i = rng.integers(0, Nt); z_i = rng.integers(0, Nx)
                y_i = rng.integers(0, Nx); x_i = rng.integers(0, Nx)
                d_i = rng.integers(0, 4)
                U = test_gauge[t_i, z_i, y_i, x_i, d_i]
                dev = np.max(np.abs(U @ U.conj().T - np.eye(Nc)))
                if np.isfinite(dev):
                    devs.append(dev)
        except Exception:
            continue

        if not devs:
            continue

        max_dev = max(devs)
        mean_dev = sum(devs) / len(devs)

        if mean_dev < best_dev and max_dev < 1e-3:
            best_dev = mean_dev
            best_offset = off_bytes
            best_gauge = test_gauge

        # Early exit on excellent match near expected offset
        if max_dev < 1e-6 and mean_dev < 1e-6 and abs(off_bytes - approx_hdr) < 4096:
            return test_gauge.astype(np.complex128, copy=False)

    if best_gauge is not None:
        return best_gauge.astype(np.complex128, copy=False)

    raise ValueError(f"No valid gauge data in {filepath}. "
                     f"Scanned {len(candidates)} offsets.")


def validate_gauge(gauge: np.ndarray, tag: str = "", logger=None) -> dict:
    """Validate gauge config (CPU)."""
    Nt, Nz, Ny, Nx, Nd, Nc, _ = gauge.shape
    results = {"shape": list(gauge.shape), "tag": tag}
    rng = np.random.default_rng(42)
    n_check = min(200, Nt * Nz * Ny * Nx)

    devs = []
    for _ in range(n_check):
        ti, zi, yi, xi = rng.integers(0, Nt), rng.integers(0, Nz), rng.integers(0, Ny), rng.integers(0, Nx)
        d = rng.integers(0, Nd)
        U = gauge[ti, zi, yi, xi, d]
        devs.append(np.max(np.abs(U @ U.conj().T - np.eye(Nc))))

    results["unitary_dev_max"] = float(np.max(devs))
    results["unitary_dev_mean"] = float(np.mean(devs))
    results["unitary_dev_median"] = float(np.median(devs))

    traces = [np.trace(gauge[rng.integers(0, Nt), rng.integers(0, Nz),
                           rng.integers(0, Ny), rng.integers(0, Nx), rng.integers(0, Nd)])
              for _ in range(n_check)]
    results["trace_mean_re"] = float(np.real(np.mean(traces)))
    results["trace_mean_im"] = float(np.imag(np.mean(traces)))

    # Plaquette trace
    plaq_traces = []
    for _ in range(50):
        ti, zi, yi, xi = rng.integers(0, Nt), rng.integers(0, Nz), rng.integers(0, Ny), rng.integers(0, Nx)
        for mu in range(3):
            for nu in range(mu + 1, 4):
                U1 = gauge[ti, zi, yi, xi, mu]
                idx2 = [ti, zi, yi, xi]; idx2[3 - nu] = (idx2[3 - nu] + 1) % Nx
                U2 = gauge[tuple(idx2 + [nu])]
                idx3 = [ti, zi, yi, xi]; idx3[3 - mu] = (idx3[3 - mu] + 1) % Nx
                U3 = gauge[tuple(idx3 + [mu])].conj().T
                U4 = gauge[ti, zi, yi, xi, nu].conj().T
                plaq_traces.append(np.trace(U1 @ U2 @ U3 @ U4))
    results["plaq_trace_mean_re"] = float(np.real(np.mean(plaq_traces)))
    results["plaq_trace_mean_im"] = float(np.imag(np.mean(plaq_traces)))

    if logger:
        logger.info(f"  Gauge [{tag}]: unitarity={results['unitary_dev_max']:.2e}, "
                    f"trace_re={results['trace_mean_re']:.4f}, plaq_re={results['plaq_trace_mean_re']:.6f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# OPE computation per component — GPU complex64
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ope_per_component_gpu(
    gauge_gpu: cp.ndarray,       # (Nt,Nz,Ny,Nx,4,3,3) complex64 on GPU
    mu: int, nu: int,
    z_dir: int, delta_z: int,
    Nt: int, Nx: int,
    _einsum, logger,
    save_fmunu: bool = False,
    fmunu_output_dir: Optional[Path] = None,
) -> np.ndarray:
    r"""OPE operator on GPU — vectorized Wilson lines via cumulative transporters.

    Algorithm (optimized):
      Let C(x) = Π_{s=0}^{x-1} U(s, z_dir) be the forward transporter.
      Then W^dag(z→0)(x) = C(x)^dag @ C(x-z)  (batch matmul, no Python loops!)
      O(z) = Σ_{x_⊥} Tr[F(x+z)·C(x+z)^dag·C(x)·F(x)·C(x)^dag·C(x+z)]

    Returns: (delta_z, Nt) complex64.
    """
    compute_dt = get_compute_dtype()
    logger.info(f"  OPE (GPU, vectorized, {compute_dt}): mu={mu}, nu={nu}, "
                f"z_dir={z_dir}, dz={delta_z}")
    if mu == nu:
        return np.zeros((delta_z, Nt), dtype=compute_dt)

    t0 = time.perf_counter()
    z_axis = 3 - z_dir  # axis in (t,z,y,x) for the z_dir direction

    # ── Step 1: F_{μν} via Clover (GPU) ────────────────────────────────────
    with Timer(f"Fmunu_GPU_mu{mu}_nu{nu}", logger):
        F = plaquette_clover_gpu(gauge_gpu, mu, nu, _einsum)
    logger.info(f"    F_{{{mu},{nu}}} (GPU): shape={F.shape}, mem={F.nbytes/1024**2:.1f} MB")

    if save_fmunu and fmunu_output_dir:
        fmunu_output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(fmunu_output_dir / f"Fmunu_mu{mu}_nu{nu}.npz", F_munu=cp.asnumpy(F))
        logger.info(f"    [SAVE] Fmunu_mu{mu}_nu{nu}.npz")

    # ── Step 2: Cumulative forward transporter C along z_dir ───────────────
    # C(x) = Π_{s=0}^{x-1} U(x_src + s*e_z, z_dir)
    # Computed iteratively along z_axis: C[n+1] = U[n] @ C[n]
    with Timer(f"wilson_cumulative_mu{mu}_nu{nu}", logger):
        U_z = gauge_gpu[..., z_dir, :, :]  # (Nt,Nz,Ny,Nx,3,3)
        C = cp.zeros_like(U_z)

        # Initialize C with identity at first position along z_axis
        C = _set_first_slice_identity(C, z_axis, compute_dt)

        # Iterative cumulative product: C[n] = U[n-1] @ C[n-1] for n=1..Nx-1
        for nz in range(1, Nx):
            idx_prev = [slice(None)] * 5; idx_prev[z_axis] = nz - 1
            idx_curr = [slice(None)] * 5; idx_curr[z_axis] = nz
            C[tuple(idx_curr)] = _einsum("...ab,...bc->...ac",
                                         U_z[tuple(idx_prev)], C[tuple(idx_prev)])
        logger.debug(f"    Cumulative transporter C done in {time.perf_counter()-t0:.1f}s")

    # ── Step 3: OPE for each z using precomputed C ────────────────────────
    # Spatial perpendicular axes (exclude z_axis and time axis 0)
    all_spatial = [1, 2, 3]
    perp_axes = tuple(a for a in all_spatial if a != z_axis)

    # Pre-compute: C_dag = C^dag at all positions
    C_dag = C.conj().transpose(0, 1, 2, 3, 5, 4)

    # Pre-compute: Tr[F·F] at each point (needed for z=0 and debugging)
    # F_trace = Tr_c[F * F] = einsum("tzyxab,tzyxba->tzyx", F, F)

    ope = np.zeros((delta_z, Nt), dtype=compute_dt)

    for zi in range(delta_z):
        tz0 = time.perf_counter()

        if zi == 0:
            Fsq = _einsum("tzyxab,tzyxba->tzyx", F, F)
            s = cp.sum(Fsq, axis=perp_axes)
            ope[zi, :] = cp.asnumpy(s[:, 0] if s.ndim == 2 else s)
            logger.debug(f"    z=0: {time.perf_counter()-tz0:.1f}s, |O|={np.abs(ope[0]).max():.2e}")
            continue

        # Shift F forward by zi along z_dir: F(x+zi)
        F_shifted = cp.roll(F, -zi, axis=z_axis)

        # Shift C and C_dag forward by zi
        C_shifted = cp.roll(C, -zi, axis=z_axis)
        C_dag_shifted = cp.roll(C_dag, -zi, axis=z_axis)

        # W_dag(x+zi→x) = C(x+zi)^dag @ C(x)
        # Evaluated at each x: F(x+zi) connects to F(x) via Wilson lines
        W_dag = _einsum("...ab,...bc->...ac", C_dag_shifted, C)

        # W_fwd(x→x+zi) = C(x)^dag @ C(x+zi) = W_dag^dag
        W_fwd = _einsum("...ab,...bc->...ac", C_dag, C_shifted)
        # Actually W_dag.conj().transpose(...,5,4) but using C_dag and C_shifted is cleaner

        # O(z) = Tr[F(x+zi) · W_dag · F(x) · W_fwd]
        Fz_Wd = _einsum("...ab,...bc->...ac", F_shifted, W_dag)
        Fz_Wd_F0 = _einsum("...ab,...bc->...ac", Fz_Wd, F)
        Fz_Wd_F0_W = _einsum("...ab,...ba->...", Fz_Wd_F0, W_fwd)

        s = cp.sum(Fz_Wd_F0_W, axis=perp_axes)
        ope[zi, :] = cp.asnumpy(s[:, 0] if s.ndim == 2 else s)

        if zi % 4 == 0 or zi == delta_z - 1:
            gpu_mem = get_gpu_memory_mb()
            logger.debug(f"    z={zi:3d}/{delta_z}  |O|_max={np.abs(ope[zi]).max():.4e}  "
                         f"{time.perf_counter()-tz0:.1f}s  GPU free={gpu_mem['free_mb']:.0f}MB")

    elapsed = time.perf_counter() - t0
    logger.info(f"    OPE (GPU vec) mu={mu},nu={nu} done in {elapsed:.1f}s  "
                f"|O| range: [{np.abs(ope).min():.2e},{np.abs(ope).max():.2e}]")
    assert np.all(np.isfinite(ope)), f"OPE mu={mu},nu={nu} has NaN/inf"
    return ope


def _set_first_slice_identity(C: cp.ndarray, axis: int, dtype) -> cp.ndarray:
    """Set the first slice along `axis` of C[...,3,3] to identity via broadcasting."""
    # C has shape like (Nt, Nz, Ny, Nx, 3, 3)
    # We need to broadcast eye(3,3) across all non-axis batch dims (excl last 2 matrix dims)
    eye = cp.eye(3, dtype=dtype)
    # Build broadcast shape: batch dims (no axis, no matrix) + [3, 3]
    batch_sizes = [s for i, s in enumerate(C.shape[:-2]) if i != axis]
    eye_big = cp.broadcast_to(eye, batch_sizes + [3, 3])
    idx = [slice(None)] * C.ndim
    idx[axis] = 0
    C[tuple(idx)] = eye_big
    return C


def save_ope_data(ope, output_dir, mu, nu, conf_id, delta_z, logger) -> Path:
    """Save OPE .npz in standard naming."""
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
    """Compute OPE for all configs FROM SCRATCH on GPU (complex64)."""
    params = config["parameters"]
    paths = config["data_paths"]
    ensemble = config["ensemble"]

    Nt, Nx, Nc = ensemble["Nt"], ensemble["Nx"], ensemble["Nc"]
    conf_ids = params["conf_ids"]
    delta_z, z_dir = params["delta_z"], params["z_dir"]
    gauge_base = paths["gauge_config_base"]
    gauge_pattern = paths["gauge_config_pattern"]
    save_fmunu = params.get("save_intermediate_fmunu", True)

    components = [(0, 1), (3, 0), (3, 1)]  # F_xy, F_tx, F_ty

    print_banner("Step 02: Compute OPE from Gauge Configs (GPU, complex64)", logger)
    log_gpu_status(logger, "  ")
    logger.info(f"  Compute dtype: {get_compute_dtype()}")
    logger.info(f"  Ensemble: {ensemble['full_name']} | {Nt}×{Nx}³ | Nc={Nc}")
    logger.info(f"  Components: {components} | dz={delta_z} | z_dir={z_dir}")
    logger.info(f"  Configs: {conf_ids}")

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

        # ── Read gauge CPU → GPU (downcast complex128 → complex64) ──────────
        try:
            with Timer(f"read_gauge_conf{conf_id}", logger, output_dir.parent,
                      extra={"conf_id": conf_id}):
                try:
                    gauge_cpu = read_gauge_lime(gauge_file, Nt, Nx, Nc)
                except Exception:
                    logger.warning("  Direct read failed — should not happen with ILDG format")
                    raise

                logger.info(f"  Gauge CPU: shape={gauge_cpu.shape}, dtype={gauge_cpu.dtype}, "
                           f"mem={gauge_cpu.nbytes/1024**2:.1f} MB")

                # Downcast to compute dtype and transfer to GPU
                compute_dt = get_compute_dtype()
                gauge_gpu = cp.asarray(gauge_cpu.astype(compute_dt, copy=False))
                del gauge_cpu
                gpu_sync()
                gpu_mem = get_gpu_memory_mb()
                logger.info(f"  Gauge GPU: dtype={gauge_gpu.dtype}, "
                           f"mem={gauge_gpu.nbytes/1024**2:.1f} MB, "
                           f"GPU free={gpu_mem['free_mb']:.0f} MB")
        except Exception as e:
            logger.error(f"  [ERROR] Gauge read failed: {e}")
            log_exception(logger, e, f"Reading gauge {conf_id}")
            all_results[conf_id] = {"status": "error", "reason": str(e)}
            continue

        # ── Validate gauge (CPU) ─────────────────────────────────────────
        # Need CPU copy for validation; re-read briefly
        gauge_cpu_val = cp.asnumpy(gauge_gpu)  # small, just for validation
        with Timer(f"validate_gauge_conf{conf_id}", logger, output_dir.parent,
                  extra={"conf_id": conf_id}):
            val = validate_gauge(gauge_cpu_val, f"conf_{conf_id}", logger)
        save_intermediate(val, conf_out, f"gauge_validation_conf{conf_id}.json", logger)
        del gauge_cpu_val

        # ── Compute OPE components (GPU) ─────────────────────────────────
        conf_results = {"status": "ok", "components": {}, "validation": val}
        all_ok = True

        for mu, nu in components:
            key = f"mu{mu}_nu{nu}"
            try:
                with Timer(f"ope_GPU_{key}_conf{conf_id}", logger, output_dir.parent,
                          extra={"conf_id": conf_id, "mu": mu, "nu": nu, "device": "gpu",
                                 "dtype": str(get_compute_dtype())}):
                    ope = compute_ope_per_component_gpu(
                        gauge_gpu, mu, nu, z_dir, delta_z,
                        Nt, Nx, cp.einsum, logger,
                        save_fmunu=save_fmunu, fmunu_output_dir=conf_out)

                path = save_ope_data(ope, conf_out, mu, nu, conf_id, delta_z, logger)
                conf_results["components"][key] = {
                    "status": "ok", "output": str(path),
                    "shape": list(ope.shape),
                    "re_range": [float(ope.real.min()), float(ope.real.max())],
                    "im_range": [float(ope.imag.min()), float(ope.imag.max())],
                    "re_mean": float(ope.real.mean()),
                    "im_mean": float(ope.imag.mean()),
                    "nonzero_fraction": float(np.count_nonzero(ope) / ope.size),
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

        del gauge_gpu
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

        gpu_mem = get_gpu_memory_mb()
        logger.info(f"  [PROGRESS] OPE: {iconf+1}/{len(conf_ids)} done, GPU free={gpu_mem['free_mb']:.0f}MB")

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info(f"\n{'═'*60}")
    logger.info(f"OPE (GPU, {get_compute_dtype()}) Summary:")
    required = {f"mu{mu}_nu{nu}" for mu, nu in components}
    for conf_id, result in all_results.items():
        found = {k for k, v in result.get("components", {}).items() if v.get("status") == "ok"}
        missing = required - found
        s = "✓" if result["status"] == "ok" else "⚠"
        parts = []
        for k in sorted(found):
            c = result["components"][k]
            parts.append(f"{k}:|O|∈[{c['re_range'][0]:.2e},{c['re_range'][1]:.2e}] {c.get('dtype','?')}")
        logger.info(f"  {s} conf={conf_id}: {'; '.join(parts)}")
        if missing:
            logger.warning(f"    Missing: {sorted(missing)}")
    logger.info(f"{'═'*60}")
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Compute OPE FROM SCRATCH (GPU, complex64)")
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

    summary_path = output_dir / "compute_ope_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Summary saved to {summary_path}")

    all_ok = all(r["status"] == "ok" for r in results.values())
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
