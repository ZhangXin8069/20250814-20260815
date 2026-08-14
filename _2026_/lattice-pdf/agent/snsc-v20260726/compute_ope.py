#!/usr/bin/env python3
"""
OPE (Operator Product Expansion) computation from gauge configurations.

Reads gauge configurations (.lime ILDG format, big-endian float64),
computes field strength tensor F_{μν} via clover plaquette, then
constructs the nonlocal gluon OPE operator:

    O_{mu,nu}(z, t) = Σ_{x_perp} Tr[F_{mu,nu}(z, x_perp, t)
                       * W(z→0, x_perp, t)
                       * F_{mu,nu}(0, x_perp, t)
                       * W(0→z, x_perp, t)]

for each (mu,nu) component and saves results per configuration.

Usage (standalone):
    python compute_ope.py --run-dir /path/to/output

Usage (imported):
    from compute_ope import compute_ope_all_configs
    results = compute_ope_all_configs(config, output_dir, logger)
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

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent  # /root/lattice-pdf
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from utils import Timer, print_banner

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

    Args:
        filepath: Path to .lime file.
        Nt: Temporal extent.
        Nx: Spatial extent.
        Nc: Number of colors (default 3).

    Returns:
        gauge: shape (Nt, Nx, Nx, Nx, 4, Nc, Nc), dtype complex128.
    """
    with open(filepath, "rb") as f:
        raw = np.fromfile(f, dtype=">f8")  # big-endian float64

    expected = Nt * Nx * Nx * Nx * 4 * Nc * Nc * 2
    if raw.size != expected:
        raise ValueError(
            f"Gauge file size mismatch: expected {expected} doubles, got {raw.size}\n"
            f"  File: {filepath}\n"
            f"  Expected shape: ({Nt},{Nx},{Nx},{Nx},4,{Nc},{Nc},2)"
        )

    raw = raw.reshape(Nt, Nx, Nx, Nx, 4, Nc, Nc, 2)
    gauge = raw[..., 0] + 1j * raw[..., 1]
    return gauge.astype(np.complex128)


def validate_gauge(gauge: np.ndarray, tag: str = "") -> dict:
    """Quick validation checks on a gauge configuration.

    Returns dict with keys: unitary_deviation, trace_mean, shape.
    """
    Nt, Nz, Ny, Nx, Nd, Nc, _ = gauge.shape
    results = {"shape": list(gauge.shape), "tag": tag}

    # Check unitarity: U @ U^dag ≈ I for a few random sites
    rng = np.random.default_rng(42)
    devs = []
    for _ in range(100):
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

    # Trace mean (should be ~0 for SU(3))
    traces = []
    for _ in range(100):
        t = rng.integers(0, Nt)
        z = rng.integers(0, Nz)
        y = rng.integers(0, Ny)
        x = rng.integers(0, Nx)
        d = rng.integers(0, Nd)
        traces.append(np.trace(gauge[t, z, y, x, d]))
    results["trace_mean_re"] = float(np.real(np.mean(traces)))
    results["trace_mean_im"] = float(np.imag(np.mean(traces)))

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
) -> np.ndarray:
    r"""Compute OPE operator for a single (mu, nu) component.

    Algorithm:
      1. Compute F_{mu,nu}(x) via clover plaquette → shape (Nt,Nz,Ny,Nx,3,3)
      2. For each z ∈ [0, delta_z-1]:
         a. Shift F to position z along z_dir
         b. Build Wilson line W(z→0) as product of U_z^dag
         c. Contract: O(z) = Σ_{x_perp} Tr[F(z) * W^dag * F(0) * W]
      3. Return ope of shape (Nx, Nt)

    Notes:
      - Perpendicular directions (not z_dir) are summed over.
      - The field strength at 0 is at the unshifted position.
      - Wilson line is a product of gauge links in z_dir.

    Args:
        gauge: Gauge links, shape (Nt, Nz, Ny, Nx, 4, 3, 3), donghx convention.
        mu, nu: Lorentz indices of F_{mu,nu}.
        z_dir: Wilson line direction (0=x, 1=y, 2=z).
        delta_z: Maximum Wilson line length in lattice units.
        Nt, Nx: Lattice extents.
        contract_fn: Einsum-like contraction function.
        logger: Logger instance.

    Returns:
        ope: shape (Nx, Nt), complex128. OPE operator vs z and t.
             (summed over the 2 perpendicular spatial directions.)
    """
    logger.info(f"  Computing OPE: mu={mu}, nu={nu}, z_dir={z_dir}, delta_z={delta_z}")

    if mu == nu:
        logger.warning(f"  mu=nu={mu}, F is identically zero. Returning zeros.")
        return np.zeros((Nx, Nt), dtype=complex)

    t_start = time.perf_counter()

    # Step 1: Field strength tensor F_{mu,nu}
    F_munu = plaquette_clover(gauge, mu, nu, contract_fn)
    # Shape: (Nt, Nz, Ny, Nx, 3, 3)

    # The 4 spatial axes in donghx convention are: (t=0, z=1, y=2, x=3)
    # z_dir maps: 0→x (axis 3), 1→y (axis 2), 2→z (axis 1)
    z_axis = 3 - z_dir  # Axis in the (t,z,y,x) ordering

    # Identify perpendicular axes (the two spatial axes NOT z_dir)
    spatial_axes = [1, 2, 3]  # (z, y, x) in donghx ordering
    perp_axes = [a for a in spatial_axes if a != z_axis]

    ope = np.zeros((Nx, Nt), dtype=complex)

    for zi in range(delta_z):
        if zi == 0:
            # z=0: F(0) * F(0)^dag (identity-like) — trace of F^2 at origin
            # Sum over perpendicular directions, trace over color
            F0 = F_munu  # (Nt, Nz, Ny, Nx, 3, 3)
            # F(0) * F(0) — actually this is Tr[F * F]
            F0_trace = np.einsum("tzyxaa->tzyx", F0)  # Color trace
            # Sum over perpendicular directions → (Nt, Nx_zdir)
            ope[zi, :] = np.sum(F0_trace, axis=perp_axes)
            continue

        # Step 2a: Shift F to position z = zi along z_dir
        F_at_z = np.roll(F_munu, -zi, axis=z_axis)  # F at z = zi

        # Step 2b: Wilson line W(z→0) = product of U_z^dag from z to 0
        # W(z→0) = Π_{s=0}^{zi-1} U_z^dag(z - s)
        W_dag = np.zeros_like(F_at_z)  # (Nt,Nz,Ny,Nx,3,3)
        for i in range(Nt):
            for z in range(Nx):
                for y in range(Nx):
                    for x in range(Nx):
                        W = np.eye(3, dtype=complex)
                        for s in range(zi):
                            # Position at z - s along z_dir
                            idx = [i, z, y, x]
                            idx[z_axis] = (idx[z_axis] - s) % Nx
                            U_link = gauge[tuple(idx) + (z_dir,)]
                            W = U_link @ W
                        W_dag[i, z, y, x] = W.conj().T

        # Step 2c: F(0) at origin (no shift)
        F_at_0 = F_munu

        # Step 2d: O(z) = Tr[F(z) * W^dag * F(0) * W]
        # F_at_z(ab) * W_dag(bc) * F_at_0(cd) * W(da)^dag
        # = F_at_z(ab) * W_dag(bc) * F_at_0(cd) * W_dag(ad)^*
        # We need W (not W_dag) for the forward path
        W_fwd = W_dag.conj().transpose(0, 1, 2, 3, 5, 4)  # inverse of W_dag

        # Contract: F(z,ab) @ W_dag(bc) @ F(0,cd) @ W_fwd(da)
        # This is: sum_{abcd} F_z[ab] * W_dag[bc] * F_0[cd] * W_fwd[da]
        # = Tr(F_z @ W_dag @ F_0 @ W_fwd)
        Fz_Wdag = contract_fn("tzyxab,tzyxbc->tzyxac", F_at_z, W_dag)
        Fz_Wdag_F0 = contract_fn("tzyxab,tzyxbc->tzyxac", Fz_Wdag, F_at_0)
        Fz_Wdag_F0_W = contract_fn("tzyxab,tzyxba->tzyx", Fz_Wdag_F0, W_fwd)

        # Step 2e: Sum over perpendicular directions
        ope[zi, :] = np.sum(Fz_Wdag_F0_W, axis=tuple(perp_axes))

        if zi % 8 == 0:
            logger.debug(f"    z={zi:3d}/{delta_z}  |O|_max={np.abs(ope[zi]).max():.4e}")

    # Fill remaining z (delta_z .. Nx-1) with zeros (or mirror if needed)
    # Actually delta_z = Nx typically (delta_z=24 = Nx=24)
    if delta_z < Nx:
        logger.debug(f"  delta_z={delta_z} < Nx={Nx}, remaining z set to zero")

    elapsed = time.perf_counter() - t_start
    logger.info(
        f"    OPE complete in {elapsed:.1f}s  "
        f"shape={ope.shape}  "
        f"|O| range: [{np.abs(ope).min():.2e}, {np.abs(ope).max():.2e}]"
    )

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
    """Save OPE data in donghx's naming convention.

    Format: ops_mu{mu}_nu{nu}_dz{delta_z}_conf{conf_id}.npz
    With key 'ops' containing the (Nx, Nt) complex array.
    """
    fname = f"ops_mu{mu}_nu{nu}_dz{delta_z}_conf{conf_id}.npz"
    path = output_dir / fname
    np.savez(path, ops=ope)
    logger.info(f"    Saved {fname}: shape={ope.shape}, "
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
    """Compute OPE for all configurations from gauge configs.

    For each configuration:
      1. Read gauge config (.lime format)
      2. Compute F_{mu,nu} via clover plaquette for each component
      3. Compute OPE operator for each (mu,nu)
      4. Save as .npz

    Args:
        config: Full configuration dict (from run_config.json).
        output_dir: Output directory (data saved under conf_{id}/).
        logger: Logger instance.

    Returns:
        dict: conf_id → {status, components: {mu_nu: {path, shape, ...}}}
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

    # OPE tensor components needed for unpolarized gluon:
    # (0,1) = F_xy (spatial-spatial, dominant)
    # (3,0) = F_tx (time-spatial)
    # (3,1) = F_ty (time-spatial)
    components = [(0, 1), (3, 0), (3, 1)]

    # Try to import opt_einsum
    try:
        from opt_einsum import contract as _opt_contract
        contract_fn = _opt_contract
        logger.info("Using opt_einsum for contractions")
    except ImportError:
        contract_fn = np.einsum
        logger.info("Using numpy.einsum for contractions")

    print_banner("Step 02: Compute OPE from Gauge Configs", logger)
    logger.info(f"  Lattice: {Nt}×{Nx}³, Nc={Nc}")
    logger.info(f"  Components: mu,nu ∈ {components}")
    logger.info(f"  delta_z = {delta_z}, z_dir = {z_dir}")
    logger.info(f"  Configs: {conf_ids}")
    logger.info(f"  Gauge base: {gauge_base}")
    logger.info(f"  Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for conf_id in conf_ids:
        conf_out_dir = output_dir / f"conf_{conf_id}"
        conf_out_dir.mkdir(parents=True, exist_ok=True)

        # ── Read gauge configuration ───────────────────────────────────────
        gauge_file = os.path.join(gauge_base, gauge_pattern.format(conf_id=conf_id))
        logger.info(f"\n{'─'*60}")
        logger.info(f"  conf_id={conf_id}")
        logger.info(f"  Gauge file: {gauge_file}")

        if not os.path.exists(gauge_file):
            logger.error(f"  [ERROR] Gauge file not found: {gauge_file}")
            all_results[conf_id] = {
                "status": "missing",
                "reason": f"gauge file not found: {gauge_file}",
            }
            continue

        # Read and validate gauge
        with Timer(f"read_gauge_conf{conf_id}", logger, output_dir):
            try:
                gauge = read_gauge_lime(gauge_file, Nt, Nx, Nc)
                logger.info(f"  Gauge shape: {gauge.shape}, dtype: {gauge.dtype}")
                logger.info(f"  Gauge size: {gauge.nbytes / 1024**2:.1f} MB")
            except Exception as e:
                logger.error(f"  [ERROR] Failed to read gauge: {e}")
                all_results[conf_id] = {"status": "error", "reason": str(e)}
                continue

        # Validate
        val = validate_gauge(gauge, f"conf_{conf_id}")
        logger.info(f"  Unitarity check: max_dev={val['unitary_dev_max']:.2e}, "
                    f"mean_dev={val['unitary_dev_mean']:.2e}")
        logger.info(f"  Trace check: re={val['trace_mean_re']:.4f}, "
                    f"im={val['trace_mean_im']:.4f}")

        # ── Compute OPE for each component ─────────────────────────────────
        conf_results = {"status": "ok", "components": {}}
        all_ok = True

        for mu, nu in components:
            key = f"mu{mu}_nu{nu}"

            with Timer(f"ope_{key}_conf{conf_id}", logger, output_dir):
                try:
                    ope = compute_ope_per_component(
                        gauge, mu, nu, z_dir, delta_z,
                        Nt, Nx, contract_fn, logger,
                    )

                    # Validate
                    assert np.all(np.isfinite(ope)), f"OPE {key} contains NaN/inf"
                    assert ope.shape == (Nx, Nt), f"OPE shape mismatch: expected ({Nx},{Nt}), got {ope.shape}"

                    # Save
                    path = save_ope_data(ope, conf_out_dir, mu, nu, conf_id, delta_z, logger)

                    conf_results["components"][key] = {
                        "status": "ok",
                        "output": str(path),
                        "shape": list(ope.shape),
                        "re_range": [float(ope.real.min()), float(ope.real.max())],
                        "im_range": [float(ope.imag.min()), float(ope.imag.max())],
                    }

                except Exception as e:
                    logger.error(f"  [ERROR] {key}: {e}", exc_info=True)
                    conf_results["components"][key] = {
                        "status": "error",
                        "message": str(e),
                    }
                    all_ok = False

        if not all_ok:
            conf_results["status"] = "partial"

        all_results[conf_id] = conf_results

        # Free gauge memory
        del gauge
        import gc
        gc.collect()

    # ── Summary ─────────────────────────────────────────────────────────────
    logger.info(f"\n{'═'*60}")
    logger.info("OPE Computation Summary:")
    required = {f"mu{mu}_nu{nu}" for mu, nu in components}
    for conf_id, result in all_results.items():
        found_keys = {
            k for k, v in result["components"].items()
            if v.get("status") == "ok"
        }
        missing = required - found_keys
        status = "✓" if result["status"] == "ok" else "⚠"
        # Print shape info
        info_parts = []
        for k in found_keys:
            comp = result["components"][k]
            re_r = comp.get("re_range", [0, 0])
            info_parts.append(f"{k}: |O|∈[{re_r[0]:.2e},{re_r[1]:.2e}]")
        logger.info(f"  {status} conf={conf_id}: {'; '.join(info_parts)}")
        if missing:
            logger.warning(f"    Missing: {missing}")
    logger.info(f"{'═'*60}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Compute OPE from gauge configurations")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory")
    parser.add_argument("--config", type=str, default=None, help="Path to run_config.json")
    parser.add_argument("--conf-id", type=int, default=None,
                       help="Compute for a single config (overrides config list)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = args.config or (run_dir / "run_config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Override conf list if single config requested
    if args.conf_id is not None:
        config["parameters"]["conf_ids"] = [args.conf_id]
        config["parameters"]["Nconf"] = 1

    output_dir = run_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    from utils import setup_logging
    logger = setup_logging(run_dir / "run.log", "compute_ope")

    results = compute_ope_all_configs(config, output_dir, logger)

    # Write summary
    summary_path = output_dir / "compute_ope_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Summary saved to {summary_path}")

    all_ok = all(r["status"] == "ok" for r in results.values())
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
