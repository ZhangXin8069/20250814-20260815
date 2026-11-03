#!/usr/bin/env python3
"""
Generate synthetic sample data matching donghx's output format.

Creates data for Nconf=3 configurations on L24x72 ensemble:
- 2pt correlator: (Nt, Nt) complex, exponential decay + noise
- OPE data: (Nx, Nt) complex per (mu,nu) component

The synthetic data uses physically-motivated functional forms to ensure
the downstream huangcl analysis produces reasonable-looking ratio plots.

Usage:
    python generate.py --run-dir /path/to/run_dir
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


# ─── Physical model parameters ───────────────────────────────────────────────

def generate_sample_2pt(
    Nt: int,
    conf_id: int,
    Px: int = 0,
    Py: int = 0,
    Pz: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic proton 2pt correlator.

    Returns:
        ndarray of shape (Nt, Nt) with complex dtype (complex128).

    Model: C2(t_src, t_snk) ~ A^2 * [exp(-E0*dt) + exp(-E0*(Nt-dt))] / (2*E0)
    with dt = (t_snk - t_src) mod Nt, plus smooth correlated noise.
    """
    rng = np.random.default_rng(seed + conf_id)

    E0 = 0.45  # Proton ground-state energy (lattice units, ~1 GeV * 0.105 fm)
    A0_sq = 1.0  # Overlap factor squared
    noise_level = 0.03  # Relative noise level
    imag_fraction = 0.02  # Imaginary part fraction of real part

    # Configuration-specific scale factor (5% variation between configs)
    cfg_scale = 1.0 + 0.05 * np.sin(conf_id * 0.01)

    corr = np.zeros((Nt, Nt), dtype=complex)

    for t_src in range(Nt):
        for dt_abs in range(Nt):
            t_snk = (t_src + dt_abs) % Nt
            dt = dt_abs

            # Forward + backward propagating exponential
            forward = np.exp(-E0 * dt)
            backward = np.exp(-E0 * (Nt - dt))
            val = A0_sq * (forward + backward) / (2.0 * E0)

            # Add smooth correlated noise in t_src and dt
            noise_t = 1.0 + noise_level * rng.normal()
            noise_scale = cfg_scale * noise_t

            # Make imaginary part small fraction of real
            imag_val = val * imag_fraction * rng.normal()

            corr[t_src, dt_abs] = val * noise_scale + 1j * imag_val

    # Smooth the noise by convolving along both axes
    kernel = np.array([0.25, 0.5, 0.25])
    for axis in [0, 1]:
        real_part = corr.real.copy()
        imag_part = corr.imag.copy()
        for i in range(corr.shape[1 - axis]):
            if axis == 0:
                col = real_part[:, i]
                smoothed = np.convolve(col, kernel, mode="same")
                real_part[:, i] = smoothed
                col_i = imag_part[:, i]
                smoothed_i = np.convolve(col_i, kernel, mode="same")
                imag_part[:, i] = smoothed_i
            else:
                row = real_part[i, :]
                smoothed = np.convolve(row, kernel, mode="same")
                real_part[i, :] = smoothed
                row_i = imag_part[i, :]
                smoothed_i = np.convolve(row_i, kernel, mode="same")
                imag_part[i, :] = smoothed_i
        corr = real_part + 1j * imag_part

    return corr.astype(complex)


def generate_sample_ope(
    Nt: int,
    Nx: int,
    mu: int,
    nu: int,
    conf_id: int,
    delta_z: int = 24,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic OPE (operator product expansion) dataset.

    Returns:
        ndarray of shape (Nx, Nt) with complex dtype (complex128).

    The OPE operator is F_{mu,nu}(z) * U_link(z,0) * F_{mu,nu}(0).
    For (mu,nu) = (0,1): spatial-spatial (F_xy), real and dominant
    For (mu,nu) = (3,0): time-spatial (F_tx), smaller magnitude
    For (mu,nu) = (3,1): time-spatial (F_ty), similar to F_tx

    The combined OPE = -F_tx - F_ty + 2*F_xy is what huangcl uses.
    """
    rng = np.random.default_rng(seed + conf_id + mu * 10 + nu)

    # Different amplitude and structure for each tensor component
    if mu == 0 and nu == 1:
        # F_xy: spatial plaquette, ~1 in magnitude
        base_amplitude = 0.8
        z_variation_scale = 0.05
    elif (mu == 3 and nu == 0) or (mu == 3 and nu == 1):
        # F_tx, F_ty: time-spatial, smaller
        base_amplitude = 0.15
        z_variation_scale = 0.02
    else:
        base_amplitude = 0.1
        z_variation_scale = 0.01

    noise_level = 0.08
    imag_fraction = 0.03
    cfg_scale = 1.0 + 0.08 * np.sin(conf_id * 0.01 + mu + nu)

    ope = np.zeros((Nx, Nt), dtype=complex)

    for z_idx in range(Nx):
        for t in range(Nt):
            # Base value with smooth z-dependence
            z_factor = 1.0 + z_variation_scale * np.sin(2 * np.pi * z_idx / Nx + mu)
            # Time dependence: slight exponential decay from source
            t_factor = 1.0 - 0.1 * np.exp(-0.3 * min(t, Nt - t))
            # Spatial modulation
            x_mod = 0.02 * np.sin(2 * np.pi * z_idx / 8.0)

            val = base_amplitude * z_factor * t_factor * (1.0 + x_mod)
            val *= cfg_scale * (1.0 + noise_level * rng.normal())
            imag_val = val * imag_fraction * rng.normal()

            ope[z_idx, t] = val + 1j * imag_val

    return ope.astype(complex)


def save_2pt_data(
    corr: np.ndarray,
    output_dir: Path,
    conf_id: int,
    Px: int = 0,
    Py: int = 0,
    Pz: int = 2,
    mom_smear: int = -2,
    element: str = "_Cg5g4",
) -> Path:
    """Save 2pt correlator in donghx's naming convention."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = (
        f"twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}"
        f"_eginphase{abs(mom_smear)}"
        f"{element}_nopol_ss_conf{conf_id}.npy"
    )
    path = output_dir / fname
    np.save(path, corr)
    return path


def save_ope_data(
    ope: np.ndarray,
    output_dir: Path,
    mu: int,
    nu: int,
    conf_id: int,
    delta_z: int = 24,
) -> Path:
    """Save OPE data in donghx's naming convention (as .npz with 'ops' key)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"ops_mu{mu}_nu{nu}_dz{delta_z}_conf{conf_id}.npz"
    path = output_dir / fname
    np.savez(path, ops=ope)
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sample data")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory")
    parser.add_argument("--config", type=str, default=None, help="Path to run_config.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data_dir = run_dir / "02_sample_data"

    # Load config
    config_path = args.config or (run_dir / "run_config.json")
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        params = config["parameters"]
    else:
        params = {
            "Nt": 72, "Nx": 24, "Nconf": 3,
            "conf_ids": [6250, 6450, 6650],
            "Px": 0, "Py": 0, "Pz": 2,
            "mom_smear": -2, "delta_z": 24,
            "element": "_Cg5g4",
        }

    Nt = params["Nt"]
    Nx = params["Nx"]
    conf_ids = params["conf_ids"]
    Px = params.get("Px", 0)
    Py = params.get("Py", 0)
    Pz = params.get("Pz", 2)
    mom_smear = params.get("mom_smear", -2)
    delta_z = params.get("delta_z", 24)
    element = params.get("element", "_Cg5g4")
    seed = params.get("seed", 42)

    print(f"[generate_data] Nt={Nt}, Nx={Nx}, Nconf={len(conf_ids)}")
    print(f"[generate_data] conf_ids={conf_ids}")
    print(f"[generate_data] P=({Px},{Py},{Pz}), delta_z={delta_z}")

    # Generate data for each configuration
    for conf_id in conf_ids:
        print(f"\n[generate_data] Processing conf_id={conf_id}...")

        # 2pt correlator
        conf_dir = data_dir / f"conf_{conf_id}"
        corr = generate_sample_2pt(Nt, conf_id, Px, Py, Pz, seed=seed)
        path = save_2pt_data(corr, conf_dir, conf_id, Px, Py, Pz, mom_smear, element)
        print(f"  [2pt] shape={corr.shape}, dtype={corr.dtype}, saved to {path.name}")

        # Validate 2pt
        assert corr.shape == (Nt, Nt), f"2pt shape mismatch: {corr.shape}"
        assert np.all(np.isfinite(corr)), "2pt contains NaN/inf"
        assert np.abs(corr).max() > 0, "2pt is all zeros"

        # OPE components
        ope_components = [(0, 1), (3, 0), (3, 1)]
        for mu, nu in ope_components:
            ope = generate_sample_ope(Nt, Nx, mu, nu, conf_id, delta_z, seed=seed)
            path = save_ope_data(ope, conf_dir, mu, nu, conf_id, delta_z)
            print(f"  [OPE mu={mu},nu={nu}] shape={ope.shape}, dtype={ope.dtype}, saved to {path.name}")

            # Validate OPE
            assert ope.shape == (Nx, Nt), f"OPE shape mismatch: {ope.shape}"
            assert np.all(np.isfinite(ope)), f"OPE mu={mu},nu={nu} contains NaN/inf"

    print(f"\n[generate_data] All data generated successfully in {data_dir}")

    # Write a manifest of generated files
    manifest = {"2pt_files": [], "ope_files": []}
    for conf_id in conf_ids:
        conf_dir = data_dir / f"conf_{conf_id}"
        for f in sorted(conf_dir.iterdir()):
            if f.suffix == '.npy':
                manifest["2pt_files"].append(str(f.relative_to(data_dir)))
            elif f.suffix == '.npz':
                manifest["ope_files"].append(str(f.relative_to(data_dir)))
    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"[generate_data] Manifest written to {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
