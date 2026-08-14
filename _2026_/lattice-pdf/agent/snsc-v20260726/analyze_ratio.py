#!/usr/bin/env python3
"""
huangcl-style ratio analysis for the disconnected gluon PDF.

Computes the disconnected 3pt/2pt ratio:
    R(z, t_sep, tau) = C3_disc(z, t_sep, tau) / C2(t_sep)

using jackknife resampling, then generates publication-quality plots.

Adapted from:
  - /root/lattice-pdf/examples/huangcl/code.py
  - /root/lattice-pdf/agent/snsc/runs/.../04_huangcl_analysis/analyze_ratio.py

Usage (standalone):
    python analyze_ratio.py --run-dir /path/to/output

Usage (imported):
    from analyze_ratio import run_analysis
    results = run_analysis(config, data_dir, output_dir, logger)
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))
from utils import Timer, print_banner


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical utilities
# ═══════════════════════════════════════════════════════════════════════════════

def sem(data: np.ndarray, jack: bool) -> np.ndarray:
    """Standard error of the mean.

    For jackknife, multiply by sqrt(N-1).
    """
    error = data.std(axis=0)
    if jack:
        error *= np.sqrt(data.shape[0] - 1)
    return error


def resample(
    corr: np.ndarray,
    jack: bool,
    Nsample: int,
    seed: int = 0,
) -> np.ndarray:
    """Jackknife or bootstrap resampling along axis=0 (config index).

    Args:
        corr: Input data, shape (n_conf, ...).
        jack: If True, jackknife; otherwise bootstrap.
        Nsample: Number of resamples.
        seed: Random seed for bootstrap.

    Returns:
        Resampled data, shape (Nsample, ...).
    """
    n_conf = corr.shape[0]
    if jack:
        # Jackknife: leave-one-out
        total = n_conf * corr.mean(axis=0)
        re_corr = (total - corr) / (n_conf - 1)
    else:
        rng = np.random.default_rng(seed=seed)
        idx = rng.integers(0, n_conf, size=(Nsample, n_conf))
        re_corr = corr[idx].mean(axis=1)
    return re_corr


# ═══════════════════════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis(
    config: dict,
    data_dir: Path,
    output_dir: Path,
    logger,
) -> dict:
    """Run the full huangcl-style ratio analysis.

    Workflow:
      1. Load 2pt correlators: shape (Nconf, Nt, Nt)
      2. Load OPE components: shape (Nconf, Nx, Nt) for each (mu,nu)
      3. Combine OPE = -F_tx - F_ty + 2*F_xy
      4. Build relative-time arrays
      5. Construct 3pt disconnected correlator
      6. Jackknife/Bootstrap resampling
      7. Compute ratio R(z) = C3_disc / C2
      8. Plot and save

    Args:
        config: Full configuration dict.
        data_dir: Directory containing conf_{id}/ subdirectories.
        output_dir: Directory for output plots and data.
        logger: Logger instance.

    Returns:
        dict of results including paths to output files.
    """
    params = config["parameters"]
    ensemble = config["ensemble"]

    Nt = ensemble["Nt"]
    Nx = ensemble["Nx"]
    Nconf = params["Nconf"]
    conf_ids = params["conf_ids"]
    conf_start = params["conf_start"]
    conf_step = params["conf_step"]
    Px, Py = params["Px"], params["Py"]
    Pz = params["Pz"]  # Use first Pz
    mom_smear = params["mom_smear"]
    element = params["element"]
    delta_z = params["delta_z"]
    jack = params["jackknife"]
    max_t = params["max_t"]
    target_z = params["target_z"]
    dt_list = params["dt_list"]
    seed = params["seed"]

    Nsample = Nconf if jack else 3000  # Bootstrap samples if not jackknife

    print_banner("Step 03: huangcl Ratio Analysis", logger)
    logger.info(f"  Ensemble: {ensemble['full_name']} ({ensemble['name']})")
    logger.info(f"  Nconf={Nconf}, Nt={Nt}, Nx={Nx}")
    logger.info(f"  Momentum: P=({Px},{Py},{Pz})")
    logger.info(f"  Jackknife: {jack}, Nsample={Nsample}")
    logger.info(f"  max_t={max_t}, target_z={target_z}, dt_list={dt_list}")
    logger.info(f"  Data dir: {data_dir}")
    logger.info(f"  Output dir: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()

    # ── Step 1: Load data ──────────────────────────────────────────────────
    logger.info("Loading 2pt and OPE data...")

    _corr = np.zeros((Nconf, Nt, Nt), dtype=complex)
    _ope_01 = np.zeros((Nconf, Nx, Nt), dtype=complex)
    _ope_30 = np.zeros((Nconf, Nx, Nt), dtype=complex)
    _ope_31 = np.zeros((Nconf, Nx, Nt), dtype=complex)

    load_errors = []

    for i in range(Nconf):
        conf_id = conf_ids[i]
        conf_dir = data_dir / f"conf_{conf_id}"

        # 2pt correlator
        fname_2pt = (
            f"twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}"
            f"_eginphase{abs(mom_smear)}{element}"
            f"_nopol_ss_conf{conf_id}.npy"
        )
        path_2pt = conf_dir / fname_2pt
        if not path_2pt.exists():
            msg = f"2pt file not found: {path_2pt}"
            logger.error(f"  {msg}")
            load_errors.append(msg)
            continue
        _corr[i] = np.load(path_2pt)
        logger.debug(f"  Loaded 2pt[{i}]: {path_2pt.name}, shape={_corr[i].shape}")

        # OPE components
        for mu, nu, arr in [(0, 1, _ope_01), (3, 0, _ope_30), (3, 1, _ope_31)]:
            fname_ope = f"ops_mu{mu}_nu{nu}_dz{delta_z}_conf{conf_id}.npz"
            path_ope = conf_dir / fname_ope
            if not path_ope.exists():
                msg = f"OPE file not found: {path_ope}"
                logger.error(f"  {msg}")
                load_errors.append(msg)
                continue
            arr[i] = np.load(path_ope)["ops"]
            logger.debug(f"  Loaded OPE[{i}] mu={mu},nu={nu}: shape={arr[i].shape}")

    if load_errors:
        logger.error(f"Failed to load {len(load_errors)} file(s). Aborting.")
        return {"status": "error", "errors": load_errors}

    logger.info(f"Data loaded: 2pt shape={_corr.shape}, OPE shapes={_ope_01.shape}")

    # Validate
    for name, data in [("2pt", _corr), ("ope_01", _ope_01),
                        ("ope_30", _ope_30), ("ope_31", _ope_31)]:
        assert np.all(np.isfinite(data)), f"{name} contains NaN/inf"
        assert np.abs(data).max() > 0, f"{name} is all zeros"
    logger.info("Data validation passed: all arrays finite and non-zero")

    # ── Step 2: Combine OPE ────────────────────────────────────────────────
    _ope = -_ope_30 - _ope_31 + 2 * _ope_01
    _ope = _ope.transpose(0, 2, 1)  # (Nconf, tau, z)
    logger.info(f"Combined OPE shape: {_ope.shape}")
    logger.info(f"  Combined OPE range: re=[{_ope.real.min():.4e}, {_ope.real.max():.4e}]")

    # ── Step 3: Build relative-time arrays ──────────────────────────────────
    _corr2_rel = np.zeros((Nconf, Nt, max_t), dtype=complex)
    _ope_rel = np.zeros((Nconf, Nt, max_t, Nx), dtype=complex)

    for ti in range(Nt):
        corr2_shift = np.roll(_corr[:, :, ti], shift=-ti, axis=1)
        _corr2_rel[:, ti, :] = corr2_shift[:, :max_t]

        ope_shift = np.roll(_ope, shift=-ti, axis=1)
        _ope_rel[:, ti, :, :] = ope_shift[:, :max_t, :]

    logger.info(f"Relative-time: corr2_rel={_corr2_rel.shape}, ope_rel={_ope_rel.shape}")

    # ── Step 4: Build 3pt disconnected correlator ──────────────────────────
    # C3(t_src, dt, dtau, z) = C2_rel(t_src, dt) * OPE_rel(t_src, dtau, z)
    _corr3 = np.zeros((Nconf, Nt, max_t, max_t, Nx), dtype=complex)
    for dt in range(max_t):
        for dtau in range(dt + 1):
            c2_slice = _corr2_rel[:, :, dt]
            ope_slice = _ope_rel[:, :, dtau, :]
            _corr3[:, :, dt, dtau, :] = ope_slice * c2_slice[:, :, np.newaxis]

    logger.info(f"3pt correlator shape: {_corr3.shape}")
    logger.info(f"  C3 range: re=[{_corr3.real.min():.4e}, {_corr3.real.max():.4e}]")

    # Free memory
    del _corr, _ope, _ope_30, _ope_31, _ope_01
    gc.collect()
    logger.debug(f"Memory after freeing raw data: peak RSS")

    # ── Step 5: Jackknife resampling ───────────────────────────────────────
    with Timer("jackknife_resample", logger):
        corr2 = resample(_corr2_rel, jack, Nsample, seed)
        ope_rs = resample(_ope_rel, jack, Nsample, seed)
        corr3 = resample(_corr3, jack, Nsample, seed)

    logger.info(f"Resampled: corr2={corr2.shape}, ope={ope_rs.shape}, corr3={corr3.shape}")

    del _corr2_rel, _ope_rel, _corr3
    gc.collect()

    # ── Step 6: Disconnected part and ratio ────────────────────────────────
    # C3_disc = C3 - C2 * OPE
    # R = mean_{t_src}(C3_disc / C2)
    corr3_disc = (
        corr3 - corr2[:, :, :, np.newaxis, np.newaxis] * ope_rs[:, :, np.newaxis, :, :]
    )
    ratio = np.mean(
        corr3_disc / corr2[:, :, :, np.newaxis, np.newaxis],
        axis=1,
    )  # → (Nsample, dt, dtau, z)

    logger.info(f"Ratio shape: {ratio.shape}")

    del corr3, corr2, ope_rs, corr3_disc
    gc.collect()

    # ── Step 7: Statistics ─────────────────────────────────────────────────
    ratio_mean = ratio.mean(axis=0)  # (dt, dtau, z)
    ratio_err = sem(ratio, jack)     # (dt, dtau, z)

    logger.info(f"Ratio mean shape: {ratio_mean.shape}")
    logger.info(f"Ratio mean stats at z={target_z}:")
    for dt in dt_list:
        vals = ratio_mean[dt, :dt + 1, target_z]
        logger.info(
            f"  dt={dt}: re=[{vals.real.min():.4f}, {vals.real.max():.4f}], "
            f"im=[{vals.imag.min():.4e}, {vals.imag.max():.4e}]"
        )

    # ── Step 8: Save numerical results ─────────────────────────────────────
    np.savez(
        output_dir / "ratio_results.npz",
        ratio=ratio,
        ratio_mean=ratio_mean,
        ratio_err=ratio_err,
        dt_list=np.array(dt_list),
        target_z=target_z,
        Nconf=Nconf,
        jack=jack,
    )
    logger.info(f"Numerical results saved to {output_dir / 'ratio_results.npz'}")

    # ── Step 9: Plot ───────────────────────────────────────────────────────
    plot_ratio(ratio_mean, ratio_err, dt_list, target_z, Nx, Px, Py, Pz,
               Nconf, jack, output_dir, logger)

    plot_diagnostics(ratio_mean, dt_list, Nx, output_dir, logger)

    plot_effective_mass(data_dir, config, output_dir, logger)

    # ── Step 10: Summary ───────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    logger.info(f"Analysis complete in {elapsed:.1f}s")

    return {
        "status": "ok",
        "ratio_path": str(output_dir / "ratio.png"),
        "diag_path": str(output_dir / "ratio_diagnostics.png"),
        "meff_path": str(output_dir / "effective_mass.png"),
        "results_path": str(output_dir / "ratio_results.npz"),
        "elapsed_seconds": elapsed,
        "ratio_mean_shape": list(ratio_mean.shape),
        "ratio_mean_stats": {
            f"z{target_z}_dt{dt}": {
                "re_range": [
                    float(ratio_mean[dt, :dt + 1, target_z].real.min()),
                    float(ratio_mean[dt, :dt + 1, target_z].real.max()),
                ],
                "re_mean": float(ratio_mean[dt, :dt + 1, target_z].real.mean()),
            }
            for dt in dt_list
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ratio(
    ratio_mean, ratio_err, dt_list, target_z, Nx,
    Px, Py, Pz, Nconf, jack, output_dir, logger,
):
    """Plot ratio R(z) at fixed z for multiple t_sep values."""
    colors = [
        "#d3d3d3", "#f38152", "#4caf50", "#00bcd4",
        "#e65100", "#ffb300", "#757575",
    ]

    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150)

    for i, dt in enumerate(dt_list):
        tau_vals = np.arange(0, dt + 1)
        x_vals = tau_vals - dt / 2.0
        y_vals = ratio_mean[dt, tau_vals, target_z]
        y_errs = ratio_err[dt, tau_vals, target_z]

        color = colors[i % len(colors)]
        ax.errorbar(
            x_vals, y_vals.real, yerr=y_errs.real,
            fmt="x", color=color, ecolor=color,
            capsize=0, markersize=7, markeredgewidth=1.8,
            linewidth=1.2,
            label=f"z={target_z}, tsep={dt}",
        )

    ax.set_title(
        f"Unpolarized Gluon, P({Px},{Py},{Pz}), z={target_z}, "
        f"Nconf={Nconf}, jackknife={jack}",
        fontsize=14, pad=12,
    )
    ax.set_xlabel(r"$\tau - t_{\rm sep}/2$", fontsize=16, labelpad=8)
    ax.set_ylabel(r"$C_3^{\rm disc} / C_2$", fontsize=16, labelpad=8)
    ax.set_xlim(-7, 7)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "ratio.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Ratio plot saved to {path}")


def plot_diagnostics(
    ratio_mean, dt_list, Nx, output_dir, logger,
):
    """Plot diagnostic 2D panels (Real and Imag parts for multiple z)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, dt in enumerate([4, 6, 8]):
        ax1 = axes[0, idx]
        ax2 = axes[1, idx]

        tau_vals = np.arange(0, dt + 1)
        x_vals = tau_vals - dt / 2.0

        for z_val in range(0, min(6, Nx)):
            y_vals = ratio_mean[dt, tau_vals, z_val]
            ax1.plot(x_vals, y_vals.real, 'o-', ms=3, label=f'z={z_val}')
            ax2.plot(x_vals, y_vals.imag, 's-', ms=3, label=f'z={z_val}')

        ax1.set_title(f"Re[Ratio], dt={dt}")
        ax2.set_title(f"Im[Ratio], dt={dt}")
        ax1.set_xlabel(r"$\tau - t_{\rm sep}/2$")
        ax2.set_xlabel(r"$\tau - t_{\rm sep}/2$")
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        if idx == 0:
            ax1.legend(fontsize=6)
            ax2.legend(fontsize=6)

    plt.tight_layout()
    path = output_dir / "ratio_diagnostics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Diagnostic plot saved to {path}")


def plot_effective_mass(
    data_dir, config, output_dir, logger,
):
    """Plot effective mass from 2pt correlator."""
    params = config["parameters"]
    ensemble = config["ensemble"]
    Nt = ensemble["Nt"]
    Nconf = params["Nconf"]
    conf_ids = params["conf_ids"]
    Px, Py = params["Px"], params["Py"]
    Pz = params["Pz"]
    mom_smear = params["mom_smear"]
    element = params["element"]
    jack = params["jackknife"]
    alttc = ensemble["alttc"]

    # Load 2pt and extract diagonal
    diag_corr = np.zeros((Nconf, Nt), dtype=complex)

    for i in range(Nconf):
        conf_id = conf_ids[i]
        fname = (
            f"twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}"
            f"_eginphase{abs(mom_smear)}{element}"
            f"_nopol_ss_conf{conf_id}.npy"
        )
        path = data_dir / f"conf_{conf_id}" / fname
        if path.exists():
            corr = np.load(path)
            diag_corr[i] = np.array([corr[t, t] for t in range(Nt)])
        else:
            logger.warning(f"  2pt file not found for meff: {path}")

    # Use original (non-resampled) for effective mass display
    diag_mean = diag_corr.mean(axis=0).real
    diag_std = diag_corr.std(axis=0).real * (np.sqrt(Nconf - 1) if jack and Nconf > 1 else 1.0)

    # Cosh effective mass
    fm2GeV = 0.1973
    meff = np.full(Nt - 2, np.nan)
    for t in range(1, Nt - 1):
        ct = abs(diag_mean[t])
        if ct > 1e-30:
            arg = (abs(diag_mean[t - 1]) + abs(diag_mean[t + 1])) / (2 * ct)
            if arg >= 1.0:
                meff[t - 1] = np.arccosh(arg) * fm2GeV / alttc

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Correlator
    ax1.errorbar(
        range(Nt), diag_mean, yerr=diag_std,
        fmt='o', ms=3, capsize=2, label=f'Pz={Pz}',
    )
    ax1.set_title(f"2pt Correlator (diagonal), P=({Px},{Py},{Pz})")
    ax1.set_xlabel("t")
    ax1.set_ylabel("C(t, t)")
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Effective mass
    valid = ~np.isnan(meff)
    ax2.errorbar(
        np.arange(1, Nt - 1)[valid], meff[valid],
        fmt='o', ms=3, capsize=2, label=f'Pz={Pz}',
    )
    ax2.set_title(f"Effective Mass (cosh, a={alttc} fm)")

    # Estimate ground state from plateau
    ps, pe = Nt // 4, Nt // 2
    plateau_mask = valid[ps:pe]
    if np.any(plateau_mask):
        plateau_val = np.mean(meff[ps:pe][plateau_mask])
        ax2.axhline(y=plateau_val, color='r', linestyle='--', alpha=0.5,
                    label=f'Plateau[{ps},{pe}]≈{plateau_val:.3f} GeV')
        ax2.legend()

    ax2.set_xlabel("t")
    ax2.set_ylabel("a m_eff [GeV]")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "effective_mass.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Effective mass plot saved to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="huangcl ratio analysis")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory")
    parser.add_argument("--config", type=str, default=None, help="Path to run_config.json")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory override")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory override")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = args.config or (run_dir / "run_config.json")
    with open(config_path) as f:
        config = json.load(f)

    data_dir = Path(args.data_dir) if args.data_dir else (run_dir / "data")
    output_dir = Path(args.output_dir) if args.output_dir else (run_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    from utils import setup_logging
    logger = setup_logging(run_dir / "run.log", "huangcl_analysis")

    results = run_analysis(config, data_dir, output_dir, logger)

    # Save summary
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Summary saved to {summary_path}")

    return 0 if results.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
