#!/usr/bin/env python3
"""
huangcl ratio analysis — adapted from /root/lattice-pdf/examples/huangcl/code.py

Computes the disconnected 3pt/2pt ratio for the unpolarized gluon PDF
using jackknife resampling, then plots the results.

Differences from original:
- Reads from configurable data paths (not hardcoded cluster paths)
- Nconf is configurable (default 3 for testing)
- More diagnostic output saved (numerical results as .npz, detailed logs)
- Function-based structure for use in pipeline

Usage:
    python analyze_ratio.py --run-dir /path/to/run_dir [--data-dir /path/to/data]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import resource
import sys
import time
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


# ─── Logging setup ───────────────────────────────────────────────────────────

def setup_logging(log_file: Path) -> logging.Logger:
    """Configure logging to both file and stdout."""
    logger = logging.getLogger("huangcl_analysis")
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ─── Statistical utilities ───────────────────────────────────────────────────

def sem(data: np.ndarray, jack: bool) -> np.ndarray:
    """Standard error of the mean. For jackknife, multiply by sqrt(N-1)."""
    error = data.std(axis=0)
    if jack:
        error = error * np.sqrt(data.shape[0] - 1)
    return error


def resample(corr: np.ndarray, jack: bool, Nsample: int, seed: int = 0) -> np.ndarray:
    """Jackknife or bootstrap resampling along axis=0 (config index).

    Args:
        corr: Input data, shape (n_conf, ...)
        jack: If True, use jackknife; otherwise bootstrap
        Nsample: Number of resamples (for jackknife, equals n_conf)
        seed: Random seed for bootstrap

    Returns:
        Resampled data with same leading dimension replaced by resample index.
    """
    n_conf = corr.shape[0]
    if jack:
        # Jackknife: leave-one-out estimates
        re_corr = (n_conf * corr.mean(axis=0) - corr) / (n_conf - 1)
    else:
        rng = np.random.default_rng(seed=seed)
        idx = rng.integers(0, n_conf, size=(Nsample, n_conf))
        re_corr = corr[idx].mean(axis=1)
    return re_corr


def get_peak_memory_gb() -> float:
    """Return peak RSS memory usage in GB."""
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max_rss / (1024 ** 2)


# ─── Main analysis ───────────────────────────────────────────────────────────

def run_analysis(
    data_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    *,
    Nconf: int = 3,
    Nt: int = 72,
    Nx: int = 24,
    Px: int = 0,
    Py: int = 0,
    Pz: int = 2,
    conf_start: int = 6250,
    conf_step: int = 200,
    conf_name: str = "beta6.20_mu-0.2770_ms-0.2400_L24x72",
    conf_short: str = "L24x72",
    jack: bool = True,
    Nsample: int | None = None,
    max_t: int = 20,
    target_z: int = 2,
    dt_list: list[int] | None = None,
    element: str = "_Cg5g4",
    mom_smear: int = -2,
    delta_z: int = 24,
) -> dict:
    """Run the full huangcl-style ratio analysis.

    Args:
        data_dir: Directory containing conf_XXXX/ subdirectories with .npy/.npz files.
        output_dir: Directory for output plots and data.
        logger: Logger instance.

    Returns:
        Dictionary of results including paths to output files.
    """
    if dt_list is None:
        dt_list = [4, 5, 6, 7, 8, 9, 10]

    if Nsample is None:
        Nsample = Nconf if jack else 3000

    logger.info("=" * 70)
    logger.info("huangcl Ratio Analysis")
    logger.info(f"  Ensemble: {conf_name} ({conf_short})")
    logger.info(f"  Nconf={Nconf}, Nt={Nt}, Nx={Nx}")
    logger.info(f"  Momentum: P=({Px},{Py},{Pz})")
    logger.info(f"  Jackknife: {jack}, Nsample={Nsample}")
    logger.info(f"  max_t={max_t}, target_z={target_z}, dt_list={dt_list}")
    logger.info(f"  Data dir: {data_dir}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info("=" * 70)

    t_start = time.time()

    # ── Load data ────────────────────────────────────────────────────────
    _corr = np.zeros((Nconf, Nt, Nt), dtype=complex)
    _ope_01 = np.zeros((Nconf, Nx, Nt), dtype=complex)
    _ope_30 = np.zeros((Nconf, Nx, Nt), dtype=complex)
    _ope_31 = np.zeros((Nconf, Nx, Nt), dtype=complex)

    for i in range(Nconf):
        conf_id = conf_start + i * conf_step
        conf_dir = data_dir / f"conf_{conf_id}"

        # 2pt correlator
        fname_2pt = (
            f"twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}"
            f"_eginphase{abs(mom_smear)}"
            f"{element}_nopol_ss_conf{conf_id}.npy"
        )
        path_2pt = conf_dir / fname_2pt
        if not path_2pt.exists():
            raise FileNotFoundError(f"2pt file not found: {path_2pt}")
        _corr[i] = np.load(path_2pt)
        logger.debug(f"  Loaded 2pt[{i}]: {path_2pt.name}, shape={_corr[i].shape}")

        # OPE components
        for mu, nu, arr in [(0, 1, _ope_01), (3, 0, _ope_30), (3, 1, _ope_31)]:
            fname_ope = f"ops_mu{mu}_nu{nu}_dz{delta_z}_conf{conf_id}.npz"
            path_ope = conf_dir / fname_ope
            if not path_ope.exists():
                raise FileNotFoundError(f"OPE file not found: {path_ope}")
            arr[i] = np.load(path_ope)["ops"]
            logger.debug(f"  Loaded OPE[{i}] mu={mu},nu={nu}: shape={arr[i].shape}")

    logger.info(f"Data loaded: 2pt shape={_corr.shape}, OPE shapes={_ope_01.shape}")

    # Validate loaded data
    for name, data in [("2pt", _corr), ("ope_01", _ope_01), ("ope_30", _ope_30), ("ope_31", _ope_31)]:
        assert np.all(np.isfinite(data)), f"{name} contains NaN/inf"
        assert np.abs(data).max() > 0, f"{name} is all zeros"
    logger.info("Data validation passed: all arrays finite and non-zero")

    # ── Combine OPE ──────────────────────────────────────────────────────
    _ope = -_ope_30 - _ope_31 + 2 * _ope_01
    _ope = _ope.transpose(0, 2, 1)  # (Nconf, tau, z)
    logger.info(f"Combined OPE shape: {_ope.shape}")

    # ── Build relative-time offsets ──────────────────────────────────────
    _corr2_rel = np.zeros((Nconf, Nt, max_t), dtype=complex)
    _ope_rel = np.zeros((Nconf, Nt, max_t, Nx), dtype=complex)

    for ti in range(Nt):
        corr2_shift = np.roll(_corr[:, :, ti], shift=-ti, axis=1)
        _corr2_rel[:, ti, :] = corr2_shift[:, :max_t]

        ope_shift = np.roll(_ope, shift=-ti, axis=1)
        _ope_rel[:, ti, :, :] = ope_shift[:, :max_t, :]

    logger.info(f"Relative-time arrays: corr2_rel={_corr2_rel.shape}, ope_rel={_ope_rel.shape}")

    # ── Build 3pt correlator ─────────────────────────────────────────────
    _corr3 = np.zeros((Nconf, Nt, max_t, max_t, Nx), dtype=complex)
    for dt in range(max_t):
        for dtau in range(dt + 1):
            c2_slice = _corr2_rel[:, :, dt]
            ope_slice = _ope_rel[:, :, dtau, :]
            _corr3[:, :, dt, dtau, :] = ope_slice * c2_slice[:, :, np.newaxis]

    logger.info(f"3pt correlator shape: {_corr3.shape}")

    # Free memory
    del _corr, _ope, _ope_30, _ope_31, _ope_01
    gc.collect()
    logger.debug(f"Memory after freeing raw data: {get_peak_memory_gb():.3f} GB")

    # ── Resampling ───────────────────────────────────────────────────────
    corr2 = resample(_corr2_rel, jack, Nsample)
    ope_rs = resample(_ope_rel, jack, Nsample)
    corr3 = resample(_corr3, jack, Nsample)

    logger.info(f"Resampled shapes: corr2={corr2.shape}, ope={ope_rs.shape}, corr3={corr3.shape}")

    del _corr2_rel, _ope_rel, _corr3
    gc.collect()

    # ── Disconnected part and ratio ──────────────────────────────────────
    corr3_disc = (
        corr3 - corr2[:, :, :, np.newaxis, np.newaxis] * ope_rs[:, :, np.newaxis, :, :]
    )
    ratio = np.mean(
        corr3_disc / corr2[:, :, :, np.newaxis, np.newaxis],
        axis=1,
    )
    # ratio shape: (Nsample, dt, dtau, z)

    logger.info(f"Ratio shape: {ratio.shape}")

    del corr3, corr2, ope_rs, corr3_disc
    gc.collect()

    # ── Statistics ───────────────────────────────────────────────────────
    ratio_mean = ratio.mean(axis=0)  # (dt, dtau, z)
    ratio_err = sem(ratio, jack)     # (dt, dtau, z)

    logger.info(f"Ratio mean shape: {ratio_mean.shape}")
    logger.info(f"Ratio mean stats at z={target_z}:")
    for dt in dt_list:
        vals = ratio_mean[dt, :dt + 1, target_z]
        logger.info(f"  dt={dt}: real range [{vals.real.min():.4f}, {vals.real.max():.4f}]")

    # ── Save numerical results ───────────────────────────────────────────
    np.savez(
        output_dir / "ratio_results.npz",
        ratio=ratio,
        ratio_mean=ratio_mean,
        ratio_err=ratio_err,
        dt_list=dt_list,
        target_z=target_z,
        Nconf=Nconf,
        jack=jack,
    )
    logger.info(f"Numerical results saved to {output_dir / 'ratio_results.npz'}")

    # ── Plot ─────────────────────────────────────────────────────────────
    colors = [
        "#d3d3d3", "#f38152", "#4caf50", "#00bcd4",
        "#e65100", "#ffb300", "#757575",
    ]
    markers = ["x"] * 7

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
        f"Unpolarized, P({Px},{Py},{Pz}), z={target_z}, "
        f"Nconf={Nconf}, jackknife={jack}",
        fontsize=14, pad=12,
    )
    ax.set_xlabel(r"$\tau - t_{\rm sep}/2$", fontsize=16, labelpad=8)
    ax.set_ylabel(r"$C_3^{\rm disc} / C_2$", fontsize=16, labelpad=8)
    ax.set_xlim(-7, 7)
    ax.set_ylim(-0.1, 1.2)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ratio_path = output_dir / "ratio.png"
    plt.savefig(ratio_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Ratio plot saved to {ratio_path}")

    # ── Additional diagnostic plots ──────────────────────────────────────

    # 2D heatmap of ratio at fixed dt
    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
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
        if idx == 0:
            ax1.legend(fontsize=6)
    plt.tight_layout()
    diag_path = output_dir / "ratio_diagnostics.png"
    fig2.savefig(diag_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Diagnostic plot saved to {diag_path}")

    # ── Effective mass from 2pt ──────────────────────────────────────────
    _corr_2pt_mean = np.zeros((Nconf, Nt, Nt), dtype=complex)
    for i in range(Nconf):
        conf_id = conf_start + i * conf_step
        conf_dir = data_dir / f"conf_{conf_id}"
        fname_2pt = (
            f"twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}"
            f"_eginphase{abs(mom_smear)}"
            f"{element}_nopol_ss_conf{conf_id}.npy"
        )
        _corr_2pt_mean[i] = np.load(conf_dir / fname_2pt)

    # Diagonal: C(t, t)
    diag_corr = np.array([_corr_2pt_mean[i, t, t] for i in range(Nconf) for t in range(Nt)]).reshape(Nconf, Nt)
    diag_corr_jk = resample(diag_corr, jack, Nsample)
    diag_mean = diag_corr_jk.mean(axis=0).real
    diag_err = sem(diag_corr_jk, jack).real

    # Cosh effective mass
    meff = np.zeros(Nt - 2)
    meff_err = np.zeros(Nt - 2)
    for t in range(1, Nt - 1):
        ct = diag_mean[t]
        ctp1 = diag_mean[t + 1]
        ctm1 = diag_mean[t - 1]
        if ct > 0 and (ctp1 + ctm1) / (2 * ct) > 1:
            meff[t - 1] = np.arccosh((ctp1 + ctm1) / (2 * ct))
        else:
            meff[t - 1] = np.nan
    meff_err[t - 1] = diag_err[t] / diag_mean[t] * 0.5  # Approximate

    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.errorbar(range(Nt), diag_mean, yerr=diag_err, fmt='o', ms=3, capsize=2)
    ax1.set_title("2pt Correlator (diagonal)")
    ax1.set_xlabel("t")
    ax1.set_ylabel("C(t, t)")
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    valid = ~np.isnan(meff)
    ax2.errorbar(np.arange(1, Nt - 1)[valid], meff[valid],
                 yerr=meff_err[valid], fmt='o', ms=3, capsize=2)
    ax2.set_title("Effective Mass (cosh)")
    ax2.set_xlabel("t")
    ax2.set_ylabel("a m_eff")
    ax2.axhline(y=0.45, color='r', linestyle='--', alpha=0.5, label='E0=0.45 (input)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    meff_path = output_dir / "effective_mass.png"
    fig3.savefig(meff_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Effective mass plot saved to {meff_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    mem_gb = get_peak_memory_gb()
    logger.info(f"Analysis complete in {elapsed:.1f}s, peak memory: {mem_gb:.3f} GB")

    return {
        "ratio_path": str(ratio_path),
        "diag_path": str(diag_path),
        "meff_path": str(meff_path),
        "results_path": str(output_dir / "ratio_results.npz"),
        "elapsed_seconds": elapsed,
        "peak_memory_gb": mem_gb,
        "ratio_mean_shape": list(ratio_mean.shape),
        "ratio_mean_stats": {
            f"z{target_z}_dt{dt}_re_range": [
                float(ratio_mean[dt, :dt + 1, target_z].real.min()),
                float(ratio_mean[dt, :dt + 1, target_z].real.max()),
            ]
            for dt in dt_list
        },
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="huangcl ratio analysis")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory (default: RUN_DIR/02_sample_data)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: RUN_DIR/04_huangcl_analysis/output)")
    parser.add_argument("--Nconf", type=int, default=3, help="Number of configurations")
    parser.add_argument("--conf-start", type=int, default=6250, help="Starting config ID")
    parser.add_argument("--conf-step", type=int, default=200, help="Config ID step")
    parser.add_argument("--no-jack", action="store_true", help="Disable jackknife")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir) if args.data_dir else run_dir / "02_sample_data"
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "04_huangcl_analysis" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        params = config["parameters"]
    else:
        params = {}

    # Setup logging
    log_file = run_dir / "04_huangcl_analysis" / "analysis.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file)

    results = run_analysis(
        data_dir=data_dir,
        output_dir=output_dir,
        logger=logger,
        Nconf=args.Nconf,
        Nt=params.get("Nt", 72),
        Nx=params.get("Nx", 24),
        Px=params.get("Px", 0),
        Py=params.get("Py", 0),
        Pz=params.get("Pz", 2),
        conf_start=args.conf_start,
        conf_step=args.conf_step,
        conf_name=params.get("conf_name", "beta6.20_mu-0.2770_ms-0.2400_L24x72"),
        conf_short=params.get("ensemble", "L24x72"),
        jack=not args.no_jack,
        max_t=params.get("max_t", 20),
        target_z=params.get("target_z", 2),
        dt_list=params.get("dt_list", [4, 5, 6, 7, 8, 9, 10]),
    )

    # Save results summary
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Summary saved to {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
