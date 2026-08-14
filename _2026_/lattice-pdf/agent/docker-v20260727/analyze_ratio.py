#!/usr/bin/env python3
"""
huangcl-style ratio analysis for the disconnected gluon PDF.

Enhanced version (docker-v20260727):
  - Computes the disconnected 3pt/2pt ratio: R(z, t_sep, tau) = C3_disc / C2
  - Jackknife resampling with detailed error analysis
  - Fixed plots: uses source-averaged C2pt_1d (not diagonal C(t,t)=0)
  - All intermediate analysis results saved
  - Comprehensive logging

Adapted from:
  - /root/lattice-pdf/examples/huangcl/code.py
  - /root/lattice-pdf/agent/docker-v20260726/analyze_ratio.py

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
from utils import Timer, print_banner, save_intermediate


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical utilities
# ═══════════════════════════════════════════════════════════════════════════════

def sem(data: np.ndarray, jack: bool) -> np.ndarray:
    """Standard error of the mean. For jackknife, multiply by sqrt(N-1)."""
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
    """Jackknife or bootstrap resampling along axis=0 (config index)."""
    n_conf = corr.shape[0]
    if jack:
        if n_conf <= 1:
            return corr.copy()  # Cannot jackknife with 1 config
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
      8. Plot and save (ratio, diagnostics, effective mass, field strength)

    Returns:
        dict of results including paths to output files.
    """
    params = config["parameters"]
    ensemble = config["ensemble"]

    Nt = ensemble["Nt"]
    Nx = ensemble["Nx"]
    Nconf = params["Nconf"]
    conf_ids = params["conf_ids"]
    Px, Py = params["Px"], params["Py"]
    Pz = params["Pz"]
    mom_smear = params["mom_smear"]
    element = params["element"]
    delta_z = params["delta_z"]
    jack = params["jackknife"]
    max_t = params["max_t"]
    target_z = params["target_z"]
    dt_list = params["dt_list"]
    seed = params["seed"]

    Nsample = Nconf if jack else 3000

    print_banner("Step 03: huangcl Ratio Analysis", logger)
    logger.info(f"  Ensemble: {ensemble['full_name']} ({ensemble['name']})")
    logger.info(f"  Nconf={Nconf}, Nt={Nt}, Nx={Nx}")
    logger.info(f"  Momentum: P=({Px},{Py},{Pz})")
    logger.info(f"  Jackknife: {jack}, Nsample={Nsample}")
    logger.info(f"  max_t={max_t}, target_z={target_z}, dt_list={dt_list}")

    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()

    # ── Step 1: Load data ──────────────────────────────────────────────────
    logger.info("Loading 2pt and OPE data...")

    _corr = np.zeros((Nconf, Nt, Nt), dtype=complex)
    _ope_01 = np.zeros((Nconf, Nx, Nt), dtype=complex)
    _ope_30 = np.zeros((Nconf, Nx, Nt), dtype=complex)
    _ope_31 = np.zeros((Nconf, Nx, Nt), dtype=complex)

    load_errors = []
    loaded_confs = []

    for i in range(Nconf):
        conf_id = conf_ids[i]
        conf_dir = data_dir / f"conf_{conf_id}"

        if not conf_dir.exists():
            msg = f"Config directory not found: {conf_dir}"
            logger.error(f"  {msg}")
            load_errors.append(msg)
            continue

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

        ope_loaded = 0
        for mu, nu, arr in [(0, 1, _ope_01), (3, 0, _ope_30), (3, 1, _ope_31)]:
            fname_ope = f"ops_mu{mu}_nu{nu}_dz{delta_z}_conf{conf_id}.npz"
            path_ope = conf_dir / fname_ope
            if not path_ope.exists():
                msg = f"OPE file not found: {path_ope}"
                logger.error(f"  {msg}")
                load_errors.append(msg)
                continue
            arr[i] = np.load(path_ope)["ops"]
            ope_loaded += 1

        if ope_loaded == 3:
            loaded_confs.append(conf_id)

    if load_errors:
        logger.error(f"Failed to load {len(load_errors)} file(s).")
        if len(loaded_confs) == 0:
            return {"status": "error", "errors": load_errors}
        logger.warning(f"Proceeding with {len(loaded_confs)}/{Nconf} configs: {loaded_confs}")

    logger.info(f"Data loaded: 2pt shape={_corr.shape}, "
                f"OPE shapes={_ope_01.shape}, loaded_confs={loaded_confs}")

    for name, data in [("2pt", _corr), ("ope_01", _ope_01),
                        ("ope_30", _ope_30), ("ope_31", _ope_31)]:
        assert np.all(np.isfinite(data)), f"{name} contains NaN/inf"
        assert np.abs(data).max() > 0, f"{name} is all zeros"
    logger.info("Data validation passed: all arrays finite and non-zero")

    # ── Step 2: Combine OPE ────────────────────────────────────────────────
    _ope = -_ope_30 - _ope_31 + 2 * _ope_01
    _ope = _ope.transpose(0, 2, 1)
    logger.info(f"Combined OPE shape: {_ope.shape}")
    logger.info(f"  Combined OPE range: re=[{_ope.real.min():.4e}, {_ope.real.max():.4e}], "
                f"im=[{_ope.imag.min():.4e}, {_ope.imag.max():.4e}]")

    save_intermediate(
        {"ope_combined": _ope, "formula": "-ope_30 - ope_31 + 2*ope_01"},
        output_dir, "ope_combined.npz", logger,
    )

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
    logger.info("Building 3pt disconnected correlator...")
    _corr3 = np.zeros((Nconf, Nt, max_t, max_t, Nx), dtype=complex)
    for dt in range(max_t):
        for dtau in range(dt + 1):
            c2_slice = _corr2_rel[:, :, dt]
            ope_slice = _ope_rel[:, :, dtau, :]
            _corr3[:, :, dt, dtau, :] = ope_slice * c2_slice[:, :, np.newaxis]

    logger.info(f"3pt correlator shape: {_corr3.shape}")
    logger.info(f"  C3 range: re=[{_corr3.real.min():.4e}, {_corr3.real.max():.4e}]")

    save_intermediate(
        {"corr3": _corr3, "corr2_rel": _corr2_rel, "ope_rel": _ope_rel},
        output_dir, "correlators_rel_time.npz", logger,
    )

    del _corr, _ope, _ope_30, _ope_31, _ope_01
    gc.collect()

    # ── Step 5: Jackknife resampling ───────────────────────────────────────
    with Timer("jackknife_resample", logger):
        corr2 = resample(_corr2_rel, jack, Nsample, seed)
        ope_rs = resample(_ope_rel, jack, Nsample, seed)
        corr3 = resample(_corr3, jack, Nsample, seed)

    logger.info(f"Resampled: corr2={corr2.shape}, ope={ope_rs.shape}, corr3={corr3.shape}")

    del _corr2_rel, _ope_rel, _corr3
    gc.collect()

    # ── Step 6: Disconnected part and ratio ────────────────────────────────
    logger.info("Computing disconnected ratio...")
    corr3_disc = (
        corr3 - corr2[:, :, :, np.newaxis, np.newaxis] * ope_rs[:, :, np.newaxis, :, :]
    )

    eps = 1e-30
    ratio = np.mean(
        corr3_disc / (corr2[:, :, :, np.newaxis, np.newaxis] + eps),
        axis=1,
    )

    logger.info(f"Ratio shape: {ratio.shape}")

    del corr3, corr2, ope_rs, corr3_disc
    gc.collect()

    # ── Step 7: Statistics ─────────────────────────────────────────────────
    ratio_mean = ratio.mean(axis=0)
    ratio_err = sem(ratio, jack)

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
        loaded_confs=np.array(loaded_confs),
    )
    logger.info(f"Numerical results saved to {output_dir / 'ratio_results.npz'}")

    # ── Step 9: Plots ──────────────────────────────────────────────────────
    plot_ratio(ratio_mean, ratio_err, dt_list, target_z, Nx, Px, Py, Pz,
               Nconf, jack, output_dir, logger)

    plot_diagnostics(ratio_mean, dt_list, Nx, output_dir, logger)

    plot_effective_mass(data_dir, config, output_dir, logger)

    plot_field_strength_diagnostics(config, data_dir, output_dir, logger)

    # ── Step 10: Summary ───────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    logger.info(f"Analysis complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return {
        "status": "ok",
        "ratio_path": str(output_dir / "ratio.png"),
        "diag_path": str(output_dir / "ratio_diagnostics.png"),
        "meff_path": str(output_dir / "effective_mass.png"),
        "field_strength_path": str(output_dir / "field_strength_diagnostics.png"),
        "results_path": str(output_dir / "ratio_results.npz"),
        "elapsed_seconds": elapsed,
        "ratio_mean_shape": list(ratio_mean.shape),
        "loaded_confs": loaded_confs,
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

        # Skip if all NaN
        if np.all(np.isnan(y_vals.real)):
            continue

        color = colors[i % len(colors)]
        ax.errorbar(
            x_vals, y_vals.real, yerr=y_errs.real,
            fmt="x", color=color, ecolor=color,
            capsize=0, markersize=7, markeredgewidth=1.8,
            linewidth=1.2,
            label=f"z={target_z}, tsep={dt}",
        )

    ax.set_title(
        f"Unpolarized Gluon PDF, P({Px},{Py},{Pz}), z={target_z}, "
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
        if dt >= ratio_mean.shape[0]:
            continue
        ax1 = axes[0, idx]
        ax2 = axes[1, idx]

        tau_vals = np.arange(0, dt + 1)
        x_vals = tau_vals - dt / 2.0

        for z_val in range(0, min(6, Nx)):
            y_vals = ratio_mean[dt, tau_vals, z_val]
            if np.all(np.isnan(y_vals.real)):
                continue
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


# ═══════════════════════════════════════════════════════════════════════════════
# EFFECTIVE MASS PLOT — FIXED
# Uses C2pt_1d from saved meff files (source-averaged), NOT C(t,t) diagonal
# ═══════════════════════════════════════════════════════════════════════════════

def plot_effective_mass(
    data_dir, config, output_dir, logger,
):
    """Plot effective mass using source-averaged C2pt_1d from saved .npz files.

    The 2pt correlator diagonal C(t,t) is identically zero (parity projection
    artifact at this momentum). We use the properly source-averaged C2pt_1d
    that was saved in meff_Pz{N}_conf{id}.npz during the 2pt computation step.
    """
    params = config["parameters"]
    ensemble = config["ensemble"]
    Nt = ensemble["Nt"]
    Nconf = params["Nconf"]
    conf_ids = params["conf_ids"]
    Pz = params["Pz"]
    alttc = ensemble["alttc"]

    # ── Load C2pt_1d and meff from saved files ───────────────────────────
    C2pt_all = np.zeros((Nconf, Nt))
    meff_all = np.zeros((Nconf, Nt - 2))
    plateau_vals = []
    loaded_ok = 0

    for i in range(Nconf):
        conf_id = conf_ids[i]
        meff_path = data_dir / f"conf_{conf_id}" / f"meff_Pz{Pz}_conf{conf_id}.npz"
        if meff_path.exists():
            data = np.load(meff_path)
            C2pt_all[i] = data["C2pt_1d"]
            meff_all[i] = data["meff_gev"]
            plateau_vals.append(float(data["meff_plateau_gev"]))
            loaded_ok += 1
        else:
            logger.warning(f"  meff file not found: {meff_path}")

    if loaded_ok == 0:
        logger.error("No meff files found — cannot plot effective mass")
        return

    C2pt_mean = C2pt_all[:loaded_ok].mean(axis=0)
    C2pt_std = C2pt_all[:loaded_ok].std(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))

    # ── Panel 1: Source-averaged C2pt_1d (signed, linear) ──────────────────
    ax1 = axes[0]
    t_vals = np.arange(Nt)
    # Plot absolute value with sign as color
    pos = C2pt_mean >= 0
    neg = C2pt_mean < 0
    if np.any(pos):
        ax1.errorbar(t_vals[pos], C2pt_mean[pos], yerr=C2pt_std[pos],
                    fmt='o', ms=4, capsize=2, color='#2196F3',
                    label=f'C>0 ({np.sum(pos)})')
    if np.any(neg):
        ax1.errorbar(t_vals[neg], -C2pt_mean[neg], yerr=C2pt_std[neg],
                    fmt='s', ms=4, capsize=2, color='#f44336',
                    label=f'C<0 (|×|, {np.sum(neg)})')
    ax1.set_title(f"2pt Correlator (source-averaged), Pz={Pz}")
    ax1.set_xlabel("Δt")
    ax1.set_ylabel("C(Δt) [signed]")
    ax1.axhline(y=0, color='gray', lw=0.5, ls='--')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Effective mass per config + mean ──────────────────────────
    ax2 = axes[1]
    fm2GeV = 0.1973
    colors = plt.cm.tab10(np.linspace(0, 1, max(loaded_ok, 3)))

    t_meff = np.arange(1, Nt - 1)
    for i in range(loaded_ok):
        valid = ~np.isnan(meff_all[i])
        if np.sum(valid) > 0:
            ax2.plot(t_meff[valid], meff_all[i][valid], 'o-', ms=3, alpha=0.4,
                    color=colors[i], linewidth=0.8)

    # Compute jackknife mean and error of meff
    meff_jk = meff_all[:loaded_ok]
    meff_mean_jk = np.full(Nt - 2, np.nan)
    meff_err_jk = np.full(Nt - 2, np.nan)
    for t in range(Nt - 2):
        vals = meff_jk[:, t]
        finite = vals[~np.isnan(vals)]
        if len(finite) >= 2:
            meff_mean_jk[t] = np.mean(finite)
            meff_err_jk[t] = np.std(finite) * np.sqrt(loaded_ok - 1)

    valid_mean = ~np.isnan(meff_mean_jk)
    if np.any(valid_mean):
        ax2.errorbar(t_meff[valid_mean], meff_mean_jk[valid_mean],
                    yerr=meff_err_jk[valid_mean],
                    fmt='ko-', ms=5, capsize=2, linewidth=1.5,
                    label=f'Mean ({loaded_ok} cfgs)')

    # Plateau estimate
    ps, pe = Nt // 4, min(Nt // 2, Nt - 2)
    plateau_mask = valid_mean[ps:pe]
    if np.any(plateau_mask):
        plateau_val = np.mean(meff_mean_jk[ps:pe][plateau_mask])
        ax2.axhline(y=plateau_val, color='r', linestyle='--', alpha=0.7, lw=1.5,
                   label=f'Plateau[{ps},{pe}]≈{plateau_val:.3f} GeV')
        # Proton mass reference
        ax2.axhline(y=0.938, color='g', linestyle=':', alpha=0.5, lw=1,
                   label='m_proton ≈ 0.938 GeV')

    ax2.set_title(f"Effective Mass (cosh, a={alttc} fm)")
    ax2.set_xlabel("t")
    ax2.set_ylabel("a·m_eff [GeV]")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # ── Panel 3: C2pt per config (zoomed, linear) ──────────────────────────
    ax3 = axes[2]
    t_zoom = min(20, Nt)  # Zoom to first 20 time slices
    for i in range(loaded_ok):
        ax3.plot(np.arange(t_zoom), np.abs(C2pt_all[i, :t_zoom]), 'o-',
                ms=3, alpha=0.5, color=colors[i], linewidth=0.8,
                label=f'conf {conf_ids[i]}')
    ax3.set_title(f"|C(Δt)| per Config (t≤{t_zoom})")
    ax3.set_xlabel("Δt")
    ax3.set_ylabel("|C(Δt)|")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    plt.tight_layout()
    path = output_dir / "effective_mass.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Effective mass plot saved to {path} ({loaded_ok} configs)")


# ═══════════════════════════════════════════════════════════════════════════════
# FIELD STRENGTH DIAGNOSTICS PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def plot_field_strength_diagnostics(
    config, data_dir, output_dir, logger,
):
    """Plot field strength diagnostics from OPE and F_{mu,nu} data."""
    params = config["parameters"]
    conf_ids = params["conf_ids"]
    ensemble = config["ensemble"]
    Nt = ensemble["Nt"]
    Nx = ensemble["Nx"]

    # Load OPE combined data per config
    ope_data = {}
    for conf_id in conf_ids:
        conf_dir = data_dir / f"conf_{conf_id}"
        ope_01_path = conf_dir / f"ops_mu0_nu1_dz{params['delta_z']}_conf{conf_id}.npz"
        ope_30_path = conf_dir / f"ops_mu3_nu0_dz{params['delta_z']}_conf{conf_id}.npz"
        ope_31_path = conf_dir / f"ops_mu3_nu1_dz{params['delta_z']}_conf{conf_id}.npz"

        if ope_01_path.exists() and ope_30_path.exists() and ope_31_path.exists():
            ope_01 = np.load(ope_01_path)["ops"]
            ope_30 = np.load(ope_30_path)["ops"]
            ope_31 = np.load(ope_31_path)["ops"]
            ope_combined = -ope_30 - ope_31 + 2 * ope_01
            ope_data[conf_id] = ope_combined

    if not ope_data:
        logger.warning("No OPE data available for field strength plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. |OPE(z,t)| averaged over z, vs t
    ax1 = axes[0, 0]
    for conf_id, ope in ope_data.items():
        ax1.plot(range(Nt), np.abs(ope).mean(axis=0), 'o-', ms=2,
                label=f'conf {conf_id}')
    ax1.set_title(r"⟨|O(z,t)|⟩$_z$ vs t")
    ax1.set_xlabel("t")
    ax1.set_ylabel(r"⟨|O(z,t)|⟩$_z$")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # 2. Re[OPE(z)] vs z at fixed t=Nt/2
    ax2 = axes[0, 1]
    t_fixed = Nt // 2
    for conf_id, ope in ope_data.items():
        ax2.plot(range(ope.shape[0]), ope[:, t_fixed].real,
                'o-', ms=2, label=f'conf {conf_id}')
    ax2.set_title(f"Re[O(z, t={t_fixed})] vs z")
    ax2.set_xlabel("z")
    ax2.set_ylabel("Re[O(z)]")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # 3. Im[OPE(z,t)] at fixed t
    ax3 = axes[1, 0]
    for conf_id, ope in ope_data.items():
        ax3.plot(range(ope.shape[0]), ope[:, t_fixed].imag,
                's-', ms=2, label=f'conf {conf_id}')
    ax3.set_title(f"Im[O(z, t={t_fixed})] vs z")
    ax3.set_xlabel("z")
    ax3.set_ylabel("Im[O(z)]")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # 4. Re[OPE(z=0,t)] vs t (local operator)
    ax4 = axes[1, 1]
    for conf_id, ope in ope_data.items():
        ax4.plot(range(Nt), ope[0, :].real, 'o-', ms=2,
                label=f'conf {conf_id}')
    ax4.set_title("Re[O(z=0, t)] vs t (local)")
    ax4.set_xlabel("t")
    ax4.set_ylabel("Re[O(z=0)]")
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "field_strength_diagnostics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Field strength diagnostic plot saved to {path}")


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

    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Summary saved to {summary_path}")

    return 0 if results.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
