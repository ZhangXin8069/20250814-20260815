#!/usr/bin/env python3
"""
Statistical analysis module — Jackknife, effective mass, ratio_3pt.

This module provides self-contained implementations of the lattice QCD
statistical analysis toolkit, following the patterns established in:
  - lqcddb.analyse.analyse (Jackknife, meff, ratio_3pt)
  - zhangxin/include.py (data_analyse class)

All functions operate on numpy arrays. GPU acceleration is used for
large tensor operations where beneficial.

Key functions:
  - Jackknife: resampling with covariance estimation
  - meff: effective mass extraction (log, cosh methods)
  - ratio_3pt: 3pt/2pt ratio for PDF/FF extraction
  - plot_correlator: publication-quality correlator plots
  - plot_meff: effective mass plateau plots
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.optimize import curve_fit

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ═══════════════════════════════════════════════════════════════════════════════
# Physical constants
# ═══════════════════════════════════════════════════════════════════════════════

fm2GeV = 0.1973269804  # ℏc in GeV·fm  (conversion: 1/fm → GeV)

# ═══════════════════════════════════════════════════════════════════════════════
# Jackknife resampling
# ═══════════════════════════════════════════════════════════════════════════════


def Jackknife(
    data: np.ndarray,
    Nconf_axes: int = 0,
    only_sample: bool = False,
    cov_axes: Optional[Union[int, Tuple[int, ...]]] = None,
) -> dict:
    """Jackknife resampling with optional covariance estimation.

    The jackknife method estimates the mean and error of an observable
    by systematically leaving out one configuration at a time:
        O^{(k)} = (1/(N-1)) Σ_{i≠k} O_i
        mean = (1/N) Σ_k O^{(k)}
        error = √((N-1)/N Σ_k (O^{(k)} - mean)²)

    This is identical to the lqcddb.analyse.analyse.Jackknife function
    but is re-implemented here for self-containedness.

    Args:
        data: Input array with at least Nconf_axes dimension.
        Nconf_axes: Axis index for the configuration (jackknife) dimension.
        only_sample: If True, return only jackknife samples.
        cov_axes: Axis/axes for covariance matrix computation.

    Returns:
        dict with keys:
          'data_sample': jackknife samples (same shape as input)
          'data_mean': mean over configurations
          'data_err': standard error = √(N-1) × std(samples)
          'data_cov': covariance matrix (if cov_axes specified)
    """
    ndim = data.ndim
    Nconf_axes = Nconf_axes % ndim
    Nconf = data.shape[Nconf_axes]

    # 1. Jackknife samples: leave-one-out mean
    data_sum = np.sum(data, axis=Nconf_axes, keepdims=True)
    data_sample = -(data - data_sum) / (Nconf - 1)

    if only_sample:
        return {"data_sample": data_sample}

    # 2. Mean and standard error
    data_mean = np.mean(data, axis=Nconf_axes)
    data_err = np.sqrt(Nconf - 1) * np.std(data_sample, axis=Nconf_axes)

    result = {
        "data_sample": data_sample,
        "data_mean": data_mean,
        "data_err": data_err,
    }

    # 3. Covariance matrix
    if cov_axes is not None:
        if isinstance(cov_axes, int):
            cov_axes = (cov_axes % ndim,)
        else:
            cov_axes = tuple(ax % ndim for ax in cov_axes)

        residual = data_sample - data_mean

        # Reorder: Nconf → other → cov
        all_axes = list(range(ndim))
        other_axes = [
            ax for ax in all_axes
            if ax != Nconf_axes and ax not in cov_axes
        ]
        new_order = [Nconf_axes] + other_axes + list(cov_axes)
        r = np.transpose(residual, new_order)

        shape_other = [residual.shape[ax] for ax in other_axes]
        shape_cov = [residual.shape[ax] for ax in cov_axes]
        N_cov = int(np.prod(shape_cov))
        r_flat = r.reshape([Nconf] + shape_other + [N_cov])

        cov_sum = np.einsum(
            "n...i,n...j->...ij", r_flat, r_flat, optimize=True
        )
        cov = cov_sum * (Nconf - 1) / Nconf
        result["data_cov"] = cov.reshape(
            tuple(shape_other) + tuple(shape_cov) + tuple(shape_cov)
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Effective mass
# ═══════════════════════════════════════════════════════════════════════════════


def meff(
    data_sample: np.ndarray,
    alttc: float,
    Nconf_axes: int = 0,
    Nt_axes: int = 1,
    meff_type: Literal["log", "cosh"] = "log",
) -> dict:
    """Compute the effective mass from a 2pt correlator.

    Effective mass definitions:
      log:  m_eff(t) = ln(C(t) / C(t+1)) × (1/a)
      cosh: m_eff(t) = arccosh((C(t+2) + C(t)) / (2 C(t+1))) × (1/a)

    The log method is appropriate for large t where the correlator
    is dominated by the ground state: C(t) ~ A e^{-E t}.
    The cosh method accounts for backward-propagating states and
    is preferred for smaller lattices.

    Args:
        data_sample: Jackknife/Bootstrap samples of the correlator.
            Shape: (Nconf, Nt, ...) where Nconf_axes=0, Nt_axes=1.
        alttc: Lattice spacing in fm.
        Nconf_axes: Configuration axis index.
        Nt_axes: Time axis index.
        meff_type: 'log' or 'cosh'.

    Returns:
        dict with keys:
          'data_sample': effective mass jackknife samples
          'data_mean': mean effective mass vs t
          'data_err': error on effective mass vs t
    """
    if np.iscomplexobj(data_sample):
        # Take real part for effective mass
        data_sample = np.real(data_sample)

    Nconf = data_sample.shape[Nconf_axes]
    Nt = data_sample.shape[Nt_axes]
    a_inv_GeV = fm2GeV / alttc  # lattice spacing → GeV conversion

    ndim = data_sample.ndim
    meff_sample = np.zeros_like(data_sample)

    with np.errstate(divide="ignore", invalid="ignore"):
        if meff_type == "log":
            # m_eff(t) = ln(C(t)/C(t+1)) / a
            for t in range(Nt - 1):
                idx_now = [slice(None)] * ndim
                idx_now[Nt_axes] = t
                idx_next = [slice(None)] * ndim
                idx_next[Nt_axes] = t + 1

                ratio = data_sample[tuple(idx_now)] / data_sample[tuple(idx_next)]
                with np.errstate(invalid="ignore"):
                    meff_sample[tuple(idx_now)] = np.log(np.abs(ratio)) * a_inv_GeV

        elif meff_type == "cosh":
            # m_eff(t) = arccosh((C(t+2) + C(t)) / (2 C(t+1))) / a
            for t in range(Nt - 2):
                idx_t = [slice(None)] * ndim
                idx_t[Nt_axes] = t
                idx_t1 = [slice(None)] * ndim
                idx_t1[Nt_axes] = t + 1
                idx_t2 = [slice(None)] * ndim
                idx_t2[Nt_axes] = t + 2

                num = data_sample[tuple(idx_t2)] + data_sample[tuple(idx_t)]
                den = 2 * data_sample[tuple(idx_t1)]
                ratio = num / den
                # Clamp to [1, inf) for arccosh
                ratio = np.maximum(ratio, 1.0)
                meff_sample[tuple(idx_t)] = np.arccosh(ratio) * a_inv_GeV

    # Compute mean and error
    meff_mean = np.mean(meff_sample, axis=Nconf_axes)
    meff_err = (
        np.std(meff_sample, axis=Nconf_axes) * np.sqrt(Nconf - 1)
    )

    return {
        "data_sample": meff_sample,
        "data_mean": meff_mean,
        "data_err": meff_err,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Ratio 3pt/2pt
# ═══════════════════════════════════════════════════════════════════════════════


def ratio_3pt(
    data_3pt: np.ndarray,
    data_2pt: np.ndarray,
    t_sep: int = 12,
    Nconf_axes: int = 0,
    Nt_axes: int = -1,
) -> dict:
    """Compute the ratio R(τ) = C₃(τ) / C₂(t_sep).

    This ratio isolates the matrix element of interest by canceling
    the source/sink overlap factors.

    For the disconnected gluon PDF: R(z, Pz, τ) isolates the
    quasi-PDF matrix element h(Pz, z).

    Args:
        data_3pt: 3pt correlator jackknife samples.
            Shape: (Nconf, Nt, ...) with Nt time slices.
        data_2pt: 2pt correlator jackknife samples.
            Shape: (Nconf, Nt, ...).
        t_sep: Source-sink separation.
        Nconf_axes: Configuration axis index.
        Nt_axes: Time axis index for the ratio.

    Returns:
        dict with keys:
          'data_sample': ratio jackknife samples
          'data_mean': mean ratio vs time
          'data_err': error on ratio vs time
    """
    ndim_3pt = data_3pt.ndim
    ndim_2pt = data_2pt.ndim

    Nt_axes_3pt = Nt_axes % ndim_3pt
    Nt_axes_2pt = Nt_axes % ndim_2pt

    Nconf_3pt = data_3pt.shape[Nconf_axes]
    Nconf_2pt = data_2pt.shape[Nconf_axes]
    assert Nconf_3pt == Nconf_2pt, "Nconf mismatch between 3pt and 2pt"

    # Extract C₂(t_sep): fixed-time value from 2pt
    idx_2pt = [slice(None)] * ndim_2pt
    idx_2pt[Nt_axes_2pt] = t_sep
    C2_tsep = data_2pt[tuple(idx_2pt)]

    # Ratio: R(τ) = C₃(τ) / C₂(t_sep)
    # Need to broadcast C2_tsep over the τ axis
    expand_shape = list(data_3pt.shape)
    ratio_sample = data_3pt / np.expand_dims(
        C2_tsep, axis=Nt_axes_3pt
    )

    ratio_mean = np.mean(np.real(ratio_sample), axis=Nconf_axes)
    ratio_err = (
        np.std(np.real(ratio_sample), axis=Nconf_axes)
        * np.sqrt(Nconf_3pt - 1)
    )

    return {
        "data_sample": ratio_sample,
        "data_mean": ratio_mean,
        "data_err": ratio_err,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Plateau fitting
# ═══════════════════════════════════════════════════════════════════════════════


def fit_plateau(
    t_vals: np.ndarray,
    meff_vals: np.ndarray,
    meff_errs: np.ndarray,
    t_min: int,
    t_max: int,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Fit a constant to the effective mass plateau region.

    Args:
        t_vals: Time slice values.
        meff_vals: Effective mass values (mean).
        meff_errs: Effective mass errors.
        t_min, t_max: Fit range [t_min, t_max].
        logger: Optional logger.

    Returns:
        dict with keys:
          'E0': fitted ground-state energy (GeV)
          'E0_err': error on E0
          'chi2_dof': χ² per degree of freedom
          'fit_range': [t_min, t_max]
    """
    mask = (t_vals >= t_min) & (t_vals <= t_max)
    t_fit = t_vals[mask]
    y_fit = meff_vals[mask]
    dy_fit = meff_errs[mask]

    if len(t_fit) < 2:
        if logger:
            logger.warning(f"Insufficient points for fit: {len(t_fit)}")
        return {"E0": np.nan, "E0_err": np.nan, "chi2_dof": np.nan}

    # Weighted average
    w = 1.0 / (dy_fit**2 + 1e-15)
    E0 = np.sum(w * y_fit) / np.sum(w)
    E0_err = np.sqrt(1.0 / np.sum(w))

    # χ²
    chi2 = np.sum(((y_fit - E0) / dy_fit) ** 2)
    dof = len(t_fit) - 1
    chi2_dof = chi2 / dof if dof > 0 else np.nan

    if logger:
        logger.info(
            f"  Plateau fit [{t_min},{t_max}]: "
            f"E0 = {E0:.4f} ± {E0_err:.4f} GeV, "
            f"χ²/dof = {chi2_dof:.2f}"
        )

    return {
        "E0": E0,
        "E0_err": E0_err,
        "chi2_dof": chi2_dof,
        "fit_range": [t_min, t_max],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════


def plot_correlator(
    corr_looped: np.ndarray,
    alttc: float,
    title: str,
    output_path: str,
    logger: Optional[logging.Logger] = None,
    log_scale: bool = True,
) -> None:
    """Plot a time-averaged 2pt correlator.

    Args:
        corr_looped: 1D array of correlator values C(dt).
        alttc: Lattice spacing in fm.
        title: Plot title.
        output_path: Output file path (.png).
        logger: Optional logger.
        log_scale: If True, use log y-axis.
    """
    if not HAS_MPL:
        if logger:
            logger.warning("matplotlib not available, skipping plot")
        return

    Nt = len(corr_looped)
    t_vals = np.arange(Nt) * alttc

    fig, ax = plt.subplots(figsize=(10, 6))

    # Take real part
    y = np.real(corr_looped)
    ax.plot(t_vals, np.abs(y), "o-", markersize=4, linewidth=1.5,
            color="#3498DB", label="|C(dt)|")

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("t [fm]", fontsize=12)
    ax.set_ylabel("C(dt)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    if logger:
        logger.info(f"  Plot saved: {output_path}")


def plot_meff(
    meff_result: dict,
    alttc: float,
    title: str,
    output_path: str,
    E_expected: Optional[float] = None,
    fit_result: Optional[Dict] = None,
    logger: Optional[logging.Logger] = None,
    t_max_plot: Optional[int] = None,
) -> None:
    """Plot effective mass with error bars, expected value, and plateau fit.

    Args:
        meff_result: dict from meff() with 'data_mean' and 'data_err'.
        alttc: Lattice spacing in fm.
        title: Plot title.
        output_path: Output file path (.png).
        E_expected: Expected ground-state energy (GeV) for reference line.
        fit_result: Optional plateau fit result from fit_plateau().
        logger: Optional logger.
        t_max_plot: Maximum time slice to plot.
    """
    if not HAS_MPL:
        if logger:
            logger.warning("matplotlib not available, skipping plot")
        return

    meff_mean = np.real(meff_result["data_mean"]).ravel()
    meff_err = np.real(meff_result["data_err"]).ravel()
    Nt = len(meff_mean)
    t_vals = np.arange(Nt) * alttc

    if t_max_plot is not None:
        meff_mean = meff_mean[:t_max_plot]
        meff_err = meff_err[:t_max_plot]
        t_vals = t_vals[:t_max_plot]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Effective mass with error bars
    ax.errorbar(
        t_vals, meff_mean, yerr=meff_err,
        fmt="o-", markersize=4, linewidth=1.5,
        color="#3498DB", capsize=2, label="m_eff(t)",
    )

    # Expected energy
    if E_expected is not None:
        ax.axhline(
            y=E_expected, color="#E74C3C", linestyle="--",
            linewidth=1.5, label=f"Expected: {E_expected:.3f} GeV",
        )

    # Plateau fit
    if fit_result is not None and not np.isnan(fit_result.get("E0", np.nan)):
        t_min, t_max = fit_result["fit_range"]
        t_fit_vals = np.array([t_min * alttc, t_max * alttc])
        ax.fill_between(
            t_fit_vals, fit_result["E0"] - fit_result["E0_err"],
            fit_result["E0"] + fit_result["E0_err"],
            alpha=0.3, color="#2ECC71",
            label=f"Fit: {fit_result['E0']:.3f}±{fit_result['E0_err']:.3f} GeV",
        )

    ax.set_xlabel("t [fm]", fontsize=12)
    ax.set_ylabel("m_eff [GeV]", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Auto-set y-range to exclude outliers
    valid = np.isfinite(meff_mean) & (meff_err > 0) & (meff_err < 10)
    if np.any(valid):
        y_min = max(0, np.min(meff_mean[valid]) - 0.5)
        y_max = np.max(meff_mean[valid]) + 1.0
        ax.set_ylim(y_min, min(y_max, 5.0))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    if logger:
        logger.info(f"  Plot saved: {output_path}")


def plot_ratio(
    ratio_result: dict,
    alttc: float,
    title: str,
    output_path: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot the 3pt/2pt ratio R(τ).

    Args:
        ratio_result: dict from ratio_3pt() with 'data_mean' and 'data_err'.
        alttc: Lattice spacing in fm.
        title: Plot title.
        output_path: Output file path (.png).
        logger: Optional logger.
    """
    if not HAS_MPL:
        if logger:
            logger.warning("matplotlib not available, skipping plot")
        return

    ratio_mean = np.real(ratio_result["data_mean"]).ravel()
    ratio_err = np.real(ratio_result["data_err"]).ravel()
    Nt = len(ratio_mean)
    t_vals = np.arange(Nt) * alttc

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        t_vals, ratio_mean, yerr=ratio_err,
        fmt="o-", markersize=4, linewidth=1.5,
        color="#2ECC71", capsize=2, label="R(τ) = C₃/C₂",
    )

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlabel("τ [fm]", fontsize=12)
    ax.set_ylabel("R(τ)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    if logger:
        logger.info(f"  Plot saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-config analysis
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_multi_config(
    corr_list: List[np.ndarray],
    alttc: float,
    meff_type: str = "log",
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Perform jackknife analysis over multiple configurations.

    Args:
        corr_list: List of (Nt,) correlator arrays, one per config.
        alttc: Lattice spacing in fm.
        meff_type: 'log' or 'cosh' effective mass method.
        logger: Optional logger.

    Returns:
        dict with keys:
          'corr_jk': jackknife result for correlator
          'meff_jk': jackknife result for effective mass
          'Nconf': number of configurations
    """
    Nconf = len(corr_list)
    if Nconf < 2:
        if logger:
            logger.warning(f"Only {Nconf} configs — computing meff without jackknife errors")
        # Single config: compute meff on raw data
        corr_arr = np.asarray(corr_list[0]).ravel()
        # Pad to (1, Nt) for meff compatibility
        corr_stacked = corr_arr.reshape(1, -1)
        # Direct effective mass (no jackknife)
        corr_mean = corr_arr
        corr_err = np.zeros_like(corr_arr)

        # Compute meff on raw data
        meff_data = meff(
            corr_stacked, alttc, Nconf_axes=0, Nt_axes=1,
            meff_type=meff_type,
        )

        if logger:
            logger.info(
                f"  Corr (t=0): {np.real(corr_mean[0]):.6e}"
            )
            valid = np.isfinite(meff_data["data_mean"])
            if np.any(valid):
                logger.info(
                    f"  Meff range: [{np.min(meff_data['data_mean'][valid]):.3f}, "
                    f"{np.max(meff_data['data_mean'][valid]):.3f}] GeV"
                )

        return {
            "corr_jk": {
                "data_sample": corr_stacked,
                "data_mean": corr_mean,
                "data_err": corr_err,
            },
            "meff_jk": meff_data,
            "Nconf": Nconf,
        }

    # Stack configs along axis 0 → (Nconf, Nt)
    corr_stacked = np.stack([np.asarray(c).ravel() for c in corr_list], axis=0)
    assert corr_stacked.ndim == 2, f"Expected (Nconf, Nt), got {corr_stacked.shape}"

    if logger:
        logger.info(
            f"Multi-config analysis: Nconf={Nconf}, "
            f"shape={corr_stacked.shape}"
        )

    # Jackknife
    corr_jk = Jackknife(corr_stacked, Nconf_axes=0)

    # Effective mass
    meff_jk = meff(
        corr_jk["data_sample"],
        alttc,
        Nconf_axes=0,
        Nt_axes=1,
        meff_type=meff_type,
    )

    if logger:
        logger.info(
            f"  Corr mean (t=0): {np.real(corr_jk['data_mean'][0]):.6e} "
            f"± {np.real(corr_jk['data_err'][0]):.6e}"
        )
        valid_meff = np.isfinite(meff_jk["data_mean"])
        if np.any(valid_meff):
            logger.info(
                f"  Meff range: [{np.min(meff_jk['data_mean'][valid_meff]):.3f}, "
                f"{np.max(meff_jk['data_mean'][valid_meff]):.3f}] GeV"
            )

    return {
        "corr_jk": corr_jk,
        "meff_jk": meff_jk,
        "Nconf": Nconf,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Expected values for validation
# ═══════════════════════════════════════════════════════════════════════════════

def expected_energy(
    particle: str,
    momentum: List[int],
    Nx: int,
    alttc: float,
    M0: float = 0.938,  # proton mass in GeV
) -> float:
    """Compute the expected energy for validation.

    E(P) = √(m₀² + |p|²) where |p| = 2π|P|/(Nx·a)

    Reference values from the user spec:
      proton E(P=0) ≈ 1.0 GeV (with ~60 MeV discretization offset)
      proton E(Pz=-2) ≈ 1.4 GeV (from √(m₀² + p²))

    Args:
        particle: 'proton' or 'pion'.
        momentum: 3-vector [Pz, Py, Px] in units of 2π/L.
        Nx: Spatial lattice extent.
        alttc: Lattice spacing in fm.
        M0: Rest mass in GeV.

    Returns:
        Expected energy in GeV.
    """
    if particle == "pion":
        M0 = 0.140  # pion mass ~140 MeV (approximate)

    p_phys = 2 * np.pi / (Nx * alttc) * fm2GeV  # momentum unit in GeV
    p_sq = sum(m**2 for m in momentum) * p_phys**2
    E = np.sqrt(M0**2 + p_sq)

    return E


# ═══════════════════════════════════════════════════════════════════════════════
# Momentum to GeV conversion
# ═══════════════════════════════════════════════════════════════════════════════

def Mom2GeV(
    Nx: int, alttc: float, Mom, M0: Union[float, List[float]] = 0.938
) -> float:
    """Convert lattice momentum to physical energy in GeV.

    E = Σ_i √( (2π/Nx · 1/a · ℏc)² · |P_i|² + M0_i² )

    Args:
        Nx: Spatial lattice extent.
        alttc: Lattice spacing in fm.
        Mom: Momentum 3-vector [Pz, Py, Px] or squared momentum.
        M0: Mass in GeV (float or list for multi-particle states).

    Returns:
        Energy in GeV.
    """
    single_Q2 = 2 * np.pi / Nx * (fm2GeV / alttc)

    if isinstance(Mom, (int, float)):
        mom_sq = Mom
    elif isinstance(Mom, list):
        if not Mom:
            mom_sq = 0.0
        elif isinstance(Mom[0], (int, float)):
            mom_sq = sum(x**2 for x in Mom)
        elif isinstance(Mom[0], list):
            mom_sq = [sum(x**2 for x in sub) for sub in Mom]
    else:
        raise TypeError(f"Unsupported Mom type: {type(Mom)}")

    if isinstance(M0, (int, float)):
        if isinstance(mom_sq, list):
            return [(single_Q2**2 * msq + M0**2) ** 0.5 for msq in mom_sq]
        return (single_Q2**2 * mom_sq + M0**2) ** 0.5
    elif isinstance(M0, list):
        if isinstance(mom_sq, list):
            result = []
            for msq in mom_sq:
                total = sum(
                    (single_Q2**2 * msq + m**2) ** 0.5 for m in M0
                )
                result.append(total)
            return result
        else:
            return sum(
                (single_Q2**2 * mom_sq + m**2) ** 0.5 for m in M0
            )

    raise TypeError(f"Unsupported M0 type: {type(M0)}")
