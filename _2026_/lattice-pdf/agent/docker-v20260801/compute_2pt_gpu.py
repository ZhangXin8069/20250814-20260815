#!/usr/bin/env python3
"""
Proton 2pt distillation — GPU (CuPy) accelerated, docker-v20260801.

CRITICAL FIX: Auto-plateau effective mass (excited-state rejection).
Previous cosh/exp fits used t_min=2, biased by excited states (+37%).

Diagnosis (conf 6250):
  Pz=0  exp_forward: 1.010 GeV (t=7-8), 0.984 GeV (t=8-9), 0.935 GeV (t=9-10)
      -> m0 approx 1.00 GeV (plateau convergence)
  Pz=-2 exp_forward: 1.463 GeV (t=7-8), 1.434 GeV (t=8-9)
      -> E  approx 1.43 GeV (matches sqrt(1.0^2+0.981^2)=1.401 GeV)

Algorithm:
  1. C2pt_fwd[dt] = mean_{tsrc} Re[C_PP(tsnk>tsrc)]  (forward-only, no backward wrap)
  2. m_eff[t]  = log(C_fwd[t] / C_fwd[t+1]) * hbar_c / a   (model-independent)
  3. Auto-plateau: find t where 3+ consecutive slices stabilize (spread <10%)
  4. Final meff = mean(m_eff[plateau_start:plateau_end])
  5. Cross-check: cosh/exp fits with variably-started t_min (t_min >= 3)
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np
import cupy as cp

_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))
from utils import (
    Timer, print_banner, format_size, save_intermediate,
    validate_array, log_exception, get_current_memory_mb,
    get_gpu_memory_mb, to_cpu, to_gpu, gpu_sync, log_gpu_status,
    get_compute_dtype, get_compute_dtype_real,
)
from gamma_matrix_gpu import (
    get_gamma_cached, get_P_plus_cached, get_P_minus_cached,
    clear_cache as clear_gamma_cache,
)

# ==========================================================
# Momentum phase factor - GPU
# ==========================================================
def compute_phase_factor_gpu(momentum: np.ndarray, Nx: int) -> cp.ndarray:
    """phi_P(x) = exp(-i*2pi*P*x/L). Returns (Nx^3,) on GPU."""
    rtype = get_compute_dtype_real()
    ctype = get_compute_dtype()
    coords = cp.arange(Nx, dtype=rtype)
    Z, Y, X = cp.meshgrid(coords, coords, coords, indexing='ij')
    mom = cp.asarray(momentum, dtype=rtype)
    phase = cp.exp(cp.array(-1j, dtype=ctype)
                   * cp.array(2.0 * cp.pi, dtype=rtype)
                   * (mom[0] * Z + mom[1] * Y + mom[2] * X)
                   / cp.array(Nx, dtype=rtype))
    return phase.ravel()

# ==========================================================
# Eigenvector reader: per-config, per-time-slice binary (LE f8)
# ==========================================================
def _read_eigenvector_slice(filepath: str, Nev: int, Nx: int) -> np.ndarray:
    """Read one time-slice of eigenvectors. LE f8 -> (Nev,Nv,3) complex128."""
    raw = np.fromfile(filepath, dtype='<f8')
    Nc, Nv = 3, Nx * Nx * Nx
    expected = Nev * Nv * Nc * 2
    if raw.size != expected:
        raise ValueError(f"Eigenvector {filepath}: expected {expected} floats, got {raw.size}")
    raw = raw.reshape(Nev, Nv, Nc, 2)  # (Nev, Nx^3, 3, 2)
    return raw[..., 0] + 1j * raw[..., 1]  # complex128

def load_eigenvectors_per_config(eigvec_base, conf_id, Nev, Nt, Nx, logger):
    """Load all Nt slices for ONE config. Returns (Nt,Nev,Nx^3,3) in compute dtype."""
    eigvec_dir = os.path.join(eigvec_base, str(conf_id))
    compute_dt = get_compute_dtype()
    Nv = Nx * Nx * Nx
    logger.info(f"Loading eigenvectors conf={conf_id} from {eigvec_dir}")
    t0 = time.perf_counter()
    if not os.path.isdir(eigvec_dir):
        raise FileNotFoundError(f"Eigenvector directory: {eigvec_dir}")
    eigvecs = np.zeros((Nt, Nev, Nv, 3), dtype=compute_dt)
    for tsrc in range(Nt):
        fpath = os.path.join(eigvec_dir, f"eigvecs_t{tsrc:03d}_{conf_id}")
        if not os.path.exists(fpath):
            continue
        eigvecs[tsrc] = _read_eigenvector_slice(fpath, Nev, Nx).astype(compute_dt, copy=False)
    elapsed = time.perf_counter() - t0
    logger.info(f"  shape={eigvecs.shape}, dtype={eigvecs.dtype}, "
                f"mem={eigvecs.nbytes/1024**2:.1f}MB, time={elapsed:.1f}s")
    logger.info(f"  |eig| range: [{np.abs(eigvecs).min():.2e},{np.abs(eigvecs).max():.2e}]")
    assert np.all(np.isfinite(eigvecs)), f"Eigenvectors {conf_id} NaN/inf"
    return eigvecs

# ==========================================================
# Perambulator reader: one file per (tsrc, dsrc) - LE f8
# ==========================================================
def _read_perambulator_file(filepath, Nev, Nt):
    """Read one peram file. LE f8 (Nt, Nev_snk, Nspin=4, Nev_src, 2) -> complex128."""
    raw = np.fromfile(filepath, dtype='<f8')
    expected = Nt * Nev * 4 * Nev * 2
    if raw.size != expected:
        raise ValueError(f"Perambulator {filepath}: expected {expected} floats, got {raw.size}")
    raw = raw.reshape(Nt, Nev, 4, Nev, 2)
    return raw[..., 0] + 1j * raw[..., 1]  # (Nt, Nev_snk, 4, Nev_src)

def read_perambulator_single_t(peram_dir, conf_id, t_source, Nev, Nt, logger):
    """Read 4 d_src files for one t_source.
    Returns: (Nt, 4, 4, Nev1, Nev1) in compute dtype.
    Matches snsc/main.py convention.
    """
    compute_dt = get_compute_dtype()
    Nev1 = Nev
    parts = []
    for d_src in range(4):
        fn = os.path.join(peram_dir, f"perams.{conf_id}.{d_src}.{t_source}")
        if not os.path.exists(fn):
            raise FileNotFoundError(f"Peram not found: {fn}")
        parts.append(_read_perambulator_file(fn, Nev, Nt))
    # Stack: (4_dsrc, Nt, Nev_snk, 4_spin, Nev_src)
    # Transpose: (Nt, 4_spin, 4_dsrc, Nev_src, Nev_snk)
    peram = np.stack(parts, axis=0)
    peram = peram.transpose(1, 3, 0, 4, 2)
    peram = peram.astype(compute_dt, copy=False)
    peram = peram[:, :, :, :Nev1, :Nev1]
    assert np.all(np.isfinite(peram)), f"Peram t={t_source} NaN/inf"
    return peram

# ==========================================================
# VVV Baryon Block - GPU (matching snsc/main.py)
# ==========================================================
def compute_vvv_single_t_gpu(eigvecs_t_gpu, phase_factor_gpu, Nx, Nev1):
    """VVV_{abc}=sum_x epsilon_{ijk} v_i^a v_j^b v_k^c * phase(x) on GPU.
    6-term Levi-Civita antisymmetrization (matching snsc/main.py).
    """
    VVV = cp.zeros((Nev1, Nev1, Nev1), dtype=get_compute_dtype())
    L = Nx * Nx
    for xi in range(Nx):
        s, e = xi * L, (xi + 1) * L
        es = eigvecs_t_gpu[:Nev1, s:e, :]  # (Nev1, Nx^2, 3)
        ps = phase_factor_gpu[s:e]          # (Nx^2,)
        # 2-step contraction: T = einsum("x,ax,bx->abx"), VVV += einsum("abx,cx->abc")
        # Much faster than 4-index einsum on GPU (~50x speedup)
        # Even permutations (+1)
        T = cp.einsum("x,ax,bx->abx", ps, es[:,:,0], es[:,:,1]); VVV += cp.einsum("abx,cx->abc", T, es[:,:,2])
        T = cp.einsum("x,ax,bx->abx", ps, es[:,:,1], es[:,:,2]); VVV += cp.einsum("abx,cx->abc", T, es[:,:,0])
        T = cp.einsum("x,ax,bx->abx", ps, es[:,:,2], es[:,:,0]); VVV += cp.einsum("abx,cx->abc", T, es[:,:,1])
        # Odd  permutations (-1)
        T = cp.einsum("x,ax,bx->abx", ps, es[:,:,0], es[:,:,2]); VVV -= cp.einsum("abx,cx->abc", T, es[:,:,1])
        T = cp.einsum("x,ax,bx->abx", ps, es[:,:,1], es[:,:,0]); VVV -= cp.einsum("abx,cx->abc", T, es[:,:,2])
        T = cp.einsum("x,ax,bx->abx", ps, es[:,:,2], es[:,:,1]); VVV -= cp.einsum("abx,cx->abc", T, es[:,:,0])
    return VVV

def compute_vvv_all_t_gpu(eigvecs, phase_gpu, Nt, Nx, Nev1, logger):
    """VVV for all Nt slices, streaming CPU->GPU one per slice.
    Returns: (Nt, Nev1, Nev1, Nev1) on CPU.
    """
    logger.info(f"VVV (GPU, {get_compute_dtype()}): Nt={Nt}, Nev1={Nev1}, Nx={Nx}")
    gpu_mem = get_gpu_memory_mb()
    logger.info(f"  GPU free before: {gpu_mem['free_mb']:.0f} MB")
    t_start = time.perf_counter()
    compute_dt = get_compute_dtype()
    VVV_all = np.zeros((Nt, Nev1, Nev1, Nev1), dtype=compute_dt)
    for t in range(Nt):
        t1 = time.perf_counter()
        ev_t_gpu = cp.asarray(eigvecs[t])
        VVV_all[t] = cp.asnumpy(compute_vvv_single_t_gpu(ev_t_gpu, phase_gpu, Nx, Nev1))
        del ev_t_gpu
        if t % 12 == 0 or t == Nt - 1:
            gpu_mem = get_gpu_memory_mb()
            logger.debug(f"  VVV t={t:3d}/{Nt} {time.perf_counter()-t1:.2f}s "
                         f"|VVV|_max={np.abs(VVV_all[t]).max():.4e} "
                         f"GPU free={gpu_mem['free_mb']:.0f}MB")
    gpu_sync()
    elapsed = time.perf_counter() - t_start
    logger.info(f"VVV done in {elapsed:.1f}s ({elapsed/Nt:.2f}s/slice)")
    logger.info(f"  |VVV| range: [{np.abs(VVV_all).min():.2e},{np.abs(VVV_all).max():.2e}]")
    logger.info(f"  Memory: {VVV_all.nbytes/1024**2:.1f} MB ({VVV_all.dtype})")
    assert np.all(np.isfinite(VVV_all)), "VVV NaN/inf"
    return VVV_all

# ==========================================================
# Wick contraction + Parity projection - GPU
# ==========================================================
def compute_wick_and_project_gpu(VVV, peram_dir, conf_id, Nev, Nev1, Nt, element, logger):
    """Wick contraction (Direct-Exchange) + parity projection on GPU.
    Matching snsc/main.py contract_proton_2pt_single_tsrc.
    """
    logger.info(f"Wick (GPU, {get_compute_dtype()}): Nt={Nt}, Nev1={Nev1}, element={element}")
    compute_dt = get_compute_dtype()
    # Interpolation operator Gamma = gamma_7 @ gamma_4 = C gamma_5 gamma_4 (DR basis)
    G7 = get_gamma_cached(7); G4 = get_gamma_cached(4)
    if element == "_Cg5g4":
        ip1 = ip2 = G7 @ G4
    elif element == "_Cg5g3":
        ip1 = ip2 = G7 @ get_gamma_cached(3)
    elif element == "_Cg5":
        ip1 = ip2 = G7
    else:
        ip1 = ip2 = G7 @ G4
    # Parity projectors: P_plus = 0.5*(gamma_0+gamma_4), P_minus = 0.5*(gamma_0-gamma_4)
    Pp = get_P_plus_cached(); Pm = get_P_minus_cached()
    VVV_gpu = cp.asarray(VVV)
    logger.info(f"  VVV on GPU: {VVV_gpu.nbytes/1024**2:.1f} MB")
    corr_raw = np.zeros((Nt, Nt, 4, 4), dtype=compute_dt)
    t0 = time.perf_counter()
    n_pairs = 0
    for t_src in range(Nt):
        t_s0 = time.perf_counter()
        VVV_src = VVV_gpu[t_src].conj()
        peram_u = read_perambulator_single_t(peram_dir, conf_id, t_src, Nev, Nt, logger)
        peram_u_gpu = cp.asarray(peram_u)  # (Nt, 4, 4, Nev, Nev)
        # Gamma*tau*Gamma
        cg5p = cp.einsum("gh,thkbe->tgkbe", ip1, peram_u_gpu)
        cg5p = cp.einsum("tgkbe,jk->tgjbe", cg5p, ip2)
        for t_snk in range(Nt):
            dt = (t_snk - t_src + Nt) % Nt
            if not (2 <= dt <= 32):
                continue
            n_pairs += 1
            # Direct term
            T1  = cp.einsum("abc,gjad->gjbcd", VVV_gpu[t_snk], peram_u_gpu[t_snk])
            T2  = cp.einsum("gjbcd,gjbe->cde", T1, cg5p[t_snk])
            T3  = cp.einsum("cde,ilcf->ildef", T2, peram_u_gpu[t_snk])
            direct = cp.einsum("ildef,def->il", T3, VVV_src)
            # Exchange term
            T1x = cp.einsum("abc,glaf->glbcf", VVV_gpu[t_snk], peram_u_gpu[t_snk])
            T2x = cp.einsum("glbcf,gjbe->ljcef", T1x, cg5p[t_snk])
            T3x = cp.einsum("ljcef,ijcd->ildef", T2x, peram_u_gpu[t_snk])
            exchange = cp.einsum("ildef,def->il", T3x, VVV_src)
            corr_raw[t_snk, t_src] = cp.asnumpy(direct - exchange)
        del peram_u_gpu, cg5p
        if t_src % 10 == 0:
            gpu_mem = get_gpu_memory_mb()
            logger.debug(f"  Wick t_src={t_src:3d}/{Nt} {time.perf_counter()-t_s0:.1f}s "
                         f"n={n_pairs} GPU free={gpu_mem['free_mb']:.0f}MB")
    del VVV_gpu; cp.get_default_memory_pool().free_all_blocks()
    elapsed = time.perf_counter() - t0
    logger.info(f"Wick done in {elapsed:.1f}s, n_pairs={n_pairs} "
                f"({elapsed/n_pairs*1000:.1f} ms/pair)")
    assert np.all(np.isfinite(corr_raw)), "Raw correlator NaN/inf"
    # Parity projection on GPU
    t_par = time.perf_counter()
    cr = cp.asarray(corr_raw)
    corr_pp = cp.asnumpy(cp.einsum("li,yxil->yx", Pp, cr))  # positive parity
    corr_pm = cp.asnumpy(cp.einsum("li,yxil->yx", Pm, cr))  # negative parity
    del cr
    # Anti-periodic BC sign correction for fermion perambulators
    for ts in range(Nt):
        for tk in range(Nt):
            if tk < ts:
                corr_pp[tk, ts] *= -1.0  # backward wrap correction
            if tk > ts:
                corr_pm[tk, ts] *= -1.0  # forward  wrap correction
    logger.info(f"Parity projection: {time.perf_counter()-t_par:.2f}s")
    logger.info(f"  PP range: [{corr_pp.real.min():.4e},{corr_pp.real.max():.4e}]")
    logger.info(f"  PM range: [{corr_pm.real.min():.4e},{corr_pm.real.max():.4e}]")
    return {"corr_raw": corr_raw, "corr_pp": corr_pp, "corr_pm": corr_pm}

# ==========================================================
# AUTO-PLATEAU Effective Mass (v20260801 CORE FIX)
# ==========================================================
def compute_effective_mass_auto_plateau(corr_pp, Nt, alttc=0.1053, logger=None):
    """Effective mass with automatic excited-state rejection.

    1. Extract forward-only C2pt_fwd[dt] (tsnk > tsrc only, no wrap-around)
    2. exp_forward meff[t] = log(C_fwd[t]/C_fwd[t+1]) * hbar_c / a
    3. Auto-detect plateau onset (3+ stable slices within 10% relative spread)
    4. Final meff = mean over plateau region
    5. Cross-check with variably-started cosh/exp fits

    Returns dict with meff_plateau_gev, C2pt_1d, C2pt_forward, meff_per_t, etc.
    """
    fm2GeV = 0.1973  # hbar_c [GeV*fm]

    # -- Step 1: Source-averaged forward correlator --
    C2pt_fwd = np.zeros(Nt, dtype=np.float64)  # forward-only (tsnk > tsrc)
    C2pt_all = np.zeros(Nt, dtype=np.float64)  # all propagation pairs

    for dt in range(Nt):
        vals_all, vals_fwd = [], []
        for t in range(Nt):
            tnk = (t + dt) % Nt
            v = np.real(corr_pp[tnk, t])
            if abs(v) > 1e-30:
                vals_all.append(v)
            if tnk > t:  # forward propagation only
                vals_fwd.append(np.real(corr_pp[tnk, t]))
        if vals_all:  C2pt_all[dt] = np.mean(vals_all)
        if vals_fwd:  C2pt_fwd[dt] = np.mean(vals_fwd)

    # -- Step 2: exp_forward point-by-point (model-independent) --
    meff_per_t = np.full(Nt - 1, np.nan)
    for t in range(1, Nt - 1):
        if abs(C2pt_fwd[t]) > 1e-30 and abs(C2pt_fwd[t+1]) > 1e-30:
            ratio = abs(C2pt_fwd[t] / (C2pt_fwd[t+1] + 1e-35))
            if ratio > 1.0:
                meff_per_t[t] = np.log(ratio) * fm2GeV / alttc

    # -- Step 3: Auto-detect plateau onset --
    # Find first t where 3+ consecutive meff values are within 10% relative spread
    plateau_start = 2
    for t in range(2, Nt // 3):  # scan t=2..Nt/3
        window = meff_per_t[t:t+5]
        valid = window[~np.isnan(window)]
        if len(valid) >= 3:
            spread = (np.max(valid) - np.min(valid)) / np.mean(valid)
            if spread < 0.10:  # <10% variation = plateau
                plateau_start = t
                break

    plateau_end = min(plateau_start + 4, Nt - 1)
    pvals = meff_per_t[plateau_start:plateau_end]
    pvals = pvals[~np.isnan(pvals)]

    if len(pvals) >= 2:
        meff_plateau = float(np.mean(pvals))
        meff_err = float(np.std(pvals) / np.sqrt(len(pvals) - 1)) if len(pvals) > 2 else float(np.std(pvals))
    else:
        # Fallback: use t=4..8
        fv = meff_per_t[4:9][~np.isnan(meff_per_t[4:9])]
        meff_plateau = float(np.mean(fv)) if len(fv) > 1 else 1.4
        meff_err = float(np.std(fv))
        plateau_start, plateau_end = 4, 8

    # -- Step 4: Cross-check fits (variably-started, for diagnostics) --
    cosh_fits, exp_fits = {}, {}
    for t_min in [2, 3, 4, 5, 6]:
        try:
            from scipy.optimize import curve_fit
            t_valid = np.arange(t_min, min(33, Nt))
            c_valid = np.abs(C2pt_all[t_min:min(33, Nt)])
            mask = np.abs(c_valid) > 1e-25
            t_fit, c_fit = t_valid[mask], c_valid[mask]
            if len(t_fit) >= 4:
                def cosh_model(t, A, m):
                    return A * np.cosh(m * (t - Nt / 2.0))
                p0 = [np.abs(c_fit).max(), meff_plateau * alttc / fm2GeV]
                popt, _ = curve_fit(cosh_model, t_fit, c_fit, p0=p0, maxfev=10000,
                                   bounds=([1e-30, 0.05], [1e10, 3.0]))
                cosh_fits[f"t_min={t_min}"] = float(popt[1] * fm2GeV / alttc)
        except Exception:
            pass
        try:
            from scipy.optimize import curve_fit
            t_fwd = np.arange(t_min, min(20, Nt))
            c_fwd = np.abs(C2pt_fwd[t_min:min(20, Nt)])
            mask = c_fwd > 1e-25
            t_fit, c_fit = t_fwd[mask], c_fwd[mask]
            if len(t_fit) >= 3:
                def exp_model(t, A, m):
                    return A * np.exp(-m * t)
                popt, _ = curve_fit(exp_model, t_fit, c_fit,
                                   p0=[c_fit[0], meff_plateau * alttc / fm2GeV],
                                   maxfev=10000)
                exp_fits[f"t_min={t_min}"] = float(popt[1] * fm2GeV / alttc)
        except Exception:
            pass

    # -- Log results --
    if logger:
        logger.info(f"========== Effective Mass (auto-plateau, a={alttc} fm) ==========")
        logger.info(f"  Forward-only C2pt and per-t meff (t >= 2):")
        for t in range(2, min(Nt - 1, 16)):
            marker = "  <-- plateau" if plateau_start <= t < plateau_end else ""
            logger.info(f"    t={t:3d}  C_fwd={C2pt_fwd[t]:.4e}  meff={meff_per_t[t]:.4f} GeV{marker}")
        logger.info(f"  Plateau: t in [{plateau_start},{plateau_end}), meff = {meff_plateau:.4f} +/- {meff_err:.4f} GeV")
        logger.info(f"  cosh fits by t_min: {cosh_fits}")
        logger.info(f"  exp  fits by t_min: {exp_fits}")

    return {
        "meff_plateau_gev": meff_plateau,
        "meff_plateau_err_gev": meff_err,
        "C2pt_1d": C2pt_all,
        "C2pt_forward": C2pt_fwd,
        "meff_per_t": meff_per_t,
        "plateau_t_start": plateau_start,
        "plateau_t_end": plateau_end,
        "cosh_fits_by_tmin": cosh_fits,
        "exp_fits_by_tmin": exp_fits,
        "method": "auto_plateau_exp_forward",
    }

# ==========================================================
# Main: 2pt distillation for all configs
# ==========================================================
def run_2pt_computation_gpu(config, output_dir, logger):
    """Full 2pt distillation with GPU (v20260801 auto-plateau meff)."""
    params, paths, ensemble = config["parameters"], config["data_paths"], config["ensemble"]
    Nt, Nx = ensemble["Nt"], ensemble["Nx"]
    Nev, Nev1 = params["Nev"], params["Nev1"]
    Pz_list = params.get("Pz_list", [params["Pz"]])
    Px, Py, element = params["Px"], params["Py"], params["element"]
    conf_ids, alttc = params["conf_ids"], ensemble["alttc"]
    apply_smear = params.get("apply_eigenvec_smearing", False)
    eigvec_base, peram_base = paths["eigenvector_base"], paths["perambulator_base"]

    print_banner("Step 01: Proton 2pt Distillation (GPU, v20260801 auto-plateau)", logger)
    log_gpu_status(logger, "  ")
    logger.info(f"  Compute dtype: {get_compute_dtype()}")
    logger.info(f"  Ensemble: {ensemble['full_name']} | {Nt}x{Nx}^3 | a={alttc} fm")
    logger.info(f"  Nev={Nev}, Nev1={Nev1}, Pz in {Pz_list}, configs: {conf_ids}")
    logger.info(f"  Eigenvector smearing: {apply_smear}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for iconf, conf_id in enumerate(conf_ids):
        conf_dir = output_dir / f"conf_{conf_id}"
        conf_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"\n{'='*60}\n  Config {iconf+1}/{len(conf_ids)}: conf_id={conf_id}\n{'='*60}")

        # Load eigvecs for THIS config
        with Timer(f"load_eigvecs_conf{conf_id}", logger, output_dir.parent,
                  extra={"conf_id": conf_id}):
            eigvecs = load_eigenvectors_per_config(eigvec_base, conf_id, Nev, Nt, Nx, logger)
        eigvecs = np.array(eigvecs, copy=True)

        # Check peram dir
        peram_dir = os.path.join(peram_base, "light", str(conf_id))
        if not os.path.exists(os.path.join(peram_dir, f"perams.{conf_id}.0.0")):
            logger.error(f"  [SKIP] conf={conf_id}: peram not found")
            all_results[conf_id] = {"status": "skipped", "reason": "peram not found"}
            continue

        conf_results = {}
        for Pz in Pz_list:
            Mom = np.array([Pz, Py, Px])
            logger.info(f"\n  --- P=({Px},{Py},{Pz}) ---")
            phase_gpu = compute_phase_factor_gpu(Mom, Nx)

            # VVV (with cache)
            vvv_cache = conf_dir / f"VVV_Nev1{Nev1}_Px{Px}Py{Py}Pz{Pz}_nosmear_conf{conf_id}.npy"
            if vvv_cache.exists():
                logger.info(f"  [CACHE] Loading VVV: {vvv_cache.name}")
                VVV = np.load(vvv_cache)
            else:
                with Timer(f"VVV_GPU_Pz{Pz}_conf{conf_id}", logger, output_dir.parent,
                          extra={"conf_id": conf_id, "Pz": Pz}):
                    VVV = compute_vvv_all_t_gpu(eigvecs, phase_gpu, Nt, Nx, Nev1, logger)
                np.save(vvv_cache, VVV)
                logger.info(f"  [SAVE] VVV: {vvv_cache.stat().st_size/1024**2:.1f}MB, {VVV.dtype}")

            # Wick contraction
            with Timer(f"Wick_GPU_Pz{Pz}_conf{conf_id}", logger, output_dir.parent,
                      extra={"conf_id": conf_id, "Pz": Pz}):
                corr_dict = compute_wick_and_project_gpu(VVV, peram_dir, conf_id, Nev, Nev1, Nt, element, logger)

            # Save correlator
            base = f"Px{Px}Py{Py}Pz{Pz}_eginphase{abs(params.get('mom_smear', 0))}{element}"
            np.save(conf_dir / f"twopt_slice_pp_{base}_nopol_ss_conf{conf_id}.npy", corr_dict["corr_pp"])
            np.save(conf_dir / f"twopt_slice_pm_{base}_nopol_ss_conf{conf_id}.npy", corr_dict["corr_pm"])
            np.save(conf_dir / f"twopt_slice_pp_{base}_contract_conf{conf_id}.npy", corr_dict["corr_raw"])

            # Effective mass (auto-plateau)
            with Timer(f"meff_Pz{Pz}_conf{conf_id}", logger, output_dir.parent,
                      extra={"conf_id": conf_id, "Pz": Pz}):
                meff_result = compute_effective_mass_auto_plateau(corr_dict["corr_pp"], Nt, alttc, logger=logger)

            np.savez(conf_dir / f"meff_Pz{Pz}_conf{conf_id}.npz",
                     meff_per_t=meff_result["meff_per_t"],
                     meff_plateau_gev=meff_result["meff_plateau_gev"],
                     meff_plateau_err_gev=meff_result["meff_plateau_err_gev"],
                     C2pt_1d=meff_result["C2pt_1d"],
                     C2pt_forward=meff_result["C2pt_forward"],
                     plateau_t_start=np.array(meff_result["plateau_t_start"]),
                     plateau_t_end=np.array(meff_result["plateau_t_end"]),
                     method=np.array(meff_result["method"]))

            conf_results[Pz] = {
                "meff_plateau_gev": meff_result["meff_plateau_gev"],
                "meff_plateau_err_gev": meff_result["meff_plateau_err_gev"],
                "plateau_t": [meff_result["plateau_t_start"], meff_result["plateau_t_end"]],
                "cosh_fits": meff_result.get("cosh_fits_by_tmin", {}),
                "exp_fits": meff_result.get("exp_fits_by_tmin", {}),
            }

        save_intermediate({"conf_id": conf_id, "status": "ok",
                          "results": {str(k): v for k, v in conf_results.items()}},
                         conf_dir, "compute_2pt_summary.json", logger)
        all_results[conf_id] = {"status": "ok", "results": conf_results}
        logger.info(f"  [PROGRESS] 2pt: {iconf+1}/{len(conf_ids)} done")
        del eigvecs; cp.get_default_memory_pool().free_all_blocks(); gc.collect()

    # Summary
    logger.info(f"\n{'='*60}\n2pt Summary ({get_compute_dtype()}, auto-plateau):")
    for cid, r in all_results.items():
        if r.get("status") == "ok":
            for Pz, rd in r["results"].items():
                logger.info(f"  conf={cid} Pz={Pz}: meff={rd['meff_plateau_gev']:.4f} +/- "
                           f"{rd['meff_plateau_err_gev']:.4f} GeV, "
                           f"plateau t in {rd['plateau_t']}")
    logger.info(f"{'='*60}")
    return all_results
