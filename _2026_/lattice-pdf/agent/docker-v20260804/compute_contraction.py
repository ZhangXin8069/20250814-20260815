#!/usr/bin/env python3
"""
Wick contraction — GPU-accelerated, factorized Direct-Exchange.

CORRECTED implementation following docker-v20260802/compute_2pt_gpu.py.

Perambulator shape: (Nt, 4, 4, Nev_src, Nev_snk)
  = (tsnk, spin_snk, spin_src, Nev_src, Nev_snk) — MATCHES sush einsum convention.

VVV shape: (Nt, N_mom, Nev1, Nev1, Nev1) — baryon block with momentum projection.

Key contraction chain (proton 2pt, _Cg5g4 operator):
  1. ip = γ₇ @ γ₄ = γ₃γ₁γ₄  (interpolation operator)
  2. cg5p = ip · peram · ip  (spin-space contraction)
  3. T1 = VVV_snk ⊗ peram_snk  (direct) or VVV_snk ⊗ peram_snk† (exchange)
  4. T2 = T1 ⊗ cg5p_snk
  5. T3 = T2 ⊗ peram_snk
  6. direct/exchange = T3 ⊗ VVV*_src
  7. corr_raw = Direct - Exchange
  8. Parity project: P₊·corr·P₊ → scalar correlator
"""
from __future__ import annotations
import gc, logging, os, time
from typing import Dict, List, Optional
import numpy as np
try: import cupy as cp; HAS_CUPY = True
except ImportError: HAS_CUPY = False; cp = np

def _xp(): return cp if HAS_CUPY else np
def _gpu(a): return cp.asarray(a) if HAS_CUPY else np.asarray(a)
def _cpu(a):
    if HAS_CUPY and isinstance(a, cp.ndarray): return cp.asnumpy(a)
    return np.asarray(a)
def _free():
    if HAS_CUPY: cp.get_default_memory_pool().free_all_blocks(); gc.collect()

from gamma_matrix_gpu import get_gamma_cached, get_P_plus_cached, get_P_minus_cached, clear_cache as clear_gamma_cache

# ═══════════════════════════════════════════════════════════════════════════════
# Interpolation operator
# ═══════════════════════════════════════════════════════════════════════════════

def _interp_operator(element: str = "_Cg5g4"):
    """Build interpolation operator on GPU.

    _Cg5g4: ip = γ₇ @ γ₄ = γ₃γ₁γ₄  (standard proton operator)
    _Cg5g3: ip = γ₇ @ γ₃
    _Cg5:   ip = γ₇
    """
    G7 = get_gamma_cached(7)  # γ₃γ₁
    if element == "_Cg5g4":
        return G7 @ get_gamma_cached(4)
    elif element == "_Cg5g3":
        return G7 @ get_gamma_cached(3)
    elif element == "_Cg5":
        return G7
    else:
        return G7 @ get_gamma_cached(4)  # default

# ═══════════════════════════════════════════════════════════════════════════════
# Proton 2pt — factorized Direct - Exchange
# ═══════════════════════════════════════════════════════════════════════════════

def compute_2pt_proton(peram_dir: str, conf_id: int, VVV: np.ndarray,
                       Nt: int, Nev: int, Nev1: int = None,
                       momentum_list: List[List[int]] = None,
                       element: str = "_Cg5g4", logger=None) -> Dict:
    """Proton 2pt <P(t_snk) P†(t_src)> — factorized Wick contraction.

    Peram loaded per t_src from disk (4 dsrc combined), used for all t_snk.
    VVV is pre-computed for the analysis momenta.

    Returns dict: corr_raw (N_mom,Nt,Nt,4,4), corr_pp (N_mom,Nt,Nt),
                  C2pt_1d (N_mom,Nt) — source-averaged 1D correlator.
    """
    if Nev1 is None: Nev1 = Nev
    if momentum_list is None: momentum_list = [[0,0,0]]
    N_mom = VVV.shape[1]
    # Use analysis momenta indices
    mom_indices = list(range(N_mom))
    if N_mom > len(momentum_list):
        mom_indices = list(range(len(momentum_list)))
        N_mom = len(momentum_list)

    xp = _xp()
    if logger: logger.info(f"Proton 2pt: Nt={Nt} Nev1={Nev1} N_mom={N_mom}")

    # Build interpolation operator on GPU
    ip = _interp_operator(element)  # (4,4) on GPU
    Pp = get_P_plus_cached()        # P₊ = (γ₀+γ₄)/2

    # Transfer VVV to GPU (all time slices, all momenta)
    VVV_g = _gpu(VVV[:, :N_mom, :Nev1, :Nev1, :Nev1].astype(np.complex64))
    t0 = time.perf_counter()

    # Raw correlator per momentum: (N_mom, Nt, Nt, 4, 4)
    corr_raw = np.zeros((N_mom, Nt, Nt, 4, 4), dtype=np.complex64)

    for t_src in range(Nt):
        # ── Load peram for this source time ─────────────────────────
        from data_io import read_perambulator_single_t
        peram = read_perambulator_single_t(peram_dir, conf_id, t_src, Nev, Nt, logger)
        # peram: (Nt, 4, 4, Nev_src, Nev_snk)
        peram = peram[:, :, :, :Nev1, :Nev1]  # truncate to Nev1
        peram_g = _gpu(peram.astype(np.complex64))  # → GPU

        # VVV* at source
        VVV_src = xp.conj(VVV_g[t_src])  # (N_mom, Nev1, Nev1, Nev1)

        # ── cg5p = ip · peram · ip ─────────────────────────────────
        # peram: (Nt, 4, 4, Nev_src, Nev_snk) = (t, h, k, b, e)
        # ip: (4,4) = (g,h), cg5p[t,g,k,b,e] = Σ_h ip[g,h]·peram[t,h,k,b,e]
        cg5p = xp.einsum('gh,thkbe->tgkbe', ip, peram_g, optimize=True)
        # cg5p[t,g,j,b,e] = Σ_k cg5p[t,g,k,b,e]·ip[j,k]
        cg5p = xp.einsum('tgkbe,jk->tgjbe', cg5p, ip, optimize=True)
        # cg5p shape: (Nt, 4, 4, Nev_src, Nev_snk) = (t, g, j, b, e)

        for t_snk in range(Nt):
            dt = (t_snk - t_src + Nt) % Nt
            # Only relevant time separations (2 ≤ dt ≤ 32)
            if not (2 <= dt <= 32):
                continue

            VVV_snk = VVV_g[t_snk]  # (N_mom, Nev1, Nev1, Nev1)

            # Peram at sink time
            peram_s = peram_g[t_snk]  # (4, 4, Nev_src, Nev_snk) = (g, j, a, d)
            cg5p_s = cg5p[t_snk]     # (4, 4, Nev_src, Nev_snk) = (g, j, b, e)

            for m in range(N_mom):
                v_snk = VVV_snk[m]  # (Nev1, Nev1, Nev1) = (a,b,c)
                v_src = VVV_src[m]  # (Nev1, Nev1, Nev1) = (d,e,f)

                # Direct diagram
                T1 = xp.einsum('abc,gjad->gjbcd', v_snk, peram_s, optimize=True)  # VVV⊗peram
                T2 = xp.einsum('gjbcd,gjbe->cde', T1, cg5p_s, optimize=True)      # ⊗cg5p
                T3 = xp.einsum('cde,ilcf->ildef', T2, peram_s, optimize=True)     # ⊗peram
                direct = xp.einsum('ildef,def->il', T3, v_src, optimize=True)      # ⊗VVV*

                # Exchange diagram
                T1x = xp.einsum('abc,glaf->glbcf', v_snk, peram_s, optimize=True)
                T2x = xp.einsum('glbcf,gjbe->ljcef', T1x, cg5p_s, optimize=True)
                T3x = xp.einsum('ljcef,ijcd->ildef', T2x, peram_s, optimize=True)
                exchange = xp.einsum('ildef,def->il', T3x, v_src, optimize=True)

                corr_raw[m, t_snk, t_src] = _cpu(direct - exchange)

        del peram_g, cg5p
        if t_src % 10 == 0 and logger:
            logger.debug(f"  Wick t_src={t_src}/{Nt} ({time.perf_counter()-t0:.0f}s)")

    del VVV_g; _free(); clear_gamma_cache()

    # ── Source average → 1D correlator ───────────────────────────────
    C2pt = np.zeros((N_mom, Nt), dtype=np.float64)
    corr_pp = np.zeros((N_mom, Nt, Nt), dtype=np.float64)

    for m in range(N_mom):
        # Parity project: Pp·corr·Pp → scalar
        cr_proj = np.einsum('li,yxil->yx', _cpu(Pp), corr_raw[m], optimize=True)
        # Anti-periodic BC: sign flip when tsnk < tsrc
        for ts in range(Nt):
            for tk in range(Nt):
                if tk < ts:
                    cr_proj[tk, ts] *= -1.0
        corr_pp[m] = np.real(cr_proj)
        # Source average
        for dt in range(Nt):
            vals = []
            for t in range(Nt):
                tsnk = (t + dt) % Nt
                v = corr_pp[m, tsnk, t]
                if abs(v) > 1e-30:
                    vals.append(v)
            C2pt[m, dt] = np.mean(vals) if vals else 0.0

    elapsed = time.perf_counter() - t0
    if logger:
        logger.info(f"  Proton 2pt done in {elapsed:.1f}s")
        for m in range(N_mom):
            logger.info(f"  Mom[{m}]: C(dt=2)={C2pt[m,2]:.6e} ... C(dt=15)={C2pt[m,15]:.6e}")

    return {'corr_raw': corr_raw, 'corr_pp': corr_pp, 'C2pt_1d': C2pt, 'N_mom': N_mom}

# ═══════════════════════════════════════════════════════════════════════════════
# Pion 2pt — Tr[γ₅τ · γ₅τ†]
# ═══════════════════════════════════════════════════════════════════════════════

def compute_2pt_pion(peram: np.ndarray, Nt: int, Nev: int, logger=None) -> Dict:
    """Pion 2pt <π(t_snk) π†(0)> — single t_src=0.

    C_π(tsnk) = Tr[γ₅ τ(tsnk;0) γ₅ τ†(tsnk;0)]
              = Σ_{k,l,α,β} (γ₅)_{αγ} τ_{γβ}^{kl}(tsnk) (γ₅)_{βδ} τ_{αδ}^{kl*}(tsnk)
              = Σ |γ₅ @ τ|²

    peram: (Nt, 4, 4, Nev, Nev) at fixed t_src.
    """
    xp = _xp()
    g5 = get_gamma_cached(5)  # γ₅
    peram_g = _gpu(peram.astype(np.complex64))
    g5_g = g5  # already on GPU if HAS_CUPY

    # γ₅τ: contract spin_snk with g5
    # peram: (Nt, 4, 4, Nev_src, Nev_snk) = (t, a, b, k, l)
    # g5τ[t, g, b, k, l] = Σ_a g5[g,a] * peram[t, a, b, k, l]
    g5tau = xp.einsum('ga,tabkl->tgbkl', g5_g, peram_g, optimize=True)
    # C_π(tsnk, tsrc) = -Σ |g5τ|²
    # For tsrc=0: C(tsnk) = -Σ_{t,b,k,l} g5τ[tsnk,g,b,k,l] * conj(g5τ[tsnk,g,b,k,l])
    corr = -xp.einsum('tgbkl,tgbkl->t', g5tau, xp.conj(g5tau), optimize=True).real

    corr_cpu = _cpu(corr)
    _free(); clear_gamma_cache()

    if logger:
        logger.info(f"  Pion 2pt: C(0)={corr_cpu[0]:.6e} C(36)={corr_cpu[Nt//2]:.6e}")
    return {'C2pt_1d': corr_cpu}  # (Nt,)

# ═══════════════════════════════════════════════════════════════════════════════
# Effective mass — cosh method with source averaging
# ═══════════════════════════════════════════════════════════════════════════════

def compute_effective_mass(C2pt: np.ndarray, Nt: int, alttc: float = 0.1053,
                           method: str = 'fit_cosh', logger=None) -> Dict:
    """Compute effective mass from source-averaged 1D correlator.

    Methods:
      cosh:     arccosh on source-averaged C(t)
      fit_cosh: Non-linear fit A·cosh(m·(t-Nt/2)) — most robust
      fit_exp:  Single exponential A·exp(-m·t)
    """
    fm2GeV = 0.1973
    if C2pt.ndim > 1:
        C = C2pt.ravel()  # ensure 1D
    else:
        C = C2pt
    C_abs = np.abs(C) + 1e-30

    meff = np.full(Nt-2, np.nan)

    if method == 'cosh':
        cosh_arg = (C_abs[2:] + C_abs[:-2]) / (2.0 * C_abs[1:-1])
        valid = cosh_arg >= 1.0
        meff[valid] = np.arccosh(np.minimum(cosh_arg[valid], 1e15)) * fm2GeV / alttc

    elif method == 'fit_cosh':
        from scipy.optimize import curve_fit
        t_valid = np.arange(2, min(33, Nt))
        c_valid = C_abs[2:min(33, Nt)]
        mask = c_valid > 1e-25
        t_fit, c_fit = t_valid[mask], c_valid[mask]
        if len(t_fit) >= 5:
            try:
                def cosh_model(t, A, m):
                    return A * np.cosh(m * (t - Nt/2.0))
                popt, _ = curve_fit(cosh_model, t_fit, c_fit, p0=[c_fit.max(), 0.5],
                                   maxfev=10000, bounds=([1e-30,0.01],[1e10,5.0]))
                m_phys = popt[1] * fm2GeV / alttc
                meff[:] = m_phys
                if logger: logger.info(f"fit_cosh: m_latt={popt[1]:.4f} m_phys={m_phys:.3f} GeV")
            except Exception as e:
                if logger: logger.warning(f"fit_cosh failed: {e}")
                return compute_effective_mass(C2pt, Nt, alttc, 'cosh', logger)

    elif method == 'fit_exp':
        from scipy.optimize import curve_fit
        t_fwd = np.arange(2, min(33, Nt))
        c_fwd = C_abs[2:min(33, Nt)]
        mask = c_fwd > 1e-25
        t_fit, c_fit = t_fwd[mask], c_fwd[mask]
        if len(t_fit) >= 4:
            try:
                def exp_model(t, A, m):
                    return A * np.exp(-m * t)
                popt, _ = curve_fit(exp_model, t_fit, c_fit, p0=[c_fit[0], 0.5], maxfev=10000)
                m_phys = popt[1] * fm2GeV / alttc
                meff[:] = m_phys
                if logger: logger.info(f"fit_exp: m_phys={m_phys:.3f} GeV")
            except Exception as e:
                if logger: logger.warning(f"fit_exp failed: {e}")
                return compute_effective_mass(C2pt, Nt, alttc, 'cosh', logger)

    # Plateau estimate
    ps, pe = Nt//4, Nt//2
    pmask = ~np.isnan(meff[ps:pe])
    plateau = float(np.mean(meff[ps:pe][pmask])) if np.any(pmask) else np.nan

    if method in ('fit_cosh', 'fit_exp') and not np.isnan(meff[0]):
        plateau = float(meff[0])

    if logger:
        logger.info(f"Effective mass ({method}): plateau={plateau:.4f} GeV [{ps},{pe}]")
        for t in range(1, min(16, Nt-1)):
            logger.info(f"  t={t:3d} m_eff={meff[t-1]:.6f} GeV C={C_abs[t]:.4e}")

    return {'meff': meff, 'C2pt': C_abs, 'plateau_GeV': plateau, 'method': method}
