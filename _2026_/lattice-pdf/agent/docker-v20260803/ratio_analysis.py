#!/usr/bin/env python3
"""
3pt/2pt Ratio Analysis — All Configs, All Momenta
==================================================

Computes C₃(τ) via dynamic contraction and the ratio
  R(τ) = C₃(τ)/C₂(t_sep) × √[C₂^I(t_sep-τ)·C₂^F(τ)·C₂^F(t_sep)
                                  / (C₂^F(t_sep-τ)·C₂^I(τ)·C₂^I(t_sep))]
using Jackknife errors across Nconf=3 configurations.

Channels: proton P=0, proton P=(0,0,2), pion P=0, pion P=(0,0,2)
"""

import os, sys, time, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils import setup_logging, Timer, log_gpu_memory, save_array
from lib.constants import fm2GeV as _fm2GeV

T_SEP_3PT = 8


def compute_3pt(logger, conf_ids, vertex_data, t_sep):
    import cupy as cp
    from lib.backend import set_backend, get_backend
    from lib.dynamic import PeramRegistry, VRegistry, GammaRegistry, dynamic_contraction, clear_plan_cache
    from lib.gamma_matrix import gamma
    from lib.io_readers import readin_peram_time_slice
    from lib.seqperam import seq_peram

    set_backend('cupy'); backend = get_backend()
    Ntau = t_sep + 1

    projector = backend.asarray((gamma(0)+gamma(4))/2.0, dtype=backend.complex64)
    gmu = backend.asarray([gamma(1),gamma(2),gamma(3),gamma(4)], dtype=backend.complex64)

    logger.info("="*60)
    logger.info(f"3pt correlators: t_sep={t_sep}, Nconf={len(conf_ids)}")
    logger.info("="*60)

    all_3pt = {}
    for conf_id in conf_ids:
        logger.info(f"\n--- Config {conf_id} ---")
        peram_dir = get_peram_dir(conf_id)
        VdV = vertex_data[conf_id]['VdV']; VVV = vertex_data[conf_id]['VVV']

        # Accumulators: tau=0..t_sep, gamma_mu=0..3
        proton_P0 = np.zeros((Ntau,4), dtype=np.complex64)
        proton_P2 = np.zeros((Ntau,4), dtype=np.complex64)
        pion_P0   = np.zeros((Ntau,4), dtype=np.complex64)
        pion_P2   = np.zeros((Ntau,4), dtype=np.complex64)

        t0 = time.perf_counter()
        for t_src in range(NT):
            t_sink = (t_src + t_sep) % NT

            # Perams at source and sink (reuse across tau loop)
            p_src  = backend.asarray(readin_peram_time_slice(peram_dir, str(conf_id), t_src, NT, NEV), dtype=backend.complex64)
            p_snk  = backend.asarray(readin_peram_time_slice(peram_dir, str(conf_id), t_sink, NT, NEV), dtype=backend.complex64)
            p_srcS = seq_peram(p_src); p_snkS = seq_peram(p_snk)

            for tau in range(Ntau):
                t_cur = (t_src + tau) % NT
                p_cur  = backend.asarray(readin_peram_time_slice(peram_dir, str(conf_id), t_cur, NT, NEV), dtype=backend.complex64)
                p_curS = seq_peram(p_cur)

                # ── Peram registry (covers all 3pt Wick output needs) ──
                PR = PeramRegistry()
                PR.register('light', ('tsink','tsrc'),  backend.asarray(p_src[t_sink], dtype=backend.complex64))
                PR.register('light', ('tcur0','tsrc'),  backend.asarray(p_src[t_cur], dtype=backend.complex64))
                PR.register('light', ('tsrc','tsrc'),   backend.asarray(p_src[t_src], dtype=backend.complex64))
                PR.register('light', ('tsink','tcur0'), backend.asarray(p_cur[t_sink], dtype=backend.complex64))
                PR.register('light', ('tcur0','tcur0'), backend.asarray(p_cur[t_cur], dtype=backend.complex64))
                PR.register('light', ('tsrc','tcur0'),  backend.asarray(p_cur[t_src], dtype=backend.complex64))
                PR.register('light', ('tcur0','tsink'), backend.asarray(p_snk[t_cur], dtype=backend.complex64))
                PR.register('light', ('tsink','tsink'), backend.asarray(p_snk[t_sink], dtype=backend.complex64))
                PR.register('light', ('tsrc','tsink'),  backend.asarray(p_srcS[t_sink], dtype=backend.complex64))
                PR.register('light', ('tsrc','tcur0'),  backend.asarray(p_srcS[t_cur], dtype=backend.complex64))
                PR.register('light', ('tcur0','tsink'), backend.asarray(p_curS[t_sink], dtype=backend.complex64))
                PR.register('light', ('tsink','tcur0'), backend.asarray(p_snkS[t_cur], dtype=backend.complex64))

                GRp = GammaRegistry()  # proton gammas
                GRp.register('gamma_7', backend.asarray(gamma(7), dtype=backend.complex64))
                GRp.register('gamma_mu', gmu)
                GRp.register('Projector', (projector, projector))

                GRpi = GammaRegistry()  # pion gammas
                GRpi.register('gamma_5', backend.asarray(gamma(5), dtype=backend.complex64))
                GRpi.register('gamma_mu', gmu)
                GRpi.register('Projector', (projector, projector))

                cpx = backend.complex64
                # ── Proton P=0 ──
                VR = VRegistry()
                VR.register('VVV_0','tsrc',  backend.asarray(VVV[t_src,0:1].conj(), dtype=cpx))
                VR.register('VDV_0','tcur0', backend.asarray(VdV[t_cur,0:1], dtype=cpx))
                VR.register('VVV_0','tsink', backend.asarray(VVV[t_sink,0:1], dtype=cpx))
                r = dynamic_contraction(
                    [(PROTON_SINK_3PT, PROTON_SRC_3PT, CURR_3PT)],
                    peram_registry=PR,v_registry=VR,gamma_registry=GRp,
                    Cpt='3pt',Vindex=['M','M','M'],Gindex=['','G',''],
                    use_equivalence=False,ignore_dis=False,Projection=True,
                    verbose=False).calculate_all()
                v = r.get() if hasattr(r,'get') else r
                vn = np.asarray(v).ravel()
                proton_P0[tau,:min(4,len(vn))] += np.real(vn[:min(4,len(vn))])/NT

                # ── Proton P=(0,0,2) ──
                VR2 = VRegistry()
                VR2.register('VVV_0','tsrc',  backend.asarray(VVV[t_src,1:2].conj(), dtype=cpx))
                VR2.register('VDV_0','tcur0', backend.asarray(VdV[t_cur,1:2], dtype=cpx))
                VR2.register('VVV_0','tsink', backend.asarray(VVV[t_sink,1:2], dtype=cpx))
                r2 = dynamic_contraction(
                    [(PROTON_SINK_3PT, PROTON_SRC_3PT, CURR_3PT)],
                    peram_registry=PR,v_registry=VR2,gamma_registry=GRp,
                    Cpt='3pt',Vindex=['M','M','M'],Gindex=['','G',''],
                    use_equivalence=False,ignore_dis=False,Projection=True,
                    verbose=False).calculate_all()
                v2 = r2.get() if hasattr(r2,'get') else r2
                v2n = np.asarray(v2).ravel()
                proton_P2[tau,:min(4,len(v2n))] += np.real(v2n[:min(4,len(v2n))])/NT

                # ── Pion P=0 ──
                VRp = VRegistry()
                VRp.register('VDV_0','tsrc',  backend.asarray(VdV[t_src,0:1].conj(), dtype=cpx))
                VRp.register('VDV_0','tcur0', backend.asarray(VdV[t_cur,0:1], dtype=cpx))
                VRp.register('VDV_0','tsink', backend.asarray(VdV[t_sink,0:1], dtype=cpx))
                rp = dynamic_contraction(
                    [(PION_SINK, PION_SRC, CURR_3PT_U)],
                    peram_registry=PR,v_registry=VRp,gamma_registry=GRpi,
                    Cpt='3pt',Vindex=['M','M','M'],Gindex=['','G',''],
                    use_equivalence=False,ignore_dis=False,Projection=True,
                    verbose=False).calculate_all()
                vp = rp.get() if hasattr(rp,'get') else rp
                vpn = np.asarray(vp).ravel()
                pion_P0[tau,:min(4,len(vpn))] += np.real(vpn[:min(4,len(vpn))])/NT

                # ── Pion P=(0,0,2) ──
                VRp2 = VRegistry()
                VRp2.register('VDV_0','tsrc',  backend.asarray(VdV[t_src,1:2].conj(), dtype=cpx))
                VRp2.register('VDV_0','tcur0', backend.asarray(VdV[t_cur,1:2], dtype=cpx))
                VRp2.register('VDV_0','tsink', backend.asarray(VdV[t_sink,1:2], dtype=cpx))
                rp2 = dynamic_contraction(
                    [(PION_SINK, PION_SRC, CURR_3PT_U)],
                    peram_registry=PR,v_registry=VRp2,gamma_registry=GRpi,
                    Cpt='3pt',Vindex=['M','M','M'],Gindex=['','G',''],
                    use_equivalence=False,ignore_dis=False,Projection=True,
                    verbose=False).calculate_all()
                vp2 = rp2.get() if hasattr(rp2,'get') else rp2
                vp2n = np.asarray(vp2).ravel()
                pion_P2[tau,:min(4,len(vp2n))] += np.real(vp2n[:min(4,len(vp2n))])/NT

            if t_src % 10 == 0:
                e = time.perf_counter()-t0
                logger.info(f"  t_src={t_src:3d}/{NT}  {e:.0f}s  "
                           f"pP0[0,3]={proton_P0[0,3].real:.3e}  "
                           f"piP0[0,3]={pion_P0[0,3].real:.3e}")

        conf_dir = os.path.join(DATA_DIR, str(conf_id))
        os.makedirs(conf_dir, exist_ok=True)
        for n, a in [('proton_P0_3pt',proton_P0),('proton_P2_3pt',proton_P2),
                     ('pion_P0_3pt',pion_P0),('pion_P2_3pt',pion_P2)]:
            save_array(os.path.join(conf_dir, f'{n}_{conf_id}.npy'), a, logger)

        all_3pt[conf_id] = {'proton_P0_3pt':proton_P0,'proton_P2_3pt':proton_P2,
                            'pion_P0_3pt':pion_P0,'pion_P2_3pt':pion_P2}
        logger.info(f"  C₃(0,γ₃): protP0={proton_P0[0,3].real:.4e}  "
                   f"pionP0={pion_P0[0,3].real:.4e}")
    return all_3pt


def compute_ratios(logger, conf_ids, corr_2pt, corr_3pt, t_sep):
    from lib.analyse import Jackknife, ratio_3pt

    Nconf = len(conf_ids)
    logger.info(f"\n{'='*60}")
    logger.info(f"3pt/2pt Ratio: t_sep={t_sep}, Nconf={Nconf}")
    logger.info("="*60)

    analysis_dir = os.path.join(DATA_DIR, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    ratio_results = {}

    channels = [
        ('proton','P0','pp_P0','proton_P0_3pt'),
        ('proton','P2','pp_P2','proton_P2_3pt'),
        ('pion','P0','pi_P0','pion_P0_3pt'),
        ('pion','P2','pi_P2','pion_P2_3pt'),
    ]

    for had, mom, k2, k3 in channels:
        logger.info(f"\n--- {had} P={mom} ---")

        # Stack 3pt: gamma_mu=3 (z-direction) component
        s3 = np.stack([corr_3pt[c][k3][:,3].real for c in conf_ids])
        s2 = np.stack([corr_2pt[c][k2].real for c in conf_ids])
        logger.info(f"  C₃ shape={s3.shape}  C₂ shape={s2.shape}")
        logger.info(f"  C₃(0,γ₃) = {np.array2string(s3[:,0], precision=5)}")
        logger.info(f"  C₂({t_sep}) = {np.array2string(s2[:,t_sep], precision=5)}")

        jk3 = Jackknife(s3, Nconf_axes=0)
        jk2 = Jackknife(s2, Nconf_axes=0)

        try:
            ratio = ratio_3pt(jk3['data_sample'], jk2['data_sample'],
                             data_2ptF_sample=None, t_sep=t_sep,
                             Nconf_axes=0, tau_axes=1, t_sink_axes=1)
            rm, re = ratio['data_mean'], ratio['data_err']

            for t in range(min(len(rm), t_sep+1)):
                logger.info(f"  R({t:2d}) = {rm[t]:+.6f} ± {re[t]:.6f}")

            save_array(os.path.join(analysis_dir, f'ratio_{had}_{mom}_mean.npy'), rm, logger)
            save_array(os.path.join(analysis_dir, f'ratio_{had}_{mom}_err.npy'), re, logger)
            ratio_results[f'{had}_{mom}'] = ratio
        except Exception as e:
            logger.error(f"  Ratio failed: {e}")
            traceback.print_exc()

    return ratio_results


def generate_plots(logger, ratio_results):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, (ch, title, color) in enumerate([
        ('proton_P0','Proton P=(0,0,0)','#3498DB'),
        ('proton_P2','Proton P=(0,0,2)','#2980B9'),
        ('pion_P0','Pion P=(0,0,0)','#2ECC71'),
        ('pion_P2','Pion P=(0,0,2)','#27AE60')]):
        ax = axes[i//2, i%2]
        if ch in ratio_results:
            r = ratio_results[ch]; tau = np.arange(len(r['data_mean']))
            ax.errorbar(tau, r['data_mean'], yerr=r['data_err'],
                       fmt='o-', color=color, markersize=5, capsize=3, lw=1.5)
            ax.axhline(y=0, color='gray', ls='-', alpha=.3)
            ax.axhline(y=1, color='black', ls=':', alpha=.3)
        ax.set_xlabel('τ / a'); ax.set_ylabel('R(τ)')
        ax.set_title(f'{title} — 3pt/2pt Ratio', fontweight='bold')
        ax.grid(True, alpha=.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'ratio_3pt_all_channels.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    logger.info(f"  Saved: {path}")


def main():
    import argparse
    p = argparse.ArgumentParser(description='3pt/2pt Ratio Analysis')
    p.add_argument('--conf-ids', type=int, nargs='+', default=CONF_IDS)
    p.add_argument('--t-sep', type=int, default=T_SEP_3PT)
    p.add_argument('--skip-3pt', action='store_true')
    p.add_argument('--skip-plots', action='store_true')
    args = p.parse_args()
    t_sep_val = args.t_sep

    t0 = time.perf_counter()
    logger = setup_logging(LOGS_DIR, 'ratio_3pt')
    logger.info(f"3pt/2pt Ratio: t_sep={t_sep_val}, Nconf={len(args.conf_ids)}")

    # Load vertex & 2pt data
    vd = {}; c2 = {}
    for cid in args.conf_ids:
        d = os.path.join(DATA_DIR, str(cid))
        vd[cid] = {'VdV': np.load(f'{d}/VdV_mom_{cid}.npy'),
                   'VVV': np.load(f'{d}/VVV_mom_{cid}.npy')}
        c2[cid] = {k: np.load(f'{d}/corr_{k}_{cid}.npy')
                   for k in ['pp_P0','pp_P2','pi_P0','pi_P2'] if os.path.exists(f'{d}/corr_{k}_{cid}.npy')}
        logger.info(f"  Config {cid}: VdV{vd[cid]['VdV'].shape} 2pt keys={list(c2[cid].keys())}")

    # 3pt
    if not args.skip_3pt:
        with Timer("3pt correlators", logger):
            c3 = compute_3pt(logger, args.conf_ids, vd, t_sep_val)
    else:
        c3 = {}
        for cid in args.conf_ids:
            d = os.path.join(DATA_DIR, str(cid))
            c3[cid] = {k: np.load(f'{d}/{k}_{cid}.npy')
                       for k in ['proton_P0_3pt','proton_P2_3pt','pion_P0_3pt','pion_P2_3pt']
                       if os.path.exists(f'{d}/{k}_{cid}.npy')}
            logger.info(f"  Loaded 3pt config {cid}: {list(c3[cid].keys())}")

    # Ratio
    with Timer("Ratio analysis", logger):
        rr = compute_ratios(logger, args.conf_ids, c2, c3, t_sep_val)

    # Plots
    if not args.skip_plots:
        generate_plots(logger, rr)

    logger.info(f"\nDone: {time.perf_counter()-t0:.0f}s")


if __name__ == '__main__':
    main()
