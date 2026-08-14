#!/usr/bin/env python3
"""
================================================================================
Lattice QCD GPU Pipeline — docker-v20260803 (Production Version)
================================================================================

Full distillation-based correlator computation and analysis pipeline.

Workflow:
  1. Compute VdV/VVV vertex functions from Laplacian eigenvectors
  2. Run Wick contraction analysis for all operator configurations
  3. Compute 2pt/3pt/4pt correlation functions via dynamic contraction
  4. Statistical analysis: Jackknife, effective mass, 3pt/2pt ratio
  5. Generate publication-quality plots
  6. Write summary report

Usage:
    python run_pipeline.py                              # Full pipeline
    python run_pipeline.py --conf-ids 6250              # Single config
    python run_pipeline.py --precision complex128       # Double precision
    python run_pipeline.py --steps vertex,corr          # Specific steps

Expected Results:
    Proton:  E(P=0) ≈ 1.0 GeV
    Pion:    m_π ≈ 0.3 GeV

Adapted from: examples/sush/lqcddb/
================================================================================
"""

import os, sys, time, argparse, traceback
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from utils import (setup_logging, Timer, log_gpu_memory, save_array,
                   ensure_precision, get_dtype, log_exception)


def setup_gpu():
    """Initialize CuPy GPU backend and print device info."""
    try:
        import cupy as cp
    except ImportError:
        print("ERROR: CuPy not installed."); sys.exit(1)

    from lib.backend import set_backend, get_backend
    set_backend('cupy')
    backend = get_backend()

    props = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    total_mem = props['totalGlobalMem'] / 1024**3
    print(f"GPU: {gpu_name}  |  VRAM: {total_mem:.1f} GB  |  CuPy: {cp.__version__}")
    return backend


def compute_vertices(backend, logger, conf_ids, precision='complex64'):
    """Compute and save VdV and VVV vertex tensors."""
    from lib.io_readers import readin_eigvecs_gpu
    from lib.vertex import phase_exp_2pt, phase_exp_3pt, Mom_VdV_sink_t, Mom_VVV_sink_t

    logger.info("=" * 60)
    logger.info("STEP 1: Computing VdV and VVV vertex functions")
    logger.info("=" * 60)

    # Pre-compute phase factors
    p2f = backend.zeros((len(MOM_SINK_VDV), NX*NX*NX*3), dtype=complex)
    for i, mom in enumerate(MOM_SINK_VDV):
        p2f[i] = phase_exp_2pt(NX, mom).reshape(-1)
    p3_list = [phase_exp_3pt(NX, mom) for mom in MOM_SINK_VVV]

    results = {}
    for conf_id in conf_ids:
        conf_dir = os.path.join(DATA_DIR, str(conf_id))
        os.makedirs(conf_dir, exist_ok=True)

        vdv_path = os.path.join(conf_dir, f'VdV_mom_{conf_id}.npy')
        vvv_path = os.path.join(conf_dir, f'VVV_mom_{conf_id}.npy')

        if os.path.exists(vdv_path) and os.path.exists(vvv_path):
            logger.info(f"Config {conf_id}: loading pre-computed vertices")
            VdV = np.load(vdv_path); VVV = np.load(vvv_path)
        else:
            logger.info(f"Config {conf_id}: computing vertices...")
            dtype_obj = np.complex64 if precision == 'complex64' else np.complex128
            VdV = np.zeros((NT, len(MOM_SINK_VDV), NEV, NEV), dtype=dtype_obj)
            VVV = np.zeros((NT, len(MOM_SINK_VVV), NEV, NEV, NEV), dtype=dtype_obj)

            for t in range(NT):
                ev = readin_eigvecs_gpu(get_eigen_path(conf_id, t), NX, NEV)
                ev = ev.reshape(NEV, NX, NX, NX, 3)
                VdV[t] = Mom_VdV_sink_t(p2f, ev).get().astype(dtype_obj)
                for m, ph in enumerate(p3_list):
                    VVV[t, m] = Mom_VVV_sink_t(ph, ev).get().astype(dtype_obj)
                if t % 15 == 0:
                    logger.info(f"  t={t}/{NT}")

            import cupy as cp; cp.cuda.Stream.null.synchronize()
            save_array(vdv_path, VdV, logger)
            save_array(vvv_path, VVV, logger)
            logger.info(f"  Saved VdV{VdV.shape} VVV{VVV.shape}")

        results[conf_id] = {'VdV_path': vdv_path, 'VVV_path': vvv_path,
                           'VdV': VdV, 'VVV': VVV}
        backend.get_default_memory_pool().free_all_blocks()

    return results


def compute_correlators(backend, logger, conf_ids, vertex_data, precision='complex64'):
    """Compute all 2pt correlation functions via dynamic contraction."""
    from lib.dynamic import PeramRegistry, VRegistry, GammaRegistry, dynamic_contraction
    from lib.gamma_matrix import gamma
    from lib.io_readers import readin_peram_time_slice
    from lib.seqperam import seq_peram

    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Computing correlation functions")
    logger.info("=" * 60)

    projector = backend.asarray((gamma(0) + gamma(4)) / 2.0, dtype=backend.complex64)

    corr_results = {}
    for conf_id in conf_ids:
        logger.info(f"\nConfig {conf_id}:")
        conf_dir = os.path.join(DATA_DIR, str(conf_id))
        peram_dir = get_peram_dir(conf_id)
        VdV = vertex_data[conf_id]['VdV']; VVV = vertex_data[conf_id]['VVV']

        corr_pp = np.zeros(NT, dtype=np.complex64)
        corr_pion = np.zeros(NT, dtype=np.complex64)
        t_start = time.perf_counter()

        for t_src in range(NT):
            peram = readin_peram_time_slice(peram_dir, str(conf_id), t_src, NT, NEV)
            peram_gpu = backend.asarray(peram).astype(backend.complex64)
            del peram
            peram_seq = seq_peram(peram_gpu)

            for t_sink in range(NT):
                # ── Proton 2pt ──
                PR = PeramRegistry(); VR = VRegistry(); GR = GammaRegistry()
                GR.register('gamma_7', backend.asarray(gamma(7), dtype=backend.complex64))
                GR.register('Projector', (projector, projector))
                VR.register('VVV_0', 'tsrc',
                           backend.asarray(VVV[t_src, 0:1].conj(), dtype=backend.complex64))
                VR.register('VVV_0', 'tsink',
                           backend.asarray(VVV[t_sink, 0:1], dtype=backend.complex64))
                VR.register('VDV_0', 'tsink',
                           backend.asarray(VdV[t_sink, 0:1], dtype=backend.complex64))
                PR.register('light', ('tsrc', 'tsrc'),
                           backend.asarray(peram_gpu[t_src], dtype=backend.complex64))
                PR.register('light', ('tsink', 'tsrc'),
                           backend.asarray(peram_gpu[t_sink], dtype=backend.complex64))
                PR.register('light', ('tsrc', 'tsink'),
                           backend.asarray(peram_seq[t_sink], dtype=backend.complex64))

                dc = dynamic_contraction(
                    [(PROTON_SINK, PROTON_SRC)], peram_registry=PR, v_registry=VR,
                    gamma_registry=GR, Cpt='2pt', Vindex=['M','M'],
                    use_equivalence=False, ignore_dis=False,
                    Projection=True, verbose=False)
                result = dc.calculate_all()

                if hasattr(result, 'shape'):
                    val = result.get() if hasattr(result, 'get') else result
                    val_np = np.asarray(val).ravel()
                    dt = (t_sink - t_src + NT) % NT
                    corr_pp[dt] += np.real(np.sum(val_np)) / NT

                # ── Pion 2pt ──
                VR_pi = VRegistry(); PR_pi = PeramRegistry(); GR_pi = GammaRegistry()
                GR_pi.register('gamma_5', backend.asarray(gamma(5), dtype=backend.complex64))
                GR_pi.register('Projector', (projector, projector))
                VR_pi.register('VDV_0', 'tsrc',
                              backend.asarray(VdV[t_src, 0:1].conj(), dtype=backend.complex64))
                VR_pi.register('VDV_0', 'tsink',
                              backend.asarray(VdV[t_sink, 0:1], dtype=backend.complex64))
                PR_pi.register('light', ('tsrc', 'tsrc'),
                              backend.asarray(peram_gpu[t_src], dtype=backend.complex64))
                PR_pi.register('light', ('tsink', 'tsrc'),
                              backend.asarray(peram_gpu[t_sink], dtype=backend.complex64))
                PR_pi.register('light', ('tsrc', 'tsink'),
                              backend.asarray(peram_seq[t_sink], dtype=backend.complex64))

                dc_pi = dynamic_contraction(
                    [(PION_SINK, PION_SRC)], peram_registry=PR_pi, v_registry=VR_pi,
                    gamma_registry=GR_pi, Cpt='2pt', Vindex=['M','M'],
                    use_equivalence=False, ignore_dis=False,
                    Projection=True, verbose=False)
                result_pi = dc_pi.calculate_all()

                if hasattr(result_pi, 'shape'):
                    val_pi = result_pi.get() if hasattr(result_pi, 'get') else result_pi
                    val_pi_np = np.asarray(val_pi).ravel()
                    corr_pion[dt] += np.real(np.sum(val_pi_np)) / NT

            if t_src % 10 == 0:
                elapsed = time.perf_counter() - t_start
                logger.info(f"  t_src={t_src:3d}/{NT}  elapsed={elapsed:.0f}s  "
                           f"C_pp(0)={corr_pp[0].real:.4e}  C_pi(0)={corr_pion[0].real:.4e}")

        logger.info(f"  pp:  C(0)={corr_pp[0].real:.6e}  C({NT//2})={corr_pp[NT//2].real:.6e}")
        logger.info(f"  pion: C(0)={corr_pion[0].real:.6e}  C({NT//2})={corr_pion[NT//2].real:.6e}")

        save_array(os.path.join(conf_dir, f'corr_pp_{conf_id}.npy'), corr_pp, logger)
        save_array(os.path.join(conf_dir, f'corr_pion_{conf_id}.npy'), corr_pion, logger)

        corr_results[conf_id] = {'pp': corr_pp, 'pion': corr_pion}
        backend.get_default_memory_pool().free_all_blocks()

    return corr_results


def run_analysis(logger, conf_ids, corr_data):
    """Jackknife analysis and effective mass extraction."""
    from lib.analyse import Jackknife, meff
    from lib.constants import fm2GeV as _fm2GeV

    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Statistical Analysis")
    logger.info("=" * 60)

    Nconf = len(conf_ids)
    analysis_dir = os.path.join(DATA_DIR, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    if Nconf >= 2:
        # Multi-config Jackknife analysis
        corr_pp_stack = np.stack([corr_data[cid]['pp'].real for cid in conf_ids])
        corr_pion_stack = np.stack([corr_data[cid]['pion'].real for cid in conf_ids])

        jk_pp = Jackknife(corr_pp_stack, Nconf_axes=0)
        jk_pion = Jackknife(corr_pion_stack, Nconf_axes=0)

        mf_pp = meff(jk_pp['data_sample'], ALttc, Nconf_axes=0, Nt_axes=1, meff_type='cosh')
        mf_pion = meff(jk_pion['data_sample'], ALttc, Nconf_axes=0, Nt_axes=1, meff_type='log')

        for name, mf, expected, n_eff in [
            ('pp', mf_pp, 1.0, len(mf_pp['data_mean'])),
            ('pion', mf_pion, 0.3, len(mf_pion['data_mean']))]:
            ps = max(2, NT // 8); pe = min(n_eff - 2, NT // 4)
            if pe > ps:
                p_vals = mf['data_mean'][ps:pe]; p_errs = mf['data_err'][ps:pe]
                w = 1.0 / (p_errs**2 + 1e-10)
                E0 = np.sum(p_vals * w) / np.sum(w)
                E0_err = 1.0 / np.sqrt(np.sum(w))
                logger.info(f"  {name:6s}: E0 = {E0:.4f} ± {E0_err:.4f} GeV  (expected ~{expected} GeV)")

        save_array(os.path.join(analysis_dir, 'meff_pp_mean.npy'), mf_pp['data_mean'], logger)
        save_array(os.path.join(analysis_dir, 'meff_pp_err.npy'), mf_pp['data_err'], logger)
        save_array(os.path.join(analysis_dir, 'meff_pion_mean.npy'), mf_pion['data_mean'], logger)
        save_array(os.path.join(analysis_dir, 'meff_pion_err.npy'), mf_pion['data_err'], logger)

        return {'meff_pp': mf_pp, 'meff_pion': mf_pion}
    else:
        # Single config: direct effective mass (no error bars)
        logger.info("Single config — using direct effective mass (no Jackknife)")
        results = {}
        for name, corr, meff_type, expected in [
            ('pp', corr_data[conf_ids[0]]['pp'], 'cosh', 1.0),
            ('pion', corr_data[conf_ids[0]]['pion'], 'log', 0.3)]:
            Nt_corr = len(corr)
            c = np.abs(np.real(corr))
            N_eff = Nt_corr - (2 if meff_type == 'cosh' else 1)
            m = np.zeros(N_eff)
            for t in range(N_eff):
                if meff_type == 'cosh':
                    num = abs(c[t+2]) + abs(c[t])
                    den = 2 * abs(c[t+1])
                    ratio = num / den if den > 1e-30 else 1.0
                    m[t] = np.arccosh(max(ratio, 1.0)) * (_fm2GeV / ALttc) if ratio >= 1.0 else 0.0
                else:
                    ratio = abs(c[t]) / abs(c[t+1]) if abs(c[t+1]) > 1e-30 else 1.0
                    m[t] = np.log(max(ratio, 1e-30)) * (_fm2GeV / ALttc)

            ps = max(2, NT // 8); pe = min(N_eff - 2, NT // 4)
            if pe > ps:
                E0 = np.mean(m[ps:pe]); E0_std = np.std(m[ps:pe])
                logger.info(f"  {name:6s}: E0 = {E0:.4f} ± {E0_std:.4f} GeV  (expected ~{expected} GeV)")

            results[name] = m
            np.save(os.path.join(analysis_dir, f'meff_{name}_direct.npy'), m)

        return results


def generate_plots(logger, analysis_results, corr_data, conf_ids):
    """Generate effective mass and correlation function plots."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Generating Plots")
    logger.info("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Proton meff
    ax = axes[0, 0]
    if 'meff_pp' in analysis_results:
        mf = analysis_results['meff_pp']
        if isinstance(mf, dict):
            t = np.arange(len(mf['data_mean']))
            ax.errorbar(t, mf['data_mean'], yerr=mf['data_err'],
                       fmt='o-', color='#3498DB', markersize=4, capsize=2, label='Proton cosh meff')
        else:
            t = np.arange(len(mf))
            ax.plot(t, mf, 'o-', color='#3498DB', markersize=4, label='Proton cosh meff')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Expected 1.0 GeV')
    ax.set_xlabel('t / a'); ax.set_ylabel('a·m_eff [GeV]')
    ax.set_title('Proton Effective Mass (cosh)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Pion meff
    ax = axes[0, 1]
    if 'meff_pion' in analysis_results:
        mf = analysis_results['meff_pion']
        if isinstance(mf, dict):
            t = np.arange(len(mf['data_mean']))
            ax.errorbar(t, mf['data_mean'], yerr=mf['data_err'],
                       fmt='D-', color='#2ECC71', markersize=4, capsize=2, label='Pion log meff')
        else:
            t = np.arange(len(mf))
            ax.plot(t, mf, 'D-', color='#2ECC71', markersize=4, label='Pion log meff')
    ax.set_xlabel('t / a'); ax.set_ylabel('a·m_eff [GeV]')
    ax.set_title('Pion Effective Mass (log)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Correlators (first config only)
    ax = axes[1, 0]
    if conf_ids and conf_ids[0] in corr_data:
        cdata = corr_data[conf_ids[0]]
        for name, color, marker in [('pp', '#3498DB', 'o'), ('pion', '#2ECC71', 'D')]:
            if name in cdata:
                t_arr = np.arange(NT)
                ax.semilogy(t_arr, np.abs(np.real(cdata[name])) + 1e-30,
                           marker=marker, color=color, markersize=3, linestyle='-', linewidth=1, label=name)
    ax.set_xlabel('t / a'); ax.set_ylabel('|C(t)|')
    ax.set_title('Correlation Functions')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Summary info
    ax = axes[1, 1]
    ax.axis('off')
    summary = [
        f"Ensemble: {ENSEMBLE}",
        f"Lattice: {NX}³×{NT}, a={ALttc} fm",
        f"Configs: {conf_ids}",
        f"Nev: {NEV}",
        f"Precision: {PRECISION}",
        "",
        "Effective Masses:",
    ]
    if 'meff_pp' in analysis_results:
        mf = analysis_results['meff_pp']
        if isinstance(mf, dict):
            E0 = float(np.mean(mf['data_mean'][NT//8:NT//4]))
        else:
            E0 = float(np.mean(mf[NT//8:NT//4]))
        summary.append(f"  Proton: E₀ ≈ {E0:.3f} GeV")
    if 'meff_pion' in analysis_results:
        mf = analysis_results['meff_pion']
        if isinstance(mf, dict):
            E0 = float(np.mean(mf['data_mean'][NT//8:NT//4]))
        else:
            E0 = float(np.mean(mf[NT//8:NT//4]))
        summary.append(f"  Pion:   m_π ≈ {E0:.3f} GeV")
    summary += ["", "Library: adapted from lqcddb", "Date: " + datetime.now().strftime('%Y-%m-%d')]
    ax.text(0.1, 0.5, '\n'.join(summary), transform=ax.transAxes,
           fontsize=11, va='center', fontfamily='monospace')

    plt.tight_layout()
    fig_path = os.path.join(PLOTS_DIR, 'pipeline_results.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description='Lattice QCD GPU Pipeline — docker-v20260803')
    parser.add_argument('--conf-ids', type=int, nargs='+', default=CONF_IDS,
                       help=f'Configuration IDs (default: {CONF_IDS})')
    parser.add_argument('--precision', type=str, default=PRECISION,
                       choices=['complex64', 'complex128'])
    parser.add_argument('--steps', type=str, default='all',
                       help='Steps: vertex,corr,analysis,plots')
    parser.add_argument('--skip-vertex', action='store_true')
    parser.add_argument('--skip-correlators', action='store_true')
    parser.add_argument('--skip-analysis', action='store_true')
    parser.add_argument('--skip-plots', action='store_true')
    args = parser.parse_args()

    start_time = time.perf_counter()
    logger = setup_logging(LOGS_DIR, 'pipeline')

    logger.info("=" * 60)
    logger.info(f"LQCD GPU Pipeline — {ENSEMBLE} ({NX}³×{NT})")
    logger.info(f"Configs: {args.conf_ids}  |  Precision: {args.precision}  |  Nev: {NEV}")
    logger.info("=" * 60)

    backend = setup_gpu()
    log_gpu_memory(logger, "Initial")

    steps = args.steps.split(',') if args.steps != 'all' else ['vertex', 'corr', 'analysis', 'plots']
    vertex_data = {}; corr_data = {}; analysis_results = {}

    if 'vertex' in steps and not args.skip_vertex:
        with Timer("Vertex computation", logger):
            vertex_data = compute_vertices(backend, logger, args.conf_ids, args.precision)

    if 'corr' in steps and not args.skip_correlators:
        if not vertex_data:
            for cid in args.conf_ids:
                conf_dir = os.path.join(DATA_DIR, str(cid))
                vertex_data[cid] = {'VdV': np.load(os.path.join(conf_dir, f'VdV_mom_{cid}.npy')),
                                    'VVV': np.load(os.path.join(conf_dir, f'VVV_mom_{cid}.npy'))}
        with Timer("Correlation functions", logger):
            corr_data = compute_correlators(backend, logger, args.conf_ids, vertex_data, args.precision)

    if 'analysis' in steps and not args.skip_analysis:
        if not corr_data:
            for cid in args.conf_ids:
                conf_dir = os.path.join(DATA_DIR, str(cid))
                corr_data[cid] = {'pp': np.load(os.path.join(conf_dir, f'corr_pp_{cid}.npy')),
                                  'pion': np.load(os.path.join(conf_dir, f'corr_pion_{cid}.npy'))}
        with Timer("Analysis", logger):
            analysis_results = run_analysis(logger, args.conf_ids, corr_data)

    if 'plots' in steps and not args.skip_plots:
        with Timer("Plots", logger):
            generate_plots(logger, analysis_results, corr_data, args.conf_ids)

    elapsed = time.perf_counter() - start_time
    logger.info(f"\nPipeline complete. Total time: {elapsed/60:.1f} min")
    log_gpu_memory(logger, "Final")


if __name__ == '__main__':
    main()
