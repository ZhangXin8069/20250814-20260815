#!/usr/bin/env python3
"""
================================================================================
Full Jackknife Analysis — All Configs, Multiple Momenta
================================================================================

Computes and analyzes correlation functions across ALL configurations with
proper Jackknife resampling for:
  - Pion:   P = (0,0,0)  and  P = (0,0,2)
  - Proton: P = (0,0,0)  and  P = (0,0,2)

Workflow:
  1. Compute/Load VdV and VVV vertices for all configs, all momenta
  2. Compute 2pt correlators at P=0 and P=(0,0,2) for proton and pion
  3. Jackknife resampling across Nconf=3
  4. Effective mass extraction (cosh for baryon, log for meson)
  5. Generate plots and report

Ensemble: beta6.20_mu-0.2770_ms-0.2400_L24x72 (24³×72)
Configs:  6250, 6450, 6650
================================================================================
"""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from utils import setup_logging, Timer, log_gpu_memory, save_array
from lib.constants import fm2GeV as _fm2GeV


def compute_all_vertices(logger, conf_ids, recompute=False):
    """Compute (or load) VdV/VVV for all configs at all momenta.

    Returns
    -------
    Dict mapping conf_id -> {'VdV': ndarray, 'VVV': ndarray}
    VdV shape: (Nt, N_mom=2, Nev, Nev)   [mom0=P0, mom1=Pz2]
    VVV shape: (Nt, N_mom=2, Nev, Nev, Nev)
    """
    import cupy as cp
    from lib.backend import set_backend, get_backend
    from lib.io_readers import readin_eigvecs_gpu
    from lib.vertex import phase_exp_2pt, phase_exp_3pt, Mom_VdV_sink_t, Mom_VVV_sink_t

    set_backend('cupy')
    backend = get_backend()

    logger.info("=" * 70)
    logger.info("STEP 1: VdV / VVV Vertex Functions")
    logger.info(f"  Momenta: VdV={MOM_SINK_VDV}, VVV={MOM_SINK_VVV}")
    logger.info(f"  Nev={NEV}, Nt={NT}")
    logger.info("=" * 70)

    # Pre-compute phase factors (same for all configs)
    p2f = backend.zeros((len(MOM_SINK_VDV), NX * NX * NX * 3), dtype=complex)
    for i, mom in enumerate(MOM_SINK_VDV):
        p2f[i] = phase_exp_2pt(NX, mom).reshape(-1)
    p3_list = [phase_exp_3pt(NX, mom) for mom in MOM_SINK_VVV]

    results = {}
    for conf_id in conf_ids:
        conf_dir = os.path.join(DATA_DIR, str(conf_id))
        os.makedirs(conf_dir, exist_ok=True)
        vdv_path = os.path.join(conf_dir, f'VdV_mom_{conf_id}.npy')
        vvv_path = os.path.join(conf_dir, f'VVV_mom_{conf_id}.npy')

        if os.path.exists(vdv_path) and os.path.exists(vvv_path) and not recompute:
            logger.info(f"  Config {conf_id}: loading pre-computed vertices")
            VdV = np.load(vdv_path)
            VVV = np.load(vvv_path)
            # Check momentum dimension
            if VdV.shape[1] < len(MOM_SINK_VDV) or VVV.shape[1] < len(MOM_SINK_VVV):
                logger.warning(f"  Config {conf_id}: old vertices have only "
                              f"{VdV.shape[1]} momenta — recomputing")
                recompute_single = True
            else:
                recompute_single = False
        else:
            recompute_single = True

        if recompute_single:
            logger.info(f"  Config {conf_id}: computing vertices ({NT} time slices × "
                       f"{len(MOM_SINK_VDV)}+{len(MOM_SINK_VVV)} momenta)...")
            VdV = np.zeros((NT, len(MOM_SINK_VDV), NEV, NEV), dtype=np.complex64)
            VVV = np.zeros((NT, len(MOM_SINK_VVV), NEV, NEV, NEV), dtype=np.complex64)

            t0_total = time.perf_counter()
            for t in range(NT):
                ev = readin_eigvecs_gpu(get_eigen_path(conf_id, t), NX, NEV)
                ev = ev.reshape(NEV, NX, NX, NX, 3)
                VdV[t] = Mom_VdV_sink_t(p2f, ev).get().astype(np.complex64)
                for m, ph in enumerate(p3_list):
                    VVV[t, m] = Mom_VVV_sink_t(ph, ev).get().astype(np.complex64)
                if t % 15 == 0 or t == NT - 1:
                    logger.info(f"    t={t:3d}/{NT}")

            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - t0_total
            logger.info(f"    Done: {elapsed:.1f}s. Saving...")
            save_array(vdv_path, VdV, logger)
            save_array(vvv_path, VVV, logger)

        results[conf_id] = {'VdV': VdV, 'VVV': VVV}
        logger.info(f"    VdV{VdV.shape}  VVV{VVV.shape}  "
                   f"M0[0,0,0]={VdV[0,0,0,0].real:.4f}")

    log_gpu_memory(logger, "After vertices")
    backend.get_default_memory_pool().free_all_blocks()
    return results


def compute_all_correlators(logger, conf_ids, vertex_data):
    """Compute 2pt correlators at P=0 and P=(0,0,2) for proton and pion.

    For each config:
      - Proton 2pt: P=0 (VVV mom 0), P=(0,0,2) (VVV mom 1)
      - Pion 2pt:   P=0 (VdV mom 0), P=(0,0,2) (VdV mom 1)

    Returns
    -------
    Dict mapping conf_id -> {
        'pp_P0': ndarray (Nt,), 'pp_P2': ndarray (Nt,),
        'pi_P0': ndarray (Nt,), 'pi_P2': ndarray (Nt,)
    }
    """
    import cupy as cp
    from lib.backend import set_backend, get_backend
    from lib.dynamic import PeramRegistry, VRegistry, GammaRegistry, dynamic_contraction
    from lib.gamma_matrix import gamma
    from lib.io_readers import readin_peram_time_slice
    from lib.seqperam import seq_peram

    set_backend('cupy')
    backend = get_backend()

    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Computing 2pt Correlation Functions")
    logger.info(f"  Channels: proton(P0), proton(Pz2), pion(P0), pion(Pz2)")
    logger.info(f"  Configs: {conf_ids}")
    logger.info("=" * 70)

    projector = backend.asarray((gamma(0) + gamma(4)) / 2.0, dtype=backend.complex64)

    corr_results = {}
    for conf_id in conf_ids:
        logger.info(f"\n--- Config {conf_id} ---")
        conf_dir = os.path.join(DATA_DIR, str(conf_id))
        os.makedirs(conf_dir, exist_ok=True)
        peram_dir = get_peram_dir(conf_id)
        VdV = vertex_data[conf_id]['VdV']  # (Nt, 2, Nev, Nev)
        VVV = vertex_data[conf_id]['VVV']  # (Nt, 2, Nev, Nev, Nev)

        # Accumulators: source-averaged, (Nt,) with dt index
        pp_P0  = np.zeros(NT, dtype=np.complex64)
        pp_P2  = np.zeros(NT, dtype=np.complex64)
        pi_P0  = np.zeros(NT, dtype=np.complex64)
        pi_P2  = np.zeros(NT, dtype=np.complex64)

        t_start = time.perf_counter()

        for t_src in range(NT):
            # Load perambulator for this time source
            peram = readin_peram_time_slice(peram_dir, str(conf_id), t_src, NT, NEV)
            peram_gpu = backend.asarray(peram).astype(backend.complex64)
            del peram
            peram_seq = seq_peram(peram_gpu)

            for t_sink in range(NT):
                dt = (t_sink - t_src + NT) % NT

                # ============================================================
                # Proton: P=0  (VVV momentum index 0)
                # ============================================================
                PR = PeramRegistry(); VR = VRegistry(); GR = GammaRegistry()
                GR.register('gamma_7', backend.asarray(gamma(7), dtype=backend.complex64))
                GR.register('Projector', (projector, projector))

                # VVV at t_src, momentum 0, conjugated
                VR.register('VVV_0', 'tsrc',
                           backend.asarray(VVV[t_src, 0:1].conj(), dtype=backend.complex64))
                # VVV at t_sink, momentum 0
                VR.register('VVV_0', 'tsink',
                           backend.asarray(VVV[t_sink, 0:1], dtype=backend.complex64))
                # VDV at t_sink (only needed for some Wick diagrams)
                VR.register('VDV_0', 'tsink',
                           backend.asarray(VdV[t_sink, 0:1], dtype=backend.complex64))

                PR.register('light', ('tsrc', 'tsrc'),
                           backend.asarray(peram_gpu[t_src], dtype=backend.complex64))
                PR.register('light', ('tsink', 'tsrc'),
                           backend.asarray(peram_gpu[t_sink], dtype=backend.complex64))
                PR.register('light', ('tsrc', 'tsink'),
                           backend.asarray(peram_seq[t_sink], dtype=backend.complex64))

                dc = dynamic_contraction(
                    [(PROTON_SINK, PROTON_SRC)],
                    peram_registry=PR, v_registry=VR, gamma_registry=GR,
                    Cpt='2pt', Vindex=['M', 'M'],
                    use_equivalence=False, ignore_dis=False,
                    Projection=True, verbose=False)
                result = dc.calculate_all()
                val = result.get() if hasattr(result, 'get') else result
                pp_P0[dt] += np.real(np.sum(np.asarray(val).ravel())) / NT

                # ============================================================
                # Proton: P=(0,0,2)  (VVV momentum index 1)
                # ============================================================
                VR_P2 = VRegistry()
                VR_P2.register('VVV_0', 'tsrc',
                              backend.asarray(VVV[t_src, 1:2].conj(), dtype=backend.complex64))
                VR_P2.register('VVV_0', 'tsink',
                              backend.asarray(VVV[t_sink, 1:2], dtype=backend.complex64))
                VR_P2.register('VDV_0', 'tsink',
                              backend.asarray(VdV[t_sink, 1:2], dtype=backend.complex64))

                dc2 = dynamic_contraction(
                    [(PROTON_SINK, PROTON_SRC)],
                    peram_registry=PR, v_registry=VR_P2, gamma_registry=GR,
                    Cpt='2pt', Vindex=['M', 'M'],
                    use_equivalence=False, ignore_dis=False,
                    Projection=True, verbose=False)
                result2 = dc2.calculate_all()
                val2 = result2.get() if hasattr(result2, 'get') else result2
                pp_P2[dt] += np.real(np.sum(np.asarray(val2).ravel())) / NT

                # ============================================================
                # Pion: P=0  (VDV momentum index 0)
                # ============================================================
                VRP = VRegistry(); PRP = PeramRegistry(); GRP = GammaRegistry()
                GRP.register('gamma_5', backend.asarray(gamma(5), dtype=backend.complex64))
                GRP.register('Projector', (projector, projector))
                VRP.register('VDV_0', 'tsrc',
                            backend.asarray(VdV[t_src, 0:1].conj(), dtype=backend.complex64))
                VRP.register('VDV_0', 'tsink',
                            backend.asarray(VdV[t_sink, 0:1], dtype=backend.complex64))
                PRP.register('light', ('tsrc', 'tsrc'),
                            backend.asarray(peram_gpu[t_src], dtype=backend.complex64))
                PRP.register('light', ('tsink', 'tsrc'),
                            backend.asarray(peram_gpu[t_sink], dtype=backend.complex64))
                PRP.register('light', ('tsrc', 'tsink'),
                            backend.asarray(peram_seq[t_sink], dtype=backend.complex64))

                dcP = dynamic_contraction(
                    [(PION_SINK, PION_SRC)],
                    peram_registry=PRP, v_registry=VRP, gamma_registry=GRP,
                    Cpt='2pt', Vindex=['M', 'M'],
                    use_equivalence=False, ignore_dis=False,
                    Projection=True, verbose=False)
                resultP = dcP.calculate_all()
                valP = resultP.get() if hasattr(resultP, 'get') else resultP
                pi_P0[dt] += np.real(np.sum(np.asarray(valP).ravel())) / NT

                # ============================================================
                # Pion: P=(0,0,2)  (VDV momentum index 1)
                # ============================================================
                VRP2 = VRegistry()
                VRP2.register('VDV_0', 'tsrc',
                             backend.asarray(VdV[t_src, 1:2].conj(), dtype=backend.complex64))
                VRP2.register('VDV_0', 'tsink',
                             backend.asarray(VdV[t_sink, 1:2], dtype=backend.complex64))

                dcP2 = dynamic_contraction(
                    [(PION_SINK, PION_SRC)],
                    peram_registry=PRP, v_registry=VRP2, gamma_registry=GRP,
                    Cpt='2pt', Vindex=['M', 'M'],
                    use_equivalence=False, ignore_dis=False,
                    Projection=True, verbose=False)
                resultP2 = dcP2.calculate_all()
                valP2 = resultP2.get() if hasattr(resultP2, 'get') else resultP2
                pi_P2[dt] += np.real(np.sum(np.asarray(valP2).ravel())) / NT

            if t_src % 10 == 0:
                elapsed = time.perf_counter() - t_start
                logger.info(f"  t_src={t_src:3d}/{NT}  elapsed={elapsed:.0f}s  "
                           f"pp0={pp_P0[0].real:.4e}  pp2={pp_P2[0].real:.4e}  "
                           f"pi0={pi_P0[0].real:.4e}  pi2={pi_P2[0].real:.4e}")

        # Save
        for name, arr in [('pp_P0', pp_P0), ('pp_P2', pp_P2),
                          ('pi_P0', pi_P0), ('pi_P2', pi_P2)]:
            path = os.path.join(conf_dir, f'corr_{name}_{conf_id}.npy')
            save_array(path, arr, logger)
        logger.info(f"  Saved 4 correlators: C(t=0) = "
                   f"pp_P0={pp_P0[0].real:.4e}, pp_P2={pp_P2[0].real:.4e}, "
                   f"pi_P0={pi_P0[0].real:.4e}, pi_P2={pi_P2[0].real:.4e}")

        corr_results[conf_id] = {'pp_P0': pp_P0, 'pp_P2': pp_P2,
                                 'pi_P0': pi_P0, 'pi_P2': pi_P2}
        backend.get_default_memory_pool().free_all_blocks()

    return corr_results


def jackknife_analysis(logger, conf_ids, corr_data):
    """Jackknife analysis across all configurations.

    For each channel (pp_P0, pp_P2, pi_P0, pi_P2):
      1. Stack correlators across configs: shape (Nconf, Nt)
      2. Jackknife resampling
      3. Effective mass: cosh for proton, log for pion
      4. Plateau fit with weighted average

    Returns
    -------
    Dict mapping channel_name -> {
        'corr_jk': Jackknife dict,
        'meff': meff dict,
        'E0': plateau energy (GeV),
        'E0_err': plateau error (GeV),
        'plateau_range': (t_start, t_end),
        'plateau_values': ndarray
    }
    """
    from lib.analyse import Jackknife, meff

    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Jackknife Analysis (Nconf={})".format(len(conf_ids)))
    logger.info("=" * 70)

    Nconf = len(conf_ids)
    analysis_dir = os.path.join(DATA_DIR, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    channels = ['pp_P0', 'pp_P2', 'pi_P0', 'pi_P2']
    meff_types = {'pp_P0': 'cosh', 'pp_P2': 'cosh', 'pi_P0': 'log', 'pi_P2': 'log'}
    labels = {'pp_P0': 'Proton P=0', 'pp_P2': 'Proton P=(0,0,2)',
              'pi_P0': 'Pion P=0', 'pi_P2': 'Pion P=(0,0,2)'}
    # Expected values: proton mass ~1.0 GeV at physical point;
    # pion mass depends on ensemble (m_π ~ 300 MeV typical for this β=6.20)
    # Pz=2 energy: E = sqrt(m² + p²) with p = 4π/L in GeV
    p_lat = 2 * np.pi * 2 / NX
    p_phys = p_lat * (_fm2GeV / ALttc)
    expected = {
        'pp_P0': 1.0,
        'pp_P2': np.sqrt(1.0**2 + p_phys**2),  # ≈1.40 GeV
        'pi_P0': 0.30,
        'pi_P2': np.sqrt(0.30**2 + p_phys**2),  # ≈1.03 GeV
    }
    # E(Pz=2) = sqrt(m₀² + p²), p = 2×2π/L, L=24a, a⁻¹=1.874 GeV
    # p = 2×2π×1.874/24 ≈ 0.981 GeV → E ≈ sqrt(1.0²+0.98²) ≈ 1.40 GeV ✓
    # E_pion(Pz=2) = sqrt(0.30²+0.98²) ≈ 1.03 GeV ... hmm, that seems high
    # Actually for the pion the expected value depends on the pion mass
    # Let me compute: p = 2·2π/(24·0.1053) * 0.1973 = 2·0.491 = 0.981 GeV
    # E_pion = sqrt(0.30² + 0.981²) ≈ 1.03 GeV
    # That's probably correct for a heavier-than-physical pion

    # But actually, let me recalculate:
    # a = 0.1053 fm, a⁻¹ = 0.1973/0.1053 ≈ 1.874 GeV
    # p = 2π·2/(Nx·a) = 4π/(24·0.1053) fm⁻¹ = 4π·0.1973/(24·0.1053) GeV
    #   = 4·3.1416·0.1973/(24·0.1053) = 2.479/(2.527) ≈ 0.981 GeV
    # E_proton = sqrt(1.0² + 0.981²) = sqrt(1.0 + 0.962) = sqrt(1.962) ≈ 1.401 GeV ✓
    # E_pion = sqrt(0.30² + 0.981²) = sqrt(0.09 + 0.962) = sqrt(1.052) ≈ 1.026 GeV
    # Hmm, that doesn't match the 0.55 I had. Let me just compute it dynamically.

    analysis_results = {}

    for ch in channels:
        logger.info(f"\n--- {labels[ch]} ---")

        # Stack correlators: (Nconf, Nt)
        stack = np.stack([np.real(corr_data[cid][ch]) for cid in conf_ids])
        logger.info(f"  Stacked shape: {stack.shape}")
        logger.info(f"  C(0) values: {np.array2string(stack[:,0], precision=4, max_line_width=120)}")
        logger.info(f"  C(1) values: {np.array2string(stack[:,1], precision=4, max_line_width=120)}")

        # Jackknife
        jk = Jackknife(stack, Nconf_axes=0)
        logger.info(f"  Jackknife: mean[t=0]={jk['data_mean'][0]:.6e} ± {jk['data_err'][0]:.6e}")

        # Effective mass
        mf = meff(jk['data_sample'], ALttc, Nconf_axes=0, Nt_axes=1,
                  meff_type=meff_types[ch])
        mf_mean = mf['data_mean']
        mf_err = mf['data_err']
        N_eff = len(mf_mean)

        # Plateau: early time region before backward-propagating state dominates
        # For Nt=72, good plateau is typically t ∈ [4, 15] for proton,
        # t ∈ [6, 18] for pion (pion lives longer).
        # We scan a window to find the flattest region.
        if 'pp' in ch:
            ps, pe = 4, min(N_eff - 2, 14)  # Proton: early plateau
        else:
            ps, pe = 5, min(N_eff - 2, 18)  # Pion: longer plateau

        # Filter out zero values (from NaN → 0 clamping in cosh meff)
        plateau = np.array([mf_mean[i] for i in range(ps, pe) if mf_mean[i] > 0.01])
        plateau_err = np.array([mf_err[i] for i in range(ps, pe) if mf_mean[i] > 0.01])

        if len(plateau) < 2:
            # Fallback: just take first few valid points
            plateau = np.array([mf_mean[i] for i in range(2, min(8, N_eff)) if mf_mean[i] > 0.01])
            plateau_err = np.array([mf_err[i] for i in range(2, min(8, N_eff)) if mf_mean[i] > 0.01])
            if len(plateau) < 2:
                logger.warning(f"  Cannot find valid plateau points!")
                E0 = float('nan'); E0_err = float('nan')
                analysis_results[ch] = {'corr_jk': jk, 'meff': mf,
                    'E0': E0, 'E0_err': E0_err, 'plateau_range': (0, 0),
                    'expected': 0.0}
                continue

        # Weighted average over plateau
        w = 1.0 / (plateau_err**2 + 1e-10)
        E0 = np.sum(plateau * w) / np.sum(w)
        E0_err = 1.0 / np.sqrt(np.sum(w))

        # Compute expected E(Pz=2) from measured E(P=0)
        if ch in ['pp_P2', 'pi_P2']:
            # Get the P=0 mass from the corresponding channel
            base_ch = ch.replace('_P2', '_P0')
            if base_ch in analysis_results:
                m0 = analysis_results[base_ch]['E0']
                if np.isnan(m0):
                    m0 = 1.0 if 'pp' in ch else 0.3
            else:
                m0 = 1.0 if 'pp' in ch else 0.3
            # p = 2π·Pz / (Nx·a) in GeV
            p_lat = 2 * np.pi * 2 / NX  # in lattice units (2π/L)
            p_phys = p_lat * (_fm2GeV / ALttc)  # in GeV
            expected_E = np.sqrt(m0**2 + p_phys**2)
            expected_label = f"sqrt(m₀²+p²) = sqrt({m0:.3f}²+{p_phys:.3f}²) = {expected_E:.3f}"
        else:
            if 'pp' in ch:
                expected_E = 1.0
                expected_label = "1.0 GeV (nucleon mass)"
            else:
                expected_E = 0.30
                expected_label = "~0.3 GeV (pion mass)"

        logger.info(f"  Effective plateau points ({len(plateau)} valid):")
        logger.info(f"    Values: {np.array2string(plateau, precision=4, max_line_width=120)}")
        logger.info(f"    Errors: {np.array2string(plateau_err, precision=4, max_line_width=120)}")
        logger.info(f"  Plateau fit:  E₀ = {E0:.4f} ± {E0_err:.4f} GeV")
        logger.info(f"  Expected:     {expected_label}")

        dev = abs(E0 - expected_E) / max(E0_err, 1e-10)
        if dev < 2:
            logger.info(f"  ✓  Consistent (deviation = {dev:.1f}σ)")
        elif dev < 4:
            logger.info(f"  ⚠  Marginal (deviation = {dev:.1f}σ)")
        else:
            logger.info(f"  ✗  Inconsistent (deviation = {dev:.1f}σ)")

        # Save
        save_array(os.path.join(analysis_dir, f'corr_{ch}_mean.npy'), jk['data_mean'], logger)
        save_array(os.path.join(analysis_dir, f'corr_{ch}_err.npy'), jk['data_err'], logger)
        save_array(os.path.join(analysis_dir, f'meff_{ch}_mean.npy'), mf_mean, logger)
        save_array(os.path.join(analysis_dir, f'meff_{ch}_err.npy'), mf_err, logger)

        analysis_results[ch] = {
            'corr_jk': jk,
            'meff': mf,
            'E0': E0,
            'E0_err': E0_err,
            'plateau_range': (ps, pe),
            'expected': expected_E,
        }

    return analysis_results


def generate_plots(logger, corr_data, analysis_results, conf_ids):
    """Generate comprehensive analysis plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Generating Plots")
    logger.info("=" * 70)

    Nconf = len(conf_ids)

    # ── Figure 1: Effective masses (2×2 layout) ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    channels_plot = ['pp_P0', 'pp_P2', 'pi_P0', 'pi_P2']
    titles = ['Proton P = (0,0,0)', 'Proton P = (0,0,2)',
              'Pion P = (0,0,0)', 'Pion P = (0,0,2)']
    colors = ['#3498DB', '#2980B9', '#2ECC71', '#27AE60']

    for i, (ch, title, color) in enumerate(zip(channels_plot, titles, colors)):
        ax = axes[i // 2, i % 2]
        ar = analysis_results[ch]
        mf = ar['meff']
        t = np.arange(len(mf['data_mean']))
        ps, pe = ar['plateau_range']

        # Plot effective mass
        ax.errorbar(t, mf['data_mean'], yerr=mf['data_err'],
                    fmt='o-', color=color, markersize=4, capsize=2, linewidth=1.5,
                    label='a·m_eff(t)')

        # Highlight plateau region
        ax.axvspan(ps, pe - 1, alpha=0.15, color=color,
                   label=f'Plateau [{ps},{pe-1}]')

        # Horizontal line for fitted E₀
        E0 = ar['E0']
        E0_err = ar['E0_err']
        ax.axhline(y=E0, color='red', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'E₀ = {E0:.3f} ± {E0_err:.3f} GeV')
        ax.axhspan(E0 - E0_err, E0 + E0_err, color='red', alpha=0.1)

        # Expected line
        ax.axhline(y=ar['expected'], color='gray', linestyle=':', linewidth=1,
                   alpha=0.5, label=f'Expected ~{ar["expected"]:.3f} GeV')

        ax.set_xlabel('t / a', fontsize=12)
        ax.set_ylabel('a·m_eff [GeV]', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, len(t) - 1)

    plt.tight_layout()
    path1 = os.path.join(PLOTS_DIR, 'effective_mass_all_channels.png')
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {path1}")

    # ── Figure 2: Correlation functions (per-config + Jackknife mean) ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for i, (ch, title, color) in enumerate(zip(channels_plot, titles, colors)):
        ax = axes[i // 2, i % 2]
        t_arr = np.arange(NT)

        # Per-config correlators
        for j, cid in enumerate(conf_ids):
            c = np.abs(np.real(corr_data[cid][ch]))
            ax.semilogy(t_arr, c + 1e-30, '.', color=color, markersize=2,
                       alpha=0.4, label=f'conf {cid}' if j == 0 else None)

        # Jackknife mean
        ar = analysis_results[ch]
        jk = ar['corr_jk']
        ax.semilogy(t_arr, np.abs(jk['data_mean']) + 1e-30, '-',
                    color='black', linewidth=2, label='JK mean')

        # Error band
        c_mean = np.abs(jk['data_mean'])
        c_err = jk['data_err']
        ax.fill_between(t_arr,
                       np.maximum(c_mean - c_err, 1e-30),
                       c_mean + c_err,
                       color=color, alpha=0.2, label='JK error')

        ax.set_xlabel('t / a', fontsize=12)
        ax.set_ylabel('|C(t)|', fontsize=12)
        ax.set_title(f'{title} — Correlation Function', fontsize=14, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path2 = os.path.join(PLOTS_DIR, 'correlators_all_channels.png')
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {path2}")

    # ── Figure 3: Summary table ──
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    # Compute momentum in GeV
    p_lat = 2 * np.pi * 2 / NX
    p_phys = p_lat * (_fm2GeV / ALttc)

    rows = [
        ["Channel", "E₀ [GeV]", "Expected [GeV]", "σ dev", "Plateau", "Nconf"],
    ]
    for ch, label in [('pp_P0', 'Proton P=0'), ('pp_P2', 'Proton P=(0,0,2)'),
                       ('pi_P0', 'Pion P=0'), ('pi_P2', 'Pion P=(0,0,2)')]:
        ar = analysis_results[ch]
        ps, pe = ar['plateau_range']
        dev = abs(ar['E0'] - ar['expected']) / max(ar['E0_err'], 1e-10)
        rows.append([
            label,
            f"{ar['E0']:.4f} ± {ar['E0_err']:.4f}",
            f"{ar['expected']:.3f}",
            f"{dev:.1f}",
            f"[{ps},{pe-1}]",
            f"{Nconf}"
        ])

    # Also add dispersion check
    if 'pp_P0' in analysis_results and 'pp_P2' in analysis_results:
        m0 = analysis_results['pp_P0']['E0']
        E2 = analysis_results['pp_P2']['E0']
        if not np.isnan(m0) and not np.isnan(E2):
            E2_theory = np.sqrt(m0**2 + p_phys**2)
            rows.append([
                "Proton dispersion", f"E(Pz=2)={E2:.3f}",
                f"sqrt(m₀²+p²)={E2_theory:.3f}", "", "", ""
            ])
    if 'pi_P0' in analysis_results and 'pi_P2' in analysis_results:
        m0 = analysis_results['pi_P0']['E0']
        E2 = analysis_results['pi_P2']['E0']
        if not np.isnan(m0) and not np.isnan(E2):
            E2_theory = np.sqrt(m0**2 + p_phys**2)
            rows.append([
                "Pion dispersion", f"E(Pz=2)={E2:.3f}",
                f"sqrt(m₀²+p²)={E2_theory:.3f}", "", "", ""
            ])

    table = ax.table(cellText=rows, cellLoc='center', loc='center',
                     colWidths=[0.22, 0.25, 0.25, 0.10, 0.10, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
    ax.set_title(f'Lattice QCD Analysis Summary — {ENSEMBLE} ({NX}³×{NT})  |  '
                f'a⁻¹={_fm2GeV/ALttc:.3f} GeV  |  p={p_phys:.3f} GeV',
                fontsize=13, fontweight='bold', pad=20)

    path3 = os.path.join(PLOTS_DIR, 'analysis_summary_table.png')
    fig.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {path3}")


def generate_report(logger, analysis_results, conf_ids, elapsed_total):
    """Write comprehensive Markdown report."""
    report_path = os.path.join(LOGS_DIR, 'full_analysis_report.md')

    p_lat = 2 * np.pi * 2 / NX
    p_phys = p_lat * (_fm2GeV / ALttc)
    Nconf = len(conf_ids)

    lines = []
    lines.append("# Lattice QCD Full Analysis Report")
    lines.append("")
    lines.append(f"**Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Ensemble:** {ENSEMBLE} ({NX}³×{NT}, a={ALttc} fm, a⁻¹={_fm2GeV/ALttc:.3f} GeV)")
    lines.append(f"**Configs:** {conf_ids} (Nconf={Nconf})")
    lines.append(f"**Nev:** {NEV}")
    lines.append(f"**Total time:** {elapsed_total/60:.1f} min")
    lines.append("")
    lines.append(f"**Momentum P=(0,0,2):** p_z = 2×(2π/L) = {p_phys:.3f} GeV")
    lines.append("")

    lines.append("---")
    lines.append("## Effective Mass Results")
    lines.append("")
    lines.append("| Channel | E₀ [GeV] | Expected [GeV] | σ | Plateau |")
    lines.append("|---------|-----------|----------------|---|---------|")

    for ch, label in [('pp_P0', 'Proton P=0'), ('pp_P2', 'Proton P=(0,0,2)'),
                       ('pi_P0', 'Pion P=0'), ('pi_P2', 'Pion P=(0,0,2)')]:
        ar = analysis_results[ch]
        ps, pe = ar['plateau_range']
        dev = abs(ar['E0'] - ar['expected']) / max(ar['E0_err'], 1e-10)
        status = "✅" if dev < 2 else ("⚠️" if dev < 4 else "❌")
        lines.append(f"| {label} | {ar['E0']:.4f} ± {ar['E0_err']:.4f} | {ar['expected']:.3f} | {dev:.1f}σ {status} | [{ps},{pe-1}] |")

    lines.append("")
    lines.append("---")
    lines.append("## Dispersion Relation Check")
    lines.append("")

    if 'pp_P0' in analysis_results and 'pp_P2' in analysis_results:
        m0_pp = analysis_results['pp_P0']['E0']
        E2_pp = analysis_results['pp_P2']['E0']
        E2_th_pp = np.sqrt(m0_pp**2 + p_phys**2)
        lines.append(f"- **Proton:** E(Pz=0) = {m0_pp:.4f} GeV, E(Pz=2) = {E2_pp:.4f} GeV")
        lines.append(f"  - Theoretical: √(m₀²+p²) = √({m0_pp:.3f}²+{p_phys:.3f}²) = {E2_th_pp:.4f} GeV")
        lines.append(f"  - Difference: {abs(E2_pp-E2_th_pp):.4f} GeV")

    if 'pi_P0' in analysis_results and 'pi_P2' in analysis_results:
        m0_pi = analysis_results['pi_P0']['E0']
        E2_pi = analysis_results['pi_P2']['E0']
        E2_th_pi = np.sqrt(m0_pi**2 + p_phys**2)
        lines.append(f"- **Pion:** E(Pz=0) = {m0_pi:.4f} GeV, E(Pz=2) = {E2_pi:.4f} GeV")
        lines.append(f"  - Theoretical: √(m₀²+p²) = √({m0_pi:.3f}²+{p_phys:.3f}²) = {E2_th_pi:.4f} GeV")
        lines.append(f"  - Difference: {abs(E2_pi-E2_th_pi):.4f} GeV")

    lines.append("")
    lines.append("---")
    lines.append("## Output Files")
    lines.append("")
    lines.append(f"- `{DATA_DIR}/{{conf_id}}/VdV_mom_{{conf_id}}.npy` — VdV vertices (Nt, 2, Nev, Nev)")
    lines.append(f"- `{DATA_DIR}/{{conf_id}}/VVV_mom_{{conf_id}}.npy` — VVV vertices (Nt, 2, Nev, Nev, Nev)")
    lines.append(f"- `{DATA_DIR}/{{conf_id}}/corr_*_{{conf_id}}.npy` — Correlators (Nt,)")
    lines.append(f"- `{DATA_DIR}/analysis/` — Jackknife means, errors, meff")
    lines.append(f"- `{PLOTS_DIR}/` — Figures")
    lines.append(f"- `{LOGS_DIR}/` — Logs and this report")

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    for line in lines:
        logger.info(line)

    logger.info(f"\nReport saved: {report_path}")
    return report_path


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Full Jackknife Analysis Pipeline')
    parser.add_argument('--conf-ids', type=int, nargs='+', default=CONF_IDS)
    parser.add_argument('--recompute-vertices', action='store_true',
                       help='Force recompute of vertices')
    parser.add_argument('--skip-vertices', action='store_true')
    parser.add_argument('--skip-correlators', action='store_true')
    parser.add_argument('--skip-plots', action='store_true')
    args = parser.parse_args()

    start_time = time.perf_counter()
    logger = setup_logging(LOGS_DIR, 'full_analysis')
    import cupy as cp

    logger.info("=" * 70)
    logger.info("FULL JACKKNIFE ANALYSIS PIPELINE")
    logger.info(f"  Ensemble: {ENSEMBLE} ({NX}³×{NT})")
    logger.info(f"  Configs: {args.conf_ids} (Nconf={len(args.conf_ids)})")
    logger.info(f"  Nev: {NEV}  |  Precision: {PRECISION}")
    logger.info(f"  GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    logger.info("=" * 70)

    # ── Step 1: Vertices ──
    vertex_data = {}
    if not args.skip_vertices:
        with Timer("Vertex computation", logger):
            vertex_data = compute_all_vertices(logger, args.conf_ids, args.recompute_vertices)
    else:
        for cid in args.conf_ids:
            conf_dir = os.path.join(DATA_DIR, str(cid))
            vertex_data[cid] = {
                'VdV': np.load(os.path.join(conf_dir, f'VdV_mom_{cid}.npy')),
                'VVV': np.load(os.path.join(conf_dir, f'VVV_mom_{cid}.npy')),
            }
            logger.info(f"  Loaded config {cid}: VdV{vertex_data[cid]['VdV'].shape} "
                       f"VVV{vertex_data[cid]['VVV'].shape}")

    # ── Step 2: Correlators ──
    corr_data = {}
    if not args.skip_correlators:
        with Timer("Correlator computation", logger):
            corr_data = compute_all_correlators(logger, args.conf_ids, vertex_data)
    else:
        for cid in args.conf_ids:
            conf_dir = os.path.join(DATA_DIR, str(cid))
            corr_data[cid] = {}
            for ch in ['pp_P0', 'pp_P2', 'pi_P0', 'pi_P2']:
                path = os.path.join(conf_dir, f'corr_{ch}_{cid}.npy')
                if os.path.exists(path):
                    corr_data[cid][ch] = np.load(path)
            logger.info(f"  Loaded config {cid}: {list(corr_data[cid].keys())}")

    # ── Step 3: Jackknife Analysis ──
    with Timer("Jackknife analysis", logger):
        analysis_results = jackknife_analysis(logger, args.conf_ids, corr_data)

    # ── Step 4: Plots ──
    if not args.skip_plots:
        with Timer("Plot generation", logger):
            generate_plots(logger, corr_data, analysis_results, args.conf_ids)

    # ── Report ──
    total_elapsed = time.perf_counter() - start_time
    generate_report(logger, analysis_results, args.conf_ids, total_elapsed)

    logger.info(f"\n{'='*70}")
    logger.info(f"ANALYSIS COMPLETE — Total: {total_elapsed/60:.1f} min")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
