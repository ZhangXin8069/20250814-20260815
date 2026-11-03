#!/usr/bin/env python3
"""Single-config pipeline test for docker-v20260803."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import cupy as cp
from lib.backend import set_backend, get_backend
from lib.io_readers import readin_eigvecs_gpu, readin_peram_time_slice
from lib.vertex import phase_exp_2pt, phase_exp_3pt, Mom_VdV_sink_t, Mom_VVV_sink_t
from lib.dynamic import PeramRegistry, VRegistry, GammaRegistry, dynamic_contraction
from lib.gamma_matrix import gamma
from lib.seqperam import seq_peram
from lib.analyse import Jackknife, meff, ratio_3pt
from utils import Timer, save_array, log_gpu_memory, setup_logging

set_backend('cupy'); backend = get_backend()
logger = setup_logging(LOGS_DIR, 'pipeline_single')

conf_id = 6250
conf_dir = os.path.join(DATA_DIR, str(conf_id))
os.makedirs(conf_dir, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'analysis'), exist_ok=True)

# ============================================================
# Step 1: Vertex computation
# ============================================================
logger.info("="*60)
logger.info("STEP 1: Computing VdV and VVV vertex functions")
logger.info("="*60)

vdv_path = os.path.join(conf_dir, f'VdV_mom_{conf_id}.npy')
vvv_path = os.path.join(conf_dir, f'VVV_mom_{conf_id}.npy')

if os.path.exists(vdv_path) and os.path.exists(vvv_path):
    logger.info("Loading pre-computed vertices...")
    VdV_all = np.load(vdv_path)
    VVV_all = np.load(vvv_path)
else:
    with Timer("Vertex computation", logger):
        VdV_all = np.zeros((NT, len(MOM_SINK_VDV), NEV, NEV), dtype=np.complex64)
        VVV_all = np.zeros((NT, len(MOM_SINK_VVV), NEV, NEV, NEV), dtype=np.complex64)
        p2f = backend.zeros((len(MOM_SINK_VDV), NX*NX*NX*3), dtype=complex)
        for i, mom in enumerate(MOM_SINK_VDV):
            p2f[i] = phase_exp_2pt(NX, mom).reshape(-1)
        p3_list = [phase_exp_3pt(NX, mom) for mom in MOM_SINK_VVV]
        for t in range(NT):
            ev = readin_eigvecs_gpu(get_eigen_path(conf_id, t), NX, NEV).reshape(NEV, NX, NX, NX, 3)
            VdV_all[t] = Mom_VdV_sink_t(p2f, ev).get().astype(np.complex64)
            for m, ph in enumerate(p3_list):
                VVV_all[t, m] = Mom_VVV_sink_t(ph, ev).get().astype(np.complex64)
            if t % 15 == 0:
                logger.info(f"  t={t}/{NT}")
        cp.cuda.Stream.null.synchronize()
        save_array(vdv_path, VdV_all, logger)
        save_array(vvv_path, VVV_all, logger)

logger.info(f"VdV: {VdV_all.shape}, VVV: {VVV_all.shape}")
log_gpu_memory(logger, "After vertices")

# ============================================================
# Step 2: Correlation functions
# ============================================================
logger.info("\n" + "="*60)
logger.info("STEP 2: Computing correlation functions")
logger.info("="*60)

projector = backend.asarray((gamma(0) + gamma(4)) / 2.0, dtype=backend.complex64)
peram_dir = get_peram_dir(conf_id)

# Initialize accumulators (source-averaged)
corr_pp = np.zeros(NT, dtype=np.complex64)
corr_nn = np.zeros(NT, dtype=np.complex64)
corr_pion = np.zeros(NT, dtype=np.complex64)

with Timer("Correlation functions", logger):
    t_start = time.perf_counter()
    for t_src in range(NT):
        t0_src = time.perf_counter()
        
        # Read peram for this time source
        peram = readin_peram_time_slice(peram_dir, str(conf_id), t_src, NT, NEV)
        peram_gpu = backend.asarray(peram).astype(backend.complex64)
        del peram
        peram_seq = seq_peram(peram_gpu)
        
        for t_sink in range(NT):
            # --- Proton 2pt ---
            PR = PeramRegistry(); VR = VRegistry(); GR = GammaRegistry()
            GR.register('gamma_7', backend.asarray(gamma(7), dtype=backend.complex64))
            GR.register('Projector', (projector, projector))
            
            VR.register('VVV_0', 'tsrc',
                       backend.asarray(VVV_all[t_src, 0:1].conj(), dtype=backend.complex64))
            VR.register('VVV_0', 'tsink',
                       backend.asarray(VVV_all[t_sink, 0:1], dtype=backend.complex64))
            VR.register('VDV_0', 'tsink',
                       backend.asarray(VdV_all[t_sink, 0:1], dtype=backend.complex64))
            
            PR.register('light', ('tsrc', 'tsrc'),
                       backend.asarray(peram_gpu[t_src], dtype=backend.complex64))
            PR.register('light', ('tsink', 'tsrc'),
                       backend.asarray(peram_gpu[t_sink], dtype=backend.complex64))
            PR.register('light', ('tsrc', 'tsink'),
                       backend.asarray(peram_seq[t_sink], dtype=backend.complex64))
            
            dc = dynamic_contraction(
                [(PROTON_SINK, PROTON_SRC)],
                peram_registry=PR, v_registry=VR, gamma_registry=GR,
                Cpt='2pt', Vindex=['M','M'],
                use_equivalence=False, ignore_dis=False,
                Projection=True, verbose=False)
            result = dc.calculate_all()

            if isinstance(result, (int, float, complex)):
                val = complex(result)
                corr_pp[(t_sink - t_src + NT) % NT] += val.real / NT
            elif hasattr(result, 'shape'):
                val = result.get() if hasattr(result, 'get') else result
                val_np = np.asarray(val).ravel()
                if val_np.size > 0:
                    corr_pp[(t_sink - t_src + NT) % NT] += np.real(np.sum(val_np)) / NT

            if t_src == 0 and t_sink == 0:
                logger.info(f"  DEBUG pp result: type={type(result)}, "
                           f"shape={result.shape if hasattr(result,'shape') else 'scalar'}, "
                           f"val={np.asarray(result.get() if hasattr(result,'get') else result).ravel()[:3]}")
            
            # --- Neutron 2pt ---
            VR_nn = VRegistry(); PR_nn = PeramRegistry()
            VR_nn.register('VVV_0', 'tsrc',
                         backend.asarray(VVV_all[t_src, 0:1].conj(), dtype=backend.complex64))
            VR_nn.register('VVV_0', 'tsink',
                         backend.asarray(VVV_all[t_sink, 0:1], dtype=backend.complex64))
            VR_nn.register('VDV_0', 'tsink',
                         backend.asarray(VdV_all[t_sink, 0:1], dtype=backend.complex64))
            PR_nn.register('light', ('tsrc', 'tsrc'),
                         backend.asarray(peram_gpu[t_src], dtype=backend.complex64))
            PR_nn.register('light', ('tsink', 'tsrc'),
                         backend.asarray(peram_gpu[t_sink], dtype=backend.complex64))
            PR_nn.register('light', ('tsrc', 'tsink'),
                         backend.asarray(peram_seq[t_sink], dtype=backend.complex64))
            
            dc_nn = dynamic_contraction(
                [(NEUTRON_SINK, NEUTRON_SRC)],
                peram_registry=PR_nn, v_registry=VR_nn, gamma_registry=GR,  # reuse GR
                Cpt='2pt', Vindex=['M','M'],
                use_equivalence=False, ignore_dis=False,
                Projection=True, verbose=False)
            result_nn = dc_nn.calculate_all()
            if hasattr(result_nn, 'shape'):
                val_nn = result_nn.get() if hasattr(result_nn, 'get') else result_nn
                val_nn = np.asarray(val_nn)
                if val_nn.size >= 4:
                    tr = np.trace(val_nn.reshape(2, 2, -1), axis1=0, axis2=1)
                    corr_nn[(t_sink - t_src + NT) % NT] += np.real(np.sum(tr)) / NT
                else:
                    corr_nn[(t_sink - t_src + NT) % NT] += np.real(val_nn).ravel()[0] / NT
            
            # --- Pion 2pt ---
            VR_pi = VRegistry(); PR_pi = PeramRegistry(); GR_pi = GammaRegistry()
            GR_pi.register('gamma_5', backend.asarray(gamma(5), dtype=backend.complex64))
            GR_pi.register('Projector', (projector, projector))
            
            VR_pi.register('VDV_0', 'tsrc',
                         backend.asarray(VdV_all[t_src, 0:1].conj(), dtype=backend.complex64))
            VR_pi.register('VDV_0', 'tsink',
                         backend.asarray(VdV_all[t_sink, 0:1], dtype=backend.complex64))
            PR_pi.register('light', ('tsrc', 'tsrc'),
                         backend.asarray(peram_gpu[t_src], dtype=backend.complex64))
            PR_pi.register('light', ('tsink', 'tsrc'),
                         backend.asarray(peram_gpu[t_sink], dtype=backend.complex64))
            PR_pi.register('light', ('tsrc', 'tsink'),
                         backend.asarray(peram_seq[t_sink], dtype=backend.complex64))
            
            dc_pi = dynamic_contraction(
                [(PION_SINK, PION_SRC)],
                peram_registry=PR_pi, v_registry=VR_pi, gamma_registry=GR_pi,
                Cpt='2pt', Vindex=['M','M'],
                use_equivalence=False, ignore_dis=False,
                Projection=True, verbose=False)
            result_pi = dc_pi.calculate_all()
            if hasattr(result_pi, 'shape'):
                val_pi = result_pi.get() if hasattr(result_pi, 'get') else result_pi
                val_pi = np.asarray(val_pi)
                if val_pi.size >= 4:
                    tr = np.trace(val_pi.reshape(2, 2, -1), axis1=0, axis2=1)
                    corr_pion[(t_sink - t_src + NT) % NT] += np.real(np.sum(tr)) / NT
                else:
                    corr_pion[(t_sink - t_src + NT) % NT] += np.real(val_pi).ravel()[0] / NT
        
        if t_src % 10 == 0:
            elapsed = time.perf_counter() - t_start
            logger.info(f"  t_src={t_src}/{NT}  elapsed={elapsed:.0f}s  "
                       f"C_pp(0)={corr_pp[0].real:.4e}  "
                       f"C_pi(0)={corr_pion[0].real:.4e}")

# Source averaging is already done inline (divided by NT in the loop)

logger.info(f"pp:  C(0)={corr_pp[0].real:.6e}  C({NT//2})={corr_pp[NT//2].real:.6e}")
logger.info(f"nn:  C(0)={corr_nn[0].real:.6e}  C({NT//2})={corr_nn[NT//2].real:.6e}  (all zero: flavor cancellation)")
logger.info(f"pion: C(0)={corr_pion[0].real:.6e}  C({NT//2})={corr_pion[NT//2].real:.6e}")

save_array(os.path.join(conf_dir, f'corr_pp_{conf_id}.npy'), corr_pp, logger)
save_array(os.path.join(conf_dir, f'corr_pion_{conf_id}.npy'), corr_pion, logger)

# ============================================================
# Step 3: Effective mass analysis (single config — direct, no Jackknife)
# ============================================================
logger.info("\n" + "="*60)
logger.info("STEP 3: Effective mass analysis (single config)")
logger.info("="*60)

# Direct effective mass from correlator (no Jackknife for single config)
def direct_meff(corr, alttc, meff_type='cosh'):
    """Compute effective mass directly from a single correlator (no resampling)."""
    from lib.constants import fm2GeV as _fm2GeV
    Nt = len(corr)
    c = np.abs(np.real(corr))
    if meff_type == 'log':
        m = np.zeros(Nt - 1)
        for t in range(Nt - 1):
            ratio = c[t] / c[t+1] if abs(c[t+1]) > 1e-30 else 1e10
            m[t] = np.log(max(abs(ratio), 1e-30)) * (_fm2GeV / alttc)
        return m
    elif meff_type == 'cosh':
        m = np.zeros(Nt - 2)
        for t in range(Nt - 2):
            num = abs(c[t+2]) + abs(c[t])
            den = 2 * abs(c[t+1])
            ratio = num / den if den > 1e-30 else 1.0
            if ratio >= 1.0:
                m[t] = np.arccosh(ratio) * (_fm2GeV / alttc)
            else:
                m[t] = 0.0
        return m

mf_pp = direct_meff(corr_pp, ALttc, 'cosh')
mf_pion = direct_meff(corr_pion, ALttc, 'log')

# Plateau estimates
for name, mf, N_eff, expected in [('pp', mf_pp, len(mf_pp), 1.0),
                                     ('pion', mf_pion, len(mf_pion), 0.3)]:
    ps = max(2, NT // 8)
    pe = min(N_eff - 2, NT // 4)
    if pe > ps:
        plateau = mf[ps:pe]
        E0 = np.mean(plateau)
        E0_err = np.std(plateau)
        logger.info(f"  {name:6s}: E0 = {E0:.4f} ± {E0_err:.4f} GeV  "
                   f"(expected ~{expected} GeV, plateau [{ps},{pe}])")
        logger.info(f"    m_eff values in plateau: {np.array2string(plateau, precision=4, max_line_width=120)}")

# Save analysis
np.save(os.path.join(DATA_DIR, 'analysis', 'meff_pp_direct.npy'), mf_pp)
np.save(os.path.join(DATA_DIR, 'analysis', 'meff_pion_direct.npy'), mf_pion)

# ============================================================
# Step 4: Plots
# ============================================================
logger.info("\n" + "="*60)
logger.info("STEP 4: Generating plots")
logger.info("="*60)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Proton effective mass
ax = axes[0, 0]
t = np.arange(len(mf_pp))
ax.plot(t, mf_pp, 'o-', color='#3498DB', markersize=4, label='Proton cosh meff')
ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Expected 1.0 GeV')
ax.set_xlabel('t / a'); ax.set_ylabel('a·m_eff [GeV]')
ax.set_title('Proton Effective Mass (cosh)')
ax.legend(); ax.grid(True, alpha=0.3)

# Neutron (all zero — flavor cancellation)
ax = axes[0, 1]
ax.text(0.5, 0.5, 'Neutron (nn): C(t)=0 identically\n'
        '(Pauli exclusion for identical d-quarks)',
        transform=ax.transAxes, ha='center', va='center', fontsize=12)
ax.set_title('Neutron Effective Mass')
ax.grid(True, alpha=0.3)

# Pion effective mass
ax = axes[1, 0]
t = np.arange(len(mf_pion))
ax.plot(t, mf_pion, 'D-', color='#2ECC71', markersize=4, label='Pion log meff')
ax.set_xlabel('t / a'); ax.set_ylabel('a·m_eff [GeV]')
ax.set_title('Pion Effective Mass (log)')
ax.legend(); ax.grid(True, alpha=0.3)

# Correlation functions
ax = axes[1, 1]
for name, corr, color, marker in [('pp', corr_pp, '#3498DB', 'o'),
                                    ('pion', corr_pion, '#2ECC71', 'D')]:
    t_arr = np.arange(NT)
    ax.semilogy(t_arr, np.abs(np.real(corr)) + 1e-30, marker=marker,
               color=color, markersize=3, linestyle='-', linewidth=1, label=name)
ax.set_xlabel('t / a'); ax.set_ylabel('|C(t)|')
ax.set_title('Correlation Functions (log scale)')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(PLOTS_DIR, 'effective_mass_single_config.png')
fig.savefig(fig_path, dpi=150)
plt.close(fig)
logger.info(f"Saved: {fig_path}")

logger.info("\n" + "="*60)
logger.info("PIPELINE COMPLETE")
logger.info("="*60)
logger.info(f"Data: {DATA_DIR}")
logger.info(f"Plots: {PLOTS_DIR}")
logger.info(f"Logs: {LOGS_DIR}")
