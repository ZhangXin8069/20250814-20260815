"""
neutron + pi-  -> neutron + pi-  2pt correlation function
I = 3/2 channel, E32P29, Nev=100, P^2 <= 3
Single-threaded CuPy with lqcddb API
All perams pre-loaded into CPU (2D array), explicit contractions.
Correct Wick time mapping [t_q, t_aq] → peram[t_aq, t_q].
Reference: contraction.pn.n.2pt.cupy.py
"""
import sys, time, pathlib
import numpy as np

sys.path.insert(0, '/public/home/sush/distillation/lqcddb/src/')

from lqcddb.base import *
from lqcddb.eigvectors import vertex_creator
from lqcddb.constant import gamma
from lqcddb.analyse import loop_tsrc

# ── Backend ──────────────────────────────────────────────────
set_backend('cupy')
backend = get_backend()

# ── Parameters ───────────────────────────────────────────────
lattice_size = [32, 32, 32, 64]
grid_size = [1, 1, 1, 1]
Nx, Ny, Nz, Nt = lattice_size
Nev_src = 100
conf_id = sys.argv[1]

eigen_base = '/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64'
peram_base = '/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light'
eigen_tag = 'e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy'

# ── MPI init (single rank) ───────────────────────────────────
mpinit(grid_size=grid_size, latt_size=lattice_size, backend=backend.__name__)
rank = getMPIRank()
size = getMPISize()
Lt = Nt // size

t0 = time.perf_counter()
if rank == 0:
    print(f'=== contraction.Np.I1.5.2pt.cupy.py ===')
    print(f'conf_id={conf_id}, Nev_src={Nev_src}, lattice={lattice_size}')
    print(f'Process: neutron + pi- -> neutron + pi-  (I=3/2)')

# ── Gamma & projection ───────────────────────────────────────
gamma_7  = backend.asarray(gamma(7))
gamma_5m = backend.asarray(gamma(5))
projection = backend.asarray((gamma(0) + gamma(4)) / 2.0)

# ── Momentum lists (P^2 <= 3) ─────────────────────────────────
Mom_sink_VDV = (
    [[0, 0, 0]]
    + sorted(creat_mom_list(Mom=[0, 0, 1], fix_Q2=True))
    + sorted(creat_mom_list(Mom=[0, 1, 1], fix_Q2=True))
    + sorted(creat_mom_list(Mom=[1, 1, 1], fix_Q2=True))
)
Mom_sink_VVV = (
    [[0, 0, 0]]
    + sorted(creat_mom_list(Mom=[0, 0, 1], fix_Q2=True))[::-1]
    + sorted(creat_mom_list(Mom=[0, 1, 1], fix_Q2=True))[::-1]
    + sorted(creat_mom_list(Mom=[1, 1, 1], fix_Q2=True))[::-1]
)

Mom_len = len(Mom_sink_VDV)
Mom_group_size = 3
Mom_groups = Mom_len // Mom_group_size
if rank == 0:
    print(f'Momentum: {Mom_len} total (P^2<=3), groups={Mom_groups}')

# ── Phase factors ─────────────────────────────────────────────
vtx = vertex_creator(Nx=Nx)
phase_exp_2pt = backend.zeros((Mom_len, Nx, Nx, Nx, 3), dtype=complex)
phase_exp_3pt = backend.zeros((Mom_len, Nx, Nx, Nx), dtype=complex)
for mi in range(Mom_len):
    phase_exp_2pt[mi] = vtx.phase_exp_2pt(Mom=Mom_sink_VDV[mi])
    phase_exp_3pt[mi] = vtx.phase_exp_3pt(Mom=Mom_sink_VVV[mi])

# ── Precompute VdV (GPU) and VVV (CPU) ────────────────────────
sink_VdV = backend.zeros((Nt, Mom_len, Nev_src, Nev_src), dtype=complex)
sink_VVV = np.zeros((Nt, Mom_len, Nev_src, Nev_src, Nev_src), dtype=complex)

st_eigen = time.perf_counter()
for t in range(Nt):
    eigvecs = backend.load(f'{eigen_base}/{conf_id}/{conf_id}_t{t:03d}_{eigen_tag}.npy')
    eigvecs_src = eigvecs[:Nev_src]
    sink_VdV[t] = vtx.Mom_VdV_sink_t(phase_exp=phase_exp_2pt, eigvecs=eigvecs_src)
    sink_VVV[t] = backend.asnumpy(
        vtx.Mom_VVV_sink_t(phase_exp=backend.asarray(phase_exp_3pt), eigvecs=eigvecs_src)
    )

del eigvecs

if rank == 0:
    print(f'VdV/VVV: {(time.perf_counter() - st_eigen):.1f}s')

# ── Pre-load ALL perams into CPU (2D array, MPI-distributed) ──
# peram_all[t_from, t_to, d_sink, d_src, ev_sink, ev_src]
# Index: peram_all[t_aq, t_q] = peram from anti-quark time to quark time
st_peram = time.perf_counter()

peram_all = np.zeros((Nt, Nt, 4, 4, Nev_src, Nev_src), dtype=complex)
for t_from in range(Nt):
    data = np.load(f'{peram_base}/{conf_id}/t{t_from:03d}_{eigen_tag}.npy')
    peram_all[t_from] = data[:, :, :, :Nev_src, :Nev_src]

if rank == 0:
    mem_gb = peram_all.nbytes / 1e9
    print(f'Load perams: {(time.perf_counter() - st_peram):.1f}s  ({mem_gb:.1f} GB CPU)')

# ── Output array ──────────────────────────────────────────────
corr = backend.zeros((12, Mom_len, Mom_len, Nt, Nt), dtype=complex)

# ── Time mapping (Wick → physical, for peram[t_aq, t_q]) ─────
#   tsink → t_sink            tcur0 → t_sink
#   tcur1 → t_src             tsrc  → t_src
# Peram[t_aq, t_q]:
#   [tsink,tcur1] → peram[t_src,t_sink]  fwd
#   [tsink,tcur0] → peram[t_sink,t_sink] eq-sink
#   [tcur0,tsrc]  → peram[t_src,t_sink]  fwd
#   [tsrc,tcur1]  → peram[t_src,t_src]   eq-src
#   [tsink,tsrc]  → peram[t_src,t_sink]  fwd
#   [tcur0,tcur1] → peram[t_src,t_sink]  fwd
#   [tsrc,tcur0]  → peram[t_sink,t_src]  BACKWARD

# ── Main loop ─────────────────────────────────────────────────
# V-structure momentum (same for all diagrams):
#   VVV_0(bdf) @ tsink=t_sink → M (sink, full 27)
#   VDV_1(np)  @ tcur0=t_sink → M (sink, full 27)
#   VVV_2(hjl) @ tcur1=t_src  → N (source, 9-mom slice)
#   VDV_3(rt)  @ tsrc=t_src   → N (source, 9-mom slice)

if rank == 0:
    print(f'Starting main loop over {Nt} source times...')

for t_src in range(Nt):
    st_loop = time.perf_counter()

    # Source vertices (Bcast from t_src owner, single-rank acts as no-op)
    source_VdV = get_mpi_data(sink_VdV[t_src // size], mdtype='Bcast', root=t_src % size).transpose(0, 2, 1).conj()
    source_VVV = backend.asarray(get_mpi_data(sink_VVV[t_src // size], mdtype='Bcast', root=t_src % size)).conj()

    if rank == 0:
        print(f'[t_src={t_src:02d}] setup: {(time.perf_counter() - st_loop):.1f}s', end='')

    st_cal = time.perf_counter()

    # Sink times distributed via get_mpi_tlist (full Nt)
    t_sink_list_rank, _, t_sink_list_indx = get_mpi_tlist(Nt=Nt, t=range(t_src, t_src + Nt // 2, 1), gtype='TScatter')
    for t_sink_indx, t_sink in enumerate(t_sink_list_indx):
        t_real_sink = t_sink_list_rank[t_sink_indx]

        # Sink V-structures (LOCAL index)
        sink_VVV_ts = backend.asarray(sink_VVV[t_sink])
        sink_VdV_ts = sink_VdV[t_sink]

        # Peram slices: peram_all[t_from, t_to] = peram from t_from to t_to
        peram_src_sink = backend.asarray(peram_all[t_src,  t_sink])
        peram_src_src  = backend.asarray(peram_all[t_src,  t_src ])
        peram_sink_sink= backend.asarray(peram_all[t_sink, t_sink])
        peram_sink_src = backend.asarray(peram_all[t_sink, t_src ])

        for mi in range(Mom_groups):
            N_slice = slice(mi * Mom_group_size, (mi + 1) * Mom_group_size)

            # ── Diagram 0  sign=-1 ──
            # perams: agbh fwd, ekfl fwd, oqpr fwd, cmdn eq-sink, sitj eq-src
            corr[0, N_slice, :, t_src, t_sink] += -1.0 * cached_contract(
                'agbh,ekfl,oqpr,cmdn,sitj,ce,mo,gi,qs,Nbdf,Nnp,Mhjl,Mrt,ak->MN',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src,
                gamma_7, gamma_5m, gamma_7, gamma_5m,
                sink_VVV_ts, sink_VdV_ts, source_VVV[N_slice], source_VdV[N_slice],
                projection, optimize=['auto', 'greedy', 'dp'],
            )

            # ── Diagram 1  sign=+1 ──
            # perams: agbh fwd, ekfl fwd, oqpr fwd, cidj fwd, smtn BACKWARD
            corr[1, N_slice, :, t_src, t_sink] += +1.0 * cached_contract(
                'agbh,ekfl,oqpr,cidj,smtn,ce,mo,gi,qs,Nbdf,Nnp,Mhjl,Mrt,ak->MN',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src,
                gamma_7, gamma_5m, gamma_7, gamma_5m,
                sink_VVV_ts, sink_VdV_ts, source_VVV[N_slice], source_VdV[N_slice],
                projection, optimize=['auto', 'greedy', 'dp'],
            )

            # ── Diagram 2  sign=+1 ──
            # perams: agbh fwd, eqfr fwd, okpl fwd, cmdn eq-sink, sitj eq-src
            corr[2, N_slice, :, t_src, t_sink] += +1.0 * cached_contract(
                'agbh,eqfr,okpl,cmdn,sitj,ce,mo,gi,qs,Nbdf,Nnp,Mhjl,Mrt,ak->MN',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src,
                gamma_7, gamma_5m, gamma_7, gamma_5m,
                sink_VVV_ts, sink_VdV_ts, source_VVV[N_slice], source_VdV[N_slice],
                projection, optimize=['auto', 'greedy', 'dp'],
            )

            # ── Diagram 3  sign=-1 ──
            # perams: agbh fwd, eqfr fwd, okpl fwd, cidj fwd, smtn BACKWARD
            corr[3, N_slice, :, t_src, t_sink] += -1.0 * cached_contract(
                'agbh,eqfr,okpl,cidj,smtn,ce,mo,gi,qs,Nbdf,Nnp,Mhjl,Mrt,ak->MN',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src,
                gamma_7, gamma_5m, gamma_7, gamma_5m,
                sink_VVV_ts, sink_VdV_ts, source_VVV[N_slice], source_VdV[N_slice],
                projection, optimize=['auto', 'greedy', 'dp'],
            )

            # ── Diagram 4  sign=+1 ──
            # perams: akbl fwd, egfh fwd, oqpr fwd, cmdn eq-sink, sitj eq-src
            corr[4, N_slice, :, t_src, t_sink] += +1.0 * cached_contract(
                'akbl,egfh,oqpr,cmdn,sitj,ce,mo,gi,qs,Nbdf,Nnp,Mhjl,Mrt,ak->MN',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src,
                gamma_7, gamma_5m, gamma_7, gamma_5m,
                sink_VVV_ts, sink_VdV_ts, source_VVV[N_slice], source_VdV[N_slice],
                projection, optimize=['auto', 'greedy', 'dp'],
            )

            # ── Diagram 5  sign=-1 ──
            # perams: akbl fwd, egfh fwd, oqpr fwd, cidj fwd, smtn BACKWARD
            corr[5, N_slice, :, t_src, t_sink] += -1.0 * cached_contract(
                'akbl,egfh,oqpr,cidj,smtn,ce,mo,gi,qs,Nbdf,Nnp,Mhjl,Mrt,ak->MN',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src,
                gamma_7, gamma_5m, gamma_7, gamma_5m,
                sink_VVV_ts, sink_VdV_ts, source_VVV[N_slice], source_VdV[N_slice],
                projection, optimize=['auto', 'greedy', 'dp'],
            )

            # ── Diagram 6  sign=-1 ──
            # perams: akbl fwd, eqfr fwd, ogph fwd, cmdn eq-sink, sitj eq-src
            corr[6, N_slice, :, t_src, t_sink] += -1.0 * cached_contract(
                'akbl,eqfr,ogph,cmdn,sitj,ce,mo,gi,qs,Nbdf,Nnp,Mhjl,Mrt,ak->MN',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src,
                gamma_7, gamma_5m, gamma_7, gamma_5m,
                sink_VVV_ts, sink_VdV_ts, source_VVV[N_slice], source_VdV[N_slice],
                projection, optimize=['auto', 'greedy', 'dp'],
            )

            # ── Diagram 7  sign=+1 ──
            # perams: akbl fwd, eqfr fwd, ogph fwd, cidj fwd, smtn BACKWARD
            corr[7, N_slice, :, t_src, t_sink] += +1.0 * cached_contract(
                'akbl,eqfr,ogph,cidj,smtn,ce,mo,gi,qs,Nbdf,Nnp,Mhjl,Mrt,ak->MN',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src,
                gamma_7, gamma_5m, gamma_7, gamma_5m,
                sink_VVV_ts, sink_VdV_ts, source_VVV[N_slice], source_VdV[N_slice],
                projection, optimize=['auto', 'greedy', 'dp'],
            )

            # ── Diagram 8  sign=-1 ──
            # perams: aqbr fwd, egfh fwd, okpl fwd, cmdn eq-sink, sitj eq-src
            corr[8, N_slice, :, t_src, t_sink] += -1.0 * cached_contract(
                'aqbr,egfh,okpl,cmdn,sitj,ce,mo,gi,qs,Nbdf,Nnp,Mhjl,Mrt,ak->MN',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src,
                gamma_7, gamma_5m, gamma_7, gamma_5m,
                sink_VVV_ts, sink_VdV_ts, source_VVV[N_slice], source_VdV[N_slice],
                projection, optimize=['auto', 'greedy', 'dp'],
            )

            # ── Diagram 9  sign=+1 ──
            # perams: aqbr fwd, egfh fwd, okpl fwd, cidj fwd, smtn BACKWARD
            corr[9, N_slice, :, t_src, t_sink] += +1.0 * cached_contract(
                'aqbr,egfh,okpl,cidj,smtn,ce,mo,gi,qs,Nbdf,Nnp,Mhjl,Mrt,ak->MN',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src,
                gamma_7, gamma_5m, gamma_7, gamma_5m,
                sink_VVV_ts, sink_VdV_ts, source_VVV[N_slice], source_VdV[N_slice],
                projection, optimize=['auto', 'greedy', 'dp'],
            )

            # ── Diagram 10  sign=+1 ──
            # perams: aqbr fwd, ekfl fwd, ogph fwd, cmdn eq-sink, sitj eq-src
            corr[10, N_slice, :, t_src, t_sink] += +1.0 * cached_contract(
                'aqbr,ekfl,ogph,cmdn,sitj,ce,mo,gi,qs,Nbdf,Nnp,Mhjl,Mrt,ak->MN',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src,
                gamma_7, gamma_5m, gamma_7, gamma_5m,
                sink_VVV_ts, sink_VdV_ts, source_VVV[N_slice], source_VdV[N_slice],
                projection, optimize=['auto', 'greedy', 'dp'],
            )

            # ── Diagram 11  sign=-1 ──
            # perams: aqbr fwd, ekfl fwd, ogph fwd, cidj fwd, smtn BACKWARD
            corr[11, N_slice, :, t_src, t_sink] += -1.0 * cached_contract(
                'aqbr,ekfl,ogph,cidj,smtn,ce,mo,gi,qs,Nbdf,Nnp,Mhjl,Mrt,ak->MN',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src,
                gamma_7, gamma_5m, gamma_7, gamma_5m,
                sink_VVV_ts, sink_VdV_ts, source_VVV[N_slice], source_VdV[N_slice],
                projection, optimize=['auto', 'greedy', 'dp'],
            )

    if rank == 0:
        dt = time.perf_counter() - st_cal
        n_calls = Nt * Mom_groups * 12
        print(f'  contract: {dt:.1f}s ({n_calls}c)  total: {(time.perf_counter()-t0):.0f}s')

# ── Time-source average ───────────────────────────────────────
if rank == 0:
    print(f'\nMain loop done ({time.perf_counter()-t0:.0f}s). Running loop_tsrc...')

corr_avg = loop_tsrc(
    backend.asnumpy(corr), indx=[-2, -1],
    Boundary_Conditions='Antiperiodic',
)


# ── Save ──────────────────────────────────────────────────────
if rank == 0:
    corr_total = backend.sum(backend.asarray(corr_avg), axis=0)
    print(corr_total[0, 0, 0])
    
    save_dir = f'/public/home/sush/distillation/0v2b/result/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}'
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    save_path = (
        f'{save_dir}/corr_ud_2pt_gamma0505_{eigen_tag}'
        f'_neutron_pi-_neutron_pi-_src{Nev_src}.npy'
    )
    corr_save = backend.asnumpy(corr_total)
    np.save(save_path, corr_save)
    print(f'\nSaved: {save_path}')
    print(f'Shape (Mom_src, Mom_snk, tau): {corr_save.shape}')
    print(f'Done. Total time: {(time.perf_counter() - t0):.1f}s')
