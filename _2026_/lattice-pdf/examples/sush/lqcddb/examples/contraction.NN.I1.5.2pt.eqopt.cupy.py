import sys, time
import numpy as np

from lqcddb.base import *
from lqcddb.eigvectors import vertex_creator
from lqcddb.constant import gamma
from lqcddb.analyse import loop_tsrc  # kept for user's post-processing
from lqcddb.constant.sigma_matrix import Mom_times_sigma
from lqcddb.io import check_dir_path
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
    print(f'=== contraction.NN.I1.5.2pt.cupy.py ===')
    print(f'conf_id={conf_id}, Nev_src={Nev_src}, lattice={lattice_size}')
    print(f'Process: neutron + pi- -> neutron + pi-  (I=3/2)')

# ── Gamma & projection ───────────────────────────────────────
gamma_7 = backend.asarray(gamma(7))
gamma_5 = backend.asarray(gamma(5))
projection = backend.asarray((gamma(0) + gamma(4)) / 2.0)[:, :2]

# ── Momentum lists (P^2 <= 3) ─────────────────────────────────
Mom_sink_VDV = (
    [[0, 0, 0]]
    + sorted(creat_mom_list(Mom=[0, 0, 1], fix_Q2=True))
    + sorted(creat_mom_list(Mom=[0, 1, 1], fix_Q2=True))
    + sorted(creat_mom_list(Mom=[1, 1, 1], fix_Q2=True))
)
Mom_sink_VVV = (
    [[0, 0, 0]]
    + sorted(creat_mom_list(Mom=[0, 0, 1], fix_Q2=True))
    + sorted(creat_mom_list(Mom=[0, 1, 1], fix_Q2=True))
    + sorted(creat_mom_list(Mom=[1, 1, 1], fix_Q2=True))
)

Mom_len = len(Mom_sink_VDV)
if rank == 0:
    print(f'Momentum: {Mom_len} total (P^2<=3)')

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

# ── Pre-load ALL perams into CPU (2D array) ───────────────────
st_peram = time.perf_counter()
peram_all = np.zeros((Nt, Nt, 4, 4, Nev_src, Nev_src), dtype=complex)
for t_from in range(Nt):
    data = np.load(f'{peram_base}/{conf_id}/t{t_from:03d}_{eigen_tag}.npy')
    peram_all[t_from] = data[:, :, :, :Nev_src, :Nev_src]
if rank == 0:
    mem_gb = peram_all.nbytes / 1e9
    print(f'Load perams: {(time.perf_counter() - st_peram):.1f}s  ({mem_gb:.1f} GB CPU)')

# ── Pre-compute p̂·σ⃗ for all momenta ───────────────────────────
psigma_all = backend.zeros((Mom_len, 4, 4), dtype=complex)
psigma_dag_all = backend.zeros((Mom_len, 4, 4), dtype=complex)
for k in range(Mom_len):
    p = backend.asarray(Mom_times_sigma(Mom_sink_VVV[k], upto4dim=True))
    psigma_all[k] = p
    psigma_dag_all[k] = p.conj().T
if rank == 0:
    print(f'p̂·σ⃗ precomputed: {Mom_len} momenta')

# ── Output: 3x3 GEVP matrix, 27 momenta, Nt x Nt time ────────
corr = backend.zeros((3, 3, Mom_len, Nt, Nt, 2, 2), dtype=complex)

# ── Main loop ─────────────────────────────────────────────────
if rank == 0:
    print(f'Starting main loop over {Nt} source times...')

for t_src in range(0, 1, 1):
    st_loop = time.perf_counter()

    # Source vertices (Bcast from t_src owner)
    source_VdV_full = get_mpi_data(sink_VdV[t_src // size], mdtype='Bcast', root=t_src % size).transpose(0, 2, 1).conj()
    source_VVV_full = backend.asarray(get_mpi_data(sink_VVV[t_src // size], mdtype='Bcast', root=t_src % size)).conj()

    t_sink_list_rank, _, t_sink_list_indx = get_mpi_tlist(Nt=Nt, t=range(t_src, t_src + Nt // 2, 1), gtype='TScatter')

    for t_sink_indx, t_sink in enumerate(t_sink_list_indx):
        t_real_sink = t_sink_list_rank[t_sink_indx]

        # Sink V-structures (full 27-mom arrays)
        sink_VVV_ts = backend.asarray(sink_VVV[t_sink])
        sink_VdV_ts = sink_VdV[t_sink]

        # Peram slices
        peram_src_sink = backend.asarray(peram_all[t_src,  t_sink])
        peram_sink_src = backend.asarray(peram_all[t_sink, t_src])
        peram_src_src  = backend.asarray(peram_all[t_src,  t_src])
        peram_sink_sink= backend.asarray(peram_all[t_sink, t_sink])

        # V tensors at momentum 0 → broadcast to (Mom_len, ...) for K-index
        snkVVV0 = sink_VVV_ts[0:1]     # (1, Nev, Nev, Nev)
        snkVDV0 = sink_VdV_ts[0:1]     # (1, Nev, Nev)
        srcVVV0 = source_VVV_full[0:1]
        srcVDV0 = source_VdV_full[0:1]
        
        for Mom_indx in range(3):
            Mom_list = [x for x in range(Mom_indx, Mom_len, 3)]
            
            # V tensors at all K (already (Mom_len, ...))
            snkVVVK = sink_VVV_ts[Mom_list]                  # (len(Mom_list), Nev, Nev, Nev)
            snkVDVK = sink_VdV_ts[Mom_list]                  # (len(Mom_list), Nev, Nev)
            srcVVVK = source_VVV_full[Mom_list]              # (len(Mom_list), Nev, Nev, Nev)
            srcVDVK = source_VdV_full[Mom_list]              # (len(Mom_list), Nev, Nev)
            
            # ==== M^{1,1} ====
            # Group 0, CG=1.0
            # diag0 sign=+1.0  aobp(fwd), egfh(fwd), cmdn(fwd)
            corr[0,0,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
                'aobp,egfh,cmdn,ce,mo,Kbdf,Knph,ay,gz->Kyz',
                peram_src_sink, peram_src_sink, peram_src_sink, gamma_7, gamma_7, snkVVVK, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # diag1 sign=-1.0  agbh(fwd), eofp(fwd), cmdn(fwd)
            corr[0,0,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
                'agbh,eofp,cmdn,ce,mo,Kbdf,Knph,ay,gz->Kyz',
                peram_src_sink, peram_src_sink, peram_src_sink, gamma_7, gamma_7, snkVVVK, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            
            # ==== M^{1,2} ====
            # Group 1, CG=1.0
            # diag0 sign=-1.0  aobp(fwd), egfh(fwd), cmdn(fwd), kilj(eq_src)
            corr[0,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
                'eofp,ambn,cgdh,kilj,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # diag1 sign=+1.0  aobp(fwd), egfh(fwd), cidj(fwd), kmln(eq_src)
            corr[0,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
                'eofp,ambn,cidj,kglh,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # diag2 sign=+1.0  agbh(fwd), eofp(fwd), cmdn(fwd), kilj(eq_src)
            corr[0,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
                'eofp,agbh,cmdn,kilj,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # diag3 sign=-1.0  agbh(fwd), eofp(fwd), cidj(fwd), kmln(eq_src)
            corr[0,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
                'eofp,agbh,cidj,kmln,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # diag4 sign=-1.0  agbh(fwd), eofp(fwd), cidj(fwd), kmln(eq_src)
            corr[0,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
                'eofp,aibj,cmdn,kglh,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # diag5 sign=1.0  agbh(fwd), eofp(fwd), cidj(fwd), kmln(eq_src)
            corr[0,1,Mom_list,t_src,t_sink] += 1.0 * cached_contract(
                'eofp,aibj,cgdh,kmln,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            
            # Group 2, CG=1.0
            # diag0 sign=+1.0  aobp(fwd), egfh(fwd), kilj(eq_src), cmdn(fwd)
            # corr[0,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eofp,kilj,ambn,cgdh,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_src, peram_src_sink, peram_src_sink, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            corr[0,1,Mom_list,t_src,t_sink] += 1.0 * cached_contract(
                'eofp,ambn,cgdh,kilj,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            
            # diag1 sign=-1.0  aobp(fwd), eifj(fwd), kglh(eq_src), cmdn(fwd)
            # corr[0,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eofp,kilj,agbh,cmdn,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_src, peram_src_sink, peram_src_sink, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            corr[0,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
                'eofp,agbh,cmdn,kilj,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            
            # diag2 sign=-1.0  agbh(fwd), eofp(fwd), kilj(eq_src), cmdn(fwd)
            corr[0,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
                'eifj,kolp,ambn,cgdh,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
                peram_src_sink, peram_src_src, peram_src_sink, peram_src_sink, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # diag3 sign=+1.0  agbh(fwd), eifj(fwd), kolp(eq_src), cmdn(fwd)
            # corr[0,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,kolp,agbh,cmdn,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_src, peram_src_sink, peram_src_sink, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            corr[0,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
                'eofp,agbh,cidj,kmln,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            
            # Group 3, CG=1.0
            # diag0 sign=+1.0  aobp(fwd), eifj(fwd), cmdn(fwd), kglh(eq_src)
            # corr[0,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eofp,kglh,ambn,cidj,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_src, peram_src_sink, peram_src_sink, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            corr[0,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
                'eofp,ambn,cidj,kglh,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # diag1 sign=-1.0  aobp(fwd), eifj(fwd), cgdh(fwd), kmln(eq_src)
            # corr[0,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eofp,kglh,aibj,cmdn,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_src, peram_src_sink, peram_src_sink, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            corr[0,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
                'eofp,aibj,cmdn,kglh,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # diag2 sign=-1.0  aibj(fwd), eofp(fwd), cmdn(fwd), kglh(eq_src)
            # corr[0,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'egfh,kolp,ambn,cidj,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_src, peram_src_sink, peram_src_sink, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            corr[0,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
                'eifj,kolp,ambn,cgdh,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
                peram_src_sink, peram_src_src, peram_src_sink, peram_src_sink, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # diag3 sign=+1.0  aibj(fwd), eofp(fwd), cgdh(fwd), kmln(eq_src)
            # corr[0,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'egfh,kolp,aibj,cmdn,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_src, peram_src_sink, peram_src_sink, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            corr[0,1,Mom_list,t_src,t_sink] += 1.0 * cached_contract(
                'eofp,aibj,cgdh,kmln,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
                peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            
            # # ==== M^{1,3} ====
            # # Group 1, CG=1.0
            # # diag0 sign=-1.0  aobp(fwd), egfh(fwd), cmdn(fwd), kilj(eq_src)
            # corr[0,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aobp,egfh,cmdn,kilj,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=+1.0  aobp(fwd), egfh(fwd), cidj(fwd), kmln(eq_src)
            # corr[0,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aobp,egfh,cidj,kmln,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=+1.0  agbh(fwd), eofp(fwd), cmdn(fwd), kilj(eq_src)
            # corr[0,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'agbh,eofp,cmdn,kilj,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=-1.0  agbh(fwd), eofp(fwd), cidj(fwd), kmln(eq_src)
            # corr[0,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'agbh,eofp,cidj,kmln,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # Group 2, CG=1.0
            # # diag0 sign=+1.0  aobp(fwd), egfh(fwd), kilj(eq_src), cmdn(fwd)
            # corr[0,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aobp,egfh,kilj,cmdn,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  aobp(fwd), eifj(fwd), kglh(eq_src), cmdn(fwd)
            # corr[0,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aobp,eifj,kglh,cmdn,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  agbh(fwd), eofp(fwd), kilj(eq_src), cmdn(fwd)
            # corr[0,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'agbh,eofp,kilj,cmdn,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  agbh(fwd), eifj(fwd), kolp(eq_src), cmdn(fwd)
            # corr[0,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'agbh,eifj,kolp,cmdn,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  aibj(fwd), eofp(fwd), kglh(eq_src), cmdn(fwd)
            # corr[0,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eofp,kglh,cmdn,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  aibj(fwd), egfh(fwd), kolp(eq_src), cmdn(fwd)
            # corr[0,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,egfh,kolp,cmdn,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # Group 3, CG=1.0
            # # diag0 sign=+1.0  aobp(fwd), eifj(fwd), cmdn(fwd), kglh(eq_src)
            # corr[0,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aobp,eifj,cmdn,kglh,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  aobp(fwd), eifj(fwd), cgdh(fwd), kmln(eq_src)
            # corr[0,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aobp,eifj,cgdh,kmln,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  aibj(fwd), eofp(fwd), cmdn(fwd), kglh(eq_src)
            # corr[0,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eofp,cmdn,kglh,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  aibj(fwd), eofp(fwd), cgdh(fwd), kmln(eq_src)
            # corr[0,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eofp,cgdh,kmln,ce,mo,ik,Kbdf,Knph,Kjl,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_7, gamma_5, snkVVVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # ==== M^{2,1} ====
            # # Group 4, CG=1.0
            # # diag0 sign=+1.0  aibj(fwd), ekfl(fwd), cmdn(eq_sink), ogph(fwd)
            # corr[1,0,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,cmdn,ogph,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, snkVVVK, snkVDV0, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  aibj(fwd), ekfl(fwd), cgdh(fwd), ompn(eq_sink)
            # corr[1,0,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,cgdh,ompn,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, snkVVVK, snkVDV0, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  akbl(fwd), eifj(fwd), cmdn(eq_sink), ogph(fwd)
            # corr[1,0,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,cmdn,ogph,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, snkVVVK, snkVDV0, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  akbl(fwd), eifj(fwd), cgdh(fwd), ompn(eq_sink)
            # corr[1,0,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,cgdh,ompn,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, snkVVVK, snkVDV0, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # Group 5, CG=1.0
            # # diag0 sign=+1.0  ambn(eq_sink), eifj(fwd), okpl(fwd), cgdh(fwd)
            # corr[1,0,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eifj,okpl,cgdh,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, snkVVVK, snkVDV0, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  ambn(eq_sink), ekfl(fwd), oipj(fwd), cgdh(fwd)
            # corr[1,0,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,ekfl,oipj,cgdh,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, snkVVVK, snkVDV0, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  aibj(fwd), emfn(eq_sink), okpl(fwd), cgdh(fwd)
            # corr[1,0,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,emfn,okpl,cgdh,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, snkVVVK, snkVDV0, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  aibj(fwd), ekfl(fwd), ompn(eq_sink), cgdh(fwd)
            # corr[1,0,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,ompn,cgdh,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, snkVVVK, snkVDV0, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  akbl(fwd), emfn(eq_sink), oipj(fwd), cgdh(fwd)
            # corr[1,0,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,emfn,oipj,cgdh,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, snkVVVK, snkVDV0, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  akbl(fwd), eifj(fwd), ompn(eq_sink), cgdh(fwd)
            # corr[1,0,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,ompn,cgdh,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, snkVVVK, snkVDV0, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # Group 6, CG=1.0
            # # diag0 sign=-1.0  eifj(fwd), okpl(fwd), ambn(eq_sink), cgdh(fwd)
            # corr[1,0,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,okpl,ambn,cgdh,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, snkVVVK, snkVDV0, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=+1.0  eifj(fwd), okpl(fwd), agbh(fwd), cmdn(eq_sink)
            # corr[1,0,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,okpl,agbh,cmdn,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, snkVVVK, snkVDV0, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=+1.0  ekfl(fwd), oipj(fwd), ambn(eq_sink), cgdh(fwd)
            # corr[1,0,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oipj,ambn,cgdh,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, snkVVVK, snkVDV0, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=-1.0  ekfl(fwd), oipj(fwd), agbh(fwd), cmdn(eq_sink)
            # corr[1,0,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oipj,agbh,cmdn,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, snkVVVK, snkVDV0, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # ==== M^{2,2} ====
            # # Group 7, CG=1.0
            # # diag0 sign=-1.0  eifj(fwd), oqpr(fwd), ambn(eq_sink), cgdh(fwd), sktl(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,oqpr,ambn,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=+1.0  eifj(fwd), oqpr(fwd), ambn(eq_sink), ckdl(fwd), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,oqpr,ambn,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=+1.0  eifj(fwd), oqpr(fwd), agbh(fwd), cmdn(eq_sink), sktl(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,oqpr,agbh,cmdn,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=-1.0  eifj(fwd), oqpr(fwd), agbh(fwd), ckdl(fwd), smtn(bwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,oqpr,agbh,ckdl,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=-1.0  eifj(fwd), oqpr(fwd), akbl(fwd), cmdn(eq_sink), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,oqpr,akbl,cmdn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=+1.0  eifj(fwd), oqpr(fwd), akbl(fwd), cgdh(fwd), smtn(bwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,oqpr,akbl,cgdh,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=+1.0  eqfr(fwd), oipj(fwd), ambn(eq_sink), cgdh(fwd), sktl(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eqfr,oipj,ambn,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=-1.0  eqfr(fwd), oipj(fwd), ambn(eq_sink), ckdl(fwd), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eqfr,oipj,ambn,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=-1.0  eqfr(fwd), oipj(fwd), agbh(fwd), cmdn(eq_sink), sktl(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eqfr,oipj,agbh,cmdn,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=+1.0  eqfr(fwd), oipj(fwd), agbh(fwd), ckdl(fwd), smtn(bwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eqfr,oipj,agbh,ckdl,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=+1.0  eqfr(fwd), oipj(fwd), akbl(fwd), cmdn(eq_sink), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eqfr,oipj,akbl,cmdn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=-1.0  eqfr(fwd), oipj(fwd), akbl(fwd), cgdh(fwd), smtn(bwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eqfr,oipj,akbl,cgdh,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # Group 8, CG=1.0
            # # diag0 sign=-1.0  eifj(fwd), okpl(fwd), sqtr(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,okpl,sqtr,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=+1.0  eifj(fwd), okpl(fwd), sqtr(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,okpl,sqtr,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=+1.0  eifj(fwd), oqpr(fwd), sktl(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,oqpr,sktl,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=-1.0  eifj(fwd), oqpr(fwd), sktl(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,oqpr,sktl,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  ekfl(fwd), oipj(fwd), sqtr(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oipj,sqtr,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  ekfl(fwd), oipj(fwd), sqtr(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oipj,sqtr,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=-1.0  ekfl(fwd), oqpr(fwd), sitj(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oqpr,sitj,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=+1.0  ekfl(fwd), oqpr(fwd), sitj(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oqpr,sitj,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=-1.0  eqfr(fwd), oipj(fwd), sktl(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eqfr,oipj,sktl,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=+1.0  eqfr(fwd), oipj(fwd), sktl(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eqfr,oipj,sktl,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=+1.0  eqfr(fwd), okpl(fwd), sitj(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eqfr,okpl,sitj,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=-1.0  eqfr(fwd), okpl(fwd), sitj(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eqfr,okpl,sitj,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])

            # # Group 9, CG=1.0
            # # diag0 sign=+1.0  eifj(fwd), okpl(fwd), ambn(eq_sink), cgdh(fwd), sqtr(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,okpl,ambn,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  eifj(fwd), okpl(fwd), ambn(eq_sink), cqdr(fwd), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,okpl,ambn,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  eifj(fwd), okpl(fwd), agbh(fwd), cmdn(eq_sink), sqtr(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,okpl,agbh,cmdn,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  eifj(fwd), okpl(fwd), agbh(fwd), cqdr(fwd), smtn(bwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,okpl,agbh,cqdr,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  eifj(fwd), okpl(fwd), aqbr(fwd), cmdn(eq_sink), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,okpl,aqbr,cmdn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  eifj(fwd), okpl(fwd), aqbr(fwd), cgdh(fwd), smtn(bwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,okpl,aqbr,cgdh,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=-1.0  ekfl(fwd), oipj(fwd), ambn(eq_sink), cgdh(fwd), sqtr(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oipj,ambn,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=+1.0  ekfl(fwd), oipj(fwd), ambn(eq_sink), cqdr(fwd), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oipj,ambn,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=+1.0  ekfl(fwd), oipj(fwd), agbh(fwd), cmdn(eq_sink), sqtr(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oipj,agbh,cmdn,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=-1.0  ekfl(fwd), oipj(fwd), agbh(fwd), cqdr(fwd), smtn(bwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oipj,agbh,cqdr,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=-1.0  ekfl(fwd), oipj(fwd), aqbr(fwd), cmdn(eq_sink), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oipj,aqbr,cmdn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=+1.0  ekfl(fwd), oipj(fwd), aqbr(fwd), cgdh(fwd), smtn(bwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oipj,aqbr,cgdh,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # Group 10, CG=1.0
            # # diag0 sign=+1.0  aibj(fwd), eqfr(fwd), cmdn(eq_sink), ogph(fwd), sktl(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,cmdn,ogph,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  aibj(fwd), eqfr(fwd), cmdn(eq_sink), okpl(fwd), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,cmdn,okpl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  aibj(fwd), eqfr(fwd), cgdh(fwd), ompn(eq_sink), sktl(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,cgdh,ompn,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  aibj(fwd), eqfr(fwd), cgdh(fwd), okpl(fwd), smtn(bwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,cgdh,okpl,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  aibj(fwd), eqfr(fwd), ckdl(fwd), ompn(eq_sink), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,ckdl,ompn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  aibj(fwd), eqfr(fwd), ckdl(fwd), ogph(fwd), smtn(bwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,ckdl,ogph,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=-1.0  aqbr(fwd), eifj(fwd), cmdn(eq_sink), ogph(fwd), sktl(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,cmdn,ogph,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=+1.0  aqbr(fwd), eifj(fwd), cmdn(eq_sink), okpl(fwd), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,cmdn,okpl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=+1.0  aqbr(fwd), eifj(fwd), cgdh(fwd), ompn(eq_sink), sktl(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,cgdh,ompn,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=-1.0  aqbr(fwd), eifj(fwd), cgdh(fwd), okpl(fwd), smtn(bwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,cgdh,okpl,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=-1.0  aqbr(fwd), eifj(fwd), ckdl(fwd), ompn(eq_sink), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,ckdl,ompn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=+1.0  aqbr(fwd), eifj(fwd), ckdl(fwd), ogph(fwd), smtn(bwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,ckdl,ogph,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # Group 11, CG=1.0
            # # diag0 sign=+1.0  ambn(eq_sink), eifj(fwd), oqpr(fwd), cgdh(fwd), sktl(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eifj,oqpr,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  ambn(eq_sink), eifj(fwd), oqpr(fwd), ckdl(fwd), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,eifj,oqpr,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  ambn(eq_sink), eqfr(fwd), oipj(fwd), cgdh(fwd), sktl(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,eqfr,oipj,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  ambn(eq_sink), eqfr(fwd), oipj(fwd), ckdl(fwd), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eqfr,oipj,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=-1.0  aibj(fwd), emfn(eq_sink), oqpr(fwd), cgdh(fwd), sktl(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,emfn,oqpr,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=+1.0  aibj(fwd), emfn(eq_sink), oqpr(fwd), ckdl(fwd), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,emfn,oqpr,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=+1.0  aibj(fwd), eqfr(fwd), ompn(eq_sink), cgdh(fwd), sktl(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,ompn,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=-1.0  aibj(fwd), eqfr(fwd), ompn(eq_sink), ckdl(fwd), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,ompn,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=+1.0  aqbr(fwd), emfn(eq_sink), oipj(fwd), cgdh(fwd), sktl(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,emfn,oipj,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=-1.0  aqbr(fwd), emfn(eq_sink), oipj(fwd), ckdl(fwd), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,emfn,oipj,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=-1.0  aqbr(fwd), eifj(fwd), ompn(eq_sink), cgdh(fwd), sktl(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,ompn,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=+1.0  aqbr(fwd), eifj(fwd), ompn(eq_sink), ckdl(fwd), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,ompn,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # Group 12, CG=1.0
            # # diag0 sign=+1.0  ambn(eq_sink), eifj(fwd), okpl(fwd), sqtr(eq_src), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eifj,okpl,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  ambn(eq_sink), eifj(fwd), oqpr(fwd), sktl(eq_src), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,eifj,oqpr,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  ambn(eq_sink), ekfl(fwd), oipj(fwd), sqtr(eq_src), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,ekfl,oipj,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  ambn(eq_sink), ekfl(fwd), oqpr(fwd), sitj(eq_src), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,ekfl,oqpr,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  ambn(eq_sink), eqfr(fwd), oipj(fwd), sktl(eq_src), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eqfr,oipj,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  ambn(eq_sink), eqfr(fwd), okpl(fwd), sitj(eq_src), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,eqfr,okpl,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=-1.0  aibj(fwd), emfn(eq_sink), okpl(fwd), sqtr(eq_src), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,emfn,okpl,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=+1.0  aibj(fwd), emfn(eq_sink), oqpr(fwd), sktl(eq_src), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,emfn,oqpr,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=+1.0  aibj(fwd), ekfl(fwd), ompn(eq_sink), sqtr(eq_src), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,ompn,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=-1.0  aibj(fwd), ekfl(fwd), oqpr(fwd), smtn(bwd), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,oqpr,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=-1.0  aibj(fwd), eqfr(fwd), ompn(eq_sink), sktl(eq_src), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,ompn,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=+1.0  aibj(fwd), eqfr(fwd), okpl(fwd), smtn(bwd), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,okpl,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag12 sign=+1.0  akbl(fwd), emfn(eq_sink), oipj(fwd), sqtr(eq_src), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,emfn,oipj,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag13 sign=-1.0  akbl(fwd), emfn(eq_sink), oqpr(fwd), sitj(eq_src), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,emfn,oqpr,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag14 sign=-1.0  akbl(fwd), eifj(fwd), ompn(eq_sink), sqtr(eq_src), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,ompn,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag15 sign=+1.0  akbl(fwd), eifj(fwd), oqpr(fwd), smtn(bwd), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,oqpr,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag16 sign=+1.0  akbl(fwd), eqfr(fwd), ompn(eq_sink), sitj(eq_src), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eqfr,ompn,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag17 sign=-1.0  akbl(fwd), eqfr(fwd), oipj(fwd), smtn(bwd), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eqfr,oipj,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag18 sign=-1.0  aqbr(fwd), emfn(eq_sink), oipj(fwd), sktl(eq_src), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,emfn,oipj,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag19 sign=+1.0  aqbr(fwd), emfn(eq_sink), okpl(fwd), sitj(eq_src), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,emfn,okpl,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag20 sign=+1.0  aqbr(fwd), eifj(fwd), ompn(eq_sink), sktl(eq_src), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,ompn,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag21 sign=-1.0  aqbr(fwd), eifj(fwd), okpl(fwd), smtn(bwd), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,okpl,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag22 sign=-1.0  aqbr(fwd), ekfl(fwd), ompn(eq_sink), sitj(eq_src), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,ekfl,ompn,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag23 sign=+1.0  aqbr(fwd), ekfl(fwd), oipj(fwd), smtn(bwd), cgdh(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,ekfl,oipj,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # Group 13, CG=1.0
            # # diag0 sign=-1.0  ambn(eq_sink), eifj(fwd), okpl(fwd), cgdh(fwd), sqtr(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,eifj,okpl,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=+1.0  ambn(eq_sink), eifj(fwd), okpl(fwd), cqdr(fwd), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eifj,okpl,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=+1.0  ambn(eq_sink), ekfl(fwd), oipj(fwd), cgdh(fwd), sqtr(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,ekfl,oipj,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=-1.0  ambn(eq_sink), ekfl(fwd), oipj(fwd), cqdr(fwd), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,ekfl,oipj,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  aibj(fwd), emfn(eq_sink), okpl(fwd), cgdh(fwd), sqtr(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,emfn,okpl,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  aibj(fwd), emfn(eq_sink), okpl(fwd), cqdr(fwd), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,emfn,okpl,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=-1.0  aibj(fwd), ekfl(fwd), ompn(eq_sink), cgdh(fwd), sqtr(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,ompn,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=+1.0  aibj(fwd), ekfl(fwd), ompn(eq_sink), cqdr(fwd), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,ompn,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=-1.0  akbl(fwd), emfn(eq_sink), oipj(fwd), cgdh(fwd), sqtr(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,emfn,oipj,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=+1.0  akbl(fwd), emfn(eq_sink), oipj(fwd), cqdr(fwd), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,emfn,oipj,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=+1.0  akbl(fwd), eifj(fwd), ompn(eq_sink), cgdh(fwd), sqtr(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,ompn,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=-1.0  akbl(fwd), eifj(fwd), ompn(eq_sink), cqdr(fwd), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,ompn,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # Group 14, CG=1.0
            # # diag0 sign=+1.0  aibj(fwd), ekfl(fwd), sqtr(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,sqtr,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  aibj(fwd), ekfl(fwd), sqtr(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,sqtr,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  aibj(fwd), eqfr(fwd), sktl(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,sktl,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  aibj(fwd), eqfr(fwd), sktl(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,sktl,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=-1.0  akbl(fwd), eifj(fwd), sqtr(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,sqtr,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=+1.0  akbl(fwd), eifj(fwd), sqtr(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,sqtr,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=+1.0  akbl(fwd), eqfr(fwd), sitj(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eqfr,sitj,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=-1.0  akbl(fwd), eqfr(fwd), sitj(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eqfr,sitj,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=+1.0  aqbr(fwd), eifj(fwd), sktl(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,sktl,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=-1.0  aqbr(fwd), eifj(fwd), sktl(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,sktl,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=-1.0  aqbr(fwd), ekfl(fwd), sitj(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,ekfl,sitj,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=+1.0  aqbr(fwd), ekfl(fwd), sitj(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,ekfl,sitj,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # Group 15, CG=1.0
            # # diag0 sign=-1.0  aibj(fwd), ekfl(fwd), cmdn(eq_sink), ogph(fwd), sqtr(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,cmdn,ogph,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=+1.0  aibj(fwd), ekfl(fwd), cmdn(eq_sink), oqpr(fwd), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,cmdn,oqpr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=+1.0  aibj(fwd), ekfl(fwd), cgdh(fwd), ompn(eq_sink), sqtr(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,cgdh,ompn,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=-1.0  aibj(fwd), ekfl(fwd), cgdh(fwd), oqpr(fwd), smtn(bwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,cgdh,oqpr,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=-1.0  aibj(fwd), ekfl(fwd), cqdr(fwd), ompn(eq_sink), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,cqdr,ompn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=+1.0  aibj(fwd), ekfl(fwd), cqdr(fwd), ogph(fwd), smtn(bwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,cqdr,ogph,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=+1.0  akbl(fwd), eifj(fwd), cmdn(eq_sink), ogph(fwd), sqtr(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,cmdn,ogph,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=-1.0  akbl(fwd), eifj(fwd), cmdn(eq_sink), oqpr(fwd), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,cmdn,oqpr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=-1.0  akbl(fwd), eifj(fwd), cgdh(fwd), ompn(eq_sink), sqtr(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,cgdh,ompn,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=+1.0  akbl(fwd), eifj(fwd), cgdh(fwd), oqpr(fwd), smtn(bwd)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,cgdh,oqpr,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=+1.0  akbl(fwd), eifj(fwd), cqdr(fwd), ompn(eq_sink), sgth(eq_src)
            # corr[1,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,cqdr,ompn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=-1.0  akbl(fwd), eifj(fwd), cqdr(fwd), ogph(fwd), smtn(bwd)
            # corr[1,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,cqdr,ogph,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # ==== M^{2,3} ====
            # # Group 7, CG=1.0
            # # diag0 sign=-1.0  eifj(fwd), oqpr(fwd), ambn(eq_sink), cgdh(fwd), sktl(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,oqpr,ambn,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=+1.0  eifj(fwd), oqpr(fwd), ambn(eq_sink), ckdl(fwd), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,oqpr,ambn,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=+1.0  eifj(fwd), oqpr(fwd), agbh(fwd), cmdn(eq_sink), sktl(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,oqpr,agbh,cmdn,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=-1.0  eifj(fwd), oqpr(fwd), agbh(fwd), ckdl(fwd), smtn(bwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,oqpr,agbh,ckdl,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=-1.0  eifj(fwd), oqpr(fwd), akbl(fwd), cmdn(eq_sink), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,oqpr,akbl,cmdn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=+1.0  eifj(fwd), oqpr(fwd), akbl(fwd), cgdh(fwd), smtn(bwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,oqpr,akbl,cgdh,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=+1.0  eqfr(fwd), oipj(fwd), ambn(eq_sink), cgdh(fwd), sktl(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eqfr,oipj,ambn,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=-1.0  eqfr(fwd), oipj(fwd), ambn(eq_sink), ckdl(fwd), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eqfr,oipj,ambn,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=-1.0  eqfr(fwd), oipj(fwd), agbh(fwd), cmdn(eq_sink), sktl(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eqfr,oipj,agbh,cmdn,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=+1.0  eqfr(fwd), oipj(fwd), agbh(fwd), ckdl(fwd), smtn(bwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eqfr,oipj,agbh,ckdl,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=+1.0  eqfr(fwd), oipj(fwd), akbl(fwd), cmdn(eq_sink), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eqfr,oipj,akbl,cmdn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=-1.0  eqfr(fwd), oipj(fwd), akbl(fwd), cgdh(fwd), smtn(bwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eqfr,oipj,akbl,cgdh,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # Group 8, CG=1.0
            # # diag0 sign=-1.0  eifj(fwd), okpl(fwd), sqtr(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,okpl,sqtr,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=+1.0  eifj(fwd), okpl(fwd), sqtr(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,okpl,sqtr,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=+1.0  eifj(fwd), oqpr(fwd), sktl(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,oqpr,sktl,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=-1.0  eifj(fwd), oqpr(fwd), sktl(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,oqpr,sktl,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  ekfl(fwd), oipj(fwd), sqtr(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oipj,sqtr,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  ekfl(fwd), oipj(fwd), sqtr(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oipj,sqtr,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=-1.0  ekfl(fwd), oqpr(fwd), sitj(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oqpr,sitj,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=+1.0  ekfl(fwd), oqpr(fwd), sitj(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oqpr,sitj,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=-1.0  eqfr(fwd), oipj(fwd), sktl(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eqfr,oipj,sktl,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=+1.0  eqfr(fwd), oipj(fwd), sktl(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eqfr,oipj,sktl,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=+1.0  eqfr(fwd), okpl(fwd), sitj(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eqfr,okpl,sitj,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=-1.0  eqfr(fwd), okpl(fwd), sitj(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eqfr,okpl,sitj,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])

            # # Group 9, CG=1.0
            # # diag0 sign=+1.0  eifj(fwd), okpl(fwd), ambn(eq_sink), cgdh(fwd), sqtr(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,okpl,ambn,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  eifj(fwd), okpl(fwd), ambn(eq_sink), cqdr(fwd), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,okpl,ambn,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  eifj(fwd), okpl(fwd), agbh(fwd), cmdn(eq_sink), sqtr(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,okpl,agbh,cmdn,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  eifj(fwd), okpl(fwd), agbh(fwd), cqdr(fwd), smtn(bwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,okpl,agbh,cqdr,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  eifj(fwd), okpl(fwd), aqbr(fwd), cmdn(eq_sink), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,okpl,aqbr,cmdn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  eifj(fwd), okpl(fwd), aqbr(fwd), cgdh(fwd), smtn(bwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,okpl,aqbr,cgdh,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=-1.0  ekfl(fwd), oipj(fwd), ambn(eq_sink), cgdh(fwd), sqtr(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oipj,ambn,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=+1.0  ekfl(fwd), oipj(fwd), ambn(eq_sink), cqdr(fwd), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oipj,ambn,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=+1.0  ekfl(fwd), oipj(fwd), agbh(fwd), cmdn(eq_sink), sqtr(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oipj,agbh,cmdn,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=-1.0  ekfl(fwd), oipj(fwd), agbh(fwd), cqdr(fwd), smtn(bwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oipj,agbh,cqdr,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=-1.0  ekfl(fwd), oipj(fwd), aqbr(fwd), cmdn(eq_sink), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oipj,aqbr,cmdn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=+1.0  ekfl(fwd), oipj(fwd), aqbr(fwd), cgdh(fwd), smtn(bwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oipj,aqbr,cgdh,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # Group 10, CG=1.0
            # # diag0 sign=+1.0  aibj(fwd), eqfr(fwd), cmdn(eq_sink), ogph(fwd), sktl(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,cmdn,ogph,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  aibj(fwd), eqfr(fwd), cmdn(eq_sink), okpl(fwd), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,cmdn,okpl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  aibj(fwd), eqfr(fwd), cgdh(fwd), ompn(eq_sink), sktl(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,cgdh,ompn,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  aibj(fwd), eqfr(fwd), cgdh(fwd), okpl(fwd), smtn(bwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,cgdh,okpl,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  aibj(fwd), eqfr(fwd), ckdl(fwd), ompn(eq_sink), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,ckdl,ompn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  aibj(fwd), eqfr(fwd), ckdl(fwd), ogph(fwd), smtn(bwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,ckdl,ogph,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=-1.0  aqbr(fwd), eifj(fwd), cmdn(eq_sink), ogph(fwd), sktl(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,cmdn,ogph,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=+1.0  aqbr(fwd), eifj(fwd), cmdn(eq_sink), okpl(fwd), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,cmdn,okpl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=+1.0  aqbr(fwd), eifj(fwd), cgdh(fwd), ompn(eq_sink), sktl(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,cgdh,ompn,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=-1.0  aqbr(fwd), eifj(fwd), cgdh(fwd), okpl(fwd), smtn(bwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,cgdh,okpl,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=-1.0  aqbr(fwd), eifj(fwd), ckdl(fwd), ompn(eq_sink), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,ckdl,ompn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=+1.0  aqbr(fwd), eifj(fwd), ckdl(fwd), ogph(fwd), smtn(bwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,ckdl,ogph,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # Group 11, CG=1.0
            # # diag0 sign=+1.0  ambn(eq_sink), eifj(fwd), oqpr(fwd), cgdh(fwd), sktl(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eifj,oqpr,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  ambn(eq_sink), eifj(fwd), oqpr(fwd), ckdl(fwd), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,eifj,oqpr,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  ambn(eq_sink), eqfr(fwd), oipj(fwd), cgdh(fwd), sktl(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,eqfr,oipj,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  ambn(eq_sink), eqfr(fwd), oipj(fwd), ckdl(fwd), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eqfr,oipj,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=-1.0  aibj(fwd), emfn(eq_sink), oqpr(fwd), cgdh(fwd), sktl(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,emfn,oqpr,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=+1.0  aibj(fwd), emfn(eq_sink), oqpr(fwd), ckdl(fwd), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,emfn,oqpr,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=+1.0  aibj(fwd), eqfr(fwd), ompn(eq_sink), cgdh(fwd), sktl(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,ompn,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=-1.0  aibj(fwd), eqfr(fwd), ompn(eq_sink), ckdl(fwd), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,ompn,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=+1.0  aqbr(fwd), emfn(eq_sink), oipj(fwd), cgdh(fwd), sktl(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,emfn,oipj,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=-1.0  aqbr(fwd), emfn(eq_sink), oipj(fwd), ckdl(fwd), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,emfn,oipj,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=-1.0  aqbr(fwd), eifj(fwd), ompn(eq_sink), cgdh(fwd), sktl(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,ompn,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=+1.0  aqbr(fwd), eifj(fwd), ompn(eq_sink), ckdl(fwd), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,ompn,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # Group 12, CG=1.0
            # # diag0 sign=+1.0  ambn(eq_sink), eifj(fwd), okpl(fwd), sqtr(eq_src), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eifj,okpl,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  ambn(eq_sink), eifj(fwd), oqpr(fwd), sktl(eq_src), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,eifj,oqpr,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  ambn(eq_sink), ekfl(fwd), oipj(fwd), sqtr(eq_src), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,ekfl,oipj,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  ambn(eq_sink), ekfl(fwd), oqpr(fwd), sitj(eq_src), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,ekfl,oqpr,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  ambn(eq_sink), eqfr(fwd), oipj(fwd), sktl(eq_src), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eqfr,oipj,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  ambn(eq_sink), eqfr(fwd), okpl(fwd), sitj(eq_src), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,eqfr,okpl,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=-1.0  aibj(fwd), emfn(eq_sink), okpl(fwd), sqtr(eq_src), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,emfn,okpl,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=+1.0  aibj(fwd), emfn(eq_sink), oqpr(fwd), sktl(eq_src), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,emfn,oqpr,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=+1.0  aibj(fwd), ekfl(fwd), ompn(eq_sink), sqtr(eq_src), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,ompn,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=-1.0  aibj(fwd), ekfl(fwd), oqpr(fwd), smtn(bwd), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,oqpr,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=-1.0  aibj(fwd), eqfr(fwd), ompn(eq_sink), sktl(eq_src), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,ompn,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=+1.0  aibj(fwd), eqfr(fwd), okpl(fwd), smtn(bwd), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,okpl,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag12 sign=+1.0  akbl(fwd), emfn(eq_sink), oipj(fwd), sqtr(eq_src), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,emfn,oipj,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag13 sign=-1.0  akbl(fwd), emfn(eq_sink), oqpr(fwd), sitj(eq_src), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,emfn,oqpr,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag14 sign=-1.0  akbl(fwd), eifj(fwd), ompn(eq_sink), sqtr(eq_src), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,ompn,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag15 sign=+1.0  akbl(fwd), eifj(fwd), oqpr(fwd), smtn(bwd), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,oqpr,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag16 sign=+1.0  akbl(fwd), eqfr(fwd), ompn(eq_sink), sitj(eq_src), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eqfr,ompn,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag17 sign=-1.0  akbl(fwd), eqfr(fwd), oipj(fwd), smtn(bwd), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eqfr,oipj,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag18 sign=-1.0  aqbr(fwd), emfn(eq_sink), oipj(fwd), sktl(eq_src), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,emfn,oipj,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag19 sign=+1.0  aqbr(fwd), emfn(eq_sink), okpl(fwd), sitj(eq_src), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,emfn,okpl,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag20 sign=+1.0  aqbr(fwd), eifj(fwd), ompn(eq_sink), sktl(eq_src), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,ompn,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag21 sign=-1.0  aqbr(fwd), eifj(fwd), okpl(fwd), smtn(bwd), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,okpl,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag22 sign=-1.0  aqbr(fwd), ekfl(fwd), ompn(eq_sink), sitj(eq_src), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,ekfl,ompn,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag23 sign=+1.0  aqbr(fwd), ekfl(fwd), oipj(fwd), smtn(bwd), cgdh(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,ekfl,oipj,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # Group 13, CG=1.0
            # # diag0 sign=-1.0  ambn(eq_sink), eifj(fwd), okpl(fwd), cgdh(fwd), sqtr(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,eifj,okpl,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=+1.0  ambn(eq_sink), eifj(fwd), okpl(fwd), cqdr(fwd), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eifj,okpl,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=+1.0  ambn(eq_sink), ekfl(fwd), oipj(fwd), cgdh(fwd), sqtr(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,ekfl,oipj,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=-1.0  ambn(eq_sink), ekfl(fwd), oipj(fwd), cqdr(fwd), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,ekfl,oipj,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  aibj(fwd), emfn(eq_sink), okpl(fwd), cgdh(fwd), sqtr(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,emfn,okpl,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  aibj(fwd), emfn(eq_sink), okpl(fwd), cqdr(fwd), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,emfn,okpl,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=-1.0  aibj(fwd), ekfl(fwd), ompn(eq_sink), cgdh(fwd), sqtr(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,ompn,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=+1.0  aibj(fwd), ekfl(fwd), ompn(eq_sink), cqdr(fwd), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,ompn,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=-1.0  akbl(fwd), emfn(eq_sink), oipj(fwd), cgdh(fwd), sqtr(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,emfn,oipj,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=+1.0  akbl(fwd), emfn(eq_sink), oipj(fwd), cqdr(fwd), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,emfn,oipj,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=+1.0  akbl(fwd), eifj(fwd), ompn(eq_sink), cgdh(fwd), sqtr(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,ompn,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=-1.0  akbl(fwd), eifj(fwd), ompn(eq_sink), cqdr(fwd), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,ompn,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # Group 14, CG=1.0
            # # diag0 sign=+1.0  aibj(fwd), ekfl(fwd), sqtr(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,sqtr,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  aibj(fwd), ekfl(fwd), sqtr(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,sqtr,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  aibj(fwd), eqfr(fwd), sktl(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,sktl,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  aibj(fwd), eqfr(fwd), sktl(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,sktl,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=-1.0  akbl(fwd), eifj(fwd), sqtr(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,sqtr,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=+1.0  akbl(fwd), eifj(fwd), sqtr(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,sqtr,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=+1.0  akbl(fwd), eqfr(fwd), sitj(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eqfr,sitj,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=-1.0  akbl(fwd), eqfr(fwd), sitj(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eqfr,sitj,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=+1.0  aqbr(fwd), eifj(fwd), sktl(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,sktl,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=-1.0  aqbr(fwd), eifj(fwd), sktl(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,sktl,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=-1.0  aqbr(fwd), ekfl(fwd), sitj(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,ekfl,sitj,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=+1.0  aqbr(fwd), ekfl(fwd), sitj(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,ekfl,sitj,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # Group 15, CG=1.0
            # # diag0 sign=-1.0  aibj(fwd), ekfl(fwd), cmdn(eq_sink), ogph(fwd), sqtr(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,cmdn,ogph,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=+1.0  aibj(fwd), ekfl(fwd), cmdn(eq_sink), oqpr(fwd), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,cmdn,oqpr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=+1.0  aibj(fwd), ekfl(fwd), cgdh(fwd), ompn(eq_sink), sqtr(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,cgdh,ompn,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=-1.0  aibj(fwd), ekfl(fwd), cgdh(fwd), oqpr(fwd), smtn(bwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,cgdh,oqpr,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=-1.0  aibj(fwd), ekfl(fwd), cqdr(fwd), ompn(eq_sink), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,cqdr,ompn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=+1.0  aibj(fwd), ekfl(fwd), cqdr(fwd), ogph(fwd), smtn(bwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,cqdr,ogph,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=+1.0  akbl(fwd), eifj(fwd), cmdn(eq_sink), ogph(fwd), sqtr(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,cmdn,ogph,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=-1.0  akbl(fwd), eifj(fwd), cmdn(eq_sink), oqpr(fwd), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,cmdn,oqpr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=-1.0  akbl(fwd), eifj(fwd), cgdh(fwd), ompn(eq_sink), sqtr(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,cgdh,ompn,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=+1.0  akbl(fwd), eifj(fwd), cgdh(fwd), oqpr(fwd), smtn(bwd)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,cgdh,oqpr,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=+1.0  akbl(fwd), eifj(fwd), cqdr(fwd), ompn(eq_sink), sgth(eq_src)
            # corr[1,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,cqdr,ompn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=-1.0  akbl(fwd), eifj(fwd), cqdr(fwd), ogph(fwd), smtn(bwd)
            # corr[1,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,cqdr,ogph,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVVK, snkVDV0, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # ==== M^{3,1} ====
            # # Group 4, CG=1.0
            # # diag0 sign=+1.0  aibj(fwd), ekfl(fwd), cmdn(eq_sink), ogph(fwd)
            # corr[2,0,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,cmdn,ogph,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, snkVVV0, snkVDVK, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  aibj(fwd), ekfl(fwd), cgdh(fwd), ompn(eq_sink)
            # corr[2,0,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,cgdh,ompn,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, snkVVV0, snkVDVK, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  akbl(fwd), eifj(fwd), cmdn(eq_sink), ogph(fwd)
            # corr[2,0,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,cmdn,ogph,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, snkVVV0, snkVDVK, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  akbl(fwd), eifj(fwd), cgdh(fwd), ompn(eq_sink)
            # corr[2,0,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,cgdh,ompn,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, snkVVV0, snkVDVK, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # Group 5, CG=1.0
            # # diag0 sign=+1.0  ambn(eq_sink), eifj(fwd), okpl(fwd), cgdh(fwd)
            # corr[2,0,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eifj,okpl,cgdh,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, snkVVV0, snkVDVK, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  ambn(eq_sink), ekfl(fwd), oipj(fwd), cgdh(fwd)
            # corr[2,0,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,ekfl,oipj,cgdh,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, snkVVV0, snkVDVK, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  aibj(fwd), emfn(eq_sink), okpl(fwd), cgdh(fwd)
            # corr[2,0,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,emfn,okpl,cgdh,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, snkVVV0, snkVDVK, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  aibj(fwd), ekfl(fwd), ompn(eq_sink), cgdh(fwd)
            # corr[2,0,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,ompn,cgdh,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, snkVVV0, snkVDVK, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  akbl(fwd), emfn(eq_sink), oipj(fwd), cgdh(fwd)
            # corr[2,0,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,emfn,oipj,cgdh,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, snkVVV0, snkVDVK, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  akbl(fwd), eifj(fwd), ompn(eq_sink), cgdh(fwd)
            # corr[2,0,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,ompn,cgdh,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, snkVVV0, snkVDVK, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # Group 6, CG=1.0
            # # diag0 sign=-1.0  eifj(fwd), okpl(fwd), ambn(eq_sink), cgdh(fwd)
            # corr[2,0,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,okpl,ambn,cgdh,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, snkVVV0, snkVDVK, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=+1.0  eifj(fwd), okpl(fwd), agbh(fwd), cmdn(eq_sink)
            # corr[2,0,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,okpl,agbh,cmdn,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, snkVVV0, snkVDVK, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=+1.0  ekfl(fwd), oipj(fwd), ambn(eq_sink), cgdh(fwd)
            # corr[2,0,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oipj,ambn,cgdh,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, snkVVV0, snkVDVK, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=-1.0  ekfl(fwd), oipj(fwd), agbh(fwd), cmdn(eq_sink)
            # corr[2,0,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oipj,agbh,cmdn,ce,mo,gi,Kbdf,Knp,Khjl,ay,kz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, snkVVV0, snkVDVK, srcVVVK, projection, projection, optimize=['greedy', 'dp'])
            # # ==== M^{3,2} ====
            # # Group 7, CG=1.0
            # # diag0 sign=-1.0  eifj(fwd), oqpr(fwd), ambn(eq_sink), cgdh(fwd), sktl(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,oqpr,ambn,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=+1.0  eifj(fwd), oqpr(fwd), ambn(eq_sink), ckdl(fwd), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,oqpr,ambn,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=+1.0  eifj(fwd), oqpr(fwd), agbh(fwd), cmdn(eq_sink), sktl(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,oqpr,agbh,cmdn,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=-1.0  eifj(fwd), oqpr(fwd), agbh(fwd), ckdl(fwd), smtn(bwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,oqpr,agbh,ckdl,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=-1.0  eifj(fwd), oqpr(fwd), akbl(fwd), cmdn(eq_sink), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,oqpr,akbl,cmdn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=+1.0  eifj(fwd), oqpr(fwd), akbl(fwd), cgdh(fwd), smtn(bwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,oqpr,akbl,cgdh,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=+1.0  eqfr(fwd), oipj(fwd), ambn(eq_sink), cgdh(fwd), sktl(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eqfr,oipj,ambn,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=-1.0  eqfr(fwd), oipj(fwd), ambn(eq_sink), ckdl(fwd), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eqfr,oipj,ambn,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=-1.0  eqfr(fwd), oipj(fwd), agbh(fwd), cmdn(eq_sink), sktl(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eqfr,oipj,agbh,cmdn,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=+1.0  eqfr(fwd), oipj(fwd), agbh(fwd), ckdl(fwd), smtn(bwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eqfr,oipj,agbh,ckdl,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=+1.0  eqfr(fwd), oipj(fwd), akbl(fwd), cmdn(eq_sink), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eqfr,oipj,akbl,cmdn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=-1.0  eqfr(fwd), oipj(fwd), akbl(fwd), cgdh(fwd), smtn(bwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eqfr,oipj,akbl,cgdh,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # Group 8, CG=1.0
            # # diag0 sign=-1.0  eifj(fwd), okpl(fwd), sqtr(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,okpl,sqtr,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=+1.0  eifj(fwd), okpl(fwd), sqtr(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,okpl,sqtr,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=+1.0  eifj(fwd), oqpr(fwd), sktl(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,oqpr,sktl,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=-1.0  eifj(fwd), oqpr(fwd), sktl(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,oqpr,sktl,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  ekfl(fwd), oipj(fwd), sqtr(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oipj,sqtr,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  ekfl(fwd), oipj(fwd), sqtr(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oipj,sqtr,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=-1.0  ekfl(fwd), oqpr(fwd), sitj(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oqpr,sitj,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=+1.0  ekfl(fwd), oqpr(fwd), sitj(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oqpr,sitj,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=-1.0  eqfr(fwd), oipj(fwd), sktl(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eqfr,oipj,sktl,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=+1.0  eqfr(fwd), oipj(fwd), sktl(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eqfr,oipj,sktl,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=+1.0  eqfr(fwd), okpl(fwd), sitj(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eqfr,okpl,sitj,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=-1.0  eqfr(fwd), okpl(fwd), sitj(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eqfr,okpl,sitj,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])

            # # Group 9, CG=1.0
            # # diag0 sign=+1.0  eifj(fwd), okpl(fwd), ambn(eq_sink), cgdh(fwd), sqtr(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,okpl,ambn,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  eifj(fwd), okpl(fwd), ambn(eq_sink), cqdr(fwd), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,okpl,ambn,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  eifj(fwd), okpl(fwd), agbh(fwd), cmdn(eq_sink), sqtr(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,okpl,agbh,cmdn,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  eifj(fwd), okpl(fwd), agbh(fwd), cqdr(fwd), smtn(bwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,okpl,agbh,cqdr,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  eifj(fwd), okpl(fwd), aqbr(fwd), cmdn(eq_sink), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,okpl,aqbr,cmdn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  eifj(fwd), okpl(fwd), aqbr(fwd), cgdh(fwd), smtn(bwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,okpl,aqbr,cgdh,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=-1.0  ekfl(fwd), oipj(fwd), ambn(eq_sink), cgdh(fwd), sqtr(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oipj,ambn,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=+1.0  ekfl(fwd), oipj(fwd), ambn(eq_sink), cqdr(fwd), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oipj,ambn,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=+1.0  ekfl(fwd), oipj(fwd), agbh(fwd), cmdn(eq_sink), sqtr(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oipj,agbh,cmdn,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=-1.0  ekfl(fwd), oipj(fwd), agbh(fwd), cqdr(fwd), smtn(bwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oipj,agbh,cqdr,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=-1.0  ekfl(fwd), oipj(fwd), aqbr(fwd), cmdn(eq_sink), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oipj,aqbr,cmdn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=+1.0  ekfl(fwd), oipj(fwd), aqbr(fwd), cgdh(fwd), smtn(bwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oipj,aqbr,cgdh,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # Group 10, CG=1.0
            # # diag0 sign=+1.0  aibj(fwd), eqfr(fwd), cmdn(eq_sink), ogph(fwd), sktl(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,cmdn,ogph,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  aibj(fwd), eqfr(fwd), cmdn(eq_sink), okpl(fwd), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,cmdn,okpl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  aibj(fwd), eqfr(fwd), cgdh(fwd), ompn(eq_sink), sktl(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,cgdh,ompn,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  aibj(fwd), eqfr(fwd), cgdh(fwd), okpl(fwd), smtn(bwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,cgdh,okpl,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  aibj(fwd), eqfr(fwd), ckdl(fwd), ompn(eq_sink), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,ckdl,ompn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  aibj(fwd), eqfr(fwd), ckdl(fwd), ogph(fwd), smtn(bwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,ckdl,ogph,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=-1.0  aqbr(fwd), eifj(fwd), cmdn(eq_sink), ogph(fwd), sktl(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,cmdn,ogph,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=+1.0  aqbr(fwd), eifj(fwd), cmdn(eq_sink), okpl(fwd), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,cmdn,okpl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=+1.0  aqbr(fwd), eifj(fwd), cgdh(fwd), ompn(eq_sink), sktl(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,cgdh,ompn,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=-1.0  aqbr(fwd), eifj(fwd), cgdh(fwd), okpl(fwd), smtn(bwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,cgdh,okpl,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=-1.0  aqbr(fwd), eifj(fwd), ckdl(fwd), ompn(eq_sink), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,ckdl,ompn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=+1.0  aqbr(fwd), eifj(fwd), ckdl(fwd), ogph(fwd), smtn(bwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,ckdl,ogph,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # Group 11, CG=1.0
            # # diag0 sign=+1.0  ambn(eq_sink), eifj(fwd), oqpr(fwd), cgdh(fwd), sktl(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eifj,oqpr,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  ambn(eq_sink), eifj(fwd), oqpr(fwd), ckdl(fwd), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,eifj,oqpr,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  ambn(eq_sink), eqfr(fwd), oipj(fwd), cgdh(fwd), sktl(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,eqfr,oipj,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  ambn(eq_sink), eqfr(fwd), oipj(fwd), ckdl(fwd), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eqfr,oipj,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=-1.0  aibj(fwd), emfn(eq_sink), oqpr(fwd), cgdh(fwd), sktl(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,emfn,oqpr,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=+1.0  aibj(fwd), emfn(eq_sink), oqpr(fwd), ckdl(fwd), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,emfn,oqpr,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=+1.0  aibj(fwd), eqfr(fwd), ompn(eq_sink), cgdh(fwd), sktl(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,ompn,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=-1.0  aibj(fwd), eqfr(fwd), ompn(eq_sink), ckdl(fwd), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,ompn,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=+1.0  aqbr(fwd), emfn(eq_sink), oipj(fwd), cgdh(fwd), sktl(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,emfn,oipj,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=-1.0  aqbr(fwd), emfn(eq_sink), oipj(fwd), ckdl(fwd), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,emfn,oipj,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=-1.0  aqbr(fwd), eifj(fwd), ompn(eq_sink), cgdh(fwd), sktl(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,ompn,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=+1.0  aqbr(fwd), eifj(fwd), ompn(eq_sink), ckdl(fwd), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,ompn,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # Group 12, CG=1.0
            # # diag0 sign=+1.0  ambn(eq_sink), eifj(fwd), okpl(fwd), sqtr(eq_src), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eifj,okpl,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  ambn(eq_sink), eifj(fwd), oqpr(fwd), sktl(eq_src), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,eifj,oqpr,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  ambn(eq_sink), ekfl(fwd), oipj(fwd), sqtr(eq_src), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,ekfl,oipj,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  ambn(eq_sink), ekfl(fwd), oqpr(fwd), sitj(eq_src), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,ekfl,oqpr,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  ambn(eq_sink), eqfr(fwd), oipj(fwd), sktl(eq_src), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eqfr,oipj,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  ambn(eq_sink), eqfr(fwd), okpl(fwd), sitj(eq_src), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,eqfr,okpl,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=-1.0  aibj(fwd), emfn(eq_sink), okpl(fwd), sqtr(eq_src), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,emfn,okpl,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=+1.0  aibj(fwd), emfn(eq_sink), oqpr(fwd), sktl(eq_src), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,emfn,oqpr,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=+1.0  aibj(fwd), ekfl(fwd), ompn(eq_sink), sqtr(eq_src), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,ompn,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=-1.0  aibj(fwd), ekfl(fwd), oqpr(fwd), smtn(bwd), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,oqpr,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=-1.0  aibj(fwd), eqfr(fwd), ompn(eq_sink), sktl(eq_src), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,ompn,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=+1.0  aibj(fwd), eqfr(fwd), okpl(fwd), smtn(bwd), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,okpl,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag12 sign=+1.0  akbl(fwd), emfn(eq_sink), oipj(fwd), sqtr(eq_src), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,emfn,oipj,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag13 sign=-1.0  akbl(fwd), emfn(eq_sink), oqpr(fwd), sitj(eq_src), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,emfn,oqpr,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag14 sign=-1.0  akbl(fwd), eifj(fwd), ompn(eq_sink), sqtr(eq_src), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,ompn,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag15 sign=+1.0  akbl(fwd), eifj(fwd), oqpr(fwd), smtn(bwd), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,oqpr,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag16 sign=+1.0  akbl(fwd), eqfr(fwd), ompn(eq_sink), sitj(eq_src), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eqfr,ompn,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag17 sign=-1.0  akbl(fwd), eqfr(fwd), oipj(fwd), smtn(bwd), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eqfr,oipj,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag18 sign=-1.0  aqbr(fwd), emfn(eq_sink), oipj(fwd), sktl(eq_src), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,emfn,oipj,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag19 sign=+1.0  aqbr(fwd), emfn(eq_sink), okpl(fwd), sitj(eq_src), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,emfn,okpl,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag20 sign=+1.0  aqbr(fwd), eifj(fwd), ompn(eq_sink), sktl(eq_src), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,ompn,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag21 sign=-1.0  aqbr(fwd), eifj(fwd), okpl(fwd), smtn(bwd), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,okpl,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag22 sign=-1.0  aqbr(fwd), ekfl(fwd), ompn(eq_sink), sitj(eq_src), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,ekfl,ompn,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag23 sign=+1.0  aqbr(fwd), ekfl(fwd), oipj(fwd), smtn(bwd), cgdh(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,ekfl,oipj,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # Group 13, CG=1.0
            # # diag0 sign=-1.0  ambn(eq_sink), eifj(fwd), okpl(fwd), cgdh(fwd), sqtr(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,eifj,okpl,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=+1.0  ambn(eq_sink), eifj(fwd), okpl(fwd), cqdr(fwd), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eifj,okpl,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=+1.0  ambn(eq_sink), ekfl(fwd), oipj(fwd), cgdh(fwd), sqtr(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,ekfl,oipj,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=-1.0  ambn(eq_sink), ekfl(fwd), oipj(fwd), cqdr(fwd), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,ekfl,oipj,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  aibj(fwd), emfn(eq_sink), okpl(fwd), cgdh(fwd), sqtr(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,emfn,okpl,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  aibj(fwd), emfn(eq_sink), okpl(fwd), cqdr(fwd), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,emfn,okpl,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=-1.0  aibj(fwd), ekfl(fwd), ompn(eq_sink), cgdh(fwd), sqtr(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,ompn,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=+1.0  aibj(fwd), ekfl(fwd), ompn(eq_sink), cqdr(fwd), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,ompn,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=-1.0  akbl(fwd), emfn(eq_sink), oipj(fwd), cgdh(fwd), sqtr(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,emfn,oipj,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=+1.0  akbl(fwd), emfn(eq_sink), oipj(fwd), cqdr(fwd), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,emfn,oipj,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=+1.0  akbl(fwd), eifj(fwd), ompn(eq_sink), cgdh(fwd), sqtr(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,ompn,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=-1.0  akbl(fwd), eifj(fwd), ompn(eq_sink), cqdr(fwd), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,ompn,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # Group 14, CG=1.0
            # # diag0 sign=+1.0  aibj(fwd), ekfl(fwd), sqtr(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,sqtr,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  aibj(fwd), ekfl(fwd), sqtr(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,sqtr,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  aibj(fwd), eqfr(fwd), sktl(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,sktl,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  aibj(fwd), eqfr(fwd), sktl(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,sktl,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=-1.0  akbl(fwd), eifj(fwd), sqtr(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,sqtr,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=+1.0  akbl(fwd), eifj(fwd), sqtr(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,sqtr,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=+1.0  akbl(fwd), eqfr(fwd), sitj(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eqfr,sitj,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=-1.0  akbl(fwd), eqfr(fwd), sitj(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eqfr,sitj,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=+1.0  aqbr(fwd), eifj(fwd), sktl(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,sktl,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=-1.0  aqbr(fwd), eifj(fwd), sktl(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,sktl,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=-1.0  aqbr(fwd), ekfl(fwd), sitj(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,ekfl,sitj,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=+1.0  aqbr(fwd), ekfl(fwd), sitj(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,ekfl,sitj,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # Group 15, CG=1.0
            # # diag0 sign=-1.0  aibj(fwd), ekfl(fwd), cmdn(eq_sink), ogph(fwd), sqtr(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,cmdn,ogph,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=+1.0  aibj(fwd), ekfl(fwd), cmdn(eq_sink), oqpr(fwd), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,cmdn,oqpr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=+1.0  aibj(fwd), ekfl(fwd), cgdh(fwd), ompn(eq_sink), sqtr(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,cgdh,ompn,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=-1.0  aibj(fwd), ekfl(fwd), cgdh(fwd), oqpr(fwd), smtn(bwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,cgdh,oqpr,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=-1.0  aibj(fwd), ekfl(fwd), cqdr(fwd), ompn(eq_sink), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,cqdr,ompn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=+1.0  aibj(fwd), ekfl(fwd), cqdr(fwd), ogph(fwd), smtn(bwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,cqdr,ogph,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=+1.0  akbl(fwd), eifj(fwd), cmdn(eq_sink), ogph(fwd), sqtr(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,cmdn,ogph,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=-1.0  akbl(fwd), eifj(fwd), cmdn(eq_sink), oqpr(fwd), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,cmdn,oqpr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=-1.0  akbl(fwd), eifj(fwd), cgdh(fwd), ompn(eq_sink), sqtr(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,cgdh,ompn,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=+1.0  akbl(fwd), eifj(fwd), cgdh(fwd), oqpr(fwd), smtn(bwd)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,cgdh,oqpr,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=+1.0  akbl(fwd), eifj(fwd), cqdr(fwd), ompn(eq_sink), sgth(eq_src)
            # corr[2,1,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,cqdr,ompn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=-1.0  akbl(fwd), eifj(fwd), cqdr(fwd), ogph(fwd), smtn(bwd)
            # corr[2,1,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,cqdr,ogph,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVVK, srcVDV0, projection, projection, optimize=['greedy', 'dp'])
            # # ==== M^{3,3} ====
            # # Group 7, CG=1.0
            # # diag0 sign=-1.0  eifj(fwd), oqpr(fwd), ambn(eq_sink), cgdh(fwd), sktl(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,oqpr,ambn,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=+1.0  eifj(fwd), oqpr(fwd), ambn(eq_sink), ckdl(fwd), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,oqpr,ambn,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=+1.0  eifj(fwd), oqpr(fwd), agbh(fwd), cmdn(eq_sink), sktl(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,oqpr,agbh,cmdn,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=-1.0  eifj(fwd), oqpr(fwd), agbh(fwd), ckdl(fwd), smtn(bwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,oqpr,agbh,ckdl,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=-1.0  eifj(fwd), oqpr(fwd), akbl(fwd), cmdn(eq_sink), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,oqpr,akbl,cmdn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=+1.0  eifj(fwd), oqpr(fwd), akbl(fwd), cgdh(fwd), smtn(bwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,oqpr,akbl,cgdh,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=+1.0  eqfr(fwd), oipj(fwd), ambn(eq_sink), cgdh(fwd), sktl(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eqfr,oipj,ambn,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=-1.0  eqfr(fwd), oipj(fwd), ambn(eq_sink), ckdl(fwd), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eqfr,oipj,ambn,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=-1.0  eqfr(fwd), oipj(fwd), agbh(fwd), cmdn(eq_sink), sktl(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eqfr,oipj,agbh,cmdn,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=+1.0  eqfr(fwd), oipj(fwd), agbh(fwd), ckdl(fwd), smtn(bwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eqfr,oipj,agbh,ckdl,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=+1.0  eqfr(fwd), oipj(fwd), akbl(fwd), cmdn(eq_sink), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eqfr,oipj,akbl,cmdn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=-1.0  eqfr(fwd), oipj(fwd), akbl(fwd), cgdh(fwd), smtn(bwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eqfr,oipj,akbl,cgdh,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # Group 8, CG=1.0
            # # diag0 sign=-1.0  eifj(fwd), okpl(fwd), sqtr(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,okpl,sqtr,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=+1.0  eifj(fwd), okpl(fwd), sqtr(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,okpl,sqtr,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=+1.0  eifj(fwd), oqpr(fwd), sktl(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,oqpr,sktl,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=-1.0  eifj(fwd), oqpr(fwd), sktl(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,oqpr,sktl,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  ekfl(fwd), oipj(fwd), sqtr(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oipj,sqtr,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  ekfl(fwd), oipj(fwd), sqtr(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oipj,sqtr,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=-1.0  ekfl(fwd), oqpr(fwd), sitj(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oqpr,sitj,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=+1.0  ekfl(fwd), oqpr(fwd), sitj(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oqpr,sitj,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=-1.0  eqfr(fwd), oipj(fwd), sktl(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eqfr,oipj,sktl,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=+1.0  eqfr(fwd), oipj(fwd), sktl(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eqfr,oipj,sktl,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=+1.0  eqfr(fwd), okpl(fwd), sitj(eq_src), ambn(eq_sink), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eqfr,okpl,sitj,ambn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=-1.0  eqfr(fwd), okpl(fwd), sitj(eq_src), agbh(fwd), cmdn(eq_sink)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eqfr,okpl,sitj,agbh,cmdn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])

            # # Group 9, CG=1.0
            # # diag0 sign=+1.0  eifj(fwd), okpl(fwd), ambn(eq_sink), cgdh(fwd), sqtr(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,okpl,ambn,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  eifj(fwd), okpl(fwd), ambn(eq_sink), cqdr(fwd), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,okpl,ambn,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  eifj(fwd), okpl(fwd), agbh(fwd), cmdn(eq_sink), sqtr(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,okpl,agbh,cmdn,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  eifj(fwd), okpl(fwd), agbh(fwd), cqdr(fwd), smtn(bwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,okpl,agbh,cqdr,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  eifj(fwd), okpl(fwd), aqbr(fwd), cmdn(eq_sink), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'eifj,okpl,aqbr,cmdn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  eifj(fwd), okpl(fwd), aqbr(fwd), cgdh(fwd), smtn(bwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'eifj,okpl,aqbr,cgdh,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=-1.0  ekfl(fwd), oipj(fwd), ambn(eq_sink), cgdh(fwd), sqtr(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oipj,ambn,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=+1.0  ekfl(fwd), oipj(fwd), ambn(eq_sink), cqdr(fwd), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oipj,ambn,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=+1.0  ekfl(fwd), oipj(fwd), agbh(fwd), cmdn(eq_sink), sqtr(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oipj,agbh,cmdn,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=-1.0  ekfl(fwd), oipj(fwd), agbh(fwd), cqdr(fwd), smtn(bwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oipj,agbh,cqdr,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=-1.0  ekfl(fwd), oipj(fwd), aqbr(fwd), cmdn(eq_sink), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ekfl,oipj,aqbr,cmdn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=+1.0  ekfl(fwd), oipj(fwd), aqbr(fwd), cgdh(fwd), smtn(bwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ekfl,oipj,aqbr,cgdh,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # Group 10, CG=1.0
            # # diag0 sign=+1.0  aibj(fwd), eqfr(fwd), cmdn(eq_sink), ogph(fwd), sktl(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,cmdn,ogph,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  aibj(fwd), eqfr(fwd), cmdn(eq_sink), okpl(fwd), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,cmdn,okpl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  aibj(fwd), eqfr(fwd), cgdh(fwd), ompn(eq_sink), sktl(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,cgdh,ompn,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  aibj(fwd), eqfr(fwd), cgdh(fwd), okpl(fwd), smtn(bwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,cgdh,okpl,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  aibj(fwd), eqfr(fwd), ckdl(fwd), ompn(eq_sink), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,ckdl,ompn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  aibj(fwd), eqfr(fwd), ckdl(fwd), ogph(fwd), smtn(bwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,ckdl,ogph,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=-1.0  aqbr(fwd), eifj(fwd), cmdn(eq_sink), ogph(fwd), sktl(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,cmdn,ogph,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=+1.0  aqbr(fwd), eifj(fwd), cmdn(eq_sink), okpl(fwd), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,cmdn,okpl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=+1.0  aqbr(fwd), eifj(fwd), cgdh(fwd), ompn(eq_sink), sktl(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,cgdh,ompn,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=-1.0  aqbr(fwd), eifj(fwd), cgdh(fwd), okpl(fwd), smtn(bwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,cgdh,okpl,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=-1.0  aqbr(fwd), eifj(fwd), ckdl(fwd), ompn(eq_sink), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,ckdl,ompn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=+1.0  aqbr(fwd), eifj(fwd), ckdl(fwd), ogph(fwd), smtn(bwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,ckdl,ogph,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # Group 11, CG=1.0
            # # diag0 sign=+1.0  ambn(eq_sink), eifj(fwd), oqpr(fwd), cgdh(fwd), sktl(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eifj,oqpr,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  ambn(eq_sink), eifj(fwd), oqpr(fwd), ckdl(fwd), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,eifj,oqpr,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  ambn(eq_sink), eqfr(fwd), oipj(fwd), cgdh(fwd), sktl(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,eqfr,oipj,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  ambn(eq_sink), eqfr(fwd), oipj(fwd), ckdl(fwd), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eqfr,oipj,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=-1.0  aibj(fwd), emfn(eq_sink), oqpr(fwd), cgdh(fwd), sktl(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,emfn,oqpr,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=+1.0  aibj(fwd), emfn(eq_sink), oqpr(fwd), ckdl(fwd), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,emfn,oqpr,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=+1.0  aibj(fwd), eqfr(fwd), ompn(eq_sink), cgdh(fwd), sktl(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,ompn,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=-1.0  aibj(fwd), eqfr(fwd), ompn(eq_sink), ckdl(fwd), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,ompn,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=+1.0  aqbr(fwd), emfn(eq_sink), oipj(fwd), cgdh(fwd), sktl(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,emfn,oipj,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=-1.0  aqbr(fwd), emfn(eq_sink), oipj(fwd), ckdl(fwd), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,emfn,oipj,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=-1.0  aqbr(fwd), eifj(fwd), ompn(eq_sink), cgdh(fwd), sktl(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,ompn,cgdh,sktl,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=+1.0  aqbr(fwd), eifj(fwd), ompn(eq_sink), ckdl(fwd), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,ompn,ckdl,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # Group 12, CG=1.0
            # # diag0 sign=+1.0  ambn(eq_sink), eifj(fwd), okpl(fwd), sqtr(eq_src), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eifj,okpl,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  ambn(eq_sink), eifj(fwd), oqpr(fwd), sktl(eq_src), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,eifj,oqpr,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  ambn(eq_sink), ekfl(fwd), oipj(fwd), sqtr(eq_src), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,ekfl,oipj,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  ambn(eq_sink), ekfl(fwd), oqpr(fwd), sitj(eq_src), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,ekfl,oqpr,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  ambn(eq_sink), eqfr(fwd), oipj(fwd), sktl(eq_src), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eqfr,oipj,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  ambn(eq_sink), eqfr(fwd), okpl(fwd), sitj(eq_src), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,eqfr,okpl,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=-1.0  aibj(fwd), emfn(eq_sink), okpl(fwd), sqtr(eq_src), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,emfn,okpl,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=+1.0  aibj(fwd), emfn(eq_sink), oqpr(fwd), sktl(eq_src), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,emfn,oqpr,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=+1.0  aibj(fwd), ekfl(fwd), ompn(eq_sink), sqtr(eq_src), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,ompn,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=-1.0  aibj(fwd), ekfl(fwd), oqpr(fwd), smtn(bwd), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,oqpr,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=-1.0  aibj(fwd), eqfr(fwd), ompn(eq_sink), sktl(eq_src), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,ompn,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=+1.0  aibj(fwd), eqfr(fwd), okpl(fwd), smtn(bwd), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,okpl,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag12 sign=+1.0  akbl(fwd), emfn(eq_sink), oipj(fwd), sqtr(eq_src), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,emfn,oipj,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag13 sign=-1.0  akbl(fwd), emfn(eq_sink), oqpr(fwd), sitj(eq_src), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,emfn,oqpr,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag14 sign=-1.0  akbl(fwd), eifj(fwd), ompn(eq_sink), sqtr(eq_src), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,ompn,sqtr,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag15 sign=+1.0  akbl(fwd), eifj(fwd), oqpr(fwd), smtn(bwd), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,oqpr,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag16 sign=+1.0  akbl(fwd), eqfr(fwd), ompn(eq_sink), sitj(eq_src), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eqfr,ompn,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag17 sign=-1.0  akbl(fwd), eqfr(fwd), oipj(fwd), smtn(bwd), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eqfr,oipj,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag18 sign=-1.0  aqbr(fwd), emfn(eq_sink), oipj(fwd), sktl(eq_src), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,emfn,oipj,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag19 sign=+1.0  aqbr(fwd), emfn(eq_sink), okpl(fwd), sitj(eq_src), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,emfn,okpl,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag20 sign=+1.0  aqbr(fwd), eifj(fwd), ompn(eq_sink), sktl(eq_src), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,ompn,sktl,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag21 sign=-1.0  aqbr(fwd), eifj(fwd), okpl(fwd), smtn(bwd), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,okpl,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag22 sign=-1.0  aqbr(fwd), ekfl(fwd), ompn(eq_sink), sitj(eq_src), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,ekfl,ompn,sitj,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag23 sign=+1.0  aqbr(fwd), ekfl(fwd), oipj(fwd), smtn(bwd), cgdh(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,ekfl,oipj,smtn,cgdh,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # Group 13, CG=1.0
            # # diag0 sign=-1.0  ambn(eq_sink), eifj(fwd), okpl(fwd), cgdh(fwd), sqtr(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,eifj,okpl,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=+1.0  ambn(eq_sink), eifj(fwd), okpl(fwd), cqdr(fwd), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,eifj,okpl,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=+1.0  ambn(eq_sink), ekfl(fwd), oipj(fwd), cgdh(fwd), sqtr(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'ambn,ekfl,oipj,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=-1.0  ambn(eq_sink), ekfl(fwd), oipj(fwd), cqdr(fwd), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'ambn,ekfl,oipj,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=+1.0  aibj(fwd), emfn(eq_sink), okpl(fwd), cgdh(fwd), sqtr(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,emfn,okpl,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=-1.0  aibj(fwd), emfn(eq_sink), okpl(fwd), cqdr(fwd), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,emfn,okpl,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=-1.0  aibj(fwd), ekfl(fwd), ompn(eq_sink), cgdh(fwd), sqtr(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,ompn,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=+1.0  aibj(fwd), ekfl(fwd), ompn(eq_sink), cqdr(fwd), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,ompn,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=-1.0  akbl(fwd), emfn(eq_sink), oipj(fwd), cgdh(fwd), sqtr(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,emfn,oipj,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=+1.0  akbl(fwd), emfn(eq_sink), oipj(fwd), cqdr(fwd), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,emfn,oipj,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=+1.0  akbl(fwd), eifj(fwd), ompn(eq_sink), cgdh(fwd), sqtr(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,ompn,cgdh,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=-1.0  akbl(fwd), eifj(fwd), ompn(eq_sink), cqdr(fwd), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,ompn,cqdr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # Group 14, CG=1.0
            # # diag0 sign=+1.0  aibj(fwd), ekfl(fwd), sqtr(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,sqtr,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=-1.0  aibj(fwd), ekfl(fwd), sqtr(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,sqtr,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=-1.0  aibj(fwd), eqfr(fwd), sktl(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,eqfr,sktl,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=+1.0  aibj(fwd), eqfr(fwd), sktl(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,eqfr,sktl,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=-1.0  akbl(fwd), eifj(fwd), sqtr(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,sqtr,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=+1.0  akbl(fwd), eifj(fwd), sqtr(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,sqtr,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=+1.0  akbl(fwd), eqfr(fwd), sitj(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eqfr,sitj,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=-1.0  akbl(fwd), eqfr(fwd), sitj(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eqfr,sitj,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=+1.0  aqbr(fwd), eifj(fwd), sktl(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,eifj,sktl,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=-1.0  aqbr(fwd), eifj(fwd), sktl(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,eifj,sktl,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=-1.0  aqbr(fwd), ekfl(fwd), sitj(eq_src), cmdn(eq_sink), ogph(fwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aqbr,ekfl,sitj,cmdn,ogph,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_sink_sink, peram_src_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=+1.0  aqbr(fwd), ekfl(fwd), sitj(eq_src), cgdh(fwd), ompn(eq_sink)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aqbr,ekfl,sitj,cgdh,ompn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_src, peram_src_sink, peram_sink_sink, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # Group 15, CG=1.0
            # # diag0 sign=-1.0  aibj(fwd), ekfl(fwd), cmdn(eq_sink), ogph(fwd), sqtr(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,cmdn,ogph,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag1 sign=+1.0  aibj(fwd), ekfl(fwd), cmdn(eq_sink), oqpr(fwd), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,cmdn,oqpr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag2 sign=+1.0  aibj(fwd), ekfl(fwd), cgdh(fwd), ompn(eq_sink), sqtr(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,cgdh,ompn,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag3 sign=-1.0  aibj(fwd), ekfl(fwd), cgdh(fwd), oqpr(fwd), smtn(bwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,cgdh,oqpr,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag4 sign=-1.0  aibj(fwd), ekfl(fwd), cqdr(fwd), ompn(eq_sink), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'aibj,ekfl,cqdr,ompn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag5 sign=+1.0  aibj(fwd), ekfl(fwd), cqdr(fwd), ogph(fwd), smtn(bwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'aibj,ekfl,cqdr,ogph,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag6 sign=+1.0  akbl(fwd), eifj(fwd), cmdn(eq_sink), ogph(fwd), sqtr(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,cmdn,ogph,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag7 sign=-1.0  akbl(fwd), eifj(fwd), cmdn(eq_sink), oqpr(fwd), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,cmdn,oqpr,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag8 sign=-1.0  akbl(fwd), eifj(fwd), cgdh(fwd), ompn(eq_sink), sqtr(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,cgdh,ompn,sqtr,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag9 sign=+1.0  akbl(fwd), eifj(fwd), cgdh(fwd), oqpr(fwd), smtn(bwd)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,cgdh,oqpr,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag10 sign=+1.0  akbl(fwd), eifj(fwd), cqdr(fwd), ompn(eq_sink), sgth(eq_src)
            # corr[2,2,Mom_list,t_src,t_sink] += +1.0 * cached_contract(
            #     'akbl,eifj,cqdr,ompn,sgth,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_sink, peram_src_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            # # diag11 sign=-1.0  akbl(fwd), eifj(fwd), cqdr(fwd), ogph(fwd), smtn(bwd)
            # corr[2,2,Mom_list,t_src,t_sink] += -1.0 * cached_contract(
            #     'akbl,eifj,cqdr,ogph,smtn,ce,mo,gi,qs,Kbdf,Knp,Khjl,Krt,ay,gz->Kyz',
            #     peram_src_sink, peram_src_sink, peram_src_sink, peram_src_sink, peram_sink_src, gamma_7, gamma_5, gamma_7, gamma_5, snkVVV0, snkVDVK, srcVVV0, srcVDVK, projection, projection, optimize=['greedy', 'dp'])
            
    if rank == 0:
        dt = time.perf_counter() - st_loop
        print(f'[t_src={t_src:02d}] {dt:.1f}s  total: {(time.perf_counter()-t0):.0f}s')

# ── Save ──────────────────────────────────────────────────────
if rank == 0:
    save_dir = (f'/public/home/sush/distillation/0v2b/result/E32P29/'
                f'Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}')
    check_dir_path(save_dir)
    save_path = (
        f'{save_dir}/corr_NN_I15_GEVP_3x3_src{Nev_src}.npy'
    )
    
    corr_save = backend.asnumpy(corr)
    np.save(save_path, corr_save)
    print(f'\nSaved: {save_path}')
    print(f'Shape (3,3,27,Nt,Nt): {corr_save.shape}')
    print(f'Done. Total time: {(time.perf_counter() - t0):.1f}s')
