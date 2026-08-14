"""
contraction.pp.2pt.numpy.py
E32P29 pi+ meson 2pt correlation function, momentum=0, Nev=100, 8 cores.

Reference: lqcddb/src/lqcddb/test/contraction.pi.mpi.py
"""

import os
import sys
import time

sys.path.insert(0, '/public/home/sush/distillation/')

from lqcddb.base import set_backend, get_backend, cached_contract
from lqcddb.base import mpinit, getMPIRank, getMPISize, getMPIComm
from lqcddb.base import get_mpi_tlist, get_mpi_data
from lqcddb.eigvectors import vertex_creator
from lqcddb.constant import gamma, Nc, Nd
from lqcddb.contraction import seq_peram
from lqcddb.io import check_dir_path
# ── Backend ────────────────────────────────────────────────────────────────
set_backend('numpy')
backend = get_backend()

# ── MPI init ───────────────────────────────────────────────────────────────
lattice_size = [32, 32, 32, 64]   # Nz, Ny, Nx, Nt
grid_size    = [1, 1, 1, 16]      # 8 cores, time-only decomposition

mpinit(grid_size=grid_size, latt_size=lattice_size, backend=backend.__name__)
rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()

Nx, Ny, Nz, Nt = lattice_size
Lx, Ly, Lz, Lt = [lattice_size[i] // grid_size[i] for i in range(4)]

# ── Parameters ─────────────────────────────────────────────────────────────
conf_id  = sys.argv[1]
Nev_src  = 100                 # number of eigenvectors for source/sink

# ── Data paths ─────────────────────────────────────────────────────────────
eigen_dir = '/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64'
peram_dir = '/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light'

# File name suffix common to both eigenvector and perambulator files
file_suffix = 'e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy'

eigen_path  = f'{eigen_dir}/{conf_id}/{conf_id}_t{{t:03d}}_{file_suffix}.npy'
peram_path  = f'{peram_dir}/{conf_id}/t{{t:03d}}_{file_suffix}.npy'

# ── Output ─────────────────────────────────────────────────────────────────
result_dir = f'/public/home/sush/distillation/0v2b/result/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}'
if rank == 0:
    check_dir_path(result_dir)

# ── Momentum phase factor (p = 0) ──────────────────────────────────────────
fun_eigen = vertex_creator(Nx=Nx)
phase_exp = fun_eigen.phase_exp_2pt(Mom=[0, 0, 0])

# ── Local time slices for this rank ────────────────────────────────────────
t_rank, _, _ = get_mpi_tlist(Nt=Nt, t=range(Nt), gtype='TScatter')

# ── Compute sink VdV for local time slices ─────────────────────────────────
st_eigen = time.perf_counter()
sink = backend.zeros((Lt, Nev_src, Nev_src), dtype=complex)

for t_local, t_global in enumerate(t_rank):
    eigvecs = backend.load(eigen_path.format(t=t_global))
    sink[t_local] = fun_eigen.Mom_VdV_sink_t(
        phase_exp=phase_exp,
        eigvecs=eigvecs[:Nev_src],
    )
    
if rank == 0:
    print(f'load eigen & compute sink: {time.perf_counter() - st_eigen:.3f} s')

# ── 2pt contraction ────────────────────────────────────────────────────────
corr_2pt = backend.zeros((Nt, Lt), dtype=complex)

for t_src in range(Nt):                                          # global t_src
    st_cal = time.perf_counter()

    # Source VdV: broadcast from owning rank, then conjugate
    source = get_mpi_data(
        data=sink[t_src // size], mdtype='Bcast', root=t_src % size
    ).transpose(1, 0).conj()

    # Local sink time indices for this t_src
    _, _, t_sink_list_indx = get_mpi_tlist(
        Nt=Nt, t=range(t_src, t_src + Nt//2, 1), gtype='TScatter'
    )

    # Load perambulator at t_src and compute seq_peram (tau_d)
    if rank == 0:
        peram_u_src = backend.load(peram_path.format(t=t_src))
    else:
        peram_u_src = None

    peram_u_src = get_mpi_data(data=peram_u_src, mdtype='TScatter', root=0, axis=0)
    peram_d_src = seq_peram(peram_u_src)

    for t_sink in t_sink_list_indx:                              # local t_sink
        corr_2pt[t_src, t_sink] = cached_contract(
            'manb,cedf,bd,fn,ac,em->',
            peram_u_src[t_sink, :, :, :Nev_src, :Nev_src],
            peram_d_src[t_sink, :, :, :Nev_src, :Nev_src],
            source,
            sink[t_sink],
            backend.asarray(gamma(5)),
            backend.asarray(gamma(5)),
        )
        
    if rank == 0:
        print(f'cal 2pt use time: {time.perf_counter() - st_cal:.3f} s of tsrc {t_src}')

# ── MPI gather ─────────────────────────────────────────────────────────────
corr_2pt = get_mpi_data(data=corr_2pt, mdtype='TGather', root=0, axis=-1)

# ── Save ───────────────────────────────────────────────────────────────────
if rank == 0:
    out_name = f'corr_2pt_pp_gamma0505_e100_src{Nev_src}.npy'
    backend.save(f'{result_dir}/{out_name}', corr_2pt)
    print(f'Saved: {result_dir}/{out_name}')
    print(f'Shape: {corr_2pt.shape}')
