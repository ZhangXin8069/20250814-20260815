import sys

from lqcddb import *

import time

set_backend('cupy')
backend = get_backend()

lattice_size = [32, 32, 32, 64]
grid_size = [1, 1, 1, 4]

def Readin_gauge(conf_file, lattice_size):
    
    Nz, Ny, Nx, Nt = lattice_size
    
    f = open("%s" % conf_file, "rb")
    gauge = backend.fromfile(f, dtype=">f8")
    gauge = backend.array(gauge)

    gauge = gauge.reshape(Nt, Nx, Nx, Nx, 4, 3, 3, 2)
    gauge = gauge[..., 0] + gauge[..., 1] * 1j
    f.close()

    return gauge

conf_id = sys.argv[1]

mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend.__name__)

rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()

Nx, Ny, Nz, Nt = lattice_size
Lx, Ly, Lz, Lt = [lattice_size[x]//grid_size[x] for x in range(len(lattice_size))]

fun_eigen = vertex_creator(Nx = Nx)

Mom_sink_VDV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))
Mom_sink_VVV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))
# Mom_sink_VVV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))[::-1]

Mom_len = len(Mom_sink_VDV)

phase_exp_2pt = backend.zeros((Mom_len, Nx, Nx, Nx, Nc), dtype = complex)
phase_exp_3pt = backend.zeros((Mom_len, Nx, Nx, Nx), dtype = complex)

for Mom_indx in range(Mom_len):
    phase_exp_2pt[Mom_indx] = fun_eigen.phase_exp_2pt(Mom = Mom_sink_VDV[Mom_indx])
    phase_exp_3pt[Mom_indx] = fun_eigen.phase_exp_3pt(Mom = Mom_sink_VVV[Mom_indx])

Nev_src = 100
Nev_max = 400
tsep = 24
tgap = 6

Tn = tsep - tgap + 2

trank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'Scatter')

import numpy as np

VdV_sink = backend.zeros((Lt, Mom_len, Nev_src, Nev_src), dtype = complex)
sink_VVV = np.zeros((Lt, Mom_len, Nev_src, Nev_src, Nev_src), dtype = complex)

st_eigen = time.perf_counter()
for tsrc_indx, tsrc in enumerate(trank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')
    
    for Mom_indx in range(Mom_len):
        # VdV_sink[tsrc_indx, (Mom_indx):((Mom_indx + 1))] = fun_eigen.Mom_VdV_sink_t(phase_exp = phase_exp_2pt[(Mom_indx):((Mom_indx + 1))], eigvecs = eigvecs)
        VdV_sink[tsrc_indx, (Mom_indx):((Mom_indx + 1))] = fun_eigen.Mom_VdV_sink_t(phase_exp = phase_exp_2pt[(Mom_indx):((Mom_indx + 1))], eigvecs = eigvecs[:Nev_src])
        sink_VVV[tsrc_indx, Mom_indx] = fun_eigen.Mom_VVV_sink_t(phase_exp = phase_exp_3pt[Mom_indx], eigvecs = eigvecs[:Nev_src]).get()

# if rank == 0:
#     sink_VVV = np.load(f'/public/home/sush/distillation/0v2b/VVV/{conf_id}_VVV.npy')
#     print(sink_VVV.shape)
    
# else:
#     sink_VVV = None

# sink_VVV = get_mpi_data(sink_VVV, mdtype = 'TScatter', axis = 0)

del phase_exp_2pt, phase_exp_3pt

if rank == 0:
    print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')

projection = (gamma(0) + gamma(4))/2


operator_sink = [[]]

GR = GammaRegistry()
VR = VRegistry()
PR = PeramRegistry()

GR.register('gamma_7',  gamma(7))
GR.register('gamma_5',  gamma(5))
GR.register('Projector',  (gamma(4) + gamma(0))/2.0)
sink_operator = [
    [['|', 'u', 'u', 'gamma_7', 'd', '|']],
    [[-np.sqrt(1/3), '|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'], [np.sqrt(1/3), '|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'], [np.sqrt(2/3), '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'u', '|']],
    [[-np.sqrt(1/3), '|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'], [np.sqrt(1/3), '|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'], [np.sqrt(2/3), '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'u', '|']]
    ]

source_operator = [
    [[-1, '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|']],
    [[-np.sqrt(1/3), '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'u^d', 'gamma_5', 'u', '|'], [np.sqrt(1/3), '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'd^d', 'gamma_5', 'd', '|'], [np.sqrt(2/3), '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'u^d', 'gamma_5', 'd', '|']],
    [[-np.sqrt(1/3), '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'u^d', 'gamma_5', 'u', '|'], [np.sqrt(1/3), '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'd^d', 'gamma_5', 'd', '|'], [np.sqrt(2/3), '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'u^d', 'gamma_5', 'd', '|']]
    ]

# ── MPI: pre-load perams only for local tsrc ──
# peram_u_local[local_tsrc_idx, tsink_global] — local source, global sink
st_load = time.perf_counter()
peram_u_local = np.zeros((Lt, Nt, Ns, Ns, Nev_src, Nev_src), dtype=complex)
for tsrc_indx, tsrc in enumerate(trank):
    peram_u_local[tsrc_indx] = np.load(
        f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy'
        )[..., :Nev_src, :Nev_src]

if rank == 0:
    print(f'load peram use time {time.perf_counter() - st_load:.3f} s')

# ── MPI: correlator — global tsrc × local tsink ──
corr_NN_local = backend.zeros((Nt, Lt, Ns, Ns, Mom_len, len(sink_operator), len(source_operator)), dtype=complex)

for tsrc in range(Nt):
    st_cal = time.perf_counter()

    # TScatter peram file for this tsrc from rank 0 → (Lt, Ns, Ns, Nev, Nev) per rank
    if rank == 0:
        peram_tsrc_full = np.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy'
            )[..., :Nev_src, :Nev_src]
        peram_tsrc_full = np.ascontiguousarray(peram_tsrc_full)
    else:
        peram_tsrc_full = None
    peram_tsrc = get_mpi_data(peram_tsrc_full, mdtype='TScatter', root=0, axis=0)
    # peram_tsrc shape: (Lt, Ns, Ns, Nev_src, Nev_src) — local tsink only

    # Register ('tsrc', 'tsrc'): peram from tsrc→tsrc, Bcast from tsrc's owner
    if rank == tsrc % size:
        p_tt_tt = np.ascontiguousarray(peram_u_local[(tsrc - rank) // size, tsrc])
    else:
        p_tt_tt = np.empty((Ns, Ns, Nev_src, Nev_src), dtype=complex)
    comm.Bcast(p_tt_tt, root=tsrc % size)
    PR.register('light', ('tsrc', 'tsrc'), backend.asarray(p_tt_tt))

    # Get local tsink indices for this tsrc
    _, _, tsink_indx = get_mpi_tlist(Nt=Nt, t=[x for x in range(tsrc + 1, tsrc + 2, 1)], gtype='TScatter')

    # Bcast source V from tsrc's owner (one slice per tsrc, not per tsink)
    src_VdV, src_VVV = None, None
    src_owner = tsrc % size
    if src_owner == rank:
        src_indx = tsrc // size
        src_VdV_np = VdV_sink[src_indx].get()
        src_VVV_np = np.ascontiguousarray(sink_VVV[src_indx])
        # send shapes first
        shape_vdv = src_VdV_np.shape
        shape_vvv = src_VVV_np.shape
    else:
        shape_vdv = None
        shape_vvv = None
    shape_vdv = comm.bcast(shape_vdv, root=src_owner)
    shape_vvv = comm.bcast(shape_vvv, root=src_owner)
    if rank != src_owner:
        src_VdV_np = np.empty(shape_vdv, dtype=complex)
        src_VVV_np = np.empty(shape_vvv, dtype=complex)
    comm.Bcast(src_VdV_np, root=src_owner)
    comm.Bcast(src_VVV_np, root=src_owner)
    src_VdV = backend.asarray(src_VdV_np)
    src_VVV = src_VVV_np

    for i in range(1, 3):
        for j in range(1, 3):

            for tsink in tsink_indx:
                tsink_global = tsink * size + rank

                # Resolve perams for this (tsrc, tsink_global)
                # ('tsink', 'tsrc'): from TScatter'd tsrc file → local tsink
                PR.register('light', ('tsink', 'tsrc'), backend.asarray(peram_tsrc[tsink]))
                # ('tsrc', 'tsink'): from local tsink file → sink=tsrc (global)
                PR.register('light', ('tsrc', 'tsink'), backend.asarray(peram_u_local[tsink, tsrc]))
                # ('tsink', 'tsink'): from local tsink file → sink=tsink_global
                PR.register('light', ('tsink', 'tsink'), backend.asarray(peram_u_local[tsink, tsink_global]))

                # V arrays: sink V is local, source V is Bcast'd
                if i == 0:
                    VR.register('VVV_0', 'tsink', backend.asarray(sink_VVV[tsink, :]))
                elif i == 1:
                    VR.register('VDV_0', 'tsink', VdV_sink[tsink, 0:1])
                    VR.register('VVV_0', 'tsink', backend.asarray(sink_VVV[tsink, :]))
                elif i == 2:
                    VR.register('VDV_0', 'tsink', VdV_sink[tsink])
                    VR.register('VVV_0', 'tsink', backend.asarray(sink_VVV[tsink, 0:1]))

                if j == 0:
                    VR.register('VVV_0', 'tsrc', backend.asarray(src_VVV).conj())
                elif j == 1:
                    VR.register('VDV_0', 'tsrc', src_VdV[0:1].transpose(0, 2, 1).conj())
                    VR.register('VVV_0', 'tsrc', backend.asarray(src_VVV).conj())
                elif j == 2:
                    VR.register('VDV_0', 'tsrc', src_VdV.transpose(0, 2, 1).conj())
                    VR.register('VVV_0', 'tsrc', backend.asarray(src_VVV[0:1]).conj())

                dycn_NN = dynamic_contraction(
                    [[s, t] for s in sink_operator[i] for t in source_operator[j]],
                    peram_registry=PR,
                    v_registry=VR,
                    gamma_registry=GR,
                    Cpt='2pt',
                    Vindex=['M']*2 if (i + 1) * (j + 1) == 1 else ['M']*3 if (i + 1) * (j + 1)<4 else ['M']*4,
                    use_equivalence=True,
                    ignore_dis=False,
                    Projection=False,
                    optimize=['greedy', 'dp', 'auto']
                    )

                corr_NN_local[tsrc, tsink, ..., i, j] = dycn_NN.calculate_all()

    if rank == 0:
        print(f'calculate 4pt of tsrc {tsrc} use time {(time.perf_counter() - st_cal):.3f} s')

# ── MPI: gather correlator (interleave local tsink contributions) ──
corr_NN = get_mpi_data(corr_NN_local.get(), mdtype='TGather', root=0, axis=1)
if rank == 0:
    corr_NN = backend.asarray(corr_NN)

if rank == 0:
    corr_save_path = f'/nexdata/project/lqcd/sush/result/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}'
    import pathlib
    path = pathlib.Path(corr_save_path)
    if path.exists():
        print('save_path:', corr_save_path)
    else:
        path.mkdir(parents=True, exist_ok=True)
        print('mkdir_save_path:', corr_save_path)

    backend.save(f'{corr_save_path}/corr_3pt_NN_2pt_src100.npy', corr_NN[:])