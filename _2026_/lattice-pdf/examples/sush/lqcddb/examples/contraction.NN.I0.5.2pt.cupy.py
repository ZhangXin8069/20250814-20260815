import sys

from lqcddb import *

import time

set_backend('cupy')
backend = get_backend()

lattice_size = [32, 32, 32, 64]
grid_size = [1, 1, 1, 2]

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
    [['|', 'd', 'u', 'gamma_7', 'd', '|']], 
    [[-np.sqrt(1/3), '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'], [np.sqrt(1/3), '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'], [np.sqrt(2/3), '|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'd', '|']],
    [[-np.sqrt(1/3), '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'], [np.sqrt(1/3), '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'], [np.sqrt(2/3), '|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'd', '|']]
    ]

source_operator = [
    [[-1, '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|']], 
    [[-np.sqrt(1/3), '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'u^d', 'gamma_5', 'u', '|'], [np.sqrt(1/3), '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'd^d', 'gamma_5', 'd', '|'], [np.sqrt(2/3), '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'd^d', 'gamma_5', 'u', '|']],
    [[-np.sqrt(1/3), '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'u^d', 'gamma_5', 'u', '|'], [np.sqrt(1/3), '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'd^d', 'gamma_5', 'd', '|'], [np.sqrt(2/3), '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'd^d', 'gamma_5', 'u', '|']]
    ]

corr_NN = backend.zeros((Nt, Lt, Ns, Ns, Mom_len, len(sink_operator), len(source_operator)), dtype = complex)

st_load = time.perf_counter()
if rank == 0:
    peram_u = np.zeros((Nt, Nt, Ns, Ns, Nev_src, Nev_src), dtype = complex)
    for tsrc in range(Nt):
        peram_u[tsrc] = np.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')[..., :Nev_src, :Nev_src]
else:
    peram_u = None
    
peram_u = get_mpi_data(peram_u, mdtype = 'TScatter', root = 0, axis = 1)
    
print(f'load peram use time {time.perf_counter() - st_load :.3f} s')

for tsrc in range(Nt):
    st_cal = time.perf_counter()
    tsink_list, _, tsink_indx = get_mpi_tlist(Nt = Nt, t = [x for x in range(tsrc, tsrc + Nt//2, 1)], gtype = 'TScatter')
    
    VVV_src = backend.asarray(get_mpi_data(data = sink_VVV[tsrc//size, :], mdtype = 'Bcast', root = tsrc%size))
    VDV_src = backend.asarray(get_mpi_data(data = VdV_sink[tsrc//size, :, :Nev_src, :Nev_src], mdtype = 'Bcast', root = tsrc%size))
    
    peram_u_src_src = get_mpi_data(data = peram_u[tsrc, tsrc//size, ..., :Nev_src, :Nev_src], mdtype = 'Bcast', root = tsrc%size)
    PR.register('light', ('tsrc', 'tsrc' ), backend.asarray(peram_u_src_src))
    for i in range(0, 3):
        for j in range(0, 3):
            for _tsink, tsink in enumerate(tsink_indx):
                PR.register('light', ('tsink', 'tsrc'), backend.asarray(peram_u[tsrc, tsink]))
                PR.register('light', ('tsrc', 'tsink' ), seq_peram(backend.asarray(peram_u[tsrc, tsink])))
                PR.register('light', ('tsink', 'tsink' ), backend.asarray(peram_u[tsink_list[_tsink], tsink]))

                if i == 0:
                    VR.register('VVV_0', 'tsink',  backend.asarray(sink_VVV[tsink, :]))

                elif i == 1:
                    VR.register('VDV_0', 'tsink',  VdV_sink[tsink, 0:1])
                    VR.register('VVV_0', 'tsink',  backend.asarray(sink_VVV[tsink, :]))

                elif i == 2:
                    VR.register('VDV_0', 'tsink',  VdV_sink[tsink])
                    VR.register('VVV_0', 'tsink',  backend.asarray(sink_VVV[tsink, 0:1]))

                if j == 0:
                    VR.register('VVV_0', 'tsrc',  backend.asarray(VVV_src).conj())

                elif j == 1:
                    VR.register('VDV_0', 'tsrc',  VDV_src[0:1].transpose(0, 2, 1).conj())
                    VR.register('VVV_0', 'tsrc',  backend.asarray(VVV_src).conj())
                    
                elif j == 2:
                    VR.register('VDV_0', 'tsrc',  VDV_src.transpose(0, 2, 1).conj())
                    VR.register('VVV_0', 'tsrc',  backend.asarray(VVV_src[0:1]).conj())
                
                dycn_NN = dynamic_contraction(
                    [[s, t] for s in sink_operator[i] for t in source_operator[j]], 
                    peram_registry = PR,
                    v_registry = VR,
                    gamma_registry = GR,
                    Cpt = '2pt',
                    Vindex = ['M']*2 if (i + 1) * (j + 1) == 1 else ['M']*3 if (i + 1) * (j + 1)<4 else ['M']*4,
                    use_equivalence = True,
                    ignore_dis = False,    
                    Projection = False,
                    optimize = ['greedy', 'dp', 'auto']
                    )

                corr_NN[tsrc, tsink, ..., i, j] = dycn_NN.calculate_all()
                
    if rank == 0:
        print(f'calculate 4pt of tsrc {tsrc} use time {(time.perf_counter() - st_cal):.3f} s')

corr_NN = get_mpi_data(corr_NN, mdtype = 'TGather', root = 0, axis = 1)

if rank == 0:
    corr_save_path = f'/public/home/sush/distillation/0v2b/result_nex/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}'
    import pathlib
    path = pathlib.Path(corr_save_path)
    if path.exists():
        print('save_path:',corr_save_path)
    
    else:
        path.mkdir(parents = True, exist_ok = True)
        print('mkdir_save_path:',corr_save_path)
        
    backend.save(f'{corr_save_path}/corr_3pt_NN_2pt_src100.npy', loop_tsrc(corr_NN[:], indx = [0, 1], Boundary_Conditions = 'Antiperiodic'))