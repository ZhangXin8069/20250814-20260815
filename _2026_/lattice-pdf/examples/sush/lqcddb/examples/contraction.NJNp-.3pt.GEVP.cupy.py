import sys

from lqcddb import *

import time

set_backend('cupy')
backend = get_backend()

lattice_size = [32, 32, 32, 64]
grid_size = [1, 1, 1, 8]

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
# Mom_sink_VVV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))
Mom_sink_VVV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))[::-1]

Mom_len = len(Mom_sink_VDV)

phase_exp_2pt = backend.zeros((Mom_len, Nx, Nx, Nx, Nc), dtype = complex)
phase_exp_3pt = backend.zeros((Mom_len, Nx, Nx, Nx), dtype = complex)

for Mom_indx in range(Mom_len):
    phase_exp_2pt[Mom_indx] = fun_eigen.phase_exp_2pt(Mom = Mom_sink_VDV[Mom_indx])
    phase_exp_3pt[Mom_indx] = fun_eigen.phase_exp_3pt(Mom = Mom_sink_VVV[Mom_indx])

Nev_src = 100
Nev_max = 400
tsep = 14

trank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'TScatter')

import numpy as np

VdV_link = backend.zeros((Lt, Mom_len, Nev_max, Nev_max), dtype = complex)
sink_VVV = np.zeros((Lt, Mom_len, Nev_src, Nev_src, Nev_src), dtype = complex)

st_eigen = time.perf_counter()
for tsrc_indx, tsrc in enumerate(trank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')
    
    for Mom_indx in range(Mom_len):
        # VdV_link[tsrc_indx, (Mom_indx):((Mom_indx + 1))] = fun_eigen.Mom_VdV_sink_t(phase_exp = phase_exp_2pt[(Mom_indx):((Mom_indx + 1))], eigvecs = eigvecs)
        VdV_link[tsrc_indx, (Mom_indx):((Mom_indx + 1))] = fun_eigen.Mom_VdV_sink_t(phase_exp = phase_exp_2pt[(Mom_indx):((Mom_indx + 1))], eigvecs = eigvecs)
        sink_VVV[tsrc_indx, Mom_indx] = fun_eigen.Mom_VVV_sink_t(phase_exp = phase_exp_3pt[Mom_indx], eigvecs = eigvecs[:Nev_src]).get()

VdV_link *= fun_eigen.create_omega_accelerate(
    exact = 100,
    N_eigen = [100, 200, 400, 700, 1100],
    N_sum = [20, 20, 20, 20, 20],
    N_extract = [10, 10, 10, 10, 10],
    noise = 200,
)

# if rank == 0:
#     sink_VVV = np.load(f'/public/home/sush/distillation/0v2b/VVV/{conf_id}_VVV.npy')
#     print(sink_VVV.shape)
    
# else:
#     sink_VVV = None

# sink_VVV = get_mpi_data(sink_VVV, mdtype = 'TScatter', axis = 0)

del phase_exp_2pt, phase_exp_3pt

if rank == 0:
    print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')

operator_sink = [[]]

GR = GammaRegistry()
VR = VRegistry()
PR = PeramRegistry()

GR.register('gamma_7',  gamma(7))
GR.register('gamma_5',  gamma(5))
GR.register('Projector',  [((gamma(4) + gamma(0))/2.0)[:, :2], ((gamma(4) + gamma(0))/2.0)[:, :2]])

sink_operator = [
    [[-np.sqrt(1/3), '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'], [np.sqrt(1/3), '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'], [np.sqrt(2/3), '|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'd', '|']],
    [[-np.sqrt(1/3), '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'], [np.sqrt(1/3), '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'], [np.sqrt(2/3), '|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'd', '|']],
    [['|', 'd', 'u', 'gamma_7', 'd', '|']]
    ]

source_operator = [
    [['|', 'd^d', 'gamma_5', 'u', '|', '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|']]
    ]

corr_NJN_GEVP = np.zeros((3, Nt, Lt, 4, Ns//2, Ns//2, Mom_len, Mom_len), dtype = complex)

st_load = time.perf_counter()
if rank == 0:
    peram_u = np.zeros((Nt, Nt, Ns, Ns, Nev_max, Nev_src), dtype = complex)
    for tsrc in range(Nt):
        peram_u[tsrc] = np.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy'
            )[..., :Nev_max, :Nev_src]
        
    print(f'load peram use time {time.perf_counter() - st_load :.3f} s')

else:
    peram_u = None

peram_u = get_mpi_data(data = peram_u, mdtype = 'TScatter', root = 0, axis = 1)

for tsrc in range(Nt):
    st_cal = time.perf_counter()
    tsink = (tsrc + tsep)%Nt
    
    VVV_src = backend.asarray(get_mpi_data(data = sink_VVV[tsrc//size, :], mdtype = 'Bcast', root = tsrc%size))
    VDV_src = backend.asarray(get_mpi_data(data = VdV_link[tsrc//size, :, :Nev_src, :Nev_src], mdtype = 'Bcast', root = tsrc%size))
    VVV_src = VVV_src.conj()
    VDV_src = VDV_src.conj().transpose(0, 2, 1)
    
    VVV_sink = backend.asarray(get_mpi_data(data = sink_VVV[tsink//size, :], mdtype = 'Bcast', root = tsink%size))
    VDV_sink = backend.asarray(get_mpi_data(data = VdV_link[tsink//size, :, :Nev_src, :Nev_src], mdtype = 'Bcast', root = tsink%size))
    VDV_sink = VDV_sink.conj().transpose(0, 2, 1)
    
    _, _, tsink_indx = get_mpi_tlist(Nt = Nt, t = [x for x in range(tsrc, tsrc + tsep + 1, 1)], gtype = 'TScatter')
    
    peram_u_src_src = get_mpi_data(data = peram_u[tsrc, tsrc//size, ..., :Nev_src, :Nev_src], mdtype = 'Bcast', root = tsrc%size)
    peram_u_sink_src = get_mpi_data(data = peram_u[tsrc, tsink//size, ..., :Nev_src, :Nev_src], mdtype = 'Bcast', root = tsink%size)
    peram_u_sink_sink = get_mpi_data(data = peram_u[tsink, tsink//size, ..., :Nev_src, :Nev_src], mdtype = 'Bcast', root = tsink%size)

    PR.register('light', ('tsrc', 'tsrc' ), backend.asarray(peram_u_src_src))
    PR.register('light', ('tsink', 'tsrc'), backend.asarray(peram_u_sink_src))
    PR.register('light', ('tsrc', 'tsink' ), seq_peram(backend.asarray(peram_u_sink_src)))
    PR.register('light', ('tsink', 'tsink' ), backend.asarray(peram_u_sink_sink))
    
    VR.register('VDV_0', 'tsrc',  VDV_src[:])
    VR.register('VVV_0', 'tsrc',  VVV_src[:])
    
    for tcur0 in tsink_indx:
        st_cal_cur0 = time.perf_counter()
        peram_u_tcur0_tsrc = backend.asarray(peram_u[tsrc, tcur0])
        peram_u_tcur0_tsink = backend.asarray(peram_u[tsink, tcur0])
        
        PR.register('light', ('tcur0', 'tsrc'), peram_u_tcur0_tsrc)
        PR.register('light', ('tsrc', 'tcur0'), seq_peram(peram_u_tcur0_tsrc))

        PR.register('light', ('tcur0', 'tsink'), peram_u_tcur0_tsink)
        PR.register('light', ('tsink', 'tcur0'), seq_peram(peram_u_tcur0_tsink))
        for Mom_indx in range(27):
            VR.register('VDV_0', 'tcur0',  VdV_link[tcur0, Mom_indx:Mom_indx+1])
            for i in range(0, 3):
                if i == 0:
                    VR.register('VDV_0', 'tsink',  VDV_sink[0:1])
                    VR.register('VVV_0', 'tsink',  VVV_sink[Mom_indx:Mom_indx+1])

                elif i == 1:
                    VR.register('VDV_0', 'tsink',  VDV_sink[Mom_indx:Mom_indx+1])
                    VR.register('VVV_0', 'tsink',  VVV_sink[0:1])
                
                elif i == 2:
                    VR.register('VVV_0', 'tsink',  VVV_sink[Mom_indx:Mom_indx+1])
                    
                for j in range(4):
                    GR.register('gamma_1',  gamma(j + 1) @ gamma(5))
                    dycn_NN = dynamic_contraction(
                        [[s, t] + [['|', 'u^d', 'gamma_1', 'd', '|']] for s in sink_operator[i] for t in source_operator[0]], 
                        peram_registry = PR,
                        v_registry = VR,
                        gamma_registry = GR,
                        Cpt = '3pt',
                        Vindex = ['N', 'N', 'N', 'M', 'M'] if i < 2 else ['N', 'N', 'M', 'M'],
                        use_equivalence = True,
                        ignore_dis = False,    
                        Projection = True,
                        optimize = ['greedy', 'dp', 'auto'],
                        # plot = './test/'
                        )
                    
                    corr_NJN_GEVP[i, tsrc, tcur0, j, ..., Mom_indx:Mom_indx + 1, :] = dycn_NN.calculate_all().get()
                
        if rank == 0:
            print(f'calculate 4pt of tsrc {tsrc} of tcur0 {tcur0} use time {(time.perf_counter() - st_cal_cur0):.3f} s')
    
    if rank == 0:
        print(f'calculate 4pt of tsrc {tsrc} use time {(time.perf_counter() - st_cal):.3f} s')

corr_NJN_GEVP = get_mpi_data(corr_NJN_GEVP, mdtype = 'TGather', root = 0, axis = 2)

# if rank == 0:
#     print(corr_NJN_GEVP)
    
if rank == 0:
    corr_save_path = f'/public/home/sush/distillation/0v2b/result_nex/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}'
    import pathlib
    path = pathlib.Path(corr_save_path)
    if path.exists():
        print('save_path:',corr_save_path)
    
    else:
        path.mkdir(parents = True, exist_ok = True)
        print('mkdir_save_path:',corr_save_path)
        
    backend.save(f'{corr_save_path}/corr_3pt_NJN_GEVP_src100.npy', loop_tsrc(corr_NJN_GEVP, indx = [1, 2], Boundary_Conditions = 'Antiperiodic', Ctype = '3pt', t_sep = tsep))