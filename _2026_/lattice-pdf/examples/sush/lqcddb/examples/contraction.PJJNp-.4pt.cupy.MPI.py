import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

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
Mom_sink_VVV = [[0, 0, 0]][::-1] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))[::-1]

Mom_len = len(Mom_sink_VDV)

phase_exp_2pt = backend.zeros((Mom_len, Nx, Nx, Nx, Nc), dtype = complex)
phase_exp_3pt = backend.zeros((Mom_len, Nx, Nx, Nx), dtype = complex)

for Mom_indx in range(Mom_len):
    phase_exp_2pt[Mom_indx] = fun_eigen.phase_exp_2pt(Mom = Mom_sink_VDV[Mom_indx])
    phase_exp_3pt[Mom_indx] = fun_eigen.phase_exp_3pt(Mom = Mom_sink_VVV[Mom_indx])

Nev_src = 100
Nev_max = 400
tsep = 24

trank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'Scatter')

import numpy as np

VdV_link = backend.zeros((Lt, Mom_len, Nev_max, Nev_max), dtype = complex)
# sink_VVV = np.zeros((Lt, Mom_len, Nev_src, Nev_src, Nev_src), dtype = complex)

st_eigen = time.perf_counter()
for tsrc_indx, tsrc in enumerate(trank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')
    
    for Mom_indx in range(Mom_len):
        VdV_link[tsrc_indx, (Mom_indx):((Mom_indx + 1))] = fun_eigen.Mom_VdV_sink_t(phase_exp = phase_exp_2pt[(Mom_indx):((Mom_indx + 1))], eigvecs = eigvecs)
        # sink_VVV[tsrc_indx, Mom_indx] = fun_eigen.Mom_VVV_sink_t(phase_exp = phase_exp_3pt[Mom_indx], eigvecs = eigvecs[:Nev_src]).get()

VdV_link = VdV_link * fun_eigen.create_omega_accelerate(
    exact = 100,
    N_eigen = [100, 200, 400, 700, 1100],
    N_sum = [20, 20, 20, 20, 20],
    N_extract = [10, 10, 10, 10, 10],
    noise = 200,
)

if rank == 0:
    sink_VVV = np.load(f'/public/home/sush/distillation/0v2b/VVV/{conf_id}_VVV.npy')
    print(sink_VVV.shape)
    
else:
    sink_VVV = None

sink_VVV = get_mpi_data(sink_VVV, mdtype = 'TScatter', axis = 0)

del phase_exp_2pt, phase_exp_3pt

if rank == 0:
    print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')

projection = (gamma(0) + gamma(4))/2


operator_groups = [[['|', 'u', 'u', 'gamma_7', 'd', '|'], ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'd^d', 'gamma_5', 'u','|'], ['|', 'u^d', 'gamma_mu', 'd','|', '|', 'u^d', 'gamma_mu', 'd','|']]]

GR = GammaRegistry()
VR = VRegistry()
PR = PeramRegistry()
gamma_curr = backend.asarray([gamma(5) @ x for x in [gamma(1), gamma(2), gamma(3), gamma(4)]])

GR.register('gamma_7',  gamma(7))
GR.register('gamma_5',  gamma(5))
GR.register('Projector',  (gamma(4) + gamma(0))/2.0)
GR.register('gamma_mu',  gamma_curr)

corr_4pt = np.zeros((Nt, tsep + 2, tsep + 2, 4, 1, Mom_len, Mom_len), dtype = complex)

for tsrc in range(Nt):
    st_peram = time.perf_counter()
    if rank == 0:
        peram_u_src = backend.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')

    else:
        peram_u_src = None

    if rank == 0:
        peram_u_sep = backend.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{(tsrc + tsep)%Nt:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')

    else:
        peram_u_sep = None

    peram_u_src = get_mpi_data(peram_u_src[..., :Nev_max, :Nev_src], mdtype = 'TScatter', root = 0)
    
    peram_u_sep = get_mpi_data(peram_u_sep[..., :Nev_max, :Nev_src], mdtype = 'TScatter', root = 0)
    peram_u_sep = seq_peram(peram_u_sep)
    
    peram_u_src_src  = get_mpi_data(data = peram_u_src[tsrc//size], mdtype = 'Bcast', root = tsrc%size)
    peram_u_src_sink = get_mpi_data(data = peram_u_src[(tsrc + tsep)%Nt//size], mdtype = 'Bcast', root = (tsrc + tsep)%Nt%size)

    if rank == 0:
        print(f'Load perams: {(time.perf_counter() - st_peram):.1f}s ')

    st_cal = time.perf_counter()
    source_VdV = get_mpi_data(data = VdV_link[tsrc//size, :, :Nev_src, :Nev_src], mdtype = 'Bcast', root = tsrc%size).transpose(0, 2, 1).conj()
    source_VVV = backend.asarray(get_mpi_data(data = sink_VVV[tsrc//size], mdtype = 'Bcast', root = tsrc%size)).conj()
    sink_VVV_t = backend.asarray(get_mpi_data(data = sink_VVV[(tsrc + tsep)%Nt//size], mdtype = 'Bcast', root = (tsrc + tsep)%Nt%size))
    
    _, _, tlist_indx = get_mpi_tlist(Nt = Nt, t = range(tsrc, tsrc + tsep + 2, 1), gtype = 'TScatter')
    
    VR.register('VVV_0', 'tsink', sink_VVV_t[0:1])
    VR.register('VVV_0', 'tsrc',  source_VVV)
    VR.register('VDV_0', 'tsrc',  source_VdV)
    
    PR.register('light', ('tsrc', 'tsrc' ), peram_u_src_src[..., :Nev_src, :Nev_src] )
    PR.register('light', ('tsink', 'tsrc'), peram_u_src_sink[..., :Nev_src, :Nev_src])
    PR.register('light', ('tsrc', 'tsink'), peram_u_src_sink[..., :Nev_src, :Nev_src])
    
    # 非MPI程序！！！！！！！！！！！！！！！！！！
    
    for tcur0_indx, tcur0 in enumerate(tlist_indx):
        st_tcur0 = time.perf_counter()
        
        PR.register('light', ('tcur0', 'tsrc' ), peram_u_src[tcur0, ..., :Nev_max, :Nev_src])
        PR.register('light', ('tsrc', 'tcur0' ), seq_peram(peram_u_src[tcur0, ..., :Nev_max, :Nev_src]))
        PR.register('light', ('tsink', 'tcur0'), peram_u_sep[tcur0, ..., :Nev_src, :Nev_max])
        
        for tcur1_indx, tcur1 in enumerate(tlist_indx):
            st_tcur1 = time.perf_counter()
            PR.register('light', ('tcur1', 'tsrc' ), peram_u_src[tcur1, ..., :Nev_max, :Nev_src])
            PR.register('light', ('tsrc' , 'tcur1'), seq_peram(peram_u_src[tcur1, ..., :Nev_max, :Nev_src]))
            PR.register('light', ('tsink', 'tcur1'), peram_u_sep[tcur1, ..., :Nev_src, :Nev_max])
                
            for _Mom in range(0, Mom_len, 3):
                VR.register('VDV_0', 'tcur0', VdV_link[tcur0, _Mom:(_Mom+3)])
                VR.register('VDV_0', 'tcur1', VdV_link[tcur1, _Mom:(_Mom+3)].transpose(0, 2, 1).conj())
                
                dycn = dynamic_contraction(
                    operator_groups, 
                    peram_registry = PR,
                    v_registry = VR,
                    gamma_registry = GR,
                    Cpt = '4pt',
                    Vindex = ['N', 'K', 'K', 'M', 'M'],
                    Gindex = ['', 'G', 'G', '', '', ''],
                    use_equivalence = False,
                    ignore_dis = False,    
                    Projection = True,
                    optimize = ['greedy', 'dp', 'auto']
                    )

                corr_4pt[tsrc, tcur0_indx, tcur1_indx, :, :, _Mom:(_Mom+3)] = dycn.calculate_all().get()
                    
                    # corr_4pt[tsrc, tcur0_indx, tcur1_indx, :, _Gindx*1:(_Gindx+1)*1, _Mom*1:(_Mom+1)*1] = cached_contract('ab...,ab->...', A, projection).get()

            if rank == 0:
                print(f'calculate 4pt of tsrc {tsrc} of tcur0 {tcur0} use time {(time.perf_counter() - st_tcur1):.3f} s')
                
        if rank == 0:
            print(f'calculate 4pt of tsrc {tsrc} of tcur0 {tcur0} use time {(time.perf_counter() - st_tcur0):.3f} s')
            
            
    if rank == 0:
        print(f'calculate 4pt of tsrc {tsrc} use time {(time.perf_counter() - st_cal):.3f} s')

    exit()

