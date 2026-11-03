import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from lqcddb import *

import time

set_backend('cupy')
backend = get_backend()

lattice_size = [32, 32, 32, 64]
grid_size = [1, 1, 1, 1]

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
tgap = 6

Tn = tsep - tgap + 2

trank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'Scatter')

import numpy as np

VdV_link = backend.zeros((Lt, Mom_len, Nev_max, Nev_max), dtype = complex)
sink_VVV = np.zeros((Lt, Mom_len, Nev_src, Nev_src, Nev_src), dtype = complex)

st_eigen = time.perf_counter()
for tsrc_indx, tsrc in enumerate(trank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')
    
    for Mom_indx in range(Mom_len):
        VdV_link[tsrc_indx, (Mom_indx):((Mom_indx + 1))] = fun_eigen.Mom_VdV_sink_t(phase_exp = phase_exp_2pt[(Mom_indx):((Mom_indx + 1))], eigvecs = eigvecs)
        # VdV_link[tsrc_indx, (Mom_indx):((Mom_indx + 1))] = fun_eigen.VdV_sink_t_link(
        #     eigvecs = eigvecs, link_dir = 
        # )
        
        sink_VVV[tsrc_indx, Mom_indx] = fun_eigen.Mom_VVV_sink_t(phase_exp = phase_exp_3pt[Mom_indx], eigvecs = eigvecs[:Nev_src]).get()

VdV_link = VdV_link * fun_eigen.create_omega_accelerate(
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

projection = (gamma(0) + gamma(4))/2


operator_NJNp = [[['|', 'd', 'u', 'gamma_7', 'd', '|'], ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'd^d', 'gamma_5', 'u','|'], ['|', 'u^d', 'gamma_mu', 'd','|']]]
operator_PJN = [[['|', 'u', 'u', 'gamma_7', 'd', '|'], ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|'], ['|', 'u^d', 'gamma_mu', 'd','|']]]

GR = GammaRegistry()
VR = VRegistry()
PR = PeramRegistry()
gamma_curr = backend.asarray([gamma(5) @ x for x in [gamma(1), gamma(2), gamma(3), gamma(4)]])

GR.register('gamma_7',  gamma(7))
GR.register('gamma_5',  gamma(5))
GR.register('Projector',  (gamma(4) + gamma(0))/2.0)
GR.register('gamma_mu',  gamma_curr)

corr_NJNp = backend.zeros((Nt, Tn, Tn, len(gamma_curr), Mom_len, Mom_len), dtype = complex)
corr_PJN = backend.zeros((Nt, Tn, Tn, len(gamma_curr), Mom_len), dtype = complex)

# peram_u = np.zeros((Nt, Nt, Ns, Ns, Nev_max, Nev_src), dtype = complex)
# for tsrc in range(Nt):
#     peram_u_src[= np.load(
#         f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy'
#         )[..., :Nev_src]

for tsrc in range(Nt):
    st_cal = time.perf_counter()
    peram_u_src = backend.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')[..., :Nev_src]
    peram_u_sep = backend.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{(tsep + tsrc)%Nt:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')[..., :Nev_src]
    
    VR.register('VDV_0', 'tsrc',  VdV_link[tsrc, ..., :Nev_src, :Nev_src].transpose(0, 2, 1).conj())
    PR.register('light', ('tsrc', 'tsrc' ), backend.asarray(peram_u_src[tsrc, ..., :Nev_src, :Nev_src]))
    
    for tn in [x%Nt for x in range(tgap + tsrc, (tsep + 1 + tsrc), 1)]:
        peram_u_n = backend.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{tn:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')[..., :Nev_src]
        
        st_tcur0x = time.perf_counter()
        VR.register('VVV_0', 'tsrc',  backend.asarray(sink_VVV[tsrc].conj()))
        
        PR.register('light', ('tsink', 'tsrc'), backend.asarray(peram_u_src[tn, ..., :Nev_src, :Nev_src]))
        
        for tcur0x in [x%Nt for x in range(tgap + tsrc, tn, 1)]:
            for _Mom in range(0, 7, 1):
                VR.register('VDV_0', 'tcur0',  VdV_link[tcur0x, _Mom:_Mom+1])
                VR.register('VVV_0', 'tsink',  backend.asarray(sink_VVV[tn, _Mom:_Mom+1]))
                
                PR.register('light', ('tcur0', 'tsrc'), backend.asarray(peram_u_src[tcur0x, ..., :Nev_max, :Nev_src]))
                PR.register('light', ('tsrc', 'tcur0'), seq_peram(backend.asarray(peram_u_src[tcur0x, ..., :Nev_max, :Nev_src])))
                PR.register('light', ('tsink', 'tcur0'), seq_peram(backend.asarray(peram_u_n[tcur0x, ..., :Nev_max, :Nev_src])))
                for G in range(4):
                    GR.register('gamma_mu',  gamma_curr[G:G+1])
                    dycn_NJNp = dynamic_contraction(
                        operator_NJNp, 
                        peram_registry = PR,
                        v_registry = VR,
                        gamma_registry = GR,
                        Cpt = '3pt',
                        Vindex = ['N', 'N', 'M', 'M'],
                        Gindex = ['', 'G', '', ''],
                        use_equivalence = False,
                        ignore_dis = False,    
                        Projection = True,
                        optimize = ['greedy', 'dp', 'auto']
                        )

                    corr_NJNp[tsrc, tn - tsrc - tgap, tcur0x - tsrc - tgap, G:G+1, _Mom:_Mom+1, :] = dycn_NJNp.calculate_all()

        if rank == 0:
            print(f'calculate NJNp of tsrc {tsrc} use time {(time.perf_counter() - st_tcur0x):.3f} s')
            
        st_tcur0y = time.perf_counter()
        
        GR.register('gamma_mu',  gamma_curr)
        VR.register('VVV_0', 'tsrc',  backend.asarray(sink_VVV[tn, 0:7].conj()))
        VR.register('VVV_0', 'tsink',  backend.asarray(sink_VVV[(tsep + tsrc)%Nt, 0:1]))

        PR.register('light', ('tsink', 'tsrc'), backend.asarray(peram_u_n[(tsep + tsrc)%Nt, ..., :Nev_src, :Nev_src]))
        
        for tcur0y in [x%Nt for x in range(tn, tsep + tsrc + 1, 1)]:
            VR.register('VDV_0', 'tcur0',  VdV_link[tcur0y, 0:7].transpose(0, 2, 1).conj())

            PR.register('light', ('tcur0', 'tsrc'), backend.asarray(peram_u_n[tcur0y, ..., :Nev_max, :Nev_src]))
            PR.register('light', ('tsink', 'tcur0'), seq_peram(backend.asarray(peram_u_sep[tcur0y, ..., :Nev_max, :Nev_src])))
            
            dycn_PJN = dynamic_contraction(
                operator_PJN, 
                peram_registry = PR,
                v_registry = VR,
                gamma_registry = GR,
                Cpt = '3pt',
                Vindex = ['N', 'M', 'M'],
                Gindex = ['', 'G', '', ''],
                use_equivalence = False,
                ignore_dis = False,    
                Projection = True,
                optimize = ['greedy', 'dp', 'auto']
                )
            
            corr_PJN[tsrc, tn - tsrc - tgap, tcur0y - tsrc - tgap, :, 0:7] = dycn_PJN.calculate_all()[..., 0, :]

        if rank == 0:
            print(f'calculate PJN of tsrc {tsrc} use time {(time.perf_counter() - st_tcur0y):.3f} s')
            
    if rank == 0:
        print(f'calculate 4pt of tsrc {tsrc} use time {(time.perf_counter() - st_cal):.3f} s')

if rank == 0:

    corr_save_path = f'/nexdata/project/lqcd/sush/result/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}'

    import pathlib

    path = pathlib.Path(corr_save_path)

    if path.exists():
        print('save_path:',corr_save_path)
    
    else:
        path.mkdir(parents = True, exist_ok = True)
        print('mkdir_save_path:',corr_save_path)
    
    backend.save(
        f'{corr_save_path}/corr_3pt_proton_J_mu_neutron_tsep{tsep}_of_4pt_src100.npy', 
        corr_PJN[:]
        )
    
    backend.save(
        f'{corr_save_path}/corr_3pt_neutron_J_mu_neutron_pi-_tsep{tsep}_of_4pt_src100.npy', 
        corr_NJNp[:]
        )

