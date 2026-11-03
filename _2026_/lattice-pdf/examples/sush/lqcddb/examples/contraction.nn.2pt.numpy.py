import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from function_contraction import *

from opt_einsum import contract, contract_path

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

fun_eigen = corr_eigvecs(Nx = Nx, backend = backend)

Mom_sink_VDV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))
Mom_sink_VVV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))

Mom_sqrt = backend.asarray([sum([y**2 for y in x]) for x in Mom_sink_VDV])

Mom_len = len(Mom_sink_VDV)

phase_exp_2pt = backend.zeros((Mom_len, Nx, Nx, Nx, Nc), dtype = complex)
phase_exp_3pt = backend.zeros((Mom_len, Nx, Nx, Nx), dtype = complex)

for Mom_indx in range(Mom_len):
    phase_exp_2pt[Mom_indx] = fun_eigen.phase_exp_2pt(Mom = Mom_sink_VDV[Mom_indx])
    phase_exp_3pt[Mom_indx] = fun_eigen.phase_exp_3pt(Mom = Mom_sink_VVV[Mom_indx])

Nev_src = 100
Nev_src_list = [100]

t_rank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'Scatter')

import numpy as np

sink_VdV = backend.zeros((Lt, Mom_len, Nev_src, Nev_src), dtype = complex)
sink_VVV = np.zeros((Lt, Mom_len, Nev_src, Nev_src, Nev_src), dtype = complex)

st_eigen = time.perf_counter()
for t_src_indx, t_src in enumerate(t_rank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t{t_src:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')
    
    for Mom_indx in range(Mom_len):
        sink_VdV[t_src_indx, (Mom_indx):((Mom_indx + 1))] = fun_eigen.Mom_VdV_sink_t_3(phase_exp = phase_exp_2pt[(Mom_indx):((Mom_indx + 1))], eigvecs = eigvecs[:Nev_src])
        sink_VVV[t_src_indx, Mom_indx] = fun_eigen.Mom_VVV_sink_t_3(phase_exp = phase_exp_3pt[Mom_indx], eigvecs = eigvecs[:Nev_src]).get()
        
if rank == 0:
    print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')

# if rank == 0:
#     sink_VVV = np.load(f'/public/home/sush/distillation/0v2b/VVV/{conf_id}_VVV.npy')
#     print(sink_VVV.shape)
    
# else:
#     sink_VVV = None

sink_VVV = get_mpi_data(sink_VVV, mdtype = 'TScatter', axis = 0)
Psigma = Mom_times_sigma(Mom_sink_VVV, upto4dim = True)

projection = (gamma(0) + gamma(4))/2

Psigma_projection = Psigma@projection

corr_2pt_nep1_nep1 = backend.zeros((12, Mom_len, 4, Nt, Lt), dtype = complex)
corr_2pt_nep1_prp0d = backend.zeros((12, Mom_len, 4, Nt, Lt), dtype = complex)
corr_2pt_prp0u_prp0u = backend.zeros((24, Mom_len, 4, Nt, Lt), dtype = complex)
corr_2pt_prp0d_prp0d = backend.zeros((12, Mom_len, 4, Nt, Lt), dtype = complex)

corr_2pt_proton_pi0u_proton = backend.zeros((6, Mom_len, 2, Nt, Lt), dtype = complex)
corr_2pt_proton_pi0d_proton = backend.zeros((4, Mom_len, 2, Nt, Lt), dtype = complex)
corr_2pt_proton_pi1_neutron = backend.zeros((4, Mom_len, 2, Nt, Lt), dtype = complex)

corr_2pt_pi0u_proton_proton = backend.zeros((6, 2, Mom_len, Nt, Lt), dtype = complex)
corr_2pt_pi0d_proton_proton = backend.zeros((4, 2, Mom_len, Nt, Lt), dtype = complex)
corr_2pt_pi1_neutron_proton = backend.zeros((4, 2, Mom_len, Nt, Lt), dtype = complex)

corr_2pt_pp = backend.zeros((2, Mom_len, Nt, Lt), dtype = complex)

st_peram = time.perf_counter()
if rank == 0:
    peram_u = backend.zeros((Nt, Nt, Ns, Ns, Nev_src, Nev_src), dtype = complex)

    for t_src in range(Nt):
        peram_u[t_src] = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{t_src:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy'
            )[..., :Nev_src, :Nev_src]
        
else:
    peram_u = None

peram_u = get_mpi_data(data = peram_u, mdtype = 'TScatter', root = 0, axis = 1)

if rank == 0:
    print(f'load peram use time {(time.perf_counter() - st_peram):.3f} s')
    
for t_src in range(Nt):
    source_VdV = get_mpi_data(data = sink_VdV[t_src//size], mdtype = 'Bcast', root = t_src%size).transpose(0, 2, 1).conj()
    source_VVV = backend.asarray(get_mpi_data(data = sink_VVV[t_src//size], mdtype = 'Bcast', root = t_src%size)).conj()
    
    t_sink_list_rank, _, t_sink_list_indx = get_mpi_tlist(Nt = Nt, t = range(t_src, t_src + 32, 1), gtype = 'Scatter')
    
    st_cal = time.perf_counter()
        
    for t_sink in t_sink_list_indx:
        sink_VVV_t_sink = backend.asarray(sink_VVV[t_sink, (9 * Mom_indx_A):(9 * (Mom_indx_A + 1))])
        sink_VdV_t_sink = backend.asarray(sink_VdV[t_sink, 0:1])
        source_VdV_t_src = source_VdV[0:1]
        source_VVV_t_src = source_VVV[(9 * Mom_indx_A):(9 * (Mom_indx_A + 1))]
            
        corr_2pt_pp[0, (9 * Mom_indx_A):(9 * (Mom_indx_A + 1)), t_src, t_sink] = contract(
            'eofp,ambn,cgdh,ce,mo,Mbdf,Mnph,ga->M',
            peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
            gamma(7), gamma(7), 
            sink_VVV_t_sink, source_VVV_t_src,
            projection
        )
        
        corr_2pt_pp[1, (9 * Mom_indx_A):(9 * (Mom_indx_A + 1)), t_src, t_sink] = contract(
            'eofp,agbh,cmdn,ce,mo,Mbdf,Mnph,ga->M',
            peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
            gamma(7), gamma(7), 
            sink_VVV_t_sink, source_VVV_t_src,
            projection
        )

    if rank == 0:
        print(f'calculate 2pt of t_src {t_src} use time {(time.perf_counter() - st_cal):.3f} s')

corr_2pt_pp = get_mpi_data(data = corr_2pt_pp, mdtype = 'TGather', root = 0, axis = -1)

if rank == 0:
    corr_save_path = f'/public/home/sush/distillation/0v2b/result/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}'

    import pathlib

    path = pathlib.Path(corr_save_path)

    if path.exists():
        print('save_path:',corr_save_path)
    
    else:
        path.mkdir(parents = True, exist_ok = True)
        print('mkdir_save_path:',corr_save_path)

    corr_2pt_pp = loop_tsrc(corr_2pt_pp, indx = [-2, -1], Boundary_Conditions = 'Antiperiodic')
    
    backend.save(
        f'{corr_save_path}/corr_2pt_gamma0505_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy_proton_proton_nn_src100.npy', 
        corr_2pt_pp
        )