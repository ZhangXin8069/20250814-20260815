import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from function_contraction import *

from opt_einsum import contract, contract_path

from cupy.cuda.runtime import getDeviceCount as cudaGetDeviceCount
A = cudaGetDeviceCount()
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
Mom_sink_VVV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))[::-1]

# Mom_sink_VDV = [[0, 0, 0]]
# Mom_sink_VVV = [[0, 0, 0]]

Mom_sink_link = Mom_sink_VDV

Mom_len = len(Mom_sink_VDV)

phase_exp_2pt = backend.zeros((Mom_len, Nx, Nx, Nx, Nc), dtype = complex)
phase_exp_3pt = backend.zeros((Mom_len, Nx, Nx, Nx), dtype = complex)

for Mom_indx in range(Mom_len):
    phase_exp_2pt[Mom_indx] = fun_eigen.phase_exp_2pt(Mom = Mom_sink_VDV[Mom_indx])
    phase_exp_3pt[Mom_indx] = fun_eigen.phase_exp_3pt(Mom = Mom_sink_VVV[Mom_indx])

Nev_src = 100
Nev_link = 400
link_max = 0
t_sep = 12

if link_max > 0:
    if rank == 0:
        gauge_link = Readin_gauge()
        
    else:
        gauge_link = None
    
    gauge_link = get_mpi_data(gauge_link, mdtype = 'TScatter', root = 0, axis = 0)
    
else:
    gauge_link = False
    
t_rank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'Scatter')

import numpy as np

VdV_link = backend.zeros((Lt, Mom_len, 2*link_max + 1, Nev_link, Nev_link), dtype = complex)
# sink_VVV = np.zeros((Lt, Mom_len, Nev_src, Nev_src, Nev_src), dtype = complex)

st_eigen = time.perf_counter()
for t_src_indx, t_src in enumerate(t_rank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t{t_src:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')
    
    for Mom_indx in range(Mom_len):
        VdV_link[t_src_indx, (Mom_indx):((Mom_indx + 1))] = fun_eigen.VdV_sink_t_link(eigvecs = eigvecs[:], link_dir = 'Z', link_max = link_max, phase_exp = phase_exp_2pt[(Mom_indx):((Mom_indx + 1))], gauge_link = gauge_link, t = t_src_indx)
        # sink_VVV[t_src_indx, Mom_indx] = fun_eigen.Mom_VVV_sink_t_3(phase_exp = phase_exp_3pt[Mom_indx], eigvecs = eigvecs[:Nev_src]).get()

if rank == 0:
    sink_VVV = np.load(f'/public/home/sush/distillation/0v2b/VVV/{conf_id}_VVV.npy')
    print(sink_VVV.shape)
    
else:
    
    sink_VVV = None

sink_VVV = get_mpi_data(sink_VVV, mdtype = 'TScatter', axis = 0)

if rank == 0:
    print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')
    
projection = (gamma(0) + gamma(4))/2
gamma_curr = (gamma(0) - gamma(5)) @ backend.asarray([gamma(1), gamma(2), gamma(3), gamma(4)])

corr_3pt_matrix = backend.zeros((Mom_len, Mom_len, len(gamma_curr), 2*link_max + 1, Nt, Lt), dtype = complex)

for t_src in range(Nt):
    st_cal = time.perf_counter()
    
    source_VdV = get_mpi_data(data = VdV_link[t_src//size, :, link_max, :Nev_src, :Nev_src], mdtype = 'Bcast', root = t_src%size).transpose(0, 2, 1).conj()
    source_VVV = backend.asarray(get_mpi_data(data = sink_VVV[t_src//size], mdtype = 'Bcast', root = t_src%size)).conj()
    
    t_curr_list_rank, _, t_curr_list_indx = get_mpi_tlist(Nt = Nt, t = range(t_src, t_src + 2 * t_sep + 2, 1), gtype = 'Scatter')
    
    st_peram = time.perf_counter()
    if rank == 0:
        peram_u_src = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{t_src:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy'
            )[..., :Nev_link, :Nev_src]
        
    else:
        peram_u_src = None
    
    peram_u_src = get_mpi_data(data = peram_u_src, mdtype = 'TScatter', root = 0, axis = 0)
     
    t_sink = (t_src + t_sep) % Nt
    
    if rank == 0:
        peram_u_sink_seq = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{t_sink:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy'
            )[..., :Nev_link, :Nev_src]
    else:
        peram_u_sink_seq = None
    
    peram_u_sink_seq = get_mpi_data(data = peram_u_sink_seq, mdtype = 'TScatter', root = 0, axis = 0)
    peram_u_sink_seq = seq_peram(peram_u_sink_seq)
    
    if rank == 0:
        print(f'load peram use time {(time.perf_counter() - st_peram):.3f} s')
        
    for t_curr in t_curr_list_indx:
        peram_u_src_seq = seq_peram(peram_u_src[t_curr])
        
        sink_VVV_t_sink = backend.asarray(sink_VVV[t_sink])
        for Mom_indx in range(max(1, int(backend.ceil(Mom_len/3)))):
            corr_3pt_matrix[Mom_indx * 3:(Mom_indx + 1) * 3, ..., t_src, t_curr] = (contract(
                'eifj,okpl,ambn,cgdh,ce,Gmo,gi,Nbdf,NLnp,Mhjl,ka->MNGL',
                peram_u_src[t_sink, :, :, :Nev_src, :Nev_src], peram_u_src[t_curr, :, :, :Nev_link, :Nev_src], peram_u_sink_seq[t_curr, :, :, :Nev_src, :Nev_link], peram_u_src[t_sink, :, :, :Nev_src, :Nev_src],
                gamma(7), gamma_curr, gamma(7),
                sink_VVV_t_sink[:], VdV_link[t_curr], source_VVV[Mom_indx * 3:(Mom_indx + 1) * 3],
                projection
                ) * -1.0) + corr_3pt_matrix[Mom_indx * 3:(Mom_indx + 1) * 3, ..., t_src, t_curr]

            corr_3pt_matrix[Mom_indx * 3:(Mom_indx + 1) * 3, ..., t_src, t_curr] = (contract(
                'eifj,okpl,agbh,cmdn,ce,Gmo,gi,Nbdf,NLnp,Mhjl,ka->MNGL',
                peram_u_src[t_sink, :, :, :Nev_src, :Nev_src], peram_u_src[t_curr, :, :, :Nev_link, :Nev_src], peram_u_src[t_sink, :, :, :Nev_src, :Nev_src], peram_u_sink_seq[t_curr, :, :, :Nev_src, :Nev_link],
                gamma(7), gamma_curr, gamma(7),
                sink_VVV_t_sink[:], VdV_link[t_curr], source_VVV[Mom_indx * 3:(Mom_indx + 1) * 3],
                projection
                ) * 1.0) + corr_3pt_matrix[Mom_indx * 3:(Mom_indx + 1) * 3, ..., t_src, t_curr]

            corr_3pt_matrix[Mom_indx * 3:(Mom_indx + 1) * 3, ..., t_src, t_curr] = (contract(
                'ekfl,oipj,ambn,cgdh,ce,Gmo,gi,Nbdf,NLnp,Mhjl,ka->MNGL',
                peram_u_src[t_sink, :, :, :Nev_src, :Nev_src], peram_u_src[t_curr, :, :, :Nev_link, :Nev_src], peram_u_sink_seq[t_curr, :, :, :Nev_src, :Nev_link], peram_u_src[t_sink, :, :, :Nev_src, :Nev_src],
                gamma(7), gamma_curr, gamma(7),
                sink_VVV_t_sink[:], VdV_link[t_curr], source_VVV[Mom_indx * 3:(Mom_indx + 1) * 3],
                projection
                ) * 1.0) + corr_3pt_matrix[Mom_indx * 3:(Mom_indx + 1) * 3, ..., t_src, t_curr]

            corr_3pt_matrix[Mom_indx * 3:(Mom_indx + 1) * 3, ..., t_src, t_curr] = (contract(
                'ekfl,oipj,agbh,cmdn,ce,Gmo,gi,Nbdf,NLnp,Mhjl,ka->MNGL',
                peram_u_src[t_sink, :, :, :Nev_src, :Nev_src], peram_u_src[t_curr, :, :, :Nev_link, :Nev_src], peram_u_src[t_sink, :, :, :Nev_src, :Nev_src], peram_u_sink_seq[t_curr, :, :, :Nev_src, :Nev_link],
                gamma(7), gamma_curr, gamma(7),
                sink_VVV_t_sink[:], VdV_link[t_curr], source_VVV[Mom_indx * 3:(Mom_indx + 1) * 3],
                projection
                ) * -1.0) + corr_3pt_matrix[Mom_indx * 3:(Mom_indx + 1) * 3, ..., t_src, t_curr]

    free, total = backend.cuda.runtime.memGetInfo()
    
    if rank == 0:
        print(f'calculate 2pt of t_src {t_src} use time {(time.perf_counter() - st_cal):.3f} s. device mem: {(total - free) / 1024**3} GB, free:{free / 1024**3} GB.')


corr_3pt_matrix = get_mpi_data(data = corr_3pt_matrix, mdtype = 'TGather', root = 0, axis = -1)

if rank == 0:

    corr_save_path = f'/public/home/sush/distillation/0v2b/result/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}'

    import pathlib

    path = pathlib.Path(corr_save_path)

    if path.exists():
        print('save_path:',corr_save_path)
    
    else:
        path.mkdir(parents = True, exist_ok = True)
        print('mkdir_save_path:',corr_save_path)
        
    corr_3pt_matrix = loop_tsrc(
        corr_3pt_matrix,
        indx = [-2, -1],
        Boundary_Conditions = 'Antiperiodic',
        Ctype = '3pt',
        t_sep = t_sep
    )
    
    backend.save(
        f'{corr_save_path}/corr_3pt_neutron_J_mu_neutron_pi-_src100.npy', 
        corr_3pt_matrix[:]
        )
    
