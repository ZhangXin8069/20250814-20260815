import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from function_contraction import *

from opt_einsum import contract, contract_path

import time

set_backend('numpy')
backend = get_backend()

lattice_size = [32, 32, 32, 64]
grid_size = [1, 1, 1, 16]

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

# phase_exp_2pt = backend.zeros((Mom_len, Nx, Nx, Nx, Nc), dtype = complex)
phase_exp_3pt = backend.zeros((Mom_len, Nx, Nx, Nx), dtype = complex)

for Mom_indx in range(Mom_len):
    # phase_exp_2pt[Mom_indx] = fun_eigen.phase_exp_2pt(Mom = Mom_sink_VDV[Mom_indx])
    phase_exp_3pt[Mom_indx] = fun_eigen.phase_exp_3pt(Mom = Mom_sink_VVV[Mom_indx])

Nev_src = 100
Nev_src_list = [100]

t_rank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'Scatter')

import numpy as np

# sink_VdV = backend.zeros((Lt, Mom_len, Nev_src, Nev_src), dtype = complex)
sink_VVV = np.zeros((Lt, Mom_len, Nev_src, Nev_src, Nev_src), dtype = complex)

st_eigen = time.perf_counter()
for t_src_indx, t_src in enumerate(t_rank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t{t_src:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')
    
    for Mom_indx in range(Mom_len):
        # sink_VdV[t_src_indx, (Mom_indx):((Mom_indx + 1))] = fun_eigen.Mom_VdV_sink_t_3(phase_exp = phase_exp_2pt[(Mom_indx):((Mom_indx + 1))], eigvecs = eigvecs[:Nev_src])
        sink_VVV[t_src_indx, Mom_indx] = fun_eigen.Mom_VVV_sink_t_3(phase_exp = phase_exp_3pt[Mom_indx], eigvecs = eigvecs[:Nev_src])

# if rank == 0:
#     sink_VVV = np.load(f'/public/home/sush/distillation/0v2b/VVV/{conf_id}_VVV.npy')
#     print(sink_VVV.shape)
    
# else:
#     sink_VVV = None

# sink_VVV = get_mpi_data(sink_VVV, mdtype = 'TScatter', axis = 0)
    
if rank == 0:
    print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')

projection = (gamma(0) + gamma(4))/2

corr_2pt_PP = backend.zeros((2, Mom_len, Mom_len, Nt, Lt), dtype = complex)
corr_2pt_NN = backend.zeros((2, Mom_len, Mom_len, Nt, Lt), dtype = complex)

    
for t_src in range(Nt):
    # source_VdV = get_mpi_data(data = sink_VdV[t_src//size], mdtype = 'Bcast', root = t_src%size).transpose(0, 2, 1).conj()
    source_VVV = backend.asarray(get_mpi_data(data = sink_VVV[t_src//size], mdtype = 'Bcast', root = t_src%size)).conj()
    if rank == 0:
        peram_u = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{t_src:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy'
            )[..., :Nev_src, :Nev_src]
    else:
        peram_u = None
    
    peram_u = get_mpi_data(data = peram_u, mdtype = 'TScatter', root = 0)
    
    t_sink_list_rank, _, t_sink_list_indx = get_mpi_tlist(Nt = Nt, t = range(t_src, t_src + 32, 1), gtype = 'TScatter')
    
    st_cal = time.perf_counter()
        
    for t_sink in t_sink_list_indx:
        for Mom_indx in range(max(1, int(backend.ceil(Mom_len/3)))):
            sink_VVV_t_sink = backend.asarray(sink_VVV[t_sink])
            source_VVV_t_src = source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))]

            corr_2pt_PP[0, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                'eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN',
                peram_u[t_sink], peram_u[t_sink], peram_u[t_sink], 
                gamma(7), gamma(7), 
                sink_VVV_t_sink, source_VVV_t_src,
                projection
            )
            
            corr_2pt_PP[1, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                'eofp,agbh,cmdn,ce,mo,Nbdf,Mnph,ga->MN',
                peram_u[t_sink], peram_u[t_sink], peram_u[t_sink], 
                gamma(7), gamma(7), 
                sink_VVV_t_sink, source_VVV_t_src,
                projection
            )
            
            # corr_2pt_NN[0, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
            #     'aobp,egfh,cmdn,ce,mo,Nbdf,Mnph,ga->MN',
            #     peram_u[t_sink], peram_u[t_sink], peram_u[t_sink], 
            #     gamma(7), gamma(7), 
            #     sink_VVV_t_sink, source_VVV_t_src,
            #     projection
            # )
            
            # corr_2pt_NN[1, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
            #     'agbh,eofp,cmdn,ce,mo,Nbdf,Mnph,ga->MN',
            #     peram_u[t_sink], peram_u[t_sink], peram_u[t_sink], 
            #     gamma(7), gamma(7), 
            #     sink_VVV_t_sink, source_VVV_t_src,
            #     projection
            # )
            
    if rank == 0:
        print(f'calculate 2pt of t_src {t_src} use time {(time.perf_counter() - st_cal):.3f} s')

corr_2pt_PP = get_mpi_data(data = corr_2pt_PP, mdtype = 'TGather', root = 0, axis = -1)
corr_2pt_NN = get_mpi_data(data = corr_2pt_NN, mdtype = 'TGather', root = 0, axis = -1)

if rank == 0:
    corr_save_path = f'/public/home/sush/distillation/0v2b/result/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}'
    check_dir_path(corr_save_path)
    backend.save(f'{corr_save_path}/corr_2pt_PP', corr_2pt_PP)
    # backend.save(f'{corr_save_path}/corr_2pt_NN', corr_2pt_NN)
    