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

param = creat_param(
    sink_operators = [['d^d', 'gamma_sink', 'u', 1]],
    source_operators = [['u^d', 'gamma_source', 'd', -1]]
)

corr = distillation_func(
    conf_id = conf_id,
    param = param.data(),
    
    lattice_size = lattice_size, # Nx, Ny, Nz, Nt 
    grid_size = grid_size, 
    
    backend = backend.__name__
)

# mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend)
rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()

Nx, Ny, Nz, Nt = lattice_size
Lx, Ly, Lz, Lt = [lattice_size[x]//grid_size[x] for x in range(len(lattice_size))]

fun_eigen = corr_eigvecs(Nx = Nx, backend = backend)

Mom_sink_VDV = [[0, 0, 0]]
Mom_sink_VVV = [[0, 0, 0]]

# Mom_sink_VDV = [[0, 0, 1]]
# Mom_sink_VVV = [[0, 0, -1]]

Mom_len = len(Mom_sink_VDV)

phase_exp_2pt = backend.zeros((Mom_len, Nx, Nx, Nx, Nc), dtype = complex)
phase_exp_3pt = backend.zeros((Mom_len, Nx, Nx, Nx), dtype = complex)

for Mom_indx in range(Mom_len):
    phase_exp_2pt[Mom_indx] = fun_eigen.phase_exp_2pt(Mom = Mom_sink_VDV[Mom_indx])
    phase_exp_3pt[Mom_indx] = fun_eigen.phase_exp_3pt(Mom = Mom_sink_VVV[Mom_indx])

Nev_src = 100
Nev_src_list = [100]

t_rank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'Scatter')

sink_VdV = backend.zeros((Lt, Mom_len, Nev_src, Nev_src), dtype = complex)
sink_VVV = backend.zeros((Lt, Mom_len, Nev_src, Nev_src, Nev_src), dtype = complex)

st_eigen = time.perf_counter()
for t_src_indx, t_src in enumerate(t_rank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t{t_src:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')
    for Mom_indx in range(Mom_len):
        sink_VdV[t_src_indx, Mom_indx] = fun_eigen.Mom_VdV_sink_t_2(phase_exp = phase_exp_2pt[Mom_indx], eigvecs = eigvecs[:Nev_src])
        sink_VVV[t_src_indx, Mom_indx] = fun_eigen.Mom_VVV_sink_t_2(phase_exp = phase_exp_3pt[Mom_indx], eigvecs = eigvecs[:Nev_src])
        
if rank == 0:
    print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')
    
print(f'load and calculate sink_VdV use time {(time.perf_counter() - st_eigen):.3f} s')
projection = (gamma(0) + gamma(4))/2
corr_2pt = backend.zeros((12, len(Nev_src_list), Mom_len, Nt, Lt), dtype = complex)

for t_src in range(1):
    st_peram = time.perf_counter()
    
    source_VdV = get_mpi_data(data = sink_VdV[t_src//size], mdtype = 'Bcast', root = t_src%size).transpose(0, 2, 1).conj()
    source_VVV = get_mpi_data(data = sink_VVV[t_src//size], mdtype = 'Bcast', root = t_src%size).conj()
    
    t_sink_list_rank, _, t_sink_list_indx = get_mpi_tlist(Nt = Nt, t = range(t_src, t_src + 32, 1), gtype = 'Scatter')
    
    if rank == 0:
        peram_u_src = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{t_src:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy'
            )[..., :Nev_src, :Nev_src]

    else:
        peram_u_src = None
        
    peram_u_src = get_mpi_data(data = peram_u_src, mdtype = 'TScatter', root = 0, axis = 0)
    peram_d_src = seq_peram(peram_u_src)
    
    if rank == 0:
        print(f'load peram of t_src {t_src} use time {(time.perf_counter() - st_peram):.3f} s')
        
    st_cal = time.perf_counter()
    for _Nev_src_indx, _Nev_src in enumerate(Nev_src_list):
        for Mom_indx in range(Mom_len):
            for t_sink in t_sink_list_indx:
                corr_2pt[0, _Nev_src_indx, Mom_indx, t_src, t_sink] = contract(
                    'ambn,egfh,iqjr,ckdl,ospt,ce,mo,gi,kq,bdf,np,hj,lrt,sa->',
                    peram_u_src[t_sink],peram_d_src[t_sink], peram_u_src[t_src], peram_u_src[t_sink], peram_u_src[t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV[t_sink, Mom_indx], sink_VdV[t_sink, Mom_indx], source_VdV[Mom_indx], source_VVV[Mom_indx], projection
                    )
                
                corr_2pt[1, _Nev_src_indx, Mom_indx, t_src, t_sink] = contract(
                    'ambn,eqfr,ikjl,cgdh,ospt,ce,mo,gi,kq,bdf,np,hj,lrt,sa->',
                    peram_u_src[t_sink],peram_u_src[t_sink], peram_d_src[t_sink], peram_u_src[t_sink], peram_u_src[t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV[t_sink, Mom_indx], sink_VdV[t_sink, Mom_indx], source_VdV[Mom_indx], source_VVV[Mom_indx], projection
                    )
                

    if rank == 0:
        print(f'calculate 2pt of t_src {t_src} use time {(time.perf_counter() - st_cal):.3f} s')

corr_2pt = get_mpi_data(data = corr_2pt, mdtype = 'TGather', root = 0, axis = -1)

if rank == 0:
    for _Nev_src_indx, _Nev_src in enumerate(Nev_src_list):
        for Mom_indx in range(Mom_len):
            print(corr_2pt[0, 0, 0, 0])
            
            # _Mom = Mom_sink_VDV[Mom_indx]
            # corr_save_path = f'/public/home/sush/distillation/0v2b/result/E32P29/Px{_Mom[2]}Py{_Mom[1]}Pz{_Mom[0]}/ENV_{Nev_src}/conf{conf_id}'

            # import pathlib

            # path = pathlib.Path(corr_save_path)

            # if path.exists():
            #     print('save_path:',corr_save_path)
            
            # else:
            #     path.mkdir(parents = True, exist_ok = True)
            #     print('mkdir_save_path:',corr_save_path)
                        
            # for i in range(6):
            #         backend.save(f'{corr_save_path}/corr_ud_2pt_gamma0505_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy_I1.5_Iz1.5_type{i}_src{_Nev_src}.npy', corr_2pt[i, _Nev_src_indx, Mom_indx])
        