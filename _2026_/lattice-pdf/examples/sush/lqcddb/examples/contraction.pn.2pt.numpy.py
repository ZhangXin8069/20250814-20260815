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
grid_size = [1, 1, 1, 32]

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

# param = creat_param(
#     sink_operators = [['d^d', 'gamma_sink', 'u', 1]],
#     source_operators = [['u^d', 'gamma_source', 'd', -1]]
# )

# corr = distillation_func(
#     conf_id = conf_id,
#     param = param.data(),
    
#     lattice_size = lattice_size, # Nx, Ny, Nz, Nt 
#     grid_size = grid_size, 
    
#     backend = backend.__name__
# )

mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend.__name__)

rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()

Nx, Ny, Nz, Nt = lattice_size
Lx, Ly, Lz, Lt = [lattice_size[x]//grid_size[x] for x in range(len(lattice_size))]

fun_eigen = corr_eigvecs(Nx = Nx, backend = backend)

Mom_sink_VDV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))
Mom_sink_VVV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))[::-1]

# Mom_sink_VDV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True))
# Mom_sink_VVV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True))[::-1]

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
# sink_VVV = np.zeros((Lt, Mom_len, Nev_src, Nev_src, Nev_src), dtype = complex)


for t_src_indx, t_src in enumerate(t_rank):
    st_eigen = time.perf_counter()
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t{t_src:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')[:Nev_src]
    
    for Mom_indx in range(Mom_len):
        sink_VdV[t_src_indx, (Mom_indx):((Mom_indx + 1))] = fun_eigen.Mom_VdV_sink_t_3(phase_exp = phase_exp_2pt[(Mom_indx):((Mom_indx + 1))], eigvecs = eigvecs[:Nev_src])
        # sink_VVV[t_src_indx, Mom_indx] = fun_eigen.Mom_VVV_sink_t_3(phase_exp = phase_exp_3pt[Mom_indx], eigvecs = eigvecs[:Nev_src]).get()
        
    if rank == 0:
        print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')
if rank == 0:
    sink_VVV = np.load(f'/public/home/sush/distillation/0v2b/VVV/{conf_id}_VVV.npy')

projection = (gamma(0) + gamma(4))/2
corr_2pt_nep1_nep1 = backend.zeros((12, len(Nev_src_list), Mom_len, Mom_len, Nt, Lt), dtype = complex)
corr_2pt_nep1_prp0d = backend.zeros((12, len(Nev_src_list), Mom_len, Mom_len, Nt, Lt), dtype = complex)
corr_2pt_prp0u_prp0u = backend.zeros((24, len(Nev_src_list), Mom_len, Mom_len, Nt, Lt), dtype = complex)
corr_2pt_prp0d_prp0d = backend.zeros((12, len(Nev_src_list), Mom_len, Mom_len, Nt, Lt), dtype = complex)
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
peram_u_seq = seq_peram(peram_u)

if rank == 0:
    print(f'load peram of t_src {t_src} use time {(time.perf_counter() - st_peram):.3f} s')
    
for t_src in range(Nt):
    source_VdV = get_mpi_data(data = sink_VdV[t_src//size], mdtype = 'Bcast', root = t_src%size).transpose(0, 2, 1).conj()
    source_VVV = backend.asarray(get_mpi_data(data = sink_VVV[t_src//size], mdtype = 'Bcast', root = t_src%size)).conj()
    
    t_sink_list_rank, _, t_sink_list_indx = get_mpi_tlist(Nt = Nt, t = range(t_src, t_src + 32, 1), gtype = 'Scatter')
    peram_u_src_src = get_mpi_data(peram_u[t_src, t_src//size], mdtype = 'Bcast', root = t_src%size)
    
    st_cal = time.perf_counter()
        
    for _Nev_src_indx, _Nev_src in enumerate(Nev_src_list):
        for t_sink_indx, t_sink in enumerate(t_sink_list_indx):
            t_real_sink = t_sink_list_rank[t_sink_indx]
            
            sink_VVV_t_sink = backend.asarray(sink_VVV[t_sink])
            
            for Mom_indx in range(max(1, Mom_len//3)):
                # neutron pi+ neutron pi+
                # (0)
                corr_2pt_nep1_nep1[0, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'ambn,eqfr,isjt,cgdh,okpl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (1)
                corr_2pt_nep1_nep1[1, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'ambn,eqfr,isjt,ckdl,ogph,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (2)
                corr_2pt_nep1_nep1[2, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'ambn,esft,iqjr,cgdh,okpl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (3)
                corr_2pt_nep1_nep1[3, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'ambn,esft,iqjr,ckdl,ogph,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (4)
                corr_2pt_nep1_nep1[4, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'aqbr,emfn,isjt,cgdh,okpl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (5)
                corr_2pt_nep1_nep1[5, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'aqbr,emfn,isjt,ckdl,ogph,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (6)
                corr_2pt_nep1_nep1[6, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'aqbr,esft,imjn,cgdh,okpl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_seq[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (7)
                corr_2pt_nep1_nep1[7, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'aqbr,esft,imjn,ckdl,ogph,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_seq[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (8)
                corr_2pt_nep1_nep1[8, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'asbt,emfn,iqjr,cgdh,okpl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (9)
                corr_2pt_nep1_nep1[9, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'asbt,emfn,iqjr,ckdl,ogph,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (10)
                corr_2pt_nep1_nep1[10, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'asbt,eqfr,imjn,cgdh,okpl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_seq[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (11)
                corr_2pt_nep1_nep1[11, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'asbt,eqfr,imjn,ckdl,ogph,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_seq[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )

                # neutron pi+ proton pi0d
                # (12)
                corr_2pt_nep1_prp0d[0, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] =contract(
                    'ambn,egfh,iqjr,ckdl,ospt,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (13)
                # corr_2pt_nep1_prp0d[1, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'ambn,egfh,iqjr,csdt,okpl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_nep1_prp0d[1, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_nep1[2, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()
                
                # (14)
                corr_2pt_nep1_prp0d[2, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'ambn,eqfr,igjh,ckdl,ospt,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (15)
                corr_2pt_nep1_prp0d[3, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'ambn,eqfr,igjh,csdt,okpl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (16)
                corr_2pt_nep1_prp0d[4, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'agbh,emfn,iqjr,ckdl,ospt,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (17)
                corr_2pt_nep1_prp0d[5, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'agbh,emfn,iqjr,csdt,okpl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (18)
                corr_2pt_nep1_prp0d[6, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'agbh,eqfr,imjn,ckdl,ospt,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_seq[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (19)
                corr_2pt_nep1_prp0d[7, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'agbh,eqfr,imjn,csdt,okpl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_seq[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (20)
                corr_2pt_nep1_prp0d[8, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'aqbr,emfn,igjh,ckdl,ospt,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (21)
                corr_2pt_nep1_prp0d[9, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'aqbr,emfn,igjh,csdt,okpl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (22)
                corr_2pt_nep1_prp0d[10, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'aqbr,egfh,imjn,ckdl,ospt,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_seq[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (23)
                # corr_2pt_nep1_prp0d[11, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'aqbr,egfh,imjn,csdt,okpl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_seq[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_nep1_prp0d[11, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_nep1[6, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()
                
                # proton pi0u proton pi0u
                # (24)
                # corr_2pt_prp0u_prp0u[0, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'eqfr,ambn,cgdh,okpl,isjt,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_src_src, 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0u_prp0u[0, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_nep1[0, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()
                
                # (25)
                # corr_2pt_prp0u_prp0u[1, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'eqfr,ambn,cgdh,ospt,ikjl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_src_src, 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0u_prp0u[1, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_prp0d[0, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()
                
                # (26)
                # corr_2pt_prp0u_prp0u[2, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'eqfr,ambn,ckdl,ogph,isjt,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_src_src, 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0u_prp0u[2, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_nep1[1, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()
                
                # (27)
                # corr_2pt_prp0u_prp0u[3, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'eqfr,ambn,ckdl,ospt,igjh,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_src_src, 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0u_prp0u[3, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_prp0d[2, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()
                
                # (28)
                # corr_2pt_prp0u_prp0u[4, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'eqfr,ambn,csdt,ogph,ikjl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_src_src, 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0u_prp0u[4, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_nep1[3, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()
                
                # (29)
                # corr_2pt_prp0u_prp0u[5, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'eqfr,ambn,csdt,okpl,igjh,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_src_src, 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0u_prp0u[5, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_prp0d[3, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()

                # (30)
                corr_2pt_prp0u_prp0u[6, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'eqfr,agbh,cmdn,okpl,isjt,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u_src_src, 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )
                
                # (31)
                # corr_2pt_prp0u_prp0u[7, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'eqfr,agbh,cmdn,ospt,ikjl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u_src_src, 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0u_prp0u[7, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_prp0d[4, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()
                
                # (32)
                corr_2pt_prp0u_prp0u[8, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'eqfr,agbh,ckdl,ompn,isjt,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u_src_src, 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )

                # (33)
                # corr_2pt_prp0u_prp0u[9, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'eqfr,agbh,ckdl,ospt,imjn,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_seq[t_src, t_sink], 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0u_prp0u[9, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_prp0d[6, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()

                # (34)
                corr_2pt_prp0u_prp0u[10, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'eqfr,agbh,csdt,ompn,ikjl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u_src_src, 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )

                # (35)
                # corr_2pt_prp0u_prp0u[11, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'eqfr,agbh,csdt,okpl,imjn,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_seq[t_src, t_sink], 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0u_prp0u[11, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_prp0d[7, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()

                # (36)
                # corr_2pt_prp0u_prp0u[12, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'eqfr,akbl,cmdn,ogph,isjt,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u_src_src, 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0u_prp0u[12, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_nep1[5, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()
                
                # (37)
                # corr_2pt_prp0u_prp0u[13, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'eqfr,akbl,cmdn,ospt,igjh,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u_src_src, 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0u_prp0u[13, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_prp0d[8, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()

                # (38)
                corr_2pt_prp0u_prp0u[14, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'eqfr,akbl,cgdh,ompn,isjt,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u_src_src, 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )

                # (39)
                # corr_2pt_prp0u_prp0u[15, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'eqfr,akbl,cgdh,ospt,imjn,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_seq[t_src, t_sink], 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0u_prp0u[15, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_prp0d[10, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()

                # (40)
                corr_2pt_prp0u_prp0u[16, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'eqfr,akbl,csdt,ompn,igjh,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u_src_src, 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )

                # (41)
                # corr_2pt_prp0u_prp0u[17, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'eqfr,akbl,csdt,ogph,imjn,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_seq[t_src, t_sink], 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0u_prp0u[17, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_nep1[7, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()
                
                # (42)
                # corr_2pt_prp0u_prp0u[18, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'eqfr,asbt,cmdn,ogph,ikjl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u_src_src, 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0u_prp0u[18, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_nep1[9, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()

                # (43)
                corr_2pt_prp0u_prp0u[19, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'eqfr,asbt,cmdn,okpl,igjh,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u_src_src, 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )

                # (44)
                corr_2pt_prp0u_prp0u[20, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'eqfr,asbt,cgdh,ompn,ikjl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u_src_src, 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )

                # (45)
                # corr_2pt_prp0u_prp0u[21, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'eqfr,asbt,cgdh,okpl,imjn,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_seq[t_src, t_sink], 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0u_prp0u[21, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_nep1[10, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()

                # (46)
                corr_2pt_prp0u_prp0u[22, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'eqfr,asbt,ckdl,ompn,igjh,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u_src_src, 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )

                # (47)
                # corr_2pt_prp0u_prp0u[23, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'eqfr,asbt,ckdl,ogph,imjn,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_seq[t_src, t_sink], 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0u_prp0u[23, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_nep1[11, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()
                
                # proton pi0d proton pi0d
                # (48)
                corr_2pt_prp0d_prp0d[0, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] =contract(
                    'emfn,ogph,iqjr,akbl,csdt,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )

                # (49)
                # corr_2pt_prp0d_prp0d[1, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'emfn,ogph,iqjr,asbt,ckdl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0d_prp0d[1, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_nep1[9, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()

                # (50)
                # corr_2pt_prp0d_prp0d[2, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'emfn,oqpr,igjh,akbl,csdt,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0d_prp0d[2, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_prp0d[9, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()
                
                # (51)
                # corr_2pt_prp0d_prp0d[3, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'emfn,oqpr,igjh,asbt,ckdl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0d_prp0d[3, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_prp0u_prp0u[19, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()

                # (52)
                corr_2pt_prp0d_prp0d[4, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                    'egfh,ompn,iqjr,akbl,csdt,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                    peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                    gamma(7), gamma(5), gamma(5), gamma(7),
                    sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                    )

                # (53)
                # corr_2pt_prp0d_prp0d[5, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'egfh,ompn,iqjr,asbt,ckdl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0d_prp0d[5, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_prp0u_prp0u[20, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()
                
                # (54)
                # corr_2pt_prp0d_prp0d[6, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'egfh,oqpr,imjn,akbl,csdt,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_seq[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_nep1_prp0d[11, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_nep1[6, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()
                
                # (55)
                # corr_2pt_prp0d_prp0d[7, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'egfh,oqpr,imjn,asbt,ckdl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_seq[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0d_prp0d[7, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_nep1[10, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()

                # (56)
                # corr_2pt_prp0d_prp0d[8, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'eqfr,ompn,igjh,akbl,csdt,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0d_prp0d[8, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_prp0u_prp0u[16, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()

                # (57)
                # corr_2pt_prp0d_prp0d[9, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'eqfr,ompn,igjh,asbt,ckdl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0d_prp0d[9, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_prp0u_prp0u[22, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()
                
                # (58)
                # corr_2pt_prp0d_prp0d[10, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'eqfr,ogph,imjn,akbl,csdt,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_seq[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0d_prp0d[10, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_nep1[7, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()
                
                # (59)
                # corr_2pt_prp0d_prp0d[11, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                #     'eqfr,ogph,imjn,asbt,ckdl,ce,mo,gi,kq,Nbdf,Nnp,Mhj,Mlrt,sa->MN',
                #     peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_seq[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                #     gamma(7), gamma(5), gamma(5), gamma(7),
                #     sink_VVV_t_sink, sink_VdV[t_sink], source_VdV[(3 * Mom_indx):(3 * (Mom_indx + 1))], source_VVV[(3 * Mom_indx):(3 * (Mom_indx + 1))], projection
                #     )
                corr_2pt_prp0d_prp0d[11, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_nep1_nep1[11, _Nev_src_indx, (3 * Mom_indx):(3 * (Mom_indx + 1)), :, t_src, t_sink].copy()

    if rank == 0:
        print(f'calculate 2pt of t_src {t_src} use time {(time.perf_counter() - st_cal):.3f} s')


corr_2pt_nep1_nep1 = get_mpi_data(data = corr_2pt_nep1_nep1, mdtype = 'TGather', root = 0, axis = -1)
corr_2pt_nep1_prp0d = get_mpi_data(data = corr_2pt_nep1_prp0d, mdtype = 'TGather', root = 0, axis = -1)
corr_2pt_prp0u_prp0u = get_mpi_data(data = corr_2pt_prp0u_prp0u, mdtype = 'TGather', root = 0, axis = -1)
corr_2pt_prp0d_prp0d = get_mpi_data(data = corr_2pt_prp0d_prp0d, mdtype = 'TGather', root = 0, axis = -1)

if rank == 0:
    for _Nev_src_indx, _Nev_src in enumerate(Nev_src_list):
        corr_save_path = f'/public/home/sush/distillation/0v2b/result/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}'

        import pathlib

        path = pathlib.Path(corr_save_path)

        if path.exists():
            print('save_path:',corr_save_path)
        
        else:
            path.mkdir(parents = True, exist_ok = True)
            print('mkdir_save_path:',corr_save_path)
        
        corr_2pt_nep1_nep1 = loop_tsrc(corr_2pt_nep1_nep1, indx = [-2, -1], Boundary_Conditions = 'Antiperiodic')
        corr_2pt_nep1_prp0d = loop_tsrc(corr_2pt_nep1_prp0d, indx = [-2, -1], Boundary_Conditions = 'Antiperiodic')
        corr_2pt_prp0u_prp0u = loop_tsrc(corr_2pt_prp0u_prp0u, indx = [-2, -1], Boundary_Conditions = 'Antiperiodic')
        corr_2pt_prp0d_prp0d = loop_tsrc(corr_2pt_prp0d_prp0d, indx = [-2, -1], Boundary_Conditions = 'Antiperiodic')
        
        # np.save(
        #     f'{corr_save_path}/corr_ud_2pt_gamma0505_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy_neutron_pi+_neutron_pi+_src{_Nev_src}.npy', 
        #     corr_2pt_nep1_nep1[:, _Nev_src_indx]
        #     )
    
        # np.save(
        #     f'{corr_save_path}/corr_ud_2pt_gamma0505_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy_neutron_pi+_proton_pi0d_src{_Nev_src}.npy', 
        #     corr_2pt_nep1_prp0d[:, _Nev_src_indx]
        #     )
        
        # np.save(
        #     f'{corr_save_path}/corr_ud_2pt_gamma0505_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy_proton_pi0u_proton_pi0u_src{_Nev_src}.npy', 
        #     corr_2pt_prp0u_prp0u[:, _Nev_src_indx]
        #     )
        
        # np.save(
        #     f'{corr_save_path}/corr_ud_2pt_gamma0505_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy_proton_pi0d_proton_pi0d_src{_Nev_src}.npy', 
        #     corr_2pt_prp0d_prp0d[:, _Nev_src_indx]
        #     )
        