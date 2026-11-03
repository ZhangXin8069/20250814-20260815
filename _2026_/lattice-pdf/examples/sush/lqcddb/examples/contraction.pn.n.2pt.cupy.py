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

# Mom_sink_VDV = [[0, 0, 0]]
# Mom_sink_VVV = [[0, 0, 0]]

Mom_len = len(Mom_sink_VDV)

phase_exp_2pt = backend.zeros((Mom_len, Nx, Nx, Nx, Nc), dtype = complex)
phase_exp_3pt = backend.zeros((Mom_len, Nx, Nx, Nx), dtype = complex)

for Mom_indx in range(Mom_len):
    phase_exp_2pt[Mom_indx] = fun_eigen.phase_exp_2pt(Mom = Mom_sink_VDV[Mom_indx])
    phase_exp_3pt[Mom_indx] = fun_eigen.phase_exp_3pt(Mom = Mom_sink_VVV[Mom_indx])

Nev_src = 100

t_rank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'TScatter')

import numpy as np

sink_VdV = backend.zeros((Lt, Mom_len, Nev_src, Nev_src), dtype = complex)
sink_VVV = np.zeros((Lt, Mom_len, Nev_src, Nev_src, Nev_src), dtype = complex)

st_eigen = time.perf_counter()
for t_src_indx, t_src in enumerate(t_rank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t{t_src:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')[:Nev_src]
    
    for Mom_indx in range(Mom_len):
        sink_VdV[t_src_indx, (Mom_indx):((Mom_indx + 1))] = fun_eigen.Mom_VdV_sink_t_3(phase_exp = phase_exp_2pt[(Mom_indx):((Mom_indx + 1))], eigvecs = eigvecs[:Nev_src])
        sink_VVV[t_src_indx, Mom_indx] = fun_eigen.Mom_VVV_sink_t_3(phase_exp = phase_exp_3pt[Mom_indx], eigvecs = eigvecs[:Nev_src]).get()
        
# if rank == 0:
#     sink_VVV = np.load(f'/public/home/sush/distillation/0v2b/VVV/{conf_id}_VVV.npy')
#     print(sink_VVV.shape)
    
# else:
#     sink_VVV = None

# sink_VVV = get_mpi_data(sink_VVV, mdtype = 'TScatter', axis = 0)

if rank == 0:
    print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')
    
projector_1 = (gamma(0) + gamma(4))/2
projector_2 = (gamma(0) - gamma(4))/2
gamma_proton = gamma(7)
gamma_pion = gamma(5)

corr_2pt_proton_pi0u_proton = backend.zeros((6, Mom_len, 1, Nt, Lt), dtype = complex)
corr_2pt_proton_pi0d_proton = backend.zeros((4, Mom_len, 1, Nt, Lt), dtype = complex)
corr_2pt_proton_pi1_neutron = backend.zeros((4, Mom_len, 1, Nt, Lt), dtype = complex)

corr_2pt_pi0u_proton_proton = backend.zeros((6, 1, Mom_len, Nt, Lt), dtype = complex)
corr_2pt_pi0d_proton_proton = backend.zeros((4, 1, Mom_len, Nt, Lt), dtype = complex)
corr_2pt_pi1_neutron_proton = backend.zeros((4, 1, Mom_len, Nt, Lt), dtype = complex)

corr_2pt_proton_proton = backend.zeros((4, Mom_len, Nt, Lt), dtype = complex)

st_peram = time.perf_counter()

peram_u = backend.zeros((Lt, Nt, Ns, Ns, Nev_src, Nev_src), dtype = complex)

for t_src_indx, t_src in enumerate(t_rank):
    peram_u[t_src_indx] = backend.load(
        f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{t_src:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy'
        )[..., :Nev_src, :Nev_src]

if rank == 0:
    print(f'load peram of t_src {t_src} use time {(time.perf_counter() - st_peram):.3f} s')

# peram_u = contract('ab,Ttbcef,cd->tTdafe', gamma(5), peram_u.conj(), gamma(5))

for t_src in range(Nt):
    source_VdV = get_mpi_data(data = sink_VdV[t_src//size], mdtype = 'Bcast', root = t_src%size).transpose(0, 2, 1).conj()
    source_VVV = backend.asarray(get_mpi_data(data = sink_VVV[t_src//size], mdtype = 'Bcast', root = t_src%size)).conj()
    
    peram_u_src_src = get_mpi_data(peram_u[t_src, t_src//size], mdtype = 'Bcast', root = t_src%size)
    
    st_cal = time.perf_counter()
    t_sink_list_rank, _, t_sink_list_indx = get_mpi_tlist(Nt = Nt, t = range(t_src, t_src + 32, 1), gtype = 'TScatter')
    for t_sink_indx, t_sink in enumerate(t_sink_list_indx):
        t_real_sink = t_sink_list_rank[t_sink_indx]
        sink_VVV_t_sink = backend.asarray(sink_VVV[t_sink])
        
        for Mom_indx in range(max(1, Mom_len//9)):
            corr_2pt_proton_proton[2, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = contract(
                'eofp,ambn,cgdh,ce,mo,Mbdf,Mnph,ga->M',
                peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                gamma_proton, gamma_proton,
                sink_VVV_t_sink[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[(9 * Mom_indx):(9 * (Mom_indx + 1))],
                projector_2 @ projector_2
            )
            
            corr_2pt_proton_proton[3, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = contract(
                'eofp,agbh,cmdn,ce,mo,Mbdf,Mnph,ga->M',
                peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                gamma_proton, gamma_proton,
                sink_VVV_t_sink[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[(9 * Mom_indx):(9 * (Mom_indx + 1))],
                projector_2 @ projector_2
            )
            
    t_sink_list_rank, _, t_sink_list_indx = get_mpi_tlist(Nt = Nt, t = range(t_src, t_src - 32, -1), gtype = 'TScatter')
    for t_sink_indx, t_sink in enumerate(t_sink_list_indx):
        t_real_sink = t_sink_list_rank[t_sink_indx]
        sink_VVV_t_sink = backend.asarray(sink_VVV[t_sink])
        
        for Mom_indx in range(max(1, Mom_len//9)):
            corr_2pt_proton_pi0u_proton[0, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                'eifj,ambn,cgdh,okpl,ce,mo,gi,Nbdf,Mnp,Mhjl,ka->MN',
                peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_src_src,
                gamma_proton, gamma_pion, gamma_proton,
                sink_VVV_t_sink[0:1], source_VdV[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[(9 * Mom_indx):(9 * (Mom_indx + 1))],
                projector_1
            )
            
            corr_2pt_proton_pi0u_proton[1, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                'eifj,ambn,ckdl,ogph,ce,mo,gi,Nbdf,Mnp,Mhjl,ka->MN',
                peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_src_src,
                gamma_proton, gamma_pion, gamma_proton,
                sink_VVV_t_sink[0:1], source_VdV[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[(9 * Mom_indx):(9 * (Mom_indx + 1))],
                projector_1
            )
            
            corr_2pt_proton_pi0u_proton[2, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                'eifj,agbh,cmdn,okpl,ce,mo,gi,Nbdf,Mnp,Mhjl,ka->MN',
                peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_src_src,
                gamma_proton, gamma_pion, gamma_proton,
                sink_VVV_t_sink[0:1], source_VdV[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[(9 * Mom_indx):(9 * (Mom_indx + 1))],
                projector_1
            )
            
            corr_2pt_proton_pi0u_proton[3, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                'eifj,agbh,ckdl,ompn,ce,mo,gi,Nbdf,Mnp,Mhjl,ka->MN',
                peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_src_src,
                gamma_proton, gamma_pion, gamma_proton,
                sink_VVV_t_sink[0:1], source_VdV[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[(9 * Mom_indx):(9 * (Mom_indx + 1))],
                projector_1
            )
            
            corr_2pt_proton_pi0u_proton[4, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                'eifj,akbl,cmdn,ogph,ce,mo,gi,Nbdf,Mnp,Mhjl,ka->MN',
                peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_src_src,
                gamma_proton, gamma_pion, gamma_proton,
                sink_VVV_t_sink[0:1], source_VdV[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[(9 * Mom_indx):(9 * (Mom_indx + 1))],
                projector_1
            )
            
            corr_2pt_proton_pi0u_proton[5, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                'eifj,akbl,cgdh,ompn,ce,mo,gi,Nbdf,Mnp,Mhjl,ka->MN',
                peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u_src_src,
                gamma_proton, gamma_pion, gamma_proton,
                sink_VVV_t_sink[0:1], source_VdV[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[(9 * Mom_indx):(9 * (Mom_indx + 1))],
                projector_1
            )
            
            corr_2pt_proton_pi0d_proton[0, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
                'emfn,oipj,agbh,ckdl,ce,mo,gi,Nbdf,Mnp,Mhjl,ka->MN',
                peram_u[t_src, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                gamma_proton, gamma_pion, gamma_proton,
                sink_VVV_t_sink[0:1], source_VdV[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[(9 * Mom_indx):(9 * (Mom_indx + 1))],
                projector_1
            )
            
            # corr_2pt_proton_pi0d_proton[1, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
            #     'emfn,oipj,akbl,cgdh,ce,mo,gi,Nbdf,Mnp,Mhjl,ka->MN',
            #     peram_u[t_src, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
            #     gamma_proton, gamma_pion, gamma_proton,
            #     sink_VVV_t_sink[0:1], source_VdV[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[(9 * Mom_indx):(9 * (Mom_indx + 1))],
            #     projector_1
            # )
            corr_2pt_proton_pi0d_proton[1, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_proton_pi0u_proton[4, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink]
            
            # corr_2pt_proton_pi0d_proton[2, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
            #     'eifj,ompn,agbh,ckdl,ce,mo,gi,Nbdf,Mnp,Mhjl,ka->MN',
            #     peram_u[t_src, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
            #     gamma_proton, gamma_pion, gamma_proton,
            #     sink_VVV_t_sink[0:1], source_VdV[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[(9 * Mom_indx):(9 * (Mom_indx + 1))],
            #     projector_1
            # )
            corr_2pt_proton_pi0d_proton[2, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_proton_pi0u_proton[3, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink]
            
            # corr_2pt_proton_pi0d_proton[3, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
            #     'eifj,ompn,akbl,cgdh,ce,mo,gi,Nbdf,Mnp,Mhjl,ka->MN',
            #     peram_u[t_src, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
            #     gamma_proton, gamma_pion, gamma_proton,
            #     sink_VVV_t_sink[0:1], source_VdV[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[(9 * Mom_indx):(9 * (Mom_indx + 1))],
            #     projector_1
            # )
            corr_2pt_proton_pi0d_proton[3, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_proton_pi0u_proton[5, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink]
            # corr_2pt_proton_pi1_neutron[0, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
            #     'eifj,okpl,ambn,cgdh,ce,mo,gi,Nbdf,Mnp,Mhjl,ka->MN',
            #     peram_u[t_src, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
            #     gamma_proton, gamma_pion, gamma_proton,
            #     sink_VVV_t_sink[0:1], source_VdV[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[(9 * Mom_indx):(9 * (Mom_indx + 1))],
            #     projector_1
            # )
            corr_2pt_proton_pi1_neutron[0, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_proton_pi0u_proton[0, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink]
            
            # corr_2pt_proton_pi1_neutron[1, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
            #     'eifj,okpl,agbh,cmdn,ce,mo,gi,Nbdf,Mnp,Mhjl,ka->MN',
            #     peram_u[t_src, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
            #     gamma_proton, gamma_pion, gamma_proton,
            #     sink_VVV_t_sink[0:1], source_VdV[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[(9 * Mom_indx):(9 * (Mom_indx + 1))],
            #     projector_1
            # )
            corr_2pt_proton_pi1_neutron[1, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_proton_pi0u_proton[2, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink]
            
            # corr_2pt_proton_pi1_neutron[2, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
            #     'ekfl,oipj,ambn,cgdh,ce,mo,gi,Nbdf,Mnp,Mhjl,ka->MN',
            #     peram_u[t_src, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
            #     gamma_proton, gamma_pion, gamma_proton,
            #     sink_VVV_t_sink[0:1], source_VdV[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[(9 * Mom_indx):(9 * (Mom_indx + 1))],
            #     projector_1
            # )
            corr_2pt_proton_pi1_neutron[2, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_proton_pi0u_proton[1, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink]
            
            # corr_2pt_proton_pi1_neutron[3, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = contract(
            #     'ekfl,oipj,agbh,cmdn,ce,mo,gi,Nbdf,Mnp,Mhjl,ka->MN',
            #     peram_u[t_src, t_sink], peram_u_src_src, peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
            #     gamma_proton, gamma_pion, gamma_proton,
            #     sink_VVV_t_sink[0:1], source_VdV[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[(9 * Mom_indx):(9 * (Mom_indx + 1))],
            #     projector_1
            # )
            corr_2pt_proton_pi1_neutron[3, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink] = corr_2pt_proton_pi0d_proton[0, (9 * Mom_indx):(9 * (Mom_indx + 1)), :, t_src, t_sink]

            #*******************************************************************************************************************************************************************************************
            
            corr_2pt_pi0u_proton_proton[0, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = contract(
                'oipj,cadb,egfh,mknl,ac,mo,gi,Nbd,Nfnp,Mhjl,ke->MN',
                peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink],
                gamma_pion, gamma_proton, gamma_proton, 
                sink_VdV[t_sink, (9 * Mom_indx):(9 * (Mom_indx + 1))], sink_VVV_t_sink[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[0:1],
                projector_2
            )
            
            corr_2pt_pi0u_proton_proton[1, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = contract(
                'oipj,cadb,ekfl,mgnh,ac,mo,gi,Nbd,Nfnp,Mhjl,ke->MN',
                peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink],
                gamma_pion, gamma_proton, gamma_proton, 
                sink_VdV[t_sink, (9 * Mom_indx):(9 * (Mom_indx + 1))], sink_VVV_t_sink[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[0:1],
                projector_2
            )
            
            corr_2pt_pi0u_proton_proton[2, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = contract(
                'oipj,cgdh,eafb,mknl,ac,mo,gi,Nbd,Nfnp,Mhjl,ke->MN',
                peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink],
                gamma_pion, gamma_proton, gamma_proton, 
                sink_VdV[t_sink, (9 * Mom_indx):(9 * (Mom_indx + 1))], sink_VVV_t_sink[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[0:1],
                projector_2
            )
            
            corr_2pt_pi0u_proton_proton[3, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = contract(
                'oipj,cgdh,ekfl,manb,ac,mo,gi,Nbd,Nfnp,Mhjl,ke->MN',
                peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink],
                gamma_pion, gamma_proton, gamma_proton, 
                sink_VdV[t_sink, (9 * Mom_indx):(9 * (Mom_indx + 1))], sink_VVV_t_sink[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[0:1],
                projector_2
            )
            
            corr_2pt_pi0u_proton_proton[4, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = contract(
                'oipj,ckdl,eafb,mgnh,ac,mo,gi,Nbd,Nfnp,Mhjl,ke->MN',
                peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink],
                gamma_pion, gamma_proton, gamma_proton, 
                sink_VdV[t_sink, (9 * Mom_indx):(9 * (Mom_indx + 1))], sink_VVV_t_sink[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[0:1],
                projector_2
            )
            
            corr_2pt_pi0u_proton_proton[5, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = contract(
                'oipj,ckdl,egfh,manb,ac,mo,gi,Nbd,Nfnp,Mhjl,ke->MN',
                peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink],
                gamma_pion, gamma_proton, gamma_proton, 
                sink_VdV[t_sink, (9 * Mom_indx):(9 * (Mom_indx + 1))], sink_VVV_t_sink[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[0:1],
                projector_2
            )

            # corr_2pt_pi0d_proton_proton[0, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = contract(
            #     'cadb,oipj,egfh,mknl,ac,mo,gi,Nbd,Nfnp,Mhjl,ke->MN',
            #     peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink],
            #     gamma_pion, gamma_proton, gamma_proton, 
            #     sink_VdV[t_sink, (9 * Mom_indx):(9 * (Mom_indx + 1))], sink_VVV_t_sink[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[0:1],
            #     projector_2
            # )
            corr_2pt_pi0d_proton_proton[0, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = corr_2pt_pi0u_proton_proton[0, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink]
            
            # corr_2pt_pi0d_proton_proton[1, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = contract(
            #     'cadb,oipj,ekfl,mgnh,ac,mo,gi,Nbd,Nfnp,Mhjl,ke->MN',
            #     peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink],
            #     gamma_pion, gamma_proton, gamma_proton, 
            #     sink_VdV[t_sink, (9 * Mom_indx):(9 * (Mom_indx + 1))], sink_VVV_t_sink[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[0:1],
            #     projector_2
            # )
            corr_2pt_pi0d_proton_proton[1, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = corr_2pt_pi0u_proton_proton[1, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink]
            
            corr_2pt_pi0d_proton_proton[2, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = contract(
                'cidj,oapb,egfh,mknl,ac,mo,gi,Nbd,Nfnp,Mhjl,ke->MN',
                peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink],
                gamma_pion, gamma_proton, gamma_proton, 
                sink_VdV[t_sink, (9 * Mom_indx):(9 * (Mom_indx + 1))], sink_VVV_t_sink[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[0:1],
                projector_2
            )
            
            # corr_2pt_pi0d_proton_proton[3, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = contract(
            #     'cidj,oapb,ekfl,mgnh,ac,mo,gi,Nbd,Nfnp,Mhjl,ke->MN',
            #     peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink],
            #     gamma_pion, gamma_proton, gamma_proton, 
            #     sink_VdV[t_sink, (9 * Mom_indx):(9 * (Mom_indx + 1))], sink_VVV_t_sink[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[0:1],
            #     projector_2
            # )
            corr_2pt_pi0d_proton_proton[3, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = corr_2pt_pi0u_proton_proton[3, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink]
            
            # corr_2pt_pi1_neutron_proton[0, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = contract(
            #     'eafb,oipj,cgdh,mknl,ac,mo,gi,Nbd,Nfnp,Mhjl,ke->MN',
            #     peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink],
            #     gamma_pion, gamma_proton, gamma_proton, 
            #     sink_VdV[t_sink, (9 * Mom_indx):(9 * (Mom_indx + 1))], sink_VVV_t_sink[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[0:1],
            #     projector_2
            # )
            corr_2pt_pi1_neutron_proton[0, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = corr_2pt_pi0u_proton_proton[2, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink]
            
            # corr_2pt_pi1_neutron_proton[1, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = contract(
            #     'eafb,oipj,ckdl,mgnh,ac,mo,gi,Nbd,Nfnp,Mhjl,ke->MN',
            #     peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink],
            #     gamma_pion, gamma_proton, gamma_proton, 
            #     sink_VdV[t_sink, (9 * Mom_indx):(9 * (Mom_indx + 1))], sink_VVV_t_sink[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[0:1],
            #     projector_2
            # )
            corr_2pt_pi1_neutron_proton[1, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = corr_2pt_pi0u_proton_proton[4, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink]
            
            # corr_2pt_pi1_neutron_proton[2, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = contract(
            #     'eifj,oapb,cgdh,mknl,ac,mo,gi,Nbd,Nfnp,Mhjl,ke->MN',
            #     peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink],
            #     gamma_pion, gamma_proton, gamma_proton, 
            #     sink_VdV[t_sink, (9 * Mom_indx):(9 * (Mom_indx + 1))], sink_VVV_t_sink[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[0:1],
            #     projector_2
            # )
            corr_2pt_pi1_neutron_proton[2, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = corr_2pt_pi0d_proton_proton[2, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink]
            
            # corr_2pt_pi1_neutron_proton[3, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = contract(
            #     'eifj,oapb,ckdl,mgnh,ac,mo,gi,Nbd,Nfnp,Mhjl,ke->MN',
            #     peram_u[t_src, t_sink], peram_u[t_real_sink, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink],
            #     gamma_pion, gamma_proton, gamma_proton, 
            #     sink_VdV[t_sink, (9 * Mom_indx):(9 * (Mom_indx + 1))], sink_VVV_t_sink[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[0:1],
            #     projector_2
            # )
            corr_2pt_pi1_neutron_proton[3, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = corr_2pt_pi0u_proton_proton[5, :, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink]

            corr_2pt_proton_proton[0, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = contract(
                'eofp,ambn,cgdh,ce,mo,Mbdf,Mnph,ga->M',
                peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                gamma_proton, gamma_proton,
                sink_VVV_t_sink[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[(9 * Mom_indx):(9 * (Mom_indx + 1))],
                projector_1 @ projector_1
            )
            
            corr_2pt_proton_proton[1, (9 * Mom_indx):(9 * (Mom_indx + 1)), t_src, t_sink] = contract(
                'eofp,agbh,cmdn,ce,mo,Mbdf,Mnph,ga->M',
                peram_u[t_src, t_sink], peram_u[t_src, t_sink], peram_u[t_src, t_sink], 
                gamma_proton, gamma_proton,
                sink_VVV_t_sink[(9 * Mom_indx):(9 * (Mom_indx + 1))], source_VVV[(9 * Mom_indx):(9 * (Mom_indx + 1))],
                projector_1 @ projector_1
            )
        
    if rank == 0:
        print(f'calculate 2pt of t_src {t_src} use time {(time.perf_counter() - st_cal):.3f} s')

corr_2pt_proton_pi0u_proton = get_mpi_data(data = corr_2pt_proton_pi0u_proton, mdtype = 'TGather', root = 0, axis = -1)
corr_2pt_proton_pi0d_proton = get_mpi_data(data = corr_2pt_proton_pi0d_proton, mdtype = 'TGather', root = 0, axis = -1)
corr_2pt_proton_pi1_neutron = get_mpi_data(data = corr_2pt_proton_pi1_neutron, mdtype = 'TGather', root = 0, axis = -1)

corr_2pt_pi0u_proton_proton = get_mpi_data(data = corr_2pt_pi0u_proton_proton, mdtype = 'TGather', root = 0, axis = -1)
corr_2pt_pi0d_proton_proton = get_mpi_data(data = corr_2pt_pi0d_proton_proton, mdtype = 'TGather', root = 0, axis = -1)
corr_2pt_pi1_neutron_proton = get_mpi_data(data = corr_2pt_pi1_neutron_proton, mdtype = 'TGather', root = 0, axis = -1)


corr_2pt_proton_proton = get_mpi_data(data = corr_2pt_proton_proton, mdtype = 'TGather', root = 0, axis = -1)

if rank == 0:
    corr_save_path = f'/public/home/sush/distillation/0v2b/result/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}'

    import pathlib

    path = pathlib.Path(corr_save_path)

    if path.exists():
        print('save_path:',corr_save_path)
    
    else:
        path.mkdir(parents = True, exist_ok = True)
        print('mkdir_save_path:',corr_save_path)
    
    corr_2pt_proton_pi0u_proton = loop_tsrc(corr_2pt_proton_pi0u_proton, indx = [-2, -1], Boundary_Conditions = 'Antiperiodic')
    corr_2pt_proton_pi0d_proton = loop_tsrc(corr_2pt_proton_pi0d_proton, indx = [-2, -1], Boundary_Conditions = 'Antiperiodic')
    corr_2pt_proton_pi1_neutron = loop_tsrc(corr_2pt_proton_pi1_neutron, indx = [-2, -1], Boundary_Conditions = 'Antiperiodic')
    
    corr_2pt_pi0u_proton_proton = loop_tsrc(corr_2pt_pi0u_proton_proton, indx = [-2, -1], Boundary_Conditions = 'Antiperiodic')
    corr_2pt_pi0d_proton_proton = loop_tsrc(corr_2pt_pi0d_proton_proton, indx = [-2, -1], Boundary_Conditions = 'Antiperiodic')
    corr_2pt_pi1_neutron_proton = loop_tsrc(corr_2pt_pi1_neutron_proton, indx = [-2, -1], Boundary_Conditions = 'Antiperiodic')
    
    corr_2pt_proton_proton = loop_tsrc(corr_2pt_proton_proton, indx = [-2, -1], Boundary_Conditions = 'Antiperiodic')
    
    backend.save(f'{corr_save_path}/corr_2pt_proton-_proton-_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy', corr_2pt_proton_proton)
    
    backend.save(f'{corr_save_path}/corr_2pt_proton-_pi0u_proton_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy', corr_2pt_proton_pi0u_proton[..., ::-1])
    backend.save(f'{corr_save_path}/corr_2pt_proton-_pi0d_proton_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy', corr_2pt_proton_pi0d_proton[..., ::-1])
    backend.save(f'{corr_save_path}/corr_2pt_proton-_pi1_neutron_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy', corr_2pt_proton_pi1_neutron[..., ::-1])

    backend.save(f'{corr_save_path}/corr_2pt_pi0u_proton_proton-_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy', corr_2pt_pi0u_proton_proton[..., ::-1])
    backend.save(f'{corr_save_path}/corr_2pt_pi0d_proton_proton-_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy', corr_2pt_pi0d_proton_proton[..., ::-1])
    backend.save(f'{corr_save_path}/corr_2pt_pi1_neutron_proton-_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy', corr_2pt_pi1_neutron_proton[..., ::-1])