"""
这是一个mpi计算pion 矢量流的例子
"""

import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from function_contraction import *

from opt_einsum import contract, contract_path

import time

set_backend('numpy')
backend = get_backend()

lattice_size = [12, 12, 12, 32]
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

mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend)
rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()

Nx, Ny, Nz, Nt = lattice_size
Lx, Ly, Lz, Lt = [lattice_size[x]//grid_size[x] for x in range(len(lattice_size))]

fun_eigen = corr_eigvecs(Nx = Nx, backend = backend)
phase_exp = fun_eigen.phase_exp_2pt(Mom = [0, 0, 0])


Nev_src = 50
Nev_link = 200
link_max = 10

t_sep = 8

t_rank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'Scatter')

if rank == 0:
    gauge_link = Readin_gauge(
        f'/nexdata/project/lqcd/sush/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L12x32/beta6.20_mu-0.2770_ms-0.2400_L12x32_cfg_{conf_id}.lime.contents/msg02.rec04.ildg-binary-data', 
        lattice_size = lattice_size
        )
else:
    gauge_link = None

gauge_link = get_mpi_data(data = gauge_link, mdtype = 'TScatter', root = 0, axis = 0)
gauge_link = gauge_link.transpose(4, 0, 1, 2, 3, 5, 6)

sink = backend.zeros((Lt, Nev_src, Nev_src), dtype = complex)
VdV_link = backend.zeros((Lt, 2 * link_max + 1, Nev_link, Nev_link), dtype = complex)

for t_src_indx, t_src in enumerate(t_rank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L12x32/{conf_id}/{conf_id}_t{t_src:03d}_e50_s[0]_[0]_[0]_BI_n150_V1_stout_smear_20_0.12.npy')
    sink[t_src_indx] = fun_eigen.Mom_VdV_sink_t_2(phase_exp = phase_exp, eigvecs = eigvecs[:Nev_src])
    
    VdV_link[t_src_indx] = fun_eigen.VdV_sink_t_link(
        eigvecs = eigvecs[:Nev_link],
        link_dir = 'Z',
        link_max = link_max,
        phase_exp = phase_exp,
        gauge_link = gauge_link,
        t = t_src_indx
    )
    
VdV_link = VdV_link * fun_eigen.create_omega_accelerate(
    exact = 50,
    N_eigen = [0],
    N_sum = [0],
    N_extract = [0],
    noise = 150,
)

corr_2pt = backend.zeros((Nt, Lt), dtype = complex)
corr_3pt_con = backend.zeros((2 * link_max + 1, Nt, Lt), dtype = complex)
bubble = backend.zeros((2 * link_max + 1, Nt), dtype = complex)

for t_src in range(32):
    source = get_mpi_data(data = sink[t_src//size], mdtype = 'Bcast', root = t_src%size).transpose(1, 0).conj()
    t_sink_list_rank, _, t_sink_list_indx = get_mpi_tlist(Nt = Nt, t = range(t_src, t_src + 20, 1), gtype = 'Scatter')
    
    if rank == 0:
        peram_u_src = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L12x32/light/{conf_id}/t{t_src:03d}_e50_s[0]_[0]_[0]_BI_n150_V1_stout_smear_20_0.12.npy'
            )

    else:
        peram_u_src = None
        
    peram_u_src = get_mpi_data(data = peram_u_src, mdtype = 'TScatter', root = 0, axis = 0)
    peram_d_src = seq_peram(peram_u_src)
    
    for t_sink in t_sink_list_indx:
        corr_2pt[t_src, t_sink] = contract(
            'manb,cedf,bd,fn,ac,em->',
            peram_u_src[t_sink, :, :, :Nev_src, :Nev_src],
            peram_d_src[t_sink, :, :, :Nev_src, :Nev_src],
            source,
            sink[t_sink],  
            backend.asarray(gamma(5)),
            backend.asarray(gamma(5)),
            )
    
    if rank == 0:
        peram_u_sep = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L12x32/light/{conf_id}/t{(t_src + t_sep)%Nt:03d}_e50_s[0]_[0]_[0]_BI_n150_V1_stout_smear_20_0.12.npy'
            )

    else:
        peram_u_sep = None
    
    peram_u_sep = get_mpi_data(data = peram_u_sep, mdtype = 'TScatter', root = 0, axis = 0)
    peram_u_sep = seq_peram(peram_u_sep)
    peram_d_2pt = get_mpi_data(data = peram_d_src[(t_src + t_sep)%Nt//size], mdtype = 'Bcast', root = (t_src + t_sep)%size)
    sink_3pt = get_mpi_data(data = sink[(t_src + t_sep)%Nt//size], mdtype = 'Bcast', root = (t_src + t_sep)%size)
    
    for t_curr in t_sink_list_indx:
        corr_3pt_con[:, t_src, t_curr] = contract(
            'manb,gehf,codp,bd,Lfn,ph,ac,em,og->L',
            peram_u_src[t_curr, :, :, :Nev_link, :Nev_src],
            peram_u_sep[t_curr, :, :, :Nev_src, :Nev_link],
            peram_d_2pt[:, :, :Nev_src, :Nev_src],
            source,
            VdV_link[t_curr],
            sink_3pt,
            backend.asarray(gamma(5)),
            backend.asarray(gamma(4)),
            backend.asarray(gamma(5)),
        )
    
    bubble[:, t_src] = contract(
        'menf,Lfn,em->L',
        peram_u_src[t_src//size],
        VdV_link[t_src//size],
        backend.asarray(gamma(4))
    )
    
corr_2pt = get_mpi_data(data = corr_2pt, mdtype = 'TGather', root = 0, axis = -1)
corr_3pt_con = get_mpi_data(data = corr_3pt_con, mdtype = 'TGather', root = 0, axis = -1)
bubble = get_mpi_data(data = bubble[:, rank::size], mdtype = 'TGather', root = 0, axis = -1)

if rank == 0:
    backend.save(f'/public/home/sush/distillation/pion/result/12x32/Px0Py0Pz0/ENV_200/conf{conf_id}/corr_ud_2pt_gamma0505_u_e50_s[0]_[0]_[0]_BI_n150_stout_smear_20_0.12_dul_vector_False_src{Nev_src}.npy', corr_2pt)
    
    for link_indx in range(-link_max, link_max + 1, 1):
        backend.save(f'/public/home/sush/distillation/pion/result/12x32/Px0Py0Pz0/ENV_200/conf{conf_id}/corr_ud_3pt_gamma050405_tseq{t_sep}_link_indx{link_indx}_u_e50_s[0]_[0]_[0]_BI_n150_stout_smear_20_0.12_dul_vector_False_src{Nev_src}_curr{Nev_link}.npy', corr_3pt_con[link_indx + link_max])
        backend.save(f'/public/home/sush/distillation/pion/result/12x32/Px0Py0Pz0/ENV_200/conf{conf_id}/corr_ud_bubble_gamma04_link_indx{link_indx}_u_e50_s[0]_[0]_[0]_BI_n150_stout_smear_20_0.12_dul_vector_False_curr{Nev_link}.npy', bubble[link_indx + link_max])
        