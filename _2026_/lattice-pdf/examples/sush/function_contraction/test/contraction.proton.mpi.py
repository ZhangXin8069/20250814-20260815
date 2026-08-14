"""
这是一个利用mpi计算质子的两点和矢量流三点的例子
"""

import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from function_contraction import *

from opt_einsum import contract, contract_path

import time
from typing import Literal

def Readin_gauge(conf_file, lattice_size):
    
    backend = get_backend()
    Nz, Ny, Nx, Nt = lattice_size
    
    f = open("%s" % conf_file, "rb")
    gauge = backend.fromfile(f, dtype=">f8")
    gauge = backend.array(gauge)

    gauge = gauge.reshape(Nt, Nx, Nx, Nx, 4, 3, 3, 2)
    gauge = gauge[..., 0] + gauge[..., 1] * 1j
    f.close()

    return gauge

def transpose_gauge(gauge_link, link_indx:int, link_dir:Literal["Z", "Y", "X"]):
    backend = get_backend()
    
    if link_dir=='Z':   axis_dir=1
    elif link_dir=='Y':   axis_dir=2
    elif link_dir=='X':   axis_dir=3
    
    _gauge_link = gauge_link[3 - axis_dir]

    gauge_link_rolled = backend.zeros_like(_gauge_link)
    gauge_link_rolled[:] = backend.identity(3, dtype=complex)
    
    if link_indx < 0:
        for link_indx_2 in range(abs(link_indx)):
            gauge_link_rolled = gauge_link_rolled @ backend.roll(_gauge_link, abs(link_indx)-link_indx_2, axis = axis_dir)

        gauge_link_rolled = gauge_link_rolled.transpose(0, 1, 2, 3, 5, 4).conj()
        
    elif link_indx > 0:
        for link_indx_2 in range(abs(link_indx)):
            gauge_link_rolled = gauge_link_rolled @ backend.roll(_gauge_link, -1* link_indx_2, axis = axis_dir)
            
    return gauge_link_rolled

conf_id = sys.argv[1]

import numpy as np
# 初始化numpy or cupy
set_backend('numpy')
backend = get_backend()

# 格子数据和 mpi 
lattice_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 24]
mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend.__name__)

conf_id = sys.argv[1]

rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()

Nx, Ny, Nz, Nt = lattice_size

t_sep = 10
link_max = 10
Nev_src = 150

#读取规范场
if rank == 0:
    gauge_link = Readin_gauge(
        f'/public/home/sush/share_work/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{conf_id}.lime.contents/msg02.rec04.ildg-binary-data', 
        lattice_size = lattice_size
        )

    #将规范场的维度设置为(Nd, Nt, Nz, Ny, Nx, Nc, Nc)即link方向，时间，Z, Y, X, c, c
    gauge_link = gauge_link.transpose(4, 0, 1, 2, 3, 5, 6)

else:
    gauge_link = None

gauge_link = get_mpi_data(data = gauge_link, mdtype = 'TScatter', root = 0, axis = 1)

fun_eigen = corr_eigvecs(Nx = Nx, backend = backend)
# sink Mom
Mom_sink = creat_mom_list(Mom = [0, 0, 0], fix_Q2 = True) + creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)# P = [0, 0, 0]
phase_exp_3pt = backend.empty((len(Mom_sink), Nx, Nx, Nx), dtype = complex) # e^{-ipx}
phase_exp_2pt = backend.empty((1, Nx, Nx, Nx, 3), dtype = complex) # e^{-ipx}

for i in range(len(Mom_sink)):
    phase_exp_3pt[i] = backend.asarray(fun_eigen.phase_exp_3pt(Mom = Mom_sink[i]))
phase_exp_2pt[0] = backend.asarray(fun_eigen.phase_exp_2pt(Mom = Mom_sink[0]))

t_rank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'Scatter')

sink_VVV = np.zeros((Nt//size, len(Mom_sink), Nev_src, Nev_src, Nev_src), dtype = complex)
curr_VdV = np.zeros((Nt//size, 1, 2 * link_max + 1, 650, 650), dtype = complex)

st_eigen = time.perf_counter()
for t_src_indx, t_src in enumerate(t_rank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/{conf_id}_t{t_src:03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy')
    
    for Mom_indx in range(len(Mom_sink)):
        sink_VVV[t_src_indx] = fun_eigen.Mom_VVV_sink_t_3(phase_exp = phase_exp_3pt[Mom_indx], eigvecs = eigvecs[:Nev_src])
    
    curr_VdV[t_src_indx] = fun_eigen.VdV_sink_t_link(eigvecs = eigvecs, link_dir = 'Z', link_max = link_max, phase_exp = phase_exp_2pt, t = t_src_indx, gauge_link = gauge_link)
    
curr_VdV = curr_VdV * fun_eigen.create_omega_accelerate(
    exact = 50,
    N_eigen = [100, 200, 300, 500, 800],
    N_sum = [20, 20, 20, 20, 20],
    N_extract = [10, 10, 10, 10, 10],
    noise = 500
)

if rank == 0:
    print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')

gamma_curr = backend.asarray([gamma(4)])

projection = backend.asarray([
    (gamma(0) + gamma(4))/2
    ])

corr_2pt_matrix = backend.zeros((len(Mom_sink), Nt, Nt//size), dtype = complex)
corr_3pt_matrix_d = backend.zeros((len(gamma_curr), len(Mom_sink), 2 * link_max + 1, Nt, Nt//size), dtype = complex)
corr_3pt_matrix_u = backend.zeros((len(gamma_curr), len(Mom_sink), 2 * link_max + 1, Nt, Nt//size), dtype = complex)


for t_src in range(Nt):
    st_mpi_time = time.perf_counter()
    src_VVV = get_mpi_data(data = sink_VVV[t_src//size], mdtype = 'Bcast', root = t_src%size).conj()
    _, _, t_sink_list_indx = get_mpi_tlist(Nt = Nt, t = range(t_src, t_src + t_sep + 1, 1), gtype = 'Scatter')
    
    if rank == 0:
        peram_u_src = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{t_src:03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy'
            )[..., :Nev_src]

    else:
        peram_u_src = None
        
    peram_u_src = get_mpi_data(data = peram_u_src, mdtype = 'TScatter', root = 0, axis = 0)
    for t_sink in t_sink_list_indx:
        corr_2pt_matrix[:, t_src, t_sink] = 1.0 * cached_contract(
            'eofp,ambn,cgdh,ce,mo,Mbdf,Mnph,ga->M',
            peram_u_src[t_sink, ..., :Nev_src, :Nev_src], peram_u_src[t_sink, ..., :Nev_src, :Nev_src], peram_u_src[t_sink, ..., :Nev_src, :Nev_src],
            gamma(7), gamma(7),
            sink_VVV[t_sink], src_VVV, projection[0]
        )
        
        corr_2pt_matrix[:, t_src, t_sink] += -1.0 * cached_contract(
            'eofp,agbh,cmdn,ce,mo,Mbdf,Mnph,ga->M',
            peram_u_src[t_sink, ..., :Nev_src, :Nev_src], peram_u_src[t_sink, ..., :Nev_src, :Nev_src], peram_u_src[t_sink, ..., :Nev_src, :Nev_src],
            gamma(7), gamma(7),
            sink_VVV[t_sink], src_VVV, projection[0]
        )
        
    sink_3pt_VVV = get_mpi_data(data = sink_VVV[(t_src + t_sep)%Nt//size], mdtype = 'Bcast', root = ((t_src + t_sep)%Nt)%size)
    peram_u_src_sink = get_mpi_data(data = peram_u_src[(t_src + t_sep)%Nt//size, ..., :Nev_src, :Nev_src], mdtype = 'Bcast', root = ((t_src + t_sep)%Nt)%size)
    
    if rank == 0:
        peram_u_sep = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{(t_src+t_sep)%Nt:03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy'
            )[..., :Nev_src]

    else:
        peram_u_sep = None
        
    peram_u_sep = get_mpi_data(data = peram_u_sep, mdtype = 'TScatter', root = 0, axis = 0)
    peram_u_sep = seq_peram(peram_u_sep)
    
    for t_curr in t_sink_list_indx:
        for Mom_indx in range(len(Mom_sink)):
            corr_3pt_matrix_u[..., t_src, t_curr] = -1.0 * cached_contract(
                'eifj,ambn,cgdh,okpl,ce,Gmo,gi,Mbdf,Lnp,Mhjl,Gka->GML',
                peram_u_src_sink, peram_u_sep[t_curr], peram_u_src_sink, peram_u_src[t_curr],
                gamma(7), gamma_curr, gamma(7),
                sink_3pt_VVV[Mom_indx], curr_VdV[t_curr, 0], src_VVV[Mom_indx], projection
            )
            
            corr_3pt_matrix_u[..., t_src, t_curr] += 1.0 * cached_contract(
                'eifj,ambn,ckdl,ogph,ce,Gmo,gi,Mbdf,Lnp,Mhjl,Gka->GML',
                peram_u_src_sink, peram_u_sep[t_curr], peram_u_src_sink, peram_u_src[t_curr],
                gamma(7), gamma_curr, gamma(7),
                sink_3pt_VVV[Mom_indx], curr_VdV[t_curr, 0], src_VVV[Mom_indx], projection
            )
            corr_3pt_matrix_u[..., t_src, t_curr] += 1.0 * cached_contract(
                'eifj,agbh,cmdn,okpl,ce,Gmo,gi,Mbdf,Lnp,Mhjl,Gka->GML',
                peram_u_src_sink, peram_u_src_sink, peram_u_sep[t_curr], peram_u_src[t_curr],
                gamma(7), gamma_curr, gamma(7),
                sink_3pt_VVV[Mom_indx], curr_VdV[t_curr, 0], src_VVV[Mom_indx], projection
            )
            
            corr_3pt_matrix_u[..., t_src, t_curr] += -1.0 * cached_contract(
                'eifj,akbl,cmdn,ogph,ce,Gmo,gi,Mbdf,Lnp,Mhjl,Gka->GML',
                peram_u_src_sink, peram_u_src_sink, peram_u_sep[t_curr], peram_u_src[t_curr],
                gamma(7), gamma_curr, gamma(7),
                sink_3pt_VVV[Mom_indx], curr_VdV[t_curr, 0], src_VVV[Mom_indx], projection
            )
            
            corr_3pt_matrix_d[:, Mom_indx, ..., t_src, t_curr] = 1.0 * cached_contract(
                'emfn,oipj,agbh,ckdl,ce,Gmo,gi,bdf,Lnp,hjl,Gka->GL',
                peram_u_sep[t_curr], peram_u_src[t_curr], peram_u_src_sink, peram_u_src_sink,
                gamma(7), gamma_curr, gamma(7),
                sink_3pt_VVV[Mom_indx], curr_VdV[t_curr, 0], src_VVV[Mom_indx], projection
            )
            
            corr_3pt_matrix_d[:, Mom_indx, ..., t_src, t_curr] += -1.0 * cached_contract(
                'emfn,oipj,akbl,cgdh,ce,Gmo,gi,bdf,Lnp,hjl,Gka->GL',
                peram_u_sep[t_curr], peram_u_src[t_curr], peram_u_src_sink, peram_u_src_sink,
                gamma(7), gamma_curr, gamma(7),
                sink_3pt_VVV[Mom_indx], curr_VdV[t_curr, 0], src_VVV[Mom_indx], projection
            )

        
    if rank == 0:
        print(f'mpidata and lexico prop use time {(time.perf_counter() - st_mpi_time):.3f} s of tsrc {t_src}')
        

corr_3pt_matrix_u = get_mpi_data(data = corr_3pt_matrix_u, mdtype = 'TGather', root = 0, axis = -1)
corr_3pt_matrix_d = get_mpi_data(data = corr_3pt_matrix_d, mdtype = 'TGather', root = 0, axis = -1)
corr_2pt_matrix = get_mpi_data(data = corr_2pt_matrix, mdtype = 'TGather', root = 0, axis = -1)

if rank == 0:

    # 存入数据
    for Mom_indx, Mom_list in enumerate(Mom_sink):
        corr_save_path = f'/public/home/sush/distillation/proton/result_gen/{Nx}x{Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_src}/conf{conf_id}'
        import pathlib

        path = pathlib.Path(corr_save_path)

        if path.exists():
            print('save_path:',corr_save_path)
        
        else:
            path.mkdir(parents = True, exist_ok = True)
            print('mkdir_save_path:',corr_save_path)

        # backend.save(f'{corr_save_path}/corr_2pt_gamma0707_stout_smear_20_0.12_src{Nev_src}_sink{Nev_src}.npy', corr_2pt_matrix[Mom_indx])
            
        # for link_indx in range(-link_max, link_max + 1, 1):
        #     # backend.save(f'{corr_save_path}/corr_3pt_u_tsep{t_sep}_link_indx{link_indx}_blending_stout_smear_20_0.12_src{Nev_src}_sink{Nev_src}.npy', corr_3pt_matrix_u[:, Mom_indx, link_indx])
        #     backend.save(f'{corr_save_path}/corr_3pt_d_tsep{t_sep}_link_indx{link_indx}_blending_stout_smear_20_0.12_src{Nev_src}_sink{Nev_src}.npy', corr_3pt_matrix_d[:, Mom_indx, link_indx + link_max])
        