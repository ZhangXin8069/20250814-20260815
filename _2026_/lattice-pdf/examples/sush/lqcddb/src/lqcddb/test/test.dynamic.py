"""
E32P29 质子 2pt + 3pt — 基于 dynamic_contraction 的完整计算
=============================================================

物理参数:
- 组态: E32P29 (beta6.308, L32x64), conf 1040
- 动量: p² ≤ 1  (7 个)
- 2pt: 质子 -> 反质子
- 3pt: 矢量流 γ₄, link=-2..2, p_curr=0
- 投影: Γ = (γ₀ + γ₄) / 2
- 后端: CuPy, 2 线程
"""

import os, sys, time

from lqcddb.base import *
from lqcddb.constant import gamma, Nd, Nc, Ns
from lqcddb.eigvectors import vertex_creator
from lqcddb.contraction import seq_peram,PeramRegistry, VRegistry, GammaRegistry, dynamic_contraction
from lqcddb.analyse import loop_tsrc

set_backend('numpy')  # 生产环境改为 'cupy'
backend = get_backend()

# ═══════════════════════════════════════════════════════════════
# 参数
# ═══════════════════════════════════════════════════════════════

lattice_size = [32, 32, 32, 64]
grid_size = [1, 1, 1, 32]
Nx, Ny, Nz, Nt = lattice_size
Nev_src = 100
Nev_max = 400

conf_id = sys.argv[1]
tsep = 10
link_max = 0
link_dir = 'Z'

mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend.__name__)

rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()

# ── 动量: p² ≤ 1 ──────────────────────────────────────────
Mom_sink = creat_mom_list([0, 0, 0], fix_Q2 = True)# + creat_mom_list([0, 0, 1], fix_Q2 = True)
Mom_curr = creat_mom_list([0, 0, 0], fix_Q2 = True)
n_mom = len(Mom_sink)  # 7

# ── 算符 ─────────────────────────────────────────────────
sink_op = ['|', 'u',  'u',  'gamma_7', 'd',  '|']
src_op  = ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|']
curr_op = ['|', 'u^d', 'gamma_4', 'u', '|']

# ── 数据路径 ─────────────────────────────────────────────
GAUGE_FILE = f'/public/home/sush/share_work/configurations/CLOVER/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64_cfg_{conf_id}.lime.contents/msg02.rec04.ildg-binary-data'

def Readin_gauge(conf_file, lattice_size):
    
    backend = get_backend()
    Nz, Ny, Nx, Nt = lattice_size
    
    f = open("%s" % conf_file, "rb")
    gauge = backend.fromfile(f, dtype=">f8")
    gauge = backend.array(gauge)

    gauge = gauge.reshape(Nt, Nx, Nx, Nx, Nd, Nc, Nc, 2)
    gauge = gauge[..., 0] + gauge[..., 1] * 1j
    f.close()

    return gauge.transpose(4, 0, 1, 2, 3, 5, 6)

fun_eigen = vertex_creator(Nx = Nx)

phase_exp_sink  = backend.empty((len(Mom_sink), Nx, Nx, Nx),    dtype = complex)   #  无颜色维度
phase_exp_curr  = backend.empty((len(Mom_curr), Nx, Nx, Nx, 3), dtype = complex)   #  有颜色维度

for i in range(len(Mom_sink)):
    phase_exp_sink[i] = backend.asarray(fun_eigen.phase_exp_3pt(Mom = Mom_sink[i]))
    
for i in range(len(Mom_curr)):
    phase_exp_curr[i] = backend.asarray(fun_eigen.phase_exp_2pt(Mom = Mom_curr[i]))
    
t_rank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'Scatter')

gauge_link = backend.ones((Nd, Nx, Nx, Nx, Nc, Nc), dtype = complex)

sink_VVV = backend.zeros((Nt//size, len(Mom_sink), Nev_src, Nev_src, Nev_src), dtype = complex)
curr_VdV = backend.zeros((Nt//size, len(Mom_curr), 2 * link_max + 1, Nev_max, Nev_max), dtype = complex)
st_eigen = time.perf_counter()

for tsrc_indx, tsrc in enumerate(t_rank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')
    sink_VVV[tsrc_indx] = fun_eigen.Mom_VVV_sink_t(phase_exp = phase_exp_sink, eigvecs = eigvecs[:Nev_src])
    curr_VdV[tsrc_indx] = fun_eigen.VdV_sink_t_link(eigvecs = eigvecs, link_dir = 'Z', link_max = link_max, phase_exp = phase_exp_curr, gauge_link = gauge_link)
    
if rank == 0:
    print(f'load eigen and cal VVV use time {(time.perf_counter() - st_eigen):.3f} s')
    # curr_VdV[tsrc_indx] = backend.random.random((3, 21, 650, 650)) + 1j*backend.random.random((3, 21, 650, 650))
       
curr_VdV = curr_VdV * fun_eigen.create_omega_accelerate(
    exact = 100,
    N_eigen = [100, 200, 400, 700, 1100],
    N_sum = [20, 20, 20, 20, 20],
    N_extract = [10, 10, 10, 10, 10],
    noise = 200
)

corr_2pt_uud_matrix = backend.zeros((Nt, Nt//size, Ns, Ns, len(Mom_sink)), dtype = complex)
corr_2pt_ud_matrix = backend.zeros((Nt, Nt//size, len(Mom_sink)), dtype = complex)

corr_2pt_uud_matrix_2 = backend.zeros((Nt, Nt//size, len(Mom_sink)), dtype = complex)
corr_2pt_ud_matrix_2 = backend.zeros((Nt, Nt//size, len(Mom_sink)), dtype = complex)

corr_3pt_matrix_d = backend.zeros((Nt, Nt//size, Ns, Ns, len(Mom_sink), 2 * link_max + 1), dtype = complex)
corr_3pt_matrix_u = backend.zeros((Nt, Nt//size, Ns, Ns, len(Mom_sink), 2 * link_max + 1), dtype = complex)

corr_3pt_matrix_u_2 = backend.zeros((Nt, Nt//size, len(Mom_sink), 2 * link_max + 1), dtype = complex)

Gamma = GammaRegistry()
Gamma.register('gamma_5', gamma(5))
Gamma.register('gamma_7', gamma(7))
Gamma.register('gamma_4', gamma(4))

Peram = PeramRegistry()
Ver = VRegistry()

for tsrc in range(32):
    st_mpi_time = time.perf_counter()
    src_VVV_t = get_mpi_data(data = sink_VVV[tsrc//size], mdtype = 'Bcast', root = tsrc%size).conj()
    _, _, tsink_list_indx = get_mpi_tlist(Nt = Nt, t = range(tsrc, tsrc + 12, 1), gtype = 'TScatter')

    if rank == 0:
        peram_u_src = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy'
            )[..., :Nev_src]

    else:
        peram_u_src = None

    peram_u_src = get_mpi_data(data = peram_u_src, mdtype = 'TScatter', root = 0, axis = 0)
    
    Ver.register('VVV_0', 'tsrc', src_VVV_t)
    
    for tsink in tsink_list_indx:
        Peram.register('light', ('tsink', 'tsrc'), peram_u_src[tsink, :, :, :Nev_src, :Nev_src])
        Peram.register('light', ('tsrc', 'tsink'), seq_peram(peram_u_src[tsink, :, :, :Nev_src, :Nev_src]))
        
        Ver.register('VVV_0', 'tsink', sink_VVV[tsink])
        Ver.register('VDV_0', 'tsink', backend.identity(Nev_src, dtype = complex).reshape(1, Nev_src, Nev_src))
        Ver.register('VDV_0', 'tsrc', backend.identity(Nev_src, dtype = complex).reshape(1, Nev_src, Nev_src))
        
        dycn_uud = dynamic_contraction(
            operator_groups = [(['|', 'u', 'u', 'gamma_7', 'd', '|'], ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|'])], 
            peram_registry = Peram, v_registry = Ver, gamma_registry = Gamma, 
            Cpt = '2pt', 
            Vindex = ['M', 'M'], use_equivalence = True, verbose = True, ignore_dis = True
            )
        
        corr_2pt_uud_matrix[tsrc, tsink] = dycn_uud.calculate_all(optimize = ['dp', 'greedy', 'auto'])
        
        corr_2pt_uud_matrix_2[tsrc, tsink] = 1.0 * cached_contract(
            'eofp,ambn,cgdh,ce,mo,Mbdf,Mnph,ga->M',
            peram_u_src[tsink, ..., :Nev_src, :Nev_src], peram_u_src[tsink, ..., :Nev_src, :Nev_src], peram_u_src[tsink, ..., :Nev_src, :Nev_src],
            gamma(7), gamma(7),
            sink_VVV[tsink], src_VVV_t, (gamma(0) + gamma(4))/2
        )
        
        corr_2pt_uud_matrix_2[tsrc, tsink] += -1.0 * cached_contract(
            'eofp,agbh,cmdn,ce,mo,Mbdf,Mnph,ga->M',
            peram_u_src[tsink, ..., :Nev_src, :Nev_src], peram_u_src[tsink, ..., :Nev_src, :Nev_src], peram_u_src[tsink, ..., :Nev_src, :Nev_src],
            gamma(7), gamma(7),
            sink_VVV[tsink], src_VVV_t, (gamma(0) + gamma(4))/2
        )
        
        dycn_ud = dynamic_contraction(
            operator_groups = [(['|', 'u^d', 'gamma_5', 'd', '|'], ['|', 'd^d', 'gamma_5', 'u', '|'])], 
            peram_registry = Peram, v_registry = Ver, gamma_registry = Gamma, 
            Cpt = '2pt', 
            Vindex = ['M', 'M'], use_equivalence = True, verbose = True, ignore_dis = True
            )
        
        corr_2pt_ud_matrix[tsrc, tsink] = dycn_ud.calculate_all(optimize = ['dp', 'greedy', 'auto'])
        corr_2pt_ud_matrix_2[tsrc, tsink] = cached_contract(
            'manb,cedf,Mbd,Mfn,ac,em->M',
            peram_u_src[tsink, :, :, :Nev_src, :Nev_src],
            seq_peram(peram_u_src[tsink, :, :, :Nev_src, :Nev_src]),
            backend.identity(Nev_src, dtype = complex).reshape(1, Nev_src, Nev_src),
            backend.identity(Nev_src, dtype = complex).reshape(1, Nev_src, Nev_src),  
            backend.asarray(gamma(5)),
            backend.asarray(gamma(5)),
        )
        
    if rank == 0:
        peram_u_sep = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{(tsrc + tsep)%Nt:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy'
            )[..., :Nev_src]

    else:
        peram_u_sep = None
    
    peram_u_sep = get_mpi_data(data = peram_u_sep, mdtype = 'TScatter', root = 0, axis = 0)
    peram_u_sep = seq_peram(peram_u_sep)
    
    peram_u_src_sink = get_mpi_data(data = peram_u_src[(tsrc + tsep)%Nt//size, ..., :Nev_src, :Nev_src], mdtype = 'Bcast', root = ((tsrc + tsep)%Nt)%size)
    
    sink_3pt_VVV = get_mpi_data(data = sink_VVV[(tsrc + tsep)%Nt//size], mdtype = 'Bcast', root = ((tsrc + tsep)%Nt)%size)
    Ver.register('VVV_0', 'tsink', sink_3pt_VVV)
    
    for tcur in tsink_list_indx:
        Peram.register('light', ('tsink', 'tcur0'), peram_u_sep[tcur, :, :, :Nev_src, :])
        Peram.register('light', ('tcur0', 'tsrc'), peram_u_src[tcur, :, :, :, :Nev_src])
        Peram.register('light', ('tsink', 'tsrc'), peram_u_src_sink)
        
        Ver.register('VDV_0', 'tcur0', curr_VdV[tcur, 0])
        
        dycn_uud_3pt = dynamic_contraction(
            operator_groups = [(['|', 'u', 'u', 'gamma_7', 'd', '|'], ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|'], ['|', 'u^d', 'gamma_4', 'u', '|'])], 
            peram_registry = Peram, v_registry = Ver, gamma_registry = Gamma, 
            Cpt = '3pt', 
            Vindex = ['M', 'L', 'M'], use_equivalence = True, verbose = True, ignore_dis = True
            )

        corr_3pt_matrix_u[tsrc, tcur] = dycn_uud_3pt.calculate_all(optimize = ['dp', 'greedy', 'auto'])
        
        corr_3pt_matrix_u_2[tsrc, tcur] = -1.0 * cached_contract(
            'eifj,ambn,cgdh,okpl,ce,mo,gi,Mbdf,Lnp,Mhjl,ka->ML',
            peram_u_src_sink, peram_u_sep[tcur], peram_u_src_sink, peram_u_src[tcur],
            gamma(7), gamma(4), gamma(7),
            sink_3pt_VVV[:], curr_VdV[tcur, 0], src_VVV_t[:], (gamma(0) + gamma(4))/2, optimize = ['dp', 'greedy']
        )
        corr_3pt_matrix_u_2[tsrc, tcur] += 1.0 * cached_contract(
            'eifj,ambn,ckdl,ogph,ce,mo,gi,Mbdf,Lnp,Mhjl,ka->ML',
            peram_u_src_sink, peram_u_sep[tcur], peram_u_src_sink, peram_u_src[tcur],
            gamma(7), gamma(4), gamma(7),
            sink_3pt_VVV[:], curr_VdV[tcur, 0], src_VVV_t[:], (gamma(0) + gamma(4))/2, optimize = ['dp', 'greedy']
        )
        
        corr_3pt_matrix_u_2[tsrc, tcur] += 1.0 * cached_contract(
            'eifj,agbh,cmdn,okpl,ce,mo,gi,Mbdf,Lnp,Mhjl,ka->ML',
            peram_u_src_sink, peram_u_src_sink, peram_u_sep[tcur], peram_u_src[tcur],
            gamma(7), gamma(4), gamma(7),
            sink_3pt_VVV[:], curr_VdV[tcur, 0], src_VVV_t[:], (gamma(0) + gamma(4))/2, optimize = ['dp', 'greedy']
        )
        
        corr_3pt_matrix_u_2[tsrc, tcur] += -1.0 * cached_contract(
            'eifj,akbl,cmdn,ogph,ce,mo,gi,Mbdf,Lnp,Mhjl,ka->ML',
            peram_u_src_sink, peram_u_src_sink, peram_u_sep[tcur], peram_u_src[tcur],
            gamma(7), gamma(4), gamma(7),
            sink_3pt_VVV[:], curr_VdV[tcur, 0], src_VVV_t[:], (gamma(0) + gamma(4))/2, optimize = ['dp', 'greedy']
        )
        
    if rank == tsrc%size:
        print(f'mpidata peram_u_src use time {(time.perf_counter() - st_mpi_time):.4f} s tsrc {tsrc}')
        
corr_2pt_uud_matrix = get_mpi_data(corr_2pt_uud_matrix, mdtype = 'TGather', root = 0, axis = 1)
corr_2pt_ud_matrix = get_mpi_data(corr_2pt_ud_matrix, mdtype = 'TGather', root = 0, axis = 1)
corr_2pt_uud_matrix_2 = get_mpi_data(corr_2pt_uud_matrix_2, mdtype = 'TGather', root = 0, axis = 1)
corr_2pt_ud_matrix_2 = get_mpi_data(corr_2pt_ud_matrix_2, mdtype = 'TGather', root = 0, axis = 1)
corr_3pt_matrix_u = get_mpi_data(corr_3pt_matrix_u, mdtype = 'TGather', root = 0, axis = 1)
corr_3pt_matrix_u_2 = get_mpi_data(corr_3pt_matrix_u_2, mdtype = 'TGather', root = 0, axis = 1)
