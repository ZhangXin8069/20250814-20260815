"""
利用 dynamic_contraction 计算质子两点和矢量流三点关联函数
=============================================================
轻量级 MPI 版本: 2 线程, numpy CPU 后端, 小 Nev, p=0 单动量, 无 gauge link

参考:
- contraction.NJNp-.3pt.GEVP.cupy.na800.py (dynamic_contraction + registry 模式)
- contraction.proton.mpi.py             (质子算符 + peram/MPI 处理)
"""

import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from lqcddb.base import *
from lqcddb.contraction import seq_peram, GammaRegistry, VRegistry, PeramRegistry
from lqcddb.contraction.dynamic import dynamic_contraction
from lqcddb.constant import gamma, Nc, Ns
from lqcddb.analyse import loop_tsrc
from lqcddb.eigvectors import vertex_creator


import time

# ============================================================
# 1. 初始化
# ============================================================
conf_id = sys.argv[1]

set_backend('numpy')
backend = get_backend()

lattice_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 2]        # 只用 2 线程
mpinit(grid_size=grid_size, latt_size=lattice_size, backend='numpy')

rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()

Nx, Ny, Nz, Nt = lattice_size
Lt = Nt // grid_size[-1]

# ---- 轻量级参数 ----
Nev_src  = 30
Nev_max  = 650
tsep     = 6
Mom_len  = 1                     # 仅 p=0

# ============================================================
# 2. 预计算 V 结构 (VVV + VdV)
# ============================================================
fun_eigen = vertex_creator(Nx=Nx)

# 仅零动量
Mom_sink_VVV = [[0, 0, 0]]

phase_exp_VDV = backend.zeros((1, Nx, Nx, Nx, Nc), dtype=complex)
phase_exp_VVV = backend.zeros((1, Nx, Nx, Nx), dtype=complex)

phase_exp_VVV[0] = backend.asarray(fun_eigen.phase_exp_3pt(Mom=[0, 0, 0]))
phase_exp_VDV[0] = backend.asarray(fun_eigen.phase_exp_2pt(Mom=[0, 0, 0]))

trank, _, _ = get_mpi_tlist(Nt=Nt, t=range(Nt), gtype='TScatter')

sink_VVV = backend.zeros((Lt, Mom_len, Nev_src, Nev_src, Nev_src), dtype=complex)
VdV_curr  = backend.zeros((Lt, Mom_len, Nev_max, Nev_max), dtype=complex)

st_eigen = time.perf_counter()
# for tsrc_indx, tsrc in enumerate(trank):
#     eigvecs = backend.load(
#         f'/nexdata/project/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/'
#         f'{conf_id}/{conf_id}_t{tsrc:03d}_e50_s{[100,200,300,500,800]}_'
#         f'{[20,20,20,20,20]}_{[10,10,10,10,10]}_BI_n500_V1_stout_smear_20_0.12.npy'
#     )
#     # 汇端重子 VVV: 只用 Nev_src
#     sink_VVV[tsrc_indx, 0:1] = fun_eigen.Mom_VVV_sink_t(
#         phase_exp=phase_exp_VVV[0], eigvecs=eigvecs[:Nev_src]
#     )
#     # 流插入 VdV: 使用更大 Nev (用于流端)
#     VdV_curr[tsrc_indx, 0:1] = fun_eigen.Mom_VdV_sink_t(
#         phase_exp=phase_exp_VDV[0], eigvecs=eigvecs
#     )[0]

# Omega 加速权重
VdV_curr *= fun_eigen.create_omega_accelerate(
    exact=50,
    N_eigen=[100, 200, 300, 500, 800],
    N_sum=[20, 20, 20, 20, 20],
    N_extract=[10, 10, 10, 10, 10],
    noise=500,
)

del phase_exp_VDV

if rank == 0:
    print(f'[rank {rank}] load eigen + VVV/VdV: {(time.perf_counter() - st_eigen):.3f} s')

# ============================================================
# 3. 算符定义
# ============================================================

# 质子 sink 算符: u (Cγ₅) d → u
sink_op = ['|', 'u', 'u', 'gamma_7', 'd', '|']

# 质子 source 算符 (共轭): u^d (Cγ₅) d^d → u^d
src_op  = ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|']

# 矢量流 (γ₄ 时间分量)
curr_op_u = ['|', 'u^d', 'gamma_4', 'u', '|']   # u-quark current
curr_op_d = ['|', 'd^d', 'gamma_4', 'd', '|']   # d-quark current

# ============================================================
# 4. 注册表 (时间无关部分)
# ============================================================
GR = GammaRegistry()
GR.register('gamma_7', backend.asarray(gamma(7)))
GR.register('gamma_4', backend.asarray(gamma(4)))

# 重子投影算符: P₊ = (γ₀ + γ₄) / 2
# dynamic_contraction 要求 Projector 为 [proj_sink, proj_src] 两个算符的序列
# 每个投影算符形状为 (4, Ns//2=2)，将 4 分量 Dirac 旋量投影到 2 分量
proj = backend.asarray((gamma(0) + gamma(4)) / 2.0)
GR.register('Projector', [proj, proj])

VR = VRegistry()
PR = PeramRegistry()

# ============================================================
# 5. 关联函数数组分配
# ============================================================
# 2pt: (Ns, Ns, Mom_sink, Mom_src, Nt, Lt)
corr_2pt = backend.zeros((Ns, Ns, Mom_len, Mom_len, Nt, Lt), dtype=complex)

# 3pt: (Ns, Ns, Proj_G, Gamma_G, Mom_sink, VDV_mom, Mom_src, Nt, Lt)
corr_3pt_u = backend.zeros(
    (Ns, Ns, Mom_len, Mom_len, Mom_len, Nt, Lt), dtype=complex
)
corr_3pt_d = backend.zeros_like(corr_3pt_u)

# ============================================================
# 6. 主循环
# ============================================================
for t_src in range(Nt):
    st_mpi = time.perf_counter()

    # --- 6a. 源端 V 结构 (Bcast) ---
    VVV_src = backend.asarray(get_mpi_data(
        data=sink_VVV[t_src // size, :], mdtype='Bcast', root=t_src % size
    )).conj()

    # --- 6b. 加载 peram at t_src (TScatter) ---
    if rank == 0:
        peram_u_src = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/'
            f'beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/'
            f't{t_src:03d}_e50_s{[100,200,300,500,800]}_'
            f'{[20,20,20,20,20]}_{[10,10,10,10,10]}_BI_n500_V1_stout_smear_20_0.12.npy'
        )[..., :Nev_max, :Nev_src]          # (Nt, 4, 4, Nev_max, Nev_src)
    else:
        peram_u_src = None
    peram_u_src = get_mpi_data(data=peram_u_src, mdtype='TScatter', root=0, axis=0)
    # 本地形状: (Lt, 4, 4, Nev_max, Nev_src)

    # --- 6c. t_src 等时 peram (Bcast from owner rank) ---
    # 从 peram_u_src 中取 t_src 等时: peram[t_src → t_src]
    # 在 owner rank 上 local index = t_src // size 访问绝对时间 t_src
    peram_src_src = backend.asarray(get_mpi_data(
        data=peram_u_src[t_src // size, :, :, :Nev_src, :Nev_src],
        mdtype='Bcast', root=t_src % size
    ))

    # 注册源端 perams + V
    PR.register('light', ('tsrc', 'tsrc'), peram_src_src)
    VR.register('VVV_0', 'tsrc', VVV_src)

    # --- 6d. 获取 t_sink 本地索引 ---
    _, _, t_sink_list_indx = get_mpi_tlist(
        Nt=Nt, t=[x for x in range(t_src, t_src + tsep + 1)], gtype='TScatter'
    )

    # --- 6e. 2pt 循环 ---
    for t_sink in t_sink_list_indx:
        # 汇端 VVV (本地索引: sink_VVV 预计算时按 Scatter 分布)
        VVV_sink = sink_VVV[t_sink, :]

        # 汇端 perams (本地索引: peram_u_src TScattered 后本地访问)
        # forward: t_src → t_sink (peram 文件 at t_src, 绝对 sink 时间 = rank + t_sink*size)
        peram_sink_src = backend.asarray(
            peram_u_src[t_sink, :, :, :Nev_src, :Nev_src]
        )
        # # backward: seq_peram(forward)
        # peram_src_sink = seq_peram(peram_sink_src)

        # 注册汇端 perams + V (注: 质子 2pt 不需要等时 peram)
        PR.register('light', ('tsink', 'tsrc'), peram_sink_src)
        # PR.register('light', ('tsrc', 'tsink'), peram_src_sink)
        VR.register('VVV_0', 'tsink', VVV_sink)

        # --- 2pt dynamic_contraction ---
        dc_2pt = dynamic_contraction(
            [[sink_op, src_op]],
            peram_registry=PR,
            v_registry=VR,
            gamma_registry=GR,
            Cpt='2pt',
            Vindex=['M', 'N'],
            use_equivalence=True,
            ignore_dis=False,
            Projection=False,
            optimize=['greedy', 'dp', 'auto'],
        )

        corr_2pt[..., t_src, t_sink] = dc_2pt.calculate_all()

    # --- 6f. 3pt 准备: t_sink = t_src + tsep ---
    t_sink_global = (t_src + tsep) % Nt

    # 汇端 VVV (Bcast)
    VVV_sink_3pt = backend.asarray(get_mpi_data(
        data=sink_VVV[t_sink_global // size, :],
        mdtype='Bcast', root=t_sink_global % size
    ))

    # peram from t_src to t_sink (Bcast)
    peram_src_sink_3pt = backend.asarray(get_mpi_data(
        data=peram_u_src[t_sink_global // size, :, :, :Nev_src, :Nev_src],
        mdtype='Bcast', root=t_sink_global % size
    ))
    
    # 加载 sequential peram at t_sink (TScatter)
    if rank == 0:
        peram_u_sep = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/'
            f'beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/'
            f't{t_sink_global:03d}_e50_s{[100,200,300,500,800]}_'
            f'{[20,20,20,20,20]}_{[10,10,10,10,10]}_BI_n500_V1_stout_smear_20_0.12.npy'
        )[..., :Nev_max, :Nev_src]
    else:
        peram_u_sep = None
    peram_u_sep = get_mpi_data(data=peram_u_sep, mdtype='TScatter', root=0, axis=0)
    peram_u_sep_seq = seq_peram(peram_u_sep)
    # seq_peram 交换 ev 指标: (Nev_max, Nev_src) → (Nev_src, Nev_max)

    # 注册 3pt 汇端 perams + V
    PR.register('light', ('tsink', 'tsrc'), peram_src_sink_3pt)
    # PR.register('light', ('tsrc', 'tsink'), seq_peram(peram_src_sink_3pt))
    VR.register('VVV_0', 'tsink', VVV_sink_3pt)

    # --- 6g. 3pt t_curr 循环 ---
    for t_curr in t_sink_list_indx:
        # 流 VDV (本地索引, 预计算时已按 Scatter 分布)
        VDV_curr = VdV_curr[t_curr, 0:1]          # (1, Nev_max, Nev_max)

        # peram: t_src → t_curr (本地, forward)
        peram_curr_src = backend.asarray(
            peram_u_src[t_curr, :, :, :Nev_max, :Nev_src]
        )
        # # peram: t_curr → t_src (backward, seq)
        # peram_src_curr = seq_peram(peram_curr_src)

        # peram: t_sink → t_curr (本地, from sequential peram)
        # peram_curr_sink = backend.asarray(
        #     peram_u_sep[t_curr, :, :, :Nev_max, :Nev_src]
        # )
        # peram: t_curr → t_sink (backward through sink, seq)
        peram_sink_curr = backend.asarray(
            peram_u_sep_seq[t_curr, :, :, :Nev_src, :Nev_max]
        )

        # 注册流端 perams + V
        PR.register('light', ('tcur0', 'tsrc'),  peram_curr_src)
        # PR.register('light', ('tsrc', 'tcur0'),  peram_src_curr)
        # PR.register('light', ('tcur0', 'tsink'), peram_curr_sink)
        PR.register('light', ('tsink', 'tcur0'), peram_sink_curr)
        VR.register('VDV_0', 'tcur0', VDV_curr)

        # --- u-quark 3pt ---
        dc_3pt_u = dynamic_contraction(
            [[sink_op, src_op, curr_op_u]],
            peram_registry=PR,
            v_registry=VR,
            gamma_registry=GR,
            Cpt='3pt',
            Vindex=['M', 'L', 'N'],
            use_equivalence=True,
            ignore_dis=True,
            Projection=False,
            optimize=['greedy', 'dp', 'auto'],
        )

        corr_3pt_u[..., t_src, t_curr] = dc_3pt_u.calculate_all()
        
        # --- d-quark 3pt ---
        dc_3pt_d = dynamic_contraction(
            [[sink_op, src_op, curr_op_d]],
            peram_registry=PR,
            v_registry=VR,
            gamma_registry=GR,
            Cpt='3pt',
            Vindex=['M', 'L', 'N'],
            use_equivalence=True,
            ignore_dis=True,
            Projection=False,
            optimize=['greedy', 'dp', 'auto'],
        )

        corr_3pt_d[..., t_src, t_curr] = dc_3pt_d.calculate_all()

    if rank == 0:
        print(f'[rank {rank}], t_src={t_src:03d}, t_curr={t_curr:03d}: '
            f'mpi+contraction {(time.perf_counter() - st_mpi):.3f} s')
        
# ============================================================
# 7. MPI Gather + 保存
# ============================================================
# TGather on the local-time axis (axis=-1 = Lt)
corr_2pt   = get_mpi_data(data=corr_2pt,   mdtype='TGather', root=0, axis=-1)
corr_3pt_u = get_mpi_data(data=corr_3pt_u, mdtype='TGather', root=0, axis=-1)
corr_3pt_d = get_mpi_data(data=corr_3pt_d, mdtype='TGather', root=0, axis=-1)

# if rank == 0:
#     # 时间源平均 (loop_tsrc)
#     corr_2pt_avg = loop_tsrc(
#         corr_2pt, indx=[-2, -1],
#         Boundary_Conditions='Antiperiodic', Ctype='2pt'
#     )

#     corr_3pt_u_avg = loop_tsrc(
#         corr_3pt_u, indx=[-2, -1],
#         Boundary_Conditions='Antiperiodic', Ctype='3pt', t_sep=tsep
#     )

#     corr_3pt_d_avg = loop_tsrc(
#         corr_3pt_d, indx=[-2, -1],
#         Boundary_Conditions='Antiperiodic', Ctype='3pt', t_sep=tsep
#     )

#     print(f'\n2pt  shape (after TGather):  {corr_2pt.shape}')
#     print(f'2pt  shape (after loop_tsrc): {corr_2pt_avg.shape}')
#     print(f'3pt_u shape (after TGather):  {corr_3pt_u.shape}')
#     print(f'3pt_u shape (after loop_tsrc): {corr_3pt_u_avg.shape}')

#     # 保存
#     corr_save_path = (
#         f'/public/home/sush/distillation/lqcddb/src/lqcddb/test/result/'
#         f'Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}'
#     )
#     import pathlib
#     path = pathlib.Path(corr_save_path)
#     if not path.exists():
#         path.mkdir(parents=True, exist_ok=True)
#         print(f'mkdir: {corr_save_path}')
#     else:
#         print(f'save_path exists: {corr_save_path}')

#     backend.save(f'{corr_save_path}/corr_2pt_proton_dynamic_src{Nev_src}.npy', corr_2pt_avg)
#     backend.save(f'{corr_save_path}/corr_3pt_u_proton_dynamic_src{Nev_src}.npy', corr_3pt_u_avg)
#     backend.save(f'{corr_save_path}/corr_3pt_d_proton_dynamic_src{Nev_src}.npy', corr_3pt_d_avg)

#     print(f'\nResults saved to {corr_save_path}/')
#     print('Done.')
