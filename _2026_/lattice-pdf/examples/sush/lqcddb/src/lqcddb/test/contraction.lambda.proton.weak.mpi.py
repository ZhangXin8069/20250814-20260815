"""
Λ → p 弱流三点关联函数 + 两点关联函数  MPI 程序
系综: C24P29 (24³×72, a=0.1053 fm)
Λ 算子: 仅第一项 ['s', 'u', 'C*gamma_5', 'd'] (系数 +2)

用法: python contraction.lambda.proton.weak.mpi.py <conf_id>
"""

import os
import sys
import time

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from lqcddb.base import *
from lqcddb.eigvectors import vertex_creator
from lqcddb.constant import gamma, Nc, Nd
from lqcddb.contraction import seq_peram


def Readin_gauge(conf_file, lattice_size):
    backend = get_backend()
    Nz, Ny, Nx, Nt = lattice_size
    _ = Nz, Ny  # used only for lattice shape

    f = open("%s" % conf_file, "rb")
    gauge = backend.fromfile(f, dtype=">f8")
    gauge = backend.array(gauge)

    gauge = gauge.reshape(Nt, Nx, Nx, Nx, Nd, Nc, Nc, 2)
    gauge = gauge[..., 0] + gauge[..., 1] * 1j
    f.close()

    return gauge


# ============================================================
# 1. 参数配置
# ============================================================
conf_id = sys.argv[1]

set_backend('numpy')
backend = get_backend()

lattice_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 24]
mpinit(grid_size=grid_size, latt_size=lattice_size, backend=backend.__name__)

rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()

Nx, Ny, Nz, Nt = lattice_size

t_sep = 10
link_max = 10
Nev_src = 150
Nev_link = 650

# ============================================================
# 2. 读取规范场
# ============================================================
if rank == 0:
    gauge_link = Readin_gauge(
        f'/public/home/sush/share_work/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{conf_id}.lime.contents/msg02.rec04.ildg-binary-data',
        lattice_size=lattice_size
    )
    gauge_link = gauge_link.transpose(4, 0, 1, 2, 3, 5, 6)  # -> (Nd, Nt, Nz, Ny, Nx, Nc, Nc)
else:
    gauge_link = None

gauge_link = get_mpi_data(data=gauge_link, mdtype='TScatter', root=0, axis=1)

# ============================================================
# 3. 动量 & 相位因子 (p=0  only)
# ============================================================
fun_eigen = vertex_creator(Nx=Nx, backend=backend)

Mom_sink = [[0, 0, 0]]
phase_exp_3pt = backend.empty((len(Mom_sink), Nx, Nx, Nx), dtype=complex)
phase_exp_2pt = backend.empty((1, Nx, Nx, Nx, Nc), dtype=complex)

phase_exp_3pt[0] = backend.asarray(fun_eigen.phase_exp_3pt(Mom=[0, 0, 0]))
phase_exp_2pt[0] = backend.asarray(fun_eigen.phase_exp_2pt(Mom=[0, 0, 0]))

# ============================================================
# 4. 预计算 VVV (baryon sink/source) 和 VdV (current)
# ============================================================
t_rank, _, _ = get_mpi_tlist(Nt=Nt, t=range(Nt), gtype='TScatter')

sink_VVV = backend.zeros((Nt // size, Nev_src, Nev_src, Nev_src), dtype=complex)
curr_VdV = backend.zeros((Nt // size, 2 * link_max + 1, Nev_link, Nev_link), dtype=complex)

st_eigen = time.perf_counter()
for t_src_indx, t_src in enumerate(t_rank):
    eigvecs = backend.load(
        f'/nexdata/project/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/{conf_id}_t{t_src:03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy'
    )

    sink_VVV[t_src_indx] = fun_eigen.Mom_VVV_sink_t_3(
        phase_exp=phase_exp_3pt[0], eigvecs=eigvecs[:Nev_src]
    )

    curr_VdV[t_src_indx] = fun_eigen.VdV_sink_t_link(
        eigvecs=eigvecs,
        link_dir='Z',
        link_max=link_max,
        phase_exp=phase_exp_2pt,
        gauge_link=gauge_link,
        t=t_src_indx
    )

# Omega 加速
curr_VdV = curr_VdV * fun_eigen.create_omega_accelerate(
    exact=50,
    N_eigen=[100, 200, 300, 500, 800],
    N_sum=[20, 20, 20, 20, 20],
    N_extract=[10, 10, 10, 10, 10],
    noise=500
)

if rank == 0:
    print(f'Load eigen and compute VVV VdV: {(time.perf_counter() - st_eigen):.3f} s')

# ============================================================
# 5. Gamma 矩阵
# ============================================================
# Baryon diquark structure: gamma_7 (= C*γ₅ in DR basis)
gamma_baryon = backend.asarray(gamma(7))  # shape (4, 4)

# Weak current: γ_w(μ) = γ_μ + γ_μ γ₅,  μ=1,2,3,4
# gamma(μ+11) = γ_μ γ₅,  so γ_w = gamma(μ) + gamma(μ+11)
gamma_curr = backend.zeros((4, 4, 4), dtype=complex)
for mu in range(1, 5):
    gamma_curr[mu - 1] = gamma(mu) + gamma(mu + 11)

# Baryon spin projector: (γ₀ + γ₄) / 2
proj = (gamma(0) + gamma(4)) / 2
# For 2pt: shape (4, 4)
projection_2pt = backend.asarray(proj)
# For 3pt: expand to (4, 4, 4) to match gamma_curr G index
projection_3pt = backend.array([proj] * 4)

# ============================================================
# 6. 输出数组
# ============================================================
# 2pt: scalar in spin space, scalar in momentum (p=0)
corr_2pt_lambda = backend.zeros((Nt, Nt // size), dtype=complex)
corr_2pt_proton = backend.zeros((Nt, Nt // size), dtype=complex)

# 3pt: (G=4, L=2*link_max+1) per (t_src, t_curr)
corr_3pt_weak = backend.zeros((4, 2 * link_max + 1, Nt, Nt // size), dtype=complex)

# ============================================================
# 7. 主循环
# ============================================================
for t_src in range(Nt):
    st_mpi_time = time.perf_counter()

    # --- 7a. Bcast source VVV (both Λ and P source at same t_src, p=0) ---
    src_VVV = get_mpi_data(
        data=sink_VVV[t_src // size], mdtype='Bcast', root=t_src % size
    ).conj()  # source = sink†, shape (Nev, Nev, Nev)

    # t_sink loop range: [0, t_sep]
    _, _, t_sink_list_indx = get_mpi_tlist(
        Nt=Nt, t=range(t_src, t_src + t_sep + 1, 1), gtype='TScatter'
    )

    # --- 7b. 加载 perambulators at t_src ---
    if rank == 0:
        peram_light_src = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{t_src:03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy'
        )[..., :Nev_src]

        peram_strange_src = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/strange/{conf_id}/t{t_src:03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy'
        )[..., :Nev_src]
    else:
        peram_light_src = None
        peram_strange_src = None

    peram_light_src = get_mpi_data(data=peram_light_src, mdtype='TScatter', root=0, axis=0)
    peram_strange_src = get_mpi_data(data=peram_strange_src, mdtype='TScatter', root=0, axis=0)

    # --- 7c. Λ 2pt (1 diagram, sign=+2) ---
    # Wick: eofp,agbh,cmdn,ce,mo,bdf,nph->ag
    # peram_d[eofp], peram_s[agbh], peram_u[cmdn], gamma_7[ce], C*gamma_5[mo], VVV_0[bdf], VVV_1[nph], proj[ga]
    for t_sink in t_sink_list_indx:
        corr_2pt_lambda[t_src, t_sink] = 2.0 * cached_contract(
            'eofp,agbh,cmdn,ce,mo,bdf,nph,ga->',
            peram_light_src[t_sink, :, :, :Nev_src, :Nev_src],   # peram_d: d d-bar
            peram_strange_src[t_sink, :, :, :Nev_src, :Nev_src],  # peram_s: s s-bar
            peram_light_src[t_sink, :, :, :Nev_src, :Nev_src],   # peram_u: u u-bar
            gamma_baryon,                                          # gamma_7 at tsink
            gamma_baryon,                                          # C*gamma_5 at tsrc
            sink_VVV[t_sink],                                      # VVV at tsink
            src_VVV,                                               # VVV at tsrc (.conj() already)
            projection_2pt,                                        # projector
        )

    # --- 7d. P 2pt (2 diagrams) ---
    # Diagram 0: eofp,ambn,cgdh,ce,mo,bdf,nph->ag  sign=-1
    # Diagram 1: eofp,agbh,cmdn,ce,mo,bdf,nph->ag  sign=+1
    for t_sink in t_sink_list_indx:
        corr_2pt_proton[t_src, t_sink] = -1.0 * cached_contract(
            'eofp,ambn,cgdh,ce,mo,bdf,nph,ga->',
            peram_light_src[t_sink, :, :, :Nev_src, :Nev_src],
            peram_light_src[t_sink, :, :, :Nev_src, :Nev_src],
            peram_light_src[t_sink, :, :, :Nev_src, :Nev_src],
            gamma_baryon,
            gamma_baryon,
            sink_VVV[t_sink],
            src_VVV,
            projection_2pt,
        )

        corr_2pt_proton[t_src, t_sink] += 1.0 * cached_contract(
            'eofp,agbh,cmdn,ce,mo,bdf,nph,ga->',
            peram_light_src[t_sink, :, :, :Nev_src, :Nev_src],
            peram_light_src[t_sink, :, :, :Nev_src, :Nev_src],
            peram_light_src[t_sink, :, :, :Nev_src, :Nev_src],
            gamma_baryon,
            gamma_baryon,
            sink_VVV[t_sink],
            src_VVV,
            projection_2pt,
        )

    # --- 7e. Λ→P 3pt ---
    # 需要 peram at t_sink = t_src + t_sep (Bcast from owning rank)
    t_sink_global = (t_src + t_sep) % Nt
    t_sink_root = t_sink_global % size

    sink_3pt_VVV = get_mpi_data(
        data=sink_VVV[t_sink_global // size], mdtype='Bcast', root=t_sink_root
    )

    # Perams from t_src → t_sink: Bcast from the rank that owns t_sink_global
    peram_light_src_sink = get_mpi_data(
        data=peram_light_src[t_sink_global // size, :, :, :Nev_src, :Nev_src],
        mdtype='Bcast', root=t_sink_root
    )

    # Load sequential perams at t_sink_global (for t_curr → t_sink via seq_peram)
    if rank == 0:
        peram_light_sep = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{t_sink_global:03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy'
        )[..., :Nev_src]
    else:
        peram_light_sep = None

    peram_light_sep = get_mpi_data(data=peram_light_sep, mdtype='TScatter', root=0, axis=0)
    peram_light_sep = seq_peram(peram_light_sep)  # → propagator from t_curr → t_sink

    for t_curr in t_sink_list_indx:
        # Diagram 0: sign=+2
        # eifj(d[tsink,tsrc]), okpl(s[tcur0,tsrc]), ambn(u[tsink,tcur0]), cgdh(u[tsink,tsrc]),
        # ce(γ₇), Gmo(γ_w), gi(γ₇), bdf(VVV_sink), Lnp(VDV_curr), hjl(VVV_src), Gka(proj)
        corr_3pt_weak[:, :, t_src, t_curr] = 2.0 * cached_contract(
            'eifj,okpl,ambn,cgdh,ce,Gmo,gi,bdf,Lnp,hjl,Gka->GL',
            peram_light_src_sink,                                                   # peram_d  [tsink,tsrc] (Bcast)
            peram_strange_src[t_curr, :, :, :Nev_link, :Nev_src],                   # peram_s  [tcur0,tsrc]
            peram_light_sep[t_curr, :, :, :Nev_src, :Nev_link],                     # peram_u  [tsink,tcur0] (seq)
            peram_light_src_sink,                                                   # peram_u  [tsink,tsrc] (Bcast)
            gamma_baryon,                                                            # gamma_7 at tsink
            gamma_curr,                                                              # gamma_w at tcur0 (G index)
            gamma_baryon,                                                            # C*gamma_5 at tsrc
            sink_3pt_VVV,                                                            # VVV at tsink
            curr_VdV[t_curr],                                                        # VDV at tcur0 (L index)
            src_VVV,                                                                 # VVV at tsrc (.conj())
            projection_3pt,                                                          # projector (G index)
        )

        # Diagram 1: sign=-2 (exchange)
        # eifj(d), okpl(s), agbh(u[tsink,tsrc]), cmdn(u[tsink,tcur0]), ...
        # NOTE: agbh is FORWARD [tsink,tsrc], cmdn is BACKWARD [tsink,tcur0] (needs seq_peram)
        corr_3pt_weak[:, :, t_src, t_curr] += -2.0 * cached_contract(
            'eifj,okpl,agbh,cmdn,ce,Gmo,gi,bdf,Lnp,hjl,Gka->GL',
            peram_light_src_sink,                                                   # peram_d  [tsink,tsrc] (Bcast)
            peram_strange_src[t_curr, :, :, :Nev_link, :Nev_src],                   # peram_s  [tcur0,tsrc]
            peram_light_src_sink,                                                   # peram_u  [tsink,tsrc] (Bcast) — agbh: FORWARD
            peram_light_sep[t_curr, :, :, :Nev_src, :Nev_link],                     # peram_u  [tsink,tcur0] (seq) — cmdn: BACKWARD
            gamma_baryon,
            gamma_curr,
            gamma_baryon,
            sink_3pt_VVV,
            curr_VdV[t_curr],
            src_VVV,
            projection_3pt,
        )

    if rank == 0:
        print(f'MPI data + contraction t_src={t_src}: {(time.perf_counter() - st_mpi_time):.3f} s')


# ============================================================
# 8. MPI Gather
# ============================================================
corr_2pt_lambda = get_mpi_data(data=corr_2pt_lambda, mdtype='TGather', root=0, axis=-1)
corr_2pt_proton = get_mpi_data(data=corr_2pt_proton, mdtype='TGather', root=0, axis=-1)
corr_3pt_weak = get_mpi_data(data=corr_3pt_weak, mdtype='TGather', root=0, axis=-1)

# ============================================================
# 9. 保存结果
# ============================================================
if rank == 0:
    import pathlib

    corr_save_path = f'/public/home/sush/distillation/lqcddb/lambda/result/{Nx}x{Nt}/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}'
    path = pathlib.Path(corr_save_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f'mkdir: {corr_save_path}')
    else:
        print(f'save_path: {corr_save_path}')

    backend.save(
        f'{corr_save_path}/corr_2pt_lambda_gamma0707_stout_smear_20_0.12_src{Nev_src}_sink{Nev_src}.npy',
        corr_2pt_lambda
    )
    backend.save(
        f'{corr_save_path}/corr_2pt_proton_gamma0707_stout_smear_20_0.12_src{Nev_src}_sink{Nev_src}.npy',
        corr_2pt_proton
    )

    for link_indx in range(-link_max, link_max + 1, 1):
        backend.save(
            f'{corr_save_path}/corr_3pt_weak_tsep{t_sep}_link_indx{link_indx}_stout_smear_20_0.12_src{Nev_src}_sink{Nev_src}.npy',
            corr_3pt_weak[:, link_indx + link_max]
        )

    print('Done.')
