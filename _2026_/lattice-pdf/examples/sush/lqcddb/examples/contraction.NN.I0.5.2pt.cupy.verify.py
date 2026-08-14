import sys, time
import numpy as np

from lqcddb.base import *
from lqcddb.eigvectors import vertex_creator
from lqcddb.constant import gamma
from lqcddb.analyse import loop_tsrc  # 保留供用户后处理使用
from lqcddb.io import check_dir_path
from lqcddb.contraction.autowick import wick_contraction

# ── 后端设置 ──────────────────────────────────────────────────
set_backend('cupy')
backend = get_backend()

# ── 参数 ─────────────────────────────────────────────────────
lattice_size = [32, 32, 32, 64]
grid_size = [1, 1, 1, 1]
Nx, Ny, Nz, Nt = lattice_size
Nev_src = 100
conf_id = sys.argv[1]

eigen_base = '/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64'
peram_base = '/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light'
eigen_tag = 'e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy'

# ── MPI 初始化（单进程） ─────────────────────────────────────
mpinit(grid_size=grid_size, latt_size=lattice_size, backend=backend.__name__)
rank = getMPIRank()
size = getMPISize()
Lt = Nt // size

t0 = time.perf_counter()
if rank == 0:
    print(f'=== contraction.NN.I0.5.2pt.cupy.verify.py ===')
    print(f'conf_id={conf_id}, Nev_src={Nev_src}, lattice={lattice_size}')
    print(f'过程: Nπ I=1/2  (逐图验证 — 不使用等价图优化)')

# ── Gamma 矩阵与投影算符 ─────────────────────────────────────
gamma_7 = backend.asarray(gamma(7))
gamma_5 = backend.asarray(gamma(5))
projection = backend.asarray((gamma(0) + gamma(4)) / 2.0)[:, :2]

# ── 动量列表 (P^2 <= 3) ──────────────────────────────────────
Mom_sink_VDV = (
    [[0, 0, 0]]
    + sorted(creat_mom_list(Mom=[0, 0, 1], fix_Q2=True))
    + sorted(creat_mom_list(Mom=[0, 1, 1], fix_Q2=True))
    + sorted(creat_mom_list(Mom=[1, 1, 1], fix_Q2=True))
)
Mom_sink_VVV = (
    [[0, 0, 0]]
    + sorted(creat_mom_list(Mom=[0, 0, 1], fix_Q2=True))
    + sorted(creat_mom_list(Mom=[0, 1, 1], fix_Q2=True))
    + sorted(creat_mom_list(Mom=[1, 1, 1], fix_Q2=True))
)

Mom_len = len(Mom_sink_VDV)
if rank == 0:
    print(f'动量: 共 {Mom_len} 个 (P^2<=3)')

# ── 相位因子 ──────────────────────────────────────────────────
vtx = vertex_creator(Nx=Nx)
phase_exp_2pt = backend.zeros((Mom_len, Nx, Nx, Nx, 3), dtype=complex)
phase_exp_3pt = backend.zeros((Mom_len, Nx, Nx, Nx), dtype=complex)
for mi in range(Mom_len):
    phase_exp_2pt[mi] = vtx.phase_exp_2pt(Mom=Mom_sink_VDV[mi])
    phase_exp_3pt[mi] = vtx.phase_exp_3pt(Mom=Mom_sink_VVV[mi])

# ── 预计算 VdV（GPU）和 VVV（CPU） ────────────────────────────
sink_VdV = backend.zeros((Nt, Mom_len, Nev_src, Nev_src), dtype=complex)
sink_VVV = np.zeros((Nt, Mom_len, Nev_src, Nev_src, Nev_src), dtype=complex)

st_eigen = time.perf_counter()
for t in range(Nt):
    eigvecs = backend.load(f'{eigen_base}/{conf_id}/{conf_id}_t{t:03d}_{eigen_tag}.npy')
    eigvecs_src = eigvecs[:Nev_src]
    sink_VdV[t] = vtx.Mom_VdV_sink_t(phase_exp=phase_exp_2pt, eigvecs=eigvecs_src)
    sink_VVV[t] = backend.asnumpy(
        vtx.Mom_VVV_sink_t(phase_exp=backend.asarray(phase_exp_3pt), eigvecs=eigvecs_src)
    )
del eigvecs
if rank == 0:
    print(f'VdV/VVV: {(time.perf_counter() - st_eigen):.1f}s')

# ── 预加载全部 peram 到 CPU（二维数组） ────────────────────────
st_peram = time.perf_counter()
peram_all = np.zeros((Nt, Nt, 4, 4, Nev_src, Nev_src), dtype=complex)
for t_from in range(Nt):
    data = np.load(f'{peram_base}/{conf_id}/t{t_from:03d}_{eigen_tag}.npy')
    peram_all[t_from] = data[:, :, :, :Nev_src, :Nev_src]
if rank == 0:
    mem_gb = peram_all.nbytes / 1e9
    print(f'加载 peram: {(time.perf_counter() - st_peram):.1f}s  ({mem_gb:.1f} GB CPU)')

# ═══════════════════════════════════════════════════════════════
# Wick 收缩分析 — 运行全部 16 组
# ═══════════════════════════════════════════════════════════════
if rank == 0:
    print('运行 Wick 收缩分析 (16 组)...')

wick_diag = []

# 第0组: P -> Pbar  (1->1)
wick_diag.append(wick_contraction(
    sink_operators=['|', 'u', 'u', 'gamma_7', 'd', '|'],
    source_operators=['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|'],
    Cpt='2pt', curr_operators=[]))

# 第1组: P -> Pbar + pi0(u)  (1->2)
wick_diag.append(wick_contraction(
    sink_operators=['|', 'u', 'u', 'gamma_7', 'd', '|'],
    source_operators=['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'u^d', 'gamma_5', 'u', '|'],
    Cpt='2pt', curr_operators=[]))

# 第2组: P -> -1 x (Pbar + pi0(d))  (1->2)
wick_diag.append(wick_contraction(
    sink_operators=['|', 'u', 'u', 'gamma_7', 'd', '|'],
    source_operators=[-1.0, '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'd^d', 'gamma_5', 'd', '|'],
    Cpt='2pt', curr_operators=[]))

# 第3组: P -> Nbar + pi-  (1->2)
wick_diag.append(wick_contraction(
    sink_operators=['|', 'u', 'u', 'gamma_7', 'd', '|'],
    source_operators=['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'u^d', 'gamma_5', 'd', '|'],
    Cpt='2pt', curr_operators=[]))

# 第4组: P + pi0(u) -> Pbar  (2->1)
wick_diag.append(wick_contraction(
    sink_operators=['|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'],
    source_operators=['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|'],
    Cpt='2pt', curr_operators=[]))

# 第5组: -1 x (P + pi0(d)) -> Pbar  (2->1)
wick_diag.append(wick_contraction(
    sink_operators=[-1, '|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'],
    source_operators=['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|'],
    Cpt='2pt', curr_operators=[]))

# 第6组: N + pi+ -> Pbar  (2->1)
wick_diag.append(wick_contraction(
    sink_operators=['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'u', '|'],
    source_operators=['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|'],
    Cpt='2pt', curr_operators=[]))

# 第7组: N + pi+ -> Nbar + pi-  (2->2)
wick_diag.append(wick_contraction(
    sink_operators=['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'u', '|'],
    source_operators=['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'u^d', 'gamma_5', 'd', '|'],
    Cpt='2pt', curr_operators=[]))

# 第8组: N + pi+ -> -1 x (Pbar + pi0(d))  (2->2)
wick_diag.append(wick_contraction(
    sink_operators=['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'u', '|'],
    source_operators=[-1, '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'd^d', 'gamma_5', 'd', '|'],
    Cpt='2pt', curr_operators=[]))

# 第9组: N + pi+ -> Pbar + pi0(u)  (2->2)
wick_diag.append(wick_contraction(
    sink_operators=['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'u', '|'],
    source_operators=['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'u^d', 'gamma_5', 'u', '|'],
    Cpt='2pt', curr_operators=[]))

# 第10组: P + pi0(u) -> Nbar + pi-  (2->2)
wick_diag.append(wick_contraction(
    sink_operators=['|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'],
    source_operators=['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'u^d', 'gamma_5', 'd', '|'],
    Cpt='2pt', curr_operators=[]))

# 第11组: -1 x (P + pi0(d)) -> Nbar + pi-  (2->2)
wick_diag.append(wick_contraction(
    sink_operators=[-1, '|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'],
    source_operators=['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'u^d', 'gamma_5', 'd', '|'],
    Cpt='2pt', curr_operators=[]))

# 第12组: -1 x (P + pi0(d)) -> -1 x (Pbar + pi0(d))  (2->2)
wick_diag.append(wick_contraction(
    sink_operators=[-1, '|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'],
    source_operators=[-1, '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'd^d', 'gamma_5', 'd', '|'],
    Cpt='2pt', curr_operators=[]))

# 第13组: -1 x (P + pi0(d)) -> Pbar + pi0(u)  (2->2)
wick_diag.append(wick_contraction(
    sink_operators=[-1, '|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'],
    source_operators=['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'u^d', 'gamma_5', 'u', '|'],
    Cpt='2pt', curr_operators=[]))

# 第14组: P + pi0(u) -> -1 x (Pbar + pi0(d))  (2->2)
wick_diag.append(wick_contraction(
    sink_operators=['|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'],
    source_operators=[-1, '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'd^d', 'gamma_5', 'd', '|'],
    Cpt='2pt', curr_operators=[]))

# 第15组: P + pi0(u) -> Pbar + pi0(u)  (2->2)
wick_diag.append(wick_contraction(
    sink_operators=['|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'],
    source_operators=['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'u^d', 'gamma_5', 'u', '|'],
    Cpt='2pt', curr_operators=[]))

if rank == 0:
    total_diag = sum(len(w['result_indx']) for w in wick_diag)
    print(f'  共 {total_diag} 个图（将对每个图逐一计算，不做等价优化）')

# ═══════════════════════════════════════════════════════════════
# 将 Wick 组分类映射到 GEVP 矩阵元素
# ═══════════════════════════════════════════════════════════════

def count_separators(op_list):
    """统计 '|' 分隔符个数，用于判断算符类型。"""
    return sum(1 for x in op_list if x == '|')

def classify_operator(n_seps):
    """分类: 2 个分隔符 = 类型1（单重子），>=3 个 = 类型2/3（重子+介子）。"""
    if n_seps == 2:
        return 1
    return 2

def gevp_elements(sink_type, source_type):
    """根据汇/源算符类型，确定该组对哪些 GEVP 矩阵元素有贡献。"""
    if sink_type == 1 and source_type == 1:
        return [(0, 0)]
    elif sink_type == 1 and source_type == 2:
        return [(0, 1), (0, 2)]
    elif sink_type == 2 and source_type == 1:
        return [(1, 0), (2, 0)]
    elif sink_type == 2 and source_type == 2:
        return [(1, 1), (1, 2), (2, 1), (2, 2)]
    return []

# 对每组进行分类
group_info = []
for w in wick_diag:
    sink_seps = count_separators(w['sink_operators'])
    src_seps = count_separators(w['source_operators'])
    snk_type = classify_operator(sink_seps)
    src_type = classify_operator(src_seps)
    elements = gevp_elements(snk_type, src_type)
    group_info.append({
        'wick': w,
        'sink_type': snk_type,
        'source_type': src_type,
        'elements': elements,
        'ndiag': len(w['result_indx']),
    })

# ── 打印分类信息 ────────────────────────────────────────────
if rank == 0:
    total_contractions = 0
    for gi, info in enumerate(group_info):
        n_cont = info['ndiag'] * len(info['elements'])
        total_contractions += n_cont
        elems = [f'M^{{{n+1},{m+1}}}' for n, m in info['elements']]
        print(f'  第{gi:2d}组: {info["ndiag"]:2d} 图 × {len(info["elements"])} 个GEVP元素 = {n_cont:3d} 次收缩  → {", ".join(elems)}')
    print(f'  总计: {total_contractions} 次收缩（每个图 × 每个GEVP元素独立计算）')

# ═══════════════════════════════════════════════════════════════
# V 张量动量分配表
# 对每个 GEVP 元素 (n,m): (汇_VVV, 汇_VDV, 源_VVV, 源_VDV)
# 'K' = 动量列表（9个动量）, '0' = 零动量（1个）, None = 不存在
# ═══════════════════════════════════════════════════════════════
V_ASSIGN = {
    (0, 0): ('K', None, 'K', None),
    (0, 1): ('K', None, 'K', '0'),
    (0, 2): ('K', None, '0', 'K'),
    (1, 0): ('K', '0',  'K', None),
    (2, 0): ('0', 'K',  'K', None),
    (1, 1): ('K', '0',  'K', '0'),
    (1, 2): ('K', '0',  '0', 'K'),
    (2, 1): ('0', 'K',  'K', '0'),
    (2, 2): ('0', 'K',  '0', 'K'),
}

# Peram 时间标签 → 变量名映射
PERAM_TIME_MAP = {
    ('tsink', 'tsrc'): 'peram_src_sink',     # 前向传播子
    ('tsrc', 'tsink'): 'peram_sink_src',     # 反向传播子
    ('tsink', 'tsink'): 'peram_sink_sink',   # 汇端等时
    ('tsrc', 'tsrc'): 'peram_src_src',       # 源端等时
}

# Gamma 名称 → 张量变量映射
GAMMA_MAP = {'gamma_7': 'gamma_7', 'gamma_5': 'gamma_5'}

# ═══════════════════════════════════════════════════════════════
# 动态收缩构建器
# ═══════════════════════════════════════════════════════════════

def build_contraction(wick, diag_idx, n, m, tensor_dict):
    """为 GEVP 元素 (n,m) 中的某个图构建 einsum 字符串和张量列表。"""
    einsum_raw = wick['result_indx'][diag_idx][0]
    perams = wick['peram'][diag_idx]
    v_info = wick['V']
    gamma_info = wick['gamma_pos']

    # ── 将 peram 时间标签映射为张量变量 ──
    peram_vars = []
    for p in perams:
        t_key = tuple(p[4])
        pname = PERAM_TIME_MAP.get(t_key)
        if pname is None:
            raise ValueError(f"未知的 peram 时间标签: {t_key}")
        peram_vars.append(tensor_dict[pname])

    # ── 将 V 结构映射为 V 张量变量 ──
    snk_vvv, snk_vdv, src_vvv, src_vdv = V_ASSIGN[(n, m)]
    v_vars = []
    for v in v_info:
        vtype = v[1]
        vtime = v[3]
        if 'VVV' in vtype:
            if vtime == 'tsink':
                key = 'snkVVVK' if snk_vvv == 'K' else 'snkVVV0'
            else:
                key = 'srcVVVK' if src_vvv == 'K' else 'srcVVV0'
        else:  # VDV
            if vtime == 'tsink':
                key = 'snkVDVK' if snk_vdv == 'K' else 'snkVDV0'
            else:
                key = 'srcVDVK' if src_vdv == 'K' else 'srcVDV0'
        v_vars.append(tensor_dict[key])

    # ── 将 gamma 名称映射为 gamma 张量 ──
    gamma_vars = [tensor_dict[GAMMA_MAP.get(g[1], g[1])] for g in gamma_info]

    # ── 构建带 K 前缀的 einsum 字符串 ──
    lhs, rhs = einsum_raw.split('->')
    parts = lhs.split(',')
    n_perams = len(perams)
    n_gammas = len(gamma_info)

    peram_parts = parts[:n_perams]
    gamma_parts = parts[n_perams:n_perams + n_gammas]
    v_parts_raw = parts[n_perams + n_gammas:]

    # 给 V 指标添加 K 前缀（动量维度）
    v_parts_k = ['K' + vp for vp in v_parts_raw]

    # 从原始 RHS 提取小写自由指标用于投影
    rhs_lower = ''.join(c for c in rhs if c.islower())
    sink_proj = f'{rhs_lower[0]}y'   # 汇端投影指标
    src_proj = f'{rhs_lower[1]}z'    # 源端投影指标

    # 组装完整 einsum 字符串
    einsum_parts = peram_parts + gamma_parts + v_parts_k + [sink_proj, src_proj]
    einsum = ','.join(einsum_parts) + '->Kyz'

    # ── 组装张量参数列表 ──
    tensor_args = peram_vars + gamma_vars + v_vars + [tensor_dict['projection'], tensor_dict['projection']]

    return einsum, tensor_args

# ═══════════════════════════════════════════════════════════════
# 输出: 3x3 GEVP 矩阵, 27 个动量, Nt x Nt 时间, 2x2 自旋
# ═══════════════════════════════════════════════════════════════
corr = backend.zeros((3, 3, Mom_len, Nt, Nt, 2, 2), dtype=complex)

# ── 主循环 ────────────────────────────────────────────────────
if rank == 0:
    print(f'\n开始主循环（逐图验证模式），共 {Nt} 个源时间...')

contraction_count = 0  # 统计收缩次数

for t_src in range(0, 1, 1):
    st_loop = time.perf_counter()

    # 源端顶点（从 t_src 持有者 Bcast）
    source_VdV_full = get_mpi_data(sink_VdV[t_src // size], mdtype='Bcast', root=t_src % size).transpose(0, 2, 1).conj()
    source_VVV_full = backend.asarray(get_mpi_data(sink_VVV[t_src // size], mdtype='Bcast', root=t_src % size)).conj()

    t_sink_list_rank, _, t_sink_list_indx = get_mpi_tlist(Nt=Nt, t=range(t_src, t_src + Nt // 2, 1), gtype='TScatter')

    for t_sink_indx, t_sink in enumerate(t_sink_list_indx):
        t_real_sink = t_sink_list_rank[t_sink_indx]

        # 汇端 V 结构
        sink_VVV_ts = backend.asarray(sink_VVV[t_sink])
        sink_VdV_ts = sink_VdV[t_sink]

        # Peram 切片
        peram_src_sink = backend.asarray(peram_all[t_src,  t_sink])
        peram_sink_src = backend.asarray(peram_all[t_sink, t_src])
        peram_src_src  = backend.asarray(peram_all[t_src,  t_src])
        peram_sink_sink = backend.asarray(peram_all[t_sink, t_sink])

        # 零动量 V 张量 → 形状 (1, ...)
        snkVVV0 = sink_VVV_ts[0:1]
        snkVDV0 = sink_VdV_ts[0:1]
        srcVVV0 = source_VVV_full[0:1]
        srcVDV0 = source_VdV_full[0:1]

        for Mom_indx in range(3):
            Mom_list = [x for x in range(Mom_indx, Mom_len, 3)]

            # 完整动量 K 的 V 张量 → 形状 (len(Mom_list), ...)
            snkVVVK = sink_VVV_ts[Mom_list]
            snkVDVK = sink_VdV_ts[Mom_list]
            srcVVVK = source_VVV_full[Mom_list]
            srcVDVK = source_VdV_full[Mom_list]

            # ── 张量名称 → 实际数组查找表 ──
            tensor_dict = {
                'peram_src_sink': peram_src_sink,
                'peram_sink_src': peram_sink_src,
                'peram_src_src': peram_src_src,
                'peram_sink_sink': peram_sink_sink,
                'snkVVVK': snkVVVK, 'snkVVV0': snkVVV0,
                'snkVDVK': snkVDVK, 'snkVDV0': snkVDV0,
                'srcVVVK': srcVVVK, 'srcVVV0': srcVVV0,
                'srcVDVK': srcVDVK, 'srcVDV0': srcVDV0,
                'gamma_7': gamma_7, 'gamma_5': gamma_5,
                'projection': projection,
            }

            # ── 逐图逐 GEVP 元素计算（不做等价优化）──
            for gi, info in enumerate(group_info):
                w = info['wick']
                for n, m in info['elements']:
                    for di in range(info['ndiag']):
                        sign = w['result_sign'][di]
                        einsum, tensors = build_contraction(w, di, n, m, tensor_dict)
                        corr[n, m, Mom_list, t_src, t_sink] += sign * cached_contract(
                            einsum, *tensors, optimize=['greedy', 'dp'])
                        contraction_count += 1

    if rank == 0:
        dt = time.perf_counter() - st_loop
        print(f'[t_src={t_src:02d}] {dt:.1f}s  累计收缩次数: {contraction_count}  总用时: {(time.perf_counter()-t0):.0f}s')

# ── 保存结果 ──────────────────────────────────────────────────
if rank == 0:
    print(f'\n总收缩次数: {contraction_count}')
    save_dir = (f'/public/home/sush/distillation/0v2b/result/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}')
    check_dir_path(save_dir)
    save_path = (
        f'{save_dir}/corr_NN_I05_GEVP_3x3_src{Nev_src}_verify.npy'
    )

    corr_save = backend.asnumpy(corr)
    np.save(save_path, corr_save)
    print(f'\n已保存: {save_path}')
    print(f'形状 (3,3,27,Nt,Nt,2,2): {corr_save.shape}')
    print(f'完成。总用时: {(time.perf_counter() - t0):.1f}s')
