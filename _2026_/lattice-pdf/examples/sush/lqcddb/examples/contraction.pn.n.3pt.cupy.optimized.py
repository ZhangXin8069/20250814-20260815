import os
import sys
import time
import pathlib
import numpy as np

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from function_contraction import *

from opt_einsum import contract, contract_path, contract_expression

from cupy.cuda.runtime import getDeviceCount as cudaGetDeviceCount
A = cudaGetDeviceCount()

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

mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend.__name__)

rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()

Nx, Ny, Nz, Nt = lattice_size
Lx, Ly, Lz, Lt = [lattice_size[x]//grid_size[x] for x in range(len(lattice_size))]

fun_eigen = corr_eigvecs(Nx = Nx, backend = backend)

Mom_sink_VDV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))
Mom_sink_VVV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))[::-1]

Mom_sink_link = Mom_sink_VDV

Mom_len = len(Mom_sink_VDV)

phase_exp_2pt = backend.zeros((Mom_len, Nx, Nx, Nx, Nc), dtype = complex)
phase_exp_3pt = backend.zeros((Mom_len, Nx, Nx, Nx), dtype = complex)

for Mom_indx in range(Mom_len):
    phase_exp_2pt[Mom_indx] = fun_eigen.phase_exp_2pt(Mom = Mom_sink_VDV[Mom_indx])
    phase_exp_3pt[Mom_indx] = fun_eigen.phase_exp_3pt(Mom = Mom_sink_VVV[Mom_indx])

Nev_src = 100
Nev_link = 400
link_max = 0
t_sep = 12

if link_max > 0:
    if rank == 0:
        gauge_link = Readin_gauge()

    else:
        gauge_link = None

    gauge_link = get_mpi_data(gauge_link, mdtype = 'TScatter', root = 0, axis = 0)

else:
    gauge_link = False

t_rank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'Scatter')

VdV_link = backend.zeros((Lt, Mom_len, 2*link_max + 1, Nev_link, Nev_link), dtype = complex)
sink_VVV = np.zeros((Lt, Mom_len, Nev_src, Nev_src, Nev_src), dtype = complex)

# ============================================================
# Step 1: Load eigenvectors and compute VdV, VVV projections
# ============================================================
st_eigen = time.perf_counter()
for t_src_indx, t_src in enumerate(t_rank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t{t_src:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')

    for Mom_indx in range(Mom_len):
        VdV_link[t_src_indx, (Mom_indx):((Mom_indx + 1))] = fun_eigen.VdV_sink_t_link(eigvecs = eigvecs[:], link_dir = 'Z', link_max = link_max, phase_exp = phase_exp_2pt[(Mom_indx):((Mom_indx + 1))], gauge_link = gauge_link, t = t_src_indx)
        sink_VVV[t_src_indx, Mom_indx] = fun_eigen.Mom_VVV_sink_t_3(phase_exp = phase_exp_3pt[Mom_indx], eigvecs = eigvecs[:Nev_src]).get()

if rank == 0:
    print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')

# ============================================================
# OPTIMIZATION 1: Pre-compute gamma matrices on GPU once
# ============================================================
gamma7_gpu  = backend.asarray(gamma(7))   # (4, 4)  gamma3*gamma1
gamma5_gpu  = backend.asarray(gamma(5))   # (4, 4)  gamma5
gamma0_gpu  = backend.asarray(gamma(0))   # (4, 4)  identity

gamma_curr = (gamma0_gpu - gamma5_gpu) @ backend.asarray([gamma(1), gamma(2), gamma(3), gamma(4)])  # (4, 4, 4)

projection = (gamma(0) + gamma(4))/2

corr_3pt_matrix = backend.zeros((Mom_len, Mom_len, len(gamma_curr), 2*link_max + 1, Ns, Ns, Nt, Lt), dtype = complex)

# ============================================================
# OPTIMIZATION 2: Pre-compiled contraction expressions
# ============================================================
# 12 contraction patterns: 6 in Group 1 + 6 in Group 2
# Group 1: D = peram_u_sink_seq[t_curr],  E = peram_u_src[t_src]
# Group 2: D = peram_u_src[t_sink],       E = peram_u_src_seq (seq_peram of t_curr)
#
# The gamma tensors (args 5-8: gamma7, gamma_curr, gamma7, gamma7)
# are marked as constants and baked into the expressions.

_CONTRACT_SPECS = [
    # (einsum_str, sign, group_id)
    # Group 1 (group_id=0): 6 patterns
    ('agbh,eqfr,ospt,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas', -1.0, 0),
    ('agbh,esft,oqpr,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas', -1.0, 0),
    ('aqbr,egfh,ospt,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas', -1.0, 0),
    ('aqbr,esft,ogph,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas', +1.0, 0),
    ('asbt,egfh,oqpr,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas', +1.0, 0),
    ('asbt,eqfr,ogph,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas', -1.0, 0),
    # Group 2 (group_id=1): 6 patterns
    ('agbh,eqfr,ospt,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas', +1.0, 1),
    ('agbh,esft,oqpr,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas', +1.0, 1),
    ('aqbr,egfh,ospt,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas', -1.0, 1),
    ('aqbr,esft,ogph,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas', -1.0, 1),
    ('asbt,egfh,oqpr,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas', -1.0, 1),
    ('asbt,eqfr,ogph,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas', +1.0, 1),
]

# Will be filled after first peram load (we need shapes)
_EXPR_CACHE_G1 = None  # list of 6 compiled expressions for Group 1
_EXPR_CACHE_G2 = None  # list of 6 compiled expressions for Group 2
_EXPR_USES_CONSTANTS = False  # True if gamma tensors are baked in (only 9 args needed)


def _build_contraction_expressions(shapes_G1, shapes_G2):
    """
    Build all 12 contract_expression objects once.
    Gamma matrices (positions 5,6,7,8) are stored as constant tensors
    so they stay on GPU and are reused across calls.

    When constants=[5,6,7,8], we pass the actual tensor at those positions
    and shapes at the other positions. The returned expression only needs
    the 9 non-constant tensors at call time.

    Returns: (list_of_6_expr_G1, list_of_6_expr_G2)
    """
    const_idx = [5, 6, 7, 8]

    # Build arg lists with actual gamma tensors at constant positions
    const_G1 = list(shapes_G1)
    const_G1[5] = gamma7_gpu
    const_G1[6] = gamma_curr
    const_G1[7] = gamma7_gpu
    const_G1[8] = gamma7_gpu

    const_G2 = list(shapes_G2)
    const_G2[5] = gamma7_gpu
    const_G2[6] = gamma_curr
    const_G2[7] = gamma7_gpu
    const_G2[8] = gamma7_gpu

    exprs_G1 = []
    exprs_G2 = []
    uses_constants = True

    for einsum_str, sign, group_id in _CONTRACT_SPECS:
        args = const_G1 if group_id == 0 else const_G2

        try:
            expr = contract_expression(
                einsum_str,
                *args,
                constants=const_idx,
                optimize='optimal'
            )
        except Exception:
            # Fallback: without constants if constant folding fails
            uses_constants = False
            shapes = shapes_G1 if group_id == 0 else shapes_G2
            expr = contract_expression(
                einsum_str,
                *shapes,
                optimize='optimal'
            )

        if group_id == 0:
            exprs_G1.append((expr, sign))
        else:
            exprs_G2.append((expr, sign))

    return exprs_G1, exprs_G2, uses_constants


# ============================================================
# Main computation loop over source times
# ============================================================
for t_src in range(Nt):
    st_cal = time.perf_counter()

    source_VdV = get_mpi_data(data = VdV_link[t_src//size, :, link_max, :Nev_src, :Nev_src], mdtype = 'Bcast', root = t_src%size).transpose(0, 2, 1).conj()
    source_VVV = backend.asarray(get_mpi_data(data = sink_VVV[t_src//size], mdtype = 'Bcast', root = t_src%size)).conj()

    t_curr_list_rank, _, t_curr_list_indx = get_mpi_tlist(Nt = Nt, t = range(t_src, t_src + 2 * t_sep + 2, 1), gtype = 'Scatter')

    st_peram = time.perf_counter()
    if rank == 0:
        peram_u_src = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{t_src:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy'
            )[..., :Nev_link, :Nev_src]

    else:
        peram_u_src = None

    peram_u_src = get_mpi_data(data = peram_u_src, mdtype = 'TScatter', root = 0, axis = 0)

    t_sink = (t_src + t_sep) % Nt

    if rank == 0:
        peram_u_sink_seq = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{t_sink:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy'
            )[..., :Nev_link, :Nev_src]
    else:
        peram_u_sink_seq = None

    peram_u_sink_seq = get_mpi_data(data = peram_u_sink_seq, mdtype = 'TScatter', root = 0, axis = 0)
    peram_u_sink_seq = seq_peram(peram_u_sink_seq)

    if rank == 0:
        print(f'load peram use time {(time.perf_counter() - st_peram):.3f} s')

    # ============================================================
    # OPTIMIZATION 3: Hoist loop-invariant tensors out of t_curr loop
    # ============================================================
    # These do NOT depend on t_curr:
    peram_A  = peram_u_src[t_sink, :, :, :Nev_src, :Nev_src]   # used 2-3x per contraction
    peram_E1 = peram_u_src[t_src,  :, :, :Nev_src, :Nev_src]   # Group 1 only
    # NOTE: peram_E2 = seq_peram(peram_u_src[t_curr]) depends on t_curr — kept inside loop
    sink_VVV_t_sink = backend.asarray(sink_VVV[t_sink])          # GPU, full momentum range

    # Pre-slice source tensors by Mom group (each group has 3 momenta)
    Mom_groups = max(1, int(backend.ceil(Mom_len / 1)))
    source_VdV_slices = [
        source_VdV[i * 3:(i + 1) * 3] for i in range(Mom_groups)
    ]
    source_VVV_slices = [
        source_VVV[i * 3:(i + 1) * 3] for i in range(Mom_groups)
    ]

    # ============================================================
    # OPTIMIZATION 4: Build contraction expressions on first iteration
    # ============================================================
    if _EXPR_CACHE_G1 is None and len(t_curr_list_indx) > 0:
        # Use first t_curr to determine shapes
        t0 = t_curr_list_indx[0]
        peram_C_sample  = peram_u_src[t0, :, :, :Nev_link, :Nev_src]
        peram_D1_sample = peram_u_sink_seq[t0, :, :, :Nev_src, :Nev_link]
        peram_D2_sample = peram_A  # same shape as peram_A

        VdV_sample = VdV_link[t0]
        src_dV_sample = source_VdV_slices[0]  # (3, Nev_src, Nev_src)
        src_VVV_sample = source_VVV_slices[0]  # (3, Nev_src, Nev_src, Nev_src)

        # Shapes for Group 1
        shapes_G1 = [
            peram_A.shape, peram_A.shape, peram_C_sample.shape,
            peram_D1_sample.shape, peram_E1.shape,
            gamma7_gpu.shape, gamma_curr.shape, gamma7_gpu.shape, gamma7_gpu.shape,
            sink_VVV_t_sink.shape, VdV_sample.shape,
            src_dV_sample.shape, src_VVV_sample.shape,
        ]

        # Shapes for Group 2 (peram_E2 has same shape as peram_D1_sample)
        shapes_G2 = [
            peram_A.shape, peram_A.shape, peram_C_sample.shape,
            peram_D2_sample.shape, peram_D1_sample.shape,  # peram_E2 shape = peram_D1 shape
            gamma7_gpu.shape, gamma_curr.shape, gamma7_gpu.shape, gamma7_gpu.shape,
            sink_VVV_t_sink.shape, VdV_sample.shape,
            src_dV_sample.shape, src_VVV_sample.shape,
        ]

        _EXPR_CACHE_G1, _EXPR_CACHE_G2, _EXPR_USES_CONSTANTS = _build_contraction_expressions(shapes_G1, shapes_G2)

        if rank == 0:
            print(f'[Optimization] Built {len(_EXPR_CACHE_G1)} + {len(_EXPR_CACHE_G2)} pre-compiled contraction expressions')

    # ============================================================
    # Inner loops over t_curr and Mom
    # ============================================================
    for t_curr in t_curr_list_indx:
        peram_u_src_seq_t = seq_peram(peram_u_src[t_curr])

        # Tensors that depend on t_curr
        peram_C  = peram_u_src[t_curr, :, :, :Nev_link, :Nev_src]
        peram_D1 = peram_u_sink_seq[t_curr, :, :, :Nev_src, :Nev_link]
        VdV_curr = VdV_link[t_curr]

        for Mom_indx in range(Mom_groups):
            src_dV  = source_VdV_slices[Mom_indx]
            src_VVV = source_VVV_slices[Mom_indx]
            out_slice = (slice(Mom_indx * 3, (Mom_indx + 1) * 3),)

            # peram_E2 depends on t_curr (seq_peram of peram_u_src[t_curr])
            peram_E2 = peram_u_src_seq_t[:, :, :Nev_src, :Nev_link]

            # ----------------------------------------------------
            # OPTIMIZATION 5: Accumulate locally before writing to
            # the large corr_3pt_matrix (reduces GPU memory traffic)
            # ----------------------------------------------------

            if _EXPR_CACHE_G1 is not None:
                if _EXPR_USES_CONSTANTS:
                    # Gamma tensors are baked in — only 9 non-constant args
                    args_G1 = [
                        peram_A, peram_A, peram_C, peram_D1, peram_E1,
                        sink_VVV_t_sink, VdV_curr, src_dV, src_VVV,
                    ]
                    args_G2 = [
                        peram_A, peram_A, peram_C, peram_A, peram_E2,
                        sink_VVV_t_sink, VdV_curr, src_dV, src_VVV,
                    ]
                else:
                    # Fallback: all 13 args including gamma tensors
                    args_G1 = [
                        peram_A, peram_A, peram_C, peram_D1, peram_E1,
                        gamma7_gpu, gamma_curr, gamma7_gpu, gamma7_gpu,
                        sink_VVV_t_sink, VdV_curr, src_dV, src_VVV,
                    ]
                    args_G2 = [
                        peram_A, peram_A, peram_C, peram_A, peram_E2,
                        gamma7_gpu, gamma_curr, gamma7_gpu, gamma7_gpu,
                        sink_VVV_t_sink, VdV_curr, src_dV, src_VVV,
                    ]

                # First pattern initializes the accumulator
                acc = _EXPR_CACHE_G1[0][0](*args_G1) * _EXPR_CACHE_G1[0][1]
                for expr, sign in _EXPR_CACHE_G1[1:]:
                    acc += expr(*args_G1) * sign

                # --- Group 2: 6 patterns (D=peram_A, E=peram_E2) ---
                for expr, sign in _EXPR_CACHE_G2:
                    acc += expr(*args_G2) * sign

                # Single write-back to the large array
                corr_3pt_matrix[out_slice + (..., t_src, t_curr)] += acc

            else:
                # Fallback: original contract() calls
                # (Only used if no t_curr values exist, which shouldn't happen)
                corr_3pt_matrix[out_slice + (..., t_src, t_curr)] = (contract(
                    'agbh,eqfr,ospt,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas',
                    peram_A, peram_A, peram_C, peram_D1, peram_E1,
                    gamma7_gpu, gamma_curr, gamma7_gpu, gamma7_gpu,
                    sink_VVV_t_sink[:], VdV_curr, src_dV, src_VVV,
                    ) * -1.0) + corr_3pt_matrix[out_slice + (..., t_src, t_curr)]

                corr_3pt_matrix[out_slice + (..., t_src, t_curr)] = (contract(
                    'agbh,eqfr,ospt,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas',
                    peram_A, peram_A, peram_C, peram_A, peram_E2,
                    gamma7_gpu, gamma_curr, gamma7_gpu, gamma7_gpu,
                    sink_VVV_t_sink[:], VdV_curr, src_dV, src_VVV,
                    ) * 1.0) + corr_3pt_matrix[out_slice + (..., t_src, t_curr)]

                corr_3pt_matrix[out_slice + (..., t_src, t_curr)] = (contract(
                    'agbh,esft,oqpr,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas',
                    peram_A, peram_A, peram_C, peram_D1, peram_E1,
                    gamma7_gpu, gamma_curr, gamma7_gpu, gamma7_gpu,
                    sink_VVV_t_sink[:], VdV_curr, src_dV, src_VVV,
                    ) * -1.0) + corr_3pt_matrix[out_slice + (..., t_src, t_curr)]

                corr_3pt_matrix[out_slice + (..., t_src, t_curr)] = (contract(
                    'agbh,esft,oqpr,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas',
                    peram_A, peram_A, peram_C, peram_A, peram_E2,
                    gamma7_gpu, gamma_curr, gamma7_gpu, gamma7_gpu,
                    sink_VVV_t_sink[:], VdV_curr, src_dV, src_VVV,
                    ) * 1.0) + corr_3pt_matrix[out_slice + (..., t_src, t_curr)]

                corr_3pt_matrix[out_slice + (..., t_src, t_curr)] = (contract(
                    'aqbr,egfh,ospt,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas',
                    peram_A, peram_A, peram_C, peram_D1, peram_E1,
                    gamma7_gpu, gamma_curr, gamma7_gpu, gamma7_gpu,
                    sink_VVV_t_sink[:], VdV_curr, src_dV, src_VVV,
                    ) * -1.0) + corr_3pt_matrix[out_slice + (..., t_src, t_curr)]

                corr_3pt_matrix[out_slice + (..., t_src, t_curr)] = (contract(
                    'aqbr,egfh,ospt,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas',
                    peram_A, peram_A, peram_C, peram_A, peram_E2,
                    gamma7_gpu, gamma_curr, gamma7_gpu, gamma7_gpu,
                    sink_VVV_t_sink[:], VdV_curr, src_dV, src_VVV,
                    ) * -1.0) + corr_3pt_matrix[out_slice + (..., t_src, t_curr)]

                corr_3pt_matrix[out_slice + (..., t_src, t_curr)] = (contract(
                    'aqbr,esft,ogph,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas',
                    peram_A, peram_A, peram_C, peram_D1, peram_E1,
                    gamma7_gpu, gamma_curr, gamma7_gpu, gamma7_gpu,
                    sink_VVV_t_sink[:], VdV_curr, src_dV, src_VVV,
                    ) * 1.0) + corr_3pt_matrix[out_slice + (..., t_src, t_curr)]

                corr_3pt_matrix[out_slice + (..., t_src, t_curr)] = (contract(
                    'aqbr,esft,ogph,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas',
                    peram_A, peram_A, peram_C, peram_A, peram_E2,
                    gamma7_gpu, gamma_curr, gamma7_gpu, gamma7_gpu,
                    sink_VVV_t_sink[:], VdV_curr, src_dV, src_VVV,
                    ) * -1.0) + corr_3pt_matrix[out_slice + (..., t_src, t_curr)]

                corr_3pt_matrix[out_slice + (..., t_src, t_curr)] = (contract(
                    'asbt,egfh,oqpr,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas',
                    peram_A, peram_A, peram_C, peram_D1, peram_E1,
                    gamma7_gpu, gamma_curr, gamma7_gpu, gamma7_gpu,
                    sink_VVV_t_sink[:], VdV_curr, src_dV, src_VVV,
                    ) * 1.0) + corr_3pt_matrix[out_slice + (..., t_src, t_curr)]

                corr_3pt_matrix[out_slice + (..., t_src, t_curr)] = (contract(
                    'asbt,egfh,oqpr,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas',
                    peram_A, peram_A, peram_C, peram_A, peram_E2,
                    gamma7_gpu, gamma_curr, gamma7_gpu, gamma7_gpu,
                    sink_VVV_t_sink[:], VdV_curr, src_dV, src_VVV,
                    ) * -1.0) + corr_3pt_matrix[out_slice + (..., t_src, t_curr)]

                corr_3pt_matrix[out_slice + (..., t_src, t_curr)] =(contract(
                    'asbt,eqfr,ogph,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas',
                    peram_A, peram_A, peram_C, peram_D1, peram_E1,
                    gamma7_gpu, gamma_curr, gamma7_gpu, gamma7_gpu,
                    sink_VVV_t_sink[:], VdV_curr, src_dV, src_VVV,
                    ) * -1.0) + corr_3pt_matrix[out_slice + (..., t_src, t_curr)]

                corr_3pt_matrix[out_slice + (..., t_src, t_curr)] =(contract(
                    'asbt,eqfr,ogph,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt->MNGLas',
                    peram_A, peram_A, peram_C, peram_A, peram_E2,
                    gamma7_gpu, gamma_curr, gamma7_gpu, gamma7_gpu,
                    sink_VVV_t_sink[:], VdV_curr, src_dV, src_VVV,
                    ) * 1.0) + corr_3pt_matrix[out_slice + (..., t_src, t_curr)]

    free, total = backend.cuda.runtime.memGetInfo()

    if rank == 0:
        print(f'calculate 2pt of t_src {t_src} use time {(time.perf_counter() - st_cal):.3f} s. device mem: {(total - free) / 1024**3} GB, free:{free / 1024**3} GB.')


corr_3pt_matrix = get_mpi_data(data = corr_3pt_matrix, mdtype = 'TGather', root = 0, axis = -1)

if rank == 0:

    corr_save_path = f'/public/home/sush/distillation/0v2b/result/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}'

    path = pathlib.Path(corr_save_path)

    if path.exists():
        print('save_path:',corr_save_path)

    else:
        path.mkdir(parents = True, exist_ok = True)
        print('mkdir_save_path:',corr_save_path)
    corr_3pt_matrix = loop_tsrc(
        corr_3pt_matrix,
        indx = [-2, -1],
        Boundary_Conditions = 'Antiperiodic',
        Ctype = '3pt',
        t_sep = t_sep
    )

    backend.save(
        f'{corr_save_path}/corr_3pt_neutron_J_mu_neutron_pi-_src100.npy',
        corr_3pt_matrix[:]
        )
