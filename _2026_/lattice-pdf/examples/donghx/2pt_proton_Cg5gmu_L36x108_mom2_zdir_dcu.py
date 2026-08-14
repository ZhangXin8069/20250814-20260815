#!/usr/bin/env python3
import numpy as np
import math
import os
import sys
import fileinput
import time
import torch

from gamma_matrix_DR_torch import *
from input_output_4_torch import *
from opt_einsum import contract


# ==============================================================================
# Basic setup
# ==============================================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Job started")
print("device =", device)

Nc = 3

# ------------------------------------------------------------------------------
# If you want to read parameters from an input file later, you can restore this
# block.  Here I keep your original hard-coded setup.
# ------------------------------------------------------------------------------
# infile = fileinput.input()
# for line in infile:
#     tmp = line.split()
#     if tmp[0] == "Nt":
#         Nt = int(tmp[1])
#     if tmp[0] == "Nx":
#         Nx = int(tmp[1])
#     if tmp[0] == "conf_id":
#         conf_id = tmp[1]
#     if tmp[0] == "Nev":
#         Nev = int(tmp[1])
#     if tmp[0] == "Nev1":
#         Nev1 = int(tmp[1])
#     if tmp[0] == "Px":
#         Px = int(tmp[1])
#     if tmp[0] == "Py":
#         Py = int(tmp[1])
#     if tmp[0] == "Pz":
#         Pz = int(tmp[1])
#     if tmp[0] == "mom_smear":
#         mom_smear = int(tmp[1])
#     if tmp[0] == "path_eigenvectors":
#         eig_dir = tmp[1]
#     if tmp[0] == "peram_u_dir":
#         peram_u_dir = tmp[1]
#     if tmp[0] == "peram_u_dir_tsr":
#         peram_u_dir_tsr = tmp[1]
#     if tmp[0] == "VVV_dir":
#         VVV_dir = tmp[1]
#     if tmp[0] == "corr_nucl_dir":
#         corr_nucl_dir = tmp[1]


Nt = 108
Nx = 36
conf_id = sys.argv[1]
Nev = 200
Nev1 = 200
Px = 0
Py = 0
Pz = 0
mom_smear = 2
momsmear_phase = -2
# mom_smear = 0
# momsmear_phase = 0

eig_dir = "/public/group/lqcd/eigensystem/beta6.498_mu-0.2150_ms-1926_L36x108/%s/" % (
    conf_id
)
peram_u_dir = (
    "/public/home/sunp/sunpeng_new_disk/mom_smear_perambulators/"
    "beta6.498_mu-0.2150_ms-1926_L36x108/output_dir_data/mz-2_my0_mx0/%s/" % (conf_id)
)
corr_nucl_dir = (
    "/public/group/lqcd/donghx/2pt_Result/"
    "beta6.498_mu-0.2150_ms-1926_L36x108/momsmear2z/%s/" % (conf_id)
)
os.makedirs(corr_nucl_dir, exist_ok=True)

st00 = time.time()


# ==============================================================================
# Small helpers
# ==============================================================================
def gpu_sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


def phase_calc_fast(Mom, Nx, device):
    """
    Vectorized phase calculation on GPU/CPU.

    Your code uses Mom = [Pz, Py, Px] and coordinate order [z, y, x].
    This returns a flattened array with index z * Nx * Nx + y * Nx + x,
    matching your original phase_calc.
    """
    Mom = torch.as_tensor(Mom, device=device, dtype=torch.float32)

    z = torch.arange(Nx, device=device, dtype=torch.float32)
    y = torch.arange(Nx, device=device, dtype=torch.float32)
    x = torch.arange(Nx, device=device, dtype=torch.float32)

    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    phase_arg = Mom[0] * zz + Mom[1] * yy + Mom[2] * xx

    return torch.exp(-2.0j * torch.pi * phase_arg / Nx).reshape(-1).to(torch.cfloat)


# ==============================================================================
# Projectors and interpolating operators
# ==============================================================================
matrix_pplus = 0.5 * (gamma(0) + gamma(4))  # positive parity projection
matrix_pplus = matrix_pplus.to(device=device, dtype=torch.cfloat)

# The negative-parity projector is not used below because only pp is saved.
# Keeping the line here makes it easy to restore pm if needed.
# matrix_pminus = 0.5 * (gamma(0) - gamma(4))
# matrix_pminus = matrix_pminus.to(device=device, dtype=torch.cfloat)

# interpolator0 = "Cg5"
# interpolator1 = "Cg5g3"
# interpolator2 = "Cg5g4"

# element = ""
# element = "_offdiag01"
# element = "_offdiag02"
# element = "_offdiag12"
# element = "_Cg5g3"
element = "_Cg5g4"

if element == "_offdiag01":
    interProject1 = gamma(7) @ gamma(3)
    interProject2 = gamma(7)
elif element == "_offdiag02":
    interProject1 = gamma(7) @ gamma(4)
    interProject2 = gamma(7)
elif element == "_offdiag12":
    interProject1 = gamma(7) @ gamma(3)
    interProject2 = gamma(7) @ gamma(4)
elif element == "_Cg5g3":
    interProject1 = gamma(7) @ gamma(3)
    interProject2 = gamma(7) @ gamma(3)
elif element == "_Cg5g4":
    interProject1 = gamma(7) @ gamma(4)
    interProject2 = gamma(7) @ gamma(4)
else:
    interProject1 = gamma(7)
    interProject2 = gamma(7)

print("interpolator1 =   ", interProject1)
print("interpolator2 =   ", interProject2)
print("Project = %s" % (element))

interProject1_cupy = interProject1.to(device=device, dtype=torch.cfloat)
interProject2_cupy = interProject2.to(device=device, dtype=torch.cfloat)


# ==============================================================================
# Mom-smearing phase for eigenvectors
# ==============================================================================
phase_mom_smear = torch.tensor(
    [momsmear_phase, 0, 0], device=device, dtype=torch.float32
)
phase_factor_mom_smear = phase_calc_fast(phase_mom_smear, Nx, device)


# ==============================================================================
# I/O routines
# ==============================================================================
def readin_eigvecs(eig_dir, t, Nev1, conf_id, Nx):
    """
    Read eigenvectors and apply the mom-smearing phase.

    Main changes compared with the original version:
      1. Use Nev_file for file size, but reshape the sliced data with Nev1.
      2. Convert to complex64 before moving to GPU.
      3. Use the precomputed phase_factor_mom_smear.
    """
    filename = "%s/eigvecs_t%03d_%s" % (eig_dir, t, conf_id)

    file_size = os.path.getsize(filename)
    element_size = 8
    Nev_file = int(file_size / element_size / (Nx * Nx * Nx * 3 * 2))

    data = np.fromfile(filename, dtype=np.float64)
    data = data.reshape(Nev_file, Nx * Nx * Nx, 3, 2)
    data = data[:Nev1]

    eigvec = data[..., 0] + 1j * data[..., 1]
    eigvec = np.ascontiguousarray(eigvec.astype(np.complex64))

    eigvecs_gpu = torch.from_numpy(eigvec).to(device=device, dtype=torch.cfloat)

    # Apply mom-smearing phase.  Shape: (Nev1, Nx^3, 3)
    eigvecs_gpu = eigvecs_gpu * phase_factor_mom_smear[None, :, None]

    return eigvecs_gpu


def readin_VVVsink(VVV_dir, Nev1, Nt, conf_id, Px, Py, Pz, t_source):
    f = open(
        "%s/VVV.t%03i.Px%iPy%iPz%i.conf%s" % (VVV_dir, t_source, Px, Py, Pz, conf_id),
        "rb",
    )
    temp = np.fromfile(f, dtype="f8")
    f.close()

    Nev_file = int(np.cbrt(temp.size / 2))
    temp = temp.reshape(Nev_file, Nev_file, Nev_file, 2)
    temp = temp[..., 0] + temp[..., 1] * 1j
    temp = temp[:Nev1, :Nev1, :Nev1]

    return np.ascontiguousarray(temp.astype(np.complex64))


def readin_peram(peram_dir, conf_id, Nt, Nev1, t_source):
    """
    Read perambulator for one t_source.

    Main change compared with the original version:
      - replace repeated np.append by list + np.concatenate.
    """
    parts = []
    for d_source in range(4):
        filename = "%s/perams.%s.%i.%i" % (peram_dir, conf_id, d_source, t_source)
        parts.append(np.fromfile(filename, dtype=np.float64))

    peram = np.concatenate(parts, axis=0)

    peram_size = peram.size
    Nev_file = int(np.sqrt(peram_size / (4 * 4 * Nt * 2)))

    peram = peram.reshape(
        4, Nt, Nev_file, 4, Nev_file, 2
    )  # d_source, t_sink, ev_source, d_sink, ev_sink, complex
    peram = peram.transpose(
        1, 3, 0, 4, 2, 5
    )  # t_sink, d_sink, d_source, ev_sink, ev_source, complex

    peram = peram[..., 0] + 1j * peram[..., 1]
    peram = peram[:, :, :, :Nev1, :Nev1]
    peram = np.ascontiguousarray(peram.astype(np.complex64))

    peram_gpu = torch.from_numpy(peram).to(device=device, dtype=torch.cfloat)
    return peram_gpu


# ==============================================================================
# VVV calculation
# ==============================================================================
def VVV_Calc_cupy(Mom):
    st = time.time()

    Mom = torch.as_tensor(Mom, device=device, dtype=torch.float32)

    # Important optimization: calculate the sink momentum phase only once per Pz,
    # not once per t.
    phase_factor_cupy = phase_calc_fast(Mom, Nx, device)

    VVV_sink = torch.zeros((Nt, Nev1, Nev1, Nev1), device=device, dtype=torch.cfloat)

    for t in range(Nt):
        st1 = time.time()
        eigvecs_cupy = readin_eigvecs(eig_dir, t, Nev1, conf_id, Nx)
        gpu_sync()
        ed1 = time.time()
        print("Read-in eigenvector done, t %d, time used: %.3f s" % (t, ed1 - st1))

        st2 = time.time()

        # Use a local accumulator and assign once. This avoids repeatedly reading
        # and writing VVV_sink[t] inside the xi loop.
        VVV_t = torch.zeros((Nev1, Nev1, Nev1), device=device, dtype=torch.cfloat)

        for xi in range(Nx):
            sl = slice(xi * (Nx**2), (xi + 1) * (Nx**2))

            ph = phase_factor_cupy[sl]
            e0 = eigvecs_cupy[:, sl, 0]
            e1 = eigvecs_cupy[:, sl, 1]
            e2 = eigvecs_cupy[:, sl, 2]

            VVV_t = (
                VVV_t
                + contract("x,ax,bx,cx->abc", ph, e0, e1, e2)
                + contract("x,ax,bx,cx->abc", ph, e1, e2, e0)
                + contract("x,ax,bx,cx->abc", ph, e2, e0, e1)
                - contract("x,ax,bx,cx->abc", ph, e2, e1, e0)
                - contract("x,ax,bx,cx->abc", ph, e0, e2, e1)
                - contract("x,ax,bx,cx->abc", ph, e1, e0, e2)
            )

        VVV_sink[t] = VVV_t

        del eigvecs_cupy, VVV_t
        gpu_sync()
        ed2 = time.time()
        print("t %d VVV Contraction done, time used: %.3f s" % (t, ed2 - st2))

    gpu_sync()
    ed = time.time()
    print(
        "****************all VVV complete, time used: %.3f s****************"
        % (ed - st)
    )

    return VVV_sink


# ==============================================================================
# Main calculation
# ==============================================================================
Pzlist = [2, 3, 4, 5, 6]

for Pz in Pzlist:
    Mom = torch.tensor([Pz, Py, Px], device=device, dtype=torch.float32)
    print("Mom:", Mom)

    VVV_sink = VVV_Calc_cupy(Mom)

    contrac_nucl_matrix = torch.zeros((Nt, Nt, 4, 4), device=device, dtype=torch.cfloat)

    print("Contract start")
    st = time.time()

    for t_source in range(Nt):
        st0 = time.time()

        VVV_source = torch.conj(VVV_sink[t_source])

        peram_u = readin_peram(peram_u_dir, conf_id, Nt, Nev1, t_source)

        CG5peram_uCG5 = contract(
            "gh,thkbe,jk->tgjbe", interProject1_cupy, peram_u, interProject2_cupy
        )

        for t_sink in range(Nt):
            deltat = (t_sink - t_source + Nt) % Nt
            # if 10 <= deltat <= 40 and deltat % 2 == 0:
            if 2 <= deltat <= 30:
                contrac_nucl_matrix[t_sink, t_source] = contract(
                    "abc,gjad,gjbe,ilcf,def->il",
                    VVV_sink[t_sink],
                    peram_u[t_sink],
                    CG5peram_uCG5[t_sink],
                    peram_u[t_sink],
                    VVV_source,
                ) - contract(
                    "abc,glaf,gjbe,ijcd,def->il",
                    VVV_sink[t_sink],
                    peram_u[t_sink],
                    CG5peram_uCG5[t_sink],
                    peram_u[t_sink],
                    VVV_source,
                )

        del peram_u, CG5peram_uCG5, VVV_source
        gpu_sync()

        ed0 = time.time()
        print(
            "****************all complete for t_source %d, time used: %.3f s****************"
            % (t_source, ed0 - st0)
        )

    gpu_sync()
    ed_contract = time.time()
    print(
        "****************contract all t_source done, time used: %.3f s****************"
        % (ed_contract - st)
    )

    # --------------------------------------------------------------------------
    # Positive-parity projection.
    # We do this before saving contract_conf.npy to avoid CPU -> GPU re-upload.
    # --------------------------------------------------------------------------
    st0 = time.time()

    contrac_nucl_pp = contract("li,yxil->yx", matrix_pplus, contrac_nucl_matrix)

    # Original sign convention for positive parity:
    # if t_sink < t_source: contrac_nucl_pp[t_sink, t_source] *= -1
    idx = torch.arange(Nt, device=device)
    mask_pp = idx[:, None] < idx[None, :]
    contrac_nucl_pp = torch.where(mask_pp, -contrac_nucl_pp, contrac_nucl_pp)

    gpu_sync()
    ed0 = time.time()
    print(
        "****************pp projection on GPU done, time used: %.3f s****************"
        % (ed0 - st0)
    )

    # --------------------------------------------------------------------------
    # Save contract_conf.npy as requested.
    # --------------------------------------------------------------------------
    st_save = time.time()

    contrac_nucl_matrix_np = contrac_nucl_matrix.cpu().numpy()
    np.save(
        "%s/twopt_slice_pp_Px%sPy%sPz%s_eginphase%s%s_contract_conf%s.npy"
        % (corr_nucl_dir, Px, Py, Pz, mom_smear, element, conf_id),
        contrac_nucl_matrix_np,
    )

    contrac_nucl_pp_np = contrac_nucl_pp.cpu().numpy()
    np.save(
        "%s/twopt_slice_pp_Px%sPy%sPz%s_eginphase%s%s_nopol_ss_conf%s.npy"
        % (corr_nucl_dir, Px, Py, Pz, mom_smear, element, conf_id),
        contrac_nucl_pp_np,
    )

    ed_save = time.time()
    print(
        "****************save files done, time used: %.3f s****************"
        % (ed_save - st_save)
    )

    ed00 = time.time()
    print("All io and calculation done, time used: %.3f s" % (ed00 - st00))

    del VVV_sink
    del contrac_nucl_matrix
    del contrac_nucl_pp
    del contrac_nucl_matrix_np
    del contrac_nucl_pp_np

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(Pz, "succeed")

print("JOB: ran successfully")
