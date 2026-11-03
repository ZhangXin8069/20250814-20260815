import numpy as np
import cupy as cp
import math
import os
import sys
import fileinput
from gamma_matrix_cupy_DR import *
from input_output_4_cupy import *
from opt_einsum import contract
import time


Nt = 72
Nx = 24
conf_id = sys.argv[1]
Nev = 100
Nev1 = 100
Px = 0
Py = 0
Pz = 0
mom_smear = 2
momsmear_phase = -2
eig_dir = "/public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/%s/" % (
    conf_id
)


def phase_calc(Mom):
    phase_factor = np.zeros(Nx * Nx * Nx, dtype=complex)
    for z in range(0, Nx):
        for y in range(0, Nx):
            for x in range(0, Nx):
                Pos = np.array([z, y, x])
                phase_factor[z * Nx * Nx + y * Nx + x] = np.exp(
                    -np.dot(Mom, Pos) * 2 * np.pi * 1j / Nx
                )
    phase_factor_cupy = cp.asarray(phase_factor)
    return phase_factor_cupy


phase = np.array([0, 0, momsmear_phase], dtype=complex)
phase_factor = phase_calc(phase)


def readin_eigvecs(eig_dir, t, Nev1, conf_id, Nx):
    filename = f"%s/eigvecs_t%03d_%s" % (eig_dir, t, conf_id)
    with open(filename, "rb") as f:
        file_size = os.path.getsize(filename)
        element_size = 8
        Nev = int(file_size / element_size / (Nx * Nx * Nx * 3 * 2))
        # print("Nev222", Nev)
        data = np.fromfile(f, dtype="f8").reshape(Nev, Nx, Nx, Nx, 3, 2)
    Eigvec = data[..., 0] + 1j * data[..., 1]
    Eigvec = Eigvec[0:Nev1]
    # eigvecs = eigvecs.reshape(Nev1, Nx, Nx, Nx, Nc)
    eigvecs_cupy = cp.asarray(Eigvec)
    eigvecs_cupy = eigvecs_cupy.reshape(Nev, Nx * Nx * Nx, 3)
    # eigvecs_mom2 = contract("vxa,x->vxa", eigvecs_cupy, phase_factor)
    return eigvecs_cupy


def VVV_Calc_cupy(Mom):

    st = time.time()

    VVV_sink = cp.zeros((Nt, Nev1, Nev1, Nev1), dtype=complex)

    for t in range(0, Nt):
        st1 = time.time()
        eigvecs_cupy = readin_eigvecs(eig_dir, t, Nev1, conf_id, Nx)
        ed1 = time.time()
        print("Read-in eigenvector done , time used: %.3f s" % (ed1 - st1))

        st2 = time.time()
        phase_factor_cupy = phase_calc(Mom)
        for xi in range(
            0, Nx
        ):  # I did this becasue the intermediate array is too large for a single GPU to handle
            VVV_sink[t] = (
                VVV_sink[t]
                + contract(
                    "x,ax,bx,cx->abc",
                    phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                )
                + contract(
                    "x,ax,bx,cx->abc",
                    phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                )
                + contract(
                    "x,ax,bx,cx->abc",
                    phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                )
                - contract(
                    "x,ax,bx,cx->abc",
                    phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                )
                - contract(
                    "x,ax,bx,cx->abc",
                    phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                )
                - contract(
                    "x,ax,bx,cx->abc",
                    phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                )
            )
        ed2 = time.time()
        print("t %d VVV Contraction done , time used: %.3f s" % (t, ed2 - st2))
        # VVV.tofile('%s/eigvecs_t%03d_%s'%(output_dir,t,conf_id))
        # VVV_sink[t].tofile(
        #     "%s/VVV.t%03d.Px%dPy%dPz%d.conf%s" % (VVV_dir, t, Px, Py, Pz, conf_id)
        # )

    ed = time.time()
    print(
        "****************all complete , time used: %.3f s****************" % (ed - st)
    )

    return VVV_sink


for Pz in range(6):
    Mom = np.array([Pz, Py, Px])
    VVV_sink = VVV_Calc_cupy(Mom)
