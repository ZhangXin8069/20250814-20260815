import numpy as np
import math
import os
import sys
import fileinput
from gamma_matrix_DR_torch import *
from input_output_4_torch import *
from opt_einsum import contract
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Job started")
Nc = 3
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
#     if tmp[0] == "Nev1":  # number of eigenvectors used in contraction
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


Nt = 72
Nx = 24
conf_id = sys.argv[1]
Nev = 100
Nev1 = 100
Px = 0
Py = 0
Pz = 0
# mom_smear = 2
# momsmear_phase = -2
mom_smear = -2
momsmear_phase = 2
# mom_smear = 0
# momsmear_phase = 0
eig_dir = "/public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/%s/" % (
    conf_id
)
# peram_u_dir = (
#     "/public/home/sunp/sunpeng_2025_04_23/mom_smear_perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/output_dir_data/mz-2_my0_mx0/%s/"
#     % (conf_id)
# )
# corr_nucl_dir = (
#     "/public/group/lqcd/donghx/2pt_Result/beta6.20_mu-0.2770_ms-0.2400_L24x72_momsmear2z/%s/"
#     % (conf_id)
# )

peram_u_dir = (
    "/public/home/sunp/sunpeng_new_disk/mom_smear_perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/output_dir_data/mz2_my0_mx0/%s/"
    % (conf_id)
)
corr_nucl_dir = (
    "/public/group/lqcd/donghx/2pt_Result/beta6.20_mu-0.2770_ms-0.2400_L24x72/momsmear-2z/%s/"
    % (conf_id)
)

# peram_u_dir = (
#     "/public/group/lqcd/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/%s/"
#     % (conf_id)
# )
# corr_nucl_dir = "/public/group/lqcd/donghx/2pt_L24x72_momsmear0_Liu_Cg5gmu/%s/" % (
#     conf_id
# )

st00 = time.time()

matrix_pplus = 0.5 * (gamma(0) + gamma(4))  # positive parity projection
# matrix_pplus = 0.5 * (gamma(0) + gamma(4))

matrix_pminus = 0.5 * (gamma(0) - gamma(4))  # negative parity projection

matrix_pplus = matrix_pplus.to(dtype=torch.cfloat)
matrix_pminus = matrix_pminus.to(dtype=torch.cfloat)
# print("Pol = g3g5")

corr_nucl_pp = torch.zeros((1, Nt), device=device, dtype=torch.cfloat)
corr_nucl_pm = torch.zeros((1, Nt), device=device, dtype=torch.cfloat)

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

interProject1_cupy = interProject1.to(dtype=torch.cfloat)
interProject2_cupy = interProject2.to(dtype=torch.cfloat)

##______________________________________________________________________
##______________________________________________________________________

# calculate the exp part


def phase_calc(Mom):
    phase_factor = torch.zeros(Nx * Nx * Nx, device=device, dtype=torch.cfloat)
    for z in range(0, Nx):
        for y in range(0, Nx):
            for x in range(0, Nx):
                Pos = torch.tensor([z, y, x], dtype=torch.float32)
                phase_factor[z * Nx * Nx + y * Nx + x] = torch.exp(
                    -torch.dot(Mom, Pos) * 2 * torch.pi * 1j / Nx
                )
    return phase_factor


phase = torch.tensor([momsmear_phase, 0, 0], dtype=torch.float32)
phase_factor = phase_calc(phase)
phase_factor = phase_factor.cuda()


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
    eigvecs_cupy = torch.from_numpy(Eigvec).to(dtype=torch.cfloat)
    eigvecs_cupy = eigvecs_cupy.cuda()
    eigvecs_cupy = eigvecs_cupy.reshape(Nev, Nx * Nx * Nx, 3)
    # print(eigvecs_cupy.shape)
    eigvecs_mom2 = contract("vxa,x->vxa", eigvecs_cupy, phase_factor)
    return eigvecs_mom2


def readin_VVVsink(VVV_dir, Nev1, Nt, conf_id, Px, Py, Pz, t_source):
    VVV = np.zeros((Nev1, Nev1, Nev1), dtype=complex)
    f = open(
        "%s/VVV.t%03i.Px%iPy%iPz%i.conf%s" % (VVV_dir, t_source, Px, Py, Pz, conf_id),
        "rb",
    )
    temp = np.fromfile(f, dtype="f8")
    Nev = int(np.cbrt(temp.size / 2))
    temp = temp.reshape(Nev, Nev, Nev, 2)
    temp = temp[..., 0] + temp[..., 1] * 1j
    temp = temp[0:Nev1, 0:Nev1, 0:Nev1]
    VVV = np.copy(temp)
    f.close()
    return VVV


def readin_peram(peram_dir, conf_id, Nt, Nev1, t_source):
    f = open("%s/perams.%s.0.%i" % (peram_dir, conf_id, t_source), "rb")
    peram = np.fromfile(f, dtype="f8")
    f.close()

    for d_source in range(1, 4):
        f = open("%s/perams.%s.%i.%i" % (peram_dir, conf_id, d_source, t_source), "rb")
        temp = np.fromfile(f, dtype="f8")
        peram = np.append(peram, temp)
        temp = None
        f.close()
    peram_size = peram.size
    Nev = int(np.sqrt(peram_size / (4 * 4 * Nt * 2)))
    peram = peram.reshape(
        4, Nt, Nev, 4, Nev, 2
    )  # d_source, t_sink, ev_source, d_sink, ev_sink, complex
    peram = peram.transpose(
        1, 3, 0, 4, 2, 5
    )  # t_sink, d_sink, d_source, ev_sink, ev_souce,  complex
    peram = peram[..., 0] + peram[..., 1] * 1j
    peram = peram[:, :, :, 0:Nev1, 0:Nev1]

    peram = peram.astype(np.complex64)
    peram_cupy = torch.from_numpy(peram).to(dtype=torch.cfloat)
    peram_cupy = peram_cupy.cuda()
    return peram_cupy


Mom = torch.tensor([Pz, Py, Px], dtype=torch.float32)

print("Mom:", Mom)


def VVV_Calc_cupy(Mom):

    st = time.time()

    VVV_sink = torch.zeros((Nt, Nev1, Nev1, Nev1), device=device, dtype=torch.cfloat)

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


##______________________________________________________________________
##______________________________________________________________________

# st_vvv = time.time()
# VVV_sink = readin_VVV_all_device(VVV_dir, Nev, Nt, conf_id, Px, Py, Pz)
# ed_vvv = time.time()
# print("VVV read done, time used: %.3f s" % (ed_vvv - st_vvv))


##______________________________________________________________________
##______________________________________________________________________

# st = time.time()

# peram_cpu_all = np.zeros((Nt, Nt, 4, 4, Nev1, Nev1), dtype=complex)

# for t_source in range(0, Nt):
#     peram_temp = np.load(
#         "%s/%s_peramb_t%s_mom%s.npy" % (peram_u_dir, conf_id, t_source, mom_smear)
#     )
#     peram_cpu_all[t_source] = peram_temp[:, :, :, 0:Nev1, 0:Nev1]

# ed = time.time()
# print("read peram_all done, time used: %.3f s" % (ed - st))

##______________________________________________________________________
##______________________________________________________________________

Pzlist = [-2, -3, -4, -5, -6]

for Pz in Pzlist:
    Mom = torch.tensor([Pz, Py, Px], dtype=torch.float32)
    print("Mom:", Mom)
    VVV_sink = VVV_Calc_cupy(Mom)

    contrac_nucl_matrix = torch.zeros((Nt, Nt, 4, 4), device=device, dtype=torch.cfloat)

    print("Contract start")
    st = time.time()
    for t_source in range(0, Nt):

        st0 = time.time()
        VVV_source = torch.conj(VVV_sink[t_source])  # [t_source].T
        # peram_u = torch.asarray(peram_cpu_all[t_source])
        peram_u = readin_peram(peram_u_dir, conf_id, Nt, Nev1, t_source)
        CG5peram_uCG5 = contract(
            "gh,thkbe,jk->tgjbe", interProject1_cupy, peram_u, interProject2_cupy
        )
        for t_sink in range(0, Nt):
            deltat = (t_sink - t_source + Nt) % Nt
            # if 10 <= deltat <= 40 and deltat % 2 == 0:
            if 2 <= deltat <= 32:
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

        # del peram_u

        torch.cuda.synchronize()

        ed0 = time.time()
        print(
            "****************all complete for t_source %d, time used: %.3f s****************"
            % (t_source, ed0 - st0)
        )

    st0 = time.time()

    contrac_nucl_matrix = contrac_nucl_matrix.cpu().numpy()

    np.save(
        "%s/twopt_slice_pp_Px%sPy%sPz%s_eginphase%s%s_contract_conf%s.npy"
        % (corr_nucl_dir, Px, Py, Pz, mom_smear, element, conf_id),
        contrac_nucl_matrix,
    )

    contrac_nucl_matrix = torch.from_numpy(contrac_nucl_matrix).to("cuda")

    contrac_nucl_pp = contract("li,yxil->yx", matrix_pplus, contrac_nucl_matrix)
    contrac_nucl_pm = contract("li,yxil->yx", matrix_pminus, contrac_nucl_matrix)

    # corr_proton_matrix_fw=torch.asnumpy(corr_proton_matrix_fw)
    # corr_proton_matrix_re=torch.asnumpy(corr_proton_matrix_re)  #Returns an array on the host memory from an arbitrary source array.

    for t_source in range(0, Nt):
        for t_sink in range(0, Nt):
            if t_sink < t_source:
                contrac_nucl_pp[t_sink, t_source] = (
                    -1.0 * contrac_nucl_pp[t_sink, t_source]
                )

            if t_sink > t_source:
                contrac_nucl_pm[t_sink, t_source] = (
                    -1.0 * contrac_nucl_pm[t_sink, t_source]
                )

            # corr_nucl_pp[0, (t_sink - t_source + Nt) % Nt] = (
            #     corr_nucl_pp[0, (t_sink - t_source + Nt) % Nt]
            #     + contrac_nucl_pp[t_sink, t_source]
            # )
            # corr_nucl_pm[0, (t_sink - t_source + Nt) % Nt] = (
            #     corr_nucl_pm[0, (t_sink - t_source + Nt) % Nt]
            #     + contrac_nucl_pm[t_sink, t_source]
            # )
    ed0 = time.time()
    print(
        "****************final calculation on GPU done, time used: %.3f s****************"
        % (ed0 - st0)
    )

    ed00 = time.time()
    print("All io and calculation done, time used: %.3f s" % (ed00 - st00))

    contrac_nucl_pp = contrac_nucl_pp.cpu().numpy()

    np.save(
        "%s/twopt_slice_pp_Px%sPy%sPz%s_eginphase%s%s_nopol_ss_conf%s.npy"
        % (corr_nucl_dir, Px, Py, Pz, mom_smear, element, conf_id),
        contrac_nucl_pp,
    )

    del VVV_sink
    del VVV_source
    # torch.save(
    #     "%s/twopt_slice_pm_Px%sPy%sPz%s_eginphase%s%s_nopol_ss_conf%s.npy"
    #     % (corr_nucl_dir, Px, Py, Pz, mom_smear, element, conf_id),
    #     contrac_nucl_pm,
    # )

    # corr_nucl_pp_np = torch.asnumpy(corr_nucl_pp)
    # corr_nucl_pm_np = torch.asnumpy(corr_nucl_pm)

    # write_data_ascii(
    #     corr_nucl_pp_np,
    #     Nt,
    #     Nx,
    #     "%s/corr_Nucleon_pp_Px%sPy%sPz%s_eginphase%s%s_nopol.conf%s.dat"
    #     % (corr_nucl_dir, Px, Py, Pz, mom_smear, element, conf_id),
    # )
    # write_data_ascii(
    #     corr_nucl_pm_np,
    #     Nt,
    #     Nx,
    #     "%s/corr_Nucleon_pm_Px%sPy%sPz%s_eginphase%s%s_nopol.conf%s.dat"
    #     % (corr_nucl_dir, Px, Py, Pz, mom_smear, element, conf_id),
    # )

    print(Pz, "succeed")

print("JOB: ran successfully")
