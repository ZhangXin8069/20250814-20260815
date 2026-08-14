import numpy as np
import math
import os
import sys
import fileinput
import psutil
from mpi4py import MPI
from mpi4py.util import pkl5
from gamma_matrix import *
from input_output import *
from opt_einsum import contract
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
#     if tmp[0] == "VVV_dir":
#         VVV_dir = tmp[1]
#     if tmp[0] == "corr_nucl_dir":
#         corr_nucl_dir = tmp[1]


Nt = 96
Nx = 32
conf_id = sys.argv[1]
Nev = 100
Nev1 = 100
Px = 0
Py = 0
Pz = 0
mom_smear = 2
momsmear_phase = -2
eig_dir = "/public/group/lqcd/eigensystem/beta6.41_mu-0.2295_ms-0.2050_L32x96/%s/" % (
    conf_id
)
peram_u_dir = (
    "/public/home/sunp/sunpeng_2025_04_23/mom_smear_perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/output_dir_data/mz0_my-2_mx0/%s/"
    % (conf_id)
)
corr_nucl_dir = (
    "/public/group/lqcd/donghx/2pt_Result/beta6.41_mu-0.2295_ms-0.2050_L32x96_momsmear2y/%s/"
    % (conf_id)
)
VVV_dir = (
    "/public/group/lqcd/donghx/VVV/beta6.41_mu-0.2295_ms-0.2050_L32x96_momsmear2y/%s/"
    % (conf_id)
)

st00 = time.time()

# matrix_pplus = (
#     0.5 * (gamma(0) + gamma(4)) @ (1j * gamma(3) @ gamma(5))
# )  # positive parity projection
matrix_pplus = 0.5 * (gamma(0) + gamma(4))

# matrix_pminus = (
#     0.5 * (gamma(0) - gamma(4)) @ (1j * gamma(3) @ gamma(5))
# )  # negative parity projection
matrix_pminus = 0.5 * (gamma(0) - gamma(4))

corr_nucl_pp = np.zeros((1, Nt), dtype=complex)
corr_nucl_pm = np.zeros((1, Nt), dtype=complex)

# corr_nucl_dir = "/public/group/lqcd/donghx/2pt_cpu/Test_for_check"
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
if rank == 0:
    print("interpolator1 =   ", interProject1)
    print("interpolator2 =   ", interProject2)

print("Project = %s" % (element))

####################################################

# 0:x,1:y,2:z,3:t


##______________________________________________________________________
##______________________________________________________________________

# calculate the exp part


def phase_calc(Mom):
    phase_factor = np.zeros(Nx * Nx * Nx, dtype=complex)
    for z in range(0, Nx):
        for y in range(0, Nx):
            for x in range(0, Nx):
                Pos = np.array([z, y, x])
                phase_factor[z * Nx * Nx + y * Nx + x] = np.exp(
                    -np.dot(Mom, Pos) * 2 * np.pi * 1j / Nx
                )
    return phase_factor


# z y x
phase = np.array([0, momsmear_phase, 0], dtype=complex)
phase_factor = phase_calc(phase)


def readin_eigvecs(eig_dir, t, Nev1, conf_id, Nx):
    f = open("%s/eigvecs_t%03d_%s" % (eig_dir, t, conf_id), "rb")
    eigvecs = np.fromfile(f, dtype="f8")
    eigvecs_size = eigvecs.size
    Nev = int(eigvecs_size / (Nx * Nx * Nx * 3 * 2))
    eigvecs = eigvecs.reshape(Nev, Nx * Nx * Nx, 3, 2)
    eigvecs = eigvecs[..., 0] + eigvecs[..., 1] * 1j
    eigvecs = eigvecs[0:Nev1, :, :]
    # eigvecs = eigvecs.reshape(Nev1, Nx, Nx, Nx, Nc)
    eigvecs_mom2 = contract("vxa,x->vxa", eigvecs, phase_factor)
    return eigvecs_mom2


# def readin_eigvecs(eig_dir, t, Nev, Nev1, conf_id, Nx):
#     f = np.load("%s" % (eig_dir))
#     eigvecs = f[t, :, :, :, :, :]
#     eigvecs = eigvecs.reshape(Nev, Nx * Nx * Nx, 3)
#     eigvecs = eigvecs[Nev1:Nev-1, :, :]
#     return eigvecs


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

    # diag_indices = np.arange(Nev1)

    # for diag_index in diag_indices:
    #     start_index = max(diag_index - 20, 0)
    #     end_index = min(diag_index + 21, Nev1)
    #     peram[:, :, :, start_index:end_index, diag_index] = 0
    #     peram[:, :, :, diag_index, start_index:end_index] = 0

    # peram_cupy=cp.array(peram)
    return peram


Pzlist = [6]

for Py in Pzlist:

    Mom = np.array([Pz, Py, Px])

    print("Mom:", Mom)

    st = time.time()

    VVV_sink = np.zeros((Nt, Nev1, Nev1, Nev1), dtype=complex)

    if rank != 0:
        for t in range(
            int(Nt * (rank - 1) / (size - 1)), int(Nt * (rank) / (size - 1))
        ):
            st1 = time.time()
            eigvecs_numpy = readin_eigvecs(eig_dir, t, Nev1, conf_id, Nx)
            ed1 = time.time()
            print("Read-in eigenvector done , time used: %.3f s" % (ed1 - st1))

            st2 = time.time()
            phase_factor_numpy = phase_calc(Mom)
            for xi in range(
                0, Nx
            ):  # I did this becasue the intermediate array is too large for a single GPU to handle
                VVV_sink[t] = (
                    VVV_sink[t]
                    + contract(
                        "x,ax,bx,cx->abc",
                        phase_factor_numpy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                        eigvecs_numpy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                        eigvecs_numpy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                        eigvecs_numpy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                    )
                    + contract(
                        "x,ax,bx,cx->abc",
                        phase_factor_numpy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                        eigvecs_numpy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                        eigvecs_numpy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                        eigvecs_numpy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                    )
                    + contract(
                        "x,ax,bx,cx->abc",
                        phase_factor_numpy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                        eigvecs_numpy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                        eigvecs_numpy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                        eigvecs_numpy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                    )
                    - contract(
                        "x,ax,bx,cx->abc",
                        phase_factor_numpy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                        eigvecs_numpy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                        eigvecs_numpy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                        eigvecs_numpy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                    )
                    - contract(
                        "x,ax,bx,cx->abc",
                        phase_factor_numpy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                        eigvecs_numpy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                        eigvecs_numpy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                        eigvecs_numpy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                    )
                    - contract(
                        "x,ax,bx,cx->abc",
                        phase_factor_numpy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                        eigvecs_numpy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                        eigvecs_numpy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                        eigvecs_numpy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                    )
                )
            ed2 = time.time()
            print("t %d VVV Contraction done , time used: %.3f s" % (t, ed2 - st2))
            # VVV.tofile('%s/eigvecs_t%03d_%s'%(output_dir,t,conf_id))
            VVV_sink[t].tofile(
                "%s/VVV.t%03d.Px%dPy%dPz%d.conf%s" % (VVV_dir, t, Px, Py, Pz, conf_id)
            )

    ed = time.time()
    print(
        "****************all complete , time used: %.3f s****************" % (ed - st)
    )

    # del VVV_sink

    comm.Barrier()  ## wait for all complete

    # # del VVV_sink

    ##______________________________________________________________________
    ##______________________________________________________________________
    if rank != 0:
        VVV_sink = np.zeros((Nt, Nev1, Nev1, Nev1), dtype=complex)

        st_vvv = time.time()
        VVV_sink = readin_VVV_all_device(VVV_dir, Nev1, Nt, conf_id, Px, Py, Pz)
        ed_vvv = time.time()
        print("VVV read done, time used: %.3f s" % (ed_vvv - st_vvv))

    if rank == 0:
        print("Rank 0: Starting calculation...")  # 检查是否进入主逻辑

    ##______________________________________________________________________
    ##______________________________________________________________________

    # if rank == 0:
    #     st = time.time()
    #     peram_cpu_all = np.zeros((Nt, Nt, 4, 4, Nev1, Nev1), dtype=complex)
    #     for t_source in range(0, Nt):
    #         peram_temp = np.load(
    #             "%s/%s_peramb_t%s_mom2.npy" % (peram_u_dir, conf_id, t_source)
    #         )
    #         peram_cpu_all[t_source] = peram_temp[:, :, :, Nev1:Nev-1, Nev1:Nev-1]
    #     ed = time.time()
    #     print("read peram_all done, time used: %.3f s" % (ed - st))

    ##______________________________________________________________________
    ##______________________________________________________________________

    contrac_nucl_matrix = np.zeros((Nt, Nt, 4, 4), dtype=complex)

    print("Contract start")
    if rank != 0:
        st = time.time()
        for t_source in range(
            int(Nt * (rank - 1) / (size - 1)), int(Nt * (rank) / (size - 1))
        ):
            st0 = time.time()
            VVV_source = np.conj(VVV_sink[t_source])  # [t_source].T
            # peram_u = np.load(
            #     "%s/%s_peramb_t%s_mom%s.npy" % (peram_u_dir, conf_id, t_source, mom_smear)
            # )
            peram_u = readin_peram(peram_u_dir, conf_id, Nt, Nev1, t_source)

            CG5peram_uCG5 = contract(
                "gh,thkbe,jk->tgjbe", interProject1, peram_u, interProject2
            )
            for t_sink in range(0, Nt):
                deltat = (t_sink - t_source + Nt) % Nt
                if 2 <= deltat <= 48:
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

            # cp._default_memory_pool.free_all_blocks()
            # cp.cuda.Device().synchronize()

            ed0 = time.time()
            print(
                "****************all complete for t_source %d, time used: %.3f s****************"
                % (t_source, ed0 - st0)
            )
        ed = time.time()
        print("All calculation on CPU done, time used: %.3f s" % (ed - st))
        comm.Send(contrac_nucl_matrix, dest=0, tag=rank)
        print(f"Rank {rank} send successfully")
    print("Reached line 387 - before Barrier")

    if rank == 0:
        print("Rank0 Reached line 387 - before Barrier")
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_usage_mb = mem_info.rss / (1024**2)  # RSS (Resident Set Size) in MB

    print(f"Rank {rank}: Memory Usage = {mem_usage_mb:.2f} MB")

    # comm.Barrier()
    if rank == 0:
        print("Passed Barrier at line 387")

    if rank == 0:
        contrac_nucl_matrix_all = np.zeros((Nt, Nt, 4, 4), dtype=complex)
        for source_rank in range(1, size):
            comm.Recv(contrac_nucl_matrix, source=source_rank, tag=source_rank)
            contrac_nucl_matrix_all += contrac_nucl_matrix

        st0 = time.time()

        np.save(
            "%s/twopt_slice_pp_Px%sPy%sPz%s_eginphase%s%s_contract_conf%s.npy"
            % (corr_nucl_dir, Px, Py, Pz, mom_smear, element, conf_id),
            contrac_nucl_matrix_all,
        )

        contrac_nucl_pp = contract("li,yxil->yx", matrix_pplus, contrac_nucl_matrix_all)
        contrac_nucl_pm = contract(
            "li,yxil->yx", matrix_pminus, contrac_nucl_matrix_all
        )

        # corr_proton_matrix_fw=cp.asnumpy(corr_proton_matrix_fw)
        # corr_proton_matrix_re=cp.asnumpy(corr_proton_matrix_re)  #Returns an array on the host memory from an arbitrary source array.

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

        np.save(
            "%s/twopt_slice_pp_Px%sPy%sPz%s_eginphase%s%s_nopol_ss_conf%s.npy"
            % (corr_nucl_dir, Px, Py, Pz, mom_smear, element, conf_id),
            contrac_nucl_pp,
        )

        # np.save(
        #     "%s/twopt_slice_pm_Px%sPy%sPz%s_eginphase%s%s_nopol_ss_conf%s.npy"
        #     % (corr_nucl_dir, Px, Py, Pz, mom_smear, element, conf_id),
        #     contrac_nucl_pm,
        # )

        # corr_nucl_pp_np = cp.asnumpy(corr_nucl_pp)
        # corr_nucl_pm_np = cp.asnumpy(corr_nucl_pm)

        # write_data_ascii(
        #     corr_nucl_pp,
        #     Nt,
        #     Nx,
        #     "%s/corr_Nucleon_pp_Px%sPy%sPz%s_eginphase%s%s_nopol.conf%s.dat"
        #     % (corr_nucl_dir, Px, Py, Pz, mom_smear, element, conf_id),
        # )
        # write_data_ascii(
        #     corr_nucl_pm,
        #     Nt,
        #     Nx,
        #     "%s/corr_Nucleon_pm_Px%sPy%sPz%s_eginphase%s%s_nopol.conf%s.dat"
        #     % (corr_nucl_dir, Px, Py, Pz, mom_smear, element, conf_id),
        # )

        print("succeed")

    comm.Barrier()

print("JOB: ran successfully")
