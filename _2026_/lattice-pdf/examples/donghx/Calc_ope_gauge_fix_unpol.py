##!/public/home/xinghy/anaconda3/bin/python
import numpy as np
import os
import time
import sys
import fileinput
from opt_einsum import contract
from mpi4py import MPI
from mpi4py.util import pkl5

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# from functions_cpu import *
from Operator import *

# read in global parameters
infile = fileinput.input()
for line in infile:
    tmp = line.split()
    if tmp[0] == "Nt":
        Nt = int(tmp[1])
    if tmp[0] == "Nx":
        Nx = int(tmp[1])
    if tmp[0] == "delta_z":
        delta_z = int(tmp[1])
    if tmp[0] == "conf_id":
        conf_id = tmp[1]
    if tmp[0] == "conf_file":
        conf_file = tmp[1]
    if tmp[0] == "output_dir":
        output_dir = tmp[1]
# Nt = 72
# Nx = 24
# mu = 1
# nu = 2
# delta_z = 15
# conf_id = 46000
# output_dir = "/public/home/donghx/Lattice/DeltaG/test_check"


# ------------------------------------------------------------------------------------------------
# mu,nv: 0(x),1(y),2(z),3(t)

ops = np.zeros((delta_z, Nt), dtype=complex)
ops_mz = np.zeros((delta_z, Nt), dtype=complex)

# sysm = "_AS"
sysm = ""

if rank == 0:
    _mu = 3
    _nu = 0
    _mu2 = 3
    _nu2 = 0

    st = time.time()
    # conf_file = "/public/home/donghx/Lattice/DeltaG/test_check/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_46000.lime.contents/msg02.rec04.ildg-binary-data"
    f = open("%s" % conf_file, "rb")
    gauge = np.fromfile(f, dtype=">f8")
    gauge = np.array(gauge)

    gauge = gauge.reshape(Nt, Nx, Nx, Nx, 4, 3, 3, 2)
    gauge = gauge[..., 0] + gauge[..., 1] * 1j
    print(gauge.shape)
    f.close()
    ed = time.time()
    print("read gauge done, time used: %.3f s" % (ed - st))

    pla = np.load("%s/ops_pla_conf%s.npz" % (output_dir, conf_id))["pla"]
    # pla_tilde = plaquette_clover_all_tilde(pla, Nt, Nx)
    pla_tilde = pla
    # print(pla_tilde.shape)
    ed = time.time()
    print("calculate plaquette done, time used: %.3f s" % (ed - st))

    print("calculate op at mu:%d, nu:%d start" % (_mu, _nu))
    st = time.time()
    for _dz in range(delta_z):
        print(_dz)
        ops[_dz] = operators_FF_z0(gauge, pla, pla, _dz, _mu, _nu, _mu2, _nu2)
        ops_mz[_dz] = operators_FF_z0_mz(gauge, pla, pla, _dz, _mu, _nu, _mu2, _nu2)
    ed = time.time()
    print("calculate op at mu:%d, nu:%d done, time used: %.3f s" % (_mu, _nu, ed - st))
    np.savez(
        "%s/ops_mu%d_nu%d_mu%d_nu%d_dz%d_conf%s%s.npz"
        % (output_dir, _mu, _nu, _mu2, _nu2, delta_z, conf_id, sysm),
        ops=ops,
    )
    np.savez(
        "%s/ops_minus_mu%d_nu%d_mu%d_nu%d_dz%d_conf%s%s.npz"
        % (output_dir, _mu, _nu, _mu2, _nu2, delta_z, conf_id, sysm),
        ops=ops_mz,
    )


if rank == 1:
    _mu = 3
    _nu = 1
    _mu2 = 3
    _nu2 = 1

    st = time.time()
    # conf_file = "/public/home/donghx/Lattice/DeltaG/test_check/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_46000.lime.contents/msg02.rec04.ildg-binary-data"
    f = open("%s" % conf_file, "rb")
    gauge = np.fromfile(f, dtype=">f8")
    gauge = np.array(gauge)

    gauge = gauge.reshape(Nt, Nx, Nx, Nx, 4, 3, 3, 2)
    gauge = gauge[..., 0] + gauge[..., 1] * 1j
    print(gauge.shape)
    f.close()
    ed = time.time()
    print("read gauge done, time used: %.3f s" % (ed - st))

    pla = np.load("%s/ops_pla_conf%s.npz" % (output_dir, conf_id))["pla"]
    # pla_tilde = plaquette_clover_all_tilde(pla, Nt, Nx)
    pla_tilde = pla
    # print(pla_tilde.shape)
    ed = time.time()
    print("calculate plaquette done, time used: %.3f s" % (ed - st))

    print("calculate op at mu:%d, nu:%d start" % (_mu, _nu))
    st = time.time()
    for _dz in range(delta_z):
        print(_dz)
        ops[_dz] = operators_FF_z0(gauge, pla, pla, _dz, _mu, _nu, _mu2, _nu2)
        ops_mz[_dz] = operators_FF_z0_mz(gauge, pla, pla, _dz, _mu, _nu, _mu2, _nu2)
    ed = time.time()
    print("calculate op at mu:%d, nu:%d done, time used: %.3f s" % (_mu, _nu, ed - st))
    np.savez(
        "%s/ops_mu%d_nu%d_mu%d_nu%d_dz%d_conf%s%s.npz"
        % (output_dir, _mu, _nu, _mu2, _nu2, delta_z, conf_id, sysm),
        ops=ops,
    )
    np.savez(
        "%s/ops_minus_mu%d_nu%d_mu%d_nu%d_dz%d_conf%s%s.npz"
        % (output_dir, _mu, _nu, _mu2, _nu2, delta_z, conf_id, sysm),
        ops=ops_mz,
    )


if rank == 2:
    _mu = 0
    _nu = 1
    _mu2 = 0
    _nu2 = 1

    st = time.time()
    # conf_file = "/public/home/donghx/Lattice/DeltaG/test_check/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_46000.lime.contents/msg02.rec04.ildg-binary-data"
    f = open("%s" % conf_file, "rb")
    gauge = np.fromfile(f, dtype=">f8")
    gauge = np.array(gauge)

    gauge = gauge.reshape(Nt, Nx, Nx, Nx, 4, 3, 3, 2)
    gauge = gauge[..., 0] + gauge[..., 1] * 1j
    print(gauge.shape)
    f.close()
    ed = time.time()
    print("read gauge done, time used: %.3f s" % (ed - st))

    pla = np.load("%s/ops_pla_conf%s.npz" % (output_dir, conf_id))["pla"]
    # pla_tilde = plaquette_clover_all_tilde(pla, Nt, Nx)
    pla_tilde = pla
    # print(pla_tilde.shape)
    ed = time.time()
    print("calculate plaquette done, time used: %.3f s" % (ed - st))

    print("calculate op at mu:%d, nu:%d start" % (_mu, _nu))
    st = time.time()
    for _dz in range(delta_z):
        print(_dz)
        ops[_dz] = operators_FF_z0(gauge, pla, pla, _dz, _mu, _nu, _mu2, _nu2)
        ops_mz[_dz] = operators_FF_z0_mz(gauge, pla, pla, _dz, _mu, _nu, _mu2, _nu2)
    ed = time.time()
    print("calculate op at mu:%d, nu:%d done, time used: %.3f s" % (_mu, _nu, ed - st))
    np.savez(
        "%s/ops_mu%d_nu%d_mu%d_nu%d_dz%d_conf%s%s.npz"
        % (output_dir, _mu, _nu, _mu2, _nu2, delta_z, conf_id, sysm),
        ops=ops,
    )
    np.savez(
        "%s/ops_minus_mu%d_nu%d_mu%d_nu%d_dz%d_conf%s%s.npz"
        % (output_dir, _mu, _nu, _mu2, _nu2, delta_z, conf_id, sysm),
        ops=ops_mz,
    )
