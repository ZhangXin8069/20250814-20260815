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
    if tmp[0] == "link_dir":
        link_dir = tmp[1]
    if tmp[0] == "pla_dir":
        pla_dir = tmp[1]
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

ops = np.zeros((Nt, Nx, Nx, delta_z), dtype=complex)
ops_mz = np.zeros((Nt, Nx, Nx, delta_z), dtype=complex)


# sysm = "_AS"
sysm = ""

if link_dir == "x":
    zdir = 0
    output_dir = output_dir + "/xdir/%s" % (conf_id)
elif link_dir == "y":
    zdir = 1
    output_dir = output_dir + "/ydir/%s" % (conf_id)
elif link_dir == "z":
    zdir = 2
    output_dir = output_dir + "/zdir/%s" % (conf_id)
else:
    raise ValueError(
        f"Invalid value for link_dir: {link_dir}. Expected 'x', 'y', or 'z'."
    )
print(f"link_dir: {link_dir}, zdir: {zdir}, output_dir: {output_dir}")

_mu = 0
_mu2 = 0
_nu = 0
_nu2 = 0

if rank == 0:
    _mu = 3
    _mu2 = 3
    _nu = (zdir + 1) % 3
    _nu2 = (zdir + 1) % 3

if rank == 1:
    _mu = 3
    _mu2 = 3
    _nu = (zdir + 2) % 3
    _nu2 = (zdir + 2) % 3

if rank == 2:
    _mu = (zdir + 1) % 3
    _mu2 = (zdir + 1) % 3
    _nu = (zdir + 2) % 3
    _nu2 = (zdir + 2) % 3

if rank in {0, 1, 2}:

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

    pla = plaquette_clover_all_new(gauge, Nt, Nx)
    ed = time.time()
    print("calculate plaquette done, time used: %.3f s" % (ed - st))

    print("calculate op at mu:%d, nu:%d start" % (_mu, _nu))
    # ------------------------------------------------------------------------------------------------

    ops = np.zeros((Nt, delta_z), dtype=complex)
    ops_mz = np.zeros((Nt, delta_z), dtype=complex)

    # ops = np.zeros((Nt, Nx, Nx, delta_z), dtype=complex)
    # ops_mz = np.zeros((Nt, Nx, Nx, delta_z), dtype=complex)

    # ------------------------------------------------------------------------------------------------

    st = time.time()
    for _dz in range(delta_z):
        print(_dz)
        ops[:, _dz] = operators_new_z0_mu2(
            gauge, zdir, pla, pla, _dz, _mu, _nu, _mu2, _nu2, Nt
        )
    ed = time.time()
    print("calculate op at mu:%d, nu:%d done, time used: %.3f s" % (_mu, _nu, ed - st))
    np.save(
        "%s/ops_mu%d_nu%d_dz%d_conf%s%s.npy"
        % (output_dir, _mu, _nu, delta_z, conf_id, sysm),
        ops,
    )
    # np.save(
    #     "%s/ops_minus_mu%d_nu%d_dz%d_conf%s%s.npz"
    #     % (output_dir, _mu, _nu, delta_z, conf_id, sysm),
    #     ops_mz,
    # )

    # ------------------------------------------------------------------------------------------------

    # st = time.time()
    # for _dz in range(delta_z):
    #     print(_dz)
    #     ops[:, :, :, _dz] = operators_new_z0_mu2_xy(
    #         gauge, zdir, pla, pla, _dz, _mu, _nu, _mu2, _nu2, Nt, Nx
    #     )
    #     # ops_mz[_dz] = operators_new_z0_mz_mu2(
    #     #     gauge, zdir, pla, pla_tilde, _dz, _mu, _nu, _mu2, _nu2, Nt
    #     # )
    # ed = time.time()
    # print("calculate op at mu:%d, nu:%d done, time used: %.3f s" % (_mu, _nu, ed - st))
    # np.save(
    #     "%s/ops_mu%d_nu%d_dz%d_conf%s%s.npy"
    #     % (output_dir, _mu, _nu, delta_z, conf_id, sysm),
    #     ops,
    # )
    # # np.save(
    # #     "%s/ops_minus_mu%d_nu%d_dz%d_conf%s%s.npz"
    # #     % (output_dir, _mu, _nu, delta_z, conf_id, sysm),
    # #     ops_mz,
    # # )
