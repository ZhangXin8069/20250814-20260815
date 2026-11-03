import numpy as np
import os
import time
import sys
import fileinput
from opt_einsum import contract


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
    if tmp[0] == "pla_dir":
        output_dir = tmp[1]

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
## t z y x 4(xyzt) 3(row) 3(col)


# ------------------------------------------------------------------------------------------------

st = time.time()
pla = plaquette_clover_all_new(gauge, Nt, Nx)
# print(pla.shape)
np.savez(
    "%s/ops_pla_conf%s.npz" % (output_dir, conf_id),
    pla=pla,
)
