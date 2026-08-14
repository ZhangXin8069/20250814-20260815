import numpy as np
import os
import time
import sys
import fileinput
import matplotlib
import itertools

# ope_dir = "/public/group/lqcd/donghx/Ope_Gluon/TMD_Ope_Gluon/Result/beta6.20_mu-0.2770_ms-0.2400_L24x72/zdir/"
# # sysm = ""
# sysm = "_AS"

# exist_list = []
# need_list = []


# # for conf in range(4050, 48000, 50):
# # for conf in range(11000, 60300, 50):
# # for conf in range(2700, 9860, 50):
# for conf in range(4050, 9050, 20):
#     if (
#         os.path.exists(
#             "%s/%s/ops_mu3_nu0_dz24_conf%s%s.npy" % (ope_dir, conf, conf, sysm)
#         )
#         # and os.path.exists(
#         #     "%s/%s/ops_mu3_nu1_dz24_conf%s%s.npy" % (ope_dir, conf, conf, sysm)
#         # )
#         # and os.path.exists(
#         #     "%s/%s/ops_mu1_nu2_dz24_conf%s%s.npy" % (ope_dir, conf, conf, sysm)
#         # )
#         # and
#         # os.path.exists("%s/%s/ops_pla_conf%s.npz" % (ope_dir, conf, conf))
#     ):
#         exist_list.append(conf)
#     else:
#         need_list.append(conf)

# for i in range(len(need_list)):
#     print(need_list[i])


ope_dir = "/public/group/lqcd/donghx/Ope_Gluon/TMD_Ope_Gluon/Result/beta6.20_mu-0.2770_ms-0.2400_L24x72/zdir/"
# sysm = ""
sysm = "_AS"

exist_list = []
need_list = []


# for conf in range(4050, 48000, 50):
# for conf in range(11000, 60300, 50):
# for conf in range(2700, 9860, 50):
for conf in range(4050, 48050, 50):
    if (
        # os.path.exists("%s/%s/ops_pla_conf%s.npz" % (ope_dir, conf, conf))
        # and
        os.path.exists(
            "%s/%s/ops_mu3_nu1_dz15_conf%s%s.npy" % (ope_dir, conf, conf, sysm)
        )
        # and os.path.exists(
        #     "%s/%s/ops_mu1_nu2_dz24_conf%s%s.npy" % (ope_dir, conf, conf, sysm)
        # )
        # and
        # os.path.exists("%s/%s/ops_pla_conf%s.npz" % (ope_dir, conf, conf))
    ):
        exist_list.append(conf)
    else:
        need_list.append(conf)

for i in range(len(need_list)):
    print(need_list[i])
