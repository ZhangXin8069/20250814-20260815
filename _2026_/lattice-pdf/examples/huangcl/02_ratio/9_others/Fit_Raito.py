import gvar as gv
import lsqfit
import numpy as np
import os
import time
import sys
import fileinput
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import matplotlib.ticker as ticker
from itertools import chain
from matplotlib.backends.backend_pdf import PdfPages

from prettytable import PrettyTable
import lsqfit
import gvar as gv


config = {
    # "font.family": "serif",
    # "font.size": 8,  # 相当于小四大小
    "mathtext.fontset": "stix",  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    "font.serif": ["SimHei"],  # 宋体
    "axes.unicode_minus": False,  # 处理负号，即-号
}

rcParams.update(config)


def jack(x):
    nsample = x.shape[0]
    y = np.zeros_like(x)
    sum = x[0]
    for n in range(1, nsample):
        sum = sum + x[n]
    for n in range(nsample):
        y[n] = sum - x[n]
        y[n] = y[n] / (nsample - 1)
    return y


def errLL_resam(a, resam):  # correlated case
    a = a.transpose(1, 0)
    N_j = a.shape[1]  # nsample
    N_p = a.shape[0]  # Nt
    x = np.mean(a, axis=1)  # Nt
    y = np.empty((N_p, N_j))
    for i in range(0, N_p):
        y[i] = np.array([x[i]] * N_j)  # N_j
    z = np.empty((N_p, N_p))
    for i in range(0, N_p):
        for j in range(0, N_p):
            if resam == 0:
                z[i][j] = (
                    sum((a - y)[i] * (a - y)[j]) / N_j * (N_j - 1)
                )  # jacknife error
            elif resam == 1:
                z[i][j] = sum((a - y)[i] * (a - y)[j]) / (N_j - 1)  # bootstrap error
    return z


def Para_to_array(P_A):
    center_values = np.array([value.mean for value in P_A.values()])  # 中心值
    errors = np.array([value.sdev for value in P_A.values()])  # 误差

    # 将中心值和误差合并为一个数组
    result_array = np.vstack((center_values, errors)).T  # 转置为 (N, 2) 形状

    return result_array


pzlist = [2]
delta_z = 24
Nt = 72
Nx = 24
element = "_Cg5g4"
Nconf = 879
t_sep = np.arange(5, 16)
ndt = t_sep.shape[0]
max_dt = t_sep.max()

# data_dir = "../Result"
# threept = np.mean(
#     [
#         np.load(f"{data_dir}/threept_{dir}_N{Nconf}.npy")
#         for dir in ["zdir", "ydir", "xdir"]
#     ],
#     axis=0,
# )
# Res_2pt = np.mean(
#     [np.load(f"{data_dir}/Res_2pt_{dir}.npy") for dir in ["zdir", "ydir", "xdir"]],
#     axis=0,
# )

# threept_jack = jack(threept)
# twopt_jack = jack(Res_2pt)

# ratio = np.zeros(
#     (Nconf, delta_z, ndt, Nt, max_dt + 1, len(pzlist)),
#     dtype=float,
# )

# for t_source in range(0, Nt):
#     for _dt in range(ndt):
#         dt = t_sep[_dt]
#         t_sink = (t_source + dt) % Nt
#         for conf in range(Nconf):
#             ratio[conf, :, _dt, t_source, :] = np.real(
#                 (
#                     threept_jack[conf, :, _dt, t_source, :, :]
#                     / twopt_jack[conf, t_sink, t_source]
#                 )
#             )

data_tsep9 = np.load("./ratio_ud_t9_data.npy")
data_tsep10 = np.load("./ratio_ud_t10_data.npy")
data_tsep11 = np.load("./ratio_ud_t11_data.npy")

data = np.zeros([48, 12, 3])
data[:, :10, 0] = data_tsep9
data[:, :11, 1] = data_tsep10
data[:, :12, 2] = data_tsep11
print(data_tsep9.shape)

temp_data = data.transpose(0, 2, 1)

ratio_ave_T = temp_data
Nconf = ratio_ave_T.shape[0]
ratio_diagram_ave = np.mean(ratio_ave_T, axis=0)
ratio_diagram_std = np.std(ratio_ave_T, axis=0) * np.sqrt(Nconf - 1)

print("ratio shape", ratio_ave_T.shape)

clr = [
    "#800000",
    "#5BC2E7",  # 浅蓝色
    "#FFC0CB",  # 粉红色
    "#FFD700",  # 金色
    "#34C759",  # 绿色
    "#FF6347",  # 番茄红
    "#8A2BE2",  # 蓝紫罗兰色
    "#FFA500",  # 橙色
    "#FF4500",  # 橙红色
    "#7CFC00",  # 春绿色
    "#483D8B",  # 深紫色
    "#FF69B4",  # 热粉红色
    "#00FFFF",  # 青色
]


def Fit_model(t_fit, p):
    tsep = t_fit[:, 0]
    tinsert = t_fit[:, 1]
    modle = p["A"] + p["B"] * (
        np.exp(-p["deltaE"] * (tsep - tinsert)) + np.exp(-p["deltaE"] * tinsert)
    )
    return modle


t_end = 11
t_start = 9
t_minus = 0

t_fit = np.array(
    [
        [t_sep_idx + t_start, tg_idx]
        for t_sep_idx in range(t_end - t_start + 1)
        # for tg_idx in range(t_sep_idx + t_start + 1)
        for tg_idx in range(1, t_sep_idx + t_start)
    ]
)

Ratio_fit = np.array(
    [
        ratio_ave_T[:, t_sep_idx + t_minus, tg_idx]
        for t_sep_idx in range(t_end - t_start + 1)
        # for tg_idx in range(t_sep_idx + t_start + 1)
        for tg_idx in range(1, t_sep_idx + t_start)
    ]
)

print(t_fit)

print("Ratio_fit = ", Ratio_fit.shape)
Ratio_fit = Ratio_fit.transpose(1, 0)
output_dir = "./"
outfile = "%s/Unpol_L%sx%s_conf%s_Fit_cov_C3C2_tg!=tss.pdf" % (
    output_dir,
    Nx,
    Nt,
    Nconf,
)
pdfplot = PdfPages(outfile)

chis = {}
P_A = {}
P_B = {}
P_deltaE = {}


ini_prr = {"A": "1(2.0)", "B": "0(2.0)", "deltaE": "0.5(2.0)"}
fit_parameter = np.zeros(3, dtype=float)

_dz = 0

ratio_ave = np.mean(
    Ratio_fit[:],
    axis=0,
)
ratio_cov = errLL_resam(Ratio_fit[:, :], 0)

data_fit = gv.gvar(ratio_ave, ratio_cov)

fit = lsqfit.nonlinear_fit(
    data=(t_fit, data_fit), fcn=Fit_model, prior=ini_prr, debug=True
)

chis[_dz] = fit.chi2 / (fit.dof - 3)
P_A[_dz] = fit.p["A"]
P_B[_dz] = fit.p["B"]
P_deltaE[_dz] = fit.p["deltaE"]

fig, ax = plt.subplots()

for t_sep_idx in range(t_end - t_start + 1):
    _dt = t_sep_idx + t_start
    X = np.arange(_dt + 1) - _dt / 2.0
    plt.errorbar(
        X - t_sep_idx * 0.07,
        np.real(ratio_diagram_ave[t_sep_idx + t_minus, 0 : _dt + 1]),
        np.real(ratio_diagram_std[t_sep_idx + t_minus, 0 : _dt + 1]),
        fmt="x" + "b",
        ecolor=clr[t_sep_idx],
        mec=clr[t_sep_idx],
        label=" tsep=%d" % (_dt),
    )

    t_ins_diag = np.linspace(0, _dt, 20)
    t_fit_diag = np.vstack((np.full_like(t_ins_diag, _dt), t_ins_diag)).T
    data_fit_fcn_gvar = fit.fcn(t_fit_diag, fit.p)
    data_fit_mean = np.array([para.mean for para in data_fit_fcn_gvar])
    data_fit_err = np.array([para.sdev for para in data_fit_fcn_gvar])

    plt.fill_between(
        t_ins_diag - _dt / 2.0,
        data_fit_mean - data_fit_err,
        data_fit_mean + data_fit_err,
        color=clr[t_sep_idx],
        alpha=0.2,
    )

X_plate = np.arange(-6, 7)

plt.fill_between(
    X_plate - 0.05,
    P_A[_dz].mean - P_A[_dz].sdev,
    P_A[_dz].mean + P_A[_dz].sdev,
    color=clr[2],
    alpha=0.5,
)
print(fit.format(True))
plt.xlabel(r"$t_i - t_f /2$", fontsize=15)
plt.ylabel(r"$C_3/C_2$", fontsize=15)
plt.title("Nconf=%d" % (Nconf))
plt.xlim([-7, 7])
plt.ylim(1.2, 1.4)
# plt.yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8])
# plt.axhline(y=0, color="grey", linestyle="--")
plt.legend(loc="upper left", fontsize=6, framealpha=0.7)
pdfplot.savefig(fig)
plt.close(fig)
pdfplot.close()
