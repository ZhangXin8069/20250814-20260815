#!/public/home/huangcl/.venv/bin/python
import gc
import resource

import matplotlib.pyplot as plt
import numpy as np

result_dir = "/public/home/zhangxin/lattice-pdf/examples/huangcl/result"


conf_name = "beta6.20_mu-0.2770_ms-0.2400_L24x72"
conf_short = "L24x72"
Nconf = 200  # test == False
Nt = 72
Nx = 24


Px = 0
Py = 0
Pz = 2

test = False
jack = True  # test == False
Nsample = 3000  # test == False && jack == False

if test:
    print("test")
    Nconf = 3
    jack = True

if jack:
    Nsample = Nconf

print("jackknife:", jack)
print("Nconf:", Nconf)
print("Nsample:", Nsample)


_corr = np.zeros((Nconf, Nt, Nt), complex)
_ope_01 = np.zeros((Nconf, Nx, Nt), complex)
_ope_30 = np.zeros((Nconf, Nx, Nt), complex)
_ope_31 = np.zeros((Nconf, Nx, Nt), complex)


########################################################################################
def sem(data, jack):
    error = data.std(0)
    if jack:
        error = error * np.sqrt(data.shape[0] - 1)
    return error


def resample(corr, jack, Nsample):
    # axis = 0 is conf index
    seed = 0
    n_conf = corr.shape[0]
    if jack:
        re_corr = (n_conf * corr.mean(0) - corr) / (n_conf - 1)
    else:
        rng = np.random.default_rng(seed=seed)
        idx = rng.integers(0, n_conf, size=(Nsample, n_conf))
        re_corr = corr[idx].mean(1)
    print(f"resample shape: {re_corr.shape}")
    return re_corr


########################################################################################


# load data
for i in range(Nconf):
    conf_id = 6250 + i * 200 # 6200 -> 6250
    _corr[i] = np.load(
        f"/public/group/lqcd/donghx/2pt_Result/{conf_name}/momsmear2z/{conf_id}/twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}_eginphase2_Cg5g4_nopol_ss_conf{conf_id}.npy"
    )
    _ope_01[i] = np.load(
        f"/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/{conf_short}/zdir/{conf_id}/ops_mu0_nu1_dz24_conf{conf_id}.npz"
    )["ops"]
    _ope_30[i] = np.load(
        f"/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/{conf_short}/zdir/{conf_id}/ops_mu3_nu0_dz24_conf{conf_id}.npz"
    )["ops"]
    _ope_31[i] = np.load(
        f"/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/{conf_short}/zdir/{conf_id}/ops_mu3_nu1_dz24_conf{conf_id}.npz"
    )["ops"]
print("load finish")
print("2pt shape:", _corr.shape)
print("ope shape:", _ope_01.shape)

_ope = -_ope_30 - _ope_31 + 2 * _ope_01
_ope = _ope.transpose(0, 2, 1)  # shape: Nconf, tau, z

max_t = 20
_corr2_rel = np.zeros((Nconf, Nt, max_t), dtype=complex)  # para: conf, ti(loop), dt
_ope_rel = np.zeros(
    (Nconf, Nt, max_t, Nx), dtype=complex
)  # para: conf, ti(loop), dtau, z

# loop ti
for ti in range(Nt):
    corr2_shift = np.roll(_corr[:, :, ti], shift=-ti, axis=1)
    _corr2_rel[:, ti, :] = corr2_shift[:, :max_t]

    ope_shift = np.roll(_ope, shift=-ti, axis=1)
    _ope_rel[:, ti, :, :] = ope_shift[:, :max_t, :]

_corr3 = np.zeros(
    (Nconf, Nt, max_t, max_t, Nx), dtype=complex
)  # para: conf, ti(loop),dt, dtau, z
for dt in range(max_t):
    for dtau in range(dt + 1):
        c2_slice = _corr2_rel[:, :, dt]  # para: conf, ti(loop)
        ope_slice = _ope_rel[:, :, dtau, :]  # para: conf. ti(loop), z
        c3_instance = ope_slice * c2_slice[:, :, np.newaxis]
        _corr3[:, :, dt, dtau, :] = c3_instance

del _corr, _ope, _ope_30, _ope_31, _ope_01
gc.collect()


corr2 = resample(_corr2_rel, jack, Nsample)  # para: sample, ti(loop), dt
ope = resample(_ope_rel, jack, Nsample)  # para: sample, ti(loop), dtau, z
corr3 = resample(_corr3, jack, Nsample)  # para: sample, ti(loop),dt, dtau, z

del _corr2_rel, _ope_rel, _corr3
gc.collect()

corr3_disc = (
    corr3 - corr2[:, :, :, np.newaxis, np.newaxis] * ope[:, :, np.newaxis, :, :]
)  # para: sample, ti(loop), dt, dtau, z
ratio = np.mean(
    (corr3_disc / corr2[:, :, :, np.newaxis, np.newaxis]), axis=1
)  # para: sample, dt, dtau, z

del corr3, corr2, ope, corr3_disc
gc.collect()

print("ratio finish")
print("ratio shape:", ratio.shape)

# ---------------------------------------------------------
# 假设的数据维度说明：
# ratio 形状为: (Nsample, dt_max, dtau_max, z_max)
# Nsample = ratio.shape[0]
# ---------------------------------------------------------

# 1. 设置你想绘制的特定 z 和 dt 范围
target_z = 2  # 示例: z = 2
dt_list = [4, 5, 6, 7, 8, 9, 10]  # 想画的 tsep (dt) 列表，比如 4 到 10

ratio_mean = ratio.mean(0)  # shape: (dt, dtau, z)
ratio_err = sem(ratio, jack)

# 3. 开始绘图
fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150)

# 颜色与标记配置
colors = [
    "#d3d3d3",
    "#f38152",
    "#4caf50",
    "#00bcd4",
    "#e65100",
    "#ffb300",
    "#757575",
]
markers = ["x", "x", "x", "x", "x", "x", "x"]

for i, dt in enumerate(dt_list):
    tau_vals = np.arange(0, dt + 1)

    # 横坐标: tau - dt / 2
    x_vals = tau_vals - dt / 2.0

    # 纵坐标: ratio 均值与误差
    # 提取 shape 对应 (dt, dtau, z) 中的点
    y_vals = ratio_mean[dt, tau_vals, target_z]
    y_errs = ratio_err[dt, tau_vals, target_z]

    color = colors[i % len(colors)]
    ax.errorbar(
        x_vals,
        y_vals,
        yerr=y_errs,
        fmt="x",
        color=color,
        ecolor=color,
        capsize=0,
        markersize=7,
        markeredgewidth=1.8,
        linewidth=1.2,
        label=f"z={target_z}, tsep={dt}",
    )


ax.set_title(
    f"Unpolarized, P({Px},{Py},{Pz}), z = {target_z}, Nconf={Nconf}, jackknife = {jack}",
    fontsize=14,
    pad=12,
)
ax.set_xlabel("t_ins - t_sep/2", fontsize=16, labelpad=8)
ax.set_ylabel("C3 / C2", fontsize=16, labelpad=8)
ax.set_xlim(-7, 7)
ax.set_ylim(-0.1, 1.2)
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(f"{result_dir}/ratio.png")


def get_peak_memory_gb():
    # maxrss 单位：Linux 下通常是 KB，macOS 下是 Bytes
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max_rss / (1024**2)  # 转换为 GB


print(f"Peak Memory: {get_peak_memory_gb():.3f} GB")

print("job finish")
