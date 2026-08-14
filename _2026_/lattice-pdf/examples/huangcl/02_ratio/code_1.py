#!/public/home/huangcl/.venv/bin/python
import argparse
import gc
import os
import resource
import sys
import time
from dataclasses import dataclass, field

from prettytable import PrettyTable
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import numpy as np

# ===== 独立开关，方便调试时修改 =====
debug = False  # 在登录节点跑, 方便排除错误, 结果输出到 0_debug 文件夹
jack = True  # debug == False
# ===================================


# ===== 定义四个 dataclass =====

@dataclass
class SampleParams:
    conf_short: str
    conf_name: str
    Nconf: int
    Nt: int
    Nx: int
    Px: int
    Py: int
    Pz: int
    Nsample: int
    dt_max: int


@dataclass
class FitParams:
    p0: dict
    dt_start: int
    dt_end: int
    cut: int


@dataclass
class PlotParams:
    plot_z: int
    dt_list: list[int]
    z_list: list[int]
    xlim: list[float]
    ylim: list[float]
    c0_ylim: list[float]  # c0 vs z 图的纵轴范围
    colors: list[str] = field(default_factory=lambda: [
        "#b3d9ff",  # 1 very light blue (浅蓝)
        "#c7e9c0",  # 2 very light green (浅绿)
        "#fdd49e",  # 3 light orange (浅橙)
        "#d4b9da",  # 4 light purple (浅紫)
        "#99d8c9",  # 5 light teal (浅青)
        "#6baed6",  # 6 medium blue (中蓝)
        "#41ab5d",  # 7 medium green (中绿)
        "#ef6548",  # 8 medium red (中红)
        "#8856a7",  # 9 medium purple (中紫)
        "#08306b",  # 10 dark navy (深藏青)
    ])


@dataclass
class OutputParams:
    """路径与输出文件名，在 if 块外用 conf_short 自动构造"""
    result_dir: str   # ratio 存放目录: result/L24x72/
    fit_dir: str      # fit + 图片存放目录: result/L24x72/fit_Pz*_Nsam*_.../
    ratio_file: str   # ratio 数组保存文件名（带 Pz）
    fit_file: str     # fit 结果文件名（简洁，因参数在路径中）
    report_file: str  # fit 报告文件名（简洁）


# ===== 从命令行参数读取 conf_short =====
parser = argparse.ArgumentParser(
    description="Gluon unpolarized PDF ratio calculation")
parser.add_argument("-c", type=str, default="L24x72",
                    help="conf_short, e.g. L24x72 (default: L24x72)")
parser.add_argument("-s", type=int, default=1,
                    choices=[1, 2, 3], help="start part: 1=ratio, 2=fit, 3=plot (default: 1)")
parser.add_argument("-e", type=int, default=3,
                    choices=[1, 2, 3], help="end part: 1=ratio, 2=fit, 3=plot (default: 3)")
args = parser.parse_args()
conf_short = args.c
part_start = args.s
part_end = args.e
# =========================================


if conf_short == "L24x72":
    # ===== 一次性构造三个 dataclass =====
    sampa = SampleParams(
        conf_short="L24x72",
        conf_name="beta6.20_mu-0.2770_ms-0.2400_L24x72",
        Nconf=200,
        Nt=72,
        Nx=24,
        Px=0,
        Py=0,
        Pz=2,
        Nsample=1000,
        dt_max=20,
    )

    fitpa = FitParams(
        p0={"c0": 0.6, "c1": -2, "dE": 1},
        dt_start=7,
        dt_end=10,
        cut=6,
    )

    plotpa = PlotParams(
        plot_z=2,
        dt_list=list(range(6, 13)),
        z_list=list(range(7, 13)),
        xlim=[-7, 7],
        ylim=[-0.1, 0.5],
        c0_ylim=[-1, 1],
    )

else:
    print(f"conf {conf_short} not exist.")
    sys.exit()

# 确保 dt_list 中的最大值不超过 dt_max - 1（ratio 数组的 axis 0 大小为 dt_max）
dt_max_valid = sampa.dt_max - 1
if plotpa.dt_list[-1] > dt_max_valid:
    print(
        f"Warning: dt_list max {plotpa.dt_list[-1]} exceeds dt_max-1={dt_max_valid}, truncating")
    plotpa.dt_list = [dt for dt in plotpa.dt_list if dt <= dt_max_valid]


# debug / jack 调整
if debug:
    print("debug")
    sampa.Nconf = 5
    jack = True

if jack:
    sampa.Nsample = sampa.Nconf


# ===== OutputParams 在 if 块外用 conf_short 自动构造 =====
_base_dir = "0_debug" if debug else "1_result"
_ratio_dir = os.path.join(os.getcwd(), _base_dir, conf_short)
_fit_dir_name = (
    f"fit_Pz{sampa.Pz}_Nsam{sampa.Nsample}_dtmax{sampa.dt_max}"
    f"_tsep{fitpa.dt_start}_{fitpa.dt_end}_cut{fitpa.cut}"
)
_fit_dir = os.path.join(_ratio_dir, _fit_dir_name)
outpa = OutputParams(
    result_dir=_ratio_dir,
    fit_dir=_fit_dir,
    ratio_file=f"ratio_Pz{sampa.Pz}_Nsam{sampa.Nsample}_dtmax{sampa.dt_max}.npy",
    fit_file="0_fit_data.npz",
    report_file="1_fit_report.txt",
)

# 创建结果目录
os.makedirs(outpa.result_dir, exist_ok=True)
os.makedirs(outpa.fit_dir, exist_ok=True)
# =========================================================


def sem(data, jackknife):
    error = data.std(0)
    if jackknife:
        error = error * np.sqrt(data.shape[0] - 1)
    return error


def resample(corr, jackknife, Nsample):
    # axis = 0 is conf index
    seed = 0
    n_conf = corr.shape[0]
    if jackknife:
        re_corr = (n_conf * corr.mean(0) - corr) / (n_conf - 1)
    else:
        rng = np.random.default_rng(seed=seed)
        idx = rng.integers(0, n_conf, size=(Nsample, n_conf))
        re_corr = corr[idx].mean(1)
    print(f"resample shape: {re_corr.shape}")
    return re_corr


def cov_mat(arr, jackknife):
    # 2-dimensional array, axis0 = sample, axis1 = dependent variables
    diff = arr - arr.mean(0)
    n = arr.shape[0]
    if jackknife:
        cov = np.matmul(diff.T, diff) / n * (n - 1)
    else:
        cov = np.matmul(diff.T, diff) / n
    eig = np.linalg.eigvalsh(cov)
    cond = eig[-1] / eig[0]
    return cov, cond
########################################################################################


def compute_ratio(sampa: SampleParams, outpa: OutputParams, jack: bool):
    """加载数据并计算 ratio 数组"""
    print("==================== compute ratio start ====================")

    _corr = np.zeros((sampa.Nconf, sampa.Nt, sampa.Nt), dtype=complex)
    _ope_01 = np.zeros((sampa.Nconf, sampa.Nx, sampa.Nt), dtype=complex)
    _ope_30 = np.zeros((sampa.Nconf, sampa.Nx, sampa.Nt), dtype=complex)
    _ope_31 = np.zeros((sampa.Nconf, sampa.Nx, sampa.Nt), dtype=complex)

    # load data
    for i in range(sampa.Nconf):
        conf_id = 6200 + i * 200
        _corr[i] = np.load(
            f"/public/group/lqcd/donghx/2pt_Result/{sampa.conf_name}/momsmear2z/{conf_id}/twopt_slice_pp_Px{sampa.Px}Py{sampa.Py}Pz{sampa.Pz}_eginphase2_Cg5g4_nopol_ss_conf{conf_id}.npy"
        )
        _ope_01[i] = np.load(
            f"/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/{sampa.conf_short}/zdir/{conf_id}/ops_mu0_nu1_dz24_conf{conf_id}.npz"
        )["ops"]
        _ope_30[i] = np.load(
            f"/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/{sampa.conf_short}/zdir/{conf_id}/ops_mu3_nu0_dz24_conf{conf_id}.npz"
        )["ops"]
        _ope_31[i] = np.load(
            f"/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/{sampa.conf_short}/zdir/{conf_id}/ops_mu3_nu1_dz24_conf{conf_id}.npz"
        )["ops"]
    print("load finish")
    print("2pt shape:", _corr.shape)
    print("ope shape:", _ope_01.shape)

    _ope = -_ope_30 - _ope_31 + 2 * _ope_01
    _ope = _ope.transpose(0, 2, 1)  # shape: Nconf, tau, z

    # para: conf, ti(loop), dt
    _corr2_rel = np.zeros((sampa.Nconf, sampa.Nt, sampa.dt_max), dtype=complex)
    _ope_rel = np.zeros(
        (sampa.Nconf, sampa.Nt, sampa.dt_max, sampa.Nx), dtype=complex
    )  # para: conf, ti(loop), dtau, z

    # loop ti
    for ti in range(sampa.Nt):
        corr2_shift = np.roll(_corr[:, :, ti], shift=-ti, axis=1)
        _corr2_rel[:, ti, :] = corr2_shift[:, :sampa.dt_max]

        ope_shift = np.roll(_ope, shift=-ti, axis=1)
        _ope_rel[:, ti, :, :] = ope_shift[:, :sampa.dt_max, :]

    _corr3 = np.zeros(
        (sampa.Nconf, sampa.Nt, sampa.dt_max,
         sampa.dt_max, sampa.Nx), dtype=complex
    )  # para: conf, ti(loop),dt, dtau, z
    for _dt in range(sampa.dt_max):
        for _dtau in range(_dt + 1):
            c2_slice = _corr2_rel[:, :, _dt]  # para: conf, ti(loop)
            ope_slice = _ope_rel[:, :, _dtau, :]  # para: conf. ti(loop), z
            c3_instance = ope_slice * c2_slice[:, :, np.newaxis]
            _corr3[:, :, _dt, _dtau, :] = c3_instance

    del _corr, _ope, _ope_30, _ope_31, _ope_01
    gc.collect()

    # para: sample, ti(loop), dt
    corr2 = resample(_corr2_rel, jack, sampa.Nsample)
    # para: sample, ti(loop), dtau, z
    ope = resample(_ope_rel, jack, sampa.Nsample)
    # para: sample, ti(loop),dt, dtau, z
    corr3 = resample(_corr3, jack, sampa.Nsample)

    del _corr2_rel, _ope_rel, _corr3
    gc.collect()

    corr3_disc = (
        corr3 - corr2[:, :, :, np.newaxis, np.newaxis] *
        ope[:, :, np.newaxis, :, :]
    )  # para: sample, ti(loop), dt, dtau, z
    ratio = np.mean(
        (corr3_disc / corr2[:, :, :, np.newaxis, np.newaxis]), axis=1
    )  # para: sample, dt, dtau, z

    ratio = ratio.real

    del corr3, corr2, ope, corr3_disc
    gc.collect()
    print("ratio shape:", ratio.shape)

    # 保存 ratio 数组
    np.save(f"{outpa.result_dir}/{outpa.ratio_file}", ratio)
    print(f"ratio saved to {outpa.result_dir}/{outpa.ratio_file}")
    print("==================== compute ratio end ====================")

    return ratio


def get_peak_memory_gb():
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max_rss / (1024**2)


def model(x, p):
    dt = np.array([_x[0] for _x in x])
    dtau = np.array([_x[1] for _x in x])

    return (np.ones(len(x)) * p["c0"]
            + p["c1"] * np.exp(-p["dE"] * dtau)
            + p["c1"] * np.exp(-p["dE"] * (dt - dtau)))


def do_fit(ratio: np.ndarray, fitpa: FitParams, sampa: SampleParams, outpa: OutputParams, jack: bool):
    """执行拟合，返回字典 {c0, c1, dE, chi2}"""
    print("==================== do_fit start ====================")

    x_coor = []
    front_remove = fitpa.cut // 2
    back_remove = fitpa.cut - front_remove
    for dt in range(fitpa.dt_start, fitpa.dt_end+1):
        for dtau in range(front_remove, dt - back_remove + 1):
            x_coor.append((dt, dtau))
    Ndata = len(x_coor)

    para_c0 = np.zeros((sampa.Nsample, sampa.Nx))
    para_c1 = np.zeros_like(para_c0)
    para_dE = np.zeros_like(para_c0)
    chi2 = np.zeros_like(para_c0)

    # ---- 准备报告文件 ----
    report_file = outpa.report_file
    report_lines = []
    sep_line = "=" * 72
    report_lines.append(sep_line)
    report_lines.append(f"  Fit Report: {sampa.conf_short}")
    report_lines.append(sep_line)
    report_lines.append(f"  t_sep range : [{fitpa.dt_start}, {fitpa.dt_end}]")
    report_lines.append(f"  cut         : {fitpa.cut}")
    report_lines.append(f"  Nsample     : {sampa.Nsample}")
    report_lines.append(f"  jackknife   : {jack}")
    report_lines.append(sep_line)
    report_lines.append("")

    for _z in range(sampa.Nx):
        t0_fit = time.perf_counter()
        sub_sample = np.zeros((sampa.Nsample, Ndata))
        for i, (dt, dtau) in enumerate(x_coor):
            sub_sample[:, i] = ratio[:, dt, dtau, _z]

        cov, cond = cov_mat(sub_sample, jack)

        for _id in range(sampa.Nsample):
            y_coor = gv.gvar(sub_sample[_id], cov)
            # y_coor = gv.gvar(sub_sample[_id], np.diag(np.diag(cov)))
            _fit = lsqfit.nonlinear_fit(
                data=(x_coor, y_coor), p0=fitpa.p0, fcn=model, svdcut=1e-6
            )
            para_c0[_id, _z] = _fit.pmean["c0"]
            para_c1[_id, _z] = _fit.pmean["c1"]
            para_dE[_id, _z] = _fit.pmean["dE"]
            chi2[_id, _z] = _fit.chi2 / _fit.dof

        # ---- stdout 精简输出 ----
        print(f'z={_z}')
        print(
            f'c0 = {para_c0[:, _z].mean():.3g} +- {sem(para_c0[:, _z], jack):.3g}')
        print(
            f'c1 = {para_c1[:, _z].mean():.3g} +- {sem(para_c1[:, _z], jack):.3g}')
        print(
            f'dE = {para_dE[:, _z].mean():.3g} +- {sem(para_dE[:, _z], jack):.3g}')
        print(f'chi2 = {chi2[:, _z].mean():.3g}')

        t1_fit = time.perf_counter()
        print(f"fit z = {_z}, time: {(t1_fit - t0_fit):.2f}s\n")

        # ---- 报告：每个 z 的 _fit.format ----
        report_lines.append(f"z = {_z}")
        report_lines.append("-" * 72)
        report_lines.append(f"condition number = {cond:.3g}")
        report_lines.append("")
        report_lines.append(_fit.format(maxline=True))
        report_lines.append("")

    # ---- 报告末尾：汇总表格 ----
    report_lines.append("=" * 72)
    report_lines.append("  Summary Table")
    report_lines.append("=" * 72)
    from prettytable import PrettyTable
    summary_tbl = PrettyTable()
    summary_tbl.field_names = ["z", "c0", "c1", "dE", "chi2/dof"]
    summary_tbl.align["z"] = "c"
    summary_tbl.align["c0"] = "c"
    summary_tbl.align["c1"] = "c"
    summary_tbl.align["dE"] = "c"
    summary_tbl.align["chi2/dof"] = "c"
    for _z in range(sampa.Nx):
        c0_str = f"{para_c0[:, _z].mean():.3f}({sem(para_c0[:, _z], jack) * 1e3:.0f})"
        c1_str = f"{para_c1[:, _z].mean():.3f}({sem(para_c1[:, _z], jack) * 1e3:.0f})"
        dE_str = f"{para_dE[:, _z].mean():.3f}({sem(para_dE[:, _z], jack) * 1e3:.0f})"
        chi2_str = f"{chi2[:, _z].mean():.2g}"
        summary_tbl.add_row([_z, c0_str, c1_str, dE_str, chi2_str])
    report_lines.append(str(summary_tbl))
    report_lines.append("")

    # 写入报告文件
    with open(f"{outpa.fit_dir}/{report_file}", "w") as f:
        f.write("\n".join(report_lines))
    print(f"report saved to {outpa.fit_dir}/{report_file}")

    # 保存 fit 结果（c0, c1, dE, chi2）为一个字典 → npz
    fit_result = {
        "c0": para_c0,
        "c1": para_c1,
        "dE": para_dE,
        "chi2": chi2,
    }
    np.savez(f"{outpa.fit_dir}/{outpa.fit_file}", **fit_result)
    print(f"fit result saved to {outpa.fit_dir}/{outpa.fit_file}")
    print("==================== do_fit end ====================")

    return fit_result


def plot_ratio(ratio: np.ndarray, fit_result: dict, sampa: SampleParams, fitpa: FitParams, plotpa: PlotParams, outpa: OutputParams, jack: bool):
    """绘制 ratio 散点与拟合结果子图"""
    print("==================== plot_ratio start ====================")
    time0 = time.perf_counter()

    # 从字典解包拟合结果
    para_c0 = fit_result["c0"]
    # para_c1 = fit_result["c1"]
    # para_dE = fit_result["dE"]
    chi2 = fit_result["chi2"]

    ratio_mean = ratio.mean(0)  # para: dt, dtau, z
    ratio_err = sem(ratio, jack)

    c0_mean = para_c0.mean(0)  # para: z
    c0_err = sem(para_c0, jack)
    chi2_mean = chi2.mean(0)

    n_z = len(plotpa.z_list)
    n_cols = 3
    n_rows = int(np.ceil(n_z / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10.5, 4 * n_rows),
                             sharex=True, sharey=True, dpi=150)
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, _z in enumerate(plotpa.z_list):
        ax = axes_flat[idx]

        band_down = (c0_mean[_z] - c0_err[_z])
        band_up = (c0_mean[_z] + c0_err[_z])

        # 绘制各 dt 的 ratio 散点与误差棒
        for i, dt in enumerate(plotpa.dt_list):
            tau_vals = np.arange(0, dt + 1)

            # 横坐标: tau - dt / 2
            x_vals = tau_vals - dt / 2.0

            # 纵坐标: ratio 均值与误差
            y_vals = ratio_mean[dt, tau_vals, _z]
            y_errs = ratio_err[dt, tau_vals, _z]

            color = plotpa.colors[i % len(plotpa.colors)]
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
                zorder=3,
                label=f"dt={dt}",
            )

        # fit 色带 (c0 plateau)
        x_band = np.array(plotpa.xlim)
        y1_band = np.array([band_down, band_down])
        y2_band = np.array([band_up, band_up])
        ax.fill_between(
            x_band,
            y1_band,
            y2_band,
            color="gray",
            alpha=0.35,
            linewidth=0,
            zorder=1,
            label="Fit c0",
        )

        c0_str = f"{para_c0[:, _z].mean():.3f}({sem(para_c0[:, _z], jack) * 1e3:.0f})"
        ax.set_title(
            f"z={_z}, c0={c0_str}, chi2={chi2_mean[_z]:.2f}", fontsize=12)
        ax.set_xlim(plotpa.xlim[0], plotpa.xlim[1])
        ax.set_ylim(plotpa.ylim[0], plotpa.ylim[1])
        ax.set_box_aspect(3/4)  # 4:3 宽高比，与 matplotlib 默认单图一致
        ax.legend(loc="upper right", fontsize=8)

    # 隐藏多余的子图
    for idx in range(n_z, n_rows * n_cols):
        axes_flat[idx].set_visible(False)

    # 统一坐标轴标签
    fig.supxlabel("t_ins - t_sep/2", fontsize=16)
    fig.supylabel("C3 / C2", fontsize=16)

    fig.suptitle(
        f"Unpolarized, P({sampa.Px},{sampa.Py},{sampa.Pz}), Nconf={sampa.Nconf}, Nsample = {sampa.Nsample}\n"
        f"fit: tsep = [{fitpa.dt_start}, {fitpa.dt_end}], cut = {fitpa.cut}",
        fontsize=13,
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(f"{outpa.fit_dir}/ratio.png", bbox_inches="tight")
    print(f"plot saved to {outpa.fit_dir}/ratio.png")

    # ---- c0 vs z 图 ----
    fig2, ax2 = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    z_vals = np.arange(sampa.Nx)
    ax2.errorbar(
        z_vals, c0_mean, yerr=c0_err,
        fmt="o", color="blue", ecolor="blue",
        capsize=3, markersize=3, label="c0"
    )
    ax2.set_xlim(-0.5, sampa.Nx - 0.5)
    ax2.set_ylim(plotpa.c0_ylim[0], plotpa.c0_ylim[1])
    ax2.set_xlabel("z", fontsize=16, labelpad=8)
    ax2.set_ylabel("c0", fontsize=16, labelpad=8)
    ax2.set_title(
        f"Unpolarized, P({sampa.Px},{sampa.Py},{sampa.Pz}), Nconf={sampa.Nconf}, Nsample = {sampa.Nsample}\n"
        f"fit: tsep = [{fitpa.dt_start}, {fitpa.dt_end}], cut = {fitpa.cut}",
        fontsize=13, pad=12,
    )
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{outpa.fit_dir}/c0.png", bbox_inches="tight")
    print(f"plot saved to {outpa.fit_dir}/c0.png")
    plt.close(fig2)

    # ---- chi2 vs z 图 ----
    fig3, ax3 = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    ax3.scatter(
        z_vals, chi2_mean,
        color="red", s=30, label="chi2/dof"
    )
    ax3.axhline(y=1, color="orange", linestyle="--", alpha=0.5,
                linewidth=1, label="chi2/dof = 1")
    ax3.set_xlim(-0.5, sampa.Nx - 0.5)
    ax3.set_ylim(0, 2)
    ax3.set_xlabel("z", fontsize=16, labelpad=8)
    ax3.set_ylabel("chi2/dof", fontsize=16, labelpad=8)
    ax3.set_title(
        f"Unpolarized, P({sampa.Px},{sampa.Py},{sampa.Pz}), Nconf={sampa.Nconf}, Nsample = {sampa.Nsample}\n"
        f"fit: tsep = [{fitpa.dt_start}, {fitpa.dt_end}], cut = {fitpa.cut}",
        fontsize=13, pad=12,
    )
    ax3.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{outpa.fit_dir}/chi2.png",
                bbox_inches="tight")
    print(f"plot saved to {outpa.fit_dir}/chi2.png")
    plt.close(fig3)

    time1 = time.perf_counter()
    print(f"plot done, time: {(time1 - time0):.2f}s")
    print("==================== plot_ratio end ====================")


if __name__ == "__main__":

    print("jackknife:", jack)
    print("Nconf:", sampa.Nconf)
    print("Nsample:", sampa.Nsample)
    print("conf_short:", sampa.conf_short)
    print("result_dir:", outpa.result_dir)

    # ---- Part 1: compute ratio ----
    if part_start <= 1:
        time0 = time.perf_counter()
        ratio = compute_ratio(sampa, outpa, jack)
        time1 = time.perf_counter()
        print(f"Peak Memory: {get_peak_memory_gb():.3f} GB")
        print(f"spend time: {(time1 - time0):.2f}s\n\n")
        if part_end == 1:
            print("job finish")
            sys.exit(0)
    else:
        print("===== skip compute ratio, loading ratio from file =====")
        ratio = np.load(f"{outpa.result_dir}/{outpa.ratio_file}")
        print(f"ratio loaded, shape: {ratio.shape}")

    # ---- Part 2: do fit ----
    if part_start <= 2:
        time0 = time.perf_counter()
        fit_result = do_fit(ratio, fitpa, sampa, outpa, jack)
        time1 = time.perf_counter()
        print(f"spend time: {(time1 - time0):.2f}s\n\n")
        if part_end == 2:
            print("job finish")
            sys.exit(0)
    else:
        print("===== skip do_fit, loading fit result from file =====")
        fit_result = dict(np.load(f"{outpa.fit_dir}/{outpa.fit_file}"))
        print(f"fit result loaded")

    # ---- Part 3: plot ----
    if part_start <= 3:
        plot_ratio(ratio, fit_result, sampa, fitpa, plotpa, outpa, jack)

    print("job finish")
