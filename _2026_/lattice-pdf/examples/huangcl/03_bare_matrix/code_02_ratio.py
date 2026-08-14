#!/public/home/huangcl/.venv/bin/python
import numpy as np
import matplotlib.pyplot as plt
import gvar as gv
from prettytable import PrettyTable
import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---- 从上级目录 98_tools 导入通用函数 ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "98_tools"))
from analysis_tools import (  # noqa: E402
    sem, resample, calc_cov, calc_chi2, calc_chi2_dof,
    fit, get_peak_memory_gb,
    plot_single_errbar, plot_single_chi2,
)


# ===== 独立开关，方便调试时修改 =====
debug = False  # 在登录节点跑, 方便排除错误, 结果输出到 0_debug 文件夹
jack = False  # debug == False
# ===================================


# ===== 定义四个 dataclass =====

@dataclass
class SampleParams:
    conf_short: str
    conf_name: str
    conf_ids: list[int]
    Nt: int
    Nx: int
    momP: int
    Px: int
    Py: int
    Pz: int
    Nsample: int
    dt_max: int

    @property
    def Nconf(self):
        return len(self.conf_ids)


@dataclass
class FitParams:
    p0: dict
    prior: dict
    dt_start: int
    dt_end: int
    nex: int
    svdcut: float = None


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
    """路径与输出文件名"""
    result_dir: str   # ratio 存放目录完整路径
    fit_dir: str
    ratio_file: str   # ratio 数组保存文件名
    fit_file: str     # fit 结果文件名
    report_file: str  # fit 报告文件名


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
    # 12300 是空文件夹
    _conf_ids = [x for x in range(4050, 48001, 50) if x != 12300]
    sampa = SampleParams(
        conf_short="L24x72",
        conf_name="beta6.20_mu-0.2770_ms-0.2400_L24x72",
        conf_ids=_conf_ids,
        Nt=72,
        Nx=24,
        momP=2,
        Px=0,
        Py=0,
        Pz=4,
        Nsample=3000,
        dt_max=20,
    )

    # 公共拟合参数（p0, prior 只需定义一次）
    _fit_p0 = {"c0": 0.6, "c1": -2, "dE": 1}
    _fit_prior = {
        "c0": gv.gvar(0.6, 0.5),
        "c1": gv.gvar(-2, 2),
        "dE": gv.gvar(1, 0.3),
    }
    # 只变化的参数：dt_start, dt_end, nex
    _fit_range_list = [
        (6, 11, 2),
        (7, 11, 2),
        (7, 11, 3),
        (8, 11, 3),
        (8, 11, 4),
        (9, 11, 4),
    ]
    fitpa_list = [
        FitParams(p0=_fit_p0, prior=_fit_prior,
                  dt_start=dt_start, dt_end=dt_end, nex=nex)
        for dt_start, dt_end, nex in _fit_range_list
    ]

    plotpa = PlotParams(
        plot_z=2,
        dt_list=list(range(6, 12)),
        z_list=list(range(0, sampa.Nx, 4)),
        xlim=[-7, 7],
        ylim=[-0.1, 1.0],
        c0_ylim=[-0.2, 1],
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
    sampa.conf_ids = sampa.conf_ids[:5]
    sampa.Nsample = sampa.Nconf + 5

if jack:
    sampa.Nsample = sampa.Nconf


# ===== OutputParams 在 if 块外用 conf_short 自动构造 =====
_base_dir = "0_debug" if debug else "1_result"
_ratio_dir = os.path.join(os.getcwd(), _base_dir, conf_short, f"Pz{sampa.Pz}")
outpa = OutputParams(
    result_dir=_ratio_dir,
    fit_dir="",  # 占位，__main__ 循环中动态赋值
    ratio_file=f"ratio_dtmax{sampa.dt_max}.npy",
    fit_file="0_fit_data.npz",
    report_file="1_fit_report.txt",
)

# 创建 ratio 目录
os.makedirs(outpa.result_dir, exist_ok=True)
# =========================================================


########################################################################################
def compute_ratio(sampa: SampleParams, dir: str, jack: bool):
    """加载数据并计算 ratio 数组"""
    print("==================== compute ratio start ====================")
    print(f"dir = {dir}")
    # mu, nu = 0(x), 1(y), 2(z), 3(t)
    if dir == 'x':
        Px = sampa.Pz
        Py = sampa.Px
        Pz = sampa.Py
        tdir1 = 1
        tdir2 = 2
    elif dir == 'y':
        Px = sampa.Py
        Py = sampa.Pz
        Pz = sampa.Px
        tdir1 = 2
        tdir2 = 0
    elif dir == 'z':
        Px = sampa.Px
        Py = sampa.Py
        Pz = sampa.Pz
        tdir1 = 0
        tdir2 = 1
    else:
        print('not this dir')
        sys.exit(1)

    _corr = np.zeros((sampa.Nconf, sampa.Nt, sampa.Nt), dtype=complex)
    _ope_01 = np.zeros((sampa.Nconf, sampa.Nx, sampa.Nt), dtype=complex)
    _ope_30 = np.zeros((sampa.Nconf, sampa.Nx, sampa.Nt), dtype=complex)
    _ope_31 = np.zeros((sampa.Nconf, sampa.Nx, sampa.Nt), dtype=complex)

    # load data
    for i, conf_id in enumerate(sampa.conf_ids):
        _corr[i] = np.load(
            f"/public/group/lqcd/donghx/2pt_Result/{sampa.conf_name}/momsmear{sampa.momP}{dir}/{conf_id}/twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}_eginphase2_Cg5g4_nopol_ss_conf{conf_id}.npy"
        )
        _ope_01[i] = np.load(
            f"/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/{sampa.conf_short}/{dir}dir/{conf_id}/ops_mu{tdir1}_nu{tdir2}_dz{sampa.Nx}_conf{conf_id}.npz"
        )["ops"]
        _ope_30[i] = np.load(
            f"/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/{sampa.conf_short}/{dir}dir/{conf_id}/ops_mu3_nu{tdir1}_dz{sampa.Nx}_conf{conf_id}.npz"
        )["ops"]
        _ope_31[i] = np.load(
            f"/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/{sampa.conf_short}/{dir}dir/{conf_id}/ops_mu3_nu{tdir2}_dz{sampa.Nx}_conf{conf_id}.npz"
        )["ops"]
    print("load finish")

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

    print("loop ti finish")
    del _corr, _ope, _ope_30, _ope_31, _ope_01
    gc.collect()

    # para: sample, ti(loop), dt
    corr2 = resample(_corr2_rel, jack, sampa.Nsample)
    # para: sample, ti(loop), dtau, z
    ope = resample(_ope_rel, jack, sampa.Nsample)

    del _corr2_rel, _ope_rel
    gc.collect()

    print("start to compute ratio")
    # 逐 dt 循环计算 ratio, 避免创建完整的 corr3 和 corr3_disc 导致爆内存
    ratio = np.zeros((sampa.Nsample, sampa.dt_max,
                     sampa.dt_max, sampa.Nx), dtype=float)
    for _dt in range(sampa.dt_max):
        _corr3_dt = _corr3[:, :, _dt, :, :]  # (Nconf, Nt, dt_max, Nx)
        # (Nsample, Nt, dt_max, Nx)
        corr3_dt = resample(_corr3_dt, jack, sampa.Nsample)

        corr2_dt = corr2[:, :, _dt]  # (Nsample, Nt)

        # corr3_disc_dt: (Nsample, Nt, dt_max, Nx)
        corr3_disc_dt = corr3_dt - corr2_dt[:, :, np.newaxis, np.newaxis] * ope

        # ratio_dt: (Nsample, dt_max, Nx)
        ratio_dt = (corr3_disc_dt /
                    corr2_dt[:, :, np.newaxis, np.newaxis]).mean(axis=1)
        ratio[:, _dt, :, :] = ratio_dt.real

    print("ratio shape:", ratio.shape)
    print("==================== compute ratio end ====================")
    return ratio


def model(x, p):
    dt = np.array([_x[0] for _x in x])
    dtau = np.array([_x[1] for _x in x])

    return (np.ones(len(x)) * p["c0"]
            + p["c1"] * np.exp(-p["dE"] * dtau)
            + p["c1"] * np.exp(-p["dE"] * (dt - dtau)))


# def check_chi2(chi2_manual, Ndata, _fit):
#     """比较手动计算的 chi2 与 _fit.chi2"""
#     chi2_fit = _fit.chi2
#
#     print(f"  chi2 (manual)       = {chi2_manual:.6g}")
#     print(f"  chi2 (fit)          = {chi2_fit:.6g}")
#     print(f"  num of data         = {Ndata}")
#     print(f"  dof (fit)           = {_fit.dof}")


def do_fit(ratio: np.ndarray, fitpa: FitParams, sampa: SampleParams, outpa: OutputParams, jack: bool):
    """执行拟合，返回字典 {c0, c1, dE, chi2}"""
    print("==================== do_fit start ====================")

    x_coor = []
    for dt in range(fitpa.dt_start, fitpa.dt_end+1):
        for dtau in range(fitpa.nex, dt - fitpa.nex + 1):
            x_coor.append((dt, dtau))
    Ndata = len(x_coor)

    # ---- 准备报告文件 ----
    report_file = outpa.report_file
    report_lines = []
    sep_line = "=" * 72
    report_lines.append(sep_line)
    report_lines.append(f"  Fit Report: {sampa.conf_short}")
    report_lines.append(sep_line)
    report_lines.append(f"  t_sep range : [{fitpa.dt_start}, {fitpa.dt_end}]")
    report_lines.append(f"  nex         : {fitpa.nex}")
    report_lines.append(f"  Nsample     : {sampa.Nsample}")
    report_lines.append(f"  jackknife   : {jack}")
    report_lines.append(sep_line)
    report_lines.append("")

    # 对每个 z 做拟合
    param_names = list(fitpa.p0.keys())  # ["c0", "c1", "dE"]
    all_fit_result = {name: np.zeros((sampa.Nsample, sampa.Nx))
                      for name in param_names + ["chi2"]}
    all_cond = np.zeros(sampa.Nx)

    for _z in range(sampa.Nx):
        t0_fit = time.perf_counter()
        sub_sample = np.zeros((sampa.Nsample, Ndata))
        for i, (dt, dtau) in enumerate(x_coor):
            sub_sample[:, i] = ratio[:, dt, dtau, _z]

        # 使用 analysis_tools.fit 替代手动循环
        _fit_result, _cov, _cond, _last_fit_info = fit(
            sub_sample, x_coor, model, fitpa, jack)
        for name in param_names + ["chi2"]:
            all_fit_result[name][:, _z] = _fit_result[name]
        all_cond[_z] = _cond

        # ---- stdout 精简输出 ----
        print(f'z={_z}')
        for name in param_names:
            mean = all_fit_result[name][:, _z].mean()
            err = sem(all_fit_result[name][:, _z], jack)
            print(f'{name} = {mean:.3g} +- {err:.3g}')
        print(f'chi2 = {all_fit_result["chi2"][:, _z].mean():.3g}')

        t1_fit = time.perf_counter()
        print(f"fit z = {_z}, time: {(t1_fit - t0_fit):.2f}s\n")

        # ---- 报告：每个 z 的 _fit.format ----
        report_lines.append(f"z = {_z}")
        report_lines.append("-" * 72)
        report_lines.append(f"condition number = {_cond:.3g}")
        report_lines.append("")
        report_lines.append(_last_fit_info.format(maxline=True))
        report_lines.append("")

    # ---- 报告末尾：汇总表格 ----
    report_lines.append("=" * 72)
    report_lines.append("  Summary Table")
    report_lines.append("=" * 72)
    summary_tbl = PrettyTable()
    summary_tbl.field_names = ["z"] + param_names + ["chi2/dof"]
    for name in summary_tbl.field_names:
        summary_tbl.align[name] = "c"
    for _z in range(sampa.Nx):
        row = [f"{_z}"]
        for name in param_names:
            mean = all_fit_result[name][:, _z].mean()
            err = sem(all_fit_result[name][:, _z], jack)
            row.append(f"{mean:.3f}({err * 1e3:.0f})")
        row.append(f"{all_fit_result['chi2'][:, _z].mean():.2g}")
        summary_tbl.add_row(row)
    report_lines.append(str(summary_tbl))
    report_lines.append("")

    # 写入报告文件
    with open(f"{outpa.fit_dir}/{report_file}", "w") as f:
        f.write("\n".join(report_lines))
    print(f"report saved to {outpa.fit_dir}/{report_file}")

    # 保存 fit 结果为一个字典 → npz
    np.savez(f"{outpa.fit_dir}/{outpa.fit_file}", **all_fit_result)
    print(f"fit result saved to {outpa.fit_dir}/{outpa.fit_file}")
    print("==================== do_fit end ====================")

    return all_fit_result


def plot_ratio(ratio: np.ndarray, fit_result: dict, sampa: SampleParams, fitpa: FitParams, plotpa: PlotParams, outpa: OutputParams, jack: bool):
    """绘制 ratio 散点与拟合结果子图"""
    print("==================== plot_ratio start ====================")
    time0 = time.perf_counter()

    # 从字典解包拟合结果 (all_fit_result 格式: {name: (Nsample, Nx)})
    para_c0 = fit_result["c0"]
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
                label=f"tsep={dt}",
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
        f"fit: tsep = [{fitpa.dt_start}, {fitpa.dt_end}], nex = {fitpa.nex}",
        fontsize=13,
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(f"{outpa.fit_dir}/ratio.png", bbox_inches="tight")
    print(f"plot saved to {outpa.fit_dir}/ratio.png")

    # ---- c0 vs z 图 (使用 plot_single_errbar) ----
    z_vals = np.arange(sampa.Nx)
    _title = (
        f"Unpolarized, P({sampa.Px},{sampa.Py},{sampa.Pz}), Nconf={sampa.Nconf}, Nsample = {sampa.Nsample}\n"
        f"fit: tsep = [{fitpa.dt_start}, {fitpa.dt_end}], nex = {fitpa.nex}"
    )
    plot_single_errbar(
        z_vals, c0_mean, yerr=c0_err,
        save_path=f"{outpa.fit_dir}/c0.png",
        xlabel="z", ylabel="c0",
        xlim=[-0.5, sampa.Nx - 0.5],
        ylim=plotpa.c0_ylim,
        title=_title,
        label="c0",
        legend_loc="upper right",
        figsize=(6.4, 4.8),
        dpi=150,
    )

    # ---- chi2 vs z 图 (使用 plot_single_chi2) ----
    plot_single_chi2(
        z_vals, chi2_mean,
        save_path=f"{outpa.fit_dir}/chi2.png",
        xlabel="z",
        xlim=[-0.5, sampa.Nx - 0.5],
        title=_title,
        label="chi2/dof",
        legend_loc="upper right",
        figsize=(6.4, 4.8),
        dpi=150,
    )

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
        ratio = compute_ratio(sampa, "x", jack)
        ratio += compute_ratio(sampa, "y", jack)
        ratio += compute_ratio(sampa, "z", jack)
        ratio /= 3
        # 保存 ratio 数组
        np.save(f"{outpa.result_dir}/{outpa.ratio_file}", ratio)
        print(f"ratio saved to {outpa.result_dir}/{outpa.ratio_file}")
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

    # ---- Part 2: do fit (loop over fitpa_list) ----
    for _fitpa in fitpa_list:
        _fit_dir_name = f"tsep{_fitpa.dt_start}_{_fitpa.dt_end}_nex{_fitpa.nex}"
        outpa.fit_dir = os.path.join(outpa.result_dir, _fit_dir_name)
        os.makedirs(outpa.fit_dir, exist_ok=True)

        if part_start <= 2:
            time0 = time.perf_counter()
            fit_result = do_fit(ratio, _fitpa, sampa, outpa, jack)
            time1 = time.perf_counter()
            print(f"spend time: {(time1 - time0):.2f}s\n\n")
            if part_end == 2:
                print("job finish")
                sys.exit(0)
        else:
            print(
                f"===== skip do_fit ({_fit_dir_name}), loading from file =====")
            fit_result = dict(np.load(f"{outpa.fit_dir}/{outpa.fit_file}"))
            print(f"fit result loaded")

        # ---- Part 3: plot (per fitpa) ----
        if part_start <= 3:
            plot_ratio(ratio, fit_result, sampa, _fitpa, plotpa, outpa, jack)

    print("job finish")
