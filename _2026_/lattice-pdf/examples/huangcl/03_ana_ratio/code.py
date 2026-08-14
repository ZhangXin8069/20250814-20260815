#!/public/home/huangcl/.venv/bin/python
"""
纯画图脚本, 在登录节点直接运行.
读取 02_ratio/1_result/ 下已有的 ratio .npy 和 fit .npz 数据,
为每个 z 画一张 ratio 大图 + c0 vs z + chi2 vs z, 保存到 03_ana_ratio/0_result/.
同时画对比图: c0 vs z, dE vs z, chi2 vs z, 所有拟合区间画在一起比较.
"""
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---- 从上级目录 98_tools 导入通用画图函数 ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "98_tools"))

# noqa: E402 抑制 autopep8 的 import 顺序检查
from analysis_tools import (  # noqa: E402
    DEFAULT_PLOT_COLORS,
    plot_multi_chi2,
    plot_multi_errbars,
    plot_single_chi2,
    plot_single_errbar,
    sem,
)


# ===== 定义 dataclass =====

@dataclass
class SampleParams:
    """样本参数"""
    conf_short: str
    Pz: int
    Nsample: int
    dt_max: int
    Nx: int
    jack: bool


@dataclass
class FitParams:
    """拟合参数 (只与画图标题相关)"""
    dt_start: int
    dt_end: int
    nex: int


@dataclass
class PlotParams:
    """画图参数"""
    # ---- ratio 散点图 ----
    dt_list: list[int]
    ratio_xlim: list[float]   # 横轴 (t_ins - t_sep/2) 范围
    ratio_z_ylim: list[tuple[int, list[float]]]  # (z, [ymin, ymax]) 列表, 纵轴范围

    # ---- c0/chi2 vs z 图 (单次 fit) ----
    zval: np.ndarray      # 横坐标, 如 np.arange(Nx)
    zval_xlim: list[float]  # 横轴范围, 如 [-0.5, Nx-0.5]
    c0_ylim: list[float]    # c0 纵轴范围

    # ---- 对比图 (所有 fit 画在一起) ----
    cmp_ylim: dict[str, list[float]]  # 纵轴范围, 按物理量名索引


@dataclass
class OutputParams:
    """路径参数"""
    ratio_dir: Path    # 读取 ratio/fit 的目录
    pic_dir: Path      # 保存图片的目录


# ===== 命令行参数 =====
parser = argparse.ArgumentParser(
    description="Plot ratio, c0, chi2 from existing fit results")
parser.add_argument("-c", type=str, default="L24x72",
                    help="conf_short (default: L24x72)")
parser.add_argument("-s", type=int, default=1, choices=[1, 2],
                    help="1=all plots (default), 2=skip single-fit plots, only comparison")
args = parser.parse_args()
conf_short = args.c
plot_mode = args.s  # 1: 全部画; 2: 跳过单次 fit 图, 只画对比图


# ===== 参数配置 (每个 conf 一个 if 分支)=====
if conf_short == "L24x72":
    # ---------- 样本参数 ----------
    sampa = SampleParams(
        conf_short="L24x72",
        Pz=2,
        Nsample=200,    # jackknife 时 Nsample = Nconf
        dt_max=20,
        Nx=24,
        jack=True,
    )

    # ---------- 要画图的拟合参数组合 ----------
    # fit 参数: dt_start, dt_end, nex
    _fit_range_list = [
        (6, 11, 2),
        (7, 11, 2),
        (7, 11, 3),
        (8, 11, 3),
        (8, 11, 4),
        (9, 11, 4),
    ]
    fitpa_list = [
        FitParams(dt_start=dt_start, dt_end=dt_end, nex=nex)
        for dt_start, dt_end, nex in _fit_range_list
    ]

    # ---------- 画图参数 ----------
    plotpa = PlotParams(
        # ratio 散点图
        dt_list=list(range(5, 14)),
        ratio_xlim=[-7, 7],
        ratio_z_ylim=[
            (0,  [-0.1, 0.8]),
            (4,  [-0.1, 0.6]),
            (8,  [-0.1, 0.4]),
            (12, [-0.3, 0.2]),
            (16, [-0.3, 0.2]),
            (20, [-0.3, 0.2]),
        ],
        # c0/chi2 vs z 图
        zval=np.arange(sampa.Nx),
        zval_xlim=[-1, sampa.Nx],
        c0_ylim=[-0.2, 1],
        # 对比图
        cmp_ylim={
            "c0":   [-0.2, 0.8],
            "dE":   [0.5, 1.8],
            "chi2": [0, 2],
        },
    )

else:
    print(f"conf {conf_short} not supported")
    sys.exit(1)


# ===== 路径组装 =====
BASE_DIR = Path(__file__).resolve().parent.parent
RATIO_RESULT_DIR = BASE_DIR / "02_ratio" / "1_result"
ANA_RESULT_DIR = BASE_DIR / "03_ana_ratio" / "0_result"

outpa = OutputParams(
    ratio_dir=RATIO_RESULT_DIR,
    pic_dir=ANA_RESULT_DIR / sampa.conf_short / f"Pz{sampa.Pz}",
)


def load_ratio(sampa: SampleParams, outpa: OutputParams):
    """
    加载 ratio 数组. 
    返回 shape: (Nsample, dt, dtau, z)
        - Nsample: 样本数
        - dt:      t_sep, 范围 0 ~ dt_max-1
        - dtau:    t_ins, 范围 0 ~ dt
        - z:       格点坐标, 范围 0 ~ Nx-1
    """
    ratio_file = f"ratio_Pz{sampa.Pz}_Nsam{sampa.Nsample}_dtmax{sampa.dt_max}.npy"
    ratio_path = outpa.ratio_dir / sampa.conf_short / ratio_file
    if not ratio_path.exists():
        print(f"Error: ratio file not found: {ratio_path}")
        sys.exit(1)
    ratio = np.load(ratio_path)
    print(f"ratio loaded: {ratio_path}")
    print(f"  shape: {ratio.shape}  (Nsample, dt, dtau, z)")
    return ratio


def load_fit_result(sampa: SampleParams, fitpa: FitParams, outpa: OutputParams):
    """
    加载拟合结果 .npz 文件. 
    返回字典, 每个 value shape: (Nsample, Nx)
        - c0:   拟合参数 c0
        - c1:   拟合参数 c1
        - dE:   拟合参数 dE
        - chi2: chi2/dof
    """
    fit_dir_name = (
        f"fit_Pz{sampa.Pz}_Nsam{sampa.Nsample}_dtmax{sampa.dt_max}"
        f"_tsep{fitpa.dt_start}_{fitpa.dt_end}_nex{fitpa.nex}"
    )
    fit_path = outpa.ratio_dir / sampa.conf_short / fit_dir_name / "0_fit_data.npz"
    if not fit_path.exists():
        print(f"Error: fit file not found: {fit_path}")
        sys.exit(1)
    fit_npz = np.load(fit_path)
    fit_result = {
        "c0": fit_npz["c0"],
        "c1": fit_npz["c1"],
        "dE": fit_npz["dE"],
        "chi2": fit_npz["chi2"],
    }
    print(f"fit loaded: {fit_path}")
    print(f"  c0 shape: {fit_result['c0'].shape}  (Nsample, Nx)")
    return fit_result


# ===== 专用画图函数 (ratio 散点图, 泛化性不强, 保持独立)=====

def plot_ratio_one_z(ratio_mean, ratio_err, c0_mean, c0_err, z, _ylim,
                     save_path, plotpa: PlotParams, sampa: SampleParams,
                     fitpa: FitParams):
    """
    画单个 z 的 ratio 散点图. 

    参数:
        ratio_mean: (dt, dtau) — 该 z 的 ratio 均值
        ratio_err:  (dt, dtau) — 该 z 的 ratio 误差
        c0_mean:    scalar — 该 z 的 c0 均值
        c0_err:     scalar — 该 z 的 c0 误差
        z:          int — 当前 z 值
        _ylim:      list[float] — 纵轴范围
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    band_down = c0_mean - c0_err
    band_up = c0_mean + c0_err

    # 绘制各 dt 的 ratio 散点与误差棒（只画到参与 fit 的最大 tsep）
    dt_max_fit = fitpa.dt_end
    dt_list_fit = [dt for dt in plotpa.dt_list if dt <= dt_max_fit]
    for i, dt in enumerate(dt_list_fit):
        tau_vals = np.arange(0, dt + 1)
        x_vals = tau_vals - dt / 2.0
        y_vals = ratio_mean[dt, tau_vals]
        y_errs = ratio_err[dt, tau_vals]

        color = DEFAULT_PLOT_COLORS[i % len(DEFAULT_PLOT_COLORS)]
        ax.errorbar(x_vals, y_vals, yerr=y_errs,
                    fmt="x", color=color, ecolor=color,
                    capsize=0, markeredgewidth=1.8,
                    linewidth=1.2, zorder=3, label=f"tsep={dt}")

    # fit 色带 (c0 的 ±1σ 带)
    x_band = np.array(plotpa.ratio_xlim)
    y1_band = np.array([band_down, band_down])
    y2_band = np.array([band_up, band_up])
    ax.fill_between(x_band, y1_band, y2_band,
                    color="gray", alpha=0.35, linewidth=0,
                    zorder=1, label="Fit c0")

    if _ylim is not None:
        ax.set_ylim(_ylim[0], _ylim[1])
    ax.set_xlim(plotpa.ratio_xlim[0], plotpa.ratio_xlim[1])
    ax.set_xlabel("t_ins - t_sep/2", fontsize=16, labelpad=8)
    ax.set_ylabel("C3 / C2", fontsize=16, labelpad=8)
    ax.set_title(
        f"Unpolarized, Pz={sampa.Pz}, z={z}, Nconf={sampa.Nsample}, Nsample={sampa.Nsample}\n"
        f"fit: tsep=[{fitpa.dt_start},{fitpa.dt_end}], nex={fitpa.nex}",
        fontsize=13, pad=12,
    )
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"  ratio_z{z}.png saved: {save_path}")
    plt.close(fig)


# ===== 主函数 =====

if __name__ == "__main__":
    # ---- 一次性加载所有数据 ----
    ratio = load_ratio(sampa, outpa)
    ratio_mean = ratio.mean(0)  # (dt, dtau, z)
    ratio_err = sem(ratio, sampa.jack)  # (dt, dtau, z)

    # 加载所有 fit 结果
    all_results = []
    for fitpa in fitpa_list:
        fit_result = load_fit_result(sampa, fitpa, outpa)
        all_results.append({
            "fitpa": fitpa,
            "c0": fit_result["c0"],
            "dE": fit_result["dE"],
            "chi2": fit_result["chi2"],
        })

    # ---- Part 1: 单次 fit 画图 (ratio 散点图 + c0 + dE + chi2)----
    if plot_mode == 1:
        for res in all_results:
            fitpa = res["fitpa"]
            c0 = res["c0"]
            dE = res["dE"]
            chi2 = res["chi2"]

            # 构造保存目录
            save_dir = outpa.pic_dir / \
                f"tsep{fitpa.dt_start}_{fitpa.dt_end}_nex{fitpa.nex}"
            save_dir.mkdir(parents=True, exist_ok=True)

            # 1. 为每个 z 画一张 ratio 散点图 (含 bestfit 色带)
            for z, _ylim in plotpa.ratio_z_ylim:
                ratio_mean_z = ratio_mean[:, :, z]  # (dt, dtau)
                ratio_err_z = ratio_err[:, :, z]    # (dt, dtau)
                c0_mean_z = c0[:, z].mean()          # scalar
                c0_err_z = sem(c0[:, z], sampa.jack)  # scalar

                save_path = save_dir / f"ratio_z{z}.png"
                plot_ratio_one_z(
                    ratio_mean_z, ratio_err_z,
                    c0_mean_z, c0_err_z, z, _ylim,
                    save_path, plotpa, sampa, fitpa,
                )

            # 2. c0 vs z (散点图)
            c0_mean = c0.mean(0)
            c0_err = sem(c0, sampa.jack)
            save_path = save_dir / "c0.png"
            plot_single_errbar(
                plotpa.zval, c0_mean, c0_err, save_path,
                xlabel="z", ylabel="c0",
                ylim=plotpa.c0_ylim, xlim=plotpa.zval_xlim,
                title=(
                    f"Unpolarized, Pz={sampa.Pz}, Nconf={sampa.Nsample}, "
                    f"Nsample={sampa.Nsample}\n"
                    f"fit: tsep=[{fitpa.dt_start},{fitpa.dt_end}], nex={fitpa.nex}"
                ),
                label="c0",
            )

            # 3. dE vs z (散点图)
            dE_mean = dE.mean(0)
            dE_err = sem(dE, sampa.jack)
            save_path = save_dir / "dE.png"
            plot_single_errbar(
                plotpa.zval, dE_mean, dE_err, save_path,
                xlabel="z", ylabel="dE",
                ylim=plotpa.cmp_ylim.get("dE"), xlim=plotpa.zval_xlim,
                title=(
                    f"Unpolarized, Pz={sampa.Pz}, Nconf={sampa.Nsample}, "
                    f"Nsample={sampa.Nsample}\n"
                    f"fit: tsep=[{fitpa.dt_start},{fitpa.dt_end}], nex={fitpa.nex}"
                ),
                label="dE",
            )

            # 4. chi2 vs z (散点图, 含 chi2/dof=1 横线)
            chi2_mean = chi2.mean(0)
            save_path = save_dir / "chi2.png"
            plot_single_chi2(
                plotpa.zval, chi2_mean, save_path,
                xlabel="z",
                xlim=plotpa.zval_xlim,
                title=(
                    f"Unpolarized, Pz={sampa.Pz}, Nconf={sampa.Nsample}, "
                    f"Nsample={sampa.Nsample}\n"
                    f"fit: tsep=[{fitpa.dt_start},{fitpa.dt_end}], nex={fitpa.nex}"
                ),
                label="chi2/dof",
            )
    else:
        print("===== skip single-fit plots (plot_mode=2) =====")

    # ---- Part 2: 对比图 (所有 fit 画在一起)----
    z_step = 3
    cmp_z_vals = np.arange(0, sampa.Nx, z_step)

    for qty in ["c0", "dE", "chi2"]:
        y_data = {}
        for res in all_results:
            data = res[qty]  # (Nsample, Nx)
            mean = data.mean(0)[cmp_z_vals]
            err = sem(data, sampa.jack)[cmp_z_vals]
            label = (
                f"tsep={res['fitpa'].dt_start}~{res['fitpa'].dt_end}, "
                f"nex={res['fitpa'].nex}"
            )
            y_data[label] = (mean, err)

        if qty == "chi2":
            save_path = outpa.pic_dir / f"cmp_{qty}.png"
            plot_multi_chi2(
                cmp_z_vals,
                {k: v[0] for k, v in y_data.items()},
                save_path,
                xlabel="z",
                xlim=plotpa.zval_xlim,
                title=f"Unpolarized, Pz={sampa.Pz}, Nconf={sampa.Nsample}",
            )
        else:
            save_path = outpa.pic_dir / f"cmp_{qty}.png"
            plot_multi_errbars(
                cmp_z_vals, y_data, save_path,
                xlabel="z", ylabel=qty,
                ylim=plotpa.cmp_ylim.get(qty),
                xlim=plotpa.zval_xlim,
                title=f"Unpolarized, Pz={sampa.Pz}, Nconf={sampa.Nsample}",
            )

    # ---- Part 3: Pz=2 整体 ratio 散点图 (无 bestfit)----
    # 对每个 ratio_z_ylim 中的 z, 画所有 dt 的 ratio 散点, 不含色带
    for z, _ylim in plotpa.ratio_z_ylim:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        for i, dt in enumerate(plotpa.dt_list):
            tau_vals = np.arange(0, dt + 1)
            x_vals = tau_vals - dt / 2.0
            y_vals = ratio_mean[dt, tau_vals, z]
            y_errs = ratio_err[dt, tau_vals, z]
            color = DEFAULT_PLOT_COLORS[i % len(DEFAULT_PLOT_COLORS)]
            ax.errorbar(x_vals, y_vals, yerr=y_errs,
                        fmt="x", color=color, ecolor=color,
                        capsize=0, markeredgewidth=1.8,
                        linewidth=1.2, zorder=3, label=f"tsep={dt}")

        if _ylim is not None:
            ax.set_ylim(_ylim[0], _ylim[1])
        ax.set_xlim(plotpa.ratio_xlim[0], plotpa.ratio_xlim[1])
        ax.set_xlabel("t_ins - t_sep/2", fontsize=16, labelpad=8)
        ax.set_ylabel("C3 / C2", fontsize=16, labelpad=8)
        ax.set_title(
            f"Unpolarized, Pz={sampa.Pz}, z={z}, Nconf={sampa.Nsample}, Nsample={sampa.Nsample}",
            fontsize=13, pad=12,
        )
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        save_path = outpa.pic_dir / f"ratio_z{z}_nofit.png"
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  ratio_z{z}_nofit.png saved: {save_path}")
        plt.close(fig)

    print("job finish")
