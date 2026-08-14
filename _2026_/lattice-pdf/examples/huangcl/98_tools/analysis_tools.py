#!/usr/bin/env python3
"""
# analysis_tool.py

# 通用分析工具模块和常用的模板, 供脚本调用. 调用方法如下:
# ---- 从上级目录 98_tools 导入通用画图函数 ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "98_tools"))

# noqa: E402 抑制 autopep8 的 import 顺序检查
from analysis_tools import (  # noqa: E402
    # funtions
)
"""

import os
import resource
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

PROJECT_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# 默认颜色配置
# ============================================================

DEFAULT_PLOT_COLORS = [
    "#4ba3e3",  # 1. 鲜亮浅蓝
    "#58b368",  # 2. 鲜亮草绿
    "#f29e38",  # 3. 暖橙
    "#a569bd",  # 4. 中紫
    "#2075bc",  # 5. 深蓝
    "#196f3d",  # 6. 深绿
    "#d9534f",  # 7. 砖红
    "#6c3483",  # 8. 暗紫
    "#21618c",  # 9. 藏青
    "#117864",  # 10. 深墨绿
]


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class SampleParams:
    """样本参数"""
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
    """拟合参数"""
    p0: dict
    prior: dict
    dt_start: int
    dt_end: int
    svdcut: float = None


@dataclass
class PlotParams:
    """画图参数"""
    # ---- 通用单图 ----
    xlim: list = None
    ylim: list = None


@dataclass
class OutputParams:
    """路径与输出文件名"""
    base_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent)
    result_dir: Path = None


# ============================================================
# 统计 / 重采样
# ============================================================

def sem(data: np.ndarray, jackknife: bool) -> np.ndarray:
    """
    计算标准误. 

    Parameters
    ----------
    data : np.ndarray
        输入数组, axis=0 为 sample 维度. 
    jackknife : bool
        是否为 jackknife 样本. 

    Returns
    -------
    np.ndarray
        标准误. 
    """
    error = data.std(0)
    if jackknife:
        error = error * np.sqrt(data.shape[0] - 1)
    return error


def resample(corr: np.ndarray, jackknife: bool, Nsample: int) -> np.ndarray:
    """
    对 corr 进行重采样. 

    Parameters
    ----------
    corr : np.ndarray
        原始数据, axis=0 为 conf 维度. 
    jackknife : bool
        True 表示 jackknife, False 表示 bootstrap. 
    Nsample : int
        bootstrap 样本数; jackknife 时该值被忽略. 

    Returns
    -------
    np.ndarray
        重采样后的数组, axis=0 为 sample 维度. 
    """
    seed = 0
    n_conf = corr.shape[0]
    if jackknife:
        re_corr = (n_conf * corr.mean(0) - corr) / (n_conf - 1)
    else:
        rng = np.random.default_rng(seed=seed)
        idx = rng.integers(0, n_conf, size=(Nsample, n_conf))
        # 矩阵乘法加速 bootstrap:
        #   构建计数矩阵 counts[s,k] = conf k 在样本 s 中出现次数
        #   re_corr = (1/n_conf) * counts @ corr.reshape(n_conf, -1)
        # 一次 BLAS 运算完成所有样本均值, 比 Python 循环快 10-20 倍.
        counts = np.zeros((Nsample, n_conf), dtype=np.float64)
        np.add.at(counts, (np.arange(Nsample)[:, None], idx), 1.0)
        corr_flat = corr.reshape(n_conf, -1)
        re_corr_flat = (1.0 / n_conf) * (counts @ corr_flat)
        re_corr = re_corr_flat.reshape(Nsample, *corr.shape[1:])
    return re_corr


def calc_cov(arr: np.ndarray, jackknife: bool) -> Tuple[np.ndarray, float]:
    """
    计算协方差矩阵与条件数. 

    Parameters
    ----------
    arr : np.ndarray
        二维数组, axis=0 为 sample, axis=1 为依赖变量. 
    jackknife : bool
        是否为 jackknife 样本. 

    Returns
    -------
    cov : np.ndarray
        协方差矩阵. 
    cond : float
        条件数 (最大特征值 / 最小特征值). 
    """
    diff = arr - arr.mean(0)
    n = arr.shape[0]
    cov = diff.T @ diff
    if jackknife:
        cov *= (n - 1) / n
    else:
        cov /= n
    eig = np.linalg.eigvalsh(cov)
    cond = eig[-1] / eig[0]
    return cov, cond


def calc_chi2(
    y_data: np.ndarray,
    y_fit: np.ndarray,
    cov: np.ndarray,
    svdcut: Optional[float] = None,
) -> float:
    """
    计算 chi2 = diff^T C^{-1} diff, 支持 lsqfit 风格的 svdcut. 

    Parameters
    ----------
    y_data : np.ndarray
        数据. 
    y_fit : np.ndarray
        拟合值. 
    cov : np.ndarray
        协方差矩阵. 
    svdcut : float, optional
        SVD cut, None 表示不截断. 

    Returns
    -------
    float
        chi2 值. 
    """
    diff = y_data - y_fit

    if svdcut is None:
        return diff @ np.linalg.solve(cov, diff)

    eigval, eigvec = np.linalg.eigh(cov)
    cut = eigval.max() * svdcut
    mask = eigval > cut
    eig_inv = 1.0 / eigval[mask]
    cov_inv = eigvec[:, mask] @ np.diag(eig_inv) @ eigvec[:, mask].T
    return diff @ cov_inv @ diff


def calc_chi2_dof(
    y_data: np.ndarray,
    y_fit: np.ndarray,
    cov: np.ndarray,
    n_params: int,
    svdcut: Optional[float] = None,
) -> Tuple[float, float, int]:
    """
    计算 chi2/dof. 

    Returns
    -------
    chi2_dof : float
        chi2 / dof. 
    chi2 : float
        chi2 值. 
    dof : int
        自由度. 
    """
    chi2 = calc_chi2(y_data, y_fit, cov, svdcut)
    dof = len(y_data) - n_params
    return chi2 / dof, chi2, dof


# ============================================================
# 拟合
# ============================================================

def fit(
    y_coor: np.ndarray,
    x_coor,
    model: Callable,
    fitpa: FitParams,
    jackknife: bool,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, float, lsqfit.nonlinear_fit]:
    """
    对每个 sample 做 lsqfit 非线性拟合. 

    Parameters
    ----------
    y_coor : np.ndarray
        数据数组, shape 为 (Nsample, Ndata). 
    x_coor : array-like
        拟合点坐标, 传给 lsqfit 的 data. 
    model : Callable
        模型函数, 签名 model(x, p) -> np.ndarray. 
    fitpa : FitParams
        拟合参数 dataclass, 包含 p0, prior, svdcut. 
        优先使用 prior; 若 prior 为 None 或空, 则退化为使用 p0. 
    jackknife : bool
        是否为 jackknife 样本. 

    Returns
    -------
    fit_result : Dict[str, np.ndarray]
        拟合结果, 包含每个参数及 chi2/dof. 
        key 为参数名, 额外包含 "chi2" 键. 
    cov : np.ndarray
        协方差矩阵. 
    cond : float
        条件数. 
    last_fit_info : lsqfit.nonlinear_fit
        最后一个 sample 的拟合对象. 
    """
    Nsample, _ = y_coor.shape
    param_names = list(fitpa.p0.keys())
    n_params = len(param_names)

    fit_result = {name: np.zeros(Nsample) for name in param_names}
    fit_result["chi2"] = np.zeros(Nsample)

    cov, cond = calc_cov(y_coor, jackknife)

    # 优先使用 prior, 否则退化为 p0
    use_prior = fitpa.prior is not None and len(fitpa.prior) > 0

    last_fit_info = None
    for _id in range(Nsample):
        y_gvar = gv.gvar(y_coor[_id], cov)

        if use_prior:
            _fit = lsqfit.nonlinear_fit(
                data=(x_coor, y_gvar),
                prior=fitpa.prior,
                fcn=model,
                svdcut=fitpa.svdcut,
            )
        else:
            _fit = lsqfit.nonlinear_fit(
                data=(x_coor, y_gvar),
                p0=fitpa.p0,
                fcn=model,
                svdcut=fitpa.svdcut,
            )
        last_fit_info = _fit

        for name in param_names:
            fit_result[name][_id] = _fit.pmean[name]

        chi2_dof_val, _, _ = calc_chi2_dof(
            y_coor[_id],
            model(x_coor, _fit.pmean),
            cov,
            n_params,
            fitpa.svdcut,
        )
        fit_result["chi2"][_id] = chi2_dof_val

    return fit_result, cov, cond, last_fit_info


# ============================================================
# Fit Report Demo 框架
# ============================================================

"""
# ---- 以下是一个可直接复制到实际 code.py 中补充的 fit report 框架 ----

def do_fit_and_report(
    y_coor: np.ndarray,
    x_coor,
    model: Callable,
    fitpa: FitParams,
    sampa: SampleParams,
    out_path: str,
    jack: bool,
):
    '''
    拟合 + 写报告 demo. 
    根据实际分析需求修改 for 循环维度, 参数名, 输出路径等. 
    '''
    # ---- 1. 报告头部: 样本与拟合设置信息 ----
    lines = []
    sep_line = "=" * 72
    lines.append(sep_line)
    lines.append("  Fit Report")
    lines.append(sep_line)
    lines.append(f"  conf_short  : {sampa.conf_short}")
    lines.append(f"  Nconf       : {sampa.Nconf}")
    lines.append(f"  Nsample     : {sampa.Nsample}")
    lines.append(f"  jackknife   : {jack}")
    lines.append(f"  data shape  : {y_coor.shape}")
    lines.append(f"  fitpa       : {fitpa}")  # 直接打印整个 dataclass
    lines.append(f"  model       : {model.__name__}")
    lines.append(sep_line)
    lines.append("")

    # ---- 2. 中间分析: 对每个 i 做 fit ----
    # 示例: 假设 y_coor 形状为 (Nsample, Ni, Ndata), 对每个 i 做拟合
    # 请根据实际分析替换 i 的含义 (如 z, dt_start 等)
    # all_fit_result[name] 的维度为 (Nsample, Ni), 先 sample 后 i
    param_names = list(fitpa.p0.keys())
    Ni = y_coor.shape[1]
    all_fit_result = {name: np.zeros((sampa.Nsample, Ni))
                      for name in param_names + ["chi2"]}
    all_cond = np.zeros(Ni)

    for i in range(Ni):
        _fit_result, _cov, _cond, _last_fit_info = fit(
            y_coor[:, i, :], x_coor, model, fitpa, jack)
        for name in param_names + ["chi2"]:
            all_fit_result[name][:, i] = _fit_result[name]
        all_cond[i] = _cond
        lines.append(f"i = {i}: condition number = {_cond:.3g}")
        # 输出本次 fit 最后一个 sample 的 lsqfit 详细报告
        if _last_fit_info is not None:
            lines.append(_last_fit_info.format(maxline=True))
            lines.append("")
    lines.append("")

    # ---- 3. 汇总表格: 每个 i 的 mean ± err 和 chi2/dof ----
    lines.append(sep_line)
    lines.append("  Summary Table")
    lines.append(sep_line)

    summary_tbl = PrettyTable()
    summary_tbl.field_names = ["i"] + param_names + ["chi2/dof"]
    for name in summary_tbl.field_names:
        summary_tbl.align[name] = "c"

    for i in range(Ni):
        row = [f"{i}"]
        for name in param_names:
            mean = all_fit_result[name][:, i].mean()
            err = sem(all_fit_result[name][:, i], jack)
            row.append(f"{mean:.3f}({err * 1e3:.0f})")
        row.append(f"{all_fit_result['chi2'][:, i].mean():.2g}")
        summary_tbl.add_row(row)

    lines.append(str(summary_tbl))
    lines.append("")

    # ---- 4. 保存报告 ----
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"report saved to {out_path}")

    # ---- 5. 返回整个拟合结果 ----
    return all_fit_result, all_cond

# ---- 框架结束 ----
"""


# ============================================================
# 画图
# ============================================================

def plot_single_errbar(
    x: np.ndarray,
    y: np.ndarray,
    yerr: Optional[np.ndarray],
    save_path: str,
    *,
    xlabel: str = "x",
    ylabel: str = "y",
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    title: Optional[str] = None,
    label: Optional[str] = None,
    legend_loc: str = "upper right",
    figsize: Tuple[float, float] = (8, 6),
    dpi: float = 150,
    plot_colors: Optional[List[str]] = None,
    # 阴影色带开关与参数
    show_band: bool = False,
    band_x: Optional[np.ndarray] = None,
    band_y_down: Optional[np.ndarray] = None,
    band_y_up: Optional[np.ndarray] = None,
    band_color: str = "gray",
    band_alpha: float = 0.35,
    band_label: str = "Fit band",
):
    """
    通用单条误差棒图. 

    固定参数: 
        - capsize = 0 (误差棒无帽子)
        - 默认颜色为 DEFAULT_PLOT_COLORS[0]

    show_band=True 时, 在 band_x 与 [band_y_down, band_y_up] 之间填充色带. 
    该色带通常用于表示某个拟合参数 (如 E0 或 c0) 的 ±1σ 范围. 
    调用方需要自行构造 band_x, band_y_down, band_y_up, 例如: 
        band_x = np.array([fitpa.dt_start, fitpa.dt_end])
        band_y_down = np.array([E0_mean - E0_err, E0_mean - E0_err])
        band_y_up = np.array([E0_mean + E0_err, E0_mean + E0_err])
    """
    if plot_colors is None:
        plot_colors = DEFAULT_PLOT_COLORS

    default_color = plot_colors[0]
    color = default_color
    ecolor = default_color

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.errorbar(
        x, y, yerr=yerr,
        fmt='x',
        color=color,
        ecolor=ecolor,
        capsize=0,
        label=label,
    )

    if show_band:
        ax.fill_between(
            band_x,
            band_y_down,
            band_y_up,
            color=band_color,
            alpha=band_alpha,
            linewidth=0,
            zorder=1,
            label=band_label,
        )

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.set_xlabel(xlabel, fontsize=16, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=16, labelpad=8)
    if title is not None:
        ax.set_title(title, fontsize=13, pad=12)
    if label is not None or show_band:
        ax.legend(loc=legend_loc)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"  saved: {save_path}")
    plt.close(fig)


def plot_multi_errbars(
    x: np.ndarray,
    y_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: str,
    *,
    xlabel: str = "x",
    ylabel: str = "y",
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    title: Optional[str] = None,
    x_offset: float = 0.3,
    legend_loc: str = "upper right",
    figsize: Tuple[float, float] = (10, 6),
    dpi: float = 150,
    plot_colors: Optional[List[str]] = None,
):
    """
    通用多条误差棒对比图. 

    固定参数: 
        - capsize = 0 (误差棒无帽子)
        - 颜色从 plot_colors 按顺序循环取
        - 同一横坐标的多条误差棒自动左右错开, 相邻间隔为 x_offset

    Parameters
    ----------
    x : np.ndarray
        公共横坐标. 
    y_data : Dict[str, Tuple[np.ndarray, np.ndarray]]
        每条误差棒的数据, key 为图例标签 label, 
        value 为 (y_mean, y_err) 元组. 
    """
    if plot_colors is None:
        plot_colors = DEFAULT_PLOT_COLORS

    labels = list(y_data.keys())
    n = len(labels)
    if n > 1:
        offsets = np.linspace(-(n - 1) * x_offset / 2,
                              (n - 1) * x_offset / 2, n)
    else:
        offsets = [0.0]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for i, label in enumerate(labels):
        y_mean, y_err = y_data[label]
        color = plot_colors[i % len(plot_colors)]
        x_shift = offsets[i]
        ax.errorbar(
            np.array(x) + x_shift,
            y_mean,
            yerr=y_err,
            fmt='x',
            color=color,
            ecolor=color,
            capsize=0,
            label=label,
        )

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.set_xlabel(xlabel, fontsize=16, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=16, labelpad=8)
    if title is not None:
        ax.set_title(title, fontsize=14, pad=12)
    ax.legend(loc=legend_loc)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"  saved: {save_path}")
    plt.close(fig)


def plot_single_scatter(
    x: np.ndarray,
    y: np.ndarray,
    save_path: str,
    *,
    xlabel: str = "x",
    ylabel: str = "y",
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    title: Optional[str] = None,
    label: Optional[str] = None,
    legend_loc: str = "upper right",
    figsize: Tuple[float, float] = (8, 6),
    dpi: float = 150,
    plot_colors: Optional[List[str]] = None,
    # ---- 水平参考线 ----
    show_hline: bool = False,
    hline_y: float = 1.0,
    hline_label: Optional[str] = None,
    hline_color: Optional[str] = None,
    hline_style: str = "--",
    hline_width: float = 1.5,
):
    """
    通用单条散点图.

    默认是任意散点图, 可通过参数调整为特定用途.
    例如画 chi2/dof 散点图时:
        ylabel = r"$\\chi^2 / \\mathrm{d.o.f.}$"
        ylim   = [0, 2]
        show_hline = True, hline_y = 1.0,
        hline_label = r"$\\chi^2 / \\mathrm{d.o.f.} = 1$"

    Parameters
    ----------
    x : np.ndarray
        自变量 (横坐标).
    y : np.ndarray
        因变量 (纵坐标).
    save_path : str
        图片保存路径.
    show_hline : bool
        是否画水平参考线, 默认 False.
    hline_y : float
        水平线纵坐标, 默认 1.0.
    hline_label : str, optional
        水平线图例标签, 默认 None (不显示).
    hline_color : str, optional
        水平线颜色, 默认使用 plot_colors[0].
    hline_style : str
        水平线线型, 默认 '--'.
    hline_width : float
        水平线线宽, 默认 1.5.
    """
    if plot_colors is None:
        plot_colors = DEFAULT_PLOT_COLORS

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 水平参考线
    if show_hline:
        _hline_color = hline_color if hline_color is not None else plot_colors[0]
        ax.axhline(y=hline_y, color=_hline_color, linestyle=hline_style,
                   linewidth=hline_width, label=hline_label)

    # 散点 (第二种颜色, 'x' 标记)
    color = plot_colors[1 % len(plot_colors)]
    ax.scatter(x, y, marker="x", color=color, s=40, linewidths=1.5,
               label=label, zorder=3)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.set_xlabel(xlabel, fontsize=16, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=16, labelpad=8)
    if title is not None:
        ax.set_title(title, fontsize=13, pad=12)
    # 有数据标签或水平线标签时显示图例
    if label is not None or hline_label is not None:
        ax.legend(loc=legend_loc)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"  saved: {save_path}")
    plt.close(fig)


def plot_multi_scatter(
    x: np.ndarray,
    y_data: Dict[str, np.ndarray],
    save_path: str,
    *,
    xlabel: str = "x",
    ylabel: str = "y",
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    title: Optional[str] = None,
    x_offset: float = 0.3,
    legend_loc: str = "upper right",
    figsize: Tuple[float, float] = (10, 6),
    dpi: float = 150,
    plot_colors: Optional[List[str]] = None,
    # ---- 水平参考线 ----
    show_hline: bool = False,
    hline_y: float = 1.0,
    hline_label: Optional[str] = None,
    hline_color: Optional[str] = None,
    hline_style: str = "--",
    hline_width: float = 1.5,
):
    """
    通用多条散点对比图.

    默认是任意散点对比图, 可通过参数调整为特定用途.
    例如画 chi2/dof 散点对比图时:
        ylabel = r"$\\chi^2 / \\mathrm{d.o.f.}$"
        ylim   = [0, 2]
        show_hline = True, hline_y = 1.0,
        hline_label = r"$\\chi^2 / \\mathrm{d.o.f.} = 1$"

    Parameters
    ----------
    x : np.ndarray
        公共自变量 (横坐标).
    y_data : Dict[str, np.ndarray]
        每条散点数据, key 为图例标签, value 为 y 值数组.
    save_path : str
        图片保存路径.
    x_offset : float
        相邻曲线横坐标偏移量, 默认 0.3.
    show_hline : bool
        是否画水平参考线, 默认 False.
    hline_y : float
        水平线纵坐标, 默认 1.0.
    hline_label : str, optional
        水平线图例标签, 默认 None (不显示).
    hline_color : str, optional
        水平线颜色, 默认使用 plot_colors[0].
    hline_style : str
        水平线线型, 默认 '--'.
    hline_width : float
        水平线线宽, 默认 1.5.
    """
    if plot_colors is None:
        plot_colors = DEFAULT_PLOT_COLORS

    labels = list(y_data.keys())
    n = len(labels)
    if n > 1:
        offsets = np.linspace(-(n - 1) * x_offset / 2,
                              (n - 1) * x_offset / 2, n)
    else:
        offsets = [0.0]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 水平参考线
    if show_hline:
        _hline_color = hline_color if hline_color is not None else plot_colors[0]
        ax.axhline(y=hline_y, color=_hline_color, linestyle=hline_style,
                   linewidth=hline_width, label=hline_label)

    # 每个散点 (颜色从第二种开始循环, 'x' 标记, 横坐标错开)
    for i, label in enumerate(labels):
        color = plot_colors[(i + 1) % len(plot_colors)]
        x_shift = offsets[i]
        ax.scatter(np.array(x) + x_shift, y_data[label],
                   marker="x", color=color, s=40,
                   linewidths=1.5, label=label, zorder=3)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.set_xlabel(xlabel, fontsize=16, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=16, labelpad=8)
    if title is not None:
        ax.set_title(title, fontsize=14, pad=12)
    ax.legend(loc=legend_loc)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"  saved: {save_path}")
    plt.close(fig)


# ============================================================
# 系统
# ============================================================

def get_peak_memory_gb():
    """返回当前进程峰值内存占用 (GB). """
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max_rss / (1024 ** 2)
