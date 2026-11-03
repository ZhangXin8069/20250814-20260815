#!/public/home/huangcl/.venv/bin/python
import numpy as np
import matplotlib.pyplot as plt
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
    sem, resample, get_peak_memory_gb, DEFAULT_PLOT_COLORS,
    plot_multi_errbars, plot_multi_scatter, PROJECT_DIR,
    calc_cov,
)


# ===== 独立开关，方便调试时修改 =====
jack = False
# ===================================


# ===== 定义 dataclass =====

@dataclass
class AnaParams:
    """分析参数"""
    conf_short: str
    Px: int
    Py: int
    Pz: int
    dt: int   # t_sep
    dtau: int  # t_ins (0 <= dtau <= dt)
    z: int     # z 坐标


@dataclass
class DirParams:
    """路径管理：区分读取路径和输出路径"""
    conf_short: str = ""
    Pz: int = 0

    @property
    def ratio_read_dir(self):
        """读取 ratio 数据的路径 (来自 02_ratio 的结果)"""
        return os.path.join(
            str(PROJECT_DIR), "02_ratio", "1_result",
            self.conf_short, f"Pz{self.Pz}")

    @property
    def corr2_read_dir(self):
        """读取 corr2 数据的路径 (来自 04_proton_energy 的结果)"""
        return os.path.join(
            str(PROJECT_DIR), "04_proton_energy", "1_result",
            self.conf_short, f"Pz{self.Pz}")

    @property
    def ratio_save_dir(self):
        """ratio 直方图输出路径 (05_ana_3dir_diff_sem/1_result/conf_short/ratio)"""
        d = os.path.join(
            str(PROJECT_DIR), "05_ana_3dir_diff_sem", "1_result",
            self.conf_short, "ratio")
        os.makedirs(d, exist_ok=True)
        return d

    @property
    def corr2_save_dir(self):
        """corr2 直方图输出路径 (05_ana_3dir_diff_sem/1_result/conf_short/corr2)"""
        d = os.path.join(
            str(PROJECT_DIR), "05_ana_3dir_diff_sem", "1_result",
            self.conf_short, "corr2")
        os.makedirs(d, exist_ok=True)
        return d

    @property
    def eff_mass_save_dir(self):
        """eff_mass 直方图输出路径 (05_ana_3dir_diff_sem/1_result/conf_short/eff_mass)"""
        d = os.path.join(
            str(PROJECT_DIR), "05_ana_3dir_diff_sem", "1_result",
            self.conf_short, "eff_mass")
        os.makedirs(d, exist_ok=True)
        return d


# ===== 从命令行参数读取 conf_short =====
parser = argparse.ArgumentParser(
    description="Analyze ratio differences among 3 directions")
parser.add_argument("-c", type=str, default="L24x72",
                    help="conf_short, e.g. L24x72 (default: L24x72)")
args = parser.parse_args()
conf_short = args.c
# =========================================


if conf_short == "L24x72":
    anapa = AnaParams(
        conf_short="L24x72",
        Px=0, Py=0,
        Pz=2,
        dt=6,
        dtau=3,
        z=0,
    )
else:
    print(f"conf {conf_short} not exist.")
    sys.exit()


# ===== DirParams =====
dirpa = DirParams(
    conf_short=conf_short,
    Pz=anapa.Pz,
)
print(f"ratio_read_dir: {dirpa.ratio_read_dir}")
print(f"corr2_read_dir: {dirpa.corr2_read_dir}")
print(f"ratio_save_dir: {dirpa.ratio_save_dir}")
print(f"corr2_save_dir: {dirpa.corr2_save_dir}")
print(f"eff_mass_save_dir: {dirpa.eff_mass_save_dir}")
# =====================


# ===== 通用画图函数 =====

def plot_histogram(data_dict: dict, save_dir: str, jack: bool,
                   xlabel: str = "value",
                   title_prefix: str = "",
                   filename_prefix: str = "hist"):
    """
    通用直方图画图函数.

    Parameters
    ----------
    data_dict : dict
        key 为 label 字符串 (如 "x_dir", "y_dir"),
        value 为 1D array, 即该 label 对应的样本值.
    save_dir : str
        图片保存目录.
    jack : bool
        是否使用 jackknife SEM.
    xlabel : str
        横轴标签.
    title_prefix : str
        标题前缀 (会附加 Nsample 等信息).
    filename_prefix : str
        文件名前缀.
    """
    print(f"===== plotting histogram: {filename_prefix} =====")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    colors = DEFAULT_PLOT_COLORS
    labels = list(data_dict.keys())

    # 收集样本值, 计算 mean ± sem
    all_vals = []
    summaries = []  # (mean, sem, label_str)
    for i, label in enumerate(labels):
        vals = data_dict[label]
        all_vals.append(vals)
        _mean = vals.mean()
        _sem = sem(vals, jack)
        summaries.append((_mean, _sem, label))

    # 统一横轴范围
    vmin = min(v.min() for v in all_vals)
    vmax = max(v.max() for v in all_vals)
    margin = (vmax - vmin) * 0.15 if vmax > vmin else 0.5
    x_range = (vmin - margin, vmax + margin)

    # bins 数量
    n_bins = int(np.sqrt(len(all_vals[0])))

    # 画直方图, label 中直接包含 mean(sem) 信息
    for i, label in enumerate(labels):
        color = colors[i % len(colors)]
        _mean, _sem, _ = summaries[i]
        label_text = f"{label} {_mean:.3g}({_sem:.3g})"
        ax.hist(all_vals[i], bins=n_bins, range=x_range,
                color=color, alpha=0.35, edgecolor=color,
                linewidth=0.8, label=label_text)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("frequency", fontsize=14)
    ax.set_title(
        f"{title_prefix}, Nsample={len(all_vals[0])}",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(
        save_dir,
        f"{filename_prefix}.png")
    fig.savefig(save_path, bbox_inches="tight")
    print(f"  saved: {save_path}")
    plt.close(fig)

    # 控制台输出汇总
    print("  Summary (mean(sem)):")
    for _mean, _sem, _label in summaries:
        print(f"    {_label}: {_mean:.3g}({_sem:.3g})")

    print(f"===== {filename_prefix} plot end =====")


# ===== 加载数据函数 =====

def load_ratio(dirpa: DirParams):
    """
    读取三个方向和平均的 ratio 数据.

    Returns
    -------
    dict
        key 为方向 ("x", "y", "z", "ave"),
        value 为 ratio 数组, shape: (Nsample, dt, dtau, z)
          axis=0: sample 维度 (bootstrap/jackknife)
          axis=1: t_sep (dt) 维度
          axis=2: t_ins (dtau) 维度, 0 <= dtau <= dt
          axis=3: z 维度 (0 ~ Nx-1)
    """
    ratio_data = {}
    for _dir in ["x", "y", "z"]:
        load_path = os.path.join(
            dirpa.ratio_read_dir, f"{_dir}_dir", "ratio.npy")
        print(f"===== loading ratio ({_dir}) from: {load_path} =====")
        ratio = np.load(load_path)
        ratio_data[_dir] = ratio

    # 读取 ave 的 ratio
    load_path = os.path.join(dirpa.ratio_read_dir, "ave_dir", "ratio.npy")
    print(f"===== loading ratio (ave) from {load_path} =====")
    ratio_ave = np.load(load_path)
    print(f"  ratio (ave) loaded, shape: {ratio_ave.shape}")
    ratio_data["ave"] = ratio_ave

    return ratio_data


def load_corr2(dirpa: DirParams):
    """
    读取四个方向 (x, y, z, ave) 的 corr2 (2pt) 数据.

    Returns
    -------
    dict
        key 为方向 ("x", "y", "z", "ave"),
        value 为 corr2 数组, shape: (Nsample, Ntsep)
          axis=0: sample 维度 (bootstrap/jackknife)
          axis=1: t_sep (dt) 维度, 0 ~ Ntsep-1
    """
    corr2_data = {}
    for _dir in ["x", "y", "z"]:
        load_path = os.path.join(
            dirpa.corr2_read_dir, f"corr2_{_dir}.npy")
        print(f"===== loading corr2 ({_dir}) from: {load_path} =====")
        corr2 = np.load(load_path)
        print(f"  corr2 ({_dir}) loaded, shape: {corr2.shape}")
        corr2_data[_dir] = corr2

    # 读取 ave 的 corr2
    load_path = os.path.join(dirpa.corr2_read_dir, "corr2_ave.npy")
    print(f"===== loading corr2 (ave) from {load_path} =====")
    corr2_ave = np.load(load_path)
    print(f"  corr2 (ave) loaded, shape: {corr2_ave.shape}")
    corr2_data["ave"] = corr2_ave

    return corr2_data


# ===== 有效质量计算 =====

def compute_eff_mass(corr2_data: dict) -> dict:
    """
    从 corr2 数据计算有效质量 (effective mass).

    mass = log( C(t) / C(t+1) ), 使用 np.roll 向量化计算.
    边界点 (最后一列) 无意义, 但保留以保持 shape 一致.

    Parameters
    ----------
    corr2_data : dict
        key 为方向 ("x", "y", "z", "ave"),
        value 为 corr2 数组, shape: (Nsample, Ntsep)

    Returns
    -------
    dict
        key 为方向 ("x", "y", "z", "ave"),
        value 为 eff_mass 数组, shape: (Nsample, Ntsep)
          axis=0: sample 维度
          axis=1: t 维度 (0 ~ Ntsep-1), 最后一列无意义
    """
    print("===== computing effective mass =====")
    eff_mass_data = {}
    for _dir in ["x", "y", "z", "ave"]:
        _corr2 = corr2_data[_dir]
        mass = np.log(_corr2 / np.roll(_corr2, shift=-1, axis=1))
        eff_mass_data[_dir] = mass
        print(f"  eff_mass ({_dir}) computed, shape: {mass.shape}")
    print("===== eff mass computation end =====")
    return eff_mass_data


# ===== 归一化协方差 (相关系数) 计算 =====

def print_normalized_cov(eff_mass_data: dict, anapa: AnaParams, jack: bool):
    """
    计算 eff_mass 在指定 dt 下, 不同方向 (x/y/z) 两两之间的归一化协方差 (相关系数).

    归一化协方差: corr = cov[i,j] / sqrt(cov[i,i] * cov[j,j])
    """
    print("===== normalized covariance (correlation) of eff_mass =====")
    print(f"  Pz={anapa.Pz}, tsep={anapa.dt}")

    dirs = ["x", "y", "z"]
    # 构建二维数组: (Nsample, 3), 每列对应一个方向的 eff_mass 在 dt 处的值
    arr_list = []
    for _dir in dirs:
        vals = eff_mass_data[_dir][:, anapa.dt]
        arr_list.append(vals)
    arr = np.column_stack(arr_list)  # (Nsample, 3)

    cov_mat, cond = calc_cov(arr, jack)
    print(f"  covariance matrix (shape={cov_mat.shape}):")
    print(cov_mat)
    print(f"  condition number: {cond:.3f}")

    # 归一化 -> 相关系数矩阵
    diag_std = np.sqrt(np.diag(cov_mat))
    corr_mat = cov_mat / np.outer(diag_std, diag_std)
    print("  correlation matrix:")
    print(corr_mat)

    # 打印两两之间的相关系数
    for i in range(3):
        for j in range(i + 1, 3):
            print(
                f"    corr({dirs[i]}_dir, {dirs[j]}_dir) = {corr_mat[i, j]:.4f}")

    print("===== normalized cov end =====\n")


# ===== 提取特定切片并画图的封装 =====

def plot_ratio_histogram(ratio_data: dict, anapa: AnaParams,
                         save_dir: str, jack: bool):
    """
    从 ratio_data 中提取指定 (dt, dtau, z) 的切片, 调用通用画图函数.
    """
    data_dict = {}
    for _dir in ["x", "y", "z"]:
        vals = ratio_data[_dir][:, anapa.dt, anapa.dtau, anapa.z]
        data_dict[f"{_dir}_dir"] = vals

    title = (f"P=({anapa.Px},{anapa.Py},{anapa.Pz}), "
             f"tsep={anapa.dt}, tins={anapa.dtau}, z={anapa.z}")
    filename = (f"hist_ratio_P{anapa.Pz}_z{anapa.z}"
                f"_tsep{anapa.dt}_tins{anapa.dtau}")

    plot_histogram(
        data_dict, save_dir, jack,
        xlabel="ratio value",
        title_prefix=title,
        filename_prefix=filename,
    )


def plot_corr2_histogram(corr2_data: dict, anapa: AnaParams,
                         save_dir: str, jack: bool):
    """
    从 corr2_data 中提取指定 dt (tsep) 的切片, 调用通用画图函数.
    """
    data_dict = {}
    for _dir in ["x", "y", "z"]:
        vals = corr2_data[_dir][:, anapa.dt]
        data_dict[f"{_dir}_dir"] = vals

    title = f"corr2, Pz={anapa.Pz}, tsep={anapa.dt}"
    filename = f"hist_corr2_P{anapa.Pz}_tsep{anapa.dt}"

    plot_histogram(
        data_dict, save_dir, jack,
        xlabel="corr2 value",
        title_prefix=title,
        filename_prefix=filename,
    )


def plot_eff_mass_histogram(eff_mass_data: dict, anapa: AnaParams,
                            save_dir: str, jack: bool):
    """
    从 eff_mass_data 中提取指定 dt (tsep) 的切片, 调用通用画图函数.

    eff_mass 的 shape 为 (Nsample, Ntsep-1), dt 索引对应 t = dt.
    """
    data_dict = {}
    for _dir in ["x", "y", "z"]:
        vals = eff_mass_data[_dir][:, anapa.dt]
        data_dict[f"{_dir}_dir"] = vals

    title = f"eff_mass, Pz={anapa.Pz}, tsep={anapa.dt}"
    filename = f"hist_eff_mass_P{anapa.Pz}_tsep{anapa.dt}"

    plot_histogram(
        data_dict, save_dir, jack,
        xlabel="effective mass",
        title_prefix=title,
        filename_prefix=filename,
    )


if __name__ == "__main__":

    print("jackknife:", jack)

    time0 = time.perf_counter()

    # ---- ratio 分析 ----
    ratio_data = load_ratio(dirpa)
    plot_ratio_histogram(ratio_data, anapa, dirpa.ratio_save_dir, jack)

    # ---- corr2 分析 ----
    corr2_data = load_corr2(dirpa)
    plot_corr2_histogram(corr2_data, anapa, dirpa.corr2_save_dir, jack)

    # ---- eff_mass 分析 ----
    eff_mass_data = compute_eff_mass(corr2_data)
    plot_eff_mass_histogram(eff_mass_data, anapa,
                            dirpa.eff_mass_save_dir, jack)

    # ---- 归一化协方差 ----
    print_normalized_cov(eff_mass_data, anapa, jack)

    print(f"total time: {(time.perf_counter() - time0):.2f}s")
    print("job finish")
