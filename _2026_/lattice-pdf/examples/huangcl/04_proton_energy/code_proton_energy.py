#!/public/home/huangcl/.venv/bin/python
import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---- 从上级目录 98_tools 导入通用函数 ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "98_tools"))
from analysis_tools import (  # noqa: E402
    sem, resample, get_peak_memory_gb, DEFAULT_PLOT_COLORS,
    plot_multi_errbars, plot_multi_scatter,
)

# ===== 独立开关，方便调试时修改 =====
debug = False  # 在登录节点跑, 方便排除错误, 结果输出到 0_debug 文件夹
jack = False  # debug == False
# ===================================


# ===== 定义 dataclass =====

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
class PlotParams:
    """画图参数"""
    xlim: list[float]
    ylim: list[float]
    sem_ylim: list[float] = None
    x_offset: float = 0.1


@dataclass
class OutputParams:
    """路径管理"""
    base_dir: str = "1_result"
    conf_short: str = ""
    Pz: int = 0

    @property
    def result_dir(self):
        return os.path.join(os.getcwd(), self.base_dir,
                            self.conf_short, f"Pz{self.Pz}")

    def get_sub_dir(self, name):
        """返回子目录路径并创建"""
        d = os.path.join(self.result_dir, name)
        os.makedirs(d, exist_ok=True)
        return d

    def sub_dir_path(self, name):
        """仅返回子目录路径, 不创建"""
        return os.path.join(self.result_dir, name)


# ===== 从命令行参数读取 =====
parser = argparse.ArgumentParser(
    description="Gluon unpolarized PDF effective mass calculation")
parser.add_argument("-c", type=str, default="L24x72",
                    help="conf_short, e.g. L24x72 (default: L24x72)")
parser.add_argument("-s", type=int, default=1,
                    choices=[1, 2, 3], help="start part: 1=corr2, 2=fit, 3=plot (default: 1)")
parser.add_argument("-e", type=int, default=3,
                    choices=[1, 2, 3], help="end part: 1=corr2, 2=fit, 3=plot (default: 3)")
args = parser.parse_args()
conf_short = args.c
part_start = args.s
part_end = args.e
# ============================


# ===== 画图参数配置（按 Pz 索引）=====
_plotpa_config = {
    # Pz : (ylim, sem_ylim)
    2: ([0.5, 1], [-0.01, 0.1]),
    3: ([0.7, 1.2], [-0.01, 0.1]),
    4: ([0.9, 1.4], [-0.01, 0.1]),
    5: ([1.1, 1.6], [-0.01, 0.1]),
    6: ([1.3, 1.8], [-0.01, 0.1]),
}
# =====================================

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
        Pz=6,
        Nsample=3000,
        dt_max=20,
    )

    _ylim, _sem_ylim = _plotpa_config[sampa.Pz]
    plotpa = PlotParams(
        xlim=[-0.5, 15.5],
        ylim=_ylim,
        sem_ylim=_sem_ylim,
        x_offset=0.2,
    )

else:
    print(f"conf {conf_short} not exist.")
    sys.exit()


# debug / jack 调整
if debug:
    print("debug")
    sampa.conf_ids = sampa.conf_ids[:5]
    jack = True

if jack:
    sampa.Nsample = sampa.Nconf


# ===== OutputParams =====
_base_dir = "0_debug" if debug else "1_result"
outpa = OutputParams(
    base_dir=_base_dir,
    conf_short=conf_short,
    Pz=sampa.Pz,
)
os.makedirs(outpa.result_dir, exist_ok=True)
# =========================


def compute_corr2(sampa: SampleParams, dir: str, jack: bool):
    """
    加载指定方向的两点关联函数，计算 corr2 并 resample。
    类似 ratio 代码中按方向读取数据的方式。
    """
    print(
        f"==================== compute_corr2 ({dir}) start ====================")

    # 根据方向确定动量分量（与 ratio 代码一致）
    if dir == 'x':
        Px = sampa.Pz
        Py = sampa.Px
        Pz = sampa.Py
    elif dir == 'y':
        Px = sampa.Py
        Py = sampa.Pz
        Pz = sampa.Px
    elif dir == 'z':
        Px = sampa.Px
        Py = sampa.Py
        Pz = sampa.Pz
    else:
        print('not this dir')
        sys.exit(1)

    _corr = np.zeros((sampa.Nconf, sampa.Nt, sampa.Nt), dtype=complex)

    # load data (使用 momsmear{ momP }{ dir } 路径，与 ratio 一致)
    for i, conf_id in enumerate(sampa.conf_ids):
        _corr[i] = np.load(
            f"/public/group/lqcd/donghx/2pt_Result/{sampa.conf_name}/momsmear{sampa.momP}{dir}/{conf_id}/twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}_eginphase2_Cg5g4_nopol_ss_conf{conf_id}.npy"
        )

    print("load finish")
    print("2pt shape:", _corr.shape)

    # para: conf, ti(loop), dt
    _corr2_rel = np.zeros((sampa.Nconf, sampa.Nt, sampa.dt_max), dtype=complex)

    # loop ti
    for ti in range(sampa.Nt):
        corr2_shift = np.roll(_corr[:, :, ti], shift=-ti, axis=1)
        _corr2_rel[:, ti, :] = corr2_shift[:, :sampa.dt_max]

    _corr2_ave = _corr2_rel.mean(1)

    # para: sample, dt
    corr2 = resample(_corr2_ave, jack, sampa.Nsample).real

    del _corr, _corr2_rel, _corr2_ave
    gc.collect()

    print(f"Peak Memory: {get_peak_memory_gb():.3f} GB")
    print(
        f"==================== compute_corr2 ({dir}) end ====================")

    return corr2


if __name__ == "__main__":

    print("jackknife:", jack)
    print("Nconf:", sampa.Nconf)
    print("Nsample:", sampa.Nsample)
    print("conf_short:", sampa.conf_short)
    print("result base:", outpa.result_dir)

    # ---- Part 1: compute corr2 ----
    if part_start <= 1:
        time0 = time.perf_counter()
        corr2_x = compute_corr2(sampa, "x", jack)
        corr2_y = compute_corr2(sampa, "y", jack)
        corr2_z = compute_corr2(sampa, "z", jack)

        corr2_ave = (corr2_x + corr2_y + corr2_z) / 3.0

        np.save(os.path.join(outpa.result_dir, "corr2_x.npy"), corr2_x)
        np.save(os.path.join(outpa.result_dir, "corr2_y.npy"), corr2_y)
        np.save(os.path.join(outpa.result_dir, "corr2_z.npy"), corr2_z)
        np.save(os.path.join(outpa.result_dir, "corr2_ave.npy"), corr2_ave)
        print(f"corr2 arrays saved to {outpa.result_dir}")
        time1 = time.perf_counter()
        print(f"Peak Memory: {get_peak_memory_gb():.3f} GB")
        print(f"corr2 time: {(time1 - time0):.2f}s\n")
        if part_end == 1:
            print("job finish")
            sys.exit(0)
    else:
        print("===== skip compute corr2, loading from file =====")
        corr2_x = np.load(os.path.join(outpa.result_dir, "corr2_x.npy"))
        corr2_y = np.load(os.path.join(outpa.result_dir, "corr2_y.npy"))
        corr2_z = np.load(os.path.join(outpa.result_dir, "corr2_z.npy"))
        corr2_ave = np.load(os.path.join(outpa.result_dir, "corr2_ave.npy"))
        print(f"corr2 arrays loaded, shape: {corr2_x.shape}")

    # ---- Part 2: fit (placeholder) ----
    if part_start <= 2:
        # TODO: 拟合代码将在后续添加
        # 需要从 corr2 提取有效质量，然后对有效质量做拟合
        # 拟合模型: aE(dt) = aE0 + ...
        pass
        if part_end == 2:
            print("fit finish")
            sys.exit(0)

    # ---- Part 3: plot ----
    if part_start <= 3:
        time0 = time.perf_counter()
        x_vals = np.arange(sampa.dt_max)
        effmass_data = {}
        sem_data = {}
        for _dir, _corr2 in [("xdir", corr2_x), ("ydir", corr2_y), ("zdir", corr2_z), ("ave", corr2_ave)]:
            mass = np.log(_corr2 / np.roll(_corr2, shift=-1, axis=1))
            effmass_data[_dir] = (mass.mean(0), sem(mass, jack))
            sem_data[_dir] = sem(mass, jack)

        # 图1: eff mass 对比
        plot_multi_errbars(
            x_vals, effmass_data,
            save_path=os.path.join(outpa.result_dir, "eff_mass.png"),
            xlabel="t/a", ylabel="aE",
            xlim=plotpa.xlim,
            ylim=plotpa.ylim,
            x_offset=plotpa.x_offset,
            title=(
                f"{conf_short}, P=({sampa.Px},{sampa.Py},{sampa.Pz}), "
                f"Nconf={sampa.Nconf}, Nsample={sampa.Nsample}"
            ),
        )

        # 图2: SEM 对比散点图 (t=0~14)
        t_max_sem = 15  # t=0~14
        x_sem = np.arange(t_max_sem)
        sem_scatter = {}
        for _dir in ["xdir", "ydir", "zdir", "ave"]:
            sem_scatter[_dir] = sem_data[_dir][:t_max_sem]
        plot_multi_scatter(
            x_sem, sem_scatter,
            save_path=os.path.join(outpa.result_dir, "sem_comparison.png"),
            xlabel="t/a", ylabel="SEM(aE)",
            xlim=plotpa.xlim,
            ylim=plotpa.sem_ylim,
            x_offset=plotpa.x_offset,
            title=(
                f"{conf_short}, P=({sampa.Px},{sampa.Py},{sampa.Pz}), "
                f"Nconf={sampa.Nconf}, Nsample={sampa.Nsample}"
            ),
        )

        time1 = time.perf_counter()
        print(f"plot time: {(time1 - time0):.2f}s\n")

    print("job finish")
