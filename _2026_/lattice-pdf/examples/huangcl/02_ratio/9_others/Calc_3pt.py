import numpy as np
import math
import os
import fileinput
from opt_einsum import contract
import time
from pathlib import Path

# ========== 参数配置 ==========
base_dir = Path(
    "/public/group/lqcd/donghx/2pt_Result/beta6.20_mu-0.2770_ms-0.2400_L24x72"
)
two_point_dirs = {
    "z": base_dir / "momsmear2z",
    "y": base_dir / "momsmear2y",
    "x": base_dir / "momsmear2x",
    "-z": base_dir / "momsmear-2z",
    "-y": base_dir / "momsmear-2y",
    "-x": base_dir / "momsmear-2x",
}
opedir = Path(
    "/public/group/lqcd/donghx/Strange_PDF/Result/beta6.20_mu-0.2770_ms-0.2400_L24x72/hyp_3D_5times/"
)

pol2 = "nopol"
element = "_Cg5g4"
mom = 2
pz_list = [2, 3, 4, 5, 6]  # 动量列表
Nt = 72
t_sep = np.arange(5, 16)  # 可能用于后续分析，当前未使用
delta_z = 16

# 组态列表：10000 到 50000 步长 200，排除 13100（如果存在）
configs = list(range(10000, 50200, 200))
# configs.remove(13100)  # 若需要排除则取消注释

# 文件名模板（两点函数）
twopt_template = (
    "twopt_slice_pp_Px{px}Py{py}Pz{pz}_eginphase{mom}{element}_{pol}_ss_conf{conf}.npy"
)
# OPE 文件名模板
ope_filename = "Ope_dzyx16_g4_conf{conf}_hypsmear_srcdist_VDU_z0.npy"
ope_filename_mz = "Ope_dzyx16_g4_conf{conf}_hypsmear_srcdist_VDU_0z.npy"

# ========== 数据结构初始化 ==========
# 存储两点函数：data_by_dir[方向][动量] = 列表（每个元素是一个组态加载的数组）
data_by_dir = {d: {p: [] for p in pz_list} for d in ["z", "y", "x", "-z", "-y", "-x"]}
# 存储 OPE 数据（仅当对应组态两点函数齐全时添加）
ope_list = []  # 顺序与 data_by_dir 中的组态顺序一致
ope_mz_list = []  # 顺序与 data_by_dir 中的组态顺序一致
conf_used = []  # 记录实际使用的组态编号（用于调试）
missing_configs = []  # 记录缺失文件的组态

# ========== 主循环 ==========
for conf in configs:
    print(f"Processing configuration {conf}")

    # 1. 检查并加载 OPE 文件
    ope_file = opedir / str(conf) / ope_filename.format(conf=conf)
    ope_file_mz = opedir / str(conf) / ope_filename_mz.format(conf=conf)
    if not ope_file.exists():
        if not ope_file_mz.exists():
            print(f"  OPE file missing for {conf}, skipping")
            missing_configs.append(conf)
            continue
    ope_data = np.load(ope_file)
    ope_data_mz = np.load(ope_file_mz)

    # 2. 检查当前组态下所有动量、所有方向（包括正负）的两点函数文件是否都存在
    all_exist = True
    for pz in pz_list:
        for d in ["z", "y", "x"]:  # 只遍历正方向，负方向由对应的正方向构造
            # 正动量文件（动量 +pz）
            if d == "z":
                px, py, pz_val = 0, 0, pz
            elif d == "y":
                px, py, pz_val = 0, pz, 0
            else:  # 'x'
                px, py, pz_val = pz, 0, 0

            filename = twopt_template.format(
                px=px, py=py, pz=pz_val, mom=mom, element=element, pol=pol2, conf=conf
            )
            filepath = two_point_dirs[d] / str(conf) / filename
            if not filepath.exists():
                all_exist = False
                break

            # 负动量文件（动量 -pz），注意 mom 取负，目录为 "-"+d
            filename_mz = twopt_template.format(
                px=-px,
                py=-py,
                pz=-pz_val,
                mom=-mom,
                element=element,
                pol=pol2,
                conf=conf,
            )
            filepath_mz = two_point_dirs["-" + d] / str(conf) / filename_mz
            if not filepath_mz.exists():
                all_exist = False
                break

        if not all_exist:
            break

    if not all_exist:
        print(f"  Some two-point files missing for {conf}, skipping")
        missing_configs.append(conf)
        continue

    # 3. 所有两点函数文件都存在，加载数据（正负方向）
    for pz in pz_list:
        for d in ["z", "y", "x"]:
            if d == "z":
                px, py, pz_val = 0, 0, pz
            elif d == "y":
                px, py, pz_val = 0, pz, 0
            else:
                px, py, pz_val = pz, 0, 0

            # 正动量
            filename = twopt_template.format(
                px=px, py=py, pz=pz_val, mom=mom, element=element, pol=pol2, conf=conf
            )
            filepath = two_point_dirs[d] / str(conf) / filename
            data = np.load(filepath)
            data_by_dir[d][pz].append(data)

            # 负动量
            filename_mz = twopt_template.format(
                px=-px,
                py=-py,
                pz=-pz_val,
                mom=-mom,
                element=element,
                pol=pol2,
                conf=conf,
            )
            filepath_mz = two_point_dirs["-" + d] / str(conf) / filename_mz
            data_mz = np.load(filepath_mz)
            data_by_dir["-" + d][pz].append(data_mz)

    # 4. 保存 OPE 数据
    ope_list.append(ope_data)
    ope_mz_list.append(ope_data_mz)
    conf_used.append(conf)
    print(f"  Loaded data for configuration {conf}")

# ========== 数据后处理 ==========
print(f"Total configurations with both OPE and two-point data: {len(conf_used)}")
print("Used configs:", conf_used)

# 检查每个动量下的组态数是否一致（可选）
for d in ["z", "y", "x", "-z", "-y", "-x"]:
    for pz in pz_list:
        nconf = len(data_by_dir[d][pz])
        print(f"Direction {d}, pz={pz}: {nconf} configurations")

# 将正负方向的数据平均，得到每个方向（z,y,x）每个动量下的平均两点函数
avg_by_dir = {d: [] for d in ["z", "y", "x", "-z", "-y", "-x"]}
for d in ["z", "y", "x", "-z", "-y", "-x"]:
    for pz in pz_list:
        # data_by_dir[d][pz] 和 data_by_dir["-"+d][pz] 长度相同，且组态顺序一致
        avg_list = []
        for i in range(len(data_by_dir[d][pz])):
            avg = data_by_dir[d][pz][i]
            avg_list.append(avg)
        avg_by_dir[d].append(np.array(avg_list))  # 形状 (nconf, Nt, ...)

# 转换为 NumPy 数组，形状 (len(pz), nconf, Nt, ...)
zdir_array = np.array(avg_by_dir["z"])  # (len(pz), nconf, Nt, ...)
ydir_array = np.array(avg_by_dir["y"])
xdir_array = np.array(avg_by_dir["x"])
mzdir_array = np.array(avg_by_dir["-z"])  # (len(pz), nconf, Nt, ...)
mydir_array = np.array(avg_by_dir["-y"])
mxdir_array = np.array(avg_by_dir["-x"])

# 合并三个方向，新轴在最前 (3, len(pz), nconf, Nt, ...)
Res_2pt_pol = np.stack(
    [zdir_array, ydir_array, xdir_array, mzdir_array, mydir_array, mxdir_array], axis=0
)

# OPE 数据转换
ope_list = np.array(ope_list)
ope_mz_list = np.array(ope_mz_list)
Res_ope = (ope_list - ope_mz_list) / 2.0
Res_ope = Res_ope.transpose(1, 0, 2, 3)

# 转置两点函数数组为后续计算方便的形状 (3, nconf, Nt, Nt, len(pz))
# 原代码中假设两点函数形状为 (Nt, Nt, ...)，此处沿用
Res_2pt_pol = Res_2pt_pol.transpose(0, 2, 3, 4, 1)

print("Res_ope shape:  ", Res_ope.shape)
# (3, 191, 72, 11)
print("Res_2pt shape:  ", Res_2pt_pol.shape)
# (6, 191, 72, 72, 5)

Nconf = Res_2pt_pol.shape[1]
print("Nconf :", Nconf)

np.save(
    "./Ratio_data/Ope_N%s_nosmear_2z.npy" % (Nconf),
    Res_ope,
)
np.save(
    "./Ratio_data/Res_2pt_all_6p_N%s.npy" % (Nconf),
    Res_2pt_pol,
)

Res_ope_duplicated = np.tile(Res_ope, (2, 1, 1, 1))

mean_ope = np.mean(Res_ope_duplicated, axis=1)
mean_2pt_pol = np.mean(Res_2pt_pol, axis=1)

# # ###___________________________________________________________________
# # ###___________________________________________________________________

centered_ope = Res_ope_duplicated - mean_ope[:, np.newaxis, :, :]
centered_2pt = Res_2pt_pol - mean_2pt_pol[:, np.newaxis, :, :, :]

ndt = t_sep.shape[0]
max_dt = t_sep.max()
threept = np.zeros(
    (6, Nconf, delta_z, ndt, Nt, max_dt + 1, len(pz_list)), dtype=np.complex64
)

for i_dt, dt in enumerate(t_sep):
    for t_source in range(Nt):
        t_sink = (t_source + dt) % Nt
        for i_t_insert in range(dt + 1):
            t_insert = (t_source + i_t_insert) % Nt
            threept[:, :, :, i_dt, t_source, i_t_insert, :] = contract(
                "dcz,dcp->dczp",
                centered_ope[:, :, t_insert, :],
                centered_2pt[:, :, t_sink, t_source, :],
            )

np.save("./Ratio_data/threept_N%s_hyp3D5_6p_2z.npy" % (Nconf), threept)
