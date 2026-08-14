import sys
sys.path.append('/public/group/imp/zengch/LQCD/input_file')
sys.path.append('/public/group/imp/zengch/LQCD/tool')

import pandas as pd
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
import time
import matplotlib.pyplot as plt
import os
from tool import *
from data_chose import data_chose
from C_pt_load import this_path
import shutil
import pdb  # 用于调试代码 pdb.set_trace()

# ===================== 新增MPI并行部分 =====================
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# ==========================================================

file_path = '/public/group/imp/zengch/LQCD/renorma/result/Ratio_data/FeynmanHellman'


def fit_ratio_FeynmenHellman(z_fit, nr_, t_sep_do, t_sep_up, Pz_, conf_, note_name_, 
                             z0_t_sep_do, z0_t_sep_up, save_csv=False):
    file_read = f'{file_path}/Delta_Ratio_data_{conf_}{note_name_}_pz{Pz_}_nr{nr_}.npz'
    data = np.load(file_read)

    delta = data['delta']
    t_sep_vals = data['t_sep_vals']
    
    mask = (t_sep_vals >= t_sep_do) & (t_sep_vals <= t_sep_up)
    c0_data = delta[z_fit, mask, :]
    
    n_data = np.shape(c0_data)[0]
    n_samples = np.shape(c0_data)[1]
    fit_results_list = []
    

    c0_in, chi_in = fit_ratio_mean_FeynmenHellman(z_fit, nr_, t_sep_do, t_sep_up, Pz_, conf_, note_name_)
    c_inv = np.linalg.inv(covariance_matrix(c0_data, 'boot'))
    c_inv_nocov = np.diag(1./np.std(c0_data, axis=1)**2.)
    
    # 提前计算所有tsep点的均值谱
    c0_data_mean = np.mean(c0_data, axis=1)

    for i_sam in range(n_samples):
        c0_data_i = c0_data[:, i_sam]
        def cost_function(c0_th):
            r_c = c0_data_i - c0_th
            chi2_all = r_c.T @ c_inv @ r_c
            return chi2_all / len(r_c - m.nfit)        
        m = Minuit(cost_function, c0_th=c0_in)
        m.migrad()
        c0 = m.values['c0_th']
        chi2_mean = cost_function(c0)
        fit_results_list.append({"sample_i": i_sam, "c0": c0, "chi2_mean": chi2_mean})

    

    # 路径改用 z=0 对应的原始 t_sep 拼接目录
    if save_csv:
        c0_save_path = (f'{this_path}/result/ratio_tsep_tisep_FeynmenHellman/'
                        f'{conf_}{note_name_}_pz{Pz_}/ex{nr_}/{z0_t_sep_do}_{z0_t_sep_up}/')
       
        os.makedirs(c0_save_path, exist_ok=True)
        fit_results = pd.DataFrame(fit_results_list)
        fit_results.to_csv(f"{c0_save_path}/z{z_fit}_tsep{t_sep_do}_{t_sep_up}.csv", index=False)

   
    
    c0_mean = np.mean([item["c0"] for item in fit_results_list])
    c0_std = np.std([item["c0"] for item in fit_results_list])
    chi2_mean = np.mean([item["chi2_mean"] for item in fit_results_list])

    # 新增：均值谱对应的chi2均值与误差

    
    r_mean = c0_data_mean - c0_mean
    chi2_vs_mean = r_mean.T @ c_inv @ r_mean
    chi2_vs_mean = chi2_vs_mean / (len(r_mean) - m.nfit)

    chi2_vs_mean_nocov = r_mean.T @ c_inv_nocov @ r_mean
    chi2_vs_mean_nocov = chi2_vs_mean_nocov / (len(r_mean) - m.nfit)

    

    print('.....................................................')
    print(f'z = {z_fit}:Pz={Pz_}:tsep.....{t_sep_do}-{t_sep_up}')
    print(f'c0_mean = {c0_mean}')
    print(f'chi2_mean (sample self) = {chi2_mean}')
    print(f'chi2_mean (fit vs data mean) = {chi2_vs_mean}')
    print(f'chi2_mean (fit vs data mean nocov) = {chi2_vs_mean_nocov}')
    print('.....................................................') 
    return c0_mean, c0_std, chi2_vs_mean_nocov


def fit_ratio_mean_FeynmenHellman(z_fit, nr_, t_set_do, t_set_up, Pz_, conf_, note_name_):
    file_read = f'{file_path}/Delta_Ratio_data_{conf_}{note_name_}_pz{Pz_}_nr{nr_}.npz'
    data = np.load(file_read)

    delta = data['delta']
    t_sep_vals = data['t_sep_vals']

    mask = (t_sep_vals >= t_set_do) & (t_sep_vals <= t_set_up)
    c0_data = delta[z_fit, mask, :]

    std_err = np.std(c0_data, axis=1)
    n_data = np.shape(c0_data)[0]
    n_samples = np.shape(c0_data)[1]
    c_inv = np.linalg.inv(covariance_matrix(c0_data, 'boot'))
    c0_data_mean = np.mean(c0_data, axis=1)

    def cost_function(c0_th):
        r_c = c0_data_mean - c0_th
        chi2_all = r_c.T @ c_inv @ r_c
        return chi2_all / (len(r_c) - m.nfit)

    c0_in = 0.54
    m = Minuit(cost_function, c0_th=c0_in)
    m.migrad()
    c0 = m.values['c0_th']
    chi2_mean = cost_function(c0)
    return c0, chi2_mean


def c0_vs_z_FeynmenHellman(nr_, t_sep_do, t_sep_up, Pz_, conf_, note_name_, chi2_limt_):

    c0_save_path = (f'{this_path}/result/ratio_tsep_tisep_FeynmenHellman/'
                        f'{conf_}{note_name_}_pz{Pz_}/ex{nr_}/{t_sep_do}_{t_sep_up}/')
    # 先删除原有整个文件夹，再新建
    if os.path.exists(c0_save_path):
        shutil.rmtree(c0_save_path)

    res_list = []
    chi2_history = []
    # 判定区间
   
    chi2_ratio_upper_limit = chi2_limt_

    # 全局记录：上一轮 z 最终收敛的 t_sep，下一轮 z 以此为起点
    last_t_do = t_sep_do
    last_t_up = t_sep_up

    ratio_limit = 2.
    if (conf_ == 'L24x72') and (Pz_ == 4):
        ratio_limit = 0.8 
    
    z_i_charge = 6
    if (conf_ == 'L48x96_C48P14') and (Pz_ == 9):
        chi2_ratio_upper_limit = 2.2
        ratio_limit = 2.5
        z_i_charge = 7
    
    if (conf_ == 'L32x64_C32P23') and (Pz_ == 6):
        chi2_ratio_upper_limit = 2.2
        ratio_limit = 2.5
        z_i_charge = 4
    

    for z_i in range(20):
        curr_t_do = last_t_do
        curr_t_up = last_t_up

        save_type = False
        if z_i < z_i_charge:
            save_type = True
        
        # 首次拟合
        c0_z_mean, c0_z_std, chi2mean = fit_ratio_FeynmenHellman(
            z_i, nr_, curr_t_do, curr_t_up, Pz_, conf_, note_name_,
            t_sep_do, t_sep_up,
            save_csv=save_type
        )

        if z_i > z_i_charge - 1:
            # 中间迭代全部不保存文件
            while True:
                chi2_prev = chi2_history[0]
                chi2_ratio = chi2mean #/ chi2_prev

                if (curr_t_do - 1 < nr_ * 2):
                    if curr_t_do  == t_sep_do:
                        break
                    if (chi2_ratio > ratio_limit):
                        curr_t_do += 1
                        curr_t_up += 1
                    break


                # 比值超限，右移区间
                if chi2_ratio > chi2_ratio_upper_limit:
                    if curr_t_do < last_t_do:
                        curr_t_do += 1
                        curr_t_up += 1
                        break
                    else:
                        break

                # 比值偏小，左移区间继续尝试
                curr_t_do -= 1
                curr_t_up -= 1
                c0_z_mean, c0_z_std, chi2mean = fit_ratio_FeynmenHellman(
                    z_i, nr_, curr_t_do, curr_t_up, Pz_, conf_, note_name_,
                    t_sep_do, t_sep_up,
                    save_csv=False
                )
            
            # ========== 循环结束，当前 curr_t_do / curr_t_up 为最终区间，单独执行一次并保存CSV ==========
            c0_z_mean, c0_z_std, chi2mean = fit_ratio_FeynmenHellman(
                z_i, nr_, curr_t_do, curr_t_up, Pz_, conf_, note_name_,
                t_sep_do, t_sep_up,
                save_csv=True
            )

        # 保存当前 z 最终结果到列表
        res_list.append({
            'z': z_i,
            'c0_z_mean': c0_z_mean,
            'c0_z_std': c0_z_std,
            'chi2_mean': chi2mean
        })
        chi2_history.append(chi2mean)

        # 传给下一轮z作为初始t_sep
        last_t_do = curr_t_do
        last_t_up = curr_t_up

        # 总结果 csv 写入
        fit_result = pd.DataFrame(res_list)
        ratio_fit_path = f'{this_path}/result/ratio_fit_result_FeynmenHellman/{conf_}{note_name_}_pz{Pz_}/ex{nr_}/tsep_{t_sep_do}_{t_sep_up}/'
        os.makedirs(ratio_fit_path, exist_ok=True)
        fit_result.to_csv(f'{ratio_fit_path}/c0_vs_z_{t_sep_do}_{t_sep_up}.csv', index=False)

    return 0

# ===================== MPI 并行入口函数（核心改造） =====================
def run_task_by_rank():
    # 1. 定义所有组态、标签、区间、nr、动量列表（和原main_pzN完全一致）
    conf_set = [
        'L24x72', 'L32x64_C32P23', 'L32x64_C32P29', 'L48x96_C48P14',
        'L32x64', 'L32x96', 'L36x108', 'L48x144'
    ]
    note_name_set = [
        '_dhxmeang1', '_plus', '', '_moms4',
        '_dhxplusg1', '_dhx', '', '_dhx'
    ]
    t_sep_star_set = [7, 7, 7, 6, 8, 9, 10, 12]
    t_sep_end_set = [10, 10, 10, 9, 11, 11, 12, 15]
    nr_test_set = [2, 2, 2, 2, 2, 4, 4, 6]
    Pz_conf_set = [
        [4, 5, 6], [4, 5, 6], [4, 5, 6], [7, 8, 9, 10],
        [4, 5, 6], [4, 5, 6], [4, 5, 6], [4, 5, 6]
    ]
    #chi2_limt_set = [0.6, 1.0, 2.0, 1.0, 1.0, 0.6, 0.4, 1.0]
    #chi2_limt_set = [0.8, 0.9, 1.0, 1.0, 1.0, 0.6, 0.4, 1.0]
    chi2_limt_set = [1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0]
    # 2. 平铺所有任务
    task_list = []
    for idx in range(len(conf_set)):
        conf = conf_set[idx]
        note = note_name_set[idx]
        t_star = t_sep_star_set[idx]
        t_end = t_sep_end_set[idx]
        nr = nr_test_set[idx]
        pz_list = Pz_conf_set[idx]
        chi2_lim = chi2_limt_set[idx]

        for pz in pz_list:
            task_list.append((conf, note, t_star, t_end, nr, pz, chi2_lim))

    total_tasks = len(task_list)
    if rank < total_tasks:
        task = task_list[rank]
        conf, note, t_star, t_end, nr, pz, chi2_lim = task

        # ========== 新增：每个任务单独日志文件 ==========
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        # 日志名格式：conf_pz{动量}_rank{进程号}.log
        log_name = f"{conf}_pz{pz}_rank{rank}.log"
        log_path = os.path.join(log_dir, log_name)

        # 重定向 stdout 到当前任务专属日志
        import sys
        sys.stdout = open(log_path, "w", encoding="utf-8")

        print(f"===== Rank {rank} | Task Start: conf={conf}, Pz={pz} =====")
        c0_vs_z_FeynmenHellman(nr, t_star, t_end, pz, conf, note, chi2_lim)
        print(f"===== Rank {rank} | Task Finish: conf={conf}, Pz={pz} =====")

        # 关闭日志文件
        sys.stdout.close()
    else:
        print(f"Rank {rank}: No task to run")

if __name__ == "__main__":
    run_task_by_rank()