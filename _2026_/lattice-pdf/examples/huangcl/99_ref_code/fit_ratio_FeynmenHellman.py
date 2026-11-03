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

import pdb # 用于调试代码 pdb.set_trace()


file_path =  '/public/group/imp/zengch/LQCD/renorma/result/Ratio_data/FeynmanHellman'


def fit_ratio_FeynmenHellman(z_fit, nr_, t_set_do, t_set_up, Pz_, conf_, note_name_):
    # 读取 npz 文件（包含数据 + 真实 t_sep）
    file_read = f'{file_path}/Delta_Ratio_data_{conf_}{note_name_}_pz{Pz_}_nr{nr_}.npz'
    data = np.load(file_read)
    
    delta = data['delta']          # 数据 [z, t_sep_idx, samples]
    t_sep_vals = data['t_sep_vals']  # 真实 t_sep 数值 [4,5,6,...]
    
    # ✅ 关键：用真实 t_sep 数值找到对应索引
    mask = (t_sep_vals >= t_set_do) & (t_sep_vals <= t_set_up)
    c0_data = delta[z_fit, mask, :]  # 只取真实 t_sep 在 [do, up] 之间的数据
    
    std_err = np.std(c0_data, axis=1)
    n_data = np.shape(c0_data)[0]
    n_samples = np.shape(c0_data)[1]
    fit_results_list = []
    c0_in, chi_in = fit_ratio_mean_FeynmenHellman(z_fit, nr_, t_set_do, t_set_up, Pz_, conf_, note_name_)
    c_inv = np.linalg.inv(covariance_matrix(c0_data, 'boot'))

    
    #pdb.set_trace()

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

    fit_results = pd.DataFrame(fit_results_list)
    c0_save_path = f'{this_path}/result/ratio_tsep_tisep_FeynmenHellman/{conf_}{note_name_}_pz{Pz_}/ex{nr_}'
    os.makedirs(c0_save_path, exist_ok=True)
    fit_results.to_csv(f"{c0_save_path}/z{z_fit}_tsep{t_set_do}_{t_set_up}.csv", index=False)
    
    c0_mean = np.mean(fit_results['c0'])
    c0_std = np.std(fit_results['c0'])
    return c0_mean, c0_std, chi_in

def fit_ratio_mean_FeynmenHellman(z_fit, nr_, t_set_do, t_set_up, Pz_, conf_, note_name_):
    # 读取 npz 文件（包含数据 + 真实 t_sep）
    file_read = f'{file_path}/Delta_Ratio_data_{conf_}{note_name_}_pz{Pz_}_nr{nr_}.npz'
    data = np.load(file_read)
    
    delta = data['delta']
    t_sep_vals = data['t_sep_vals']
    
    # ✅ 关键：用真实 t_sep 数值筛选
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

        #chi2_all =np.sum( (r_c) **2. / std_err **2.)

        #print(c0_data_mean)
        #print(c0_th)
        #print(r_c)
        #print(chi2_all/ len(r_c))

        #pdb.set_trace()

        #print(m.nfit)

        return chi2_all / (len(r_c) - m.nfit) 
    
    c0_in = 0.54
    m = Minuit(cost_function, c0_th=c0_in)
    m.migrad()
    c0 = m.values['c0_th']
    chi2_mean = cost_function(c0)

    print('z=', z_fit)
    print('c0=', c0)
    print('chi2_mean=', chi2_mean)
    return c0, chi2_mean

def c0_vs_z_FeynmenHellman(nr_, t_sep_do, t_sep_up, Pz_, conf_, note_name_):

    #t_sep_list = [8, 9, 10, 11, 12, 13, 14] 
    #n_remove =  3 # 拟合时去掉头尾的点数

    res_list = []
    for z_i in range(20):

        c0_z_mean, c0_z_std, chi2mean = fit_ratio_FeynmenHellman(z_i, nr_, t_sep_do, t_sep_up,  Pz_, conf_, note_name_)
       
        res_list.append({ 'z'       : z_i,
                         'c0_z_mean': c0_z_mean, 
                         'c0_z_std' : c0_z_std,
                         'chi2_mean': chi2mean               
        }
        )

        fit_result  = pd.DataFrame(res_list)
        ratio_fit_path = f'{this_path}/result/ratio_fit_result_FeynmenHellman/{conf_}{note_name_}_pz{Pz_}/ex{nr_}'
        os.makedirs(ratio_fit_path, exist_ok=True)
        fit_result.to_csv(f'{ratio_fit_path}/c0_vs_z_{t_sep_do}_{t_sep_up}.csv', index=False)
        #fit_result.to_csv(f'ratio_fit_result/a_test'+ data_name +f'_ex{n_remove_}_test{t_sep_list_[0]}_{t_sep_list_[-1]}.csv', index=False)
     
    return 0

def main_pz0():

    conf_set =       ['L24x72', 'L32x64', 'L32x96', 'L36x108', 'L48x144']
    note_name_set =  ['',        '',       '',       '',        '_dhx0']
    t_sep_star_set = [7,         9,        9,        9,         12]
    t_sep_end_set  = [14,        16,       17,       19,         24]
    nr_test_set    = [0,         0,        0,        0,          0] 
    Pz = 0


    #conf_set =       ['L32x64_C32P29', 'L32x64', 'L32x96', 'L36x108', 'L48x144']
    #note_name_set =  ['',        '',       '',       '',        '_dhx0']
    #t_sep_star_set = [7,         8,        8,        9,          12]
    #t_sep_end_set  = [13,        14,       14,       15,         22]
    #nr_test_set    = [3,         3,        4,        4,          6] 


    conf_set =       ['L36x108']
    note_name_set =  [   '',   ]
    t_sep_star_set = [   9,   ]
    t_sep_end_set  = [   19,   ]
    nr_test_set    = [    0,  ] 
    Pz = 0




    #fit_ratio_mean_FeynmenHellman(0, nr_test, t_sep_star, t_sep_end, Pz, conf)
    #fit_ratio_FeynmenHellman(0, nr_test, t_sep_star, t_sep_end, Pz, conf)
    for i in range(len(conf_set)):
        conf = conf_set[i]
        note_name = note_name_set[i]
        t_sep_star = t_sep_star_set[i]
        t_sep_end =  t_sep_end_set[i]
        nr_test   = nr_test_set[i]


        c0_vs_z_FeynmenHellman(nr_test, t_sep_star, t_sep_end, Pz, conf, note_name)

    return 0 

def main_pzN():

  
  
    #Pz_set = [3, 4]

    
    #conf_set =       ['L24x72',   'L32x64_C32P23',  'L32x64_C32P29',  'L32x64', 'L32x96', 'L48x144']
    #note_name_set =  ['_dhxmeang1',        '_plus',      '',        '_dhxplusg1', '_dhx',  '_dhx']
    #t_sep_star_set = [7,         7,   7,  8,  9 , 12]
    #t_sep_end_set  = [13,        13,  13,  15, 17, 24]
    #nr_test_set    = [2,         3,   3,  3,  4,   6] 

    conf_set =       ['L24x72',   'L32x64_C32P23',    'L32x64_C32P29',   'L48x96_C48P14',  'L32x64', 'L32x96','L36x108', 'L48x144']
    note_name_set =  ['_dhxmeang1',        '_plus',      '',                '_moms4',      '_dhxplusg1', '_dhx', '',  '_dhx']
    #t_sep_star_set = [7,   7,   7,  6,   9,    9 ,  12]
    #t_sep_end_set  = [14,  14,  14, 14,  16,   17,  24]
    #nr_test_set    = [0,   0,   0,  0,   0,    0,    0]
    t_sep_star_set = [7,   7,   7,  6,   9,    9 , 10,  12]
    t_sep_end_set  = [10,  10,  10, 9,  12,   12,  13, 15]
    nr_test_set    = [2,   2,   2,  2,  3,    4,   4,  5]            
    Pz_conf_set    = [ [3, 4, 5, 6],  [3, 4, 5, 6], [3, 4, 5, 6], [7, 8, 9, 10], [3, 4, 5, 6], [3, 4, 5, 6],[3, 4, 5, 6], [3, 4, 5, 6]]
    

    
    conf_set =       [ 'L24x72'   ]  
    note_name_set =  [ '_dhxmeang1']
    t_sep_star_set = [7 ]
    t_sep_end_set  = [10 ]
    nr_test_set    = [ 2 ] 
    Pz_conf_set    = [ [3, 4, 5, 6]]
    

    

  





    #fit_ratio_mean_FeynmenHellman(0, nr_test, t_sep_star, t_sep_end, Pz, conf)
    #fit_ratio_FeynmenHellman(0, nr_test, t_sep_star, t_sep_end, Pz, conf)
    for i in range(len(conf_set)):
        Pz_set = Pz_conf_set[i]
        for Pz in Pz_set:
            conf = conf_set[i]
            note_name = note_name_set[i]
            t_sep_star = t_sep_star_set[i]
            t_sep_end =  t_sep_end_set[i]
            nr_test   = nr_test_set[i]


            c0_vs_z_FeynmenHellman(nr_test, t_sep_star, t_sep_end, Pz, conf, note_name)

    return 0



if __name__ == "__main__":

    #main_pz0()
    main_pzN()

    