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
from scipy.interpolate import interp1d
import pdb # 用于调试代码 pdb.set_trace()
from C_pt_load import this_path
import glob


fm_to_GeV = 0.197

data_name_set = ['C24P29', 'L24x72',       'L32x64',        'L32x96',    'L48x144', 'L36x108',  'L32x64_C32P23', 'L32x64_C32P29', 'L48x96_C48P14']

a_len_set     = {'C24P29':0.105/fm_to_GeV, 'L24x72':0.105/fm_to_GeV,      'L32x64':0.0897/fm_to_GeV,     'L32x96':0.0775/fm_to_GeV,  'L48x144':0.0519/fm_to_GeV,  
                  'L36x108':0.0688/fm_to_GeV, 'L32x64_C32P23':0.105/fm_to_GeV, 'L32x64_C32P29':0.105/fm_to_GeV, 'L48x96_C48P14':0.105/fm_to_GeV}

Nl_set        = {'C24P29':24, 'L24x72':24, 'L32x64': 32, 'L32x96':32, 'L48x144':48, 'L36x108': 36, 'L32x64_C32P23':32, 'L32x64_C32P29':32, 'L48x96_C48P14':48}

pion_mass_set  = {'C24P29':0.293, 'L24x72':0.293, 'L32x64': 0.285, 'L32x96':0.303, 'L36x108':0.297, 'L48x144':0.317, 
                 'L32x64_C32P23':0.228, 'L32x64_C32P29':0.292, 'L48x96_C48P14':0.136}


def hB_data(conf, note_name, Pz_, exn, tsep_star, tsep_end):

    
    data_name = f'{conf}{note_name}_pz{Pz_}'


    hb_z_set = []
    z_set    = []

    a_len     = a_len_set[conf]
    
    save_path = f'{this_path}/result/hB_data/new_res/{data_name}_FeynmenHellman_ex{exn}_tsep{tsep_star}_{tsep_end}.npz'
    #os.makedirs(save_path, exist_ok=True)

    for z_i in range(20):
        file_dir = f'{this_path}/result/ratio_tsep_tisep_FeynmenHellman/{data_name}/ex{exn}/{tsep_star}_{tsep_end}/'
        fit_file_list = glob.glob(os.path.join(file_dir, f'z{z_i}_tsep*_*.csv'))
    
        if not fit_file_list:
            print(f"警告: z{z_i} 未找到匹配csv文件")
            return None, None, None, None
            
        file_path = fit_file_list[0]


        data = pd.read_csv(file_path)
        hb = data['c0'].values
        #log_hb = np.log(hb)

        hb_z_set.append(hb)

        #print(z_i, hb.shape)
       
        z_set.append(z_i * a_len * fm_to_GeV) # fm
    

    #pdb.set_trace()
    hb_z_set_zn = np.array(hb_z_set)

    
    
    #pdb.set_trace()
    z_set    = np.array(z_set)

    # z_0 归一化
    hb_z_set_z0 = hb_z_set_zn[0:1, :]
    hb_z_set    = hb_z_set_zn / hb_z_set_z0

    #pdb.set_trace()

    interpolation_function = interp1d(z_set, hb_z_set, kind='linear', axis=0)
   
    #print(z_set)
    
    z_set_new    = np.arange(0.15,   1.0 + 0.05, 0.05)

    if conf =='L48x144':
        z_set_new    = np.arange(0.15,   0.95 + 0.05, 0.05)
   
    hb_z_set_new = interpolation_function(z_set_new)

    

    hb_z_set_new_log = np.log(hb_z_set_new)

   
    np.savez(save_path, 
             z = z_set_new,                  # 插值以后的 z (fm)
             loghB = hb_z_set_new_log,       # 插值以后的 loghB  已归一化
             hB = hb_z_set_new,              # 插值以后的 hB     已归一化
             z_o= z_set,                     # 原本的 z (fm)
             hB_o = hb_z_set,                # 原本的 hB      已归一化
             hB_o_zn = hb_z_set_zn)          # 原本的 hB      未归一化
    
    #pdb.set_trace()
   

    return  z_set_new, hb_z_set_new # z fm

def load_hB_data_FeynmenHellman_new(data_name, exn, tsep_star, tsep_end):

    save_path = f'{this_path}/result/hB_data/new_res/{data_name}_FeynmenHellman_ex{exn}_tsep{tsep_star}_{tsep_end}.npz'
    all_data = np.load(save_path)
    return all_data 

def test():

    
   
    Pz_set = [4, 5, 6]
    for Pz in Pz_set:
        #hB_data(f'L24x72', '', Pz , 0,  7, 14)
        #hB_data(f'L32x64', '', Pz , 0,  9, 16)
        #hB_data(f'L32x96', '', Pz , 0,  9,  17)
        #hB_data(f'L36x108', '', Pz , 0,  9,  19)
        #hB_data(f'L48x144', '_dhx0', Pz , 0,  12, 24)

        

        hB_data(f'L24x72', '_dhxmeang1', Pz , 2,  7, 10)
        hB_data(f'L32x64_C32P23', '_plus', Pz , 2,  7, 10)
        hB_data(f'L32x64_C32P29', '', Pz , 2,  7, 10)
        #hB_data(f'L48x96_C48P14', '_moms4', Pz , 2,  6, 9)
        hB_data(f'L32x64', '_dhxplusg1', Pz , 2,  8, 11)
        hB_data(f'L32x96', '_dhx', Pz , 4,  9, 11)
        hB_data(f'L36x108', '', Pz , 4,  10, 12)
        hB_data(f'L48x144', '_dhx', Pz , 6,  12, 15)

    Pz_set = [7, 8, 9, 10]
    for Pz in Pz_set:
        hB_data(f'L48x96_C48P14', '_moms4', Pz , 2,  6, 9)
        
        
        



  

    return 0

if __name__ == "__main__":
    test()
    #hB_data(f'L24x72_pz{Pz}', 8, 16)
    #hB_data(f'L32x64_pz{Pz}', 8, 16)
    #hB_data(f'L32x96_pz{Pz}', 8, 16)
    #hB_data('L48x144', 0,  5,  12, 20)

    #Pz = 0
    #hB_data('L24x72',  Pz,  3,  6,  12)
    #hB_data('L24x72',  Pz,  3,  7,  13)
    #hB_data('L32x64',  Pz,  3,  7,  13)
    #hB_data('L32x64',  Pz,  3,  8,  14)
    #hB_data('L32x96',  Pz,  4,  8,  14)
    #hB_data('L36x108',  Pz,  4,  9,  15)
    #hB_data('L48x144', Pz,  6,  12, 18)

    
   

    #data = load_hB_data_FeynmenHellman_new('L32x96_dhx_pz4' , 4,  9, 12)
    #print(data['z'], np.exp(data['loghB'][:, 0]))
    #print(data['z_o'], data['hB_o'][:, 0])

