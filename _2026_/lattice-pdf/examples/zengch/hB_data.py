import sys
sys.path.append('/public/home/zengch/LQCD/input_file')
sys.path.append('/public/home/zengch/LQCD/tool')

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


fm_to_GeV = 0.197

data_name_set = ['L24x72',       'L32x64',        'L32x96',    'L48x144', 'L36x108',  'L32x64_C32P23']

a_len_set     = {'L24x72':0.105/fm_to_GeV,      'L32x64':0.0897/fm_to_GeV,     'L32x96':0.0775/fm_to_GeV,  'L48x144':0.0519/fm_to_GeV,  'L36x108':0.0688/fm_to_GeV, 'L32x64_C32P23':0.105/fm_to_GeV}
Nl_set        = {'L24x72':24, 'L32x64': 32, 'L32x96':32, 'L48x144':48, 'L36x108': 36, 'L32x64_C32P23':32}



def hB_data(conf, Pz_, exn, tsep_star, tsep_end):

    
    data_name = f'{conf}_dhxmean_pz{Pz_}'
    if Pz_ == 0:
        data_name = f'{conf}_pz{Pz_}'

    hb_z_set = []
    z_set    = []

    a_len     = a_len_set[conf]
    
    save_path = f'{this_path}/result/hB_data/{data_name}_ex{exn}_tsep{tsep_star}_{tsep_end}.npz'
    #os.makedirs(save_path, exist_ok=True)

    for z_i in range(20):
        file_path = f'{this_path}/result/ratio_tsep_tisep/{data_name}/ex{exn}/boot_tsep{tsep_star}_{tsep_end}_z{z_i}.csv'
        data = pd.read_csv(file_path)
        hb = data['c0_z1'].values
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
 
    #z_set_new    = np.arange(0.15,   0.75 + 0.05, 0.05)
    z_set_new    = np.arange(0.15,   1.0 + 0.05, 0.05)
    #z_set_new    = np.arange(0.15,   0.75 + 0.05, 0.05)
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

def load_hB_data(data_name, exn, tsep_star, tsep_end):

    save_path = f'{this_path}/result/hB_data/{data_name}_ex{exn}_tsep{tsep_star}_{tsep_end}.npz'
    all_data = np.load(save_path)
    return all_data 

def test():
    #Pz = 0
    
   

    Pz = 2
    hB_data(f'L24x72',Pz , 3,  7, 14)
    #hB_data(f'L32x64',Pz , 3,  7, 14)
    #hB_data(f'L32x96',Pz , 3,  8, 14)
    #hB_data(f'L48x144',Pz , 4,  12, 18)
    #hB_data('L32x64_C32P23', Pz, 2,  6, 14)


    Pz = 3
    hB_data(f'L24x72',Pz , 3,  7, 12)
    #hB_data(f'L32x64',Pz , 3,  7, 14)
    #hB_data(f'L32x96',Pz , 3,  8, 14)
    #hB_data(f'L48x144',Pz , 4,  12, 18)
    #hB_data('L32x64_C32P23', Pz, 2,  6, 12)


    Pz = 4
    hB_data(f'L24x72',Pz , 2,  6, 10)
    #hB_data(f'L32x64',Pz , 3,  7, 12)
    #hB_data(f'L32x96',Pz , 3,  8, 14)
    #hB_data(f'L48x144',Pz , 4,  12, 16)
    #hB_data('L32x64_C32P23', Pz, 2,  6, 11)


    Pz = 5
    #hB_data(f'L24x72',Pz , 2,  6, 9)
    #hB_data(f'L32x64',Pz , 3,  7, 12)
    #hB_data(f'L32x96',Pz , 3,  8, 12)
    #hB_data(f'L48x144',Pz , 4,  12, 16)
    #hB_data('L32x64_C32P23', Pz, 2,  6, 10)


    Pz = 6
    #hB_data(f'L24x72',Pz , 2,  6, 8)
    #hB_data(f'L32x64',Pz , 3,  7, 12)
    #hB_data(f'L32x96',Pz , 3,  8, 12)
    #hB_data(f'L48x144',Pz , 4,  12, 14)

    return 0

if __name__ == "__main__":
    test()
    #hB_data(f'L24x72_pz{Pz}', 8, 16)
    #hB_data(f'L32x64_pz{Pz}', 8, 16)
    #hB_data(f'L32x96_pz{Pz}', 8, 16)
    #hB_data('L48x144', 0,  5,  12, 20)

    #Pz = 0
    #hB_data('L24x72',  Pz,  3,  6,  16)
    #hB_data('L32x64',  Pz,  3,  7,  20)
    #hB_data('L32x96',  Pz,  4,  8,  19)
    #hB_data('L36x108',  Pz,  4,  9,  20)
    #hB_data('L48x144', Pz,  5,  12, 23)

    
   

    #data = load_hB_data('L24x72_pz0')
    #print(data['z'], np.exp(data['loghB'][:, 0]))
    #print(data['z_o'], data['hB_o'][:, 0])

    #data = load_hB_data('L32x64_pz0')
    #print(data['z'], np.exp(data['loghB'][:, 0]))

    #data = load_hB_data('L32x96_pz0')
    #print(data['z'], np.exp(data['loghB'][:, 0]))