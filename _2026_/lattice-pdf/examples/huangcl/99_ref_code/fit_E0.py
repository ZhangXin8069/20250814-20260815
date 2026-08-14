import sys
sys.path.append('/public/home/zengch/LQCD/input_file')
sys.path.append('/public/home/zengch/LQCD/tool')
sys.path.append('/public/home/zengch/All_TMD_dependence')
import pandas as pd
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
import time
import matplotlib.pyplot as plt
import os
from tool import *
from constant import CA, CF, gammaE, pi, A_s
import pdb # 用于调试代码 pdb.set_trace()
from hB_data import load_hB_data, a_len_set,  Nl_set, fm_to_GeV # fm / fm_to_GeV = GeV^{-1}
from C_pt_load import C2pt, load_C2pt_deltat

input_name = 'L32x64'

Nl    = Nl_set[input_name]
a_GeV = a_len_set[input_name]


 # 拟合的参数化形式
def th_E0(Pz_,  m_, k2_, k3_):

    return (m_ **2. + k2_ * Pz_**2. + k3_ * Pz_ **4. * a_GeV **2.) ** 0.5

def fit_E0_mean(par_ini): # ini, 拟合开始的t, fin, 拟合介绍的t

    data = pd.read_csv(f'par_2pt/{input_name}.csv')
    E0_Pz0 = data['E0'] / a_GeV
    data = pd.read_csv(f'par_2pt/{input_name}_pz2.csv')
    E0_Pz2 = data['E0'] / a_GeV
    data = pd.read_csv(f'par_2pt/{input_name}_pz3.csv')
    E0_Pz3 = data['E0'] / a_GeV
    data = pd.read_csv(f'par_2pt/{input_name}_pz4.csv')
    E0_Pz4 = data['E0'] / a_GeV
    data = pd.read_csv(f'par_2pt/{input_name}_pz5.csv')
    E0_Pz5 = data['E0'] / a_GeV

    Pz_set = np.array([0,      2,      3,      4,       5])
    Pz_GeV    = Pz_set * 2.* pi / (Nl * a_GeV) 

    E0_set = np.array([E0_Pz0, E0_Pz2, E0_Pz3, E0_Pz4,  E0_Pz5])

    E0_mean = np.mean(E0_set, axis=1)
    E0_std  = np.std(E0_set, axis=1)
   
    
    c_inv     = covariance_matrix_inv(E0_set, 'boot')
    cnot_inv  = np.linalg.inv(np.diag(E0_std**2.))

    #visualize_matrix_num( np.linalg.inv(cnot_inv),   'test')

    data_num = len(Pz_set)

    #visualize_matrix_num( c_inv_L24x72,   'cov_yes')

    print(Pz_GeV)
    print(E0_mean)
    print(E0_std)

    
    def cost_function(par_set_):
        m_, k2_, k3_ = par_set_
            
        th_ = th_E0(Pz_GeV,  m_, k2_, k3_) 
        del_ = (th_ - E0_mean)
        chi2 = del_.T @ cnot_inv @ del_
       
        mean_chi2 = chi2 / data_num

        
        #pdb.set_trace()


        return mean_chi2
    
    
     
    par_name = ('m',   'k2',   'k3')
    
    
    m = Minuit(cost_function, par_ini, name=par_name)

    m.limits["m"] = (None, None)  
    m.limits["k2"] = (None, None) 
    m.limits["k3"] = (None, None)  
 

    #m.fixed['k3']= True
    
            
    
    m.migrad()

    chi_all_mean = cost_function(m.values)
  
    m_fit, k2_fit, k3_fit = m.values

    print('m_ini =', m_fit)
    print('k2_ini =', k2_fit)
    print('k3_ini =', k3_fit)
    print('chi2_mean =', chi_all_mean)
   
    return m.values

def fit_E0(par_ini): # ini, 拟合开始的t, fin, 拟合介绍的t

    data = pd.read_csv(f'par_2pt/{input_name}.csv')
    E0_Pz0 = data['E0'] / a_GeV
    data = pd.read_csv(f'par_2pt/{input_name}_pz2.csv')
    E0_Pz2 = data['E0'] / a_GeV
    data = pd.read_csv(f'par_2pt/{input_name}_pz3.csv')
    E0_Pz3 = data['E0'] / a_GeV
    data = pd.read_csv(f'par_2pt/{input_name}_pz4.csv')
    E0_Pz4 = data['E0'] / a_GeV
    data = pd.read_csv(f'par_2pt/{input_name}_pz5.csv')
    E0_Pz5 = data['E0'] / a_GeV

    Pz_set = np.array([0,      2,      3,      4,       5])
    Pz_GeV    = Pz_set * 2.* pi / (Nl * a_GeV) 

    E0_set = np.array([E0_Pz0, E0_Pz2, E0_Pz3, E0_Pz4,  E0_Pz5])

    E0_mean = np.mean(E0_set, axis=1)
    E0_std  = np.std(E0_set, axis=1)
   
    
    c_inv     = covariance_matrix_inv(E0_set, 'boot')
    cnot_inv  = np.linalg.inv(np.diag(E0_std**2.))

    #visualize_matrix_num( np.linalg.inv(cnot_inv),   'test')

    data_num = len(Pz_set)

    #visualize_matrix_num( c_inv_L24x72,   'cov_yes')

    #print(Pz_GeV)
    #print(E0_mean)
    #print(E0_std)

    fit_results_list = []

    for i in range(np.shape(E0_set)[1]):

        def cost_function(par_set_):
            m_, k2_, k3_ = par_set_
                
            th_ = th_E0(Pz_GeV,  m_, k2_, k3_) 
            del_ = (th_ - E0_set[:, i])
            chi2 = del_.T @ cnot_inv @ del_
        
            mean_chi2 = chi2 / data_num

            
            #pdb.set_trace()


            return mean_chi2
        
        
        
        par_name = ('m',   'k2',   'k3')
        
        
        m = Minuit(cost_function, par_ini, name=par_name)

        m.limits["m"] = (None, None)  
        m.limits["k2"] = (None, None) 
        m.limits["k3"] = (None, None)  
    

        #m.fixed['k3']= True
        
                
        
        m.migrad()

        chi_all_mean = cost_function(m.values)
    
        m_fit, k2_fit, k3_fit = m.values


        fit_results_list.append({
            "sample_i": i,
            "m": m_fit,
            "k2": k2_fit,
            "k3": k3_fit,
            "chi2" : chi_all_mean
        })

        # 保存拟合结果到 CSV 文件
        fit_results = pd.DataFrame(fit_results_list)
        fit_results.to_csv(f"par_E0/{input_name}.csv", index=False)
   
    return 0



if __name__ == "__main__":
   
    
    m_ini = 1.0137366990819108
    k2_ini = 0.4629363848738383
    k3_ini = -0.2028755628447637
    

    par_ini = [m_ini, k2_ini, k3_ini] 

    par_ini = np.array(par_ini)


    #print(par_ini)

    fit_E0(par_ini)
    #fit_2pt_mean(par_ini, 9, 14, 'L32x64_pz4')



    #print(th_log_hB(z_test, 0.1, 2, par_set))


