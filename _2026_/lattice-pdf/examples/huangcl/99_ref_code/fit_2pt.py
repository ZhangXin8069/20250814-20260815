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
from hB_data import load_hB_data, a_len_set, fm_to_GeV # fm / fm_to_GeV = GeV^{-1}
from C_pt_load import C2pt, load_C2pt_deltat

 # 拟合的参数化形式
def th_2pt(t_,  c4_, c5_, E0_, E_delta_):
   
    return c4_ * np.exp(- E0_ * t_) * (1. + c5_ * np.exp(- E_delta_ * t_) )

def fit_2pt_mean(par_ini, ini, fin, input_name): # ini, 拟合开始的t, fin, 拟合介绍的t

    data_2pt = load_C2pt_deltat(f'delta_matrix/{input_name}_2pt_deltat_matrix.npy').T
    ndelta_t = np.shape(data_2pt )[1]

    t_delta  = np.arange(ndelta_t)
    mean_2pt = np.mean(data_2pt , axis= 1)
    std_2pt  = np.std(data_2pt , axis= 1)

    data_2pt_fit = data_2pt[ini:fin,:]
    t_fit        = t_delta[ini:fin]
    mean_2pt_fit = mean_2pt[ini:fin]
    std_2pt_fit  = std_2pt[ini:fin]
    c_inv_2pt    = covariance_matrix_inv(data_2pt_fit, 'boot')
    cnot_inv_2pt = np.linalg.inv(np.diag(std_2pt_fit**2.))

    #visualize_matrix_num( np.linalg.inv(c_inv_2pt),   'test')

    data_num = len(t_fit)

    #visualize_matrix_num( c_inv_L24x72,   'cov_yes')

    #pdb.set_trace()


    def cost_function(par_set_):
        c4_, c5_, E0_, E_delta_ = par_set_
            
        th_ = th_2pt(t_fit, c4_, c5_, E0_, E_delta_) 
        del_2pt = (th_ - mean_2pt_fit)
        chi2 = del_2pt.T @ c_inv_2pt @ del_2pt
       
        mean_chi2 = chi2 / data_num

        
        #pdb.set_trace()


        return mean_chi2
    
    def chi2_cal(par_set_):
        c4_, c5_, E0_, E_delta_ = par_set_
            
        th_ = th_2pt(t_fit, c4_, c5_, E0_, E_delta_) 
        del_2pt = (th_ - mean_2pt_fit)
        chi2 = del_2pt.T @ cnot_inv_2pt @ del_2pt

        
        
       
        mean_chi2 = chi2 / data_num

        #pdb.set_trace()


        return mean_chi2
     
    par_name = ('c4',   'c5',   'E0',   'E_delta')
    
    
    m = Minuit(cost_function, par_ini, name=par_name)

    m.limits["c4"] = (None, None)  
    m.limits["c5"] = (None, None) 
    m.limits["E0"] = (0, None)  
    m.limits["E_delta"] = (0, None) 

    #m.fixed['c4']= True
    
            
    
    m.migrad()

    #chi_all_mean = cost_function(m.values)
    chi_all_mean = chi2_cal(m.values)
    c4_fit, c5_fit, E0_fit, E_delta_fit = m.values

    print('c4_ini =', c4_fit)
    print('c5_ini =', c5_fit)
    print('E0_ini =', E0_fit)
    print('E_delta_ini =', E_delta_fit)
    print('chi2_mean =', chi_all_mean)
   
    return m.values

def fit_2pt(par_ini_0, ini, fin, input_name):

    #input_name = 'L32x64_pz5'

    #ini = 9
    #fin = 14

    data_2pt = load_C2pt_deltat(f'delta_matrix/{input_name}_2pt_deltat_matrix.npy').T
    ndelta_t = np.shape(data_2pt )[1]

    t_delta  = np.arange(ndelta_t)
    mean_2pt = np.mean(data_2pt , axis= 1)
    std_2pt  = np.std(data_2pt , axis= 1)

    data_2pt_fit = data_2pt[ini:fin,:]
    t_fit        = t_delta[ini:fin]
    mean_2pt_fit = mean_2pt[ini:fin]
    std_2pt_fit  = std_2pt[ini:fin]
    c_inv_2pt    = covariance_matrix_inv(data_2pt_fit, 'boot')
    cnot_inv_2pt = np.linalg.inv(np.diag(std_2pt_fit**2.))


    par_ini = fit_2pt_mean(par_ini_0, ini, fin, input_name)

    #visualize_matrix_num( np.linalg.inv(c_inv_2pt),   'test')

    data_num = len(t_fit)

    #visualize_matrix_num( c_inv_L24x72,   'cov_yes')

    #pdb.set_trace()
    fit_results_list = []

    for i in range(data_2pt_fit.shape[1]):
        def cost_function(par_set_):
            c4_, c5_, E0_, E_delta_ = par_set_
                
            th_ = th_2pt(t_fit, c4_, c5_, E0_, E_delta_) 
            del_2pt = (th_ - data_2pt_fit[:, i])
            chi2 = del_2pt.T @ c_inv_2pt @ del_2pt
        
            mean_chi2 = chi2 / data_num

            
            #pdb.set_trace()


            return mean_chi2
        

        
        par_name = ('c4',   'c5',   'E0',   'E_delta')
        
        
        m = Minuit(cost_function, par_ini, name=par_name)

        m.limits["c4"] = (None, None)  
        m.limits["c5"] = (None, None) 
        m.limits["E0"] = (0, None)  
        m.limits["E_delta"] = (0, None) 

        #m.fixed['c4']= True
        
                
        
        m.migrad()

        chi_all_mean = cost_function(m.values)
        c4_fit, c5_fit, E0_fit, E_delta_fit = m.values

        fit_results_list.append({
            "sample_i": i,
            "c4": c4_fit,
            "c5": c5_fit,
            "E0": E0_fit,
            "E_delta_fit": E_delta_fit,
            "chi2" : chi_all_mean
        })

        # 保存拟合结果到 CSV 文件
        fit_results = pd.DataFrame(fit_results_list)
        fit_results.to_csv(f"par_2pt/{input_name}.csv", index=False)
   
    return m.values


input_name = 'L32x64'
a_GeV = a_len_set[input_name]

if __name__ == "__main__":
   
    # pz = 0
    c4_ini = 0.013361275315495368
    c5_ini = 1.2977301597692157
    E0_ini = 0.47031201594892524
    E_delta_ini = 0.3951952425378958
    par_ini = [c4_ini, c5_ini, E0_ini, E_delta_ini] 
    fit_2pt(par_ini, 6, 22, 'L32x64')

    # pz = 2
    c4_ini = 0.007191326329180693
    c5_ini = 1.3711349707078369
    E0_ini = 0.6110508387546296
    E_delta_ini = 0.3546087492591182
    par_ini = [c4_ini, c5_ini, E0_ini, E_delta_ini] 
    fit_2pt(par_ini, 6, 22, 'L32x64_pz2')

    # pz = 3
    c4_ini = 0.00254534063308937
    c5_ini = 1.4950831540206349
    E0_ini = 0.7489328408833628
    E_delta_ini = 0.3146488600340617
    par_ini = [c4_ini, c5_ini, E0_ini, E_delta_ini] 
    fit_2pt(par_ini, 6, 22, 'L32x64_pz3')

    # pz = 4
    c4_ini = 0.00045060710237850086
    c5_ini = 2.114353566896884
    E0_ini = 0.8973820519968336
    E_delta_ini = 0.3061722683310514
    par_ini = [c4_ini, c5_ini, E0_ini, E_delta_ini] 
    fit_2pt(par_ini, 6, 22, 'L32x64_pz4')

    # pz =5
    c4_ini = 0.00012043628967839538
    c5_ini = 28.012362815933443
    E0_ini = 1.1393241537323875
    E_delta_ini = 2.4192900812233713
    par_ini = [c4_ini, c5_ini, E0_ini, E_delta_ini] 
    fit_2pt(par_ini, 6, 22, 'L32x64_pz5')

    #c4_ini = 0.00012043628967839538
    #c5_ini = 28.012362815933443
    #E0_ini = 1.0345561048741034 * a_GeV
    #E_delta_ini = 2.4192900812233713

   

    par_ini = np.array(par_ini)


    #print(par_ini)

    print(a_GeV)

    

    #fit_2pt(par_ini, 9, 14, 'L32x96')
    #fit_2pt(par_ini, 9, 14, 'L32x96_pz2')
    #fit_2pt(par_ini, 9, 14, 'L32x96_pz3')
    #fit_2pt(par_ini, 9, 14, 'L32x96_pz4')
    #fit_2pt(par_ini, 9, 14, 'L32x96_pz5')
    #fit_2pt_mean(par_ini, 6, 22, 'L32x64')



    #print(th_log_hB(z_test, 0.1, 2, par_set))


