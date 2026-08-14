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

start_time = time.time()

#data_name_test = 'L32x96_dhx_pz5'
data_name_test = 'L24x72_dhxmeang1_pz2'
#data_name_test = 'L32x64_pz0'
#data_name_test = 'L48x144_pz0'
#data_name = 'L32x96'

#type_fit = '_new' 
#type_fit = '_s'  
type_fit = ''


 # 拟合的参数化形式
def R_model(z_,  tsep_, ti_, z_list_, c0_z0, c1_z0, c2_z0, c0_z1, c1_z1, c2_z1, deltaE):
    # 初始化结果数组
    res = np.zeros_like(ti_, dtype=float)
    
    # 计算 z_ == z_list[0] 的部分
    mask_z0 = (z_ == z_list_[0])
    res[mask_z0] = c0_z0 + c1_z0 * np.exp(-deltaE * (tsep_[mask_z0] - ti_[mask_z0])) + c1_z0 * np.exp(-deltaE * ti_[mask_z0]) + c2_z0  * np.exp(-deltaE * tsep_[mask_z0])
    
    # 计算 z_ == z_list[1] 的部分
    mask_z1 = (z_ == z_list_[1])
    res[mask_z1] = c0_z1 + c1_z1 * np.exp(-deltaE * (tsep_[mask_z1] - ti_[mask_z1])) + c1_z1 * np.exp(-deltaE * ti_[mask_z1]) + c2_z1  * np.exp(-deltaE * tsep_[mask_z1])
    
    # 检查是否有不满足条件的 z_ 值
    if not np.all(mask_z0 | mask_z1):
        raise ValueError(f'z only {z_list_[0]} or {z_list_[1]}')
    
    return res

def fit_ratio( data_for_fit,  resam_type,  data_name_):
    z_set_, tsep_set_, ti_sep_set_, ratio_mean_set_, err_set, ratio_samples_fit_, z_list_, t_sep_list_, n_remove_ = data_for_fit
    #后面3个用于保存文件

    c_inv =  covariance_matrix_inv(ratio_samples_fit_,  resam_type) # 计算协方差矩阵的逆

    ini_c0_z0, ini_c1_z0,  ini_c2_z0, ini_c0_z1, ini_c1_z1, ini_c2_z1, ini_deltaE, ini_chi2_mean = fit_ratio_mean(data_for_fit,  resam_type , c_inv)

    # 初始化一个空的 DataFrame 用于存储拟合结果
    fit_results_list = []
    
    data_num = len(z_set_)

    n_samples = ratio_samples_fit_.shape[1]

    for sample_i in range(n_samples):

        ratio_sample_i_set  =  ratio_samples_fit_[:, sample_i]

        #print(sample_i)

        def cost_function(c0_z0, c1_z0, c2_z0, c0_z1, c1_z1, c2_z1, deltaE):
                    
            R_th = R_model( z_set_, tsep_set_, ti_sep_set_, z_list_, c0_z0, c1_z0, c2_z0,  c0_z1, c1_z1, c2_z1, deltaE) 
            
            del_ratio = (ratio_sample_i_set - R_th )
            chi2 = del_ratio.T @ c_inv @ del_ratio
                
            mean_chi2 = chi2 / data_num

            #print('par', c0_z0, c1_z0, c0_z1, c1_z1, deltaE)
            #print('chi_mean', mean_chi2)

            
            return mean_chi2

        # 初始化 Minuit
        #m = Minuit(cost_function, c0_z0=0.512, c1_z0=-1.33767, c0_z1=0.0557, c1_z1=-0.11829, deltaE=0.9096) 
        #m = Minuit(cost_function, c0_z0=0.490763, c1_z0=-1.792308, c0_z1=0.425081, c1_z1=-1.504943, deltaE=1.02788) 
        #m = Minuit(cost_function, c0_z0=0.69, c1_z0=-22., c0_z1=0.66, c1_z1=-20., deltaE=1.59)
        #m = Minuit(cost_function, c0_z0=0.649778, c1_z0=-2.843502, c0_z1= 0.618377 , c1_z1=-2.6613, deltaE=1.231724)
        m = Minuit(cost_function, c0_z0= ini_c0_z0, c1_z0= ini_c1_z0, c2_z0 =ini_c2_z0, c0_z1= ini_c0_z1, c1_z1=ini_c1_z1, c2_z1 =ini_c2_z1, deltaE=ini_deltaE)

        #m.fixed["c2_z0"] = True
        #m.fixed["c2_z1"] = True
       
        
        m.limits["c0_z0"] = (None, None)  
        m.limits["c1_z0"] = (None, None) 
        m.limits["c2_z0"] = (None, None) 
        m.limits["c0_z1"] = (None, None)  
        m.limits["c1_z1"] = (None, None)
        m.limits["c2_z1"] = (None, None)  
        m.limits["deltaE"] = (-20, 20) 
        '''
        m.limits["c0_z0"] = (-5, 5)  
        m.limits["c1_z0"] = (-5, 5)
        m.limits["c2_z0"] = (-50, 200)
        m.limits["c0_z1"] = (-5, 5)  
        m.limits["c1_z1"] = (-5, 5)
        m.limits["c2_z0"] = (-50, 200) 
        m.limits["deltaE"] = (-20, 20)   
        '''

        # 运行拟合
        m.migrad()  # 最小化

        c0_z0 , c1_z0, c2_z0, c0_z1, c1_z1, c2_z1, deltaE = m.values
        chi2_mean = cost_function(c0_z0 , c1_z0, c2_z0, c0_z1, c1_z1, c2_z1, deltaE)

        
        # 输出拟合结果和运行时间
        #print("\n=== 拟合结果 ===")
        #print("par:", m.values)
        #print("data number:", data_num )
        #print("chi/N:", chi2_mean)
        #print(f"run time: {elapsed_time:.2f} 秒")
            
        # 将拟合结果存储到 DataFrame 中
        fit_results_list.append({
            "sample_i": sample_i,
            "c0_z0": c0_z0,
            "c1_z0": c1_z0,
            "c2_z0": c2_z0,
            "c0_z1": c0_z1,
            "c1_z1": c1_z1,
            "c2_z1": c2_z1,
            "deltaE": deltaE,
            "chi2_mean": chi2_mean
        })

    # 保存拟合结果到 CSV 文件
    fit_results = pd.DataFrame(fit_results_list)
    ratio_tsep_tisep_path = f'{this_path}/result/ratio_tsep_tisep/{data_name_}/ex{n_remove_}{type_fit}'
    os.makedirs(ratio_tsep_tisep_path, exist_ok=True)
    fit_results.to_csv(f"{ratio_tsep_tisep_path}/{resam_type}_tsep{int(tsep_set_[0])}_{int(tsep_set_[-1])}_z{int(z_list_[-1])}.csv", index=False)
    
    c0_z0  =  np.mean(fit_results['c0_z0'])
    c0_z1  =  np.mean(fit_results['c0_z1'])
    c1_z0  =  np.mean(fit_results['c1_z0'])
    c1_z1  =  np.mean(fit_results['c1_z1'])
    c2_z0  =  np.mean(fit_results['c2_z0'])
    c2_z1  =  np.mean(fit_results['c2_z1'])
    deltaE =  np.mean(fit_results['deltaE'])

    c0_z0_std  =  np.std(fit_results['c0_z0'])
    c0_z1_std  =  np.std(fit_results['c0_z1'])
    c1_z0_std  =  np.std(fit_results['c1_z0'])
    c1_z1_std  =  np.std(fit_results['c1_z1'])
    c2_z0_std  =  np.std(fit_results['c2_z0'])
    c2_z1_std  =  np.std(fit_results['c2_z1'])
    deltaE_std =  np.std(fit_results['deltaE'])

    if  resam_type == 'jack':

        #pdb.set_trace()

        c0_z0_std  =  c0_z0_std * np.sqrt(n_samples - 1)
        c0_z1_std  =  c0_z1_std * np.sqrt(n_samples - 1)
        c1_z0_std  =  c1_z0_std * np.sqrt(n_samples - 1)
        c1_z1_std  =  c1_z1_std * np.sqrt(n_samples - 1)
        c2_z0_std  =  c2_z0_std * np.sqrt(n_samples - 1)
        c2_z1_std  =  c2_z1_std * np.sqrt(n_samples - 1)
        deltaE_std =  deltaE_std * np.sqrt(n_samples - 1)
    
    #pdb.set_trace()
    '''
    width = 13

    # 定义输出文件路径
    
    ratio_fit_path = f'{this_path}/result/ratio_fit_result/{data_name_}'
    os.makedirs(ratio_fit_path, exist_ok=True)
    output_file = f"{ratio_fit_path}/c0_vs_z_{resam_type}_ex{n_remove_}_test{t_sep_list_[0]}_{t_sep_list_[-1]}.txt"

    # 打开文件并写入内容
    
    with open(output_file, "a") as file:
        file.write("\n=== 拟合结果 ===\n")
        file.write(f"{'par,':<{width}} {'c0_z0':<{width}} {'c1_z0':<{width}} {'c0_z1':<{width}} {'c1_z1':<{width}} {'deltaE':<{width}}\n")
        file.write(f"{'mean:':<{width}} {c0_z0:<{width}.6f} {c1_z0:<{width}.6f} {c0_z1:<{width}.6f} {c1_z1:<{width}.6f} {deltaE:<{width}.6f}\n")
        file.write(f"{'std:':<{width}} {c0_z0_std:<{width}.6f} {c1_z0_std:<{width}.6f} {c0_z1_std:<{width}.6f} {c1_z1_std:<{width}.6f} {deltaE_std:<{width}.6f}\n")
        file.write(f"{'data number:':<{width}} {data_num}\n")
        file.write(f"{'chi/N:':<{width}} {chi2_mean:.6f}\n")
        file.write(f"{'z=:':<{width}} {z_list_[1]}\n")
    '''
    return c0_z1, c0_z1_std, ini_chi2_mean

def fit_ratio_mean(data_for_fit, resam_type , c_inv = None):
    z_set_, tsep_set_, ti_sep_set_, ratio_mean_set_, err_set, ratio_samples_fit_, z_list_, t_sep_list_, n_remove_ = data_for_fit
    # 初始化一个空的 DataFrame 用于存储拟合结果
    if c_inv is None:
        c_inv =  covariance_matrix_inv(ratio_samples_fit_, resam_type) # 计算协方差矩阵的逆
   
    data_num = len(ratio_mean_set_)

    #visualize_matrix(np.linalg.inv(c_inv) @ c_inv)

    #print(sample_i)

    def cost_function(c0_z0, c1_z0, c2_z0, c0_z1, c1_z1, c2_z1, deltaE):
                
        R_th = R_model(z_set_, tsep_set_, ti_sep_set_, z_list_, c0_z0, c1_z0, c2_z0, c0_z1, c1_z1, c2_z1, deltaE) 
        del_ratio = (ratio_mean_set_ - R_th )
        chi2 = del_ratio.T @ c_inv @ del_ratio
            
        mean_chi2 = chi2 / data_num

        #pdb.set_trace()

        #print('par', c0_z0, c1_z0, c0_z1, c1_z1, deltaE)
        #print('chi_mean', mean_chi2)

        
        return mean_chi2

    # 初始化 Minuit
    m = Minuit(cost_function, c0_z0=0.490763, c1_z0=-1.792308, c2_z0=0, c0_z1=0.425081, c1_z1=-1.504943, c2_z1=0, deltaE=1.02788)
    #m = Minuit(cost_function, c0_z0=0.5, c1_z0=1., c2_z0=0, c0_z1=0.5, c1_z1=1., c2_z1=0,  deltaE=1.02788) 
    #m = Minuit(cost_function,  c0_z0=0.67,  c1_z0=-3.8, c2_z0=0,  c0_z1=0.64,  c1_z1=-3.7,c2_z1=0, deltaE=1.26)

    #m.fixed["c2_z0"] = True
    #m.fixed["c2_z1"] = True

    # 设置参数范围（可选）
    m.limits["c0_z0"] = (None, None)  
    m.limits["c1_z0"] = (None, None)
    m.limits["c2_z0"] = (None, None)
    m.limits["c0_z1"] = (None, None)  
    m.limits["c1_z1"] = (None, None)
    m.limits["c2_z0"] = (None, None) 
    m.limits["deltaE"] = (-20, 20)  

    # 运行拟合
    m.migrad()  # 最小化

    c0_z0 , c1_z0, c2_z0, c0_z1, c1_z1, c2_z1, deltaE = m.values
    chi2_mean = cost_function(c0_z0 , c1_z0, c2_z0, c0_z1, c1_z1, c2_z1, deltaE)

    
    if chi2_mean > 2:
        #m = Minuit(cost_function, c0_z0=0.490763, c1_z0=-1.792308, c2_z0=0, c0_z1=0.425081, c1_z1=-1.504943, c2_z1=0, deltaE=1.02788)
        m = Minuit(cost_function, c0_z0=0.5, c1_z0=-1.5, c2_z0=0, c0_z1=0.5, c1_z1=-1.5, c2_z1=0,  deltaE=1.02788) 
        #m = Minuit(cost_function,  c0_z0=0.67,  c1_z0=-3.8, c2_z0=0,  c0_z1=0.64,  c1_z1=-3.7,c2_z1=0, deltaE=1.26)

        #m.fixed["c2_z0"] = True
        #m.fixed["c2_z1"] = True

        # 设置参数范围（可选）
        
        m.limits["c0_z0"] = (None, None)  
        m.limits["c1_z0"] = (None, None)
        m.limits["c2_z0"] = (None, None)
        m.limits["c0_z1"] = (None, None)  
        m.limits["c1_z1"] = (None, None)
        m.limits["c2_z0"] = (None, None) 
        m.limits["deltaE"] = (-20, 20)
        

        

        # 运行拟合
        m.migrad()  # 最小化

        c0_z0 , c1_z0, c2_z0, c0_z1, c1_z1, c2_z1, deltaE = m.values
        chi2_mean = cost_function(c0_z0 , c1_z0, c2_z0, c0_z1, c1_z1, c2_z1, deltaE)
    

    # 检查拟合是否收敛

    # 输出拟合结果和运行时间
    print("...............................................................")
    print(f"z for fit:{z_list_}")
    print("Fit parameters:")
    print(f"c0_z0  = {m.values['c0_z0']} ± {m.errors['c0_z0']}")
    print(f"c1_z0  = {m.values['c1_z0']} ± {m.errors['c1_z0']}")
    print(f"c2_z0  = {m.values['c2_z0']} ± {m.errors['c2_z0']}")
    print(f"c0_z1  = {m.values['c0_z1']} ± {m.errors['c0_z1']}")
    print(f"c1_z1  = {m.values['c1_z1']} ± {m.errors['c1_z1']}")
    print(f"c2_z1  = {m.values['c2_z1']} ± {m.errors['c2_z1']}")
    print(f"deltaE = {m.values['deltaE']} ± {m.errors['deltaE']}")
    

    # Get the covariance matrix of the parameters
    cov_matrix = m.covariance
    print("Covariance matrix of the parameters from Minuit")
    print(cov_matrix)

    print("data number:", data_num )
    print("chi/N:", chi2_mean)
    print("...............................................................")
    #print(f"run time: {elapsed_time:.2f} 秒")
        
    return c0_z0 , c1_z0, c2_z0, c0_z1, c1_z1, c2_z1, deltaE, chi2_mean

def c0_vs_z(n_remove_, t_sep_list_, data_name_, resam_type_ ):

    #t_sep_list = [8, 9, 10, 11, 12, 13, 14] 
    #n_remove =  3 # 拟合时去掉头尾的点数

    res_list = []
    for z_i in range(20):

        c0_z_mean, c0_z_std, chi2mean = fit_ratio( data_chose( z_i , t_sep_list_ , n_remove_ , data_name_), resam_type_, data_name_ )

        res_list.append({ 'z'       : z_i,
                         'c0_z_mean': c0_z_mean, 
                         'c0_z_std' : c0_z_std,
                         'chi2_mean': chi2mean               
        }
        )

        fit_result  = pd.DataFrame(res_list)
        ratio_fit_path = f'{this_path}/result/ratio_fit_result/{data_name_}/ex{n_remove_}{type_fit}'
        os.makedirs(ratio_fit_path, exist_ok=True)
        fit_result.to_csv(f'{ratio_fit_path}/c0_vs_z_'+ resam_type_+f'_{t_sep_list_[0]}_{t_sep_list_[-1]}.csv', index=False)
        #fit_result.to_csv(f'ratio_fit_result/a_test'+ data_name +f'_ex{n_remove_}_test{t_sep_list_[0]}_{t_sep_list_[-1]}.csv', index=False)
     
    return 0



if __name__ == "__main__":

    star = time.time()
    #t_sep_list = [12, 13, 14,15, 16, 17, 18, 19, 20, 21, 22, 23] 
    t_sep_list = [7, 8, 9, 10, 11, 12, 13] 
    n_remove =  3
    
    
    #fit_ratio( data_chose(9,  t_sep_list, n_remove, data_name_test),  'boot', data_name_test )
    #fit_ratio_mean(data_chose(7,  t_sep_list, n_remove, data_name_test),  'boot')
    c0_vs_z(n_remove, t_sep_list, data_name_test, 'boot')
    
    
    #test(data_name, 'boot')
   

    end = time.time()
    #print('run time:', end - star)

    

