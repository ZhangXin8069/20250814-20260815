import sys
sys.path.append('/public/group/imp/zengch/LQCD/input_file')
sys.path.append('/public/group/imp/zengch/LQCD/tool')
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
from hB_data import load_hB_data, hB_data, a_len_set, fm_to_GeV # fm / fm_to_GeV = GeV^{-1}
from hB_data_FeynmenHellman import load_hB_data_FeynmenHellman
 # 拟合的参数化形式

this_path = '/public/group/imp/zengch/LQCD/renorma'


data_L24x72 = load_hB_data_FeynmenHellman('L24x72_pz0',  3, 7, 13)
data_L32x64 = load_hB_data_FeynmenHellman('L32x64_pz0', 3, 8, 14)
data_L32x96 = load_hB_data_FeynmenHellman('L32x96_pz0',  4, 8, 14)
data_L36x108 = load_hB_data_FeynmenHellman('L36x108_pz0',  4, 9, 15)  
data_L48x144 = load_hB_data_FeynmenHellman('L48x144_dhx0_pz0',  6, 12, 18)

def Z_MS(z_GeV_, mu_):   #z GeV^{-1} , mu GeV

    #alpha_s = A_s(mu_) * 4. * np.pi
    alpha_s =  0.296

    Z_MS = 1. + alpha_s * CA / (4. * np.pi) * (5./3. *  np.log(  (z_GeV_ **2 * mu_**2) / (4. * np.exp(-2. * gammaE)) ) + 3)

    return Z_MS

def th_hB(z_, a_, mu_, par_g_set, f_set): 
    """
    计算对数hB函数，z_必须完全等于预定义的z_set_new数组
    
    参数:
        z_:  必须完全等于z_set_new数组 fm 
        a_:  标量                      GeV^{-1}
        mu_: 标量                      GeV
        par_set: 参数集合              GeV (m0, lambda_QCD)
        
    返回:
        计算结果，形状与z_set_new相同
    """
    z_set_new    = np.arange(0.15, 1.0 + 0.05, 0.05) # 对z 进行插值计算的点


    # 严格检查z_是否完全等于z_set_new
    if not np.array_equal(z_, z_set_new[:len(z_)]):
        raise ValueError("z_ must be exactly equal to the predefined z_set_new array")
    
    z_set_new = z_

    # 解包参数
    k, d, m0, Lamda_QCD, *g_params = par_g_set
    f_set = np.array(f_set)
    nf = 3.
    
    b0 = 11. - 2. * nf / 3.

    #ZR = th_ZR(z_set_new, a_, mu_, k, d, m0, Lamda_QCD)

    z_set_new_GeV = z_set_new / fm_to_GeV # 将z 的单位换算成 GeV^{-1}

    #pdb.set_trace()
    #pert_NLO = Z_MS(z_set_new_GeV, mu_) 

    def B(z_i):

        z1 = 0.301 / fm_to_GeV
        result = np.zeros_like(z_i, dtype=float)
        mask = (z_i <= z1)
        
        # 处理z_i < z1的情况（已向量化）
        z_masked = z_i[mask]
        Z_MS_NLO = Z_MS(z_masked, mu_) 
        result[mask] = np.log( Z_MS_NLO) + m0 * z_masked
        
        #pdb.set_trace()
        # 处理z_i >= z1的情况（向量化优化）
        result[~mask] = np.array(g_params)[:len(result[~mask])]
  
        return result

    # 主计算部分
    log_hB = (k * z_set_new_GeV) / (a_ * np.log(a_ * Lamda_QCD))
    log_hB += 5. * CA / (3. * b0) * np.log(np.log(1./(a_ * Lamda_QCD)) / np.log(mu_/ Lamda_QCD))
    log_hB +=  np.log(   (1. + d / np.log(a_ * Lamda_QCD)) **2.   ) / 2.
    log_hB += f_set[:len(z_)] * a_ **2.

    

    #pdb.set_trace()
    
    # 添加B(z_)部分
    log_hB += B(z_set_new_GeV)

    

    #pdb.set_trace()

    #return np.exp(log_hB)
    return log_hB

def th_ZR(z_, a_, mu_, k, d, m0, Lamda_QCD, f_set):
   
    z_set_new    = z_ 
    
    # 解包参数

    z_set_new_GeV = z_set_new / fm_to_GeV
    
    
    nf = 3.
    
    b0 = 11. - 2. * nf / 3.
    
    # 主计算部分
    log_hB =  (k * z_set_new_GeV) / (a_ * np.log(a_ * Lamda_QCD))
    log_hB += 5. * CA / (3. * b0) * np.log(np.log(1./(a_ * Lamda_QCD)) / np.log(mu_/ Lamda_QCD))
    log_hB += np.log(   (1. + d / np.log(a_ * Lamda_QCD)) **2.   ) / 2. + m0 * z_set_new_GeV
    #pdb.set_trace()
    log_hB +=  f_set[:len(z_)] * a_ ** 2.

    #pdb.set_trace()
    
    #return log_hB
    return np.exp(log_hB)

def fit_ZR_mean(par_ini):
    mu = 2. 
    
    hB_L24x72         = data_L24x72['loghB']
    z_L24x72          = data_L24x72['z']
    hB_mean_L24x72    = np.mean(hB_L24x72, axis=1)
    hB_std_L24x72     = np.std(hB_L24x72, axis=1)
    #c_inv_L24x72      = covariance_matrix_inv(hB_L24x72, 'boot')
    c_inv_L24x72      = np.linalg.inv(np.diag(hB_std_L24x72**2.))

    
    
    hB_L32x64         = data_L32x64['loghB']
    z_L32x64          = data_L32x64['z']
    hB_mean_L32x64    = np.mean(hB_L32x64, axis=1)
    hB_std_L32x64     = np.std(hB_L32x64, axis=1)
    #c_inv_L32x64      = covariance_matrix_inv(hB_L32x64, 'boot')
    c_inv_L32x64      = np.linalg.inv(np.diag(hB_std_L32x64**2.))

    
    hB_L32x96         = data_L32x96['loghB']
    z_L32x96          = data_L32x96['z']
    hB_mean_L32x96    = np.mean(hB_L32x96, axis=1)
    hB_std_L32x96     = np.std(hB_L32x96, axis=1)
    #c_inv_L32x96      = covariance_matrix_inv(hB_L32x96, 'boot')
    c_inv_L32x96      = np.linalg.inv(np.diag(hB_std_L32x96**2.))

    hB_L36x108         = data_L36x108['loghB']
    z_L36x108          = data_L36x108['z']
    hB_mean_L36x108    = np.mean(hB_L36x108, axis=1)
    hB_std_L36x108     = np.std(hB_L36x108, axis=1)
    #c_inv_L36x108     = covariance_matrix_inv(hB_L36x108, 'boot')
    c_inv_L36x108     = np.linalg.inv(np.diag(hB_std_L36x108**2.))

    hB_L48x144         = data_L48x144['loghB']
    z_L48x144          = data_L48x144['z']
    hB_mean_L48x144    = np.mean(hB_L48x144, axis=1)
    hB_std_L48x144     = np.std(hB_L48x144, axis=1)
    #c_inv_L48x144      = covariance_matrix_inv(hB_L48x144, 'boot')
    c_inv_L48x144      = np.linalg.inv(np.diag(hB_std_L48x144**2.))


    

    #pdb.set_trace()


    def cost_function(z_set_, hB_data, c_inv, a_, mu_, par_set):
        data_num = len(z_set_)

       
        par_g_set = par_set[:18]    
        f_set = par_set[18:]   
            
        hB_th = th_hB(z_set_, a_, mu_, par_g_set, f_set) 
        del_hB = (hB_th - hB_data )
        chi2 = del_hB.T @ c_inv @ del_hB
       
        mean_chi2 = chi2 / data_num

        
        #pdb.set_trace()


        return mean_chi2
    
    def cost_function_all(par_set):
            
        chi2_L24x72  = cost_function(z_L24x72,   hB_mean_L24x72,  c_inv_L24x72,  a_len_set['L24x72'],  mu, par_set)
        chi2_L32x64  = cost_function(z_L32x64,   hB_mean_L32x64,  c_inv_L32x64,  a_len_set['L32x64'],  mu, par_set)
        chi2_L32x96  = cost_function(z_L32x96,   hB_mean_L32x96,  c_inv_L32x96,  a_len_set['L32x96'],  mu, par_set)
        chi2_L36x108 = cost_function(z_L36x108,  hB_mean_L36x108, c_inv_L36x108, a_len_set['L36x108'], mu, par_set)
        chi2_L48x144 = cost_function(z_L48x144,  hB_mean_L48x144, c_inv_L48x144, a_len_set['L48x144'], mu, par_set)
        #return chi2_L24x72  
        n1 = len(z_L24x72)
        n2 = len(z_L32x64)
        n3 = len(z_L32x96)
        n4 = len(z_L36x108)
        n5 = len(z_L48x144)
    
        dof = n1 + n2 + n3 + n4 + n5
        #dof = n1 + n2 + n3 
        return ( chi2_L24x72 * n1 + chi2_L32x64 * n2 + chi2_L32x96 * n3  + chi2_L36x108 * n4 + chi2_L48x144 * n5) / dof
        #return ( chi2_L24x72 * n1 + chi2_L32x64 * n2 + chi2_L32x96 * n3) / dof


    
    par_name = ('k',   'd',   'm0',   'Lamda_QCD', 
                'g1',  'g2',  'g3',   'g4', 
                'g5',  'g6',  'g7',   'g8', 
                'g9',  'g10', 'g11',  'g12',
                'g13', 'g14', 'f1',   'f2', 'f3', 'f4', 'f5',
                'f6',  'f7',  'f8',  'f9', 'f10', 'f11', 'f12',
                'f13',  'f14', 'f15', 'f16', 'f17', 'f18')
    
   
    m = Minuit(cost_function_all, par_ini, name=par_name)

    m.limits["k"] = (None, None)  
    m.limits["d"] = (None, None) 
    m.limits["m0"] = (None, None)  
    m.limits["Lamda_QCD"] = (0, None) 

    #m.fixed['f1',   'f2', 'f3', 'f4', 'f5',
    #        'f6',  'f7',  'f8',  'f9', 'f10', 'f11', 'f12',
    #        'f13',  'f14', 'f15', 'f16', 'f17', 'f18']= True
    
            
    
    m.migrad()

    chi_all_mean = cost_function_all(m.values)
    k_fit, d_fit, m0_fit, lambda_QCD_fit, *g_params_fit = m.values

    print('k_fit =', k_fit)
    print('d_fit =', d_fit)
    print('m0_fit =', m0_fit)
    print('lambda_QCD_fit =', lambda_QCD_fit)
    print('chi2_mean =', chi_all_mean)
    print(k_fit, d_fit, m0_fit, lambda_QCD_fit, *g_params_fit)
    
    return m.values

def fit_ZR(par_ini):
    mu = 2. 
    
    hB_L24x72         = data_L24x72['loghB']
    z_L24x72          = data_L24x72['z']
    hB_mean_L24x72    = np.mean(hB_L24x72, axis=1)
    hB_std_L24x72     = np.std(hB_L24x72, axis=1)
    #c_inv_L24x72      = covariance_matrix_inv(hB_L24x72, 'boot')
    c_inv_L24x72      = np.linalg.inv(np.diag(hB_std_L24x72**2.))

    hB_L32x64         = data_L32x64['loghB']
    z_L32x64          = data_L32x64['z']
    hB_mean_L32x64    = np.mean(hB_L32x64, axis=1)
    hB_std_L32x64     = np.std(hB_L32x64, axis=1)
    #c_inv_L32x64      = covariance_matrix_inv(hB_L32x64, 'boot')
    c_inv_L32x64      = np.linalg.inv(np.diag(hB_std_L32x64**2.))

    hB_L32x96         = data_L32x96['loghB']
    z_L32x96          = data_L32x96['z']
    hB_mean_L32x96    = np.mean(hB_L32x96, axis=1)
    hB_std_L32x96     = np.std(hB_L32x96, axis=1)
    #c_inv_L32x96      = covariance_matrix_inv(hB_L32x96, 'boot')
    c_inv_L32x96      = np.linalg.inv(np.diag(hB_std_L32x96**2.))

    hB_L36x108         = data_L36x108['loghB']
    z_L36x108          = data_L36x108['z']
    hB_mean_L36x108    = np.mean(hB_L36x108, axis=1)
    hB_std_L36x108     = np.std(hB_L36x108, axis=1)
    #c_inv_L36x108     = covariance_matrix_inv(hB_L36x108, 'boot')
    c_inv_L36x108     = np.linalg.inv(np.diag(hB_std_L36x108**2.))

    hB_L48x144         = data_L48x144['loghB']
    z_L48x144          = data_L48x144['z']
    hB_mean_L48x144    = np.mean(hB_L48x144, axis=1)
    hB_std_L48x144     = np.std(hB_L48x144, axis=1)
    #c_inv_L48x144      = covariance_matrix_inv(hB_L48x144, 'boot')
    c_inv_L48x144      = np.linalg.inv(np.diag(hB_std_L48x144**2.))

   

    def cost_function(z_set_, hB_data, c_inv, a_, mu_, par_set):

        par_g_set = par_set[:18]    
        f_set = par_set[18:]
       
        data_num = len(z_set_)
        
        hB_th = th_hB(z_set_, a_, mu_, par_g_set, f_set) 
        del_hB = (hB_th - hB_data )
        chi2 = del_hB.T @ c_inv @ del_hB
       
        mean_chi2 = chi2 / data_num

        #pdb.set_trace()

        return mean_chi2
    
    fit_results_list = []

    for i in range(hB_L24x72.shape[1]):
    #for i in range(1):


        def cost_function_all(par_set):
            
            chi2_L24x72  = cost_function(z_L24x72,   hB_L24x72[:,i],  c_inv_L24x72,  a_len_set['L24x72'],  mu, par_set)
            chi2_L32x64  = cost_function(z_L32x64,   hB_L32x64[:,i],  c_inv_L32x64,  a_len_set['L32x64'],  mu, par_set)
            chi2_L32x96  = cost_function(z_L32x96,   hB_L32x96[:,i],  c_inv_L32x96,  a_len_set['L32x96'],  mu, par_set)
            chi2_L36x108 = cost_function(z_L36x108,  hB_L36x108[:,i], c_inv_L36x108, a_len_set['L36x108'], mu, par_set)
            chi2_L48x144 = cost_function(z_L48x144,  hB_L48x144[:,i], c_inv_L48x144, a_len_set['L48x144'], mu, par_set)
            #return chi2_L24x72  
            n1 = len(z_L24x72)
            n2 = len(z_L32x64)
            n3 = len(z_L32x96)
            n4 = len(z_L36x108)
            n5 = len(z_L48x144)
       
            dof = n1 + n2 + n3 + n4 + n5
            #dof = n1 + n2 + n3 
            return ( chi2_L24x72 * n1 + chi2_L32x64 * n2 + chi2_L32x96 * n3  + chi2_L36x108 * n4 + chi2_L48x144 * n5) / dof
            #return ( chi2_L24x72 * n1 + chi2_L32x64 * n2 + chi2_L32x96 * n3) / dof


        
        par_name = ('k',   'd',   'm0',   'Lamda_QCD', 
                'g1',  'g2',  'g3',   'g4', 
                'g5',  'g6',  'g7',   'g8', 
                'g9',  'g10', 'g11',  'g12',
                'g13', 'g14', 'f1',   'f2', 'f3', 'f4', 'f5',
                'f6',  'f7',  'f8',  'f9', 'f10', 'f11', 'f12',
                'f13',  'f14', 'f15', 'f16', 'f17', 'f18')
        
        
        m = Minuit(cost_function_all, par_ini, name=par_name)

        m.limits["k"] = (None, None)  
        m.limits["d"] = (None, None) 
        m.limits["m0"] = (None, None)  
        m.limits["Lamda_QCD"] = (0, None) 

        m.fixed['f1',   'f2', 'f3', 'f4', 'f5',
            'f6',  'f7',  'f8',  'f9', 'f10', 'f11', 'f12',
            'f13',  'f14', 'f15', 'f16', 'f17', 'f18']= True
        
                
        
        m.migrad()
        par_all_fit = m.values
        k_fit, d_fit, m0_fit, lambda_QCD_fit = par_all_fit[:4]
        g_params_fit = par_all_fit[4:18]
        f_params_fit = par_all_fit[18:]
        chi_all_mean = cost_function_all(m.values)

        fit_results_list.append({
            "sample_i": i,
            "k": k_fit,
            "d": d_fit,
            "m0": m0_fit,
            "lambda_QCD": lambda_QCD_fit,
            **{f"g{j+1}": g_val for j, g_val in enumerate(g_params_fit)},
            **{f"f{j+1}": f_val for j, f_val in enumerate(f_params_fit)},
            "chi2" : chi_all_mean
        })

        # 保存拟合结果到 CSV 文件
        fit_results = pd.DataFrame(fit_results_list)
        #fit_results.to_csv(f"{this_path}/result/ZR_fit_result/ZR_a5_FeynmenHellman_f2all.csv", index=False)
        fit_results.to_csv(f"{this_path}/result/ZR_fit_result/ZR_a5_FeynmenHellman_new_test.csv", index=False)
    
    return m.values


if __name__ == "__main__":
    count = 0

    
    k_ini = 1.89
    d_ini =  -0.047
    m0_ini =  14.5 * 0.197
    lambda_QCD_ini =   0.642

    g_par  = [ 5.45, 6.17, 6.89, 7.6, 
               8.3,  9.0,  9.7,  10.4, 
               11.1, 11.7, 12.4, 13.1, 
               13.7, 14.4]
    

    k_fit = 1.8163627877699116
    d_fit = 0.007463893236205944
    m0_fit = 2.6970204571342604
    lambda_QCD_fit = 0.6294431618831456

    g_par  = [ 5.162634001824635, 5.843450994544796, 6.514895838209571, 7.181288254946188,
               7.836199081382588, 8.489270492436495, 9.132685844239024, 9.769474680227532, 10.401182308512647, 
               11.025564507990834, 11.641616309735282, 12.254499692428325, 12.858163418772612, 13.465173999132338]
    
    f_par = [0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0]
    
    par_ini = [k_fit, d_fit, m0_fit, lambda_QCD_fit] + g_par + f_par

    par_ini = np.array(par_ini)


    #print(par_ini)

    #fit_ZR_mean(par_ini)
    fit_ZR(par_ini)



    #print(th_log_hB(z_test, 0.1, 2, par_set))


