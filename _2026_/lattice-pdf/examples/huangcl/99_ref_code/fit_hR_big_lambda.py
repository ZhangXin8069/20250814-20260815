import sys
import json
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
from fit_zr_new import th_ZR
from hB_data import load_hB_data
from hB_data_FeynmenHellman import load_hB_data_FeynmenHellman, a_len_set, fm_to_GeV, Nl_set
from constant import CA, CF, gammaE, pi, A_s
from scipy.interpolate import interp1d
import pdb # 用于调试代码 pdb.set_trace()
import warnings

this_path = '/public/group/imp/zengch/LQCD/renorma'

cov = 'Yes' #是否考虑关联拟合 Yes 考虑，否则不考虑

cov_name = ''
if cov == 'Yes':
    cov_name = 'c'

nbin = 10



# 检查命令行参数
if len(sys.argv) < 2:
    print("用法: python fit_hR_big_lambda.py <输入文件.json>")
    sys.exit(1)

# 读取JSON配置文件
input_file = sys.argv[1]
with open(input_file, 'r') as f:
    config = json.load(f)

# 从配置文件读取参数

conf = config['conf']
note_name = config['note_name']
exn = config['exn']
tsep_star = config['tsep_star']
tsep_end = config['tsep_end']
a_input = config['a_input']
lamb_do_set = config['lamb_do_set']
lamb_up_set = config['lamb_up_set']
mu_input = config['mu']
append_note = config['append_note']
zs_input =  config['zs_input']
pz_set = config['pz_set']
lambda_extra_value = config['lambda_extra']

# 选择小区域的重整化因子
# ....... a = 0.105
if a_input == 0.105:
    data_pz0 = load_hB_data_FeynmenHellman('L24x72_pz0', 0, 7, 14)
    hB_pz0   = data_pz0['hB_o'][:20 ,:]
    z_fm_pz0 = data_pz0['z_o'][:20]

elif a_input == 0.0897:
    #........ a = 0.0897
    data_pz0 = load_hB_data_FeynmenHellman('L32x64_pz0', 0, 9, 16)
    hB_pz0   = data_pz0['hB_o'][:20 ,:]
    z_fm_pz0 = data_pz0['z_o'][:20]


elif a_input == 0.0775:
    #........ a = 0.0775
    data_pz0 = load_hB_data_FeynmenHellman('L32x96_pz0', 0, 9, 17)
    hB_pz0   = data_pz0['hB_o'][:20 ,:]
    z_fm_pz0 = data_pz0['z_o'][:20]

elif a_input == 0.0688:
    #........ a = 0.0688
    data_pz0 = load_hB_data_FeynmenHellman('L36x108_pz0', 0, 9, 19)
    hB_pz0   = data_pz0['hB_o'][:20 ,:]
    z_fm_pz0 = data_pz0['z_o'][:20]

elif a_input == 0.0519:
    #........ a = 0.0775
    data_pz0 = load_hB_data_FeynmenHellman('L48x144_dhx0_pz0', 0, 12, 24)
    hB_pz0   = data_pz0['hB_o'][:20 ,:]
    z_fm_pz0 = data_pz0['z_o'][:20]

else:
    print('error in a_input' )


hR_lamb_name = f'{conf}{note_name}'
ZR_use = f'ZR_a5_FeynmenHellman_new_mu{mu_input}' # 选择重整化参数


hR_pz = {}
z_set = {}

z_set[0] = z_fm_pz0
hR_pz[0] = hB_pz0

for idx, pz_i in enumerate(pz_set):
    data_pz = load_hB_data_FeynmenHellman(f'{conf}{note_name}_pz{pz_i}', exn[idx], tsep_star[idx], tsep_end[idx])
    hB_pz_i   = data_pz['hB_o']
    z_fm_pz = data_pz['z_o']

    hR_pz[pz_i] = hB_pz_i
    z_set[pz_i] = z_fm_pz


if not all(np.array_equal(vec, z_set[0]) for vec in z_set.values()):
    raise ValueError("Not all vectors in z_set are identical!")



# 输入拟合号的 ZR 
par_fit = pd.read_csv(f'{this_path}/result/ZR_fit_result/{ZR_use}.csv')

k_fit  = par_fit['k'].values
d_fit  = par_fit['d'].values
m0_fit = par_fit['m0'].values
m2_fit = par_fit['m2'].values
lambda_QCD_fit = par_fit['lambda_QCD'].values
f_columns = [f'f{i}' for i in range(1, 3)]  # 创建f1到f2的列名列表
f_set_fit = par_fit[f_columns].values.T


ZR_fit = th_ZR(z_set[0][:, None], a_len_set[conf], mu_input, k_fit[None ,:], d_fit[None ,:], m0_fit[None ,:], m2_fit[None ,:], lambda_QCD_fit[None ,:], f_set_fit)




def hR_z_Pz(z_, Pz_ , hR_pz, hR_0, conf_):

    zs = zs_input

    a_len = a_len_set[conf_]
    Nl    = Nl_set[conf_]

    Pz_GeV    = Pz_ * 2.* pi / (Nl * a_len) 
    z_GeV     = z_ / fm_to_GeV

    lambda_ = z_GeV * Pz_GeV

    res  =  np.zeros_like(ZR_fit, dtype=float)
    mask = ( z_ < zs)

   

    res[mask]  = hR_pz[mask] / hR_0[mask]

    eta_s =   ZR_fit[~mask][0] / hR_0[~mask][0]  
    res[~mask] = ( hR_pz[~mask] / ZR_fit[~mask] ) * eta_s

    
   
    return lambda_ , res

def hR_lamb_fit_data(lambda_, Pz_):
    lamb, hR_lamb = hR_z_Pz(z_set[0], Pz_ , hR_pz[Pz_], hR_pz[0], conf)
    
    # 取输入lambda_的最值
    lam_min = lambda_.min()
    lam_max = lambda_.max()
    
    # 筛选原lamb在 [lam_min, lam_max] 范围内的数据
    mask = (lamb >= lam_min) & (lamb <= lam_max)
    lamb_sel = lamb[mask]
    hR_lamb_sel = hR_lamb[mask]
    
    # 直接返回对应区间的hR_lamb数据（不再插值）
    return hR_lamb_sel

def hR_lambda_fit_form(lambda_, l1_, a1_, lambda0_):
    res = l1_ * lambda_ **(-a1_) * np.exp(-lambda_/lambda0_)
    #print(res)
    #pdb.set_trace()
    return res

def fit_hR_lambda(par_ini, pz_, lab_do, lab_up):
    pz_fit = pz_
    lambda_fit = np.linspace(lab_do, lab_up, nbin)
    #lambda_fit = np.array([4.3196899, 4.71238898, 5.10508806, 5.49778714, 5.89048623])
    hR_data = hR_lamb_fit_data(lambda_fit, pz_fit)

    #lambda_fit = cc_lamb[11:16]
    #hR_data =    cc_input[11:16]

    #pdb.set_trace()


    hR_data_mean = np.mean(hR_data, axis=1)
    hR_data_std  = np.std( hR_data, axis=1)

    c_inv = np.linalg.inv(np.diag(hR_data_std**2.))

    if cov == 'Yes':
        c_inv = covariance_matrix(hR_data,'boot')
        c_inv = replace_small_singular_values(c_inv)
        c_inv = np.linalg.inv(c_inv)
    

    #cov_matrix = covariance_matrix(hR_data,'boot')
    #visualize_matrix_num(c_inv @ cov_matrix)


    data_num = len(lambda_fit)

    
    fit_results_list = []

    id_for_fit = range(hR_data.shape[1])
    
    
    for i in id_for_fit:
        #hR_data_i = hR_data[:, i]
        #print(hR_data_i)

        #pdb.set_trace() 

        def cost_function(par):
            l1, a1, lambda0 = par 
            hR_data_i = hR_data[:, i]
       
            hR_th = hR_lambda_fit_form(lambda_fit, l1, a1, lambda0)

            

            del_hR = (hR_th - hR_data_i )
            chi2 = del_hR.T @ c_inv @ del_hR

            mean_chi2 = chi2 / data_num
           
           
            return abs(mean_chi2)


        
        par_name = ('l1',   'a1',   'lambda0')
        
        
        m = Minuit(cost_function, par_ini, name=par_name)

        m.limits["lambda0"] = (0.0, 100) 
        #m.limits["l1"] = (-50, 50)  
        m.limits["a1"] = (-100, 100)  
        #m.fixed[  'k']= True
        
        m.migrad()

        l1_fit, a1_fit, lambda0_fit = m.values
        chi_all_mean = cost_function(m.values)

        fit_results_list.append({
            "sample_i": i,
            "l1": l1_fit,
            "a1": a1_fit,
            "lambda0": lambda0_fit,
            "chi2" : chi_all_mean
        })

        # 保存拟合结果到 CSV 文件
        fit_results = pd.DataFrame(fit_results_list)
        fit_results.to_csv(f"{this_path}/result/hR_lambda_fit_result/{hR_lamb_name}{append_note}_FeynmenHellman_mu{mu_input}pz{pz_fit}_lambda{lab_do}_{lab_up}_n{nbin}{cov_name}.csv", index=False)
    
    return m.values

def fit_hR_lambda_mean(par_ini, pz_, lab_do, lab_up):
    pz_fit = pz_
    lambda_fit = np.linspace(lab_do, lab_up, nbin)
    #lambda_fit = np.array([4.3196899, 4.71238898, 5.10508806, 5.49778714, 5.89048623])
    hR_data = hR_lamb_fit_data(lambda_fit, pz_fit)

    #lambda_fit = cc_lamb[11:16]
    #hR_data =    cc_input[11:16]

    


    hR_data_mean = np.mean(hR_data, axis=1)
    hR_data_std  = np.std( hR_data, axis=1)
    
    c_inv = np.linalg.inv(np.diag(hR_data_std**2.))
  
    data_num = len(lambda_fit)

    
    fit_results_list = []

    id_for_fit = range(hR_data.shape[1])
    
    
    

    def cost_function(par):
        l1, a1, lambda0 = par 
        hR_data_i = hR_data_mean
    
        hR_th = hR_lambda_fit_form(lambda_fit, l1, a1, lambda0)

        

        del_hR = (hR_th - hR_data_i )
        chi2 = del_hR.T @ c_inv @ del_hR

        mean_chi2 = chi2 / data_num
        
        
        return mean_chi2


    
    par_name = ('l1',   'a1',   'lambda0')
    
    
    m = Minuit(cost_function, par_ini, name=par_name)

    m.limits["lambda0"] = (0.0, 100) 
    #m.limits["l1"] = (-50, 50)  
    m.limits["a1"] = (-100, 100)  
    #m.fixed[  'k']= True
    
    m.migrad()

    l1_fit, a1_fit, lambda0_fit = m.values
    chi_all_mean = cost_function(m.values)
    print('.........................')
    print('l1=', l1_fit)
    print('a1=', a1_fit)
    print('lambda0=', lambda0_fit)
    print('chi_mean=', chi_all_mean)
    print('.........................')

 
    return [l1_fit, a1_fit, lambda0_fit], chi_all_mean

def fit_hR_lambda_meanc(par_ini, pz_, lab_do, lab_up):
    pz_fit = pz_
    lambda_fit = np.linspace(lab_do, lab_up, nbin)
    #lambda_fit = np.array([4.3196899, 4.71238898, 5.10508806, 5.49778714, 5.89048623])
    hR_data = hR_lamb_fit_data(lambda_fit, pz_fit)

    #lambda_fit = cc_lamb[11:16]
    #hR_data =    cc_input[11:16]

    hR_data_mean = np.mean(hR_data, axis=1)
    hR_data_std  = np.std( hR_data, axis=1)
    
    c_inv = covariance_matrix(hR_data,'boot')
    c_inv = replace_small_singular_values(c_inv)
    c_inv = np.linalg.inv(c_inv)

    
    
    plot_scatter_with_correlation(hR_data, [0, 1], f'{pz_fit}', [lambda_fit, par_ini])
    
    data_num = len(lambda_fit)

    
    fit_results_list = []

    id_for_fit = range(hR_data.shape[1])
    
    
    

    def cost_function(par):
        l1, a1, lambda0 = par 
        hR_data_i = hR_data_mean
    
        hR_th = hR_lambda_fit_form(lambda_fit, l1, a1, lambda0)

        

        del_hR = (hR_th - hR_data_i )
        chi2 = del_hR.T @ c_inv @ del_hR

        mean_chi2 = chi2 / data_num

        print(l1, a1, lambda0, mean_chi2)
        pdb.set_trace()
        
        return mean_chi2


    
    par_name = ('l1',   'a1',   'lambda0')
    
    
    m = Minuit(cost_function, par_ini, name=par_name)

    m.limits["lambda0"] = (0.0, 100) 
    #m.limits["l1"] = (-50, 50)  
    m.limits["a1"] = (-100, 100)  
    #m.fixed[  'k']= True
    
    m.migrad()

    l1_fit, a1_fit, lambda0_fit = m.values
    chi_all_mean = cost_function(m.values)
    print('.........................')
    print('l1c=', l1_fit)
    print('a1c=', a1_fit)
    print('lambda0c=', lambda0_fit)
    print('chi_meanc=', chi_all_mean)
    print('.........................')

 
    return [l1_fit, a1_fit, lambda0_fit], chi_all_mean


def hR_lambda(lambda_, Pz_, lambda_extra_, l1_fit_, a1_fit_, lambda0_fit_):

    lamb, hR_lamb = hR_z_Pz(z_set[0], Pz_ , hR_pz[Pz_], hR_pz[0], conf)

    res = np.zeros( (len(lambda_), np.shape(hR_lamb)[1])  ) 
    interpolation_function = interp1d(lamb, hR_lamb, kind='linear', axis=0)
    
    mask = ( lambda_ < lambda_extra_)
    res[mask]  = interpolation_function(lambda_[mask])
    res[~mask] = hR_lambda_fit_form(lambda_[~mask], l1_fit_[:, None], a1_fit_[:, None], lambda0_fit_[:, None]).T

    #pdb.set_trace()

    #pdb.set_trace()

    
    return res

def hR_x(x_, Pz_, lambda_extra_, l1_fit_, a1_fit_, lambda0_fit_): #傅里叶变换后的 hR_x，输入x_必须为向量
    inte_down  = 0
    inte_up    = 300.
    bin_num    = 1600


    lambda_bin = np.linspace(inte_down, inte_up, bin_num)

    

    delta_bin  = (inte_up - inte_down) / bin_num 

    #pdb.set_trace()
    #广播操作设置
    hR_broadcast     = hR_lambda(lambda_bin, Pz_, lambda_extra_, l1_fit_, a1_fit_, lambda0_fit_)[:, :, None]
    x_broadcast      = x_[None, None, :]
    lambda_broadcast = lambda_bin[:, None, None]

    #pdb.set_trace()

    hR_x   =  2.* delta_bin * hR_broadcast * np.cos( x_broadcast  * lambda_broadcast)/ 2. / pi
    hR_x   = np.sum(hR_x, axis=0).T

    np.savez(f'{this_path}/result/hR_x/{hR_lamb_name}{append_note}_FeynmenHellman_mu{mu_input}pz{Pz_}'  , x = x_ , hRx = hR_x)
   
    return hR_x

def lambda_extrapolation():
    #par_ini = [0.46191917633411583,-8.287029492467427,0.3552916269882499]
    par_ini = [1.0748787457241349,-1.6393289803758786,1.4677585459926972]
    #par_ini = [0.07755464732083407,-3.884829888972699,1.2078597825532202]
    
 
    
    for i in range(len(pz_set)):
        
        Pz_i = pz_set[i]
        print(Pz_i)
        lamb_do = lamb_do_set[i]
        lamb_up = lamb_up_set[i]
        #par_input, chi2_test = fit_hR_lambda_mean(par_ini, Pz_i, lamb_do, lamb_up)
        #par_inputc, chi2_testc = fit_hR_lambda_meanc(par_input, Pz_i, lamb_do, lamb_up)
        fit_hR_lambda(par_ini, Pz_i, lamb_do, lamb_up)
    return 0

def draw_hR_lambda():

    for i in range(len(pz_set)):

        Pz_i = pz_set[i]

        lamb_do = lamb_do_set[i]
        lamb_up = lamb_up_set[i]
        #lambda_extra = (lamb_do + lamb_up)/2.
        lambda_extra = lambda_extra_value


        par_fit = pd.read_csv(f'{this_path}/result/hR_lambda_fit_result/{hR_lamb_name}{append_note}_FeynmenHellman_mu{mu_input}pz{Pz_i}_lambda{lamb_do}_{lamb_up}_n{nbin}{cov_name}.csv')
        l1_fit = par_fit['l1'].values
        a1_fit = par_fit['a1'].values
        lambda0_fit = par_fit['lambda0'].values


        Pz_ = Pz_i # int

        a_len = a_len_set[conf]
        Nl    = Nl_set[conf]
        Pz_Gev = Pz_ * 2.* np.pi / (Nl * a_len) 

        lambda_test = np.linspace(0, 25, 300)

        res = hR_lambda(lambda_test, Pz_, lambda_extra,  l1_fit, a1_fit, lambda0_fit)

        hbmean = np.mean(res, axis=1)
        hbstd  = np.std(res, axis=1)

      
      
        # 创建平滑曲线版本
        plt.figure(figsize=(10, 6))

        plt.axvspan(lamb_do, lamb_up, alpha=0.2, color='gray')
        plt.axvline(x=lambda_extra, color='red', linestyle='--', linewidth=2)
        plt.text(lambda_extra + 0.1, plt.ylim()[1]*0.8, f'λ = {lambda_extra}', fontsize=16, color='red')

        # 绘制误差区域（更美观）
        plt.plot(lambda_test, hbmean, 'b-', linewidth=2, label='mean')
        plt.fill_between(lambda_test, hbmean - hbstd, hbmean + hbstd, 
                        alpha=0.3, color='blue', label='±1 std')

        plt.title(f'{conf} Pz = {Pz_Gev:.2f} GeV', fontsize=24)
        plt.xlabel(r'$\lambda$', fontsize=24)
        plt.ylabel(r'$h_R(\lambda)$', fontsize=24)
        plt.xlim(0, 25)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        file_save = f'{this_path}/result/hR_lambda_fit_result/draw/a_res_test/{hR_lamb_name}{append_note}mu{mu_input}'
        os.makedirs(file_save , exist_ok = True)

        # 保存为PDF文件
        plt.savefig(f'{file_save}/hR_lambda_Pz{Pz_}.pdf', dpi=300, bbox_inches='tight')

    return 0

def quasi_pdf():
    
    for i in range(len(pz_set)):
        
        Pz_i = pz_set[i]
        print(f'qusi_pdf:{Pz_i}')
        lamb_do = lamb_do_set[i]
        lamb_up = lamb_up_set[i]
        #lambda_extra  = (lamb_do + lamb_up)/ 2.
        lambda_extra  = lambda_extra_value 
        par_fit = pd.read_csv(f'{this_path}/result/hR_lambda_fit_result/{hR_lamb_name}{append_note}_FeynmenHellman_mu{mu_input}pz{Pz_i}_lambda{lamb_do}_{lamb_up}_n{nbin}{cov_name}.csv')
        l1_fit = par_fit['l1'].values
        a1_fit = par_fit['a1'].values
        lambda0_fit = par_fit['lambda0'].values

        x_set = np.linspace(-2, 2, 600)
        res = hR_x(x_set, Pz_i, lambda_extra,  l1_fit, a1_fit, lambda0_fit)

        hbmean = np.mean(res, axis=1)
        hbstd  = np.std(res, axis=1)

        #print(x_set)
        #print(hbmean)
        #print(hbstd)


    return 0

if __name__ == "__main__":

    
    # lamda 外推拟合
    lambda_extrapolation()
    
    # 画lambda 外推的结果
    draw_hR_lambda()
    
    # .....画 quasi - pdf
    quasi_pdf()
    
    

    
    

 
    
