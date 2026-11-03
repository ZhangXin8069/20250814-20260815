import sys
sys.path.append('/public/home/zengch/LQCD/input_file')
sys.path.append('/public/home/zengch/LQCD/tool')
import matplotlib.pyplot as plt
#from fit_ratio import R_model, fit_ratio_mean
#from C_pt_load import load_Ratio_data
import numpy as np
import pandas as pd
import os
from data_chose import data_chose
import pdb # 用于调试代码 pdb.set_trace()
#from hB_data import load_hB_data, a_len_set, fm_to_GeV
#from fit_ratio import fit_ratio_mean
#from fit_zr import th_hB, th_ZR, Z_MS
#from fit_hR_big_lambda import hR_z_Pz, hR_lamb_fit_data
#from fit_2pt import th_2pt, fit_2pt_mean
from tool import *
from C_pt_load import C2pt, load_C2pt_deltat
#from fit_E0 import th_E0

def plot_fit_results(): # 画 ratio 的图

    # 拟合的数据
    # .......................................
    # .......................................
    #t_sep_list = [8, 9, 10,11, 12,13, 14,15, 16, 17,18] 
    t_sep_list = [5, 6, 7, 8, 9, 10,11, 12, 13] 
   
    z_chose =  1    # !!! charge
    z_list = [0, z_chose]
    n_remove = 3 # 拟合时去掉头尾的点数
    ratio_name = 'L24x72_dhx_pz2'
   
    all_data = data_chose(z_chose,  t_sep_list, 0 , ratio_name, only_z = 'yes')

    z_set          = all_data[0]
    tsep_set       = all_data[1]
    ti_sep_set     = all_data[2]
    ratio_mean_set = all_data[3]
    err_set        = all_data[4]

    fit_data = {}
    draw_data = {}
    condition_z     = ( z_set[:] == int(z_chose))
    for tsep in t_sep_list:
        condition_tsep  = ( tsep_set[:] ==  int(tsep))
        mask = (condition_z & condition_tsep)
        
        z_draw_all       =  z_set[mask]
        ti_draw_all      =  ti_sep_set[mask]
        ratio_draw_all   =  ratio_mean_set[mask]
        err_draw_all     =  err_set[mask]

        draw_data[tsep] =  (ti_draw_all, ratio_draw_all , err_draw_all)

        z_draw           =  z_set[mask][n_remove : -n_remove]
        ti_draw          =  ti_sep_set[mask][n_remove : -n_remove]
        ratio_draw       =  ratio_mean_set[mask][n_remove : -n_remove]
        err_draw         =  err_set[mask][n_remove : -n_remove]

        fit_data[tsep]   = (z_draw, ti_draw, ratio_draw , err_draw )
    # .......................................
    # .......................................


    # 定义不同 t_sep 的颜色和形状
    colors = [
        'blue', 'orange', 'green', 'red', 
        'purple', 'brown', 'pink', 'gray', 
        'olive', 'cyan', 'magenta', 'lime',
        'teal', 'navy', 'maroon', 'gold', 
        'indigo', 'coral', 'sienna', 'tan', 'beige'
    ]  # 20种颜色

    markers = [
        'o', 's', '^', 'D', 
        'v', 'p', '*', 'h', 
        'H', '8', '<', '>',
        'd', 'P', 'X', 'o', 
        's', '^', 'D', 'v', 'p', '*'
    ]  # 20种标记

    for idx, t_sep in enumerate(t_sep_list):
        ti_sep, mean, err = draw_data[t_sep]  # 原始数据
        
        z_fit, ti_fit, ratio_fit, err_fit = fit_data[t_sep]  # 拟合数据
        
        

        shift_scale = 0.05 # 
        shift = (t_sep - 6 ) * shift_scale

        #print(ti_sep)
        #print(ti_fit)

        # 绘制数据点
        plt.errorbar(ti_sep - t_sep / 2. + shift , mean, yerr=err, fmt=markers[idx], 
                     label=f't_sep={t_sep} data', color=colors[idx], alpha=0.7, capsize=0.01, markersize=1.2, elinewidth=1)

       
 
    # 添加标题和标签
    plt.xlabel('t - t_sep/2', fontsize=14)
    plt.ylabel('Ratio', fontsize=14)
    

    # 设置图例
    plt.legend(fontsize=6)
    #plt.ylim(-0.1, 0.5)

    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.5)

    # 保存图像
    plt.title(f"Pz=2, z={z_chose}")
    plt.savefig(f'picture/ratio/{ratio_name}_z{z_chose}.pdf', bbox_inches='tight')

   

    return 0

def plot_c0_vs_z():
    # 定义文件路径和文件名
    base_path = 'ratio_fit_result'
    '''
    file_names = [
        'c0_vs_z_ex2_test6_18.csv',
        'c0_vs_z_ex2_test7_18.csv',
        'c0_vs_z_ex2_test8_18.csv',
        'c0_vs_z_ex3_test8_18.csv',
        'c0_vs_z_ex3_test9_18.csv',
        'c0_vs_z_ex3_test10_18.csv',
        'c0_vs_z_ex4_test10_18.csv',
        'c0_vs_z_ex4_test11_18.csv',
        'c0_vs_z_ex4_test12_18.csv'
    ]
    
    file_names = [
        'c0_vs_z_3pt_ex2_test7_16.csv',
        'c0_vs_z_3pt_ex2_test8_16.csv',
        'c0_vs_z_3pt_ex3_test8_16.csv',
        'c0_vs_z_3pt_ex3_test9_16.csv',
        'c0_vs_z_3pt_ex3_test10_16.csv',
        'c0_vs_z_3pt_ex4_test9_16.csv',
        'c0_vs_z_3pt_ex4_test10_16.csv',
    ]

    
    file_names = [
        'c0_vs_z_4pt_ex2_test6_14.csv',
        'c0_vs_z_4pt_ex2_test7_14.csv',
        'c0_vs_z_4pt_ex2_test8_14.csv',
        'c0_vs_z_4pt_ex3_test8_14.csv',
        'c0_vs_z_4pt_ex3_test9_14.csv',
        'c0_vs_z_4pt_ex3_test10_14.csv',
        'c0_vs_z_4pt_ex4_test10_14.csv',
    ]
    '''

    file_names = [
        'c0_vs_z_4pt_cc_ex3_test8_14.csv',
        'c0_vs_z_4pt_cc_ex3_test9_14.csv',
    ]

    # 定义颜色列表
    colors = ['red', 'green', 'blue', 'purple', 'yellow', 'black', 'cyan', 'magenta', 'brown']
    #colors = ['red', 'green', 'blue', 'purple', 'yellow', 'black']

    # 创建一个图像
    plt.figure(figsize=(10, 6))  # 设置图像大小

    # 定义偏移量
    offset = 0.1  # 总偏移量
    

    # 遍历每个文件并读取数据
    for i, file_name in enumerate(file_names):
        # 构造完整的文件路径
        file_path = os.path.join(base_path, file_name)
        
        # 读取CSV文件
        data = pd.read_csv(file_path)

        # 提取数据
        z = data['z']
        mean_values = data['c0_z_mean']
        std_values = data['c0_z_std']

        # 修改label，去掉开头10个字符和结尾4个字符
        label = file_name[8:-4]

        # 添加水平偏移量
        z_offset = z + i * offset 

        # 绘制中心值和误差棒
        plt.errorbar(z_offset, mean_values, yerr=std_values, fmt='o', label=label, color=colors[i], alpha=0.7, capsize=1, markersize=2, elinewidth=2)  # 绘制误差棒图

    # 设置图像的标签和标题
    plt.xlabel('z')  # 设置x轴标签
    plt.ylim(-0.2, 0.82)
    #plt.ylim(-0.09, 0.65)
    plt.xlim(0, 20)
    plt.ylabel('c0')  # 设置y轴标签
    plt.title('Mean Values with Error Bars for Each z')  # 设置标题
    plt.legend()  # 添加图例
    plt.grid(True)  # 添加网格

    # 保存图像
    output_file_name = 'picture/c0_vs_z_plot_test.png'  # 定义保存的文件名
    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')  # 保存图像，设置分辨率和边框

    # 显示图像
    plt.show()

    return 0

def par_ratio_analys(file_path = 'ratio_tsep_tisep/L32x96_pz2/boot_tsep9.0_16.0_z1.csv'):
    # 读取文件
    data = pd.read_csv(file_path)

    bins = 80

    save_dir = 'picture/par_distribution/'

    t_sep_list = [ 9, 10, 11, 12, 13, 14, 15, 16] 
    n_remove =  3
    fit_par  = fit_ratio_mean( data_chose(1,  t_sep_list, n_remove, 'L32x96_pz2'), 'boot' )

    count = 0

    # 检查并创建文件夹
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 遍历每个参数并绘制柱状图
    for column in data.columns[1:]:  # 跳过 sample_i 列
        # 提取当前参数的数据
        values = data[column]

        
        fit_mean = fit_par[count]
        count = count + 1

        
        
        # 绘制柱状图
        fig = plt.figure()
        plt.hist(values, bins=bins, edgecolor='black', alpha=0.7)

        # 所有样本的平均值
        mean_value = values.mean()
        plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'{mean_value:.2f}')
        #在平均值线旁边显示平均值的大小
        plt.text(mean_value, plt.ylim()[1] * 0.9, f'All samples mean = {mean_value:.2f}', color='red', ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        # 拟合的平均值
        plt.axvline(fit_mean, color='blue', linestyle='--', linewidth=2, label=f'{fit_mean:.2f}')
        #在平均值线旁边显示平均值的大小
        plt.text(fit_mean, plt.ylim()[1] * 0.7, f'Fit mean  = {fit_mean:.2f}', color='blue', ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        plt.xlabel(f'{column} values')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {column}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 保存图片
        output_path = f'histogram_{column}.png'
        plt.savefig(os.path.join(save_dir, output_path), dpi=300)
        
        # 关闭图像窗口
        plt.close()

def par_analys(file_name = 'L24x72_dhx_pz2'):
    # 读取文件
    file_path = f'par_E0/{file_name}.csv'
    data = pd.read_csv(file_path)

    bins = 80

    save_dir = 'picture/par_distribution/'

    count = 0

    #k_fit = data['k']
    #d_fit = data['d']
    #m0_fit = data['m0']
    #lambda_QCD_fit = data['lambda_QCD']
    #chi2_fit = data['chi2']


    
    #pdb.set_trace()
    # 检查并创建文件夹
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 遍历每个参数并绘制柱状图
    for column in data.columns[1:]:  # 跳过 sample_i 列
        # 提取当前参数的数据
        values = data[column]
        #values_t = values[ values > -10]
        #print(values_t.shape)

        
        count = count + 1

        
        
        # 绘制柱状图
        fig = plt.figure()
        plt.hist(values, bins=bins, edgecolor='black', alpha=0.7)

        # 所有样本的平均值
        mean_value = values.mean()
        std = values.std()
        print(f'std of {column} = {std}')
        plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'{mean_value:.2f}')
        #在平均值线旁边显示平均值的大小
        plt.text(mean_value, plt.ylim()[1] * 0.9, f'All samples mean = {mean_value:.2f}', color='red', ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        plt.xlabel(f'{column} values')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {column}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 保存图片
        output_path = f'histogram_{column}_{file_name}.png'
        plt.savefig(os.path.join(save_dir, output_path), dpi=300)
        
        # 关闭图像窗口
        plt.close()

def hR_vs_z():


    mu  = 2.

    data_L24x72       = load_hB_data('L24x72')
    hB_L24x72         = np.exp(data_L24x72['hB'])
    z_L24x72          = data_L24x72['z']
    hB_mean_L24x72    = np.mean(hB_L24x72, axis=1)
    hB_std_L24x72     = np.std(hB_L24x72, axis=1)
    #c_inv_L24x72      = covariance_matrix_inv(hB_L24x72, 'boot')
    #c_inv_L24x72      = np.linalg.inv(np.diag(hB_std_L24x72**2.))

    
    data_L32x64       = load_hB_data('L32x64')
    hB_L32x64         = np.exp(data_L32x64['hB'])
    z_L32x64          = data_L32x64['z']
    hB_mean_L32x64    = np.mean(hB_L32x64, axis=1)
    hB_std_L32x64     = np.std(hB_L32x64, axis=1)
    #c_inv_L32x64      = covariance_matrix_inv(hB_L32x64, 'boot')
    #c_inv_L32x64      = np.linalg.inv(np.diag(hB_std_L32x64**2.))

    data_L32x96       = load_hB_data('L32x96')
    hB_L32x96         = np.exp(data_L32x96['hB'])
    z_L32x96          = data_L32x96['z']
    hB_mean_L32x96    = np.mean(hB_L32x96, axis=1)
    hB_std_L32x96     = np.std(hB_L32x96, axis=1)
    #c_inv_L32x96      = covariance_matrix_inv(hB_L32x96, 'boot')
    #c_inv_L32x96      = np.linalg.inv(np.diag(hB_std_L32x96**2.))

    
    par_fit = pd.read_csv('ZR_fit_result/res_log_18_cov.csv')

    k_fit  = par_fit['k'].values
    d_fit  = par_fit['d'].values
    m0_fit = par_fit['m0'].values
    lambda_QCD_fit = par_fit['lambda_QCD'].values
    
    ZR_L24x72 = th_ZR(z_L24x72 ,  a_len_set['L24x72'], mu, k_fit[:, None], d_fit[:, None], m0_fit[:, None], lambda_QCD_fit[:, None]).T 
    ZR_L32x64 = th_ZR(z_L32x64 ,  a_len_set['L32x64'], mu, k_fit[:, None], d_fit[:, None], m0_fit[:, None], lambda_QCD_fit[:, None]).T 
    ZR_L32x96 = th_ZR(z_L32x96 ,  a_len_set['L32x96'], mu, k_fit[:, None], d_fit[:, None], m0_fit[:, None], lambda_QCD_fit[:, None]).T 
    
    

    hR_L24x72_data = hB_L24x72 / ZR_L24x72
    hR_L24x72_data_mean = np.mean(hR_L24x72_data, axis=1)
    hR_L24x72_data_std  = np.std(hR_L24x72_data, axis=1)

    hR_L32x64_data = hB_L32x64 / ZR_L32x64
    hR_L32x64_data_mean = np.mean(hR_L32x64_data, axis=1)
    hR_L32x64_data_std  = np.std(hR_L32x64_data, axis=1)


    hR_L32x96_data = hB_L32x96 / ZR_L32x96
    hR_L32x96_data_mean = np.mean(hR_L32x96_data, axis=1)
    hR_L32x96_data_std  = np.std(hR_L32x96_data, axis=1)

    #pdb.set_trace()

    jitter = 0.005  # 调整此值控制错开程度
    z_SET = z_L24x72
    z_jittered_L24x72 = z_SET 
    z_jittered_L32x64 = z_SET + jitter
    z_jittered_L32x96 = z_SET + 2 * jitter

    z_zms = np.linspace(0.05, 1, 20) 
    z_Gev = z_zms / fm_to_GeV

    

    # 绘图设置
    plt.figure(figsize=(10, 6))
    plt.xlabel("z(fm)", fontsize=14)
    plt.ylabel(r"$h_B(z, P_z=0, a)/Z_R(z, a)$", fontsize=14)
    plt.title("hR vs z with Error Bars", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.plot(z_zms, Z_MS(z_Gev, mu), label = 'Pert.NLO')

    # 绘制带抖动的误差棒
    plt.errorbar(z_jittered_L24x72, hR_L24x72_data_mean, yerr=hR_L24x72_data_std,
                fmt='o', color='blue', markersize=1, capsize=1, capthick=1,
                label='L24x72, a = 0.105fm', alpha=0.8)

    plt.errorbar(z_jittered_L32x64, hR_L32x64_data_mean, yerr=hR_L32x64_data_std,
                fmt='s', color='red', markersize=1, capsize=1, capthick=1,
                label='L32x64, a= 0.0897fm', alpha=0.8)

    plt.errorbar(z_jittered_L32x96, hR_L32x96_data_mean, yerr=hR_L32x96_data_std,
                fmt='^', color='green', markersize=1, capsize=1, capthick=1,
                label='L32x96, a= 0.0775fm', alpha=0.8)

    # 添加图例和调整坐标轴范围
    plt.legend(fontsize=12, framealpha=1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.xlim(z_SET.min() - 2*jitter, z_SET.max() + 2*jitter)  # 确保抖动后数据在视图内
    plt.xlim(0, 1.1)
    plt.ylim(0, 4.5)
    # 保存为PDF并显示
    plt.tight_layout()
    plt.savefig("picture/hR.pdf", format='pdf', bbox_inches='tight', dpi=300)





    #pdb.set_trace()
    return 0

def test():
    a =  np.load("ex_co_mon2.npy")
    hist_draw(a[:, 0], bins=80, filename='picture/par_distribution/cc_l1.pdf')
    hist_draw(a[:, 1], bins=80, filename='picture/par_distribution/cc_a1.pdf')

    lamb = a[:, 2]
    hist_draw(lamb, bins=80, filename='picture/par_distribution/cc_lambda0_0_10.pdf')

    hist_draw(a[:, 3], bins=80, filename='picture/par_distribution/cc_chi2.pdf')
    print(np.mean(a,axis=0))
    return 0

def draw_2pt_vs_t():
    input_name = 'L32x64_pz4'
    L32x64_pt0 = load_C2pt_deltat(f'delta_matrix/{input_name}_2pt_deltat_matrix.npy')

    ndelta_t = np.shape(L32x64_pt0)[1]

    ini = 2
    fin = 16

    t_delta  = np.arange(ndelta_t)
    mean_2pt = np.mean(L32x64_pt0, axis= 0)
    std_2pt  = np.std(L32x64_pt0, axis= 0)


    c4_ini =  0.014491864127737421
    c5_ini = 2.6807380682393998
    E0_ini = 0.4763851555753921
    E_delta_ini = 0.5946359779418238

    plt.figure(figsize=(10, 6))
    c4_fit, c5_fit, E0_fit, E_delta_fit = fit_2pt_mean([c4_ini, c5_ini, E0_ini, E_delta_ini], 6, 14, input_name)
    plt.plot(t_delta[ini:fin], th_2pt(t_delta[ini:fin], c4_fit, c5_fit, E0_fit, E_delta_fit))


    plt.errorbar(t_delta[ini:fin], mean_2pt[ini:fin], yerr=std_2pt[ini:fin], 
                fmt='o',  # 数据点样式：圆圈+实线连接
                color='b',  # 颜色
                ecolor='r',  # 误差棒颜色
                capsize=5,  # 误差棒端帽大小
                label='2pt function ±1σ')

    # 添加图例和标签
    plt.xlabel(r'$\Delta t$', fontsize=14)
    plt.ylabel(r'$C^{2pt}(P_z, t)$', fontsize=14)
    plt.title(f'{input_name}', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'picture/two_point_function_{input_name}.png', dpi=300, bbox_inches='tight')
    #pdb.set_trace()
    return 0

def draw_E0():
    par_E0 = pd.read_csv('par_E0/L32x64.csv')

    m_set  = par_E0['m'].values
    k2_set = par_E0['k2'].values
    k3_set = par_E0['k3'].values
    Pz_set = np.linspace(0, 2.2, 30)

    res =  th_E0(Pz_set[:, None], m_set[None, :], k2_set[None, :], k3_set[None, :])

    E0_mean = np.mean(res, axis=1)
    E0_std = np.std(res, axis=1)

    plt.figure(figsize=(8, 6))
    
    # 绘制均值曲线
    plt.plot(Pz_set, E0_mean, 'b-', linewidth=2, label='$E_0$ mean')
    
    # 用 fill_between 绘制标准差范围
    plt.fill_between(Pz_set, 
                     E0_mean - E0_std, 
                     E0_mean + E0_std, 
                     color='blue', alpha=0.2, label='$\pm 1\sigma$')

    # 添加图例和标签
    plt.legend(fontsize=12)
    plt.xlabel('$P_z(GeV)$', fontsize=14)
    plt.ylabel('$E_0(GeV)$', fontsize=14)
    plt.title('L32x64', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('picture/E0_vs_Pz.png', dpi=300, bbox_inches='tight') 

    # 显示图形
    #plt.tight_layout()
   
    return 0

plot_fit_results()
#plot_c0_vs_z()
#par_ratio_analys()
#test()        
#par_analys('L32x64')
#par_analys('L32x64_pz2')
#par_analys('L32x64_pz3')
#par_analys('L32x64_pz4')
#par_analys('L32x64_pz5')


#draw_E0()
#draw_2pt_vs_t()
#hR_vs_z()