import sys
sys.path.append('/public/group/imp/zengch/LQCD/input_file')
sys.path.append('/public/group/imp/zengch/LQCD/tool')
sys.path.append('/public/group/imp/zengch/LQCD/renorma')
sys.path.append('/public/home/zengch/All_TMD_dependence')
import numpy as np
import matplotlib.pyplot as plt
import os
from tool import * 
from iminuit import Minuit
from iminuit.cost import LeastSquares
from hB_data_FeynmenHellman import load_hB_data_FeynmenHellman, a_len_set, fm_to_GeV, Nl_set, pion_mass_set
from constant import inte
import pdb # 用于调试代码 pdb.set_trace()

mu = 2.0
append_note = '_Nremovenew7syslamm1'
FILE_PATH_READ = '/public/group/imp/zengch/LQCD/renorma/result/hR_PDF'
FILE_PATH_SAVE = f'/public/group/imp/zengch/LQCD/renorma/result/hR_res/new_res_mu{mu}_gfdk{append_note}_c.npz'  # 'gx', 'fx', 'lx', 'hx', 'dx', 'bx',  'kx', 'cx'

# CC 映射字典（键值对调版）
conf_to_conf = {
    'C24P29': 'L24x72',
    'E32P29': 'L32x64',
    'F32P30': 'L32x96',
    'H48P32': 'L48x144',
    'C32P23': 'L32x64_C32P23',
    'C32P29': 'L32x64_C32P29',
    'C48P14': 'L48x96_C48P14'
}

def hR_form(var, par):
    mpi_phy = 0.135
    xg0_, fx_, lx_, hx_, dx_, bx_,  kx_, cx_ = par
    a_, pz_, mpi, L_ = var
    #return xg0_ + a_**2 * fx_ + a_ **4 * lx_ + a_**2 * pz_**2 * hx_ + dx_ / pz_**2 + bx_ / pz_**4 + kx_ * (mpi**2 - mpi_phy**2) + cx_ * np.exp(-L_ * a_ * mpi) 
    return xg0_ + a_**2 * fx_ + a_ **4 * lx_ + a_**2 * pz_**2 * hx_ + dx_ / pz_**2 + bx_ / pz_ + kx_ * (mpi**2 - mpi_phy**2) + cx_ * np.exp(-L_ * a_ * mpi)                # 1/pz 次方项
    #return xg0_ + a_**2 * fx_ + a_ **4 * lx_ + (np.sin(a_ * pz_)**2. / (a_**2 * pz_**2) - 1.) * hx_ + dx_ / pz_**2 + bx_ / pz_**4 + kx_ * (mpi**2 - mpi_phy**2) + cx_ * np.exp(-L_ * a_ * mpi) 

def hR_PDF_extrap(conf_, note_name_, pz_, mu_):
    # ===================== 自动判断：是原始数据 还是 CC 数据 =====================
    is_cc = conf_ in conf_to_conf

    if is_cc:
        # CC 数据格式
        conf_real = conf_to_conf[conf_]
        a_len = a_len_set[conf_real]
        Nl = Nl_set[conf_real]
        pion_mass_GeV = pion_mass_set[conf_real]
        Pz_GeV = pz_ * 2 * np.pi / (Nl * a_len)
        data_path = f'{FILE_PATH_READ}/{conf_}{note_name_}_pz{pz_}.npz'
    else:
        # 原始数据格式
        a_len = a_len_set[conf_]
        Nl = Nl_set[conf_]
        pion_mass_GeV = pion_mass_set[conf_]
        Pz_GeV = pz_ * 2 * np.pi / (Nl * a_len)
        data_path = f'{FILE_PATH_READ}/{conf_}{note_name_}{append_note}_FeynmenHellman_mu{mu_}pz{pz_}.npz'

    try:
        data = np.load(data_path)
    except FileNotFoundError:
        print(f"警告: 文件不存在 → {data_path}")
        return None, None, None, None, None

    xx = data['x']
    hR_PDF = data['hR_PDF'] * 2  # 统一乘2

    mask = (xx >= 0) & (xx <= 1)
    xx_filtered = xx[mask]
    hR_PDF_filtered = hR_PDF[mask, :]

    return xx_filtered, hR_PDF_filtered, a_len, Pz_GeV, pion_mass_GeV, Nl

def fit_hR_PDF_extrap():
    """
    万能外推拟合：
    直接在下面 conf_set 里混合写 Lxxx / Cxxx 都可以自动识别！
    """
    # ===================== 你要的输入：混合两种数据 =====================


    conf_set =     ['L24x72',      'L32x64_C32P23', 'L32x64_C32P29', 'L48x96_C48P14',   'L32x64',    'L32x96', 'L36x108', 'L48x144' ]
    note_name_set = ['_dhxmeang1', '_plus',         '',    '_moms4',    '_dhxplusg1',   '_dhx',  '',  '_dhx' ]
  
    #pz_set =       [[4, 5], [4, 5, 6], [4, 5, 6], [7, 8, 9, 10], [4, 5], [3, 4, 5, 6], [4, 5, 6]] # v0 >1.5
    pz_set =       [[4, 5, 6], [5, 6], [5, 6], [7, 8, 9, 10], [4, 5, 6], [4, 5, 6], [4, 5, 6], [4, 5, 6]] # v1 >1.7
  

   

    # 你也可以混用 CC 数据，完全没问题！
    # conf_set =     ['C24P29', 'C32P23', 'L32x64', 'F32P30']
    # note_name_set = ['_cc', '_cc', '_dhxplusg1', '_dhx']
    # pz_set =       [[3,4], [3,4,5], [3,4], [3]]

    xx_set = []
    hR_PDF_set = []
    a_len_array = []
    Pz_GeV_array = []
    pion_mass_GeV_array = []
    Nl_array = []

    par_name = ('xg0', 'fx', 'lx', 'hx', 'dx', 'bx',  'kx', 'cx')
   

    print("===== 开始加载数据（自动识别原始/CC） =====")
    for i in range(len(conf_set)):
        confi = conf_set[i]
        note_namei = note_name_set[i]
        for pz_i in pz_set[i]:
            xx_, hR_PDF_, a_len_, Pz_GeV_, pion_mass_GeV_, Nl_ = hR_PDF_extrap(confi, note_namei, pz_i, mu)
            
            if xx_ is None:
                print(f"跳过: {confi}{note_namei} pz={pz_i}")
                continue
                
            xx_set.append(xx_)
            hR_PDF_set.append(hR_PDF_)
            a_len_array.append(a_len_)
            Pz_GeV_array.append(Pz_GeV_)
            pion_mass_GeV_array.append(pion_mass_GeV_)
            Nl_array.append(Nl_)
            print(f"✅ 已加载: {confi}{note_namei} pz={pz_i}")

    if not xx_set:
        print("❌ 无数据可拟合")
        return None

    xx_set = np.array(xx_set)
    hR_PDF_set = np.array(hR_PDF_set)
    a_len_array = np.array(a_len_array)
    Pz_GeV_array = np.array(Pz_GeV_array)
    pion_mass_GeV_array = np.array(pion_mass_GeV_array)
    Nl_array = np.array(Nl_array)

    xlen_num = xx_set.shape[1]
    data_num = xx_set.shape[0]
    sample_num = hR_PDF_set.shape[2]

    res_set_x = []
    print("\n===== 开始拟合 =====")

    for i_num in range(xlen_num):
        xx_fit = xx_set[:, i_num]
        hR_fit = hR_PDF_set[:, i_num, :]
        hR_fit_std = np.std(hR_fit, axis=1)

        # 检查协方差矩阵是否可逆
        try:
            c_inv = covariance_matrix(hR_fit, 'boot')
            #visualize_matrix_num(c_inv , 'hR_res_c')
            #c_inv = block_diagonal_mask(c_inv, [len(sub) for sub in pz_set])
            #visualize_matrix_num(c_inv , 'hR_res_c_new')

            #c_inv = np.diag(hR_fit_std**2.)
            #visualize_matrix_num(c_inv , 'hR_res_')
           

            c_inv = np.linalg.inv(c_inv)
            #pdb.set_trace()
        except np.linalg.LinAlgError:
            print(f"警告: 协方差矩阵在x点 {i_num} 不可逆，使用单位矩阵")
        
        

        res_set_x_sample = []
        if i_num % 10 == 0:
            print(f"进度: {i_num}/{xlen_num}")
        
        for j_num in range(sample_num):
            hR_data_j = hR_fit[:, j_num]
            
            def cost_function(par):
                var_ = (a_len_array, Pz_GeV_array, pion_mass_GeV_array, Nl_array)
                hR_th = hR_form(var_, par)
                del_hR = hR_th - hR_data_j
                chi2 = del_hR.T @ c_inv @ del_hR
                #print(m.nfit)
                return chi2 / (data_num - m.nfit)

            par_ini = (np.mean(hR_data_j), 0, 0, 0, 0, 0, 0, 0)
            m = Minuit(cost_function, par_ini, name=par_name)

            m.fixed['lx'] = True
            #m.fixed['dx'] = True
            m.fixed['hx'] = True
            m.fixed['bx'] = True
            #m.fixed['kx'] = True
            m.fixed['cx'] = True
            m.migrad()

            res = [xx_fit[0], *m.values, cost_function(m.values)]
            res_set_x_sample.append(res)

        res_set_x.append(res_set_x_sample)

    res_set_x = np.array(res_set_x)
    chi2_all_mean = np.mean(res_set_x[:, :, -1])

    print(f"\n✅ 拟合完成！平均卡方: {chi2_all_mean:.3f}")
    os.makedirs(os.path.dirname(FILE_PATH_SAVE), exist_ok=True)
    np.savez(FILE_PATH_SAVE, hR=res_set_x, name=['x', 'xg0', 'fx', 'lx', 'hx', 'dx', 'bx', 'kx', 'cx', 'chi2mean'])
    print(f"💾 已保存 → {FILE_PATH_SAVE}")
    
    return res_set_x

if __name__ == "__main__":
    print("=" * 60)
    print("      万能联合外推（支持原始数据 + CC 数据混合拟合）")
    print("=" * 60)
    fit_hR_PDF_extrap()