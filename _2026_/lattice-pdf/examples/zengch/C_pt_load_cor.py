import sys
sys.path.append('/public/home/zengch/LQCD/input_file')
sys.path.append('/public/home/zengch/LQCD/tool')
import numpy as np
import os
import matplotlib.pyplot as plt
from tool import *
import pandas as pd
#from  input_file_L24x72 import *
#from  input_file_L32x64 import *
#from  input_file_L32x96 import *
#from  input_file_L48x144 import *
from  input_file_L32x64_C32P23 import *
#from  input_file_L32x96_pt import *
#from  input_file_L32x64_pt import *
#from  input_file_L24x72_dhx import *
#from  input_file_L32x64_dhx import *
#from  input_file_L32x96_dhx import *
import glob
import time
import pdb # pdb.set_trace()

this_path = '/public/home/zengch/LQCD/renorma'



def test_len(test_conf=None):
    file_path_2pt = glob.glob(twopt_list[0])
    file_path_ops = glob.glob(ope_list[0])

    # 提取配置编号的函数
    def extract_conf_number_2pt(file_path):
        return int(file_path.split(".npy")[0].split("conf")[-1])

    def extract_conf_number_ops(file_path):
        return int(file_path.split(".npz")[0].split("conf")[-1])

    # 提取所有配置编号
    confs_2pt = set(extract_conf_number_2pt(path) for path in file_path_2pt)
    confs_ops = set(extract_conf_number_ops(path) for path in file_path_ops)

    print(f"2pt文件数量: {len(confs_2pt)}, ops文件数量: {len(confs_ops)}")

    # 找出只在2pt中存在的配置编号
    only_in_2pt = confs_2pt - confs_ops
    # 找出只在ops中存在的配置编号
    only_in_ops = confs_ops - confs_2pt
    # 找出共有的配置编号
    common_confs = confs_2pt & confs_ops

    if only_in_2pt:
        print(f"只在2pt中存在的配置编号: {sorted(only_in_2pt)}")
    else:
        print("没有只在2pt中存在的配置编号")

    if only_in_ops:
        print(f"只在ops中存在的配置编号: {sorted(only_in_ops)}")
    else:
        print("没有只在ops中存在的配置编号")

    print(f"共有的配置编号数量: {len(common_confs)}")
    
    # 新增的查询功能
    if test_conf is not None:
        in_2pt = test_conf in confs_2pt
        in_ops = test_conf in confs_ops
        
        print(f"\n查询结果 - 配置编号 {test_conf}:")
        print(f"  在2pt文件中: {'存在' if in_2pt else '不存在'}")
        print(f"  在ops文件中: {'存在' if in_ops else '不存在'}")
        
        if in_2pt and in_ops:
            print("  状态: 两个文件中都存在")
        elif in_2pt:
            print("  状态: 只在2pt文件中存在")
        elif in_ops:
            print("  状态: 只在ops文件中存在")
        else:
            print("  状态: 两个文件中都不存在")
    
    return 0


def average_configurations(Nconf_Nt, N):
    """
    简化版本：将相邻N个配置取平均
    """
    
    num_configs = Nconf_Nt.shape[0]
    num_full_groups = num_configs // N
    
    # 处理完整的分组
    result_full = np.array([np.mean(Nconf_Nt[i*N:(i+1)*N], axis=0) 
                           for i in range(num_full_groups)])
    
    # 处理剩余部分
    remainder = num_configs % N
    if remainder > 0:
        result_remainder = np.mean(Nconf_Nt[-remainder:], axis=0, keepdims=True)
        result = np.vstack([result_full, result_remainder]) if result_full.size > 0 else result_remainder
    else:
        result = result_full
    
    return result


class C2pt:
    def __init__(self):
        """
        访问两点函数数据
        用法: C_2pt.data[w, i, a, b] 其中:
            w: pt 方向      (3)   x, y, z
            i: 组态索引  (N_conf)
            a: 第1维度索引 (Nt) tsource
            b: 第2维度索引 (Nt) tsink
        """
        self.data = []

        for ndir in twopt_list:
            ndir_data = []
            file_path = glob.glob(ndir)
            
            #pdb.set_trace()
            
            #sort_file_path = sorted(file_path, key=lambda x: int(x.split(".npz")[0].split("conf")[-1])) # for cc code

            # dhx 代码
            #...................................
            # 过滤掉包含 "14950" 的文件路径:dhz 代码中Pz = 0 的没有 14950 #只有24x72
            # file_path = [path for path in file_path if "14950" not in path]
            sort_file_path = sorted(file_path, key=lambda x: int(x.split(".npy")[0].split("conf")[-1])) # for dhx code
            #...................................
            
            for filename in sort_file_path:
                
                # filename 是npz 文件时：
                
                #with np.load(filename) as f:
                #     ndir_data.append(f['arr_0'].T)
                     #ndir_data.append(f['twopt'])
                
                f = np.load(filename)
                ndir_data.append(f.T)
                   
                 
            self.data.append(ndir_data)
        self.data = np.array(self.data)  # 形状: (ndir, n_configs, Nt, Nt)
         
      
    def data_delta(self):
        """
        返回应用了相对时间偏移的数据（向量化实现）
        等效于 data[ndir, i, j, (k+j)% Nt]，但使用NumPy广播实现
        
        返回:
            numpy.ndarray: 形状 (ndir, n_configs, Nt, Nt) 的数组
        """
        # 创建k+j的索引网格
        j_indices = np.arange(Nt)[:, np.newaxis]  # shape (Nt, 1)
        k_indices = np.arange(Nt)[np.newaxis, :]  # shape (1, Nt)
        total_indices = (j_indices + k_indices) % Nt  # shape (Nt, Nt)
        
        # 使用高级索引获取所有配置的数据
        return self.data[:, :, np.arange(Nt)[:, np.newaxis], total_indices]

class C3pt:
    """
    访问三点函数数据
    用法: C_3pt.data[w, i, a, b, c, d] 其中:
        w: pt 方向      (3)   x, y, z
        i: 组态索引  (N_conf)
        a: 第一维度索引 (?) z
        b: 第二维度索引 (?) tsink - tsource
        c: 第三维度索引 (Nt) tsource
        d: 第四维度索引 (?) ti    - tsource
    """

    def __init__(self):
        
        self.n_xyz  = len(ope_list)
        self.data   = self.compute_data()

        #pdb.set_trace()

    def ops(self): # w, conf_n, z, ti
        ops_set = []
        for ndir in range(self.n_xyz):
            ndir_data = []
            file_path = glob.glob(ope_list[ndir])

            
        
            #pdb.set_trace()

            sort_file_path = sorted(file_path, key=lambda x: int(x.split(".npz")[0].split("conf")[-1]))

            for filename in sort_file_path:
                
                   
                with np.load(filename) as f:
                    #print(f['ops'][2,4])
                    ndir_data.append(f['ops'])
                    
            ops_set.append(ndir_data)
        
        ops_set = np.array(ops_set)
        ops_set_mean = np.mean(ops_set, axis = 1, keepdims=True)


        ops_ti = ops_set - ops_set_mean # (n_xyz, Nconf, z, ti)

        #visualize_matrix(ops_ti[0, 0], name = 'two_point/ops_L24x72_dhx_ti')

        

        # 增加一个维度t_source, 使得最后一个维度从ti 变为 ti - t_source
        N_ti = ops_ti.shape[-1]
        ops_res = np.zeros(ops_ti.shape + (N_ti,)) 

        delta_ti_indices = np.arange(N_ti)  # [0, 1, 2, ..., N-1]
        l_indices = np.arange(N_ti)[:, None]  # [0, 1, 2, ..., N-1]，增加一个维度用于广播
        
        # 计算 (l + delta_d) % N
        new_indices = (l_indices + delta_ti_indices) % N_ti

        ops_res = ops_ti[..., new_indices] # (n_xyz, Nconf, z, t_source, ti - t_source)
        return  ops_res
    
    def c2pt(self): # w, confi, tsource, tsink - tsource  
        c2pt_data = C2pt().data_delta()
        c2pt_data_mean = np.mean(c2pt_data, axis = 1, keepdims=True)
        #pdb.set_trace()
        return c2pt_data - c2pt_data_mean
    
    def compute_data(self):
        """计算三维相关函数数据"""
        ops_data = self.ops()
        c2pt_data = self.c2pt()

        #pdb.set_trace()

       

        # 扩展维度以匹配形状 # t_i, tsink 只取前31个
        ops_expanded  = ops_data[ :, :,  :z_max,  :, np.newaxis, :N_tsep_3pt]            # (n_xyz, Nconf, z,  t_source,      1          , ti -tsource)


        c2pt_expanded = c2pt_data[:, :, np.newaxis, :, :N_tsep_3pt, np.newaxis]  #    (n_xyz, Nconf, 1,  tsource, tsink - tsource , 1  )

        # 计算元素乘积
        data = ops_expanded * c2pt_expanded           # (n_xyz, Nconf, z,  tsource,  tsink - tsource,  ti-tsource )
        
        data = np.transpose(data, (0, 1, 2, 4, 3, 5)) # (n_xyz, Nconf, z,  tsink - tsource,  tsource,  ti-tsource)
        
        return data
        
def C2pt_deltat(N):
    # 预加载所有数据（形状: [w, N_conf,Nt,Nt]）并进行重采样

    save_path=f'{this_path}/result/delta_matrix/{input_name}_2pt_deltat_matrix_cor{N}.npy'
    all_data = C2pt().data_delta() 

    shape_all_data = np.shape(all_data)
    print(shape_all_data) # [w, N_conf,Nt,Nt]
    
   
    # 沿ts轴(第0轴)求平均
    Nconf_Nt_Nt = np.mean(all_data, axis=0)  # 结果形状: [N_conf,Nt, Nt]
    Nconf_Nt = np.mean(Nconf_Nt_Nt, axis=1)  # 结果形状: [N_conf, Nt]

    result = average_configurations(Nconf_Nt, N)

    #c2pt_chose =np.real( Nconf_Nt )
    c2pt_chose =np.real( result )
    print(np.shape(result))

    #print(np.shape(c2pt_chose))  # [N_conf, Nt]

    if Resam_type == 1:
      
        c2pt = bootstrap_new( c2pt_chose, n_resamples= Nconf_n) 
        #c2pt = boot( c2pt_chose,  Nconf_n)
    elif Resam_type ==0 :
        c2pt = jackknife( c2pt_chose )  

    else:
        raise ValueError('Resam_type must 0 of 1.')

    print(np.shape(c2pt))  # [N_conf,Nt] 
    #pdb.set_trace()
    np.save(save_path, c2pt)
    return c2pt

def C3pt_deltat(N):
    # 预加载所有数据（形状: [w, N_conf, Nz_3pt, N_tsep_3pt, Nt, N_tisep_3pt]）
    all_data = C3pt().data 

    save_path=f'{this_path}/result/delta_matrix/{input_name}_3pt_deltat_matrix_cor{N}.npy'
    shape_all_data = np.shape(all_data)
    print(shape_all_data) #[w, N_conf, Nz_3pt, N_tsep_3pt, Nt, N_tisep_3pt]
   
    # 对x,y,z方向，及t_source求平均
    res1 = np.mean(all_data, axis=0)  # 结果形状: [N_conf, Nz_3pt, N_tsep_3pt, Nt, N_tisep_3pt]
    res2 = np.mean(res1, axis=3)      # 结果形状: [N_conf, Nz_3pt, N_tsep_3pt, N_tisep_3pt]

    res3 = average_configurations(res2, N)
    #c3pt_chose =np.real( res2 )
    c3pt_chose =np.real( res3 )

    

    print(np.shape(c3pt_chose))

    if Resam_type == 1:
      
        c3pt = bootstrap_new( c3pt_chose, n_resamples= Nconf_n) 
        #c3pt = boot( c3pt_chose, Nconf_n)
    elif Resam_type ==0 :
        c3pt = jackknife( c3pt_chose )  

    else:
        raise ValueError('Resam_type must 0 of 1.')
    
    print(np.shape(c3pt))

    np.save(save_path, c3pt)

    return c3pt

def load_C2pt_deltat(N):
    """
    从文件加载矩阵并确保格式与C2pt_deltat()输出一致
    返回格式：numpy数组 shape=(N_conf, Nt)
    """

    file_path=f'{this_path}/result/delta_matrix/{input_name}_2pt_deltat_matrix_cor{N}.npy'


    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在，请先运行C2pt_deltat()生成数据")
    
    matrix = np.load(file_path)
    
    
    return matrix

def load_C3pt_deltat(N):
    """
    从文件加载三点关联函数矩阵
    返回:
        numpy数组，形状为(N_conf, z, tsep, tisep)
    
    """
    file_path = f'{this_path}/result/delta_matrix/{input_name}_3pt_deltat_matrix_cor{N}.npy'
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"文件 {file_path} 不存在\n"
            "请先运行C3pt_deltat()生成数据"
        )
    
    # 加载数据
    matrix = np.load(file_path)
    
    return matrix

def Ratio(z: int, t_sep: int, ti_sep: int,  c2pt, c3pt) -> tuple:
    """
    计算三点关联函数与两点关联函数的比值及其误差（使用Jackknife方法）
    
    参数:
        z: z方向的距离索引 
        t_sep: 时间分割索引 
        ti_sep: 插入时间分割索引 
        
    返回:
        (ratio_mean, ratio_std): 
            - ratio_samples : 所有组态的比值
            - ratio_mean: 比值的平均值
            - ratio_std: 比值的标准差
    
    计算步骤:
        1. 加载重采样后的三点和两点关联函数数据
        2. 对每个组态计算比值 c3pt/c2pt
        3. 计算比值的均值和标准差
    """
    # 加载数据
    
    N_conf = c2pt.shape[0]
   
    
    # 计算每个组态的比值
    ratio_samples = [ 
        c3pt[i, z, t_sep , ti_sep] / c2pt[i, t_sep ] for i in range(N_conf)
    ]
    
    # 计算统计量
    n = len(ratio_samples)
    ratio_mean = np.mean(ratio_samples)
    
    if Resam_type == 0:
        # Jackknife标准差公式 (sqrt[(n-1)/n * sum((θ_i - θ_mean)^2)])
        ratio_std = np.sqrt( (n-1) * np.sum((ratio_samples - ratio_mean)**2) /n)
    
    if Resam_type == 1:
        # boststrap标准差公式 (sqrt[1/n * sum((θ_i - θ_mean)^2)])
        ratio_std = np.sqrt(         np.sum((ratio_samples - ratio_mean)**2) /n)
    
    return ratio_samples, ratio_mean, ratio_std

def Ratio_data(N):

    # data:    [z, t_sep, ti_sep, mean, std]
    # samples: [samples0, samples1, ..., samples3000]
    save_path=f'{this_path}/result/Ratio_data/Ratio_data_{input_name}_cor{N}.npz'
    
    c2pt_load = load_C2pt_deltat(N)
    c3pt_load = load_C3pt_deltat(N)

    ratio_samples = []
    ratio_data = []

    for z in range(Nz_3pt):
        for t_sep in range(N_tsep_3pt):
            for t_isep in range(t_sep  + 1):
                samples, mean, std = Ratio(z, t_sep, t_isep, c2pt_load, c3pt_load)
                data = [z, t_sep , t_isep,  mean, std]  
                #pdb.set_trace()    
                ratio_data.append(data)
                ratio_samples.append(samples)
    
    ratio_samples = np.array( ratio_samples)
    ratio_data    = np.array( ratio_data )

    np.savez(save_path, samples = ratio_samples, data = ratio_data)

    return ratio_samples, ratio_data 

def load_Ratio_data(file_path=f'{this_path}/result/Ratio_data/Ratio_data_{input_name}.npz'):

     # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"文件 {file_path} 不存在\n"
            "请先运行Ratio_data()生成数据"
        )
    
    # 加载数据

    
    all_data = np.load(file_path)

    return all_data

if __name__ == "__main__":
    #test_len()

    star = time.time()
    

    # 看两点关联函数的样子
    #a = C2pt().data
    #visualize_matrix(a[0, 0], name = 'two_point/two_point_L48x144_dhx')
    #pdb.set_trace()  

    # 看三点关联函数的样子
    #b = C3pt().ops()
    #a = C3pt().c2pt()
    
    #pdb.set_trace() 

    
    # 初始化
    
    #c2pt = C2pt_deltat()
    #c3pt = C3pt_deltat()
    for i in range(20):
        #c2pt = C2pt_deltat(i+1)
        #c3pt = C3pt_deltat(i+1)
        Ratio_data(i + 1)
   

   

    
    
    # 使用示例
    # data:    [z, t_sep, ti_sep, mean, std]
    # samples: [samples0, samples1, ..., samples1N]
    #save_path_a=f'{this_path}/result/Ratio_data/Ratio_data_L48x144_cor_pz0.npz'
    #save_path_b=f'{this_path}/result/Ratio_data/Ratio_data_L48x144_pz0.npz'

    #a = load_Ratio_data( save_path_a )
    #b = load_Ratio_data( save_path_b )

    #print(a['data'][0])
    #print(b['data'][0])


    
    
    #print(np.shape(a['data']))
    #print(np.shape(a['samples']))

    #pdb.set_trace()

    #pdb.set_trace()


    #for i in range(100):
    #    print(a['data'][i],a['samples'][i][0])
    #print(a['samples'][0])

  

    
    




    

    


   
