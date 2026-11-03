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
from fit_zr_new import th_ZR
from hB_data_FeynmenHellman import load_hB_data_FeynmenHellman, a_len_set, fm_to_GeV, Nl_set
from constant import CA, CF, gammaE, pi, A_s, inte
from scipy.interpolate import interp1d
import pdb # 用于调试代码 pdb.set_trace()
#from sympy import Si
from scipy.special import sici

from mpi4py import MPI
#from mpi4py.util import pkl5
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


this_path = '/public/group/imp/zengch/LQCD/renorma'

mu = 4.
append_note = '_Nremovenew7'

def Si(x_):
    return sici(x_)[0]

def hR_tilde(x_, pz_, mu_, conf_, note_name_, append_note_): # x_ 必须为向量
    data_name = f'{conf_}{note_name_}{append_note_}_FeynmenHellman_mu{mu_}pz{pz_}'
    x_hRx = np.load(f'{this_path}/result/hR_x/{data_name}.npz')
    x_ = np.array(x_)

    x_set   = x_hRx['x']

    #pdb.set_trace()
    hRx_set = x_hRx['hRx']
   
    interpolation_function = interp1d(x_set, hRx_set, kind='linear', axis=0)
    res =  interpolation_function(x_)
    
    #hist_draw(res[1])
    return res

def hR_PDF(xx, Pz_, mu_,  conf_,  note_name_, append_note_):  #    m:(np.abs(x_[j])*Zs*a*mom_val)/GeVfm,   r:mu/(np.abs(x_[j])*mom_val)
    data_name = f'{conf_}{note_name_}{append_note_}_FeynmenHellman_mu{mu_}pz{Pz_}'
    dx_ = xx[1]-xx[0]
    hR_0 = hR_tilde( xx, Pz_, mu_, conf_, note_name_, append_note_) 
    
    
    type_ = conf_
   

    #xi_set_, y_,  par_, 
    
    alpha_s = A_s(mu_) * 4. * pi
    a_len = a_len_set[type_]
    Nl    = Nl_set[type_]

    Pz_GeV    = Pz_ * 2.* pi / (Nl * a_len)

    lambda_s = 0.3 / fm_to_GeV * Pz_GeV

    
    yy= xx
    if np.any(yy == 0):
        print("警告: y中包含零元素，将导致g_xy除以零错误")

    n_len = len(xx)

    x_col = xx[:, None]
    y_col = yy[None, :]

    x_matr = np.tile(x_col, (1, n_len))  # 水平复制 n 次，得到 n×n 矩阵
    y_matr = np.tile(y_col, (n_len, 1))

    cxi = x_matr / y_matr

    #visualize_matrix_num(x_matr, name = 'x_matr')
    #visualize_matrix_num(y_matr, name = 'y_matr')
    #visualize_matrix_num(cxi, name = 'cxi')
    #pdb.set_trace()

    def g_1(xi_, y_): # xi < 0 
        res = -2.*(1. - xi_ + xi_**2.) **2. / (1. - xi_)  * np.log(xi_/(xi_-1.))  -  (11.-28.* xi_ + 18. * xi_ **2. - 12. * xi_**3.)/ (6.*(1.- xi_))
        return res

    def g_2(xi_, y_): # 0 < xi <1
        res = 2. * (1.- xi_ +xi_ **2.)**2. / (1.- xi_)    *  (-np.log(mu_**2./ (4. * y_ **2. * Pz_GeV**2.)) + np.log( xi_ *(1. - xi_))) 
        res = res - (15.- 56. * xi_ +102. * xi_ **2. -96. * xi_**3. + 48.* xi_ ** 4.) / ( 6. * ( 1. - xi_))
        return res
    
    def g_3(xi_, y_): # xi > 1
        return 2. * (1. - xi_ +xi_**2)**2/(1-xi_) * np.log(xi_/(xi_-1.))  +  (11.-28.* xi_+18. * xi_** 2.- 12.* xi_**3.)/ (6.*(1.- xi_))
    
    def g_0(xi_, y_):
       
        return 5./6. * ( -1./np.abs(1. - xi_)  + 2.* Si(((1. - xi_)*np.abs(y_) * lambda_s )) /(pi*(1-xi_)) )

    
    mask_cxi_1 = (cxi < 0)
    mask_cxi_2 = (0 < cxi ) & (cxi < 1)
    mask_cxi_3 = (cxi > 1)

    g_xy = np.zeros_like(cxi)
    

    g_xy[mask_cxi_1] = -g_0(cxi[mask_cxi_1] ,  y_matr[mask_cxi_1] ) - g_1( cxi[mask_cxi_1] ,  y_matr[mask_cxi_1] ) 
    g_xy[mask_cxi_2] =  g_0(cxi[mask_cxi_2] ,  y_matr[mask_cxi_2] ) + g_2( cxi[mask_cxi_2] ,  y_matr[mask_cxi_2] ) 
    g_xy[mask_cxi_3] =  g_0(cxi[mask_cxi_3] ,  y_matr[mask_cxi_3] ) + g_3( cxi[mask_cxi_3] ,  y_matr[mask_cxi_3] ) 

    visualize_matrix_num(g_xy, name = 'gxy')
    g_ij = g_xy
    Z_ij = np.zeros_like(g_ij)
    delta_ij = np.identity(len(xx))

    dx_matr = np.diag(xx/abs(xx)) * dx_


    C_alp_LO   = dx_matr * alpha_s * CA / 2. / pi 
    M_alp_LO   = g_ij / y_matr - delta_ij * y_matr *  np.sum( g_ij / (y_matr **2.) , axis = 1 )
    z_ij = delta_ij + C_alp_LO @ M_alp_LO

    hR_PDF = np.linalg.inv(z_ij) @ hR_0 

    #visualize_matrix_num(z_ij, name = 'z_ij')
    #visualize_matrix_num(np.linalg.inv(z_ij), name = 'z_ij_inv')


    np.savez(f'{this_path}/result/hR_PDF/{data_name}',x=xx, hR_PDF = hR_PDF, hR_tilde = hR_0)
    #pdb.set_trace()
    
    return hR_PDF

def light_cone_PDF(rank_):

    conf_set = ['L24x72', 'L32x64_C32P23','L32x64_C32P29', 'L48x96_C48P14',  'L32x64', 'L32x96','L36x108', 'L48x144']
    note_name_set = ['_dhxmeang1', '_plus', '', '_moms4', '_dhxplusg1', '_dhx','', '_dhx']
    pz_set_conf   = [ [4, 5, 6],  [5, 6],  [5, 6], [7, 8, 9, 10] , [4, 5, 6], [4, 5, 6],  [4, 5, 6],  [4, 5, 6]]
    
    #conf_set = ['L24x72', 'L24x72', 'L32x64_C32P23','L32x64_C32P29', 'L32x64', 'L32x96', 'L48x144']
    #note_name_set = ['_cctest', '_dhxmeang1', '_plusm3', '', '_dhxplusg1', '_dhx', '_dhx']
    #pz_set_conf   = [ [3, 4],  [3, 4, 5, 6],  [5, 6, 7, 8],  [3, 4, 5, 6],  [3, 4, 5, 6], [3, 4, 5, 6],  [3, 4, 5, 6]]
    
    #x_test = np.linspace(-2 + 0.0001, 2 -0.0001, 398)
    
    
    conf = conf_set[rank_]
    pz_set = pz_set_conf[rank_]
    note_name = note_name_set[rank_]
    print(conf + note_name)
    for pz in pz_set:

        print(pz)
    
        dx = 0.02
        x_test = np.arange(-1.99,1.99+0.005,dx)
        #x_test = np.linspace(-1.99,1.99, 16)
        #pdb.set_trace()  
        hR_PDF(x_test, pz, mu, conf, note_name, append_note)

    return 0 


light_cone_PDF(rank)

