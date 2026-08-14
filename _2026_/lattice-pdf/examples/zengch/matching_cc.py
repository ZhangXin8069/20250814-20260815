#!/usr/bin/env python3
# encoding: utf-8
# cd /public/home/sunpeng/chen_c/Deutron/03_12_check_s16x3_mom3_sm0_5/ratio_all_particle conda activate mypython3      python3 ratio_deu_sm0.py 0/5 deuteron/proton_u/proton_d
from ctypes import *
import os
import struct
import numpy as np
import h5py as h5
import pandas as pd
import subprocess
import re
import lsqfit
import gvar as gv
import sys
from scipy import interpolate
from scipy import integrate
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
from timeit import default_timer as timer
from sympy import Si




mom=int(sys.argv[1])

mom_val=mom*0.492    #change!


dx=0.02

#x_pos=np.arange(-1.99-0.001,-0.01-0.001+0.005,dx)  
#x_neg=np.arange(0.01+0.001,1.99+0.001+0.005,dx)    

#x_=np.append(x_pos,x_neg)

x_=np.arange(-1.99,1.99+0.005,dx)  

Ca=3
mu=2     #change!
alpha_s=0.296  #change!
Zs=3    #change！
a=0.105    #change!
GeVfm=0.197327





def C(ksi,m,r):
    if ksi>1:
        ker=(1+np.square(ksi))/(1-ksi)*np.log(ksi/(ksi-1))+1+3*Si(((1-ksi)*np.abs(m)))/(np.pi*(1-ksi))
    elif ksi>0 and ksi<1:
        ker=(1+np.square(ksi))/(1-ksi)*(-np.log(np.square(r))+np.log(4*ksi*(1-ksi))-1)+1+3*Si(((1-ksi)*np.abs(m)))/(np.pi*(1-ksi))
    elif ksi<0:
        ker=-(1+np.square(ksi))/(1-ksi)*np.log(-ksi/(1-ksi))-1+3*Si(((1-ksi)*np.abs(m)))/(np.pi*(1-ksi))
    return alpha_s*Cf*ker/(2*np.pi)


def C_gluon_ratio(ksi,m,r):    #m:(np.abs(x_[j])*Zs*a*mom_val)/GeVfm,   r:mu/(np.abs(x_[j])*mom_val)
    if ksi>1:
        ker=2*(1-ksi+ksi**2)**2/(1-ksi) * np.log(ksi/(ksi-1))  +  (11-28*ksi+18*ksi**2-12*ksi**3)/(6*(1-ksi))
    elif ksi>0 and ksi<1:
        ker=2*(1-ksi+ksi**2)**2/(1-ksi)    *  (-np.log(np.square(r)/4)+np.log(ksi*(1-ksi)))   - (15-56*ksi+102*ksi**2-96*ksi**3+ 48*ksi**4) /(6*(1-ksi))
    elif ksi<0:
        ker=-2*(1-ksi+ksi**2)**2/(1-ksi)  * np.log(ksi/(ksi-1))  -  (11-28*ksi+18*ksi**2-12*ksi**3)/(6*(1-ksi))
    ker+=5/6* (  -1/np.abs(1-ksi)  + 2* Si(((1-ksi)*np.abs(m))) /(np.pi*(1-ksi)) )
    return alpha_s*Ca*ker/(2*np.pi)





time1 = timer()
z=np.zeros((x_.shape[0],x_.shape[0]))
for i in range(x_.shape[0]):
    for j in range(x_.shape[0]):
        if i==j:
            z[i][j]=1
            for k in range(x_.shape[0]):
                if k!=i:
                    z[i][j]+=-dx*C_gluon_ratio(x_[k]/x_[i],(np.abs(x_[i])*Zs*a*mom_val)/GeVfm,mu/(np.abs(x_[i])*mom_val))/np.abs(x_[i])
        elif i!=j:
            z[i][j]=dx*C_gluon_ratio(x_[i]/x_[j],(np.abs(x_[j])*Zs*a*mom_val)/GeVfm,mu/(np.abs(x_[j])*mom_val))/np.abs(x_[j])
print("z calculated")
z_inv=np.linalg.inv(z)
#z_inv=z
        
time2 = timer()
print("calculate kernel time :"+str(time2-time1))
        
filename_z_inv="./save_z_inv_mom"+str(mom)+".npy"   #change!
np.save(filename_z_inv,z_inv)

plt_y_list=[-0.15,-0.11,0.11,0.15,0.21,0.31,0.41,0.61,0.71,0.81]


for y in plt_y_list:
    out_plt_z_inv_y="z_inv_slice_mom"+str(mom)+"_y_"+str(y)+".pdf"
    pdfplot=PdfPages(out_plt_z_inv_y)
    plt.plot(x_, z_inv[int((y-(-1.99))/0.02)])
    pdfplot.savefig()
    plt.clf()
    pdfplot.close()   

'''
    i=0
    for n_a,a in enumerate(a_list):

        mom_list=a_mom_dic[a]
        mom_val_square=1/np.square(mom_list) 

        data_ori=data_assmb[i:i+len(mom_list),:,int((x-(-1.99))/0.02)]    # len(mom_list),Nconf
        i=i+len(mom_list)
        data_ori_mean=np.mean(data_ori,axis=1)
        if Resam_type==0:
            data_ori_error=np.std(data_ori,axis=1)*np.sqrt(Nconf-1)
        elif Resam_type==1:
            data_ori_error=np.std(data_ori,axis=1)*np.sqrt(Nconf)/np.sqrt(Nconf-1)
        plt.errorbar(mom_val_square,data_ori_mean,data_ori_error,fmt='x',ms=3,ecolor=a_col_dic[a],elinewidth=1,mfc=a_col_dic[a],mec=a_col_dic[a],capsize=2,label="a="+str(a)) 
        #plt.errorbar(mom_val_square,data_ori_mean,data_ori_error,fmt='.',ms=3,ecolor="green",elinewidth=1,mfc="green",mec="green",capsize=2,label="finite momentum")


        mom_band_square=np.arange(0,mom_val_square[0]+0.015,0.01)
        #print("mom_band_square")
        #print(mom_band_square)   #result_gather=result_gather.reshape(-1,Nx,3+len(a_list))
        
        lc_band=np.zeros((mom_band_square.shape[0],Nconf))

        for n in range(Nconf):
            fit_para={'q':[result_gather[n,int((x-(-1.99))/0.02),0]],'f':[result_gather[n,int((x-(-1.99))/0.02),1]],'h':[result_gather[n,int((x-(-1.99))/0.02),2]],'d':[result_gather[n,int((x-(-1.99))/0.02),3]]}
            lc_band[:,n]=fcn_extrap_momentum_a(1/(np.sqrt(mom_band_square)),fit_para, a)

        lc_band_mean=np.mean(lc_band,axis=1)
        if Resam_type==0:
            lc_band_error=np.std(lc_band,axis=1)*np.sqrt(Nconf-1)
        elif Resam_type==1:
            lc_band_error=np.std(lc_band,axis=1)*np.sqrt(Nconf)/np.sqrt(Nconf-1)
        plt.fill_between(mom_band_square,lc_band_mean-lc_band_error,lc_band_mean+lc_band_error,color=a_col_dic[a], alpha=0.2)





    #result_gather=result_gather.reshape(-1,Nx,4)
    q_inf_mean=np.mean(result_gather[:,int((x-(-1.99))/0.02),0],axis=0)
    if Resam_type==1:
        q_inf_std=np.std(result_gather[:,int((x-(-1.99))/0.02),0],axis=0)*np.sqrt(Nconf)/np.sqrt(Nconf-1)      
    plt.errorbar(np.array([0]),np.array([q_inf_mean]),np.array([q_inf_std]),fmt=".",ms=3,ecolor="black",elinewidth=1,mfc="black",mec="black",capsize=2,label="infinite") 
    #dic_save={"mom_val_square":mom_val_square,"data_ori":data_ori,"mom_band_square":mom_band_square,"lc_band":lc_band}
    #file_save="mom_extrapolated_res_x_"+str(x)+".pickle"
    #with open(file_save, "wb") as file:
        #   pickle.dump(dic_save, file)        
    plt.ylim([-0.5,3])

    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    x_relative = 0.2
    y_relative = 0.3
    x_absolute = (x_relative * (x_max - x_min)) + x_min
    y_absolute = (y_relative * (y_max - y_min)) + y_min

    text ="x="+str(x)
    text+="\n"
    text +="chi2="+"%0.2f"%(mom_extra_sta_mean_chi2[int((x-(-1.99))/0.02)])
    plt.text(x_absolute, y_absolute, text, fontsize=12, color='black', ha='center', va='center')

    plt.xlabel(r"$1/(P_z)^2(GeV^{-2})$",fontsize=14)
    plt.ylabel(r"$xg(x,P_z)$ ",fontsize=14)
    if(x==0.71):
        plt.legend(loc="lower right",fontsize=12)
    else:
        plt.legend(loc="upper right",fontsize=12)

    pdfplot.savefig()
    plt.clf()
    pdfplot.close()    










print("calculate kernel  successfully")
'''
