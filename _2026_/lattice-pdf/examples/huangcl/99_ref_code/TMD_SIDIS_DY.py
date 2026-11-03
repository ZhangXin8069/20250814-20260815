import sys
sys.path.append('/public/home/zengch/All_TMD_dependence')
import numpy as np
from evolution import evo, b_inte_max, evo_pi
import math
from CNLO import  f1, d1
import lha
from scipy.special import gamma
from scipy.special import jv
from scipy.special import jn_zeros
from constant import inte_nquad, inte
import Parametric_form as Pf
import matplotlib.pyplot as plt
import time
#程序中涉及的常数

gammaE=np.euler_gamma
pi=math.pi

PpD=0.925
PnD=0.925
Pp=-0.028
Pn=0.86

sivers_form = 'Y_f1' #  Y_f1 和 transversity 文章一致
#sivers_form = 'N_f1' # N_f1 和sivers 文章一致

flavor_use_up=[2, 1, 3, -2, -1, -3]
flavor_use_down=[2, 1, 3, -2, -1, -3]

flavor_use_w_plu=[2,  -1]
#flavor_use_w_min=[1,  -2]
flavor_use_w_min=[1, -2]

sw2=0.231
cw2=0.769

Mz=91.2
Mw=80.4

Gz=2.5
Gw=2.1

nzero=8
jn0=jn_zeros(0, nzero)[-1]
jn1=jn_zeros(1, nzero)[-1]


flavor_charge={2:2.0/3.0, 1:-1.0/3.0, 3:-1.0/3.0, 4:2.0/3.0, 5:-1.0/3.0, -2:-2.0/3.0, -1:1.0/3.0, -3:1.0/3.0, -4:-2.0/3.0, -5:1.0/3.0}
flavor_iso_spin={2: 1./2., 1:-1.0/2.0, 3:-1.0/2.0,  -2: -1.0/2.0, -1: 1.0/2.0, -3: 1.0/2.0}


#m_target={'deuteron':1.876, 'proton':0.938, 'neutron':0.940, '3He':2.808}
m_target={'deuteron':0.939, 'proton':0.939, 'neutron':0.939, '3He':0.939}
par12=[[0.08, 0.12, 0.01, 0.01, 0.01, 0.01], [-0.53, -0.98, 0.16, 0.16, 0.16, 0.16], [21, 29], [3, 3, 5, 5, 5, 5], [-0.07, 0.80, -0.01, 0.01, -0.01, -0.02]]

parwordtest=[[0.0673, 0.1027, 0, 0.0152,  0.0152, 0],  #fit355=4.687270246254235, 1.601705232051371 1.5160502902908 new
            [-0.3707, -0.9841, 0, 0.1303,  0.1303, 0], #fit355=4.719655383187265, 1.6081822594379762 1.5225273176774528
            [10.0129, 24.0757, 0, 0], #fit323=5.034241127341193, 1.7045753055678714 1.6104344253047571
            [3., 3., 5, 5, 5, 5],      #fit277=2.9286045875881435, 1.00709633902043, 0.9887223778905267
            [-0.0618, 0.7004, 0, 0.0118, -0.0106, 0]]

Z0_inte_bin={73 : 13.5,   86.5 : 3.5,   91.2 : 4,   94 : 6.,   100 : 14 } 

#Z0_inte_bin={73 : 10,  83 : 3.5,  86.5 : 2.5,  89 : 1,  90 : 1,  91 : 1,  92 : 1,  93 : 1.,  94 : 1.5, 95.5 : 3.5, 100 : 4.5, 114 :14}

'''
partest=[[0.58, 4.8, 192.],
         [-0.35, -0.51, 2.5],
         [-3.9, 9.4],
         [-0.020, 0.40, 0.90, -0.51]]
'''
partest=[[0.75, 2.8, 203.],
         [-0.34, -0.7, 2.4],
         [-3.9, 5.9],
         [-0.016, 0.36, 0.64, -0.36]]


def CKM(fla1, fla2):
    if fla1==2 and fla2==1:
        return 
    return 0

def flab(V_boson,fla):
    if V_boson=='photon' or V_boson== 'Z0':
        return -fla
    
    elif V_boson =='W+':
        if fla ==2:
            return -1
        if fla ==-1:
            return 2
        else:
            return 'erro in TMD_SIDIS_py/flab'
    
    elif V_boson =='W-':
        if fla ==1:
            return -2
        if fla ==-2:
            return 1
        else:
            return 'erro in TMD_SIDIS_py/flab'
        
    else:
        return 'erro in TMD_SIDIS_py/flab'

'''
def Delta(V_boson, Q):
    if V_boson=='w+':
        Delta=Q**4./((Q**2. -Mw**2.)**2.+Gw**2. *Mw**2.)
    return Delta
'''

def ele2_Q(fla, Q):
    ele=flavor_charge[fla]
    Ti=flavor_iso_spin[fla]

    ele_Q=ele**2 * (1.+ Q**4/((Q**2.-Mz**2.)**2. + Gz**2. * Mz**2. )) 
    ele_Q=ele_Q + (Ti - 2.*ele *sw2)/(2.* sw2 * cw2) * 2.* Q**2. *(Q**2. - Mz**2.)/((Q**2 - Mz**2.)**2. + Gz**2 * Mz**2.)

    return ele_Q 

#..............................sivers fit..................................

def fsiver_all(var, par, flavor):
    x=var[0]
    b=var[1]
    target=var[2]
    mu0=2.
   
    [ru,    rd,    rs,    rub,    rdb,    rsb]=par[0]
    [betau, betad, betas, betaub, betadb, betasb]=par[1]
    [epsu,  epsd,  epss,  epsub,  epsdb,  epssb]=par[2]
    [alpu,  alpd,  alps,  alpub,  alpdb,  alpsb]=par[3]
    [Nu,    Nd,    Ns,    Nub,    Ndb,    Nsb]=par[4]
    
    par_u=[ ru,  betau,  epsu,  alpu, Nu]
    par_d=[ rd,  betad,  epsd,  alpd, Nd]
    par_s=[ rs,  betas,  epss,  alps, Ns]
    par_ub=[rub, betaub, epsub, alpub, Nub]
    par_db=[rdb, betadb, epsdb, alpdb, Ndb]
    par_sb=[rsb, betasb, epssb, alpsb, Nsb]
    
    #定义质子和中子的sivers参数
    p_par={2:par_u, 1:par_d, 3:par_s,  -2:par_ub, -1:par_db, -3:par_sb }
    n_par={2:par_d, 1:par_u, 3:par_s,  -2:par_db, -1:par_ub, -3:par_sb }
    
    
    def fsiver(flapar):#sivers函数，其中有5个参数
        
        fsiver = 0
        if sivers_form == 'Y_f1':
            fsiver = Pf.TMD_PDF(flapar, x,  b,  mu0, flavor)   # 包含f1 shape 与transversity 文章相同 
        elif sivers_form == 'N_f1':
            fsiver = Pf.TMD_PDF_noshape(flapar, x,  b)         # 不包含f1 shape 与sivers 文章相同
        else:
            raise ValueError("sivers_form must be Y_f1 or N_f1")
        return fsiver
    
    if target=='proton':
        return fsiver(p_par[flavor]) 
    elif target=='neutron':
        return fsiver(n_par[flavor]) 
    elif target=='deuteron':
        return (PnD*fsiver(n_par[flavor])+PpD*fsiver(p_par[flavor]))/2.0 
    elif target=='3He':
        return (Pn*fsiver(n_par[flavor])+Pp*2.0*fsiver(p_par[flavor]))/3.0 
    else:
        print('error:target input error')
        return 0

#.............................AUT for theory..........................

def aut_th_up(var, par,  model, order):#最终得到的AUT,由演化因子(evo_fac),TMDpdf(f1,fsiver),以及TMDff(d1)决定
    x=var[0]
    Qf=var[1]
    z=var[2]
    ph=var[3]
    target=var[4]
    hadron=var[5]
    charge=var[6]

    
    Q=math.sqrt(Qf)
    
    
    aut_th_up=0

    def ff_fsi(b):
        ff_fsi=0
        for i in flavor_use_up:
            ff=d1([z,b,hadron,charge], model, i, order)
            fsi=fsiver_all([x,b,target],par, i)
            elec=flavor_charge[i] ** 2. 
            ff_fsi=elec * ff*fsi+ff_fsi
            #print(i, b, fsi, ff)
        return ff_fsi
        
    def aut_inte1(b):
        aut1=b**2*jv(1,b*ph/z)*evo([Q,b], model)*ff_fsi(b)
        return aut1
    
    
    aut_th_up=inte(aut_inte1, 0, b_inte_max , 'none')
    
    
    return -m_target[target]*aut_th_up

def aut_th_down(var, model, order):#最终得到的AUT,由演化因子(evo_fac),TMDpdf(f1,fsiver),以及TMDff(d1)决定
    x=var[0]
    Qf=var[1]
    z=var[2]
    ph=var[3]
    target=var[4]
    hadron=var[5]
    charge=var[6]

    
    Q=math.sqrt(Qf)
    
    
   
    aut_th_down=0
    
        
    def ff_pdf1(b):
        ff_pdf1=0
        for i in flavor_use_down:
            ff=d1([z,b,hadron,charge], model, i, order) 
            pdf1=f1([x,b,target], model, i, order)
            elec=flavor_charge[i] **2.
            ff_pdf1=elec * ff*pdf1+ff_pdf1
        return ff_pdf1
    
    def aut_inte2(b):
        aut2=b*jv(0,b*ph/z)*ff_pdf1(b) * evo([Q,b], model) 
        return aut2
    
   
    aut_th_down=inte(aut_inte2, 0, b_inte_max , 'none')
    
    return aut_th_down

def aut_DY_up(var, par,  model, order):
    exp=var[0]
    V_boson=var[1]
    binn=var[2]

    if V_boson=='W+' or V_boson=='W-':
        Q=np.where(len(var)==7, Mw, var[-1]) # 如果数据长度为7，则Q为w质量，否则默认最后一个值为Q
    elif V_boson=='Z0':
        Q=np.where(len(var)==7, Mz, var[-1]) # 如果数据长度为7，则Q为z质量，否则默认最后一个值为Q
    else:
        Q=var[7]

    
    if binn=='y':
        y=var[3]
        pt=5.3
    elif binn=='pt':
        y=0
        pt=var[3]
    else:
        pt=var[6]

    
    if exp=='star2016':
        sqrt_s=500
        x1=np.exp(y) * Q/sqrt_s
        x2=np.exp(-y) * Q/sqrt_s
        #print(x1, x2)
        hadron='proton'
        target='proton'

    elif exp=='star2024':
        sqrt_s=510
        x1=np.exp(y) * Q/sqrt_s
        x2=np.exp(-y)* Q/sqrt_s
        hadron='proton'
        target='proton'

    else: #exp = compasss
        xN=var[3]
        xpi=var[4]
        x1=xpi
        x2=xN
        hadron='pi'
        target='proton'

    flavor_use=0

    if V_boson=='W-':
        flavor_use=flavor_use_w_min
       
    if V_boson=='W+':
        flavor_use=flavor_use_w_plu
        #flavor_use=[-1]
        
    if V_boson=='Z0' or V_boson=='photon':
        flavor_use=flavor_use_up
    
    
    aut_th_up=0
    
    def ff_pdf1(b):
        ff_pdf1=0
        for i in flavor_use:
            #print(flavor_use)
            if exp=='DYcompass2017' or exp=='DYcompass2023':
                
                if target != 'proton' and hadron != 'pi':
                    raise ValueError("compassDY target must be proton and beam must be pi")
                
                f1_beam=  f1([x1, b, hadron], model, i, order) *evo_pi([Q,b], model) ** 0.5
                f1_targ=  fsiver_all([x2,b, target], par, flab(V_boson, i)) *evo([Q,b], model) ** 0.5
            else:
                f1_beam=  fsiver_all([x1, b, hadron], par, i) *evo([Q,b], model) ** 0.5
                f1_targ=  f1([x2,b, target], model, flab(V_boson, i), order) *evo([Q,b], model) ** 0.5
            
            if V_boson=='W+' or V_boson=='W-':
                ff_pdf1=f1_beam*f1_targ+ff_pdf1
            else:
                elec=flavor_charge[i] ** 2.
                ff_pdf1=f1_beam*f1_targ*elec + ff_pdf1
           
        return ff_pdf1
    
    def aut_inte1(b):
        aut1=b**2.*jv(1,b*pt)*ff_pdf1(b)
        return aut1

    aut_th_up=-m_target[target]*inte(aut_inte1, 0, b_inte_max ,'none')
    #aut_th_up=-m_target[target]*inte(aut_inte1, 0, jn0/qt , 0)
  
    return aut_th_up

def aut_DY_down(var, model, order):
    exp=var[0]
    V_boson=var[1]
    binn=var[2]

    
    if V_boson=='W+' or V_boson=='W-':
        Q=np.where(len(var)==7, Mw, var[-1]) # 如果数据长度为7，则Q为w质量，否则默认最后一个值为Q
    elif V_boson=='Z0':
        Q=np.where(len(var)==7, Mz, var[-1]) # 如果数据长度为7，则Q为z质量，否则默认最后一个值为Q
    else:
        Q=var[7]

    
    if binn=='y':
        y=var[3]
        pt=5.3
    elif binn=='pt':
        y=0
        pt=var[3]
    else:
        pt=var[6]

    
    if exp=='star2016':
        sqrt_s=500
        x1=np.exp(y) * Q/sqrt_s
        x2=np.exp(-y) * Q/sqrt_s
        hadron='proton'
        target='proton'
    elif exp=='star2024':
        sqrt_s=510
        x1=np.exp(y) * Q/sqrt_s
        x2=np.exp(-y)* Q/sqrt_s
        hadron='proton'
        target='proton'
    else: #exp = compasss
        xN=var[3]
        xpi=var[4]
        x1=xpi
        x2=xN
        hadron='pi'
        target='proton'

    flavor_use=0

    if V_boson=='W-':
        flavor_use=flavor_use_w_min
    
    if V_boson=='W+':
        flavor_use=flavor_use_w_plu
    
    if V_boson=='Z0' or V_boson=='photon':
        flavor_use=flavor_use_up
    
    #print(flavor_use)
    #print(var, x1, x2, Q, pt)
    aut_th_down=0
    
    def ff_pdf1(b):
        ff_pdf1=0
        for i in flavor_use:
            if hadron == 'pi':
                f1_beam= f1([x1, b, hadron], model, i, order) * evo_pi([Q,b], model) ** 0.5
            else:
                f1_beam= f1([x1, b, hadron], model, i, order) * evo([Q,b], model) ** 0.5
           
            
            f1_targ= f1([x2, b, target], model, flab(V_boson, i), order) * evo([Q,b], model) ** 0.5
            
            if V_boson=='W+' or V_boson=='W-':
                ff_pdf1=f1_beam * f1_targ + ff_pdf1
            elif  V_boson=='photon' or V_boson=='Z0':
                elec=flavor_charge[i]**2.
                ff_pdf1=f1_beam * f1_targ * elec +ff_pdf1  
        return ff_pdf1
    
    def aut_inte2(b):
        aut1=b*jv(0,b*pt)*ff_pdf1(b)
        return aut1

    aut_th_down=inte(aut_inte2, 0, b_inte_max , 'none')
    #aut_th_down=inte(aut_inte2, 0, jn0/qt , 0)
    
    return aut_th_down

def aut_DY_up_Z0(var, par, model, order):
    exp=var[0]
    V_boson=var[1]
    Q=np.where(len(var)==7, Mz, var[-1]) # 如果数据长度为7，则Q为z质量，否则默认最后一个值为Q
    y=0
 
    if exp=='star2016':
        sqrt_s=500
        x1=np.exp(y) * Q/sqrt_s
        x2=np.exp(-y) * Q/sqrt_s
        hadron='proton'
        target='proton'
    elif exp=='star2024':
        sqrt_s=510
        x1=np.exp(y) * Q/sqrt_s
        x2=np.exp(-y)* Q/sqrt_s
        hadron='proton'
        target='proton'
    
    else:
        print('errror in aut_DY_up_Z0')
 
   
    flavor_use=flavor_use_up
    
    
    aut_th_up=0
    
    def ff_pdf1(b):
        ff_pdf1=0
        for i in flavor_use:
            f1_beam=  fsiver_all([x1, b, hadron], par, i)
            f1_targ=  f1([x2,b, target], model, flab(V_boson, i), order)
            elec=ele2_Q(i, Q)
            #elec=flavor_charge[i]**2.
            ff_pdf1=f1_beam*f1_targ*elec + ff_pdf1
           
        return ff_pdf1
    
    def aut_inte1(b, pt):
        aut1=b**2.*jv(1,b*pt)*evo([Q,b], model)*ff_pdf1(b)
        return aut1

    aut_th_up=-m_target[target]*inte_nquad(aut_inte1, [0, b_inte_max] , [0.5, 10])
    #aut_th_up=-m_target[target]*inte(aut_inte1, 0, jn0/qt , 0)
  
    return aut_th_up

def aut_DY_down_Z0(var, model, order):
    exp=var[0]
    V_boson=var[1]
    Q=np.where(len(var)==7, Mz, var[-1]) # 如果数据长度为7，则Q为z质量，否则默认最后一个值为Q
    y=0
    
    if exp=='star2016':
        sqrt_s=500
        x1=np.exp(y) * Q/sqrt_s
        x2=np.exp(-y) * Q/sqrt_s
        hadron='proton'
        target='proton'
    elif exp=='star2024':
        sqrt_s=510
        x1=np.exp(y) * Q/sqrt_s
        x2=np.exp(-y)* Q/sqrt_s
        hadron='proton'
        target='proton'
    
    flavor_use=flavor_use_up
    
    aut_th_down=0
    
    def ff_pdf1(b):
        ff_pdf1=0
        for i in flavor_use:
            f1_beam= f1([x1, b, hadron], model, i, order)
            f1_targ= f1([x2, b, target], model, flab(V_boson, i), order)
            elec=ele2_Q(i, Q)
            #elec=flavor_charge[i]**2.
            ff_pdf1=f1_beam*f1_targ*elec+ff_pdf1
           
            
        return ff_pdf1
    
    def aut_inte2(b, pt):
        aut1=b*jv(0,b*pt)*evo([Q,b], model)*ff_pdf1(b)
        return aut1

    aut_th_down=inte_nquad(aut_inte2, [0, b_inte_max ], [0.5, 10])
    
    return aut_th_down

def aut_DY_up_Z0_inteQ(var, par, model, order):

    #...............直接对 Q 积分...............
    #def inte_Q(Q):
    #    var.append(Q)
    #    return aut_DY_up_Z0(var, par, model)
    #inteQ=inte(inte_Q, 73, 114, 0)
    #..........................................

    #............选某些特定点对 Q 积分...........
    inteQ=0
    varb=var[:]
    for Q in Z0_inte_bin:
        varb.append(Q)
        bin_len=Z0_inte_bin[Q]
        aut= aut_DY_up_Z0(varb, par, model, order)
        inteQ= bin_len * aut+ inteQ
    #..........................................
    return inteQ

def aut_DY_down_Z0_inteQ(var,  model, order):

    #...............直接对 Q 积分...............
    #def inte_Q(Q):
    #    var.append(Q)
    #    return aut_DY_down_Z0(var,  model)
    #inteQ=inte(inte_Q, 73, 114, 0)
    #..........................................

    #............选某些特定点对 Q 积分...........
    inteQ=0
    varb=var[:]
    for Q in Z0_inte_bin:
        varb.append(Q)
        bin_len=Z0_inte_bin[Q]
        aut= aut_DY_down_Z0(varb,  model, order)
        inteQ= bin_len * aut+ inteQ
    #..........................................
    return inteQ


if __name__ =="__main__":
    var=['star2024', 'Z0', 'y', 0, 0.056, 0.081, 0.05] 
    '''
    fig=plt.figure()
    Q=np.linspace(73, 114, 100)
    up=[]
    for  i in Q:
        varQ=var+[i]
        print(varQ)
        up.append(aut_DY_up_Z0(varQ, parwordtest, 'SV19'))
    plt.plot(Q, up)
    plt.savefig('picture/FUT.png')
    '''
    star=time.time()
    #aa=aut_DY_up_Z0_inteQ(var, parwordtest, 'SV19')
    aa=aut_DY_up_Z0_inteQ(var, parwordtest, 'SV19')/aut_DY_down_Z0_inteQ(var, 'SV19')
    print(aa)
    end=time.time()
    print(end-star)

 
    
    