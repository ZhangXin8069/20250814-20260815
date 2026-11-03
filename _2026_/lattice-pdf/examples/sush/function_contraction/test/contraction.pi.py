# 引入对应的包，地址：/public/home/sush/distillation/function_contraction
import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from function_contraction import *

from opt_einsum import contract, contract_path

import time

#将组态号输入到程序中
conf_id = sys.argv[1]

#设置一个全局变量，调整使用的包 numpy or cupy
set_backend('numpy')
backend = get_backend()

#设置格子大小和进程数量，默认为1
lattice_size = [12, 12, 12, 32]
grid_size = [1, 1, 1, 1]

#读取规范场的函数
def Readin_gauge(conf_file, lattice_size):
    
    Nz, Ny, Nx, Nt = lattice_size
    
    f = open("%s" % conf_file, "rb")
    gauge = backend.fromfile(f, dtype=">f8")
    gauge = backend.array(gauge)

    gauge = gauge.reshape(Nt, Nx, Nx, Nx, 4, 3, 3, 2)
    gauge = gauge[..., 0] + gauge[..., 1] * 1j
    f.close()

    return gauge


Nx, Ny, Nz, Nt = lattice_size
Lx, Ly, Lz, Lt = [lattice_size[x]//grid_size[x] for x in range(len(lattice_size))]

# 创建动量对应的exp^{-ipx}
fun_eigen = corr_eigvecs(Nx = Nx, backend = backend)
phase_exp = fun_eigen.phase_exp_2pt(Mom = [0, 0, 0])

#sink source端用的eigen数量
Nev_src = 50

#current端用的eigen数量
Nev_link = 200

#current端link正反向长度（正反都会计算-link_max:link_max+1）
link_max = 10

#sink与source端的距离
t_sep = 8

#读取规范场
gauge_link = Readin_gauge(
    f'/nexdata/project/lqcd/sush/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L12x32/beta6.20_mu-0.2770_ms-0.2400_L12x32_cfg_{conf_id}.lime.contents/msg02.rec04.ildg-binary-data', 
    lattice_size = lattice_size
    )

#将规范场的维度设置为(Nd, Nt, Nz, Ny, Nx, Nc, Nc)即link方向，时间，Z, Y, X, c, c
gauge_link = gauge_link.transpose(4, 0, 1, 2, 3, 5, 6)

#初始化sink 和 link
sink = backend.zeros((Lt, Nev_src, Nev_src), dtype = complex)
VdV_link = backend.zeros((Lt, 2 * link_max + 1, Nev_link, Nev_link), dtype = complex)

st_eigen = time.perf_counter()
for t_src_indx, t_src in enumerate([x for x in range(Nt)]):
    # 读取eigen
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L12x32/{conf_id}/{conf_id}_t{t_src:03d}_e50_s[0]_[0]_[0]_BI_n150_V1_stout_smear_20_0.12.npy')

    #计算sink端VdV
    sink[t_src_indx] = fun_eigen.Mom_VdV_sink_t_2(phase_exp = phase_exp, eigvecs = eigvecs[:Nev_src])
    
    #计算current端VdV
    VdV_link[t_src_indx] = fun_eigen.VdV_sink_t_link(
        eigvecs = eigvecs[:Nev_link],
        link_dir = 'Z',
        link_max = link_max,
        phase_exp = phase_exp,
        gauge_link = gauge_link,
        t = t_src_indx
    )
print(f'load eigen and calculate sink and VdV of all t use time {(time.perf_counter() - st_eigen):.3f} s')

# 将current端的VdV乘以权重
VdV_link = VdV_link * fun_eigen.create_omega_accelerate(
    exact = 50,
    N_eigen = [0],
    N_sum = [0],
    N_extract = [0],
    noise = 150,
)

# 初始化结果
corr_2pt = backend.zeros((Nt, Lt), dtype = complex)
corr_3pt_con = backend.zeros((2 * link_max + 1, Nt, Lt), dtype = complex)
bubble = backend.zeros((2 * link_max + 1, Nt), dtype = complex)

for t_src in range(Nt):
    st_cal = time.perf_counter()
    
    # 通过sink端VdV计算source端VdV
    source = data = sink[t_src].transpose(1, 0).conj()

    # 读取从source开始传播的传播子
    peram_u_src = backend.load(
        f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L12x32/light/{conf_id}/t{t_src:03d}_e50_s[0]_[0]_[0]_BI_n150_V1_stout_smear_20_0.12.npy'
        )
    
    # 对其进行\gamma_5 P^{\dagger} \gamma_5 反向
    peram_d_src = seq_peram(peram_u_src)
    
    for t_sink in range(Nt):
        corr_2pt[t_src, t_sink] = contract(
            'manb,cedf,bd,fn,ac,em->',
            peram_u_src[t_sink, :, :, :Nev_src, :Nev_src],
            peram_d_src[t_sink, :, :, :Nev_src, :Nev_src],
            source,
            sink[t_sink],  
            backend.asarray(gamma(5)),
            backend.asarray(gamma(5)),
            )
    
    # 读取从sink端开始传播的传播子
    peram_u_sep = backend.load(
        f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L12x32/light/{conf_id}/t{(t_src + t_sep)%Nt:03d}_e50_s[0]_[0]_[0]_BI_n150_V1_stout_smear_20_0.12.npy'
        )
    
    # 对其进行\gamma_5 P^{\dagger} \gamma_5 反向
    peram_u_sep = seq_peram(peram_u_sep)
    
    # 提取sink到source的传播子
    peram_d_2pt = peram_d_src[(t_src + t_sep)%Nt]
    
    #提取sink端VdV
    sink_3pt = sink[(t_src + t_sep)%Nt]
    
    #计算一个GV的流
    for t_curr in range(Nt):
        corr_3pt_con[:, t_src, t_curr] = contract(
            'manb,gehf,codp,bd,Lfn,ph,ac,em,og->L',
            peram_u_src[t_curr, :, :, :Nev_link, :Nev_src],
            peram_u_sep[t_curr, :, :, :Nev_src, :Nev_link],
            peram_d_2pt[:, :, :Nev_src, :Nev_src],
            source,
            VdV_link[t_curr],
            sink_3pt,
            backend.asarray(gamma(5)),
            backend.asarray(gamma(4)),
            backend.asarray(gamma(5)),
        )
    
    # 计算一个quark loop
    bubble[:, t_src] = contract(
        'menf,Lfn,em->L',
        peram_u_src[t_src],
        VdV_link[t_src],
        backend.asarray(gamma(4))
    )
    
    print(f'load peram and calculate 2pt, 3pt and bubble of t_src{t_src} use time {(time.perf_counter() - st_cal):.3f} s')
    
# 存入数据 需要自己输入，修改！
# backend.save(f'.../conf{conf_id}/corr_ud_2pt_gamma0505_u_e50_s[0]_[0]_[0]_BI_n150_stout_smear_20_0.12_dul_vector_False_src{Nev_src}.npy', corr_2pt)

# for link_indx in range(-link_max, link_max + 1, 1):
#     backend.save(f'.../conf{conf_id}/corr_ud_3pt_gamma050405_tseq{t_sep}_link_indx{link_indx}_u_e50_s[0]_[0]_[0]_BI_n150_stout_smear_20_0.12_dul_vector_False_src{Nev_src}_curr{Nev_link}.npy', corr_3pt_con[link_indx + link_max])
#     backend.save(f'.../conf{conf_id}/corr_ud_bubble_gamma04_link_indx{link_indx}_u_e50_s[0]_[0]_[0]_BI_n150_stout_smear_20_0.12_dul_vector_False_curr{Nev_link}.npy', bubble[link_indx + link_max])
        