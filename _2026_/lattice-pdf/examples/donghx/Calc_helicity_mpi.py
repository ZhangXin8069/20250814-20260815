import numpy as np
import os
import time
import sys
import fileinput
from opt_einsum import contract
from mpi4py import MPI
from mpi4py.util import pkl5

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def plaquette_clover_single_time(gauge_slice, mu, nu, periodic=True):
    """
    计算单个时间片上的clover plaquette

    Args:
        gauge_slice: 单个时间片的规范场，形状为 (Nz, Ny, Nx, 4, 3, 3)
        mu, nu: 方向索引 (0,1,2,3)
        periodic: 是否使用周期性边界条件
    """
    Nz, Ny, Nx = gauge_slice.shape[:3]

    # 为每个方向创建滚动的规范场
    if periodic:
        roll_func = np.roll
    else:
        # 如果不需要周期性边界条件，可以使用自定义函数
        roll_func = lambda x, shift, axis: np.roll(x, shift, axis)

    # 对空间维度进行滚动（注意：这里只有3个空间维度）
    gauge_rightup = gauge_slice

    # 计算滚动后的规范场
    # 注意：这里我们需要处理4维索引到3维索引的映射
    dim_map = {0: 2, 1: 1, 2: 0}  # 将4D索引映射到3D索引
    if mu < 3:
        mu_3d = dim_map[mu]
        gauge_leftup = roll_func(gauge_slice, 1, 2 - mu_3d)
    else:
        # mu=3是时间方向，对于单个时间片，时间方向的滚动需要特殊处理
        # 这里我们假设gauge_slice已经包含了时间方向的信息
        gauge_leftup = gauge_slice  # 时间方向需要外部处理

    if nu < 3:
        nu_3d = dim_map[nu]
        gauge_rightdown = roll_func(gauge_slice, 1, 2 - nu_3d)
        gauge_leftdown = roll_func(gauge_leftup, 1, 2 - nu_3d)
    else:
        # nu=3是时间方向
        gauge_rightdown = gauge_slice
        gauge_leftdown = gauge_leftup

    # 计算四个方向的plaquette
    # P_mu,nu
    if mu < 3 and nu < 3:
        mu_3d = dim_map[mu]
        nu_3d = dim_map[nu]

        # pla_rightup = U_mu(x) * U_nu(x+mu) * U_mu^†(x+nu) * U_nu^†(x)
        U1 = gauge_rightup[..., mu, :, :]
        U2 = roll_func(gauge_rightup, -1, 2 - mu_3d)[..., nu, :, :]
        U3 = roll_func(gauge_rightup, -1, 2 - nu_3d)[..., mu, :, :].conj()
        U4 = gauge_rightup[..., nu, :, :].conj()

        pla_rightup = np.einsum("zyxab,zyxbc->zyxac", U1, U2)
        pla_rightup = np.einsum("zyxab,zyxcb->zyxac", pla_rightup, U3)
        pla_rightup = np.einsum("zyxab,zyxcb->zyxac", pla_rightup, U4)

        # P_nu,-mu
        U1 = roll_func(gauge_leftup, -1, 2 - mu_3d)[..., nu, :, :]
        U2 = roll_func(gauge_leftup, -1, 2 - nu_3d)[..., mu, :, :].conj()
        U3 = gauge_leftup[..., nu, :, :].conj()
        U4 = gauge_leftup[..., mu, :, :]

        pla_leftup = np.einsum("zyxab,zyxcb->zyxac", U1, U2)
        pla_leftup = np.einsum("zyxab,zyxcb->zyxac", pla_leftup, U3)
        pla_leftup = np.einsum("zyxab,zyxbc->zyxac", pla_leftup, U4)

        # P_-mu,-nu
        U1 = roll_func(gauge_leftdown, -1, 2 - nu_3d)[..., mu, :, :].conj()
        U2 = gauge_leftdown[..., nu, :, :].conj()
        U3 = gauge_leftdown[..., mu, :, :]
        U4 = roll_func(gauge_leftdown, -1, 2 - mu_3d)[..., nu, :, :]

        pla_leftdown = np.einsum("zyxba,zyxcb->zyxac", U1, U2)
        pla_leftdown = np.einsum("zyxab,zyxbc->zyxac", pla_leftdown, U3)
        pla_leftdown = np.einsum("zyxab,zyxbc->zyxac", pla_leftdown, U4)

        # P_-nu,mu
        U1 = gauge_rightdown[..., nu, :, :].conj()
        U2 = gauge_rightdown[..., mu, :, :]
        U3 = roll_func(gauge_rightdown, -1, 2 - mu_3d)[..., nu, :, :]
        U4 = roll_func(gauge_rightdown, -1, 2 - nu_3d)[..., mu, :, :].conj()

        pla_rightdown = np.einsum("zyxba,zyxbc->zyxac", U1, U2)
        pla_rightdown = np.einsum("zyxab,zyxbc->zyxac", pla_rightdown, U3)
        pla_rightdown = np.einsum("zyxab,zyxcb->zyxac", pla_rightdown, U4)

        # 计算反厄米部分
        ans = (
            pla_rightup
            - pla_rightup.conj().transpose(0, 1, 2, 5, 4)
            + pla_leftup
            - pla_leftup.conj().transpose(0, 1, 2, 5, 4)
            + pla_leftdown
            - pla_leftdown.conj().transpose(0, 1, 2, 5, 4)
            + pla_rightdown
            - pla_rightdown.conj().transpose(0, 1, 2, 5, 4)
        )

        return -1j * ans / 8.0

    else:
        # 处理包含时间方向的情况
        # 对于包含时间方向的plaquette，需要特殊处理
        # 这里返回零数组作为占位符
        return np.zeros((Nz, Ny, Nx, 3, 3), dtype=complex)


def plaquette_clover_all_tilde(pla_all, Nt, Nx):  # mu,nv: 0(x),1(y),2(z),3(t)
    pla_tilde = np.zeros((4, 4, Nx, Nx, Nx, 3, 3), dtype=complex)
    pla_tilde = contract(
        "opmn,mnzyxab->opzyxab",
        Tensor4,
        pla_all,
    )
    return pla_tilde


def operators_new_z0_mu2(gauge, zdir, pla, pla_tilde, delta_z, mu, nu, mu2, nu2, Nt):
    op = np.zeros(Nt, dtype=complex)
    z_dir = zdir
    ope = np.roll(pla[mu, nu], -delta_z, 3 - z_dir)
    for _dz in range(delta_z):
        ope = np.einsum(
            "tzyxab,tzyxcb->tzyxac",
            ope,
            np.roll(gauge, -(delta_z - 1 - _dz), 3 - z_dir)[
                :, :, :, :, z_dir, :, :
            ].conj(),
        )
    ope = np.einsum(
        "tzyxab,tzyxbc->tzyxac",
        ope,
        pla_tilde[mu2, nu2],
    )
    for _dz in range(delta_z):
        ope = np.einsum(
            "tzyxab,tzyxbc->tzyxac",
            ope,
            np.roll(gauge, -_dz, 3 - z_dir)[:, :, :, :, z_dir, :, :],
        )
    # print(ope.shape)
    ans = np.trace(ope, axis1=4, axis2=5)
    # print("ans=  ", ans.shape)
    op = np.sum(ans, axis=(1, 2, 3))
    # print("Trace_zp_z0=  ", op)

    return op


def operators_new_z0_mz_mu2_optimized(
    gauge, zdir, pla, pla_tilde, delta_z, mu, nu, mu2, nu2, Nt
):
    """
    内存优化版本，循环Nt维度处理反向位移算符

    Args:
        gauge: 规范场，形状为 (Nt, Nz, Ny, Nx, 4, 3, 3)
        zdir: 方向索引 (0:x, 1:y, 2:z, 3:t)
        pla: clover plaquette数组，形状为 (4, 4, Nt, Nz, Ny, Nx, 3, 3)
        pla_tilde: 另一个clover plaquette数组
        delta_z: 位移大小（传入正数，函数内会取负）
        mu, nu: pla的方向索引
        mu2, nu2: pla_tilde的方向索引
        Nt: 时间维度大小
    """
    # 获取空间维度大小
    Nz, Ny, Nx = gauge.shape[1:4]

    # 初始化结果数组
    op = np.zeros(Nt, dtype=complex)

    # 将delta_z取负（反向）
    delta_z_neg = -delta_z

    # 循环每个时间片
    for t in range(Nt):
        # 获取当前时间片的数据
        gauge_t = gauge[t]  # 形状: (Nz, Ny, Nx, 4, 3, 3)
        pla_t = pla[mu, nu, t]  # 形状: (Nz, Ny, Nx, 3, 3)
        pla_tilde_t = pla_tilde[mu2, nu2, t]  # 形状: (Nz, Ny, Nx, 3, 3)

        # 根据方向选择滚动轴
        if zdir < 3:  # 空间方向
            # 映射4维索引到3维索引
            spatial_axis_map = {0: 2, 1: 1, 2: 0}  # x->2, y->1, z->0
            axis = spatial_axis_map[zdir]

            # 第一步: 滚动pla_t
            # 注意: delta_z_neg是负数，所以-delta_z_neg是正数
            # 原代码: np.roll(pla[mu, nu], -delta_z, 3 - z_dir)
            ope = np.roll(pla_t, -delta_z_neg, axis=axis)

            # 第二步: 进行delta_z_neg次收缩（注意delta_z_neg是负数）
            # 原代码: for _dz in range(0, delta_z, -1):
            for _dz in range(0, delta_z_neg, -1):
                # 计算滚动量
                shift_amount = -(delta_z_neg - _dz)  # 注意: delta_z_neg是负数

                # 获取滚动的规范场
                gauge_shifted = np.roll(gauge_t, shift_amount, axis=axis)

                # 获取zdir方向的规范场
                gauge_dir = gauge_shifted[..., zdir, :, :]

                # 收缩操作: ope * gauge_dir
                ope = np.einsum("zyxab,zyxbc->zyxac", ope, gauge_dir)

            # 第三步: 与pla_tilde_t收缩
            ope = np.einsum("zyxab,zyxbc->zyxac", ope, pla_tilde_t)

            # 第四步: 再次进行delta_z_neg次收缩
            for _dz in range(0, delta_z_neg, -1):
                # 计算滚动量
                shift_amount = -_dz + 1  # 注意: delta_z_neg是负数

                # 获取滚动的规范场
                gauge_shifted = np.roll(gauge_t, shift_amount, axis=axis)

                # 获取zdir方向的规范场并取共轭
                gauge_dir_conj = gauge_shifted[..., zdir, :, :].conj()

                # 收缩操作: ope * gauge_dir_conj^†
                ope = np.einsum("zyxab,zyxcb->zyxac", ope, gauge_dir_conj)

            # 第五步: 计算迹并求和
            ans_t = np.trace(ope, axis1=3, axis2=4)
            op[t] = np.sum(ans_t)

    return op
