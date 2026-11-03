import os
import sys
import numpy as np
import time
from opt_einsum import contract

# dir_mu,nv: 0(x),1(y),2(z),3(t)

Tensor4 = np.zeros((4, 4, 4, 4), dtype=complex)

for i in range(0, 4):
    for j in range(0, 4):
        a = 0
        if i > j:
            a += 1.0
        for k in range(0, 4):
            b = 0
            if i > k:
                b += 1.0
            if j > k:
                b += 1.0
            for l in range(0, 4):
                c = 0
                if i > l:
                    c += 1.0
                if j > l:
                    c += 1.0
                if k > l:
                    c += 1.0
                if (a + b + c) % 2 == 0:
                    Tensor4[i, j, k, l] = 1
                elif (a + b + c) % 2 == 1:
                    Tensor4[i, j, k, l] = -1
                lst = [i, j, k, l]
                set_lst = set(lst)
                if len(set_lst) != len(lst):
                    Tensor4[i, j, k, l] = 0

Tensor4 = 0.5 * Tensor4

# 1/2 epsilon_mu,nu,rho,sigma


# __________________________________________________________
# __________________________________________________________


def plaquette_new(gauge, mu, nu, Nt, Nx):
    pla = np.zeros((Nt, Nx, Nx, Nx, 3, 3), dtype=complex)
    pla = np.einsum(
        "tzyxab,tzyxbc->tzyxac",
        gauge[:, :, :, :, mu, :, :],
        np.roll(gauge, -1, 3 - mu)[:, :, :, :, nu, :, :],
    )
    pla = np.einsum(
        "tzyxab,tzyxcb->tzyxac",
        pla,
        np.roll(gauge, -1, 3 - nu)[:, :, :, :, mu, :, :].conj(),
    )
    # ab,cb->ac <=> .T
    pla = np.einsum(
        "tzyxab,tzyxcb->tzyxac",
        pla,
        gauge[:, :, :, :, nu, :, :].conj(),
    )
    # ab,cb->ac <=> .T
    pla = np.einsum(
        "tzyxab,tzyxcb->tzyxac",
        gauge[:, :, :, :, mu, :, :],
        np.roll(gauge, -1, 3 - nu)[:, :, :, :, mu, :, :].conj(),
    )
    # pla = np.roll(gauge.conj(), -1, 3 - nu)[:, :, :, :, mu, :, :]
    # print(pla.shape)
    return pla


def plaquette_clover_new(gauge, mu, nu):
    gauge_rightup = gauge
    gauge_leftup = np.roll(gauge, 1, 3 - mu)
    gauge_rightdown = np.roll(gauge, 1, 3 - nu)
    gauge_leftdown = np.roll(gauge_leftup, 1, 3 - nu)

    pla_rightup = np.einsum(
        "tzyxab,tzyxbc->tzyxac",
        gauge_rightup[:, :, :, :, mu, :, :],
        np.roll(gauge_rightup, -1, 3 - mu)[:, :, :, :, nu, :, :],
    )
    pla_rightup = np.einsum(
        "tzyxab,tzyxcb->tzyxac",
        pla_rightup,
        np.roll(gauge_rightup, -1, 3 - nu)[:, :, :, :, mu, :, :].conj(),
    )
    pla_rightup = np.einsum(
        "tzyxab,tzyxcb->tzyxac",
        pla_rightup,
        gauge_rightup[:, :, :, :, nu, :, :].conj(),
    )
    # P_mu,nu

    pla_leftup = np.einsum(
        "tzyxab,tzyxcb->tzyxac",
        np.roll(gauge_leftup, -1, 3 - mu)[:, :, :, :, nu, :, :],
        np.roll(gauge_leftup, -1, 3 - nu)[:, :, :, :, mu, :, :].conj(),
    )
    pla_leftup = np.einsum(
        "tzyxab,tzyxcb->tzyxac",
        pla_leftup,
        gauge_leftup[:, :, :, :, nu, :, :].conj(),
    )
    pla_leftup = np.einsum(
        "tzyxab,tzyxbc->tzyxac",
        pla_leftup,
        gauge_leftup[:, :, :, :, mu, :, :],
    )
    # P_nu,-mu

    pla_leftdown = np.einsum(
        "tzyxba,tzyxcb->tzyxac",
        np.roll(gauge_leftdown, -1, 3 - nu)[:, :, :, :, mu, :, :].conj(),
        gauge_leftdown[:, :, :, :, nu, :, :].conj(),
    )
    pla_leftdown = np.einsum(
        "tzyxab,tzyxbc->tzyxac",
        pla_leftdown,
        gauge_leftdown[:, :, :, :, mu, :, :],
    )
    pla_leftdown = np.einsum(
        "tzyxab,tzyxbc->tzyxac",
        pla_leftdown,
        np.roll(gauge_leftdown, -1, 3 - mu)[:, :, :, :, nu, :, :],
    )
    # P_-mu,-nu

    pla_rightdown = np.einsum(
        "tzyxba,tzyxbc->tzyxac",
        gauge_rightdown[:, :, :, :, nu, :, :].conj(),
        gauge_rightdown[:, :, :, :, mu, :, :],
    )
    pla_rightdown = np.einsum(
        "tzyxab,tzyxbc->tzyxac",
        pla_rightdown,
        np.roll(gauge_rightdown, -1, 3 - mu)[:, :, :, :, nu, :, :],
    )
    pla_rightdown = np.einsum(
        "tzyxab,tzyxcb->tzyxac",
        pla_rightdown,
        np.roll(gauge_rightdown, -1, 3 - nu)[:, :, :, :, mu, :, :].conj(),
    )
    # P_-nu,mu
    ans = (
        pla_rightup
        - pla_rightup.conj().transpose(0, 1, 2, 3, 5, 4)
        + pla_leftup
        - pla_leftup.conj().transpose(0, 1, 2, 3, 5, 4)
        + pla_leftdown
        - pla_leftdown.conj().transpose(0, 1, 2, 3, 5, 4)
        + pla_rightdown
        - pla_rightdown.conj().transpose(0, 1, 2, 3, 5, 4)
    )
    return -1j * ans / 8.0
    # return pla_rightup


def plaquette_clover_all_new(gauge, Nt, Nx):  # mu,nv: 0(x),1(y),2(z),3(t)
    pla = np.zeros((4, 4, Nt, Nx, Nx, Nx, 3, 3), dtype=complex)
    Temp_zeros = np.zeros((Nt, Nx, Nx, Nx, 3, 3), dtype=complex)
    for _mu in range(4):
        for _nu in range(4):
            #     pla[_mu, _nu] = plaquette_clover_new(gauge, _mu, _nu)
            if _mu == _nu:
                pla[_mu, _nu] = Temp_zeros
            else:
                pla[_mu, _nu] = plaquette_clover_new(gauge, _mu, _nu)
    return pla


def plaquette_clover_all_tilde(pla_all, Nt, Nx):  # mu,nv: 0(x),1(y),2(z),3(t)
    pla_tilde = np.zeros((4, 4, Nt, Nx, Nx, Nx, 3, 3), dtype=complex)
    pla_tilde = contract(
        "opmn,mntzyxab->optzyxab",
        Tensor4,
        pla_all,
    )
    return pla_tilde


def operators_FF_z0(gauge, pla, pla_tilde, delta_z, mu, nu, mu2, nu2):
    op = 0.0 + 0.0 * 1j
    z_dir = 2
    ope = np.roll(pla[mu, nu], -delta_z, 3 - z_dir)
    ope = np.einsum("tzyxab,tzyxbc->tzyxac", ope, pla_tilde[mu2, nu2])
    ans = np.trace(ope, axis1=4, axis2=5)
    # print("ans=  ", ans.shape)
    op = np.sum(ans, axis=(1, 2, 3))
    # print("Trace_zp_z0=  ", op)

    return op


def operators_FF_z0_mz(gauge, pla, pla_tilde, delta_z, mu, nu, mu2, nu2):
    op = 0.0 + 0.0 * 1j
    z_dir = 2
    delta_z = -1 * delta_z
    ope = np.roll(pla[mu, nu], -delta_z, 3 - z_dir)
    ope = np.einsum("tzyxab,tzyxbc->tzyxac", ope, pla_tilde[mu2, nu2])
    ans = np.trace(ope, axis1=4, axis2=5)
    # print("ans=  ", ans.shape)
    op = np.sum(ans, axis=(1, 2, 3))
    # print("Trace_zp_z0=  ", op)

    return op


def operators_new(gauge, pla, pla_tilde, delta_z, mu, nu):
    op = 0.0 + 0.0 * 1j
    ope = pla[mu, nu]
    z_dir = 2
    for _dz in range(delta_z):
        ope = np.einsum(
            "tzyxab,tzyxbc->tzyxac",
            ope,
            np.roll(gauge, -_dz, 3 - z_dir)[:, :, :, :, z_dir, :, :],
        )
    ope = np.einsum(
        "tzyxab,tzyxbc->tzyxac",
        ope,
        np.roll(pla_tilde[mu, nu], -delta_z, 3 - z_dir),
    )
    for _dz in range(delta_z):
        ope = np.einsum(
            "tzyxab,tzyxcb->tzyxac",
            ope,
            np.roll(gauge, -(delta_z - _dz - 1), 3 - z_dir)[
                :, :, :, :, z_dir, :, :
            ].conj(),
        )
    # print(ope.shape)
    ans = np.trace(ope, axis1=4, axis2=5)
    # print("ans=  ", ans.shape)
    op = np.sum(ans, axis=(1, 2, 3))
    # print("Trace_zp=  ", op)

    return ope


def operators_new_shift(gauge, pla, pla_tilde, delta_z, mu, nu):
    op = 0.0 + 0.0 * 1j
    z_dir = 2
    ope = np.roll(pla[mu, nu], -1, 3 - z_dir)
    for _dz in range(delta_z):
        ope = np.einsum(
            "tzyxab,tzyxbc->tzyxac",
            ope,
            np.roll(gauge, -_dz - 1, 3 - z_dir)[:, :, :, :, z_dir, :, :],
        )
    ope = np.einsum(
        "tzyxab,tzyxbc->tzyxac",
        ope,
        np.roll(pla_tilde[mu, nu], -delta_z - 1, 3 - z_dir),
    )
    for _dz in range(delta_z):
        ope = np.einsum(
            "tzyxab,tzyxcb->tzyxac",
            ope,
            np.roll(gauge, -(delta_z - _dz), 3 - z_dir)[:, :, :, :, z_dir, :, :].conj(),
        )
    # print(ope.shape)
    ans = np.trace(ope, axis1=4, axis2=5)
    # print("ans=  ", ans.shape)
    op = np.sum(ans, axis=(1, 2, 3))
    # print("Trace_zp_shift=  ", op)

    return ope


def operators_new_z0(gauge, pla, pla_tilde, delta_z, mu, nu):
    op = 0.0 + 0.0 * 1j
    z_dir = 2
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
        pla_tilde[mu, nu],
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


def operators_new_z0_mz(gauge, pla, pla_tilde, delta_z, mu, nu):
    op = 0.0 + 0.0 * 1j
    z_dir = 2
    delta_z = -1 * delta_z
    ope = np.roll(pla[mu, nu], -delta_z, 3 - z_dir)
    for _dz in range(0, delta_z, -1):
        ope = np.einsum(
            "tzyxab,tzyxbc->tzyxac",
            ope,
            np.roll(gauge, -(delta_z - _dz), 3 - z_dir)[:, :, :, :, z_dir, :, :],
        )
    ope = np.einsum(
        "tzyxab,tzyxbc->tzyxac",
        ope,
        pla_tilde[mu, nu],
    )
    for _dz in range(0, delta_z, -1):
        ope = np.einsum(
            "tzyxab,tzyxcb->tzyxac",
            ope,
            np.roll(gauge, -_dz + 1, 3 - z_dir)[:, :, :, :, z_dir, :, :].conj(),
        )
    # print(ope.shape)
    ans = np.trace(ope, axis1=4, axis2=5)
    # print("ans=  ", ans.shape)
    op = np.sum(ans, axis=(1, 2, 3))
    # print("Trace_zp_z0_mz=  ", op)

    return op


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


def operators_new_z0_mz_mu2(gauge, zdir, pla, pla_tilde, delta_z, mu, nu, mu2, nu2, Nt):
    op = np.zeros(Nt, dtype=complex)
    z_dir = zdir
    delta_z = -1 * delta_z
    ope = np.roll(pla[mu, nu], -delta_z, 3 - z_dir)
    for _dz in range(0, delta_z, -1):
        ope = np.einsum(
            "tzyxab,tzyxbc->tzyxac",
            ope,
            np.roll(gauge, -(delta_z - _dz), 3 - z_dir)[:, :, :, :, z_dir, :, :],
        )
    ope = np.einsum(
        "tzyxab,tzyxbc->tzyxac",
        ope,
        pla_tilde[mu2, nu2],
    )
    for _dz in range(0, delta_z, -1):
        ope = np.einsum(
            "tzyxab,tzyxcb->tzyxac",
            ope,
            np.roll(gauge, -_dz + 1, 3 - z_dir)[:, :, :, :, z_dir, :, :].conj(),
        )
    # print(ope.shape)
    ans = np.trace(ope, axis1=4, axis2=5)
    # print("ans=  ", ans.shape)
    op = np.sum(ans, axis=(1, 2, 3))
    # print("Trace_zp_z0_mz=  ", op)

    return op


def operators_new_z0_mu2_xy(
    gauge, zdir, pla, pla_tilde, delta_z, mu, nu, mu2, nu2, Nt, Nx
):
    op = np.zeros((Nt, Nx, Nx), dtype=complex)
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
    op = np.sum(ans, axis=3 - z_dir)
    # print("Trace_zp_z0=  ", op)

    return op


def operators_new_z0_mz_mu2_xy(
    gauge, zdir, pla, pla_tilde, delta_z, mu, nu, mu2, nu2, Nt, Nx
):
    op = np.zeros((Nt, Nx, Nx), dtype=complex)
    z_dir = zdir
    delta_z = -1 * delta_z
    ope = np.roll(pla[mu, nu], -delta_z, 3 - z_dir)
    for _dz in range(0, delta_z, -1):
        ope = np.einsum(
            "tzyxab,tzyxbc->tzyxac",
            ope,
            np.roll(gauge, -(delta_z - _dz), 3 - z_dir)[:, :, :, :, z_dir, :, :],
        )
    ope = np.einsum(
        "tzyxab,tzyxbc->tzyxac",
        ope,
        pla_tilde[mu2, nu2],
    )
    for _dz in range(0, delta_z, -1):
        ope = np.einsum(
            "tzyxab,tzyxcb->tzyxac",
            ope,
            np.roll(gauge, -_dz + 1, 3 - z_dir)[:, :, :, :, z_dir, :, :].conj(),
        )
    # print(ope.shape)
    ans = np.trace(ope, axis1=4, axis2=5)
    # print("ans=  ", ans.shape)
    op = np.sum(ans, axis=3 - z_dir)
    # print("Trace_zp_z0_mz=  ", op)

    return op
