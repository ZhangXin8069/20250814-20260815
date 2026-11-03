# sigma matrix in Pauli basis
import numpy as np
from ..base.backend import get_backend

# 0
s0 = np.zeros((2, 2), dtype = complex)
s0[0, 0] =  1.0+0.0*1j
s0[1, 1] =  1.0+0.0*1j

# X
s1 = np.zeros((2, 2), dtype = complex)
s1[0, 1] =  1.0+0.0*1j
s1[1, 0] =  1.0+0.0*1j

# Y
s2 = np.zeros((2, 2), dtype = complex)
s2[0, 1] =  0.0-1.0*1j
s2[1, 0] =  0.0+1.0*1j

# Z
s3 = np.zeros((2, 2), dtype = complex)
s3[0, 0] =  1.0+0.0*1j
s3[1, 1] = -1.0+0.0*1j

def sigma(i):
    '''
    The Pauli matrix 
    
    Param:
        i the number of sigma matrix
        0: 1; 1: X; 2: Y; 3: Z
        
    return:
        sigma matrix of Pauli basis
    '''
    backend = get_backend()
    
    if i == 0:
        return backend.asarray(s0)

    elif i == 1:
        return backend.asarray(s1)
    
    elif i == 2:
        return backend.asarray(s2)
    
    elif i == 3:
        return backend.asarray(s3)

def Mom_times_sigma(Mom: list = [0, 0, 0], upto4dim: bool = False):
    '''
    P times sigma matrix [Z, Y, X]
    
    Param:
        Mom: list of Mom it must set the last dim is len 3
        upto4dim: up the dim of matrix from 2 dims of sigma matrix to 4 dims
            [[S, 0], 
             [0, S]] 
        
    return:
        P times vec S
        
    '''
    backend = get_backend()
    from ..base import cached_contract

    # 将 Mom 转换为后端数组（支持任意前导维度）
    Mom_array = backend.asarray(Mom)
    
    # 计算最后一维的范数（模长），保持维度以便广播
    norm = backend.sqrt(backend.sum(Mom_array ** 2, axis=-1, keepdims=True))
    eps = 1e-12  # 小量，用于判断零向量
    
    # 归一化：除以模长，模长接近零时直接置零
    Mom_normalized = backend.where(
        norm > eps,
        Mom_array / norm,
        backend.zeros_like(Mom_array)
    )
    
    # 原始的 sigma 矩阵列表：[σ_z, σ_y, σ_x]
    sigma_array = backend.asarray([sigma(3), sigma(2), sigma(1)])
    # 张量缩并：...a 与 abc 缩并得到 ...bc
    PSigma = cached_contract('...a,abc->...bc', Mom_normalized, sigma_array)
    
    if upto4dim:
        # 将 2x2 结果扩展到 4x4 块对角形式
        PSigma_shape = PSigma.shape[:-2] + (4, 4)
        result = backend.zeros(PSigma_shape, dtype=PSigma.dtype)
        result[..., :2, :2] = PSigma
        result[..., 2:, 2:] = PSigma
    else:
        result = PSigma

    return result

def Mom_cross_sigma(Mom: list = [0, 0, 0], upto4dim: bool = False):
    '''
    P cross sigma matrix [Z, Y, X]

    Param:
        Mom: list or array, last dim must be length 3, ordered as [Z, Y, X]
        upto4dim: if True, expand each 2x2 matrix to 4x4 block diagonal
            [[S, 0],
             [0, S]]

    return:
        P × σ : shape (..., 3, 2, 2) or (..., 3, 4, 4) if upto4dim
        The component axis (size 3) corresponds to Z, Y, X.
    '''
    
    backend = get_backend()
    from ..base import cached_contract, levi_civita_tensor
    # Pauli matrices in order: sigma_z, sigma_y, sigma_x
    sigma_array = backend.asarray([sigma(3), sigma(2), sigma(1)])  # (3, 2, 2)

    # Levi-Civita tensor for cross product in Z, Y, X basis
    # Indices: 0=Z, 1=Y, 2=X
    levi = levi_civita_tensor(3)

    # Contract: P_j * σ_k * ε_{ijk} -> (i, b, c)
    PSigma = cached_contract('...j,kbc,ijk->...ibc',
                      backend.asarray(Mom), sigma_array, levi)

    if upto4dim:
        PSigma_shape = PSigma.shape[:-2] + (4, 4)
        result = backend.zeros(PSigma_shape, dtype=PSigma.dtype)
        result[..., :, :2, :2] = PSigma
        result[..., :, 2:, 2:] = PSigma
    else:
        result = PSigma

    return result