# sigma matrix in Pauli basis
import numpy as np
from .backend import get_backend

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

def Mom_times_sigma(Mom:list = [0, 0, 0], upto4dim:bool = False):
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
    from opt_einsum import contract
    
    sigma_array = backend.asarray([sigma(3), sigma(2), sigma(1)])
    PSigma = contract('...a,abc->...bc', backend.asarray(Mom), sigma_array)
    
    if upto4dim:
        PSigma_shape = PSigma.shape[:-2] + (4, 4)
        result = backend.zeros(PSigma_shape, dtype = PSigma.dtype)
        result[..., :2, :2] = PSigma
        result[..., 2:, 2:] = PSigma
        
    else:
        result = PSigma

    return result