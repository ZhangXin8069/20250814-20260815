from opt_einsum import contract
from .gamma_matrix import gamma

def seq_peram(peram):    
    if type(peram).__module__ == 'cupy':
        import cupy
        return contract('ab,...bcef,cd->...dafe', cupy.asarray(gamma(5)), peram.conj(), cupy.asarray(gamma(5)))
    
    else:
        return contract('ab,...bcef,cd->...dafe', gamma(5), peram.conj(), gamma(5))
        