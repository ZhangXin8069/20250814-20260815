from opt_einsum import contract
from ..constant.gamma_matrix import gamma

def seq_peram(peram):
    return contract('ab,...bcef,cd->...dafe', gamma(5), peram.conj(), gamma(5))