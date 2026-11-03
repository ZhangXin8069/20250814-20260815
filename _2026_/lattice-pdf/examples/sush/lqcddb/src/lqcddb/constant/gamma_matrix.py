# gamma matrix in DR basis
import numpy as np
import sys
from ..base.backend import get_backend
#identity
g0=np.zeros((4,4),dtype=complex)
g0[0,0]=1.0+0.0*1j
g0[1,1]=1.0+0.0*1j
g0[2,2]=1.0+0.0*1j
g0[3,3]=1.0+0.0*1j

#gamma1
g1=np.zeros((4,4),dtype=complex)
g1[0,3]=0.0+1.0*1j
g1[1,2]=0.0+1.0*1j
g1[2,1]=0.0-1.0*1j
g1[3,0]=0.0-1.0*1j

#gamma2
g2=np.zeros((4,4),dtype=complex)
g2[0,3]=-1.0+0.0*1j
g2[1,2]=1.0+0.0*1j
g2[2,1]=1.0+0.0*1j
g2[3,0]=-1.0+0.0*1j

#gamma3
g3=np.zeros((4,4),dtype=complex)
g3[0,2]=0.0+1.0*1j
g3[1,3]=0.0-1.0*1j
g3[2,0]=0.0-1.0*1j
g3[3,1]=0.0+1.0*1j

#gamma4
g4=np.zeros((4,4),dtype=complex)
g4[0,2]=1.0+0.0*1j
g4[1,3]=1.0+0.0*1j
g4[2,0]=1.0+0.0*1j
g4[3,1]=1.0+0.0*1j

#gamma5
g5=np.zeros((4,4),dtype=complex)
g5[0,0]=1.0+0.0*1j
g5[1,1]=1.0+0.0*1j
g5[2,2]=-1.0+0.0*1j
g5[3,3]=-1.0+0.0*1j

def gamma(i):
	backend = get_backend()

	if i==0: #identity
		return backend.asarray(g0)
		
	elif i==1: #gamma1
		return backend.asarray(g1)
		
	elif i==2: #gamma2
		return backend.asarray(g2)

	elif i==3: #gamma3
		return backend.asarray(g3)
		
	elif i==4: #gamma4
		return backend.asarray(g4)
		
	elif i==5: #gamma5
		return backend.asarray(g5)

	elif i==6: #-gamma1*gamma4*gamma5 (gamma2*gamma3)
		return backend.asarray(np.matmul(g2,g3))
		
	elif i==7: #-gamma2*gamma4*gamma5 (gamma3*gamma1)
		return backend.asarray(np.matmul(g3,g1))
 
	elif i==8: #-gamma3*gamma4*gamma5 (gamma1*gamma2)
		return backend.asarray(np.matmul(g1,g2))
 
	elif i==9: #gamma1*gamma4
		return backend.asarray(np.matmul(g1,g4))
 
	elif i==10: #gamma2*gamma4
		return backend.asarray(np.matmul(g2,g4))
 
	elif i==11: #gamma3*gamma4
		return backend.asarray(np.matmul(g3,g4))
 
	elif i==12: #gamma1*gamma5
		return backend.asarray(np.matmul(g1,g5))
 
	elif i==13: #gamma2*gamma5
		return backend.asarray(np.matmul(g2,g5))
 
	elif i==14: #gamma3*gamma5
		return backend.asarray(np.matmul(g3,g5))
 
	elif i==15: #gamma4*gamma5
		return backend.asarray(np.matmul(g4,g5))

	elif i==16: #(gamma3*gamma1)
		m1=np.matmul(g3,g1)
		m2=0.5*(g0+g4)
		return backend.asarray(np.matmul(m1,m2))

	elif i==17: #(gamma3*gamma1)
		m1=np.matmul(g3,g1)
		m2=0.5*(g0-g4)
		return backend.asarray(np.matmul(m1,m2))

	else:
		print("wrong gamma index")
		sys.exit(-3)
 
def gamma_index(g):
	value=np.zeros((4),dtype=complex)
	row=np.zeros((4),dtype=int)
	col=np.zeros((4),dtype=int)
	count=0
	for i in range(4):
		for j in range(4):
			if(np.abs(g[i,j]) != 0.0):
				value[count]=g[i,j]
				row[count]=i
				col[count]=j
				count=count+1
	return value, row, col


def tran_indx_to_gamma(indx):
	if type(indx) == list:
		indx = np.asarray(indx)

	indx_shape = list(indx.shape)
	indx = indx.reshape(-1)
	_gamma = np.asarray([gamma(x) for x in indx]).reshape(indx_shape + [4] * 2)

	return _gamma

def PFF_Mom_to_gamma_new(Mom, allow_t:bool = False):
	from ..base.base_functions import levi_civita_tensor
	from opt_einsum import contract
	from itertools import combinations

	gamma_indx_list_matrix = [[[]]]
		

	if allow_t == False:
		lc_tensor = levi_civita_tensor(3)
		Mom_list = [x[::-1] for x in Mom]

	else:
		lc_tensor = levi_civita_tensor(4)
		Mom_list = [([1] + x)[::-1] for x in Mom]

	if Mom_list == [[0, 0, 0]]:
		gamma_indx_list_matrix = np.asarray([[[x, y] for x in range(1, 5) for y in range(1, 5)]])

	else:
		for _Mom in Mom_list:
			k = []
			k = [x_indx for x_indx, x in enumerate(_Mom) if x != 0]

			for l in np.asarray(list(combinations(k, lc_tensor.ndim - 2))):	  
				gamma_indx_list = [[]]
				if lc_tensor.ndim - 2 == 1:
					gamma_indx_matrix = lc_tensor[..., l[0]]
				
				elif lc_tensor.ndim - 2 == 2:
					gamma_indx_matrix = lc_tensor[..., l[0], l[1]]

				gamma_indx = np.argwhere(gamma_indx_matrix != 0) + 1
				for i in list(gamma_indx):
					i = [int(x) for x in i]
					gamma_indx_list += [i]

				gamma_indx_list_matrix += [gamma_indx_list[1:]]

		gamma_indx_list_matrix = np.asarray(gamma_indx_list_matrix[1:]).reshape(-1, len(list(combinations(k, lc_tensor.ndim - 2)))*2, 2)
			
	gamma_indx_list_all = np.asarray([[x for x in [1, 2, 3, 4] if x in gamma_indx_list_matrix[y]] for y in range(len(Mom))])

	return (gamma_indx_list_matrix, tran_indx_to_gamma(gamma_indx_list_matrix), gamma_indx_list_all, tran_indx_to_gamma(gamma_indx_list_all))