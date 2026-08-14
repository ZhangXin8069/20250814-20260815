from opt_einsum import contract
import numpy as np
import random as ra
from typing import Literal
from ..base.backend import set_backend, get_backend

class vector_creator:
    def __init__(
            self,
            ) -> None:
            
        self.backend = get_backend()

    def inner_product(
            self, 
            init_vector, 
            test_vector, 
            dtype:Literal['', 'abs'] = ''
            ):
        
        shape_init = list(init_vector.shape)
        shape_test = list(test_vector.shape)

        if np.prod(shape_init[1:]) != np.prod(shape_test[1:]):
            raise ValueError("the init and test vectors not have the the same Volume shape. ")
        
        V = np.prod(shape_init[1:])
        
        C = contract('NV,nV', init_vector.reshape(-1, V).conj(), test_vector.reshape(-1, V))
        
        if dtype == 'abs':
            return C*C.conj()
        
        else:
            return C
            
    def check(self, eigvecs, dtype:Literal['find', 'print'] = 'find', tol = 1e-10, check_normal: bool = True):
        """
        this function will check the vectors normal and orth.
        """
        if self.backend.isnan(eigvecs).any():
            print('eigen have nan')
            return False
        
        if (self.backend.abs(eigvecs) == 0).any():
            print('eigen has zero')
        
        shape_init = list(eigvecs.shape)
        V = np.prod(shape_init[1:])

        eigvecs = eigvecs.reshape((eigvecs.shape[0], V))

        A = contract('nV,NV->nN', eigvecs, self.backend.conj(eigvecs))
        if check_normal:
            B = True
            
            for i in range(eigvecs.shape[0]):
                if ((A[i, i] - 1) * (A[i, i] - 1).conj()).real >= tol:
                    B = False
                    print(f"eigen don't normal, position is {(i)}, vector norm is {A[i,i]}")

                else:    
                    A[i,i] = A[i,i] - 1

            if B:
                if dtype == 'print':
                    print(f"normal in the tol: {tol}")

            else:
                print(f"eigen don't normal")
                return False
                    
        else:
            A = A - self.backend.identity(eigvecs.shape[0])
        

        if dtype == 'find':
            if (A >= tol).any():
                print("don't orth")
                return False
            
            else:
                return True
            
        elif dtype == 'print':
            position = self.backend.argwhere(A >= tol)
            if position.reshape(-1).shape[0] != 0:
                print("don't orth")
                print(position)
                print(A[A>=tol])
                return False
            
            else:
                print(f'orth in the tol: {tol}')
                return True
            
    def normal(self, vectors):
        shape_init = list(vectors.shape)
        V = np.prod(shape_init[:])//shape_init[0]
        
        if V == 1:
            V = np.prod(shape_init[:])

        vectors = vectors.reshape(-1, V)
        N = contract('nv,nv->n', vectors, self.backend.conj(vectors)).reshape(-1, 1)
        vectors = vectors / self.backend.sqrt(N)
        
        return vectors.reshape(shape_init)

    def orthnormal(
            self,
            vectors_init, 
            vector
            ):
        
        shape_init = list(vectors_init.shape)
        shape_test = shape_init.copy()
        shape_test[0] = 1
        V = np.prod(shape_init[1:])
        eigvecs_test = vector.reshape(-1, V)
        
        eigvecs_test = self.normal(eigvecs_test)
        eigvecs_coeff = contract('NV,nV->nN', (vectors_init.conj()).reshape(-1, V), eigvecs_test.reshape(-1, V))
        eigvecs_orth = eigvecs_test.reshape(-1, V) - contract('nN,Nv->nv', eigvecs_coeff.reshape(-1, shape_init[0]), vectors_init.reshape(-1, V))
        eigvecs_orthnormal = self.normal(eigvecs_orth)

        vectors_init = self.backend.append(vectors_init, eigvecs_orthnormal.reshape(tuple(shape_test)), axis=0)
        
        return vectors_init

    def creat_noise(
            self,
            vectors_init, 
            N, 
            dtype:Literal['complex', 'float'] = 'complex'
            ):
        
        vectors_init = self.normal(vectors_init)

        shape_init = list(vectors_init.shape)
        shape_test = shape_init.copy()
        shape_test[0] = 1
        V = np.prod(shape_init[1:])

        for _ in range(N):
            
            if dtype == 'complex':
                eigvecs_test = (2 * self.backend.random.random((1, V)) - 1) + 1j * (2 * self.backend.random.random((1, V)) - 1)

            elif dtype == 'float':
                eigvecs_test = 2 * self.backend.random.random((1, V)) - 1

            eigvecs_test = eigvecs_test / self.backend.sqrt(contract('NV,NV->N', eigvecs_test.conj(), eigvecs_test)).reshape(-1, 1)
            eigvecs_coeff = contract('NV,nV->nN', (vectors_init.conj()).reshape(-1, V), eigvecs_test.reshape(-1, V))
            # eigvecs_orth = eigvecs_test.reshape(-1, V) - self.backend.sum(eigvecs_coeff.reshape(-1, 1) * vectors_init.reshape(-1, V), axis=0)
            eigvecs_orth = eigvecs_test.reshape(-1, V) - contract('nN,Nv->nv', eigvecs_coeff.reshape(-1, shape_init[0]), vectors_init.reshape(-1, V))
            eigvecs_orthnormal = eigvecs_orth / self.backend.sqrt(contract('NV,NV->N', eigvecs_orth.conj(), eigvecs_orth)).reshape(-1, 1)

            vectors_init = self.backend.append(vectors_init, eigvecs_orthnormal.reshape(tuple(shape_test)), axis=0)

            shape_init[0] += 1
        
        return vectors_init
    
    # this version is sum the eigen
    def compress_matrix_V1(
            self,
            eigenvectors, 
            N_eigen:list=[0], 
            N_sum:list=[0], 
            Ctype:str='I'
            ):
        eigen_shape = eigenvectors.shape
        eigen_BI = self.backend.zeros((sum(N_sum), eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1]), "<c16")
        if Ctype == 'I':
            if len(N_eigen) != 1 or len(N_sum) != 1:
                raise print('interlace Ctype must use 1 dimensions of N_eigen and N_sum')
            
            # eigen_BI[:N_sum[0]] = eigenvectors[:N_eigen[0]]
            eigen_BI[:] = self.backend.sum(
                eigenvectors[:].reshape(N_eigen[-1]//N_sum[-1], N_sum[-1], eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1]), axis=0
                ) / self.backend.sqrt(N_eigen[-1]//N_sum[-1])
            
        elif Ctype == 'B':
            for i in range(len(N_eigen)):
                eigen_BI[sum(N_sum[:i]) : sum(N_sum[:(i + 1)])] = self.backend.sum(
                    eigenvectors[sum(N_eigen[:i]) : sum(N_eigen[:(i+1)])].reshape(N_sum[i], N_eigen[i]//N_sum[i], eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1]), axis=1
                    ) / self.backend.sqrt(N_eigen[i]//N_sum[i])  
                
        elif Ctype == 'BI':
            # this Ctype only use block for first dimension, use interlace for second dimension
            
            for i in range(len(N_eigen) - 1):
                eigen_BI[sum(N_sum[:i]) : sum(N_sum[:(i + 1)])] = self.backend.sum(
                    eigenvectors[sum(N_eigen[:i]) : sum(N_eigen[:(i+1)])].reshape(N_sum[i], N_eigen[i]//N_sum[i], eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1]), axis=1
                    ) / self.backend.sqrt(N_eigen[i]//N_sum[i])  
            
            eigen_BI[sum(N_sum[:(i+1)]):] = self.backend.sum(
                eigenvectors[sum(N_eigen[:(i+1)]):].reshape(N_eigen[-1]//N_sum[-1], N_sum[-1], eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1]), axis=0
                ) / self.backend.sqrt(N_eigen[-1]//N_sum[-1])
            
        else:
            raise print('plase use Ctype B mean block, I mean interlace or BI mean block for first dimension interlace for second dimension')
        
        return eigen_BI

    # this version is to extract the eigen 
    def compress_matrix_V2(
            self,
            eigenvectors, 
            N_eigen:list=[0], 
            N_sum:list=[0], 
            N_extract:list=[0], 
            Ctype:str='I'
            ):
        eigen_shape = eigenvectors.shape
        eigen_BI = self.backend.zeros((sum(N_sum), eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1]), "<c16")
        if Ctype == 'I':
            
            if len(N_eigen) != 1 or len(N_sum) != 1 or len(N_extract)!=1:
                raise print('interlace Ctype must use 1 dimensions of N_eigen and N_sum')
            
            if N_extract[-1] < 2:
                raise print('extract eigen must > 2 of a part')
            
            for i in range(N_sum[-1]//N_extract[-1]):
                random = np.int32(np.random.random(N_extract[-1]) * (N_extract[-1] * N_eigen[-1]//N_sum[-1]))
                
                while np.unique(random).shape[0] != N_extract[-1]:
                    
                    random = np.int32(np.random.random(N_extract[-1]) * (N_extract[-1] * N_eigen[-1]//N_sum[-1]))
                
                eigen_BI[i * N_extract[-1]:(i+1) * N_extract[-1]] = (
                    eigenvectors[:].reshape(N_extract[-1] * N_eigen[-1]//N_sum[-1], N_sum[-1]//N_extract[-1], eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1])
                    )[random, i].reshape(N_extract[-1], eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1])
            
            
        elif Ctype == 'B':
            
            for i in range(len(N_eigen)):
                
                if N_extract[i] < 1:
                    raise print('extract eigen must > 1 of a part')
                
                for j in range(N_sum[i]//N_extract[i] - 1):
                    
                    random = np.int32(np.random.random(N_extract[i]) * (N_extract[i] * N_eigen[i]//N_sum[i]))
                    
                    while np.unique(random).shape[0] != N_extract[i]:
                        
                        random = np.int32(np.random.random(N_extract[i]) * (N_extract[i] * N_eigen[i]//N_sum[i]))

                    eigen_BI[(sum(N_sum[:i]) + j * N_extract[i])  : (sum(N_sum[:i]) + (j + 1) * N_extract[i])] = (
                        eigenvectors[sum(N_eigen[:i]) : sum(N_eigen[:(i+1)])][j * (N_extract[i] * N_eigen[i]//N_sum[i]) : (j + 1) * (N_extract[i] * N_eigen[i]//N_sum[i])]
                        )[random].reshape(N_extract[i], eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1])
                    
                        # eigenvectors[sum(N_eigen[:i]) : sum(N_eigen[:(i+1)])].reshape(N_sum[i]//N_extract[i], N_extract[i] * N_eigen[i]//N_sum[i], eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1])
                        # )[j, random].reshape(N_extract[i], eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1])
                
                j = N_sum[i]//N_extract[i] - 1
                
                random = np.int32(np.random.random(N_extract[i]) * (N_eigen[i] - j * (N_extract[i] * N_eigen[i]//N_sum[i])))
                
                while np.unique(random).shape[0] != N_extract[i]:
                    
                    random = np.int32(np.random.random(N_extract[i]) * (N_eigen[i] - j * (N_extract[i] * N_eigen[i]//N_sum[i])))

                eigen_BI[(sum(N_sum[:i]) + j * N_extract[i])  : (sum(N_sum[:i]) + (j + 1) * N_extract[i])] = (
                    eigenvectors[sum(N_eigen[:i]) : sum(N_eigen[:(i+1)])][j * (N_extract[i] * N_eigen[i]//N_sum[i]): ]
                    )[random].reshape(N_extract[i], eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1])
            
                
            # for i in range(len(N_eigen)):
                
        elif Ctype == 'BI':
            # this Ctype only use block for first dimension, use interlace for second dimension
            
            for i in range(len(N_eigen)):
                eigen_BI[sum(N_sum[:i]) : sum(N_sum[:(i + 1)])] = self.compress_matrix_V2(eigenvectors = eigenvectors[sum(N_eigen[:i]) : sum(N_eigen[:(i+1)])], N_eigen = [N_eigen[i]], N_sum = [N_sum[i]], N_extract = [N_extract[i]], Ctype = 'I')
        else:
            raise print('plase use Ctype B mean block, I mean interlace or BI mean block for first dimension interlace for second dimension')
        
        return eigen_BI

    # this version is to creat the orth random eigen, use vectors
    def compress_matrix_V3(
            self,
            eigenvectors, 
            N_eigen:list=[0], 
            N_sum:list=[0], 
            N_extract:list=[1], 
            Ctype:str='I', 
            adjcent:bool = False
            ):
        
        eigen_shape = eigenvectors.shape
        eigen_BI = self.backend.zeros((sum(N_sum), eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1]), "<c16")
        if Ctype == 'I':
            if len(N_eigen) != 1 or len(N_sum) != 1 or len(N_extract)!=1:
                raise print('interlace Ctype must use 1 dimensions of N_eigen and N_sum')
            
            for i in range(N_sum[-1]//N_extract[-1]):
                random = self.backend.random.random((1, N_extract[-1] * N_eigen[-1]//N_sum[-1])) + 1j * self.backend.random.random((1, N_extract[-1] * N_eigen[-1]//N_sum[-1]))
                random = random / self.backend.sqrt(contract('Na,Na->N', random.conj(), random).reshape(-1, 1))
                random = self.creat_noise(vectors_init = random, N = N_extract[-1] - 1)

                if adjcent == True:
                    eigen_BI[i*N_extract[-1]:(i+1)*N_extract[-1]] = contract('Nn,nxyzc->Nxyzc', random,  (
                        ((eigenvectors[:].reshape(N_eigen[-1]//N_sum[-1], N_sum[-1], eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1]))[:, i * N_extract[-1]: (i + 1) * N_extract[-1]]).reshape(-1, eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1])
                        )) #.reshape(N_extract[-1], eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1])
                
                else:
                    eigen_BI[i*N_extract[-1]:(i+1)*N_extract[-1]] = contract('Nn,nxyzc->Nxyzc', random,  (
                        eigenvectors[:].reshape(N_extract[-1] * N_eigen[-1]//N_sum[-1], N_sum[-1]//N_extract[-1], eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1])
                        )[:,i]) #.reshape(N_extract[-1], eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1])
            
        elif Ctype == 'B':
            for i in range(len(N_eigen)):
                if N_extract[i] < 1:
                    raise print('extract eigen must > 1 of a part')
                
                for j in range(N_sum[i]//N_extract[i] - 1):
                    random = self.backend.random.random((1, N_extract[i] * N_eigen[i]//N_sum[i])) + 1j * self.backend.random.random((1, N_extract[i] * N_eigen[i]//N_sum[i]))
                    random = random / self.backend.sqrt(contract('Na,Na->N', random.conj(), random).reshape(-1, 1))
                    random = self.creat_noise(vectors_init = random, N = N_extract[i] - 1)

                        
                    eigen_BI[(sum(N_sum[:i]) + j * N_extract[i])  : (sum(N_sum[:i]) + (j + 1) * N_extract[i])] = contract('Nn,nxyzc->Nxyzc', random,  (
                        eigenvectors[sum(N_eigen[:i]) : sum(N_eigen[:(i+1)])][j * (N_extract[i] * N_eigen[i]//N_sum[i]) : (j + 1) * (N_extract[i] * N_eigen[i]//N_sum[i])]
                        ))
                    
                j = N_sum[i]//N_extract[i] - 1
                
                random = self.backend.random.random((1, (N_eigen[i] - j * (N_extract[i] * N_eigen[i]//N_sum[i])))) + 1j * self.backend.random.random((1, (N_eigen[i] - j * (N_extract[i] * N_eigen[i]//N_sum[i]))))
                random = random / self.backend.sqrt(contract('Na,Na->N', random.conj(), random).reshape(-1, 1))
                random = self.creat_noise(vectors_init = random, N = N_extract[i] - 1)
                    
                eigen_BI[(sum(N_sum[:i]) + j * N_extract[i])  : (sum(N_sum[:i]) + (j + 1) * N_extract[i])] = contract('Nn,nxyzc->Nxyzc', random,  (
                    eigenvectors[sum(N_eigen[:i]) : sum(N_eigen[:(i+1)])][j * (N_extract[i] * N_eigen[i]//N_sum[i]): ]
                    ))

        elif Ctype == 'BI':
            # this Ctype only use block for first dimension, use interlace for second dimension
            
            for i in range(len(N_eigen)):
                eigen_BI[sum(N_sum[:i]) : sum(N_sum[:(i + 1)])] = self.compress_matrix_V3(eigenvectors = eigenvectors[sum(N_eigen[:i]) : sum(N_eigen[:(i+1)])], N_eigen = [N_eigen[i]], N_sum = [N_sum[i]], N_extract = [N_extract[i]], Ctype = 'I', adjcent = adjcent)
        
        else:
            raise print('plase use Ctype B mean block, I mean interlace or BI mean block for first dimension interlace for second dimension')
        
        if self.check(eigen_BI, dtype = 'find') == 'don\'t orth':
            raise print('eigen don\'t orth in compress_matrix_V3')
        
        return eigen_BI
    
    def compress_matrix_V4(
            self,
            eigenvectors, 
            N_eigen:list=[0], 
            N_sum:list=[0], 
            N_extract:list=[1], 
            Ctype:Literal['I', 'B', 'BI'] = 'I', 
            adjcent:bool = False,
            random_type:Literal["orthnormal", 'Z_N'] = "orthnormal"
            ):
        
        eigen_shape = eigenvectors.shape
        eigen_BI = self.backend.zeros((sum(N_sum), eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1]), "<c16")
        def creat_random_vectors(V, k):
            if random_type == 'orthnormal':
                random = self.backend.random.random((1, V)) + 1j * self.backend.random.random((1, V))
                random = random / self.backend.sqrt(contract('Na,Na->N', random.conj(), random).reshape(-1, 1))
                random = self.creat_noise(vectors_init = random, N = V - 1)[ra.sample([x for x in range(V)], k = k)]

            elif "Z_" in random_type:
                random = self.backend.zeros((k, V), dtype = complex)

                N = int(random_type[2:])
                random_value_list = [1, -1, 1j, -1j, float(self.backend.sqrt(2)) + 1j * float(self.backend.sqrt(2)), float(self.backend.sqrt(2)) - 1j * float(self.backend.sqrt(2)), -float(self.backend.sqrt(2)) + 1j * float(self.backend.sqrt(2)), -float(self.backend.sqrt(2)) - 1j * float(self.backend.sqrt(2))][:N]
                
                for i in range(k):
                    for j in range(V):
                        random[i, j] = ra.sample(random_value_list, k = 1)[0]

            return random
            
        if Ctype == 'I':
            if len(N_eigen) != 1 or len(N_sum) != 1 or len(N_extract)!=1:
                raise print('interlace Ctype must use 1 dimensions of N_eigen and N_sum')
            
            for i in range(N_sum[-1]//N_extract[-1]):
                random = creat_random_vectors(V = N_extract[-1] * N_eigen[-1]//N_sum[-1], k = N_extract[-1])

                if adjcent == True:
                    eigen_BI[i*N_extract[-1]:(i+1)*N_extract[-1]] = contract('Nn,nxyzc->Nxyzc', random,  (
                        ((eigenvectors[:].reshape(N_eigen[-1]//N_sum[-1], N_sum[-1], eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1]))[:, i * N_extract[-1]: (i + 1) * N_extract[-1]]).reshape(-1, eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1])
                        )) #.reshape(N_extract[-1], eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1])
                
                else:
                    eigen_BI[i*N_extract[-1]:(i+1)*N_extract[-1]] = contract('Nn,nxyzc->Nxyzc', random,  (
                        eigenvectors[:].reshape(N_extract[-1] * N_eigen[-1]//N_sum[-1], N_sum[-1]//N_extract[-1], eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1])
                        )[:,i]) #.reshape(N_extract[-1], eigen_shape[-4], eigen_shape[-3], eigen_shape[-2], eigen_shape[-1])
            
        elif Ctype == 'B':
            for i in range(len(N_eigen)):
                if N_extract[i] < 1:
                    raise print('extract eigen must > 1 of a part')
                
                for j in range(N_sum[i]//N_extract[i] - 1):
                    random = creat_random_vectors(V = N_extract[i] * N_eigen[i]//N_sum[i], k = N_extract[i])

                    eigen_BI[(sum(N_sum[:i]) + j * N_extract[i])  : (sum(N_sum[:i]) + (j + 1) * N_extract[i])] = contract('Nn,nxyzc->Nxyzc', random,  (
                        eigenvectors[sum(N_eigen[:i]) : sum(N_eigen[:(i+1)])][j * (N_extract[i] * N_eigen[i]//N_sum[i]) : (j + 1) * (N_extract[i] * N_eigen[i]//N_sum[i])]
                        ))
                    
                j = N_sum[i]//N_extract[i] - 1

                random = creat_random_vectors(V = (N_eigen[i] - j * (N_extract[i] * N_eigen[i]//N_sum[i])), k = N_extract[i])
                eigen_BI[(sum(N_sum[:i]) + j * N_extract[i])  : (sum(N_sum[:i]) + (j + 1) * N_extract[i])] = contract('Nn,nxyzc->Nxyzc', random,  (
                    eigenvectors[sum(N_eigen[:i]) : sum(N_eigen[:(i+1)])][j * (N_extract[i] * N_eigen[i]//N_sum[i]): ]
                    ))

        elif Ctype == 'BI':
            # this Ctype only use block for first dimension, use interlace for second dimension
            
            for i in range(len(N_eigen)):
                eigen_BI[sum(N_sum[:i]) : sum(N_sum[:(i + 1)])] = self.compress_matrix_V4(
                    eigenvectors = eigenvectors[sum(N_eigen[:i]) : sum(N_eigen[:(i+1)])], 
                    N_eigen = [N_eigen[i]], 
                    N_sum = [N_sum[i]], 
                    N_extract = [N_extract[i]], 
                    Ctype = 'I', 
                    adjcent = adjcent,
                    random_type = random_type
                    )
        
        else:
            raise print('plase use Ctype B mean block, I mean interlace or BI mean block for first dimension interlace for second dimension')
        
        if random_type == 'orthnormal':
            if self.check(eigen_BI, dtype = 'find') == 'don\'t orth':
                raise print('eigen don\'t orth in compress_matrix_V4')
        
        return eigen_BI