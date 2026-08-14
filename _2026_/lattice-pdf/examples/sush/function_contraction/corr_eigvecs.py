from opt_einsum import contract
from .backend import get_backend, set_backend
import numpy as np
from typing import List, Literal
from itertools import combinations
from .constant import Nd

class corr_eigvecs:
    
    def __init__(
        self,
        Nx,
        backend:str = 'numpy',
        Nc:int = 3
        ) -> None:
        
        self.backend_name = backend

        set_backend(backend)
        self.backend = get_backend()
        self.Nx = Nx
        self.Nc = Nc

    def check(self, eigvecs, dtype: str = 'find', tol = 1e-10, check_normal: bool = True):

        if self.backend.isnan(eigvecs).any():
            raise print('eigen have nan')
        
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
                    print(f"eigen don't normal, position is {(i)}, vector norm is {A[i,i]}")
                    B = False

                else:    
                    A[i,i] = A[i,i] - 1

            if B:
                if dtype == 'print':
                    print(f"normal in the tol: {tol}")
                    
        else:
            A = A - self.backend.identity(eigvecs.shape[0])
        

        if dtype == 'find':
            if (A >= tol).any():
                return "don't orth"
            
            else:
                return 'orth'
            
        elif dtype == 'print':
            position = self.backend.argwhere(A >= tol)
            if position.reshape(-1).shape[0] != 0:
                print("don't orth")
                print(position)
                print(A[A>=tol])
                return "don't orth"
            
            else:
                print(f'orth in the tol: {tol}')
                return 'orth'

    def normal(self, vectors):
        shape_init = list(vectors.shape)
        V = np.prod(shape_init[-4:])
        vectors = vectors.reshape(-1, V)
        N = contract('nv,nv->n', vectors, self.backend.conj(vectors)).reshape(-1, 1)
        vectors = vectors / np.sqrt(N)
        
        return vectors.reshape(shape_init)
    
    def src_sink_MPI_tran(self, src_sink, mpi_size, trtype:Literal['forward', 'backward'] = 'forward'):
        
        if 'numpy' in str(type(src_sink)):
            set_backend('numpy')
        
        elif 'cupy' in str(type(src_sink)):
            set_backend('cupy')

        backend = get_backend()
        if trtype == 'forward':
            src_sink_shape_tran = [mpi_size] + list(src_sink.shape)

            if (src_sink_shape_tran[1] / mpi_size) %1 != 0:
                raise print(f'Nt: {src_sink_shape_tran[1]}, mpi_size: {mpi_size}, Nt/mpi_size = {(src_sink_shape_tran[1] / mpi_size)}')
            
            src_sink_shape_tran[1] = int(src_sink_shape_tran[1] / mpi_size)
            src_sink = backend.ascontiguousarray(src_sink.reshape(src_sink_shape_tran).swapaxes(0, 1))
            
            set_backend(self.backend_name)

            return src_sink
        
        elif trtype == 'backward':
            src_sink_shape_tran = list(src_sink.shape)
            
            src_sink_shape_tran[1] = int(src_sink_shape_tran[0] * mpi_size)
            src_sink = backend.ascontiguousarray(src_sink.swapaxes(0, 1).reshape(src_sink_shape_tran[1:]))

            set_backend(self.backend_name)

            return src_sink
    
    def perm_comb(self, N:float, M:int = 1, dtype:Literal['perm', 'comb'] = 'perm', renormal:bool  =False):
        import numpy


        if (renormal == False and M >= 0) or (renormal == True and M >= 1):
            if dtype == 'perm':
                return float(numpy.prod([N - x for x in range(len([N]*M))]))
            
            elif dtype == 'comb':
                return float(numpy.prod([N - x for x in range(len([N]*M))]) / numpy.prod([x for x in range(1, M + 1, 1)]))

        elif M <= 0 and renormal == True:
            return self.perm_comb(N = N, M = 1, dtype = dtype, renormal = False)
        
        else:
            raise print('mistake')
        
    def create_omega_accelerate(
            self,
            exact:int = 0, 
            N_eigen:list = [0], 
            N_sum:list = [0], 
            N_extract:list = [0], 
            noise:int = 0, 
            conserved:str = False, 
            normal:bool = False,
            fixed_first_pos:list = [-1],
            dim:int = 2
            ):
        """
        创建任意维度的 Ω 张量（2D, 3D, 4D）
        
        参数:
            exact: 精确本征矢的数量
            N_eigen: block中各本征矢组压缩前的数量列表
            N_sum: block中各本征矢组压缩后的数量列表
            N_extract: block中各本征矢组压缩后的每个block中提取的数量
            noise: noise vector 数量
            conserved: 是否守恒模式（守恒时维度固定为2）
            dim: 输出张量的维度
            
        返回:
            omega: 构建的Ω张量，权重矩阵
        """
        if conserved:
            dim = 2
        
        backend = get_backend()
        
        Nv = self.Nx**3 * self.Nc
        Nbin = 1
        # tranlate the shape of extract 
        if N_eigen != [0]:
            tran_N_eigen = []
            tran_N_sum = []

            for i in range(len(N_sum)):
                for j in range(int(N_sum[i] / N_extract[i])):
                    if (N_sum[i] / N_extract[i])%1 != 0:
                        raise print(f"extract: {N_extract[i]} can't to be divisible by sum vector :{N_sum[i]}, please check the block is right")
                    
                    tran_N_sum += [N_extract[i]]

                    if (N_eigen[i]/(N_sum[i] / N_extract[i]))%1 != 0:
                        raise print(f"sum vectors / extract vectors:{N_sum[i] / N_extract[i]} can't to be divisible by block vectors:{N_eigen[i]}, please check the block is right")
                    
                    tran_N_eigen += [int(N_eigen[i]/(N_sum[i] / N_extract[i]))]

        else:
            tran_N_sum = N_sum
            tran_N_eigen = N_eigen

        # str_indx = ''.join([['a', 'b', 'c', 'd', 'e', 'f'][x] for x in range(dim)])
        # lists ev_space or ev_sum is to create the space of every block vectors or the used block vectors
        # we call the corresponding elements ot the ev_ are part, then it will remove the zero elements
        ev_space = [x for x in ([exact] + tran_N_eigen + [Nv / Nbin - sum([exact] + tran_N_eigen)]) if x != 0]
        ev_sum = [x for x in ([exact] + tran_N_sum + [noise]) if x != 0]
        
        # this is the nums of parts we need to care about
        len_of_space = len(ev_sum)
        
        # the whole vectors we used include the exact, block and noise 
        Nev = sum(ev_sum)    

        # coordinate_position_slice is the position of every part's vectors. the order is exact, block, noise
        coordinate_position_slice = []
        coordinate_position_range = []
        for i in range(len_of_space):
            coordinate_position_slice += [slice(sum(ev_sum[:i]), sum(ev_sum[:(i+1)]))]
            coordinate_position_range += [range(sum(ev_sum[:i]), sum(ev_sum[:(i+1)]))]
            
            if fixed_first_pos[0] in range(sum(ev_sum[:i]), sum(ev_sum[:(i+1)])):
                fixed_first_pos_indx = i

        # weights is a 2 dim arr of the weight of every parts. 
        # the first dim 'i' is the len_of_space num of parts, the second dim 'j' is the dim with (space - j) / (sum - j)
        weights = np.empty((len_of_space, dim), dtype = float)
        for i in range(len_of_space):
            for j in range(dim):
                weights[i, j] = (ev_space[i] - j) / (ev_sum[i] - j)
                
        # print(weights)
        # weights = weights * Nbin**(1/dim)

        # all_position is what the parts we used in the whole dims 
        if dim <= 3:
            all_position = np.unique(np.asarray(list(combinations(range(dim * len_of_space), dim)))%len_of_space, axis = 0)

        elif dim == 4:
            all_position = np.asarray([[x, y, z, l] for x in range(len_of_space) for y in range(len_of_space) for z in range(len_of_space) for l in range(len_of_space)])
        
        # initialize the weight array
        if fixed_first_pos[0] >= 0:
            omega = backend.empty(([len(fixed_first_pos)] + [Nev] * (dim - 1)), dtype=float)
        
        else:
            omega = backend.empty([Nev]*dim, dtype=float)

        if fixed_first_pos[0] >= 0:
            all_position[:, 0] = fixed_first_pos_indx
            all_position = np.unique(all_position, axis = 0)

        # the main calculation of the func we will loop all situations of all_position
        
        for indx_of_position in all_position:
            # sum_of_position will record the num of the parts we used in all dims
            sum_of_position = [0] * len_of_space
            for j in indx_of_position:
                sum_of_position[j] += 1
            # print(sum_of_position)
            # convert each part we used to the corresponding position
            _position = []
            len_position = []

            for dim_indx in range(dim):
                _position += [coordinate_position_slice[indx_of_position[dim_indx]]]
                len_position += [coordinate_position_range[indx_of_position[dim_indx]]]
            # print(_position)
            
            _position_of_indx = _position.copy()
            # initialize the weight of the the situation and put the weight to the weight arr
            _weights = 1.0
            for space_indx in range(len_of_space):
                _weights *= np.prod(weights[space_indx, :sum_of_position[space_indx]])
            
            if fixed_first_pos[0] >= 0:
                _position[0] = slice(len(fixed_first_pos))
                _position_of_indx[0] = slice(fixed_first_pos[0], fixed_first_pos[-1] + 1)
                len_position[0] = range(len(fixed_first_pos))
            _W_bool = backend.empty(tuple([len(len_position[x]) for x in range(dim)]), dtype = bool)
            omega[tuple(_position)] = _weights
            
            # create the index grid
            indices = backend.ogrid[_position_of_indx]
            # the whole cycle is to make the equal of N parts in the dim 
            for extra_dim in range(1, len_of_space):
                # sum_of_position[extra_dim] means the one part is used in N dim,
                # if N >=2 this cycle will stare
                for i in range(sum_of_position[extra_dim] - 1):
                    # this will find what the part is used in the N dim 
                    # then find where the part
                    # then create the combinations of the position
                    for j in list(combinations((np.argwhere(indx_of_position == extra_dim).reshape(-1)).tolist(), 2 + i)):
                        # initialize the arr of the each part we used to the corresponding position
                        _W = omega[tuple(_position)]
                        
                        # initialize the grid use to find what position need to put the unique weight
                        # _W_bool[:] = 1
                        _W_bool.fill(1)
                        # initialize the diag element weight
                        _weights_diag = _weights
                        
                        # find what position need to be changed the weight of diag
                        for k in range(len(j) - 1):
                            _W_bool &= indices[j[k]] == indices[j[k + 1]]
                            _weights_diag = _weights_diag / weights[extra_dim, sum_of_position[extra_dim] - k - 1]
                        
                        # assignment
                        _W[_W_bool] = _weights_diag

                        # change the diagonal element weight between any two parts
                        if len(j) == 2 and dim == 4:
                            m, n = [x for x in range(dim) if x not in j]
                            
                            if indx_of_position[m] == indx_of_position[n] and indx_of_position[m] != extra_dim:
                                _W_bool &= indices[m] == indices[n]
                                _weights_diag = _weights_diag / weights[indx_of_position[m], 1]
                                _W[_W_bool] = _weights_diag
                        
                        # change the diagonal element weights within one part
                        if sum_of_position[extra_dim] - (2 + i) == 2:
                            m, n = [x for x in range(dim) if x not in j]
                            _W_bool &= indices[m] == indices[n]

                            _weights_diag = _weights_diag / weights[extra_dim, sum_of_position[extra_dim] - k - 2]
                            _W[_W_bool] = _weights_diag
                            
        if normal:
            if dim == 2:
                omega_shape = omega.shape
            
                for i in range(0, omega_shape[0], 1):
                    omega[i] = omega[i] * omega_shape[0] / backend.sum(omega[i])
            
                omega.T[backend.tril_indices_from(omega, -1)] = omega[backend.tril_indices_from(omega, -1)]
                
        return omega.astype(complex)
    
    def phase_exp_2pt(self, Mom: list = [0, 0, 0]):
        backend = get_backend()
        
        if all(x == 0 for x in Mom):
            return backend.ones((self.Nx, self.Nx, self.Nx, self.Nc), dtype=complex)
        
        # 更高效的方法：使用单独的一维数组和广播
        mom_array = backend.asarray(Mom, dtype=complex)
        factor = -2j * backend.pi / self.Nx
        
        # 创建一维坐标数组
        z = backend.arange(self.Nx, dtype=complex)
        y = backend.arange(self.Nx, dtype=complex)
        x = backend.arange(self.Nx, dtype=complex)
        
        # 分别计算三个方向的贡献
        # 使用reshape和广播避免meshgrid的内存开销
        z_phase = backend.exp(factor * mom_array[0] * z[:, None, None])
        y_phase = backend.exp(factor * mom_array[1] * y[None, :, None])
        x_phase = backend.exp(factor * mom_array[2] * x[None, None, :])
        
        # 组合相位因子（乘法对应相位相加）
        phase_3d = z_phase * y_phase * x_phase
        
        # 扩展到颜色维度

        phase_exp = backend.stack([phase_3d, phase_3d, phase_3d], axis=-1)
        

        return phase_exp

    def phase_exp_3pt(self, Mom=[0, 0, 0]):
        backend = get_backend()

        if all(x == 0 for x in Mom):
            phase_exp = backend.ones(self.Nx**3, dtype=complex)
            return phase_exp.reshape(self.Nx, self.Nx, self.Nx)

        # 生成坐标数组 [0, 1, ..., Nx-1]
        coords = backend.arange(self.Nx, dtype=complex)

        # indexing='ij' 保证展平后索引顺序为 z*Nx^2 + y*Nx + x
        zz, yy, xx = backend.meshgrid(coords, coords, coords, indexing='ij')

        # 向量化点积: Mom[0]*z + Mom[1]*y + Mom[2]*x，逐元素
        dot = Mom[0] * zz + Mom[1] * yy + Mom[2] * xx

        # 逐元素计算相位因子
        phase_exp = backend.exp(-2j * backend.pi * dot / self.Nx)
        
        return phase_exp
    def VdV_sink_t_link_new(
        self,
        eigvecs,
        link_dir,
        link_max,
        phase_exp,
        gauge_link,
        t: int = 0,
        eigvecs_min = None,
        conserved = False,
    ):
        """计算 sink 端带规范链接的 V† D V 关联函数。

        该函数在 distillation 框架下计算如下 Wick 收缩：

            V_{mn}(p, Δx) = Σ_x e^{-ip·x} φ_m†(x) U(x, x+Δx) φ_n(x+Δx)

        其中 φ 为本征矢量( eigenvectors)，U(x, x+Δx) 为规范链接构成的平行输运路径。

        本函数是 VdV_sink_t_link_old 的优化版本，主要优化点：
        - 将动量循环合并为单次 einsum 调用，避免逐动量迭代
        - 预计算 eigvecs_conj_T 等重复使用的张量

        根据输入参数，分为三种计算情形：

        【情形 1】无规范链接
            条件：gauge_link 为 bool 类型，或 link_dir == '0'
            计算：V_{mn}(p) = Σ_x e^{-ip·x} φ_m†(x) φ_n(x)
            输出形状：(N_mom, 1, Nev, Nev)

        【情形 2】守恒流 / 时间方向
            条件：conserved == True 或 link_dir == 'T'
            需要提供 eigvecs_min（另一组本征矢量）
            输出形状：(2, Nev, Nev)
            下标 0：eigvecs_min† @ U_t @ eigvecs
            下标 1：eigvecs† @ U_t† @ eigvecs_min

        【情形 3】空间方向
            条件：link_dir ∈ {'X', 'Y', 'Z', 'all'}，且非守恒流
            沿空间方向构建规范路径，计算动量为 p 的关联函数
            输出形状：(N_mom, 2*link_max+1, Nev, Nev)
            link_max 控制路径的最大位移长度

        Parameters
        ----------
        self : object
            用于兼容类方法签名的占位参数（函数体内未使用）。
        eigvecs : ndarray, shape (Nev, Nx, Ny, Nz, 3)
            本征矢量（distillation eigenvectors）。最后一维为颜色指标。
        link_dir : str
            链接方向。可选值：'0'（无链接）、'T'（时间）、'X'、'Y'、'Z'（空间单方向）、
            'all'（所有空间方向求和）。
        link_max : int
            空间链接的最大位移长度（仅情形 3 使用）。
        phase_exp : ndarray
            动量相因子 e^{-ip·x}，形状任意，会被展平为 (N_mom, Nx*Ny*Nz*3)。
        gauge_link : ndarray or bool
            规范场链接。形状 (Nd, Nt, Nx, Ny, Nz, 3, 3)。
            若为 bool 类型（通常为 False），则跳过规范链接计算（情形 1）。
        t : int, optional
            要读取的时间片索引，默认为 0。
        eigvecs_min : ndarray or None, optional
            第二组本征矢量，仅在情形 2（守恒流/时间方向）中使用。
            形状与 eigvecs 相同。
        conserved : bool, optional
            是否为守恒流计算，默认为 False。

        Returns
        -------
        VDV : ndarray, dtype=np.complex128
            关联函数矩阵。形状取决于计算情形：
            - 情形 1: (N_mom, 1, Nev, Nev)
            - 情形 2: (2, Nev, Nev)
            - 情形 3: (N_mom, 2*link_max+1, Nev, Nev)
        """
        backend = get_backend()
        
        # ---------------- Preprocessing ----------------
        eigvecs = backend.asarray(eigvecs)                # (Nev, Nx, Ny, Nz, 3)
        phase_exp = backend.asarray(phase_exp)            # arbitrary shape, will be flattened
        Nev = eigvecs.shape[0]
        Nx = eigvecs.shape[1]                        # assuming isotropic spatial lattice Nx = Ny = Nz
        V_full = Nx * Nx * Nx * 3                    # total dimension: spatial sites × color

        # Precompute reused tensors to avoid repeated reshaping inside loops
        eigvecs_flat = eigvecs.reshape(Nev, V_full)  # (Nev, V)
        eigvecs_conj_T = eigvecs_flat.conj().T       # (V, Nev) — conjugate transpose, used in every contraction
        phase_exp = phase_exp.reshape(-1, V_full)    # (N_mom, V)
        N_mom = phase_exp.shape[0]

        # ---------------- Case 1: No gauge link ----------------
        # Triggered when gauge_link is a boolean (sentinel for "no link")
        # or when link_dir explicitly set to '0'.
        # Computes: V_{mn}(p) = sum_x e^{-ipx} phi_m*(x) phi_n(x)
        if isinstance(gauge_link, bool) or link_dir == '0':
            if isinstance(gauge_link, bool):
                link_dir = '0'
            A = 1.0
            VDV = backend.zeros((N_mom, 1, Nev, Nev), dtype=backend.complex128)
            VDV[:, 0, :, :] = contract(
                'VN,mV,nV->mNn',
                eigvecs_conj_T,
                phase_exp,
                eigvecs_flat
            )
            return VDV / A

        # ---------------- Read gauge link at time slice t ----------------
        gauge_link_t = backend.asarray(gauge_link[:, t].copy())   # (Nd, Nx, Ny, Nz, 3, 3)
        gauge_link_t = gauge_link_t.reshape(Nd, Nx, Nx, Nx, 3, 3)

        # Map link_dir string to spatial axis index (used for backend.roll on eigvecs)
        if link_dir == 'T':      axis_dir = 0
        elif link_dir == 'Z':    axis_dir = 1
        elif link_dir == 'Y':    axis_dir = 2
        elif link_dir == 'X':    axis_dir = 3
        elif link_dir == 'all':  axis_dir = 4
        else:
            raise ValueError("Invalid link_dir")

        # ---------------- Case 2: Conserved current or temporal direction ----------------
        # For conserved currents or time-direction links, the contraction involves
        # two sets of eigenvectors (eigvecs and eigvecs_min) coupled by the temporal
        # gauge link (index 3 in the Nd=4 gauge link array).
        # Output has shape (2, Nev, Nev):
        #   [0] = eigvecs_min^dag @ U_t @ eigvecs
        #   [1] = eigvecs^dag @ U_t^dag @ eigvecs_min
        if conserved or link_dir == 'T':
            eigvecs_min = backend.asarray(eigvecs_min)
            A = 1.0
            _gauge_link = gauge_link_t[3]   # temporal gauge link is at Nd-index 3
            glink = _gauge_link.reshape(Nx**3, 3, 3)
            ev_min = eigvecs_min.reshape(Nev, Nx**3, 3)
            ev     = eigvecs.reshape(Nev, Nx**3, 3)

            VDV = backend.zeros((2, Nev, Nev), dtype=backend.complex128)
            VDV[0] = contract('nvc,vcb,Nvb->nN', ev_min.conj(), glink, ev)
            VDV[1] = contract('nvc,vbc,Nvb->nN', ev.conj(), glink.conj(), ev_min)
            return VDV / A

        # ---------------- Case 3: Spatial directions ----------------
        # For each spatial direction (or all three for 'all'), we build a gauge
        # transport path of length |link_indx| along that direction, apply it to
        # the rolled eigenvectors, and contract with the phase factor and
        # conjugate eigenvectors.
        #
        # Key optimization: instead of looping over momenta (old code), we use a
        # single einsum('VN,mV,nV->mNn', ...) that computes all momenta at once.

        # Determine which gauge-link spatial indices and eigenvector roll axes to iterate over
        if axis_dir == 4:          # 'all': sum over X, Y, Z spatial directions
            A = 3.0
            gauge_indices = [0, 1, 2]          # spatial gauge-link indices: X=0, Y=1, Z=2
            roll_axes = [1, 2, 3]              # corresponding axes in eigvecs array
        else:                      # single spatial direction: X, Y, or Z
            A = 1.0
            B = 3 - axis_dir                  # map: X(axis=3)->gauge[0], Y(2)->gauge[1], Z(1)->gauge[2]
            gauge_indices = [B]
            roll_axes = [axis_dir]

        VDV = backend.zeros((N_mom, 2 * link_max + 1, Nev, Nev), dtype=backend.complex128)

        # Identity matrix template for initializing the gauge path accumulator
        eye3 = backend.eye(3, dtype=backend.complex128)

        for g_idx, roll_ax in zip(gauge_indices, roll_axes):
            _gauge_link = gauge_link_t[g_idx]                # (Nx, Nx, Nx, 3, 3)

            for link_indx in range(-link_max, link_max + 1):
                # Roll eigenvectors by -link_indx along the spatial direction.
                # This shifts the "source" position so that the gauge path connects
                # x (sink) to x + link_indx (source).
                eig_rolled = backend.roll(eigvecs, -link_indx, axis=roll_ax)  # (Nev, Nx, Nx, Nx, 3)

                if link_indx == 0:
                    # Zero displacement: no gauge transport needed
                    link_rolled = eig_rolled.reshape(Nev, Nx**3, 3)
                else:
                    # Build the gauge transport path by multiplying gauge links
                    # along the spatial direction. Start from the identity matrix
                    # at each spatial point.
                    gauge_path = backend.tile(eye3, (Nx**3, 1, 1))

                    if link_indx < 0:
                        # Negative displacement: path goes backward, so we use the
                        # Hermitian conjugate of the gauge links.
                        # The product builds U(x-1)^dag @ U(x-2)^dag @ ... @ U(x+link_indx)^dag
                        steps = abs(link_indx)
                        for step in range(steps):
                            shift = steps - step
                            # Roll BEFORE reshape to respect spatial structure
                            U_shifted = backend.roll(_gauge_link, shift, axis=(roll_ax - 1)).reshape(Nx**3, 3, 3)
                            gauge_path = gauge_path @ U_shifted
                        # Hermitian conjugate: transpose color indices and complex-conjugate
                        gauge_path = gauge_path.transpose(0, 2, 1).conj()
                    else:  # link_indx > 0
                        # Positive displacement: path goes forward.
                        # The product builds U(x) @ U(x+1) @ ... @ U(x+link_indx-1)
                        for step in range(link_indx):
                            U_shifted = backend.roll(_gauge_link, -step, axis=(roll_ax - 1)).reshape(Nx**3, 3, 3)
                            gauge_path = gauge_path @ U_shifted

                    # Apply the gauge path to the rolled eigenvectors:
                    #   phi_n(x+link_indx) -> sum_cb U_path(x, cb) * phi_n(x+link_indx, b)
                    link_rolled = contract('vcb,Nvb->Nvc', gauge_path, eig_rolled.reshape(Nev, Nx**3, 3))

                # Compute contribution for ALL momenta at once (key optimization):
                #   V_{mn}(p, link_indx) = sum_x e^{-ipx} phi_m*(x) * [U_path phi_n](x)
                link_flat = link_rolled.reshape(Nev, V_full)         # (Nev, V)
                contrib = contract('VN,mV,nV->mNn', eigvecs_conj_T, phase_exp, link_flat)
                VDV[:, link_indx + link_max, :, :] += contrib

        return VDV / A
    
    def VdV_sink_t_link(
        self, 
        eigvecs,
        link_dir, 
        link_max, 
        phase_exp,
        gauge_link,
        t:int = 0,
        eigvecs_min = None,
        conserved = False,
        ):
        backend = get_backend()

        eigvecs = backend.asarray(eigvecs)
        phase_exp = backend.asarray(phase_exp)

        Nev = eigvecs.shape[0]
        Nx = eigvecs.shape[1]
        Nd = 4
        
        phase_exp = phase_exp.reshape(-1, Nx*Nx*Nx*3)
        
        if type(gauge_link) == bool:
            link_dir = '0'
            A = 1
            
        if (link_dir == '0'):
            VDV_link_cupy = backend.zeros((phase_exp.shape[0], 1, Nev, Nev), dtype=backend.complex128)
            VDV_link_cupy[:, 0] = contract(
                "VN,mV,nV->mNn", 
                backend.conj((eigvecs).reshape(Nev, Nx * Nx * Nx * 3).T), 
                phase_exp.reshape(-1, Nx*Nx*Nx*3),
                (eigvecs).reshape(Nev, Nx * Nx * Nx * 3)
                )
            
        else:
            gauge_link_t = backend.asarray(gauge_link[:, t].copy())

            gauge_link_t = gauge_link_t.reshape(Nd, Nx, Nx, Nx, 3, 3)
            if link_dir=='T':   axis_dir=0
            if link_dir=='Z':   axis_dir=1
            if link_dir=='Y':   axis_dir=2
            if link_dir=='X':   axis_dir=3
            if link_dir=='all': axis_dir=4

            if conserved == True or link_dir == 'T':
                eigvecs_min = backend.asarray(eigvecs_min)
                A = 1
                _gauge_link = gauge_link_t[3]
                VDV_link_cupy = backend.zeros((2,Nev,Nev), dtype=backend.complex128)
                VDV_link_cupy[0] = contract("nvc,vcb,Nvb->nN", backend.conj((eigvecs_min).reshape(Nev,Nx*Nx*Nx,3)), _gauge_link.reshape(Nx**3, 3, 3), ((eigvecs).reshape(Nev,Nx*Nx*Nx,3))) 
                VDV_link_cupy[1] = contract("nvc,vbc,Nvb->nN", backend.conj((eigvecs).reshape(Nev,Nx*Nx*Nx,3)), _gauge_link.reshape(Nx**3, 3, 3).conj(), ((eigvecs_min).reshape(Nev,Nx*Nx*Nx,3))) 
                
            else:
                VDV_link_cupy = backend.zeros((phase_exp.shape[0], 2*link_max+1, Nev, Nev), dtype=backend.complex128)

                if axis_dir == 4:
                    A = 3; B = 0

                else:
                    A = 1; B = 3 - axis_dir

                for axis_indx in range(A):
                    _gauge_link = gauge_link_t[axis_indx + B]
                    axis_dir = 3 - B if A == 1 else axis_indx + 1
                    
                    for link_indx in range(-link_max,link_max+1,1):
                        link_rolled = backend.zeros((Nev,Nx*Nx*Nx,3), dtype=backend.complex128)
                        gauge_link_rolled = backend.zeros((Nx*Nx*Nx,3,3), dtype=backend.complex128)
                        gauge_link_rolled[:] = backend.identity(3,dtype=backend.complex128)
                        eigvecs_link_rolled = backend.roll(eigvecs, -1*link_indx, axis=axis_dir)
                        
                        if link_indx == 0: 
                            link_rolled = eigvecs_link_rolled.reshape(Nev,Nx*Nx*Nx,3)
                                
                        else:
                            if link_indx < 0:
                                for link_indx_2 in range(abs(link_indx)):
                                    gauge_link_rolled = gauge_link_rolled @ backend.roll(_gauge_link, abs(link_indx)-link_indx_2, axis=(axis_dir-1)).reshape(Nx*Nx*Nx,3,3)

                                gauge_link_rolled = gauge_link_rolled.transpose(0,2,1).conj()
                                
                            if link_indx > 0:
                                for link_indx_2 in range(abs(link_indx)):
                                    gauge_link_rolled = gauge_link_rolled @ backend.roll(_gauge_link, -1* link_indx_2, axis=(axis_dir-1)).reshape(Nx*Nx*Nx,3,3)

                            link_rolled = contract('Vcb,NVb->NVc', gauge_link_rolled, eigvecs_link_rolled.reshape(Nev,Nx*Nx*Nx,3))
                            # for i in range(3):
                            #     link_rolled[:,:,i] = backend.sum(gauge_link_rolled[:,i,:] * eigvecs_link_rolled.reshape(Nev,Nx*Nx*Nx,3), axis=-1)
                        for mom_indx in range(phase_exp.shape[0]):
                            VDV_link_cupy[mom_indx, link_indx + link_max] = (
                                contract("VN,V,nV->Nn", backend.conj((eigvecs).reshape(Nev,Nx*Nx*Nx*3).T), (phase_exp.reshape(-1, Nx*Nx*Nx*3))[mom_indx], ((link_rolled).reshape(Nev,Nx*Nx*Nx*3)))
                                ) + VDV_link_cupy[mom_indx, link_indx + link_max]

        return (VDV_link_cupy / A)

        
    def Mom_VdV_sink_t(
        self,
        phase_exp,
        eigvecs = None,
        ):
        
        VdV_mom = self.Mom_VdV_sink_t_3(phase_exp = phase_exp, eigvecs = eigvecs)
        return VdV_mom

    # calculate the VVV mom part

    def Mom_VVV_sink_t(
        self,
        phase_exp,
        eigvecs,
        ):
        VVV_timeslice_t = self.Mom_VVV_sink_t_3(phase_exp = phase_exp, eigvecs = eigvecs)
        return VVV_timeslice_t

    def Mom_VdV_sink_t_2(
        self,
        phase_exp,
        eigvecs = None,
        ):
        
        VdV_mom = self.Mom_VdV_sink_t_3(phase_exp = phase_exp, eigvecs = eigvecs)
        return VdV_mom

    # calculate the VVV mom part

    def Mom_VVV_sink_t_2(
        self,
        phase_exp,
        eigvecs,
        ):
        
        VVV_timeslice_t = self.Mom_VVV_sink_t_3(phase_exp = phase_exp, eigvecs = eigvecs)
        return VVV_timeslice_t
        
    def Mom_VdV_sink_t_3(
        self,
        phase_exp,
        eigvecs = None,
        ):
        backend = get_backend()
        
        Nev_src_sink, Nz, Ny, Nx = eigvecs.shape[:4]
        phase_exp = phase_exp.reshape(-1, Nz*Ny*Nx*3)
        num_Mom = phase_exp.shape[0]
        
        VdV_mom = backend.zeros((num_Mom, Nev_src_sink, Nev_src_sink), dtype=complex)
        VdV_mom[:] = contract("bV,MV,cV->Mbc", backend.conj(((eigvecs).reshape(Nev_src_sink, -1))), phase_exp, (eigvecs).reshape(Nev_src_sink, -1))

        return VdV_mom

    # calculate the VVV mom part
    def Mom_VVV_sink_t_3(
        self,
        phase_exp,
        eigvecs,
        ):
        backend = get_backend()
    
        Nev_src_sink, Nz, Ny, Nx = eigvecs.shape[:4]
        phase_exp = phase_exp.reshape(-1, Nz, Ny, Nx)
        num_Mom = phase_exp.shape[0]
        
        VVV_timeslice_t = backend.zeros((num_Mom, Nev_src_sink, Nev_src_sink, Nev_src_sink), dtype=complex)
        
        for dir in range(1, Nx+1):    
            Z = dir ;Y = Nx + 1 ;X = Nx + 1
            
            Z0 = (Z-1) % Nx ; Y0 = (Y-1) % Nx ; X0 = (X-1) % Nx
            VVV_timeslice_t_1 = contract("Mzyx,azyx,bzyx,czyx->Mabc", phase_exp[:, Z0:Z, Y0:Y, X0:X] , eigvecs[:,Z0:Z, Y0:Y, X0:X, 0], eigvecs[:,Z0:Z, Y0:Y, X0:X, 1], eigvecs[:,Z0:Z, Y0:Y, X0:X, 2])
            VVV_timeslice_t_2 = contract("Mzyx,azyx,bzyx,czyx->Mabc", phase_exp[:, Z0:Z, Y0:Y, X0:X] , eigvecs[:,Z0:Z, Y0:Y, X0:X, 1], eigvecs[:,Z0:Z, Y0:Y, X0:X, 2], eigvecs[:,Z0:Z, Y0:Y, X0:X, 0])
            VVV_timeslice_t_3 = contract("Mzyx,azyx,bzyx,czyx->Mabc", phase_exp[:, Z0:Z, Y0:Y, X0:X] , eigvecs[:,Z0:Z, Y0:Y, X0:X, 2], eigvecs[:,Z0:Z, Y0:Y, X0:X, 0], eigvecs[:,Z0:Z, Y0:Y, X0:X, 1])
            VVV_timeslice_t_4 = contract("Mzyx,azyx,bzyx,czyx->Mabc", phase_exp[:, Z0:Z, Y0:Y, X0:X] , eigvecs[:,Z0:Z, Y0:Y, X0:X, 0], eigvecs[:,Z0:Z, Y0:Y, X0:X, 2], eigvecs[:,Z0:Z, Y0:Y, X0:X, 1])
            VVV_timeslice_t_5 = contract("Mzyx,azyx,bzyx,czyx->Mabc", phase_exp[:, Z0:Z, Y0:Y, X0:X] , eigvecs[:,Z0:Z, Y0:Y, X0:X, 1], eigvecs[:,Z0:Z, Y0:Y, X0:X, 0], eigvecs[:,Z0:Z, Y0:Y, X0:X, 2])
            VVV_timeslice_t_6 = contract("Mzyx,azyx,bzyx,czyx->Mabc", phase_exp[:, Z0:Z, Y0:Y, X0:X] , eigvecs[:,Z0:Z, Y0:Y, X0:X, 2], eigvecs[:,Z0:Z, Y0:Y, X0:X, 1], eigvecs[:,Z0:Z, Y0:Y, X0:X, 0])
            VVV_timeslice_t += VVV_timeslice_t_1 + VVV_timeslice_t_2 + VVV_timeslice_t_3 - VVV_timeslice_t_4 - VVV_timeslice_t_5 - VVV_timeslice_t_6

        return VVV_timeslice_t