from .backend import get_backend
from typing import Literal, Union, List
from .constant import fm2GeV
from .corr_base_functions import ArraySlicer

def Mom2GeV(Nx: int, alttc: float, Mom: Union[float, List[float], List[List[float]]], 
            M0: Union[float, List[float]]):
    """
    将格点QCD动量转换为真实能量(GeV)，支持 M0 为 float 或 list。
    
    参数
    ----------
    Nx : int
        格子空间方向长度
    alttc : float
        格距
    Mom : float 或 List[float] 或 List[List[float]]
        动量输入：
        - 数值：直接作为动量模平方
        - 一维列表 [px, py, pz]：计算模平方 sum(pi^2)
        - 二维列表 [[...], ...]：对每个子列表计算模平方，返回结果列表
    M0 : float 或 List[float]
        质量项：
        - float：原逻辑，能量 = sqrt(single_Q2^2 * mom_sq + M0^2)
        - list：能量 = sum_i sqrt(single_Q2^2 * mom_sq + M0_i^2)
    
    返回
    -------
    float 或 List[float]
        转换后的能量（GeV），类型取决于 Mom 和 M0 的组合。
    """
    single_Q2 = 2 * 3.14159265359 / Nx * (fm2GeV / alttc)
    
    # ---------- 计算动量模平方 mom_sq ----------
    if isinstance(Mom, (int, float)):
        mom_sq = Mom
    elif isinstance(Mom, list):
        if not Mom:                     # 空列表
            mom_sq = 0.0
        elif isinstance(Mom[0], (int, float)):
            mom_sq = sum(x**2 for x in Mom)   # 一维列表 -> 模平方（标量）
        elif isinstance(Mom[0], list):
            mom_sq = [sum(x**2 for x in sub) for sub in Mom]  # 二维列表 -> 模平方列表
        else:
            raise TypeError(f"Mom 列表元素类型不支持: {type(Mom[0])}")
    else:
        raise TypeError(f"Mom 类型不支持: {type(Mom)}")
    
    # ---------- 根据 M0 类型计算能量 ----------
    if isinstance(M0, (int, float)):
        # M0 为标量
        if isinstance(mom_sq, list):
            return [(single_Q2**2 * msq + M0**2) ** 0.5 for msq in mom_sq]
        else:
            return (single_Q2**2 * mom_sq + M0**2) ** 0.5
    
    elif isinstance(M0, list):
        # M0 为列表：对每个 M0_i 计算能量并求和
        if not M0:
            # 空列表：和为零
            total = 0.0
            if isinstance(mom_sq, list):
                return [0.0] * len(mom_sq)
            else:
                return 0.0
        
        if isinstance(mom_sq, list):
            # 多个动量，每个动量对应一个求和值
            result = []
            for msq in mom_sq:
                total = 0.0
                for m in M0:
                    total += (single_Q2**2 * msq + m**2) ** 0.5
                result.append(total)
            return result
        else:
            # 单个动量模平方
            total = 0.0
            for m in M0:
                total += (single_Q2**2 * mom_sq + m**2) ** 0.5
            return total
    
    else:
        raise TypeError(f"M0 类型不支持: {type(M0)}。请使用 float 或 list。")
    
def loop_tsrc(data, indx:list = [-2, -3], Boundary_Conditions: Literal['Periodic', 'Antiperiodic'] = 'Periodic', Ctype: Literal['2pt', '3pt'] = '2pt', t_sep:int = 0):
    import numpy as np
    type_cupy = False
    
    if type(data).__module__ == 'cupy':
        data = data.get()
        type_cupy = True
        
        
    data_init = data.copy()
    data_shape = data_init.shape
    
    if len(indx) != 2:
        raise ValueError('indx must have 2 elements for tsrc, tsink')

    if data_shape[indx[0]] != data_shape[indx[1]]:
        raise ValueError('data[indx] must have the same size')
    
    Nt = data_shape[indx[0]]
    data_looped = np.zeros_like(np.sum(data_init, axis = indx[0], keepdims = True))
    if Ctype == '2pt':
        if Boundary_Conditions == 'Antiperiodic':
            for tsrc in range(Nt):
                for tsink in range(Nt):
                    data_slicer = ArraySlicer(data_init)

                    if(tsink < tsrc):
                        data_init = data_slicer.assign(
                            dims = [indx[0], indx[1]], 
                            indices = [[tsrc], [tsink]], 
                            values = - 1.0 * data_slicer.slice(dims = indx, indices = [[tsrc], [tsink]])
                            )

        for tsrc in range(Nt):
            for tsink in range(Nt):

                data_looped_slicer = ArraySlicer(data_looped)
                data_slicer = ArraySlicer(data_init)

                data_looped = data_looped_slicer.assign(
                    dims = [indx[1]], 
                    indices = [(tsink - tsrc + Nt) % Nt], 
                    values = data_slicer.slice(dims = indx, indices = [[tsrc], [tsink]]) + data_looped_slicer.slice(dims = [indx[1]], indices = [(tsink - tsrc + Nt) % Nt])
                    )
                
                # print(data_looped)
    elif Ctype == '3pt':
        data_slicer = ArraySlicer(data_init)

        data_init = data_slicer.assign(
            dims = [indx[0]], 
            indices = [[x for x in range(Nt - t_sep, Nt, 1)]], 
            values = - 1.0 * data_slicer.slice(dims = [indx[0]], indices = [[x for x in range(Nt - t_sep, Nt, 1)]])
            )
        
        for tsrc in range(Nt):
            for tsink in range(Nt):

                data_looped_slicer = ArraySlicer(data_looped)
                data_slicer = ArraySlicer(data_init)

                data_looped = data_looped_slicer.assign(
                    dims = [indx[1]], 
                    indices = [(tsink - tsrc + Nt) % Nt], 
                    values = data_slicer.slice(dims = indx, indices = [[tsrc], [tsink]]) + data_looped_slicer.slice(dims = [indx[1]], indices = [(tsink - tsrc + Nt) % Nt])
                    )
    
    backend = get_backend()
    if backend.__name__ == 'cupy' and type_cupy == False:
        return data_looped
    
    return backend.asarray(data_looped)

plot_analyse_marker = ['s','*','+','x','p','h','v','X','D','P','H','o']
plot_analyse_color = ['#3498DB', '#ff7f0e', '#2ECC71', '#E74C3C', '#9467bd', '#8c564b', '#CB4335','#e377c2', '#7f7f7f', '#F1C40F', '#17becf', '#2ca02c']

def get_data_info(data, dtype, ttype:Literal['t_src', 'all_t'] = 't_src'):
    '''
    get data info 
    '''
    if dtype == complex:
        if ttype == 't_src':
            Nconf_axes = -2
            
        elif ttype == 'all_t':
            Nconf_axes = -3
    else:
        if ttype == 't_src':
            Nconf_axes = -3
            
        elif ttype == 'all_t':
            Nconf_axes = -4
            
    return data.shape[Nconf_axes], data.shape[Nconf_axes + 1], Nconf_axes

def sum_over_array_of_list(arr, axes, groupings):
    """
    Sum over specified axes according to index groupings.

    For each axis in `axes`, the corresponding list in `groupings` defines groups
    of original indices. All indices in a group are summed together, reducing the
    axis size to the number of groups.

    Parameters
    ----------
    arr : backend.ndarray
        Input array (any shape).
    axes : tuple or list of int
        Axes to aggregate (0‑based).
    groupings : list of list of list of int
        For each axis, a list of groups, each group is a list of indices.
        All indices must be covered exactly once.

    Returns
    -------
    backend.ndarray
        Aggregated array. Shape: original shape with each aggregated axis replaced
        by the number of groups for that axis.

    Examples
    --------
    >>> a = backend.arange(24).reshape(2,3,4)
    >>> axes = (1,2)
    >>> groupings = ([[0,2],[1]], [[0,3],[1,2]])   axis1: 2 groups, axis2: 2 groups
    >>> aggregate_by_groups(a, axes, groupings).shape
    (2, 2, 2)
    """
    backend = get_backend()
    arr = backend.asarray(arr)
    # validate inputs
    if len(axes) != len(groupings):
        raise ValueError("axes and groupings must have same length")
    
    for ax, groups in zip(axes, groupings):
        if ax < 0 or ax >= arr.ndim:
            raise ValueError(f"axis {ax} out of range")
        
        all_idx = set()
        for g in groups:
            for i in g:
                if i < 0 or i >= arr.shape[ax]:
                    raise ValueError(f"index {i} out of range on axis {ax}")
        
                if i in all_idx:
                    raise ValueError(f"duplicate index {i} on axis {ax}")
        
                all_idx.add(i)
        
        if len(all_idx) != arr.shape[ax]:
            missing = set(range(arr.shape[ax])) - all_idx
            raise ValueError(f"axis {ax}: indices {missing} not covered")

    # process each axis sequentially
    for ax, groups in zip(axes, groupings):
        # build group id for each original index
        group_id = backend.empty(arr.shape[ax], dtype=int)
        for gid, g in enumerate(groups):
            group_id[g] = gid

        # sort axis so that indices of the same group become contiguous
        sort_idx = backend.argsort(group_id)
        arr = backend.take(arr, sort_idx, axis=ax)

        # find boundaries where group id changes
        _, starts = backend.unique(group_id[sort_idx], return_index=True)

        # sum each contiguous block
        arr = backend.add.reduceat(arr, starts, axis=ax)

    return arr

def Jackknife(data, Nconf_axes=0, only_sample:bool=False, cov_axes = None):
    """
    Jackknife 重采样函数，支持沿指定轴构建协方差矩阵。

    参数
    ----------
    data : array
        输入数据数组，至少包含组态轴。
    Nconf_axes : int
        组态所在的轴编号（默认为 0）。
    only_sample : bool
        若为 True，仅返回 Jackknife 样本，不计算均值、误差和协方差。
    cov_axes : int 或 tuple 或 None
        指定构建协方差矩阵的轴。对该轴求外积，其他轴保持不变。
        若为 None，则不计算协方差矩阵。

    返回
    ----------
    dict:
        'data_sample'  — Jackknife 样本，形状与输入相同
        'data_mean'    — 数据均值（去掉组态轴）
        'data_err'     — 标准误差 sqrt(Nconf-1) * std(samples)
        'data_cov'     — 协方差矩阵（仅当 cov_axes 不为 None 时返回）
    """
    backend = get_backend()
    ndim = data.ndim
    Nconf_axes = Nconf_axes % ndim
    Nconf = data.shape[Nconf_axes]

    # 1. 所有组态的 Jackknife 样本（去掉第 k 个组态的均值）
    data_sum = backend.sum(data, axis=Nconf_axes, keepdims=True)
    data_sample = -(data - data_sum) / (Nconf - 1)

    if only_sample:
        return {'data_sample': data_sample}

    # 2. 均值与标准误差
    data_mean = backend.mean(data, axis=Nconf_axes)
    data_err = backend.sqrt(Nconf - 1) * backend.std(data_sample, axis=Nconf_axes)

    result = {'data_sample': data_sample,
              'data_mean': data_mean,
              'data_err': data_err}

    # 3. 协方差矩阵（均值的协方差）
    if cov_axes is not None:
        if isinstance(cov_axes, int):
            cov_axes = (cov_axes % ndim,)
        else:
            cov_axes = tuple(ax % ndim for ax in cov_axes)

        # 残差：Jackknife 样本 - 均值（自动广播）
        residual = data_sample - data_mean

        # 确定轴顺序：组态轴 -> 其他轴 -> 协方差轴
        all_axes = list(range(ndim))
        other_axes = [ax for ax in all_axes if ax != Nconf_axes and ax not in cov_axes]
        new_order = [Nconf_axes] + other_axes + list(cov_axes)
        r = backend.transpose(residual, new_order)

        # 展平协方差轴以便使用 einsum
        shape_other = [residual.shape[ax] for ax in other_axes]
        shape_cov = [residual.shape[ax] for ax in cov_axes]
        N_cov = 1
        for s in shape_cov:
            N_cov *= s
        r_flat = r.reshape([Nconf] + shape_other + [N_cov])

        # 对每个组态计算外积，再求和；可以启用优化路径加速
        cov_sum = backend.einsum('n...i,n...j->...ij', r_flat, r_flat, optimize=True)

        # 均值协方差矩阵 = (N-1)/N * 外积和
        cov = cov_sum * (Nconf - 1) / Nconf
        result['data_cov'] = cov.reshape(shape_other + shape_cov + shape_cov)

    return result

#Bootstrap: 随机抽取N次原样本中的M个数据，
def Bootstrap(data, Nconf_axes=0, only_sample=False, cov_axes=None, M=0, N=0):
    """
    Bootstrap 重采样函数，支持沿指定轴构建协方差矩阵。

    参数
    ----------
    data : array
        输入数据数组，至少包含组态轴。
    Nconf_axes : int
        组态所在的轴编号（默认为 0）。
    only_sample : bool
        若为 True，仅返回 Bootstrap 样本，不计算均值、误差和协方差。
    cov_axes : int 或 tuple 或 None
        指定构建协方差矩阵的轴。对该轴求外积，其他轴保持不变。
        若为 None，则不计算协方差矩阵。
    M : int
        每个 Bootstrap 样本抽取的组态数。默认为 Nconf - 5。
    N : int
        Bootstrap 样本总数。默认为 Nconf * 4。

    返回
    ----------
    dict:
        'data_sample'  — Bootstrap 样本，形状 (N, dim1, dim2, ...)
        'data_mean'    — Bootstrap 样本均值
        'data_err'     — Bootstrap 样本标准差
        'data_cov'     — 协方差矩阵（仅当 cov_axes 不为 None 时返回）
    """
    backend = get_backend()
    ndim = data.ndim
    Nconf_axes = Nconf_axes % ndim

    datashape = list(data.shape)
    Nconf = datashape[Nconf_axes]

    # 默认参数
    if M == 0:
        M = max(Nconf - 5, 1)
    if N == 0:
        N = Nconf * 4

    # 参数校验
    if M > Nconf:
        raise ValueError(f'M ({M}) 不能大于 Nconf ({Nconf})')

    # 生成 bootstrap 样本：每个样本有放回地随机抽取 M 个组态并求均值
    sample_shape = tuple(datashape[:Nconf_axes] + datashape[Nconf_axes+1:])
    data_sample = backend.zeros((N,) + sample_shape)
    for i in range(N):
        indices = backend.random.choice(Nconf, size=M, replace=True)
        selected = backend.take(data, indices, axis=Nconf_axes)
        data_sample[i] = backend.mean(selected, axis=Nconf_axes)

    if only_sample:
        return {'data_sample': data_sample}

    # 计算 bootstrap 样本的均值和标准差
    data_mean = backend.mean(data_sample, axis=0)
    data_err = backend.std(data_sample, axis=0)

    result = {
        'data_sample': data_sample,
        'data_mean': data_mean,
        'data_err': data_err,
    }

    # 沿用户指定的轴构建协方差矩阵（参照 Jackknife 模式）
    if cov_axes is not None:
        if isinstance(cov_axes, int):
            cov_axes = (cov_axes % ndim,)
        else:
            cov_axes = tuple(ax % ndim for ax in cov_axes)

        # cov_axes 相对于原始数据轴编号
        # data_sample 形状为 (N, *sample_shape)，Nconf_axes 已移除，N 轴插入到位置 0
        # 映射关系：原始轴 < Nconf_axes -> 新轴 = 原始轴 + 1；原始轴 > Nconf_axes -> 新轴 = 原始轴
        cov_axes_r = tuple(ax + 1 if ax < Nconf_axes else ax for ax in cov_axes)

        # 计算残差：bootstrap 样本 - 样本均值
        residual = data_sample - backend.mean(data_sample, axis=Nconf_axes, keepdims = True)  # 形状: (N, dim1, dim2, ...)

        # 确定轴顺序：N 轴 -> 其他轴 -> 协方差轴
        ndim_sample = residual.ndim  # 包含 N 轴的总维度
        all_axes = list(range(ndim_sample))
        other_axes = [ax for ax in all_axes if ax != 0 and ax not in cov_axes_r]
        new_order = [0] + other_axes + list(cov_axes_r)
        r = backend.transpose(residual, new_order)

        # 记录各维度大小
        shape_other = [residual.shape[ax] for ax in other_axes]
        shape_cov = [residual.shape[ax] for ax in cov_axes_r]
        N_cov = 1
        for s in shape_cov:
            N_cov *= s

        # 展平：(N, *shape_other, N_cov)
        r_flat = r.reshape((N,) + tuple(shape_other) + (N_cov,))

        # 通过外积计算协方差矩阵，对 bootstrap 样本轴取均值
        cov = backend.einsum('n...i,n...j->...ij', r_flat, r_flat) / N

        # 将协方差轴恢复为原始维度
        result['data_cov'] = cov.reshape(tuple(shape_other) + tuple(shape_cov) + tuple(shape_cov))

    return result


def meff(data_sample, alttc, Nconf_axes:int = 0, Nt_axes:int = 1, meff_type:Literal['log', 'cosh', 'GEVP'] = 'log'):
    backend = get_backend()
    
    if data_sample.dtype != float:
        raise ValueError(f"data_sample's dtype must be float not {data_sample.dtype}")
    
    Nconf = data_sample.shape[Nconf_axes]
    Nt = data_sample.shape[Nt_axes]
    
    meff_sample = backend.zeros_like(data_sample)
    
    with backend.errstate(divide='ignore', invalid='ignore'):
        if meff_type == 'log':
            ArraySlicer(meff_sample).assign(
                dims = [Nt_axes], 
                indices = [[x for x in range(Nt - 1)]], 
                values = backend.log(
                    ArraySlicer(data_sample).slice(dims = [Nt_axes], indices = [[x for x in range(Nt - 1)]]) / ArraySlicer(data_sample).slice(dims = [Nt_axes], indices = [[x + 1 for x in range(Nt - 1)]])
                    ) * (fm2GeV / alttc)
                )
            
        elif meff_type == 'cosh':
            ArraySlicer(meff_sample).assign(
                dims = [Nt_axes], 
                indices = [[x for x in range(Nt - 2)]], 
                values = backend.arccosh(
                    (ArraySlicer(data_sample).slice(dims = [Nt_axes], indices = [[x + 2 for x in range(Nt - 2)]]) + ArraySlicer(data_sample).slice(dims = [Nt_axes], indices = [[x for x in range(Nt - 2)]]))/(2 * ArraySlicer(data_sample).slice(dims = [Nt_axes], indices = [[x + 1 for x in range(Nt - 2)]]))
                    ) * (fm2GeV / alttc)
                )
        
        elif meff_type == 'GEVP':
            ArraySlicer(meff_sample).assign(
                dims = [Nt_axes], 
                indices = [[x for x in range(Nt - 1)]], 
                values = backend.log(
                    ArraySlicer(data_sample).slice(dims = [Nt_axes], indices = [[x for x in range(Nt - 1)]]) / ArraySlicer(data_sample).slice(dims = [Nt_axes], indices = [[x + 1 for x in range(Nt - 1)]])
                    ) * (fm2GeV / alttc)
                )
        
    data_info = {}
    
    meff_mean = backend.mean(meff_sample, axis = Nconf_axes)
    meff_err = backend.std(meff_sample, axis = Nconf_axes) * backend.sqrt(Nconf - 1)
    
    data_info['data_sample'] = meff_sample    
    data_info['data_mean'] = meff_mean
    data_info['data_err'] = meff_err
    
    return data_info

def PDF(data_3pt_sample, data_2pt_sample, t_sep:int = 0, Nconf_axes:int = 0, link_axes:int = 1, t_src_axes:int = 2, t_sink_axes:int = 2, link_fold:bool = False): #P, ENV, t_sep, link, Ncnfg, Nt, dtype=complex
    
    backend = get_backend()
    
    Nt = data_3pt_sample.shape[t_src_axes]
    Nconf = data_3pt_sample.shape[Nconf_axes]
    N_link = data_3pt_sample.shape[link_axes]
    link_max = int(N_link//2)

    if link_fold == True:
        N_link = link_max + 1
        data_3pt_sample_fold = (
            ArraySlicer(data_3pt_sample).slice(dims = [link_axes], indices = [x for x in range(link_max, 2*link_max + 1, 1)]) + 
            ArraySlicer(data_3pt_sample).slice(dims = [link_axes], indices = [x for x in range(0, link_max + 1, 1)][::-1])
            )/2
    
    else:
        data_3pt_sample_fold = data_3pt_sample

    PDF_3pt_2pt_sample = backend.zeros_like(data_3pt_sample_fold)

    # the 3pt/2pt part
    for t_src in range(Nt):
        ArraySlicer(PDF_3pt_2pt_sample).assign(
            dims = [t_src_axes], 
            indices = [[t_src]], 
            values = ArraySlicer(data_3pt_sample_fold).slice(dims = [t_src_axes], indices = [[t_src]])/ArraySlicer(data_2pt_sample).slice(dims = [t_src_axes, t_sink_axes], indices = [[t_src], [(t_sep + t_src)%Nt]])
            )
    
    PDF_3pt_2pt_mean = backend.mean(PDF_3pt_2pt_sample[:], axis = Nconf_axes)
    
    data_info = {}
        
    data_info['data_sample'] = PDF_3pt_2pt_sample
    data_info['data_mean'] = PDF_3pt_2pt_mean
    data_info['data_err'] = backend.std(PDF_3pt_2pt_sample, axis = Nconf_axes) * backend.sqrt(Nconf - 1)
    # data_info['data_cov'] = PDF_3pt_2pt_cov

    return data_info

def solve_gevp(C, t0):
    """
    求解广义特征值问题 (GEVP)，提取格点场论中的能级。

    Parameters
    ----------
    C : backend.ndarray
        关联函数矩阵，形状为 (N, N, Nt)，其中 N 为插值场数目，Nt 为时间方向格点数。
    t0 : int
        参考时间切片，满足 GEVP 条件 t > t0。

    Returns
    -------
    eigenvalues : backend.ndarray, 形状 (N, Nt//2)
        排序后的广义特征值。
        - t < t0 时，按升序排列
        - t >= t0 时，按降序排列（最大值对应基态）

    Notes
    -----
    公式 (2.8): λ_n^(0)(t, t0) = exp(-E_n * (t - t0))

    三种情况：
    - t < t0:  λ > 1，升序排列
    - t = t0:  λ = 1，退化情况
    - t > t0:  λ < 1，降序排列，最大值对应最低能级
    """
    backend = get_backend()
    
    from scipy.linalg import eigh
    
    # 输入检查
    if C.ndim != 3:
        raise ValueError(f"C 的维度应为 3，实际为 {C.ndim}，期望形状 (N, N, Nt)")

    if C.shape[0] != C.shape[1]:
        raise ValueError(f"C 的前两个维度应相等，实际形状为 {C.shape[:2]}")

    N = C.shape[0]

    Nt = C.shape[2]

    if t0 < 0 or t0 >= Nt:
        raise ValueError(f"t0 = {t0} 超出有效范围 [0, {Nt - 1}]")

    # 对称化关联矩阵（确保厄米性）
    C = (C.conj().transpose(1, 0, 2) + C) / 2

    # 初始化输出数组
    C_GEVP = backend.zeros((N, Nt))
    C_eigenvectors = backend.zeros((N, N, Nt), dtype = complex)

    # 对每个时间切片求解 GEVP
    for t in range(0, Nt, 1):
        # 求解广义特征值问题 C(t) v = λ C(t0) v
        eigenvalues, eigenvectors = eigh(C[..., t], C[..., t0])
        eigenvalues = eigenvalues.real

        # 根据 t 与 t0 的关系确定排序方式
        #   t < t0:  λ = exp(-E_n*(t-t0)) > 1，升序
        #   t >= t0: λ = exp(-E_n*(t-t0)) <= 1，降序
        # 最大本征值对应最低能级（基态）
        if t < t0:
            order = backend.argsort(eigenvalues)          # 升序
            
        elif t >= t0:
            order = backend.argsort(eigenvalues)[::-1]    # 降序
        
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        coeff = eigenvectors.conj().T@eigenvectors
        eigenvectors = eigenvectors/backend.sqrt(backend.diagonal(coeff).real).reshape(1, -1)
        # coeff_exp = backend.diagonal(eigenvectors) / backend.sqrt(backend.diagonal(eigenvectors) * backend.diagonal(eigenvectors).conj())
        # eigenvectors = eigenvectors / coeff_exp.reshape(1, -1)
        
        # eigenvectors = eigenvectors/(coeff)**(1/2).reshape(1, -1)
        
        # 存储排序后的特征值
        C_GEVP[:, t] = eigenvalues
        C_eigenvectors[..., t] = eigenvectors
        
    return C_GEVP, C_eigenvectors

# def dis_connect(self, data_info:dict, dtype:Literal['Im', 'Re', 'Complex'] = 'Im', conf_list:list = [], compensation_matrix = 0):
#     if dtype == 'Re':
#         axis = 0

#     elif dtype == 'Im':
#         axis = 1

#     if conf_list == []:
#         conf_list = [x_indx for x_indx, x in enumerate(data_info['conf'])]

#     Nt = data_info['Nt']
#     tsep = data_info['tsep']

#     if ('2pt_mu_nu' in data_info['read_type']) and ('bubble' in data_info['read_type']):
#         if dtype == 'Complex':
#             data_2pt_mu_nu = data_info['data_2pt_mu_nu'][..., 0].copy() + 1j * data_info['data_2pt_mu_nu'][..., 1].copy()
#             data_bubble = data_info['data_bubble'][..., 0].copy() + 1j * data_info['data_bubble'][..., 1].copy() 
#             arr_dtype = complex
#         else:    
#             data_2pt_mu_nu = data_info['data_2pt_mu_nu'][..., axis].copy()
#             data_bubble = data_info['data_bubble'][..., 0].copy()
#             arr_dtype = float
            
#         data_2pt_bubble_shape = list(data_info['data_2pt_mu_nu'].shape)
#         data_2pt_bubble_shape[param_position(param = 'tsep')] = len(tsep)
#         data_2pt_bubble_shape[param_position(param = 'link_list')] = len(data_info['link_list'])
#         data_2pt_bubble_matrix = backend.zeros(data_2pt_bubble_shape, dtype = float)
        
#         data_2pt_mu_nu_shape = list(data_2pt_mu_nu.shape)
#         data_2pt_mu_nu_shape[param_position(param = 'link_list')] = len(data_info['link_list'])
#         _data_bubble = backend.zeros(data_2pt_mu_nu_shape, dtype = arr_dtype)

#         data_2pt_mu_nu_mean = backend.mean(data_2pt_mu_nu, axis = -3, keepdims = True)
#         data_bubble_mean = backend.mean(data_bubble, axis = -2, keepdims = True)

#         _data_2pt_mu_nu = data_2pt_mu_nu - data_2pt_mu_nu_mean

#         if ttype(compensation_matrix) != int:
#             for t in range(Nt):
#                 compensation_matrix[..., t, :, :] = backend.roll(compensation_matrix[..., t, :, :], axis = -2, shift = -t)
            
#         for t in range(Nt):
#             _data_2pt_mu_nu[..., t, :] = backend.roll(_data_2pt_mu_nu[..., t, :], axis = -1, shift = -t)
        
#         for t in range(Nt):
#             _data_bubble[..., t, :] = backend.roll(data_bubble - data_bubble_mean, axis = -1, shift = -t)

#         # _data_bubble = backend.roll(_data_bubble, axis = 0, shift = 3)
#         if data_info['dtype'] == 'PFF':
#             _data_2pt_mu_nu = backend.roll(_data_2pt_mu_nu, axis = 0, shift = len( data_info['Mom']) // 2)

#         data_2pt_bubble_1 = contract('MGETLCmn,MGETLCmb->MGETLCmnb', _data_2pt_mu_nu[..., tsep], _data_bubble)
        
#         if data_info['dtype'] == 'PFF':
#             # _data_bubble = backend.roll(_data_bubble, axis = 0, shift = 3)
#             data_2pt_bubble_2 = contract('MGETLCmn,MGETLCmb->MGETLCmbn', _data_2pt_mu_nu, _data_bubble[..., tsep])
            
#             for t_sep_indx, t_sep in enumerate(tsep):
#                 if dtype == 'Complex':
#                     data_2pt_bubble_matrix[:, :, :, t_sep_indx:(t_sep_indx+1), ..., 0] = (
#                         backend.append(data_2pt_bubble_1[..., t_sep_indx:(t_sep_indx+1), :t_sep] , data_2pt_bubble_2[..., t_sep_indx:(t_sep_indx+1), t_sep:], axis = -1)
#                         ).transpose(0,1,2,7,4,5,6,3,8).reshape(data_2pt_bubble_matrix[:, :, :, t_sep_indx:(t_sep_indx+1), ..., 0].shape).real

#                     data_2pt_bubble_matrix[:, :, :, t_sep_indx:(t_sep_indx+1), ..., 1] = (
#                         backend.append(data_2pt_bubble_1[..., t_sep_indx:(t_sep_indx+1), :t_sep] , data_2pt_bubble_2[..., t_sep_indx:(t_sep_indx+1), t_sep:], axis = -1)
#                         ).transpose(0,1,2,7,4,5,6,3,8).reshape(data_2pt_bubble_matrix[:, :, :, t_sep_indx:(t_sep_indx+1), ..., 0].shape).imag

#                     if ttype(compensation_matrix) != int:
#                         data_2pt_bubble_matrix[:, :, :, t_sep_indx:(t_sep_indx + 1), ..., t_sep, :] = compensation_matrix[..., t_sep, :]

#                 else:
#                     data_2pt_bubble_matrix[:, :, :, t_sep_indx:(t_sep_indx+1), ..., axis] = (
#                         backend.append(data_2pt_bubble_1[..., t_sep_indx:(t_sep_indx+1), :t_sep] , data_2pt_bubble_2[..., t_sep_indx:(t_sep_indx+1), t_sep:], axis = -1)
#                         ).transpose(0,1,2,7,4,5,6,3,8).reshape(data_2pt_bubble_matrix[:, :, :, t_sep_indx:(t_sep_indx+1), ..., 0].shape)

#                     if ttype(compensation_matrix) != int:
#                         data_2pt_bubble_matrix[:, :, :, t_sep_indx:(t_sep_indx + 1), ..., t_sep, axis] = compensation_matrix[..., t_sep, axis]

#         elif data_info['dtype'] == 'PDF':
#             link_max = (len(data_info['link_list']) - 1)//2
            
#             if dtype == 'Complex':
#                 data_2pt_bubble_matrix[..., 0] = (data_2pt_bubble_1).transpose(0,1,2,7,4,5,6,3,8).reshape(data_2pt_bubble_matrix[..., 0].shape).real
#                 data_2pt_bubble_matrix[..., 1] = (data_2pt_bubble_1).transpose(0,1,2,7,4,5,6,3,8).reshape(data_2pt_bubble_matrix[..., 0].shape).imag

#             else:
#                 data_2pt_bubble_matrix[..., axis] = (data_2pt_bubble_1).transpose(0,1,2,7,4,5,6,3,8).reshape(data_2pt_bubble_matrix[..., 0].shape)
            
#             if self.link_fold:
#                 data_2pt_bubble_matrix = backend.append(data_2pt_bubble_matrix[:, :, :, :, link_max:link_max+1], (data_2pt_bubble_matrix[:, :, :, :, link_max + 1:] - data_2pt_bubble_matrix[:, :, :, :, 0:link_max][:, :, :, :, ::-1])/2, axis = 4)
                
#         data_info['data_2pt_bubble'] = data_2pt_bubble_matrix.sum(-3)[..., conf_list, :, :]

#         data_info = Jackknife(data_info, name = 'data_2pt_bubble', only_sample = True)

#     return data_info
    
# def GEVP_ndarray(data_sample):
    