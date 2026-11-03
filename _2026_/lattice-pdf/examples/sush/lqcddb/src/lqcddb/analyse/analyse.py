from ..base.backend import get_backend
from typing import Literal, Union, List
from ..constant.constant import fm2GeV
from ..base.base_functions import ArraySlicer

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
        if Boundary_Conditions == 'Antiperiodic':
            sign_3pt = -1.0

        elif Boundary_Conditions == 'Periodic':
            sign_3pt = 1.0

        data_slicer = ArraySlicer(data_init)

        data_init = data_slicer.assign(
            dims = [indx[0]], 
            indices = [[x for x in range(Nt - t_sep, Nt, 1)]], 
            values = sign_3pt * data_slicer.slice(dims = [indx[0]], indices = [[x for x in range(Nt - t_sep, Nt, 1)]])
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

def mean_over_array_of_list(arr, axes, groupings):
    """
    Compute mean over specified axes according to index groupings.

    For each axis in `axes`, the corresponding list in `groupings` defines groups
    of original indices. The mean is computed over each group, reducing the axis
    size to the number of groups.

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
        Aggregated array with mean values. Shape: original shape with each
        aggregated axis replaced by the number of groups for that axis.

    Examples
    --------
    >>> a = backend.arange(24).reshape(2,3,4)
    >>> axes = (1,2)
    >>> groupings = ([[0,2],[1]], [[0,3],[1,2]])   # axis1: 2 groups, axis2: 2 groups
    >>> mean_over_array_of_list(a, axes, groupings).shape
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

        # find group boundaries and sizes
        _, starts, counts = backend.unique(
            group_id[sort_idx], return_index=True, return_counts=True
        )

        # sum each contiguous block
        arr = backend.add.reduceat(arr, starts, axis=ax)

        # divide by group sizes (broadcast along the aggregated axis)
        broadcast_shape = [1] * arr.ndim
        broadcast_shape[ax] = -1
        arr = arr / counts.reshape(broadcast_shape)

    return arr

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

    第 0 个样本为全部 Nconf 个组态的无放回抽取 (即原始数据均值)，
    其余 N-1 个样本为有放回随机抽取 M 个组态的均值。

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
        每个 Bootstrap 样本 (i>=1) 抽取的组态数。默认为 Nconf - 5。
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

    # # 参数校验
    # if M > Nconf:
    #     raise ValueError(f'M ({M}) 不能大于 Nconf ({Nconf})')

    # 生成 bootstrap 样本
    # 第一个样本 (i=0): 全部组态无放回抽取 (等同于原始数据均值)
    # 其余样本 (i>=1): 有放回地随机抽取 M 个组态并求均值
    sample_shape = tuple(datashape[:Nconf_axes] + datashape[Nconf_axes+1:])
    data_sample = backend.zeros((N,) + sample_shape, dtype = complex)

    # i=0: 全部组态
    indices = backend.arange(Nconf)
    selected = backend.take(data, indices, axis=Nconf_axes)
    data_sample[0] = backend.mean(selected, axis=Nconf_axes)

    for i in range(1, N):
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

def _validate_axes(data_3pt_sample, data_2ptI_sample, data_2ptF_sample,
                   Nconf_axes, tau_axes, t_sink_axes, t_src_axes, link_axes):
    """Validate axis bounds and check dimension count consistency.

    Only requires that all arrays have the same number of dimensions (ndim).
    Non-special axes (i.e., not time/config/link) are allowed to have different
    sizes and will be handled by numpy/cupy broadcasting.  The three-point
    function's dimensions are taken as primary for the output shape.
    """
    arrays = {
        'data_3pt_sample': data_3pt_sample,
        'data_2ptI_sample': data_2ptI_sample,
    }
    if data_2ptF_sample is not None:
        arrays['data_2ptF_sample'] = data_2ptF_sample

    # Axis bounds check
    for name, arr in arrays.items():
        ndim = arr.ndim
        for axis_name, axis_val in [
            ('Nconf_axes', Nconf_axes),
            ('tau_axes', tau_axes),
            ('t_sink_axes', t_sink_axes),
            ('t_src_axes', t_src_axes),
            ('link_axes', link_axes),
        ]:
            if axis_val is not None and axis_val >= ndim:
                raise IndexError(
                    f"{name}: {axis_name}={axis_val} out of bounds for ndim={ndim}"
                )

    # Dimension count consistency: all arrays must have the same ndim.
    # Sizes of non-special axes may differ (broadcasting handles the division).
    ref_ndim = data_3pt_sample.ndim
    for name, arr in arrays.items():
        if arr.ndim != ref_ndim:
            raise ValueError(
                f"Dimension count mismatch: data_3pt_sample has ndim={ref_ndim}, "
                f"{name} has ndim={arr.ndim}. All arrays must have the same number of dimensions."
            )


def _fold_link(data, link_axes):
    """Fold the link dimension: (upper_half + reversed(lower_half)) / 2.

    Mirrors lqcddb.analyse.PDF's link folding logic. Requires the link
    dimension to have an odd number of entries so that both halves align.
    """
    link_max = data.shape[link_axes] // 2

    idx_upper = [slice(None)] * data.ndim
    idx_upper[link_axes] = [x for x in range(link_max, 2 * link_max + 1)]
    upper = data[tuple(idx_upper)]

    idx_lower = [slice(None)] * data.ndim
    idx_lower[link_axes] = [x for x in range(0, link_max + 1)][::-1]
    lower_rev = data[tuple(idx_lower)]

    return (upper + lower_rev) / 2

def ratio_3pt(data_3pt_sample, data_2ptI_sample, data_2ptF_sample=None,
              t_sep=12, Nconf_axes=0,
              tau_axes=-1, t_sink_axes=-1,
              t_src_axes=None,
              link_axes=None, link_fold=False):
    """
    计算三点函数与外态的比值。
        R = C₃ / C₂^F(t_sep) × √[C₂^I(t_sep-τ) C₂^F(τ) C₂^F(t_sep) / (C₂^F(t_sep-τ) C₂^I(τ) C₂^I(t_sep))]

    支持两种模式：
      - 一维模式（t_src_axes=None）：每个数组只有一个时间轴，三点函数用 tau_axes，
        两点函数用 t_sink_axes，二者可以位于不同位置。
      - 二维模式（t_src_axes 不为 None）：所有数组共享源时间轴 t_src_axes，
        三点函数另有 tau_axes，两点函数另有 t_sink_axes。

    Parameters
    ----------
    data_3pt_sample : ndarray
        三点 Jackknife 样本，C₃。
    data_2ptI_sample : ndarray
        初态两点 Jackknife 样本，C₂^I。
    data_2ptF_sample : ndarray, optional
        末态两点 Jackknife 样本，C₂^F。若为 None 则使用 data_2ptI_sample。
    t_sep : int
        固定的源-汇时间间隔。
    Nconf_axes : int
        Jackknife 样本所在的轴。
    tau_axes : int
        data_3pt_sample 中算子插入时间 τ 所在的轴。
    t_sink_axes : int
        data_2ptI_sample 和 data_2ptF_sample 中汇时间所在的轴。
    t_src_axes : int, optional
        源时间所在的轴。提供此参数时启用二维模式：
        data_3pt_sample 具有 (t_src_axes, tau_axes) 两个时间维度；
        两点数据具有 (t_src_axes, t_sink_axes) 两个时间维度。
    link_axes : int, optional
        link 插入方向所在的轴，用于折叠。
    link_fold : bool
        若为 True，在计算比值前先对 link 轴做折叠。

    Returns
    -------
    dict，键为 data_sample, data_mean, data_err
    """
    backend = get_backend()

    if data_2ptF_sample is None:
        data_2ptF_sample = data_2ptI_sample
        
    # --- Dimension validation ---
    _validate_axes(data_3pt_sample, data_2ptI_sample, data_2ptF_sample,
                   Nconf_axes, tau_axes, t_sink_axes, t_src_axes, link_axes)

    # --- Link folding ---
    if link_fold and link_axes is not None:
        data_3pt_sample = _fold_link(data_3pt_sample, link_axes)
        data_2ptI_sample = _fold_link(data_2ptI_sample, link_axes)
        data_2ptF_sample = _fold_link(data_2ptF_sample, link_axes)

    Nconf = data_3pt_sample.shape[Nconf_axes]

    if t_src_axes is None:
        # 1D mode: single time axis per array
        Ntau = data_3pt_sample.shape[tau_axes]

        # t_sep - τ: periodic wrapping via negative indices
        idx_tshift = t_sep - backend.arange(Ntau)

        C2F_tsep_minus_tau = ArraySlicer(data_2ptF_sample).slice(dims=[t_sink_axes], indices=[list(idx_tshift)])
        C2I_tsep_minus_tau = ArraySlicer(data_2ptI_sample).slice(dims=[t_sink_axes], indices=[list(idx_tshift)])

        # C2(t_sep): fixed-time scalar, expand to match 3pt dimensions
        C2F_tsep = data_2ptF_sample.take(t_sep, axis=t_sink_axes)
        C2F_tsep = backend.expand_dims(C2F_tsep, axis=tau_axes)

        C2I_tsep = data_2ptI_sample.take(t_sep, axis=t_sink_axes)
        C2I_tsep = backend.expand_dims(C2I_tsep, axis=tau_axes)

        # sqrt term
        num = C2I_tsep_minus_tau * data_2ptF_sample * C2F_tsep
        den = C2F_tsep_minus_tau * data_2ptI_sample * C2I_tsep
        with backend.errstate(invalid='ignore', divide='ignore'):
            sqrt_term = backend.sqrt(backend.maximum(backend.nan_to_num(num / den, nan=0.0), 0))

        # Ratio
        ratio_sample = data_3pt_sample / C2F_tsep * sqrt_term

    else:
        # 2D mode: source time axis + τ/sink axis
        Nt_src = data_3pt_sample.shape[t_src_axes]
        Ntau = data_3pt_sample.shape[tau_axes]

        # After slicing along t_src_axes, axes >= t_src_axes shift down by 1.
        # Convert to positive indices first — the comparison is incorrect for
        # negative indices (e.g. -1 > -2 is True, but -1-1 = -2 would point
        # to the wrong axis in the sliced array).
        ndim = data_3pt_sample.ndim
        _src_pos = t_src_axes if t_src_axes >= 0 else ndim + t_src_axes
        _sink_pos = t_sink_axes if t_sink_axes >= 0 else ndim + t_sink_axes
        _tau_pos = tau_axes if tau_axes >= 0 else ndim + tau_axes

        _sink = (_sink_pos - 1) if _sink_pos > _src_pos else _sink_pos
        _tau  = (_tau_pos - 1) if _tau_pos > _src_pos else _tau_pos

        ratio_sample = backend.zeros_like(data_3pt_sample)

        for t_src in range(Nt_src):
            t_sep_time = (t_src + t_sep) % Nt_src

            # τ-dependent sink indices for this t_src
            t_ops = (t_src + backend.arange(Ntau)) % Nt_src
            t_diffs = (t_src + t_sep - backend.arange(Ntau)) % Nt_src

            # --- C₃ for this t_src ---
            C3_slice = data_3pt_sample.take(t_src, axis=t_src_axes)
            # Re-index C₃ by relative τ = tcurr - tsrc (mod Nt) so the
            # tau_axes position k corresponds to τ = k, matching the 2pt
            # terms in the sqrt below (which are already τ-indexed via t_ops).
            C3_by_tau = ArraySlicer(C3_slice).slice(dims=[_tau], indices=[list(t_ops)])

            # --- C₂^F terms (slice t_src, then index sink by scalar or array) ---
            C2F_tsep = data_2ptF_sample.take(t_src, axis=t_src_axes)
            C2F_tsep = C2F_tsep.take(t_sep_time, axis=_sink)
            C2F_tsep = backend.expand_dims(C2F_tsep, axis=_tau)

            C2F_tau_src = data_2ptF_sample.take(t_src, axis=t_src_axes)
            C2F_tau = ArraySlicer(C2F_tau_src).slice(dims=[_sink], indices=[list(t_ops)])

            C2F_tshift_src = data_2ptF_sample.take(t_src, axis=t_src_axes)
            C2F_tshift = ArraySlicer(C2F_tshift_src).slice(dims=[_sink], indices=[list(t_diffs)])

            # --- C₂^I terms ---
            C2I_tsep = data_2ptI_sample.take(t_src, axis=t_src_axes)
            C2I_tsep = C2I_tsep.take(t_sep_time, axis=_sink)
            C2I_tsep = backend.expand_dims(C2I_tsep, axis=_tau)

            C2I_tau_src = data_2ptI_sample.take(t_src, axis=t_src_axes)
            C2I_tau = ArraySlicer(C2I_tau_src).slice(dims=[_sink], indices=[list(t_ops)])

            C2I_tshift_src = data_2ptI_sample.take(t_src, axis=t_src_axes)
            C2I_tshift = ArraySlicer(C2I_tshift_src).slice(dims=[_sink], indices=[list(t_diffs)])

            # sqrt term
            num = C2I_tshift * C2F_tau * C2F_tsep
            den = C2F_tshift * C2I_tau * C2I_tsep
            # nan_to_num handles 0/0 → 0 for τ outside [0, t_sep] where
            # both 2pt functions are zero; within the valid range num/den
            # is well-behaved (all factors non-zero).
            with backend.errstate(invalid='ignore', divide='ignore'):
                sqrt_term = backend.sqrt(backend.maximum(backend.nan_to_num(num / den, nan=0.0), 0))

            # Ratio for this t_src
            ratio_slice = C3_by_tau / C2F_tsep * sqrt_term

            # --- Map back to absolute t_curr for output ---
            # ratio_slice is indexed by relative τ; the output tau_axes must
            # preserve the input convention (absolute t_curr).  The inverse
            # mapping: position k → τ = (k - t_src) % Nt.
            t_ops_inv = (backend.arange(Ntau) - t_src) % Nt_src
            ratio_slice_abs = ArraySlicer(ratio_slice).slice(
                dims=[_tau], indices=[list(t_ops_inv)])

            idx_assign = [slice(None)] * ratio_sample.ndim
            idx_assign[t_src_axes] = t_src
            ratio_sample[tuple(idx_assign)] = ratio_slice_abs

    # --- Mean and error ---
    ratio_mean = backend.mean(ratio_sample.real, axis=Nconf_axes)
    ratio_err = backend.std(ratio_sample.real, axis=Nconf_axes) * backend.sqrt(Nconf - 1)

    return {
        'data_sample': ratio_sample,
        'data_mean': ratio_mean,
        'data_err': ratio_err,
    }

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
    C = ((C.conj().transpose(1, 0, 2) + C) / 2).real

    # 初始化输出数组
    C_GEVP = backend.zeros((N, Nt))
    C_eigenvectors = backend.zeros((N, N, Nt), dtype = float)

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


def dis_connect(data_2pt_sample, data_bubble_sample, Nconf_axes:int, t_src_axes:int, t_sink_axes:int, tsep:int, dtype:Literal['PFF', 'PDF'] = 'PDF'):
    backend = get_backend()
    
    data_2pt_bubble_matrix = backend.zeros_like(data_2pt_sample)
    _data_2pt_mu_nu = data_2pt_sample - backend.mean(data_2pt_sample, axis = Nconf_axes, keepdims = True)

    _data_bubble = backend.zeros_like(data_2pt_sample)
    data_bubble_mean = backend.mean(data_bubble_sample, axis = Nconf_axes, keepdims = True)
    Nt = data_2pt_bubble_matrix.shape[t_src_axes]
    
    for t in range(Nt):
        ArraySlicer(_data_2pt_mu_nu).assign(dims = [t_src_axes], indices = [[t]], values = backend.roll(ArraySlicer(_data_2pt_mu_nu).slice(dims = [t_src_axes], indices = [[t]]), axis = t_sink_axes, shift = -t))
        ArraySlicer(_data_bubble).assign(dims = [t_src_axes], indices = [[t]], values = backend.roll(data_bubble_sample - data_bubble_mean, axis = t_src_axes, shift = -t))
    
    data_2pt_bubble_1 = ArraySlicer(_data_2pt_mu_nu).slice(dims = [t_sink_axes], indices = [[tsep]]) * _data_bubble
    
    if dtype == 'PFF':
        data_2pt_bubble_2 = ArraySlicer(_data_bubble).slice(dims = [t_sink_axes], indices = [[tsep]]) * _data_2pt_mu_nu
        ArraySlicer(data_2pt_bubble_matrix).assign(dims = [t_sink_axes], indices = [[x for x in range(tsep + 1)]], values = ArraySlicer(data_2pt_bubble_1).slice(dims = [t_sink_axes], indices = [[x for x in range(tsep + 1)]]))
        ArraySlicer(data_2pt_bubble_matrix).assign(dims = [t_sink_axes], indices = [[x for x in range(tsep, 2 * tsep + 1, 1)]], values = ArraySlicer(data_2pt_bubble_2).slice(dims = [t_sink_axes], indices = [[x for x in range(tsep, 2 * tsep + 1, 1)]]))
        
    else:
        data_2pt_bubble_matrix[:] = data_2pt_bubble_1

    return backend.sum(data_2pt_bubble_matrix, axis = t_src_axes, keepdims = True)