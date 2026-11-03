import numpy as np
import itertools


def levi_civita_tensor(n:int = 3):
    """
    生成 n 维空间中的 Levi-Civita 张量
    
    参数:
        n (int): 张量的阶数（也是空间的维度）
        
    返回:
        np.ndarray: n 阶张量，形状为 (n, n, ..., n) [共 n 个维度]
    """
    # 创建 n×n×...×n 的零张量
    from scipy.linalg import det

    shape = tuple([n] * n)
    epsilon = np.zeros(shape, dtype = float)
    
    # 生成所有可能的索引排列
    for perm in itertools.permutations(range(n)):
        # 检查是否包含重复索引
        if len(set(perm)) == n:
            # 创建排列矩阵
            perm_matrix = np.eye(n)[list(perm)]
            
            # 计算行列式作为符号
            sign = det(perm_matrix)
            
            # 四舍五入避免浮点误差
            sign = np.round(sign).astype(int)
            
            # 赋值
            epsilon[perm] = sign
            
    return epsilon

def creat_mom_list(Mom:list = [0, 0, 0], fix_Q2:bool = False, only_g0:bool = False):
    def add_negative_signs(lst):
    # 找到所有非零元素的索引
        nonzero_indices = [i for i, val in enumerate(lst) if val != 0]

        result = []
        # 为每个非零元素生成所有可能的正负号组合
        for signs in itertools.product([1, -1], repeat=len(nonzero_indices)):
            new_list = lst.copy()
            for idx, sign in zip(nonzero_indices, signs):
                new_list[idx] = lst[idx] * sign
            result.append(new_list)
    
        return result

    if 'array' in str(type(Mom)):
        Mom = Mom.tolist()

    if type(Mom[0]) == list:
        _num = len(Mom)

    else:
        _num = 1
        Mom = [Mom]
        
    Mom_list_all = []

    for i in range(_num):
        _Mom = Mom[i]

        min_Mom = min(_Mom)
        max_Mom = max(_Mom)

        len_Mom = max_Mom - min_Mom + 1

        Q2 = sum([x**2 for x in _Mom])

        
        coeff = [-1, 1]

        for i in range((len_Mom) ** 3):
            Mom_list = [(i//((len_Mom)**2))%len_Mom + min_Mom, (i//((len_Mom)**1))%len_Mom + min_Mom, (i//((len_Mom)**0))%len_Mom + min_Mom]
            Mom_list = add_negative_signs(Mom_list)

            if fix_Q2:
                for Mom_list_indx, Mom_list_one in enumerate(Mom_list):
                    indx = []

                    if Q2 == sum([x**2 for x in Mom_list_one]):
                        indx += [Mom_list_indx]

                    Mom_list_all += [Mom_list[x] for x in indx]

            else:
                Mom_list_all += Mom_list
                
    if only_g0:
        return sorted([x for x in Mom_list_all if all([y>=0 for y in x])])
    
    else:
        return sorted(Mom_list_all)

class ArraySlicer:
    def __init__(self, array):
        """
        Advanced array slicing and assignment utility class
        Uses np.ix_ to build indexing grids
        
        Initialize

        Args:
            array: numpy array
        """
        self.array = array
        self.ndim = array.ndim
        self.shape = array.shape

    def get_slices(self, dims, indices):
        """
        Build slice object based on specified dimensions and indices

        Args:
            dims: list of dimensions
            indices: list of indices corresponding to dimensions

        Returns:
            tuple: slice object
        """
        arr_dims = self.ndim
        arr_shape = self.shape
        
        if len(dims) != len(indices):
            raise ValueError("Dimension and index lists must have the same length")
        
        # 初始化每个维度的索引范围
        slices = [list(range(x)) for x in arr_shape]
        
        for dim, idx in zip(dims, indices):
            if dim >= arr_dims:
                raise ValueError(f"Dimension {dim} exceeds array dimension range (0-{arr_dims-1})")
            
            if isinstance(idx, (list, np.ndarray)):
                # 如果是列表或数组，直接使用
                slices[dim] = idx
            elif isinstance(idx, int):
                # 单个索引
                slices[dim] = [idx]
            elif idx == slice(None):
                # 保持原样
                slices[dim] = slices[dim]
            elif isinstance(idx, range):
                slices[dim] = list(idx)
            elif isinstance(idx, slice):
                # 将slice转换为range
                start = idx.start if idx.start is not None else 0
                stop = idx.stop if idx.stop is not None else arr_shape[dim]
                step = idx.step if idx.step is not None else 1
                slices[dim] = list(range(start, stop, step))
            else:
                raise ValueError(f"Unsupported index type: {type(idx)}")
        
        return np.ix_(*slices)

    def slice(self, dims, indices):
        """
        Slice extraction

        Args:
            dims: list of dimensions
            indices: list of indices corresponding to dimensions

        Returns:
            ndarray: sliced array
        """
        slices = self.get_slices(dims, indices)
        return self.array[slices]
    
    def assign(self, dims, indices, values, keep_dims:list = []):
        """
        Assignment operation

        Args:
            dims: list of dimensions
            indices: list of indices corresponding to dimensions
            values: values to assign

        Returns:
            ndarray: array after assignment (modified in-place)
        """
        slices = self.get_slices(dims, indices)
        if isinstance(values, int):
            self.array[slices] = values

        else:
            if keep_dims == []:
                self.array[slices] = values.reshape(self.array[slices].shape)
            
            else:
                self.array[slices] = values.reshape([x if x_indx in keep_dims or (x_indx - self.ndim) in keep_dims else 1 for x_indx, x in enumerate(self.array[slices].shape)])

        return self.array
    
    def get_slice_shape(self, dims, indices):
        """
        Get shape after slicing

        Args:
            dims: list of dimensions
            indices: list of indices corresponding to dimensions

        Returns:
            tuple: shape after slicing
        """
        slices = self.get_slices(dims, indices)
        
        # 创建一个虚拟数组来计算切片后的形状
        dummy_array = np.zeros(self.shape)
        sliced = dummy_array[slices]
        
        return sliced.shape
    
    def get_info(self):
        """Get array information"""
        return {
            'shape': self.array.shape,
            'ndim': self.array.ndim,
            'dtype': self.array.dtype
        }

"""
带自动缓存的 opt_einsum 收缩函数。

首次调用时根据张量形状编译收缩路径并缓存，
后续相同表达式、相同形状、相同优化策略的调用直接复用缓存。
"""

from typing import Any, Dict, Tuple, List, Union

# ============================================================
# 模块级缓存
# ============================================================
_expr_cache: Dict[Tuple, Any] = {}
_AUTO_OPTIMIZERS = ('auto', 'greedy', 'optimal', 'dp')


def _validate_einsum_shapes(einsum_str: str, shapes: Tuple[Tuple[int, ...], ...]):
    """
    校验 einsum 表达式与张量形状是否兼容，提前发现维度不匹配。

    在调用 opt_einsum 之前进行，避免 opt_einsum 只报告第一个不匹配的索引，
    这里会一次性检查所有索引并给出中文描述。
    """
    # ---- 解析表达式 ----
    if '->' in einsum_str:
        inputs_str = einsum_str.split('->')[0]
    else:
        inputs_str = einsum_str

    input_groups = inputs_str.split(',')

    # ---- 检查 1：输入张量数量 ----
    if len(input_groups) != len(shapes):
        raise ValueError(
            f"收缩表达式包含 {len(input_groups)} 个输入张量 "
            f"（{inputs_str}），但实际提供了 {len(shapes)} 个张量。"
        )

    # ---- 检查 2：每个张量的索引数（维度数） ----
    # 注意：'...' (ellipsis) 是单个记法，可代表任意数量的维度，不能按字符数计算
    for i, (group, shape) in enumerate(zip(input_groups, shapes)):
        if '...' in group:
            num_explicit = len(group) - 3   # 减去 '...' 的 3 个字符
            if num_explicit > len(shape):
                raise ValueError(
                    f"第 {i+1} 个变量的索引字符串 '{group}' 包含 {num_explicit} 个显式索引，"
                    f"但其形状 {shape} 只有 {len(shape)} 个维度，"
                    f"至少需要 {num_explicit} 个维度来承载显式索引。"
                )
        else:
            if len(group) != len(shape):
                raise ValueError(
                    f"第 {i+1} 个变量的索引字符串 '{group}' 包含 {len(group)} 个索引，"
                    f"但其形状 {shape} 只有 {len(shape)} 个维度，数量不匹配。"
                )

    # ---- 检查 3：同一索引在不同张量中的维度大小是否一致 ----
    # 构建映射：索引标签 → [(张量编号, 维度位置, 实际大小), ...]
    #
    # 对于 ellipsis，将其按位置展开为 _ellipsis_0, _ellipsis_1, …，
    # 确保对应位置的 ellipsis 维度在不同张量间大小一致。
    index_map: Dict[str, List[Tuple[int, int, int]]] = {}
    for i, (group, shape) in enumerate(zip(input_groups, shapes)):
        if '...' in group:
            parts = group.split('...')
            prefix = parts[0]
            suffix = parts[1] if len(parts) > 1 else ''
            n_ellipsis = len(shape) - len(prefix) - len(suffix)

            # 前缀索引
            for j, label in enumerate(prefix):
                index_map.setdefault(label, []).append((i, j, shape[j]))

            # 省略号维度（用 _ellipsis_N 标记位置）
            for k in range(n_ellipsis):
                pos = len(prefix) + k
                label = f'_ellipsis_{k}'
                index_map.setdefault(label, []).append((i, pos, shape[pos]))

            # 后缀索引
            suffix_start = len(shape) - len(suffix)
            for j, label in enumerate(suffix):
                pos = suffix_start + j
                index_map.setdefault(label, []).append((i, pos, shape[pos]))
        else:
            for j, (label, size) in enumerate(zip(group, shape)):
                index_map.setdefault(label, []).append((i, j, size))

    errors: List[str] = []
    for label, occurrences in index_map.items():
        # 内部记号 _ellipsis_N 对外显示为 '...'
        label_display = '...' if label.startswith('_ellipsis_') else label

        # numpy einsum 广播规则：同一索引的所有非 1 大小必须一致
        # 例如 [1, 10, 1, 10] → 合法（非 1 值均为 10）
        # 例如 [1, 10, 1, 5]  → 非法（10 ≠ 5）
        non_one_sizes = set(size for _, _, size in occurrences
                            if size != 1)
        if len(non_one_sizes) > 1:
            # 找出每种大小的第一个出现位置，构造错误信息
            by_size: Dict[int, Tuple[int, int]] = {}
            for tensor_idx, dim_idx, size in occurrences:
                if size != 1 and size not in by_size:
                    by_size[size] = (tensor_idx, dim_idx)

            lines = [f"索引 '{label_display}' 在不同张量中的非 1 大小不一致："]
            for sz, (ti, di) in sorted(by_size.items()):
                lines.append(
                    f"  - 第 {ti+1} 个变量的第 {di+1} 个维度，大小为 {sz}"
                )
            lines.append("这些维度无法通过广播匹配（广播要求所有非 1 大小相同）。")
            errors.append("\n".join(lines))

    if errors:
        raise ValueError(
            "收缩表达式与张量形状不匹配：\n" + "\n".join(errors)
        )


def cached_contract(
    einsum_str: str,
    *tensors,
    optimize: Union[str, bool, List[str]] = 'auto'
):
    """
    执行带缓存的张量收缩，支持自动择优。

    Parameters
    ----------
    einsum_str : str
        收缩字符串，例如 'ab,bc->ac'。
    tensors : array_like
        参与收缩的张量。
    optimize : str, bool, or list of str
        - 字符串：直接使用该策略。
        - True：自动尝试预定义策略列表，选择代价最低者。
        - 字符串列表：手动指定候选策略列表。

    Returns
    -------
    result : array_like
        收缩结果。
    """
    from opt_einsum import contract_expression, contract_path

    shapes = tuple(t.shape for t in tensors)

    # ---- 解析 optimize → opt_key（热路径，只为构造缓存键） ----
    if isinstance(optimize, str):
        opt_key = optimize
    elif optimize is True:
        opt_key = _AUTO_OPTIMIZERS
    elif isinstance(optimize, list):
        opt_key = tuple(optimize)
    else:
        raise TypeError(
            f"optimize 参数类型应为 str, bool 或 list，收到 {type(optimize)}"
        )

    key = (einsum_str, shapes, opt_key)
    expr = _expr_cache.get(key)
    if expr is not None:
        return expr(*tensors)

    # ================================================================
    # 缓存未命中 —— 以下为冷路径，仅首次调用时执行
    # ================================================================

    # ---- 解析 candidate_opts（冷路径） ----
    if isinstance(optimize, str):
        candidate_opts = [optimize]
    elif optimize is True:
        candidate_opts = list(_AUTO_OPTIMIZERS)
    else:
        candidate_opts = optimize  # list

    # ---- 提前校验形状，给出具体诊断信息 ----
    _validate_einsum_shapes(einsum_str, shapes)

    # 创建零内存广播视图作为占位符
    placeholders = [
        np.broadcast_to(np.empty((), dtype=t.dtype), t.shape)
        for t in tensors
    ]

    best_path = None
    best_cost = (float('inf'), float('inf'))  # (flops, size)
    
    for opt in candidate_opts:
        try:
            path, path_info = contract_path(
                einsum_str, *placeholders, optimize=opt
            )
            # 直接使用 path_info.opt_cost
            flops = path_info.opt_cost
            size = path_info.largest_intermediate
            cost = (flops, size)
            if cost < best_cost:   # 先比较 flops，再比较 size
                best_cost = cost
                best_path = path
                
        except Exception:
            continue

    if best_path is None:
        tried = ", ".join(candidate_opts)
        raise RuntimeError(
            f"尝试了以下优化策略 [{tried}]，均无法为给定表达式和形状"
            f"找到可行的收缩路径。表达式: {einsum_str}，形状: {shapes}"
        )

    expr = contract_expression(einsum_str, *shapes, optimize=best_path)
    _expr_cache[key] = expr
    return expr(*tensors)

def clear_cache():
    """清空全局收缩路径缓存，释放内存。"""
    _expr_cache.clear()


def get_cache_keys():
    """返回当前 :func:`cached_contract` 缓存中所有键的列表。

    每个键为 ``(einsum_str, shapes, opt_key)`` 三元组：

    - ``einsum_str``: 收缩字符串，如 ``'ab,bc->ac'``
    - ``shapes``: 参与收缩张量的形状元组
    - ``opt_key``: 优化策略标识（字符串或策略元组）

    Returns
    -------
    list of tuple
        缓存键列表，按插入顺序排列。
    """
    return list(_expr_cache.keys())