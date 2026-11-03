from typing import List,Literal,Tuple
import numpy as np
from pandas import unique
from collections import Counter
import itertools

def list_roll(arr, shift):
    """
    实现类似NumPy的roll函数
    arr: 输入列表
    shift: 移动的位数，正数向右移动，负数向左移动
    """
    if not arr:  # 处理空列表
        return arr
    
    # 处理位移大于列表长度的情况
    shift = shift % len(arr)
    
    # 使用切片实现循环移动
    return arr[-shift:] + arr[:-shift]

def fill_unequal_sequential(arr):
    """
    将多维数组中不相等的数据按照顺序填充为连续整数
    
    参数:
        arr: 输入的多维NumPy数组
    
    返回:
        处理后的数组，相等数据保持不变，不等数据按顺序填充
    """
    # 获取数组形状
    original_shape = arr.shape
    
    # 展平数组以便处理
    flat_arr = arr.flatten()
    
    # 获取唯一值和它们的索引
    unique_vals, inverse_indices = np.unique(flat_arr, return_inverse=True)

    # 创建一个新的序列，相等数据保持不变，不等数据按顺序填充
    # 首先找到所有需要保持的值（出现多次的值）
    counts = np.bincount(inverse_indices)
    keep_values = unique_vals[counts > 1]

    # 创建映射：对于需要保持的值，映射到自身；对于需要替换的值，映射到新序列
    new_values = np.arange(len(flat_arr), len(flat_arr) - len(unique_vals), -1)
    
    # 应用映射
    result_flat = np.where(np.isin(flat_arr, keep_values), flat_arr, new_values[inverse_indices])
    
    # 恢复原始形状
    return result_flat.reshape(original_shape)

def contraction_index(contraction_same_indx:List[List[int]] = [[0], [0]], dims:List[int] = [0, 0], shapes:List[Tuple] = [(0), (0)], name:List[str] = [['']]):
    contraction_indx_arr = np.asarray(list('MGABCDEFHIJKLNOPQRST'))

    '''
    创建相同与不相同指标 contraction 指标的字母索引

    参数：
        contraction_same_indx: 输入相同指标的位置
        dims: 输入数组的维度
        shapes: 输入数组维度对应的的大小
        name: 输入数组维度对应的含义
    
    返回：
        按照数组顺序输出对应的字母指标, 组合后的字母指标, 以及收缩后的形状
    '''
    
    if len(contraction_same_indx) != len(dims):
        raise ValueError("please set the params 'contraction_same_indx' and 'dims' have the same dims")
    
    str_indx = [''] * len(contraction_same_indx)
    
    lengths = [len(x) for x in contraction_same_indx]

    max_dims = max(dims)
    max_len_indx = lengths.index(max(lengths))
    max_contraction = max(max(contraction_same_indx))

    all_shapes = []
    all_name = []
    for i_indx, i in enumerate(shapes):
        for j_indx, j in enumerate(i):
            if dims[i_indx] != 0:
                all_shapes += [shapes[i_indx][j_indx]]

                if name != [['']]:
                    all_name += [name[i_indx][j_indx]]

    _sum_dims = sum(dims[:max_len_indx])
    
    all_shapes = list_roll(all_shapes, shift = -_sum_dims)

    if name != [['']]:
        all_name = list_roll(all_name, shift = -_sum_dims)

    contraction_same_indx = list_roll(contraction_same_indx, shift = -max_len_indx)
    dims = list_roll(dims, shift = -max_len_indx)

    shape_indx = []

    if max_dims > 0:
        if max_contraction >= max_dims:
            raise ValueError("the contraction_same_indx's value is larger than dims")
        
        indx = np.arange((len(contraction_same_indx) * max_dims), dtype = int).reshape(len(contraction_same_indx), max_dims)
        indx
        for i_indx, i in enumerate(contraction_same_indx):
            for j_indx, j in enumerate(i): 
                if j >= 0:
                    indx[i_indx, j] = contraction_same_indx[0][j_indx]
        
        _indx = fill_unequal_sequential(indx)
        for i in range(len(contraction_same_indx)):
            str_indx[i] = ''.join(contraction_indx_arr[_indx[i, :dims[i]]])
            shape_indx += [int(x) for x in indx[i, :dims[i]]]
    
        for i_indx, i in enumerate(shape_indx):
            if shape_indx[i_indx] > max_contraction:
                shape_indx[i_indx] = i_indx

        str_indx = list_roll(str_indx, shift = max_len_indx)
        shape_indx = list_roll(shape_indx, shift = _sum_dims)
    
    if name == [['']]:
        return *str_indx, ''.join(dict.fromkeys(''.join(str_indx))), [all_shapes[x] for x in unique(np.asarray(shape_indx))]
    
    else:
        return *str_indx, ''.join(dict.fromkeys(''.join(str_indx))), [all_shapes[x] for x in unique(np.asarray(shape_indx))], [all_name[x] for x in unique(np.asarray(shape_indx))]
    
def wick_contraction(
        sink_operators: List[str],
        source_operators: List[str],
        curr_operators: List[str],
        Cpt: Literal['bubble', '2pt', '3pt', '4pt'] = '2pt',
        Pindex:list = [], Vindex:list = [], Gindex:list = [],
        ):
    """
    Auto Wick contraction for N particles operators separated by ``'|'``.

    **Result** — a dict containing the following keys:

    ``result_indx``
        Contraction index.
    ``result_name``
        Corresponding parameter names for each indicator (e.g., ``peram``, ``VVV``, …).
    ``result_sign``
        Sign of each diagram.
    ``operators``
        Concatenation of ``sink_operators`` + ``curr_operators`` + ``source_operators``.
    ``sink_operators``
        Sink operator list (see below).
    ``source_operators``
        Source operator list.
    ``curr_operators``
        Current operator list.
    ``quark_pos``
        Tuple of ``(position in operator, quark flavour, quark index:(spin eigen))``.
    ``sep_pos``
        Position of the ``'|'`` separator.
    ``gamma_pos``
        Tuple of ``(position in operator, gamma matrix name, index, time index)``.
    ``V``
        Tuple of ``((left divider position, right divider position), name, index, time)``.
    """
    quark_list = (['u', 'd', 's', 'c', 'b', 't']
                  + [x + '^d' for x in ['u', 'd', 's', 'c', 'b', 't']])
    contraction_index = ['ab', 'cd', 'ef', 'mn', 'op', 'gh', 'ij', 'kl', 'qr', 'st', 'uv', 'wx', 'yz',
                         'AB', 'CD', 'EF', 'MN', 'OP', 'GH', 'IJ', 'KL', 'QR', 'ST', 'UV', 'WX', 'YZ']

    def creat_sink_source(param, Vindex):
        """输出VVV and VDV"""
        quark_pos = param['quark_pos']
        sep_pos = param['sep_pos']
        sorted_sep = sorted(sep_pos, reverse=False)
        pairs = [(sorted_sep[i], sorted_sep[i + 1]) for i in range(0, len(sorted_sep), 2)]
        intervals = [(min(pair), max(pair)) for pair in pairs]

        # 为每个区间分配时间标签
        time_labels = ['tsink'] + [f'tcur{i}' for i in range(n_intervals-2)] + ['tsrc']

        result = []
        for i_indx, (low, high) in enumerate(intervals):
            candidates = []
            for idx, (pos, typ, label) in enumerate(quark_pos):
                if low <= pos <= high:
                    second_letter = label[1]
                    has_carat = '^d' in typ
                    candidates.append((idx, second_letter, has_carat))
            candidates.sort(key=lambda x: (x[2], x[0]))
            sorted_letters = [letter for _, letter, _ in candidates]
            if len(sorted_letters) == 3:
                result.append([(low, high), f'VVV_{i_indx}',
                               Vindex[i_indx] + ''.join(sorted_letters), time_labels[i_indx]])
            elif len(sorted_letters) == 2:
                result.append([(low, high), f'VDV_{i_indx}',
                               Vindex[i_indx] + ''.join(sorted_letters[::-1]), time_labels[i_indx]])
        return result

    def keep_unique_letters(s):
        """计算剩余收缩指标"""
        counts_lower = Counter(ch for ch in s if ch.islower())
        counts_upper = Counter(ch for ch in s if ch.isupper())

        result_lower = []
        result_upper = []

        s_no_comma = s.replace(',', '')
        for ch in s_no_comma:
            if ch.islower():
                if counts_lower[ch] == 1:
                    result_lower.append(ch)
            else:
                if counts_upper[ch] >= 1 and ch not in result_upper:
                    result_upper.append(ch)
        
        if len(result_lower) == 2:
            _result = [None] * len(result_lower)
            segments = s.split(',')

            for i_indx, i in enumerate(result_lower):
                _result[
                    [y_indx for x in segments for y_indx, y in enumerate(x) if y == i][0]
                ] = i
            _result = _result + result_upper
            
        else:
            _result = result_lower + result_upper
    
        return ''.join(_result)
    
    def add_sep_sign(operator):
        """增加算符前后的间隔符号 | """
        if operator:
            if operator[0] != '|':
                operator = ['|'] + operator
            if operator[-1] != '|':
                operator = operator + ['|']
        return operator

    def count_inversions(lst):
        """计算列表中的逆序数"""
        inversions = 0
        n = len(lst)
        for i in range(n):
            for j in range(i + 1, n):
                if lst[i] > lst[j]:
                    inversions += 1
        return inversions
    
    def compute_fermion_sign(contraction_pairs):
        """
        计算 Wick 收缩的 Fermi 子符号。

        核心思路：
        1. 将所有被收缩的场按其在算符字符串中的位置排序
        2. 按排序后的顺序排列收缩对索引
        3. 计算逆序数 → 符号 = (-1)^逆序数

        这正确实现了 Fermi 子反对易关系：
        每次交换两个 Fermi 子场产生一个负号。
        """
        # 收集所有被收缩场的 (算符位置, 收缩对编号)
        indexed_positions = []
        for pair_idx, pair in enumerate(contraction_pairs):
            quark_op_pos = pair[0]      # 夸克在算符序列中的位置
            antiq_op_pos = pair[1]      # 反夸克在算符序列中的位置
            indexed_positions.append((quark_op_pos, pair_idx))
            indexed_positions.append((antiq_op_pos, pair_idx))

        # ★ 关键：按算符位置排序（而非按夸克/反夸克交替排列）
        # indexed_positions.sort(key=lambda x: x[0])

        # 提取排序后的收缩对编号序列
        pair_indices = [x[0] for x in indexed_positions]

        # 符号 = (-1)^(逆序数)
        return (-1) ** count_inversions(pair_indices)
    
    def generate_valid_contractions(quark_pos):
        """
        按夸克味分组，为每味生成所有合法的一一配对（完美匹配），
        再取各味之间的笛卡尔积组合。

        对于 n 个夸克 + n 个反夸克（同味），完美匹配数 = n!
        """
        # 按味分组：区分夸克和反夸克
        flavor_quarks = {}    # 味 → 夸克列表 [(位置, 味, 标签), ...]
        flavor_antiquarks = {}  # 味 → 反夸克列表

        for pos, qtype, label in quark_pos:
            if '^d' in qtype:
                base_flavor = qtype.replace('^d', '')
                flavor_antiquarks.setdefault(base_flavor, []).append(
                    (pos, qtype, label))
            else:
                flavor_quarks.setdefault(qtype, []).append(
                    (pos, qtype, label))

        # 为每味生成所有完美匹配
        flavor_matchings = []
        for flavor in sorted(set(list(flavor_quarks.keys())
                                 + list(flavor_antiquarks.keys()))):
            qs = flavor_quarks.get(flavor, [])
            aqs = flavor_antiquarks.get(flavor, [])

            if len(qs) != len(aqs):
                raise ValueError(
                    f"味 '{flavor}' 的夸克数({len(qs)}) "
                    f"≠ 反夸克数({len(aqs)})，无法形成完整收缩")

            if not qs:
                continue

            # 所有反夸克的排列 → 每种排列定义一种一一配对
            matchings = []
            for perm in itertools.permutations(range(len(aqs))):
                matching = []
                for q_idx in range(len(qs)):
                    aq_idx = perm[q_idx]
                    matching.append((qs[q_idx], aqs[aq_idx]))
                matchings.append(matching)

            flavor_matchings.append(matchings)

        # 各味之间的笛卡尔积 → 完整收缩图
        if not flavor_matchings:
            return [[]], 1

        all_diagrams = []
        for combo in itertools.product(*flavor_matchings):
            diagram = []
            for flavor_matching in combo:
                diagram.extend(flavor_matching)
            all_diagrams.append(diagram)

        return all_diagrams, len(all_diagrams)


    if Cpt == '2pt':
        curr_operators = []

    sink_operators = add_sep_sign(sink_operators)
    source_operators = add_sep_sign(source_operators)
    curr_operators = add_sep_sign(curr_operators)
    # operators = source_operators + curr_operators + sink_operators
    operators = sink_operators + curr_operators + source_operators
    
    overall_sign = np.prod([x for x in operators if isinstance(x, (int, complex, float))])
    operators = [x for x in operators if isinstance(x, str)]

    allowed_types = (int, complex, float, str)
    if [item for item in operators if not isinstance(item, allowed_types)]:
        raise ValueError(
            f"this function only support type {allowed_types}, "
            f"and type {int, complex, float} just for coeff")

    num = Counter([x for x in operators if x in quark_list])

    if not all(
        (num[x] - num[y]) == 0
        for x in num for y in num if x + '^d' == y
    ):
        raise ValueError(
            f'the number of quark and bar quark are not equal {num}')
    param = {}
    param['operators'] = operators
    param['sink_operators'] = [x for x in sink_operators if isinstance(x, str)]
    param['source_operators'] = [x for x in source_operators if isinstance(x, str)]
    param['curr_operators'] = [x for x in curr_operators if isinstance(x, str)]

    param['quark_pos'] = []
    param['sep_pos'] = []
    param['gamma_pos'] = []

    for _param_indx, _param in enumerate(operators):
        if _param in quark_list:
            param['quark_pos'] += [(_param_indx, _param)]
        elif 'gamma' in _param:
            param['gamma_pos'] += [(_param_indx, _param)]
        elif '|' == _param:
            param['sep_pos'] += [_param_indx]
        else:
            print(f"Warning: '{_param}' unidentifiable, discarded.")

    for _param_indx, _param in enumerate(param['quark_pos']):
        param['quark_pos'][_param_indx] = (
            *_param, contraction_index[_param_indx]
            )

    # 生成所有合法收缩图
    all_diagrams, num_diag = generate_valid_contractions(param['quark_pos'])

    param['peram'] = []
    param['result_sign'] = []

    sorted_sep = sorted(param['sep_pos'])
    # 沿用 creat_sink_source 的区间定义：每两个连续的 sep 为一组（步长2）
    intervals = []
    for i in range(0, len(sorted_sep), 2):
        if i+1 < len(sorted_sep):
            intervals.append((sorted_sep[i]+1, sorted_sep[i+1]-1))

    n_intervals = len(intervals)

    # 除自旋和eigen额外的指标
    Pindex = Pindex + ['' for _ in range(n_intervals - len(Pindex))]
    Vindex = Vindex + ['' for _ in range(n_intervals - len(Vindex))]
    Gindex = Gindex + ['' for _ in range(n_intervals - len(Gindex))]

    # 根据区间数量与 Cpt 构造标签列表
    time_labels = ['tsink'] + [f'tcur{i}' for i in range(n_intervals-2)] + ['tsrc']

    # 构建位置 → 时间标签的映射
    pos_to_time = {}
    for idx, (start, end) in enumerate(intervals):
        label = time_labels[idx]
        for pos in range(start, end+1):
            pos_to_time[pos] = label

    # Gamma 矩阵处理
    result = []
    for cut_indx, cut in enumerate(param['gamma_pos']):
        pos = [cut[0] - 1, cut[0] + 1]
        letters = []
        for _param in param['quark_pos']:
            if _param[0] in pos:
                letters += [_param[2][0]]
        combined = Gindex[cut_indx] + ''.join(letters)
        result += [(*cut, combined, pos_to_time[cut[0]])]
    param['gamma_pos'] = result

    for diagram in all_diagrams:
        # diagram = [(夸克信息, 反夸克信息), ...]
        # 每个元素: ((夸克位置, 夸克味, 夸克标签), (反夸克位置, 反夸克味, 反夸克标签))
        combo_entry = []

        for quark_info, antiquark_info in diagram:
            q_pos, q_type, q_label = quark_info
            aq_pos, aq_type, aq_label = antiquark_info
            combined_type = q_type + aq_type
            combined_label = q_label[0] + aq_label[0] + q_label[1] + aq_label[1]
            t_q = pos_to_time.get(q_pos, '?')
            t_aq = pos_to_time.get(aq_pos, '?')
            combo_entry.append([q_pos, aq_pos, combined_type, combined_label, [t_q, t_aq]])

        param['peram'].append(combo_entry)
        sign = compute_fermion_sign(
            [(entry[0], entry[1]) for entry in combo_entry])

        param['result_sign'].append(sign * overall_sign)

    param['V'] = creat_sink_source(param=param, Vindex = Vindex)

    param['result_indx'] = []
    param['result_name'] = []

    for diag_indx in range(num_diag):
        _result_indx = ','.join(
            [x[3] for x in param['peram'][diag_indx]]          # combined_label
            + [x[2] for x in param['gamma_pos']]
            + [x[2] for x in param['V']]
        )
        param['result_indx'].append(
            [_result_indx + '->' + keep_unique_letters(_result_indx)])

        _result_name = ', '.join(
            [f'peram_{x[2][0]}' for x in param['peram'][diag_indx]]   # combined_type
            + [x[1] for x in param['gamma_pos']]
            + [x[1] for x in param['V']]
        )
        param['result_name'].append([_result_name])

    return param


from collections import defaultdict

def identify_equivalent_diagrams(*dicts):
    """
    将多个拥有相同 key 的 dict 整合。
    输入：任意数量的 dict
    输出：一个 dict，其 value 为所有输入 dict 对应值的列表
    """


    def find_connected_groups(four_dim_list):
        """
        参数: four_dim_list - 四维列表，形状 [num_groups, num_items, ...]
        返回: 列表的列表，每个子列表包含一个连通分量的所有组索引
        """
        n = len(four_dim_list)
        if n == 0:
            return []

        # 并查集
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        # 构建从“二维列表哈希”到“组索引列表”的映射
        key_to_groups = defaultdict(list)
        for group_idx, group in enumerate(four_dim_list):
            for item in group:  # item 是一个二维列表，如 [[1,7,'uu^u'], ...]
                # 将二维列表转换为可哈希的元组元组
                key = tuple(tuple(sublist) for sublist in item)
                key_to_groups[key].append(group_idx)

        # 合并相同二维列表对应的所有组
        for groups in key_to_groups.values():
            if len(groups) > 1:
                first = groups[0]
                for g in groups[1:]:
                    union(first, g)

        # 收集每个连通分量的组索引
        components = defaultdict(list)
        for i in range(n):
            components[find(i)].append(i)

        # 输出为排序后的列表的列表
        result = [sorted(indices) for indices in components.values()]
        result.sort(key=lambda x: x[0])  # 按第一个元素排序
        return result

    def generate_all_swaps(original, swap_pairs):
        """
        参数:
            original: 二维列表，如 [[1,7],[2,12],...]
            swap_pairs: 交换对列表，如 [[2,4],[7,9],...]
        返回:
            三维列表，每个元素是应用一组交换后得到的二维列表
        """
        n = len(swap_pairs)
        results = []
        
        for enabled in itertools.product([False, True], repeat=n):
            # 构建当前组合的映射字典
            mapping = {}
            for (a, b), do_swap in zip(swap_pairs, enabled):
                if do_swap:
                    mapping[a] = b
                    mapping[b] = a
            # 应用映射到原始列表
            transformed = [[mapping.get(x, x) for x in sub] for sub in original]
            results.append(transformed)
        
        return results

    data = []
    for _dicts in dicts:
        change_quark_pos = [[int(x[0]) - 1, int(x[0]) + 1] for x in _dicts['gamma_pos']]
        peram = [[y[:2] + [y[2].replace('d', 'u')] for y in x] for x in _dicts['peram']]
        data += [[sorted(y, key=lambda x: x[0]) for y in x] for x in [generate_all_swaps(x, change_quark_pos) for x in peram]]

    equivalent_diagrams = find_connected_groups(data)

    return equivalent_diagrams

import matplotlib.patches as patches
import matplotlib.pyplot as plt

def plot_figure_wick(result_dict, diagram_index=0, Cpt:Literal['2pt', '3pt', '4pt']='2pt', plot_text:Literal[True, False] = True):

    """
    **Wick contraction diagram** — fully general for 2pt/3pt/4pt+.
    All visual parameters scale automatically with diagram complexity.
    """

    # ── Pre-compute structure to determine scale ───────────────────
    quark_pos  = result_dict['quark_pos']
    gamma_pos  = result_dict['gamma_pos']
    V_info     = result_dict['V']
    sign       = result_dict['result_sign'][diagram_index]
    peram_list = result_dict['peram'][diagram_index]
    cur_diagram = result_dict['result_indx'][diagram_index][0]
    cur_name    = result_dict['result_name'][diagram_index][0]
    sep_pos     = result_dict['sep_pos']

    is_2pt = (Cpt == '2pt')

    # ── Region intervals ───────────────────────────────────────────
    sorted_sep = sorted(sep_pos)
    intervals = [(sorted_sep[i], sorted_sep[i + 1])
                 for i in range(0, len(sorted_sep), 2)]
    n_regions = len(intervals)

    num_source_intervals = len([0 for x in result_dict['source_operators'] if x == '|']) // 2
    num_sink_intervals   = len([0 for x in result_dict['sink_operators']   if x == '|']) // 2
    n_cur = max(0, n_regions - num_source_intervals - num_sink_intervals)

    # ── Classify quarks into regions ───────────────────────────────
    src_quarks = []
    snk_quarks = []
    cur_quarks = [[] for _ in range(n_cur)]

    for (idx, qtype, label) in quark_pos:
        for ri, (low, high) in enumerate(intervals):
            if low <= idx <= high:
                if ri < num_sink_intervals:
                    snk_quarks.append((idx, qtype, label))
                elif ri >= n_regions - num_source_intervals:
                    src_quarks.append((idx, qtype, label))
                else:
                    cur_quarks[ri - num_sink_intervals].append((idx, qtype, label))
                break

    src_quarks.sort(key=lambda x: x[0])
    snk_quarks.sort(key=lambda x: x[0])
    for cq in cur_quarks:
        cq.sort(key=lambda x: x[0])

    # ── Scale factor — derived from diagram complexity ─────────────
    n_quarks = len(quark_pos)
    # More quarks / more regions → smaller base elements, but bounded
    complexity = max(n_quarks, 6) * (1 + 0.15 * (n_regions - 1))
    scale = max(0.75, min(1.0, 8.0 / complexity))

    # All visual dimensions scale together
    # ── 自适应视觉参数（全部乘以 scale 以适配不同复杂度的图）──────────────

    # 空间布局
    RAD       = 0.42 * scale   # 夸克节点圆的半径
    DY        = 1.6  * scale   # 同一区域内相邻夸克的纵向间距
    PAD       = 0.8  * scale   # 区域背景矩形在圆外的内边距
    SHRINK    = RAD             # 箭头端点收缩量，等于圆半径（保证箭头落在圆边界上）

    # 线宽
    LW_NODE   = 2.2 * scale   # 夸克圆的描边线宽
    LW_ARROW  = 3.0 * scale   # 传播子箭头线宽
    LW_GAMMA  = 2.5 * scale   # Gamma 矩阵虚线线宽
    LW_SEP    = 2.0 * scale   # 束缚态粒子分隔虚线线宽
    LW_BG     = 1.2 * scale   # 区域背景矩形边框线宽
    LW_TIME   = 3.0 * scale   # 时间方向箭头线宽
    LW_LEGEND = 3.0 * scale   # 图例中线条线宽

    # 箭头头部
    MUT_SCALE = 22  * scale   # 箭头头部大小（mutation_scale）

    # 字号
    FS_TITLE  = 17  * scale   # 顶部标题字号
    FS_REGION = 14  * scale   # 区域标签（Sink/Source/Current）字号
    FS_LABEL  = 10  * scale   # 夸克编号标签（如 [ab] S 1）字号
    FS_NODE   = 14  * scale   # 圆内夸克字母字号（u, d, ū 等）
    FS_GAMMA  = 13  * scale   # Gamma 矩阵标签字号
    FS_VERTEX = 13  * scale   # 顶点标签字号
    FS_TIME   = 13  * scale   # 时间方向文字字号
    FS_LEGEND = 9.5 * scale   # 图例文字字号
    FS_LEG_T  = 11  * scale   # 图例标题字号
    FS_INFO   = 10  * scale   # 左下角信息文字字号

    # 偏移量
    TEXT_OFF   = 0.25 * scale  # 夸克编号标签相对于圆心上方的偏移量
    TEXT_ASC   = 0.20 * scale  # 文字上沿的额外高度（用于分隔线避让计算）
    TITLE_PAD  = 18  * scale   # 标题与图的间距
    LEGEND_OFF = 1.01          # 图例横向偏移（相对坐标，无需缩放）


    # ── Figure size scales with content ────────────────────────────
    fig_w = 25 * scale
    fig_h = (10 + 2 * n_quarks / 5) * scale
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── letter → (quark_type, operator_index) ──────────────────────
    letter_to_quark = {}
    for (idx, qtype, label) in quark_pos:
        for ch in label:
            letter_to_quark[ch] = (qtype, idx)

    def get_region(idx):
        for low, high in intervals:
            if low <= idx <= high:
                return (low, high)
        return None

    # ── Colour palette ─────────────────────────────────────────────
    QC = {
        'u': '#3498DB',  'd': '#E74C3C',  's': '#2ECC71',
        'c': '#9B59B6',  'b': '#F39C12',  't': '#1ABC9C',
        'u^d': '#2471A3', 'd^d': '#CB4335', 's^d': '#1E8449',
        'c^d': '#7D3C98', 'b^d': '#D4AC0D', 't^d': '#17A589',
    }

    # ── Layout — fully dynamic x positions ─────────────────────────
    REGION_GAP = 10.0 * scale

    if is_2pt:
        SNK_X = 3.5 * scale
        SRC_X = SNK_X + REGION_GAP
        CUR_XS = []
    else:
        SNK_X = 2.5 * scale
        SRC_X = SNK_X + REGION_GAP
        if n_cur == 0:
            CUR_XS = []
        elif n_cur == 1:
            CUR_XS = [(SNK_X + SRC_X) / 2]
        else:
            CUR_XS = [SNK_X + (i + 1) * REGION_GAP / (n_cur + 1)
                      for i in range(n_cur)]

    # ── y positions ────────────────────────────────────────────────
    src_ys = [i * DY for i in range(len(src_quarks))]
    snk_ys = [i * DY for i in range(len(snk_quarks))]
    cur_yss = [[i * DY for i in range(len(cq))] for cq in cur_quarks]

    base_h = max(
        (src_ys[-1] + RAD) if src_ys else 0,
        (snk_ys[-1] + RAD) if snk_ys else 0)

    if not is_2pt:
        for ci in range(n_cur):
            offset = base_h + 2.0 * scale + ci * (DY * 2.5)
            cur_yss[ci] = [offset + i * DY for i in range(len(cur_quarks[ci]))]

    all_ys = src_ys + snk_ys
    for cys in cur_yss:
        all_ys += cys
    total_h = max(all_ys) + RAD + 1.0 * scale if all_ys else 3.0 * scale

    ymin = -2.5 * scale
    ymax = total_h + 1.5 * scale
    ax.set_xlim(-1 * scale, SRC_X + 3.0 * scale)
    ax.set_ylim(ymin, ymax)

    # ── Region backgrounds ─────────────────────────────────────────
    region_colors = [('#E8F8F5', 'Sink')] + \
                    [('#FEF9E7', f'Current {i+1}') for i in range(n_cur)] + \
                    [('#FDEDEC', 'Source')]

    for ri, (qx, qlist) in enumerate(
            [(SNK_X, src_quarks)] +
            [(CUR_XS[i], cur_quarks[i]) for i in range(n_cur)] +
            [(SRC_X, snk_quarks)]):
        if not qlist:
            continue
        ys = [snk_ys, *cur_yss, src_ys][ri]
        top = ys[-1] + RAD + PAD
        bot = ys[0]  - RAD - PAD
        h = top - bot
        color, label = region_colors[ri]
        ax.add_patch(patches.FancyBboxPatch(
            (qx - RAD - PAD, bot), 2 * (RAD + PAD), h,
            boxstyle='round,pad=0.2', facecolor=color,
            edgecolor='#BDC3C7', lw=LW_BG, alpha=0.45, zorder=0))
        ax.text(qx, top + 0.15 * scale, label,
                ha='center', va='bottom', fontsize=FS_REGION,
                fontweight='bold', color='#2C3E50', zorder=10)

    # ── Time arrow ─────────────────────────────────────────────────
    time_y = ymin + 0.3 * scale
    ax.annotate('', xy=(SRC_X + 1.5 * scale, time_y),
                xytext=(SNK_X - 1.5 * scale, time_y),
                arrowprops=dict(arrowstyle='<-', color='#2C3E50',
                                lw=LW_TIME, mutation_scale=20 * scale))
    ax.text((SNK_X + SRC_X) / 2, time_y - 0.45 * scale,
            'Time Direction  (sink <- Source)',
            ha='center', va='top', fontsize=FS_TIME,
            fontstyle='italic', fontweight='bold', color='#2C3E50')

    # ── Particle separator lines ───────────────────────────────────
    def get_particle_intervals(operators):
        sep_positions = [i for i, op in enumerate(operators) if op == '|']
        return [(sep_positions[i], sep_positions[i + 1])
                for i in range(0, len(sep_positions), 2)]

    source_particle_intervals = get_particle_intervals(result_dict['source_operators'])
    sink_particle_intervals   = get_particle_intervals(result_dict['sink_operators'])

    sink_offset    = 0
    curr_offset    = len(result_dict['sink_operators'])
    source_offset  = curr_offset + len(result_dict['curr_operators'])

    def draw_particle_separator_lines(quarks, ys, region_x, particle_intervals, offset=0):
        if len(quarks) <= 1:
            return

        particle_groups = []
        for p_low, p_high in particle_intervals:
            abs_low  = p_low + offset
            abs_high = p_high + offset
            group = [(q_idx, q_y) for (q_idx, _, _), q_y in zip(quarks, ys)
                     if abs_low <= q_idx <= abs_high]
            if group:
                group.sort(key=lambda x: x[1])
                particle_groups.append(group)

        for i in range(len(particle_groups) - 1):
            last_of_current  = particle_groups[i][-1][1]
            first_of_next    = particle_groups[i + 1][0][1]

            lower_text_top    = last_of_current + RAD + TEXT_OFF + TEXT_ASC
            upper_circle_bot  = first_of_next - RAD
            separator_y       = (lower_text_top + upper_circle_bot) / 2.0

            x_left  = region_x - RAD - PAD + 0.1 * scale
            x_right = region_x + RAD + PAD - 0.1 * scale
            ax.plot([x_left, x_right], [separator_y, separator_y],
                    color='#2C3E50', lw=LW_SEP, ls='--', alpha=0.6, zorder=1)

    draw_particle_separator_lines(src_quarks, src_ys, SRC_X,
                                  source_particle_intervals, offset=source_offset)
    draw_particle_separator_lines(snk_quarks, snk_ys, SNK_X,
                                  sink_particle_intervals, offset=sink_offset)

    # ── Helper: position by operator index ─────────────────────────
    def gq(qidx):
        for i, (idx, _, _) in enumerate(src_quarks):
            if idx == qidx:
                return SRC_X, src_ys[i]
        for ci, cq in enumerate(cur_quarks):
            for i, (idx, _, _) in enumerate(cq):
                if idx == qidx:
                    return CUR_XS[ci], cur_yss[ci][i]
        for i, (idx, _, _) in enumerate(snk_quarks):
            if idx == qidx:
                return SNK_X, snk_ys[i]
        return None, None

    # ── Draw quark nodes ───────────────────────────────────────────
    def display_name(qtype):
        if '^d' in qtype:
            base = qtype.replace('^d', '')
            return r'$\bar{' + base + '}$'
        return qtype

    def draw_node(cx, cy, qtype, lbl, rlabel, ni):
        c = QC.get(qtype, '#95A5A6')
        ax.add_patch(plt.Circle((cx, cy), RAD, color=c,
                                ec='#2C3E50', lw=LW_NODE, zorder=5))
        ax.text(cx, cy, display_name(qtype),
                ha='center', va='center',
                fontsize=FS_NODE, fontweight='bold', color='white', zorder=6)
        ax.text(cx, cy + RAD + TEXT_OFF, f'[{lbl}]  {rlabel} {ni+1}',
                ha='center', va='bottom', fontsize=FS_LABEL,
                color='#5D6D7E', zorder=6)

    for i, (idx, qt, lb) in enumerate(src_quarks):
        draw_node(SRC_X, src_ys[i], qt, lb, 'S', i)
    for ci, cq in enumerate(cur_quarks):
        for i, (idx, qt, lb) in enumerate(cq):
            draw_node(CUR_XS[ci], cur_yss[ci][i], qt, lb, f'C{ci+1}', i)
    for i, (idx, qt, lb) in enumerate(snk_quarks):
        draw_node(SNK_X, snk_ys[i], qt, lb, 'K', i)

    # ── Gamma-matrix dashed lines + Vertex labels ──────────────────
    if plot_text:
        for _, gname, gidx, _ in gamma_pos:
            if len(gidx) < 2:
                continue
            i1 = letter_to_quark.get(gidx[0])
            i2 = letter_to_quark.get(gidx[1])
            if i1 is None or i2 is None:
                continue
            x1, y1 = gq(i1[1])
            x2, y2 = gq(i2[1])
            if x1 is None or x2 is None:
                continue

            r1, r2 = get_region(i1[1]), get_region(i2[1])
            same_region = (r1 == r2)

            ax.plot([x1, x2], [y1, y2], '--', color='#E67E22',
                    lw=LW_GAMMA, alpha=0.85, zorder=2)

            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if same_region:
                mid_region_x = (SNK_X + SRC_X) / 2
                dx = -1.6 * scale if x1 < mid_region_x else 1.6 * scale
            else:
                dx = 1.0 * scale
                my += 0.55 * scale

            parts = gname.replace('gamma_', '').split('_')
            subscript = '|'.join(parts)
            label = r'$\Gamma_{\mathrm{%s}}^{\mathrm{%s}}$' % (subscript, gidx)

            ax.text(mx + dx, my, label,
                    ha='center', va='center', fontsize=FS_GAMMA,
                    fontweight='bold', color='#D35400',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#FFF3E0',
                              ec='#E67E22', alpha=0.95), zorder=8)

        VC = {'VVV': '#C0392B', 'VDV': '#8E44AD'}
        for _, vn, vi, _ in V_info:
            vt = 'VVV' if 'VVV' in vn else 'VDV'
            col = VC[vt]
            pts = []
            for l in vi:
                info = letter_to_quark.get(l)
                if info is not None:
                    xy = gq(info[1])
                    if xy[0] is not None:
                        pts.append(xy)
            if not pts:
                continue
            cx = np.mean([p[0] for p in pts])
            if vt == 'VDV':
                cy = pts[1][1]
            else:
                cy = np.mean([p[1] for p in pts])

            mid_region_x = (SNK_X + SRC_X) / 2
            if cx < mid_region_x - 1 * scale:
                dx = -1.6 * scale
            elif cx > mid_region_x + 1 * scale:
                dx = 1.6 * scale
            else:
                dx = 1.5 * scale

            ax.text(cx + dx, cy, r'$\mathrm{%s}^{\mathrm{%s}}$' % (vn, vi),
                    ha='center', va='center', fontsize=FS_VERTEX,
                    fontweight='bold', color=col,
                    bbox=dict(boxstyle='round,pad=0.25', fc='#FDEDEC',
                              ec=col, alpha=0.95), zorder=8)
    # ── Build propagator list ──────────────────────────────────────
    quark_propagators = []
    for p in peram_list:
        pstr = p[3]
        src_l = (pstr[1], pstr[3])
        snk_l = (pstr[0], pstr[2])
        si = letter_to_quark.get(src_l[0])
        ki = letter_to_quark.get(snk_l[0])
        if si is None or ki is None:
            continue
        quark_propagators.append(dict(
            src_idx=si[1], snk_idx=ki[1],
            src_qtype=si[0], snk_qtype=ki[0]))
        
    # ── Propagators — arrows land on circle boundaries ─────────────
    PC = ['#2980B9', '#8E44AD', '#D35400', '#27AE60',
          '#C0392B', '#16A085', '#7F8C8D', '#F1C40F']

    for pi, qp in enumerate(quark_propagators):
        col = PC[pi % len(PC)]
        sx, sy = gq(qp['src_idx'])
        kx, ky = gq(qp['snk_idx'])
        if sx is None or kx is None:
            continue

        dist = np.sqrt((kx - sx)**2 + (ky - sy)**2)
        if dist < 2 * RAD + 0.01 * scale:
            continue

        # Unit direction vector (source center → sink center)
        ux = (kx - sx) / dist
        uy = (ky - sy) / dist

        # Arrow endpoints on circle boundaries
        start_x = sx + RAD * ux
        start_y = sy + RAD * uy
        end_x   = kx - RAD * ux
        end_y   = ky - RAD * uy

        # Arc curvature direction
        if kx < SNK_X + 1 * scale and sx > SRC_X - 1 * scale:
            rad = -(0.20 + 0.06 * (pi % 3))
        else:
            rad = (0.20 + 0.06 * (pi % 3))

        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(
                        arrowstyle='->', color=col,
                        lw=LW_ARROW, mutation_scale=MUT_SCALE,
                        connectionstyle=f'arc3,rad={rad}'),
                    zorder=3)

    # ── Legend ─────────────────────────────────────────────────────
    seen = sorted(set(qt for _, qt, _ in quark_pos),
                  key=lambda x: x.replace('^d', 'z'))
    items = [patches.Patch(fc=QC.get(q, '#95A5A6'), ec='#2C3E50',
                           label=display_name(q))
             for q in seen]
    items += [
        plt.Line2D([], [], color='#2980B9', lw=LW_LEGEND,
                   label='Perambulator'),
        plt.Line2D([], [], color='#E67E22', lw=LW_LEGEND * 0.67, ls='--',
                   label='Gamma matrix'),
        plt.Line2D([], [], color='#2C3E50', lw=LW_LEGEND * 0.67, ls='--',
                   label='Particle separator'),
    ]

    ax.legend(handles=items, loc='upper left',
              bbox_to_anchor=(LEGEND_OFF, 1.0),
              fontsize=FS_LEGEND, framealpha=0.9,
              title='Legend', title_fontsize=FS_LEG_T)

    # ── Info text ──────────────────────────────────────────────────
    idx_str = cur_diagram
    if len(idx_str) > 45:
        parts = idx_str.split(',')
        lines, line = [], ''
        for p in parts:
            if len(line) + len(p) + 1 > 45:
                lines.append(line.rstrip(','))
                line = p + ','
            else:
                line += p + ','
        if line:
            lines.append(line.rstrip(','))
        idx_str = '\n'.join(lines)

    comp_str = cur_name
    if len(comp_str) > 55:
        parts = comp_str.split(', ')
        lines, line = [], ''
        for p in parts:
            if len(line) + len(p) + 2 > 55:
                lines.append(line.rstrip(', '))
                line = p + ', '
            else:
                line += p + ', '
        if line:
            lines.append(line.rstrip(', '))
        comp_str = '\n'.join(lines)

    info_text = (
        f"Contraction indices:\n{idx_str}\n\n"
        f"Components:\n{comp_str}\n\n"
        f"Contraction sign: {sign:.1f}"
    )

    ax.text(0.95, 0.02, info_text,
            transform=ax.transAxes, fontsize=FS_INFO,
            va='bottom', ha='left', family='monospace',
            bbox=dict(boxstyle='round', fc='#F7F9F9',
                      ec='#BDC3C7', alpha=0.95))

    tag = '2-pt' if is_2pt else f'{n_regions}-pt'
    ax.set_title(f'Wick Contraction Diagram #{diagram_index}  '
                 f'({Cpt}, sign = {sign:.1f})',
                 fontsize=FS_TITLE, fontweight='bold', pad=TITLE_PAD)
    plt.tight_layout()

    return fig, ax
