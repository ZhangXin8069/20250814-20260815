import sympy
from sympy.physics.quantum.cg import CG

"""
SU(2) 群的具体实现 (适用于自旋、同位旋等角动量代数)
"""
    
def _to_sympy(val):
    """将输入 (字符串、浮点数或整数) 转换为 SymPy 的精确有理数"""
    return sympy.S(str(val))

def SU2combine(states):
    """
    组合。输入多个 (同位旋, 第三分量)，输出最终的所有可能组合及其 CG 系数。
    
    参数:
        states: 列表，形如 [(j1, m1), (j2, m2), ...]
        
    返回:
        字典，键为 (J_total, M_total, intermediate_Js)，值为对应的 CG 系数。
        (注：当粒子数 > 2 时，必须记录中间耦合的 J 值才能唯一标识一个正交态)
    """
    if not states:
        return {}

    states = [(_to_sympy(j), _to_sympy(m)) for j, m in states]
    
    # 初始状态：第一个粒子
    j1, m1 = states[0]
    # current_distribution 存储: { (当前总J, 当前总M, 中间J的元组): 系数 }
    current_distribution = { (j1, m1, ()): sympy.S.One }

    for i in range(1, len(states)):
        j2, m2 = states[i]
        next_distribution = {}

        for (j_prev, m_prev, int_js), coeff_prev in current_distribution.items():
            if coeff_prev == 0:
                continue

            # SU(2) 角动量相加规则：|j1 - j2| <= J <= j1 + j2
            j_min = abs(j_prev - j2)
            j_max = j_prev + j2
            m_new = m_prev + m2

            J = j_min
            while J <= j_max:
                # 计算当前耦合的 CG 系数
                cg_coeff = CG(j_prev, m_prev, j2, m2, J, m_new).doit()
                
                if cg_coeff != 0:
                    # 记录除最后一步外的所有中间 J 值
                    new_int_js = int_js + (J,) if i < len(states) - 1 else int_js
                    key = (J, m_new, new_int_js)
                    # 累乘系数并叠加（多条路径可能到达同一状态）
                    next_distribution[key] = next_distribution.get(key, sympy.S.Zero) + coeff_prev * cg_coeff
                
                J += 1  # 步长为1

        current_distribution = next_distribution

    # 清理输出格式
    result = {}
    for (J, M, int_js), coeff in current_distribution.items():
        if coeff != 0:
            result[(J, M, int_js)] = coeff

    return result

def SU2decompose(j_list, target, intermediate_Js=None):
    """
    分解。输入 N 个总同位旋和最终状态，分解为直和形式。
    
    参数:
        j_list: 列表，各个粒子的总同位旋 [j1, j2, ... jN]
        target: [目标总同位旋, 目标第三分量]
        intermediate_Js: 当 N>2 时，必须提供中间耦合的 J 值来消除简并态的歧义。
        
    返回:
        字典，键为 (m1, m2, ... mN)，值为对应的 CG 系数。
    """
    j_list = [_to_sympy(j) for j in j_list]
    target_J, target_M = [_to_sympy(x) for x in target]

    if intermediate_Js is not None:
        intermediate_Js = [_to_sympy(j) for j in intermediate_Js]
    
    # 边界条件：只剩一个粒子
    if len(j_list) == 1:
        if target_J == j_list[0] and -target_J <= target_M <= target_J:
            return { (target_M,): sympy.S.One }
        return {}

    # 校验：N>2 的态是有简并的，必须指定 intermediate_Js 才能确立唯一基
    if len(j_list) > 2 and intermediate_Js is None:
        raise ValueError("对于 N > 2 的情况，必须提供 intermediate_Js 序列来唯一确定中间耦合路径。")
    elif len(j_list) == 2:
        intermediate_Js = []

    # 从最后一个粒子开始，从右向左递归拆解
    j_last = j_list[-1]
    j_prev = intermediate_Js[-1] if intermediate_Js else j_list[0]
    
    results = {}

    # 遍历最后一个粒子所有可能的 m_last
    m_last = -j_last
    while m_last <= j_last:
        m_prev = target_M - m_last
        if abs(m_prev) <= j_prev:
            # 倒推 CG 系数
            cg = CG(j_prev, m_prev, j_last, m_last, target_J, target_M).doit()
            if cg != 0:
                # 递归分解前序部分
                prev_results = SU2decompose(
                    j_list[:-1], 
                    [j_prev, m_prev],
                    intermediate_Js[:-1] if intermediate_Js else None
                )
                
                for m_tuple, prev_coeff in prev_results.items():
                    new_tuple = m_tuple + (m_last,)
                    results[new_tuple] = results.get(new_tuple, sympy.S.Zero) + prev_coeff * cg
        m_last += 1

    return results