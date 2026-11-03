"""
cached_contract 完整测试程序。
直接调用 base_functions.py 中的实现。
"""
import sys
sys.path.insert(0, '/public/home/sush/distillation/lqcddb/src/lqcddb')

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
from unittest.mock import patch

from base import base_functions
from base import cached_contract, clear_cache

def test_basic_correctness():
    """基本正确性：单次收缩结果与 np.einsum 一致。"""
    rng = np.random.default_rng(0)
    A = rng.random((3, 4))
    B = rng.random((4, 5))
    res = cached_contract('ab,bc->ac', A, B)
    expected = np.einsum('ab,bc->ac', A, B)
    assert_array_almost_equal(res, expected, decimal=12)
    print("[PASS] 基本正确性")

def test_optimize_true():
    """optimize=True：自动择优结果正确。"""
    rng = np.random.default_rng(1)
    A = rng.random((2, 3, 4))
    B = rng.random((4, 5))
    C = rng.random((5, 3))
    res = cached_contract('ijk,kl,lj->il', A, B, C, optimize=True)
    expected = np.einsum('ijk,kl,lj->il', A, B, C, optimize=True)
    assert_array_almost_equal(res, expected, decimal=12)
    print("[PASS] optimize=True 结果正确")


def test_cache_isolation():
    """不同 optimize 策略产生独立缓存条目。"""
    clear_cache()
    rng = np.random.default_rng(2)
    A = rng.random((2, 3))
    B = rng.random((3, 4))

    res1 = cached_contract('ab,bc->ac', A, B, optimize=True)
    cache_len = len(base_functions._expr_cache)

    # 相同参数应命中缓存
    res2 = cached_contract('ab,bc->ac', A, B, optimize=True)
    assert len(base_functions._expr_cache) == cache_len, "True 缓存应命中"
    assert_array_almost_equal(res1, res2, decimal=12)

    # 不同 optimize 参数应新建缓存
    cached_contract('ab,bc->ac', A, B, optimize='greedy')
    assert len(base_functions._expr_cache) == cache_len + 1, "greedy 应新增缓存"
    print("[PASS] 缓存隔离")


def test_true_vs_explicit_list():
    """optimize=True 与显式传入全部四种策略共享缓存（顺序敏感）。"""
    clear_cache()
    rng = np.random.default_rng(3)
    A = rng.random((3, 4))
    B = rng.random((4, 2))

    res_true = cached_contract('ab,bc->ac', A, B, optimize=True)
    cache_after_true = len(base_functions._expr_cache)

    # 相同顺序的列表应与 True 共享缓存
    res_list = cached_contract(
        'ab,bc->ac', A, B,
        optimize=['auto', 'greedy', 'optimal', 'dp']
    )
    assert len(base_functions._expr_cache) == cache_after_true, (
        "应共享 True 缓存（opt_key 相同）"
    )
    assert_array_almost_equal(res_true, res_list, decimal=12)

    # 不同顺序应新建缓存
    cached_contract(
        'ab,bc->ac', A, B,
        optimize=['greedy', 'auto', 'optimal', 'dp']
    )
    assert len(base_functions._expr_cache) == cache_after_true + 1, (
        "顺序不同应新建缓存"
    )
    print("[PASS] True 与列表缓存共享（顺序敏感）")


def test_fallback_on_optimal_failure():
    """某个策略失败时自动回退到其他策略。"""
    clear_cache()
    rng = np.random.default_rng(4)
    A = rng.random((2, 3, 4))
    B = rng.random((4, 5))
    C = rng.random((5, 3))

    import opt_einsum
    real_cp = opt_einsum.contract_path

    def mock_cp(einsum_str, *ops, optimize):
        if optimize == 'optimal':
            raise MemoryError("模拟 optimal 失败")
        return real_cp(einsum_str, *ops, optimize=optimize)

    with patch('opt_einsum.contract_path', mock_cp):
        res = cached_contract('ijk,kl,lj->il', A, B, C, optimize=True)
        expected = np.einsum('ijk,kl,lj->il', A, B, C, optimize=True)
        assert_array_almost_equal(res, expected, decimal=12)
    print("[PASS] optimal 失败时自动回退")


def test_no_viable_path():
    """全部策略失败时抛出明确错误。"""
    clear_cache()
    rng = np.random.default_rng(5)
    A = rng.random((2, 3))
    B = rng.random((3, 4))

    with patch(
        'opt_einsum.contract_path',
        side_effect=RuntimeError("全部失败")
    ):
        try:
            cached_contract('ab,bc->ac', A, B, optimize=True)
        except RuntimeError as e:
            assert "尝试了以下优化策略" in str(e), f"异常信息不符: {e}"
        else:
            assert False, "应抛出 RuntimeError"
    print("[PASS] 无可行路径时正确抛出异常")


def test_multi_expression():
    """不同表达式/形状各自缓存，互不干扰。"""
    clear_cache()
    rng = np.random.default_rng(6)

    A = rng.random((2, 3, 4))
    B = rng.random((4, 5))
    C = rng.random((5, 3))
    cached_contract('ijk,kl,lj->il', A, B, C, optimize=True)

    A2 = rng.random((2, 3, 4))
    B2 = rng.random((4, 3))
    cached_contract('ijk,kj->ij', A2, B2, optimize=True)

    # 形状不同 → 不同缓存条目
    A3 = rng.random((2, 3, 5))
    B3 = rng.random((5, 3))
    cached_contract('ijk,kj->ij', A3, B3, optimize=True)

    assert len(base_functions._expr_cache) == 3, (
        f"缓存条目数应为 3，实际 {len(base_functions._expr_cache)}"
    )
    print(f"[PASS] 不同表达式建立不同缓存 ({len(base_functions._expr_cache)} 条)")


def test_type_error():
    """optimize 参数类型错误检测。"""
    clear_cache()
    rng = np.random.default_rng(7)
    A = rng.random((2, 3))
    B = rng.random((3, 2))
    try:
        cached_contract('ab,bc->ac', A, B, optimize=123)
    except TypeError as e:
        assert "optimize 参数类型应为" in str(e)
    else:
        assert False, "应抛出 TypeError"
    print("[PASS] 参数类型错误检测")


def test_shape_mismatch_detailed():
    """形状不匹配时给出具体诊断：指明第几个变量、第几个维度、实际大小。"""
    clear_cache()
    A = np.ones((3, 4))
    B = np.ones((5, 6))   # 索引 'b'：A 中是 4，B 中是 5

    try:
        cached_contract('ab,bc->ac', A, B)
    except ValueError as e:
        msg = str(e)
        assert "第 1 个变量的第 2 个维度" in msg, f"缺少定位信息: {msg}"
        assert "大小为 4" in msg, f"缺少实际大小: {msg}"
        assert "第 2 个变量的第 1 个维度" in msg, f"缺少对比定位: {msg}"
        assert "大小为 5" in msg, f"缺少对比大小: {msg}"
    else:
        assert False, "应抛出 ValueError"
    print("[PASS] 形状不匹配详细诊断")


def test_shape_mismatch_multiple():
    """同时报告多个索引的大小不一致。"""
    clear_cache()
    # 'ab,bc,cd->ad'：索引 b 和 c 都不匹配
    A = np.ones((2, 3))
    B = np.ones((5, 6))
    C = np.ones((7, 8))

    try:
        cached_contract('ab,bc,cd->ad', A, B, C)
    except ValueError as e:
        msg = str(e)
        # 应该同时报告 b 和 c 的 mismatch
        assert "索引 'b'" in msg, f"缺少索引 b: {msg}"
        assert "索引 'c'" in msg, f"缺少索引 c: {msg}"
    else:
        assert False, "应抛出 ValueError"
    print("[PASS] 同时报告多个维度不匹配")


def test_wrong_tensor_count():
    """张量数量与表达式不匹配。"""
    clear_cache()
    A = np.ones((2, 3))
    B = np.ones((3, 4))
    try:
        cached_contract('ab,bc,cd->ad', A, B)  # 3 个输入但只有 2 个张量
    except ValueError as e:
        msg = str(e)
        assert "3 个输入张量" in msg and "2 个张量" in msg, f"异常信息不符: {msg}"
    else:
        assert False, "应抛出 ValueError"
    print("[PASS] 张量数量不匹配")

# ============================================================
# 多变量收缩测试
# ============================================================

def test_many_tensors_contraction():
    """模拟 Wick 收缩风格的多变量（≥8 个张量）收缩，验证正确性与缓存行为。"""
    clear_cache()
    rng = np.random.default_rng(42)

    # 模拟一个真实的 baryon 3pt Wick 收缩：
    #   4 个 peram (4x4 狄拉克空间), 3 个 gamma (4x4),
    #   2 个 VVV (M×Nev×Nev×Nev), 1 个 VdV (L×Nev×Nev)
    # 简化版：全部用 4x4 矩阵，用 10 个张量
    #
    # einsum: 'ab,cd,ef,gh,ij,kl,mn,op,qr,st->'
    # （全收缩为标量）

    N = 4
    tensors = [rng.random((N, N)) + 1j * rng.random((N, N)) for _ in range(10)]
    sub = ','.join(
        [f'{chr(97+2*i)}{chr(97+2*i+1)}' for i in range(10)]
    ) + '->'

    # 用 np.einsum 得到参考结果（10 张量收缩浮点累积导致绝对值差异
    # 可达 1e-5 但相对误差 < 1e-14，用相对容差比较）
    expected = np.einsum(sub, *tensors, optimize=True)

    # 首次调用（编译 + 缓存）
    result1 = cached_contract(sub, *tensors, optimize=True)
    assert_allclose(result1, expected, rtol=1e-10)
    assert len(base_functions._expr_cache) == 1, "首次应新增 1 条缓存"

    # 再次调用（命中缓存）
    tensors2 = [rng.random((N, N)) + 1j * rng.random((N, N)) for _ in range(10)]
    expected2 = np.einsum(sub, *tensors2, optimize=True)
    result2 = cached_contract(sub, *tensors2, optimize=True)
    assert_allclose(result2, expected2, rtol=1e-10)
    assert len(base_functions._expr_cache) == 1, "形状相同时应命中缓存"

    # 不同形状应新建缓存
    tensors3 = [rng.random((N, N)) + 1j * rng.random((N, N)) for _ in range(8)]
    sub3 = ','.join(
        [f'{chr(97+2*i)}{chr(97+2*i+1)}' for i in range(8)]
    ) + '->'
    expected3 = np.einsum(sub3, *tensors3, optimize=True)
    result3 = cached_contract(sub3, *tensors3, optimize=True)
    assert_allclose(result3, expected3, rtol=1e-10)
    assert len(base_functions._expr_cache) == 2, "不同表达式应新建缓存"

    print(f"[PASS] 多变量收缩 (10 张量 + 8 张量)  ({len(base_functions._expr_cache)} 条缓存)")


def test_many_tensors_mismatch():
    """多变量收缩中任意一个维度不匹配都能被精确定位。"""
    clear_cache()
    rng = np.random.default_rng(99)

    N = 4
    # 链式收缩：ab,bc,cd,de,ef,fg,gh,hi,ij,ja->
    # 相邻张量共享一个索引，第 5 个张量的 'f' 维度故意给错大小
    shapes = [(N, N)] * 10
    shapes[4] = (N, 6)  # 第 5 个张量 ('ef') 的 'f' 维 = 6，但第 6 个张量 ('fg') 的 'f' 维 = 4

    tensors = [
        rng.random(s) + 1j * rng.random(s) for s in shapes
    ]

    indices = ['ab', 'bc', 'cd', 'de', 'ef', 'fg', 'gh', 'hi', 'ij', 'ja']
    sub = ','.join(indices) + '->'

    try:
        cached_contract(sub, *tensors, optimize=True)
    except ValueError as e:
        msg = str(e)
        assert "索引 'f'" in msg, f"应报告索引 'f': {msg}"
        assert "第 5 个变量" in msg, f"应定位到第 5 个变量: {msg}"
        assert "大小为 6" in msg, f"应报告实际大小 6: {msg}"
    else:
        assert False, "应抛出 ValueError"
    print("[PASS] 多变量收缩维度不匹配精确定位")


# ============================================================
# 复杂收缩网络测试
# ============================================================

def test_complex_contraction():
    """模拟真实 Wick 收缩：8 张量、多种秩（2D-4D）、非简单链式拓扑、2D 输出。

    收缩拓扑（每个字母是一个被收缩的索引，M/N 是输出索引）::

         T0: abcd ──────────────────────────────┐
                │a  │b      │c      │d          │
         T1: aefg    │      │       │           │
                │e   │      │       │           │
         T2: behi ───┘      │       │           │
                │h          │       │           │
         T3: cfhj ─────────┘       │           │
                │j                 │           │
         T4: dgiM ────────────────┘           │
                │i │g          │M             │
         T5: jklM              │              │
                │k │l          │              │
         T6: kmnN ─┐           │              │
                │m │n          │              │
         T7: lmnN ─┘───────────┘              │
                                               │
         输出: ->MN  (shape: 2×3) ─────────────┘

    总共 14 个收缩索引 + 2 个输出索引，覆盖 2D/3D/4D 张量。
    """
    clear_cache()
    rng = np.random.default_rng(123)

    # 索引维度映射（物理模拟：spin≈4, ev≈32/16, mom≈2）
    dims = {
        'a': 2, 'b': 3, 'c': 4, 'd': 5,
        'e': 3, 'f': 4, 'g': 5, 'h': 4,
        'i': 5, 'j': 5, 'k': 3, 'l': 4,
        'm': 5, 'n': 6, 'M': 2, 'N': 3,
    }

    # 8 个张量，各自有不同的秩
    spec = [
        ('abcd', (dims['a'], dims['b'], dims['c'], dims['d'])),   # 4D
        ('aefg', (dims['a'], dims['e'], dims['f'], dims['g'])),   # 4D
        ('behi', (dims['b'], dims['e'], dims['h'], dims['i'])),   # 4D
        ('cfhj', (dims['c'], dims['f'], dims['h'], dims['j'])),   # 4D
        ('dgiM', (dims['d'], dims['g'], dims['i'], dims['M'])),   # 4D
        ('jklM', (dims['j'], dims['k'], dims['l'], dims['M'])),   # 4D
        ('kmnN', (dims['k'], dims['m'], dims['n'], dims['N'])),   # 4D
        ('lmnN', (dims['l'], dims['m'], dims['n'], dims['N'])),   # 4D
    ]

    indices = [s[0] for s in spec]
    shapes  = [s[1] for s in spec]
    einsum_str = ','.join(indices) + '->MN'

    tensors = [
        rng.random(shp) + 1j * rng.random(shp) for shp in shapes
    ]

    # ---- 参考结果 ----
    expected = np.einsum(einsum_str, *tensors, optimize=True)

    # ---- 首次调用（编译 + 缓存） ----
    result1 = cached_contract(einsum_str, *tensors, optimize=True)
    assert_allclose(result1, expected, rtol=1e-10)
    assert result1.shape == (dims['M'], dims['N']), (
        f"输出形状应为 ({dims['M']}, {dims['N']})，实际 {result1.shape}"
    )
    cache_count = len(base_functions._expr_cache)
    assert cache_count == 1, f"首次应 1 条缓存，实际 {cache_count}"

    # ---- 再次调用（缓存命中） ----
    tensors2 = [
        rng.random(shp) + 1j * rng.random(shp) for shp in shapes
    ]
    expected2 = np.einsum(einsum_str, *tensors2, optimize=True)
    result2 = cached_contract(einsum_str, *tensors2, optimize=True)
    assert_allclose(result2, expected2, rtol=1e-10)
    assert len(base_functions._expr_cache) == cache_count, "形状相同应命中缓存"

    # ---- 同一拓扑但某一维大小改变 → 新缓存 ----
    dims_v2 = dict(dims, b=7)
    shapes_v2 = [
        (dims_v2['a'], dims_v2['b'], dims_v2['c'], dims_v2['d']),
        (dims_v2['a'], dims_v2['e'], dims_v2['f'], dims_v2['g']),
        (dims_v2['b'], dims_v2['e'], dims_v2['h'], dims_v2['i']),
        (dims_v2['c'], dims_v2['f'], dims_v2['h'], dims_v2['j']),
        (dims_v2['d'], dims_v2['g'], dims_v2['i'], dims_v2['M']),
        (dims_v2['j'], dims_v2['k'], dims_v2['l'], dims_v2['M']),
        (dims_v2['k'], dims_v2['m'], dims_v2['n'], dims_v2['N']),
        (dims_v2['l'], dims_v2['m'], dims_v2['n'], dims_v2['N']),
    ]
    tensors_v2 = [
        rng.random(shp) + 1j * rng.random(shp) for shp in shapes_v2
    ]
    expected_v2 = np.einsum(einsum_str, *tensors_v2, optimize=True)
    result_v2 = cached_contract(einsum_str, *tensors_v2, optimize=True)
    assert_allclose(result_v2, expected_v2, rtol=1e-10)
    assert len(base_functions._expr_cache) == cache_count + 1, (
        f"形状不同应新增缓存，预期 {cache_count + 1}，实际 {len(base_functions._expr_cache)}"
    )

    print(f"[PASS] 复杂收缩网络 (8 张量, 14 收缩索引, 2D 输出)"
          f"  ({len(base_functions._expr_cache)} 条缓存)")


def test_complex_contraction_mismatch():
    """复杂收缩网络中任意维度不匹配能被精确定位。"""
    clear_cache()
    rng = np.random.default_rng(456)

    dims = {
        'a': 2, 'b': 3, 'c': 4, 'd': 5,
        'e': 3, 'f': 4, 'g': 5, 'h': 4,
        'i': 5, 'j': 5, 'k': 3, 'l': 4,
        'm': 5, 'n': 6, 'M': 2, 'N': 3,
    }

    spec = [
        ('abcd', ['a', 'b', 'c', 'd']),
        ('aefg', ['a', 'e', 'f', 'g']),
        ('behi', ['b', 'e', 'h', 'i']),
        ('cfhj', ['c', 'f', 'h', 'j']),
        ('dgiM', ['d', 'g', 'i', 'M']),
        ('jklM', ['j', 'k', 'l', 'M']),
        ('kmnN', ['k', 'm', 'n', 'N']),
        ('lmnN', ['l', 'm', 'n', 'N']),
    ]

    # 故意让 T3 的 'h' 维度大小不对（应为 4，给成 9）
    shapes_good = [
        tuple(dims[d] for d in labels) for _, labels in spec
    ]
    shapes_bad = list(shapes_good)
    # T3 是 cfhj (c=4, f=4, h=4, j=5)，把 h 对应的 dim(2) 改成 9
    shapes_bad[3] = (dims['c'], dims['f'], 9, dims['j'])

    tensors_bad = [
        rng.random(shp) + 1j * rng.random(shp) for shp in shapes_bad
    ]

    einsum_str = ','.join(s[0] for s in spec) + '->MN'

    try:
        cached_contract(einsum_str, *tensors_bad, optimize=True)
    except ValueError as e:
        msg = str(e)
        assert "索引 'h'" in msg, f"应报告索引 'h': {msg}"
        assert "大小为 4" in msg, f"应报告参考大小 4: {msg}"
        assert "大小为 9" in msg, f"应报告错误大小 9: {msg}"
    else:
        assert False, "应抛出 ValueError"
    print("[PASS] 复杂收缩网络维度不匹配精确定位")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("开始测试 cached_contract …\n")

    test_basic_correctness()
    test_optimize_true()
    test_cache_isolation()
    test_true_vs_explicit_list()
    test_fallback_on_optimal_failure()
    test_no_viable_path()
    test_multi_expression()
    test_type_error()
    test_shape_mismatch_detailed()
    test_shape_mismatch_multiple()
    test_wrong_tensor_count()
    test_many_tensors_contraction()
    test_many_tensors_mismatch()
    test_complex_contraction()
    test_complex_contraction_mismatch()

    print("\n所有测试通过！")
