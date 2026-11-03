"""
cached_contract 完整测试程序（最终修正 mock 路径）
"""
import numpy as np
from numpy.testing import assert_array_almost_equal
from typing import Dict, Tuple, List, Union
from opt_einsum import contract_expression, contract_path
from unittest.mock import patch

# ============================================================
# 模块级缓存
# ============================================================
_expr_cache: Dict[Tuple, 'contract_expression'] = {}
_AUTO_OPTIMIZERS = ('auto', 'greedy', 'optimal', 'dp')


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
    shapes = tuple(t.shape for t in tensors)

    # ---- 解析 optimize 参数 ----
    if optimize is True:
        candidate_opts = list(_AUTO_OPTIMIZERS)
        opt_key = _AUTO_OPTIMIZERS
    elif isinstance(optimize, str):
        candidate_opts = [optimize]
        opt_key = optimize
    elif isinstance(optimize, list):
        candidate_opts = optimize
        opt_key = tuple(optimize)
    else:
        raise TypeError(
            f"optimize 参数类型应为 str, bool 或 list，收到 {type(optimize)}"
        )

    key = (einsum_str, shapes, opt_key)

    if key not in _expr_cache:
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
            raise RuntimeError(
                "无法为给定表达式和形状找到可行的收缩路径。"
            )

        _expr_cache[key] = contract_expression(
            einsum_str, *shapes, optimize=best_path
        )

    expr = _expr_cache[key]
    return expr(*tensors)


# ============================================================
# 测试函数
# ============================================================
def clear_cache():
    _expr_cache.clear()


def test_basic_correctness():
    rng = np.random.default_rng(0)
    A = rng.random((3, 4))
    B = rng.random((4, 5))
    res = cached_contract('ab,bc->ac', A, B)
    expected = np.einsum('ab,bc->ac', A, B)
    assert_array_almost_equal(res, expected, decimal=12)
    print("[PASS] 基本正确性")


def test_optimize_true():
    rng = np.random.default_rng(1)
    A = rng.random((2, 3, 4))
    B = rng.random((4, 5))
    C = rng.random((5, 3))
    res = cached_contract('ijk,kl,lj->il', A, B, C, optimize=True)
    expected = np.einsum('ijk,kl,lj->il', A, B, C, optimize=True)
    assert_array_almost_equal(res, expected, decimal=12)
    print("[PASS] optimize=True 结果正确")


def test_cache_isolation():
    clear_cache()
    rng = np.random.default_rng(2)
    A = rng.random((2, 3))
    B = rng.random((3, 4))

    res1 = cached_contract('ab,bc->ac', A, B, optimize=True)
    cache_len = len(_expr_cache)

    res2 = cached_contract('ab,bc->ac', A, B, optimize=True)
    assert len(_expr_cache) == cache_len, "True 缓存应命中"
    assert_array_almost_equal(res1, res2, decimal=12)

    cached_contract('ab,bc->ac', A, B, optimize='greedy')
    assert len(_expr_cache) == cache_len + 1, "greedy 应新增缓存"
    print("[PASS] 缓存隔离")


def test_true_vs_explicit_list():
    clear_cache()
    rng = np.random.default_rng(3)
    A = rng.random((3, 4))
    B = rng.random((4, 2))

    res_true = cached_contract('ab,bc->ac', A, B, optimize=True)
    cache_after_true = len(_expr_cache)

    res_list = cached_contract('ab,bc->ac', A, B, optimize=['auto', 'greedy', 'optimal', 'dp'])
    assert len(_expr_cache) == cache_after_true, "应共享 True 缓存"
    assert_array_almost_equal(res_true, res_list, decimal=12)

    cached_contract('ab,bc->ac', A, B, optimize=['greedy', 'auto', 'optimal', 'dp'])
    assert len(_expr_cache) == cache_after_true + 1, "顺序不同应新建缓存"
    print("[PASS] True 与列表缓存共享（顺序敏感）")


def test_fallback_on_optimal_failure():
    clear_cache()
    rng = np.random.default_rng(4)
    A = rng.random((2, 3, 4))
    B = rng.random((4, 5))
    C = rng.random((5, 3))

    original_cp = contract_path
    def mock_cp(einsum_str, *ops, optimize):
        if optimize == 'optimal':
            raise MemoryError("模拟 optimal 失败")
        return original_cp(einsum_str, *ops, optimize=optimize)

    # 正确 mock 当前模块的 contract_path
    with patch(f'{__name__}.contract_path', mock_cp):
        res = cached_contract('ijk,kl,lj->il', A, B, C, optimize=True)
        expected = np.einsum('ijk,kl,lj->il', A, B, C, optimize=True)
        assert_array_almost_equal(res, expected, decimal=12)
    print("[PASS] optimal 失败时自动回退")


def test_no_viable_path():
    clear_cache()
    rng = np.random.default_rng(5)
    A = rng.random((2, 3))
    B = rng.random((3, 4))

    # 正确 mock 当前模块的 contract_path，使所有调用均失败
    with patch(f'{__name__}.contract_path', side_effect=RuntimeError("全部失败")):
        try:
            cached_contract('ab,bc->ac', A, B, optimize=True)
        except RuntimeError as e:
            assert "无法为给定表达式" in str(e), f"异常信息不符: {e}"
        else:
            assert False, "应抛出 RuntimeError"
    print("[PASS] 无可行路径时正确抛出异常")


def test_multi_expression():
    clear_cache()
    rng = np.random.default_rng(6)
    A = rng.random((2, 3, 4))
    B = rng.random((4, 5))
    C = rng.random((5, 3))
    cached_contract('ijk,kl,lj->il', A, B, C, optimize=True)

    A2 = rng.random((2, 3, 4))
    B2 = rng.random((4, 3))
    cached_contract('ijk,kj->ij', A2, B2, optimize=True)
    
    A2 = rng.random((2, 3, 5))
    B2 = rng.random((5, 3))
    cached_contract('ijk,kj->ij', A2, B2, optimize=True)
    
    assert len(_expr_cache) >= 2, f"缓存条目数应为2，实际 {len(_expr_cache)}"
    print(f"[PASS] 不同表达式建立不同缓存, {len(_expr_cache)}")


def test_type_error():
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


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("开始测试 cached_contract（最终版）...\n")
    test_basic_correctness()
    test_optimize_true()
    test_cache_isolation()
    test_true_vs_explicit_list()
    test_fallback_on_optimal_failure()
    test_no_viable_path()
    test_multi_expression()
    test_type_error()
    print("\n所有测试通过！")
