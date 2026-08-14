#!/usr/bin/env python3
"""
DeGrand-Rossi (chiral) basis gamma matrices — GPU (CuPy).

Verified against docker-v20260802/gamma_matrix_gpu.py and
donghx/gamma_matrix_cupy_DR.py.

DR basis (Euclidean, Hermitian):
  γ₁ : i * anti-diag(1, 1, -1, -1)
  γ₂ : anti-diag(-1, 1, 1, -1)
  γ₃ : i * anti-diag(1, -1, -1, 1)
  γ₄ : anti-diag(1, 1, 1, 1)          ← τ₁⊗I₂ (time direction)
  γ₅ : diag(1, 1, -1, -1)             ← τ₃⊗I₂ (chirality)
  γ₀ : I₄ (identity)

Derived:
  6: γ₂γ₃,  7: γ₃γ₁,  8: γ₁γ₂,
  9: γ₁γ₄, 10: γ₂γ₄, 11: γ₃γ₄,
 12: γ₁γ₅, 13: γ₂γ₅, 14: γ₃γ₅, 15: γ₄γ₅,
 16: γ₃γ₁P₊, 17: γ₃γ₁P₋
"""
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = np

def _dtype():
    from utils import get_compute_dtype
    return get_compute_dtype()

def _cp_dtype():
    from utils import get_cp_dtype
    return get_cp_dtype()

_cache = {}

# ─── Fundamental gamma matrices ─────────────────────────────────────────────

def gamma_0():
    """γ₀ = I₄ (identity)."""
    return cp.eye(4, dtype=_cp_dtype())

def gamma_1():
    """γ₁ = i * anti-diag(1, 1, -1, -1)."""
    g = cp.zeros((4, 4), dtype=_cp_dtype())
    g[0, 3] = 1j; g[1, 2] = 1j; g[2, 1] = -1j; g[3, 0] = -1j
    return g

def gamma_2():
    """γ₂ = anti-diag(-1, 1, 1, -1)."""
    g = cp.zeros((4, 4), dtype=_cp_dtype())
    g[0, 3] = -1.0; g[1, 2] = 1.0; g[2, 1] = 1.0; g[3, 0] = -1.0
    return g

def gamma_3():
    """γ₃ = i * anti-diag(1, -1, -1, 1)."""
    g = cp.zeros((4, 4), dtype=_cp_dtype())
    g[0, 2] = 1j; g[1, 3] = -1j; g[2, 0] = -1j; g[3, 1] = 1j
    return g

def gamma_4():
    """γ₄ = anti-diag(1, 1, 1, 1) — τ₁⊗I₂ (time direction)."""
    g = cp.zeros((4, 4), dtype=_cp_dtype())
    g[0, 2] = 1.0; g[1, 3] = 1.0; g[2, 0] = 1.0; g[3, 1] = 1.0
    return g

def gamma_5():
    """γ₅ = diag(1, 1, -1, -1) — τ₃⊗I₂ (chirality)."""
    g = cp.zeros((4, 4), dtype=_cp_dtype())
    g[0, 0] = 1.0; g[1, 1] = 1.0; g[2, 2] = -1.0; g[3, 3] = -1.0
    return g

# ─── Derived combinations ───────────────────────────────────────────────────

def gamma_6():
    return gamma_2() @ gamma_3()

def gamma_7():
    """γ₇ = γ₃γ₁."""
    return gamma_3() @ gamma_1()

def gamma_8():
    return gamma_1() @ gamma_2()

def gamma_9():
    return gamma_1() @ gamma_4()

def gamma_10():
    return gamma_2() @ gamma_4()

def gamma_11():
    return gamma_3() @ gamma_4()

def gamma_12():
    return gamma_1() @ gamma_5()

def gamma_13():
    return gamma_2() @ gamma_5()

def gamma_14():
    return gamma_3() @ gamma_5()

def gamma_15():
    return gamma_4() @ gamma_5()

def gamma_16():
    return gamma_7() @ ((gamma_0() + gamma_4()) / 2.0)

def gamma_17():
    return gamma_7() @ ((gamma_0() - gamma_4()) / 2.0)

GAMMA_FNS = [gamma_0, gamma_1, gamma_2, gamma_3, gamma_4, gamma_5,
             gamma_6, gamma_7, gamma_8, gamma_9, gamma_10, gamma_11,
             gamma_12, gamma_13, gamma_14, gamma_15, gamma_16, gamma_17]

# ─── Projectors ─────────────────────────────────────────────────────────────

def P_plus():
    """P₊ = (γ₀ + γ₄)/2 — positive parity projector."""
    return (gamma_0() + gamma_4()) / 2.0

def P_minus():
    """P₋ = (γ₀ - γ₄)/2 — negative parity projector."""
    return (gamma_0() - gamma_4()) / 2.0

def C_matrix():
    """C = γ₄γ₅ — charge conjugation matrix."""
    return gamma_4() @ gamma_5()

# ─── Cached access ──────────────────────────────────────────────────────────

def get_gamma(idx: int):
    if 0 <= idx < len(GAMMA_FNS):
        return GAMMA_FNS[idx]()
    raise ValueError(f"Unknown gamma index: {idx} (valid: 0-17)")

def get_gamma_cached(idx: int):
    if idx not in _cache:
        _cache[idx] = get_gamma(idx)
    return _cache[idx]

def get_P_plus_cached():
    if 'P_plus' not in _cache:
        _cache['P_plus'] = P_plus()
    return _cache['P_plus']

def get_P_minus_cached():
    if 'P_minus' not in _cache:
        _cache['P_minus'] = P_minus()
    return _cache['P_minus']

def clear_cache():
    _cache.clear()
    if HAS_CUPY:
        cp.get_default_memory_pool().free_all_blocks()
