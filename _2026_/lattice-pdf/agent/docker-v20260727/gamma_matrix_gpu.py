#!/usr/bin/env python3
"""
DeGrand-Rossi (chiral) basis gamma matrices — GPU (CuPy), CORRECT donghx convention.

Verified against:
  /root/lattice-pdf/examples/donghx/gamma_matrix_cupy_DR.py
  /root/lattice-pdf/snsc/main.py (build_gamma_matrices)

DR basis (Euclidean, Hermitian):
  γ₁ : i * anti-diag(1, 1, -1, -1)
  γ₂ : anti-diag(-1, 1, 1, -1)
  γ₃ : i * anti-diag(1, -1, -1, 1)
  γ₄ : anti-diag(1, 1, 1, 1)          ← τ₁⊗I₂ (time direction)
  γ₅ : diag(1, 1, -1, -1)             ← τ₃⊗I₂ (chirality)
  C = γ₄γ₅ = γ₂γ₄ (charge conjugation)

KEY: γ₄ = τ₁⊗I₂ (anti-diagonal), γ₅ = τ₃⊗I₂ (diagonal)
     This is the STANDARD DR convention used by donghx and snsc/main.py.

Derived combinations:
  6: γ₂γ₃,  7: γ₃γ₁,  8: γ₁γ₂,
  9: γ₁γ₄, 10: γ₂γ₄, 11: γ₃γ₄,
 12: γ₁γ₅, 13: γ₂γ₅, 14: γ₃γ₅, 15: γ₄γ₅,
 16: γ₃γ₁P₊, 17: γ₃γ₁P₋
"""
import cupy as cp

def _dtype():
    try:
        from utils import get_compute_dtype
        return get_compute_dtype()
    except ImportError:
        return cp.complex64

_cache = {}

# ─── Fundamental gamma matrices (CORRECT donghx convention) ──────────────────

def gamma_0():
    """Identity γ₀ = I₄."""
    return cp.eye(4, dtype=_dtype())

def gamma_1():
    """γ₁ = i * anti-diag(1, 1, -1, -1)."""
    g = cp.zeros((4, 4), dtype=_dtype())
    g[0, 3] = 1j; g[1, 2] = 1j; g[2, 1] = -1j; g[3, 0] = -1j
    return g

def gamma_2():
    """γ₂ = anti-diag(-1, 1, 1, -1)."""
    g = cp.zeros((4, 4), dtype=_dtype())
    g[0, 3] = -1.0; g[1, 2] = 1.0; g[2, 1] = 1.0; g[3, 0] = -1.0
    return g

def gamma_3():
    """γ₃ = i * anti-diag(1, -1, -1, 1)."""
    g = cp.zeros((4, 4), dtype=_dtype())
    g[0, 2] = 1j; g[1, 3] = -1j; g[2, 0] = -1j; g[3, 1] = 1j
    return g

def gamma_4():
    """γ₄ = anti-diag(1, 1, 1, 1) — τ₁⊗I₂ (time direction)."""
    g = cp.zeros((4, 4), dtype=_dtype())
    g[0, 2] = 1.0; g[1, 3] = 1.0; g[2, 0] = 1.0; g[3, 1] = 1.0
    return g

def gamma_5():
    """γ₅ = diag(1, 1, -1, -1) — τ₃⊗I₂ (chirality)."""
    g = cp.zeros((4, 4), dtype=_dtype())
    g[0, 0] = 1.0; g[1, 1] = 1.0; g[2, 2] = -1.0; g[3, 3] = -1.0
    return g

# ─── Derived combinations ────────────────────────────────────────────────────

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
    return gamma_3() @ gamma_1() @ ((gamma_0() + gamma_4()) / 2.0)

def gamma_17():
    return gamma_3() @ gamma_1() @ ((gamma_0() - gamma_4()) / 2.0)

GAMMA_FNS = [gamma_0, gamma_1, gamma_2, gamma_3, gamma_4, gamma_5,
             gamma_6, gamma_7, gamma_8, gamma_9, gamma_10, gamma_11,
             gamma_12, gamma_13, gamma_14, gamma_15, gamma_16, gamma_17]

# ─── Projection & interpolation operators ─────────────────────────────────────

def P_plus():
    """P₊ = (γ₀ + γ₄) / 2 — positive parity projector."""
    return (gamma_0() + gamma_4()) / 2.0

def P_minus():
    """P₋ = (γ₀ - γ₄) / 2 — negative parity projector."""
    return (gamma_0() - gamma_4()) / 2.0

def C_matrix():
    """C = γ₄γ₅ = γ₂γ₄ — charge conjugation matrix."""
    return gamma_4() @ gamma_5()

def Cg5():
    return C_matrix() @ gamma_5()

def Cg5g4():
    return C_matrix() @ gamma_5() @ gamma_4()

def Cg5g3():
    return C_matrix() @ gamma_5() @ gamma_3()

# ─── Cached ──────────────────────────────────────────────────────────────────

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
    cp.get_default_memory_pool().free_all_blocks()

# ─── Validation ──────────────────────────────────────────────────────────────

def validate_gamma_identities():
    """Check Clifford algebra and hermiticity."""
    import numpy as np
    results = {}
    g = {i: cp.asnumpy(get_gamma(i)) for i in range(18)}

    for mu in range(1, 5):
        for nu in range(1, 5):
            anticomm = g[mu] @ g[nu] + g[nu] @ g[mu]
            expected = 2.0 * np.eye(4) if mu == nu else np.zeros((4, 4))
            results[f"anticomm_{mu}_{nu}"] = np.allclose(anticomm, expected, atol=1e-7)

    for mu in range(1, 5):
        anticomm = g[5] @ g[mu] + g[mu] @ g[5]
        results[f"gamma5_anticomm_{mu}"] = np.allclose(anticomm, np.zeros((4, 4)), atol=1e-7)

    results["gamma5_sq"] = np.allclose(g[5] @ g[5], np.eye(4), atol=1e-7)
    results["gamma4_sq"] = np.allclose(g[4] @ g[4], np.eye(4), atol=1e-7)

    for mu in range(1, 5):
        results[f"hermitian_{mu}"] = np.allclose(g[mu].conj().T, g[mu], atol=1e-6)

    # Verify against donghx convention (numerical check)
    # γ₄ should be anti-diagonal [0,2]=1, [1,3]=1, [2,0]=1, [3,1]=1
    g4 = g[4]
    results["g4_is_anti_diag"] = (abs(g4[0,2]-1)<1e-10 and abs(g4[1,3]-1)<1e-10
                                 and abs(g4[2,0]-1)<1e-10 and abs(g4[3,1]-1)<1e-10)
    # γ₅ should be diagonal [0,0]=1, [1,1]=1, [2,2]=-1, [3,3]=-1
    g5 = g[5]
    results["g5_is_diag"] = (abs(g5[0,0]-1)<1e-10 and abs(g5[1,1]-1)<1e-10
                            and abs(g5[2,2]+1)<1e-10 and abs(g5[3,3]+1)<1e-10)

    return results

if __name__ == "__main__":
    print(f"Gamma matrices (dtype={_dtype()})")
    print(f"CuPy {cp.__version__}, Device: {cp.cuda.Device()}")
    results = validate_gamma_identities()
    all_ok = True
    for k, ok in sorted(results.items()):
        s = "✓" if ok else "✗ FAILED"
        if not ok: all_ok = False
        print(f"  {s}: {k}")
    print(f"\n{'ALL OK' if all_ok else 'FAILURES'}")

    # Print the interpolation operator for _Cg5g4
    g7 = get_gamma(7); g4 = get_gamma(4)
    interp = g7 @ g4
    print(f"\nInterpolation operator (_Cg5g4): gamma[7]@gamma[4] =")
    print(cp.asnumpy(interp))
