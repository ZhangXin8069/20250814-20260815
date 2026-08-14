#!/usr/bin/env python3
"""
DeGrand-Rossi (chiral) basis gamma matrices for lattice QCD.

Adapted from /root/lattice-pdf/examples/donghx/gamma_matrix_cupy_DR.py
Converted from CuPy to pure NumPy for CPU execution.

The DeGrand-Rossi basis is a chiral-variant representation where:
- gamma_4 is diagonal
- gamma_5 is off-diagonal
- gamma_i (i=1,2,3) have a specific block structure

Reference: DeGrand & Rossi, Comput. Phys. Commun. 60 (1990) 211.
"""

import numpy as np


def gamma_0() -> np.ndarray:
    """Identity matrix (gamma_0 = I_4 in DR basis)."""
    return np.eye(4, dtype=complex)


def gamma_1() -> np.ndarray:
    """Gamma matrix gamma_1 in DeGrand-Rossi basis."""
    g = np.zeros((4, 4), dtype=complex)
    g[0, 3] = -1j
    g[1, 2] = -1j
    g[2, 1] = 1j
    g[3, 0] = 1j
    return g


def gamma_2() -> np.ndarray:
    """Gamma matrix gamma_2 in DeGrand-Rossi basis."""
    g = np.zeros((4, 4), dtype=complex)
    g[0, 3] = -1.0
    g[1, 2] = 1.0
    g[2, 1] = 1.0
    g[3, 0] = -1.0
    return g


def gamma_3() -> np.ndarray:
    """Gamma matrix gamma_3 in DeGrand-Rossi basis."""
    g = np.zeros((4, 4), dtype=complex)
    g[0, 2] = -1j
    g[1, 3] = 1j
    g[2, 0] = 1j
    g[3, 1] = -1j
    return g


def gamma_4() -> np.ndarray:
    """Gamma matrix gamma_4 (time direction) in DeGrand-Rossi basis."""
    return np.diag([1.0, 1.0, -1.0, -1.0])


def gamma_5() -> np.ndarray:
    """Gamma matrix gamma_5 = gamma_1 * gamma_2 * gamma_3 * gamma_4."""
    g = np.zeros((4, 4), dtype=complex)
    g[0, 2] = 1.0
    g[1, 3] = 1.0
    g[2, 0] = 1.0
    g[3, 1] = 1.0
    return g


# ─── Derived combinations (matching donghx's gamma_matrix_cupy_DR.py) ────────

# gamma[6] = gamma_2 @ gamma_3  (also equal to -gamma_1 @ gamma_4 @ gamma_5)
def gamma_6() -> np.ndarray:
    """gamma_2 * gamma_3 = -gamma_1 * gamma_4 * gamma_5."""
    return gamma_2() @ gamma_3()


# gamma[7] = gamma_3 @ gamma_1  (also equal to -gamma_2 @ gamma_4 @ gamma_5)
def gamma_7() -> np.ndarray:
    """gamma_3 * gamma_1 = -gamma_2 * gamma_4 * gamma_5."""
    return gamma_3() @ gamma_1()


# gamma[8] = gamma_1 @ gamma_2  (also equal to -gamma_3 @ gamma_4 @ gamma_5)
def gamma_8() -> np.ndarray:
    """gamma_1 * gamma_2 = -gamma_3 * gamma_4 * gamma_5."""
    return gamma_1() @ gamma_2()


# gamma[9]  = gamma_1 @ gamma_4
# gamma[10] = gamma_2 @ gamma_4
# gamma[11] = gamma_3 @ gamma_4
def gamma_9() -> np.ndarray:
    return gamma_1() @ gamma_4()


def gamma_10() -> np.ndarray:
    return gamma_2() @ gamma_4()


def gamma_11() -> np.ndarray:
    return gamma_3() @ gamma_4()


# gamma[12] = gamma_1 @ gamma_5
# gamma[13] = gamma_2 @ gamma_5
# gamma[14] = gamma_3 @ gamma_5
# gamma[15] = gamma_4 @ gamma_5
def gamma_12() -> np.ndarray:
    return gamma_1() @ gamma_5()


def gamma_13() -> np.ndarray:
    return gamma_2() @ gamma_5()


def gamma_14() -> np.ndarray:
    return gamma_3() @ gamma_5()


def gamma_15() -> np.ndarray:
    return gamma_4() @ gamma_5()


# gamma[16] = gamma_3 @ gamma_1 @ (1+gamma_4)/2  (positive parity projection)
# gamma[17] = gamma_3 @ gamma_1 @ (1-gamma_4)/2  (negative parity projection)
def gamma_16() -> np.ndarray:
    P_plus = (gamma_0() + gamma_4()) / 2.0
    return gamma_3() @ gamma_1() @ P_plus


def gamma_17() -> np.ndarray:
    P_minus = (gamma_0() - gamma_4()) / 2.0
    return gamma_3() @ gamma_1() @ P_minus


# Convenience: all gamma matrices indexed by number
GAMMA_INDEX = {
    0: gamma_0,
    1: gamma_1,
    2: gamma_2,
    3: gamma_3,
    4: gamma_4,
    5: gamma_5,
    6: gamma_6,
    7: gamma_7,
    8: gamma_8,
    9: gamma_9,
    10: gamma_10,
    11: gamma_11,
    12: gamma_12,
    13: gamma_13,
    14: gamma_14,
    15: gamma_15,
    16: gamma_16,
    17: gamma_17,
}


def get_gamma(idx: int) -> np.ndarray:
    """Get gamma matrix by index.

    Args:
        idx: 0=identity, 1-4=gamma_1..gamma_4, 5=gamma_5,
             6-17=derived combinations.
    """
    if idx not in GAMMA_INDEX:
        raise ValueError(f"Unknown gamma index: {idx} (valid: 0-17)")
    return GAMMA_INDEX[idx]()


def build_all_gammas() -> dict[int, np.ndarray]:
    """Build and cache all 18 gamma matrices."""
    return {idx: fn() for idx, fn in GAMMA_INDEX.items()}


# ─── Projection operators ────────────────────────────────────────────────────

def P_plus() -> np.ndarray:
    """Positive parity projection: (1 + gamma_4) / 2."""
    return (gamma_0() + gamma_4()) / 2.0


def P_minus() -> np.ndarray:
    """Negative parity projection: (1 - gamma_4) / 2."""
    return (gamma_0() - gamma_4()) / 2.0


# ─── Interpolation operators ─────────────────────────────────────────────────

def C_matrix() -> np.ndarray:
    """Charge conjugation matrix: C = gamma_2 * gamma_4."""
    return gamma_2() @ gamma_4()


def Cg5() -> np.ndarray:
    """C * gamma_5."""
    return C_matrix() @ gamma_5()


def Cg5g4() -> np.ndarray:
    """C * gamma_5 * gamma_4 -- proton interpolation element."""
    return C_matrix() @ gamma_5() @ gamma_4()


def Cg5g3() -> np.ndarray:
    """C * gamma_5 * gamma_3 -- alternative proton element."""
    return C_matrix() @ gamma_5() @ gamma_3()


def Cg1() -> np.ndarray:
    """C * gamma_1."""
    return C_matrix() @ gamma_1()


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_gamma_identities() -> dict[str, bool]:
    """Check fundamental gamma matrix identities."""
    results = {}

    g = {i: get_gamma(i) for i in range(18)}

    # Anti-commutation: {gamma_mu, gamma_nu} = 2 * delta_{mu,nu} * I
    for mu in range(1, 5):
        for nu in range(1, 5):
            anticomm = g[mu] @ g[nu] + g[nu] @ g[mu]
            expected = 2.0 * np.eye(4) if mu == nu else np.zeros((4, 4))
            key = f"anticomm_{mu}_{nu}"
            results[key] = np.allclose(anticomm, expected, atol=1e-10)

    # gamma_5 anti-commutes with gamma_mu
    for mu in range(1, 5):
        anticomm = g[5] @ g[mu] + g[mu] @ g[5]
        results[f"gamma5_anticomm_{mu}"] = np.allclose(anticomm, np.zeros((4, 4)), atol=1e-10)

    # gamma_5^2 = I
    results["gamma5_sq"] = np.allclose(g[5] @ g[5], np.eye(4), atol=1e-10)

    # Hermiticity: gamma_mu^dagger = gamma_mu
    for mu in range(1, 5):
        results[f"hermitian_{mu}"] = np.allclose(
            g[mu].conj().T, g[mu], atol=1e-10
        )

    return results


if __name__ == "__main__":
    print("Gamma matrix validation:")
    results = validate_gamma_identities()
    all_ok = True
    for key, ok in results.items():
        status = "✓" if ok else "✗ FAILED"
        if not ok:
            all_ok = False
        print(f"  {status}: {key}")

    if all_ok:
        print("\nAll gamma matrix identities verified.")
    else:
        print("\nWARNING: Some identities failed!")
