"""
Same interface as baroperator.py, but GAMMA_PROPERTIES (H, T, C) and
DIQUARK_TRANSPOSE_SIGN are computed directly from the gamma matrices
defined in gamma_matrix.py, instead of being hardcoded.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from ..constant.gamma_matrix import g0, g1, g2, g3, g4, g5

# Charge conjugation matrix in DR basis: C = gamma_4 gamma_2
C_mat = g4 @ g2           # C
C_inv = C_mat.conj().T    # C^{-1} = C^\dagger  (C is unitary)


def _H(M):
    """Hermitian conjugate: Gamma^H = gamma_4 Gamma^dagger gamma_4"""
    return g4 @ M.conj().T @ g4

def _T(M):
    """Transpose"""
    return M.T

def _C(M):
    """Charge conjugation: C^{-1} Gamma C"""
    return C_inv @ M @ C_mat


def _sign(M_ref, M_result):
    """Identify sign: if M_result = s * M_ref (s = +/-1), return s"""
    num = np.trace(M_ref.conj().T @ M_result)
    den = np.trace(M_ref.conj().T @ M_ref)
    ratio = num / den
    if np.abs(ratio.real - 1.0) < 1e-12:
        return +1
    if np.abs(ratio.real + 1.0) < 1e-12:
        return -1
    raise ValueError(f"Not proportional: ratio.real = {ratio.real}")


# Representative 4x4 matrices for each gamma structure.
# For index-parametrised structures (gamma_mu, sigma_mu_nu, etc.)
# use the time component (mu=0) or time-space component (mu=0, nu=1).
_gamma_vec = [g4, g1, g2, g3]          # gamma_0, gamma_1, gamma_2, gamma_3


def _sigma(mu, nu):
    """sigma_{mu,nu} = 1/2 [gamma_mu, gamma_nu] (Euclidean convention, no factor of i)"""
    return 0.5 * (_gamma_vec[mu] @ _gamma_vec[nu] - _gamma_vec[nu] @ _gamma_vec[mu])


# Mapping: name (str) -> representative 4x4 complex matrix
_MAT = {
    "1":                        g0,
    "gamma_5":                  g5,
    "gamma_0":                  g4,
    "gamma_1":                  g1,
    "gamma_2":                  g2,
    "gamma_3":                  g3,
    "gamma_4":                  g4,
    "gamma_6":                  g2 @ g3,       # gamma(6)  = gamma_2 * gamma_3
    "gamma_7":                  g3 @ g1,       # gamma(7)  = gamma_3 * gamma_1
    "gamma_8":                  g1 @ g2,       # gamma(8)  = gamma_1 * gamma_2
    "gamma_mu":                 g4,                       # mu = 0  (time)
    "gamma_5 * gamma_mu":       g5 @ g4,                  # mu = 0
    "sigma_mu_nu":              _sigma(0, 1),             # (mu,nu) = (0,1)
    "C":                        C_mat,
    "C * gamma_5":              C_mat @ g5,
    "C * gamma_mu":             C_mat @ g4,               # mu = 0
    "C * gamma_5 * gamma_mu":   C_mat @ g5 @ g4,          # mu = 0
    "C * sigma_mu_nu":          C_mat @ _sigma(0, 1),     # (mu,nu) = (0,1)
}

# Compute GAMMA_PROPERTIES: H/T/C transformation signs for each gamma structure
GAMMA_PROPERTIES = {}
for name, M in _MAT.items():
    GAMMA_PROPERTIES[name] = {
        "H": (_sign(M, _H(M)), name),
        "T": (_sign(M, _T(M)), name),
        "C": (_sign(M, _C(M)), name),
    }

# Compute DIQUARK_TRANSPOSE_SIGN: (C*Gamma)^T = eta * (C*Gamma)
DIQUARK_TRANSPOSE_SIGN = {}
_diquark_names = [
    "C",
    "C * gamma_5",
    "C * gamma_mu",
    "C * gamma_5 * gamma_mu",
    "C * sigma_mu_nu",
]
for name in _diquark_names:
    D = _MAT[name]
    DIQUARK_TRANSPOSE_SIGN[name] = _sign(D, D.T)


@dataclass
class Gamma:
    """Wrapper providing H/T/C queries for a named gamma structure."""
    expr: str

    def H(self):
        if self.expr not in GAMMA_PROPERTIES:
            raise ValueError(f"Unknown gamma structure: {self.expr}")
        return GAMMA_PROPERTIES[self.expr]["H"]

    def T(self):
        if self.expr not in GAMMA_PROPERTIES:
            raise ValueError(f"Unknown gamma structure: {self.expr}")
        return GAMMA_PROPERTIES[self.expr]["T"]

    def C(self):
        if self.expr not in GAMMA_PROPERTIES:
            raise ValueError(f"Unknown gamma structure: {self.expr}")
        return GAMMA_PROPERTIES[self.expr]["C"]


def is_separator(x: str) -> bool:
    """Check whether token is a hadron delimiter '|'."""
    return x == "|"


def is_gamma(x: str) -> bool:
    """Check whether token is a known gamma structure name."""
    return x in GAMMA_PROPERTIES


def is_quark(x: str) -> bool:
    """Check whether token is a quark field (not separator, not gamma)."""
    if is_separator(x):
        return False
    if is_gamma(x):
        return False
    return True


def dagger_quark(q: str) -> str:
    """Toggle the dagger flag on a quark name: 'q' <-> 'q^d'."""
    if "^d" in q:
        return q.replace("^d", "")
    return q + "^d"


def split_hadrons(tokens: List[str]) -> List[List[str]]:
    """Split a flat token list into individual hadrons delimited by '|'."""
    hadrons = []
    current = []
    inside = False
    for x in tokens:
        if x == "|" and not inside:
            current = ["|"]
            inside = True
        elif x == "|" and inside:
            current.append("|")
            hadrons.append(current)
            inside = False
        else:
            current.append(x)
    return hadrons


def classify_structure(body: List[str]) -> str:
    """Classify a hadron body as 'meson' (2q), 'baryon' (3q), or 'generic'."""
    n_quark = sum(is_quark(x) for x in body)
    if n_quark == 2:
        return "meson"
    if n_quark == 3:
        return "baryon"
    return "generic"


def hermitian_gamma_chain(gammas: List[str]) -> Tuple[int, List[str]]:
    """Compute the Hermitian conjugate of a gamma chain: (G1 G2)^H = G2^H G1^H."""
    sign = 1
    new_chain = []
    for g in gammas[::-1]:
        GG = Gamma(g)
        s, gg = GG.H()
        sign *= s
        new_chain.append(gg)
    return sign, new_chain


def conjugate_meson(quarks, gammas):
    """Conjugate a meson operator: qbar Gamma q -> qbar Gamma^H q."""
    sign = 1
    gamma_sign, new_gammas = hermitian_gamma_chain(gammas)
    sign *= gamma_sign
    qbar = dagger_quark(quarks[1])
    q = dagger_quark(quarks[0])
    result = ["|", qbar, *new_gammas, q, "|"]
    return [sign] + result


def conjugate_baryon(quarks, gammas):
    """Conjugate a baryon operator: (q1^T C Gamma q2) q3 -> (q1bar C Gamma q2bar) q3bar."""
    sign = -1
    gamma_sign, new_gammas = hermitian_gamma_chain(gammas)
    sign *= gamma_sign
    for g in gammas:
        if g in DIQUARK_TRANSPOSE_SIGN:
            sign *= DIQUARK_TRANSPOSE_SIGN[g]
    q1 = dagger_quark(quarks[1])
    q2 = dagger_quark(quarks[2])
    q3 = dagger_quark(quarks[0])
    result = ["|", q1, *new_gammas, q2, q3, "|"]
    return [sign] + result


def conjugate_generic(quarks, gammas):
    """Conjugate a generic multi-quark operator (reverse quark order, dagger all)."""
    sign = 1
    gamma_sign, new_gammas = hermitian_gamma_chain(gammas)
    sign *= gamma_sign
    new_quarks = [dagger_quark(q) for q in quarks[::-1]]
    result = ["|", *new_quarks, *new_gammas, "|"]
    return [sign] + result


def conjugate_single_hadron(tokens: List[str]):
    """Conjugate a single hadron token list, dispatching by structure type."""
    if tokens[0] != "|" or tokens[-1] != "|":
        raise ValueError("Hadron must start/end with '|'")
    body = tokens[1:-1]
    quarks = []
    gammas = []
    for x in body:
        if is_quark(x):
            quarks.append(x)
        elif is_gamma(x):
            gammas.append(x)
        else:
            raise ValueError(f"Unknown token: {x}")
    structure = classify_structure(body)
    if structure == "meson":
        return conjugate_meson(quarks, gammas)
    elif structure == "baryon":
        return conjugate_baryon(quarks, gammas)
    else:
        return conjugate_generic(quarks, gammas)


def conjugate_operator(tokens: List[str]):
    """Conjugate a full operator expression containing one or more hadrons. Like ```["|", "u^d", "gamma_5", "d", "|"]```"""
    hadrons = split_hadrons(tokens)
    total_sign = 1
    final_tokens = []
    for hadron in hadrons:
        result = conjugate_single_hadron(hadron)
        total_sign *= result[0]
        final_tokens.extend(result[1:])
    return [float(total_sign)] + final_tokens


def transpose_gamma(gamma_expr: str):
    """Return the transpose sign of a gamma structure: (sign, name)."""
    G = Gamma(gamma_expr)
    return G.T()


def charge_conjugation_gamma(gamma_expr: str):
    """Return the charge conjugation sign of a gamma structure: (sign, name)."""
    G = Gamma(gamma_expr)
    return G.C()


def diquark_symmetry(gamma_expr: str):
    """Return the diquark transpose symmetry sign eta: (C*Gamma)^T = eta * (C*Gamma)."""
    if gamma_expr not in DIQUARK_TRANSPOSE_SIGN:
        raise ValueError(f"{gamma_expr} is not a diquark structure")
    return DIQUARK_TRANSPOSE_SIGN[gamma_expr]
