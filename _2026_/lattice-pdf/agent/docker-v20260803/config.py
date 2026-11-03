"""
Pipeline Configuration
======================

Central configuration for the lattice-pdf GPU pipeline (docker-v20260803).
All physical parameters, data paths, and runtime options are defined here.

Ensemble: beta6.20_mu-0.2770_ms-0.2400_L24x72
Lattice: 24³×72, a ≈ 0.1053 fm, a⁻¹ ≈ 1.874 GeV
"""

import os

# ═══════════════════════════════════════════════════════════════════
# Lattice Ensemble Parameters
# ═══════════════════════════════════════════════════════════════════

ENSEMBLE = "beta6.20_mu-0.2770_ms-0.2400_L24x72"
NX = 24          # Spatial lattice size (isotropic)
NY = 24
NZ = 24
NT = 72          # Temporal lattice size
ALttc = 0.1053   # Lattice spacing in fm
A_INV = 0.1973269804 / ALttc  # Inverse lattice spacing in GeV (~1.874 GeV)

# ═══════════════════════════════════════════════════════════════════
# Configuration IDs
# ═══════════════════════════════════════════════════════════════════

CONF_IDS = [6250, 6450, 6650]

# ═══════════════════════════════════════════════════════════════════
# Data Paths
# ═══════════════════════════════════════════════════════════════════

BASE_EIGEN_DIR = "/public/group/lqcd/eigensystem"
BASE_PERAM_DIR = "/public/group/lqcd/perambulators"
BASE_GAUGE_DIR = "/public/group/lqcd/configurations/CLOVER"

EIGEN_DIR = f"{BASE_EIGEN_DIR}/{ENSEMBLE}"
PERAM_DIR = f"{BASE_PERAM_DIR}/{ENSEMBLE}/light"
GAUGE_DIR = f"{BASE_GAUGE_DIR}/{ENSEMBLE}"

# ═══════════════════════════════════════════════════════════════════
# Distillation Parameters
# ═══════════════════════════════════════════════════════════════════

NEV = 50         # Number of Laplacian eigenvectors to use (reduced for GPU testing)
                  # Use 100 for production runs on HPC clusters
NEV_MAX = 50     # Maximum eigenvectors (for link operators if needed)

# ═══════════════════════════════════════════════════════════════════
# Momentum List
# ═══════════════════════════════════════════════════════════════════
# Momenta for VdV (meson vertex): uses these for 2pt sink projections
# Momenta for VVV (baryon vertex): reversed order convention
# Note: VVV computation is expensive (~Nev³). For testing, use only P=0.

MOM_SINK_VDV = [
    [0, 0, 0],       # Rest frame — always needed
    [0, 0, 2],       # Pz = 2 (required for analysis)
]

# VVV baryon vertex at P=0 and P=(0,0,2)
MOM_SINK_VVV = [
    [0, 0, 0],       # Rest frame
    [0, 0, 2],       # Pz = 2
]

# Analysis momenta: [Pz, Py, Px] in units of 2π/L
# For pion and proton at P=(0,0,0) and P=(0,0,2)
# Note: P=(0,0,2) means Pz=2, Py=0, Px=0
ANALYSIS_MOMENTA = {
    'pion': {
        'P000': [0, 0, 0],   # Rest frame
        'P002': [0, 0, 2],   # pz = 2·2π/L
    },
    'proton': {
        'P000': [0, 0, 0],   # Rest frame
        'P002': [0, 0, 2],   # pz = 2·2π/L
    },
}

# ═══════════════════════════════════════════════════════════════════
# Hadron Operator Definitions
# ═══════════════════════════════════════════════════════════════════
# Format: ['|', quark1, quark2, gamma, quark3, '|']
# gamma_5 inserts at the quark-bilinear, gamma_7 = γ₃γ₁

# Pion: π⁺ = ū γ₅ d
PION_SINK = ['|', 'u^d', 'gamma_5', 'd', '|']
PION_SRC  = ['|', 'd^d', 'gamma_5', 'u', '|']

# Proton: p = ε_{abc} (u^T C γ₅ d) u  in DR basis
# Represented as: ['|', 'u', 'u', 'gamma_7', 'd', '|']
# gamma_7 = γ₃γ₁ provides the C*γ₅ diquark structure in DR basis
PROTON_SINK = ['|', 'u', 'u', 'gamma_7', 'd', '|']
PROTON_SRC  = ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|']

# Neutron: n = ε_{abc} (d^T C γ₅ d) u  in DR basis
# Two d-quarks form the Cγ₅ diquark, plus one u spectator quark
# Note: pn (proton-neutron) vanishes identically in flavor-conserving QCD
#       because proton (uud) and neutron (udd) have different flavor content.
#       Use nn (neutron-neutron) or pp (proton-proton) instead.
NEUTRON_SINK = ['|', 'd', 'd', 'gamma_7', 'u', '|']
NEUTRON_SRC  = ['|', 'd^d', 'gamma_7', 'd^d', 'u^d', '|']  # conjugate

# 3pt: Proton — Proton (sink), vector current (curr), Nucleon (source)
# Vector current: ū γ_μ d (isovector, flavor-changing)
PROTON_SINK_3PT = ['|', 'u', 'u', 'gamma_7', 'd', '|']
CURR_3PT = ['|', 'u^d', 'gamma_mu', 'd', '|']
PROTON_SRC_3PT = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|']

# 3pt: Pion — Pion (sink), u-quark vector current (curr), Pion (source)
# Current ūγ_μu couples to the u quark (connected piece of EM form factor)
CURR_3PT_U = ['|', 'u^d', 'gamma_mu', 'u', '|']

# Backward-compatible aliases
PJN_SINK = PROTON_SINK_3PT
PJN_CURR = CURR_3PT
PJN_SRC = PROTON_SRC_3PT

# 4pt: PJNNJNp — Proton + Pion (sink), vector current (curr), Nucleon-prime (source)
# Sink has TWO hadrons: neutron ['|','d','d','gamma_7','u','|'] + pion ['|','d^d','gamma_5','u','|']
# This represents a neutron-pion scattering state at the sink
PJNNJNP_SINK = ['|', 'd', 'd', 'gamma_7', 'u', '|', '|', 'd^d', 'gamma_5', 'u', '|']
PJNNJNP_CURR = ['|', 'u^d', 'gamma_mu', 'd', '|']
PJNNJNP_SRC  = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|']

# ═══════════════════════════════════════════════════════════════════
# Analysis Parameters
# ═══════════════════════════════════════════════════════════════════

T_SEP = 12       # Source-sink separation for 3pt
PRECISION = 'complex64'  # Default: single precision (complex64)
                         # Use 'complex128' for double precision

# ═══════════════════════════════════════════════════════════════════
# Output Paths
# ═══════════════════════════════════════════════════════════════════

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PLOTS_DIR = os.path.join(PROJECT_DIR, 'plots')
LOGS_DIR = os.path.join(PROJECT_DIR, 'logs')

# Create output directories
for d in [DATA_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)


def get_eigen_path(conf_id, t):
    """Get eigenvector file path for a given config and time slice.

    Parameters
    ----------
    conf_id : int or str
        Configuration ID.
    t : int
        Time slice index.

    Returns
    -------
    str
        Full path to eigenvector binary file.
    """
    return f"{EIGEN_DIR}/{conf_id}/eigvecs_t{t:03d}_{conf_id}"


def get_peram_dir(conf_id):
    """Get perambulator directory path for a given config.

    Parameters
    ----------
    conf_id : int or str
        Configuration ID.

    Returns
    -------
    str
        Directory containing perambulator binary files.
    """
    return f"{PERAM_DIR}/{conf_id}"


def get_peram_file(conf_id, d_source, t_source):
    """Get perambulator file path.

    Parameters
    ----------
    conf_id : int or str
        Configuration ID.
    d_source : int
        Dirac source index (0..3).
    t_source : int
        Source time slice.

    Returns
    -------
    str
        Full path to perambulator binary file.
    """
    return f"{PERAM_DIR}/{conf_id}/perams.{conf_id}.{d_source}.{t_source}"


def get_gauge_path(conf_id):
    """Get gauge configuration file path.

    Parameters
    ----------
    conf_id : int or str
        Configuration ID.

    Returns
    -------
    str
        Full path to gauge configuration (.lime) file.
    """
    return f"{GAUGE_DIR}/{ENSEMBLE}_cfg_{conf_id}.lime"
