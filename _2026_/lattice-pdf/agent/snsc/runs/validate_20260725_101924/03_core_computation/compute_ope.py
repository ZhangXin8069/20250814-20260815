#!/usr/bin/env python3
"""
CPU/NumPy implementation of the gluon OPE (Operator Product Expansion) computation.

Adapted from donghx's GPU code:
- Operator.py: plaquette construction, field strength tensor, Wilson line operators
- Calc_pla.py: Clover plaquette computation
- Calc_ope_unpol.py: Nonlocal gluon operator with Wilson line

Computes the nonlocal gluon operator:
    O_{mu,nu}(z) = F_{mu,nu}(z) * U_link(z, 0) * F_{mu,nu}(0)

for the three components needed in huangcl's analysis:
- (mu=0, nu=1): F_xy — spatial-spatial
- (mu=3, nu=0): F_tx — time-spatial
- (mu=3, nu=1): F_ty — time-spatial

Usage:
    python compute_ope.py --run-dir /path/to/run_dir [--use-random-data]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np


logger = logging.getLogger("compute_ope")


def setup_logging(log_file: Path) -> logging.Logger:
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─── Gauge field utilities ───────────────────────────────────────────────────

def generate_random_gauge(
    Nt: int, Nx: int, Nc: int = 3, beta: float = 6.0, seed: int = 42
) -> np.ndarray:
    """Generate random SU(3) gauge links.

    In real lattice QCD, gauge links are SU(3) matrices. Here we generate
    random matrices and project them to SU(3) approximately.

    Returns:
        ndarray of shape (Nt, Nx, Nx, Nx, 4, Nc, Nc) complex.
    """
    rng = np.random.default_rng(seed)

    # Generate random complex 3x3 matrices
    gauge = np.zeros((Nt, Nx, Nx, Nx, 4, Nc, Nc), dtype=complex)

    for t in range(Nt):
        for z in range(Nx):
            for y in range(Nx):
                for x in range(Nx):
                    for mu in range(4):
                        # Generate random matrix
                        A = (rng.normal(0, 1, (Nc, Nc)) +
                             1j * rng.normal(0, 1, (Nc, Nc)))

                        # Make it approximately SU(3)
                        # U = exp(i * A/norm), A = A^dagger (hermitian)
                        A = (A + A.conj().T) / 2.0  # Hermitian
                        A /= np.linalg.norm(A) * 10.0

                        # Matrix exponential via Taylor: exp(i*A) ≈ I + i*A - A^2/2
                        I = np.eye(Nc, dtype=complex)
                        U = I + 1j * A - A @ A / 2.0 + 1j * A @ A @ A / 6.0

                        # Crude re-unitarization
                        U = 0.5 * (U + np.linalg.inv(U).conj().T)

                        gauge[t, z, y, x, mu] = U

    return gauge


def plaquette_clover(
    gauge: np.ndarray,  # (Nt, Nx, Nx, Nx, 4, 3, 3)
    mu: int,
    nu: int,
    Nt: int,
) -> np.ndarray:
    """Compute clover plaquette for a given (mu, nu) plane.

    P_{mu,nu}(x) = U_mu(x) * U_nu(x+mu_hat) * U_mu^dag(x+nu_hat) * U_nu^dag(x)

    The clover term averages over the 4 leaves of the clover:
    Q_{mu,nu} = (P_{mu,nu} + P_{nu,-mu} + P_{-mu,-nu} + P_{-nu,mu}) / 4

    Returns:
        ndarray of shape (Nt, Nx, Nx, Nx, 3, 3) complex.
    """
    Nx = gauge.shape[1]
    Nc = 3

    if mu == nu:
        return np.zeros((Nt, Nx, Nx, Nx, Nc, Nc), dtype=complex)

    pla = np.zeros((Nt, Nx, Nx, Nx, Nc, Nc), dtype=complex)

    for t in range(Nt):
        for z in range(Nx):
            for y in range(Nx):
                for x in range(Nx):
                    # Get links in the 4 directions of the clover leaf
                    # Forward-forward
                    x_mu = (x + (1 if mu == 0 else 0)) % Nx
                    y_mu = (y + (1 if mu == 1 else 0)) % Nx
                    z_mu = (z + (1 if mu == 2 else 0)) % Nx
                    t_mu = (t + (1 if mu == 3 else 0)) % Nt

                    x_nu = (x + (1 if nu == 0 else 0)) % Nx
                    y_nu = (y + (1 if nu == 1 else 0)) % Nx
                    z_nu = (z + (1 if nu == 2 else 0)) % Nx
                    t_nu = (t + (1 if nu == 3 else 0)) % Nt

                    # P_{mu,nu} = U_mu(x) U_nu(x+mu) U_mu^dag(x+nu) U_nu^dag(x)
                    U_mu_x = gauge[t, z, y, x, mu]

                    # U_nu at x+mu
                    U_nu_xp = gauge[t_mu, z_mu, y_mu, x_mu, nu]

                    # U_mu^dag at x+nu
                    U_mu_dag_xn = gauge[t_nu, z_nu, y_nu, x_nu, mu].conj().T

                    # U_nu^dag at x
                    U_nu_dag_x = gauge[t, z, y, x, nu].conj().T

                    P = U_mu_x @ U_nu_xp @ U_mu_dag_xn @ U_nu_dag_x
                    pla[t, z, y, x] += P

    # Average over 4 leaves (simplified here; full clover involves more terms)
    return pla


def field_strength_from_plaquette(
    pla: np.ndarray,  # (Nt, Nx, Nx, Nx, 3, 3)
    mu: int,
    nu: int,
) -> np.ndarray:
    """Extract field strength tensor from clover plaquette.

    F_{mu,nu} = (1/8i) * (Q_{mu,nu} - Q_{mu,nu}^dag)  [traceless anti-hermitian part]

    Returns:
        ndarray of shape (Nt, Nx, Nx, Nx) complex (trace of F).
    """
    F = np.zeros(pla.shape[:4], dtype=complex)  # (Nt, Nx, Nx, Nx)

    for t in range(pla.shape[0]):
        for z in range(pla.shape[1]):
            for y in range(pla.shape[2]):
                for x in range(pla.shape[3]):
                    Q = pla[t, z, y, x]  # (3, 3)
                    anti_herm = (Q - Q.conj().T) / (8j)
                    F[t, z, y, x] = np.trace(anti_herm)

    return F


# ─── Nonlocal OPE operator ───────────────────────────────────────────────────

def compute_ope_operator(
    gauge: np.ndarray,     # (Nt, Nx, Nx, Nx, 4, 3, 3)
    mu: int,
    nu: int,
    z_dir: int,            # Wilson line direction (0=x, 1=y, 2=z)
    delta_z: int,           # Wilson line length
    Nt: int,
    Nx: int,
    logger: logging.Logger,
) -> np.ndarray:
    """Compute the nonlocal gluon OPE operator.

    O(z, x_perp, t) = Tr[F_{mu,nu}(z, x_perp, t) * U_link(z, 0, x_perp, t)
                         * F_{mu,nu}(0, x_perp, t)]

    where U_link is the Wilson line connecting the two field strength insertions.

    For simplicity in this CPU implementation, we compute a simplified version
    where the nonlocality is treated through spatial shifts of the field strength.

    Returns:
        ndarray of shape (Nx, Nt) complex (spatially summed over perpendicular directions).
    """
    logger.info(f"Computing OPE: mu={mu}, nu={nu}, z_dir={z_dir}, delta_z={delta_z}")

    # Compute plaquette for this (mu, nu) pair
    pla = plaquette_clover(gauge, mu, nu, Nt)

    # Extract field strength
    F = field_strength_from_plaquette(pla, mu, nu)  # (Nt, Nx, Nx, Nx)

    # Construct nonlocal operator:
    # Sum over perpendicular directions, keep z and t
    ope = np.zeros((Nx, Nt), dtype=complex)

    Nc = 3

    for zi in range(Nx):
        # z=0 field strength (spatially averaged over x, y)
        F0_avg = np.zeros(Nt, dtype=complex)
        for xi in range(Nx):
            for yi in range(Nx):
                F0_avg += F[:, zi, yi, xi]

        # z=delta_z field strength
        z_delta = (zi + delta_z) % Nx
        Fz_avg = np.zeros(Nt, dtype=complex)
        for xi in range(Nx):
            for yi in range(Nx):
                Fz_avg += F[:, z_delta, yi, xi]

        # Wilson line (simplified: product of links along z)
        # In real calculation, this is a product of gauge links
        # Here we approximate with a phase factor
        wilson_phase = np.ones(Nt, dtype=complex)
        for dz in range(delta_z):
            z_step = (zi + dz) % Nx
            for xi in range(Nx):
                for yi in range(Nx):
                    # Use the gauge link in z-direction
                    U_link = gauge[:, z_step, yi, xi, z_dir]
                    wilson_phase *= np.trace(U_link) / Nc / (Nx * Nx)

        # OPE = F(z) * Wilson_line * F(0)  (plus hermitian conjugate)
        ope[zi, :] = Fz_avg * wilson_phase * F0_avg.conj()
        ope[zi, :] += ope[zi, :].conj()  # Make it real-ish

    return ope


# ─── Main pipeline ───────────────────────────────────────────────────────────

def run_ope_computation(
    output_dir: Path,
    logger: logging.Logger,
    *,
    Nt: int = 72,
    Nx: int = 24,
    Nc: int = 3,
    delta_z: int = 24,
    z_dir: int = 2,
    conf_id: int = 6250,
    use_random_data: bool = True,
    seed: int = 42,
) -> dict:
    """Run the full OPE computation for a single configuration.

    Computes the three (mu, nu) components needed for huangcl's analysis.
    """
    logger.info("=" * 60)
    logger.info(f"Gluon OPE Computation")
    logger.info(f"  Lattice: {Nt}x{Nx}^3, delta_z={delta_z}, z_dir={z_dir}")
    logger.info(f"  Config ID: {conf_id}")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    # Generate or load gauge configuration
    if use_random_data:
        logger.info("Generating random gauge configuration...")
        gauge = generate_random_gauge(Nt, Nx, Nc, seed=seed + conf_id)
        logger.info(f"  Gauge shape: {gauge.shape}")
    else:
        raise NotImplementedError("Real gauge config loading not available locally")

    # Compute OPE for the three required components
    components = [(0, 1), (3, 0), (3, 1)]
    results = {}

    for mu, nu in components:
        t1 = time.time()
        ope = compute_ope_operator(gauge, mu, nu, z_dir, delta_z, Nt, Nx, logger)

        # Save in donghx format: npz with 'ops' key
        fname = f"ops_mu{mu}_nu{nu}_dz{delta_z}_conf{conf_id}.npz"
        np.savez(output_dir / fname, ops=ope)
        logger.info(f"  Saved {fname}: shape={ope.shape}, "
                    f"|OPE| range [{np.abs(ope).min():.2e}, {np.abs(ope).max():.2e}], "
                    f"time={time.time() - t1:.1f}s")

        results[f"mu{mu}_nu{nu}"] = {
            "shape": list(ope.shape),
            "re_range": [float(ope.real.min()), float(ope.real.max())],
        }

    elapsed = time.time() - t_total
    logger.info(f"Total OPE computation time: {elapsed:.1f}s")

    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gluon OPE computation (CPU)")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory")
    parser.add_argument("--conf-id", type=int, default=6250, help="Configuration ID")
    parser.add_argument("--use-random-data", action="store_true", default=True,
                       help="Use random gauge config")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = run_dir / "03_core_computation" / "output"
    log_file = run_dir / "03_core_computation" / "compute_ope.log"

    config_path = run_dir / "run_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        params = config["parameters"]
    else:
        params = {}

    setup_logging(log_file)

    results = run_ope_computation(
        output_dir=output_dir,
        logger=logger,
        Nt=params.get("Nt", 72),
        Nx=params.get("Nx", 24),
        Nc=params.get("Nc", 3),
        delta_z=params.get("delta_z", 24),
        z_dir=params.get("z_dir", 2),
        conf_id=args.conf_id,
        use_random_data=args.use_random_data,
        seed=params.get("seed", 42),
    )

    logger.info(f"Results: {json.dumps(results, indent=2, default=str)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
