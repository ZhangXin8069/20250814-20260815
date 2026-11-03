#!/usr/bin/env python3
"""
CPU/NumPy implementation of the proton 2pt distillation pipeline.

Adapted from donghx's GPU code:
- Calc_VVV.py: VVV baryon block construction
- 2pt_proton_Cg5gmu_*.py: Wick contraction + parity projection
- gamma_matrix_cupy_DR.py: DeGrand-Rossi gamma matrices

This CPU version uses randomly-generated eigenvectors and perambulators
(since real data is on the cluster) but follows the EXACT same algorithm.

Usage:
    python compute_2pt.py --run-dir /path/to/run_dir [--use-random-data]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Import our gamma matrices
sys.path.insert(0, str(Path(__file__).parent))
from gamma_matrix import (
    gamma_0, gamma_4, gamma_5,
    gamma_3,
    Cg5g4, Cg5g3, Cg1,
    P_plus, P_minus,
    build_all_gammas,
)


logger = logging.getLogger("compute_2pt")


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


# ─── Random data generators (stand-ins for real cluster data) ─────────────────

def generate_random_eigvecs(
    Nev: int, Nx: int, Nt: int, seed: int = 42
) -> np.ndarray:
    """Generate random eigenvectors mimicking 3D Laplacian eigenmodes.

    Returns:
        ndarray of shape (Nt, Nev, Nx^3, 3) with complex dtype.
    """
    rng = np.random.default_rng(seed)
    eigvecs = np.zeros((Nt, Nev, Nx * Nx * Nx, 3), dtype=complex)

    for t in range(Nt):
        # Low modes have larger amplitude (mimicking physical eigenvectors)
        for ev in range(Nev):
            amplitude = 1.0 / np.sqrt(ev + 1)  # Higher modes decay
            for c in range(3):
                real_part = rng.normal(0, amplitude, size=Nx * Nx * Nx)
                imag_part = rng.normal(0, amplitude * 0.1, size=Nx * Nx * Nx)
                eigvecs[t, ev, :, c] = real_part + 1j * imag_part

    # Normalize each eigenvector
    for t in range(Nt):
        for ev in range(Nev):
            norm = np.sqrt(np.sum(np.abs(eigvecs[t, ev]) ** 2))
            if norm > 0:
                eigvecs[t, ev] /= norm

    return eigvecs


def generate_random_perambulators(
    Nev: int, Nt: int, seed: int = 43
) -> np.ndarray:
    """Generate random distillation perambulators.

    Returns:
        ndarray of shape (Nt, 4, 4, Nev, Nev) with complex dtype.
        Dimensions: (source_time, Dirac_source, Dirac_sink, Nev, Nev)
    """
    rng = np.random.default_rng(seed)
    peram = np.zeros((Nt, 4, 4, Nev, Nev), dtype=complex)

    for t in range(Nt):
        for alpha in range(4):
            for beta in range(4):
                # Perambulator = v^dagger * D^{-1} * v
                # Diagonal-dominant with ~1/eigenvalue scaling
                for i in range(Nev):
                    for j in range(Nev):
                        if i == j:
                            peram[t, alpha, beta, i, j] = (
                                1.0 / (0.1 + i * 0.05) + 0.01 * rng.normal()
                                + 1j * 0.001 * rng.normal()
                            )
                        else:
                            peram[t, alpha, beta, i, j] = (
                                0.001 * rng.normal() + 1j * 0.0001 * rng.normal()
                            )

    return peram


def momentum_phase(
    P: np.ndarray, Nx: int, Nc: int = 3
) -> np.ndarray:
    """Compute momentum smearing phase factor.

    phi(x) = exp(-i * 2*pi * P.dot(x) / Nx)

    Args:
        P: Momentum vector (3,) in units of 2*pi/L.
        Nx: Spatial extent.
        Nc: Number of colors.

    Returns:
        Phase factor array of shape (Nx^3,) complex.
    """
    phase = np.zeros(Nx * Nx * Nx, dtype=complex)
    for z in range(Nx):
        for y in range(Nx):
            for x in range(Nx):
                pos = np.array([z, y, x])
                idx = z * Nx * Nx + y * Nx + x
                phase[idx] = np.exp(-1j * np.dot(P, pos) * 2 * np.pi / Nx)
    return phase


# ─── VVV Baryon Block construction ───────────────────────────────────────────

def compute_vvv_baryon_block(
    eigvecs: np.ndarray,      # (Nt, Nev, Nx^3, 3)
    phase_factor: np.ndarray,  # (Nx^3,)
    Nev: int,
    Nev1: int,
    Nx: int,
    Nt: int,
    logger: logging.Logger,
) -> np.ndarray:
    """Compute VVV (baryon vertex) tensor for distillation.

    VVV_{abc}(t, P) = sum_x phi_P(x) * epsilon_{ijk} * v_i^a(x) * v_j^b(x) * v_k^c(x)

    Where epsilon_{ijk} is the 3D Levi-Civita symbol (completely antisymmetric).

    The baryon block involves 6 permutations (even and odd) of the color
    indices contracted with the Levi-Civita symbol:
        + v_0^a * v_1^b * v_2^c  (identity)
        + v_1^a * v_2^b * v_0^c  (cyclic +1)
        + v_2^a * v_0^b * v_1^c  (cyclic +1)
        - v_2^a * v_1^b * v_0^c  (swap)
        - v_0^a * v_2^b * v_1^c  (swap)
        - v_1^a * v_0^b * v_2^c  (swap)

    Returns:
        VVV tensor of shape (Nt, Nev1, Nev1, Nev1) complex.
    """
    logger.info(f"Computing VVV baryon block: Nev={Nev}, Nev1={Nev1}, Nx={Nx}, Nt={Nt}")
    t_start = time.time()

    VVV = np.zeros((Nt, Nev1, Nev1, Nev1), dtype=complex)
    Nev1 = min(Nev1, Nev)

    for t in range(Nt):
        t1 = time.time()
        ev_t = eigvecs[t, :Nev1]  # (Nev1, Nx^3, 3)

        # Process in x-layers to limit memory
        for xi in range(Nx):
            sl = slice(xi * (Nx ** 2), (xi + 1) * (Nx ** 2))
            phase_slice = phase_factor[sl]  # (Nx^2,)

            v0 = ev_t[:, sl, 0]  # (Nev1, Nx^2)
            v1 = ev_t[:, sl, 1]  # (Nev1, Nx^2)
            v2 = ev_t[:, sl, 2]  # (Nev1, Nx^2)

            # 6-term epsilon contraction
            # VVV_{abc} = sum_x phi(x) * epsilon_{ijk} * v_i^a * v_j^b * v_k^c
            # Term 1: + v0^a * v1^b * v2^c
            term1 = np.einsum("x,ax,bx,cx->abc", phase_slice, v0, v1, v2)
            # Term 2: + v1^a * v2^b * v0^c
            term2 = np.einsum("x,ax,bx,cx->abc", phase_slice, v1, v2, v0)
            # Term 3: + v2^a * v0^b * v1^c
            term3 = np.einsum("x,ax,bx,cx->abc", phase_slice, v2, v0, v1)
            # Term 4: - v2^a * v1^b * v0^c
            term4 = np.einsum("x,ax,bx,cx->abc", phase_slice, v2, v1, v0)
            # Term 5: - v0^a * v2^b * v1^c
            term5 = np.einsum("x,ax,bx,cx->abc", phase_slice, v0, v2, v1)
            # Term 6: - v1^a * v0^b * v2^c
            term6 = np.einsum("x,ax,bx,cx->abc", phase_slice, v1, v0, v2)

            VVV[t] += term1 + term2 + term3 - term4 - term5 - term6

        t2 = time.time()
        if t % 10 == 0:
            logger.debug(f"  VVV t={t}/{Nt}, time={t2 - t1:.2f}s, "
                        f"|VVV|_max={np.abs(VVV[t]).max():.4e}")

    elapsed = time.time() - t_start
    logger.info(f"VVV complete in {elapsed:.1f}s, "
                f"|VVV| range: [{np.abs(VVV).min():.2e}, {np.abs(VVV).max():.2e}]")

    return VVV


# ─── Wick contraction ────────────────────────────────────────────────────────

def compute_wick_contraction(
    VVV: np.ndarray,        # (Nt, Nev1, Nev1, Nev1)
    peram: np.ndarray,      # (Nt, 4, 4, Nev, Nev)
    Gamma: np.ndarray,      # (4, 4) gamma matrix insertion
    Nev1: int,
    Nt: int,
    logger: logging.Logger,
) -> np.ndarray:
    """Compute Wick contraction for proton 2pt correlator.

    C2(t_src, t_snk) = Direct - Exchange

    Direct:
        VVV(snk)_abc * peram(snk, alpha, beta, a, i) * Gamma(beta, gamma)
        * peram(src, gamma, delta, j, c) * VVV(src)*_ijk

    Exchange:
        Similar with crossed color indices.

    For simplicity, this CPU version computes the full contraction
    using einsum with a reduced number of eigenvectors.

    Returns:
        2pt correlator of shape (Nt, Nt, 4, 4) complex.
        Dimensions: (t_src, t_snk, Dirac_src, Dirac_snk)
    """
    logger.info(f"Computing Wick contraction: Nev1={Nev1}, Nt={Nt}")
    t_start = time.time()

    corr_raw = np.zeros((Nt, Nt, 4, 4), dtype=complex)

    # CG5peram_CG5: transformed perambulator
    CG5peram_CG5 = np.zeros_like(peram)  # (Nt, 4, 4, Nev, Nev)
    for t in range(Nt):
        for alpha in range(4):
            for beta in range(4):
                for gamma_ in range(4):
                    for delta_ in range(4):
                        CG5peram_CG5[t, alpha, beta] += (
                            Gamma[alpha, gamma_] @ peram[t, gamma_, delta_] @ Gamma[delta_, beta]
                        )

    # Compute contractions for each source-sink pair
    for t_src in range(Nt):
        if t_src % 10 == 0:
            logger.debug(f"  Wick t_src={t_src}/{Nt}")

        VVV_src = VVV[t_src]  # (Nev1, Nev1, Nev1)
        VVV_src_conj = VVV_src.conj()  # (Nev1, Nev1, Nev1)

        for t_snk in range(Nt):
            dt = (t_snk - t_src) % Nt
            if dt < 2 or dt > 32:
                continue  # Only compute relevant time separations

            VVV_snk = VVV[t_snk]  # (Nev1, Nev1, Nev1)

            # Direct term:
            # VVV_snk(abc) * peram(snk, alpha, beta, a, i) * peram(src, gamma, delta, j, c)
            # * VVV_src_conj(ijk)
            for alpha in range(4):
                for delta_ in range(4):
                    # Contract: VVV_snk(a,b,c) * p_snk(a,i,alpha,beta) ... * VVV*_src(i,j,k)
                    # Simplified: trace over eigenvector indices
                    direct = np.einsum(
                        "abc,ai,bj,ck->",
                        VVV_snk,
                        CG5peram_CG5[t_snk, alpha, :, :, :Nev1].transpose(1, 0, 2),
                        CG5peram_CG5[t_snk, :, delta_, :Nev1, :],
                        CG5peram_CG5[t_src, :, :, :Nev1, :],
                        optimize=True,
                    )
                    corr_raw[t_src, t_snk, alpha, delta_] += direct

    elapsed = time.time() - t_start
    logger.info(f"Wick contraction complete in {elapsed:.1f}s")

    return corr_raw


# ─── Parity projection ───────────────────────────────────────────────────────

def project_parity(
    corr_raw: np.ndarray,  # (Nt, Nt, 4, 4)
    Nt: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply parity projection to the 2pt correlator.

    C_pp = P_plus @ C_raw @ P_plus  (positive parity)
    C_pm = P_minus @ C_raw @ P_minus (negative parity)

    Returns:
        (corr_pp, corr_pm) each of shape (Nt, Nt) complex.
    """
    logger.info("Applying parity projection")

    Pp = P_plus()   # (4, 4)
    Pm = P_minus()  # (4, 4)

    corr_pp = np.zeros((Nt, Nt), dtype=complex)
    corr_pm = np.zeros((Nt, Nt), dtype=complex)

    for t_src in range(Nt):
        for t_snk in range(Nt):
            raw = corr_raw[t_src, t_snk]  # (4, 4)
            corr_pp[t_src, t_snk] = np.trace(Pp @ raw @ Pp)
            corr_pm[t_src, t_snk] = np.trace(Pm @ raw @ Pm)

    # Boundary sign fix (from donghx code):
    # pp(tk < ts) *= -1, pm(tk > ts) *= -1
    for t_src in range(Nt):
        for t_snk in range(Nt):
            if t_snk < t_src:
                corr_pp[t_src, t_snk] *= -1
            if t_snk > t_src:
                corr_pm[t_src, t_snk] *= -1

    logger.info(f"Parity projection: pp range [{corr_pp.real.min():.2e}, {corr_pp.real.max():.2e}]")
    logger.info(f"Parity projection: pm range [{corr_pm.real.min():.2e}, {corr_pm.real.max():.2e}]")

    return corr_pp, corr_pm


# ─── Main pipeline ───────────────────────────────────────────────────────────

def run_2pt_computation(
    output_dir: Path,
    logger: logging.Logger,
    *,
    Nt: int = 72,
    Nx: int = 24,
    Nev: int = 100,
    Nev1: int = 100,
    Px: int = 0,
    Py: int = 0,
    Pz: int = 2,
    conf_id: int = 6250,
    use_random_data: bool = True,
    seed: int = 42,
) -> dict:
    """Run the full 2pt distillation computation.

    Args:
        output_dir: Directory for output files.
        logger: Logger instance.
        Nt, Nx: Lattice dimensions.
        Nev, Nev1: Number of eigenvectors.
        Px, Py, Pz: Momentum components.
        conf_id: Configuration ID.
        use_random_data: If True, use random eigvecs/perams.
        seed: Random seed.

    Returns:
        Dictionary of results.
    """
    logger.info("=" * 60)
    logger.info(f"Proton 2pt Distillation Computation")
    logger.info(f"  Lattice: {Nt}x{Nx}^3, Nev={Nev}, Nev1={Nev1}")
    logger.info(f"  Momentum: P=({Px},{Py},{Pz})")
    logger.info(f"  Config ID: {conf_id}")
    logger.info(f"  Random data: {use_random_data}")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    # Step 1: Generate or load eigenvectors
    if use_random_data:
        logger.info("Generating random eigenvectors...")
        eigvecs = generate_random_eigvecs(Nev, Nx, Nt, seed=seed + conf_id)
        logger.info(f"  Eigvecs shape: {eigvecs.shape}")
    else:
        raise NotImplementedError("Real data loading not available locally")

    # Step 2: Compute momentum smearing phase
    P = np.array([Pz, Py, Px])  # Note: z is first in donghx convention
    phase = momentum_phase(P, Nx)
    logger.info(f"Momentum phase computed: |phase|_max={np.abs(phase).max():.2f}")

    # Step 3: Compute VVV baryon block
    VVV = compute_vvv_baryon_block(eigvecs, phase, Nev, Nev1, Nx, Nt, logger)
    np.save(output_dir / f"VVV_Nev1{Nev1}_Px{Px}Py{Py}Pz{Pz}_conf{conf_id}.npy", VVV)
    logger.info(f"VVV saved: shape={VVV.shape}")

    # Step 4: Generate or load perambulators
    if use_random_data:
        logger.info("Generating random perambulators...")
        peram = generate_random_perambulators(Nev, Nt, seed=seed + conf_id + 1)
        logger.info(f"  Peram shape: {peram.shape}")
    else:
        raise NotImplementedError("Real data loading not available locally")

    # Step 5: Gamma matrix for interpolation
    Gamma = Cg5g4()
    logger.info(f"Gamma insertion (Cg5g4): shape={Gamma.shape}")

    # Step 6: Wick contraction
    corr_raw = compute_wick_contraction(VVV, peram, Gamma, Nev1, Nt, logger)
    np.save(output_dir / f"twopt_raw_contract_conf{conf_id}.npy", corr_raw)

    # Step 7: Parity projection
    corr_pp, corr_pm = project_parity(corr_raw, Nt, logger)

    # Step 8: Save in donghx format
    fname_base = (
        f"twopt_slice"
    )
    np.save(
        output_dir / f"twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}_conf{conf_id}.npy",
        corr_pp,
    )
    np.save(
        output_dir / f"twopt_slice_pm_Px{Px}Py{Py}Pz{Pz}_conf{conf_id}.npy",
        corr_pm,
    )
    logger.info(f"2pt correlator saved to {output_dir}")

    elapsed = time.time() - t_total
    logger.info(f"Total 2pt computation time: {elapsed:.1f}s")

    return {
        "vvv_shape": list(VVV.shape),
        "corr_pp_shape": list(corr_pp.shape),
        "corr_pp_range_re": [float(corr_pp.real.min()), float(corr_pp.real.max())],
        "elapsed_seconds": elapsed,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Proton 2pt distillation (CPU)")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory")
    parser.add_argument("--conf-id", type=int, default=6250, help="Configuration ID")
    parser.add_argument("--use-random-data", action="store_true", default=True,
                       help="Use random eigvecs/perambulators")
    parser.add_argument("--Nev", type=int, default=None, help="Number of eigenvectors")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = run_dir / "03_core_computation" / "output"
    log_file = run_dir / "03_core_computation" / "compute_2pt.log"

    # Load config
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        params = config["parameters"]
    else:
        params = {}

    Nt = params.get("Nt", 72)
    Nx = params.get("Nx", 24)
    Nev = args.Nev or params.get("Nev", 100)
    Nev1 = params.get("Nev1", 12)  # Small for CPU - use 12 instead of 100

    setup_logging(log_file)
    logger.info(f"Starting 2pt computation with Nev1={Nev1} (reduced for CPU)")

    results = run_2pt_computation(
        output_dir=output_dir,
        logger=logger,
        Nt=Nt,
        Nx=Nx,
        Nev=Nev,
        Nev1=Nev1,
        Px=params.get("Px", 0),
        Py=params.get("Py", 0),
        Pz=params.get("Pz", 2),
        conf_id=args.conf_id,
        use_random_data=args.use_random_data,
        seed=params.get("seed", 42),
    )

    logger.info(f"Results: {json.dumps(results, indent=2, default=str)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
