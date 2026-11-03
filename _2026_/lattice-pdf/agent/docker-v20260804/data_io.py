#!/usr/bin/env python3
"""
Data I/O — reads eigenvectors and perambulators from HPC cluster filesystem.

CORRECTED perambulator format (v20260802 convention):
  - Raw binary: perams.{cid}.{dsrc}.{tsrc} → LE float64, shape (Nt,Nev,4,Nev,2)
  - Combined for einsum: combine 4 dsrc files per t_src → transpose to (Nt,4,4,Nev,Nev)
    where axes = (tsnk, spin_snk, spin_src, Nev_snk, Nev_src)

This matches sush's lqcddb einsum convention where peram[t] has shape (4,4,Nev,Nev).
"""
from __future__ import annotations

import os, time
from typing import Optional
import numpy as np

try: import cupy; HAS_CUPY = True
except ImportError: HAS_CUPY = False

def _dtype(): from utils import get_compute_dtype; return get_compute_dtype()

# ═══════════════════════════════════════════════════════════════════════════════
# Eigenvector reader (unchanged — was correct)
# ═══════════════════════════════════════════════════════════════════════════════

def read_eigenvector_slice(filepath: str, Nev: int, Nx: int) -> np.ndarray:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Eigenvector file not found: {filepath}")
    raw = np.fromfile(filepath, dtype='<f8')
    Nc, Nv = 3, Nx**3
    expected = Nev * Nv * Nc * 2
    if raw.size != expected:
        Nev_full = (raw.size//2)//(Nv*Nc)
        if Nev > Nev_full:
            raise ValueError(f"Requested Nev={Nev} but file has Nev_full={Nev_full}")
        raw = raw[:expected]
    raw = raw.reshape(Nev, Nv, Nc, 2)
    return raw[...,0] + 1j*raw[...,1]

def load_eigenvectors(eigvec_base: str, conf_id: int, Nev: int, Nt: int, Nx: int, logger) -> np.ndarray:
    eigvec_dir = os.path.join(eigvec_base, str(conf_id))
    dt = _dtype(); Nv = Nx**3
    logger.info(f"Loading eigenvectors for conf={conf_id} from {eigvec_dir}")
    t0 = time.perf_counter()
    if not os.path.isdir(eigvec_dir):
        raise FileNotFoundError(f"Eigenvector directory not found: {eigvec_dir}")
    first_file = os.path.join(eigvec_dir, f"eigvecs_t000_{conf_id}")
    if not os.path.exists(first_file):
        raise FileNotFoundError(f"No eigenvector files in {eigvec_dir}")
    raw = np.fromfile(first_file, dtype='<f8')
    Nev_full = (raw.size//2)//(Nv*3); Nev_use = min(Nev, Nev_full)
    logger.info(f"  Nev_full={Nev_full}, using Nev={Nev_use}")
    eigvecs = np.zeros((Nt, Nev_use, Nv, 3), dtype=dt)
    missing = []
    for t in range(Nt):
        fp = os.path.join(eigvec_dir, f"eigvecs_t{t:03d}_{conf_id}")
        if not os.path.exists(fp): missing.append(t); continue
        eigvecs[t] = read_eigenvector_slice(fp, Nev_use, Nx).astype(dt, copy=False)
    if missing:
        logger.warning(f"  Missing {len(missing)}/{Nt} time slices")
        if len(missing) > Nt//2: raise FileNotFoundError(f"Too many missing eigenvector files for conf={conf_id}")
    elapsed = time.perf_counter()-t0
    logger.info(f"  Loaded: shape={eigvecs.shape}, mem={eigvecs.nbytes/1024**2:.1f} MB, time={elapsed:.1f}s")
    assert np.all(np.isfinite(eigvecs))
    return eigvecs

# ═══════════════════════════════════════════════════════════════════════════════
# Perambulator reader — CORRECTED to combine 4 dsrc → (Nt,4,4,Nev,Nev)
# ═══════════════════════════════════════════════════════════════════════════════

def _read_peram_file(filepath: str, Nev: int, Nt: int) -> np.ndarray:
    """Read one perambulator binary file → (Nt, Nev, 4, Nev) complex128."""
    raw = np.fromfile(filepath, dtype='<f8')
    expected = Nt * Nev * 4 * Nev * 2
    if raw.size != expected:
        raise ValueError(f"Peram {filepath}: expected {expected} floats, got {raw.size}")
    raw = raw.reshape(Nt, Nev, 4, Nev, 2)
    return raw[...,0] + 1j*raw[...,1]

def read_perambulator_single_t(peram_dir: str, conf_id: int, t_source: int,
                                Nev: int, Nt: int, logger) -> np.ndarray:
    """Read perambulator for ONE source time slice, all 4 Dirac sources.

    Combines: perams.{conf_id}.{dsrc}.{t_source} for dsrc=0..3
    Raw each: (Nt, Nev, 4, Nev) = (tsnk, Nev_snk, spin_snk, Nev_src)
    Stacked:   (4, Nt, Nev, 4, Nev) = (dsrc, tsnk, Nev_snk, spin_snk, Nev_src)
    Transposed: (Nt, 4, 4, Nev, Nev) = (tsnk, spin_snk, dsrc=spin_src, Nev_snk, Nev_src)

    THIS MATCHES sush's einsum convention: peram[t] → (4,4,Nev,Nev).
    """
    dt = _dtype()
    # peram_dir is the BASE path (e.g. .../light/), files are in {peram_dir}/{conf_id}/
    peram_conf_dir = os.path.join(peram_dir, str(conf_id))
    parts = []
    for dsrc in range(4):
        fn = os.path.join(peram_conf_dir, f"perams.{conf_id}.{dsrc}.{t_source}")
        if not os.path.exists(fn):
            raise FileNotFoundError(f"Peram not found: {fn}")
        p = _read_peram_file(fn, Nev, Nt)  # (Nt, Nev, 4, Nev)
        parts.append(p)
    # Stack: (4, Nt, Nev, 4, Nev) → transpose to (Nt, 4, 4, Nev, Nev)
    peram = np.stack(parts, axis=0)  # (4, Nt, Nev, 4, Nev)
    peram = peram.transpose(1, 3, 0, 4, 2)  # → (Nt, spin_snk=4, dsrc=4, Nev_src, Nev_snk)
    peram = peram.astype(dt, copy=False)
    assert np.all(np.isfinite(peram)), f"Peram t_src={t_source} contains NaN/inf"
    return peram
