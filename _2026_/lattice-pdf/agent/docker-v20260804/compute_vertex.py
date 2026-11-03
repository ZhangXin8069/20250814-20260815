#!/usr/bin/env python3
"""
Vertex computation — VdV and VVV momentum-projected vertices.

VdV_{mn}(P,t) = Σ_x e^{-iP·x} v†_m(x,t) v_n(x,t)
VVV_{abc}(P,t) = Σ_x e^{-iP·x} ε_{abc} v^a(x,t) v^b(x,t) v^c(x,t)

VVV uses the v20260802 2-step einsum (x-direction slicing) for GPU memory efficiency.
"""
from __future__ import annotations
import gc, logging, time
from typing import List, Optional, Tuple
import numpy as np
try: import cupy as cp; HAS_CUPY = True
except ImportError: HAS_CUPY = False; cp = np

def _xp(): return cp if HAS_CUPY else np
def _to_gpu(a): return cp.asarray(a) if HAS_CUPY else np.asarray(a)
def _to_cpu(a):
    if HAS_CUPY and isinstance(a, cp.ndarray): return cp.asnumpy(a)
    return np.asarray(a)

# ═══════════════════════════════════════════════════════════════════════════════
# Phase factors
# ═══════════════════════════════════════════════════════════════════════════════

def _phase_factor(momentum: List[int], Nx: int) -> np.ndarray:
    """φ(x) = exp(-i·2π·P·x/L), flattened (Nx³,) complex128."""
    if all(m==0 for m in momentum):
        return np.ones(Nx**3, dtype=np.complex128)
    coords = np.arange(Nx)
    zz, yy, xx = np.meshgrid(coords, coords, coords, indexing='ij')
    dot = momentum[0]*zz + momentum[1]*yy + momentum[2]*xx
    return np.exp(-2j*np.pi*dot/Nx).ravel()

def generate_momentum_list(Nx: int) -> List[List[int]]:
    ml = [[0,0,0]]
    # Q²=1: 6 permutations with signs
    for perm in [(0,0,1),(0,1,0),(1,0,0)]:
        for pz in [1,-1]:
            for py in [1,-1]:
                for px in [1,-1]:
                    m = [pz*perm[0],py*perm[1],px*perm[2]]
                    if m not in ml and sum(x**2 for x in m)==1: ml.append(m)
    # Q²=2: 12
    for perm in [(0,1,1),(1,0,1),(1,1,0)]:
        for pz in [1,-1]:
            for py in [1,-1]:
                for px in [1,-1]:
                    m = [pz*perm[0],py*perm[1],px*perm[2]]
                    if m not in ml and sum(x**2 for x in m)==2: ml.append(m)
    # Q²=3: 8
    for pz in [1,-1]:
        for py in [1,-1]:
            for px in [1,-1]:
                m = [pz,py,px]
                if m not in ml: ml.append(m)
    return ml

# ═══════════════════════════════════════════════════════════════════════════════
# VdV — Σ_x e^{-iP·x} v†_m(x) v_n(x)  (one einsum per time slice)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_VdV(eigvecs: np.ndarray, momentum_list: List[List[int]],
                Nx: int, logger=None) -> np.ndarray:
    Nt, Nev, Nv, Nc = eigvecs.shape; N_mom = len(momentum_list); V_full = Nv*Nc
    if logger: logger.info(f"VdV: Nt={Nt} Nev={Nev} N_mom={N_mom}")
    # Pre-compute phases on CPU
    ph = np.zeros((N_mom, Nv, Nc), dtype=np.complex128)
    for i, mom in enumerate(momentum_list):
        pf = _phase_factor(mom, Nx).reshape(Nx,Nx,Nx)
        for c in range(3): ph[i,:,c] = pf.ravel()
    ph = ph.reshape(N_mom, V_full)
    ph_g = _to_gpu(ph.astype(np.complex64))
    xp = _xp()
    VdV = np.zeros((Nt, N_mom, Nev, Nev), dtype=np.complex64)
    t0 = time.perf_counter()
    for t in range(Nt):
        ev = _to_gpu(eigvecs[t].astype(np.complex64)).reshape(Nev, V_full)
        VdV[t] = _to_cpu(xp.einsum('bV,MV,cV->Mbc', xp.conj(ev), ph_g, ev, optimize=True))
        del ev
        if t%16==0 and HAS_CUPY: cp.get_default_memory_pool().free_all_blocks()
    if HAS_CUPY: cp.get_default_memory_pool().free_all_blocks()
    if logger:
        logger.info(f"  VdV: shape={VdV.shape} time={time.perf_counter()-t0:.1f}s mem={VdV.nbytes/1024**2:.1f}MB")
        logger.info(f"  VdV(P=0,t=0)|diag|: [{np.abs(np.diag(VdV[0,0])).min():.2e},{np.abs(np.diag(VdV[0,0])).max():.2e}]")
    return VdV

# ═══════════════════════════════════════════════════════════════════════════════
# VVV — Σ_x e^{-iP·x} ε_{abc} v^a v^b v^c  (v20260802 2-step einsum)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_vvv_single_t_gpu(ev_t_gpu, ph_gpu, Nx: int, Nev1: int):
    """VVV for ONE time slice on GPU — 2-step einsum, x-direction slicing."""
    xp = _xp()
    VVV_t = xp.zeros((Nev1, Nev1, Nev1), dtype=ev_t_gpu.dtype)
    L = Nx * Nx  # sites per x-slice
    for xi in range(Nx):
        s, e = xi*L, (xi+1)*L
        es = ev_t_gpu[:Nev1, s:e, :]  # (Nev1, Nx², 3)
        ps = ph_gpu[s:e]              # (Nx²,)
        # Even permutations (cyclic) — ε sign: +1
        T = xp.einsum('x,ax,bx->abx', ps, es[...,0], es[...,1]); VVV_t += xp.einsum('abx,cx->abc', T, es[...,2])
        T = xp.einsum('x,ax,bx->abx', ps, es[...,1], es[...,2]); VVV_t += xp.einsum('abx,cx->abc', T, es[...,0])
        T = xp.einsum('x,ax,bx->abx', ps, es[...,2], es[...,0]); VVV_t += xp.einsum('abx,cx->abc', T, es[...,1])
        # Odd permutations (anti-cyclic) — ε sign: -1
        T = xp.einsum('x,ax,bx->abx', ps, es[...,0], es[...,2]); VVV_t -= xp.einsum('abx,cx->abc', T, es[...,1])
        T = xp.einsum('x,ax,bx->abx', ps, es[...,1], es[...,0]); VVV_t -= xp.einsum('abx,cx->abc', T, es[...,2])
        T = xp.einsum('x,ax,bx->abx', ps, es[...,2], es[...,1]); VVV_t -= xp.einsum('abx,cx->abc', T, es[...,0])
    return VVV_t

def compute_VVV(eigvecs: np.ndarray, momentum_list: List[List[int]],
                Nx: int, Nev1: int = None, logger=None) -> np.ndarray:
    """VVV for all time slices and all momenta (CPU→GPU streaming per slice).

    Returns (Nt, N_mom, Nev1, Nev1, Nev1) CPU array.
    """
    Nt, Nev, Nv, Nc = eigvecs.shape
    if Nev1 is None: Nev1 = Nev
    N_mom = len(momentum_list)
    if logger: logger.info(f"VVV: Nt={Nt} Nev1={Nev1} N_mom={N_mom} Nx={Nx}")
    # Pre-compute all phase factors on CPU
    phases = np.zeros((N_mom, Nv), dtype=np.complex128)
    for i, mom in enumerate(momentum_list):
        phases[i] = _phase_factor(mom, Nx)
    VVV = np.zeros((Nt, N_mom, Nev1, Nev1, Nev1), dtype=np.complex64)
    t0 = time.perf_counter()
    for t in range(Nt):
        ev_t_g = _to_gpu(eigvecs[t].astype(np.complex64))  # (Nev, Nv, 3)
        for m in range(N_mom):
            ph_g = _to_gpu(phases[m].astype(np.complex64))
            VVV[t,m] = _to_cpu(_compute_vvv_single_t_gpu(ev_t_g, ph_g, Nx, Nev1))
            del ph_g
        del ev_t_g
        if t%8==0 and HAS_CUPY: cp.get_default_memory_pool().free_all_blocks()
        if logger and t%8==0:
            elapsed = time.perf_counter()-t0
            logger.info(f"  VVV t={t}/{Nt} ({elapsed:.0f}s) |v|max={np.abs(VVV[t,0]).max():.4e}")
    if HAS_CUPY: cp.get_default_memory_pool().free_all_blocks()
    if logger:
        logger.info(f"  VVV: shape={VVV.shape} time={time.perf_counter()-t0:.1f}s mem={VVV.nbytes/1024**2:.1f}MB")
        logger.info(f"  VVV(P=0,t=0)|v|: [{np.abs(VVV[0,0]).min():.2e},{np.abs(VVV[0,0]).max():.2e}]")
    return VVV

# ═══════════════════════════════════════════════════════════════════════════════
# Convenience
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_vertices(eigvecs, momentum_list, momentum_list_vvv=None, Nx=24, logger=None):
    if momentum_list_vvv is None: momentum_list_vvv = momentum_list
    VdV = compute_VdV(eigvecs, momentum_list, Nx, logger)
    VVV = compute_VVV(eigvecs, momentum_list_vvv, Nx, logger=logger)
    return VdV, VVV
