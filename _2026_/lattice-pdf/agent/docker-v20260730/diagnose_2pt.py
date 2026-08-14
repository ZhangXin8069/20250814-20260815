#!/usr/bin/env python3
"""
Quick 2pt diagnostic: test different boundary sign fixes and parity projections
using the pre-computed raw contraction matrix.
"""
import numpy as np

# Load raw contract and gamma matrices
corr_raw = np.load('/root/lattice-pdf/agent/docker-tv20260729/output_20260727_084812/data/conf_6250/twopt_slice_pp_Px0Py0Pz-2_eginphase2_Cg5g4_contract_conf6250.npy')
Nt = 72

# Gamma matrices (DR basis, CPU reference)
gc = {}
gc[0] = np.eye(4, dtype=np.complex128)
g = np.zeros((4,4), dtype=np.complex128)
g[0,3]=1j;g[1,2]=1j;g[2,1]=-1j;g[3,0]=-1j; gc[1]=g.copy()
g.fill(0); g[0,3]=-1;g[1,2]=1;g[2,1]=1;g[3,0]=-1; gc[2]=g.copy()
g.fill(0); g[0,2]=1j;g[1,3]=-1j;g[2,0]=-1j;g[3,1]=1j; gc[3]=g.copy()
g.fill(0); g[0,2]=1;g[1,3]=1;g[2,0]=1;g[3,1]=1; gc[4]=g.copy()
g.fill(0); g[0,0]=1;g[1,1]=1;g[2,2]=-1;g[3,3]=-1; gc[5]=g.copy()
Pp = 0.5 * (gc[0] + gc[4])
Pm = 0.5 * (gc[0] - gc[4])

fm2GeV, alttc = 0.1973, 0.1053

print("=" * 70)
print("DIAGNOSTIC: 2pt correlator analysis")
print("=" * 70)

# ── Parity project raw → PP/PM ──────────────────────────────────────
corr_pp_raw = np.einsum("li,yxil->yx", Pp, corr_raw)
corr_pm_raw = np.einsum("li,yxil->yx", Pm, corr_raw)

print(f"\nRaw PP shape: {corr_pp_raw.shape}")
print(f"Raw PP range: [{corr_pp_raw.real.min():.4e}, {corr_pp_raw.real.max():.4e}]")
print(f"Raw PM range: [{corr_pm_raw.real.min():.4e}, {corr_pm_raw.real.max():.4e}]")

# ── Test 1: NO boundary sign fix ────────────────────────────────────
print("\n" + "-" * 50)
print("Test 1: NO boundary sign fix (raw PP as-is)")
corr_pp_1 = corr_pp_raw.copy()

C2pt_1 = np.zeros(Nt, dtype=np.float64)
C2pt_fwd_1 = np.zeros(Nt, dtype=np.float64)
for dt in range(Nt):
    vals = [np.real(corr_pp_1[(t+dt)%Nt, t]) for t in range(Nt) if abs(corr_pp_1[(t+dt)%Nt, t]) > 1e-30]
    if vals:
        C2pt_1[dt] = np.mean(vals)
    vals_fwd = [np.real(corr_pp_1[(t+dt)%Nt, t]) for t in range(Nt) if abs(corr_pp_1[(t+dt)%Nt, t]) > 1e-30 and (t+dt)%Nt > t]
    if vals_fwd:
        C2pt_fwd_1[dt] = np.mean(vals_fwd)

print(f"Source-avg C2pt (no fix): range [{C2pt_1[2:33].min():.4e}, {C2pt_1[2:33].max():.4e}]")
print(f"Sign changes (2..32): {np.sum(np.diff(np.sign(C2pt_1[2:33])[np.sign(C2pt_1[2:33]) != 0]) != 0)}")

# Compute effective mass (exp_forward only, no cosh)
fm2GeV, alttc = 0.1973, 0.1053
print(f"\nt   C_avg        C_fwd        m_eff_fwd(GeV)")
for dt in range(2, 16):
    c_avg = C2pt_1[dt]
    c_fwd = C2pt_fwd_1[dt]
    if dt < 32 and abs(c_fwd) > 1e-30 and abs(C2pt_fwd_1[dt+1]) > 1e-30:
        ratio = abs(c_fwd / C2pt_fwd_1[dt+1])
        if ratio > 1.0:
            meff = np.log(ratio) * fm2GeV / alttc
            print(f"{dt:2d}  {c_avg:12.4e}  {c_fwd:12.4e}  {meff:10.4f}")
        else:
            print(f"{dt:2d}  {c_avg:12.4e}  {c_fwd:12.4e}  {'ratio<1':>10s}")
    else:
        print(f"{dt:2d}  {c_avg:12.4e}  {c_fwd:12.4e}  {'---':>10s}")

# ── Test 2: Standard donghx boundary sign fix ───────────────────────
print("\n" + "-" * 50)
print("Test 2: Standard donghx boundary sign fix")
corr_pp_2 = corr_pp_raw.copy()
corr_pm_2 = corr_pm_raw.copy()
for ts in range(Nt):
    for tk in range(Nt):
        if tk < ts:
            corr_pp_2[tk, ts] *= -1.0
        if tk > ts:
            corr_pm_2[tk, ts] *= -1.0

C2pt_2 = np.zeros(Nt, dtype=np.float64)
C2pt_fwd_2 = np.zeros(Nt, dtype=np.float64)
for dt in range(Nt):
    vals = [np.real(corr_pp_2[(t+dt)%Nt, t]) for t in range(Nt) if abs(corr_pp_2[(t+dt)%Nt, t]) > 1e-30]
    if vals:
        C2pt_2[dt] = np.mean(vals)
    vals_fwd = [np.real(corr_pp_2[(t+dt)%Nt, t]) for t in range(Nt) if abs(corr_pp_2[(t+dt)%Nt, t]) > 1e-30 and (t+dt)%Nt > t]
    if vals_fwd:
        C2pt_fwd_2[dt] = np.mean(vals_fwd)

print(f"Source-avg C2pt (donghx fix): range [{C2pt_2[2:33].min():.4e}, {C2pt_2[2:33].max():.4e}]")
print(f"Sign changes (2..32): {np.sum(np.diff(np.sign(C2pt_2[2:33])[np.sign(C2pt_2[2:33]) != 0]) != 0)}")

for dt in range(2, 16):
    c_avg = C2pt_2[dt]
    c_fwd = C2pt_fwd_2[dt]
    if dt < 32 and abs(c_fwd) > 1e-30 and abs(C2pt_fwd_2[dt+1]) > 1e-30:
        ratio = abs(c_fwd / C2pt_fwd_2[dt+1])
        if ratio > 1.0:
            meff = np.log(ratio) * fm2GeV / alttc
            print(f"{dt:2d}  {c_avg:12.4e}  {c_fwd:12.4e}  {meff:10.4f}")
        else:
            print(f"{dt:2d}  {c_avg:12.4e}  {c_fwd:12.4e}  {'ratio<1':>10s}")
    else:
        print(f"{dt:2d}  {c_avg:12.4e}  {c_fwd:12.4e}  {'---':>10s}")

# ── Test 3: Folded correlator (standard nucleon analysis) ───────────
print("\n" + "-" * 50)
print("Test 3: Folded correlator C(t)=C_pp(t)+C_pm(Nt-t)")

# Forward folding: C_pp_folded(t) = C_pp(t,0) for t>0
# For source-averaged: each pair with dt contributes
C_folded = np.zeros(Nt, dtype=np.float64)
for dt in range(Nt):
    vals = []
    for t_src in range(Nt):
        t_snk = (t_src + dt) % Nt
        if abs(corr_pp_raw[t_snk, t_src]) > 1e-30:
            vals.append(np.real(corr_pp_raw[t_snk, t_src]))
    if vals:
        C_folded[dt] = np.mean(vals)

# Compute meff via direct ratio
print(f"t   C_folded      m_eff(GeV)")
for dt in range(2, 20):
    if dt < 32:
        r = abs(C_folded[dt] / (C_folded[dt+1] + 1e-30))
        if r > 1.0 and C_folded[dt+1] != 0:
            meff = np.log(r) * fm2GeV / alttc
            print(f"{dt:2d}  {C_folded[dt]:12.4e}  {meff:10.4f}")
        else:
            print(f"{dt:2d}  {C_folded[dt]:12.4e}  {'???':>10s}")

# ── Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print(f"  Test 1 (no fix)    C2pt sign changes: {np.sum(np.diff(np.sign(C2pt_1[2:33])[np.sign(C2pt_1[2:33]) != 0]) != 0)}")
print(f"  Test 2 (donghx fix) C2pt sign changes: {np.sum(np.diff(np.sign(C2pt_2[2:33])[np.sign(C2pt_2[2:33]) != 0]) != 0)}")
print(f"  Test 3 (folded)    C2pt sign changes: {np.sum(np.diff(np.sign(C_folded[2:33])[np.sign(C_folded[2:33]) != 0]) != 0)}")
print("=" * 70)
