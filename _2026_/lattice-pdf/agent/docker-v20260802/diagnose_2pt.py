#!/usr/bin/env python3
"""
2pt correlator diagnostic: test different boundary sign fixes and parity projections.

v20260802: Fixed from v20260730 — removed hardcoded path to docker-tv20260729.
Now accepts --contract-file argument or auto-discovers from the latest output directory.

Usage:
    python diagnose_2pt.py --contract-file /path/to/contract.npy
    python diagnose_2pt.py --run-dir /path/to/output_dir --conf-id 6250
"""
import argparse, os, sys, json
import numpy as np
from pathlib import Path


def run_diagnostics(corr_raw: np.ndarray, Nt: int = 72, alttc: float = 0.1053):
    """Run all 2pt diagnostics on the raw contraction matrix.

    Tests three boundary condition approaches:
      1. No boundary sign fix
      2. Standard donghx anti-periodic BC fix
      3. Folded correlator C(t)=C_pp(t)+C_pm(Nt-t)
    """
    # Gamma matrices (DR basis, CPU reference — same as gamma_matrix_gpu.py)
    gc = {}
    gc[0] = np.eye(4, dtype=np.complex128)
    g = np.zeros((4,4), dtype=np.complex128)
    g[0,3]=1j;g[1,2]=1j;g[2,1]=-1j;g[3,0]=-1j; gc[1]=g.copy()
    g.fill(0); g[0,3]=-1;g[1,2]=1;g[2,1]=1;g[3,0]=-1; gc[2]=g.copy()
    g.fill(0); g[0,2]=1j;g[1,3]=-1j;g[2,0]=-1j;g[3,1]=1j; gc[3]=g.copy()
    g.fill(0); g[0,2]=1;g[1,3]=1;g[2,0]=1;g[3,1]=1; gc[4]=g.copy()
    g.fill(0); g[0,0]=1;g[1,1]=1;g[2,2]=-1;g[3,3]=-1; gc[5]=g.copy()
    Pp = 0.5 * (gc[0] + gc[4])  # P₊ = (γ₀+γ₄)/2
    Pm = 0.5 * (gc[0] - gc[4])  # P₋ = (γ₀-γ₄)/2

    fm2GeV, alttc_val = 0.1973, alttc

    print("=" * 70)
    print("DIAGNOSTIC: 2pt correlator analysis (v20260802)")
    print(f"  Input shape: {corr_raw.shape}")
    print(f"  Nt={Nt}, a={alttc_val} fm")
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
    n_sign = np.sum(np.diff(np.sign(C2pt_1[2:33])[np.sign(C2pt_1[2:33]) != 0]) != 0)
    print(f"Sign changes (t=2..32): {n_sign}")

    print(f"\n{'t':>3s}  {'C_avg':>14s}  {'C_fwd':>14s}  {'m_eff_fwd(GeV)':>15s}")
    for dt in range(2, 16):
        c_avg = C2pt_1[dt]
        c_fwd = C2pt_fwd_1[dt]
        if dt < 32 and abs(c_fwd) > 1e-30 and abs(C2pt_fwd_1[dt+1]) > 1e-30:
            ratio = abs(c_fwd / C2pt_fwd_1[dt+1])
            if ratio > 1.0:
                meff = np.log(ratio) * fm2GeV / alttc_val
                print(f"{dt:3d}  {c_avg:14.4e}  {c_fwd:14.4e}  {meff:15.4f}")
            else:
                print(f"{dt:3d}  {c_avg:14.4e}  {c_fwd:14.4e}  {'ratio<1':>15s}")
        else:
            print(f"{dt:3d}  {c_avg:14.4e}  {c_fwd:14.4e}  {'---':>15s}")

    # ── Test 2: Standard donghx boundary sign fix ───────────────────────
    print("\n" + "-" * 50)
    print("Test 2: Standard donghx boundary sign fix (anti-periodic BC)")
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
    n_sign2 = np.sum(np.diff(np.sign(C2pt_2[2:33])[np.sign(C2pt_2[2:33]) != 0]) != 0)
    print(f"Sign changes (t=2..32): {n_sign2}")

    print(f"\n{'t':>3s}  {'C_avg':>14s}  {'C_fwd':>14s}  {'m_eff_fwd(GeV)':>15s}")
    for dt in range(2, 16):
        c_avg = C2pt_2[dt]
        c_fwd = C2pt_fwd_2[dt]
        if dt < 32 and abs(c_fwd) > 1e-30 and abs(C2pt_fwd_2[dt+1]) > 1e-30:
            ratio = abs(c_fwd / C2pt_fwd_2[dt+1])
            if ratio > 1.0:
                meff = np.log(ratio) * fm2GeV / alttc_val
                print(f"{dt:3d}  {c_avg:14.4e}  {c_fwd:14.4e}  {meff:15.4f}")
            else:
                print(f"{dt:3d}  {c_avg:14.4e}  {c_fwd:14.4e}  {'ratio<1':>15s}")
        else:
            print(f"{dt:3d}  {c_avg:14.4e}  {c_fwd:14.4e}  {'---':>15s}")

    # ── Test 3: Folded correlator (standard nucleon analysis) ───────────
    print("\n" + "-" * 50)
    print("Test 3: Folded correlator C(t)=C_pp(t)+C_pm(Nt-t)")

    C_folded = np.zeros(Nt, dtype=np.float64)
    for dt in range(Nt):
        vals = []
        for t_src in range(Nt):
            t_snk = (t_src + dt) % Nt
            if abs(corr_pp_raw[t_snk, t_src]) > 1e-30:
                vals.append(np.real(corr_pp_raw[t_snk, t_src]))
        if vals:
            C_folded[dt] = np.mean(vals)

    print(f"{'t':>3s}  {'C_folded':>14s}  {'m_eff(GeV)':>12s}")
    for dt in range(2, 20):
        if dt < 32:
            r = abs(C_folded[dt] / (C_folded[dt+1] + 1e-30))
            if r > 1.0 and C_folded[dt+1] != 0:
                meff = np.log(r) * fm2GeV / alttc_val
                print(f"{dt:3d}  {C_folded[dt]:14.4e}  {meff:12.4f}")
            else:
                print(f"{dt:3d}  {C_folded[dt]:14.4e}  {'???':>12s}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    n1 = np.sum(np.diff(np.sign(C2pt_1[2:33])[np.sign(C2pt_1[2:33]) != 0]) != 0)
    n2 = np.sum(np.diff(np.sign(C2pt_2[2:33])[np.sign(C2pt_2[2:33]) != 0]) != 0)
    n3 = np.sum(np.diff(np.sign(C_folded[2:33])[np.sign(C_folded[2:33]) != 0]) != 0)
    print(f"  Test 1 (no fix)       C2pt sign changes: {n1}")
    print(f"  Test 2 (donghx fix)    C2pt sign changes: {n2}  ← RECOMMENDED")
    print(f"  Test 3 (folded)        C2pt sign changes: {n3}")

    # Print plateau estimates
    ps, pe = Nt//4, Nt//2
    for label, C in [("Test 1 (no fix)", C2pt_1), ("Test 2 (donghx)", C2pt_2), ("Test 3 (folded)", C_folded)]:
        meffs = []
        for t in range(1, Nt-1):
            if C[t] != 0 and C[t+1] != 0:
                ratio = abs(C[t] / (C[t+1] + 1e-30))
                if ratio > 1.0:
                    meffs.append(np.log(ratio) * fm2GeV / alttc_val)
        meffs = np.array(meffs)
        valid = ~np.isnan(meffs[ps-1:pe-1])
        if np.any(valid):
            plateau = np.mean(meffs[ps-1:pe-1][valid])
            print(f"  {label:20s} plateau [{ps},{pe}]: m_eff={plateau:.4f} GeV")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="2pt correlator diagnostics (v20260802)")
    parser.add_argument("--contract-file", type=str, default=None,
                       help="Path to the twopt_slice_pp_*_contract_conf*.npy file")
    parser.add_argument("--run-dir", type=str, default=None,
                       help="Run directory containing data/conf_*/")
    parser.add_argument("--conf-id", type=int, default=6250,
                       help="Config ID to load (default: 6250)")
    parser.add_argument("--Nt", type=int, default=72,
                       help="Number of time slices (default: 72)")
    parser.add_argument("--alttc", type=float, default=0.1053,
                       help="Lattice spacing in fm (default: 0.1053)")
    args = parser.parse_args()

    contract_file = args.contract_file

    if contract_file is None and args.run_dir is not None:
        # Auto-discover from run directory
        run_dir = Path(args.run_dir)
        conf_dir = run_dir / "data" / f"conf_{args.conf_id}"
        candidates = sorted(conf_dir.glob("twopt_slice_pp_*_contract_conf*.npy"))
        if candidates:
            contract_file = str(candidates[0])
            print(f"Auto-discovered contract file: {contract_file}")

    if contract_file is None:
        # Try to find from docker-v20260802 latest output
        script_dir = Path(__file__).parent
        output_dirs = sorted(script_dir.glob("output_*"))
        if output_dirs:
            latest = output_dirs[-1]
            conf_dir = latest / "data" / f"conf_{args.conf_id}"
            candidates = sorted(conf_dir.glob("twopt_slice_pp_*_contract_conf*.npy"))
            if candidates:
                contract_file = str(candidates[0])
                print(f"Auto-discovered from latest run ({latest.name}): {contract_file}")

    if contract_file is None:
        print("ERROR: No contract file found. Specify --contract-file or --run-dir.")
        print("Example: python diagnose_2pt.py --run-dir output_20260728_080657 --conf-id 6250")
        sys.exit(1)

    if not os.path.exists(contract_file):
        print(f"ERROR: Contract file not found: {contract_file}")
        sys.exit(1)

    corr_raw = np.load(contract_file)
    run_diagnostics(corr_raw, args.Nt, args.alttc)


if __name__ == "__main__":
    main()
