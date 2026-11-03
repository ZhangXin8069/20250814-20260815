#!/usr/bin/env python3
"""
Environment checker for docker-v20260804 GPU pipeline.

Verifies:
  1. Python version and dependencies
  2. CUDA/CuPy availability
  3. GPU memory and device info
  4. Data path accessibility (eigenvectors, perambulators, gauge configs)
  5. Sample data integrity (file sizes, shapes)

Usage:
    python check_env.py
    python check_env.py --conf-id 6250
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

import numpy as np

# ── Add script directory to path ─────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))


def check_python() -> dict:
    """Check Python version."""
    vi = sys.version_info
    ok = vi >= (3, 8)
    return {
        "version": f"{vi.major}.{vi.minor}.{vi.micro}",
        "ok": ok,
        "message": "✓" if ok else f"✗ Need Python ≥ 3.8",
    }


def check_module(name: str) -> dict:
    """Check if a Python module can be imported."""
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "unknown")
        return {"name": name, "version": str(version), "ok": True}
    except ImportError:
        return {"name": name, "version": "N/A", "ok": False}


def check_cupy() -> dict:
    """Check CuPy and CUDA availability."""
    try:
        import cupy as cp

        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        device_name = (
            props["name"].decode()
            if isinstance(props["name"], bytes)
            else props["name"]
        )
        return {
            "cupy_version": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "device_name": device_name,
            "compute_capability": f"{props['major']}.{props['minor']}",
            "total_memory_gb": total_bytes / 1024**3,
            "free_memory_gb": free_bytes / 1024**3,
            "ok": True,
        }
    except ImportError:
        return {"ok": False, "error": "CuPy not installed"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def check_data_paths(conf_ids: list) -> dict:
    """Check data path accessibility and sample file integrity."""
    eigvec_base = "/public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    peram_base = "/public/group/lqcd/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light"

    results = {}

    for cid in conf_ids:
        conf_result = {"conf_id": cid, "eigvec": {}, "peram": {}}

        # ── Eigenvectors ───────────────────────────────────────────────
        eigvec_dir = os.path.join(eigvec_base, str(cid))
        conf_result["eigvec"]["dir_exists"] = os.path.isdir(eigvec_dir)

        if os.path.isdir(eigvec_dir):
            first_file = os.path.join(eigvec_dir, f"eigvecs_t000_{cid}")
            if os.path.exists(first_file):
                size_mb = os.path.getsize(first_file) / 1024**2
                raw = np.fromfile(first_file, dtype="<f8")
                Nv = 24**3
                Nev_full = (raw.size // 2) // (Nv * 3)
                conf_result["eigvec"]["first_file_size_mb"] = size_mb
                conf_result["eigvec"]["Nev_full"] = Nev_full
                conf_result["eigvec"]["first_file_ok"] = Nev_full >= 100

                # Count total files
                n_files = len([
                    f for f in os.listdir(eigvec_dir)
                    if f.startswith("eigvecs_t")
                ])
                conf_result["eigvec"]["n_files"] = n_files
                conf_result["eigvec"]["n_expected"] = 72
            else:
                conf_result["eigvec"]["first_file_exists"] = False

        # ── Perambulators ──────────────────────────────────────────────
        peram_dir = os.path.join(peram_base, str(cid))
        conf_result["peram"]["dir_exists"] = os.path.isdir(peram_dir)

        if os.path.isdir(peram_dir):
            first_file = os.path.join(peram_dir, f"perams.{cid}.0.0")
            if os.path.exists(first_file):
                size_mb = os.path.getsize(first_file) / 1024**2
                raw = np.fromfile(first_file, dtype="<f8")
                expected = 72 * 100 * 4 * 100 * 2  # Nt * Nev * Nspin * Nev * 2
                matches = raw.size == expected
                conf_result["peram"]["first_file_size_mb"] = size_mb
                conf_result["peram"]["expected_shape"] = (
                    "(72, 100, 4, 100, 2)"
                )
                conf_result["peram"]["shape_matches"] = matches

                n_files = len([
                    f for f in os.listdir(peram_dir)
                    if f.startswith(f"perams.{cid}.")
                ])
                conf_result["peram"]["n_files"] = n_files
                conf_result["peram"]["n_expected"] = 72 * 4
            else:
                conf_result["peram"]["first_file_exists"] = False

        results[str(cid)] = conf_result

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Check GPU environment for docker-v20260804"
    )
    parser.add_argument(
        "--conf-id", type=int, default=None,
        help="Check specific config ID"
    )
    args = parser.parse_args()

    conf_ids = [args.conf_id] if args.conf_id else [6250, 6450, 6650]

    print("=" * 72)
    print("  docker-v20260804 — Environment Check")
    print("=" * 72)
    print()

    # Python
    py = check_python()
    print(f"Python:      {py['version']} [{py['message']}]")

    # Modules
    for mod in ["numpy", "scipy", "matplotlib", "opt_einsum"]:
        result = check_module(mod)
        status = "✓" if result["ok"] else "✗"
        print(f"  {mod:15s} {result['version']:12s} [{status}]")

    # CuPy/GPU
    print()
    print("--- GPU ---")
    gpu = check_cupy()
    if gpu["ok"]:
        print(f"  CuPy:        {gpu['cupy_version']}")
        print(f"  CUDA:        {gpu['cuda_runtime']}")
        print(f"  Device:      {gpu['device_name']}")
        print(f"  Compute:     {gpu['compute_capability']}")
        print(
            f"  Memory:      {gpu['free_memory_gb']:.1f} GB free / "
            f"{gpu['total_memory_gb']:.1f} GB total"
        )
    else:
        print(f"  GPU: ✗ {gpu.get('error', 'not available')}")

    # Data paths
    print()
    print("--- Data Paths ---")
    data = check_data_paths(conf_ids)
    for cid_str, info in data.items():
        eig_ok = (
            info["eigvec"].get("dir_exists", False)
            and info["eigvec"].get("first_file_ok", False)
            and info["eigvec"].get("n_files", 0) >= info["eigvec"].get(
                "n_expected", 72
            )
        )
        peram_ok = (
            info["peram"].get("dir_exists", False)
            and info["peram"].get("shape_matches", False)
            and info["peram"].get("n_files", 0) >= info["peram"].get(
                "n_expected", 288
            )
        )
        print(
            f"  conf={cid_str}: "
            f"eigvec={'✓' if eig_ok else '✗'} "
            f"({info['eigvec'].get('n_files', '?')}/"
            f"{info['eigvec'].get('n_expected', 72)} files, "
            f"Nev={info['eigvec'].get('Nev_full', '?')}), "
            f"peram={'✓' if peram_ok else '✗'} "
            f"({info['peram'].get('n_files', '?')}/"
            f"{info['peram'].get('n_expected', 288)} files)"
        )

    # Summary
    print()
    print("=" * 72)
    all_ok = (
        py["ok"]
        and gpu["ok"]
        and all(
            info["eigvec"].get("first_file_ok", False)
            and info["peram"].get("shape_matches", False)
            for info in data.values()
        )
    )
    if all_ok:
        print("  ✓ ALL CHECKS PASSED — ready to run pipeline")
    else:
        print("  ✗ Some checks failed — review above before running")
    print("=" * 72)


if __name__ == "__main__":
    main()
