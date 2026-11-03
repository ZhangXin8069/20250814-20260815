#!/usr/bin/env python3
"""
Master pipeline orchestrator — docker-v20260802 GPU version (MERGED + FIXED).

MERGE NOTE (v20260802):
  This is the definitive merged version incorporating:
  - v20260730's CORRECT donghx OPE algorithm (dual F̃, Tensor4, Levi-Civita)
  - v20260731's double precision default and eigenvector format detection
  - All 30 bug fixes from the v20260730/v20260731 cross-review (2026-07-28)
  - Unified step_* function naming convention
  - Removed unused shutil import
  - Cleaned up dead config fields (eigenvector_pattern, perambulator_pattern, eigenvalue)
  - Ported check_env.py and diagnose_2pt.py diagnostic scripts

Runs the complete GPU-accelerated pipeline:
  0. Environment check (GPU detection, dependencies, data paths)
  1. Proton 2pt distillation (GPU: CuPy VVV + Wick contraction)
  2. OPE computation FROM SCRATCH (GPU: CuPy donghx F_{mu nu} + dual F̃ + Wilson line + OPE)
  3. huangcl ratio analysis (Jackknife + plotting)
  4. Final report generation

Usage:
    python run_pipeline.py                              # Default output dir
    python run_pipeline.py --conf-id 6250                # Single config test
    python run_pipeline.py --conf-id 6250 --skip-ope --skip-analysis  # 2pt only
    python run_pipeline.py --skip-2pt                    # Skip 2pt
    python run_pipeline.py --skip-ope                    # Skip OPE
    python run_pipeline.py --skip-analysis               # Data only
    python run_pipeline.py --precision complex64          # Single precision
    python run_pipeline.py --smear                       # Enable eigenvector smearing
    python run_pipeline.py --meff-method fit_exp         # Use exponential fit
    python run_pipeline.py --verbose                     # Debug logging
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils import (
    Timer, print_banner, format_size,
    get_peak_memory_gb, setup_logging, Colors, color,
    dump_config_snapshot, get_output_tree, get_current_memory_mb,
    get_gpu_device_info, get_gpu_memory_mb, log_gpu_status,
    set_compute_dtype, get_compute_dtype, HAS_CUPY,
    save_intermediate,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 0: Environment check
# ═══════════════════════════════════════════════════════════════════════════════

def step_environment_check(config: dict, logger) -> dict:
    """Verify GPU, Python deps, and data paths before running the pipeline.

    Checks: Python version, module imports (numpy, scipy, matplotlib, cupy),
    GPU/CuPy status, and accessibility of eigenvector, perambulator, and gauge
    data paths for all requested config IDs.

    Args:
        config: Full pipeline config dict.
        logger: Logger instance.

    Returns:
        dict with keys: gpu (bool), all_required_ok (bool), and per-check results.
    """
    print_banner("Step 00: Environment Check (GPU, v20260802)", logger)

    logger.info(f"Python: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Script directory: {_SCRIPT_DIR}")

    results = {}

    # ── GPU check ──────────────────────────────────────────────────────────
    if HAS_CUPY:
        import cupy as cp
        gpu_info = get_gpu_device_info()
        logger.info(f"GPU Device: {gpu_info.get('device_name', 'unknown')}")
        logger.info(f"  Compute Capability: {gpu_info.get('compute_capability', '?')}")
        logger.info(f"  Memory: {gpu_info.get('free_memory_gb', 0):.1f} GB free / "
                    f"{gpu_info.get('total_memory_gb', 0):.1f} GB total")
        logger.info(f"  CuPy: {gpu_info.get('cupy_version', '?')}")
        logger.info(f"  CUDA Runtime: {gpu_info.get('cuda_version', '?')}")
        results["gpu"] = True
    else:
        logger.error("✗ CuPy NOT AVAILABLE — GPU acceleration disabled!")
        logger.error("  Install with: pip install cupy-cuda12x")
        results["gpu"] = False

    # ── Essential modules ──────────────────────────────────────────────────
    for name, import_name in [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("h5py", "h5py"),
    ]:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "?")
            logger.info(f"  ✓ {name}: {ver}")
            results[name] = True
        except ImportError:
            logger.warning(f"  ✗ {name}: NOT AVAILABLE (optional)")
            results[name] = False

    # ── Data paths check ──────────────────────────────────────────────────
    paths = config["data_paths"]
    params = config["parameters"]
    logger.info("Data path checks:")

    # Eigenvector base directory
    eigvec_base = paths["eigenvector_base"]
    if os.path.isdir(eigvec_base):
        subdirs = sorted([d for d in os.listdir(eigvec_base)
                         if os.path.isdir(os.path.join(eigvec_base, d))])
        logger.info(f"  ✓ eigenvector_base: {eigvec_base}/ ({len(subdirs)} config dirs: {subdirs[:5]}...)")
        results["data_eigenvector_base"] = True
    else:
        logger.error(f"  ✗ eigenvector_base: {eigvec_base} — NOT ACCESSIBLE")
        results["data_eigenvector_base"] = False

    # Perambulator base directory (without /light/ — code appends it)
    peram_base = paths["perambulator_base"]
    if os.path.isdir(peram_base):
        subdirs = sorted([d for d in os.listdir(peram_base)
                         if os.path.isdir(os.path.join(peram_base, d))])
        logger.info(f"  ✓ perambulator_base: {peram_base}/ ({len(subdirs)} subdirs: {subdirs[:5]}...)")
        results["data_perambulator_base"] = True
    else:
        logger.error(f"  ✗ perambulator_base: {peram_base} — NOT ACCESSIBLE")
        results["data_perambulator_base"] = False

    # Gauge config base directory
    gauge_base = paths["gauge_config_base"]
    if os.path.isdir(gauge_base):
        files = sorted(os.listdir(gauge_base))
        logger.info(f"  ✓ gauge_config_base: {gauge_base}/ ({len(files)} files)")
        results["data_gauge_config_base"] = True
    else:
        logger.error(f"  ✗ gauge_config_base: {gauge_base} — NOT ACCESSIBLE")
        results["data_gauge_config_base"] = False

    # ── Check individual config data ──────────────────────────────────────
    for conf_id in params["conf_ids"]:
        # Eigenvector directory per config
        ev_dir = os.path.join(eigvec_base, str(conf_id))
        if os.path.isdir(ev_dir):
            n_files = len(os.listdir(ev_dir))
            logger.info(f"  ✓ eigenvecs conf={conf_id}: {n_files} files")
        else:
            logger.error(f"  ✗ eigenvecs conf={conf_id}: directory not found")

        # Perambulator directory per config (light/{conf_id}/)
        peram_dir = os.path.join(peram_base, "light", str(conf_id))
        if os.path.isdir(peram_dir):
            n_files = len(os.listdir(peram_dir))
            logger.info(f"  ✓ perams conf={conf_id}: {n_files} files")
        else:
            logger.error(f"  ✗ perams conf={conf_id}: directory not found")

        # Gauge config file
        gauge_file = os.path.join(
            gauge_base,
            paths["gauge_config_pattern"].format(conf_id=conf_id),
        )
        if os.path.isfile(gauge_file):
            size = format_size(os.path.getsize(gauge_file))
            logger.info(f"  ✓ gauge conf={conf_id}: {size}")
        else:
            logger.error(f"  ✗ gauge conf={conf_id}: file not found")

    # Summary
    required_checks = ["numpy", "matplotlib",
                       "data_eigenvector_base", "data_perambulator_base",
                       "data_gauge_config_base"]
    all_required_ok = all(results.get(k, False) for k in required_checks)
    if all_required_ok and results.get("gpu", False):
        logger.info(color("Environment check: ALL REQUIRED ITEMS OK (GPU ready)", Colors.GREEN))
    elif all_required_ok:
        logger.warning(color("Environment check: CPU ready, GPU NOT available", Colors.YELLOW))
    else:
        missing = [k for k in required_checks if not results.get(k, False)]
        if not results.get("gpu", False):
            missing.append("GPU/CuPy")
        logger.error(color(f"Environment check: MISSING: {missing}", Colors.RED))

    results["all_required_ok"] = all_required_ok
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: 2pt computation
# ═══════════════════════════════════════════════════════════════════════════════

def step_compute_2pt(config: dict, output_dir: Path, logger) -> dict:
    """Run proton 2pt distillation computation (GPU, per-config eigvecs).

    Uses v20260730's verified perambulator reading + v20260731's eigenvector
    detection + graceful missing-file handling.

    Args:
        config: Full pipeline config dict.
        output_dir: Base output directory.
        logger: Logger instance.

    Returns:
        dict mapping conf_id → results with per-Pz correlator data.
    """
    print_banner("Step 01: Proton 2pt Distillation (GPU, v20260802)", logger)

    from compute_2pt_gpu import run_2pt_computation_gpu
    data_dir = output_dir / "data"

    try:
        with Timer("01_compute_2pt_gpu", logger, output_dir):
            results = run_2pt_computation_gpu(config, data_dir, logger)

        all_ok = all(
            isinstance(r, dict) and r.get("status") == "ok"
            for r in results.values()
        )
        status_str = color("✓ ALL OK", Colors.GREEN) if all_ok else color("⚠ PARTIAL", Colors.YELLOW)
        logger.info(f"2pt computation (GPU): {status_str}")

        n_ok = sum(1 for r in results.values()
                   if isinstance(r, dict) and r.get("status") == "ok")
        logger.info(f"  Success: {n_ok}/{len(results)} configs")

        return results

    except Exception as e:
        logger.error(f"2pt computation FAILED: {e}")
        logger.debug(traceback.format_exc())
        return {"status": "error", "reason": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: OPE computation
# ═══════════════════════════════════════════════════════════════════════════════

def step_compute_ope(config: dict, output_dir: Path, logger) -> dict:
    """Compute OPE FROM SCRATCH using CORRECT donghx algorithm (dual F̃).

    Uses v20260730's full donghx operators_new_z0_mu2 algorithm:
      - Tensor4 = 0.5 * ε_{μνρσ} (Levi-Civita)
      - Dual field strength F̃_{μν} = 0.5 * ε_{μνρσ} * F_{ρσ}
      - Roll-based Wilson line transport
      - Trace + spatial sum over ALL axes

    Args:
        config: Full pipeline config dict.
        output_dir: Base output directory.
        logger: Logger instance.

    Returns:
        dict mapping conf_id → OPE results with component details.
    """
    print_banner("Step 02: Compute OPE from Gauge Configs (GPU, donghx algorithm, v20260802)", logger)

    from compute_ope_gpu import compute_ope_all_configs_gpu
    data_dir = output_dir / "data"

    try:
        with Timer("02_compute_ope_gpu", logger, output_dir):
            results = compute_ope_all_configs_gpu(config, data_dir, logger)

        all_ok = all(
            isinstance(r, dict) and r.get("status") == "ok"
            for r in results.values()
        )
        status_str = color("✓ ALL OK", Colors.GREEN) if all_ok else color("⚠ PARTIAL", Colors.YELLOW)
        logger.info(f"OPE computation (GPU, donghx): {status_str}")

        n_ok = sum(1 for r in results.values()
                   if isinstance(r, dict) and r.get("status") == "ok")
        total_components = sum(
            sum(1 for v in r.get("components", {}).values() if v.get("status") == "ok")
            for r in results.values()
        )
        logger.info(f"  Configs: {n_ok}/{len(results)}, Components: {total_components} OK")

        return results

    except Exception as e:
        logger.error(f"OPE computation FAILED: {e}")
        logger.debug(traceback.format_exc())
        return {"status": "error", "reason": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: huangcl analysis
# ═══════════════════════════════════════════════════════════════════════════════

def step_huangcl_analysis(config: dict, output_dir: Path, logger) -> dict:
    """Run huangcl-style ratio analysis with jackknife resampling.

    Loads 2pt correlators and OPE components, constructs the disconnected
    3pt/2pt ratio R(z), and generates diagnostic plots.

    Args:
        config: Full pipeline config dict.
        output_dir: Base output directory.
        logger: Logger instance.

    Returns:
        dict with status, plot paths, and numerical results.
    """
    print_banner("Step 03: huangcl Ratio Analysis", logger)

    from analyze_ratio import run_analysis
    data_dir = output_dir / "data"
    plots_dir = output_dir / "plots"

    try:
        with Timer("03_huangcl_analysis", logger, output_dir):
            results = run_analysis(config, data_dir, plots_dir, logger)

        if results.get("status") == "ok":
            logger.info(f"Analysis: {color('✓ OK', Colors.GREEN)}")
            for key in ["ratio_path", "diag_path", "meff_path", "field_strength_path"]:
                if key in results:
                    logger.info(f"  Plot: {results[key]}")
        else:
            logger.error(f"Analysis: {color('✗ FAILED', Colors.RED)}")
            if "errors" in results:
                for err in results["errors"][:5]:
                    logger.error(f"  Error: {err}")

        return results

    except Exception as e:
        logger.error(f"Analysis FAILED: {e}")
        logger.debug(traceback.format_exc())
        return {"status": "error", "reason": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Final report
# ═══════════════════════════════════════════════════════════════════════════════

def step_final_report(
    config: dict, output_dir: Path, logger,
    results_env: dict,
    results_2pt: dict,
    results_ope: dict,
    results_analysis: dict,
    elapsed_total: float,
) -> Path:
    """Generate comprehensive final Markdown report with all results.

    Includes: configuration summary, GPU info, data paths, per-step results,
    effective mass summary, OPE component summary, analysis plots, output tree,
    and timing data.

    Args:
        config: Full pipeline config dict.
        output_dir: Base output directory.
        logger: Logger instance.
        results_env, results_2pt, results_ope, results_analysis: Per-step results.
        elapsed_total: Total pipeline wall time in seconds.

    Returns:
        Path to the generated report.md file.
    """
    print_banner("Step 04: Final Report", logger)

    params = config["parameters"]
    ensemble = config["ensemble"]
    paths = config["data_paths"]

    report_path = output_dir / "final_report.md"

    lines = []
    lines.append("# Gluon PDF Validation Pipeline Report (GPU, v20260802)")
    lines.append("")
    lines.append(f"**Version**: docker-v20260802 (MERGED + FIXED — donghx OPE + double precision)")
    lines.append(f"**Run time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total elapsed**: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    lines.append(f"**Peak CPU memory**: {get_peak_memory_gb():.2f} GB")
    lines.append(f"**Output directory**: `{output_dir}`")
    lines.append("")

    # GPU info
    gpu_info = get_gpu_device_info()
    if gpu_info.get("cupy_available"):
        lines.append(f"**GPU**: {gpu_info.get('device_name', '?')} "
                    f"(CC {gpu_info.get('compute_capability', '?')}, "
                    f"{gpu_info.get('total_memory_gb', 0):.1f} GB, "
                    f"CuPy {gpu_info.get('cupy_version', '?')}, "
                    f"CUDA {gpu_info.get('cuda_version', '?')})")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Ensemble | {ensemble['full_name']} ({ensemble['name']}) |")
    lines.append(f"| Lattice | {ensemble['Nt']}×{ensemble['Nx']}³, β={ensemble['beta']} |")
    lines.append(f"| Lattice spacing | a={ensemble['alttc']} fm |")
    lines.append(f"| Configs | {params['conf_ids']} (Nconf={params['Nconf']}) |")
    lines.append(f"| Momentum | P=({params['Px']},{params['Py']},{params['Pz']}) |")
    lines.append(f"| Nev (eigvecs) / Nev1 (perams) | {params['Nev']} / {params['Nev1']} |")
    lines.append(f"| Element | {params['element']} |")
    lines.append(f"| delta_z | {params['delta_z']} |")
    lines.append(f"| Jackknife | {params['jackknife']} |")
    lines.append(f"| GPU precision | **{get_compute_dtype().__name__}** |")
    lines.append(f"| OPE algorithm | **donghx operators_new_z0_mu2 (dual F̃)** |")
    lines.append(f"| F_{μν} | **GPU (CuPy plaquette_clover)** |")
    lines.append(f"| Eigenvectors | **Per-config** (binary per time slice) |")
    lines.append("")

    # Data Paths
    lines.append("## Data Paths")
    lines.append("")
    lines.append(f"| Data | Path |")
    lines.append(f"|------|------|")
    lines.append(f"| Eigenvectors | `{paths['eigenvector_base']}/{{conf_id}}/` |")
    lines.append(f"| Perambulators | `{paths['perambulator_base']}/light/{{conf_id}}/` |")
    lines.append(f"| Gauge configs | `{paths['gauge_config_base']}/` |")
    lines.append(f"| OPE | *Computed from scratch (GPU, donghx algorithm)* |")
    lines.append("")

    # Steps
    lines.append("## Step 0: Environment Check")
    lines.append("")
    lines.append(f"- GPU available: {results_env.get('gpu', False)}")
    lines.append(f"- All required OK: {results_env.get('all_required_ok', 'N/A')}")
    lines.append("")

    lines.append("## Step 1: Proton 2pt Distillation (GPU)")
    lines.append("")
    for conf_id, result in results_2pt.items():
        if isinstance(result, dict) and result.get("status") == "ok":
            lines.append(f"### conf={conf_id} ✓")
            for Pz, r in result.get("results", {}).items():
                lines.append(f"- Pz={Pz}: PP range {r.get('corr_pp_range_re', 'N/A')}, "
                           f"m_eff(plateau)≈{r.get('meff_plateau_gev', 'N/A')} GeV")
                if 'all_meff_methods' in r:
                    methods_str = ", ".join(f"{m}={v:.3f}" for m, v in r['all_meff_methods'].items())
                    lines.append(f"  All methods: {methods_str}")
        elif isinstance(result, dict):
            lines.append(f"### conf={conf_id} ✗ — {result.get('reason', '?')}")
    lines.append("")

    lines.append("## Step 2: OPE Computation (GPU, donghx dual-F̃ algorithm)")
    lines.append("")
    for conf_id, result in results_ope.items():
        if isinstance(result, dict) and result.get("status") == "ok":
            n_ok = sum(1 for v in result.get("components", {}).values()
                      if v.get("status") == "ok")
            n_total = len(result.get("components", {}))
            lines.append(f"### conf={conf_id} ✓ ({n_ok}/{n_total} components)")
            if "validation" in result:
                val = result["validation"]
                lines.append(f"- Unitarity: max_dev={val.get('unitary_dev_max', 'N/A')}")
                lines.append(f"- Plaq trace: re={val.get('plaq_trace_mean_re', 'N/A')}")
            for key, comp in result.get("components", {}).items():
                status = "✓" if comp.get("status") == "ok" else "✗"
                re_r = comp.get("re_range", [0, 0])
                lines.append(f"  - {key} {status}: |O|∈[{re_r[0]:.2e}, {re_r[1]:.2e}]")
        elif isinstance(result, dict):
            lines.append(f"### conf={conf_id} ✗ — {result.get('reason', '?')}")
    lines.append("")

    lines.append("## Step 3: huangcl Ratio Analysis")
    lines.append("")
    if results_analysis.get("status") == "ok":
        lines.append("✓ Analysis completed successfully")
        lines.append(f"- Loaded configs: {results_analysis.get('loaded_confs', [])}")
        lines.append(f"- Ratio plot: `{results_analysis.get('ratio_path', 'N/A')}`")
        lines.append(f"- Diagnostics: `{results_analysis.get('diag_path', 'N/A')}`")
        lines.append(f"- Effective mass: `{results_analysis.get('meff_path', 'N/A')}`")
        lines.append(f"- Field strength: `{results_analysis.get('field_strength_path', 'N/A')}`")
    else:
        lines.append(f"✗ Analysis failed: {results_analysis.get('errors', [])}")
    lines.append("")

    # Output Files
    lines.append("## Output Files")
    lines.append("")
    lines.append("```")
    lines.append(get_output_tree(output_dir, max_files_per_dir=30))
    lines.append("```")
    lines.append("")

    # Notes
    lines.append("## Key Fixes in v20260802")
    lines.append("")
    lines.append("1. **OPE algorithm**: Uses donghx's dual field strength F̃ = 0.5·ε·F (v20260730, verified correct)")
    lines.append("2. **Perambulator reading**: v20260730 verified layout (Nt,Nev,Nspin,Nev) — matches snsc/main.py")
    lines.append("3. **Eigenvector format**: Auto-detection of Nev_full from file size (v20260731 feature)")
    lines.append("4. **Missing file handling**: Graceful per-time-slice fallback (v20260730 feature)")
    lines.append("5. **Double precision**: complex128 by default, configurable via --precision")
    lines.append("6. **All intermediate results saved**: VVV blocks, F_{μν} tensors, OPE components, correlators, ratio data")
    lines.append("")
    lines.append(f"---")
    lines.append(f"*Generated by docker-v20260802 (GPU, merged + fixed) pipeline on {datetime.now():%Y-%m-%d %H:%M:%S}*")

    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)

    logger.info(f"Report saved to {report_path} ({len(report_text)} chars)")
    return report_path


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="docker-v20260802 — GPU 合并修正版 (donghx OPE + 双精度 + 全部bug修复)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                              # Full pipeline (3 configs)
  python run_pipeline.py --conf-id 6250                # Single config test
  python run_pipeline.py --conf-id 6250 --skip-ope --skip-analysis  # 2pt only
  python run_pipeline.py --skip-2pt --skip-ope         # Analysis only
  python run_pipeline.py --skip-analysis               # Compute only
  python run_pipeline.py --precision complex64          # Single precision
  python run_pipeline.py --smear                       # Enable eigenvector smearing
  python run_pipeline.py --meff-method fit_exp          # Exponential effective mass
  python run_pipeline.py --verbose --conf-id 6250       # Debug mode
        """,
    )
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory [default: output_YYYYMMDD_HHMMSS/]")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to run_config.json")
    parser.add_argument("--skip-2pt", action="store_true",
                       help="Skip 2pt distillation")
    parser.add_argument("--skip-ope", action="store_true",
                       help="Skip OPE computation")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip huangcl analysis")
    parser.add_argument("--skip-report", action="store_true",
                       help="Skip final report generation")
    parser.add_argument("--conf-id", type=int, default=None,
                       help="Process a single config only")
    parser.add_argument("--conf-ids", type=int, nargs="+", default=None,
                       help="Process a list of config IDs")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose debug output")
    parser.add_argument("--precision", type=str, default="complex128",
                       choices=["complex64", "complex128"],
                       help="GPU compute precision: complex64 (single) or complex128 (double, default)")
    parser.add_argument("--smear", action="store_true",
                       help="Enable eigenvector smearing in VVV (default: OFF)")
    parser.add_argument("--meff-method", type=str, default="fit_cosh",
                       choices=["fit_cosh", "fit_exp", "exp_forward", "cosh"],
                       help="Effective mass extraction method (default: fit_cosh)")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    config_path = args.config or (_SCRIPT_DIR / "run_config.json")
    print(f"[INFO] Loading config from: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    # ── Set compute precision BEFORE any module imports ───────────────────
    set_compute_dtype(args.precision)
    config["parameters"]["dtype_compute"] = args.precision
    config["parameters"]["dtype_real"] = "float32" if args.precision == "complex64" else "float64"
    config["precision"]["compute_dtype"] = args.precision
    config["precision"]["compute_dtype_real"] = config["parameters"]["dtype_real"]
    print(f"[INFO] GPU compute precision: {args.precision} ({config['parameters']['dtype_real']})")

    # ── Apply CLI overrides ───────────────────────────────────────────────
    config["parameters"]["apply_eigenvec_smearing"] = args.smear
    print(f"[INFO] Eigenvector smearing: {'ON' if args.smear else 'OFF'}")

    config["parameters"]["meff_method"] = args.meff_method
    print(f"[INFO] Effective mass method: {args.meff_method}")

    if args.conf_id is not None:
        config["parameters"]["conf_ids"] = [args.conf_id]
        config["parameters"]["Nconf"] = 1
        print(f"[INFO] Single config mode: conf_id={args.conf_id}")

    if args.conf_ids is not None:
        config["parameters"]["conf_ids"] = args.conf_ids
        config["parameters"]["Nconf"] = len(args.conf_ids)

    # ── Setup output directory ────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("output_%Y%m%d_%H%M%S")
        output_dir = _SCRIPT_DIR / ts

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot to output dir
    dump_config_snapshot(config, output_dir)

    # ── Setup logging ─────────────────────────────────────────────────────
    log_file = output_dir / "run.log"
    logger = setup_logging(log_file, "pipeline_v20260802",
                          console_level=logging.DEBUG if args.verbose else logging.INFO)

    t_total = time.perf_counter()

    logger.info("=" * 70)
    logger.info("  Gluon PDF Validation Pipeline — docker-v20260802 (MERGED + FIXED)")
    logger.info(f"  OPE algorithm: donghx operators_new_z0_mu2 (dual F̃)")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Config: {config_path}")
    logger.info(f"  Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"  Precision: {args.precision}")
    logger.info("=" * 70)

    # Log GPU status
    log_gpu_status(logger, "")
    gpu_mem = get_gpu_memory_mb()
    logger.info(f"  GPU memory: {gpu_mem['free_mb']:.0f} MB free / "
                f"{gpu_mem['total_mb']:.0f} MB total")

    # Save GPU info snapshot
    gpu_info = get_gpu_device_info()
    save_intermediate(gpu_info, output_dir, "gpu_info.json", logger)

    # Log configuration summary
    params = config["parameters"]
    logger.info(f"Config summary:")
    logger.info(f"  Ensemble: {config['ensemble']['full_name']}")
    logger.info(f"  Lattice: {config['ensemble']['Nt']}x{config['ensemble']['Nx']}^3")
    logger.info(f"  Configs: {params['conf_ids']} (Nconf={params['Nconf']})")
    logger.info(f"  Momentum: P=({params['Px']},{params['Py']},{params['Pz']})")
    logger.info(f"  Nev={params['Nev']} (eigvecs), Nev1={params['Nev1']} (perams)")
    logger.info(f"  delta_z={params['delta_z']}")
    logger.info(f"  Eigenvector smearing: {params.get('apply_eigenvec_smearing', False)}")
    logger.info(f"  Effective mass method: {params.get('meff_method', 'fit_cosh')}")

    results = {}

    # ── Step 0: Environment check ─────────────────────────────────────────
    try:
        with Timer("00_environment_check", logger, output_dir):
            results["environment"] = step_environment_check(config, logger)

        if not results["environment"].get("all_required_ok", False):
            logger.error("Environment check FAILED. Required items missing.")
            if not results["environment"].get("gpu", False):
                logger.error("GPU/CuPy not available — cannot proceed with GPU pipeline.")
                return 1
    except Exception as e:
        logger.error(f"Environment check crashed: {e}")
        logger.debug(traceback.format_exc())
        results["environment"] = {"status": "error", "reason": str(e)}

    # ── Step 1: 2pt computation (GPU) ────────────────────────────────────
    if not args.skip_2pt:
        results["2pt"] = step_compute_2pt(config, output_dir, logger)
    else:
        logger.info("Step 01: Skipping 2pt (--skip-2pt)")
        results["2pt"] = {}

    # ── Step 2: OPE computation FROM SCRATCH (GPU, donghx) ────────────────
    if not args.skip_ope:
        results["ope"] = step_compute_ope(config, output_dir, logger)
    else:
        logger.info("Step 02: Skipping OPE (--skip-ope)")
        results["ope"] = {}

    # ── Step 3: huangcl analysis ──────────────────────────────────────────
    if not args.skip_analysis:
        results["analysis"] = step_huangcl_analysis(config, output_dir, logger)
    else:
        logger.info("Step 03: Skipping analysis (--skip-analysis)")
        results["analysis"] = {}

    # ── Step 4: Final report ──────────────────────────────────────────────
    if not args.skip_report:
        with Timer("04_final_report", logger, output_dir):
            report_path = step_final_report(
                config, output_dir, logger,
                results.get("environment", {}),
                results.get("2pt", {}),
                results.get("ope", {}),
                results.get("analysis", {}),
                time.perf_counter() - t_total,
            )
        results["report"] = str(report_path)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_total
    mem_peak = get_peak_memory_gb()
    gpu_mem_final = get_gpu_memory_mb()

    logger.info("")
    logger.info("═" * 70)
    logger.info("  Pipeline Complete (GPU, v20260802 — donghx OPE + double precision)")
    logger.info(f"  Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    logger.info(f"  Peak CPU memory: {mem_peak:.2f} GB")
    logger.info(f"  GPU memory: {gpu_mem_final['free_mb']:.0f} MB free / "
                f"{gpu_mem_final['total_mb']:.0f} MB total")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Log: {log_file}")
    logger.info("═" * 70)

    # Print output structure
    logger.info("\nOutput structure:")
    for line in get_output_tree(output_dir).split("\n")[:50]:
        logger.info(line)

    # ── Exit status ───────────────────────────────────────────────────────
    exit_code = 0

    if isinstance(results.get("2pt"), dict) and results["2pt"]:
        n_ok = sum(1 for r in results["2pt"].values()
                   if isinstance(r, dict) and r.get("status") == "ok")
        if n_ok == 0 and not args.skip_2pt:
            logger.error("2pt (GPU): ALL configs failed")
            exit_code = 1

    if isinstance(results.get("ope"), dict) and results["ope"]:
        n_ok = sum(1 for r in results["ope"].values()
                   if isinstance(r, dict) and r.get("status") == "ok")
        if n_ok == 0 and not args.skip_ope:
            logger.error("OPE (GPU, donghx): ALL configs failed")
            exit_code = 1

    if isinstance(results.get("analysis"), dict):
        if results["analysis"].get("status") != "ok" and not args.skip_analysis:
            logger.error("Analysis: FAILED")
            exit_code = 1

    if exit_code == 0:
        logger.info(color("Pipeline finished SUCCESSFULLY (GPU, v20260802)", Colors.GREEN))
    else:
        logger.error(color(f"Pipeline finished with ERRORS (exit code {exit_code})", Colors.RED))

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
