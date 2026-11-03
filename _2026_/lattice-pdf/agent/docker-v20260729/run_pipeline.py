#!/usr/bin/env python3
"""
Master pipeline orchestrator — docker-tv20260729 CORRECTED GPU version.

FIXES over v20260727:
  1. Eigenvector smearing is now OPTIONAL (default: OFF).
     When perambulator already encodes momentum smearing (mz2_my0_mx0),
     eigenvectors should NOT be additionally smeared in VVV.
  2. Effective mass uses fit-based methods (fit_cosh/fit_exp) instead of
     naive arccosh, which fails on noisy/oscillating correlators.
  3. Multiple meff methods tried and reported for comparison.
  4. Added --smear and --meff-method CLI options.

Runs the complete GPU-accelerated pipeline:
  0. Environment check (GPU detection, dependencies, data paths)
  1. Proton 2pt distillation (GPU: CuPy VVV + Wick contraction)
  2. OPE computation FROM SCRATCH (GPU: CuPy F_{mu nu} + Wilson line + OPE)
  3. huangcl ratio analysis (Jackknife + plotting)
  4. Final report generation

Usage:
    python run_pipeline.py                              # Default output dir
    python run_pipeline.py --conf-id 6250                # Single config test
    python run_pipeline.py --smear                       # Enable eigenvector smearing
    python run_pipeline.py --meff-method fit_exp         # Use exponential fit
    python run_pipeline.py --skip-2pt                    # Skip 2pt
    python run_pipeline.py --skip-ope                    # Skip OPE
    python run_pipeline.py --skip-analysis               # Data only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
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
    get_gpu_device_info, log_gpu_status, get_gpu_memory_mb, HAS_CUPY,
    set_compute_dtype, get_compute_dtype,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline steps
# ═══════════════════════════════════════════════════════════════════════════════

def step_environment_check(config: dict, logger) -> dict:
    """Check Python environment, GPU, dependencies, and data path accessibility."""
    print_banner("Step 00: Environment Check (GPU)", logger)
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
        ("opt_einsum", "opt_einsum"),
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
    logger.info("Data path checks (read-only):")

    for key, path, required in [
        ("eigenvector", paths["eigenvector"], True),
        ("eigenvalue", paths.get("eigenvalue", ""), False),
        ("perambulator_base", paths["perambulator_base"], True),
        ("gauge_config_base", paths["gauge_config_base"], True),
    ]:
        if not path:
            if required:
                logger.error(f"  ✗ {key}: path not configured (REQUIRED)")
                results[f"data_{key}"] = False
            continue
        exists = os.path.exists(path)
        if exists:
            if os.path.isfile(path):
                size = format_size(os.path.getsize(path))
                logger.info(f"  ✓ {key}: {path} ({size})")
            else:
                n_entries = len(os.listdir(path))
                logger.info(f"  ✓ {key}: {path}/ ({n_entries} entries)")
            results[f"data_{key}"] = True
        else:
            if required:
                logger.error(f"  ✗ {key}: {path} — NOT ACCESSIBLE (REQUIRED)")
                results[f"data_{key}"] = False
            else:
                logger.warning(f"  ⚠ {key}: {path} — not accessible")
                results[f"data_{key}"] = False

    # ── Check individual config data ──────────────────────────────────────
    params = config["parameters"]
    for conf_id in params["conf_ids"]:
        peram_dir = os.path.join(paths["perambulator_base"], str(conf_id))
        if os.path.isdir(peram_dir):
            n_files = len(os.listdir(peram_dir))
            logger.info(f"  ✓ perams conf={conf_id}: {n_files} files")
        else:
            logger.error(f"  ✗ perams conf={conf_id}: directory not found")

        gauge_file = os.path.join(
            paths["gauge_config_base"],
            paths["gauge_config_pattern"].format(conf_id=conf_id),
        )
        if os.path.isfile(gauge_file):
            size = format_size(os.path.getsize(gauge_file))
            logger.info(f"  ✓ gauge conf={conf_id}: {size}")
        else:
            logger.error(f"  ✗ gauge conf={conf_id}: file not found")

    # Summary
    required_checks = ["numpy", "matplotlib",
                       "data_eigenvector", "data_perambulator_base",
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


def step_compute_2pt(config: dict, output_dir: Path, logger) -> dict:
    """Run proton 2pt distillation computation (GPU)."""
    print_banner("Step 01: Proton 2pt Distillation (GPU)", logger)

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


def step_compute_ope(config: dict, output_dir: Path, logger) -> dict:
    """Compute OPE FROM SCRATCH using gauge configurations (GPU)."""
    print_banner("Step 02: Compute OPE from Gauge Configs (GPU, FROM SCRATCH)", logger)

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
        logger.info(f"OPE computation (GPU): {status_str}")

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


def step_huangcl_analysis(config: dict, output_dir: Path, logger) -> dict:
    """Run huangcl-style ratio analysis."""
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


def step_final_report(
    config: dict, output_dir: Path, logger,
    results_env: dict,
    results_2pt: dict,
    results_ope: dict,
    results_analysis: dict,
    elapsed_total: float,
) -> Path:
    """Generate comprehensive final markdown report."""
    print_banner("Step 04: Final Report", logger)

    params = config["parameters"]
    ensemble = config["ensemble"]
    paths = config["data_paths"]

    report_path = output_dir / "final_report.md"

    lines = []
    lines.append("# Gluon PDF Validation Pipeline Report (GPU)")
    lines.append("")
    lines.append(f"**Version**: docker-tv20260729 (GPU CORRECTED)")
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
    lines.append(f"| Nev / Nev1 | {params['Nev']} / {params['Nev1']} |")
    lines.append(f"| Element | {params['element']} |")
    lines.append(f"| delta_z | {params['delta_z']} |")
    lines.append(f"| Jackknife | {params['jackknife']} |")
    lines.append(f"| GPU precision | **{params.get('dtype_compute', 'complex64')}** |")
    lines.append("| OPE mode | **FROM SCRATCH (GPU)** |")
    lines.append("| VVV / Wick | **GPU (CuPy einsum)** |")
    lines.append("| F_{μν} | **GPU (CuPy plaquette_clover)** |")
    lines.append("")

    # Data Paths
    lines.append("## Data Paths")
    lines.append("")
    lines.append(f"| Data | Path |")
    lines.append(f"|------|------|")
    lines.append(f"| Eigenvectors | `{paths['eigenvector']}` |")
    lines.append(f"| Perambulators | `{paths['perambulator_base']}/{{conf_id}}/` |")
    lines.append(f"| Gauge configs | `{paths['gauge_config_base']}/` |")
    lines.append(f"| OPE | *Computed from scratch (GPU)* |")
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
                           f"m_eff(plateau)≈{r.get('meff_plateau_gev', 'N/A')}")
        elif isinstance(result, dict):
            lines.append(f"### conf={conf_id} ✗ — {result.get('reason', '?')}")
        else:
            lines.append(f"### conf={conf_id} — unexpected result")
    lines.append("")

    lines.append("## Step 2: OPE Computation (GPU, FROM SCRATCH)")
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
        else:
            lines.append(f"### conf={conf_id} — unexpected result")
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
    lines.append("## Notes")
    lines.append("")
    lines.append("1. **GPU acceleration**: VVV, Wick contraction, F_{μν}, Wilson line, and OPE contraction all computed on GPU (CuPy/CUDA).")
    lines.append("2. **Eigenvector cfg_48000 reuse**: Standard distillation practice.")
    lines.append("3. **OPE from scratch (GPU)**: Gauge config → Clover plaquette → F_{μν} → Wilson line → nonlocal OPE → .npz, all on GPU.")
    lines.append("4. **Memory strategy**: 4.5GB eigenvectors CPU-resident, streamed per time slice to GPU; gauge configs fully GPU-resident.")
    lines.append("5. **All intermediate results saved**: VVV blocks, F_{μν} tensors, OPE components, correlators, ratio data.")
    lines.append(f"6. **Peak CPU memory**: {get_peak_memory_gb():.2f} GB")
    lines.append(f"7. **v20260726 → v20260727**: GPU acceleration of core computations.")
    lines.append("")
    lines.append(f"---")
    lines.append(f"*Generated by docker-v20260727 (GPU) pipeline on {datetime.now():%Y-%m-%d %H:%M:%S}*")

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
        description="docker-v20260727 — GPU加速版 (CuPy/CUDA, OPE从头计算)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_pipeline.py                          # 完整运行 (3 组态)
  python run_pipeline.py --conf-id 6250            # 单组态测试
  python run_pipeline.py --skip-2pt --skip-ope     # 仅运行分析
  python run_pipeline.py --skip-analysis           # 仅计算
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
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose debug output")
    parser.add_argument("--cpu-only", action="store_true",
                       help="Force CPU fallback (disable GPU)")
    parser.add_argument("--precision", type=str, default="complex64",
                       choices=["complex64", "complex128"],
                       help="GPU compute precision: complex64 (single) or complex128 (double)")
    parser.add_argument("--smear", action="store_true",
                       help="Enable eigenvector smearing in VVV (default: OFF)")
    parser.add_argument("--meff-method", type=str, default="fit_cosh",
                       choices=["fit_cosh", "fit_exp", "exp_forward", "cosh"],
                       help="Effective mass extraction method (default: fit_cosh)")
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    config_path = args.config or (_SCRIPT_DIR / "run_config.json")
    print(f"[INFO] Loading config from: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    # Set compute precision BEFORE any deferred module imports
    set_compute_dtype(args.precision)
    config["parameters"]["dtype_compute"] = args.precision
    config["parameters"]["dtype_real"] = "float32" if args.precision == "complex64" else "float64"
    config["precision"]["compute_dtype"] = args.precision
    config["precision"]["compute_dtype_real"] = config["parameters"]["dtype_real"]
    print(f"[INFO] GPU compute precision: {args.precision} ({config['parameters']['dtype_real']})")

    # NEW: eigenvector smearing control
    config["parameters"]["apply_eigenvec_smearing"] = args.smear
    print(f"[INFO] Eigenvector smearing: {'ON' if args.smear else 'OFF'}")

    # NEW: effective mass method
    config["parameters"]["meff_method"] = args.meff_method
    print(f"[INFO] Effective mass method: {args.meff_method}")

    # Override conf list
    if args.conf_id is not None:
        config["parameters"]["conf_ids"] = [args.conf_id]
        config["parameters"]["Nconf"] = 1
        print(f"[INFO] Single config mode: conf_id={args.conf_id}")

    if args.cpu_only:
        config["gpu"]["enabled"] = False
        print("[INFO] CPU-only mode forced")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("output_%Y%m%d_%H%M%S")
        output_dir = _SCRIPT_DIR / ts

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, output_dir / "run_config.json")

    # Setup logging
    log_file = output_dir / "run.log"
    logger = setup_logging(log_file, "docker_pipeline_gpu",
                          console_level=logging.DEBUG if args.verbose else logging.INFO)

    t_total = time.perf_counter()

    logger.info("=" * 70)
    logger.info("  Gluon PDF Validation Pipeline — docker-tv20260729 (GPU CORRECTED)")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Config: {config_path}")
    logger.info(f"  Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"  OPE mode: FROM SCRATCH (GPU)")
    logger.info("=" * 70)

    # Log GPU status
    log_gpu_status(logger, "")
    gpu_mem = get_gpu_memory_mb()
    logger.info(f"  GPU memory: {gpu_mem['free_mb']:.0f} MB free / "
                f"{gpu_mem['total_mb']:.0f} MB total")

    # Save GPU info snapshot
    gpu_info = get_gpu_device_info()
    with open(output_dir / "gpu_info.json", "w") as f:
        json.dump(gpu_info, f, indent=2, default=str)

    # Log configuration
    params = config["parameters"]
    logger.info(f"Config summary:")
    logger.info(f"  Ensemble: {config['ensemble']['full_name']}")
    logger.info(f"  Lattice: {config['ensemble']['Nt']}x{config['ensemble']['Nx']}^3")
    logger.info(f"  Configs: {params['conf_ids']} (Nconf={params['Nconf']})")
    logger.info(f"  Momentum: P=({params['Px']},{params['Py']},{params['Pz']})")
    logger.info(f"  Nev={params['Nev']}, delta_z={params['delta_z']}")
    logger.info(f"  Eigenvector smearing: {params.get('apply_eigenvec_smearing', False)}")
    logger.info(f"  Effective mass method: {params.get('meff_method', 'fit_cosh')}")

    dump_config_snapshot(config, output_dir, logger)

    results = {}

    # ── Step 0: Environment check ─────────────────────────────────────────
    try:
        with Timer("00_environment_check", logger, output_dir):
            results["environment"] = step_environment_check(config, logger)

        if not results["environment"].get("all_required_ok", False):
            logger.error("Environment check FAILED. Required items missing.")
            if not results["environment"].get("gpu", False):
                logger.error("GPU/CuPy not available — cannot proceed with GPU pipeline.")
                logger.error("Install CuPy: pip install cupy-cuda12x")
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

    # ── Step 2: OPE computation FROM SCRATCH (GPU) ────────────────────────
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
    logger.info("  Pipeline Complete (GPU)")
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
            logger.error("OPE (GPU): ALL configs failed")
            exit_code = 1

    if isinstance(results.get("analysis"), dict):
        if results["analysis"].get("status") != "ok" and not args.skip_analysis:
            logger.error("Analysis: FAILED")
            exit_code = 1

    if exit_code == 0:
        logger.info(color("Pipeline finished SUCCESSFULLY (GPU)", Colors.GREEN))
    else:
        logger.error(color(f"Pipeline finished with ERRORS (exit code {exit_code})", Colors.RED))

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
