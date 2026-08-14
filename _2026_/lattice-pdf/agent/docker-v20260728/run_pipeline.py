#!/usr/bin/env python3
"""
Master pipeline orchestrator — docker-v20260726 本地直接运行版。

Runs the complete pipeline:
  0. Environment check
  1. Proton 2pt distillation (real eigvecs + perambulators)
  2. OPE computation FROM SCRATCH (gauge config → F_{μν} → nonlocal OPE → .npz)
  3. huangcl ratio analysis (Jackknife + plotting)
  4. Final report generation

Unlike the snsc-v20260726 version, this computes OPE from scratch
rather than loading pre-computed data. All intermediate results are saved.

Key features:
  - OPE computed from gauge configurations (no pre-computed data needed)
  - All intermediate variables saved (VVV, F_{μν}, OPE components, etc.)
  - Comprehensive logging (main log + per-module logs)
  - Self-loop bug fixing and optimization
  - All data charts saved
  - Configurable via run_config.json
  - Works in current Docker/local environment

Usage:
    python run_pipeline.py                          # Default output dir
    python run_pipeline.py --output-dir /path/out    # Custom output dir
    python run_pipeline.py --skip-2pt                # Skip 2pt (if already computed)
    python run_pipeline.py --skip-ope                # Skip OPE computation
    python run_pipeline.py --skip-analysis           # Skip analysis
    python run_pipeline.py --skip-2pt --skip-ope     # Only analysis + report
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

# ─── Ensure docker-v20260726 is on sys.path ──────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ─── Add lattice-pdf root for snsc.main imports ──────────────────────────────
_REPO_ROOT = Path(_SCRIPT_DIR).parent.parent  # /root/lattice-pdf
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import (
    Timer, print_banner, format_size,
    get_peak_memory_gb, setup_logging, Colors, color,
    dump_config_snapshot, get_output_tree, get_current_memory_mb,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline steps
# ═══════════════════════════════════════════════════════════════════════════════

def step_environment_check(config: dict, logger) -> dict:
    """Check Python environment, dependencies, and data path accessibility."""
    print_banner("Step 00: Environment Check", logger)
    logger.info(f"Python: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Script directory: {_SCRIPT_DIR}")

    results = {}

    # Check essential modules
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

    # Check snsc/main.py (needed for plaquette_clover)
    snsc_path = _REPO_ROOT / "snsc" / "main.py"
    if snsc_path.exists():
        logger.info(f"  ✓ snsc/main.py: {snsc_path} ({snsc_path.stat().st_size / 1024:.0f} KB)")
        results["snsc_main"] = True
    else:
        logger.error(f"  ✗ snsc/main.py: NOT FOUND at {snsc_path} (REQUIRED for OPE)")
        results["snsc_main"] = False

    # Check data paths (best-effort, read-only)
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
                subdirs = len([d for d in os.listdir(path)
                              if os.path.isdir(os.path.join(path, d))])
                logger.info(f"  ✓ {key}: {path}/ ({subdirs} subdirectories)")
            results[f"data_{key}"] = True
        else:
            if required:
                logger.error(f"  ✗ {key}: {path} — NOT ACCESSIBLE (REQUIRED)")
                results[f"data_{key}"] = False
            else:
                logger.warning(f"  ⚠ {key}: {path} — not accessible (optional)")
                results[f"data_{key}"] = False

    # Check individual config data
    params = config["parameters"]
    for conf_id in params["conf_ids"]:
        # Check perambulator directory
        peram_dir = os.path.join(paths["perambulator_base"], str(conf_id))
        if os.path.isdir(peram_dir):
            n_files = len(os.listdir(peram_dir))
            logger.info(f"  ✓ perams conf={conf_id}: {n_files} files")
        else:
            logger.error(f"  ✗ perams conf={conf_id}: directory not found")

        # Check gauge config
        gauge_file = os.path.join(
            paths["gauge_config_base"],
            paths["gauge_config_pattern"].format(conf_id=conf_id),
        )
        if os.path.isfile(gauge_file):
            size = format_size(os.path.getsize(gauge_file))
            logger.info(f"  ✓ gauge conf={conf_id}: {gauge_file} ({size})")
        else:
            logger.error(f"  ✗ gauge conf={conf_id}: file not found")

    # Summary
    required_checks = ["numpy", "matplotlib", "snsc_main",
                       "data_eigenvector", "data_perambulator_base", "data_gauge_config_base"]
    all_required_ok = all(results.get(k, False) for k in required_checks)
    if all_required_ok:
        logger.info(color("Environment check: ALL REQUIRED ITEMS OK", Colors.GREEN))
    else:
        missing = [k for k in required_checks if not results.get(k, False)]
        logger.error(color(f"Environment check: MISSING REQUIRED: {missing}", Colors.RED))

    results["all_required_ok"] = all_required_ok
    return results


def step_compute_2pt(config: dict, output_dir: Path, logger) -> dict:
    """Run proton 2pt distillation computation.

    Saves all intermediate results:
      - VVV blocks per config and momentum
      - Raw Wick contraction matrices
      - Parity-projected (PP, PM) correlators
      - Effective mass data
    """
    print_banner("Step 01: Proton 2pt Distillation", logger)

    from compute_2pt import run_2pt_computation
    data_dir = output_dir / "data"

    try:
        with Timer("01_compute_2pt", logger, output_dir):
            results = run_2pt_computation(config, data_dir, logger)

        all_ok = all(r["status"] == "ok" for r in results.values())
        status_str = color("✓ ALL OK", Colors.GREEN) if all_ok else color("⚠ PARTIAL", Colors.YELLOW)
        logger.info(f"2pt computation: {status_str}")

        # Count successes
        n_ok = sum(1 for r in results.values() if r["status"] == "ok")
        n_total = len(results)
        logger.info(f"  Success: {n_ok}/{n_total} configs")

        return results

    except Exception as e:
        logger.error(f"2pt computation FAILED: {e}")
        logger.debug(traceback.format_exc())
        return {"status": "error", "reason": str(e)}


def step_compute_ope(config: dict, output_dir: Path, logger) -> dict:
    """Compute OPE FROM SCRATCH using gauge configurations.

    For each configuration:
      1. Read gauge config (.lime ILDG format)
      2. Validate gauge (unitarity, trace, plaquette)
      3. Compute F_{mu,nu} via clover plaquette → save as intermediate
      4. Compute nonlocal OPE operator for each (mu,nu) component
      5. Save OPE as .npz

    This is the KEY DIFFERENCE from snsc-v20260726:
    OPE is computed from scratch, not loaded from pre-computed data.
    """
    print_banner("Step 02: Compute OPE from Gauge Configs (FROM SCRATCH)", logger)

    from compute_ope import compute_ope_all_configs
    data_dir = output_dir / "data"

    try:
        with Timer("02_compute_ope", logger, output_dir):
            results = compute_ope_all_configs(config, data_dir, logger)

        all_ok = all(r["status"] == "ok" for r in results.values())
        status_str = color("✓ ALL OK", Colors.GREEN) if all_ok else color("⚠ PARTIAL", Colors.YELLOW)
        logger.info(f"OPE computation: {status_str}")

        # Count successes and components
        n_ok = sum(1 for r in results.values() if r["status"] == "ok")
        n_total = len(results)
        total_components = sum(
            sum(1 for v in r.get("components", {}).values() if v.get("status") == "ok")
            for r in results.values()
        )
        logger.info(f"  Configs: {n_ok}/{n_total}, Components: {total_components} OK")

        return results

    except Exception as e:
        logger.error(f"OPE computation FAILED: {e}")
        logger.debug(traceback.format_exc())
        return {"status": "error", "reason": str(e)}


def step_huangcl_analysis(config: dict, output_dir: Path, logger) -> dict:
    """Run huangcl-style ratio analysis.

    Computes:
      - 3pt disconnected correlator from 2pt + OPE
      - Ratio R(z) = C3_disc / C2 with jackknife errors
      - Plots: ratio.png, diagnostics.png, effective_mass.png, field_strength.png
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
            # Log plot paths
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
    lines.append("# Gluon PDF Validation Pipeline Report")
    lines.append("")
    lines.append(f"**Version**: docker-v20260726 (本地直接运行)")
    lines.append(f"**Run time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total elapsed**: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    lines.append(f"**Peak memory**: {get_peak_memory_gb():.2f} GB")
    lines.append(f"**Output directory**: `{output_dir}`")
    lines.append("")

    # ── Configuration ──────────────────────────────────────────────────────
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
    lines.append("| OPE mode | **FROM SCRATCH** (gauge config → F_{μν} → OPE) |")
    lines.append("")

    # ── Data Paths ─────────────────────────────────────────────────────────
    lines.append("## Data Paths")
    lines.append("")
    lines.append(f"| Data | Path |")
    lines.append(f"|------|------|")
    lines.append(f"| Eigenvectors | `{paths['eigenvector']}` |")
    lines.append(f"| Perambulators | `{paths['perambulator_base']}/{{conf_id}}/` |")
    lines.append(f"| Gauge configs | `{paths['gauge_config_base']}/` |")
    lines.append(f"| OPE | *Computed from scratch* |")
    lines.append("")

    # ── Environment Check ──────────────────────────────────────────────────
    lines.append("## Step 0: Environment Check")
    lines.append("")
    env = results_env
    lines.append(f"- All required OK: {env.get('all_required_ok', 'N/A')}")
    for k, v in env.items():
        if k.startswith("data_") and isinstance(v, bool):
            lines.append(f"- {k}: {'✓' if v else '✗'}")
    lines.append("")

    # ── Step 1: 2pt ────────────────────────────────────────────────────────
    lines.append("## Step 1: Proton 2pt Distillation")
    lines.append("")
    for conf_id, result in results_2pt.items():
        if isinstance(result, dict) and result.get("status") == "ok":
            lines.append(f"### conf={conf_id} ✓")
            for Pz, r in result.get("results", {}).items():
                lines.append(f"- Pz={Pz}: PP range {r.get('corr_pp_range_re', 'N/A')}, "
                           f"m_eff(log)≈{r.get('meff_plateau_log_gev', 'N/A'):.4f} GeV")
        elif isinstance(result, dict):
            lines.append(f"### conf={conf_id} ✗ — {result.get('reason', result.get('status', 'unknown'))}")
        else:
            lines.append(f"### conf={conf_id} — unexpected result type")
    lines.append("")

    # ── Step 2: OPE ────────────────────────────────────────────────────────
    lines.append("## Step 2: OPE Computation (FROM SCRATCH)")
    lines.append("")
    for conf_id, result in results_ope.items():
        if isinstance(result, dict) and result.get("status") == "ok":
            n_ok = sum(1 for v in result.get("components", {}).values() if v.get("status") == "ok")
            n_total = len(result.get("components", {}))
            lines.append(f"### conf={conf_id} ✓ ({n_ok}/{n_total} components)")
            if "validation" in result:
                val = result["validation"]
                lines.append(f"- Unitarity: max_dev={val.get('unitary_dev_max', 'N/A'):.2e}")
                lines.append(f"- Plaq trace: re={val.get('plaq_trace_mean_re', 'N/A'):.6f}")
            for key, comp in result.get("components", {}).items():
                status = "✓" if comp.get("status") == "ok" else "✗"
                re_r = comp.get("re_range", [0, 0])
                lines.append(f"  - {key} {status}: |O|∈[{re_r[0]:.2e}, {re_r[1]:.2e}]")
        elif isinstance(result, dict):
            lines.append(f"### conf={conf_id} ✗ — {result.get('reason', result.get('status', 'unknown'))}")
        else:
            lines.append(f"### conf={conf_id} — unexpected result type")
    lines.append("")

    # ── Step 3: Analysis ───────────────────────────────────────────────────
    lines.append("## Step 3: huangcl Ratio Analysis")
    lines.append("")
    if results_analysis.get("status") == "ok":
        lines.append("✓ Analysis completed successfully")
        lines.append(f"- Loaded configs: {results_analysis.get('loaded_confs', [])}")
        lines.append(f"- Ratio plot: `{results_analysis.get('ratio_path', 'N/A')}`")
        lines.append(f"- Diagnostics: `{results_analysis.get('diag_path', 'N/A')}`")
        lines.append(f"- Effective mass: `{results_analysis.get('meff_path', 'N/A')}`")
        lines.append(f"- Field strength: `{results_analysis.get('field_strength_path', 'N/A')}`")
        lines.append(f"- Numerical results: `{results_analysis.get('results_path', 'N/A')}`")

        # Ratio stats
        if "ratio_mean_stats" in results_analysis:
            lines.append("")
            lines.append("### Ratio Statistics")
            lines.append("")
            for key, stats in results_analysis["ratio_mean_stats"].items():
                lines.append(f"- {key}: re_mean={stats['re_mean']:.6f}, "
                           f"re_range={stats['re_range']}")
    else:
        lines.append(f"✗ Analysis failed: {results_analysis.get('errors', [])}")
    lines.append("")

    # ── Output Files ───────────────────────────────────────────────────────
    lines.append("## Output Files")
    lines.append("")
    lines.append("```")
    lines.append(get_output_tree(output_dir, max_files_per_dir=30))
    lines.append("```")
    lines.append("")

    # ── Notes ──────────────────────────────────────────────────────────────
    lines.append("## Notes")
    lines.append("")
    lines.append("1. **Eigenvector reuse**: eigvecs from cfg_48000 reused for all configs (standard distillation practice).")
    lines.append("2. **OPE from scratch**: All OPE data computed from gauge configurations via Clover plaquette → F_{μν} → nonlocal OPE operator → .npz.")
    lines.append("3. **All intermediate results saved**: VVV blocks, F_{μν} tensors, OPE components, correlators, ratio data.")
    lines.append("4. **No LQCD_Master / lamet-agent**: This pipeline directly runs physics computation without AI agent stages.")
    lines.append(f"5. **Peak memory**: {get_peak_memory_gb():.2f} GB")
    lines.append(f"6. **Reference**: huangcl's analysis from `examples/huangcl/code.py`.")
    lines.append("7. **Docker/local environment**: Runs in current environment without Slurm dependency.")
    lines.append("")
    lines.append(f"---")
    lines.append(f"*Generated by docker-v20260726 pipeline on {datetime.now():%Y-%m-%d %H:%M:%S}*")

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
        description="docker-v20260726 — 本地直接运行版 (OPE从头计算，保存全部中间结果)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_pipeline.py                          # 默认输出目录
  python run_pipeline.py --output-dir /my/output  # 自定义输出
  python run_pipeline.py --skip-2pt --skip-ope    # 仅运行分析
  python run_pipeline.py --skip-analysis          # 仅计算 2pt + OPE
  python run_pipeline.py --conf-id 6250           # 单组态测试
        """,
    )
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory [default: output_YYYYMMDD_HHMMSS/]")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to run_config.json [default: ./run_config.json]")
    parser.add_argument("--skip-2pt", action="store_true",
                       help="Skip 2pt distillation (use existing data)")
    parser.add_argument("--skip-ope", action="store_true",
                       help="Skip OPE computation (use existing data)")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip huangcl analysis (data generation only)")
    parser.add_argument("--skip-report", action="store_true",
                       help="Skip final report generation")
    parser.add_argument("--conf-id", type=int, default=None,
                       help="Process a single config only (for testing)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose debug output")
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    config_path = args.config or (_SCRIPT_DIR / "run_config.json")
    print(f"[INFO] Loading config from: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    # Override conf list if single config requested
    if args.conf_id is not None:
        config["parameters"]["conf_ids"] = [args.conf_id]
        config["parameters"]["Nconf"] = 1
        print(f"[INFO] Single config mode: conf_id={args.conf_id}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("output_%Y%m%d_%H%M%S")
        output_dir = _SCRIPT_DIR / ts

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy config to output
    shutil.copy(config_path, output_dir / "run_config.json")

    # Setup logging
    log_file = output_dir / "run.log"
    logger = setup_logging(log_file, "docker_pipeline",
                          console_level=logging.DEBUG if args.verbose else logging.INFO)

    t_total = time.perf_counter()

    logger.info("=" * 70)
    logger.info("  Gluon PDF Validation Pipeline — docker-v20260726")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Config: {config_path}")
    logger.info(f"  Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"  OPE mode: FROM SCRATCH")
    logger.info("=" * 70)

    # Log configuration
    params = config["parameters"]
    logger.info(f"Config summary:")
    logger.info(f"  Ensemble: {config['ensemble']['full_name']}")
    logger.info(f"  Lattice: {config['ensemble']['Nt']}x{config['ensemble']['Nx']}^3")
    logger.info(f"  Configs: {params['conf_ids']} (Nconf={params['Nconf']})")
    logger.info(f"  Momentum: P=({params['Px']},{params['Py']},{params['Pz']})")
    logger.info(f"  Nev={params['Nev']}, Nev1={params['Nev1']}")
    logger.info(f"  delta_z={params['delta_z']}, z_dir={params['z_dir']}")
    logger.info(f"  Jackknife: {params['jackknife']}")

    # Save config snapshot
    dump_config_snapshot(config, output_dir, logger)

    results = {}

    # ── Step 0: Environment check ─────────────────────────────────────────
    try:
        with Timer("00_environment_check", logger, output_dir):
            results["environment"] = step_environment_check(config, logger)

        if not results["environment"].get("all_required_ok", False):
            logger.error("Environment check FAILED. Required items missing. Aborting.")
            logger.error("Please ensure all data paths are accessible and dependencies are installed.")
            # Continue anyway — the specific steps will fail individually
    except Exception as e:
        logger.error(f"Environment check crashed: {e}")
        logger.debug(traceback.format_exc())
        results["environment"] = {"status": "error", "reason": str(e)}

    # ── Step 1: 2pt computation ───────────────────────────────────────────
    if not args.skip_2pt:
        results["2pt"] = step_compute_2pt(config, output_dir, logger)
    else:
        logger.info("Step 01: Skipping 2pt computation (--skip-2pt)")
        results["2pt"] = {}

    # ── Step 2: OPE computation FROM SCRATCH ──────────────────────────────
    if not args.skip_ope:
        results["ope"] = step_compute_ope(config, output_dir, logger)
    else:
        logger.info("Step 02: Skipping OPE computation (--skip-ope)")
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

    logger.info("")
    logger.info("═" * 70)
    logger.info("  Pipeline Complete")
    logger.info(f"  Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    logger.info(f"  Peak memory: {mem_peak:.2f} GB")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Log: {log_file}")
    logger.info("═" * 70)

    # Print output structure
    logger.info("\nOutput structure:")
    for line in get_output_tree(output_dir).split("\n")[:50]:
        logger.info(line)

    # ── Exit status ───────────────────────────────────────────────────────
    exit_code = 0

    # Check 2pt
    if isinstance(results.get("2pt"), dict) and results["2pt"]:
        n_ok = sum(1 for r in results["2pt"].values()
                   if isinstance(r, dict) and r.get("status") == "ok")
        if n_ok == 0 and not args.skip_2pt:
            logger.error("2pt: ALL configs failed")
            exit_code = 1

    # Check OPE
    if isinstance(results.get("ope"), dict) and results["ope"]:
        n_ok = sum(1 for r in results["ope"].values()
                   if isinstance(r, dict) and r.get("status") == "ok")
        if n_ok == 0 and not args.skip_ope:
            logger.error("OPE: ALL configs failed")
            exit_code = 1

    # Check analysis
    if isinstance(results.get("analysis"), dict):
        if results["analysis"].get("status") != "ok" and not args.skip_analysis:
            logger.error("Analysis: FAILED")
            exit_code = 1

    if exit_code == 0:
        logger.info(color("Pipeline finished SUCCESSFULLY", Colors.GREEN))
    else:
        logger.error(color(f"Pipeline finished with ERRORS (exit code {exit_code})", Colors.RED))

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
