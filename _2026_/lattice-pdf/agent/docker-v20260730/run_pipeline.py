#!/usr/bin/env python3
"""
Master pipeline orchestrator — docker-v20260730 GPU version.

CHANGES from v20260729:
  1. NEW data paths: per-config eigenvectors (binary per-time-slice),
     per-config perambulators (light/{conf_id}/), gauge configs (same base).
  2. Eigenvectors loaded per-config from binary LE f8 files (not shared .npy).
  3. Perambulators read new (Nspin, Nev_snk, Nt, Nev_src) per-file format.
  4. Eigenvector smearing OFF by default (perambulator encodes momentum smearing).
  5. Effective mass: fit_cosh (default), with multiple method diagnostics.

Runs the complete GPU-accelerated pipeline:
  0. Environment check (GPU detection, dependencies, data paths)
  1. Proton 2pt distillation (GPU: CuPy VVV + Wick contraction)
  2. OPE computation FROM SCRATCH (GPU: CuPy F_{mu nu} + Wilson line + OPE)
  3. huangcl ratio analysis (Jackknife + plotting)
  4. Final report generation

Usage:
    python run_pipeline.py                              # Default output dir
    python run_pipeline.py --conf-id 6250                # Single config test
    python run_pipeline.py --skip-2pt                    # Skip 2pt
    python run_pipeline.py --skip-ope                    # Skip OPE
    python run_pipeline.py --skip-analysis               # Data only
    python run_pipeline.py --precision complex128        # Double precision
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
    get_gpu_device_info, get_gpu_memory_mb, log_gpu_status,
    set_compute_dtype, get_compute_dtype, HAS_CUPY,
    save_intermediate,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 0: Environment check
# ═══════════════════════════════════════════════════════════════════════════════

def check_environment(config: dict, logger: logging.Logger) -> dict:
    """Verify GPU, Python deps, and data paths."""
    print_banner("Step 00: Environment Check", logger)
    results = {"status": "ok", "checks": {}}

    # ── Python version ────────────────────────────────────────────────────
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    logger.info(f"Python: {py_ver}")
    results["checks"]["python"] = {"version": py_ver, "ok": True}

    # ── Module imports ────────────────────────────────────────────────────
    modules = {
        "numpy": "np", "scipy": "scipy", "matplotlib": "matplotlib",
        "cupy": "cupy", "json": "json", "pathlib": "pathlib",
    }
    for mod_name, alias in modules.items():
        try:
            m = __import__(mod_name)
            ver = getattr(m, "__version__", "?")
            ok = True
        except ImportError:
            ver = "MISSING"
            ok = False
        logger.info(f"  {'✓' if ok else '✗'} {mod_name}: {ver}")
        results["checks"][mod_name] = {"version": ver, "ok": ok}

    # ── GPU/CuPy ──────────────────────────────────────────────────────────
    gpu_info = get_gpu_device_info()
    if gpu_info.get("cupy_available"):
        logger.info(f"  GPU: {gpu_info['device_name']} (CC {gpu_info['compute_capability']})")
        logger.info(f"  VRAM: {gpu_info['free_memory_gb']:.1f}/{gpu_info['total_memory_gb']:.1f} GB free")
        logger.info(f"  CuPy: {gpu_info['cupy_version']}, CUDA: {gpu_info['cuda_version']}")
        results["gpu"] = gpu_info
        results["checks"]["gpu"] = {"ok": True, **gpu_info}
    else:
        logger.warning("  GPU/CuPy not available — will run on CPU only")
        results["checks"]["gpu"] = {"ok": False, "error": gpu_info.get("error", "unknown")}

    # ── Data paths ────────────────────────────────────────────────────────
    paths = config["data_paths"]
    conf_ids = config["parameters"]["conf_ids"]
    logger.info("Checking data paths:")

    data_checks = {}

    # Eigenvectors (per-config, per-time-slice)
    eigvec_base = paths["eigenvector_base"]
    eigvec_ok = True
    for conf_id in conf_ids:
        eigvec_dir = os.path.join(eigvec_base, str(conf_id))
        t0_file = os.path.join(eigvec_dir, f"eigvecs_t000_{conf_id}")
        if os.path.exists(t0_file):
            n_files = len([f for f in os.listdir(eigvec_dir) if f.startswith("eigvecs_")])
            logger.info(f"  ✓ eigvec conf={conf_id}: {n_files} time-slice files in {eigvec_dir}")
        else:
            logger.error(f"  ✗ eigvec conf={conf_id}: MISSING ({t0_file})")
            eigvec_ok = False
    data_checks["eigenvectors"] = {"ok": eigvec_ok, "base": eigvec_base}

    # Perambulators (light/{conf_id}/)
    peram_base = paths["perambulator_base"]
    peram_ok = True
    for conf_id in conf_ids:
        peram_dir = os.path.join(peram_base, "light", str(conf_id))
        first_peram = os.path.join(peram_dir, f"perams.{conf_id}.0.0")
        if os.path.exists(first_peram):
            n_files = len([f for f in os.listdir(peram_dir) if f.startswith(f"perams.{conf_id}")])
            logger.info(f"  ✓ peram conf={conf_id}: {n_files} files in {peram_dir}")
        else:
            logger.error(f"  ✗ peram conf={conf_id}: MISSING ({first_peram})")
            peram_ok = False
    data_checks["perambulators"] = {"ok": peram_ok, "base": peram_base}

    # Gauge configs
    gauge_base = paths["gauge_config_base"]
    gauge_pattern = paths["gauge_config_pattern"]
    gauge_ok = True
    for conf_id in conf_ids:
        gf = os.path.join(gauge_base, gauge_pattern.format(conf_id=conf_id))
        if os.path.exists(gf):
            sz = os.path.getsize(gf) / 1024**3
            logger.info(f"  ✓ gauge conf={conf_id}: {gf} ({sz:.1f} GB)")
        else:
            logger.error(f"  ✗ gauge conf={conf_id}: MISSING ({gf})")
            gauge_ok = False
    data_checks["gauge_configs"] = {"ok": gauge_ok, "base": gauge_base}

    results["checks"]["data_paths"] = data_checks

    all_ok = all(c.get("ok", True) for c in results["checks"].values()
                if isinstance(c, dict))
    results["all_ok"] = all_ok

    if all_ok:
        logger.info("All checks passed! Ready to run.")
    else:
        logger.warning("Some checks failed — pipeline may encounter errors.")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Report generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(config: dict, output_dir: Path, results: dict, logger: logging.Logger) -> Path:
    """Generate final Markdown report."""
    print_banner("Step 04: Final Report", logger)

    params = config["parameters"]
    ensemble = config["ensemble"]
    timing_path = output_dir / "timing.jsonl"

    lines = [
        f"# Gluon PDF Pipeline — docker-v20260730 GPU Report",
        f"",
        f"**Generated**: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"**Run ID**: {config['run_id']}",
        f"**Precision**: {config['precision']['compute_dtype']}",
        f"",
        f"## Ensemble",
        f"",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Name | {ensemble['full_name']} |",
        f"| L³×T | {ensemble['Nx']}³×{ensemble['Nt']} |",
        f"| β | {ensemble['beta']} |",
        f"| a (fm) | {ensemble['alttc']} |",
        f"| Nc | {ensemble['Nc']} |",
        f"",
        f"## Parameters",
        f"",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Nev | {params['Nev']} |",
        f"| Momentum | P=({params['Px']},{params['Py']},{params['Pz']}) |",
        f"| Element | {params['element']} |",
        f"| Configs | {params['conf_ids']} (Nconf={params['Nconf']}) |",
        f"| Smearing | {'ON' if params['apply_eigenvec_smearing'] else 'OFF'} |",
        f"| meff method | {params['meff_method']} |",
        f"| Jackknife | {params['jackknife']} |",
        f"",
        f"## Data Paths",
        f"",
        f"| Data | Path |",
        f"|------|------|",
        f"| Eigenvectors | `{config['data_paths']['eigenvector_base']}/{{conf_id}}/` |",
        f"| Perambulators | `{config['data_paths']['perambulator_base']}/light/{{conf_id}}/` |",
        f"| Gauge configs | `{config['data_paths']['gauge_config_base']}` |",
        f"",
    ]

    # Timing summary
    if timing_path.exists():
        lines.append("## Timing")
        lines.append("")
        lines.append("| Step | Time (s) | GPU Free (MB) |")
        lines.append("|------|----------|---------------|")
        total_time = 0.0
        with open(timing_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    step = rec.get("step", "?")
                    elapsed = rec.get("elapsed_s", 0)
                    gpu_free = rec.get("gpu_free_after_mb", "?")
                    lines.append(f"| {step} | {elapsed:.1f} | {gpu_free} |")
                    total_time += elapsed
                except Exception:
                    pass
        lines.append(f"| **TOTAL** | **{total_time:.1f}** | |")
        lines.append("")

    # 2pt results
    lines.append("## 2pt Results")
    lines.append("")
    two_pt_summary = output_dir / "data" / "compute_2pt_summary.json"
    if two_pt_summary.exists():
        with open(two_pt_summary) as f:
            tp_data = json.load(f)
        lines.append("| Config | Pz | meff (GeV) | Method |")
        lines.append("|--------|----|------------|--------|")
        for conf_id_str, conf_data in tp_data.items():
            if conf_data.get("status") == "ok":
                for Pz_str, pz_data in conf_data.get("results", {}).items():
                    lines.append(f"| {conf_id_str} | {Pz_str} | {pz_data.get('meff_plateau_gev', '?'):.4f} | {pz_data.get('meff_method', '?')} |")
                    if 'all_meff_methods' in pz_data:
                        for m, v in pz_data['all_meff_methods'].items():
                            lines.append(f"| | | {v:.4f} ({m}) |")
        lines.append("")

    # OPE results
    lines.append("## OPE Results")
    lines.append("")
    ope_summary = output_dir / "data" / "compute_ope_summary.json"
    if ope_summary.exists():
        with open(ope_summary) as f:
            ope_data = json.load(f)
        for conf_id_str, conf_data in ope_data.items():
            status = conf_data.get("status", "?")
            lines.append(f"- Config {conf_id_str}: **{status}**")
            if "components" in conf_data:
                for comp_key, comp_data in conf_data["components"].items():
                    if comp_data.get("status") == "ok":
                        lines.append(f"  - {comp_key}: shape={comp_data.get('shape')}, "
                                   f"|O|=[{comp_data.get('re_range', [0,0])[0]:.2e}, {comp_data.get('re_range', [0,0])[1]:.2e}]")
        lines.append("")

    # Analysis results
    lines.append("## huangcl Analysis")
    lines.append("")
    analysis_summary = output_dir / "plots" / "analysis_summary.json"
    if analysis_summary.exists():
        with open(analysis_summary) as f:
            ana_data = json.load(f)
        lines.append(f"- Status: **{ana_data.get('status', '?')}**")
        lines.append(f"- Loaded configs: {ana_data.get('loaded_confs', [])}")
        lines.append(f"- Elapsed: {ana_data.get('elapsed_seconds', 0):.1f}s")
        lines.append("")

    # Output tree
    lines.append("## Output Files")
    lines.append("")
    lines.append("```")
    lines.append(get_output_tree(output_dir))
    lines.append("```")
    lines.append("")

    report_path = output_dir / "final_report.md"
    report_path.write_text("\n".join(lines))
    logger.info(f"Report saved to {report_path}")
    return report_path


# ═══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="docker-v20260730 GPU pipeline — Gluon PDF disconnected analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                              # Full pipeline (3 configs)
  %(prog)s --conf-id 6250 --skip-ope --skip-analysis    # Single-config 2pt test
  %(prog)s --skip-2pt                                   # OPE + analysis only
  %(prog)s --precision complex128                        # Double precision
  %(prog)s --smear --meff-method fit_exp                # With eigenvector smearing
        """,
    )
    parser.add_argument("--conf-id", type=int, default=None,
                       help="Override: single config ID")
    parser.add_argument("--conf-ids", type=int, nargs="+", default=None,
                       help="Override: list of config IDs")
    parser.add_argument("--skip-2pt", action="store_true", help="Skip 2pt step")
    parser.add_argument("--skip-ope", action="store_true", help="Skip OPE step")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis step")
    parser.add_argument("--skip-env-check", action="store_true", help="Skip environment check")
    parser.add_argument("--smear", action="store_true", help="Enable eigenvector smearing")
    parser.add_argument("--meff-method", type=str, default=None,
                       choices=["fit_cosh", "fit_exp", "exp_forward", "cosh"],
                       help="Effective mass method")
    parser.add_argument("--precision", type=str, default="complex64",
                       choices=["complex64", "complex128"],
                       help="Compute precision (default: complex64)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Override output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    config_path = _SCRIPT_DIR / "run_config.json"
    with open(config_path) as f:
        config = json.load(f)

    # ── Apply CLI overrides ───────────────────────────────────────────────
    set_compute_dtype(args.precision)
    config["precision"]["compute_dtype"] = args.precision

    if args.conf_id is not None:
        config["parameters"]["conf_ids"] = [args.conf_id]
        config["parameters"]["Nconf"] = 1
    if args.conf_ids is not None:
        config["parameters"]["conf_ids"] = args.conf_ids
        config["parameters"]["Nconf"] = len(args.conf_ids)
    if args.smear:
        config["parameters"]["apply_eigenvec_smearing"] = True
    if args.meff_method:
        config["parameters"]["meff_method"] = args.meff_method

    # ── Setup output directory ────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        timestamp = datetime.now().strftime(config["output"]["timestamp_format"])
        output_dir = (_SCRIPT_DIR / timestamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config["output"]["base_dir"] = str(output_dir)

    # ── Setup logging ─────────────────────────────────────────────────────
    logger = setup_logging(
        output_dir / "run.log",
        name="pipeline_v20260730",
        console_level=logging.DEBUG if args.verbose else logging.INFO,
    )

    logger.info(f"{'═'*70}")
    logger.info(f"  docker-v20260730 GPU Pipeline")
    logger.info(f"  Run dir: {output_dir}")
    logger.info(f"  Precision: {args.precision}")
    logger.info(f"  Configs: {config['parameters']['conf_ids']}")
    logger.info(f"{'═'*70}")

    dump_config_snapshot(config, output_dir, logger)

    pipeline_start = time.perf_counter()
    all_results = {"run_id": config["run_id"], "steps": {}}

    # ── Step 0: Environment check ─────────────────────────────────────────
    if not args.skip_env_check:
        try:
            env_results = check_environment(config, logger)
            all_results["steps"]["00_env_check"] = env_results
            save_intermediate(env_results, output_dir, "env_check.json", logger)
        except Exception as e:
            logger.error(f"Environment check failed: {e}")
            if not args.skip_env_check:
                traceback.print_exc()
                sys.exit(1)
    else:
        logger.info("Environment check SKIPPED")

    data_dir = output_dir / "data"
    plots_dir = output_dir / "plots"

    # ── Step 1: 2pt computation ───────────────────────────────────────────
    if not args.skip_2pt:
        try:
            from compute_2pt_gpu import run_2pt_computation_gpu
            tp_results = run_2pt_computation_gpu(config, data_dir, logger)
            all_results["steps"]["01_2pt_gpu"] = {
                "status": "ok",
                "n_configs": len(tp_results),
                "configs": {str(k): v.get("status") for k, v in tp_results.items()},
            }
            save_intermediate(tp_results, data_dir, "compute_2pt_summary.json", logger)
        except Exception as e:
            logger.error(f"2pt computation failed: {e}")
            traceback.print_exc()
            all_results["steps"]["01_2pt_gpu"] = {"status": "error", "error": str(e)}
    else:
        logger.info("2pt computation SKIPPED")

    # ── Step 2: OPE computation ───────────────────────────────────────────
    if not args.skip_ope:
        try:
            from compute_ope_gpu import compute_ope_all_configs_gpu
            ope_results = compute_ope_all_configs_gpu(config, data_dir, logger)
            all_results["steps"]["02_ope_gpu"] = {
                "status": "ok",
                "n_configs": len(ope_results),
                "configs": {str(k): v.get("status") for k, v in ope_results.items()},
            }
            save_intermediate(ope_results, data_dir, "compute_ope_summary.json", logger)
        except Exception as e:
            logger.error(f"OPE computation failed: {e}")
            traceback.print_exc()
            all_results["steps"]["02_ope_gpu"] = {"status": "error", "error": str(e)}
    else:
        logger.info("OPE computation SKIPPED")

    # ── Step 3: huangcl analysis ──────────────────────────────────────────
    if not args.skip_analysis:
        try:
            from analyze_ratio import run_analysis
            ana_results = run_analysis(config, data_dir, plots_dir, logger)
            all_results["steps"]["03_analysis"] = {
                "status": ana_results.get("status", "?"),
                "loaded_confs": ana_results.get("loaded_confs", []),
            }
            save_intermediate(ana_results, plots_dir, "analysis_summary.json", logger)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            traceback.print_exc()
            all_results["steps"]["03_analysis"] = {"status": "error", "error": str(e)}
    else:
        logger.info("Analysis SKIPPED")

    # ── Step 4: Report ────────────────────────────────────────────────────
    try:
        report_path = generate_report(config, output_dir, all_results, logger)
        all_results["steps"]["04_report"] = {"status": "ok", "path": str(report_path)}
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        all_results["steps"]["04_report"] = {"status": "error", "error": str(e)}

    # ── Final summary ─────────────────────────────────────────────────────
    pipeline_elapsed = time.perf_counter() - pipeline_start
    all_results["total_elapsed_seconds"] = pipeline_elapsed

    final_path = output_dir / "run_results.json"
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\n{'═'*70}")
    logger.info(f"  Pipeline complete in {pipeline_elapsed:.1f}s ({pipeline_elapsed/60:.1f} min)")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Results: {final_path}")
    logger.info(f"  Report: {output_dir / 'final_report.md'}")
    logger.info(f"{'═'*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
