#!/usr/bin/env python3
"""
Master pipeline orchestrator — snsc-v20260726 整合版验证管线。

Runs the complete pipeline:
  0. Environment check
  1. Proton 2pt distillation (real eigvecs + perambulators)
  2. OPE data loading (donghx pre-computed)
  3. huangcl ratio analysis (Jackknife + plotting)
  4. Final report generation

Unlike the agent/snsc version, this does NOT call LQCD_Master or lamet-agent.
All computation uses real lattice data from cluster paths.

Usage:
    python run_pipeline.py                          # Use default output dir
    python run_pipeline.py --output-dir /path/out    # Custom output dir
    python run_pipeline.py --skip-2pt                # Skip 2pt (if already computed)
    python run_pipeline.py --skip-ope                # Skip OPE loading
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ─── Ensure snsc-v20260726 is on sys.path ───────────────────────────────────
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
)


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline steps
# ═══════════════════════════════════════════════════════════════════════════════

def step_environment_check(config: dict, logger) -> dict:
    """Check Python environment and dependencies."""
    print_banner("Step 00: Environment Check", logger)
    logger.info(f"Python: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")

    results = {}

    # Check essential modules
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
            logger.warning(f"  ✗ {name}: NOT AVAILABLE")
            results[name] = False

    # Check snsc/main.py
    snsc_path = _REPO_ROOT / "snsc" / "main.py"
    if snsc_path.exists():
        logger.info(f"  ✓ snsc/main.py: {snsc_path} ({snsc_path.stat().st_size:,} bytes)")
        results["snsc_main"] = True
    else:
        logger.warning(f"  ✗ snsc/main.py: NOT FOUND at {snsc_path}")
        results["snsc_main"] = False

    # Check data paths (best-effort, may not be accessible from local machine)
    paths = config["data_paths"]
    logger.info("Data path checks (read-only):")
    for key, path in [
        ("eigenvector", paths["eigenvector"]),
        ("perambulator_base", paths["perambulator_base"]),
        ("ope_base", paths["ope_base"]),
        ("gauge_config_base", paths["gauge_config_base"]),
    ]:
        exists = os.path.exists(path)
        status = "✓" if exists else "⚠ (may only exist on cluster)"
        logger.info(f"  {status} {key}: {path}")

    all_ok = all(v for v in results.values() if isinstance(v, bool))
    if all_ok:
        logger.info("Environment check: ALL OK")
    else:
        missing = [k for k, v in results.items() if not v and isinstance(v, bool)]
        logger.warning(f"Environment check: MISSING: {missing}")

    return results


def step_compute_2pt(config: dict, output_dir: Path, logger) -> dict:
    """Run proton 2pt distillation computation."""
    print_banner("Step 01: Proton 2pt Distillation", logger)

    from compute_2pt import run_2pt_computation
    data_dir = output_dir / "data"

    with Timer("2pt_total", logger, output_dir):
        results = run_2pt_computation(config, data_dir, logger)

    # Check results
    all_ok = all(r["status"] == "ok" for r in results.values())
    status_str = color("✓ ALL OK", Colors.GREEN) if all_ok else color("⚠ PARTIAL", Colors.YELLOW)
    logger.info(f"2pt computation: {status_str}")

    return results


def step_load_ope(config: dict, output_dir: Path, logger) -> dict:
    """Load pre-computed OPE data."""
    print_banner("Step 02: Load OPE Data", logger)

    from compute_ope import load_ope_data
    data_dir = output_dir / "data"

    with Timer("ope_total", logger, output_dir):
        results = load_ope_data(config, data_dir, logger)

    all_ok = all(r["status"] == "ok" for r in results.values())
    status_str = color("✓ ALL OK", Colors.GREEN) if all_ok else color("⚠ PARTIAL", Colors.YELLOW)
    logger.info(f"OPE loading: {status_str}")

    return results


def step_huangcl_analysis(config: dict, output_dir: Path, logger) -> dict:
    """Run huangcl-style ratio analysis."""
    print_banner("Step 03: huangcl Ratio Analysis", logger)

    from analyze_ratio import run_analysis
    data_dir = output_dir / "data"
    plots_dir = output_dir / "plots"

    with Timer("analysis_total", logger, output_dir):
        results = run_analysis(config, data_dir, plots_dir, logger)

    if results.get("status") == "ok":
        logger.info(f"Analysis: {color('✓ OK', Colors.GREEN)}")
    else:
        logger.error(f"Analysis: {color('✗ FAILED', Colors.RED)}")

    return results


def step_final_report(
    config: dict, output_dir: Path, logger,
    results_2pt: dict, results_ope: dict, results_analysis: dict,
    elapsed_total: float,
) -> Path:
    """Generate final markdown report."""
    print_banner("Step 04: Final Report", logger)

    params = config["parameters"]
    ensemble = config["ensemble"]
    paths = config["data_paths"]

    report_path = output_dir / "final_report.md"

    lines = []
    lines.append("# SNSC Validation Pipeline Report")
    lines.append(f"\n**Run time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total elapsed**: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    lines.append(f"**Output directory**: `{output_dir}`")

    # ── Configuration ──────────────────────────────────────────────────────
    lines.append("\n## Configuration\n")
    lines.append(f"- **Ensemble**: {ensemble['full_name']} ({ensemble['name']})")
    lines.append(f"- **Lattice**: {ensemble['Nt']}×{ensemble['Nx']}³, β={ensemble['beta']}")
    lines.append(f"- **Lattice spacing**: a={ensemble['alttc']} fm")
    lines.append(f"- **Configs**: {params['conf_ids']} (Nconf={params['Nconf']})")
    lines.append(f"- **Momentum**: P=({params['Px']},{params['Py']},{params['Pz']})")
    lines.append(f"- **Element**: {params['element']}")
    lines.append(f"- **Jackknife**: {params['jackknife']}")

    lines.append("\n### Data Paths\n")
    lines.append(f"- Eigenvectors (cfg {paths['eigenvector_cfg']}): `{paths['eigenvector']}`")
    lines.append(f"- Perambulators: `{paths['perambulator_base']}/{{conf_id}}/`")
    lines.append(f"- OPE (pre-computed): `{paths['ope_base']}/{{conf_id}}/`")
    lines.append(f"- Gauge configs: `{paths['gauge_config_base']}/`")

    # ── Pipeline Results ───────────────────────────────────────────────────
    lines.append("\n## Pipeline Results\n")

    # 2pt
    lines.append("### Step 1: Proton 2pt Distillation\n")
    for conf_id, result in results_2pt.items():
        if result["status"] == "ok":
            lines.append(f"- **conf={conf_id}**: ✓ {len(result['results'])} momenta computed")
            for Pz, r in result["results"].items():
                lines.append(f"  - Pz={Pz}: PP range {r['corr_pp_range_re']}, meff≈{r['meff_plateau_gev']:.4f} GeV")
        else:
            lines.append(f"- **conf={conf_id}**: ✗ {result.get('reason', result['status'])}")

    # OPE
    lines.append("\n### Step 2: OPE Data Loading\n")
    for conf_id, result in results_ope.items():
        ok_count = sum(1 for v in result.get("components", {}).values() if v.get("status") == "ok")
        total = len(result.get("components", {}))
        status = "✓" if result["status"] == "ok" else "⚠"
        lines.append(f"- **conf={conf_id}**: {status} {ok_count}/{total} components loaded")

    # Analysis
    lines.append("\n### Step 3: huangcl Ratio Analysis\n")
    if results_analysis.get("status") == "ok":
        lines.append("- ✓ Analysis completed successfully")
        lines.append(f"  - Ratio plot: `{results_analysis.get('ratio_path', 'N/A')}`")
        lines.append(f"  - Diagnostics: `{results_analysis.get('diag_path', 'N/A')}`")
        lines.append(f"  - Effective mass: `{results_analysis.get('meff_path', 'N/A')}`")
        lines.append(f"  - Numerical results: `{results_analysis.get('results_path', 'N/A')}`")
    else:
        lines.append(f"- ✗ Analysis failed: {results_analysis.get('errors', [])}")

    # ── Output Files ───────────────────────────────────────────────────────
    lines.append("\n## Output Files\n")
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.suffix in {".png", ".json", ".log", ".npz", ".npy", ".md"}:
            rel = path.relative_to(output_dir)
            size = format_size(path.stat().st_size)
            lines.append(f"- `{rel}` ({size})")

    # ── Notes ──────────────────────────────────────────────────────────────
    lines.append("\n## Notes\n")
    lines.append("1. **Eigenvector reuse**: eigvecs from cfg_48000 are reused for all configs (standard distillation practice).")
    lines.append("2. **Pre-computed OPE**: OPE data loaded from donghx's pre-computed directory.")
    lines.append("3. **No LQCD_Master / lamet-agent**: This pipeline skips the AI agent stages — it directly runs the physics computation.")
    lines.append(f"4. **Peak memory**: {get_peak_memory_gb():.2f} GB")
    lines.append("5. **Reference**: huangcl's analysis from `examples/huangcl/code.py`.")
    lines.append(f"\n---\n*Generated by snsc-v20260726 pipeline on {datetime.now():%Y-%m-%d %H:%M:%S}*")

    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)

    logger.info(f"Report saved to {report_path}")
    return report_path


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="snsc-v20260726 — 整合版验证管线 (无 LQCD_Master/lamet-agent)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_pipeline.py                          # 默认输出目录
  python run_pipeline.py --output-dir /my/output  # 自定义输出
  python run_pipeline.py --skip-2pt --skip-ope    # 仅运行分析
  python run_pipeline.py --skip-analysis          # 仅计算 2pt + OPE
        """,
    )
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory [default: output_YYYYMMDD_HHMMSS/]")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to run_config.json [default: ./run_config.json]")
    parser.add_argument("--skip-2pt", action="store_true",
                       help="Skip 2pt distillation (use existing data)")
    parser.add_argument("--skip-ope", action="store_true",
                       help="Skip OPE loading (use existing data)")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip huangcl analysis")
    parser.add_argument("--skip-report", action="store_true",
                       help="Skip final report generation")
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    config_path = args.config or (_SCRIPT_DIR / "run_config.json")
    with open(config_path) as f:
        config = json.load(f)

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
    logger = setup_logging(log_file, "snsc_pipeline")

    t_total = time.perf_counter()

    logger.info("=" * 70)
    logger.info("  SNSC Validation Pipeline — snsc-v20260726")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Config: {config_path}")
    logger.info(f"  Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info("=" * 70)

    results = {}

    # ── Step 0: Environment check ─────────────────────────────────────────
    with Timer("00_environment_check", logger, output_dir):
        results["environment"] = step_environment_check(config, logger)

    # ── Step 1: 2pt computation ───────────────────────────────────────────
    if not args.skip_2pt:
        with Timer("01_compute_2pt", logger, output_dir):
            results["2pt"] = step_compute_2pt(config, output_dir, logger)
    else:
        logger.info("Step 01: Skipping 2pt computation (--skip-2pt)")

    # ── Step 2: OPE loading ───────────────────────────────────────────────
    if not args.skip_ope:
        with Timer("02_load_ope", logger, output_dir):
            results["ope"] = step_load_ope(config, output_dir, logger)
    else:
        logger.info("Step 02: Skipping OPE loading (--skip-ope)")

    # ── Step 3: huangcl analysis ──────────────────────────────────────────
    if not args.skip_analysis:
        with Timer("03_huangcl_analysis", logger, output_dir):
            results["analysis"] = step_huangcl_analysis(config, output_dir, logger)
    else:
        logger.info("Step 03: Skipping analysis (--skip-analysis)")

    # ── Step 4: Final report ──────────────────────────────────────────────
    if not args.skip_report:
        with Timer("04_final_report", logger, output_dir):
            report_path = step_final_report(
                config, output_dir, logger,
                results.get("2pt", {}),
                results.get("ope", {}),
                results.get("analysis", {}),
                time.perf_counter() - t_total,
            )
        results["report"] = str(report_path)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_total
    mem_peak = get_peak_memory_gb()

    logger.info(f"\n{'═'*70}")
    logger.info(f"  Pipeline Complete")
    logger.info(f"  Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    logger.info(f"  Peak memory: {mem_peak:.2f} GB")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Log: {log_file}")
    logger.info(f"{'═'*70}")

    # Print file tree
    logger.info("\nOutput structure:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(str(output_dir), "").count(os.sep)
        indent = "  " * level
        folder = os.path.basename(root) or str(output_dir)
        logger.info(f"{indent}{folder}/")
        sub_indent = "  " * (level + 1)
        for file in sorted(files)[:20]:  # Show first 20 files per dir
            fpath = os.path.join(root, file)
            logger.info(f"{sub_indent}{file}  ({format_size(os.path.getsize(fpath))})")
        if len(files) > 20:
            logger.info(f"{sub_indent}... and {len(files) - 20} more files")

    return 0


if __name__ == "__main__":
    sys.exit(main())
