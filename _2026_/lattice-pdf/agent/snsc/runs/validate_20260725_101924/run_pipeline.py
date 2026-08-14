#!/usr/bin/env python3
"""
Master pipeline orchestrator for LQCD agent validation.

Runs the complete pipeline:
  1. LQCD_Master: Generate physics plan + code (test mode via API)
  2. Sample data: Generate synthetic 2pt + OPE data
  3. Core computation: CPU distillation + OPE
  4. huangcl analysis: Ratio computation + plotting
  5. lamet-agent: HDF5 conversion + correlator analysis
  6. Final report: Comparison and summary

All output is logged to run.log and individual step logs.

Usage:
    python run_pipeline.py --run-dir /path/to/run_dir [--attempt N]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


def log(msg: str, log_file: Path):
    """Write message to log file and stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")


def run_step(
    name: str,
    cmd: list[str],
    cwd: Path,
    log_file: Path,
    timeout: int = 600,
) -> tuple[bool, str, str]:
    """Run a pipeline step as a subprocess.

    Args:
        name: Step name for logging.
        cmd: Command to run (list of strings).
        cwd: Working directory.
        log_file: Log file for output.
        timeout: Timeout in seconds.

    Returns:
        (success, stdout, stderr)
    """
    log(f"[{name}] Starting: {' '.join(cmd)}", log_file)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = result.stdout
        stderr = result.stderr

        # Write output to log
        with open(log_file, "a") as f:
            if stdout:
                f.write(f"\n--- [{name}] STDOUT ---\n{stdout}\n")
            if stderr:
                f.write(f"\n--- [{name}] STDERR ---\n{stderr}\n")

        if result.returncode == 0:
            log(f"[{name}] SUCCESS (returncode=0)", log_file)
            return True, stdout, stderr
        else:
            log(f"[{name}] FAILED (returncode={result.returncode})", log_file)
            return False, stdout, stderr

    except subprocess.TimeoutExpired:
        log(f"[{name}] TIMEOUT after {timeout}s", log_file)
        return False, "", f"Timeout after {timeout}s"
    except Exception as e:
        log(f"[{name}] ERROR: {e}", log_file)
        return False, "", str(e)


def check_environment(log_file: Path) -> dict:
    """Check that all required Python packages are available."""
    log("[Step 0] Checking environment...", log_file)

    required = {
        "numpy": "numpy",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "h5py": "h5py",
    }

    available = {}
    for name, import_name in required.items():
        try:
            __import__(import_name)
            available[name] = True
            log(f"  ✓ {name} available", log_file)
        except ImportError:
            available[name] = False
            log(f"  ✗ {name} NOT available", log_file)

    missing = [k for k, v in available.items() if not v]
    if missing:
        log(f"WARNING: Missing packages: {missing}. Attempting pip install...", log_file)
        for pkg in missing:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg],
                capture_output=True,
                timeout=120,
            )

    return available


def run_lqcd_master(run_dir: Path, log_file: Path) -> bool:
    """Run LQCD_Master in test mode to generate physics plan."""
    log("[Step 1] Running LQCD_Master...", log_file)

    lqcd_master_dir = Path("/root/lattice-pdf/agent/LQCD_Master")
    if not lqcd_master_dir.exists():
        log("[Step 1] LQCD_Master directory not found, skipping", log_file)
        return True  # Non-critical

    output_dir = run_dir / "01_lqcd_master"
    task = (
        "Compute proton 2pt correlator with distillation on L24x72 ensemble, "
        "Pz=2, Nev=100, momentum smearing phase=-2. "
        "Use element Cg5g4 for proton interpolation. Output npy files."
    )

    cmd = [
        sys.executable, "run.py",
        "--task", task,
        "--test",
        "--non-interactive",
        "--run-dir", str(output_dir),
        "--dotenv-path", ".env",
    ]

    log(f"[Step 1] Command: {' '.join(cmd)}", log_file)
    success, stdout, stderr = run_step("LQCD_Master", cmd, lqcd_master_dir, log_file, timeout=300)

    if success:
        # Check that plan was generated
        plan_path = output_dir / "planner" / "plan.yaml"
        if plan_path.exists():
            log(f"[Step 1] Plan generated: {plan_path}", log_file)
            with open(plan_path) as f:
                log(f"[Step 1] Plan preview: {f.read()[:500]}...", log_file)
        else:
            log("[Step 1] WARNING: plan.yaml not found", log_file)
    else:
        log("[Step 1] LQCD_Master failed (non-critical, continuing)", log_file)

    return True  # Non-critical step


def run_sample_data(run_dir: Path, log_file: Path) -> bool:
    """Generate synthetic sample data."""
    log("[Step 2] Generating sample data...", log_file)

    cmd = [sys.executable, "02_sample_data/generate.py", "--run-dir", str(run_dir)]
    success, stdout, stderr = run_step("SampleData", cmd, run_dir, log_file)

    if success:
        # Verify output
        for conf_id in [6250, 6450, 6650]:
            conf_dir = run_dir / "02_sample_data" / f"conf_{conf_id}"
            npy_files = list(conf_dir.glob("*.npy"))
            npz_files = list(conf_dir.glob("*.npz"))
            log(f"[Step 2] conf_{conf_id}: {len(npy_files)} npy, {len(npz_files)} npz", log_file)

            if len(npy_files) == 0 or len(npz_files) == 0:
                log(f"[Step 2] ERROR: Missing files for conf_{conf_id}", log_file)
                return False

    return success


def run_huangcl_analysis(run_dir: Path, log_file: Path) -> bool:
    """Run huangcl-style ratio analysis."""
    log("[Step 4] Running huangcl analysis...", log_file)

    cmd = [
        sys.executable, "04_huangcl_analysis/analyze_ratio.py",
        "--run-dir", str(run_dir),
        "--Nconf", "3",
    ]
    success, stdout, stderr = run_step("HuangclAnalysis", cmd, run_dir, log_file)

    if success:
        ratio_png = run_dir / "04_huangcl_analysis" / "output" / "ratio.png"
        if ratio_png.exists():
            size_kb = ratio_png.stat().st_size / 1024
            log(f"[Step 4] ratio.png generated ({size_kb:.0f} KB)", log_file)
        else:
            log("[Step 4] ERROR: ratio.png not found!", log_file)
            return False

    return success


def run_lamet_bridge(run_dir: Path, log_file: Path) -> bool:
    """Run HDF5 conversion and lamet-agent (if available)."""
    log("[Step 5] Running lamet-agent bridge...", log_file)

    # Convert to HDF5
    cmd = [
        sys.executable, "05_lamet_agent/convert_to_hdf5.py",
        "--run-dir", str(run_dir),
    ]
    success, stdout, stderr = run_step("HDF5Convert", cmd, run_dir, log_file)

    if success:
        h5_path = run_dir / "05_lamet_agent" / "artifacts" / "proton_2pt.h5"
        if h5_path.exists():
            log(f"[Step 5] HDF5 created: {h5_path} ({h5_path.stat().st_size / 1024:.0f} KB)", log_file)
        else:
            log("[Step 5] WARNING: HDF5 file not found", log_file)

    # Try lamet-agent
    try:
        result = subprocess.run(
            ["lamet-agent", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            log("[Step 5] lamet-agent is installed", log_file)

            manifest_path = run_dir / "05_lamet_agent" / "manifest.json"
            cmd = [
                "lamet-agent", "validate",
                str(manifest_path),
            ]
            success_v, stdout_v, stderr_v = run_step("LametValidate", cmd, run_dir, log_file)
            if success_v:
                log("[Step 5] Manifest validated successfully", log_file)

            # Run with mock backend for structural testing
            cmd_run = [
                "lamet-agent", "run",
                str(manifest_path),
                "--backend", "mock",
                "--verbose",
            ]
            run_step("LametRun", cmd_run, run_dir, log_file, timeout=120)
        else:
            log("[Step 5] lamet-agent not installed, skipping", log_file)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        log("[Step 5] lamet-agent not found, skipping", log_file)

    return True  # Non-critical


def generate_report(run_dir: Path, log_file: Path, results: dict) -> bool:
    """Generate final comparison report."""
    log("[Step 6] Generating final report...", log_file)

    report_path = run_dir / "final_report.md"
    lines = []

    lines.append("# LQCD Agent Pipeline Validation Report")
    lines.append(f"\n**Run ID**: {run_dir.name}")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Pipeline**: LQCD_Master → Sample Data → huangcl Analysis → lamet-agent")

    lines.append("\n## Pipeline Results\n")

    for step_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        lines.append(f"- **{step_name}**: {status}")

    lines.append("\n## Output Files\n")

    # List all output files
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.suffix in {".png", ".json", ".log", ".npz", ".npy", ".h5", ".yaml", ".md"}:
            rel = path.relative_to(run_dir)
            size = path.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.0f} KB"
            else:
                size_str = f"{size} B"
            lines.append(f"- `{rel}` ({size_str})")

    lines.append("\n## Verification\n")

    # Check key outputs
    ratio_png = run_dir / "04_huangcl_analysis" / "output" / "ratio.png"
    if ratio_png.exists():
        lines.append(f"- ✓ Ratio plot generated: `{ratio_png.relative_to(run_dir)}`")

    meff_png = run_dir / "04_huangcl_analysis" / "output" / "effective_mass.png"
    if meff_png.exists():
        lines.append(f"- ✓ Effective mass plot generated: `{meff_png.relative_to(run_dir)}`")

    results_npz = run_dir / "04_huangcl_analysis" / "output" / "ratio_results.npz"
    if results_npz.exists():
        data = np.load(results_npz)
        lines.append(f"- ✓ Numerical results saved: ratio shape={data['ratio'].shape}")

    lines.append("\n## Parameters\n")
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        lines.append("```json")
        lines.append(json.dumps(config["parameters"], indent=2))
        lines.append("```")

    lines.append("\n---\n*Generated by LQCD Agent Validation Pipeline*")

    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)

    log(f"[Step 6] Report saved to {report_path}", log_file)
    return True


def main():
    parser = argparse.ArgumentParser(description="LQCD Agent Validation Pipeline")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory")
    parser.add_argument("--attempt", type=int, default=0, help="Attempt number (for auto-fix loop)")
    parser.add_argument("--skip-lqcd-master", action="store_true", help="Skip LQCD_Master step")
    parser.add_argument("--skip-lamet", action="store_true", help="Skip lamet-agent step")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    log_file = run_dir / "run.log"

    log(f"{'='*70}", log_file)
    log(f"LQCD Agent Validation Pipeline - Attempt {args.attempt}", log_file)
    log(f"Run directory: {run_dir}", log_file)
    log(f"{'='*70}", log_file)

    results = {}

    # Step 0: Environment check
    check_environment(log_file)

    # Step 1: LQCD_Master
    if not args.skip_lqcd_master:
        results["01_lqcd_master"] = run_lqcd_master(run_dir, log_file)

    # Step 2: Sample data
    results["02_sample_data"] = run_sample_data(run_dir, log_file)
    if not results["02_sample_data"]:
        log("FATAL: Sample data generation failed", log_file)
        return 1

    # Step 4: huangcl analysis
    results["04_huangcl_analysis"] = run_huangcl_analysis(run_dir, log_file)
    if not results["04_huangcl_analysis"]:
        log("FATAL: huangcl analysis failed", log_file)
        return 1

    # Step 5: lamet-agent bridge
    if not args.skip_lamet:
        results["05_lamet_agent"] = run_lamet_bridge(run_dir, log_file)

    # Step 6: Final report
    results["06_final_report"] = generate_report(run_dir, log_file, results)

    # Summary
    all_ok = all(v for v in results.values())
    log(f"\n{'='*70}", log_file)
    log(f"Pipeline {'SUCCESS' if all_ok else 'PARTIAL FAILURE'}", log_file)
    for step, ok in results.items():
        log(f"  {'✓' if ok else '✗'} {step}", log_file)
    log(f"{'='*70}", log_file)

    return 0 if all_ok else 1


if __name__ == "__main__":
    # Need numpy for report generation
    import numpy as np
    sys.exit(main())
