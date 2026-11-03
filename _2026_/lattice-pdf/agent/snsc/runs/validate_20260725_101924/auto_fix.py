#!/usr/bin/env python3
"""
Auto-fix script for the LQCD agent validation pipeline.

Parses error logs and attempts automatic fixes for common error patterns.

Usage:
    python auto_fix.py --run-dir /path/to/run_dir --attempt N [--log run.log]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ERROR_PATTERNS = {
    "import_error": {
        "regex": r"ImportError: No module named '(\w+)'",
        "fix": "pip_install",
    },
    "modulenotfound_error": {
        "regex": r"ModuleNotFoundError: No module named '(\w+)'",
        "fix": "pip_install",
    },
    "file_not_found": {
        "regex": r"FileNotFoundError:.*?['\"]([^'\"]+npy[^'\"]*|conf\d+[^'\"]*)['\"]",
        "fix": "generate_missing_file",
    },
    "shape_mismatch": {
        "regex": r"shape mismatch.*?(\d+).*?(\d+)",
        "fix": "fix_shape",
    },
    "broadcast_error": {
        "regex": r"operands could not be broadcast together with shapes \((\S+)\) \((\S+)\)",
        "fix": "fix_broadcast",
    },
    "key_error": {
        "regex": r"KeyError: '(\w+)'",
        "fix": "fix_key_error",
    },
    "value_error": {
        "regex": r"ValueError: (.*)",
        "fix": "fix_value_error",
    },
    "attribute_error": {
        "regex": r"AttributeError: (.*)",
        "fix": "fix_attribute_error",
    },
    "type_error": {
        "regex": r"TypeError: (.*)",
        "fix": "fix_type_error",
    },
    "memory_error": {
        "regex": r"MemoryError|unable to allocate",
        "fix": "reduce_memory",
    },
    "zero_division": {
        "regex": r"division by zero|invalid value encountered in divide",
        "fix": "add_epsilon",
    },
}


def log(msg: str, log_file: Path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [auto_fix] {msg}"
    print(line, flush=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")


def parse_errors(log_path: Path) -> list[dict]:
    """Parse log file for error patterns."""
    if not log_path.exists():
        return []

    with open(log_path) as f:
        content = f.read()

    errors = []
    for error_type, pattern in ERROR_PATTERNS.items():
        for match in re.finditer(pattern["regex"], content, re.MULTILINE):
            errors.append({
                "type": error_type,
                "fix": pattern["fix"],
                "match": match.group(0),
                "groups": match.groups(),
                "line": content[:match.start()].count("\n") + 1,
            })

    return errors


def fix_pip_install(package: str, log_file: Path) -> bool:
    """Install missing Python package."""
    log(f"Installing missing package: {package}", log_file)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            log(f"  Successfully installed {package}", log_file)
            return True
        else:
            log(f"  Failed to install {package}: {result.stderr[-200:]}", log_file)
            return False
    except Exception as e:
        log(f"  Error installing {package}: {e}", log_file)
        return False


def fix_generate_missing_file(run_dir: Path, filename: str, log_file: Path) -> bool:
    """Generate a missing data file."""
    log(f"Generating missing file: {filename}", log_file)

    try:
        # Try to determine what kind of file was missing
        if "twopt" in filename or "2pt" in filename:
            # Generate 2pt data
            from importlib import import_module
            sys.path.insert(0, str(run_dir / "02_sample_data"))
            gen = import_module("generate")

            match = re.search(r"conf(\d+)", filename)
            conf_id = int(match.group(1)) if match else 6250

            Nt, Nx = 72, 24
            corr = gen.generate_sample_2pt(Nt, conf_id, seed=42)
            conf_dir = run_dir / "02_sample_data" / f"conf_{conf_id}"
            conf_dir.mkdir(parents=True, exist_ok=True)
            np.save(conf_dir / filename, corr)
            log(f"  Generated 2pt file: {conf_dir / filename}", log_file)
            return True

        elif "ops_mu" in filename or "ope" in filename:
            gen = import_module("generate")

            match = re.search(r"conf(\d+)", filename)
            conf_id = int(match.group(1)) if match else 6250
            mu_match = re.search(r"mu(\d+)_nu(\d+)", filename)
            mu = int(mu_match.group(1)) if mu_match else 0
            nu = int(mu_match.group(2)) if mu_match else 1

            Nt, Nx = 72, 24
            ope = gen.generate_sample_ope(Nt, Nx, mu, nu, conf_id, seed=42)
            conf_dir = run_dir / "02_sample_data" / f"conf_{conf_id}"
            conf_dir.mkdir(parents=True, exist_ok=True)
            np.savez(conf_dir / filename, ops=ope)
            log(f"  Generated OPE file: {conf_dir / filename}", log_file)
            return True

        else:
            log(f"  Unknown file type: {filename}", log_file)
            return False

    except Exception as e:
        log(f"  Error generating file: {e}", log_file)
        return False


def fix_shape_mismatch(run_dir: Path, log_file: Path) -> bool:
    """Attempt to fix array shape mismatches by adjusting config."""
    log("Attempting to fix shape mismatch...", log_file)
    config_path = run_dir / "run_config.json"

    try:
        with open(config_path) as f:
            config = json.load(f)

        # Force consistent shapes
        params = config.get("parameters", {})
        params["Nt"] = 72
        params["Nx"] = 24
        params["Nconf"] = 3

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        log("  Reset parameter shapes to Nt=72, Nx=24, Nconf=3", log_file)
        return True
    except Exception as e:
        log(f"  Failed to fix shapes: {e}", log_file)
        return False


def auto_fix(run_dir: Path, attempt: int, log_file: Path) -> dict:
    """Run auto-fix based on error patterns found in the log."""
    log(f"Auto-fix attempt {attempt}", log_file)

    errors = parse_errors(log_file)
    log(f"Found {len(errors)} error pattern(s)", log_file)

    fixes_applied = {}
    fixed_count = 0

    for error in errors:
        error_type = error["type"]
        fix_type = error["fix"]

        if fix_type in fixes_applied:
            continue  # Don't apply same fix twice

        log(f"  Error: [{error_type}] {error['match'][:100]}", log_file)

        if fix_type == "pip_install" and error["groups"]:
            package = error["groups"][0]
            if fix_pip_install(package, log_file):
                fixes_applied[fix_type] = True
                fixed_count += 1

        elif fix_type == "generate_missing_file" and error["groups"]:
            filename = error["groups"][0]
            if fix_generate_missing_file(run_dir, filename, log_file):
                fixes_applied[fix_type] = True
                fixed_count += 1

        elif fix_type == "fix_shape":
            if fix_shape_mismatch(run_dir, log_file):
                fixes_applied[fix_type] = True
                fixed_count += 1

        elif fix_type == "reduce_memory":
            log("  Adding memory reduction to config", log_file)
            config_path = run_dir / "run_config.json"
            with open(config_path) as f:
                config = json.load(f)
            config["parameters"]["max_t"] = min(config["parameters"].get("max_t", 20), 15)
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            fixes_applied[fix_type] = True
            fixed_count += 1

        elif fix_type in {"fix_broadcast", "fix_key_error", "fix_value_error", "add_epsilon"}:
            log(f"  Cannot auto-fix [{error_type}] — requires code change", log_file)
            fixes_applied[fix_type] = "manual_required"

    log(f"Applied {fixed_count} fix(es), {len(fixes_applied) - fixed_count} manual fix(es) needed", log_file)
    return fixes_applied


def main():
    parser = argparse.ArgumentParser(description="Auto-fix pipeline errors")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory")
    parser.add_argument("--attempt", type=int, default=0, help="Attempt number")
    parser.add_argument("--log", type=str, default="run.log", help="Log file to parse")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    log_file = run_dir / args.log

    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return 1

    fixes = auto_fix(run_dir, args.attempt, log_file)

    # Save fix record
    fix_record_path = run_dir / f"auto_fix_{args.attempt}.json"
    with open(fix_record_path, "w") as f:
        json.dump({
            "attempt": args.attempt,
            "timestamp": datetime.now().isoformat(),
            "fixes_applied": {k: v for k, v in fixes.items()},
        }, f, indent=2)

    return 0


if __name__ == "__main__":
    import numpy as np  # noqa
    sys.exit(main())
