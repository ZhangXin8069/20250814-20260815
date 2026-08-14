#!/usr/bin/env python3
"""
Shared utilities for the snsc-v20260726 validation pipeline.

Provides:
  - setup_logging(): dual file+stdout logging with timestamps
  - Timer: context manager for wall-time + peak-memory tracking
  - run_step(): subprocess runner with logging
  - Colors for terminal output
"""

from __future__ import annotations

import gc
import logging
import os
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional


# ─── Terminal colors ───────────────────────────────────────────────────────

class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def color(text: str, color_code: str) -> str:
    return f"{color_code}{text}{Colors.RESET}"


# ─── Logging ────────────────────────────────────────────────────────────────

def setup_logging(log_file: Path, name: str = "snsc_pipeline") -> logging.Logger:
    """Configure dual logging to file (DEBUG) and stdout (INFO).

    Args:
        log_file: Path to log file (directory will be created if needed).
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on re-import
    if logger.handlers:
        return logger

    # File handler (DEBUG level)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Console handler (INFO level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ─── Memory tracking ────────────────────────────────────────────────────────

def get_peak_memory_gb() -> float:
    """Return peak RSS memory usage in GB."""
    try:
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return max_rss / (1024 * 1024)
    except Exception:
        return 0.0


def get_current_memory_mb() -> float:
    """Return current RSS memory usage in MB."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024.0
    except Exception:
        return 0.0


# ─── Timer context manager ──────────────────────────────────────────────────

@contextmanager
def Timer(name: str, logger: Optional[logging.Logger] = None, log_dir: Optional[Path] = None):
    """Context manager: track wall time and peak memory for a code block.

    Usage:
        with Timer("Step 01: 2pt Computation", logger=log):
            result = compute_2pt(...)

    Logs elapsed time and memory delta to logger (if provided) and
    writes a timing record to log_dir/timing.jsonl (if provided).
    """
    gc.collect()
    mem_before = get_current_memory_mb()
    t_start = time.perf_counter()

    yield

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    mem_after = get_current_memory_mb()
    mem_delta = mem_after - mem_before

    msg = f"[{name}] elapsed={elapsed:.2f}s  mem_before={mem_before:.1f}MB  mem_after={mem_after:.1f}MB  delta={mem_delta:+.1f}MB"
    if logger:
        logger.info(msg)
    else:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

    # Write timing record
    if log_dir:
        import json as _json
        timing_path = log_dir / "timing.jsonl"
        with open(timing_path, "a") as f:
            _json.dump({
                "step": name,
                "elapsed_s": round(elapsed, 3),
                "mem_before_mb": round(mem_before, 1),
                "mem_after_mb": round(mem_after, 1),
                "mem_delta_mb": round(mem_delta, 1),
                "timestamp": datetime.now().isoformat(),
            }, f)
            f.write("\n")


# ─── Subprocess runner ──────────────────────────────────────────────────────

def run_step(
    name: str,
    cmd: list[str],
    cwd: Path,
    logger: Optional[logging.Logger] = None,
    timeout: int = 7200,
    env: Optional[dict] = None,
) -> tuple[bool, str, str]:
    """Run a pipeline step as a subprocess.

    Args:
        name: Step name for logging.
        cmd: Command to run (list of strings).
        cwd: Working directory.
        logger: Logger instance (optional).
        timeout: Timeout in seconds (default 2h for lattice jobs).
        env: Environment variables override.

    Returns:
        (success: bool, stdout: str, stderr: str)
    """
    msg = f"[{name}] Starting: {' '.join(cmd)}"
    if logger:
        logger.info(msg)
    else:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

    t_start = time.perf_counter()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env or os.environ,
        )
        elapsed = time.perf_counter() - t_start
        stdout = result.stdout
        stderr = result.stderr

        if logger:
            if stdout:
                logger.debug(f"[{name}] STDOUT:\n{stdout[:2000]}")
            if stderr:
                logger.debug(f"[{name}] STDERR:\n{stderr[:2000]}")

        if result.returncode == 0:
            msg_ok = f"[{name}] SUCCESS (returncode=0, elapsed={elapsed:.1f}s)"
            if logger:
                logger.info(msg_ok)
            else:
                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg_ok}", flush=True)
            return True, stdout, stderr
        else:
            msg_fail = f"[{name}] FAILED (returncode={result.returncode}, elapsed={elapsed:.1f}s)"
            if logger:
                logger.error(msg_fail)
            else:
                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg_fail}", flush=True)
            return False, stdout, stderr

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t_start
        msg_to = f"[{name}] TIMEOUT after {elapsed:.1f}s (limit: {timeout}s)"
        if logger:
            logger.error(msg_to)
        return False, "", msg_to
    except Exception as e:
        msg_err = f"[{name}] ERROR: {e}"
        if logger:
            logger.error(msg_err)
        return False, "", str(e)


# ─── File size formatter ────────────────────────────────────────────────────

def format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes > 1024 * 1024 * 1024:
        return f"{size_bytes / (1024**3):.1f} GB"
    elif size_bytes > 1024 * 1024:
        return f"{size_bytes / (1024**2):.1f} MB"
    elif size_bytes > 1024:
        return f"{size_bytes / 1024:.0f} KB"
    else:
        return f"{size_bytes} B"


# ─── Banner printer ─────────────────────────────────────────────────────────

def print_banner(title: str, logger: Optional[logging.Logger] = None):
    """Print a formatted banner."""
    banner = f"\n{'═'*70}\n  {title}\n{'═'*70}"
    if logger:
        logger.info(banner)
    else:
        print(banner, flush=True)
