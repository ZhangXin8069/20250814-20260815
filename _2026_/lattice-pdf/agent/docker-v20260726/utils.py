#!/usr/bin/env python3
"""
Shared utilities for the docker-v20260726 pipeline.

Enhanced version with:
  - Dual file+stdout logging with module-level granularity
  - Timer context manager with persistent state tracking
  - Intermediate data saving utilities
  - Memory tracking
  - Progress bars for long-running operations
  - Colors for terminal output
"""

from __future__ import annotations

import gc
import json
import logging
import os
import resource
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import numpy as np


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
    GRAY = "\033[90m"
    RESET = "\033[0m"


def color(text: str, color_code: str) -> str:
    return f"{color_code}{text}{Colors.RESET}"


# ─── Logging ────────────────────────────────────────────────────────────────

def setup_logging(
    log_file: Path,
    name: str = "docker_pipeline",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Configure dual logging to file (DEBUG) and stdout (INFO).

    Args:
        log_file: Path to log file (directory will be created if needed).
        name: Logger name.
        console_level: Logging level for console output.
        file_level: Logging level for file output.

    Returns:
        Configured logger instance.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on re-import
    if logger.handlers:
        return logger

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(file_level)
    ff = logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(ff)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    cf = logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(cf)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def add_module_handler(logger: logging.Logger, log_file: Path, module_name: str):
    """Add a module-specific file handler for granular logging.

    Each computation module gets its own log file in addition to the main log.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    mh = logging.FileHandler(log_file)
    mh.setLevel(logging.DEBUG)
    mf = logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    mh.setFormatter(mf)
    mh.addFilter(lambda record: record.name == module_name)
    logger.addHandler(mh)


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
def Timer(
    name: str,
    logger: Optional[logging.Logger] = None,
    log_dir: Optional[Path] = None,
    extra: Optional[dict] = None,
):
    """Context manager: track wall time and peak memory for a code block.

    Usage:
        with Timer("Step 01: 2pt Computation", logger=log, log_dir=output_dir):
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
    mem_peak = get_peak_memory_gb()

    msg = (
        f"[{name}] elapsed={elapsed:.2f}s  "
        f"mem_before={mem_before:.1f}MB  mem_after={mem_after:.1f}MB  "
        f"delta={mem_delta:+.1f}MB  peak={mem_peak:.2f}GB"
    )
    if logger:
        logger.info(msg)
    else:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

    # Write timing record
    if log_dir:
        timing_path = log_dir / "timing.jsonl"
        record = {
            "step": name,
            "elapsed_s": round(elapsed, 3),
            "mem_before_mb": round(mem_before, 1),
            "mem_after_mb": round(mem_after, 1),
            "mem_delta_mb": round(mem_delta, 1),
            "mem_peak_gb": round(mem_peak, 2),
            "timestamp": datetime.now().isoformat(),
        }
        if extra:
            record.update(extra)
        with open(timing_path, "a") as f:
            json.dump(record, f)
            f.write("\n")


# ─── Intermediate data saving ───────────────────────────────────────────────

def save_intermediate(
    data: Any,
    output_dir: Path,
    filename: str,
    logger: Optional[logging.Logger] = None,
    metadata: Optional[dict] = None,
):
    """Save intermediate computation data with metadata.

    Supports .npy (single array), .npz (dict of arrays), .json (dict).

    Args:
        data: Data to save (np.ndarray, dict of arrays, or plain dict).
        output_dir: Output directory.
        filename: Output filename (determines format by extension).
        logger: Logger instance.
        metadata: Optional dict to save alongside data (for .npz, added as __metadata__).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    if filename.endswith(".npy"):
        np.save(path, data)
        size_mb = path.stat().st_size / 1024**2
        if logger:
            logger.debug(f"  [SAVE] {filename} ({size_mb:.1f} MB)")

    elif filename.endswith(".npz"):
        save_dict = data if isinstance(data, dict) else {"data": data}
        if metadata:
            save_dict["__metadata__"] = np.array([json.dumps(metadata)])
        np.savez(path, **save_dict)
        size_mb = path.stat().st_size / 1024**2
        if logger:
            logger.debug(f"  [SAVE] {filename} ({size_mb:.1f} MB, {len(save_dict)} keys)")

    elif filename.endswith(".json"):
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        size_kb = path.stat().st_size / 1024
        if logger:
            logger.debug(f"  [SAVE] {filename} ({size_kb:.1f} KB)")

    else:
        raise ValueError(f"Unsupported file extension: {Path(filename).suffix}")

    return path


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


# ─── Progress bar ───────────────────────────────────────────────────────────

def progress_bar(
    current: int,
    total: int,
    prefix: str = "",
    logger: Optional[logging.Logger] = None,
    bar_length: int = 40,
):
    """Print or log a progress bar.

    Args:
        current: Current iteration (0-indexed or 1-indexed).
        total: Total iterations.
        prefix: Description prefix.
        logger: If provided, log at DEBUG level; otherwise print to stdout.
        bar_length: Length of the bar in characters.
    """
    fraction = min(current / max(total, 1), 1.0)
    filled = int(bar_length * fraction)
    bar = "█" * filled + "░" * (bar_length - filled)
    pct = fraction * 100

    line = f"\r{prefix} |{bar}| {current}/{total} ({pct:.0f}%)"
    if logger:
        if current == total or current % max(1, total // 20) == 0:
            logger.debug(line.strip())
    else:
        sys.stdout.write(line)
        sys.stdout.flush()
        if current == total:
            sys.stdout.write("\n")
            sys.stdout.flush()


# ─── Error handler ──────────────────────────────────────────────────────────

def log_exception(
    logger: logging.Logger,
    e: Exception,
    context: str = "",
    reraise: bool = False,
):
    """Log an exception with full traceback.

    Args:
        logger: Logger instance.
        e: The exception.
        context: Description of what was being attempted.
        reraise: If True, re-raise the exception after logging.
    """
    tb = traceback.format_exc()
    msg = f"[ERROR] {context}: {e}" if context else f"[ERROR] {e}"
    logger.error(msg)
    logger.debug(f"Traceback:\n{tb}")
    if reraise:
        raise


# ─── Data validation ────────────────────────────────────────────────────────

def validate_array(
    arr: Any,
    name: str,
    expected_shape: Optional[tuple] = None,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Validate a numpy array.

    Checks: is ndarray, finite values, non-zero, expected shape (if given).

    Returns True if valid, False otherwise.
    """
    import numpy as np
    if not isinstance(arr, np.ndarray):
        if logger:
            logger.error(f"  [VALIDATE] {name}: not a numpy array (type={type(arr)})")
        return False

    if not np.all(np.isfinite(arr)):
        if logger:
            logger.error(f"  [VALIDATE] {name}: contains NaN or inf")
        return False

    if expected_shape is not None and arr.shape != expected_shape:
        if logger:
            logger.warning(
                f"  [VALIDATE] {name}: shape mismatch "
                f"(expected {expected_shape}, got {arr.shape})"
            )
        # Not a hard error — shape can vary

    nz = np.count_nonzero(arr)
    if nz == 0:
        if logger:
            logger.warning(f"  [VALIDATE] {name}: all zeros")
        return False

    if logger:
        logger.debug(
            f"  [VALIDATE] {name}: shape={arr.shape}, dtype={arr.dtype}, "
            f"range=[{np.abs(arr).min():.2e}, {np.abs(arr).max():.2e}], "
            f"nonzero={nz}/{arr.size}"
        )
    return True


# ─── Configuration utilities ────────────────────────────────────────────────

def dump_config_snapshot(config: dict, output_dir: Path, logger: Optional[logging.Logger] = None):
    """Save a snapshot of the current configuration."""
    path = output_dir / "run_config_snapshot.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    if logger:
        logger.info(f"Config snapshot saved to {path}")


def get_output_tree(output_dir: Path, max_files_per_dir: int = 50) -> str:
    """Generate a text tree of the output directory."""
    lines = []
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(str(output_dir), "").count(os.sep)
        indent = "  " * level
        folder = os.path.basename(root) or str(output_dir)
        lines.append(f"{indent}{folder}/")
        sub_indent = "  " * (level + 1)
        sorted_files = sorted(files)
        for f in sorted_files[:max_files_per_dir]:
            fpath = os.path.join(root, f)
            lines.append(f"{sub_indent}{f}  ({format_size(os.path.getsize(fpath))})")
        if len(sorted_files) > max_files_per_dir:
            lines.append(f"{sub_indent}... and {len(sorted_files) - max_files_per_dir} more files")
    return "\n".join(lines)
