#!/usr/bin/env python3
"""
Shared utilities for the docker-v20260804 GPU pipeline.

Provides:
  - GPU detection, memory tracking, and logging
  - Precision management (complex64/complex128)
  - Timing and progress reporting
  - Intermediate data save/load helpers
  - Banner printing and terminal colors

This module is self-contained — it does NOT import from lqcddb or any
other lattice-pdf package.  All lattice-QCD–specific logic lives in the
pipeline steps.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# GPU / CuPy detection
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import cupy as cp  # noqa: F401

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# ═══════════════════════════════════════════════════════════════════════════════
# Precision management
# ═══════════════════════════════════════════════════════════════════════════════

_COMPUTE_DTYPE = None  # set by set_compute_dtype()


def set_compute_dtype(precision: str) -> None:
    """Set the global compute dtype for the pipeline.

    Args:
        precision: 'complex64' (single) or 'complex128' (double).
    """
    global _COMPUTE_DTYPE
    if precision == "complex64":
        _COMPUTE_DTYPE = np.complex64
    elif precision == "complex128":
        _COMPUTE_DTYPE = np.complex128
    else:
        raise ValueError(f"Unknown precision: {precision}. Use 'complex64' or 'complex128'.")


def get_compute_dtype() -> np.dtype:
    """Return the global compute dtype (numpy)."""
    global _COMPUTE_DTYPE
    if _COMPUTE_DTYPE is None:
        _COMPUTE_DTYPE = np.complex64  # default: single precision
    return _COMPUTE_DTYPE


def get_compute_dtype_real() -> np.dtype:
    """Return the real counterpart of the compute dtype."""
    return np.float32 if get_compute_dtype() == np.complex64 else np.float64


# Alias for CuPy dtype string
def get_cp_dtype() -> str:
    """Return CuPy-compatible dtype string."""
    return "complex64" if get_compute_dtype() == np.complex64 else "complex128"


# ═══════════════════════════════════════════════════════════════════════════════
# GPU helpers
# ═══════════════════════════════════════════════════════════════════════════════


def get_gpu_device_info() -> dict:
    """Return GPU device info dict (empty if no CuPy)."""
    if not HAS_CUPY:
        return {}
    import cupy

    dev = cupy.cuda.Device()
    props = cupy.cuda.runtime.getDeviceProperties(dev.id)
    free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
    return {
        "device_id": dev.id,
        "device_name": props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
        "compute_capability": f"{props['major']}.{props['minor']}",
        "total_memory_gb": total_bytes / 1024**3,
        "free_memory_gb": free_bytes / 1024**3,
        "cupy_version": cupy.__version__,
        "cuda_version": cupy.cuda.runtime.runtimeGetVersion(),
    }


def get_gpu_memory_mb() -> float:
    """Return current GPU memory usage in MB."""
    if not HAS_CUPY:
        return 0.0
    import cupy

    free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
    return (total_bytes - free_bytes) / 1024**2


def gpu_sync() -> None:
    """Synchronize the GPU stream (no-op if CuPy unavailable)."""
    if HAS_CUPY:
        import cupy

        cupy.cuda.Stream.null.synchronize()


def to_gpu(arr: np.ndarray):
    """Transfer numpy array to GPU, converting to compute dtype."""
    if HAS_CUPY:
        import cupy

        return cupy.asarray(arr, dtype=get_cp_dtype())
    return arr.astype(get_compute_dtype())


def to_cpu(arr) -> np.ndarray:
    """Transfer GPU array to CPU as numpy."""
    if HAS_CUPY:
        import cupy

        if isinstance(arr, cupy.ndarray):
            return cupy.asnumpy(arr)
    return np.asarray(arr)


def log_gpu_status(logger: logging.Logger) -> None:
    """Log current GPU memory status."""
    if not HAS_CUPY:
        logger.info("GPU: not available (CuPy not installed)")
        return
    info = get_gpu_device_info()
    logger.info(
        f"GPU [{info.get('device_name', '?')}]: "
        f"{info.get('free_memory_gb', 0):.1f} GB free / "
        f"{info.get('total_memory_gb', 0):.1f} GB total"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════════


def setup_logging(log_dir: str, verbose: bool = False) -> logging.Logger:
    """Configure dual logging: file (DEBUG) + console (INFO or DEBUG).

    Args:
        log_dir: Directory for log files.
        verbose: If True, console level = DEBUG.

    Returns:
        Configured root logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger("docker-v20260804")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler — always DEBUG
    fh = logging.FileHandler(os.path.join(log_dir, f"pipeline_{timestamp}.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# Timing
# ═══════════════════════════════════════════════════════════════════════════════


class Timer:
    """Context manager for wall-clock timing with logging."""

    def __init__(self, label: str, logger: Optional[logging.Logger] = None):
        self.label = label
        self.logger = logger
        self.start: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        msg = f"{self.label} — {elapsed:.1f}s"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start


# ═══════════════════════════════════════════════════════════════════════════════
# Display helpers
# ═══════════════════════════════════════════════════════════════════════════════


class Colors:
    """ANSI terminal color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"


def color(text: str, c: str) -> str:
    return f"{c}{text}{Colors.RESET}"


def print_banner(title: str, logger: logging.Logger) -> None:
    """Print a highlighted section banner."""
    line = "═" * 78
    logger.info("")
    logger.info(color(line, Colors.CYAN))
    logger.info(color(f"  {title}", Colors.BOLD + Colors.CYAN))
    logger.info(color(line, Colors.CYAN))


def format_size(nbytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


# ═══════════════════════════════════════════════════════════════════════════════
# Intermediate data I/O
# ═══════════════════════════════════════════════════════════════════════════════


def save_intermediate(
    data: np.ndarray, filepath: str, logger: Optional[logging.Logger] = None
) -> None:
    """Save intermediate numpy array to disk.

    Args:
        data: numpy array to save.
        filepath: Target .npy path.
        logger: Optional logger.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, data)
    if logger:
        logger.debug(
            f"  Saved: {os.path.basename(filepath)} "
            f"shape={data.shape} dtype={data.dtype} "
            f"size={format_size(data.nbytes)}"
        )


def load_intermediate(filepath: str, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Load intermediate numpy array from disk."""
    data = np.load(filepath)
    if logger:
        logger.debug(
            f"  Loaded: {os.path.basename(filepath)} "
            f"shape={data.shape} dtype={data.dtype}"
        )
    return data


def save_intermediate_gpu(
    data, filepath: str, logger: Optional[logging.Logger] = None
) -> None:
    """Save GPU array to disk as .npy (transfers to CPU first)."""
    arr = to_cpu(data)
    save_intermediate(arr, filepath, logger)


# ═══════════════════════════════════════════════════════════════════════════════
# Config helpers
# ═══════════════════════════════════════════════════════════════════════════════


def dump_config_snapshot(config: dict, filepath: str, logger: logging.Logger) -> None:
    """Write pipeline configuration as JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Convert non-serializable types
    safe = {}
    for k, v in config.items():
        if isinstance(v, (np.integer,)):
            safe[k] = int(v)
        elif isinstance(v, (np.floating,)):
            safe[k] = float(v)
        elif isinstance(v, Path):
            safe[k] = str(v)
        elif isinstance(v, np.ndarray):
            safe[k] = v.tolist()
        else:
            safe[k] = v
    with open(filepath, "w") as f:
        json.dump(safe, f, indent=2, default=str)
    logger.info(f"Config saved to {filepath}")


def get_output_tree(run_dir: str) -> str:
    """Return a pretty-printed directory tree of the output."""
    lines = []
    for root, dirs, files in os.walk(run_dir):
        level = root.replace(run_dir, "").count(os.sep)
        indent = "  " * level
        lines.append(f"{indent}{os.path.basename(root) or '.'}/")
        for f in sorted(files):
            lines.append(f"{indent}  {f}")
    return "\n".join(lines)


def log_exception(logger: logging.Logger) -> None:
    """Log the current exception with full traceback."""
    logger.error(traceback.format_exc())


def validate_array(arr, name: str, logger: logging.Logger) -> bool:
    """Check array for NaN/inf and log a summary."""
    if arr is None:
        logger.warning(f"  {name}: None")
        return False
    arr_np = to_cpu(arr)
    ok = bool(np.all(np.isfinite(arr_np)))
    if ok:
        logger.debug(
            f"  {name}: shape={arr_np.shape} dtype={arr_np.dtype} "
            f"|v|∈[{np.abs(arr_np).min():.2e}, {np.abs(arr_np).max():.2e}]"
        )
    else:
        n_nan = int(np.sum(~np.isfinite(arr_np)))
        logger.error(f"  {name}: {n_nan} NaN/inf values!")
    return ok


def get_current_memory_mb() -> float:
    """Estimate current CPU memory usage (from /proc). Not precise."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


def get_peak_memory_gb() -> float:
    """Return peak GPU memory usage (CuPy mempool)."""
    if not HAS_CUPY:
        return 0.0
    import cupy
    mempool = cupy.get_default_memory_pool()
    return mempool.total_bytes() / 1024**3
