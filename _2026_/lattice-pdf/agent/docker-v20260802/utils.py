#!/usr/bin/env python3
"""
Shared utilities for the docker-v20260802 GPU-accelerated pipeline (double precision).

MERGE NOTE (v20260802):
  - Merged from v20260730 (correct perambulator/eigenvector paths, missing-file handling)
    and v20260731 (double precision default, eigenvector format detection).
  - All version strings updated to v20260802.
  - Config path patterns cleaned: perambulator_base without /light/ suffix.
  - Removed dead config fields (eigenvector_pattern, perambulator_pattern, eigenvalue).

Precision convention:
  - GPU compute dtype: complex128 by default (configurable via set_compute_dtype)
  - Input data: complex128 (read from disk) → cast to compute dtype
  - Output data: compute dtype → saved as-is
  - Real operations: float64 by default
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

# ─── GPU backend ─────────────────────────────────────────────────────────────

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

# ─── Precision helpers ───────────────────────────────────────────────────────
# Default: double precision (complex128/float64) — matching v20260731
# Change via set_compute_dtype("complex64") for single precision

_COMPUTE_DTYPE = np.complex128      # default GPU compute dtype (double precision)
_COMPUTE_DTYPE_REAL = np.float64    # default real dtype (double precision)


def set_compute_dtype(dtype_str: str):
    """Set the global compute dtype preference.

    Args:
        dtype_str: 'complex64' for single precision, 'complex128' for double.

    Sets both the complex dtype and the corresponding real dtype
    (float32 for complex64, float64 for complex128).
    """
    global _COMPUTE_DTYPE, _COMPUTE_DTYPE_REAL
    if dtype_str == 'complex128':
        _COMPUTE_DTYPE = np.complex128
        _COMPUTE_DTYPE_REAL = np.float64
    else:
        _COMPUTE_DTYPE = np.complex64
        _COMPUTE_DTYPE_REAL = np.float32


def get_compute_dtype() -> np.dtype:
    """Get the current GPU compute dtype (complex64 or complex128)."""
    return _COMPUTE_DTYPE


def get_compute_dtype_real() -> np.dtype:
    """Get the current real compute dtype (float32 or float64)."""
    return _COMPUTE_DTYPE_REAL


def is_single_precision() -> bool:
    """Check if running in single precision mode (complex64)."""
    return _COMPUTE_DTYPE == np.complex64


def to_compute_dtype(arr: np.ndarray, is_complex: bool = True) -> np.ndarray:
    """Convert input array to compute dtype for GPU.

    Args:
        arr: Input numpy array (typically complex128 from disk).
        is_complex: If True, convert to complex compute dtype; else to real.

    Returns:
        Array cast to compute dtype (complex128 or complex64 by current setting).
    """
    target = _COMPUTE_DTYPE if is_complex else _COMPUTE_DTYPE_REAL
    if arr.dtype != target:
        return arr.astype(target, copy=False)
    return arr


def to_gpu(arr: np.ndarray, is_complex: bool = True) -> Any:
    """Transfer to GPU with precision conversion.

    Steps: CPU array → cast to compute dtype → CuPy array on GPU.

    Args:
        arr: Input numpy array.
        is_complex: If True, convert to complex compute dtype; else to real.

    Returns:
        CuPy array with compute dtype on GPU.
    """
    if HAS_CUPY:
        arr_cvt = to_compute_dtype(arr, is_complex)
        return cp.asarray(arr_cvt)
    return arr


def to_cpu(arr: Any) -> np.ndarray:
    """Transfer CuPy array to CPU (numpy). Keeps GPU dtype."""
    if HAS_CUPY and hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


def ensure_numpy(arr: Any) -> np.ndarray:
    """Ensure array is a numpy array (transfer from GPU if needed)."""
    return to_cpu(arr)


def gpu_sync():
    """Synchronize GPU stream — ensures all pending GPU ops complete."""
    if HAS_CUPY:
        cp.cuda.Stream.null.synchronize()


# ─── GPU info ────────────────────────────────────────────────────────────────

def get_gpu_device_info() -> dict:
    """Get GPU device information including name, memory, CUDA version.

    Returns:
        dict with keys: cupy_available, device_name, compute_capability,
        total_memory_gb, free_memory_gb, cupy_version, cuda_version, etc.
        If CuPy is not available, returns {'cupy_available': False, 'error': ...}.
    """
    if not HAS_CUPY:
        return {"cupy_available": False, "error": "CuPy not installed"}
    try:
        device = cp.cuda.Device()
        mem = device.mem_info
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode()
        return {
            "cupy_available": True,
            "cupy_version": cp.__version__,
            "device_id": device.id,
            "device_name": name,
            "compute_capability": f"{props['major']}.{props['minor']}",
            "total_memory_gb": round(mem[1] / 1024**3, 2),
            "free_memory_gb": round(mem[0] / 1024**3, 2),
            "used_memory_gb": round((mem[1] - mem[0]) / 1024**3, 2),
            "cuda_version": cp.cuda.runtime.runtimeGetVersion(),
            "multi_gpu_count": cp.cuda.runtime.getDeviceCount(),
        }
    except Exception as e:
        return {"cupy_available": True, "error": str(e)}


def get_gpu_memory_mb() -> dict:
    """Get current GPU memory usage in MB.

    Returns:
        dict with keys: free_mb, total_mb, used_mb.
    """
    if not HAS_CUPY:
        return {"free_mb": 0, "total_mb": 0, "used_mb": 0}
    try:
        mem = cp.cuda.Device().mem_info
        return {
            "free_mb": round(mem[0] / 1024**2, 1),
            "total_mb": round(mem[1] / 1024**2, 1),
            "used_mb": round((mem[1] - mem[0]) / 1024**2, 1),
        }
    except Exception:
        return {"free_mb": 0, "total_mb": 0, "used_mb": 0}


# ─── Terminal colors ─────────────────────────────────────────────────────────

class Colors:
    """ANSI terminal color codes for formatted console output."""
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
    """Wrap text with ANSI color codes."""
    return f"{color_code}{text}{Colors.RESET}"


# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logging(
    log_file: Path,
    name: str = "docker_pipeline_v20260802",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Configure dual logging to file (DEBUG) and stdout (INFO).

    Args:
        log_file: Path to the log file (parent dirs created if needed).
        name: Logger name (unique per module to avoid handler duplication).
        console_level: Logging level for stdout handler.
        file_level: Logging level for file handler.

    Returns:
        Configured logger instance. If handlers already exist, returns existing logger.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    # File handler: detailed timestamps with module name
    fh = logging.FileHandler(log_file)
    fh.setLevel(file_level)
    ff = logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(ff)

    # Console handler: compact timestamps without module name
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


# ─── Memory tracking ─────────────────────────────────────────────────────────

def get_peak_memory_gb() -> float:
    """Get peak CPU memory usage in GB (max RSS)."""
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except Exception:
        return 0.0


def get_current_memory_mb() -> float:
    """Get current CPU memory usage in MB (max RSS)."""
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


# ─── Timer context manager ───────────────────────────────────────────────────

@contextmanager
def Timer(
    name: str,
    logger: Optional[logging.Logger] = None,
    log_dir: Optional[Path] = None,
    extra: Optional[dict] = None,
    track_gpu: bool = True,
):
    """Context manager that tracks wall time, CPU memory, and GPU memory for a code block.

    Usage:
        with Timer("my_step", logger, output_dir, extra={"key": "val"}):
            ... do work ...

    On exit, logs elapsed time and memory delta, and appends a JSON record
    to `log_dir/timing.jsonl` if log_dir is provided.

    Args:
        name: Human-readable label for this timing block.
        logger: Logger to write timing info to.
        log_dir: Directory for timing.jsonl output (optional).
        extra: Additional key-value pairs to include in the JSON record.
        track_gpu: Whether to track GPU memory (True = sync+measure before/after).
    """
    gc.collect()
    mem_before = get_current_memory_mb()
    gpu_before = get_gpu_memory_mb() if track_gpu else {}
    if track_gpu:
        gpu_sync()
    t_start = time.perf_counter()

    yield

    if track_gpu:
        gpu_sync()
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    mem_after = get_current_memory_mb()
    mem_delta = mem_after - mem_before
    mem_peak = get_peak_memory_gb()
    gpu_after = get_gpu_memory_mb() if track_gpu else {}

    # Build GPU delta string
    gpu_str = ""
    if gpu_before and gpu_after:
        gpu_delta = gpu_after.get("used_mb", 0) - gpu_before.get("used_mb", 0)
        gpu_str = (f"gpu_free={gpu_after.get('free_mb', 0):.0f}MB  "
                   f"gpu_delta={gpu_delta:+.0f}MB  ")

    msg = (f"[{name}] elapsed={elapsed:.2f}s  {gpu_str}"
           f"mem_before={mem_before:.1f}MB  mem_after={mem_after:.1f}MB  "
           f"delta={mem_delta:+.1f}MB  peak={mem_peak:.2f}GB")
    if logger:
        logger.info(msg)
    else:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

    # Write timing record to JSONL if log_dir provided
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
        if gpu_before:
            record["gpu_free_before_mb"] = gpu_before.get("free_mb", 0)
        if gpu_after:
            record["gpu_free_after_mb"] = gpu_after.get("free_mb", 0)
        if extra:
            record.update(extra)
        with open(timing_path, "a") as f:
            json.dump(record, f)
            f.write("\n")


# ─── Intermediate data saving ────────────────────────────────────────────────

def save_intermediate(
    data: Any,
    output_dir: Path,
    filename: str,
    logger: Optional[logging.Logger] = None,
    metadata: Optional[dict] = None,
):
    """Save intermediate data to disk. Auto-transfers CuPy→CPU.

    Supports .npy (single array), .npz (dict of arrays or single array),
    and .json (dict). Keeps current compute dtype.

    Args:
        data: Array, dict of arrays, or JSON-serializable dict.
        output_dir: Directory to save into (created if needed).
        filename: Output filename with extension (.npy, .npz, .json).
        logger: Logger for debug messages.
        metadata: Optional dict to embed as __metadata__ in .npz files.

    Returns:
        Path to the saved file.

    Raises:
        ValueError: If extension is not .npy/.npz/.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    if filename.endswith(".npy"):
        data_cpu = ensure_numpy(data)
        np.save(path, data_cpu)
        if logger:
            logger.debug(f"  [SAVE] {filename} ({path.stat().st_size/1024**2:.1f} MB)")

    elif filename.endswith(".npz"):
        if isinstance(data, dict):
            save_dict = {k: ensure_numpy(v) for k, v in data.items()}
        else:
            save_dict = {"data": ensure_numpy(data)}
        if metadata:
            save_dict["__metadata__"] = np.array([json.dumps(metadata)])
        np.savez(path, **save_dict)
        if logger:
            logger.debug(f"  [SAVE] {filename} ({path.stat().st_size/1024**2:.1f} MB)")

    elif filename.endswith(".json"):
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        if logger:
            logger.debug(f"  [SAVE] {filename} ({path.stat().st_size/1024:.1f} KB)")

    else:
        raise ValueError(f"Unsupported extension: {Path(filename).suffix}")
    return path


# ─── Helpers ─────────────────────────────────────────────────────────────────

def format_size(size_bytes: int) -> str:
    """Format byte count to human-readable string."""
    if size_bytes > 1024**3:
        return f"{size_bytes/(1024**3):.1f} GB"
    elif size_bytes > 1024**2:
        return f"{size_bytes/(1024**2):.1f} MB"
    elif size_bytes > 1024:
        return f"{size_bytes/1024:.0f} KB"
    return f"{size_bytes} B"


def print_banner(title: str, logger: Optional[logging.Logger] = None):
    """Print a formatted banner to logger or stdout."""
    banner = f"\n{'═'*70}\n  {title}\n{'═'*70}"
    if logger:
        logger.info(banner)
    else:
        print(banner, flush=True)


def log_exception(
    logger: logging.Logger,
    e: Exception,
    context: str = "",
    reraise: bool = False,
):
    """Log an exception with traceback. Optionally re-raise.

    Args:
        logger: Logger to write to.
        e: The exception instance.
        context: Human-readable context string.
        reraise: If True, re-raise the exception after logging.
    """
    tb = traceback.format_exc()
    logger.error(f"[ERROR] {context}: {e}" if context else f"[ERROR] {e}")
    logger.debug(f"Traceback:\n{tb}")
    if reraise:
        raise


def validate_array(
    arr: Any,
    name: str,
    expected_shape: Optional[tuple] = None,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Validate a numpy or cupy array for basic sanity.

    Checks: is ndarray, no NaN/inf, non-zero, optional shape match.

    Args:
        arr: Array to validate (numpy or cupy).
        name: Human-readable name for log messages.
        expected_shape: Optional expected shape tuple.
        logger: Logger for messages.

    Returns:
        True if all checks pass, False otherwise.
    """
    arr_check = ensure_numpy(arr) if hasattr(arr, 'get') else arr
    if not isinstance(arr_check, np.ndarray):
        if logger:
            logger.error(f"  [VALIDATE] {name}: not ndarray (type={type(arr)})")
        return False
    if not np.all(np.isfinite(arr_check)):
        if logger:
            logger.error(f"  [VALIDATE] {name}: contains NaN/inf")
        return False
    if expected_shape is not None and arr_check.shape != expected_shape:
        if logger:
            logger.warning(f"  [VALIDATE] {name}: shape {arr_check.shape} ≠ expected {expected_shape}")
    if np.count_nonzero(arr_check) == 0:
        if logger:
            logger.warning(f"  [VALIDATE] {name}: all zeros")
        return False
    if logger:
        logger.debug(
            f"  [VALIDATE] {name}: shape={arr_check.shape}, dtype={arr_check.dtype}, "
            f"range=[{np.abs(arr_check).min():.2e},{np.abs(arr_check).max():.2e}]"
        )
    return True


def dump_config_snapshot(
    config: dict,
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
):
    """Save a JSON snapshot of the current config to output_dir."""
    path = output_dir / "run_config_snapshot.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    if logger:
        logger.info(f"Config snapshot saved to {path}")


def get_output_tree(output_dir: Path, max_files_per_dir: int = 50) -> str:
    """Generate a tree-like text representation of the output directory.

    Args:
        output_dir: Root directory to walk.
        max_files_per_dir: Maximum files to list per directory.

    Returns:
        Multi-line string showing directory structure with file sizes.
    """
    lines = []
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(str(output_dir), "").count(os.sep)
        indent = "  " * level
        folder = os.path.basename(root) or str(output_dir)
        lines.append(f"{indent}{folder}/")
        sub_indent = "  " * (level + 1)
        for f in sorted(files)[:max_files_per_dir]:
            fpath = os.path.join(root, f)
            lines.append(f"{sub_indent}{f}  ({format_size(os.path.getsize(fpath))})")
        if len(files) > max_files_per_dir:
            lines.append(f"{sub_indent}... and {len(files)-max_files_per_dir} more")
    return "\n".join(lines)


def log_gpu_status(logger: logging.Logger, prefix: str = ""):
    """Log current GPU status (device name, memory, CUDA/CuPy versions).

    Args:
        logger: Logger to write to.
        prefix: String prefix for indentation in log messages.
    """
    if not HAS_CUPY:
        logger.warning(f"{prefix}GPU status: CuPy not available")
        return
    info = get_gpu_device_info()
    if "error" in info:
        logger.warning(f"{prefix}GPU status: {info['error']}")
        return
    logger.info(
        f"{prefix}GPU: {info['device_name']} (CC {info['compute_capability']}), "
        f"Mem: {info['free_memory_gb']:.1f}/{info['total_memory_gb']:.1f} GB free, "
        f"CuPy {info['cupy_version']}, CUDA {info['cuda_version']}"
    )
