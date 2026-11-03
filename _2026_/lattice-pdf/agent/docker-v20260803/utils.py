"""
Pipeline Utilities
==================

Logging, timing, and helper functions for the GPU pipeline.
"""

import os
import sys
import time
import logging
import traceback
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════
# Logging Setup
# ═══════════════════════════════════════════════════════════════════

def setup_logging(log_dir: str, name: str = 'pipeline'):
    """Set up dual logging: file + console.

    Parameters
    ----------
    log_dir : str
        Directory for log files.
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler — detailed
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)-8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'))

    # Console handler — info and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)-8s] %(message)s',
        datefmt='%H:%M:%S'))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    return logger


# ═══════════════════════════════════════════════════════════════════
# Timer Context Manager
# ═══════════════════════════════════════════════════════════════════

class Timer:
    """Context manager for wall-clock timing with GPU synchronization.

    Automatically calls ``cupy.cuda.Stream.null.synchronize()`` before
    and after the block when GPU backend is active, ensuring accurate
    GPU kernel timing.

    Usage::

        with Timer("Compute VdV", logger=log):
            result = compute_vertex(...)

    Parameters
    ----------
    name : str
        Label for the timing output.
    logger : logging.Logger, optional
        Logger for output. Uses print if None.
    """

    def __init__(self, name: str, logger=None):
        self.name = name
        self.logger = logger

    def __enter__(self):
        # Synchronize GPU before timing
        try:
            import cupy
            cupy.cuda.Stream.null.synchronize()
        except Exception:
            pass
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        try:
            import cupy
            cupy.cuda.Stream.null.synchronize()
        except Exception:
            pass
        elapsed = time.perf_counter() - self.start
        msg = f"{self.name}: {elapsed:.3f} s"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)


# ═══════════════════════════════════════════════════════════════════
# GPU Memory Reporting
# ═══════════════════════════════════════════════════════════════════

def gpu_memory_info():
    """Get current GPU memory usage.

    Returns
    -------
    tuple (total_mb, free_mb, used_mb)
        GPU memory info in MB. Returns (0, 0, 0) if no GPU.
    """
    try:
        import cupy
        mempool = cupy.get_default_memory_pool()
        used = mempool.used_bytes()
        total = mempool.total_bytes()
        free = total - used
        return total / 1024**2, free / 1024**2, used / 1024**2
    except Exception:
        return 0, 0, 0


def log_gpu_memory(logger, label: str = ""):
    """Log current GPU memory usage.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    label : str
        Optional label for the log message.
    """
    total, free, used = gpu_memory_info()
    if total > 0:
        logger.info(
            f"GPU Memory{': ' + label if label else ''}: "
            f"used={used:.1f} MB, free={free:.1f} MB, total={total:.1f} MB")


# ═══════════════════════════════════════════════════════════════════
# Precision Conversion
# ═══════════════════════════════════════════════════════════════════

def ensure_precision(arr, dtype='complex64'):
    """Convert array to the specified precision.

    Handles both numpy and cupy arrays. For 'complex64', converts to
    dtype complex64 (single precision). For 'complex128', keeps double.

    Parameters
    ----------
    arr : ndarray
        Input array (numpy or cupy).
    dtype : str
        Target dtype: 'complex64' or 'complex128'.

    Returns
    -------
    ndarray
        Array with target precision.
    """
    if dtype == 'complex64':
        target = type(arr).__module__.split('.')[0]  # 'numpy' or 'cupy'
        if target == 'cupy':
            import cupy
            return cupy.asarray(arr, dtype=cupy.complex64)
        else:
            import numpy as np
            return np.asarray(arr, dtype=np.complex64)
    elif dtype == 'complex128':
        return arr  # already double precision
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def get_dtype(dtype_str='complex64'):
    """Get the numpy/cupy dtype object for the given string.

    Parameters
    ----------
    dtype_str : str
        'complex64' or 'complex128'.

    Returns
    -------
    numpy.dtype or cupy.dtype
    """
    import numpy as np
    if dtype_str == 'complex64':
        return np.complex64
    elif dtype_str == 'complex128':
        return np.complex128
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")


# ═══════════════════════════════════════════════════════════════════
# Error Handling
# ═══════════════════════════════════════════════════════════════════

def log_exception(logger, msg: str = ""):
    """Log the current exception with full traceback.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    msg : str
        Additional context message.
    """
    exc_type, exc_value, exc_tb = sys.exc_info()
    tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logger.error(f"{msg}\n{tb_str}")


# ═══════════════════════════════════════════════════════════════════
# File I/O Helpers
# ═══════════════════════════════════════════════════════════════════

def save_array(filepath: str, arr, logger=None):
    """Save array to .npy file, handling GPU arrays.

    Parameters
    ----------
    filepath : str
        Output .npy path.
    arr : ndarray
        Array to save (numpy or cupy).
    logger : logging.Logger, optional
    """
    import numpy as np
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if hasattr(arr, 'get') and callable(arr.get):
        arr_np = arr.get()
    else:
        arr_np = arr

    np.save(filepath, arr_np)
    if logger:
        logger.debug(f"Saved: {filepath}  shape={arr_np.shape}  dtype={arr_np.dtype}")


def load_array(filepath: str, to_gpu: bool = False):
    """Load .npy file, optionally moving to GPU.

    Parameters
    ----------
    filepath : str
        .npy file path.
    to_gpu : bool
        If True, return cupy array.

    Returns
    -------
    ndarray
        Loaded array.
    """
    import numpy as np
    arr = np.load(filepath)
    if to_gpu:
        import cupy
        return cupy.asarray(arr)
    return arr
