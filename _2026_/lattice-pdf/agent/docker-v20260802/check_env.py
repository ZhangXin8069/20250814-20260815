#!/usr/bin/env python3
"""
Environment checker — docker-v20260802

Verifies:
  - GPU/CuPy availability and device info
  - Python dependencies (numpy, scipy, matplotlib, cupy, h5py)
  - Data paths: per-config eigenvectors, per-config perambulators (light/), gauge configs
  - All data accessibility for configured config IDs

Usage:
    python check_env.py
"""
import sys, os, json

# Add script directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cupy as cp

# ── GPU info ───────────────────────────────────────────────────────────────
d = cp.cuda.Device()
mem = d.mem_info
props = cp.cuda.runtime.getDeviceProperties(d.id)
name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
print(f'GPU: {name}')
print(f'  Compute Capability: {props["major"]}.{props["minor"]}')
print(f'  Memory: {mem[0]/1024**3:.1f}/{mem[1]/1024**3:.1f} GB free')
print(f'  CuPy: {cp.__version__}  |  CUDA Runtime: {cp.cuda.runtime.runtimeGetVersion()}')
print()

# ── Module imports ─────────────────────────────────────────────────────────
for mod in ['numpy', 'scipy', 'matplotlib', 'cupy', 'h5py']:
    try:
        m = __import__(mod)
        ver = getattr(m, '__version__', '?')
        print(f'  {"✓"} {mod}: {ver}')
    except ImportError:
        print(f'  ✗ {mod}: MISSING')

print()

# ── Load config for data paths ─────────────────────────────────────────────
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_config.json")
with open(config_path) as f:
    cfg = json.load(f)
paths = cfg["data_paths"]
conf_ids = cfg["parameters"]["conf_ids"]

print("Data paths (v20260802 — per-config format):")
all_ok = True

# Check eigenvectors (per-config, per-time-slice binary files)
for conf_id in conf_ids:
    eigvec_dir = os.path.join(paths["eigenvector_base"], str(conf_id))
    t0 = os.path.join(eigvec_dir, f"eigvecs_t000_{conf_id}")
    if os.path.exists(t0):
        n = len([f for f in os.listdir(eigvec_dir) if f.startswith("eigvecs_")])
        sz = sum(os.path.getsize(os.path.join(eigvec_dir, f))
                for f in os.listdir(eigvec_dir) if f.startswith("eigvecs_")) / 1024**3
        print(f'  ✓ eigvec conf={conf_id}: {n} time-slice files ({sz:.1f} GB total)')
    else:
        print(f'  ✗ eigvec conf={conf_id}: MISSING ({t0})')
        all_ok = False

# Check perambulators (light/{conf_id}/ — code appends /light/{conf_id})
for conf_id in conf_ids:
    peram_dir = os.path.join(paths["perambulator_base"], "light", str(conf_id))
    p0 = os.path.join(peram_dir, f"perams.{conf_id}.0.0")
    if os.path.exists(p0):
        n = len([f for f in os.listdir(peram_dir) if f.startswith(f"perams.{conf_id}")])
        print(f'  ✓ peram conf={conf_id}: {n} files in {peram_dir}')
    else:
        print(f'  ✗ peram conf={conf_id}: MISSING ({p0})')
        all_ok = False

# Check gauge configs (.lime files)
for conf_id in conf_ids:
    gf = os.path.join(paths["gauge_config_base"],
                     paths["gauge_config_pattern"].format(conf_id=conf_id))
    if os.path.exists(gf):
        sz = os.path.getsize(gf) / 1024**3
        print(f'  ✓ gauge conf={conf_id}: {gf} ({sz:.1f} GB)')
    else:
        print(f'  ✗ gauge conf={conf_id}: MISSING')
        all_ok = False

print()
if all_ok:
    print('All data paths accessible — ready to run!')
    print()
    print('Quick test commands:')
    print(f'  cd {os.path.dirname(os.path.abspath(__file__))}')
    print('  python run_pipeline.py --conf-id 6250 --skip-ope --skip-analysis  # 2pt only (~8 min)')
    print('  python run_pipeline.py --conf-id 6250 --skip-2pt --skip-analysis  # OPE only (~2 min)')
    print('  python run_pipeline.py --conf-id 6250                            # Full single-config')
    print('  python run_pipeline.py                                            # Full 3-config run')
else:
    print('Some data paths missing — check above.')
