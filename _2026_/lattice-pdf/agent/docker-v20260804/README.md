# docker-v20260804 — Full Distillation Pipeline (GPU)

Comprehensive GPU-accelerated lattice QCD distillation pipeline. Computes vertex functions, performs Wick contractions for multiple correlator types, and conducts full statistical analysis — all on CUDA GPUs using CuPy.

## Version Identity

**v20260804** — Full distillation toolkit with vertex computation + multi-hadron Wick contractions + statistical analysis.

Built upon patterns from:
- `LQCD_Master` (Planner→Executor pipeline)
- `lamet-agent` (5-stage analysis pipeline)
- `sush/lqcddb` (Wick contraction + vertex + analysis framework)
- `docker-v20260802` (GPU pipeline architecture)

## Features

| Feature | Status |
|---------|--------|
| VdV vertex computation | ✓ GPU (CuPy einsum) |
| VVV baryon block computation | ✓ GPU (Levi-Civita ×6 permutations) |
| Pion 2pt correlator | ✓ |
| Proton 2pt correlator (pp) | ✓ |
| Neutron 2pt correlator (nn) | ✓ |
| OPE disconnected loop | ✓ (fermion loop) |
| PJN 3pt correlator | ✓ |
| PJNNJNp 4pt correlator | △ (simplified) |
| Jackknife resampling | ✓ |
| Effective mass (log/cosh) | ✓ |
| 3pt/2pt ratio analysis | ✓ |
| Plateau fitting | ✓ |
| Validation against expected values | ✓ |
| Multi-config analysis | ✓ |

## Ensemble & Data

```
Ensemble:   beta6.20_mu-0.2770_ms-0.2400_L24x72
Lattice:    72 × 24³
a:          0.1053 fm (β = 6.20)
Nev:        100
Configs:    6250, 6450, 6650
```

### Data Paths

| Data | Path |
|------|------|
| Eigenvectors | `/public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/` |
| Perambulators | `/public/group/lqcd/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/` |
| Gauge configs | `/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/` |

## Quick Start

```bash
cd /root/lattice-pdf/agent/docker-v20260804

# 1. Check environment
python check_env.py

# 2. Run single-config test (~15 min)
python run_pipeline.py --conf-id 6250 --skip-4pt

# 3. Run full pipeline (all configs)
python run_pipeline.py

# 4. Single precision (default)
python run_pipeline.py --precision complex64

# 5. With verbose logging
python run_pipeline.py --verbose

# 6. Analysis-only (from saved intermediates)
python run_pipeline.py --skip-2pt --skip-ope --skip-3pt --skip-4pt
```

## Pipeline Steps

```
Step 0: Environment check (GPU, deps, data paths)
Step 1: Data loading (eigenvectors + perambulators from cluster)
Step 2: Vertex computation (VdV + VVV on GPU)
Step 3: Wick contraction (2pt pp/pn, OPE, 3pt PJN, 4pt PJNNJNp)
Step 4: Statistical analysis (Jackknife, meff, ratio_3pt)
Step 5: Plotting (correlators, effective masses)
Step 6: Final report (Markdown)
```

## Output Structure

```
output/output_YYYYMMDD_HHMMSS/
├── data/
│   ├── conf6250/
│   │   ├── VdV.npy                 # (Nt, N_mom, Nev, Nev)
│   │   ├── VVV.npy                 # (Nt, N_mom, Nev, Nev, Nev)
│   │   ├── pion_2pt_looped.npy     # (Nt,) time-averaged
│   │   ├── proton_2pt_looped.npy   # (N_mom, N_mom, Nt)
│   │   ├── neutron_2pt_looped.npy  # (N_mom, N_mom, Nt)
│   │   ├── ope_loop.npy
│   │   ├── pjn_3pt.npy
│   │   └── pjnnjnp_4pt.npy
│   ├── conf6450/
│   ├── conf6650/
│   └── momentum_list.npy
├── analysis/
│   └── summary.json
├── plots/
│   ├── pion_meff_P(0,0,0).png
│   ├── pion_meff_P(0,0,2).png
│   ├── proton_meff_P(0,0,0).png
│   ├── proton_meff_P(0,0,2).png
│   ├── neutron_meff_P(0,0,0).png
│   ├── neutron_meff_P(0,0,2).png
│   └── ...
├── run_config.json
├── run.log
└── REPORT.md
```

## Validation Targets

| Particle | Momentum | Expected E (GeV) | Description |
|----------|----------|-------------------|-------------|
| Pion | P=(0,0,0) | ~0.14 | Rest mass |
| Pion | P=(0,0,2) | ~0.52 | Moving frame |
| Proton | P=(0,0,0) | ~1.0 | Rest mass |
| Proton | P=(0,0,2) | ~1.4 | Moving frame (E = √(m₀² + p²)) |

## Logs

All pipeline logs are written to `/root/lattice-pdf/agent/logs/` in addition to the per-run `output_*/run.log`.

## Dependencies

- Python ≥ 3.8
- numpy, scipy, matplotlib
- CuPy (NVIDIA CUDA 12.x)
- opt_einsum (optional; falls back to numpy.einsum)

## Module Map

| File | Purpose |
|------|---------|
| `run_pipeline.py` | Main orchestrator — 6-step pipeline |
| `check_env.py` | GPU environment verification |
| `utils.py` | Shared utilities (GPU, logging, timing, I/O) |
| `gamma_matrix_gpu.py` | DeGrand-Rossi gamma matrices on GPU |
| `data_io.py` | Eigenvector, perambulator, gauge config readers |
| `compute_vertex.py` | VdV and VVV momentum-projected vertices |
| `compute_contraction.py` | Wick contraction for all correlators |
| `analyze.py` | Jackknife, effective mass, ratio_3pt, plotting |
