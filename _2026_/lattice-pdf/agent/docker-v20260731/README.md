# docker-v20260731 — Gluon PDF Pipeline (GPU Double Precision)

GPU-accelerated disconnected gluon PDF validation pipeline with **double precision (complex128)**, per-config eigenvectors, and new data paths.

## Key Changes from v20260729

| Aspect | v20260729 | v20260731 |
|--------|-----------|-----------|
| **Precision** | complex64 (single) | **complex128 (double)** |
| **Eigenvectors** | Single shared .npy (cfg_48000) | **Per-config binary files** (72 time slices each) |
| **Nev1 (perams/VVV)** | 100 | **100** |
| **Perambulators** | mom_smear (mz2_my0_mx0) | **light/** subdirectory |
| **Data paths** | sunpeng/.../... paths | **Standard eigensystem/perambulators paths** |

## Data Paths

| Data | Path |
|------|------|
| Eigenvectors | `/public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/` |
| Perambulators | `/public/group/lqcd/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/` |
| Gauge configs | `/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/` |

## Fixed Configuration

```
Ensemble:   beta6.20_mu-0.2770_ms-0.2400_L24x72 (L24x72)
Lattice:    72×24³, a=0.1053 fm, β=6.20
Nev/Nev1:   100/100 (eigenvectors 100, perambulators 100)
Configs:    6250, 6450, 6650 (Nconf=3)
Momentum:   P=(0, 0, -2)
Operator:   _Cg5g4 (Cγ₅γ₄)
Precision:  complex128 (double)
```

## Eigenvector Format

Per-config, per-time-slice binary files:
- Path: `{eigenvector_base}/{conf_id}/eigvecs_t{000-071}_{conf_id}`
- Format: little-endian float64 pairs (real, imag)
- Shape per file: (Nev=100, Nx³=13824, Nc=3) → complex128
- Total per config: 72 files × ~63 MB = ~4.5 GB

## Pipeline Steps

| Step | Description | Details |
|------|-------------|---------|
| 0 | Environment check | GPU, dependencies, data paths |
| 1 | Proton 2pt distillation | VVV + Wick + parity projection (GPU) |
| 2 | OPE computation | F_{μν} → Wilson line → OPE (GPU) |
| 3 | huangcl ratio analysis | Jackknife + R(z) plots |
| 4 | Final report | Markdown with all results |

## Usage

```bash
cd /root/lattice-pdf/agent/docker-v20260731

# Full pipeline (all 3 configs, double precision)
python run_pipeline.py

# Single config test (recommended first run)
python run_pipeline.py --conf-id 6250

# Skip certain steps
python run_pipeline.py --skip-2pt --skip-ope  # analysis only
python run_pipeline.py --skip-analysis         # compute only

# Single precision (faster, less memory)
python run_pipeline.py --precision complex64 --conf-id 6250

# Enable eigenvector smearing
python run_pipeline.py --smear

# Verbose debug output
python run_pipeline.py --verbose --conf-id 6250
```

## Output Structure

```
output_YYYYMMDD_HHMMSS/
├── run.log                         # Main log
├── run_config.json                 # Runtime configuration
├── run_config_snapshot.json        # Config snapshot
├── gpu_info.json                   # GPU device info
├── final_report.md                 # Comprehensive report
├── timing.jsonl                    # Per-step timing
├── data/
│   ├── conf_6250/
│   │   ├── VVV_Nev150_*_conf6250.npy      # VVV baryon block
│   │   ├── twopt_slice_pp_*.npy           # Correlator (PP)
│   │   ├── twopt_slice_pm_*.npy           # Correlator (PM)
│   │   ├── twopt_slice_pp_*_contract.npy  # Raw contraction
│   │   ├── meff_Pz-2_conf6250.npz        # Effective mass
│   │   ├── compute_2pt_summary.json       # 2pt summary
│   │   ├── gauge_validation_conf6250.json # Gauge validation
│   │   ├── Fmunu_mu0_nu1.npz             # F_{xy} field strength
│   │   ├── Fmunu_mu3_nu0.npz             # F_{tx} field strength
│   │   ├── Fmunu_mu3_nu1.npz             # F_{ty} field strength
│   │   ├── ops_mu*_nu*_dz24_conf6250.npz # OPE components
│   │   └── compute_ope_summary_conf6250.json
│   ├── conf_6450/...
│   └── conf_6650/...
└── plots/
    ├── ratio.png                    # R(z) ratio
    ├── ratio_diagnostics.png        # Ratio diagnostics
    ├── effective_mass.png           # Effective mass + plateau
    ├── field_strength_diagnostics.png # Field strength diagnostics
    ├── ratio_results.npz            # Numerical results
    ├── ope_combined.npz             # Combined OPE
    └── correlators_rel_time.npz     # Relative-time correlators
```

## Dependencies

- Python 3.8+
- CuPy (CUDA 12.x)
- numpy, scipy, matplotlib
- opt_einsum (optional)

## Memory Considerations

- **Double precision**: Uses ~2× more GPU memory than single precision
- **Eigenvectors**: ~4.5 GB per config (CPU-resident, streamed slice by slice to GPU)
- **Gauge config**: ~150 MB (GPU-resident during OPE)
- Peak CPU memory: ~6-8 GB per config
