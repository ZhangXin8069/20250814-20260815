# Gluon PDF Validation Pipeline Report (GPU, Double Precision)

**Version**: docker-v20260731 (GPU DOUBLE PRECISION, per-config eigvecs)
**Run time**: 2026-07-28 09:03:29
**Total elapsed**: 9088.4s (151.5 min)
**Peak CPU memory**: 9.15 GB
**Output directory**: `/root/lattice-pdf/agent/docker-v20260731/output_20260728_063202`

**GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (CC 8.9, 8.0 GB, CuPy 14.0.1, CUDA 12040)

## Configuration

| Parameter | Value |
|-----------|-------|
| Ensemble | beta6.20_mu-0.2770_ms-0.2400_L24x72 (L24x72) |
| Lattice | 72×24³, β=6.2 |
| Lattice spacing | a=0.1053 fm |
| Configs | [6250, 6450, 6650] (Nconf=3) |
| Momentum | P=(0,0,-2) |
| Nev (eigvecs) / Nev1 (perams) | 100 / 100 |
| Element | _Cg5g4 |
| delta_z | 24 |
| Jackknife | True |
| GPU precision | **complex128** |
| OPE mode | **FROM SCRATCH (GPU)** |
| VVV / Wick | **GPU (CuPy einsum)** |
| F_{μν} | **GPU (CuPy plaquette_clover)** |
| Eigenvectors | **Per-config** (binary per time slice) |

## Data Paths

| Data | Path |
|------|------|
| Eigenvectors | `/public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/` |
| Perambulators | `/public/group/lqcd/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/` |
| Gauge configs | `/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/` |
| OPE | *Computed from scratch (GPU)* |

## Step 0: Environment Check

- GPU available: True
- All required OK: True

## Step 1: Proton 2pt Distillation (GPU, per-config eigvecs)

### conf=6250 ✓
- Pz=-2: PP range [-3.109232986102396e-05, 0.009151237523364457], m_eff(plateau)≈1.7775052445957689 GeV
### conf=6450 ✓
- Pz=-2: PP range [-1.0290241675005523e-05, 0.008785220183420659], m_eff(plateau)≈1.7942484987512863 GeV
### conf=6650 ✓
- Pz=-2: PP range [-9.724464465958684e-06, 0.00897222970944172], m_eff(plateau)≈1.7728716936756235 GeV

## Step 2: OPE Computation (GPU, FROM SCRATCH, double precision)

### conf=6250 ✓ (3/3 components)
- Unitarity: max_dev=6.661338147750939e-16
- Plaq trace: re=-0.03154470865782757
  - mu0_nu1 ✓: |O|∈[-1.06e+01, 2.38e+02]
  - mu3_nu0 ✓: |O|∈[-1.06e+01, 2.34e+02]
  - mu3_nu1 ✓: |O|∈[-1.04e+01, 2.31e+02]
### conf=6450 ✓ (3/3 components)
- Unitarity: max_dev=8.881784197001252e-16
- Plaq trace: re=-0.05795538397709445
  - mu0_nu1 ✓: |O|∈[-1.00e+01, 2.35e+02]
  - mu3_nu0 ✓: |O|∈[-8.90e+00, 2.31e+02]
  - mu3_nu1 ✓: |O|∈[-1.06e+01, 2.34e+02]
### conf=6650 ✓ (3/3 components)
- Unitarity: max_dev=8.881784197001252e-16
- Plaq trace: re=-0.05220556922747037
  - mu0_nu1 ✓: |O|∈[-8.98e+00, 2.42e+02]
  - mu3_nu0 ✓: |O|∈[-1.08e+01, 2.34e+02]
  - mu3_nu1 ✓: |O|∈[-1.04e+01, 2.38e+02]

## Step 3: huangcl Ratio Analysis

✓ Analysis completed successfully
- Loaded configs: [6250, 6450, 6650]
- Ratio plot: `/root/lattice-pdf/agent/docker-v20260731/output_20260728_063202/plots/ratio.png`
- Diagnostics: `/root/lattice-pdf/agent/docker-v20260731/output_20260728_063202/plots/ratio_diagnostics.png`
- Effective mass: `/root/lattice-pdf/agent/docker-v20260731/output_20260728_063202/plots/effective_mass.png`
- Field strength: `/root/lattice-pdf/agent/docker-v20260731/output_20260728_063202/plots/field_strength_diagnostics.png`

## Output Files

```
output_20260728_063202/
  gpu_info.json  (287 B)
  run.log  (96 KB)
  run_config.json  (2 KB)
  run_config_snapshot.json  (3 KB)
  timing.jsonl  (9 KB)
  plots/
    correlators_rel_time.npz  (33.3 MB)
    effective_mass.png  (242 KB)
    field_strength_diagnostics.png  (595 KB)
    ope_combined.npz  (82 KB)
    ratio.png  (76 KB)
    ratio_diagnostics.png  (570 KB)
    ratio_results.npz  (677 KB)
  data/
    conf_6650/
      Fmunu_mu0_nu1.npz  (136.7 MB)
      Fmunu_mu3_nu0.npz  (136.7 MB)
      Fmunu_mu3_nu1.npz  (136.7 MB)
      VVV_Nev1100_Px0Py0Pz-2_nosmear_conf6650.npy  (1.1 GB)
      compute_2pt_summary.json  (717 B)
      compute_ope_summary_conf6650.json  (2 KB)
      gauge_validation_conf6650.json  (406 B)
      meff_Pz-2_conf6650.npz  (3 KB)
      ops_mu0_nu1_dz24_conf6650.npz  (28 KB)
      ops_mu3_nu0_dz24_conf6650.npz  (28 KB)
      ops_mu3_nu1_dz24_conf6650.npz  (28 KB)
      twopt_slice_pm_Px0Py0Pz-2_eginphase2_Cg5g4_nopol_ss_conf6650.npy  (81 KB)
      twopt_slice_pp_Px0Py0Pz-2_eginphase2_Cg5g4_contract_conf6650.npy  (1.3 MB)
      twopt_slice_pp_Px0Py0Pz-2_eginphase2_Cg5g4_nopol_ss_conf6650.npy  (81 KB)
    conf_6250/
      Fmunu_mu0_nu1.npz  (136.7 MB)
      Fmunu_mu3_nu0.npz  (136.7 MB)
      Fmunu_mu3_nu1.npz  (136.7 MB)
      VVV_Nev1100_Px0Py0Pz-2_nosmear_conf6250.npy  (1.1 GB)
      compute_2pt_summary.json  (719 B)
      compute_ope_summary_conf6250.json  (2 KB)
      gauge_validation_conf6250.json  (404 B)
      meff_Pz-2_conf6250.npz  (3 KB)
      ops_mu0_nu1_dz24_conf6250.npz  (28 KB)
      ops_mu3_nu0_dz24_conf6250.npz  (28 KB)
      ops_mu3_nu1_dz24_conf6250.npz  (28 KB)
      twopt_slice_pm_Px0Py0Pz-2_eginphase2_Cg5g4_nopol_ss_conf6250.npy  (81 KB)
      twopt_slice_pp_Px0Py0Pz-2_eginphase2_Cg5g4_contract_conf6250.npy  (1.3 MB)
      twopt_slice_pp_Px0Py0Pz-2_eginphase2_Cg5g4_nopol_ss_conf6250.npy  (81 KB)
    conf_6450/
      Fmunu_mu0_nu1.npz  (136.7 MB)
      Fmunu_mu3_nu0.npz  (136.7 MB)
      Fmunu_mu3_nu1.npz  (136.7 MB)
      VVV_Nev1100_Px0Py0Pz-2_nosmear_conf6450.npy  (1.1 GB)
      compute_2pt_summary.json  (719 B)
      compute_ope_summary_conf6450.json  (2 KB)
      gauge_validation_conf6450.json  (406 B)
      meff_Pz-2_conf6450.npz  (3 KB)
      ops_mu0_nu1_dz24_conf6450.npz  (28 KB)
      ops_mu3_nu0_dz24_conf6450.npz  (28 KB)
      ops_mu3_nu1_dz24_conf6450.npz  (28 KB)
      twopt_slice_pm_Px0Py0Pz-2_eginphase2_Cg5g4_nopol_ss_conf6450.npy  (81 KB)
      twopt_slice_pp_Px0Py0Pz-2_eginphase2_Cg5g4_contract_conf6450.npy  (1.3 MB)
      twopt_slice_pp_Px0Py0Pz-2_eginphase2_Cg5g4_nopol_ss_conf6450.npy  (81 KB)
```

## Notes

1. **GPU acceleration**: VVV, Wick contraction, F_{μν}, Wilson line, and OPE contraction all computed on GPU (CuPy/CUDA).
2. **Double precision (complex128)**: All computation in complex128 for maximum numerical accuracy.
3. **Per-config eigenvectors**: Loaded from binary per-time-slice files (not shared cfg_48000).
4. **Nev1=100 perambulators**: Perambulators use 100 eigenvectors, matching eigenvector count.
5. **OPE from scratch**: Gauge config → Clover plaquette → F_{μν} → Wilson line → nonlocal OPE → .npz, all on GPU.
6. **All intermediate results saved**: VVV blocks, F_{μν} tensors, OPE components, correlators, ratio data.
7. **Peak CPU memory**: 9.15 GB
8. **v20260729 → v20260731**: Per-config eigenvectors, Nev1=100, double precision, new data paths.

---
*Generated by docker-v20260731 (GPU, double precision) pipeline on 2026-07-28 09:03:29*