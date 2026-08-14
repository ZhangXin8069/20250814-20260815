# Gluon PDF Pipeline — docker-v20260730 GPU Report

**Generated**: 2026-07-28 08:23:35
**Run ID**: docker_v20260730_gpu
**Precision**: complex64

## Ensemble

| Parameter | Value |
|-----------|-------|
| Name | beta6.20_mu-0.2770_ms-0.2400_L24x72 |
| L³×T | 24³×72 |
| β | 6.2 |
| a (fm) | 0.1053 |
| Nc | 3 |

## Parameters

| Parameter | Value |
|-----------|-------|
| Nev | 100 |
| Momentum | P=(0,0,-2) |
| Element | _Cg5g4 |
| Configs | [6250, 6450, 6650] (Nconf=3) |
| Smearing | OFF |
| meff method | fit_cosh |
| Jackknife | True |

## Data Paths

| Data | Path |
|------|------|
| Eigenvectors | `/public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/` |
| Perambulators | `/public/group/lqcd/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/` |
| Gauge configs | `/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72` |

## Timing

| Step | Time (s) | GPU Free (MB) |
|------|----------|---------------|
| load_eigvecs_conf6250 | 9.0 | 7099.0 |
| VVV_GPU_Pz-2_conf6250 | 63.9 | 6915.0 |
| Wick_GPU_Pz-2_conf6250 | 173.1 | 6435.0 |
| meff_Pz-2_conf6250 | 0.2 | 6435.0 |
| load_eigvecs_conf6450 | 9.5 | 7087.0 |
| VVV_GPU_Pz-2_conf6450 | 63.7 | 6915.0 |
| Wick_GPU_Pz-2_conf6450 | 172.3 | 6435.0 |
| meff_Pz-2_conf6450 | 0.1 | 6435.0 |
| load_eigvecs_conf6650 | 9.4 | 7087.0 |
| VVV_GPU_Pz-2_conf6650 | 64.1 | 6915.0 |
| Wick_GPU_Pz-2_conf6650 | 136.4 | 6435.0 |
| meff_Pz-2_conf6650 | 0.1 | 6435.0 |
| read_gauge_conf6250 | 28.6 | 6813.0 |
| validate_gauge_conf6250 | 0.0 | 6813.0 |
| ope_GPU_mu0_nu1_conf6250 | 22.1 | 4959.0 |
| ope_GPU_mu3_nu0_conf6250 | 22.4 | 4959.0 |
| ope_GPU_mu3_nu1_conf6250 | 22.8 | 4959.0 |
| read_gauge_conf6450 | 28.3 | 6813.0 |
| validate_gauge_conf6450 | 0.0 | 6813.0 |
| ope_GPU_mu0_nu1_conf6450 | 22.5 | 4685.0 |
| ope_GPU_mu3_nu0_conf6450 | 23.0 | 4685.0 |
| ope_GPU_mu3_nu1_conf6450 | 22.4 | 4685.0 |
| read_gauge_conf6650 | 28.4 | 6813.0 |
| validate_gauge_conf6650 | 0.0 | 6813.0 |
| ope_GPU_mu0_nu1_conf6650 | 22.5 | 4685.0 |
| ope_GPU_mu3_nu0_conf6650 | 20.8 | 4685.0 |
| ope_GPU_mu3_nu1_conf6650 | 23.3 | 4685.0 |
| **TOTAL** | **989.0** | |

## 2pt Results

| Config | Pz | meff (GeV) | Method |
|--------|----|------------|--------|
| 6250 | -2 | 1.7775 | fit_cosh |
| | | 1.7775 (fit_cosh) |
| | | 1.7716 (fit_exp) |
| | | 1.4384 (exp_forward) |
| | | 6.0846 (cosh) |
| 6450 | -2 | 1.7942 | fit_cosh |
| | | 1.7942 (fit_cosh) |
| | | 1.7951 (fit_exp) |
| | | 3.4929 (exp_forward) |
| | | 6.9253 (cosh) |
| 6650 | -2 | 1.7729 | fit_cosh |
| | | 1.7729 (fit_cosh) |
| | | 1.7744 (fit_exp) |
| | | 2.1924 (exp_forward) |
| | | 7.8781 (cosh) |

## OPE Results

- Config 6250: **ok**
  - mu0_nu1: shape=[24, 72], |O|=[-7.30e+01, 6.36e+01]
  - mu3_nu0: shape=[24, 72], |O|=[-5.74e+01, 5.31e+01]
  - mu3_nu1: shape=[24, 72], |O|=[-5.25e+01, 5.53e+01]
- Config 6450: **ok**
  - mu0_nu1: shape=[24, 72], |O|=[-5.55e+01, 5.51e+01]
  - mu3_nu0: shape=[24, 72], |O|=[-5.12e+01, 4.91e+01]
  - mu3_nu1: shape=[24, 72], |O|=[-5.69e+01, 5.15e+01]
- Config 6650: **ok**
  - mu0_nu1: shape=[24, 72], |O|=[-5.54e+01, 5.28e+01]
  - mu3_nu0: shape=[24, 72], |O|=[-7.08e+01, 5.44e+01]
  - mu3_nu1: shape=[24, 72], |O|=[-4.84e+01, 4.38e+01]

## huangcl Analysis

- Status: **ok**
- Loaded configs: [6250, 6450, 6650]
- Elapsed: 2.8s

## Output Files

```
output_20260728_080657/
  env_check.json  (2 KB)
  run.log  (83 KB)
  run_config_snapshot.json  (3 KB)
  timing.jsonl  (8 KB)
  plots/
    analysis_summary.json  (2 KB)
    correlators_rel_time.npz  (33.3 MB)
    effective_mass.png  (242 KB)
    field_strength_diagnostics.png  (574 KB)
    ope_combined.npz  (82 KB)
    ratio.png  (74 KB)
    ratio_diagnostics.png  (619 KB)
    ratio_results.npz  (677 KB)
  data/
    compute_2pt_summary.json  (2 KB)
    compute_ope_summary.json  (6 KB)
    conf_6650/
      Fmunu_mu0_nu1.npz  (68.3 MB)
      Fmunu_mu3_nu0.npz  (68.3 MB)
      Fmunu_mu3_nu1.npz  (68.3 MB)
      Ftilde_mu0_nu1.npz  (68.3 MB)
      Ftilde_mu3_nu0.npz  (68.3 MB)
      Ftilde_mu3_nu1.npz  (68.3 MB)
      VVV_Nev1100_Px0Py0Pz-2_nosmear_conf6650.npy  (549.3 MB)
      compute_2pt_summary.json  (716 B)
      compute_ope_summary_conf6650.json  (2 KB)
      gauge_validation_conf6650.json  (313 B)
      meff_Pz-2_conf6650.npz  (3 KB)
      ops_mu0_nu1_dz24_conf6650.npz  (15 KB)
      ops_mu3_nu0_dz24_conf6650.npz  (15 KB)
      ops_mu3_nu1_dz24_conf6650.npz  (15 KB)
      twopt_slice_pm_Px0Py0Pz-2_eginphase2_Cg5g4_nopol_ss_conf6650.npy  (41 KB)
      twopt_slice_pp_Px0Py0Pz-2_eginphase2_Cg5g4_contract_conf6650.npy  (648 KB)
      twopt_slice_pp_Px0Py0Pz-2_eginphase2_Cg5g4_nopol_ss_conf6650.npy  (41 KB)
    conf_6250/
      Fmunu_mu0_nu1.npz  (68.3 MB)
      Fmunu_mu3_nu0.npz  (68.3 MB)
      Fmunu_mu3_nu1.npz  (68.3 MB)
      Ftilde_mu0_nu1.npz  (68.3 MB)
      Ftilde_mu3_nu0.npz  (68.3 MB)
      Ftilde_mu3_nu1.npz  (68.3 MB)
      VVV_Nev1100_Px0Py0Pz-2_nosmear_conf6250.npy  (549.3 MB)
      compute_2pt_summary.json  (718 B)
      compute_ope_summary_conf6250.json  (2 KB)
      gauge_validation_conf6250.json  (313 B)
      meff_Pz-2_conf6250.npz  (3 KB)
      ops_mu0_nu1_dz24_conf6250.npz  (15 KB)
      ops_mu3_nu0_dz24_conf6250.npz  (15 KB)
      ops_mu3_nu1_dz24_conf6250.npz  (15 KB)
      twopt_slice_pm_Px0Py0Pz-2_eginphase2_Cg5g4_nopol_ss_conf6250.npy  (41 KB)
      twopt_slice_pp_Px0Py0Pz-2_eginphase2_Cg5g4_contract_conf6250.npy  (648 KB)
      twopt_slice_pp_Px0Py0Pz-2_eginphase2_Cg5g4_nopol_ss_conf6250.npy  (41 KB)
    conf_6450/
      Fmunu_mu0_nu1.npz  (68.3 MB)
      Fmunu_mu3_nu0.npz  (68.3 MB)
      Fmunu_mu3_nu1.npz  (68.3 MB)
      Ftilde_mu0_nu1.npz  (68.3 MB)
      Ftilde_mu3_nu0.npz  (68.3 MB)
      Ftilde_mu3_nu1.npz  (68.3 MB)
      VVV_Nev1100_Px0Py0Pz-2_nosmear_conf6450.npy  (549.3 MB)
      compute_2pt_summary.json  (716 B)
      compute_ope_summary_conf6450.json  (2 KB)
      gauge_validation_conf6450.json  (314 B)
      meff_Pz-2_conf6450.npz  (3 KB)
      ops_mu0_nu1_dz24_conf6450.npz  (15 KB)
      ops_mu3_nu0_dz24_conf6450.npz  (15 KB)
      ops_mu3_nu1_dz24_conf6450.npz  (15 KB)
      twopt_slice_pm_Px0Py0Pz-2_eginphase2_Cg5g4_nopol_ss_conf6450.npy  (41 KB)
      twopt_slice_pp_Px0Py0Pz-2_eginphase2_Cg5g4_contract_conf6450.npy  (648 KB)
      twopt_slice_pp_Px0Py0Pz-2_eginphase2_Cg5g4_nopol_ss_conf6450.npy  (41 KB)
```
