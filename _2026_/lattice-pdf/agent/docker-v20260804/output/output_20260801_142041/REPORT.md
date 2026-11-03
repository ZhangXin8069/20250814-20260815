# Lattice QCD Distillation Pipeline Report

**Version:** docker-v20260804 (CORRECTED)
**Date:** 2026-08-01 14:35:37
**Total time:** 895.2s (14.9 min)

## Configuration

| Parameter | Value |
|-----------|-------|
| Ensemble | beta6.20_mu-0.2770_ms-0.2400_L24x72 |
| Lattice | 72×24³ |
| a (fm) | 0.1053 |
| Nev/Nev1 | 100/100 |
| Configs | [6250, 6450, 6650] |
| Precision | complex64 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Operator | _Cg5g4 |

## Validation: Effective Masses

| Particle | Momentum | E_expected (GeV) | E_fitted (GeV) | χ²/dof | Assessment |
|----------|----------|-------------------|----------------|---------|------------|
| pion | P(0,0,0) | 0.140 | 0.306 | 1.48 | △ Marginal |
| pion | P(0,0,2) | 0.991 | 0.306 | 1.48 | ✗ CHECK |
| proton | P(0,0,0) | 0.938 | nan | nan | ✗ No plateau |
| proton | P(0,0,2) | 1.357 | nan | nan | ✗ No plateau |

## Output Files

```
output_20260801_142041/
  run_config.json
  analysis/
    summary.json
  plots/
    pion_corr_P(0,0,0).png
    pion_corr_P(0,0,2).png
    pion_meff_P(0,0,0).png
    pion_meff_P(0,0,2).png
    proton_corr_P(0,0,0).png
    proton_corr_P(0,0,2).png
    proton_meff_P(0,0,0).png
    proton_meff_P(0,0,2).png
  data/
    momentum_list.npy
    conf6450/
      VVV.npy
      VdV.npy
      pion_C2pt_1d.npy
      proton_C2pt_1d.npy
      proton_corr_pp.npy
    conf6250/
      VVV.npy
      VdV.npy
      pion_C2pt_1d.npy
      proton_C2pt_1d.npy
      proton_corr_pp.npy
    conf6650/
      VVV.npy
      VdV.npy
      pion_C2pt_1d.npy
      proton_C2pt_1d.npy
      proton_corr_pp.npy
```

---

🤖 Generated with [Claude Code](https://claude.com/claude-code) — docker-v20260804