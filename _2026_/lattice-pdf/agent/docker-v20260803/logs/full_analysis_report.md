# Lattice QCD Full Analysis Report

**Date:** 2026-08-01 13:42:11
**Ensemble:** beta6.20_mu-0.2770_ms-0.2400_L24x72 (24³×72, a=0.1053 fm, a⁻¹=1.874 GeV)
**Configs:** [6250, 6450, 6650] (Nconf=3)
**Nev:** 50
**Total time:** 0.1 min

**Momentum P=(0,0,2):** p_z = 2×(2π/L) = 0.981 GeV

---
## Effective Mass Results

| Channel | E₀ [GeV] | Expected [GeV] | σ | Plateau |
|---------|-----------|----------------|---|---------|
| Proton P=0 | 1.0526 ± 0.0102 | 1.000 | 5.1σ ❌ | [4,13] |
| Proton P=(0,0,2) | 1.6283 ± 0.0245 | 1.439 | 7.7σ ❌ | [4,13] |
| Pion P=0 | 0.2655 ± 0.0004 | 0.300 | 81.3σ ❌ | [5,17] |
| Pion P=(0,0,2) | 0.9552 ± 0.5265 | 1.016 | 0.1σ ✅ | [5,17] |

---
## Dispersion Relation Check

- **Proton:** E(Pz=0) = 1.0526 GeV, E(Pz=2) = 1.6283 GeV
  - Theoretical: √(m₀²+p²) = √(1.053²+0.981²) = 1.4390 GeV
  - Difference: 0.1894 GeV
- **Pion:** E(Pz=0) = 0.2655 GeV, E(Pz=2) = 0.9552 GeV
  - Theoretical: √(m₀²+p²) = √(0.265²+0.981²) = 1.0165 GeV
  - Difference: 0.0613 GeV

---
## Output Files

- `/root/lattice-pdf/agent/docker-v20260803/data/{conf_id}/VdV_mom_{conf_id}.npy` — VdV vertices (Nt, 2, Nev, Nev)
- `/root/lattice-pdf/agent/docker-v20260803/data/{conf_id}/VVV_mom_{conf_id}.npy` — VVV vertices (Nt, 2, Nev, Nev, Nev)
- `/root/lattice-pdf/agent/docker-v20260803/data/{conf_id}/corr_*_{conf_id}.npy` — Correlators (Nt,)
- `/root/lattice-pdf/agent/docker-v20260803/data/analysis/` — Jackknife means, errors, meff
- `/root/lattice-pdf/agent/docker-v20260803/plots/` — Figures
- `/root/lattice-pdf/agent/docker-v20260803/logs/` — Logs and this report