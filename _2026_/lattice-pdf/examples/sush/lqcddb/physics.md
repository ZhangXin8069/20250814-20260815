### This file's gamma matrix is in lqcddb.gamma
### Common Particle Operator Library
Meson               abbreviation   Operator
pion                  p(+,-,0)     ['d^d', 'gamma_5', 'u'](+), ['u^d', 'gamma_5', 'd'](-), 1/\sqrt(2)(['u^d', 'gamma_5', 'u'] - ['d^d', 'gamma_5', 'd'])(0), 

Baryon              abbreviation   Operator
neutron                 N          ['d', 'u', 'C * gamma_5', 'd']
proton                  P          ['u', 'u', 'C * gamma_5', 'd'] 
lambda                lambda       2 * ['s', 'u', 'C * gamma_5', 'd'] + ['d', 'u', 'C * gamma_5', 's'] - ['u', 'd', 'C * gamma_5', 's']

### Current
name                Style                                       Baryon projector
Scalar              \bar{\psi} \gamma_{0} \psi                  (\gamma_{0} + \gamma_{4}) / 2
Vector              \bar{\psi} \gamma_{\mu} \psi                (\gamma_{0} + \gamma_{4}) / 2,                          \mu = 1, 2, 3, 4
Axial vector        \bar{\psi} \gamma_{5} \gamma_{i} \psi       (\gamma_{0} + \gamma_{4})\gamma_{5} \gamma_{i} / 2,     i = 1, 2, 3
Tensor              \bar{\psi} \gamma_{i} \gamma_{j} \psi       (\gamma_{0} + \gamma_{4})\gamma_{5} \gamma_{k} / 2,     \epsilon_{ijk}

About the interaction
Electromagnetic     ['q^d', 'gamma_mu', 'q'] (q means 'udscbt' quark, but In general, only the 'uds' are considered.) 
weak                ['u^d', 'gamma_w', 'd'], ['d^d', 'gamma_w', 'u']..., gamma_w = (1-gamma_5) * gamma_mu

### Configurations
init_eigen_path=/nexdata/project/lqcd/sush/eigensystem
init_peram_path=/nexdata/project/lqcd/sush/perambulators


| conf | a (fm) | n³_L × n_T | m_π (MeV) | m_π L | m_ηs (MeV)|    vector path {init_eigen_path}     |    peram path {init_peram_path}      | 
|--------|---------|--------|------------|------|------------|
| C24P34 | 0.1053  | 24×64  | 341.1(1.8) | 4.38 | 748.61(75) | None
| C24P29 | 0.1053  | 24×72  | 292.7(1.2) | 3.75 | 657.83(64) | beta6.20_mu-0.2770_ms-0.2400_L24x72  |  beta6.20_mu-0.2770_ms-0.2400_L24x72 |
| C32P29 | 0.1053  | 32×64  | 292.4(1.1) | 5.01 | 658.80(43) | None
| C24P23 | 0.1053  | 24×64  | 229.5(3.0) | 2.96 | 645.67(99) | None
| C32P23 | 0.1053  | 32×64  | 228.0(1.2) | 3.91 | 643.93(45) | None
| C48P23 | 0.1053  | 48×96  | 225.6(0.9) | 5.79 | 644.08(62) | None
| C48P14 | 0.1053  | 48×96  | 135.5(1.6) | 3.81 | 706.55(39) | beta6.20_mu-0.2825_ms-0.2310_L48x96  | beta6.20_mu-0.2825_ms-0.2310_L48x96  |
| C64P14 | 0.1053  | 64×128 | 134.5(1.6) | 4.63 | 706.55(39) | None
| E32P29 | 0.0897  | 32×64  | 286.7(1.8) | 4.19 | 701.37(92) | beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64 | beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64 |
| E32P22 | 0.0897  | 32×64  | 216.1(2.3) | 3.15 | 688.88(70) | None
| F32P30 | 0.07753 | 32×96  | 303.2(1.3) | 3.56 | 675.98(97) | beta6.41_mu-0.2295_ms-0.2050_L32x96  | beta6.41_mu-0.2295_ms-0.2050_L32x96  | 
| F48P30 | 0.07753 | 48×96  | 303.4(0.9) | 5.72 | 674.76(58) |
| F32P21 | 0.07753 | 32×64  | 210.9(2.2) | 2.67 | 658.79(94) |
| F48P21 | 0.07753 | 48×96  | 207.2(1.1) | 3.91 | 661.94(64) |
| F64P13 | 0.07753 | 64×128 | 134.1(1.5) | 3.37 | 681.48(59) |
| F64P14 | 0.07753 | 64×128 | 135.7(1.2) | 3.41 | 681.57(50) |
| G36P29 | 0.06887 | 36×108 | 297.2(0.9) | 3.73 | 693.05(46) |
| G32P35 | 0.06887 | 32×96  | 352.2(2.6) | 3.94 | 707.7(1.8) |
| H48P32 | 0.05199 | 48×144 | 317.2(0.9) | 3.99 | 691.88(65) |
| I64P30 | 0.03761 | 64×128 | 312.2(1.6) | 3.81 | 671.4(1.3) |

**Brief description of each column:**

- **Conf**: Label for each lattice QCD simulation ensemble (prefix letters C–I correspond to different lattice spacings, 24, 32... means the lattice space size, P means the pion mass of this conf)
- **a (fm)**: Lattice spacing in femtometers (fm); shared within each group of the same spacing
- **n³_L × n_T**: Spatial lattice size cubed multiplied by the temporal lattice size (i.e., lattice volume)
- **m_π (MeV)**: Pion mass in MeV; values in parentheses denote statistical errors
- **m_π L**: Dimensionless product of the pion mass and the spatial lattice extent L, used to assess finite-volume effects
- **m_ηs (MeV)**: ηs meson mass in MeV; values in parentheses denote statistical errors
- **path**: file location