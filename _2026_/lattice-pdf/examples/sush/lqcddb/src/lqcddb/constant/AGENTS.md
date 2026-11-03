# AGENTS.md — examples/sush/lqcddb/src/lqcddb/constant

lqcddb 的物理常量与自旋矩阵：DeGrand-Rossi（手征变体）基 Dirac γ 矩阵、Pauli σ 矩阵、格点/QCD 常量。

## 文件

`constant.py`（Nc=3、Ns=4、Nd=4、fm2GeV=0.197）、`gamma_matrix.py`（`gamma(i)` 18 种 DR γ 类型 + `tran_indx_to_gamma`、`PFF_Mom_to_gamma_new`）、`sigma_matrix.py`（`sigma(i)` Pauli + `Mom_times_sigma`、`Mom_cross_sigma`）、`__init__.py`/`__init__.pyi`。

## 用法

```python
from lqcddb import gamma, sigma, Nc, Ns, Nd, fm2GeV
g5 = gamma(5)   # 4×4 DR 基 gamma
```

DR（手征变体）基为整个 lqcddb/function_contraction 工具包通用约定。
