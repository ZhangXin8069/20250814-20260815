# AGENTS.md — examples/sush/lqcddb/src/lqcddb/test

lqcddb 独立测试脚本（**无 pytest/unittest**——每个脚本输出结果/图供人工核验）+ 生成的 Wick 图。按 function_contraction 规则，新脚本/图/输出放 `test/`。

## 文件

`contraction.pi.py`（pion 2pt，无需 MPI/GPU）、`contraction.pi.mpi.py`、`contraction.proton.mpi.py`、`contraction.proton.mpi.dynamic.py`（dynamic_contraction）、`contraction.lambda.proton.weak.mpi.py`、`contraction.sh`、`baroperator.py`、`test.cache.py`（cached_contract）、`test.dynamic.py`、`identify_equivalent_diagrams_test.py`、`plot_wick.py`、`fit.py`、`figure/`（生成的 Wick 图 PDF）。
