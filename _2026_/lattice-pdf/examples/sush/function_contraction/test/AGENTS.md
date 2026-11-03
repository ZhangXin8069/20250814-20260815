# AGENTS.md — examples/sush/function_contraction/test

`function_contraction` 模块的测试脚本与可视化。**全部为独立 Python 脚本**（无 pytest/unittest），输出结果与图供人工核验。

## 关键规则

这是 `function_contraction/` 下**唯一允许修改的目录**。

## 测试文件

| 文件 | 用途 | 依赖 |
|---|---|---|
| `contraction.pi.py` | Pion 2pt 收缩测试（串行 CPU） | numpy, opt_einsum |
| `contraction.pi.mpi.py` / `contraction.proton.mpi.py` | MPI 并行测试 | mpi4py |
| `plot_wick.py` | Wick 图可视化 | matplotlib |
| `test.cache.py`、`baroperator.py`、`contraction.sh` | 缓存 einsum / 算符共轭 / 提交 | — |

## 图输出（figure/）

`wick_contraction_diagram*.pdf`（一般情形、Kπ、ππ 2pt/3pt、PP 3pt 等）。
