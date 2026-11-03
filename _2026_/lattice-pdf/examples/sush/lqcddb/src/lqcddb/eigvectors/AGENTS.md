# AGENTS.md — examples/sush/lqcddb/src/lqcddb/eigvectors

蒸馏法特征向量与顶点函数代数：Laplacian 特征向量的操作/压缩（`vector_creator`）与 VdV/VVV/相位/Ω 顶点构造（`vertex_creator`）。

## 文件

`vector.py`（`vector_creator`：内积、正交性检查/归一化、Gram-Schmidt、随机噪声、4 种压缩方案 `compress_matrix_V1..V4`）、`vertex.py`（`vertex_creator`：动量相位 `phase_exp_2pt/3pt`、`Mom_VdV_sink_t`、`Mom_VVV_sink_t`、规范链接 VdV、Ω 蒸馏权重 `create_omega_accelerate`、MPI 源/汇传输）、`__init__.py`/`__init__.pyi`。

## 用法

```python
from lqcddb import vector_creator, vertex_creator
vc = vector_creator()
vertex = vertex_creator(Nx=32)
```

## 约定

- 动量顺序 `[pz, py, px]`（z 最快）；特征向量 `(Nev, Nz, Ny, Nx, Nc)`；顶点经当前后端 `cached_contract` 构建
