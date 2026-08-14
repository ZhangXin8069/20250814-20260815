# AGENTS.md — examples/sush/lqcddb

`lqcddb` 是格点 QCD 蒸馏关联函数 Python 包——Wick 收缩、特征向量/传播子代数、统计分析。是扁平 `function_contraction/` 模块的积极开发后继者。`pip install -e .` 安装，提供懒加载 `from lqcddb import *` API，numpy/cupy 后端切换。

## 安装

```bash
pip install -e .            # numpy, scipy, opt_einsum, matplotlib
pip install -e ".[gpu]" / ".[mpi]" / ".[all]"
```

需 Python ≥ 3.10。

## 架构

- **懒加载 `__init__.py`**：`__getattr__` + `_ATTR_TO_MODULE` 映射（行 27–125），首次访问属性时才 import 对应模块，避免加载重依赖（mpi4py/cupy/sympy）
- 子包：`base/`（后端、缓存 einsum、Levi-Civita、CG 系数、MPI、Stout 涂抹）、`constant/`（常量、DR γ、Pauli σ）、`contraction/`（wick 引擎、算符共轭、顺序传播子、动态收缩注册表）、`eigvectors/`（vector_creator、vertex_creator）、`analyse/`（Jackknife/Bootstrap/meff/GEVP/3pt-2pt 比值）、`io/`（二进制读写）、`test/`（独立测试脚本）

## 约定

- 收缩统一走 `cached_contract`（勿直接裸调 einsum）
- 动量顺序 `[pz, py, px]`（z 最快）；特征向量 `(Nev, Nz, Ny, Nx, Nc)`
- γ 矩阵为 DeGrand-Rossi（DR，手征变体）基
- 详细 API 参考：`README.md`（~1400 行）
