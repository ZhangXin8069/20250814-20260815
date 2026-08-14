# AGENTS.md — examples/sush/lqcddb/src/lqcddb

lqcddb Python 包根——格点 QCD 蒸馏关联函数工具包（Wick 收缩、特征向量/传播子代数、统计分析）。提供懒加载 `from lqcddb import *` API，numpy/cupy 后端切换。

## 子包

| 子包 | 用途 |
|---|---|
| `base/` | 后端切换、缓存 einsum、Levi-Civita、SU(2) CG 系数、MPI 初始化、Stout 涂抹 |
| `constant/` | 物理常量（Nc/Ns/Nd/fm2GeV）、DR γ 矩阵、Pauli σ |
| `contraction/` | 核心 Wick 引擎（wick_contraction）、算符共轭、顺序传播子、dynamic_contraction 注册表、带宽顾问 |
| `eigvectors/` | vector_creator（特征向量代数/压缩）、vertex_creator（VdV/VVV/相位/Ω 顶点） |
| `analyse/` | Jackknife、Bootstrap、有效质量、GEVP、3pt/2pt 比值、源平均 |
| `io/` | 特征向量与传播子的二进制/ASCII 读写 |
| `test/` | 独立测试脚本（无 pytest）与生成图 |

## 关键文件

`__init__.py`（懒加载 API，`_ATTR_TO_MODULE` 映射行 27–125）、`__init__.pyi`（类型桩）。完整用法见父目录 `../CLAUDE.md`（包级文档，已归档）与 `README.md`。
