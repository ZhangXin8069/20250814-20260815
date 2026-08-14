# AGENTS.md — examples/sush

苏 SH 的蒸馏收缩框架——lattice-pdf 项目最全面的 Wick 收缩与关联函数分析工具包。用蒸馏方法从传播子和特征向量计算任意强子关联函数（介子、重子、多强子），支持 CPU（numpy）/GPU（cupy）后端与 MPI 并行。

## 两套并行代码

| 代码库 | 位置 | 状态 |
|---|---|---|
| `function_contraction/` | 顶层扁平模块 | 原始——经 `sys.path.insert(0, '/public/home/sush/distillation/')` 导入 |
| `lqcddb/` | 正式包（pyproject.toml） | **积极开发**——`pip install -e .`，懒加载 `__init__.py` |

两者暴露相同核心 API（`wick_contraction`、`Jackknife`、`meff`、`set_backend` 等）。**新工作一律优先 `lqcddb`**。

## 安装 lqcddb

```bash
cd lqcddb
pip install -e .            # 基础：numpy, scipy, opt_einsum, matplotlib
pip install -e ".[gpu]"     # + cupy
pip install -e ".[mpi]"     # + mpi4py
```

## 关键规则

**只允许修改 `test/` 目录下的文件**。新脚本、图、输出都放 `test/` 下。
