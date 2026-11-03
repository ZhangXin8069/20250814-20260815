# AGENTS.md — examples/sush/lqcddb/src/lqcddb/base

lqcddb 基础层：后端抽象、缓存 einsum、基础张量工具、SU(2) 代数、MPI 基础设施、规范场涂抹。几乎所有其他子包都从这里导入。

## 文件

| 文件 | 用途 |
|---|---|
| `backend.py` | 全局 numpy/cupy 后端状态——`set_backend('numpy'\|'cupy')`、`get_backend()`。**非线程安全，临时切换后须恢复** |
| `base_functions.py` | `levi_civita_tensor`、`creat_mom_list`、`ArraySlicer`、`cached_contract`（LRU 缓存 opt_einsum.contract，回退 numpy.einsum）、`clear_cache` |
| `cg_coeff.py` | SU(2) Clebsch-Gordan 系数——`SU2combine`/`SU2decompose` |
| `mpi_init.py` | MPI 初始化与时间片分布——`mpinit`、`get_mpi_data` 等。**需 mpi4py**（懒加载） |
| `smear_gauge.py` | `stout_smear_ndarray`——Stout 规范链接涂抹。**需 opt_einsum** |

## 约定

- `cached_contract` 是整个 lqcddb 的标准收缩入口（勿直接裸调 einsum/opt_einsum.contract）
- MPI 函数仅当 mpi4py 安装时才可导入
