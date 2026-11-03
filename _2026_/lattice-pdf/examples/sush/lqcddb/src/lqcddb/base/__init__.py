"""
lqcddb.base — 后端管理、张量工具、SU(2)代数、MPI通信、规范场平滑。

按模块级懒加载：首次访问某函数时仅导入其所在模块，
避免 mpi4py / opt_einsum 等重依赖被无条件加载。
"""
_ATTR_TO_MODULE: dict[str, str] = {}

# 轻量模块（无外部重依赖）
for _name in ("get_backend", "set_backend"):
    _ATTR_TO_MODULE[_name] = ".backend"
for _name in (
    "levi_civita_tensor",
    "creat_mom_list",
    "ArraySlicer",
    "cached_contract",
    "clear_cache",
):
    _ATTR_TO_MODULE[_name] = ".base_functions"
for _name in ("SU2combine", "SU2decompose"):
    _ATTR_TO_MODULE[_name] = ".cg_coeff"

# MPI 模块 — 仅在首次访问 MPI 函数时加载 mpi4py
for _name in (
    "get_mpi_data",
    "get_mpi_tlist",
    "mpinit",
    "getMPIComm",
    "getMPIRank",
    "getMPISize",
):
    _ATTR_TO_MODULE[_name] = ".mpi_init"

# 规范场平滑 — 仅在首次访问时加载 opt_einsum
for _name in ("stout_smear_ndarray",):
    _ATTR_TO_MODULE[_name] = ".smear_gauge"

__all__ = sorted(_ATTR_TO_MODULE.keys())
_MODULES: dict = {}

del _name


def __getattr__(name: str):
    if name in _ATTR_TO_MODULE:
        mod_path = _ATTR_TO_MODULE[name]
        if mod_path not in _MODULES:
            from importlib import import_module as _import

            _MODULES[mod_path] = _import(mod_path, __name__)
        return getattr(_MODULES[mod_path], name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
