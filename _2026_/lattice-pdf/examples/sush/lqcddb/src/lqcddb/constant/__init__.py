"""
lqcddb.constant — 物理常数、gamma矩阵、Pauli矩阵。

按模块级懒加载。无外部重依赖。
"""
_ATTR_TO_MODULE: dict[str, str] = {}

for _name in ("Nc", "Ns", "Nd", "fm2GeV"):
    _ATTR_TO_MODULE[_name] = ".constant"
for _name in ("gamma", "tran_indx_to_gamma", "PFF_Mom_to_gamma_new"):
    _ATTR_TO_MODULE[_name] = ".gamma_matrix"
for _name in ("sigma", "Mom_times_sigma", "Mom_cross_sigma"):
    _ATTR_TO_MODULE[_name] = ".sigma_matrix"

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
