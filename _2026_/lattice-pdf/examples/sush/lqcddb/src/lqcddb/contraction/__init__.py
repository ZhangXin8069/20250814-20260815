"""
lqcddb.contraction — Wick收缩、算符共轭、带宽分析、顺序传播子。

按模块级懒加载。
"""
_ATTR_TO_MODULE: dict[str, str] = {}

for _name in ("conjugate_operator",):
    _ATTR_TO_MODULE[_name] = ".baroperator"
for _name in (
    "wick_contraction",
    "plot_figure_wick",
    "identify_equivalent_diagrams",
):
    _ATTR_TO_MODULE[_name] = ".autowick"
for _name in ("seq_peram",):
    _ATTR_TO_MODULE[_name] = ".seqperam"
# contractadviser — 仅在首次访问时加载 opt_einsum
for _name in ("analyze_bandwidth", "printGPUinfo"):
    _ATTR_TO_MODULE[_name] = ".contractadviser"
for _name in (
    "PeramRegistry",
    "VRegistry",
    "GammaRegistry",
    "dynamic_contraction",
    "run_wick_analysis",
    "clear_plan_cache",
    "calculate_contraction",
):
    _ATTR_TO_MODULE[_name] = ".dynamic"

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
