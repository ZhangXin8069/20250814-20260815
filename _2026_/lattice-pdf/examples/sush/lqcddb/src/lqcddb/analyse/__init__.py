"""
lqcddb.analyse — 统计分析：Jackknife/Bootstrap、有效质量、GEVP、ratio_3pt。

按模块级懒加载。scipy 仅在 solve_gevp 内部按需导入。
"""
_ATTR_TO_MODULE: dict[str, str] = {}

for _name in (
    "Mom2GeV",
    "loop_tsrc",
    "sum_over_array_of_list",
    "mean_over_array_of_list",
    "Jackknife",
    "Bootstrap",
    "meff",
    "ratio_3pt",
    "solve_gevp",
    "dis_connect",
    "plot_analyse_marker",
    "plot_analyse_color",
):
    _ATTR_TO_MODULE[_name] = ".analyse"

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
