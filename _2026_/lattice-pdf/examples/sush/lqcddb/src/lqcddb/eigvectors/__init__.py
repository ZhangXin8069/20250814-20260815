"""
lqcddb.eigvectors — 本征矢量压缩 (vector_creator) 与顶点函数 (vertex_creator)。

按模块级懒加载。opt_einsum 仅在首次访问时导入。
"""
_ATTR_TO_MODULE: dict[str, str] = {}

_ATTR_TO_MODULE["vector_creator"] = ".vector"
_ATTR_TO_MODULE["vertex_creator"] = ".vertex"

__all__ = sorted(_ATTR_TO_MODULE.keys())
_MODULES: dict = {}


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
