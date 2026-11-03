"""
Lattice QCD Distillation Correlation Function Toolkit
======================================================

面向格点 QCD 蒸馏 (distillation/blending) 方法的关联函数计算与统计分析包。

提供 Wick 收缩自动生成、本征矢量压缩与顶点函数、perambulator 操作、
gamma/sigma 矩阵、SU(2) Clebsch-Gordan 系数、规范场 Stout 平滑、
MPI 并行数据搬运，以及 Jackknife/Bootstrap 重采样、有效质量提取、
GEVP 求解等完整分析流程。

用法::

    from lqcddb import ...
    set_backend('numpy')    # 或 'cupy'
    backend = get_backend()
    
"""
__version__ = "0.0.2"

# ── 每个公开名 → 所在源模块的映射 ──────────────────────────────
# 格式: {attribute_name: "相对包路径.具体模块"}
# __getattr__ 仅 import_module 该模块，不加载整个子包。
_ATTR_TO_MODULE: dict[str, str] = {}

# ---- base subpackage ----
_base_light = (
    # backend.py — 无外部重依赖
    "get_backend",
    "set_backend",
)
_base_functions = (
    # base_functions.py — 仅依赖 numpy
    "levi_civita_tensor",
    "creat_mom_list",
    "ArraySlicer",
    "cached_contract",
    "clear_cache",
    "get_cache_keys",
)
_base_cg = (
    # cg_coeff.py — 纯 Python 数学
    "SU2combine",
    "SU2decompose",
)
_base_mpi = (
    # mpi_init.py — **需要 mpi4py**
    "get_mpi_data",
    "get_mpi_tlist",
    "mpinit",
    "getMPIComm",
    "getMPIRank",
    "getMPISize",
)
_base_smear = (
    # smear_gauge.py — **需要 opt_einsum**
    "stout_smear_ndarray",
)
for _name in _base_light:
    _ATTR_TO_MODULE[_name] = ".base.backend"
for _name in _base_functions:
    _ATTR_TO_MODULE[_name] = ".base.base_functions"
for _name in _base_cg:
    _ATTR_TO_MODULE[_name] = ".base.cg_coeff"
for _name in _base_mpi:
    _ATTR_TO_MODULE[_name] = ".base.mpi_init"
for _name in _base_smear:
    _ATTR_TO_MODULE[_name] = ".base.smear_gauge"

# ---- constant subpackage ----
for _name in ("Nc", "Ns", "Nd", "fm2GeV"):
    _ATTR_TO_MODULE[_name] = ".constant.constant"
for _name in ("gamma", "tran_indx_to_gamma", "PFF_Mom_to_gamma_new"):
    _ATTR_TO_MODULE[_name] = ".constant.gamma_matrix"
for _name in ("sigma", "Mom_times_sigma", "Mom_cross_sigma"):
    _ATTR_TO_MODULE[_name] = ".constant.sigma_matrix"

# ---- contraction subpackage ----
for _name in ("conjugate_operator",):
    _ATTR_TO_MODULE[_name] = ".contraction.baroperator"
for _name in (
    "wick_contraction",
    "plot_figure_wick",
    "identify_equivalent_diagrams",
):
    _ATTR_TO_MODULE[_name] = ".contraction.autowick"
for _name in ("seq_peram",):
    _ATTR_TO_MODULE[_name] = ".contraction.seqperam"
for _name in ("analyze_bandwidth", "printGPUinfo"):
    _ATTR_TO_MODULE[_name] = ".contraction.contractadviser"
for _name in (
    "PeramRegistry",
    "VRegistry",
    "GammaRegistry",
    "dynamic_contraction",
    "run_wick_analysis",
    "clear_plan_cache",
    "calculate_contraction",
):
    _ATTR_TO_MODULE[_name] = ".contraction.dynamic"

# ---- analyse subpackage ----
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
    _ATTR_TO_MODULE[_name] = ".analyse.analyse"

# ---- eigvectors subpackage ----
_ATTR_TO_MODULE["vector_creator"] = ".eigvectors.vector"
_ATTR_TO_MODULE["vertex_creator"] = ".eigvectors.vertex"

# ---- io subpackage ----
for _name in ("write_data_ascii", "check_dir_path", "safe_save"):
    _ATTR_TO_MODULE[_name] = ".io.write_date"

# ── 公共 API 列表 ────────────────────────────────────────────
__all__ = sorted(_ATTR_TO_MODULE.keys())

# ── 模块缓存 ──────────────────────────────────────────────────
_MODULES: dict = {}

# 清理循环变量
del _name, _base_light, _base_functions, _base_cg, _base_mpi, _base_smear


def __getattr__(name: str):
    """按需加载：仅导入目标属性所在的源模块。

    与之前按子包加载不同，本实现直接 ``import_module`` 到具体模块
    （如 ``.analyse.analyse`` 而非 ``.analyse``），避免加载同一子包
    中不需要的模块及其外部依赖。
    """
    if name in _ATTR_TO_MODULE:
        mod_path = _ATTR_TO_MODULE[name]
        if mod_path not in _MODULES:
            from importlib import import_module

            _MODULES[mod_path] = import_module(mod_path, __name__)
        return getattr(_MODULES[mod_path], name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def __dir__():
    """返回公共名称列表，支持 tab 补全。"""
    return sorted(__all__)
