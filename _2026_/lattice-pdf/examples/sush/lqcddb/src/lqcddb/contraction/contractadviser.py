"""
带宽感知的张量收缩分析器 (Bandwidth-Aware Contraction Analyzer)

分析 opt_einsum 风格的张量收缩操作，判断是否存在带宽瓶颈，
并建议切分哪些自由指标来减少数据读入压力。

核心思路：
1. 解析 einsum 下标，找出自由指标和收缩指标
2. 计算算术强度 (Arithmetic Intensity = FLOPs / Bytes)
3. 用 Roofline 模型判断是否带宽瓶颈
4. 对于带宽瓶颈，分析每个自由指标切分后的收益
5. 输出建议切分的指标及推荐切片大小

不执行实际计算，仅做分析。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# 硬件模型 (可扩展为 JSON 配置文件)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HardwareSpec:
    """硬件规格：存储各精度峰值，根据 dtype 自动选用"""
    name: str
    peak_fp16: float = 0.0            # FP16 Tensor Core 峰值 (TFLOPS)
    peak_fp32: float = 0.0            # FP32 峰值 (TFLOPS), 也用于 complex64
    peak_fp64: float = 0.0            # FP64 峰值 (TFLOPS), 也用于 complex128
    memory_bandwidth_gbs: float = 0.0 # 显存/内存带宽 (GB/s)
    total_memory_bytes: int = 0       # 总显存/内存大小 (bytes)
    l2_cache_bytes: int = 0           # L2 缓存大小 (bytes)
    compute_efficiency: float = 1.0   # 实际可达算力比例 (1.0=理论峰值), 用于 real 类型
    memory_efficiency: float = 1.0    # 实际可达带宽比例 (1.0=理论峰值)
    complex_compute_eff: float = 1.0  # complex 相对 real 的计算效率 (CPU ~0.5, GPU ~1.0)

    def peak_compute(self, dtype: str) -> float:
        """根据 dtype 返回对应的峰值算力 (TFLOPS)"""
        compute_class = _DTYPE_TABLE[dtype]['compute_class']
        return getattr(self, f'peak_{compute_class}')

    def effective_compute(self, dtype: str) -> float:
        """实际可达算力 (TFLOPS), 计入效率因子和 complex 类型修正"""
        base_eff = self.peak_compute(dtype) * self.compute_efficiency
        if 'complex' in dtype:
            base_eff *= self.complex_compute_eff
        return base_eff

    def effective_bandwidth(self) -> float:
        """实际可达带宽 (GB/s), 计入效率因子"""
        return self.memory_bandwidth_gbs * self.memory_efficiency

    def flops_per_byte(self, dtype: str) -> float:
        """Roofline 拐点 (FLOPs/byte)，根据 dtype 自动选择峰值算力"""
        peak = self.peak_compute(dtype)
        if self.memory_bandwidth_gbs == 0 or peak == 0:
            return float('inf')
        return (peak * 1e12) / (self.memory_bandwidth_gbs * 1e9)

    def effective_flops_per_byte(self, dtype: str) -> float:
        """有效 Roofline 拐点，使用实际可达算力和带宽"""
        eff_peak = self.effective_compute(dtype)
        eff_bw = self.effective_bandwidth()
        if eff_bw == 0 or eff_peak == 0:
            return float('inf')
        return (eff_peak * 1e12) / (eff_bw * 1e9)


# ═══════════════════════════════════════════════════════════════════════════════
# 硬件数据库
# ═══════════════════════════════════════════════════════════════════════════════

_GB = 1024 * 1024 * 1024
_MB = 1024 * 1024

HARDWARE_PRESETS: Dict[str, HardwareSpec] = {

    # V100 32GB — 22 节点 (20×8GPU + 2×16GPU)
    "V100": HardwareSpec(
        name="NVIDIA V100 32GB",
        peak_fp16=125.0, peak_fp32=125.0, peak_fp64=7.8,
        memory_bandwidth_gbs=900.0,
        total_memory_bytes=32 * _GB, l2_cache_bytes=6 * _MB,
        compute_efficiency=2, memory_efficiency=1,
    ),

    # A100 40GB — 1 节点 × 2GPU
    "A100_40GB": HardwareSpec(
        name="NVIDIA A100 40GB",
        peak_fp16=624.0, peak_fp32=312.0, peak_fp64=9.7,
        memory_bandwidth_gbs=1555.0,
        total_memory_bytes=40 * _GB, l2_cache_bytes=40 * _MB,
        compute_efficiency=2.5, memory_efficiency=1,
    ),

    # A100 80GB — 7 节点 × 8GPU
    "A100_80GB": HardwareSpec(
        name="NVIDIA A100 80GB",
        peak_fp16=624.0, peak_fp32=312.0, peak_fp64=9.7,
        memory_bandwidth_gbs=2039.0,
        total_memory_bytes=80 * _GB, l2_cache_bytes=40 * _MB,
        compute_efficiency=2.5, memory_efficiency=1,
    ),

    # A800 80GB — 6 节点 × 8GPU (中国特供版 A100, 仅 NVLink 受限)
    "A800": HardwareSpec(
        name="NVIDIA A800 80GB",
        peak_fp16=624.0, peak_fp32=312.0, peak_fp64=9.7,
        memory_bandwidth_gbs=2039.0,
        total_memory_bytes=80 * _GB, l2_cache_bytes=40 * _MB,
        compute_efficiency=2.5, memory_efficiency=1,
    ),

    # H20 96GB — 6 节点 × 8GPU (中国特供版 Hopper)
    "H20": HardwareSpec(
        name="NVIDIA H20 96GB",
        peak_fp16=148.0, peak_fp32=44.0, peak_fp64=1.0,
        memory_bandwidth_gbs=4000.0,
        total_memory_bytes=96 * _GB, l2_cache_bytes=60 * _MB,
        compute_efficiency=1, memory_efficiency=1,
    ),

    # 通用 CPU 参考
    "I72C512G": HardwareSpec(
        name="Typical Server CPU (DDR4)",
        peak_fp16=0.1728, peak_fp32=0.1728, peak_fp64=0.0864,
        memory_bandwidth_gbs=204/72,
        total_memory_bytes=512 * _GB / 72, l2_cache_bytes=1280 * 1024,
        compute_efficiency=0.25, memory_efficiency=1,
    ),
    
    "CPU6248R": HardwareSpec(
        name="Typical Server CPU (DDR4)",
        peak_fp16=0.224, peak_fp32=0.224, peak_fp64=0.112,
        memory_bandwidth_gbs=280/48,
        total_memory_bytes=400 * _GB / 48, l2_cache_bytes=1024 * 1024,
        compute_efficiency=0.25, memory_efficiency=1,
    ),
    "CPUEICC": HardwareSpec(
        name="Typical Server CPU (DDR4)",
        peak_fp16=0.224, peak_fp32=0.224, peak_fp64=0.112,
        memory_bandwidth_gbs=280/48,
        total_memory_bytes=512 * _GB / 48, l2_cache_bytes=1024 * 1024,
        compute_efficiency=0.25, memory_efficiency=1,
    ),
}

# -- 兼容别名 (指向主预设) --
HARDWARE_PRESETS["A100"] = HARDWARE_PRESETS["A100_80GB"]


# dtype 元数据: bytes_per_element, flops_per_multiply_add, compute_class, 显示名称
_DTYPE_TABLE: Dict[str, dict] = {
    'float16':    {'bytes': 2,  'flops_per_madd': 2,  'compute_class': 'fp16',
                   'name': 'float16'},
    'float32':    {'bytes': 4,  'flops_per_madd': 2,  'compute_class': 'fp32',
                   'name': 'float32'},
    'float64':    {'bytes': 8,  'flops_per_madd': 2,  'compute_class': 'fp64',
                   'name': 'float64'},
    'complex64':  {'bytes': 8,  'flops_per_madd': 4,  'compute_class': 'fp32',
                   'name': 'complex64'},
    'complex128': {'bytes': 16, 'flops_per_madd': 8,  'compute_class': 'fp64',
                   'name': 'complex128'},
}


# ═══════════════════════════════════════════════════════════════════════════════
# 下标解析
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParsedContraction:
    """解析后的收缩操作信息"""
    subscript: str                          # 原始下标，如 'Mabc,Nabc->MN'
    input_subs: List[str]                   # 输入下标列表，如 ['Mabc', 'Nabc']
    output_subs: str                        # 输出下标，如 'MN'
    shapes: List[Tuple[int, ...]]           # 输入张量形状
    index_size: Dict[str, int]              # 每个指标 → 大小
    free_indices: Dict[str, int]            # 自由指标（出现在输出中）
    contracted_indices: Dict[str, int]      # 收缩指标（被求和消去）
    batch_indices: Dict[str, int] = field(default_factory=dict)  # 批处理指标（可选）

    @property
    def n_inputs(self) -> int:
        return len(self.input_subs)

    @property
    def n_free(self) -> int:
        return len(self.free_indices)

    @property
    def n_contracted(self) -> int:
        return len(self.contracted_indices)


def parse_subscript(subscript: str, shapes: List[Tuple[int, ...]]) -> ParsedContraction:
    """
    解析 opt_einsum 风格的下标字符串。

    支持的格式：
        - 显式模式: 'Mabc,Nabc->MN'
        - 隐式模式 (无 ->): 'ij,jk'  等价于 'ij,jk->ik'
        - 省略号: '...ij,...jk->...ik'

    Returns:
        ParsedContraction 包含解析后的结构信息
    """
    # 分离输入和输出
    if '->' in subscript:
        inputs_part, output_part = subscript.split('->')
    else:
        # 隐式模式：自动推断输出
        inputs_part = subscript
        output_part = None

    input_subs = [s.strip() for s in inputs_part.split(',')]

    if len(input_subs) != len(shapes):
        raise ValueError(
            f"输入数量不匹配: 下标有 {len(input_subs)} 个输入, "
            f"但提供了 {len(shapes)} 个形状"
        )

    # 构建指标 → 大小的映射
    index_size: Dict[str, int] = {}

    for sub, shape in zip(input_subs, shapes):
        # 处理省略号
        if '...' in sub:
            n_ellipsis = len(shape) - (len(sub) - 3)
            sub_expanded = sub.replace('...', '⋯' * n_ellipsis)  # 用特殊字符暂代
        else:
            sub_expanded = sub
            if len(sub_expanded) != len(shape):
                raise ValueError(
                    f"指标数量与形状维度不匹配: '{sub}' 有 {len(sub)} 个指标, "
                    f"但形状是 {shape} ({len(shape)} 维)"
                )

        for idx_char, size in zip(sub_expanded, shape):
            if idx_char == '⋯':
                continue  # 省略号中的维度暂不命名
            if idx_char in index_size:
                if index_size[idx_char] != size:
                    raise ValueError(
                        f"指标 '{idx_char}' 大小不一致: "
                        f"{index_size[idx_char]} vs {size}"
                    )
            else:
                index_size[idx_char] = size

    # 确定输出下标
    if output_part is None:
        # 隐式模式：输出 = 只出现一次的指标，按字母顺序排列
        from collections import Counter
        all_indices = ''.join(input_subs)
        counter = Counter(all_indices.replace(',', '').replace(' ', ''))
        output_part = ''.join(sorted(
            c for c, count in counter.items() if count == 1
        ))
        if not output_part:
            raise ValueError(
                f"无法自动推断输出指标: 所有指标都出现了多次。"
                f"请使用 '->' 显式指定输出。"
            )

    # 分类指标
    all_input_indices = set()
    for sub in input_subs:
        all_input_indices.update(sub)

    output_indices = set(output_part)

    free_indices = {}
    contracted_indices = {}
    batch_indices = {}

    for idx, size in index_size.items():
        if idx in output_indices:
            free_indices[idx] = size
        else:
            contracted_indices[idx] = size

    return ParsedContraction(
        subscript=subscript,
        input_subs=input_subs,
        output_subs=output_part,
        shapes=shapes,
        index_size=index_size,
        free_indices=free_indices,
        contracted_indices=contracted_indices,
        batch_indices=batch_indices,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 算术强度分析
# ═══════════════════════════════════════════════════════════════════════════════

# 缓存抖动惩罚指数: 当工作集超出 L2 缓存时，实际数据读取 ≈ 理想值 × (工作集/缓存)^指数
# - 0.0: 完全忽略缓存效应 (理想模型)
# - 0.25: 温和惩罚 (单步收缩)
# - 0.40: 中等的 (多步收缩, 中间张量叠加效应)
_CACHE_PENALTY_EXPONENT_SIMPLE = 0.25    # 2-输入
_CACHE_PENALTY_EXPONENT_MULTI = 0.40     # 多输入 (中间张量在每步都产生缓存压力)

# 显存压力阈值: 当最大中间张量超过此比例的显存时，强制建议切分
_MEMORY_PRESSURE_THRESHOLD = 0.4  # 40% 显存

# L2 溢出阈值: 当峰值工作集超过 L2 缓存的此倍数时, 强制建议切分
# 即使 roofline 说 compute-bound, 巨大的中间张量也会导致严重的缓存抖动
_L2_OVERFLOW_THRESHOLD = 100  # 工作集/L2 > 100x → 强制建议切分

# 最小自由指标大小: 当自由指标过小时, GPU 缺乏并行度, 即使 AI 够高也应建议增大
_MIN_FREE_INDEX_FOR_GPU = 4  # 低于此值 → GPU SM 吃不饱
# 切太细 (几百块) → kernel launch 主导耗时 → 反而变慢
# 切太粗 (2-3块) → 缓存抖动未解决 → 收益有限
_MAX_SLICING_CHUNKS = 12  # 最多切成此块数 (保证 GPU 每块有足够工作量)


@dataclass
class ContractionCost:
    """收缩操作的成本估算"""
    total_flops: int              # 总浮点运算次数
    total_read_bytes: int         # 理想读取字节数 (每个元素只读一次)
    total_write_bytes: int        # 输出写入字节数
    arithmetic_intensity: float   # 有效算术强度 (计入缓存抖动后的 FLOPs / byte)
    ideal_arithmetic_intensity: float  # 理想算术强度 (忽略缓存抖动)
    output_size: int              # 输出张量元素数
    cache_penalty: float          # 缓存抖动惩罚因子 (1.0 = 无惩罚)
    intermediate_write_bytes: int = 0  # 中间张量写入字节数 (多步收缩中每个中间张量需写出)
    peak_working_bytes: int = 0   # 峰值工作集大小 (bytes), 含中间张量


def _estimate_cost_via_opt_einsum(
    parsed: ParsedContraction, dtype_bytes: int, flops_per_madd: int,
    optimize: str = 'auto',
) -> Tuple[int, int, int, int, int]:
    """
    使用 opt_einsum 获取最优 contraction path, 计算真实的 FLOPs 和数据量。

    对于多输入收缩 (N>2), 简单模型将所有指标一次性缩会产生错误结果。
    实际执行是沿二叉树逐步做两两收缩, 每步产生中间张量。

    Returns:
        (total_flops, total_read_bytes, total_write_bytes,
         peak_working_bytes, intermediate_write_bytes)
        total_read_bytes: 所有步骤中读取的输入+中间张量总和
        intermediate_write_bytes: 中间张量写出字节数 (每个中间张量被写出一次)
        peak_working_bytes: 执行过程中任意时刻的最大工作集
    """
    import opt_einsum as oe

    # 构建 opt_einsum 的输入: 下标和形状
    # opt_einsum 使用类似于 einsum 的格式
    # 我们需要: subscripts, *operands
    # 但我们不需要实际的 tensor, 只需要形状 → 用 views

    # 将下标转为 opt_einsum 格式
    subscript = parsed.subscript
    shapes = parsed.shapes

    # 获取 contraction path
    # opt_einsum 3.x: shapes=True 时直接传 shape 元组, 不能传 numpy 数组
    try:
        _, path_info = oe.contract_path(
            subscript, *shapes,
            shapes=True,
            optimize=optimize,
        )
    except Exception:
        # 如果 opt_einsum 失败, 回退到简单模型
        return _estimate_cost_simple(parsed, dtype_bytes, flops_per_madd)

    # path_info.opt_cost: opt_einsum 的 'FLOPs' = 标量乘法次数
    # 对于实数: 1 MAC = 1 opt_cost → 2 FLOPs → 但这里 opt_cost 就是 MAC 数
    # 实际上 opt_einsum 的 opt_cost 计算的是 "收缩操作数",
    # 对于 ij,jk->ik: opt_cost = i*j*k (即乘加次数)
    # 所以总 FLOPs = opt_cost * flops_per_madd
    total_flops = int(path_info.opt_cost * flops_per_madd)

    # path_info.size_list: 每个中间张量的元素数
    # 数据读取 = 所有输入张量 + 每个中间张量被读取时
    # 简化: 所有输入 + 所有中间输出 (每个中间写入后被后续步骤读取)
    total_input_elements = 0
    for shape in shapes:
        s = 1
        for d in shape:
            s *= d
        total_input_elements += s

    # 每个中间张量的元素数 (path_info.size_list)
    intermediate_elements = sum(path_info.size_list)
    # 最大中间张量
    peak_intermediate = max(path_info.size_list) if path_info.size_list else 0

    # 理想数据读取 = 输入 + 所有中间张量 (每个中间张量被读至少一次)
    # 实际上中间张量可能被多次读取, 但这里取理想最小值
    total_io_elements = total_input_elements + intermediate_elements
    total_read_bytes = total_io_elements * dtype_bytes
    # 输出大小
    output_size = 1
    for size in parsed.free_indices.values():
        output_size *= size
    total_write_bytes = output_size * dtype_bytes

    # 中间张量写出字节数: 每个中间张量被写出一次 (最后输出计入 total_write_bytes)
    # size_list 最后一个元素是最终输出, 不重复计入 intermediate_write_bytes
    if len(path_info.size_list) > 0:
        intermediate_write_elements = intermediate_elements - path_info.size_list[-1]
    else:
        intermediate_write_elements = 0
    intermediate_write_bytes = int(intermediate_write_elements * dtype_bytes)

    # 峰值工作集: max(所有输入, 最大中间张量 + 相关输入)
    # 简化: 总输入大小 (保守估计)
    peak_working_bytes = (total_input_elements + peak_intermediate) * dtype_bytes

    return total_flops, total_read_bytes, total_write_bytes, peak_working_bytes, intermediate_write_bytes


def _estimate_cost_simple(
    parsed: ParsedContraction, dtype_bytes: int, flops_per_madd: int
) -> Tuple[int, int, int, int, int]:
    """简单模型: 适用于 2 输入收缩或作为回退"""
    contracted_size = 1
    for size in parsed.contracted_indices.values():
        contracted_size *= size

    output_size = 1
    for size in parsed.free_indices.values():
        output_size *= size

    total_input_elements = 0
    for shape in parsed.shapes:
        s = 1
        for d in shape:
            s *= d
        total_input_elements += s

    total_flops = flops_per_madd * output_size * contracted_size
    total_read_bytes = total_input_elements * dtype_bytes
    total_write_bytes = output_size * dtype_bytes
    peak_working_bytes = total_read_bytes + total_write_bytes

    # 简单模型 (2 输入): 无中间张量
    intermediate_write_bytes = 0

    return total_flops, total_read_bytes, total_write_bytes, peak_working_bytes, intermediate_write_bytes


def estimate_cost(parsed: ParsedContraction, hw: HardwareSpec = None,
                  dtype_bytes: int = 4, flops_per_madd: int = 2,
                  optimize: str = 'auto') -> ContractionCost:
    """
    估算张量收缩的计算成本和数据搬运成本。

    - 对 2 输入: 使用解析模型 (精确)
    - 对 N>2 输入: 使用 opt_einsum 获取最优路径及其实 FLOPs
    - 缓存抖动: 当峰值工作集 > L2 时, 数据被反复从 HBM 读取
    """
    # 输出大小
    output_size = 1
    for size in parsed.free_indices.values():
        output_size *= size

    # 选择估算方法
    if parsed.n_inputs <= 2:
        total_flops, ideal_read_bytes, total_write_bytes, peak_working_bytes, \
            intermediate_write_bytes = \
            _estimate_cost_simple(parsed, dtype_bytes, flops_per_madd)
    else:
        try:
            total_flops, ideal_read_bytes, total_write_bytes, peak_working_bytes, \
                intermediate_write_bytes = \
                _estimate_cost_via_opt_einsum(parsed, dtype_bytes, flops_per_madd,
                                               optimize=optimize)
        except Exception:
            total_flops, ideal_read_bytes, total_write_bytes, peak_working_bytes, \
                intermediate_write_bytes = \
                _estimate_cost_simple(parsed, dtype_bytes, flops_per_madd)

    # 理想算术强度 (含中间张量写出)
    ideal_data_bytes = ideal_read_bytes + total_write_bytes + intermediate_write_bytes
    ideal_ai = total_flops / ideal_data_bytes if ideal_data_bytes > 0 else float('inf')

    # 缓存抖动惩罚 (基于峰值工作集 vs L2, 多步收缩用更激进指数)
    # 惩罚同时作用于读取和写出: 当工作集 >> 缓存时, 写出也导致 cache miss
    cache_penalty = 1.0
    cache_exponent = (_CACHE_PENALTY_EXPONENT_MULTI if parsed.n_inputs > 2
                      else _CACHE_PENALTY_EXPONENT_SIMPLE)
    if hw is not None and hw.l2_cache_bytes > 0:
        if peak_working_bytes > hw.l2_cache_bytes:
            cache_ratio = peak_working_bytes / hw.l2_cache_bytes
            cache_penalty = cache_ratio ** cache_exponent

    effective_read_bytes = ideal_read_bytes * cache_penalty
    effective_write_bytes = (total_write_bytes + intermediate_write_bytes) * cache_penalty
    effective_data_bytes = effective_read_bytes + effective_write_bytes
    effective_ai = total_flops / effective_data_bytes if effective_data_bytes > 0 else float('inf')

    return ContractionCost(
        total_flops=total_flops,
        total_read_bytes=ideal_read_bytes,
        total_write_bytes=total_write_bytes,
        intermediate_write_bytes=int(intermediate_write_bytes),
        arithmetic_intensity=effective_ai,
        ideal_arithmetic_intensity=ideal_ai,
        output_size=output_size,
        cache_penalty=cache_penalty,
        peak_working_bytes=int(peak_working_bytes),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 瓶颈判断 & 切分建议
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SlicingSuggestion:
    """对某个自由指标进行切分的建议"""
    index: str                          # 建议切分的指标名
    size: int                           # 该指标的总大小
    suggested_chunk_size: int           # 建议的切片大小
    n_chunks: int                       # 切分出的块数
    working_set_per_chunk_bytes: int    # 每块的工作集大小 (bytes)
    peak_memory_without_bytes: int      # 不切分时的峰值显存 (bytes)
    peak_memory_with_bytes: int         # 切分后的峰值显存 (bytes)
    reduction_in_reads: float           # 相比不切分，数据读取减少的倍数
    rationale: str                      # 切分理由


@dataclass
class BandwidthAnalysis:
    """带宽瓶颈分析的完整结果"""
    parsed: ParsedContraction
    cost: ContractionCost
    hardware: HardwareSpec
    is_bandwidth_bound: bool
    bottleneck_severity: str            # 'none' | 'mild' | 'severe'
    suggestions: List[SlicingSuggestion]
    upsizing_suggestions: List[UpsizingSuggestion] = field(default_factory=list)
    summary: str = ""
    estimated_compute_time: float = 0.0     # 预计纯计算时间 (秒)
    estimated_data_time: float = 0.0        # 预计数据读入时间 (秒)
    estimated_total_time: float = 0.0       # 预计总收缩耗时 (秒)


# ═══════════════════════════════════════════════════════════════════════════════
# 增加维度建议 (解决计算资源利用不足)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UpsizingSuggestion:
    """增加某个指标维度的建议，提升算术强度以接近 Roofline 阈值"""
    index: str                    # 建议增加维度的指标名
    index_type: str               # 'free' | 'contracted'
    current_size: int             # 当前大小
    target_size: int              # 目标大小
    scale_factor: float           # 需要的缩放倍数
    current_ai: float             # 当前有效算术强度
    target_ai: float              # 目标算术强度
    data_increase_bytes: int      # 增加的数据量
    rationale: str                # 理由


def _compute_data_for_index(parsed: ParsedContraction, index: str,
                             dtype_bytes: int) -> Tuple[int, int]:
    """计算包含某指标的数据量 (input_bytes, output_bytes_if_free)"""
    input_bytes = 0
    for sub, shape in zip(parsed.input_subs, parsed.shapes):
        if index in sub:
            s = 1
            for d in shape:
                s *= d
            input_bytes += s * dtype_bytes

    output_bytes = 0
    if index in parsed.free_indices:
        output_elements = 1
        for oi, osize in parsed.free_indices.items():
            if oi != index:
                output_elements *= osize
        output_bytes = output_elements * dtype_bytes

    return input_bytes, output_bytes


def _generate_upsizing_suggestions(
    parsed: ParsedContraction,
    cost: ContractionCost,
    hw: HardwareSpec,
    dtype: str,
    dtype_bytes: int,
) -> List[UpsizingSuggestion]:
    """
    当计算资源未被充分利用 (bandwidth-bound) 时，建议增加维度以提高算术强度。

    原理:
        - 增加收缩指标 → FLOPs 增长 (×f), 但只有涉及该指标的输入张量增大
        - 增加自由指标 → FLOPs 增长 (×f), 但涉及该指标的输入+输出都增大
        - 收缩指标的"性价比"更好 (更少的额外数据换取更多的 FLOPs)

    对每个指标求解: 需要放大多少倍 (f) 才能使 AI 达到目标阈值
        AI_new = FLOPs*f / (Data + data_i*(f-1)) ≥ target_AI
        → f ≥ target_AI*(Data - data_i) / (FLOPs - target_AI*data_i)
    """
    suggestions: List[UpsizingSuggestion] = []

    target_ai = hw.effective_flops_per_byte(dtype) * 1.2  # 20% 余量
    need_roofline_upsizing = cost.arithmetic_intensity < target_ai

    # 检查是否有自由指标太小导致 GPU 并行度不足
    tiny_free_indices = {
        idx: size for idx, size in parsed.free_indices.items()
        if size < _MIN_FREE_INDEX_FOR_GPU
    }

    if not need_roofline_upsizing and not tiny_free_indices:
        return suggestions  # 已经足够, 无需增加

    total_flops = cost.total_flops
    total_data = cost.total_read_bytes + cost.total_write_bytes

    # 遍历所有指标 (自由 + 收缩)
    all_indices = {**parsed.free_indices, **parsed.contracted_indices}
    suggested_indices: set = set()  # 避免重复建议

    # ---- Pass 1: roofline 驱动的 upsizing (AI 低于阈值时) ----
    if need_roofline_upsizing:
        for idx, size in all_indices.items():
            data_input, data_output = _compute_data_for_index(parsed, idx, dtype_bytes)
            data_with_idx = data_input + data_output

            denom = total_flops - target_ai * data_with_idx
            if denom <= 0:
                continue

            f = target_ai * (total_data - data_with_idx) / denom
            if f <= 1.01:
                continue

            target_size = max(size + 1, int(size * f + 0.5))
            idx_type = 'free' if idx in parsed.free_indices else 'contracted'
            data_increase = data_with_idx * (f - 1)

            if idx_type == 'contracted':
                rationale = (
                    f"收缩指标 '{idx}' ({size}→{target_size}, ×{f:.1f}) 增大后，"
                    f"FLOPs 增长约 {f:.1f}× 而数据仅增加 "
                    f"{_format_bytes(int(data_increase))}"
                    f"（仅涉及 '{idx}' 的输入变大）。"
                    f"AI 从 {cost.arithmetic_intensity:.1f} → {target_ai:.1f}。"
                )
            else:
                rationale = (
                    f"自由指标 '{idx}' ({size}→{target_size}, ×{f:.1f}) 增大后，"
                    f"FLOPs 增长约 {f:.1f}×，数据增加 "
                    f"{_format_bytes(int(data_increase))}"
                    f"（涉及 '{idx}' 的输入和输出变大）。"
                    f"AI 从 {cost.arithmetic_intensity:.1f} → {target_ai:.1f}。"
                )

            suggestions.append(UpsizingSuggestion(
                index=idx, index_type=idx_type,
                current_size=size, target_size=target_size,
                scale_factor=f,
                current_ai=cost.arithmetic_intensity, target_ai=target_ai,
                data_increase_bytes=int(data_increase),
                rationale=rationale,
            ))
            suggested_indices.add(idx)

    # ---- Pass 2: GPU 并行度驱动的 upsizing (自由指标过小) ----
    for idx, size in tiny_free_indices.items():
        if idx in suggested_indices:
            continue  # Pass 1 已覆盖

        target_size = _MIN_FREE_INDEX_FOR_GPU
        f = target_size / size

        data_input, data_output = _compute_data_for_index(parsed, idx, dtype_bytes)
        data_increase = (data_input + data_output) * (f - 1)

        # 计算增加后的 AI (用于显示)
        new_flops = total_flops * f
        new_data = total_data + data_increase
        new_ai = new_flops / new_data if new_data > 0 else float('inf')

        rationale = (
            f"自由指标 '{idx}' 当前大小={size} < 最小推荐={_MIN_FREE_INDEX_FOR_GPU}。"
            f"过小的维度导致 GPU 缺乏并行度 (SM 空闲)。"
            f"增加到 {target_size} 可使 GPU 利用率显著提升，"
            f"并行度提升 {f:.0f}×。"
            f"AI: {cost.arithmetic_intensity:.0f} → {new_ai:.0f} FLOPs/byte。"
        )

        suggestions.append(UpsizingSuggestion(
            index=idx, index_type='free',
            current_size=size, target_size=target_size,
            scale_factor=f,
            current_ai=cost.arithmetic_intensity, target_ai=new_ai,
            data_increase_bytes=int(data_increase),
            rationale=rationale,
        ))
        suggested_indices.add(idx)

    # 排序: 先按类型 (自由指标优先, 它们直接提升并行度), 再按 scale_factor
    suggestions.sort(key=lambda s: (0 if s.index_type == 'free' else 1, s.scale_factor))
    return suggestions


def _compute_input_size_for_index(parsed: ParsedContraction, index: str,
                                   dtype_bytes: int = 16) -> int:
    """计算包含某个自由指标的输入张量的大小 (bytes)"""
    total = 0
    for sub, shape in zip(parsed.input_subs, parsed.shapes):
        if index in sub:
            size = 1
            for d in shape:
                size *= d
            total += size * dtype_bytes
    return total


def analyze_bandwidth(
    subscript: str,
    shapes: List[Tuple[int, ...]],
    hardware: HardwareSpec | str = "A100_80GB",
    dtype: str = "complex128",
    optimize: str = "auto",
    verbose: bool = True,
) -> BandwidthAnalysis:
    """
    分析一个 opt_einsum 收缩操作的带宽瓶颈，并给出切分建议。

    Args:
        subscript: opt_einsum 风格的下标，如 'Mabc,Nabc->MN'
        shapes:    每个输入张量的形状，如 [(100, 64, 64, 64), (200, 64, 64, 64)]
        hardware:  硬件规格 (HardwareSpec 或预设名称)，默认 'A100_80GB'。
                   dtype 自动选择对应精度峰值 (fp16/fp32/fp64)。
                   (V100, A100_40GB, A100_80GB, A800, H20, I72C512G, CPUEICC, CPU6248R)
        dtype:     数据类型，默认 'complex128'。
        optimize:  opt_einsum 路径优化策略，默认 'auto'。
                   可选: 'optimal', 'greedy', 'auto', 'dp', 'branch-2', 'branch-all'
        verbose:   是否打印详细分析过程

    Returns:
        BandwidthAnalysis 包含瓶颈判断和切分建议

    Example:
        >>> result = analyze_bandwidth('Mabc,Nabc->MN', [(2, 1000, 500, 64), (2, 1000, 500, 64)])
        >>> for s in result.suggestions:
        ...     print(f"建议切分 '{s.index}': 每块 {s.suggested_chunk_size}")
    """
    # 解析 dtype 元数据
    dtype_info = _DTYPE_TABLE.get(dtype)
    if dtype_info is None:
        raise ValueError(
            f"未知数据类型: '{dtype}'。可选: {list(_DTYPE_TABLE.keys())}"
        )
    dtype_bytes = dtype_info['bytes']
    flops_per_madd = dtype_info['flops_per_madd']
    dtype_name = dtype_info['name']

    # 1. 解析硬件 (先于 cost 估算, cost 需要 hw 做缓存抖动建模)
    if isinstance(hardware, str):
        hw = HARDWARE_PRESETS.get(hardware)
        if hw is None:
            raise ValueError(f"未知硬件预设: '{hardware}'。可用: {list(HARDWARE_PRESETS.keys())}")
    else:
        hw = hardware

    # 2. 解析下标
    parsed = parse_subscript(subscript, shapes)

    # 3. 估算成本 (传入 hw 以计算缓存抖动惩罚, optimize 选择收缩路径策略)
    cost = estimate_cost(parsed, hw, dtype_bytes, flops_per_madd,
                          optimize=optimize)

    # 3b. 时间估算 (基于 Roofline 模型)
    effective_compute_tflops = hw.effective_compute(dtype)  # TFLOPS
    effective_bw_bytes_per_s = hw.effective_bandwidth() * 1e9  # bytes/s

    # 纯计算时间 = FLOPs / 有效算力 (假设计算单元 100% 利用)
    if effective_compute_tflops > 0:
        estimated_compute_time = cost.total_flops / (effective_compute_tflops * 1e12)
    else:
        estimated_compute_time = float('inf')

    # 数据读入时间 = 等效数据量 / 有效带宽 (计入缓存抖动惩罚和中间张量写出)
    # 当工作集 >> L2 缓存时, 写入也会 cache miss → 适用相同的缓存惩罚
    effective_read_bytes = cost.total_read_bytes * cost.cache_penalty
    effective_write_bytes = (cost.total_write_bytes + cost.intermediate_write_bytes) * cost.cache_penalty
    total_data_bytes = effective_read_bytes + effective_write_bytes
    if effective_bw_bytes_per_s > 0:
        estimated_data_time = total_data_bytes / effective_bw_bytes_per_s
    else:
        estimated_data_time = float('inf')

    # Roofline 模型下: 实际耗时 = max(计算时间, 数据时间)
    # 两者可部分重叠, 但瓶颈决定最小耗时
    estimated_total_time = max(estimated_compute_time, estimated_data_time)

    # 4. Roofline 判断 (使用有效阈值, 计入实际可达效率)
    threshold = hw.effective_flops_per_byte(dtype)
    is_bandwidth_bound = cost.arithmetic_intensity < threshold

    ai_ratio = cost.arithmetic_intensity / max(threshold, 1e-10)
    if cost.arithmetic_intensity == float('inf'):
        severity = 'none'
    elif ai_ratio < 0.1:
        severity = 'severe'      # AI < 10% 阈值 → >90% 时间等数据
    elif ai_ratio < 0.25:
        severity = 'significant' # AI < 25% 阈值 → 计算单元利用不足
    elif ai_ratio < 1.0:
        severity = 'mild'        # AI < 100% 阈值 → 有带宽压力
    else:
        severity = 'none'        # AI >= 100% 阈值 → compute-bound

    # 5. 生成切分建议 (仅在带宽瓶颈或显存不足时)
    total_input_elements = 0
    for shape in parsed.shapes:
        s = 1
        for d in shape:
            s *= d
        total_input_elements += s
    total_input_bytes = total_input_elements * dtype_bytes

    # 显存压力: 峰值工作集超过显存的指定比例时强制建议切分
    mem_pressure_ratio = cost.peak_working_bytes / max(hw.total_memory_bytes, 1)
    memory_bound = total_input_bytes > hw.total_memory_bytes * 0.8
    memory_pressure = mem_pressure_ratio > _MEMORY_PRESSURE_THRESHOLD

    # L2 溢出: 仅对多输入 (>2) 收缩检查。
    # 2 输入收缩 → cuBLAS GEMM 自带 tiling, 无需额外切分。
    # 多输入收缩 → 产生物化中间张量, 每步都受缓存抖动影响。
    l2_overflow = (parsed.n_inputs > 2
                   and hw.l2_cache_bytes > 0
                   and cost.peak_working_bytes / hw.l2_cache_bytes > _L2_OVERFLOW_THRESHOLD)

    if is_bandwidth_bound or memory_bound or memory_pressure or l2_overflow:
        suggestions = _generate_slicing_suggestions(parsed, hw, dtype_bytes)
    else:
        suggestions = []

    # 5b. 生成增加维度建议 (仅在没有其他瓶颈时)
    has_bottleneck = (is_bandwidth_bound or memory_bound or memory_pressure
                      or l2_overflow)
    if not has_bottleneck:
        upsizing = _generate_upsizing_suggestions(parsed, cost, hw, dtype, dtype_bytes) or []
    else:
        upsizing = []

    # 6. 生成摘要
    summary = _generate_summary(parsed, cost, hw, is_bandwidth_bound, severity,
                                suggestions, dtype, dtype_name, upsizing,
                                estimated_compute_time, estimated_data_time,
                                estimated_total_time)

    if verbose:
        print(summary)

    return BandwidthAnalysis(
        parsed=parsed,
        cost=cost,
        hardware=hw,
        is_bandwidth_bound=is_bandwidth_bound,
        bottleneck_severity=severity,
        suggestions=suggestions,
        upsizing_suggestions=upsizing,
        summary=summary,
        estimated_compute_time=estimated_compute_time,
        estimated_data_time=estimated_data_time,
        estimated_total_time=estimated_total_time,
    )


def _generate_slicing_suggestions(
    parsed: ParsedContraction,
    hw: HardwareSpec,
    dtype_bytes: int,
) -> List[SlicingSuggestion]:
    """
    分析每个自由指标切分后的收益。

    切分的核心收益：
    - 将大收缩拆成多个小收缩
    - 每次小收缩的工作集更小，更容易放入缓存
    - 减少全局内存的重复读取

    对于切分自由指标 i (大小为 S_i → chunk 大小为 C):
    - 包含 i 的输入张量在每次子收缩中只需要 C/S_i 的数据
    - 不包含 i 的输入张量可以被所有子收缩复用（如果能放在缓存里）

    工作集 = 子收缩中所有输入数据 + 输出数据在缓存中的部分

    推荐切片大小 = 使得工作集能放入 L2 缓存的最大 chunk 大小
    """
    suggestions: List[SlicingSuggestion] = []

    if not parsed.free_indices:
        return suggestions

    available_cache = hw.l2_cache_bytes * 0.7  # 留 30% 余量

    for idx, size in parsed.free_indices.items():
        # 计算包含该指标的输入张量的大小
        involved_input_bytes = _compute_input_size_for_index(parsed, idx, dtype_bytes)

        # 不包含该指标的输入张量大小（可以作为"驻留"数据被复用）
        uninvolved_input_bytes = 0
        for sub, shape in zip(parsed.input_subs, parsed.shapes):
            if idx not in sub:
                s = 1
                for d in shape:
                    s *= d
                uninvolved_input_bytes += s * dtype_bytes

        # 对于每次子收缩：
        # - 包含 idx 的输入: involved_input_bytes * (chunk / size)
        # - 不包含 idx 的输入: uninvolved_input_bytes（希望驻留在缓存中复用）
        # - 输出部分: (output_size / size) * chunk * dtype_bytes
        output_elements_per_index = 1
        for oi, osize in parsed.free_indices.items():
            if oi != idx:
                output_elements_per_index *= osize

        # 目标：让工作集适合放入 L2 缓存
        involved_per_element = involved_input_bytes / size  # bytes per unit of idx
        output_per_sub = output_elements_per_index * dtype_bytes

        can_cache_uninvolved = uninvolved_input_bytes <= available_cache

        if can_cache_uninvolved:
            # 不涉及 idx 的数据可以驻留缓存 → 子收缩数据可从缓存复用
            available_for_chunk = available_cache - uninvolved_input_bytes
        else:
            # 不涉及 idx 的数据也放不进缓存 → 切分仅能减少峰值显存
            available_for_chunk = available_cache

        # 计算建议的 chunk 大小
        denom = involved_per_element + output_per_sub
        if available_for_chunk > 0 and denom > 0:
            suggested_chunk = int(available_for_chunk / denom)
        else:
            suggested_chunk = size

        # 夹紧到合理范围
        cache_optimal_chunk = max(1, min(suggested_chunk, size))

        # 防止过度切分: 最小块大小 ≥ ceil(size / max_chunks)
        # 切太细 → kernel launch 开销主导 + GPU 利用率不足 → 反而变慢
        min_chunk = math.ceil(size / _MAX_SLICING_CHUNKS)
        suggested_chunk = max(cache_optimal_chunk, min_chunk)
        constrained_by_min = (suggested_chunk > cache_optimal_chunk)

        if suggested_chunk >= size:
            continue  # 不需要切分

        n_chunks = math.ceil(size / suggested_chunk)

        # 峰值显存估算
        total_input_bytes = 0
        for shape in parsed.shapes:
            s = 1
            for d in shape:
                s *= d
            total_input_bytes += s * dtype_bytes

        output_bytes = 1
        for _, _sz in parsed.free_indices.items():
            output_bytes *= _sz
        output_bytes *= dtype_bytes
        peak_without = total_input_bytes + output_bytes

        # 切分后：只加载一个 chunk 的 involved 数据 + 全部 uninvolved 数据
        chunk_involved = involved_per_element * suggested_chunk
        chunk_output = output_per_sub * suggested_chunk
        peak_with = chunk_involved + uninvolved_input_bytes + chunk_output

        # 每块工作集大小
        working_set_per_chunk = chunk_involved + chunk_output
        if can_cache_uninvolved:
            working_set_per_chunk += uninvolved_input_bytes

        # 数据读取减少倍数估算
        # 不切分: 理想情况每个元素读一次 = involved + uninvolved
        # 切分后: involved 仍读一次; uninvolved 如果能缓存则只读一次,
        #         否则每个 chunk 都要重新读一次
        bytes_without_slicing = involved_input_bytes + uninvolved_input_bytes
        if can_cache_uninvolved:
            bytes_with_slicing = involved_input_bytes + uninvolved_input_bytes  # 缓存复用
        else:
            bytes_with_slicing = involved_input_bytes + uninvolved_input_bytes * n_chunks

        reduction = bytes_without_slicing / max(bytes_with_slicing, 1)
        reduction = max(1.0, reduction)

        # 生成理由
        if uninvolved_input_bytes == 0:
            rationale = (
                f"指标 '{idx}' (大小 {size}) 出现在所有输入张量中。"
                f"切分成 {n_chunks} 块（每块 {suggested_chunk}）可将峰值工作集从 "
                f"{involved_input_bytes/1e9:.1f} GB 降至 "
                f"{working_set_per_chunk/1e9:.2f} GB，"
                f"使每块数据能放入 L2 缓存 ({hw.l2_cache_bytes/1e6:.0f} MB)。"
            )
        elif can_cache_uninvolved:
            rationale = (
                f"指标 '{idx}' (大小 {size}) 切分成 {n_chunks} 块（每块 ~{suggested_chunk}）后，"
                f"不包含 '{idx}' 的输入张量 ({uninvolved_input_bytes/1e6:.1f} MB) "
                f"可驻留在 L2 缓存 ({hw.l2_cache_bytes/1e6:.0f} MB) 中被所有子收缩复用，"
                f"避免重复从 HBM 读取。每块工作集约 {working_set_per_chunk/1e6:.1f} MB。"
            )
        else:
            rationale = (
                f"指标 '{idx}' (大小 {size}) 切分成 {n_chunks} 块（每块 ~{suggested_chunk}）后，"
                f"每块工作集约 {working_set_per_chunk/1e6:.1f} MB，更接近 "
                f"L2 缓存大小 ({hw.l2_cache_bytes/1e6:.0f} MB)，提高缓存命中率。"
                f"注：不包含 '{idx}' 的输入 ({uninvolved_input_bytes/1e6:.1f} MB) "
                f"超出缓存容量，每个子收缩仍需从 HBM 重新读取，但切分可降低峰值显存占用。"
            )

        # 如果被最小粒度约束, 追加说明
        if constrained_by_min:
            rationale += (
                f" (粒度约束: 最多 {_MAX_SLICING_CHUNKS} 块, "
                f"避免过度切分导致 kernel launch 开销。"
                f"缓存最优块={cache_optimal_chunk}, "
                f"实际建议块={suggested_chunk})"
            )

        suggestions.append(SlicingSuggestion(
            index=idx,
            size=size,
            suggested_chunk_size=suggested_chunk,
            n_chunks=n_chunks,
            working_set_per_chunk_bytes=int(working_set_per_chunk),
            peak_memory_without_bytes=int(peak_without),
            peak_memory_with_bytes=int(peak_with),
            reduction_in_reads=reduction,
            rationale=rationale,
        ))

    # 按切分收益排序：减少倍数最大的排前面
    suggestions.sort(key=lambda s: s.reduction_in_reads, reverse=True)
    return suggestions


def _format_bytes(b: int) -> str:
    """格式化字节数为人可读的形式"""
    if b == 0:
        return "0 B"
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    if b < 1024 * 1024 * 1024:
        return f"{b / (1024*1024):.2f} MB"
    return f"{b / (1024*1024*1024):.2f} GB"


def _format_flops(flops: int) -> str:
    """格式化 FLOPs 为人可读的形式"""
    if flops < 1e6:
        return f"{flops} FLOPs"
    if flops < 1e9:
        return f"{flops / 1e6:.2f} MFLOPs"
    if flops < 1e12:
        return f"{flops / 1e9:.2f} GFLOPs"
    return f"{flops / 1e12:.2f} TFLOPs"


def _format_time(seconds: float) -> str:
    """格式化时间为人类可读形式"""
    if seconds == float('inf'):
        return "∞"
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    if seconds < 3600:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}m {s:.1f}s"
    if seconds < 86400:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    return f"{d}d {h}h"


def _generate_summary(
    parsed: ParsedContraction,
    cost: ContractionCost,
    hw: HardwareSpec,
    is_bound: bool,
    severity: str,
    suggestions: List[SlicingSuggestion],
    dtype: str = "complex128",
    dtype_name: str = "complex128",
    upsizing: List[UpsizingSuggestion] = None,
    estimated_compute_time: float = 0.0,
    estimated_data_time: float = 0.0,
    estimated_total_time: float = 0.0,
) -> str:
    """生成人类可读的分析摘要"""
    threshold = hw.effective_flops_per_byte(dtype)
    threshold_peak = hw.flops_per_byte(dtype)
    lines = []
    lines.append("=" * 72)
    lines.append("  张量收缩带宽瓶颈分析")
    lines.append("=" * 72)
    lines.append(f"  表达式:      {parsed.subscript}")
    shapes_str = ', '.join(str(s) for s in parsed.shapes)
    lines.append(f"  形状:        {shapes_str}")
    lines.append(f"  自由指标:    {dict(parsed.free_indices)}")
    lines.append(f"  收缩指标:    {dict(parsed.contracted_indices)}")
    lines.append(f"  硬件:        {hw.name}")
    lines.append(f"  数据类型:    {dtype_name}")
    lines.append("-" * 72)
    lines.append(f"  总 FLOPs:              {_format_flops(cost.total_flops)}")
    lines.append(f"  输入数据量 (理想):     {_format_bytes(cost.total_read_bytes)}")
    lines.append(f"  输出数据量:            {_format_bytes(cost.total_write_bytes)}")
    if cost.cache_penalty > 1.0:
        lines.append(f"  缓存抖动惩罚:          {cost.cache_penalty:.1f}x "
                     f"(工作集 {_format_bytes(cost.total_read_bytes + cost.total_write_bytes)}"
                     f" >> L2 {_format_bytes(hw.l2_cache_bytes)})")
        lines.append(f"  等效数据读取:          {_format_bytes(int(cost.total_read_bytes * cost.cache_penalty))}")
    lines.append(f"  有效算术强度:          {cost.arithmetic_intensity:.1f} FLOPs/byte")
    if cost.cache_penalty > 1.0:
        lines.append(f"    (理想: {cost.ideal_arithmetic_intensity:.1f}, "
                     f"惩罚因子: {cost.cache_penalty:.1f}x)")
    lines.append(f"  Roofline 阈值 (峰值):   {threshold_peak:.1f} FLOPs/byte")
    if hw.compute_efficiency < 1.0 or hw.memory_efficiency < 1.0:
        lines.append(f"    (计算效率 {hw.compute_efficiency:.0%}, "
                     f"带宽效率 {hw.memory_efficiency:.0%})")
        lines.append(f"  有效 Roofline 阈值:     {threshold:.1f} FLOPs/byte")
    lines.append(f"  输入数据占显存:        {cost.total_read_bytes / hw.total_memory_bytes * 100:.1f}%")
    lines.append("-" * 72)

    ai_ratio = cost.arithmetic_intensity / max(threshold, 1e-10)

    if severity == 'none':
        if suggestions:
            lines.append(f"  ⚠ 结论: 计算瓶颈, 但建议切分 (非 roofline 原因)")
            lines.append(f"    有效算术强度 ({cost.arithmetic_intensity:.1f} FLOPs/byte) >= "
                         f"有效阈值 ({threshold:.1f})")
        else:
            lines.append(f"  ✓ 结论: 计算瓶颈 (compute-bound)")
            lines.append(f"    有效算术强度 ({cost.arithmetic_intensity:.1f} FLOPs/byte) >= "
                         f"有效阈值 ({threshold:.1f})")
            lines.append(f"    计算单元能被充分利用，不需要切分。")
    elif severity == 'mild':
        lines.append(f"  ⚠ 结论: 轻微带宽瓶颈")
        lines.append(f"    算术强度 ({cost.arithmetic_intensity:.1f}) 仅为阈值的 {ai_ratio:.0%}")
        lines.append(f"    有一定的带宽压力，计算单元利用率不足。")
    elif severity == 'significant':
        lines.append(f"  ✗ 结论: 显著带宽瓶颈")
        lines.append(f"    算术强度 ({cost.arithmetic_intensity:.1f}) 仅为阈值的 {ai_ratio:.0%}")
        lines.append(f"    大部分时间在等待数据搬运，计算单元利用率低。")
    else:  # severe
        lines.append(f"  ✗✗ 结论: 严重带宽瓶颈!")
        lines.append(f"    算术强度 ({cost.arithmetic_intensity:.1f}) 仅为阈值的 {ai_ratio:.1%}")
        lines.append(f"    计算单元利用率极低 ({ai_ratio:.1%})，"
                     f"几乎全部时间在等待数据搬运。")

    # 显存压力 / L2 溢出警告
    mem_usage_ratio = cost.total_read_bytes / hw.total_memory_bytes
    peak_mem_ratio = cost.peak_working_bytes / max(hw.total_memory_bytes, 1)
    l2_ratio = cost.peak_working_bytes / max(hw.l2_cache_bytes, 1)

    if mem_usage_ratio > 0.9:
        lines.append(f"  ⚠ 输入数据占显存 {mem_usage_ratio:.0%}，接近显存上限！")
        lines.append(f"    强烈建议切分以降低峰值显存占用。")
    elif l2_ratio > _L2_OVERFLOW_THRESHOLD and suggestions:
        lines.append(f"  ⚠ 峰值工作集/L2缓存 = {l2_ratio:.0f}x "
                     f"(>{_L2_OVERFLOW_THRESHOLD}x 阈值)！")
        lines.append(f"    中间张量巨大 → 每步收缩都存在严重缓存抖动 → 建议切分。")
    elif peak_mem_ratio > _MEMORY_PRESSURE_THRESHOLD:
        lines.append(f"  ⚠ 峰值工作集占显存 {peak_mem_ratio:.0%}"
                     f" ({_format_bytes(cost.peak_working_bytes)})，"
                     f"超过 {_MEMORY_PRESSURE_THRESHOLD:.0%} 阈值！")
        lines.append(f"    中间张量过大 → 显存碎片化、cache 抖动 → 建议切分。")
    elif mem_usage_ratio > 0.5:
        lines.append(f"  ⚠ 输入数据占显存 {mem_usage_ratio:.0%}，显存压力较大。")

    # 用于时间估算显示的数据量 (含中间张量写出)
    _eff_read_bytes = int(cost.total_read_bytes * cost.cache_penalty)
    _eff_write_bytes = int((cost.total_write_bytes + cost.intermediate_write_bytes) * cost.cache_penalty)
    _total_data_bytes = _eff_read_bytes + _eff_write_bytes

    lines.append("-" * 72)
    lines.append(f"  预计时间估算 (Roofline 模型):")
    lines.append(f"    纯计算时间:            {_format_time(estimated_compute_time)}")
    lines.append(f"      (FLOPs / 有效算力 = {_format_flops(cost.total_flops)}"
                 f" / {hw.effective_compute(dtype):.3g} TFLOPS)")
    lines.append(f"    数据读入时间:          {_format_time(estimated_data_time)}")
    lines.append(f"      ({_format_bytes(_total_data_bytes)}"
                 f" / {hw.effective_bandwidth():.1f} GB/s"
                 f"{' (含缓存抖动 ' + str(round(cost.cache_penalty, 1)) + 'x)' if cost.cache_penalty > 1.0 else ''})")
    if estimated_total_time == estimated_compute_time and estimated_compute_time >= estimated_data_time:
        lines.append(f"    预计总耗时:            {_format_time(estimated_total_time)} (瓶颈: 计算)")
    elif estimated_total_time == estimated_data_time and estimated_data_time >= estimated_compute_time:
        lines.append(f"    预计总耗时:            {_format_time(estimated_total_time)} (瓶颈: 带宽)")
    else:
        lines.append(f"    预计总耗时:            {_format_time(estimated_total_time)}")
    lines.append("-" * 72)

    if suggestions:
        lines.append(f"  切分建议 ({len(suggestions)} 个):")
        lines.append("")
        for i, sug in enumerate(suggestions, 1):
            mem_reduction = (sug.peak_memory_without_bytes /
                             max(sug.peak_memory_with_bytes, 1))
            lines.append(f"  [{i}] 切分指标 '{sug.index}'")
            lines.append(f"      总大小:            {sug.size}")
            lines.append(f"      建议切片大小:      {sug.suggested_chunk_size}")
            lines.append(f"      切分块数:          {sug.n_chunks}")
            lines.append(f"      峰值显存 (不切分): {_format_bytes(sug.peak_memory_without_bytes)}")
            lines.append(f"      峰值显存 (切分后): {_format_bytes(sug.peak_memory_with_bytes)}")
            lines.append(f"      显存降低:          {mem_reduction:.1f}x")
            lines.append(f"      每块 L2 工作集:    {_format_bytes(sug.working_set_per_chunk_bytes)}")
            lines.append(f"      数据读取减少:      {sug.reduction_in_reads:.1f}x")
            lines.append(f"      理由: {sug.rationale}")
            lines.append("")
    else:
        if is_bound:
            lines.append(f"  无可切分的自由指标，或切分无收益。")
            lines.append(f"  考虑减小收缩指标的大小（如通过近似/剪枝）。")

    # 增加维度建议
    if upsizing:
        lines.append("-" * 72)
        # 区分原因是 roofline 不足还是 GPU 并行度不足
        is_parallelism = any('GPU 缺乏并行度' in u.rationale for u in upsizing)
        if is_parallelism:
            lines.append(f"  GPU 并行度不足 — 增加维度建议 (共 {len(upsizing)} 个):")
        else:
            lines.append(f"  增加维度建议 (提升算术强度, 共 {len(upsizing)} 个):")
            lines.append(f"  当前 AI={cost.arithmetic_intensity:.1f} → 目标 AI={upsizing[0].target_ai:.1f} FLOPs/byte")
        lines.append("")
        for i, u in enumerate(upsizing[:5], 1):  # 最多显示 5 个
            tag = "收缩" if u.index_type == 'contracted' else "自由"
            lines.append(f"  [{i}] 增加{tag}指标 '{u.index}': "
                         f"{u.current_size} → {u.target_size} (×{u.scale_factor:.1f})")
            lines.append(f"      数据增加: {_format_bytes(u.data_increase_bytes)}")
            lines.append(f"      理由: {u.rationale}")
            lines.append("")

    lines.append("=" * 72)
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════════════════════

def quick_check(subscript: str, shapes: List[Tuple[int, ...]],
                hardware: str = "A100_80GB", dtype: str = "complex128",
                optimize: str = "auto") -> BandwidthAnalysis:
    """
    快速检查一个收缩操作是否需要切分。

    Args:
        subscript: einsum 下标
        shapes:    输入张量形状列表
        hardware:  硬件预设名称，默认 'A100_80GB'。
        dtype:     数据类型，默认 'complex128'。
        optimize:  opt_einsum 路径优化策略，默认 'auto'。

    Returns:
        BandwidthAnalysis 结果
    """
    return analyze_bandwidth(
        subscript, shapes,
        hardware=hardware,
        dtype=dtype,
        optimize=optimize,
        verbose=False,
    )

def printGPUinfo():
    print("可用 GPU 硬件预设:")
    print("-" * 72)
    for key, hw in HARDWARE_PRESETS.items():
        if key in ('A100',):  # 跳过别名
            continue
        print(f"  {key:<16s} | FP16={hw.peak_fp16:>5.1f} TF | FP32={hw.peak_fp32:>5.1f} TF | "
              f"FP64={hw.peak_fp64:>5.1f} TF | BW={hw.memory_bandwidth_gbs:>.0f} GB/s | "
              f"Mem={hw.total_memory_bytes//_GB}GB")
    print("-" * 72)