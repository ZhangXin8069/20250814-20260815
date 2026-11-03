"""
Type stub for lqcddb.io subpackage.

提供数据 I/O 功能：ASCII 格式写入、目录路径管理和安全保存。
"""
from typing import List, Optional, Union
import numpy as np

def write_data_ascii(
    data: np.ndarray,
    T: int,
    L: int,
    filename: str,
    complex: bool = True,
    verbose: bool = False,
) -> None:
    """以 L. Liu 格式写入 ASCII 数据文件。

    首行为 ``nsamples T complex L 1`` 头部，后跟数据行。
    复数时实部和虚部分列输出。

    参数:
        data: 输入数据数组，第一维为样本。
        T: 时间方向格点数。
        L: 半时间格点数。
        filename: 输出文件路径。
        complex: ``True`` 时写入复数 (实部+虚部)，``False`` 时仅实数。
        verbose: 是否打印详细写入信息。
    """
    ...
def check_dir_path(path: str) -> str:
    """检查并创建目录路径 (``mkdir -p``)。

    参数:
        path: 目录路径字符串。
    """
    ...
def safe_save(
    file: Union[str, "pathlib.Path"],
    arr: np.ndarray,
    allow_pickle: bool = True,
    fix_imports: bool = True,
    fallback_dirs: Optional[List[str]] = None,
) -> str:
    """将数组保存为 NumPy ``.npy`` 二进制文件，出错时自动回退到备用目录。

    用法与 ``numpy.save`` 一致。先尝试保存到主路径；若失败（如磁盘满、
    权限不足），依次尝试 ``fallback_dirs``（用户指定）；若仍然失败，
    自动构建 ``/nexdata/project/lqcd/${USER}/result/`` 下的路径再试。

    自动回退路径：从 ``file`` 中找到名称含 ``result`` 的目录，提取其后的
    子路径拼接到 ``/nexdata/project/lqcd/${USER}/result/`` 下。
    若路径不含 ``result`` 目录，则直接使用原文件名。

    参数:
        file: 目标文件路径。若不以 ``.npy`` 结尾，自动追加后缀。
        arr: 待保存的数组数据。
        allow_pickle: 是否允许使用 pickle 保存对象数组。
        fix_imports: 仅用于 Python 2 兼容。
        fallback_dirs: 用户指定的备用目录列表，优先级高于自动回退。

    返回:
        实际保存成功的文件路径。

    异常:
        OSError: 所有路径（主 + 用户备用 + 自动回退）都保存失败时抛出。
    """
    ...
