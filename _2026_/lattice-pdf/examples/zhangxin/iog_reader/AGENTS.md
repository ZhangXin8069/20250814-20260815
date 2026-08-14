# AGENTS.md — examples/zhangxin/iog_reader

读取 Chroma IOG（Inline Observable Group）二进制格式的 C 扩展模块，经 `ctypes` 桥接编译的 C 库到 Python。

## 文件

`iog_reader.py`（ctypes 包装）、`iog.so`（编译的 C 共享库，**gitignored**）。

## 架构

- `iog.so` 暴露两个函数：`getsize()`（记录数）、`getdat()`（整数元数据标签 + 双精度实/虚数组的结构体）
- `iog_reader.py` 的 `iog_read()`：加载 .so → 调 getsize/getdat → 转 numpy → 解释元数据标签（源/汇位置、γ 矩阵、动量）→ pandas DataFrame（Re/Im 列）

## 构建 iog.so

**必须在 HPC 集群上用 Chroma 兼容编译器构建**（需 Chroma 兼容 MPI/C++ 编译器）。仅在 `snsc/main.py --analysis-type 2pt` 及 `main-2pt.py`/`main-3pt.py`/`main_iog.py` 需要；GPU 流水线（`agent/docker-v*/`）不用 IOG。
