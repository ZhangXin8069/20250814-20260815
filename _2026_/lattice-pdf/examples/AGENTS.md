# AGENTS.md — examples

lattice-pdf 项目全部 Python 代码，按贡献者组织。各子目录有自己的 CLAUDE.md（已归档为 `.CLAUDE.md.<ts>.bak` 供参考）。

| 目录 | 贡献者 | 内容 |
|---|---|---|
| `donghx/` | 董 HX | 质子 2pt 蒸馏 + OPE 算符（64 脚本，CuPy/DCU/CPU） |
| `zhangxin/` | 张 X | 胶子 PDF 完整工作流 + 数据分析框架 |
| `huangcl/` | 黄 CL | 多步 Chroma 流水线（contract → ratio → energy） |
| `sush/` | 苏 SH | 蒸馏收缩框架 + lqcddb 包 |

## 快速参考

| 需求 | 入口 |
|---|---|
| 质子 2pt（蒸馏） | `donghx/2pt_proton_Cg5gmu_*.py` |
| OPE（胶子算符） | `donghx/Calc_ope_unpol.py` |
| 完整胶子 PDF 流水线 | `zhangxin/gluon_pdf_full_workflow.py` |
| 关联函数分析 | `zhangxin/include.py` |
| Wick 收缩 | `sush/`（lqcddb 的 `wick_contraction`） |
| 多步 Chroma 流水线 | `huangcl/` |

## 通用模式

- GPU 后端：`_gpu.py`=CuPy、`_dcu.py`=DCU(ROCm/HIP)；回退 `try: import cupy / except: cp = np`
- 参数传递：donghx/huangcl 用 stdin 重定向（`fileinput.input()`）；zhangxin 用 argparse
- γ 矩阵：DeGrand-Rossi（DR，手征变体）基
- **张量约定不同**：zhangxin 规范场 `[color,color,dir,x,y,z,t]`；donghx `[t,z,y,x,dir,color,color]`
