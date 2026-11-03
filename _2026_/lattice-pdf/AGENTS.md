# AGENTS.md — lattice-pdf

ZhangXin 的格点 QCD 研究项目：用 LaMET 框架从第一性原理格点计算质子**非极化胶子 PDF**。断连三点点关联函数分解为两部分：**质子 2pt 关联函数**（动量涂抹蒸馏）+ **OPE 部分**（非定域胶子算符）。核心方程见 `docs/Note of gluon PDFs.pdf`（Eq.20 矩阵元、Eq.25 OPE 分解）。

## 入口

| 需求 | 入口 |
|---|---|
| 完整胶子 PDF 流水线 | `snsc/main.py --analysis-type pdf` 或 `examples/zhangxin/gluon_pdf_full_workflow.py` |
| 质子 2pt（蒸馏） | `snsc/main.py --analysis-type proton-2pt` 或 `examples/donghx/2pt_proton_*.py` |
| GPU 流水线（CuPy，现行） | `agent/docker-v20260805/run_pipeline.py`（launcher: `agent/docker/run_gpu_pipeline.sh`） |
| Wick 收缩 | `examples/sush/`（`wick_contraction`，lqcddb 包） |
| 关联函数分析（jackknife/meff） | `examples/zhangxin/include.py` 或 `examples/sush/lqcddb/` |
| 编译 LaTeX | `xelatex -interaction=nonstopmode <file>.tex` 两遍 |

## 关键约定

- **参数传递两种模式并存**：`examples/zhangxin/` 用 argparse；`examples/donghx/` 用 stdin 重定向（`fileinput.input()` 解析键值对）。
- **GPU 后端**：`_gpu.py` = NVIDIA CUDA/CuPy；`_dcu.py` = AMD/Hygon DCU（ROCm/HIP，PyTorch 兼容层）。`try: import cupy; except ImportError: cp = np` 回退模式。
- **张量约定不一致**：zhangxin 规范场 `[color,color,dir,x,y,z,t]`；donghx `[t,z,y,x,dir,color,color]`；奇偶分裂张量前置 `[2]` 维。
- **γ 矩阵**：DeGrand-Rossi（手征变体）基，见 `examples/donghx/gamma_matrix_cupy_DR.py`。
- **数据在集群**：`/public/group/lqcd/` 等路径，本机不解析；`examples/donghx/` 含集群符号链接。
- 无 env.sh（不同于 PyQCU/PyQUDA）；默认 conda 环境 `zhangxin-snsc`。

## 目录

| 目录 | 内容 |
|---|---|
| `examples/` | 全部 Python 代码，按贡献者分：`donghx`(质子2pt+OPE)、`zhangxin`(胶子PDF工作流+分析)、`huangcl`(Chroma 多步流水线)、`sush`(蒸馏收缩框架+lqcddb 包)、`zengch`(GPD/PDF 数据拟合分析) |
| `代码/` | GPU 计算代码分析文档（`code_analysis.tex`/`.pdf`） |
| `docker/` | 容器化部署占位（现行流水线在 `agent/docker-v*`） |
| `snsc/` | 统一生产入口：`main.py`(3 种模式、13 步)、`sbatch.sh`/`sbatch-2pt.sh`(Slurm) |
| `docs/` | 80+ 参考论文 PDF（只读，`pdftotext` 阅读） |
| `books/` | 3 本教科书 + agent 生成的 LaTeX 转排（英/中） |
| `agent/` | AI agent 子模块（LQCD_Master、lamet-agent、PyQUDA，git submodules）+ 版本化 GPU 流水线 `docker-v*`（现行 `docker-v20260805`） |
| `refer/` | `理论解析与工作流.tex` — 理论与工作流总参考 |
| `补充/` | 38 篇中文补充 LaTeX 笔记 |
| `文档/` | 核心推导文档（`gluon_pdf_derivation.tex` 等） |
| `汇报/` | 中文报告；`reports/` 英文 Beamer 幻灯片 |
| `.opencode/skills/` | agent 生成内容的完整技能归集（latex 转写/翻译、docker 管线、文档生成等，复制自全局技能库，原位置保留） |

## 验证

无正式测试框架（无 pytest/unittest）。测试为独立脚本，输出结果与图供人工核验：`examples/sush/function_contraction/test/`。重要规则：**function_contraction 下只允许修改 `test/` 目录**。
