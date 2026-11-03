# AGENTS.md — 文档

胶子 PDF 项目核心 LaTeX 文档（主要理论参考，含完整推导）。

| 文件 | 内容 |
|---|---|
| `gluon_pdf_derivation.tex` | 胶子 PDF 完整推导（LaMET、OPE 分解、匹配核） |
| `gluon_PDF_continuum.tex` | 胶子 PDF 连续极限外推 |
| `note_of_gluon_PDFs.tex` | 内部理论笔记（Eq.20 矩阵元、Eq.25 OPE 分解） |

## 关键方程

所有 Python 代码的理论支柱：胶子矩阵元由场强张量 F_{μν} + Wilson 线构造（Eq.20）；断连三点函数分解为 质子 2pt ⊗ OPE 算符（Eq.25）。

实现处：`examples/zhangxin/gluon_pdf_full_workflow.py`、`agent/docker-v20260802/`、`snsc/main.py --analysis-type pdf`。
