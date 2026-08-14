# AGENTS.md — 汇报

格点 QCD 计算方法中文报告，聚焦胶子准算符构造与胶子 PDF 幻灯片（XeLaTeX）。

| 文件 | 内容 |
|---|---|
| `格点上计算胶子准算符.tex` | 格点上计算胶子准算符 |
| `构造胶子准算符.tex` | 构造胶子准算符 |
| `gluon_pdf_slides.tex` | Beamer 胶子 PDF 理论与工作流演示 |

## 编译

```bash
xelatex -interaction=nonstopmode -halt-on-error <file>.tex   # 两遍
```

`reports/` 为英文幻灯片，本目录为中文报告（计算方法更详细）。
