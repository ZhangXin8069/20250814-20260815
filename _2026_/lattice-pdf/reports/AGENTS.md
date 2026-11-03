# AGENTS.md — reports

胶子 PDF 项目 Beamer 演示幻灯片（英文，XeLaTeX）。

| 文件 | 内容 |
|---|---|
| `gluon_pdf_slides.tex` | 胶子 PDF 理论与工作流幻灯片 |
| `gluon_pdf_continuum_beamer.tex` | 胶子 PDF 连续极限结果幻灯片 |

## 编译

```bash
xelatex -interaction=nonstopmode -halt-on-error <file>.tex   # 两遍（TOC/交叉引用）
```

主题：`metropolis`、`Madrid`、`CambridgeUS`。`汇报/` 为中文详细报告。
