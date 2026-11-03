---
name: 量子场论导论_latex
description: 将 Peskin & Schroeder《An Introduction to Quantum Field Theory》英文 LaTeX 逐章译为中文的完整 skill — 保留全部数学与 LaTeX 结构，正文/标题译为规范学术中文，编译为 ctexbook；当前进度 frontmatter + 第 1–22 章全部完成
---

# 中文版转排指南 — 量子场论导论

中文版由英文版 LaTeX（`../An_Introduction_to_Quantum_Field_Theory_latex/`）逐章翻译而来。
译文必须**保留全部 LaTeX 结构**，只翻译文字内容。

## 硬性规则

1. **保留结构**：`\chapter`/`\section`/`\subsection` 标题翻译成中文；方程环境、
   数学公式、`\label`、`\eqref`/`\cref`、图表环境、`\includegraphics{figX.Y.png}`、
   编号一律**原样保留，不得改动**。
2. **翻译范围**：正文散文、小节标题、图题（caption）、表题与表格单元内容 → 中文；
   数学、符号、人名、文献编号、数值单位 → 保留。人名保留英文原名。
3. **公式中的文字**：`\text{...}` 内的说明性文字可译成中文；数学符号不译。
4. **术语统一**：物理术语采用标准译名——量子场论、拉格朗日量、克莱因-戈登场、
   狄拉克场、费曼图、传播子、重整化、规范不变性、自发对称性破缺等。
5. **语气**：忠实直译，不增删内容，不添加注释。
6. **章末习题**：`\problems` 命令由 `main.tex` 定义为「习题」。
7. **输出**：写到 `量子场论导论_latex/chapters/chXX.tex`（与英文版同名，一一对应）。
8. **图片**：沿用英文版 `images/` 中的 figX.Y.png（图片内容与语言无关，不重新提取）。

## 当前进度

- 已完成中文翻译：`frontmatter.tex` + 第 1–22 章全部完成（`ch01.tex`–`ch22.tex`），
  三个部分（费曼图与量子电动力学 / 重整化 / 非阿贝尔规范理论）均译毕。
- 编译产物：`build/main.pdf`（419 页），并复制到目录根部 `量子场论导论.pdf`。
- 各章文件与英文版 `chXX.tex` 结构一一对应。

## 中文排版注意事项

- 正文中的希腊字母必须用数学模式（如 `$\pi$`、`$\alpha_s$`）：中文字体缺少
  U+03C0 等字形，直接使用字面 `π` 会产生 "Missing character" 警告。
- 数学环境中的说明性 `\mathrm{}` 不应放中文（数学字体无 CJK 字形），应保留英文。
- 中文引号用「」。

## 编译

```bash
cd build
xelatex -interaction=nonstopmode -halt-on-error ../main.tex   # 需两遍
```

需要 XeLaTeX + `ctexbook` 宏包 + 中文字体（项目统一用 XeLaTeX 支持中文）。

## 质量要求

- 章节必须在项目 `main.tex`（xelatex + ctexbook）下编译通过。
- 数学与英文版逐字节一致。
- 翻译忠实、流畅，使用标准中文物理术语；不增删内容。
