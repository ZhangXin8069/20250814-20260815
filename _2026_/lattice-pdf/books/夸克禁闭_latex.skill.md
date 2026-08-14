---
name: 夸克禁闭_latex
description: 将 Wilson《Confinement of quarks》（PRD 10, 2445 (1974)）的英文 LaTeX 翻译为中文版 LaTeX 的完整 skill — 使用 ctexart/XeLaTeX 逐节翻译，保持章节、公式与插图编号与原文一致并编译出 PDF
---

# Translation guide for section agents

You are translating one section of the *English* LaTeX transcription of
*Confinement of quarks* by Kenneth G. Wilson (Phys. Rev. D 10, 2445 (1974))
into clean, compilable Chinese LaTeX.

## Input / output
- **Input**: `../Confinement_of_quarks_latex/chapters/section0N.tex` (the
  English LaTeX for your section). Read it fully.
- **Output**: write `chapters/section0N.tex` in the Chinese directory (one file
  per section — do not touch other section files).
- `main.tex` (`ctexart`, XeLaTeX) does `\input{...}` per section and sets
  `\graphicspath{{images/}}`; images are copied from the English dir.

## Translation rules
- Translate the prose into natural, accurate **Chinese** (simplified).  Keep
  proper nouns either as the original Latin names or with a common Chinese
  rendering on first use, e.g. Schwinger（施温格）、Wilson、Kogut、Susskind.
- **Do not translate the math**: keep every `equation` environment, every label,
  and every equation number identical to the English version so
  cross-references (`\eqref`) and auto-numbering stay in sync.
- Section titles: Roman numbering via `\thesection` is already set — translate
  the title text itself (e.g. `\section{夸克束缚机制}` for "QUARK BINDING MECHANISM").
- Figures: keep `\refstepcounter{figure}` + `\caption*{图 N. ...}` (Chinese
  caption text) and the same `\label{fig:N}`.  In-text refs become ``图~\ref{fig:N}``.
- Reference list stays in the original English (backmatter.tex).
- Footnote-style superscripts `${}^{N}$` map to the same reference numbers.

## Compiling
```bash
cd 夸克禁闭_latex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex   # ×2 遍
```

## Quality bar
- Must compile with XeLaTeX + `ctexart`.
- Equations and numbering identical to the English version; prose faithful to
  the original paper in Chinese.
