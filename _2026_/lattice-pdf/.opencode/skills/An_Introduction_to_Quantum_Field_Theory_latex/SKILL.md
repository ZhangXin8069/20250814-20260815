---
name: An_Introduction_to_Quantum_Field_Theory_latex
description: 将 Peskin & Schroeder《An Introduction to Quantum Field Theory》转写为 LaTeX 的完整 skill — 从 pdftotext 提取逐章重建可编译 LaTeX（22 章 3 部 + 附录），公式按章自动编号，插图提取为 figX.Y.png
---

# Conversion Guide — An Introduction to Quantum Field Theory → LaTeX

Source: M. E. Peskin and D. V. Schroeder, *An Introduction to Quantum Field
Theory* (Addison-Wesley, 1995). This repo is a faithful LaTeX transcription
(incremental). Source PDF: `../An Introduction to Quantum Field Theory.pdf`.

## Directory layout

```
main.tex                      master document (book class, XeLaTeX; already written)
chapters/chXX.tex             one chapter file per chapter (ch01–ch22)
chapters/frontmatter.tex      preface / notation / editors' preface
chapters/appendix.tex         appendix
extract/                      raw pdftotext output (INPUT, read-only)
extract_figures.py            figure extraction script (figX.Y.png)
images/figX.Y.png             figures cropped from the source PDF (X.Y = original figure number)
```

## Document structure

- `main.tex` uses `\documentclass[11pt,oneside]{book}` with three `\part`s:
  - **Part I — Feynman Diagrams and Quantum Electrodynamics**: ch01–ch07
  - **Part II — Renormalization**: ch08–ch13
  - **Part III — Non-Abelian Gauge Theories**: ch14–ch22
  - then `\appendix` (`chapters/appendix.tex`).
- `\numberwithin{equation}{chapter}` → equations auto-number `(2.1)`, `(3.7)`, …
  matching the book. **Do not hand-number or `\tag` anything.**
- Chapter-end problems use the `\problems[...]` macro defined in `main.tex`
  (`\section*{Problems}\addcontentsline{toc}{section}{Problems}`).
- Requires the `slashed` package (for Feynman slash notation) plus the standard
  `amsmath amssymb amsthm mathtools graphicx booktabs bm enumitem microtype
  hyperref cleveref`.

## Per-chapter transcription rules

1. **Input**: read `extract/` pdftotext dump for the chapter. Strip running
   heads, standalone page numbers, and any extraction artifacts.
2. **Output**: write `chapters/chXX.tex`. First line: `\chapter{<exact source
   chapter title>}`.
3. **Equations**: the pdftotext output is a mangled linear rendering. Reconstruct
   each equation into correct LaTeX using your knowledge of quantum field theory
   (Lagrangians, Dirac/Pauli matrices, Feynman rules, propagators, loops,
   renormalization, gauge theories). Preserve the order so auto-numbers match.
   Display equations → `\begin{equation}`/`align`; inline → `$...$`.
4. **Figures**: wherever the source shows a figure, insert
   `\begin{figure}[htbp] \centering \includegraphics[width=...]{figX.Y.png}
   \caption{...} \label{fig:X.Y} \end{figure}` using the exact filename from
   `images/` (figures keep their original X.Y numbers).
5. **Faithfulness**: transcribe prose as-is; fix only pdftotext artifacts
   (stray line breaks, split words, spacing). Preserve all physics content.
6. In-text references: some refs (e.g. `eq:4.48`) point to equations the book
   cites but never numbers — leave them as-is (faithful transcription).
7. **Do not** `\tag`, `\setcounter`, invent figures, or leave Unicode math
   symbols in the text.

## Quality bar

- The chapter must compile under the project's `main.tex` (xelatex).
- Math must be physically correct standard QFT expressions.
- Keep the book's structure, equation order, and figure numbering exactly.

## Self-check

```bash
cd build
xelatex -interaction=nonstopmode -halt-on-error ../main.tex   # ×2 passes
```
