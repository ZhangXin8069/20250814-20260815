---
name: Confinement_of_quarks_latex
description: 将 Wilson《Confinement of quarks》（PRD 10, 2445 (1974)）转写为 LaTeX 的完整 skill — 从 pdftotext 提取逐节重建可编译 LaTeX（6 节 + 致谢 + 参考文献），保留原文编号、公式与 10 张插图
---

# Conversion guide for section agents

You are converting one section of *Confinement of quarks* by Kenneth G.
Wilson (Phys. Rev. D 10, 2445 (1974)) from the raw `pdftotext` dump into
clean, compilable LaTeX.  This is the foundational lattice gauge theory paper.

## Input / output
- **Input**: `extract/pageNN.txt` (raw per-page pdftotext, two-column PRD scan).
  Read the pages of your section fully.
- **Output**: write `chapters/section0N.tex` (one file per section — do not
  touch any other section file).
- `main.tex` does `\input{preamble}` and includes each section;
  `\graphicspath{{images/}}` is set, so images resolve as
  `\includegraphics[width=...]{figN.png}`.

## What the raw text looks like
`pdftotext` on a 1974 scan is very noisy. Clean it up:
- **Page numbers** and **running heads** (`KENNETH G. WILSON`, `CONFINEMENT OF QUARKS`, the volume/page `10 / 2448`) must be **deleted**. The section titles are real structure — keep them.
- The paper is **two-column**; the raw text interleaves columns. Reassemble the prose in reading order.
- Pages 1 and 15 contain the **tails of the adjacent papers** (an "effective action" paper before, and Cheng/Ng/Young after) — do NOT transcribe those.
- **Math glyphs are mangled**: the gauge-field variable is written variously as `A`, `B`, `8`, `θ` — it is `θ_{μν}`. `ψ̄`/`φ̄` appear as garbled `g`, `y`, `4`, `0`. Reconstruct every equation from context using your physics knowledge (lattice QCD). Equation numbers `(3.1)`, `(4.6)`, etc. are preserved — keep them oriented but DO NOT typeset the numbers (LaTeX auto-numbers).

## Document structure
- Section files start with `\section{<EXACT UPPERCASE SECTION TITLE>}`.
  Section III has two `\subsection`s (A. Classical action on a lattice; B. Quantization).
- Keep prose **faithful** to the original English, correcting only extraction artifacts. Keep paragraph breaks.

## Equations
- Use `equation`, and `split`/`align` where the original is multi-line.
- **Preserve the order** so auto-numbers match the paper's `(2.1)`, `(3.1)`, …
  (`\theequation` is `\arabic{section}.\arabic{equation}` in the preamble).
- In-text refs like `Eq. (3.13)` stay as literal text.
- Use the preamble macros: `\thl{\mu\nu}` for `θ_{μν}`, `\ev{...}` for `⟨...⟩`,
  `\Tr`, `\order{...}`, `\dd`, `\ddth{\mu\nu}` for `dθ_{μν}`.
- The action parameters are `ε = m₀a⁴`, `κ = a²/2` (Eq. 3.12).

## Figures
The 10 figures are in `images/figN.png` (global numbering). Where the paper
shows a figure (its caption `FIG. N.` appears in the raw text), insert:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.5\textwidth]{figN}
  \refstepcounter{figure}%
  \caption*{Fig. N. <caption text, cleaned up>.}
  \label{fig:N}
\end{figure}
```

- `\refstepcounter{figure}` makes in-text `Fig.~\ref{fig:N}` resolve; the
  starred `\caption*{Fig. N. ...}` keeps the paper's exact numbering.
- Pick `width` from the paper's column proportions (the drawings are small:
  typically `0.3`–`0.55\textwidth`).

## Backmatter
`backmatter.tex` holds the Acknowledgments and the reference list (Refs. 1–20,
kept as printed). Only edit it if you are fixing the references.

## Quality bar
- The `.tex` must compile with XeLaTeX.
- Match the paper's structure and content as closely as possible.
- Math must be **physically correct** lattice gauge theory.
- If a passage is too mangled to recover, reproduce the closest reasonable
  LaTeX and add a short `% [unrecovered: ...]` comment.

## Also produce
`figures_map.md` (fig → image + caption), `chapter_sections.txt` (structure +
equation list), `CONVERSION_GUIDE.md` (transcription rules), `CLAUDE.md`.

Chinese translation: `../夸克禁闭_latex/` (translated from this English LaTeX).
