# AGENTS.md — 补充

38 篇中文补充 LaTeX 笔记（`.tex` + 编译 PDF），覆盖格点 QCD 理论全部主题：蒸馏方法、3pt 构造、标度参数、部分子分布函数、费曼图、smear 算法（Wuppertal/Jacobi/动量/蒸馏 + APE/HYP/stout/梯度流）、传播子求解、LaMET、费米子方案、Monte Carlo、误差分析、重整化、Wilson 线等。

## 编译

```bash
xelatex -interaction=nonstopmode -halt-on-error 格点QCD蒸馏方法解析.tex   # 两遍
```

辅助文件（`.aux/.log/.out/.toc`）被 gitignore，仅跟踪 `.tex` 与 `.pdf`。

## 约定

- 全中文；XeLaTeX + ctex/xeCJK；命名为主题描述性中文文件名。
