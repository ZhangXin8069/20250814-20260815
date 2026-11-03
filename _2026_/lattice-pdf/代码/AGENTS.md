# AGENTS.md — 代码

GPU 计算代码分析文档（`code_analysis.tex`/`.pdf`），分析胶子 PDF 流水线中用到的 GPU 算法。

分析对象：`examples/donghx/`（质子 2pt + OPE）、`examples/zhangxin/`（胶子 PDF 工作流）、`agent/docker-v*/`（GPU 验证流水线）。

## 编译

```bash
xelatex -interaction=nonstopmode -halt-on-error code_analysis.tex   # 两遍
```
