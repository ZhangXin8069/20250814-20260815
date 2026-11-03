# AGENTS.md — examples/donghx/docs

董 HX 格点 QCD 计算代码分析文档（LaTeX：质子 2pt 蒸馏与 OPE 算符算法分析）。

| 文件 | 说明 |
|---|---|
| `donghx_code_analysis.tex` / `.pdf` | donghx 的 CuPy/DCU GPU 代码分析 |

## 编译

```bash
xelatex -interaction=nonstopmode -halt-on-error donghx_code_analysis.tex   # 两遍
```

分析对象：`../2pt_proton_Cg5gmu_*.py`、`../Calc_ope_unpol.py`、`../Operator.py`、`../gamma_matrix_cupy_DR.py`。同类文档：`../../huangcl/docs/`、`../../代码/`。
