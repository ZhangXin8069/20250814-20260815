# AGENTS.md — examples/huangcl/02_ratio/9_others

流水线步骤 2（`02_ratio`）的**补充分析脚本**，独立于主 `code.py`/`code_1.py` 比值流水线。

## 文件

| 文件 | 用途 |
|---|---|
| `Calc_3pt.py` | Chroma IOG 3pt 提取（读 IOG 3pt/2pt，存 `.npy` 供比值分析） |
| `Fit_Raito.py` | 多参数比值拟合（`lsqfit` + `gvar`） |

## 使用

```bash
# 在 HPC 集群（需 Chroma IOG .so）
python Calc_3pt.py      # 从 IOG 文件提取 3pt
python Fit_Raito.py     # 拟合提取的比值
```
