# AGENTS.md — examples/sush/lqcddb/src/lqcddb/contraction

lqcddb 核心收缩机制：自动 Wick 收缩、算符共轭、顺序传播子、高层动态收缩工作流、roofline 模型带宽顾问。

## 文件

| 文件 | 用途 |
|---|---|
| `autowick.py` | `wick_contraction`（枚举所有 Wick 图，通配符 'q'/'l'、'\|' 强子分隔符）；`plot_figure_wick`；`identify_equivalent_diagrams` |
| `dynamic.py` | `dynamic_contraction` + `PeramRegistry`/`VRegistry`/`GammaRegistry`——推荐的高层工作流，含计划缓存；`run_wick_analysis`、`calculate_contraction`（~1000 行） |
| `seqperam.py` | `seq_peram`——γ₅ 时间反演变换 |
| `baroperator.py` | `conjugate_operator`——强子算符 Hermitian 共轭（DR 基 C = γ₄γ₂） |
| `contractadviser.py` | `analyze_bandwidth`——roofline 模型带宽分析（只分析不计算） |
| `dynamic.py.bak` | dynamic.py 备份副本 |

## 用法（动态收缩）

```python
reg = PeramRegistry();  reg.register('light', ('tsink','tsrc'), peram)
vreg = VRegistry();     vreg.register('VVV_0', 'tsrc', vvv)
greg = GammaRegistry(); greg.register('gamma_5', g5)
dc = dynamic_contraction([(sink_op, src_op)], peram_registry=reg, ...)
```
