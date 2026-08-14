# AGENTS.md — agent/docker-v20260803

生产导向 GPU 蒸馏流水线。概括 v20260802 方法：集中 `config.py` 管全部物理参数、`lib/` 分析库改编自 `examples/sush/lqcddb`、动态 Wick 收缩、Jackknife 分析、LaTeX 物理报告。

**版本身份**：v20260803 = 集中 config + sush/lqcddb 风格 `lib/` 库 + 多强子（pion/proton）2pt + 3pt/2pt 比值 + LaTeX 报告。

## 与 v20260802 主要差异

| 方面 | v20260802 | v20260803 |
|---|---|---|
| 配置 | flags + run_config.json | 集中 `config.py` |
| 分析代码 | 流水线内嵌 | 可复用 `lib/` 包 |
| NEV 默认 | 100 | 50（GPU 测试降低；生产设 100） |
| 强子 | 仅质子 | pion + proton，多动量 |
| 报告 | Markdown | LaTeX `physics_report.tex` → PDF |

## 关键文件

`config.py`（集中配置）、`run_pipeline.py`（6 步调度）、`run_single_config.py`、`full_analysis.py`、`ratio_analysis.py`、`utils.py`、`lib/`（lqcddb 改编分析库）、`physics_report.tex`、`data/`+`plots/`+`logs/`（gitignored 输出）。
