# AGENTS.md — agent/文档

三个 AI agent 子模块的中文 LaTeX/PDF 技术文档。

| 文件 | 库 | 内容 |
|---|---|---|
| `LQCD_Master_Documentation.tex`/`.pdf` | LQCD_Master | NL→PyQUDA agent：Planner→Executor 流水线、基准任务、CLI 参考 |
| `lamet_agent_Documentation.tex`/`.pdf` | lamet-agent | LaMET 分析 agent：5 阶段流水线、manifest JSON schema、关联函数分析 |
| `PyQUDA_Documentation.tex`/`.pdf` | PyQUDA | Python QUDA 包装：HMC、smearing、梯度流、multigrid、I/O |

## 编译

```bash
# 从 agent/ 目录
./update_docs.sh              # 编译全部三个 PDF
./update_docs.sh LQCD_Master  # 只编译一个
```

文档描述的子模块位于 `../LQCD_Master/`、`../lamet-agent/`、`../PyQUDA/`（独立 git 仓库，文档放在本目录而非子模块内）。
