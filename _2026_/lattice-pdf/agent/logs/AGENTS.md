# AGENTS.md — agent/logs

docker-v*/snsc-v* 验证流水线与 agent 子模块的执行日志和调试报告。**gitignored 输出**，瞬态，可删除。

| 类型 | 文件 |
|---|---|
| 流水线日志 | `pipeline_*.log`、`run-*.log`、`diagnose_*.log` |
| 调试报告 | `debug-v20260802-*.md`、`diagnose_meff_*.log` |
| 流水线报告 | `docker-v20260803_*_report.md`、`result-summary-v20260802.md` |
| 子模块日志 | `lamet-agent/`、`lqcd-master/`、`pyquda/` |

这些是诊断产物而非权威文档，物理结论以 `agent/docker-v*` 代码为准。
