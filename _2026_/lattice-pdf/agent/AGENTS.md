# AGENTS.md — agent

AI agent 子模块与 GPU 验证流水线目录。

| 目录 | 类型 | 说明 |
|---|---|---|
| `docker/` | Launcher | GPU 流水线启动器 + 数据下载/打包脚本 |
| `docker-v20260805/` | Pipeline ★现行 | 全量蒸馏工具包：自包含 `lib/` + 集中 `config.py`、VdV/VVV 顶点、pp/pn/pion 2pt、donghx OPE、PJN 3pt、PJNNJNp 4pt、code_1.py 统计、LaTeX 报告；10 组态 |
| `docker-v20260804/` | Pipeline | 全量蒸馏工具包（VdV/VVV、多强子 2pt/3pt/4pt、Jackknife） |
| `docker-v20260803/` | Pipeline | 集中 config + sush `lib/` 库、pion+proton、LaTeX 报告 |
| `docker-v20260726`–`v20260802` | Pipeline（旧） | 早期 GPU 流水线版本（历史） |
| `snsc/` | 验证 | LQCD_Master + lamet-agent 联合验证运行 |
| `snsc-v20260725/26/` | Pipeline（旧） | 早期 CPU 集群流水线版本 |
| `docs/` | 参考 | 21 篇 AI-for-science + 格点 QCD 论文（PDF） |
| `文档/` | 文档 | 三个子模块的 LaTeX/PDF 技术文档 |
| `LQCD_Master/` `lamet-agent/` `PyQUDA/` | 子模块 | 上游 AI/GPU 库（git submodule，勿改动） |
| `logs/` | 日志 | 执行日志与调试报告（gitignored，瞬态） |

## 快速开始

```bash
cd docker-v20260805
python run_pipeline.py --conf-id 6250 --skip-3pt --skip-4pt --skip-report   # 冒烟测试
# 或经 launcher
cd docker && bash run_gpu_pipeline.sh test    # ~3 分钟
```

标准数据路径（集群）：特征向量 `/public/group/lqcd/eigensystem/...`、传播子 `/public/group/lqcd/perambulators/...`、规范场 `/public/group/lqcd/configurations/CLOVER/...`。

## 各版本技能

每个版本目录有对应单文件技能 `docker-vYYYYMMDD.skill.md`（扁平副本）；版本间差异与历史教训见各目录 CLAUDE.md（已归档为 `.CLAUDE.md.<ts>.bak`，保留供参考）。
