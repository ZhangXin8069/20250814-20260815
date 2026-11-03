---
name: run-docker-pipeline
description: 运行 Docker GPU 验证管线 (CuPy/CUDA, L24x72 disconnected 胶子 PDF) — 聚合入口, 按版本调用 docker-vYYYYMMDD 技能
---

# run-docker-pipeline — Docker GPU Pipeline (aggregate)

运行 GPU 加速的 disconnected 胶子 PDF 验证管线（CuPy/CUDA，真实 L24x72 格点数据）。

**★ 当前最新版：`docker-v20260805`**（全量蒸馏工具包：自包含 lib/ + 顶点 + 多强子 2pt/3pt/4pt + OPE + code_1.py 统计 + LaTeX 报告，10 组态）。

## 每个版本一个完整技能

在 `/root/lattice-pdf/agent/.claude/skills/` 下，**每个版本目录对应一个完整技能**，直接按版本名调用即可：

| 版本 | 技能 | 说明 |
|------|------|------|
| v20260805 | `docker-v20260805` | ★当前 — 全量蒸馏工具包 (自包含 lib/ + 10 组态 + code_1.py 统计 + LaTeX 报告) |
| v20260804 | `docker-v20260804` | 完整蒸馏工具包 (顶点 + 多强子 Wick + 统计分析) |
| v20260803 | `docker-v20260803` | 集中 config.py + sush lib/ + LaTeX 报告 |
| v20260802 | `docker-v20260802` | 经典 — donghx 正确 OPE + complex128 + 30 修复 |
| v20260801 | `docker-v20260801` | 自动平台 meff + Pz=0 校准 (历史) |
| v20260731 | `docker-v20260731` | 首个双精度 (complex128) (历史) |
| v20260730 | `docker-v20260730` | 首个物理合理结果 (donghx 对偶 F̃) (历史) |
| v20260729 | `docker-v20260729` | 修正 smear/动量投影/meff 拟合 (历史) |
| v20260728 | `docker-v20260728` | CPU 有效质量修复 (历史) |
| v20260727 | `docker-v20260727` | 首个 GPU 移植 (历史) |
| v20260726 | `docker-v20260726` | CPU 基线 (历史) |

## 快速使用（当前版）

```bash
cd /root/lattice-pdf/agent/docker-v20260805
python run_pipeline.py --conf-id 6250 --skip-3pt --skip-4pt --skip-report   # 单组态冒烟测试
python run_pipeline.py                               # 完整 10 组态运行 (~5.2h)
```

详细命令、固定配置、数据路径见对应版本的 `docker-vYYYYMMDD` 技能文件或该目录的 `CLAUDE.md`。

## 数据路径（所有版本共用）

| 数据 | 路径 |
|------|------|
| Eigenvectors | `/public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/` |
| Perambulators | `/public/group/lqcd/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/` |
| Gauge configs | `/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/` |

（v20260726/27/28 例外：使用 `sunpeng/` 旧数据路径，见各自技能。）
