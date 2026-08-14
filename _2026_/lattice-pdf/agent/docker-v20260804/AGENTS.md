# AGENTS.md — agent/docker-v20260804

综合 GPU 加速**全量蒸馏工具包**：顶点函数（VdV/VVV）+ 多强子 Wick 收缩（pion/proton/neutron 2pt、OPE loop、PJN 3pt、PJNNJNp 4pt）+ 完整统计分析（Jackknife、有效质量、比值、平台拟合），全部 CuPy/CUDA。

**版本身份**：v20260804 = 全量蒸馏工具包。基于 LQCD_Master、lamet-agent、examples/sush/lqcddb、docker-v20260802 模式构建。至今功能最全的 docker-v* 版本。

## 版本对比（最近三个）

| 版本 | 范围 |
|---|---|
| v20260802 | 质子 2pt + 胶子 OPE + huangcl 比值 |
| v20260803 | 集中 config + sush `lib/` 库，pion+proton，LaTeX 报告 |
| **v20260804** | 全工具包：4 强子道 + 3pt/4pt + 验证目标 |

## 关键文件

`run_pipeline.py`（6 步调度）、`check_env.py`、`utils.py`、`gamma_matrix_gpu.py`、`data_io.py`、`compute_vertex.py`、`compute_contraction.py`、`analyze.py`、`README.md`、`output/`（gitignored）。

## 使用

```bash
python check_env.py
python run_pipeline.py --conf-id 6250 --skip-4pt    # 单组态测试（~15 分钟）
```
