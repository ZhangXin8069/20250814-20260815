# AGENTS.md — examples/huangcl

黄 CL 的 L24x72 系综数据 4 步分析流水线：CuPy GPU 计算（步骤 0–1）+ Chroma IOG 二进制分析（步骤 2、4）。

## 流水线步骤

| 步骤 | 目录 | 计算 | 分析 |
|---|---|---|---|
| 0 | `00_contract/` | CuPy OPE（Clover F_{μν}、Wilson 线）+ 质子 2pt | — |
| 1 | `01_proton_contract/` | CuPy 质子 2pt 蒸馏 | — |
| 2 | `02_ratio/` | — | 3pt/2pt 比值 R(z)+jackknife、多参数拟合、Chroma IOG 3pt 提取 |
| 4 | `04_proton_energy/` | — | IOG 2pt 质子有效能量、cosh meff + 平台拟合 |

（步骤 3 跳过——中间分析，无最终结果。）

## 目录模式

每步：`00_code/`（Python 脚本）、`01_submit/`（Slurm 模板 + `multi.sh` 任务生成）、`02_input/`（生成的逐组态输入）。模板用占位符（`=NT=`、`=NX=`、`=CONF=` 等）经 `multi.sh` sed 替换。数据路径指向集群（`/public/...`），仅集群可解析。

**重复脚本**：`2pt_proton_Cg5gmu_L32x64_mom2_xdir_gpu.py` 与 `Calc_ope_unpol.py`/`Operator.py` 同时存在于 `examples/donghx/` 与 huangcl 的 `00_contract/00_code/`——huangcl 副本为本流水线所用版本，可能独立修改。
