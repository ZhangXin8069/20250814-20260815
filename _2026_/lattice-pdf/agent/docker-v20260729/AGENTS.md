# AGENTS.md — agent/docker-v20260729

**历史版本——已被 `../docker-v20260802/` 取代。** 关键 GPU 版本，含算法修正。

| 特性 | 值 |
|---|---|
| 计算引擎 | GPU（CuPy） |
| 有效质量 | **fit_cosh + 多方法诊断** |
| 特征向量 smearing | **默认 OFF**（修正） |

## 三项关键修正

1. **特征向量 smearing 默认关闭**——`mz2_my0_mx0` 传播子已编码动量 smearing，再对 VVV 中特征向量施加 smearing 是错误的
2. **基于物理的动量投影**——VVV 相位用 `P_proj = P_phys`（传播子 smearing 与 Wick 收缩中 VVV smearing 抵消）
3. **有效质量改用拟合**（`fit_cosh`、`fit_exp`）——朴素 arccosh 在振荡/噪声关联函数上灾难性失效；四种方法同时计算供诊断

新增 `diagnose_2pt.py` 与 `report/diagnosis_report.md`。**不要用于新工作。**
