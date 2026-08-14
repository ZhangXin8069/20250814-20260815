---
name: docker-v20260729
description: 运行 docker-v20260729 历史 GPU 管线 — 修正特征向量 smearing/动量投影/meff 拟合 (历史, 已被 v20260802+ 取代)
---

# docker-v20260729 — Historical GPU Pipeline (Algorithmic Corrections)

在 /root/lattice-pdf/agent/docker-v20260729 运行 disconnected 胶子 PDF 验证管线（CuPy/CUDA，L24x72）。

**⚠️ 历史版本** — 引入三项关键算法修正；已被 `../docker-v20260802/` 取代。

## 三项关键修正

1. **特征向量 smearing 默认关闭** — perambulator 已编码动量 smearing，VVV 中再对特征向量 smearing 是错误的
2. **基于物理的动量投影** — VVV 相位用 `P_proj = P_phys`（perambulator smearing 在 Wick 收缩中与 VVV smearing 相消）
3. **有效质量用拟合**（`fit_cosh`, `fit_exp`）替代 naive arccosh — arccosh 在振荡/噪声关联函数上灾难性失败；四种方法同时计算用于诊断

新增 `diagnose_2pt.py` 独立诊断脚本和 `report/diagnosis_report.md`。

## 固定配置

```
系综:   beta6.20_mu-0.2770_ms-0.2400_L24x72
格点:   72×24³, a=0.1053 fm, β=6.20
精度:   complex64 (单精度)
特征向量: 共享 .npy (sunpeng 路径)
数据路径: sunpeng/
```

## 运行

```bash
cd /root/lattice-pdf/agent/docker-v20260729
python run_pipeline.py --conf-id 6250           # GPU 单组态
python run_pipeline.py --smear --conf-id 6250   # 启用 eigvec smearing
python diagnose_2pt.py --run-dir output_* --conf-id 6250
```

**不要用于新工作**，使用 `../docker-v20260802/` 或更新版本。
