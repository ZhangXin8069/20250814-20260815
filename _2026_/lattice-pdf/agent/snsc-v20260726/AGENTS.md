# AGENTS.md — agent/snsc-v20260726

CPU（NumPy）断连胶子 PDF 验证流水线——`agent/snsc/` 的精简版，移除 AI agent 阶段（LQCD_Master + lamet-agent）直接跑物理计算。已被 GPU 流水线 `../docker-v20260802/` 取代。

## 文件

`run_pipeline.py`（4 步调度）、`compute_2pt.py`（质子 2pt 蒸馏：特征向量+传播子、VVV 重子块、Wick、宇称投影）、`compute_ope.py`（ILDG `.lime` 规范场、Clover F_{μν}、非定域胶子 OPE）、`analyze_ratio.py`（huangcl 式比值 C3_disc/C2、jackknife）、`gamma_matrix.py`、`utils.py`、`run_config.json`、`download_data.sh`（SSH/rsync 下载集群数据）、`sbatch.sh`（8 CPU、24h、conda `zhangxin-snsc`）。

## 使用

```bash
python run_pipeline.py
python run_pipeline.py --skip-2pt --skip-ope --skip-analysis --skip-report
sbatch sbatch.sh
bash download_data.sh --yes --skip-existing
```
