# AGENTS.md — agent/docker-v20260802

现行经典版 GPU 加速断连胶子 PDF 验证流水线。合并 v20260730 的正确 OPE 算法（donghx 对偶场强）+ v20260731 的双精度改进，另加 30 项 bug 修复。

**版本身份**：v20260802 = v20260730 正确 OPE + v20260731 双精度 + 30 修复

| 特性 | 状态 |
|---|---|
| OPE 算法 | ✓ donghx（对偶 F̃ = 0.5·ε·F + roll 式 Wilson 线） |
| 精度 | complex128（可切 complex64） |
| 特征向量 | 逐组态二进制，自动检测 |
| 传播子读取 | v20260730 验证布局：(Nt,Nev,Nspin,Nev,2) → complex |

## 关键文件

`run_pipeline.py`（主调度，4 步，计时/内存跟踪）、`compute_2pt_gpu.py`（GPU 质子 2pt 蒸馏：VVV 重子块 + Wick）、`compute_ope_gpu.py`（GPU OPE：Clover F→对偶 F̃→Wilson 线→迹）、`analyze_ratio.py`（jackknife 比值）、`gamma_matrix_gpu.py`、`utils.py`、`check_env.py`、`diagnose_2pt.py`、`run_config.json`、`README.md`。

## 使用

```bash
python check_env.py
python run_pipeline.py --conf-id 6250 --skip-ope --skip-analysis -v    # 快速 2pt 测试（~8 分钟）
python run_pipeline.py --conf-id 6250 --precision complex64             # 单精度
```
