# AGENTS.md — snsc

胶子 PDF 流水线统一生产入口，整合 `examples/donghx/`、`examples/zhangxin/`、`examples/huangcl/`，三种分析模式。

## 入口：main.py

```bash
# 模式1 PDF：完整 LaMET 流水线（13 步可选）
python main.py --analysis-type pdf --steps 1,2,3,6 --xp numpy --ensemble L32x64 \
    --conf-start 20000 --conf-num 1 --gauge-file /path/to/config.dat
# 模式2 proton-2pt：蒸馏 VVV + Wick 收缩
python main.py --analysis-type proton-2pt --Nt 72 --Nx 24 --Pz-list "-2,-3,-4,-5,-6" --conf-start 46000
# 模式3 2pt：IOG 数据有效质量
python main.py --analysis-type 2pt --Nt 72 --Nx 24 --alttc 0.1053 --conf-start 10050 --conf-step 50 --conf-num 52
```

## Slurm（必须在本目录提交，脚本用 `#SBATCH --chdir`）

```bash
sbatch sbatch.sh       # PDF 全流水线（CPU）
sbatch sbatch-2pt.sh   # 质子 2pt 蒸馏（CPU）
```

## 关键特性

- `--xp numpy|cupy` 后端选择、`--dtype complex64|complex128`
- VVV 自动缓存（`.npy`）；时间戳输出 `output_YYYYMMDD_HHMMSS/{data/,plots/,run_config.json}`
- 依赖：proton-2pt 模式需 `opt_einsum`、`numpy`、`matplotlib`；2pt 模式需编译的 `examples/zhangxin/iog_reader/iog.so` 与 `include.py`
- 集群默认 conda 环境 `zhangxin-snsc`

## 流水线步骤

1 规范场I/O · 2 Plaquette+Clover F_{μν} · 3 OPE 算符（Wilson 线+迹） · 4 特征向量I/O · 5 VVV 重子块 · 6 Wick 收缩 · 7 质子 2pt · 8 3pt/2pt 比值 R(z)+jackknife · 9 傅里叶变换(z→x) · 10 微扰匹配 · 11 连续极限 · 12 g(x,μ) 输出 · 13 绘图诊断
