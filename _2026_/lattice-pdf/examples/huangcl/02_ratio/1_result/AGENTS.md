# AGENTS.md — examples/huangcl/02_ratio/1_result

流水线步骤 2（`02_ratio`）的**生产输出**。由 `code.py`/`code_1.py` 以 `jack = True` 经 `sbatch submit.sh` 写出。这些是下游使用的分析结果（比值 R(z)、拟合）。**生成输出，勿手改。**

```
1_result/
└── L24x72/
    └── fit_Pz2_Nsam200_dtmax20_tsep*_*               # 每个生产拟合配置一个目录
```

调试/测试输出在兄弟目录 `0_debug/`。
