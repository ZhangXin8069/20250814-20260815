# AGENTS.md — examples/huangcl/00_contract

黄 CL 分析流水线**步骤 0**：用 CuPy 在 L24x72 系综 GPU 计算 **OPE 部分**与**质子 2pt 关联函数**。

## 结构

```
00_contract/
├── 00_code/          # GPU Python 脚本（CuPy）
├── 01_submit/        # Slurm 模板 + 任务生成（multi.sh）
│   ├── 00_template/  # 占位 input/submit 模板（=NT=、=CONF= 等）
│   └── 01_create_task/multi.sh   # 模板替换 → 02_input/
└── 02_input/         # 生成的逐组态任务目录（如 L24x72/）
```

## 关键脚本（00_code/）

`2pt_proton_Cg5gmu_L32x64_mom2_xdir_gpu.py`（动量涂抹质子 2pt 蒸馏，CuPy，stdin 参数）、`Calc_ope_unpol.py`（F_{μν} + Wilson 线非极化胶子 OPE，CuPy）。完整流水线见父 `../CLAUDE.md`（已归档）。
