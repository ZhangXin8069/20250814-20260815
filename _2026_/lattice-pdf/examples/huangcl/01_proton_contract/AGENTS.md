# AGENTS.md — examples/huangcl/01_proton_contract

黄 CL 分析流水线**步骤 1**：L24x72 系综经蒸馏 GPU 计算**质子 2pt 关联函数**。与计算 OPE 部分的步骤 0（`../00_contract/`）互补。

## 结构

```
01_proton_contract/
├── 00_code/          # GPU Python 脚本（CuPy）
│   └── 2pt_proton_Cg5gmu_L32x64_mom2_xdir_gpu.py   # 质子 2pt 蒸馏
└── 01_submit/        # Slurm 模板 + 任务生成
```

## 关键脚本

`2pt_proton_Cg5gmu_L32x64_mom2_xdir_gpu.py`——动量涂抹质子 2pt 蒸馏（CuPy）。stdin 键值参数（`Nt 72`、`Nx 24`、`Nev 100`、`Pz -2`）。从 `examples/donghx/` 复制；本副本为流水线权威版本。

## 运行

```bash
cd 01_submit
bash 01_create_task/multi.sh    # 生成逐组态任务并提交
```
