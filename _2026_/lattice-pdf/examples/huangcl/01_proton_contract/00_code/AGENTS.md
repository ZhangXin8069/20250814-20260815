# AGENTS.md — examples/huangcl/01_proton_contract/00_code

黄 CL 流水线步骤 1 的 GPU（CuPy）Python 脚本：蒸馏质子 2pt 关联函数。与计算 OPE 的步骤 0（`../00_contract/`）互补。

## 文件

`2pt_proton_Cg5gmu_L32x64_mom2_xdir_gpu.py`——动量涂抹质子 2pt 蒸馏（CuPy）。`conf_id` 从 `sys.argv[1]` 读取；Nt/Nx/Nev、宇称投影、插值算符 `_Cg5g4` 与集群数据路径（特征向量、传播子、VVV）硬编码。从 `examples/donghx/` import `gamma_matrix_cupy_DR` 与 `input_output_4_cupy`。

## 注意

- 与 `00_contract/00_code/` 中的脚本是兄弟副本（imports、`%`-style vs f-string 文件名格式化略异）。从 `examples/donghx/` 复制；本副本为流水线权威版本
- donghx 规范场张量约定 `[t, z, y, x, dir, color, color]`
- 虽文件名写 L32x64，脚本硬编码 Nx=32、Nt=64、Nev=100、mom_smear=3——集群数据路径指向 L32x64（β=6.20）系综
