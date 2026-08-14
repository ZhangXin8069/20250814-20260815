# AGENTS.md — examples/huangcl/00_contract/00_code

黄 CL 流水线步骤 0 的 GPU（CuPy）Python 脚本：OPE 计算与质子 2pt。这些是 donghx 脚本的 huangcl 副本（从 `examples/donghx/` 复制，可能有本地修改——本副本对本流水线有权威性）。

## 文件

| 文件 | 用途 |
|---|---|
| `2pt_proton_Cg5gmu_L32x64_mom2_xdir_gpu.py` | 质子 2pt 蒸馏（CuPy）。`conf_id` 从 `sys.argv[1]` 读取；Nt/Nx/Nev 与数据路径硬编码（`fileinput` stdin 块被注释） |
| `Calc_ope_unpol.py` | 非极化胶子 OPE（CuPy + mpi4py，3 rank）。stdin 读参数，import `Operator` |
| `Operator.py` | Plaquette/场强张量 F_{μν}（`plaquette_clover_all_new`、`operators_new_z0_mu2`） |

## 约定

- 规范场张量约定 `[t, z, y, x, dir, color, color]`（donghx 约定）
- `Calc_ope_unpol.py` 将 (μ,ν) 指标对跨 3 个 MPI rank 拆分
- 2pt 脚本从 `examples/donghx/` import `gamma_matrix_cupy_DR` 与 `input_output_4_cupy`
