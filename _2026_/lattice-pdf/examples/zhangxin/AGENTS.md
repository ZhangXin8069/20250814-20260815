# AGENTS.md — examples/zhangxin

张 X 的胶子 PDF 工作流与数据分析框架。两个独立工作流实现 + 综合分析工具包。

## 两个工作流实现

### `gluon_pdf_full_workflow.py` — 自包含流水线（~1900 行）

单脚本全 10 步，dataclass 配置，argparse。适合理解完整算法或单机运行。

```bash
python gluon_pdf_full_workflow.py                                     # 默认参数
python gluon_pdf_full_workflow.py --Pz 6 --conf 20000 --delta_z 15    # 自定义
```

### `gluon_pdf_workflow.py` — 模块化 + MPI（~1600 行）

生产 HPC 运行：子命令、内建系综预设、mpi4py、向后兼容 stdin。

```bash
python gluon_pdf_workflow.py 2pt --ensemble L32x64 --conf-id 20000
python gluon_pdf_workflow.py ope --ensemble L32x64 --conf-id 20000 --gauge-file config.dat
mpirun -np 4 python gluon_pdf_workflow.py ope --ensemble L32x64 --conf-id 20000 --gauge-file config.dat
```

**内建系综预设**（`ENSEMBLES`，行 107–151）：L24x72(β6.20, 100)、L32x64(β6.20, 100)、L32x96(β6.41, 100)、L36x108(β6.498, 200)、L48x96(β6.20, 200)、L48x144(β6.72, 200)。生产用此实现；理解算法/小格点/单机用 `gluon_pdf_full_workflow.py`。

## 分析框架

- `include.py`（~900 行）：`data_analyse` 类，读 .dat 与 Chroma IOG 二进制，jackknife、有效质量、比值、Wilson 线分析
- `main.py`（pion 3pt 比值）、`main-2pt.py`（cosh 有效质量）、`main-3pt.py`（IOG 提取）、`main_iog.py`、`_main.py`、`lsq_tools.py`
- `iog_reader/`：IOG 二进制 C 扩展（`iog.so` 需在集群编译）

## 注意

- 张量约定：规范场 `[color,color,dir,x,y,z,t]`、费米场 `[spin,color,x,y,z,t]`（与 donghx 不同）
- 依赖：`lsqfit`、`gvar`、`sympy`、`matplotlib`、`proplot`、`h5py`
