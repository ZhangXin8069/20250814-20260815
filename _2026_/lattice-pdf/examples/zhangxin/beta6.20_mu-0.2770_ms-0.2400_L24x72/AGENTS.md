# AGENTS.md — examples/zhangxin/beta6.20_mu-0.2770_ms-0.2400_L24x72

**β=6.20、L24x72** 亮夸克系综上的 pion 3 点关联函数生产工作流（Chroma 格点 QCD 代码）。生成 Chroma XML、经 Slurm 提交 GPU 作业、产生 IOG 二进制输出供后续分析。

## 系综参数

β=6.20、m_light=-0.2770、m_strange=-0.2400、24³×72、a≈0.105 fm、组态 10000–12950（步长 50，60 组态）。

## 文件

`main.py`（驱动，生成 Chroma XML）、`creat_chroma.py`（XML 生成类 ~22KB：规范场读取、stout 涂抹、传播子、强子谱、顺序源）、`run.sh`（Slurm：2 GPU、2 MPI rank、CUDA 11.4 + OpenMPI 4.1.5 + Python 3.9.10）、`ssub.sh`（批量提交：60 顺序作业，最多 2 个 pending）、`clean.sh`、`XMLDAT`。

## 工作流

`main.py → Chroma XML → mpirun -n 2 chroma → IOG 二进制输出`
