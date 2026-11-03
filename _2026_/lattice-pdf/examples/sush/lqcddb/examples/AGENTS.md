# AGENTS.md — examples/sush/lqcddb/examples

演示用 `lqcddb` 包在真实格点 QCD 数据上计算强子关联函数的示例收缩脚本。**HPC 集群生产脚本**——从集群路径 import、读真实传播子/特征向量数据。

## 命名约定

```
contraction.{hadrons}.{Cpt}.{backend}.py
```

- **强子**：`pn`(p-n)、`pp`(p-p)、`nn`/`NN`(n-n)、`NJNp-`(核子-流-核子撇)、`PJN`/`PJJNp-`(带流插入)、`Np.I1.5` 等（同位旋投影）
- **Cpt**：`.2pt.`/`.3pt.`/`.4pt.`；**后端**：`.cupy.`(GPU)/`.numpy.`(CPU)
- **后缀**：`.MPI.`、`.GEVP.`、`.eqopt.`、`.verify.`

每个 `.py` 有对应 `.sh` Slurm 提交脚本。所有脚本需 HPC 集群环境（数据在集群文件系统）。
