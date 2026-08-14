# AGENTS.md — examples/sush/lqcddb/src/lqcddb/analyse

lqcddb 统计分析子包：重采样（Jackknife/Bootstrap）、有效质量提取、GEVP 求解、3pt/2pt 比值（PDF）分析。所有函数经 `get_backend()` 操作，支持 numpy/cupy。

## 文件

`analyse.py`（核心：导出 `Jackknife`、`Bootstrap`、`meff`、`ratio_3pt`、`solve_gevp`、`dis_connect`、`loop_tsrc`、`Mom2GeV` 等）、`__init__.py`/`__init__.pyi`、`test_pion_3pt_*`（空输出目录）。

## 关键用法

```python
from lqcddb import Jackknife, meff
jk = Jackknife(corr_data, Nconf_axes=0)
mean, err = jk['data_mean'], jk['data_err']   # 可选 cov_axes → data_cov
```

详细 API 参考：`../../README.md` 第 8 节。
