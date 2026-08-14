# AGENTS.md — examples/sush/lqcddb/src/lqcddb/io

lqcddb 的输入/输出：蒸馏特征向量与传播子的二进制读取器，以及 ASCII 关联函数输出。

## 文件

`write_date.py`：`readin_eigvecs(file_path, Nx)` → `(Nev, Nx³, 3)` complex；`readin_peram(peram_dir, conf_id, Nt, Nev1)` → `(Nt, Nt, 4, 4, Nev1, Nev1)` complex；`write_data_ascii(data, T, L, filename, complex=True)`（L. Liu 格式）；`check_dir_path`；`safe_save`。

## 用法

```python
# readin_eigvecs / readin_peram 不在顶层导出——直接导入模块：
from lqcddb.io.write_date import readin_eigvecs, readin_peram
eigvecs = readin_eigvecs("eigvecs.dat", Nx=32)
# write_data_ascii 在顶层：
from lqcddb import write_data_ascii
```
