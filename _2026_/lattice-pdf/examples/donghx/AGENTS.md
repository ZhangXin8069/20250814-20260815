# AGENTS.md — examples/donghx

董 HX 的断连胶子 PDF 计算代码（LaMET）：
1. **质子 2pt 关联函数**——动量涂抹蒸馏
2. **OPE 算符**——场强张量 F_{μν} + Wilson 线构成的非定域胶子算符

## 命名约定

```
2pt_proton_Cg5gmu_{L}x{T}_mom{m}_{dir}_{backend}.py
```

- `{L}x{T}`：格点尺寸（L24x72、L32x64、L32x96、L36x108、L48x96、L48x144）
- `mom{m}`：动量（2π/L 单位），mom0/mom2；`{dir}`：xdir/ydir/zdir
- `{backend}`：`gpu`=CuPy、`dcu`=DCU、无后缀=CPU

## 关键文件

| 文件 | 用途 |
|---|---|
| `2pt_proton_Cg5gmu_*.py` | ~52 脚本：各格点/动量/方向/后端的质子 2pt 蒸馏 |
| `Calc_ope_unpol.py` / `_new.py` | 非极化胶子 OPE（CuPy，非 MPI / MPI） |
| `Calc_ope_helicity*.py` | 螺旋度（极化）胶子 OPE |
| `Calc_pla.py`、`Calc_VVV.py` | 方块、VVV 收缩 |

**参数用 stdin 重定向传递**（`fileinput.input()` 键值对），非 argparse。数据路径为集群符号链接（本机不解析）。规范场张量约定 `[t,z,y,x,dir,color,color]`。
