# AGENTS.md — examples/huangcl/02_ratio

黄 CL 分析流水线**步骤 2**：胶子 PDF 的 **3pt/2pt 比值**分析 R(z)，含 Jackknife 重采样、多参数拟合与 Chroma IOG 3pt 提取。读取步骤 0–1 产生的 OPE + 质子 2pt 数据。

## 文件

| 文件 | 用途 |
|---|---|
| `code.py` | 主比值计算：读 2pt + OPE、Jackknife、算 R(z) |
| `code_1.py` | 比值分析变体 |
| `submit.sh` | Slurm 提交脚本 |
| `9_others/Calc_3pt.py` | Chroma IOG 3pt 提取 |
| `9_others/Fit_Raito.py` | 多参数比值拟合 |
| `0_debug/` | 测试运行输出（debug 模式） |
| `1_result/` | 生产输出 |

## 运行

```bash
python code.py            # 登录节点调试（文件顶部设 debug = True，输出到 0_debug/）
sbatch submit.sh          # 生产（jack = True，输出到 1_result/）
```
