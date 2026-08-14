# AGENTS.md — examples/huangcl/04_proton_energy

黄 CL 分析流水线**步骤 4**：从 Chroma IOG 2pt 关联函数提取**质子有效能量**，含 cosh 有效质量与平台拟合。

## 文件

| 文件 | 用途 |
|---|---|
| `code.py` | 有效能量提取：读 IOG 2pt、cosh meff、平台拟合 |
| `submit.sh` | Slurm 提交脚本 |
| `0_debug/` | 测试运行输出（debug 模式） |
| `1_result/` | 生产输出 |

## 运行

```bash
python code.py            # 登录节点调试（debug = True → 0_debug/）
sbatch submit.sh          # 生产（jack = True → 1_result/）
```

`code.py` 顶部的 `debug`/`jack` 开关控制输出位置。依赖：编译的 IOG `.so`（见 `examples/zhangxin/iog_reader/`）。
