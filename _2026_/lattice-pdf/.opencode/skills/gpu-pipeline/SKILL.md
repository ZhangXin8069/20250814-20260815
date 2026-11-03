---
name: gpu-pipeline
description: 一键运行 GPU 加速非极化胶子 PDF 全流程验证管线 (CuPy/CUDA, 蒸馏2pt+OPE从头计算+huangcl分析)
---

# gpu-pipeline — GPU 加速胶子 PDF 验证管线

一键在当前环境运行完整的 disconnected 胶子 PDF 验证管线：
**蒸馏两点函数 → OPE 从头计算 → huangcl 比率分析 → 图表 → 报告**。

## 核心特征

- **GPU 加速**：VVV、Wick 收缩、F_{μν}、Wilson 线、OPE 缩并全部在 GPU 上执行
- **单精度 (complex64)**：默认，内存减半，满足当前统计需求
- **OPE 从头计算**：gauge config (.lime) → Clover F_{μν} → Wilson 线 → OPE .npz
- **所有中间变量保存**：VVV、F_{μν}、Wick 收缩、有效质量
- **4 张诊断图**：ratio、diagnostics、effective mass、field strength
- **自动报告**：Markdown 报告 + LaTeX 技术报告

## 固定参数

```
系综:   beta6.20_mu-0.2770_ms-0.2400_L24x72 (L24x72)
格点:   72×24³, a=0.1053 fm, β=6.20
Nev:    100 (per-conf eigensystem)
组态:   6250, 6450, 6650 (Nconf=3)
动量:   P=(0, 0, -2), mom_smear=-2
算符:   _Cg5g4 (Cγ₅γ₄)
GPU:    CUDA/CuPy, complex64, 自动检测设备
```

## 子命令

### `gpu-pipeline run` — 运行完整 GPU 管线

```bash
cd /root/lattice-pdf/agent/docker-v20260727

# 完整运行 (3 组态, ~8.5 min)
python run_pipeline.py

# 单组态快速测试 (~3 min)
python run_pipeline.py --conf-id 6250

# 双精度模式 (complex128, ~60 min)
python run_pipeline.py --conf-id 6250 --precision complex128

# 仅重新生成图表 (已有数据)
python run_pipeline.py --skip-2pt --skip-ope

# 详细调试输出
python run_pipeline.py --conf-id 6250 --verbose
```

### `gpu-pipeline check` — GPU 环境与数据检查

```bash
cd /root/lattice-pdf/agent/docker-v20260727 && python3 -c "
import sys, os
sys.path.insert(0,'.')

# ── GPU ──
import cupy as cp
d = cp.cuda.Device(); mem = d.mem_info
props = cp.cuda.runtime.getDeviceProperties(d.id)
name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
print(f'GPU: {name}  |  Mem: {mem[0]/1024**3:.1f}/{mem[1]/1024**3:.1f} GB free')
print(f'CuPy: {cp.__version__}  |  CUDA: {cp.cuda.runtime.runtimeGetVersion()}')

# ── Python modules ──
for mod in ['numpy','scipy','matplotlib','cupy']:
    m = __import__(mod); print(f'  {\"✓\"} {mod}: {getattr(m,\"__version__\",\"?\")}')

# ── Data ──
paths = {
    'eig_6250': '/public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/6250/',
    'eig_6450': '/public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/6450/',
    'eig_6650': '/public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/6650/',
    'gauge_6250': '/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_6250.lime',
    'peram_6250': '/public/group/lqcd/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/6250/',
}
for name, p in paths.items():
    ok = os.path.exists(p)
    sz = ''
    if ok and os.path.isfile(p): sz = f' ({os.path.getsize(p)/1024**3:.1f} GB)'
    elif ok and os.path.isdir(p): sz = f' ({len(os.listdir(p))} files)'
    print(f'  {\"✓\" if ok else \"✗\"} {name}: {p}{sz}')
"
```

### `gpu-pipeline status` — 查看最新运行状态

```bash
latest=$(ls -dt /root/lattice-pdf/agent/docker-v20260727/output_*/ 2>/dev/null | head -1)
[ -n "$latest" ] && {
    echo "=== ${latest} ==="
    echo "Timing:"
    grep "elapsed=" "$latest/run.log" | grep -E "01_compute|02_compute|03_huang|Total time"
    echo ""
    echo "Summary:"
    grep -E "Pipeline Complete|ALL OK|✗|ERROR|Ratio" "$latest/run.log" | tail -10
    echo ""
    echo "Plots:"
    ls -lh "$latest/plots/"*.png 2>/dev/null
} || echo "No output found."
```

### `gpu-pipeline plots` — 仅重新生成图表

```bash
cd /root/lattice-pdf/agent/docker-v20260727
latest=$(ls -dt output_*/ 2>/dev/null | head -1)
[ -z "$latest" ] && { echo "No output found. Run gpu-pipeline run first."; exit 1; }
python3 -c "
import sys; sys.path.insert(0,'.')
from analyze_ratio import run_analysis
from utils import setup_logging
import json
from pathlib import Path
config = json.load(open('run_config.json'))
data_dir = Path('$latest') / 'data'
plots_dir = Path('$latest') / 'plots'
logger = setup_logging(Path('$latest') / 'run.log', 'plot_fix')
results = run_analysis(config, data_dir, plots_dir, logger)
print(f'Status: {results[\"status\"]}')
"
```

### `gpu-pipeline report` — 编译技术报告

```bash
cd /root/lattice-pdf/agent/docker-v20260727
xelatex -interaction=nonstopmode -halt-on-error \
    -output-directory=report report/main.tex 2>/dev/null
xelatex -interaction=nonstopmode -halt-on-error \
    -output-directory=report report/main.tex 2>/dev/null
echo "Report: $(ls -lh report/main.pdf | awk '{print $5}')"
```

### `gpu-pipeline clean` — 清理所有输出

```bash
echo "Cleaning all pipeline outputs..."
rm -rf /root/lattice-pdf/agent/docker-v20260727/output_*/
rm -f /root/lattice-pdf/agent/docker-v20260727/report/main.{aux,log,out,toc,pdf}
echo "Done."
```

### `gpu-pipeline package` — 打包输出

```bash
cd /root/lattice-pdf/agent/docker-v20260727
latest=$(ls -dt output_*/ 2>/dev/null | head -1 | sed 's:/$::')
[ -z "$latest" ] && { echo "No output to package."; exit 1; }
tar -czf "${latest}.tar.gz" "$latest"/{plots,run.log,timing.jsonl,final_report.md,run_config.json,gpu_info.json}
# Compile report
xelatex -interaction=nonstopmode -halt-on-error -output-directory=report report/main.tex >/dev/null 2>&1
xelatex -interaction=nonstopmode -halt-on-error -output-directory=report report/main.tex >/dev/null 2>&1
[ -f report/main.pdf ] && cp report/main.pdf "$latest/report.pdf"
echo "Packaged: ${latest}.tar.gz ($(du -h ${latest}.tar.gz | cut -f1))"
echo "Report:   $latest/report.pdf"
```

## 管线步骤详解

| 步骤 | 描述 | GPU 加速 | 典型耗时 (Nconf=1/Nconf=3) |
|------|------|----------|---------------------------|
| 0 | 环境检查 | — | <1s / <1s |
| 1 | 质子 2pt 蒸馏 | VVV, Wick 收缩 | 125s / 389s |
| 2 | OPE 从头计算 | F_{μν}, Wilson 线, 缩并 | 42s / 120s |
| 3 | huangcl 比率分析 | Jackknife 重采样 | 3s / 3s |
| 4 | 最终报告 | — | <1s / <1s |

## 输出结构

```
output_YYYYMMDD_HHMMSS/
├── run.log                      # 完整日志
├── timing.jsonl                 # 每步耗时与显存
├── final_report.md              # Markdown 综合报告
├── gpu_info.json                # GPU 设备信息
├── data/
│   ├── eigenvalues_Nev100.npy
│   ├── conf_6250/               # VVV, F_{μν}×3, OPE×3, 2pt, meff
│   ├── conf_6450/
│   └── conf_6650/
└── plots/
    ├── ratio.png                 # R(z) 比率图
    ├── ratio_diagnostics.png     # 多 z 值 Re/Im 诊断
    ├── effective_mass.png        # 有效质量 3-panel
    └── field_strength_diagnostics.png  # OPE 质量诊断
```

## 精度选择

| 选项 | 精度 | GPU 显存 | 耗时 | 适用场景 |
|------|------|---------|------|---------|
| `--precision complex64` (默认) | 单精度 float32 | 低 | 快 | 快速验证、统计研究 |
| `--precision complex128` | 双精度 float64 | 高 (2×) | 慢 (~20×) | 精密谱学 |

当前统计误差 ($N_{\rm conf}=3$) 远大于 complex64 的精度损失 ($\sim 10^{-3}$)，
默认单精度完全满足需求。

## 已知限制与改进方向

1. **$N_{\rm conf}=3$ 统计不足**：disconnected 图需 $\gtrsim 100$ 组态
2. **ILDG 头部扫描**：当前扫描 760 个偏移量，可缓存正确偏移
3. **Wick 收缩占 58%**：可融合为 CUDA kernel 减少中间张量分配
