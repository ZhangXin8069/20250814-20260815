---
name: docker-v20260730
description: 运行 docker-v20260730 历史 GPU 管线 — 首个产生物理合理结果的大修版本 (donghx 对偶 F̃ OPE) (历史, 已被 v20260802+ 取代)
---

# docker-v20260730 — Historical GPU Pipeline (Major Bug-Fix Release)

在 /root/lattice-pdf/agent/docker-v20260730 运行 disconnected 胶子 PDF 验证管线（CuPy/CUDA，L24x72）。

**⚠️ 历史版本** — 首个产生物理合理结果的重要修复版本；其 OPE 算法（donghx 对偶 F̃）沿用至 v20260802。

## 本版本的 4 项关键修复

1. **Perambulator 二进制布局** — 修正为 `(Nt, Nspin=4, Nev, Nev)`（原来源/汇 Nev 轴颠倒，meff ~0.2 GeV 错误 → 修复后 ~1.78 GeV 正确）
2. **OPE 对偶场强 F̃ 缺失** — 修正为对偶 F̃ = 0.5·ε_{μνρσ}·F_{ρσ}（原来某些 μ₃ν₁ 分量小 10⁷ 倍）
3. **Levi-Civita (Tensor4) 符号错误** — `epsilon[3,1,0,2]` 符号翻转 → F̃[3,1] 符号错误
4. **OPE 空间求和轴** — 应求和所有空间轴 (z, y, x)，原只求垂直轴

**验证结果**：meff ~1.78 GeV，匹配期望 E = √(1.37² + 0.98²) ≈ 1.69 GeV。

## 固定配置

```
系综:   beta6.20_mu-0.2770_ms-0.2400_L24x72
格点:   72×24³, a=0.1053 fm, β=6.20
精度:   complex64 (单精度)
特征向量: per-config 二进制 (标准路径)
```

## 运行

```bash
cd /root/lattice-pdf/agent/docker-v20260730
python run_pipeline.py --conf-id 6250    # GPU 单精度
```

**不要用于新工作**；其 OPE 算法沿用至 `../docker-v20260802/`。
