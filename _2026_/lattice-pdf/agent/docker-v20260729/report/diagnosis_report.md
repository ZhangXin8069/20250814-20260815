# 2pt 有效质量诊断报告 — docker-tv20260729

## 诊断日期: 2026-07-27

## 问题描述
docker-v20260727 中 2pt 给出的 effective mass 明显错误 (~8-10 GeV)，
实际应为 ~1.2-1.4 GeV (E = √(m_proton² + p²), p = 2π·2/(24·0.1053)·0.1973 ≈ 0.98 GeV)

## 诊断过程

### 1. 算法对比 (docker-v20260727 vs donghx DCU 参考代码)

逐行对比了以下算法组件，确认实现与 donghx 参考代码一致:
- ✓ Gamma 矩阵 (DeGrand-Rossi 基)
- ✓ 动量相位因子 (phase factor)
- ✓ Perambulator 读写
- ✓ VVV 重子块计算 (含 epsilon 张量缩并)
- ✓ Wick 缩并 (Direct - Exchange, 分解为 4 步)
- ✓ 宇称投影 (P_plus / P_minus)
- ✓ 边界符号修正 (anti-periodic BC)
- ✓ 动量涂抹约定

### 2. 原始数据诊断

#### 2.1 Perambulator 数据质量
- 文件存在且大小正确 (43.9 MB/file, 4 files per time slice)
- 格式正确: shape (72, 4, 4, 100, 100), dtype complex64
- **关键发现: Perambulator 迹 (trace) 严重振荡**
  - 单个本征矢量元素的 τ_aa(t, 0) 有 30-40 次符号翻转 (72 个时间片)
  - 低模 (a=0) 和高模 (a=99) 均振荡
  - 这是动量涂抹 (mz=2) 导致的: perambulator 包含 exp(±i P_smear·x) 相位,
    不同空间位置贡献不同相位 → 时间方向振荡

#### 2.2 2pt 关联函数
- Raw PP 矩阵: |max| ≈ 2×10⁻⁵, 正负对称分布
- 源平均 C2pt_1d: range ≈ [-1.4×10⁻⁶, 1.8×10⁻⁷]
- **符号翻转: 12-17 次 (dt ∈ [2,32])** — 无干净的指数衰减
- 前向 C2pt_forward 同样振荡
- 边界条件修正 (有/无) 不能消除振荡

### 3. 有效质量提取测试

| 条件 | fit_cosh | fit_exp | exp_forward | cosh (原始) |
|------|----------|---------|-------------|-------------|
| v20260727 (smear, complex64) | N/A | N/A | N/A | 8-10 GeV |
| tv20260729 (nosmear, complex64) | 0.24 GeV | 4.17 GeV | 14.2 GeV | 7.86 GeV |
| tv20260729 (smear, complex64) | 0.45 GeV | 0.26 GeV | N/A | N/A |

所有方法给出不一致且错误的结果。问题不在有效质量提取方法，而在关联函数本身。

## 根因分析

### 核心问题: 2pt 关联函数没有干净的核子信号

C2pt(t) 振荡而非指数衰减，说明 Wick 缩并未能正确投出核子基态。

### 可能原因 (按可能性排序):

1. **本征矢量/Perambulator 配置不匹配 (最可能)**
   - 当前使用 cfg_48000 的本征矢量 vs cfg_6250/6450/6650 的 perambulator
   - Donghx 参考代码使用**每配置独立的本征矢量**
   - 信号强度比预期弱 ~10-100×, 与本征矢量不匹配一致
   - **建议**: 使用 per-config 本征矢量 (路径: `/public/group/lqcd/eigensystem/.../{conf_id}/`)

2. **Perambulator 动量涂抹符号约定**
   - Perambulator 路径 `mz2_my0_mx0` 表示涂抹动量 mz=2
   - 涂抹阶段 exp(±i P_smear·x) 的符号约定需要与 perambulator 生成代码一致
   - **建议**: 确认 perambulator 生成时的涂抹符号约定

3. **Gamma 矩阵约定与 Perambulator 不一致**
   - 插值算符 Cγ₅γ₄ 的 Dirac 结构依赖于 gamma 矩阵基
   - 当前使用 DeGrand-Rossi (手征) 基, 与 donghx 一致
   - 但如果 perambulator 使用不同约定, 结果会错误

4. **本征矢量空间坐标排序**
   - 动量相位因子依赖空间坐标与 eigenvector 索引的对应
   - C-order (x 最快) vs Fortran-order (z 最快)
   - **已验证**: 当前代码使用正确的 C-order

### 动量涂抹理论分析

对于动量涂抹的 perambulator (P_smear) 和物理动量 P:

**有特征向量涂抹时** (当前默认):
- VVV 携带: exp(-i(P + 3P_smear)·x)
- Perambulator sink: exp(+i P_smear·x)
- 缩并后 sink 相位: exp(-i(P + 2P_smear)·x) → P_eff ≠ P (错误!)

**无特征向量涂抹时**:
- VVV 携带: exp(-i P·x)
- Perambulator sink: exp(+i P_smear·x)
- 缩并后 sink 相位: exp(-i(P - P_smear)·x) → P_eff ≠ P (也错误!)

**正确的 VVV 相位**: 需要对每个本征矢量缩并取消 perambulator 的涂抹:
- Per perambulator 缩并: VV 的 smear × peram 的 smear 应抵消
- VVV 应使用: P_proj = P_phys (涂抹自动抵消)
- 3 个夸克各贡献 1 单位涂抹, perambulator sink 也贡献 1 单位
- 对于 a 的缩并: φ̃_a(x) × φ̃*_a(x) = exp(-i P_smear·x) × exp(+i P_smear·x) = 1 ✓
- 3 个缩并全部抵消涂抹
- **结论: 必须启用特征向量涂抹**, P_eff = P_proj = P_phys ✓

## 建议的修复步骤

### 短期 (立即可做):
1. ~~✅ 使用 per-config 本征矢量~~ (需要集群数据访问)
2. 添加更多诊断输出: 打印 VVV 和 perambulator 的大小/相位
3. 检查 donghx 实际输出数据 (如果可访问) 进行对比
4. 尝试 Pz=0 的零动量测试 (消除所有相位混淆)

### 中期:
5. 与 sunpeng 确认 perambulator 生成参数和约定
6. 检查 perambulator 文件内部的数据布局
7. 尝试用 donghx 的 per-config 本征矢量重新计算

### 长期:
8. 建立单元测试验证每个子步骤
9. 与已知正确结果进行端到端对比

## 代码改进 (docker-tv20260729 已实现)
- ✅ 特征向量涂抹可选 (`--smear` 标志)
- ✅ 多种有效质量方法 (fit_cosh, fit_exp, exp_forward, cosh)
- ✅ 增强诊断输出
- ✅ 前向关联函数分离
- ✅ 详细 README
