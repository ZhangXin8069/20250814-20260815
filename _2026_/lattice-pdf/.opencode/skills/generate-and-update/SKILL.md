---
name: generate-and-update
description: 统一生成与更新项目中所有文档(LaTeX/PDF,共51个.tex)、代码和依赖库文档的主skill
---

# generate-and-update — 项目文档与代码生成更新

统一管理 `/root/lattice-pdf/` 下所有文档（LaTeX PDF）、Python 代码和依赖库文档的生成与编译。

## 项目文档地图

```
lattice-pdf/
├── 补充/     (38 个 .tex)   格点QCD各专题补充笔记（中文）
├── 文档/     (3 个 .tex)    胶子PDF推导、连续外推、内部笔记
├── 汇报/     (2 个 .tex)    胶子准算符构造、格点计算报告
├── reports/  (2 个 .tex)    Beamer 幻灯片（理论+连续极限）
├── refer/    (2 个 .tex)    理论解析与工作流
├── 代码/     (1 个 .tex)    GPU 代码分析
├── examples/zhangxin/ (1 个 .py)  gluon_pdf_full_workflow.py
├── examples/ (5+ 个 .py)   原始 GPU 代码（donghx）
└── agent/文档/(3 个 .tex)   依赖库技术文档
```

## 编译环境

- **编译器**: `xelatex`（XeTeX，支持中文 ctex 包）
- **LaTeX 包**: `ctex`, `xeCJK`, `fontspec`, `physics`, `braket`, `beamer`, `hyperref`, `listings`, `xcolor`, `enumitem`, `booktabs`, `tabularx`, `caption`, `fancyhdr`, `tcolorbox`, `amsmath`, `fontspec`
- **PDF 工具**: `pdftotext`（提取文本）, `pdfinfo`（查看页数）

所有 LaTeX 文档使用两遍 `xelatex` 编译以解析交叉引用、目录和标签。

---

## 子命令

### 补充笔记 — `generate-and-update supplements`

**编译全部 38 个格点QCD补充笔记**（`补充/` 目录）。

| # | 文件名 | 主题 |
|---|---|---|
| 1 | 格点QCD蒸馏方法解析 | 蒸馏方法（Laplacian eigenvectors, perambulators）|
| 2 | 格点QCD中的标度参数 | 标度设置（$w_0$, $t_0$, $\Lambda_{\text{QCD}}$）|
| 3 | 格点QCD中的部分子分布函数 | PDF、DA、GPD、TMD 定义与格点计算 |
| 4 | 格点QCD中的常见费曼图 | 格点QCD中常见费曼图 |
| 5 | 格点QCD中的常见名词短语与缩写 | 术语表与缩写 |
| 6 | 格点QCD中的场强张量 | $F_{\mu\nu}$ 的格点构造（Clover plaquette）|
| 7 | 格点QCD中的传播子求解 | 夸克传播子反演算法 |
| 8 | 格点QCD中的大动量有效理论 | LaMET 理论基础 |
| 9 | 格点QCD中的费米子方案 | Wilson, Clover, Staggered, Domain Wall, Overlap |
| 10 | 格点QCD中的关联函数 | 两点/三点关联函数、谱分解 |
| 11 | 格点QCD中的光锥与光前 | 光锥坐标、光前量子化 |
| 12 | 格点QCD中的光锥PDF与quasi_PDF | 光锥PDF与quasi-PDF的关系 |
| 13 | 格点QCD中的光锥PDF、准PDF、赝PDF | 三种PDF方案对比 |
| 14 | 格点QCD中的胶子极化 | 胶子自旋、极化PDF |
| 15 | 格点QCD中的胶子算符 | 胶子场算符的格点构造 |
| 16 | 格点QCD中的夸克算符 | 夸克双线性算符 |
| 17 | 格点QCD中的连通图与非连通图 | 连通/非连通收缩 |
| 18 | 格点QCD中的蒙卡方法 | HMC、RHMC、热浴算法 |
| 19 | 格点QCD中的强子谱学 | 强子谱计算 |
| 20 | 格点QCD中的衰变道 | 强子衰变 |
| 21 | 格点QCD中的外推 | 连续外推、手征外推、体积外推 |
| 22 | 格点QCD中的维克收缩与傅里叶变换 | Wick收缩、动量投影 |
| 23 | 格点QCD中的误差统计 | Jackknife、Bootstrap、gvar |
| 24 | 格点QCD中的形状因子 | 形状因子计算 |
| 25 | 格点QCD中的重采样方法 | Jackknife、Bootstrap 实现 |
| 26 | 格点QCD中的重夸克有效理论 | HQET 基础 |
| 27 | 格点QCD中的重整化 | RI/MOM, SMOM, 混合方案 |
| 28 | 格点QCD中的DGLAP演化方程 | DGLAP 演化 |
| 29 | 格点QCD中的OPE算符 | 算符乘积展开 |
| 30 | 格点QCD中的Symanzik有效理论 | Symanzik 改进 |
| 31 | 格点QCD中的TMD_PDF | TMD PDF 计算 |
| 32 | 格点QCD中的UV涨落 | 紫外发散与重整化 |
| 33 | 格点QCD中的Wilson线 | Wilson 线构造与性质 |
| 34 | 质子自旋危机解析 | 质子自旋分解 |
| 35 | 格点QCD中的3pt构造 | 三点关联函数构造 |
| 36 | 格点QCD中的色散关系 | 色散关系与能谱 |
| 37 | 格点QCD中的梯度流重整化 | 梯度流（gradient flow）重整化 |
| 38 | 格点QCD中的luscher参数化 | Lüscher 参数化与有限体积方法 |

**全部编译**：
```bash
cd "/root/lattice-pdf/补充" && \
for f in *.tex; do \
    xelatex -interaction=nonstopmode -halt-on-error "$f" > /dev/null 2>&1 && \
    xelatex -interaction=nonstopmode -halt-on-error "$f" > /dev/null 2>&1 && \
    echo "✓ $f"; \
done && \
rm -f *.aux *.log *.out *.toc
```

**编译单个**（如 "格点QCD中的重整化"）：
```bash
cd "/root/lattice-pdf/补充" && \
xelatex -interaction=nonstopmode "格点QCD中的重整化.tex" && \
xelatex -interaction=nonstopmode "格点QCD中的重整化.tex" && \
rm -f "格点QCD中的重整化.aux" "格点QCD中的重整化.log"
```

### 核心文档 — `generate-and-update docs`

编译 `文档/` 目录下的三个核心研究文档：

| 文件 | 内容 |
|------|------|
| `gluon_pdf_derivation.tex` | 胶子 PDF 的理论推导 |
| `gluon_PDF_continuum.tex` | 胶子 PDF 连续外推分析 |
| `note_of_gluon_PDFs.tex` | 胶子 PDF 内部研究笔记 |

```bash
cd "/root/lattice-pdf/文档" && \
for f in gluon_pdf_derivation gluon_PDF_continuum note_of_gluon_PDFs; do \
    xelatex -interaction=nonstopmode -halt-on-error "${f}.tex" > /dev/null 2>&1 && \
    xelatex -interaction=nonstopmode -halt-on-error "${f}.tex" > /dev/null 2>&1 && \
    echo "✓ ${f}.pdf"; \
done && \
rm -f *.aux *.log *.out *.toc
```

### Beamer 幻灯片 — `generate-and-update slides`

编译 `reports/` 目录下的两个 Beamer 幻灯片：

| 文件 | 主题 | 主题 |
|------|------|------|
| `gluon_pdf_slides.tex` | 胶子 PDF — 理论基础与计算工作流 | metropolis |
| `gluon_pdf_continuum_beamer.tex` | 胶子 PDF — 连续极限结果 | Madrid |

```bash
cd "/root/lattice-pdf/reports" && \
xelatex -interaction=nonstopmode -halt-on-error gluon_pdf_slides.tex > /dev/null 2>&1 && \
xelatex -interaction=nonstopmode -halt-on-error gluon_pdf_slides.tex > /dev/null 2>&1 && \
echo "✓ gluon_pdf_slides.pdf" && \
xelatex -interaction=nonstopmode -halt-on-error gluon_pdf_continuum_beamer.tex > /dev/null 2>&1 && \
xelatex -interaction=nonstopmode -halt-on-error gluon_pdf_continuum_beamer.tex > /dev/null 2>&1 && \
echo "✓ gluon_pdf_continuum_beamer.pdf" && \
rm -f *.aux *.log *.out *.toc *.nav *.snm
```

### 汇报 — `generate-and-update reports`

编译 `汇报/` 目录下的胶子准算符报告：

```bash
cd "/root/lattice-pdf/汇报" && \
for f in 格点上计算胶子准算符 构造胶子准算符; do \
    xelatex -interaction=nonstopmode -halt-on-error "${f}.tex" > /dev/null 2>&1 && \
    xelatex -interaction=nonstopmode -halt-on-error "${f}.tex" > /dev/null 2>&1 && \
    echo "✓ ${f}.pdf"; \
done && \
rm -f *.aux *.log *.out *.toc
```

### 理论分析与代码分析 — `generate-and-update analysis`

编译理论分析文档和代码分析文档：

```bash
# 理论解析与工作流 (refer/)
cd "/root/lattice-pdf/refer" && \
xelatex -interaction=nonstopmode -halt-on-error 理论解析与工作流.tex > /dev/null 2>&1 && \
xelatex -interaction=nonstopmode -halt-on-error 理论解析与工作流.tex > /dev/null 2>&1 && \
rm -f *.aux *.log *.out *.toc && \
echo "✓ 理论解析与工作流.pdf"

# 代码分析 (代码/)
cd "/root/lattice-pdf/代码" && \
xelatex -interaction=nonstopmode -halt-on-error code_analysis.tex > /dev/null 2>&1 && \
xelatex -interaction=nonstopmode -halt-on-error code_analysis.tex > /dev/null 2>&1 && \
rm -f *.aux *.log *.out *.toc && \
echo "✓ code_analysis.pdf"
```

### 依赖库文档 — `generate-and-update agent-deps`

编译 `agent/文档/` 下三个依赖库的技术文档 PDF（调用 `update-deps-docs` skill）：

| 库 | PDF |
|---|---|
| LQCD Master | `LQCD_Master_Documentation.pdf` |
| lamet-agent | `lamet_agent_Documentation.pdf` |
| PyQUDA | `PyQUDA_Documentation.pdf` |

等效于：`cd agent && ./update_docs.sh all`

### 全部 LaTeX — `generate-and-update all-tex`

**编译项目中所有 LaTeX 文档**（~51 个文件），按顺序：

1. `supplements` — 38 个补充笔记
2. `docs` — 3 个核心文档
3. `slides` — 2 个 Beamer 幻灯片
4. `reports` — 2 个汇报
5. `analysis` — 理论分析 + 代码分析
6. `agent-deps` — 3 个依赖库文档

```bash
# 按顺序编译所有 LaTeX 文档
failed=0

# 1. 补充笔记
cd "/root/lattice-pdf/补充"
for f in *.tex; do
    xelatex -interaction=nonstopmode -halt-on-error "$f" > /dev/null 2>&1
    xelatex -interaction=nonstopmode -halt-on-error "$f" > /dev/null 2>&1 || { echo "FAIL: $f"; failed=1; }
done
rm -f *.aux *.log *.out *.toc

# 2. 核心文档
cd "/root/lattice-pdf/文档"
for f in gluon_pdf_derivation gluon_PDF_continuum note_of_gluon_PDFs; do
    xelatex -interaction=nonstopmode -halt-on-error "${f}.tex" > /dev/null 2>&1
    xelatex -interaction=nonstopmode -halt-on-error "${f}.tex" > /dev/null 2>&1 || { echo "FAIL: ${f}.tex"; failed=1; }
done
rm -f *.aux *.log *.out *.toc

# 3. Slides
cd "/root/lattice-pdf/reports"
for f in gluon_pdf_slides gluon_pdf_continuum_beamer; do
    xelatex -interaction=nonstopmode -halt-on-error "${f}.tex" > /dev/null 2>&1
    xelatex -interaction=nonstopmode -halt-on-error "${f}.tex" > /dev/null 2>&1 || { echo "FAIL: ${f}.tex"; failed=1; }
done
rm -f *.aux *.log *.out *.toc *.nav *.snm

# 4. 汇报
cd "/root/lattice-pdf/汇报"
for f in *.tex; do
    xelatex -interaction=nonstopmode -halt-on-error "$f" > /dev/null 2>&1
    xelatex -interaction=nonstopmode -halt-on-error "$f" > /dev/null 2>&1 || { echo "FAIL: $f"; failed=1; }
done
rm -f *.aux *.log *.out *.toc

# 5. 分析
cd "/root/lattice-pdf/refer"
xelatex -interaction=nonstopmode -halt-on-error 理论解析与工作流.tex > /dev/null 2>&1
xelatex -interaction=nonstopmode -halt-on-error 理论解析与工作流.tex > /dev/null 2>&1 || { echo "FAIL: 理论解析与工作流.tex"; failed=1; }
rm -f *.aux *.log *.out *.toc
cd "/root/lattice-pdf/代码"
xelatex -interaction=nonstopmode -halt-on-error code_analysis.tex > /dev/null 2>&1
xelatex -interaction=nonstopmode -halt-on-error code_analysis.tex > /dev/null 2>&1 || { echo "FAIL: code_analysis.tex"; failed=1; }
rm -f *.aux *.log *.out *.toc

# 6. 依赖库文档
cd "/root/lattice-pdf/agent/文档"
for f in LQCD_Master_Documentation lamet_agent_Documentation PyQUDA_Documentation; do
    xelatex -interaction=nonstopmode -halt-on-error "${f}.tex" > /dev/null 2>&1
    xelatex -interaction=nonstopmode -halt-on-error "${f}.tex" > /dev/null 2>&1 || { echo "FAIL: ${f}.tex"; failed=1; }
done
rm -f *.aux *.log *.out *.toc

if [ $failed -eq 0 ]; then echo "✓ All LaTeX documents compiled"; else echo "⚠ Some documents failed"; fi
```

### Python 代码 — `generate-and-update code`

**运行 gluon_pdf_full_workflow.py**（10 步完整工作流）：

```bash
cd /root/lattice-pdf/code

# 基本运行（使用默认参数）
python3 gluon_pdf_full_workflow.py

# 带参数运行
python3 gluon_pdf_full_workflow.py \
    --conf 20000 --Pz 6 --delta_z 15 \
    --gauge_file gauge_config.dat \
    --eig_dir ./eigvecs/ --peram_dir ./perams/ \
    --output_dir ./results/
```

也可运行 `examples/` 下的原始 GPU 代码：

```bash
cd /root/lattice-pdf/examples

# 质子 2pt（蒸馏 + 动量 smearing）
python3 2pt_proton_Cg5gmu_L32x64_mom2_xdir_gpu.py < params.txt

# OPE 部分（多 GPU via MPI）
mpirun -np 4 python3 Calc_ope_unpol.py < params.txt
```

### 状态检查 — `generate-and-update status`

**检查所有文档和代码的状态**：

```bash
echo "============================================"
echo "  项目文档与代码状态"
echo "============================================"
echo ""

# LaTeX 文档统计
for dir_name in "补充" "文档" "汇报" "reports" "refer" "代码"; do
    dir_path="/root/lattice-pdf/${dir_name}"
    tex_count=$(ls "${dir_path}"/*.tex 2>/dev/null | wc -l)
    pdf_count=$(ls "${dir_path}"/*.pdf 2>/dev/null | wc -l)
    echo "  ${dir_name}/ : ${tex_count} .tex, ${pdf_count} .pdf"
done

# Agent 文档
agent_doc_dir="/root/lattice-pdf/agent/文档"
agent_tex=$(ls "${agent_doc_dir}"/*.tex 2>/dev/null | wc -l)
agent_pdf=$(ls "${agent_doc_dir}"/*.pdf 2>/dev/null | wc -l)
echo "  agent/文档/  : ${agent_tex} .tex, ${agent_pdf} .pdf"

echo ""
echo "  code/gluon_pdf_full_workflow.py : $(wc -l < /root/lattice-pdf/code/gluon_pdf_full_workflow.py) lines"
echo "  examples/ : $(ls /root/lattice-pdf/examples/*.py 2>/dev/null | wc -l) .py files"
echo "  agent/ : 3 submodules (LQCD_Master, lamet-agent, PyQUDA)"
echo "  agent/update_deps.sh : 依赖库 git pull"
echo "  agent/update_docs.sh : 依赖库 PDF 编译"
echo ""
echo "  总计 LaTeX 文档: ~51 个 .tex 文件"
```

### 子命令速查

| 子命令 | 范围 | 文件数 |
|--------|------|--------|
| `supplements` | 补充/ 全部格点QCD专题笔记 | 38 .tex |
| `docs` | 文档/ 核心研究文档 | 3 .tex |
| `slides` | reports/ Beamer 幻灯片 | 2 .tex |
| `reports` | 汇报/ 胶子准算符报告 | 2 .tex |
| `analysis` | refer/ + 代码/ | 2 .tex |
| `agent-deps` | agent/文档/ 依赖库文档 | 3 .tex |
| `all-tex` | 全部 LaTeX 文档 | ~51 .tex |
| `code` | code/ + examples/ 工作流代码 | 6 .py |
| `status` | 统计所有文档和代码状态 | — |

## 使用示例

```
/generate-and-update supplements    # 编译全部 38 个格点QCD补充笔记
/generate-and-update docs           # 编译胶子PDF推导、连续外推、内部笔记
/generate-and-update slides         # 编译 Beamer 幻灯片
/generate-and-update reports        # 编译胶子准算符汇报
/generate-and-update analysis       # 编译理论分析与代码分析
/generate-and-update agent-deps     # 编译三个依赖库技术文档
/generate-and-update all-tex        # 编译全部 ~51 个 LaTeX 文档
/generate-and-update code           # 运行 gluon_pdf_full_workflow.py
/generate-and-update status         # 查看所有文档状态
```

## 与其他 Skill 的关系

| Skill | 覆盖范围 | 关系 |
|-------|---------|------|
| `update-deps-docs` | agent/文档/ 三个依赖库 PDF | 本 skill 的 `agent-deps` 子命令等效于它 |
| `update-website` | lattice-qcd-at-imp.top 网站 | 独立网站更新流程 |
| `build-website` | lattice-qcd-at-imp.top 从零构建 | 独立网站构建流程 |

## 更新触发条件

以下情况应运行此 skill：
1. **修改了 .tex 文件内容** → 运行对应子命令重新编译
2. **添加了新的补充笔记** → 运行 `supplements` 编译全部
3. **修改了 gluon_pdf_full_workflow.py** → 运行 `code` 测试
4. **更新了依赖库源码** → 运行 `agent-deps` 或 `update-deps-docs deps`
5. **准备汇报/演讲前** → 运行 `slides` + `reports`
6. **定期全量检查** → 运行 `status` 查看整体状态
