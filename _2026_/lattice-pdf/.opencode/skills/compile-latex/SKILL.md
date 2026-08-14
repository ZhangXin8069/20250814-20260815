---
name: compile-latex
description: 编译项目中任意 LaTeX 文档 — 单文件、整个目录或全量重编译
---

# compile-latex — LaTeX 文档编译

使用 XeLaTeX（支持中文 ctex）编译项目中任意 `.tex` 文件并生成 PDF。

## 项目文档地图

| 目录 | 文件数 | 类型 | 说明 |
|------|--------|------|------|
| `文档/` | 3 | article | 胶子 PDF 推导、连续极限、内部笔记 |
| `补充/` | 34 | article | 格点 QCD 各主题补充笔记（中文） |
| `汇报/` | 2 | article | 胶子准算符构造与计算 |
| `代码/` | 1 | article | GPU 代码分析 |
| `reports/` | 2 | beamer | 胶子 PDF 幻灯片（含 metropolis/Madrid 主题） |
| `refer/` | 1 | article | 理论解析与工作流 |
| `agent/文档/` | 3 | article | 三个 agent 子模块技术文档 |

总计 46 个 `.tex` 文件。所有文档使用 **XeLaTeX** 编译，中文支持通过 `ctex` 包。

## 编译命令规范

所有 `.tex` 文件必须用 `xelatex` 编译，**两遍**以解析交叉引用：

```bash
cd <tex文件所在目录> && \
xelatex -interaction=nonstopmode -halt-on-error <文件名>.tex && \
xelatex -interaction=nonstopmode -halt-on-error <文件名>.tex
```

编译后清理辅助文件（`.aux`, `.log`, `.out`, `.toc`, `.nav`, `.snm`, `.xdv`, `.fdb_latexmk`, `.fls`）：

```bash
rm -f <文件名>.aux <文件名>.log <文件名>.out <文件名>.toc <文件名>.nav <文件名>.snm <文件名>.xdv <文件名>.fdb_latexmk <文件名>.fls
```

## 子命令

### `compile-latex <文件路径>`
**编译单个 .tex 文件**

参数 `<文件路径>` 可以是：
- 相对于项目根目录的路径：`文档/gluon_pdf_derivation.tex`
- 绝对路径：`/root/lattice-pdf/文档/gluon_pdf_derivation.tex`
- 不带扩展名：`文档/gluon_pdf_derivation`
- 仅文件名（在已知目录中查找）：`gluon_pdf_derivation`

```bash
# 示例：编译 gluon_pdf_derivation.tex
cd "/root/lattice-pdf/文档" && \
xelatex -interaction=nonstopmode -halt-on-error gluon_pdf_derivation.tex && \
xelatex -interaction=nonstopmode -halt-on-error gluon_pdf_derivation.tex && \
rm -f gluon_pdf_derivation.aux gluon_pdf_derivation.log gluon_pdf_derivation.out gluon_pdf_derivation.toc && \
ls -lh gluon_pdf_derivation.pdf
```

### `compile-latex <目录名>`
**编译某个目录下所有 .tex 文件**

支持的目录名（输入中文或英文路径均可）：
- `文档` 或 `docs` — 根目录下 `文档/`（3 个文件）
- `补充` 或 `supplement` — 根目录下 `补充/`（34 个文件）
- `汇报` 或 `reports-brief` — 根目录下 `汇报/`（2 个文件）
- `代码` 或 `code-analysis` — 根目录下 `代码/`（1 个文件）
- `reports` 或 `slides` — 根目录下 `reports/`（2 个 beamer 文件）
- `refer` 或 `theory` — 根目录下 `refer/`（1 个文件）
- `agent-docs` — `agent/文档/`（3 个 agent 文档）

```bash
# 示例：编译 补充/ 下所有 .tex 文件
cd "/root/lattice-pdf/补充" && \
for f in *.tex; do
    base="${f%.tex}"
    echo "=== 编译 ${base} ==="
    xelatex -interaction=nonstopmode -halt-on-error "$f" > /dev/null 2>&1 && \
    xelatex -interaction=nonstopmode -halt-on-error "$f" > /dev/null 2>&1 && \
    echo "  ✓ ${base}.pdf"
done && \
rm -f *.aux *.log *.out *.toc *.nav *.snm *.xdv *.fdb_latexmk *.fls && \
echo "---" && ls -lh *.pdf
```

### `compile-latex all`
**重编译项目中所有 46 个 .tex 文件**

按目录依次编译：`文档/` → `补充/` → `汇报/` → `代码/` → `reports/` → `refer/` → `agent/文档/`

```bash
PROJECT_ROOT="/root/lattice-pdf"
FAILED=""

compile_dir() {
    local dir="$1"
    local label="$2"
    echo ""
    echo "===== 编译 ${label} ($dir) ====="
    cd "$dir"
    for f in *.tex; do
        [ ! -f "$f" ] && continue
        base="${f%.tex}"
        echo -n "  ${base} ... "
        if xelatex -interaction=nonstopmode -halt-on-error "$f" > /dev/null 2>&1 && \
           xelatex -interaction=nonstopmode -halt-on-error "$f" > /dev/null 2>&1; then
            local size=$(ls -lh "${base}.pdf" 2>/dev/null | awk '{print $5}')
            echo "✓  ${size}"
        else
            echo "✗  FAILED"
            FAILED="${FAILED}  ${dir}/${f}\n"
        fi
    done
    rm -f *.aux *.log *.out *.toc *.nav *.snm *.xdv *.fdb_latexmk *.fls 2>/dev/null
}

compile_dir "${PROJECT_ROOT}/文档"     "主文档"
compile_dir "${PROJECT_ROOT}/补充"     "补充笔记"
compile_dir "${PROJECT_ROOT}/汇报"     "汇报"
compile_dir "${PROJECT_ROOT}/代码"     "代码分析"
compile_dir "${PROJECT_ROOT}/reports"  "幻灯片"
compile_dir "${PROJECT_ROOT}/refer"    "理论解析"
compile_dir "${PROJECT_ROOT}/agent/文档" "Agent文档"

echo ""
if [ -n "$FAILED" ]; then
    echo "=== 编译失败 ==="
    echo -e "$FAILED"
else
    echo "=== 全部编译成功 ==="
fi
```

### `compile-latex clean`
**清理项目中所有 LaTeX 辅助文件**（保留 .tex 和 .pdf）

```bash
cd /root/lattice-pdf && \
find . -type f \( \
    -name "*.aux" -o -name "*.log" -o -name "*.out" -o \
    -name "*.toc" -o -name "*.nav" -o -name "*.snm" -o \
    -name "*.xdv" -o -name "*.fdb_latexmk" -o -name "*.fls" \
    -o -name "*.synctex.gz" -o -name "*.bbl" -o -name "*.blg" \
    -o -name "*.bcf" -o -name "*.run.xml" \
\) -delete && \
echo "已清理所有 LaTeX 辅助文件"
```

### `compile-latex status`
**查看所有 .tex 文件及其 PDF 状态**

```bash
cd /root/lattice-pdf && \
echo "=== LaTeX 文档状态 ===" && \
echo "" && \
for dir in 文档 补充 汇报 代码 reports refer agent/文档; do
    if [ -d "$dir" ]; then
        tex_count=$(find "$dir" -maxdepth 1 -name "*.tex" 2>/dev/null | wc -l)
        pdf_count=$(find "$dir" -maxdepth 1 -name "*.pdf" 2>/dev/null | wc -l)
        missing=$((tex_count - pdf_count))
        if [ "$missing" -gt 0 ]; then
            echo "  ${dir}/  ${tex_count} tex, ${pdf_count} pdf  ← ${missing} 个 PDF 缺失"
        else
            echo "  ${dir}/  ${tex_count} tex, ${pdf_count} pdf  ✓"
        fi
    fi
done && \
echo "" && \
echo "缺失 PDF 列表:" && \
for dir in 文档 补充 汇报 代码 reports refer agent/文档; do
    if [ -d "$dir" ]; then
        for tex in "$dir"/*.tex; do
            [ ! -f "$tex" ] && continue
            base="${tex%.tex}"
            if [ ! -f "${base}.pdf" ]; then
                echo "  ✗ ${tex} → ${base}.pdf MISSING"
            fi
        done
    fi
done
```

## 关键编译参数说明

| 参数 | 用途 |
|------|------|
| `-interaction=nonstopmode` | 不停止编译，遇到错误继续（错误仍会记录到 .log） |
| `-halt-on-error` | 第一个致命错误时停止（比 nonstopmode 更严格） |
| `-output-directory=<dir>` | 输出到指定目录（从其他目录编译时使用） |

## Beamer 特殊处理

`reports/` 下的 beamer 文件需要额外注意：
- `gluon_pdf_slides.tex` 使用 metropolis 主题
- `gluon_pdf_continuum_beamer.tex` 使用 Madrid/CambridgeUS 主题
- beamer 会生成 `.nav` 和 `.snm` 辅助文件，也需清理
- 如果需要绿色进度条，可以使用 `latexmk` 自动化：

```bash
cd /root/lattice-pdf/reports && \
latexmk -xelatex -interaction=nonstopmode gluon_pdf_slides.tex && \
latexmk -c gluon_pdf_slides.tex
```

## 使用示例

```
/compile-latex 文档/gluon_pdf_derivation.tex    # 编译单个文件
/compile-latex 补充                                # 编译 补充/ 下全部 34 个文件
/compile-latex reports                             # 编译 reports/ 下 2 个 beamer
/compile-latex all                                 # 编译全部 46 个文件
/compile-latex clean                               # 清理辅助文件
/compile-latex status                              # 查看 PDF 状态
```

## 编译环境要求

- `xelatex` (XeTeX 3.14+)
- LaTeX 包: `ctex`, `xeCJK`, `fontspec`, `physics`, `braket`, `amsmath`, `hyperref`, `geometry`, `graphicx`, `xcolor`, `beamer`（含 metropolis/Madrid/CambridgeUS 主题）
- 中文字体: 系统需安装 CJK 字体（Song, Hei, Kai, FangSong）
- 可选: `latexmk`（自动化编译）、`pdfinfo`（查看 PDF 页数）
