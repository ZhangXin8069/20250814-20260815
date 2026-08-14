---
name: update-deps-docs
description: 更新三个依赖库(LQCD_Master, lamet-agent, PyQUDA)的PDF技术文档
---

# update-deps-docs — 更新依赖库技术文档

重新编译 `/root/lattice-pdf/agent/文档/` 下三个依赖库的 LaTeX PDF 文档。

## 三个依赖库

| 库 | LaTeX 源文件 | PDF 输出 |
|---|---|---|
| LQCD Master | `LQCD_Master_Documentation.tex` | `LQCD_Master_Documentation.pdf` |
| lamet-agent | `lamet_agent_Documentation.tex` | `lamet_agent_Documentation.pdf` |
| PyQUDA | `PyQUDA_Documentation.tex` | `PyQUDA_Documentation.pdf` |

所有文件位于 `/root/lattice-pdf/agent/文档/`。

## LaTeX 源文件结构

每个 `.tex` 文档包含以下章节：
- 项目概述与架构
- 安装指南（环境要求、pip/源码安装步骤）
- 配置指南（环境变量、集群配置、清单格式等）
- 命令行接口（CLI 参数表）
- 核心模块详解（类层次、方法签名、工作流）
- 详细使用示例（逐步代码演示）
- 完整 API 清单（所有函数/类/方法签名）

## 子命令

### `update-deps-docs all`
**重编译全部三个 PDF**（默认行为）

1. 检查 LaTeX 源文件是否存在
2. 对每个 `.tex` 文件执行两遍 `xelatex`（解析交叉引用）
3. 清理辅助文件（`.aux`, `.log`, `.out`, `.toc`）
4. 报告编译结果（成功/失败、PDF 页数、文件大小）

```bash
cd "/root/lattice-pdf/agent/文档" && \
for f in LQCD_Master_Documentation lamet_agent_Documentation PyQUDA_Documentation; do \
    xelatex -interaction=nonstopmode -halt-on-error "${f}.tex" > /dev/null 2>&1 && \
    xelatex -interaction=nonstopmode -halt-on-error "${f}.tex" > /dev/null 2>&1 && \
    echo "✓ ${f}.pdf"; \
done && \
rm -f *.aux *.log *.out *.toc
```

### `update-deps-docs LQCD_Master`
**仅重编译 LQCD Master 文档**

```bash
cd "/root/lattice-pdf/agent/文档" && \
xelatex -interaction=nonstopmode -halt-on-error LQCD_Master_Documentation.tex && \
xelatex -interaction=nonstopmode -halt-on-error LQCD_Master_Documentation.tex && \
rm -f LQCD_Master_Documentation.aux LQCD_Master_Documentation.log \
      LQCD_Master_Documentation.out LQCD_Master_Documentation.toc && \
echo "✓ LQCD_Master_Documentation.pdf"
```

### `update-deps-docs lamet-agent`
**仅重编译 lamet-agent 文档**

```bash
cd "/root/lattice-pdf/agent/文档" && \
xelatex -interaction=nonstopmode -halt-on-error lamet_agent_Documentation.tex && \
xelatex -interaction=nonstopmode -halt-on-error lamet_agent_Documentation.tex && \
rm -f lamet_agent_Documentation.aux lamet_agent_Documentation.log \
      lamet_agent_Documentation.out lamet_agent_Documentation.toc && \
echo "✓ lamet_agent_Documentation.pdf"
```

### `update-deps-docs PyQUDA`
**仅重编译 PyQUDA 文档**

```bash
cd "/root/lattice-pdf/agent/文档" && \
xelatex -interaction=nonstopmode -halt-on-error PyQUDA_Documentation.tex && \
xelatex -interaction=nonstopmode -halt-on-error PyQUDA_Documentation.tex && \
rm -f PyQUDA_Documentation.aux PyQUDA_Documentation.log \
      PyQUDA_Documentation.out PyQUDA_Documentation.toc && \
echo "✓ PyQUDA_Documentation.pdf"
```

### `update-deps-docs deps`
**先更新依赖库源码，再重编译全部文档**

等效于先执行 `./update_deps.sh` 再执行 `update-deps-docs all`。

```bash
cd /root/lattice-pdf/agent && ./update_deps.sh && \
cd "/root/lattice-pdf/agent/文档" && \
for f in LQCD_Master_Documentation lamet_agent_Documentation PyQUDA_Documentation; do \
    xelatex -interaction=nonstopmode -halt-on-error "${f}.tex" > /dev/null 2>&1 && \
    xelatex -interaction=nonstopmode -halt-on-error "${f}.tex" > /dev/null 2>&1 && \
    echo "✓ ${f}.pdf"; \
done && \
rm -f *.aux *.log *.out *.toc && \
echo "---" && ls -lh *.pdf
```

### `update-deps-docs status`
**查看三个 PDF 的当前状态**（文件大小、页数）

```bash
cd "/root/lattice-pdf/agent/文档" && \
for f in LQCD_Master_Documentation lamet_agent_Documentation PyQUDA_Documentation; do \
    if [ -f "${f}.pdf" ]; then \
        pages=$(pdfinfo "${f}.pdf" 2>/dev/null | grep "Pages" | awk '{print $2}'); \
        size=$(ls -lh "${f}.pdf" | awk '{print $5}'); \
        echo "  ${f}.pdf  ${size}  ${pages} pages"; \
    else \
        echo "  ${f}.pdf  MISSING"; \
    fi; \
done
```

## 文档更新触发条件

以下情况应运行此 skill 重新编译文档：

1. **依赖库源码更新后**：执行 `./update_deps.sh` 拉取了新的代码
2. **LaTeX 源文件修改后**：手动编辑了 `.tex` 文件内容
3. **环境变更后**：安装了新的 LaTeX 包或字体

## 编译环境要求

- `xelatex` (XeTeX 3.14+)
- LaTeX 包: `ctex`, `hyperref`, `listings`, `xcolor`, `enumitem`, `booktabs`, `tabularx`, `caption`, `fancyhdr`, `tcolorbox`, `amsmath`, `fontspec`
- 中文字体: 系统需安装 CJK 字体

## 使用示例

```
/update-deps-docs              # 重编译全部三个 PDF
/update-deps-docs all          # 同上
/update-deps-docs LQCD_Master  # 仅重编译 LQCD Master
/update-deps-docs lamet-agent  # 仅重编译 lamet-agent
/update-deps-docs PyQUDA       # 仅重编译 PyQUDA
/update-deps-docs deps         # 先更新依赖源码再编译全部文档
/update-deps-docs status       # 查看 PDF 状态
```

## 依赖库源码位置

| 库 | 本地路径 | 远程仓库 |
|---|---|---|
| LQCD Master | `../LQCD_Master/` | `https://github.com/sjtu-sai-agents/LQCD_Master.git` |
| lamet-agent | `../lamet-agent/` | `https://github.com/Greyyy-HJC/lamet-agent.git` |
| PyQUDA | `../PyQUDA/` | `https://github.com/CLQCD/PyQUDA.git` |

源码通过 `../update_deps.sh` 脚本统一管理（`git pull --ff-only`）。
