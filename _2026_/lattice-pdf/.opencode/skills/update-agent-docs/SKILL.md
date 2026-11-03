---
name: update-agent-docs
description: 更新三个 AI agent 子模块(LQCD_Master, lamet-agent, PyQUDA)的 PDF 技术文档 — 支持源码拉取+文档编译
---

# update-agent-docs — 更新 Agent 子模块技术文档

重新编译 `/root/lattice-pdf/agent/文档/` 下三个依赖库的 LaTeX PDF 文档。支持先拉取最新源码再编译，或仅查看状态。

> 此 skill 封装了 `agent/update_docs.sh` 脚本的全部功能。agent 子目录下还有一个同名 skill（`agent/.claude/skills/update-deps-docs.md`），功能一致。

## 三个子模块

| 库 | LaTeX 源文件 | PDF 输出 | Git 仓库 |
|---|---|---|---|
| LQCD Master | `LQCD_Master_Documentation.tex` | `LQCD_Master_Documentation.pdf` | sjtu-sai-agents/LQCD_Master |
| lamet-agent | `lamet_agent_Documentation.tex` | `lamet_agent_Documentation.pdf` | Greyyy-HJC/lamet-agent |
| PyQUDA | `PyQUDA_Documentation.tex` | `PyQUDA_Documentation.pdf` | CLQCD/PyQUDA |

所有文件位于 `/root/lattice-pdf/agent/文档/`。源码位于 `/root/lattice-pdf/agent/<name>/`。

## 子命令

### `update-agent-docs all`
**重编译全部三个 PDF**（默认行为）

两遍 xelatex 编译 + 清理辅助文件。

```bash
cd "/root/lattice-pdf/agent/文档" && \
for f in LQCD_Master_Documentation lamet_agent_Documentation PyQUDA_Documentation; do \
    echo "=== 编译 ${f} ===" && \
    xelatex -interaction=nonstopmode -halt-on-error "${f}.tex" > /dev/null 2>&1 && \
    xelatex -interaction=nonstopmode -halt-on-error "${f}.tex" > /dev/null 2>&1 && \
    echo "  ✓ ${f}.pdf"; \
done && \
rm -f *.aux *.log *.out *.toc && \
echo "---" && ls -lh *.pdf
```

### `update-agent-docs LQCD_Master`
**仅重编译 LQCD Master 文档**

```bash
cd "/root/lattice-pdf/agent/文档" && \
xelatex -interaction=nonstopmode -halt-on-error LQCD_Master_Documentation.tex > /dev/null 2>&1 && \
xelatex -interaction=nonstopmode -halt-on-error LQCD_Master_Documentation.tex > /dev/null 2>&1 && \
rm -f LQCD_Master_Documentation.aux LQCD_Master_Documentation.log LQCD_Master_Documentation.out LQCD_Master_Documentation.toc && \
echo "✓ LQCD_Master_Documentation.pdf  $(ls -lh LQCD_Master_Documentation.pdf | awk '{print $5}')"
```

### `update-agent-docs lamet-agent`
**仅重编译 lamet-agent 文档**

```bash
cd "/root/lattice-pdf/agent/文档" && \
xelatex -interaction=nonstopmode -halt-on-error lamet_agent_Documentation.tex > /dev/null 2>&1 && \
xelatex -interaction=nonstopmode -halt-on-error lamet_agent_Documentation.tex > /dev/null 2>&1 && \
rm -f lamet_agent_Documentation.aux lamet_agent_Documentation.log lamet_agent_Documentation.out lamet_agent_Documentation.toc && \
echo "✓ lamet_agent_Documentation.pdf  $(ls -lh lamet_agent_Documentation.pdf | awk '{print $5}')"
```

### `update-agent-docs PyQUDA`
**仅重编译 PyQUDA 文档**

```bash
cd "/root/lattice-pdf/agent/文档" && \
xelatex -interaction=nonstopmode -halt-on-error PyQUDA_Documentation.tex > /dev/null 2>&1 && \
xelatex -interaction=nonstopmode -halt-on-error PyQUDA_Documentation.tex > /dev/null 2>&1 && \
rm -f PyQUDA_Documentation.aux PyQUDA_Documentation.log PyQUDA_Documentation.out PyQUDA_Documentation.toc && \
echo "✓ PyQUDA_Documentation.pdf  $(ls -lh PyQUDA_Documentation.pdf | awk '{print $5}')"
```

### `update-agent-docs deps`
**先更新依赖库源码，再重编译全部文档**

```bash
cd /root/lattice-pdf/agent && \
./update_deps.sh && \
echo "" && \
cd "文档" && \
for f in LQCD_Master_Documentation lamet_agent_Documentation PyQUDA_Documentation; do \
    echo "=== 编译 ${f} ===" && \
    xelatex -interaction=nonstopmode -halt-on-error "${f}.tex" > /dev/null 2>&1 && \
    xelatex -interaction=nonstopmode -halt-on-error "${f}.tex" > /dev/null 2>&1 && \
    echo "  ✓ ${f}.pdf"; \
done && \
rm -f *.aux *.log *.out *.toc && \
echo "---" && ls -lh *.pdf
```

### `update-agent-docs status`
**查看三个 PDF 的当前状态**（文件大小、页数）

```bash
cd "/root/lattice-pdf/agent/文档" && \
echo "=== Agent 文档状态 ===" && \
echo "" && \
for f in LQCD_Master_Documentation lamet_agent_Documentation PyQUDA_Documentation; do \
    if [ -f "${f}.pdf" ]; then \
        pages=$(pdfinfo "${f}.pdf" 2>/dev/null | grep "Pages" | awk '{print $2}') && \
        size=$(ls -lh "${f}.pdf" | awk '{print $5}') && \
        echo "  ${f}.pdf  ${size}  ${pages:-?} pages"; \
    else \
        echo "  ${f}.pdf  MISSING"; \
    fi; \
done && \
echo "" && \
for f in LQCD_Master_Documentation lamet_agent_Documentation PyQUDA_Documentation; do \
    if [ -f "${f}.tex" ]; then \
        echo "  ${f}.tex  $(ls -lh "${f}.tex" | awk '{print $5}')"; \
    else \
        echo "  ${f}.tex  MISSING"; \
    fi; \
done
```

## 触发场景

以下情况应运行此 skill：

1. **子模块源码更新后**：`git pull --ff-only` 拉取了新代码，文档内容可能有变化
2. **LaTeX 源文件手动修改后**：编辑了 `.tex` 文件
3. **环境变更后**：安装了新的 LaTeX 包或字体
4. **PDF 文件丢失/损坏**：`.pdf` 被删除或编译出错

## 使用示例

```
/update-agent-docs              # 重编译全部三个 PDF
/update-agent-docs all          # 同上
/update-agent-docs LQCD_Master  # 仅 LQCD Master
/update-agent-docs lamet-agent  # 仅 lamet-agent
/update-agent-docs PyQUDA       # 仅 PyQUDA
/update-agent-docs deps         # 拉取源码 + 编译全部
/update-agent-docs status       # 查看 PDF 状态
```

## 依赖

- `xelatex` — LaTeX 编译器
- `pdfinfo` (可选) — 查看 PDF 页数
- Git — 源码更新（仅 `deps` 子命令）

## 相关脚本

- `agent/update_docs.sh` — 等效的 shell 脚本
- `agent/update_deps.sh` — 子模块源码更新脚本
