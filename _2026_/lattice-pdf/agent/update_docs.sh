#!/bin/bash
# 更新三个依赖库的 PDF 技术文档
# 用法: ./update_docs.sh [all|LQCD_Master|lamet-agent|PyQUDA|status|deps]
#
# 子命令:
#   all (默认)  — 重新编译全部三个 PDF
#   LQCD_Master  — 仅编译 LQCD_Master_Documentation.pdf
#   lamet-agent  — 仅编译 lamet_agent_Documentation.pdf
#   PyQUDA       — 仅编译 PyQUDA_Documentation.pdf
#   deps         — 先更新依赖库源码，再编译全部文档
#   status       — 查看三个 PDF 的当前状态

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCS_DIR="${SCRIPT_DIR}/文档"
SUBCMD="${1:-all}"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 检查 xelatex 可用性
check_xelatex() {
    if ! command -v xelatex &>/dev/null; then
        log_error "xelatex 未安装，无法编译 LaTeX 文档"
        exit 1
    fi
}

# 编译单个文档（两遍 xelatex）
compile_one() {
    local tex_name="$1"
    local tex_file="${DOCS_DIR}/${tex_name}.tex"
    local pdf_file="${DOCS_DIR}/${tex_name}.pdf"

    if [ ! -f "$tex_file" ]; then
        log_error "LaTeX 源文件不存在: ${tex_file}"
        return 1
    fi

    log_info "编译 ${tex_name}.tex ..."
    cd "$DOCS_DIR"

    # 第一遍
    if ! xelatex -interaction=nonstopmode -halt-on-error "${tex_name}.tex" > /dev/null 2>&1; then
        log_error "${tex_name} 第一遍编译失败，查看 ${tex_name}.log"
        return 1
    fi

    # 第二遍（解析交叉引用）
    if ! xelatex -interaction=nonstopmode -halt-on-error "${tex_name}.tex" > /dev/null 2>&1; then
        log_error "${tex_name} 第二遍编译失败，查看 ${tex_name}.log"
        return 1
    fi

    # 清理辅助文件
    rm -f "${tex_name}.aux" "${tex_name}.log" "${tex_name}.out" "${tex_name}.toc"

    # 报告结果
    local size=$(ls -lh "$pdf_file" | awk '{print $5}')
    local pages=$(pdfinfo "$pdf_file" 2>/dev/null | grep "Pages" | awk '{print $2}')
    log_ok "${tex_name}.pdf  ${size}  ${pages:-?} pages"
}

# 编译全部三个文档
compile_all() {
    local failed=0
    compile_one "LQCD_Master_Documentation" || failed=1
    compile_one "lamet_agent_Documentation" || failed=1
    compile_one "PyQUDA_Documentation" || failed=1
    return $failed
}

# 查看状态
show_status() {
    echo ""
    echo "============================================"
    echo "  依赖库技术文档状态"
    echo "============================================"
    echo ""

    local all_ok=true
    for name in LQCD_Master_Documentation lamet_agent_Documentation PyQUDA_Documentation; do
        local pdf_file="${DOCS_DIR}/${name}.pdf"
        local tex_file="${DOCS_DIR}/${name}.tex"

        echo "  ${name}:"
        if [ -f "$tex_file" ]; then
            echo "    LaTeX: $(ls -lh "$tex_file" | awk '{print $5}')"
        else
            echo "    LaTeX: MISSING"
            all_ok=false
        fi

        if [ -f "$pdf_file" ]; then
            local size=$(ls -lh "$pdf_file" | awk '{print $5}')
            local pages=$(pdfinfo "$pdf_file" 2>/dev/null | grep "Pages" | awk '{print $2}')
            echo "    PDF:   ${size}  ${pages:-?} pages"
        else
            echo "    PDF:   MISSING"
            all_ok=false
        fi
        echo ""
    done

    if $all_ok; then
        log_ok "全部文档就绪"
    else
        log_warn "部分文档缺失，运行 ./update_docs.sh 生成"
    fi
}

# ---- 主流程 ----
echo ""
echo "============================================"
echo "  依赖库技术文档更新"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""

check_xelatex

case "$SUBCMD" in
    all)
        compile_all
        ;;
    LQCD_Master)
        compile_one "LQCD_Master_Documentation"
        ;;
    lamet-agent)
        compile_one "lamet_agent_Documentation"
        ;;
    PyQUDA)
        compile_one "PyQUDA_Documentation"
        ;;
    deps)
        log_info "先更新依赖库源码..."
        if [ -f "${SCRIPT_DIR}/update_deps.sh" ]; then
            bash "${SCRIPT_DIR}/update_deps.sh"
        else
            log_warn "update_deps.sh 不存在，跳过依赖更新"
        fi
        echo ""
        log_info "开始编译全部文档..."
        compile_all
        ;;
    status)
        show_status
        exit 0
        ;;
    *)
        log_error "未知子命令: ${SUBCMD}"
        echo "用法: ./update_docs.sh [all|LQCD_Master|lamet-agent|PyQUDA|status|deps]"
        exit 1
        ;;
esac

echo ""
if [ $? -eq 0 ]; then
    log_ok "文档更新完成"
    echo ""
    ls -lh "${DOCS_DIR}"/*.pdf
else
    log_error "部分文档编译失败"
    exit 1
fi
