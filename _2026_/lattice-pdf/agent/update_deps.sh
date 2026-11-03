#!/bin/bash
# 自动更新依赖库
# 用法: ./update_deps.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  更新依赖库 - $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# --- PyQUDA ---
echo ""
echo "[1/3] 更新 PyQUDA ..."
if [ -d "$SCRIPT_DIR/PyQUDA" ]; then
    git -C "$SCRIPT_DIR/PyQUDA" pull --ff-only
    echo "  ✓ PyQUDA 更新完成"
else
    echo "  ⚠ PyQUDA 目录不存在，正在克隆..."
    git clone https://github.com/CLQCD/PyQUDA.git "$SCRIPT_DIR/PyQUDA"
    echo "  ✓ PyQUDA 克隆完成"
fi

# --- LQCD_Master ---
echo ""
echo "[2/3] 更新 LQCD_Master ..."
if [ -d "$SCRIPT_DIR/LQCD_Master" ]; then
    git -C "$SCRIPT_DIR/LQCD_Master" pull --ff-only
    echo "  ✓ LQCD_Master 更新完成"
else
    echo "  ⚠ LQCD_Master 目录不存在，正在克隆..."
    git clone https://github.com/sjtu-sai-agents/LQCD_Master.git "$SCRIPT_DIR/LQCD_Master"
    echo "  ✓ LQCD_Master 克隆完成"
fi

# --- lamet-agent ---
echo ""
echo "[3/3] 更新 lamet-agent ..."
if [ -d "$SCRIPT_DIR/lamet-agent" ]; then
    git -C "$SCRIPT_DIR/lamet-agent" pull --ff-only
    echo "  ✓ lamet-agent 更新完成"
else
    echo "  ⚠ lamet-agent 目录不存在，正在克隆..."
    git clone https://github.com/Greyyy-HJC/lamet-agent.git "$SCRIPT_DIR/lamet-agent"
    echo "  ✓ lamet-agent 克隆完成"
fi

echo ""
echo "============================================"
echo "  全部依赖库已更新完毕"
echo "============================================"
