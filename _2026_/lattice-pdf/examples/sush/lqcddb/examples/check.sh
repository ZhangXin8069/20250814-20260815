#!/bin/bash

# 检查是否提供了两个参数（源文件夹路径 和 目标文件夹路径）
if [ $# -ne 2 ]; then
    echo "用法: $0 <源文件夹> <目标文件夹>"
    exit 1
fi

SOURCE_DIR="$1"
TARGET_DIR="$2"

# 检查源文件夹是否存在
if [ -d "$SOURCE_DIR" ]; then
    echo "源文件夹 '$SOURCE_DIR' 已存在，不创建目标文件夹 '$TARGET_DIR'。"
else
    # 源文件夹不存在，创建目标文件夹（-p 确保父目录一并创建）
    mkdir -p "$TARGET_DIR"
    echo "源文件夹 '$SOURCE_DIR' 不存在，已创建目标文件夹 '$TARGET_DIR'。"
fi