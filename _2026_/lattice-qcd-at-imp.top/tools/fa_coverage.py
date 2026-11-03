#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查页面与业务 JS 中引用的所有图标名称是否都被 fontawesome-subset.js 覆盖。"""
import re
import sys

FILES = ["index.html"] + [f"static/js/{n}" for n in
         ("i18n.js", "theme.js", "music.js", "papers.js", "animations.js", "index.js")]

used = set()
for f in FILES:
    txt = open(f, encoding="utf-8").read()
    used |= set(re.findall(r'fa-([a-z0-9][a-z0-9-]*)', txt))

# 排除纯工具类
SKIP = {
    "spin", "pulse", "fw", "li", "ul", "border", "inverse", "stack",
    "layers", "lg", "xs", "sm", "2x", "3x", "4x", "5x", "6x", "7x", "8x",
    "9x", "10x", "1x", "i2svg", "pull-left", "pull-right", "rotate-90",
    "rotate-180", "rotate-270", "flip-horizontal", "flip-vertical", "flip-both",
    "w-1", "w-2", "w-3", "w-4", "w-5", "w-6", "w-7", "w-8", "w-9", "w-10",
    "w-11", "w-12", "w-13", "w-14", "w-15", "w-16", "w-17", "w-18", "w-19", "w-20",
    "sr-only", "stack-1x", "stack-2x", "swap-opacity", "transform", "mask",
    "symbol", "pseudo-element", "pseudo-element-pending", "primary", "secondary",
    "primary-color", "secondary-color", "secondary-opacity", "primary-opacity",
    "title-id", "mask-id", "layers-text", "layers-counter",
    "layers-bottom-right", "layers-bottom-left", "layers-top-right", "layers-top-left",
}
used = {u for u in used if u not in SKIP and not u.startswith("w-")}

subset = open("static/js/fontawesome-subset.js", encoding="utf-8").read()
covered = set(re.findall(r'^  "([a-z0-9-]+)":\[', subset, re.M))

missing = sorted(u for u in used if u not in covered)
print("页面使用的图标:", len(used), "个")
print("子集包含:", len(covered), "个")
print("未覆盖:", missing if missing else "无")
sys.exit(1 if missing else 0)
