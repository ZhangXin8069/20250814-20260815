#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验 fontawesome-subset.js 中每个图标的 path 与源库定义完全一致。"""
import re
import sys

def main():
    src = open("static/js/fontawesome.all.min.js", encoding="utf-8").read()
    gen = open("static/js/fontawesome-subset.js", encoding="utf-8").read()

    pairs = re.findall(r'"(\w[\w-]*)":\[(\d+),(\d+),"((?:[^"\\]|\\.)*)"\]', gen)
    tail = r'"?:\[(\d+),(\d+),\[[^\]]*\],"[^"]*","((?:[^"\\]|\\.)*)"'
    ok = bad = 0
    for name, w, h, path in pairs:
        pat = re.compile(r'(?<![\w"])"?\b' + re.escape(name) + tail)
        m = pat.search(src)
        if m and m.group(3) == path:
            ok += 1
        else:
            bad += 1
            print("不一致:", name, "| 正则匹配:", bool(m))
            if m:
                print("  源   group(3) 前40:", repr(m.group(3)[:40]))
                print("  生成 path    前40:", repr(path[:40]))
    print(f"校验: {ok}/{ok+bad} 个图标 path 与源完全一致")
    return 0 if bad == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
