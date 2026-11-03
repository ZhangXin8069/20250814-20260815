#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 custom/ 下的大图压缩为页面引用的 JPEG/WebP 版本(原文件保留)。
策略: 巨型截图/照片 -> 白底 JPEG (降采样, quality 85-88)
      带透明的小图   -> WebP (保留 alpha, quality 88)
用法: python3 tools/optimize_images.py [--dry-run]"""
import os
import sys
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# (源文件, 目标最大宽度, quality, 格式)
JOBS_JPEG = [
    ("custom/张鑫 20251227工作汇报_01.png", 1600, 85),
    ("custom/张鑫 20251227工作汇报_02.png", 1600, 85),
    ("custom/张鑫 20251227工作汇报_03.png", 1600, 85),
    ("custom/张鑫 20251227工作汇报_04.png", 1600, 85),
    ("custom/张鑫 20251227工作汇报_05.png", 1600, 85),
    ("custom/张鑫 20251227工作汇报_06.png", 1600, 85),
    ("custom/张鑫 20251227工作汇报_07.png", 1600, 85),
    ("custom/张鑫 20251227工作汇报_08.png", 1600, 85),
    ("custom/张鑫 20260531工作汇报_01.png", 1600, 85),
    ("custom/课题组聚餐20260711.png", 1600, 88),
    ("custom/课题组合照20260706.png", 1600, 88),
    ("custom/李政道坐像.png", 1200, 88),
    ("custom/近代物理研究所惠州分部办公楼.png", 1200, 88),
    ("custom/张鑫 20260706gitee贡献度.png", 1748, 85),
]
JOBS_WEBP = [
    ("custom/刘柳明头像.png", 800),
    ("custom/孙鹏头像.png", 800),
    ("custom/孙鹏讲习图.png", 1000),
    ("custom/刘柳明讲习图.png", 1000),
    ("custom/张鑫讲习图.png", 1000),
    ("custom/标准模型图.png", 500),
    ("custom/网站管理员.png", 1003),
]

def to_jpeg(src, max_w, quality):
    im = Image.open(src)
    if im.mode in ("P", "L"):
        im = im.convert("RGBA")
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    else:
        im = im.convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    out = os.path.splitext(src)[0] + ".jpg"
    im.save(out, "JPEG", quality=quality, optimize=True, progressive=True)
    return out

def to_webp(src, max_w):
    im = Image.open(src)
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    out = os.path.splitext(src)[0] + ".webp"
    im.save(out, "WEBP", quality=88, method=6)
    return out

def main():
    dry = "--dry-run" in sys.argv
    total_before = total_after = 0
    for src, max_w, q in JOBS_JPEG:
        out = to_jpeg(src, max_w, q)
        b, a = os.path.getsize(src), os.path.getsize(out)
        total_before += b; total_after += a
        print(f"{src}: {b/1048576:7.2f}MB -> {a/1048576:7.2f}MB  ({a/b*100:5.1f}%)  {out}")
    for src, max_w in JOBS_WEBP:
        out = to_webp(src, max_w)
        b, a = os.path.getsize(src), os.path.getsize(out)
        total_before += b; total_after += a
        print(f"{src}: {b/1024:8.1f}KB -> {a/1024:8.1f}KB  ({a/b*100:5.1f}%)  {out}")
    print(f"合计: {total_before/1048576:.2f}MB -> {total_after/1048576:.2f}MB  (节省 {total_after/total_before*100:.1f}%)")

if __name__ == "__main__":
    main()
