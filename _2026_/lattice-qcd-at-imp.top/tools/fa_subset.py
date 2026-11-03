#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 fontawesome.all.min.js 提取所需图标定义, 生成精简替换器 JS。
用法: python3 tools/fa_subset.py <输出js路径>"""
import re
import sys

SRC = "static/js/fontawesome.all.min.js"

ICON_NAMES = [
    "atom", "chart-line", "chevron-down", "chevron-left", "chevron-right",
    "chevron-up", "code-branch", "database", "envelope",
    "external-link-alt", "file-alt", "file-pdf", "globe", "info-circle",
    "magnet", "map-marker-alt", "microchip", "moon", "music", "pause",
    "play", "search", "spinner", "step-backward", "step-forward", "sun",
    "user", "users", "wave-square", "link",
]

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "static/js/fontawesome-subset.js"
    src = open(SRC, encoding="utf-8").read()

    icons = {}
    missing = []
    for name in ICON_NAMES:
        # FA5 格式: name:[width,height,[aliases],"unicode","path",...]
        m = re.search(r'(?<![\w"])"?\b' + re.escape(name) + r'"?:\[(\d+),(\d+),\[[^\]]*\],"[^"]*","((?:[^"\\]|\\.)*)"', src)
        if not m:
            missing.append(name)
            continue
        w, h, path = int(m.group(1)), int(m.group(2)), m.group(3)
        icons[name] = [w, h, path]

    if missing:
        sys.stderr.write("缺失图标: %s\n" % missing)
        sys.exit(1)

    # 生成替换器 JS: DOMContentLoaded + MutationObserver 扫描 i.fas 元素并替换为内联 SVG
    icon_literal = ",\n  ".join(
        '"%s":[%d,%d,"%s"]' % (n, v[0], v[1], v[2]) for n, v in sorted(icons.items())
    )
    js = """/* FontAwesome 5.15.1 精简子集替换器 (由 tools/fa_subset.py 生成)
 * 仅包含页面使用的 %d 个 solid 图标, 以内联 SVG 渲染, 替代全量 1.16MB 库。
 * Icons: CC BY 4.0, https://fontawesome.com */
(function () {
  "use strict";
  var ICONS = {
  %s
  };
  var STYLE_OK = { fas: 1, far: 1, fab: 1, fal: 1, fad: 1 };

  function iconName(clsList) {
    var name = null, style = null;
    for (var i = 0; i < clsList.length; i++) {
      var c = clsList[i];
      if (STYLE_OK[c]) style = c;
      else if (c.indexOf("fa-") === 0) {
        var n = c.slice(3);
        if (n && ICONS[n] && !/^(spin|pulse|fw|li|ul|border|inverse|stack|layers|w-)/.test(n)) name = n;
      }
    }
    return name;
  }

  function replace(el) {
    var cls = el.className && el.className.baseVal !== undefined ? el.className.baseVal : el.className;
    var name = iconName(String(cls).split(/\\s+/));
    var def = name && ICONS[name];
    if (!def) return;
    if (el.tagName === "svg") {
      if (el.getAttribute("data-fa-name") === name) return;
      el.setAttribute("viewBox", "0 0 " + def[0] + " " + def[1]);
      var p = el.querySelector("path");
      if (p) {
        p.setAttribute("d", def[2]);
      } else {
        var np = document.createElementNS("http://www.w3.org/2000/svg", "path");
        np.setAttribute("fill", "currentColor");
        np.setAttribute("d", def[2]);
        el.appendChild(np);
      }
      el.setAttribute("data-fa-name", name);
      el.setAttribute("data-fa-replaced", "");
      return;
    }
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 " + def[0] + " " + def[1]);
    svg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("style", "display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em");
    svg.setAttribute("data-fa-name", name);
    svg.setAttribute("data-fa-replaced", "");
    var path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("fill", "currentColor");
    path.setAttribute("d", def[2]);
    svg.appendChild(path);
    var title = el.getAttribute("title");
    if (title) { svg.setAttribute("role", "img"); svg.setAttribute("title", title); }
    el.parentNode.replaceChild(svg, el);
    svg.className.baseVal = cls + " svg-inline--fa";
  }

  function scan(root) {
    var nodes = root.querySelectorAll ? root.querySelectorAll("i.fas, i.far, i.fab, svg[data-fa-replaced]") : [];
    for (var i = 0; i < nodes.length; i++) replace(nodes[i]);
  }

  function init() {
    scan(document);
    if (window.MutationObserver) {
      var mo = new MutationObserver(function (muts) {
        for (var i = 0; i < muts.length; i++) {
          if (muts[i].type === "attributes" && muts[i].target.tagName === "svg") {
            replace(muts[i].target);
            continue;
          }
          var added = muts[i].addedNodes;
          for (var j = 0; j < added.length; j++) {
            if (added[j].nodeType === 1) scan(added[j]);
          }
        }
      });
      mo.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["class"] });
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
""" % (len(icons), icon_literal)
    open(out, "w", encoding="utf-8").write(js)
    print("已生成 %s: %d 个图标, %d 字节" % (out, len(icons), len(js.encode("utf-8"))))

if __name__ == "__main__":
    main()
