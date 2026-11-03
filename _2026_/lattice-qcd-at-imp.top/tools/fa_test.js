#!/usr/bin/env node
// 行为回归测试: 最小 DOM mock 运行 fontawesome-subset.js 替换器
const fs = require("fs");
const vm = require("vm");

let pass = 0, fail = 0;
function check(name, cond, extra = "") {
  if (cond) { pass++; console.log(`  ok  ${name}`); }
  else { fail++; console.log(`FAIL  ${name} ${extra}`); }
}

function makeEl(tag, attrs = {}) {
  return {
    tagName: tag,
    _attrs: { ...attrs },
    _children: [],
    parentNode: null,
    get className() {
      const self = this;
      return {
        get baseVal() { return self._attrs.class || ""; },
        set baseVal(v) { self._attrs.class = v; },
      };
    },
    set className(v) { this._attrs.class = typeof v === "string" ? v : v.baseVal; },
    getAttribute(k) { return this._attrs[k] !== undefined ? this._attrs[k] : null; },
    setAttribute(k, v) { this._attrs[k] = String(v); },
    appendChild(c) { c.parentNode = this; this._children.push(c); },
    querySelector(sel) {
      if (sel === "path") return this._children.find(c => c.tagName === "path") || null;
      return null;
    },
  };
}

function run(code, { els, withObserver }) {
  const replaced = [];
  const doc = {
    readyState: "complete",
    body: makeEl("body"),
    createElementNS(ns, tag) { return makeEl(tag); },
    addEventListener() {},
    querySelectorAll(sel) { return els; },
  };
  doc.body.replaceChild = (svg, el) => { replaced.push({ el, svg }); };
  els.forEach(el => { el.parentNode = doc.body; });
  const obsInstances = [];
  const MutationObserver = withObserver ? class {
    constructor(cb) { this.cb = cb; obsInstances.push(this); }
    observe() {}
  } : null;
  const window = { MutationObserver };
  const ctx = vm.createContext({ document: doc, window, console, MutationObserver });
  vm.runInContext(code, ctx);
  return { doc, replaced, obsInstances };
}

// ========== 测试 1: ICONS 数据完整性 ==========
const src = fs.readFileSync("static/js/fontawesome-subset.js", "utf-8");
const icons = {};
const re = /^  "([a-z0-9-]+)":\[(\d+),(\d+),"((?:[^"\\]|\\.)*)"/gm;
let m;
while ((m = re.exec(src))) icons[m[1]] = [m[2], m[3], m[4]];
console.log("测试1: ICONS 数据");
check("数量 == 30", Object.keys(icons).length === 30, `got ${Object.keys(icons).length}`);
let dataOk = true;
for (const [n, v] of Object.entries(icons)) {
  if (!(+v[0] > 0) || !(+v[1] > 0) || v[2].length < 50) { dataOk = false; console.log(`  bad: ${n} ${v[2].length}`); }
}
check("viewBox/path 数据完整", dataOk);

// ========== 测试 2: 替换器端到端 (含 MutationObserver) ==========
console.log("测试2: 替换器行为");
// 构造真实替换路径: 复现 init 流程 — 通过 mock 的 querySelectorAll + body.replaceChild
{
  const iEl = makeEl("i", { class: "fas fa-atom" });
  const els = [iEl];
  const { doc, replaced, obsInstances } = run(src, { els, withObserver: true });
  check("MutationObserver 已注册", obsInstances.length === 1);
  check("scan 已替换 i 元素", replaced.length === 1, `got ${replaced.length}`);
  if (replaced.length === 1) {
    const { svg } = replaced[0];
    check("生成 svg 元素", svg.tagName === "svg");
    check("viewBox 正确", svg.getAttribute("viewBox") === "0 0 448 512", svg.getAttribute("viewBox"));
    check("path 含真实数据", svg.querySelector("path") && svg.querySelector("path").getAttribute("d").startsWith("M223.99908"));
    check("data-fa-name", svg.getAttribute("data-fa-name") === "atom");
    check("保留原类 + svg-inline--fa", /fa-atom/.test(svg._attrs.class) && /svg-inline--fa/.test(svg._attrs.class), svg._attrs.class);
    check("内联尺寸样式", /height:1em/.test(svg.getAttribute("style")));
  }
}

// ========== 测试 3: 未知图标与工具类 ==========
console.log("测试3: 未知图标/工具类处理");
{
  const els = [makeEl("i", { class: "fas fa-circle-nodes" }), makeEl("i", { class: "fas fa-spin" }), makeEl("i", { class: "fas fa-fw" })];
  const { doc, replaced } = run(src, { els, withObserver: false });
  check("未知图标(fa-circle-nodes)保持原样不替换", replaced.length === 0, `got ${replaced.length}`);
}

console.log(`\n结果: ${pass} 通过, ${fail} 失败`);
process.exit(fail ? 1 : 0);
