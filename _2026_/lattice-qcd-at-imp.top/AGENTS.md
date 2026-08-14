# AGENTS.md — lattice-qcd-at-imp.top

格点量子色动力学课题组网站（中国科学院近代物理研究所 IMP, CAS）→ https://lattice-qcd-at-imp.top。GitHub Pages 静态站点，SPA，Vanilla JS 模块化，中英文切换（i18n），深色/浅色模式。

## 技能（权威参考）

`.claude/skills/` 下两个技能（已归集到 `/root/.opencode/skills/`，此处原文件已归档为 `.claude.md.<ts>.bak`）：

- **build-website**：从零完整构建网站（11 步：初始化、Bulma 0.9.x CSS、`index.html` SPA、`data/translations.json`、JS 模块、数据文件、离线论文、媒体资源、本地预览、部署 GitHub Pages）
- **update-website**：内容更新。子命令 `papers`（INSPIRE-HEP API → `data/papers.json`）、`add-conference`（限中国 CAS 研究所）、`add-summer-school`、`advisor`、`students`、`translations`、`software`、`theme`、`summary`、`assets`、`all`

**导师 INSPIRE-HEP ID**：孙鹏 `1659207`、刘柳明 `1259106`。

## 架构要点

- 单页应用，`index.html` 为唯一页面；JS 模块 IIFE + `window` 全局命名空间（`I18N`/`Theme`/`Papers`/`MusicPlayer`/`Animations`），`langChanged` 自定义事件通信；defer 加载顺序：i18n → theme → music → papers → animations → index
- CSS 2331 行基于 Bulma 0.9.x（本地 `bulma.min.css`），自定义属性系统 + 响应式断点 1024/768/600/480px；Canvas 动画背景：深色=星场、浅色=樱花雨
- 板块 `#hero #about #advisors #research #publications #students #conferences #summer-schools #software #gallery #help`

## 数据三层结构

```
数据.csv  ← 手动修正入口（分类、项目、中文、英文、备注）
  ▼ /update-website summary
data_summary.json ← 机器可读摘要
  ├── data/papers.json        ← 论文唯一数据源（papers.js 加载，INSPIRE-HEP API 拉取）
  ├── data/conferences.json   ← 会议（运行时按中国机构过滤，CHINA_KEYWORDS）
  ├── data/summer-schools.json← 讲习班
  └── data/translations.json  ← i18n 字典
```

修改流程：编辑 `数据.csv` → `/update-website summary` → JSON 同步 → 推送部署。

**注意**：`custom/讲习班.html` 已废弃（运行时只加载 `summer-schools.json`）；代码引用的 `要求.json` 不存在于仓库（需求规范在 build-website 技能内）。

## 本地开发

```bash
python3 -m http.server 8000    # 纯静态，无需构建
```

离线论文回退：`custom/inspirehep.net/authors/{1659207,1259106}/INSPIRE-CiteAll.html`。
