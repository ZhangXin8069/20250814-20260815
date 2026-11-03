# 格点量子色动力学课题组网站 | Lattice QCD Group Website

中国科学院近代物理研究所 (IMP, CAS) 格点量子色动力学（Lattice QCD）课题组介绍网站。

**Advisors:** 孙鹏 (Peng Sun), 刘柳明 (Liuming Liu)

🌐 **Website:** https://lattice-qcd-at-imp.top

---

## 功能 Features

- 🌐 **中英文切换** — 完整的双语支持
- 🌙 **深色模式** — 支持浅色/深色主题切换，跟随系统偏好
- 📄 **论文列表** — 实时从 INSPIRE-HEP API 获取最新论文数据
- 👨‍🏫 **导师介绍** — 孙鹏、刘柳明两位研究员的详细简介
- 🔬 **研究方向** — 粲重子散射、部分子分布函数、核子自旋结构等
- 👨‍🎓 **研究生** — 从论文作者中自动识别研究生
- 📅 **学术会议** — 格点QCD相关会议信息
- 🏫 **讲习班** — 华大QCD讲习班、暑期学校、训练营等信息
- 🖼️ **图库** — 研究相关图片与视频展示
- 🎵 **背景音乐** — 可控的背景音乐播放器
- ✨ **动态效果** — Canvas粒子动画、滚动触发动画、视差效果等

---

## 技术栈 Tech Stack

- **HTML5 / CSS3 / JavaScript (Vanilla)**
- **CSS Framework:** Bulma 0.9.x
- **Icons:** Font Awesome 5 + Academicons
- **Data:** INSPIRE-HEP REST API (CORS supported)
- **Hosting:** GitHub Pages

---

## 项目结构 Structure

```
lattice-qcd-at-imp.top/
├── index.html                    # 主页面
├── data/                         # 静态数据文件
│   ├── translations.json        # 中英文翻译字典
│   ├── papers.json              # 论文数据（INSPIRE-HEP不可用时的fallback）
│   ├── conferences.json         # 会议数据
│   └── summer-schools.json      # 讲习班数据
├── static/
│   ├── css/
│   │   ├── index.css            # 主样式（含深色模式、动画）
│   │   ├── bulma.min.css        # Bulma框架
│   │   ├── bulma-carousel.min.css
│   │   ├── bulma-slider.min.css
│   │   └── fontawesome.all.min.css
│   ├── js/
│   │   ├── index.js             # 主控制器
│   │   ├── i18n.js             # 国际化模块
│   │   ├── theme.js            # 深色模式模块
│   │   ├── papers.js           # 论文数据加载与展示
│   │   ├── music.js            # 背景音乐播放器
│   │   ├── animations.js       # 动画效果（粒子、滚动等）
│   │   ├── bulma-carousel.min.js
│   │   ├── bulma-slider.min.js
│   │   └── fontawesome.all.min.js
│   ├── images/                  # 图片资源
│   └── videos/                  # 视频资源
├── .claude/skills/
│   └── update-website.md        # 网站更新Skill
├── *.mp3                        # 背景音乐文件
├── *.png                        # 图片资源
└── *.mp4                        # 视频资源
```

---

## 更新网站 Updating

网站可通过 Claude Code Skill 进行全流程更新。使用以下命令：

```
/update-website papers        # 更新论文数据
/update-website add-conference # 添加会议
/update-website add-summer-school # 添加讲习班
/update-website advisor       # 更新导师信息
/update-website translations  # 更新翻译
/update-website all           # 全量更新
```

### 论文数据更新原理

1. 网站打开时，`papers.js` 会直接从 INSPIRE-HEP API 获取最新论文数据
2. 数据缓存至 localStorage（24小时有效）
3. 如果 INSPIRE-HEP 不可用，自动回退到 `data/papers.json`

---

## 部署 Deploy

网站通过 GitHub Pages 部署。推送到 `main` 分支即可自动更新。

```bash
git add .
git commit -m "Update website"
git push origin main
```

---

## 媒体资源 Media Credits

- 背景音乐：菊次郎的夏天 (Summer of Kikujiro)
- 模板基础：Academic Project Page Template (Nerfies-style)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
