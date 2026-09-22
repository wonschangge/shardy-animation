#!/usr/bin/env python3
import json
from pathlib import Path

def generate_portal():
    base_dir = Path("shardy-animation")
    data_file = base_dir / "lessons_data.json"
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    layers_meta = {
        "intro": {"badge": "预备篇", "color": "#38bdf8", "sub": "建立直觉", "stats": "20 幕"},
        "L1": {"badge": "L1", "color": "#818cf8", "sub": "认识构件 · 看语法", "stats": "10 课 · 28 文件"},
        "L2": {"badge": "L2", "color": "#c084fc", "sub": "理解算法 · 看传播", "stats": "12 课 · 18 文件"},
        "L3": {"badge": "L3", "color": "#f472b6", "sub": "理解准备 · 看导入", "stats": "11 课 · 23 文件"},
        "L4": {"badge": "L4", "color": "#fb7185", "sub": "理解代价 · 看通信", "stats": "17 课 · 61 文件"},
        "L5": {"badge": "L5", "color": "#fbbf24", "sub": "理解降级 · 看切分", "stats": "9 课 · 35 文件"},
        "L6": {"badge": "L6", "color": "#4ade80", "sub": "验证正确 · 看数值", "stats": "11 课 · 76 文件"},
        "L7": {"badge": "L7", "color": "#2dd4bf", "sub": "综合运用 · 看大模型", "stats": "4 课 · 复用"}
    }

    sections_html = []

    for lid, linfo in layers_meta.items():
        layer_lessons = [les for les in data["lessons"] if les["layer_id"] == lid]
        if not layer_lessons:
            continue
        layer_obj = data["layers"].get(lid, {})
        lname = layer_obj.get("name", lid)
        ldesc = layer_obj.get("desc", "")
        
        cards_html = []
        for les in layer_lessons:
            scenes_str = f"{les['scenes']} 幕场景" if les['scenes'] else "多幕场景"
            sub_desc = les['subtitle'] or "包含 IR 解析、动态可视化变换与配套练习题。"
            card = f'''
        <a class="card" href="{les['path']}" style="--card-color: {linfo['color']}" data-code="{les['code'].lower()}" data-title="{les['title'].lower()}" data-sub="{les['subtitle'].lower()}">
          <div>
            <div class="card-top">
              <span class="card-code">{les['code']}</span>
              <span class="card-scenes">{scenes_str}</span>
            </div>
            <h3 class="card-title">{les['title']}</h3>
            <p class="card-desc">{sub_desc}</p>
          </div>
          <div class="card-footer">
            <span class="card-link-txt">进入课件学习 →</span>
          </div>
        </a>'''
            cards_html.append(card)

        sec = f'''
    <section class="layer-section" data-layer-id="{lid}">
      <div class="layer-header">
        <div class="layer-title-wrap">
          <span class="layer-badge" style="background: {linfo['color']}">{linfo['badge']}</span>
          <h2 class="layer-title">{lname}</h2>
          <span class="layer-sub">{ldesc}</span>
        </div>
        <div class="layer-meta">{linfo['stats']}</div>
      </div>
      <div class="cards-grid">
        {''.join(cards_html)}
      </div>
    </section>'''
        sections_html.append(sec)

    html_template = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shardy Animations · OpenXLA SDY 交互式动画课件全景</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='7' height='7' x='1' y='1' rx='1.5' fill='%2338bdf8'/%3E%3Crect width='7' height='7' x='8' y='1' rx='1.5' fill='%23c084fc'/%3E%3Crect width='7' height='7' x='1' y='8' rx='1.5' fill='%23fbbf24'/%3E%3Crect width='7' height='7' x='8' y='8' rx='1.5' fill='%23fb7185'/%3E%3C/svg%3E">
<style>
:root {
  --bg0: #080c18;
  --bg1: #0d1428;
  --panel: rgba(255, 255, 255, .045);
  --panel-hover: rgba(255, 255, 255, .075);
  --panel-brd: rgba(255, 255, 255, .10);
  --panel-brd-hover: rgba(94, 234, 212, .35);
  --ink: #e8eefc;
  --ink-dim: #93a3c4;
  --ink-faint: #5d6d8f;
  --accent: #5eead4;
  --accent2: #818cf8;
  --warn: #fbbf24;
  --bad: #fb7185;
  --ok: #4ade80;
  --mono: "SFMono-Regular", Menlo, Consolas, "DejaVu Sans Mono", "Noto Sans Mono CJK SC", monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: radial-gradient(1200px 900px at 50% -10%, #16224a 0%, var(--bg1) 40%, var(--bg0) 100%);
  color: var(--ink);
  font-family: var(--sans);
  min-height: 100vh;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

a { color: inherit; text-decoration: none; }

.container {
  max-width: 1320px;
  margin: 0 auto;
  padding: 40px 24px 80px;
}

/* Header / Hero */
.hero {
  text-align: center;
  padding: 30px 10px 48px;
  position: relative;
}
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(94, 234, 212, 0.12);
  color: var(--accent);
  border: 1px solid rgba(94, 234, 212, 0.3);
  font-size: 13px;
  font-weight: 600;
  padding: 5px 14px;
  border-radius: 9999px;
  letter-spacing: .08em;
  text-transform: uppercase;
  margin-bottom: 20px;
}
.hero-badge::before {
  content: "";
  display: inline-block;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 10px var(--accent);
}
.hero-title {
  font-size: 46px;
  font-weight: 900;
  letter-spacing: -0.02em;
  line-height: 1.15;
  margin-bottom: 16px;
  background: linear-gradient(135deg, #ffffff 40%, #93a3c4 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hero-desc {
  font-size: 18px;
  color: var(--ink-dim);
  max-width: 820px;
  margin: 0 auto 30px;
  line-height: 1.6;
}
.hero-actions {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 36px;
}
.btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  border-radius: 10px;
  font-size: 15px;
  font-weight: 600;
  transition: all 0.2s ease;
  cursor: pointer;
  border: 1px solid transparent;
}
.btn-primary {
  background: linear-gradient(135deg, #2dd4bf 0%, #0d9488 100%);
  color: #042f2e;
  box-shadow: 0 4px 16px rgba(45, 212, 191, 0.25);
}
.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(45, 212, 191, 0.4);
}
.btn-secondary {
  background: var(--panel);
  color: var(--ink);
  border-color: var(--panel-brd);
}
.btn-secondary:hover {
  background: var(--panel-hover);
  border-color: rgba(255, 255, 255, 0.25);
  transform: translateY(-2px);
}

/* Stat Chips */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  max-width: 960px;
  margin: 0 auto;
}
.stat-card {
  background: var(--panel);
  border: 1px solid var(--panel-brd);
  border-radius: 12px;
  padding: 14px 18px;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.stat-val {
  font-size: 26px;
  font-weight: 800;
  color: var(--accent);
  font-family: var(--mono);
}
.stat-lbl {
  font-size: 13px;
  color: var(--ink-dim);
  margin-top: 2px;
}

/* Search & Filters */
.toolbar {
  position: sticky;
  top: 16px;
  z-index: 100;
  background: rgba(13, 20, 40, 0.85);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border: 1px solid var(--panel-brd);
  border-radius: 14px;
  padding: 14px 18px;
  margin-bottom: 36px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
}
.search-box {
  position: relative;
  width: 100%;
}
.search-input {
  width: 100%;
  background: rgba(8, 12, 24, 0.7);
  border: 1px solid var(--panel-brd);
  color: var(--ink);
  padding: 12px 16px 12px 42px;
  border-radius: 10px;
  font-size: 15px;
  outline: none;
  font-family: var(--sans);
  transition: border-color 0.2s;
}
.search-input:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 2px rgba(94, 234, 212, 0.2);
}
.search-icon {
  position: absolute;
  left: 14px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--ink-faint);
  pointer-events: none;
}
.filter-tabs {
  display: flex;
  align-items: center;
  gap: 8px;
  overflow-x: auto;
  padding-bottom: 4px;
}
.filter-tab {
  padding: 6px 14px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 600;
  background: transparent;
  color: var(--ink-dim);
  border: 1px solid transparent;
  cursor: pointer;
  white-space: nowrap;
  transition: all 0.2s;
}
.filter-tab:hover {
  color: var(--ink);
  background: var(--panel);
}
.filter-tab.active {
  color: var(--bg0);
  background: var(--accent);
  border-color: var(--accent);
}

/* Layer Sections */
.layer-section {
  margin-bottom: 44px;
  transition: opacity 0.3s;
}
.layer-section.hidden {
  display: none;
}
.layer-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 18px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--panel-brd);
}
.layer-title-wrap {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.layer-badge {
  font-size: 13px;
  font-weight: 800;
  padding: 4px 10px;
  border-radius: 6px;
  color: #fff;
  font-family: var(--mono);
}
.layer-title {
  font-size: 22px;
  font-weight: 700;
  color: var(--ink);
}
.layer-sub {
  font-size: 14px;
  color: var(--ink-dim);
  margin-left: 4px;
}
.layer-meta {
  font-size: 13px;
  color: var(--ink-faint);
  font-family: var(--mono);
}

/* Lesson Cards Grid */
.cards-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 16px;
}
.card {
  background: var(--panel);
  border: 1px solid var(--panel-brd);
  border-radius: 12px;
  padding: 18px 20px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
  position: relative;
  overflow: hidden;
}
.card::after {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: var(--card-color, var(--accent));
  opacity: 0;
  transition: opacity 0.25s ease;
}
.card:hover {
  background: var(--panel-hover);
  border-color: var(--panel-brd-hover);
  transform: translateY(-3px);
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
}
.card:hover::after {
  opacity: 1;
}
.card.hidden {
  display: none;
}
.card-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}
.card-code {
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 700;
  color: var(--card-color, var(--accent));
  background: rgba(255, 255, 255, 0.05);
  padding: 2px 8px;
  border-radius: 4px;
}
.card-scenes {
  font-size: 12px;
  color: var(--ink-faint);
  display: flex;
  align-items: center;
  gap: 4px;
}
.card-title {
  font-size: 16px;
  font-weight: 700;
  color: #fff;
  line-height: 1.4;
  margin-bottom: 8px;
}
.card-desc {
  font-size: 13.5px;
  color: var(--ink-dim);
  line-height: 1.55;
  margin-bottom: 16px;
  flex: 1;
}
.card-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-top: 12px;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
}
.card-link-txt {
  font-size: 13px;
  font-weight: 600;
  color: var(--card-color, var(--accent));
  display: flex;
  align-items: center;
  gap: 6px;
}
.card:hover .card-link-txt {
  text-decoration: underline;
}

/* Empty Search State */
#empty-state {
  display: none;
  text-align: center;
  padding: 60px 20px;
  color: var(--ink-dim);
}

/* Footer */
footer {
  text-align: center;
  padding: 50px 20px 20px;
  color: var(--ink-faint);
  font-size: 13.5px;
  border-top: 1px solid var(--panel-brd);
  margin-top: 60px;
}
footer a {
  color: var(--ink-dim);
  text-decoration: underline;
}
footer a:hover {
  color: var(--accent);
}
</style>
</head>
<body>

<div class="container">
  <!-- Hero Section -->
  <header class="hero">
    <div class="hero-badge">Interactive Learning Platform</div>
    <h1 class="hero-title">Shardy Animations</h1>
    <p class="hero-desc">
      把 OpenXLA <strong>Shardy (SDY 方言)</strong> 从核心语法、分片约束、启发式传播一直到分区器产出的低级设备代码，做成完全交互式的动画拆解。
    </p>
    <div class="hero-actions">
      <a class="btn btn-primary" href="intro/index.html">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
        立即开始 · 预备篇体验
      </a>
      <a class="btn btn-secondary" href="https://github.com/openxla/shardy" target="_blank" rel="noopener">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path fill-rule="evenodd" clip-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/></svg>
        OpenXLA / Shardy
      </a>
    </div>

    <!-- Quick Stats -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-val">75</div>
        <div class="stat-lbl">全交互动画课件</div>
      </div>
      <div class="stat-card">
        <div class="stat-val">241</div>
        <div class="stat-lbl">测试 IR 逐字覆盖</div>
      </div>
      <div class="stat-card">
        <div class="stat-val">7 + 1</div>
        <div class="stat-lbl">渐进式学习分层</div>
      </div>
      <div class="stat-card">
        <div class="stat-val">100%</div>
        <div class="stat-lbl">纯静态零依赖运行</div>
      </div>
    </div>
  </header>

  <!-- Sticky Filter & Search Toolbar -->
  <div class="toolbar">
    <div class="search-box">
      <svg class="search-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
      <input type="text" id="search" class="search-input" placeholder="按课号、标题、概念搜索（如 mesh, reshard, L2, L5...）快捷键 /">
    </div>
    <div class="filter-tabs" id="filter-tabs">
      <button class="filter-tab active" data-layer="all">全部 (75)</button>
      <button class="filter-tab" data-layer="intro">预备篇 (1)</button>
      <button class="filter-tab" data-layer="L1">L1 语法 (10)</button>
      <button class="filter-tab" data-layer="L2">L2 传播 (12)</button>
      <button class="filter-tab" data-layer="L3">L3 导入 (11)</button>
      <button class="filter-tab" data-layer="L4">L4 导出 (17)</button>
      <button class="filter-tab" data-layer="L5">L5 降级 (9)</button>
      <button class="filter-tab" data-layer="L6">L6 执行 (11)</button>
      <button class="filter-tab" data-layer="L7">L7 实战 (4)</button>
    </div>
  </div>

  <!-- Content Sections -->
  <main id="course-list">
''' + '\n'.join(sections_html) + '''
  </main>
  
  <div id="empty-state">
    <p style="font-size: 18px; font-weight: 600;">未找到匹配的课件</p>
    <p style="font-size: 14px; margin-top: 6px;">尝试输入不同的关键词或重置筛选条件</p>
  </div>

  <footer>
    <p>Shardy Animations · OpenXLA SDY Dialect Interactive Courseware</p>
    <p style="margin-top: 6px;">纯静态架构，无 CDN 依赖，支持双击离线浏览与 GitHub Pages 托管部署。</p>
  </footer>
</div>

<script>
// Search & Filter Controller
const searchInput = document.getElementById("search");
const filterTabs = document.querySelectorAll(".filter-tab");
const sections = document.querySelectorAll(".layer-section");
const cards = document.querySelectorAll(".card");
const emptyState = document.getElementById("empty-state");

let currentFilter = "all";
let currentQuery = "";

function applyFilters() {
  let visibleCardsCount = 0;
  
  sections.forEach(sec => {
    const secLayer = sec.getAttribute("data-layer-id");
    const matchesTab = (currentFilter === "all" || currentFilter === secLayer);
    
    let secHasVisibleCard = false;
    const secCards = sec.querySelectorAll(".card");
    
    secCards.forEach(card => {
      const code = card.getAttribute("data-code") || "";
      const title = card.getAttribute("data-title") || "";
      const sub = card.getAttribute("data-sub") || "";
      const matchesSearch = !currentQuery || 
        code.includes(currentQuery) || 
        title.includes(currentQuery) || 
        sub.includes(currentQuery);
      
      if (matchesTab && matchesSearch) {
        card.classList.remove("hidden");
        secHasVisibleCard = true;
        visibleCardsCount++;
      } else {
        card.classList.add("hidden");
      }
    });
    
    if (secHasVisibleCard) {
      sec.classList.remove("hidden");
    } else {
      sec.classList.add("hidden");
    }
  });
  
  if (visibleCardsCount === 0) {
    emptyState.style.display = "block";
  } else {
    emptyState.style.display = "none";
  }
}

filterTabs.forEach(tab => {
  tab.addEventListener("click", () => {
    filterTabs.forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    currentFilter = tab.getAttribute("data-layer");
    applyFilters();
  });
});

searchInput.addEventListener("input", (e) => {
  currentQuery = e.target.value.trim().toLowerCase();
  applyFilters();
});

// Shortcut / to focus search
window.addEventListener("keydown", (e) => {
  if (e.key === "/" && document.activeElement !== searchInput) {
    e.preventDefault();
    searchInput.focus();
  }
});
</script>
</body>
</html>'''

    out_path = base_dir / "index.html"
    out_path.write_text(html_template, encoding="utf-8")
    print(f"Generated {out_path} ({len(html_template)} bytes)")

if __name__ == "__main__":
    generate_portal()
