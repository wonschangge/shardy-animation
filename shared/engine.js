/* ==========================================================================
   Shardy 课件 — 共享动画引擎
   --------------------------------------------------------------------------
   由 intro/app.js 提炼而来，供全部课件复用（单一来源）。
   设计要点：
   - 舞台固定 1280x720，按窗口等比缩放，保证任何屏幕上排版一致。
   - 每一"幕"(scene) 由各课 lesson.js 声明，本文件负责挂载、时间轴、自动播放。
   - 时间轴只有在"播放中"才前进，因此暂停是真正的时间冻结（含幕内分步动画）。
   - 经典脚本（非 ES module）：保证 file:// 双击直接打开也能工作。
   ========================================================================== */
'use strict';

/* ---------------------------------------------------------------- 工具函数 */
const U = {
  el(tag, attrs = {}, kids = []) {
    const e = document.createElement(tag);
    for (const k in attrs) {
      if (k === 'class') e.className = attrs[k];
      else if (k === 'html') e.innerHTML = attrs[k];
      else if (k === 'text') e.textContent = attrs[k];
      else if (k === 'style') e.setAttribute('style', attrs[k]);
      else e.setAttribute(k, attrs[k]);
    }
    (Array.isArray(kids) ? kids : [kids]).forEach(c => {
      if (c == null) return;
      e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return e;
  },

  /* 轴颜色：索引 -> CSS class（c0..c5 循环） */
  axc: i => 'axc' + (i % 6),
  axcN: i => 'c' + (i % 6),

  /* 轴名 -> 颜色索引，用于 mesh 轴着色（同一幕内应保持一致映射） */
  colorMap(axes) {
    const m = {};
    axes.forEach((a, i) => { m[a] = i; });
    return m;
  },

  /* 生成带色块的轴 chip */
  chip(name, ci, extra = '') {
    return `<span class="chip c${ci % 6} ${extra}"><i class="sw sw-c${ci % 6}"></i>${name}</span>`;
  },

  /* 设备网格：axes = [['x',2],['y',2]]，返回 .mesh 元素；cell(devIndex, xi, yi) 可注入内容 */
  meshGrid(axes, opts = {}) {
    const [a0, a1] = axes;
    const cell = opts.cell || (() => '');
    const cs = opts.cellSize || 72;
    const g = U.el('div', { class: 'mesh' });
    g.style.gridTemplateColumns = `repeat(${a1[1]}, ${cs}px)`;
    g.style.gridTemplateRows = `repeat(${a0[1]}, ${cs}px)`;
    // 行 = 第 0 轴（行优先），与 sdy 的 iota 设备编号一致
    for (let i = 0; i < a0[1]; i++) {
      for (let j = 0; j < a1[1]; j++) {
        const id = i * a1[1] + j;
        const d = U.el('div', { class: `dev ${U.axc(i)}`, 'data-dev': id });
        d.style.setProperty('--c', `var(--ax${i % 6})`);
        d.innerHTML = `<span class="dev-id">${id}</span>`;
        const inner = cell(id, i, j);
        if (inner) d.appendChild(typeof inner === 'string' ? U.el('span', { html: inner }) : inner);
        g.appendChild(d);
      }
    }
    return g;
  },

  /* 张量网格：rows/cols 为格子数，owner(r,c) 返回颜色索引（-1 表示无色）
     opts.text(r,c) 返回文字则格子显示文字（用于块内标设备号） */
  tensorGrid(rows, cols, owner, opts = {}) {
    const cs = opts.cell || 15, gap = opts.gap ?? 2;
    const g = U.el('div', { class: 'tgrid' });
    g.style.gridTemplateColumns = `repeat(${cols}, ${cs}px)`;
    g.style.gridTemplateRows = `repeat(${rows}, ${cs}px)`;
    g.style.gap = gap + 'px';
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const o = owner ? owner(r, c) : -1;
        const txt = opts.text ? opts.text(r, c) : null;
        const cell = U.el('div', { class: 'tcell' + (txt ? ' lbl' : '') });
        if (o >= 0) {
          cell.style.background = `var(--ax${o % 6})`;
          cell.style.opacity = '.92';
        }
        if (txt) cell.textContent = txt;
        g.appendChild(cell);
      }
    }
    return g;
  },

  /* 在容器上叠加分片闸刀线。cuts = [{dir:'v'|'h', pos:像素}] */
  cuts(container, list) {
    list.forEach(c => {
      const e = U.el('div', { class: `cut ${c.dir}` });
      if (c.dir === 'v') e.style.left = c.pos + 'px'; else e.style.top = c.pos + 'px';
      container.appendChild(e);
    });
    return container.querySelectorAll('.cut');
  },

  /* 转义 HTML */
  esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); },

  /* 简易 MLIR 高亮：单遍分词，避免二次替换破坏已插入的 HTML 标签 */
  hl(src) {
    const esc = U.esc;
    const RE = new RegExp([
      '(\\/\\/[^\\n]*)',                                      // 1 注释
      '(#[A-Za-z_][A-Za-z0-9_.]*)',                           // 2 #sdy.xxx / #mpmd.xxx 属性
      '(-(?:sdy|mpmd)-[a-z0-9-]+)',                           // 3 命令行 flag
      '(@[A-Za-z_][A-Za-z0-9_]*)',                            // 4 @mesh 符号引用
      '("[^"\\n]*")',                                          // 5 轴名等字符串
      '\\b((?:sdy|stablehlo|mpmd|func)\\.[a-z_][a-z0-9_]*)',  // 6 带命名空间的算子
      '\\b(mesh_tensor|func\\.func|module|return|tensor)\\b'  // 7 关键字
    ].join('|'), 'g');
    const CLS = ['', 'com', 'at', 'flag', 'nm', 'st', 'kw', 'kw'];
    return src.split('\n').map(line => {
      let out = '', last = 0, m;
      RE.lastIndex = 0;
      while ((m = RE.exec(line)) !== null) {
        if (m.index > last) out += esc(line.slice(last, m.index));
        let cls = 'kw';
        for (let k = 1; k <= 7; k++) if (m[k] !== undefined) { cls = CLS[k]; break; }
        out += '<span class="' + cls + '">' + esc(m[0]) + '</span>';
        last = m.index + m[0].length;
      }
      out += esc(line.slice(last));
      return out;
    }).join('\n');
  }
};

/* -------------------------------------------------------------- 时间轴对象 */
class Timeline {
  constructor() { this.steps = []; this.reps = []; this.t = 0; this.i = 0; }
  /* 在 ms 毫秒后执行一次 */
  at(ms, fn) { this.steps.push({ ms, fn, done: false }); this.steps.sort((a, b) => a.ms - b.ms); return this; }
  /* 每 ms 毫秒执行一次（第一次在 ms 后） */
  every(ms, fn) { this.reps.push({ ms, fn, next: ms }); return this; }
  reset() { this.t = 0; this.i = 0; this.steps.forEach(s => s.done = false); this.reps.forEach(r => r.next = r.ms); }
  tick(dt) {
    this.t += dt;
    while (this.i < this.steps.length && this.steps[this.i].ms <= this.t) {
      const s = this.steps[this.i++];
      if (!s.done) { s.done = true; try { s.fn(); } catch (e) { console.error(e); } }
    }
    this.reps.forEach(r => {
      if (this.t >= r.next) { r.next += r.ms; try { r.fn(); } catch (e) { console.error(e); } }
    });
  }
}

/* ------------------------------------------------------------------- 引擎 */
const App = {
  idx: 0, playing: true, elapsed: 0, last: 0, tl: null,
  scenes: [], opts: {},
  $: id => document.getElementById(id),

  /* scenes: SCENES 数组；opts: {kicker, codeCap} */
  init(scenes, opts = {}) {
    this.scenes = scenes || [];
    this.opts = opts || {};
    this.fit();
    window.addEventListener('resize', () => this.fit());
    this.$('btn-next').onclick = () => { this.go(this.idx + 1); };
    this.$('btn-prev').onclick = () => { this.go(this.idx - 1); };
    this.$('btn-play').onclick = () => this.toggle();
    this.$('btn-restart').onclick = () => { this.go(0); this.playing = true; this.syncPlay(); };
    document.addEventListener('keydown', e => {
      if (e.key === 'ArrowRight' || e.key === 'PageDown') { e.preventDefault(); this.go(this.idx + 1); }
      else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); this.go(this.idx - 1); }
      else if (e.key === ' ') { e.preventDefault(); this.toggle(); }
      else if (e.key === 'Home') this.go(0);
    });
    this.buildDots();
    this.go(0);
    requestAnimationFrame(t => { this.last = t; this.loop(t); });
  },

  /* 舞台等比缩放 */
  fit() {
    const s = Math.min(window.innerWidth / 1280, (window.innerHeight - 54) / 720);
    this.$('stage').style.transform = `scale(${Math.min(s, 1.35)})`;
  },

  buildDots() {
    const d = this.$('dots'); d.innerHTML = '';
    this.scenes.forEach((s, i) => {
      const dot = U.el('div', { class: 'dot', title: (i + 1) + '. ' + String(s.title).replace(/<[^>]+>/g, '') });
      dot.onclick = () => this.go(i);
      d.appendChild(dot);
    });
  },

  toggle() { this.playing = !this.playing; this.syncPlay(); },
  syncPlay() { this.$('btn-play').textContent = this.playing ? '⏸' : '▶'; },

  go(i) {
    const n = this.scenes.length;
    if (!n) return;
    this.idx = ((i % n) + n) % n;
    const sc = this.scenes[this.idx];
    this.elapsed = 0;
    this.tl = new Timeline();
    this.$('scene-kicker').innerHTML = sc.kicker || this.opts.kicker || 'Shardy 课件';
    this.$('scene-title').innerHTML = sc.title;
    this.$('scene-sub').innerHTML = sc.sub || '';
    this.$('side-cap').textContent = sc.codeCap || this.opts.codeCap || '对应 IR';
    this.$('code').innerHTML = sc.code ? U.hl(sc.code) : '<span class="com">（本幕无 IR）</span>';
    this.$('caption').innerHTML = sc.caption || '';
    const v = this.$('visual'); v.innerHTML = '';
    const holder = U.el('div', { class: 'scene' });
    v.appendChild(holder);
    try { sc.build(holder, this.tl); } catch (e) { console.error('scene build failed', e); holder.textContent = '场景渲染出错: ' + e.message; }
    document.querySelectorAll('.dot').forEach((d, k) => d.classList.toggle('on', k === this.idx));
    this.$('counter').textContent = (this.idx + 1) + ' / ' + n;
  },

  loop(t) {
    const dt = Math.min(t - this.last, 100); this.last = t;
    if (this.playing && this.scenes.length) {
      this.elapsed += dt;
      this.tl.tick(dt);
      const dur = this.scenes[this.idx].duration || 11000;
      const p = Math.min(this.elapsed / dur, 1);
      this.$('progress-fill').style.width = (p * 100) + '%';
      if (p >= 1) this.go(this.idx + 1);
    }
    requestAnimationFrame(tt => this.loop(tt));
  }
};
