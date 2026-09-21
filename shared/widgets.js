/* ==========================================================================
   Shardy 课件 — 可视元件库（W）
   --------------------------------------------------------------------------
   在 engine.js 的 U（底层：el / tensorGrid / meshGrid / hl）之上，
   提供课件常用的组合元件。全部返回 HTMLElement 或 HTML 字符串。
   ========================================================================== */
'use strict';

const W = {
  /* ---------------------------------------------------------------- 容器 */
  /* 卡片：标题 + 描述（返回 HTML 字符串，便于拼进 innerHTML） */
  card(title, desc, opts = {}) {
    const w = opts.width ? `width:${opts.width}px;` : '';
    const c = opts.color ? `border-color:${opts.color}55;` : '';
    return `<div class="card" style="${w}${c}min-width:0">
      <div class="card-t" ${opts.color ? `style="color:${opts.color}"` : ''}>${title}</div>
      <div class="card-d">${desc}</div></div>`;
  },

  /* 面板：可挂子元素的 card */
  panel(title) {
    const e = U.el('div', { class: 'card', style: 'min-width:0' });
    if (title) e.innerHTML = `<div class="card-t">${title}</div>`;
    const body = U.el('div', { class: 'card-d' });
    e.appendChild(body);
    return { el: e, body };
  },

  /* 箭头（可带标签） */
  arrow(label = '', cls = 'anim') {
    return `<div class="arrow ${cls}" style="font-size:26px">${label ? `<span class="mono small dim">${label}</span> ` : ''}⟹</div>`;
  },

  /* ------------------------------------------------------------ IR 展示 */
  /* before / after 对照框。before/after 为纯文本 IR（自动高亮） */
  irPair(before, after, opts = {}) {
    const h = opts.height ? `max-height:${opts.height}px;` : '';
    const box = (side, title, code) => `
      <div class="irbox ${side}">
        <div class="irh">${title}</div>
        <pre style="${h}">${U.hl(code)}</pre>
      </div>`;
    let html = '';
    if (before != null) html += box('in', opts.inTitle || '输入 IR', before);
    if (before != null && after != null) html += `<div class="arrow" style="font-size:22px;align-self:center">⟹</div>`;
    if (after != null) html += box('out', opts.outTitle || '输出 IR', after);
    return `<div class="irpair">${html}</div>`;
  },

  /* 单个 IR 框 */
  irBox(title, code, side = '', opts = {}) {
    const h = opts.height ? `max-height:${opts.height}px;` : '';
    return `<div class="irbox ${side}"><div class="irh">${title}</div><pre style="${h}">${U.hl(code)}</pre></div>`;
  },

  /* 源文件脚注 */
  srcNote(files) {
    const list = Array.isArray(files) ? files : [files];
    return `<div class="srcnote">源测试文件：<br>${list.map(f => '· shardy/dialect/sdy/' + f).join('<br>')}</div>`;
  },

  /* -------------------------------------------------------------- 步骤 */
  /* 步骤指示器：返回 {el, set(i)} */
  stepper(n, labels = []) {
    const el = U.el('div', { class: 'stepper' });
    const dots = [];
    for (let i = 0; i < n; i++) {
      const s = U.el('div', { class: 'stp', title: labels[i] || ('步骤 ' + (i + 1)) });
      s.textContent = String(i + 1);
      el.appendChild(s); dots.push(s);
    }
    return {
      el,
      set(i) { dots.forEach((d, k) => { d.classList.toggle('on', k === i); d.classList.toggle('done', k < i); }); }
    };
  },

  /* -------------------------------------------------------------- 练习 */
  /* 练习题：问题 + 可揭晓答案。返回 HTMLElement */
  exercise(question, answer) {
    const e = U.el('div', { class: 'exercise' });
    e.innerHTML = `<div class="exq"><b class="hl-w">练习</b> ${question}</div>
      <div class="exa hidden">${answer}</div>
      <button class="exbtn">揭晓答案</button>`;
    const ans = e.querySelector('.exa'), btn = e.querySelector('.exbtn');
    btn.onclick = () => {
      const hidden = ans.classList.toggle('hidden');
      btn.textContent = hidden ? '揭晓答案' : '收起答案';
    };
    return e;
  },

  /* -------------------------------------------------------------- 表格 */
  /* headers: [..]; rows: [[..]]; opts.hot: 高亮行索引数组; opts.mono: 等宽列索引数组 */
  table(headers, rows, opts = {}) {
    const hot = opts.hot || [], mono = opts.mono || [];
    const th = headers.map(h => `<th>${h}</th>`).join('');
    const tr = rows.map((r, i) =>
      `<tr class="${hot.includes(i) ? 'hot' : ''}">` +
      r.map((c, j) => `<td class="${mono.includes(j) ? 'mono' : ''}">${c}</td>`).join('') +
      '</tr>').join('');
    return `<table class="rtable"><tr>${th}</tr>${tr}</table>`;
  },

  /* 因子表（sharding rule 用）：cols = 因子名列表，rows = [{name, cells:[..]}] */
  factorTable(cols, rows) {
    const th = ['', ...cols, '显式复制轴'].map(h => `<th>${h}</th>`).join('');
    const tr = rows.map(r =>
      `<tr><td>${r.name}</td>` +
      cols.map((_, i) => {
        const v = r.cells[i];
        if (v == null) return '<td class="na">N</td>';
        return `<td class="${r.chg && r.chg.includes(i) ? 'chg' : 'ax'}">${v}</td>`;
      }).join('') +
      `<td>${r.rep || ''}</td></tr>`).join('');
    return `<table class="ftable"><tr>${th}</tr>${tr}</table>`;
  },

  /* -------------------------------------------------------------- 徽标 */
  badge(text, kind = 'ok') { return `<span class="badge ${kind}">${text}</span>`; },

  /* 轴 chip（带开/闭标记） */
  axisChip(name, ci, opts = {}) {
    const open = opts.open ? '<span class="faint">, ?</span>' : '';
    const rep = opts.replicated ? ' style="opacity:.6"' : '';
    const sub = opts.sub ? `<span class="faint">:(${opts.sub})</span>` : '';
    const pri = opts.priority != null ? `<span class="faint">p${opts.priority}</span>` : '';
    return `<span class="chip c${ci % 6}"${rep}>"${name}"${sub}${open}${pri}</span>`;
  },

  /* 分片属性的可读展开：dimShardings = [[{axis, open}], ...] */
  shardingText(meshName, dims, opts = {}) {
    const d = dims.map(dim => {
      const inner = dim.map(a => a.sub ? `"${a.name}":(${a.sub})` : `"${a.name}"`).join(', ');
      return `{${inner}${dim.open ? ', ?' : ''}}`;
    }).join(', ');
    const rep = opts.replicated && opts.replicated.length
      ? `, replicated={${opts.replicated.join(', ')}}` : '';
    const unr = opts.unreduced && opts.unreduced.length
      ? `, unreduced={${opts.unreduced.join(', ')}}` : '';
    return `#sdy.sharding<@${meshName}, [${d}]${rep}${unr}>`;
  },

  /* ------------------------------------------------------------ 设备/张量 */
  /* 一行设备。n 个，opts.label(i) 覆盖标签，opts.size 像素 */
  devRow(n, opts = {}) {
    const row = U.el('div', { class: 'row', style: `gap:${opts.gap ?? 18}px;justify-content:center` });
    const size = opts.size || 88;
    const boxes = [];
    for (let i = 0; i < n; i++) {
      const col = U.el('div', { class: 'col', style: 'gap:6px;align-items:center' });
      const lbl = U.el('div', { class: 'small faint mono' });
      lbl.textContent = opts.label ? opts.label(i) : ('dev' + i);
      const box = U.el('div', { class: 'dev', style: `width:${size}px;height:${size}px;flex-wrap:wrap;gap:3px;padding:5px;align-content:center` });
      box.style.setProperty('--c', `var(--ax${(opts.colorBase || 0) + i})`);
      col.appendChild(lbl); col.appendChild(box);
      row.appendChild(col);
      boxes.push(box);
    }
    return { el: row, boxes };
  },

  /* 往设备盒子里填「块」 */
  fillBox(box, chunks, colorIdx, opts = {}) {
    box.innerHTML = '';
    (Array.isArray(chunks) ? chunks : [chunks]).forEach(c => {
      const e = U.el('div', { class: 'chunk' });
      e.textContent = c;
      e.style.background = `var(--ax${colorIdx % 6})`;
      if (String(c).length > 4) e.style.fontSize = '8px';
      e.style.width = (opts.cw || 26) + 'px';
      e.style.height = (opts.ch || 26) + 'px';
      box.appendChild(e);
    });
  },

  /* 配色：设备 id -> 颜色索引 */
  devColor: id => id % 6,

  /* 常用网格：2x2 设备 */
  mesh22(inner, opts = {}) {
    const g = U.el('div', { class: 'mesh' });
    const cs = opts.cellSize || 92;
    g.style.gridTemplateColumns = `repeat(2, ${cs}px)`;
    g.style.gridTemplateRows = `repeat(2, ${cs}px)`;
    for (let id = 0; id < 4; id++) {
      const xi = Math.floor(id / 2), yi = id % 2;
      const d = U.el('div', { class: 'dev', 'data-dev': id });
      d.style.setProperty('--c', `var(--ax${W.devColor(id)})`);
      d.innerHTML = `<span class="dev-id">${id}</span>`;
      if (inner) d.insertAdjacentHTML('beforeend', inner(id, xi, yi));
      g.appendChild(d);
    }
    return g;
  },

  /* ---------------------------------------------------------- 小工具 */
  /* 标题行 + 副文本 */
  head(title, sub) {
    return `<div class="col" style="gap:2px;align-items:center">
      <div class="mid" style="font-weight:700">${title}</div>
      ${sub ? `<div class="small faint">${sub}</div>` : ''}</div>`;
  },

  /* 公式条 */
  formula(html, warn = false) {
    return `<div class="formula${warn ? ' warn' : ''}">${html}</div>`;
  },

  /* 一行 key: value */
  kv(k, v) {
    return `<div class="row" style="gap:8px"><span class="faint small">${k}</span><span class="mono small">${v}</span></div>`;
  },

  /* 时间轴辅助：按 sequence 依次执行，每步间隔 stepMs 起 */
  seq(tl, steps, start = 600, stepMs = 2200) {
    steps.forEach((fn, i) => tl.at(start + i * stepMs, fn));
  }
};

/* ------------------------------------------------------------------ *
 * 错误用例渲染（L1 起通用）：stepper + 违规 IR + 报错 + 根因
 * 用法：
 *   const wrap = W.errLayout(root);
 *   W.errScenes(wrap, tl, [{ir, err, why}, ...], {finalIr, finalErr, finalWhy});
 * ------------------------------------------------------------------ */
W.errLayout = function (root, opts = {}) {
  const irTitle = opts.irTitle || '违规 IR';
  const errTitle = opts.errTitle || '校验器报错';
  const whyTitle = opts.whyTitle || '为什么';
  const bad = opts.errColor || 'var(--bad)';
  const brd = opts.errBorder || 'rgba(251,113,133,.45)';
  const fg = opts.errFg || '#ffc9d0';
  const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
  wrap.innerHTML = `
    <div class="row" style="gap:12px;align-items:center;justify-content:center" id="stepper"></div>
    <div class="row" style="gap:14px;align-items:stretch;width:100%">
      <div class="irbox in" style="flex:1.2"><div class="irh">${irTitle}<span class="faint" id="cnt" style="float:right"></span></div>
        <pre id="bad" style="min-height:120px;font-size:11.5px"></pre></div>
      <div class="col" style="flex:1;gap:9px">
        <div class="card" style="border-color:${brd}">
          <div class="card-t" style="color:${bad};font-size:12px">${errTitle}</div>
          <div class="card-d mono" id="err" style="font-size:11px;line-height:1.55;color:${fg}"></div></div>
        <div class="card"><div class="card-t" style="font-size:12px">${whyTitle}</div>
          <div class="card-d" id="why" style="font-size:12.5px"></div></div>
      </div>
    </div>`;
  root.appendChild(wrap);
  return wrap;
};

W.errScenes = function (wrap, tl, cases, opts = {}) {
  const st = W.stepper(cases.length);
  wrap.querySelector('#stepper').appendChild(st.el);
  const bad = wrap.querySelector('#bad'), err = wrap.querySelector('#err'),
    why = wrap.querySelector('#why'), cnt = wrap.querySelector('#cnt');
  const stepMs = opts.stepMs || 2400;
  cases.forEach((c, i) => tl.at(700 + i * stepMs, () => {
    st.set(i);
    if (cnt) cnt.textContent = `${i + 1} / ${cases.length}`;
    bad.innerHTML = U.hl(c.ir);
    err.innerHTML = c.err;
    why.innerHTML = c.why;
  }));
  tl.at(700 + cases.length * stepMs, () => {
    st.set(-1);
    bad.innerHTML = U.hl(opts.finalIr || '// 回顾：这些写法的共同问题是什么？');
    if (cnt) cnt.textContent = '小结';
    err.innerHTML = opts.finalErr || '';
    why.innerHTML = opts.finalWhy || '';
  });
};
