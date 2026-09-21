/* ==========================================================================
   L1-01 · mesh-and-devices
   --------------------------------------------------------------------------
   覆盖：ir/test/mesh_parse_print.mlir (31) · ir/test/mesh_verification.mlir (34)
   目标：讲透 sdy.mesh —— 逻辑网格的声明、设备编号规则、打印规范化、8 类校验错误。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------- 1 为什么需要网格 */
{
  kicker: 'L1-01 · 逻辑网格',
  title: '为什么需要 <span class="mono hl-a">sdy.mesh</span>？',
  sub: '一维的设备列表无法表达"沿哪个方向切"。把设备<b>重新排列成多维数组</b>并给每个维度起名字，才能描述分片。',
  caption: '网格是设备的<b>逻辑视图</b>：同一批设备可以有多种网格视图，设备编号决定数据实际落在哪块硬件上。',
  code: `// 4 台设备，一维列表：0 1 2 3
// 无法表达"按行切"还是"按列切"

// 排成 2x2 并命名两个轴：
sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// 之后所有分片都只引用轴名：
//   [{"x"}, {"y"}]   第 0 维沿 x 切，第 1 维沿 y 切`,
  duration: 11000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'scene-inner' });
    wrap.style.gap = '34px';
    wrap.innerHTML = `
      <div class="col" style="gap:10px;align-items:center">
        <div class="mid faint">① 一维设备列表</div>
        <div class="row" id="flat" style="gap:8px"></div>
        <div class="small faint">只能表达"第几台"</div>
      </div>
      <div class="arrow anim" style="font-size:30px">⟹</div>
      <div class="col" style="gap:10px;align-items:center">
        <div class="mid faint">② 2x2 命名网格</div>
        <div id="grid"></div>
        <div class="small faint" id="note" style="height:20px"></div>
      </div>`;
    root.appendChild(wrap);

    const flat = wrap.querySelector('#flat');
    for (let i = 0; i < 4; i++) {
      const d = U.el('div', { class: 'dev', style: 'width:52px;height:52px' });
      d.style.setProperty('--c', `var(--ax${i})`);
      d.innerHTML = `<span class="dev-id">${i}</span>`;
      flat.appendChild(d);
    }

    const g = W.mesh22(null, { cellSize: 86 });
    g.style.opacity = '.12';
    g.style.transition = 'opacity .6s';
    wrap.querySelector('#grid').appendChild(g);
    const note = wrap.querySelector('#note');

    tl.at(900, () => { note.innerHTML = '把设备排成 2 行 2 列'; });
    tl.at(1400, () => {
      g.style.opacity = '1';
      g.querySelectorAll('.dev').forEach((d, i) => setTimeout(() => {
        d.classList.add('devpop'); setTimeout(() => d.classList.remove('devpop'), 520);
      }, i * 150));
    });
    tl.at(3200, () => { note.innerHTML = '<span class="chip c0"><i class="sw sw-c0"></i>"x" = 2</span> 决定<b>行</b>'; });
    tl.at(4800, () => { note.innerHTML = '<span class="chip c1"><i class="sw sw-c1"></i>"y" = 2</span> 决定<b>列</b>'; });
    tl.at(6400, () => { note.innerHTML = '<span class="dim">轴名随便取：</span><span class="mono">"data" / "model" / "tp"</span><span class="dim"> 都行</span>'; });
    tl.at(8200, () => {
      note.innerHTML = '<span class="hl-w">轴大小 = 该方向上的设备数</span>';
      g.querySelectorAll('.dev').forEach((d, i) => setTimeout(() => {
        d.classList.add('lit'); setTimeout(() => d.classList.remove('lit'), 800);
      }, i * 130));
    });
  }
},

/* --------------------------------------------------------- 2 语法拆解 */
{
  kicker: 'L1-01 · 逻辑网格',
  title: '拆开看：<span class="mono hl-a">sdy.mesh</span> 的语法',
  sub: '一条 <span class="mono">sdy.mesh</span> 只有三个部分：符号名、轴列表、可选的设备编号列表。',
  caption: 'sdy.mesh 是 <b>Symbol</b> 操作，必须出现在 module 的符号表里；分片属性通过 <span class="mono">@名字</span> 引用它。',
  code: `sdy.mesh @mesh_xy = <["x"=2, "y"=2]>
//        ~~~~~~~~   ~~~~~~~~~~~~~~~~~~
//        符号名      轴列表：名字=大小

sdy.mesh @m = <["a"=3, "b"=2], device_ids=[0, 2, 4, 1, 3, 5]>
//                              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                              可选：显式设备顺序

// 语法（来自 SDY 文法）：
//   mesh ::= "@" name "=" "<" axes ("," device_ids)? ">"
//   axis ::= str "=" int`,
  duration: 12000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:18px;align-items:center;width:100%' });
    wrap.innerHTML = `
      <div class="mono" id="big" style="font-size:19px;padding:14px 20px;border-radius:10px;
           background:rgba(0,0,0,.3);border:1px solid var(--panel-brd);white-space:nowrap">
        sdy.mesh <span id="p1">@mesh_xy</span> = &lt;<span id="p2">["x"=2, "y"=2]</span><span id="p3">, device_ids=[0,2,4,1,3,5]</span>&gt;
      </div>
      <div class="row" style="gap:14px;align-items:stretch" id="cards"></div>
      <div class="formula" id="msg" style="min-height:44px;display:flex;align-items:center;text-align:center"></div>`;
    root.appendChild(wrap);

    const p1 = wrap.querySelector('#p1'), p2 = wrap.querySelector('#p2'), p3 = wrap.querySelector('#p3');
    const cards = wrap.querySelector('#cards');
    const msg = wrap.querySelector('#msg');
    const mk = () => {
      const c = U.el('div', { class: 'card', style: 'width:238px;opacity:.25;transition:.35s' });
      cards.appendChild(c); return c;
    };
    const c1 = mk(), c2 = mk(), c3 = mk();
    c1.innerHTML = W.card('① 符号名 <span class="mono">@mesh_xy</span>',
      '网格的名字。分片属性用 <span class="mono">@mesh_xy</span> 引用它。<br>同一个 module 里名字唯一。');
    c2.innerHTML = W.card('② 轴列表 <span class="mono">["x"=2, "y"=2]</span>',
      '轴的<span class="hl-a">名字</span>与<span class="hl-a">大小</span>。<br>大小必须 ≥ 1，名字不可重复。');
    c3.innerHTML = W.card('③ 设备编号 <span class="mono">device_ids=[…]</span>',
      '<span class="dim">可选。</span>不写 = 默认行优先编号 iota。<br>写了 = 自定义逻辑位置到设备的映射。');

    const hl = (el, on) => { el.style.color = on ? 'var(--accent)' : ''; el.style.fontWeight = on ? '700' : ''; };
    const pick = (i) => {
      [c1, c2, c3].forEach((c, k) => c.style.opacity = k === i ? '1' : '.25');
      hl(p1, i === 0); hl(p2, i === 1); hl(p3, i === 2);
    };
    tl.at(700, () => { pick(0); msg.innerHTML = '符号名：用来被 <span class="mono">#sdy.sharding&lt;@mesh_xy, …&gt;</span> 引用'; });
    tl.at(3400, () => { pick(1); msg.innerHTML = '轴列表：每个轴是 <span class="mono">"名字"=大小</span>，行优先展开成设备编号'; });
    tl.at(6200, () => { pick(2); msg.innerHTML = '设备编号可选：省略时等价于 <span class="mono">iota(轴大小之积)</span>'; });
    tl.at(9000, () => {
      pick(-1);
      msg.innerHTML = '设备总数 = <span class="mono">∏ 轴大小</span> = 2×2 = 4，必须与设备编号个数一致';
    });
  }
},

/* --------------------------------------------------------- 3 四种形态 */
{
  kicker: 'L1-01 · 逻辑网格',
  title: '四种形态：从<span class="hl-a">占位</span>到<span class="hl-a">自定义顺序</span>',
  sub: '测试文件里 10 条合法声明，可以归成四类。分清它们，后面看任何 IR 都不会卡住。',
  caption: '注意第 ④ 类的陷阱：写了 <span class="mono">device_ids=[0,1,2,3]</span> 看似显式，但它正好等于默认 iota，<b>打印时会被省略</b>。',
  code: `// ① 空网格：占位符，传播时会被替换成真实网格
sdy.mesh @empty_mesh = <[]>

// ② maximal-sharding 网格：无轴 + 单个设备 ID
sdy.mesh @maximal_mesh_0 = <[], device_ids=[0]>
sdy.mesh @maximal_mesh_3 = <[], device_ids=[3]>

// ③ 带轴 + 隐式编号（= iota）
sdy.mesh @single_axis_of_size_2 = <["a"=2]>
sdy.mesh @two_axes = <["a"=2, "b"=1]>

// ④ 带轴 + 显式编号
sdy.mesh @single_axis_explicit_device_ids = <["a"=2], device_ids=[1, 0]>
sdy.mesh @two_axes_explicit_device_ids = <["a"=2, "b"=1], device_ids=[1, 0]>`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;flex-wrap:wrap;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:44px;display:flex;align-items:center;text-align:center"></div>`;
    root.appendChild(wrap);

    const defs = [
      { t: '① 空网格', ir: '<[]>', d: '没有轴、没有设备编号。<br>只是一个<b>占位符</b>，传播阶段会被替换成真实网格。', c: '#93c5fd' },
      { t: '② maximal-sharding', ir: '<[], device_ids=[3]>', d: '无轴 + 单个设备 ID，表示"<b>全部数据都在这台设备上</b>"。<br>用于单设备计算。', c: '#4ade80' },
      { t: '③ 带轴 + 隐式编号', ir: '<["a"=2, "b"=1]>', d: '设备编号 = <span class="mono">iota(∏轴大小)</span>，行优先。<br>例：a=0,b=0→dev0；a=1,b=0→dev1。', c: '#c084fc' },
      { t: '④ 带轴 + 显式编号', ir: '<["a"=2], device_ids=[1, 0]>', d: '自定义逻辑位置到设备的映射。<br>a=0→<b>dev1</b>，a=1→<b>dev0</b>（顺序反了）。', c: '#fbbf24' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s' });
      e.innerHTML = `<div class="card-t" style="color:${d.c}">${d.t}</div>
        <div class="mono small" style="margin:5px 0 6px;color:#cfe0ff">sdy.mesh @m = ${U.esc(d.ir)}</div>
        <div class="card-d">${d.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    const tips = [
      '空网格常在传播前出现，表示"还没决定用哪个网格"。',
      'maximal-sharding 网格的总大小是 1，但它<b>不算</b>"大小为 1 的普通网格"。',
      '行优先：最后一个轴变化最快 —— 与 sdy 内部 iota 编号一致。',
      '写成 <span class="mono">device_ids=[0, 1]</span> 时是 iota，打印会被省略，看不出"显式"过。',
    ];
    defs.forEach((d, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.35'; e.style.borderColor = k === i ? d.c + '88' : ''; });
      msg.innerHTML = tips[i];
    }));
    tl.at(14400, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = '四类的共同约束：<span class="mono">∏ 轴大小 == device_ids 个数</span>'; });
  }
},

/* ----------------------------------------------------- 4 设备编号规则 */
{
  kicker: 'L1-01 · 逻辑网格',
  title: '设备编号：<span class="hl-a">逻辑位置</span> → <span class="hl-a">物理设备</span>',
  sub: '<span class="mono">device_ids</span> 决定"逻辑位置 i 上放的是哪台设备"。不写就是行优先 iota。',
  caption: '这是 SDY 支持 <span class="mono">PositionalSharding</span> 之类任意设备分配的关键：网格可以自定义设备顺序。',
  code: `// 默认（隐式 iota）：
sdy.mesh @two_axes = <["a"=2, "b"=1]>
//   (a=0,b=0) -> dev0      (a=1,b=0) -> dev1

// 显式（顺序反转）：
sdy.mesh @single_axis_explicit_device_ids = <["a"=2], device_ids=[1, 0]>
//   a=0 -> dev1            a=1 -> dev0

// 校验：排序后的 device_ids 必须正好是 iota，
// 即不能重复、不能越界、不能缺号。`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'row', style: 'gap:30px;align-items:flex-start;justify-content:center;width:100%' });
    const panel = (title, sub) => {
      const c = U.el('div', { class: 'col', style: 'gap:10px;align-items:center' });
      c.innerHTML = `<div class="mid" style="font-weight:700">${title}</div>
        <div class="small faint">${sub}</div><div class="row" data-row style="gap:10px"></div>
        <div class="small mono dim" data-msg style="height:20px"></div>`;
      wrap.appendChild(c);
      return c;
    };
    const A = panel('① 隐式 iota', '<span class="mono">&lt;["a"=2]&gt;</span>');
    const B = panel('② 显式顺序', '<span class="mono">&lt;["a"=2], device_ids=[1, 0]&gt;</span>');
    root.appendChild(wrap);

    const fill = (host, order, colorBase) => {
      const row = host.querySelector('[data-row]');
      order.forEach((dev, i) => {
        const col = U.el('div', { class: 'col', style: 'gap:5px;align-items:center;opacity:0;transition:.5s' });
        col.innerHTML = `<div class="small mono faint">a=${i}</div>
          <div class="dev" style="width:64px;height:64px"><span class="dev-id">${dev}</span></div>
          <div class="small faint">↓ dev${dev}</div>`;
        col.querySelector('.dev').style.setProperty('--c', `var(--ax${colorBase + dev})`);
        row.appendChild(col);
        setTimeout(() => col.style.opacity = '1', i * 260);
      });
    };
    tl.at(800, () => { fill(A, [0, 1], 0); A.querySelector('[data-msg]').textContent = 'i = 设备号'; });
    tl.at(3200, () => { fill(B, [1, 0], 0); B.querySelector('[data-msg]').textContent = 'i ≠ 设备号，顺序被反转'; });
    tl.at(6800, () => {
      A.querySelector('[data-msg]').innerHTML = '<span class="hl-a">a=0 → dev0</span>';
      B.querySelector('[data-msg]').innerHTML = '<span class="hl-w">a=0 → dev1</span>';
    });
    tl.at(9200, () => {
      const n = U.el('div', { class: 'formula' });
      n.style.width = '100%';
      n.innerHTML = '校验规则：把 device_ids <b>排序</b>后必须正好是 <span class="mono">[0, 1, …, n-1]</span>';
      wrap.parentElement.appendChild(n);
      n.classList.add('anim-in');
    });
  }
},

/* --------------------------------------------- 5 打印规范化（真实用例） */
{
  kicker: 'L1-01 · 逻辑网格',
  title: '真实用例：<span class="hl-a">iota 会被省略</span>',
  sub: '显式写出 <span class="mono">device_ids=[0, 1, 2, 3]</span> 并不会被打印出来 —— 因为它正好等于默认顺序。',
  caption: '这条规则来自测试文件的 CHECK 行。读懂它，你才知道"为什么我写的 device_ids 不见了"。',
  code: `// 输入（测试文件里的原文）：
sdy.mesh @iota_explicit_device_ids = <["a"=2, "b"=2], device_ids=[0, 1, 2, 3]>

// 期望输出（同一个测试的 CHECK 行）：
sdy.mesh @iota_explicit_device_ids = <["a"=2, "b"=2]>

// 同理，轴大小全为 1 时：
//   输入 <["a"=1], device_ids=[0]>
//   输出 <["a"=1]>

// 依据：MeshAttr 约束明确指出 ——
// "we also disallow specifying a device ID list that is the
//  same as iota(product(axes)); in this case, a device ID list
//  shouldn't be specified."`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:16px;width:100%' });
    wrap.innerHTML = W.irPair(
      `sdy.mesh @iota_explicit_device_ids = <["a"=2, "b"=2], device_ids=[0, 1, 2, 3]>`,
      `sdy.mesh @iota_explicit_device_ids = <["a"=2, "b"=2]>`,
      { inTitle: '输入（测试原文）', outTitle: '打印结果（CHECK 行）' }
    ) + `<div class="formula" id="msg" style="min-height:48px;display:flex;align-items:center;text-align:center"></div>
      <div id="viz" class="row" style="gap:26px;justify-content:center;align-items:center"></div>`;
    root.appendChild(wrap);
    const msg = wrap.querySelector('#msg');
    const viz = wrap.querySelector('#viz');

    tl.at(700, () => { msg.innerHTML = '输入里确实写了 4 个设备编号：<span class="mono">[0, 1, 2, 3]</span>'; });
    tl.at(2600, () => {
      // 展示 iota 展开
      const row = U.el('div', { class: 'row', style: 'gap:10px' });
      [['a=0,b=0', 0], ['a=0,b=1', 1], ['a=1,b=0', 2], ['a=1,b=1', 3]].forEach(([pos, dev], i) => {
        const c = U.el('div', { class: 'col', style: 'gap:4px;align-items:center;opacity:0;transition:.45s' });
        c.innerHTML = `<div class="small mono faint">${pos}</div>
          <div class="dev" style="width:56px;height:56px"><span class="dev-id">${dev}</span></div>`;
        c.querySelector('.dev').style.setProperty('--c', `var(--ax${dev})`);
        row.appendChild(c);
        setTimeout(() => c.style.opacity = '1', i * 200);
      });
      viz.appendChild(row);
      msg.innerHTML = '把轴按行优先展开，得到的正是 <span class="mono">iota(2×2) = [0, 1, 2, 3]</span>';
    });
    tl.at(6200, () => { msg.innerHTML = '既然和默认完全一致，<span class="hl-a">再写一遍就是冗余</span> —— 打印时被省略'; });
    tl.at(8600, () => {
      msg.innerHTML = '所以：看到不带 <span class="mono">device_ids</span> 的网格，<b>不代表它没用显式编号</b>，而是"显式编号恰好等于默认"。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '想知道顺序是否被改过，只能<b>对比轴展开后的 iota 与 device_ids</b>。';
      viz.style.transition = 'opacity .5s'; viz.style.opacity = '.45';
    });
  }
},

/* ------------------------------------------------------- 6 八类校验错误 */
{
  kicker: 'L1-01 · 逻辑网格',
  title: '八种写错的方式（<span class="hl-a">校验器</span>怎么看）',
  sub: '测试文件用 <span class="mono">-verify-diagnostics</span> 逐条验证报错。理解每条错误的<b>根因</b>，比记住报文有用。',
  caption: '这些约束的共同目的：保证网格的 <span class="mono">∏轴大小</span> 与设备编号列表<b>一一对应</b>，不存在歧义。',
  code: `// ① 轴大小必须 >= 1
sdy.mesh @mesh = <["a"=2, "b"=0]>
//   axis size must be at least 1, got: 0

// ② 轴名不可重复
sdy.mesh @mesh = <["a"=2, "b"=2, "a"=4]>
//   duplicate axis name: "a"

// ③ 设备 ID 不可为负
sdy.mesh @mesh = <[], device_ids=[-1]>
//   device id must be non-negative, got: -1

// ④ 无轴时 device_ids 至多 1 个
sdy.mesh @mesh = <[], device_ids=[1, 0]>
//   axes is empty and device_ids has more than one element

// ⑤ 数量不匹配（多给）
sdy.mesh @mesh = <["a"=2, "b"=2], device_ids=[0, 2, 1, 3, 4, 5]>
//   total product of axis sizes must match total number
//   of device ids, got: 4 != 6

// ⑥ 数量不匹配（少给）
sdy.mesh @mesh = <["a"=2], device_ids=[0]>
//   total product of axis sizes must match total number
//   of device ids, got: 2 != 1

// ⑦ 设备 ID 重复
sdy.mesh @mesh_duplicated_device_ids = <["a"=2], device_ids=[1, 1]>
//   sorted device ids must be iota(product(axes)), got: 1, 1

// ⑧ 设备 ID 越界
sdy.mesh @mesh_out_if_bound_device_ids = <["a"=2], device_ids=[2, 1]>
//   sorted device ids must be iota(product(axes)), got: 1, 2`,
  duration: 20000,
  build(root, tl) {
    const CASES = [
      { ir: 'sdy.mesh @mesh = <["a"=2, "b"=0]>', err: 'axis size must be at least 1, got: 0', why: '轴大小为 0 没有意义：它既不能放设备，也无法参与分片。' },
      { ir: 'sdy.mesh @mesh = <["a"=2, "b"=2, "a"=4]>', err: 'duplicate axis name: "a"', why: '轴名是分片属性的唯一引用方式，重名会让 <span class="mono">{"a"}</span> 产生歧义。' },
      { ir: 'sdy.mesh @mesh = <[], device_ids=[-1]>', err: 'device id must be non-negative, got: -1', why: '设备 ID 是物理设备的索引，必须非负。' },
      { ir: 'sdy.mesh @mesh = <[], device_ids=[1, 0]>', err: 'axes is empty and device_ids has more than one element', why: '无轴网格要么是空占位符，要么是单设备 maximal 网格，不能有多个设备。' },
      { ir: 'sdy.mesh @mesh = <["a"=2, "b"=2], device_ids=[0, 2, 1, 3, 4, 5]>', err: 'total product of axis sizes must match total number of device ids, got: 4 != 6', why: '轴展开需要 4 个位置，却给了 6 个设备编号 —— 多出来的位置无处安放。' },
      { ir: 'sdy.mesh @mesh = <["a"=2], device_ids=[0]>', err: "custom op 'sdy.mesh' total product of axis sizes must match total number of device ids, got: 2 != 1", why: '轴展开需要 2 个位置，只给了 1 个 —— 有位置没有设备。' },
      { ir: 'sdy.mesh @mesh_duplicated_device_ids = <["a"=2], device_ids=[1, 1]>', err: 'sorted device ids must be iota(product(axes)), got: 1, 1', why: '同一台设备被放在两个逻辑位置；排序后不是 iota，说明有重复、有缺号。' },
      { ir: 'sdy.mesh @mesh_out_if_bound_device_ids = <["a"=2], device_ids=[2, 1]>', err: 'sorted device ids must be iota(product(axes)), got: 1, 2', why: '只有 2 台设备（编号应为 0,1），却引用了设备 2 —— 越界。' },
    ];

    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `
      <div class="row" style="gap:12px;align-items:center;justify-content:center" id="stepper"></div>
      <div class="row" style="gap:16px;align-items:stretch;width:100%">
        <div class="irbox in" style="flex:1.15">
          <div class="irh">错误 IR</div>
          <pre id="badir" style="min-height:96px;font-size:12.5px"></pre>
        </div>
        <div class="col" style="flex:1;gap:10px">
          <div class="card" style="border-color:rgba(251,113,133,.45)">
            <div class="card-t" style="color:var(--bad);font-size:12px">报错信息</div>
            <div class="card-d mono" id="err" style="font-size:11.5px;line-height:1.6;color:#ffc9d0"></div>
          </div>
          <div class="card"><div class="card-t" style="font-size:12px">为什么错</div>
            <div class="card-d" id="why" style="font-size:12.5px"></div></div>
        </div>
      </div>`;
    root.appendChild(wrap);

    const st = W.stepper(CASES.length);
    wrap.querySelector('#stepper').appendChild(st.el);
    const badir = wrap.querySelector('#badir'), err = wrap.querySelector('#err'), why = wrap.querySelector('#why');

    CASES.forEach((c, i) => tl.at(700 + i * 2200, () => {
      st.set(i);
      badir.innerHTML = U.hl(c.ir);
      err.textContent = c.err;
      why.innerHTML = c.why;
    }));
    tl.at(700 + CASES.length * 2200, () => {
      st.set(-1);
      badir.innerHTML = '<span class="com">// 8 条错误全部围绕同一件事：</span>\n<span class="com">// 轴展开的每个位置，必须有且只有一个设备。</span>';
      err.textContent = '∏ 轴大小 == 设备编号个数，且排序后为 iota';
      why.innerHTML = '记住这一句，八条错误都能自己推出来。';
    });
  }
},

/* ------------------------------------------------------------ 7 练习 */
{
  kicker: 'L1-01 · 练习',
  title: '练一练：<span class="hl-a">判断合法性与设备归属</span>',
  sub: '先自己想，再点"揭晓答案"。三道题分别考：合法性、打印结果、设备归属。',
  caption: '做完这三题，L1-01 的知识点（语法 / 编号 / 规范化 / 校验）就都覆盖了。',
  code: `// 题 1：是否合法？为什么？
sdy.mesh @m = <["a"=2, "b"=3], device_ids=[0, 2, 4, 1, 3]>

// 题 2：打印出来是什么？
sdy.mesh @m = <["a"=2], device_ids=[0, 1]>

// 题 3：a=1 上的数据在哪台设备？
sdy.mesh @m = <["a"=3], device_ids=[2, 0, 1]>`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    root.appendChild(wrap);

    const qs = [
      {
        q: '<span class="mono">sdy.mesh @m = &lt;["a"=2, "b"=3], device_ids=[0, 2, 4, 1, 3]&gt;</span> 是否合法？',
        a: '<b class="badge bad">非法</b> 轴展开需要 2×3 = <b>6</b> 个位置，只给了 <b>5</b> 个设备编号。' +
           '报错：<span class="mono">total product of axis sizes must match total number of device ids, got: 6 != 5</span>'
      },
      {
        q: '<span class="mono">sdy.mesh @m = &lt;["a"=2], device_ids=[0, 1]&gt;</span> 打印出来是什么？',
        a: '<span class="mono">sdy.mesh @m = &lt;["a"=2]&gt;</span><br>' +
           '<span class="dim">因为 [0, 1] 正好是 iota(2)，按约束必须省略。</span>'
      },
      {
        q: '<span class="mono">sdy.mesh @m = &lt;["a"=3], device_ids=[2, 0, 1]&gt;</span> 中，<span class="mono">a=1</span> 上的数据在哪台设备？',
        a: '<b class="badge ok">dev0</b> 逻辑位置按轴序排列：a=0→dev2，<b>a=1→dev0</b>，a=2→dev1。<br>' +
           '<span class="dim">注意 device_ids 是"逻辑位置 → 设备号"的映射，不是设备号列表的排序。</span>'
      },
    ];
    qs.forEach((item, i) => {
      const e = W.exercise(item.q, item.a);
      e.style.opacity = '0'; e.style.transition = 'opacity .4s';
      wrap.appendChild(e);
      tl.at(500 + i * 600, () => e.style.opacity = '1');
    });
  }
},

];
