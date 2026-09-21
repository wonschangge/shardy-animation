/* ==========================================================================
   Shardy 入门动画 — 场景定义
   --------------------------------------------------------------------------
   每一幕包含：标题 / 副标题 / 底部说明 / 侧栏 IR / 可视区的分步动画。
   所有 IR 片段均取自 shardy 仓库的文档与测试用例（openxla/shardy）。
   ========================================================================== */
'use strict';

/* ------------------------------------------------------------ 局部小工具 */
const D = {
  /* 一张卡片 */
  card(title, desc) {
    return `<div class="card" style="min-width:0">
      <div class="card-t">${title}</div>
      <div class="card-d">${desc}</div></div>`;
  },
  /* 设备调色：设备 id -> 颜色索引 */
  devColor: id => id % 6,
  /* 生成 4 设备（2x2）的网格，cell 返回内部 HTML */
  mesh22(inner) {
    const g = U.el('div', { class: 'mesh' });
    g.style.gridTemplateColumns = 'repeat(2, 92px)';
    g.style.gridTemplateRows = 'repeat(2, 92px)';
    for (let id = 0; id < 4; id++) {
      const xi = Math.floor(id / 2), yi = id % 2;
      const d = U.el('div', { class: 'dev', 'data-dev': id });
      d.style.setProperty('--c', `var(--ax${D.devColor(id)})`);
      d.innerHTML = `<span class="dev-id">${id}</span>`;
      if (inner) d.insertAdjacentHTML('beforeend', inner(id, xi, yi));
      g.appendChild(d);
    }
    return g;
  },
  /* 2x2 分片下，张量格子 (r,c) 属于哪个设备（8x8 张量，每块 4x4） */
  owner22: (r, c) => Math.floor(r / 4) * 2 + Math.floor(c / 4),
  /* 一行小标签 */
  tag(txt, cls = '') { return `<span class="chip mut ${cls}">${txt}</span>`; },
};

/* ========================================================================= */
const SCENES = [

/* ------------------------------------------------------------------ 1 封面 */
{
  kicker: 'Shardy 入门动画',
  title: 'Shardy：让张量在设备上<span class="hl-a">各就各位</span>',
  sub: '一个基于 MLIR 的、与方言无关的<b>张量切分（partitioning）系统</b>。本动画从最简单的 4 个设备讲起，直到完整的传播算法与流水线。',
  caption: '按 <b>空格</b> 暂停/播放，<b>← →</b> 切换场景，圆点可直接跳转。',
  code: `// 你会学到的核心一句话：
//   用"逻辑网格轴"描述张量怎么切，
//   剩下的交给传播(propagation)自动推导。

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

%0 = ... : tensor<8x8xf32>
      {sdy.sharding = #sdy.sharding<@mesh_xy,
                        [{"x"}, {"y"}]>}`,
  duration: 9000,
  build(root, tl) {
    const inner = U.el('div', { class: 'scene-inner' });
    inner.innerHTML = `<div class="card" style="padding:18px 22px">
        <div class="card-t" style="font-size:19px">1 个张量</div>
        <div class="card-d">tensor&lt;8x8xf32&gt;</div></div>
      <div class="arrow anim" style="font-size:34px">⟹</div>
      <div id="hero-mesh"></div>`;
    root.appendChild(inner);
    const host = inner.querySelector('#hero-mesh');
    const mesh = D.mesh22();
    host.appendChild(mesh);
    const devs = mesh.querySelectorAll('.dev');
    // 依次点亮设备，模拟张量被分发到各设备
    devs.forEach((d, i) => tl.at(400 + i * 320, () => {
      d.classList.add('devpop', 'lit');
      setTimeout(() => d.classList.remove('devpop'), 520);
    }));
    tl.at(400 + 4 * 320 + 400, () => devs.forEach(d => d.classList.remove('lit')));
    [0, 1, 2, 3].forEach(i => tl.at(2400 + i * 320, () => devs[i].classList.add('lit')));
  }
},

/* -------------------------------------------------------------- 2 为什么切分 */
{
  kicker: '第 1 步 · 动机',
  title: '为什么需要<span class="hl-a">切分</span>？',
  sub: '一块张量放不进一个设备，或者放得进但太浪费。把张量<b>切开分散</b>到多个设备，才能把模型和数据做大。',
  caption: '复制：每个设备都存一整份 → 存储 ×4，计算也没变快。切分：每个设备只存 1/4 → 各算各的。',
  code: `// 复制 (replicated) —— 4 个设备各存一整份
//   device0..3 都持有 tensor<8x8xf32>

// 切分 (sharded) —— 每设备只存 tensor<4x4xf32>
%0 : tensor<8x8xf32>
    {sdy.sharding = #sdy.sharding<@mesh_xy,
                      [{"x"}, {"y"}]>}`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col' });
    wrap.style.gap = '16px';
    wrap.innerHTML = `
      <div class="row" style="gap:34px;align-items:flex-start">
        <div class="col" style="gap:8px">
          <div class="mid faint">源张量</div>
          <div id="src"></div>
        </div>
        <div class="arrow" style="font-size:30px">⟹</div>
        <div class="col" style="gap:8px">
          <div class="mid" id="mode-label">模式：<b class="hl-w">全复制</b></div>
          <div id="devs"></div>
        </div>
        <div class="col" style="gap:10px;margin-left:8px" id="notes"></div>
      </div>
      <div class="formula" id="sum">总存储 = 4 × 64 = 256 个元素</div>`;
    root.appendChild(wrap);

    // 源张量
    const src = U.tensorGrid(8, 8, () => -1, { cell: 17, gap: 2 });
    wrap.querySelector('#src').appendChild(src);

    // 设备区：4 个设备，每个里放一个小张量
    const devHost = wrap.querySelector('#devs');
    const devs = U.el('div', { class: 'mesh' });
    devs.style.gridTemplateColumns = 'repeat(2, 108px)';
    devs.style.gridTemplateRows = 'repeat(2, 108px)';
    const tiles = [];
    for (let id = 0; id < 4; id++) {
      const d = U.el('div', { class: 'dev' });
      d.style.setProperty('--c', `var(--ax${D.devColor(id)})`);
      d.innerHTML = `<span class="dev-id">${id}</span>`;
      const t = U.el('div', { style: 'display:flex;align-items:center;justify-content:center' });
      d.appendChild(t);
      devs.appendChild(d);
      tiles.push(t);
    }
    devHost.appendChild(devs);

    const notes = wrap.querySelector('#notes');
    const sum = wrap.querySelector('#sum');
    const modeLabel = wrap.querySelector('#mode-label');

    const drawReplicated = () => tiles.forEach(t => {
      t.innerHTML = '';
      t.appendChild(U.tensorGrid(6, 6, () => 6, { cell: 9, gap: 1 }));
    });
    const drawSharded = () => tiles.forEach((t, id) => {
      t.innerHTML = '';
      t.appendChild(U.tensorGrid(6, 6, () => D.devColor(id), { cell: 9, gap: 1 }));
    });

    tl.at(300, () => {
      drawReplicated();
      notes.innerHTML = D.card('存储浪费', '同一份数据被存了 4 遍。<br>显存利用率 25%。') +
        D.card('无法扩展', '模型再大就放不下了。');
      [...devs.querySelectorAll('.dev')].forEach((d, i) => tl.at(0, () => { }));
    });
    tl.at(4600, () => {
      modeLabel.innerHTML = '模式：<b class="hl-a">切分</b>';
      drawSharded();
      notes.innerHTML = D.card('存储省 4 倍', '每设备只存 1/4 的数据。') +
        D.card('可扩展', '再加设备就能装更大的模型。');
      sum.textContent = '总存储 = 64 个元素（每设备 16）';
      sum.className = 'formula';
      [...devs.querySelectorAll('.dev')].forEach((d, i) =>
        setTimeout(() => { d.classList.add('devpop'); setTimeout(() => d.classList.remove('devpop'), 520); }, i * 130));
    });
  }
},

/* ------------------------------------------------------------- 3 逻辑网格 */
{
  kicker: '第 2 步 · 设备网格',
  title: '逻辑网格 <span class="mono hl-a">sdy.mesh</span>：给设备起名字',
  sub: '把一维的设备列表<b>重新排列成多维数组</b>，每个维度起一个名字（轴）。之后所有分片描述都只引用这些轴名。',
  caption: '网格只是"设备的视图"：同一批设备可以有多种网格视图，设备编号（左上角数字）决定数据实际落在哪块硬件上。',
  code: `// 4 个设备排成 2x2，两个轴分别叫 "x" 和 "y"
sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// 设备编号按行优先(=iota)：
//   x=0,y=0 -> dev0     x=0,y=1 -> dev1
//   x=1,y=0 -> dev2     x=1,y=1 -> dev3

// 也可以显式指定设备顺序（自定义视图）
sdy.mesh @m = <["x"=2, "y"=2], device_ids=[0,2,1,3]>`,
  duration: 12000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'scene-inner' });
    wrap.style.gap = '40px';
    wrap.innerHTML = `
      <div class="col" style="gap:10px;align-items:center">
        <div class="mid faint">① 一维设备列表</div>
        <div class="row" id="flat" style="gap:8px"></div>
      </div>
      <div class="arrow anim" style="font-size:30px" id="a1">⟹</div>
      <div class="col" style="gap:10px;align-items:center">
        <div class="mid faint">② 排成 2x2 并命名轴</div>
        <div id="grid"></div>
        <div id="axis-note" class="small dim" style="height:22px"></div>
      </div>`;
    root.appendChild(wrap);

    // 一维列表
    const flat = wrap.querySelector('#flat');
    for (let i = 0; i < 4; i++) {
      const d = U.el('div', { class: 'dev', style: 'width:52px;height:52px' });
      d.style.setProperty('--c', `var(--ax${D.devColor(i)})`);
      d.innerHTML = `<span class="dev-id">${i}</span>`;
      flat.appendChild(d);
    }

    // 2x2 网格（初始透明，等待飞入）
    const gridHost = wrap.querySelector('#grid');
    const g = D.mesh22();
    g.style.opacity = '.12';
    g.style.transition = 'opacity .6s';
    gridHost.appendChild(g);
    const axisNote = wrap.querySelector('#axis-note');

    tl.at(1200, () => { flat.style.transition = 'opacity .5s'; flat.style.opacity = '.3'; });
    tl.at(1400, () => {
      g.style.opacity = '1';
      g.querySelectorAll('.dev').forEach((d, i) => setTimeout(() => {
        d.classList.add('devpop'); setTimeout(() => d.classList.remove('devpop'), 520);
      }, i * 150));
    });
    tl.at(2600, () => {
      axisNote.innerHTML = '<span class="chip c0"><i class="sw sw-c0"></i>"x" = 2</span> &nbsp;决定<b>行</b>';
    });
    tl.at(3600, () => {
      axisNote.innerHTML = '<span class="chip c1"><i class="sw sw-c1"></i>"y" = 2</span> &nbsp;决定<b>列</b>';
    });
    tl.at(4800, () => {
      axisNote.innerHTML = '<span class="dim">轴名随便取：<span class="mono">"data" / "model" / "tp"</span> 都可以</span>';
    });
    tl.at(6200, () => {
      axisNote.innerHTML = '<span class="hl-w">每个轴的大小就是该方向上的设备数</span>';
      const devs = g.querySelectorAll('.dev');
      devs.forEach((d, i) => setTimeout(() => { d.classList.add('lit'); setTimeout(() => d.classList.remove('lit'), 900); }, i * 140));
    });
  }
},

/* ------------------------------------------------------- 4 分片表示（核心） */
{
  kicker: '第 3 步 · 分片表示 ★ 核心',
  title: '轴基分片：<span class="mono hl-a">#sdy.sharding</span>',
  sub: '对张量的<b>每一个维度</b>，列出沿哪些网格轴切分（从 major 到 minor）。没被任何一个维度用到的轴，就等于<b>复制</b>。',
  caption: '这是 Shardy 最核心的概念：不是记录"哪块数据在哪台机器"，而是记录"每个维度沿哪个轴切了几刀"。',
  code: `sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

%0 : tensor<8x8xf32>
     {sdy.sharding = #sdy.sharding<@mesh_xy,
                        [{"x"},   // 第 0 维沿 "x" 切 2 份
                         {"y"}]>} // 第 1 维沿 "y" 切 2 份

// 每台设备拿到的"局部形状"：
//   dim0: 8 / size("x") = 8/2 = 4
//   dim1: 8 / size("y") = 8/2 = 4
//   => tensor<4x4xf32>`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'scene-inner' });
    wrap.style.gap = '30px';
    wrap.innerHTML = `
      <div class="col" style="gap:10px;align-items:center">
        <div class="mid faint" id="t-label">全局张量 tensor&lt;8x8xf32&gt;</div>
        <div id="tensor" class="cut-wrap"></div>
      </div>
      <div class="col" style="gap:10px;align-items:center">
        <div class="mid" id="step-label" style="height:22px;color:var(--accent)"></div>
        <div id="legend" class="small dim" style="height:20px"></div>
      </div>
      <div class="col" style="gap:10px;align-items:center">
        <div class="mid faint">各设备的局部数据</div>
        <div id="devs"></div>
      </div>`;
    root.appendChild(wrap);

    const tHost = wrap.querySelector('#tensor');
    // 初始：无色网格
    let tg = U.tensorGrid(8, 8, () => -1, { cell: 17, gap: 2 });
    tHost.appendChild(tg);

    // 设备区
    const devHost = wrap.querySelector('#devs');
    const mesh = D.mesh22();
    const held = [];
    mesh.querySelectorAll('.dev').forEach((d, id) => {
      const box = U.el('div', { style: 'opacity:0;transition:opacity .6s;display:flex;align-items:center;justify-content:center' });
      d.appendChild(box);
      held.push(box);
    });
    devHost.appendChild(mesh);

    const stepLabel = wrap.querySelector('#step-label');
    const legend = wrap.querySelector('#legend');

    // 步骤 1：沿 x 切第 0 维
    tl.at(900, () => {
      stepLabel.innerHTML = '① 第 0 维沿 <b>"x"</b> 切 → 横向切一刀';
      tHost.querySelectorAll('.cut').forEach(c => c.remove());
      const cs = U.cuts(tHost, [{ dir: 'h', pos: 8 + 4 * (17 + 2) }]);
      setTimeout(() => cs.forEach(c => c.classList.add('on')), 40);
    });
    // 步骤 2：沿 y 切第 1 维
    tl.at(3000, () => {
      stepLabel.innerHTML = '② 第 1 维沿 <b>"y"</b> 切 → 纵向切一刀';
      const cs = U.cuts(tHost, [{ dir: 'v', pos: 8 + 4 * (17 + 2) }]);
      setTimeout(() => cs.forEach(c => c.classList.add('on')), 40);
    });
    // 步骤 3：上色，按设备区分
    tl.at(5000, () => {
      stepLabel.innerHTML = '③ 4 个象限 = 4 个设备的局部数据';
      legend.innerHTML = '颜色 = 设备编号';
      const fresh = U.tensorGrid(8, 8, D.owner22, {
        cell: 17, gap: 2,
        text: (r, c) => (r % 4 === 1 && c % 4 === 1) ? 'D' + D.owner22(r, c) : ''
      });
      fresh.style.opacity = '0';
      fresh.style.transition = 'opacity .7s';
      tHost.replaceChild(fresh, tHost.querySelector('.tgrid'));
      tHost.querySelectorAll('.cut').forEach(c => c.remove());
      requestAnimationFrame(() => fresh.style.opacity = '1');
      tg = fresh;
    });
    // 步骤 4：落到设备上
    tl.at(7200, () => {
      stepLabel.innerHTML = '④ 每台设备只保存自己的那块：<b>tensor&lt;4x4xf32&gt;</b>';
      held.forEach((box, id) => {
        setTimeout(() => {
          box.appendChild(U.tensorGrid(8, 8, () => D.devColor(id), {
            cell: 5, gap: 1, text: (r, c) => (r === 2 && c === 2) ? 'D' + id : ''
          }));
          box.style.opacity = '1';
        }, id * 220);
      });
    });
    // 步骤 5：总结
    tl.at(10500, () => {
      stepLabel.innerHTML = '';
      legend.innerHTML = '<span class="hl-a">一个 sharding 属性 = 一张切分方案</span>';
    });
    tl.at(11000, () => {
      legend.innerHTML = '<span class="hl-w">轴不出现 ⇒ 该维在该轴上被复制</span>';
    });
  }
},

/* ----------------------------------------------------------- 5 局部形状 */
{
  kicker: '第 3 步 · 补充',
  title: '算一算：<span class="hl-a">局部形状</span>',
  sub: '设备上真正要分配的内存，由分片方案唯一确定。规则只有一条：<b>该维大小 ÷ 该维用到的所有轴大小之积</b>。',
  caption: '注意最后一行：8 无法被 3 整除 —— Shardy 允许这种"不可整除"的分片，导出时会自动做 padding。',
  code: `// 局部维度 = 全局维度 / ∏(该维分片轴的 size)

@m1 = <["x"=2, "y"=2]>
  tensor<8x8xf32>  [{"x"}, {"y"}]       -> tensor<4x4xf32>

@m2 = <["x"=2, "y"=4, "z"=2]>
  tensor<4x8xf32>  [{"x"}, {"z","y"}]   -> tensor<2x1xf32>

@m3 = <["x"=8, "y"=2, "z"=3]>
  tensor<7x3x8xf32> [{"x"}, {"y"}, {"z"}]
  // 7 / 8 不整除 -> 需要 padding（Shardy 允许）`,
  duration: 12000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col' });
    wrap.style.gap = '16px';
    wrap.style.width = '100%';
    wrap.innerHTML = `<div class="formula">局部维度<sub>i</sub> &nbsp;=&nbsp; 全局维度<sub>i</sub> &nbsp;÷&nbsp; <span class="hl-w">∏ size(该维分片轴)</span></div>
      <table class="ftable" style="font-size:14px" id="tb">
        <tr><th>网格</th><th>全局张量</th><th>分片</th><th>局部张量</th></tr>
        <tr id="r1"><td class="mono">x=2,y=2</td><td class="mono">tensor&lt;8x8xf32&gt;</td><td class="mono">[{"x"},{"y"}]</td><td class="mono hl-a">tensor&lt;4x4xf32&gt;</td></tr>
        <tr id="r2"><td class="mono">x=2,y=4,z=2</td><td class="mono">tensor&lt;4x8xf32&gt;</td><td class="mono">[{"x"},{"z","y"}]</td><td class="mono hl-a">tensor&lt;2x1xf32&gt;</td></tr>
        <tr id="r3"><td class="mono">x=8,y=2,z=3</td><td class="mono">tensor&lt;7x3x8xf32&gt;</td><td class="mono">[{"x"},{"y"},{"z"}]</td><td class="mono hl-w">7/8 不整除 → padding</td></tr>
      </table>
      <div class="row" id="why" style="gap:12px"></div>`;
    root.appendChild(wrap);

    const rows = ['r1', 'r2', 'r3'].map(id => wrap.querySelector('#' + id));
    rows.forEach(r => { r.style.opacity = '.18'; r.style.transition = 'opacity .5s'; });
    rows.forEach((r, i) => tl.at(600 + i * 2400, () => {
      rows.forEach(x => x.style.opacity = '.18');
      r.style.opacity = '1';
      r.style.background = 'rgba(94,234,212,.07)';
      setTimeout(() => r.style.background = '', 2000);
    }));
    tl.at(3000, () => {
      wrap.querySelector('#why').innerHTML =
        D.card('<span class="mono">"z","y"</span> 的顺序', 'major→minor。先按 z 切，再按 y 切。') +
        D.card('为什么可以乘积', '多个轴可以叠加切同一个维度，总份数就是它们 size 的乘积。');
    });
    tl.at(8000, () => {
      wrap.querySelector('#why').innerHTML =
        D.card('不整除怎么办', '允许"切不干净"的维度存在，导出时用 <span class="mono">-sdy-pad-for-divisibility</span> 补齐。');
    });
  }
},

/* --------------------------------------------------------------- 6 复制 */
{
  kicker: '第 3 步 · 补充',
  title: '没被用到的轴 = <span class="hl-a">隐式复制</span>',
  sub: '如果某个网格轴没有出现在任何维度的分片里，数据在该轴上就是<b>复制的</b>：多台设备持有完全相同的副本。',
  caption: '复制不是浪费 —— 它常常是必要的：比如卷积的权重、或者每个设备都要用到的公共输入。',
  code: `sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// 只切第 0 维，"y" 完全没有被使用
%0 : tensor<8x8xf32>
     {sdy.sharding = #sdy.sharding<@mesh_xy,
                        [{"x"},  // 第 0 维沿 x 切
                         {}]>}   // 第 1 维不切
// "y" 上的 2 台设备 → 持有相同的 tensor<4x8xf32>
// 局部形状 = tensor<4x8xf32>`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'scene-inner' });
    wrap.style.gap = '34px';
    wrap.innerHTML = `
      <div class="col" style="gap:10px;align-items:center">
        <div class="mid faint">全局张量（颜色 = 设备）</div>
        <div id="tensor"></div>
      </div>
      <div class="col" style="gap:10px;align-items:center">
        <div class="mid" id="hint" style="height:22px;color:var(--accent)"></div>
        <div id="devs"></div>
      </div>`;
    root.appendChild(wrap);

    // 只在 dim0 上色：行决定设备(0 或 1)
    const tHost = wrap.querySelector('#tensor');
    const tg = U.tensorGrid(8, 8, (r, c) => (Math.floor(r / 4) === 0 ? 0 : 1), { cell: 17, gap: 2 });
    tHost.appendChild(tg);

    const mesh = D.mesh22();
    const tiles = [];
    const devEls = [...mesh.querySelectorAll('.dev')];
    devEls.forEach((d, id) => {
      const box = U.el('div', { style: 'display:flex;align-items:center;justify-content:center' });
      d.appendChild(box); tiles.push(box);
    });
    wrap.querySelector('#devs').appendChild(mesh);

    const hint = wrap.querySelector('#hint');
    tl.at(900, () => { hint.innerHTML = '第 0 维沿 "x" 切成上下两半'; });
    tl.at(2000, () => {
      // 设备 0/1 拿上半，2/3 拿下半 → 同一行的两台设备内容相同
      tiles.forEach((t, id) => {
        const colorIdx = id < 2 ? 0 : 1;
        t.appendChild(U.tensorGrid(6, 12, () => colorIdx, { cell: 5, gap: 1 }));
      });
    });
    tl.at(4200, () => { hint.innerHTML = '"y" 轴没被用 → 沿着 y 复制'; });
    tl.at(5000, () => {
      // 高亮同一行（同一 x）的两台设备
      devEls[0].classList.add('lit'); devEls[1].classList.add('lit');
    });
    tl.at(6200, () => {
      hint.innerHTML = '设备 0 与 1 的内容<b>完全相同</b>（设备 2 与 3 也相同）';
    });
    tl.at(8400, () => {
      hint.innerHTML = '每设备 tensor&lt;4x8xf32&gt;，沿 "y" 有 2 份副本';
    });
  }
},

/* ---------------------------------------------------- 7 开维 / 闭维 (?) */
{
  kicker: '第 3 步 · 精细控制',
  title: '开维与闭维：<span class="mono hl-a">?</span> 的含义',
  sub: '维度后面加 <span class="mono">?</span> 表示它是<b>开(open)</b>的 —— 传播过程中还可以继续往上加轴。不加 <span class="mono">?</span> 则是<b>闭(closed)</b>的，锁定，谁都别想改。',
  caption: '闭维常用于"用户指定的输入输出分片不能变"（如 jax.jit 的 in_shardings）；开维则让编译器有优化空间。',
  code: `// 闭维：{"x"} —— 锁定，传播不能再加轴
[{"x"}, {"b"}]

// 开维：{"x", ?} —— 已按 x 切，但还允许再加轴
[{"x", ?}, {"b"}]

// 完全不写 sharding 属性   ==   全开
// （编译器可以自由决定怎么切）`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'scene-inner' });
    wrap.style.gap = '26px';
    wrap.style.alignItems = 'stretch';
    wrap.innerHTML = `
      <div class="card" style="flex:1;display:flex;flex-direction:column;gap:12px" id="closed">
        <div class="card-t mono">[{"x"}, {"b"}] &nbsp;<span class="chip c0">闭</span></div>
        <div class="card-d">锁定：第 0 维只能沿 "x" 切</div>
        <div class="row" style="gap:8px" id="c-axes"></div>
        <div class="formula warn" id="c-res" style="min-height:52px;display:flex;align-items:center">等待传播尝试…</div>
      </div>
      <div class="card" style="flex:1;display:flex;flex-direction:column;gap:12px" id="open">
        <div class="card-t mono">[{"x", ?}, {"b"}] &nbsp;<span class="chip c0">开</span></div>
        <div class="card-d">开放：已按 "x" 切，还能继续加</div>
        <div class="row" style="gap:8px" id="o-axes"></div>
        <div class="formula warn" id="o-res" style="min-height:52px;display:flex;align-items:center">等待传播尝试…</div>
      </div>`;
    root.appendChild(wrap);

    const cc = wrap.querySelector('#c-axes'), oc = wrap.querySelector('#o-axes');
    const cr = wrap.querySelector('#c-res'), or = wrap.querySelector('#o-res');
    const draw = (host, arr, open) => {
      host.innerHTML = arr.map((a, i) =>
        `<span class="chip c${i}">${a}</span>`).join('<span class="faint">·</span>') +
        (open ? '<span class="chip c1">?</span>' : '<span class="chip mut">🔒</span>');
    };
    draw(cc, ['"x"'], false);
    draw(oc, ['"x"'], true);

    tl.at(900, () => { cr.textContent = '传播想再加一个轴 "y" …'; or.textContent = '传播想再加一个轴 "y" …'; });
    tl.at(2600, () => {
      cc.insertAdjacentHTML('beforeend', '<span class="chip c1" style="opacity:.5">"y" ?</span>');
      oc.insertAdjacentHTML('beforeend', '<span class="chip c1">"y"</span>');
    });
    tl.at(4200, () => {
      cr.innerHTML = '<span class="hl-w">✕ 被拒绝</span>：闭维不可更改';
      or.innerHTML = '<span class="hl-a">✓ 被接受</span>：现在第 0 维沿 "x" 再沿 "y"';
      draw(oc, ['"x"', '"y"'], true);
    });
    tl.at(7000, () => {
      cr.innerHTML = '<span class="hl-w">✕ 被拒绝</span>：闭维不可更改';
      or.innerHTML = '<span class="hl-a">✓ 被接受</span>：新形状 [{"x","y", ?}, {"b"}]';
    });
    tl.at(9200, () => {
      const n = U.el('div', { class: 'formula', style: 'width:100%;text-align:center' });
      n.innerHTML = '不写 sharding 属性 &nbsp;≡&nbsp; 全开 &nbsp;⇒&nbsp; 编译器完全自由';
      wrap.parentElement.appendChild(n);
      n.classList.add('anim-in');
    });
  }
},

/* ------------------------------------------------------ 8 显式复制轴 */
{
  kicker: '第 3 步 · 精细控制',
  title: '显式复制：<span class="mono hl-a">replicated={"y"}</span>',
  sub: '隐式复制是"暂时没用这个轴"。显式复制是<b>主动锁死</b>：这个轴不许用来切任何维度，永远保持复制。',
  caption: '区别很关键：隐式复制的轴，传播还可以拿它去切某个开维；显式复制则彻底禁止。',
  code: `sdy.mesh @mesh_xyz = <["x"=2, "y"=4, "z"=2]>

// "y" 显式复制 -> 不能用来切第 1 维(即使它是开维)
// "z" 隐式复制 -> 可以被传播拿去切第 1 维
sharding<@mesh_xyz, [{"x"}, {?}], replicated={"y"}>
   : tensor<4x8xf32>            // 局部: tensor<2x8xf32>

// 对比：如果只写 [{"x"}, {?}]
//   传播可以用 "y" 或 "z" 去切第 1 维`,
  duration: 12000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'scene-inner' });
    wrap.style.gap = '30px';
    wrap.innerHTML = `
      <div class="col" style="gap:14px;align-items:center">
        <div class="mid faint">网格 <span class="mono">x=2, y=4, z=2</span> 的轴</div>
        <div class="row" style="gap:10px" id="axes"></div>
        <div class="formula" id="sh" style="min-height:44px;display:flex;align-items:center">
          <span class="mono">[{"x"}, {?}]</span>&nbsp; replicated=<span class="mono" id="rep">{}</span>
        </div>
      </div>
      <div class="col" style="gap:12px;width:330px" id="try"></div>`;
    root.appendChild(wrap);

    const axes = wrap.querySelector('#axes');
    axes.innerHTML = `<span class="chip c0">"x"=2</span><span class="chip c1">"y"=4</span><span class="chip c2">"z"=2</span>`;
    const tryBox = wrap.querySelector('#try');
    const rep = wrap.querySelector('#rep');

    tl.at(900, () => {
      rep.innerHTML = '<span class="hl-w">{"y"}</span>';
      tryBox.innerHTML = D.card('已锁定', '<span class="mono">"y"</span> 被显式复制，不参与任何分片。');
    });
    tl.at(2400, () => {
      tryBox.innerHTML = D.card('传播尝试用 "y" 切第 1 维', '第 1 维是开维 <span class="mono">{?}</span> …');
    });
    tl.at(4200, () => {
      tryBox.innerHTML = D.card('<span class="hl-w">✕ 拒绝</span>', '<span class="mono">"y"</span> 在 replicated 列表中，禁止使用。');
    });
    tl.at(6200, () => {
      tryBox.innerHTML = D.card('改用 "z" 试试', '<span class="mono">"z"</span> 只是隐式复制（未被使用）…');
    });
    tl.at(7800, () => {
      tryBox.innerHTML = D.card('<span class="hl-a">✓ 接受</span>', '第 1 维现在沿 <span class="mono">"z"</span> 切 → <span class="mono">[{"x"},{"z",?}]</span>');
      wrap.querySelector('#sh').innerHTML =
        '<span class="mono">[{"x"}, {"z", ?}]</span>&nbsp; replicated=<span class="mono hl-w">{"y"}</span>';
    });
    tl.at(9600, () => {
      tryBox.innerHTML += D.card('局部形状', '<span class="mono">8 / size("z") = 8/2 = 4</span> → <span class="mono">tensor&lt;2x4xf32&gt;</span>');
    });
  }
},

/* --------------------------------------------------------- 9 子轴切分 */
{
  kicker: '第 3 步 · 进阶',
  title: '轴的拆分与<span class="hl-a">子轴</span>：reshape 零通信',
  sub: '一个轴可以<i>在编译器内部</i>被拆成多个子轴，记为 <span class="mono">"x":(pre)size</span>。这让 reshape 这类形状变换不必通信。',
  caption: '这是 Shardy 相对 GSPMD 的重要改进之一：reshape 常常带来额外通信，子轴拆分可以把数据留在原地。',
  code: `sdy.mesh @mesh_x = <["x"=4]>

%arg0 : tensor<8xf32> {sdy.sharding = <@mesh_x, [{"x"}]>}
// 局部形状 tensor<2xf32>

// reshape 成 2x4：需要把 4 个设备看成 2x2
%0 = stablehlo.reshape %arg0
     {sdy.sharding_per_value = [<@mesh_x,
        [{"x":(1)2}, {"x":(2)2}]>]}   // 拆成两个子轴
     : (tensor<8xf32>) -> tensor<2x4xf32>

// 第 0 维 -> x:(1)2（size 2），第 1 维 -> x:(2)2
// 最内层还剩 4/2 = 2 -> 第 1 维 4 = 2 x 2 ✓ 无需通信`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col' });
    wrap.style.gap = '20px';
    wrap.style.alignItems = 'center';
    wrap.innerHTML = `
      <div class="row" style="gap:18px;align-items:center">
        <div class="col" style="gap:8px;align-items:center">
          <div class="mid faint">一维张量 tensor&lt;8xf32&gt;</div>
          <div id="t1"></div>
        </div>
        <div class="arrow anim" style="font-size:28px" id="ar">⟹</div>
        <div class="col" style="gap:8px;align-items:center">
          <div class="mid faint">reshape 后 tensor&lt;2x4xf32&gt;</div>
          <div id="t2"></div>
        </div>
      </div>
      <div class="col" style="gap:8px;align-items:center;width:100%">
        <div class="mid faint" id="lbl">设备轴 "x" = 4</div>
        <div id="axisrow" class="row" style="gap:6px"></div>
      </div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center">
        开始…
      </div>`;
    root.appendChild(wrap);

    // 8 个元素，按 x=4 分片 -> 每个设备 2 个元素
    const t1 = U.tensorGrid(1, 8, (r, c) => Math.floor(c / 2), { cell: 26, gap: 3, text: (r, c) => (c % 2 === 0 ? 'D' + Math.floor(c / 2) : '') });
    wrap.querySelector('#t1').appendChild(t1);
    const t2 = U.tensorGrid(2, 4, () => -1, { cell: 26, gap: 3 });
    wrap.querySelector('#t2').appendChild(t2);

    const ax = wrap.querySelector('#axisrow');
    const cells = [];
    for (let i = 0; i < 4; i++) {
      const c = U.el('div', { class: 'dev', style: 'width:44px;height:44px' });
      c.style.setProperty('--c', `var(--ax${i})`);
      c.innerHTML = `<span class="dev-id">${i}</span>`;
      ax.appendChild(c); cells.push(c);
    }
    const msg = wrap.querySelector('#msg');

    tl.at(700, () => { msg.innerHTML = '一个轴 "x" 覆盖 4 台设备，张量第 0 维正好切 4 份'; });
    tl.at(2800, () => {
      msg.innerHTML = '<span class="hl-w">问题</span>：reshape 成 2x4 后，第 0 维只有 2，<br>而 "x" 大小是 4 —— 直接用会切不动 / 需要通信';
    });
    tl.at(5200, () => {
      msg.innerHTML = '把轴 "x"=4 <b>拆成两个子轴</b>：<span class="mono hl-a">"x":(1)2</span> 与 <span class="mono hl-a">"x":(2)2</span>';
      // 轴可视化：4 个设备分成左右两组
      cells[0].classList.add('lit'); cells[1].classList.add('lit');
      cells[0].style.setProperty('--c', 'var(--ax0)');
      cells[1].style.setProperty('--c', 'var(--ax0)');
      cells[2].style.setProperty('--c', 'var(--ax1)');
      cells[3].style.setProperty('--c', 'var(--ax1)');
      cells[2].classList.add('lit'); cells[3].classList.add('lit');
      ax.insertAdjacentHTML('beforebegin', '<div class="row" style="gap:6px;font-size:11px" id="subax"></div>');
    });
    tl.at(7600, () => {
      msg.innerHTML = '第 0 维（大小 2）→ 子轴 <span class="mono">"x":(1)2</span>；第 1 维（大小 4 = 2×2）→ <span class="mono">"x":(2)2</span> 再叠一层';
      // 画 t2：行 0/1 由子轴决定
      const fresh = U.tensorGrid(2, 4, (r, c) => r, { cell: 26, gap: 3, text: (r, c) => (c === 1 ? 'D' + (r * 2 + Math.floor(c / 2)) : '') });
      const host = wrap.querySelector('#t2');
      host.innerHTML = ''; host.appendChild(fresh);
    });
    tl.at(10500, () => {
      msg.innerHTML = '<span class="hl-a">✓ 数据留在原地，0 次通信</span> —— 这就是轴拆分要解决的问题';
    });
    tl.at(12800, () => {
      msg.innerHTML = '子轴只在编译器内部使用；用户写的分片必须引用<b>完整的轴</b>。';
    });
  }
},

/* ---------------------------------------------------------- 10 优先级 */
{
  kicker: '第 4 步 · 控制传播',
  title: '优先级：<span class="mono hl-a">p0 → p1 → p2</span> 分批传播',
  sub: '给维度分片标上优先级，传播就会<b>按批次从小到大</b>进行。同一批内先全部传播完，再轮到下一批。',
  caption: '优先级不是"谁更重要"，而是"传播的先后顺序"。这样你能精确控制"先批并行、再张量并行、最后 ZeRO"。',
  code: `sdy.mesh @mesh_wxyz = <["w"=6, "x"=2, "y"=4, "z"=2]>

//           第0维     第1维       第2维
sharding<@mesh_wxyz, [{"x"}p1, {"y"}, {"z", ?}p2]>
//                          ↑ p0(默认)   ↑ p2

// 传播顺序：
//   第 0 轮：所有 p0 的分片先传播到全程序
//   第 1 轮：再加入 p1
//   第 2 轮：最后加入 p2
//
// 低优先级的开维不会被高优先级覆盖，
// 但可以等所有高优先级传播完后再被填充。`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col' });
    wrap.style.gap = '18px';
    wrap.style.alignItems = 'center';
    wrap.innerHTML = `
      <div class="row" style="gap:12px" id="lvls"></div>
      <div class="row" style="gap:28px;align-items:center">
        <div class="col" style="gap:6px;align-items:center">
          <div class="small faint">程序中的张量</div>
          <div class="row" style="gap:6px" id="tensors"></div>
        </div>
        <div class="col" style="gap:6px;align-items:center">
          <div class="small faint">当前轮次</div>
          <div class="big hl-a mono" id="round">—</div>
        </div>
      </div>
      <div class="formula" id="msg" style="min-height:48px;display:flex;align-items:center;text-align:center"></div>`;
    root.appendChild(wrap);

    const lv = wrap.querySelector('#lvls');
    const defs = [
      { p: 'p0', c: 0, t: '批并行 / 数据并行', d: '最高优先级，最先铺满全图' },
      { p: 'p1', c: 1, t: '张量并行 (Megatron)', d: '第二批' },
      { p: 'p2', c: 2, t: 'ZeRO / 参数切分', d: '最后一批' },
    ];
    const lvlEls = defs.map(d => {
      const e = U.el('div', { class: 'card', style: 'width:210px;opacity:.35;transition:.4s' });
      e.innerHTML = `<div class="card-t"><span class="chip c${d.c}">${d.p}</span> ${d.t}</div><div class="card-d">${d.d}</div>`;
      lv.appendChild(e); return e;
    });

    // 8 个"张量"方块
    const tHost = wrap.querySelector('#tensors');
    const blocks = [];
    for (let i = 0; i < 6; i++) {
      const b = U.el('div', { class: 'tcell', style: 'width:34px;height:34px;border-radius:6px;background:rgba(255,255,255,.1)' });
      tHost.appendChild(b); blocks.push(b);
    }
    const round = wrap.querySelector('#round');
    const msg = wrap.querySelector('#msg');

    const paint = (upto) => {
      blocks.forEach((b, i) => {
        if (i <= upto) {
          b.style.background = `var(--ax${upto})`;
          b.style.transform = 'scale(1.06)';
          setTimeout(() => b.style.transform = '', 300);
        }
      });
    };

    tl.at(700, () => { msg.innerHTML = '程序里只有少数几个张量带分片标注，其余都是待推导的。'; });
    tl.at(2400, () => {
      round.textContent = 'p0'; lvlEls[0].style.opacity = '1'; lvlEls[0].classList.add('card');
      msg.innerHTML = '第 0 轮：只传播 <b>p0</b> 的分片（<span class="mono">{"y"}</span>）';
      paint(0);
    });
    tl.at(5600, () => {
      round.textContent = 'p1'; lvlEls[1].style.opacity = '1';
      round.className = 'big mono';
      round.style.color = 'var(--ax1)';
      msg.innerHTML = '第 1 轮：<b>p0 的分片保持不变</b>，再加入 <b>p1</b>（<span class="mono">{"x"}</span>）继续铺开';
      blocks[0].style.background = 'var(--ax0)'; blocks[1].style.background = 'var(--ax0)';
      paint(1);
    });
    tl.at(8800, () => {
      round.textContent = 'p2'; lvlEls[2].style.opacity = '1';
      round.style.color = 'var(--ax2)';
      msg.innerHTML = '第 2 轮：最后加入 <b>p2</b>（<span class="mono">{"z", ?}</span>），填充还开着的维度';
      paint(2);
    });
    tl.at(11800, () => {
      msg.innerHTML = '<span class="hl-a">结果可预测</span>：你能确切知道"只做批并行时会是什么样"。';
      lvlEls.forEach((e, i) => e.style.opacity = '1');
    });
  }
},

/* --------------------------------------------------- 11 算子分片规则 */
{
  kicker: '第 5 步 · 传播的基石',
  title: '算子分片规则：把算子抽象成<span class="hl-a">因子映射</span>',
  sub: '每个算子都用一条 <span class="mono">sharding_rule</span> 描述"它的哪些维度其实是同一个东西"。传播算法只看这张表，完全不需要理解算子本身。',
  caption: '这是 Shardy 设计的精髓：把"算子语义"和"传播算法"解耦。新增一个算子，只要给它一条规则，传播就能自动工作。',
  code: `// stablehlo.dot_general 的分片规则（einsum 风格）
#sdy.op_sharding_rule<
    ([i, k], [k, j]) -> ([i, j])   // 各张量维度 -> 因子
    {i = 8, j = 16, k = 8}>        // 每个因子的大小

// 因子的四种类型（决定通信方式）：
//   pass-through      普通：所有张量上同因子同样分片
//   reduction         归约：如 matmul 的 k -> 结果需要 all-reduce
//   need_replication  必须全复制：如 sort 的被排序维
//   permutation       需要 collective-permute：如 pad 的补齐维
//   blocked_propagation 禁止沿该因子传播`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col' });
    wrap.style.gap = '16px';
    wrap.style.alignItems = 'center';
    wrap.style.width = '100%';
    wrap.innerHTML = `
      <div class="row" style="gap:16px;align-items:center" id="ops"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:940px"></div>`;
    root.appendChild(wrap);

    const mk = (title, dims) => {
      const box = U.el('div', { class: 'card', style: 'text-align:center;min-width:150px' });
      box.innerHTML = `<div class="small faint">${title}</div>
        <div class="row" style="gap:8px;justify-content:center;margin-top:8px" data-dims></div>`;
      const host = box.querySelector('[data-dims]');
      dims.forEach(d => {
        const e = U.el('div', { class: 'chip mut', style: 'font-size:15px;padding:6px 12px' });
        e.textContent = d; e.dataset.f = d; host.appendChild(e);
      });
      return box;
    };
    const ops = wrap.querySelector('#ops');
    const lhs = mk('lhs (操作数)', ['i', 'k']);
    const rhs = mk('rhs (操作数)', ['k', 'j']);
    const res = mk('result (结果)', ['i', 'j']);
    ops.appendChild(lhs);
    ops.appendChild(U.el('div', { class: 'arrow', html: '<span class="mono mid">dot</span> ⟹' }));
    ops.appendChild(rhs);
    ops.appendChild(U.el('div', { class: 'arrow', html: '⟹' }));
    ops.appendChild(res);

    const msg = wrap.querySelector('#msg');
    const dimOf = (box, f) => box.querySelector(`[data-f="${f}"]`);

    tl.at(700, () => {
      msg.innerHTML = '规则说：lhs 的 <span class="mono">i</span> 与 result 的 <span class="mono">i</span> 是同一个因子；<span class="mono">k</span> 只出现在两个操作数里。';
    });
    tl.at(3200, () => {
      msg.innerHTML = '① 把批因子 <span class="mono">i</span> 沿轴 <b>"x"</b> 分片 → lhs 第 0 维、result 第 0 维一起被切';
      dimOf(lhs, 'i').outerHTML = '<div class="chip c0" style="font-size:15px;padding:6px 12px" data-f="i">i<span class="faint">·x</span></div>';
      dimOf(res, 'i').outerHTML = '<div class="chip c0" style="font-size:15px;padding:6px 12px" data-f="i">i<span class="faint">·x</span></div>';
    });
    tl.at(6400, () => {
      msg.innerHTML = 'rhs 里<b>没有</b>因子 <span class="mono">i</span> → 它必须沿 "x" <span class="hl-w">复制</span>（每台设备都要有完整的 rhs）';
      const e = U.el('div', { class: 'chip c0 pulse', style: 'font-size:13px;padding:5px 10px' });
      e.textContent = 'replicated·x';
      rhs.appendChild(e);
    });
    tl.at(9400, () => {
      msg.innerHTML = '② 把收缩因子 <span class="mono">k</span> 沿轴 <b>"y"</b> 分片（lhs 和 rhs 都要切）';
      dimOf(lhs, 'k').outerHTML = '<div class="chip c1" style="font-size:15px;padding:6px 12px" data-f="k">k<span class="faint">·y</span></div>';
      dimOf(rhs, 'k').outerHTML = '<div class="chip c1" style="font-size:15px;padding:6px 12px" data-f="k">k<span class="faint">·y</span></div>';
    });
    tl.at(12600, () => {
      msg.innerHTML = '<span class="hl-w">k 是 reduction 因子</span>：result 里没有 k，所以每台设备只算出了<b>部分和</b>';
      const e = U.el('div', { class: 'chip c1 pulse', style: 'font-size:13px;padding:5px 10px' });
      e.textContent = 'unreduced·y';
      res.appendChild(e);
    });
    tl.at(15400, () => {
      msg.innerHTML = '<span class="mono">sdy.all_reduce {"y"}</span> 把部分和加起来 → 结果才真正可用。<b>通信就是这么来的。</b>';
    });
  }
},

/* ----------------------------------------------------- 12 传播算法 */
{
  kicker: '第 5 步 · 核心算法',
  title: '分片传播：沿 <span class="hl-a">factor</span> 而非维度流动',
  sub: '传播就是沿着数据流反复推：把已知的分片信息，顺着算子的因子映射，推到还不知道分片的张量上，直到不再变化（不动点）。',
  caption: '为什么沿 factor 而不是维度？因为一个维度可能对应多个因子（reshape），也可能只对应因子的一部分。投影到因子空间处理、再投影回维度，逻辑才自洽。',
  code: `// 传播三步走：
//   1. DimSharding   -> FactorSharding   （投影到因子空间）
//   2. 在 factor 空间里收集 / 扩展轴
//   3. FactorSharding -> DimSharding    （投影回维度）

// 每一步取"最长兼容的主分片轴"：
//   F0 上传播 ["a","b"]   F1 上传播 ["c"]   F2 上不传播

// 反复迭代直到不动点：
//   前向 (operand -> result) 与 反向 (result -> operand) 交替`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col' });
    wrap.style.gap = '14px'; wrap.style.alignItems = 'center'; wrap.style.width = '100%';
    wrap.innerHTML = `
      <div class="row" style="gap:10px" id="chain"></div>
      <table class="ftable" id="tbl">
        <tr><th></th><th>F0</th><th>F1</th><th>F2</th><th>显式复制轴</th></tr>
        <tr id="T0"><td>T0</td><td class="ax" data-c="F0"></td><td data-c="F1"></td><td data-c="F2"></td><td></td></tr>
        <tr id="T1"><td>T1</td><td class="ax" data-c="F0"></td><td class="ax" data-c="F1"></td><td data-c="F2"></td><td></td></tr>
        <tr id="T2"><td>T2</td><td data-c="F0"></td><td class="ax" data-c="F1"></td><td data-c="F2"></td><td></td></tr>
      </table>
      <div class="formula" id="msg" style="min-height:46px;display:flex;align-items:center;text-align:center;max-width:900px"></div>`;
    root.appendChild(wrap);

    const chain = wrap.querySelector('#chain');
    ['%arg0', 'add', 'dot', 'return'].forEach((n, i) => {
      const b = U.el('div', { class: 'card', style: 'padding:8px 14px;font-family:var(--mono);font-size:13px;transition:border-color .4s' });
      b.textContent = n; b.id = 'node' + i;
      chain.appendChild(b);
      if (i < 3) chain.appendChild(U.el('div', { class: 'arrow', text: '→' }));
    });

    const setCell = (row, f, txt, cls) => {
      const td = wrap.querySelector(`#${row} [data-c="${f}"]`);
      td.textContent = txt;
      if (cls !== undefined) td.className = cls;
    };
    const msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      setCell('T0', 'F0', '"a"', 'ax'); setCell('T0', 'F2', '"f"', '');
      msg.innerHTML = '初始状态：只有一部分张量带了分片信息（这里按 factor 排列）';
    });
    tl.at(2800, () => {
      setCell('T1', 'F0', '"a","b"', 'ax'); setCell('T1', 'F1', '"c","d"', 'ax'); setCell('T1', 'F2', '"g"', '');
      setCell('T2', 'F1', '"c","e"', 'ax');
      msg.innerHTML = '观察每一列（每个 factor）：哪些轴是<b>所有</b>张量都兼容的？';
      wrap.querySelector('#node0').style.borderColor = 'var(--accent)';
    });
    tl.at(5600, () => {
      msg.innerHTML = '<b>第 1 步</b>：F0 上取最长兼容轴 <span class="mono hl-a">["a","b"]</span>，F1 上取 <span class="mono hl-a">["c"]</span>，F2 上没有可传播的';
    });
    tl.at(8600, () => {
      setCell('T0', 'F0', '"a","b"', 'ax chg');
      setCell('T0', 'F1', '"c"', 'chg');
      setCell('T2', 'F0', '"a","b"', 'ax chg');
      wrap.querySelector('#node1').style.borderColor = 'var(--accent)';
      msg.innerHTML = '<b>第 2 步</b>：把这些轴<b>扩展</b>到该 factor 上的所有张量（绿色格子是新填上的）';
    });
    tl.at(12200, () => {
      msg.innerHTML = 'T2 的 F0 原本是空的，现在也被填上了 —— 分片信息沿着数据流<b>扩散</b>开了';
      wrap.querySelector('#node2').style.borderColor = 'var(--accent)';
    });
    tl.at(15000, () => {
      msg.innerHTML = '重复这个过程（前向 + 反向）直到不再变化 = <span class="hl-a">不动点</span>，传播结束';
      wrap.querySelector('#node3').style.borderColor = 'var(--accent)';
    });
  }
},

/* ------------------------------------------------- 12b 数据流算子 */
{
  kicker: '第 5 步 · 规则之外',
  title: '数据流算子：不靠因子规则的<span class="hl-a">另一条路</span>',
  sub: '像 <span class="mono">while</span>、<span class="mono">case</span> 这种带区域的算子，没法用"维度映射"描述。Shardy 改用<b>数据流边</b>：一条边上的所有 source 和 target 必须分片一致。',
  caption: '这正是 Shardy "与方言无关"的关键接口之一：任何算子只要实现 ShardableDataFlowOpInterface，传播就能穿过它 —— 不需要 Shardy 认识这个算子。',
  code: `// while 有 n 条数据流边，第 i 条连接：
//   sources: x_i(初始值), return_value_i(循环体返回值)
//   targets: y_i(结果), pred_arg_i(判定块参数),
//            body_arg_i(循环体块参数)
//
// 一条边内的所有值必须分片完全相同。

%y = stablehlo.while(%x = %init) : tensor<8xf32> {
  ^bb0(%pred_arg: tensor<8xf32>):
    ...
    stablehlo.return %next : tensor<8xf32>
}

// 实现该接口的算子都能这样传播，例如：
//   stablehlo.while / stablehlo.case
//   stablehlo.optimization_barrier
//   sdy.manual_computation / sdy.named_computation
// 传播时视作"源与目标是同一个张量"（恒等规则）。`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:16px;align-items:center;width:100%' });
    wrap.innerHTML = `
      <div class="row" style="gap:20px;align-items:center;justify-content:center">
        <div class="col" style="gap:8px;align-items:flex-end" id="src"></div>
        <div class="card" style="padding:14px 18px;min-width:150px;text-align:center">
          <div class="mono" style="font-size:15px;color:var(--accent2)">stablehlo.while</div>
          <div class="small faint" style="margin-top:6px">带区域的算子</div>
        </div>
        <div class="col" style="gap:8px;align-items:flex-start" id="tgt"></div>
      </div>
      <div class="formula" id="msg" style="min-height:48px;display:flex;align-items:center;text-align:center;max-width:900px"></div>`;
    root.appendChild(wrap);

    const mk = (txt) => {
      const e = U.el('div', { class: 'chip mut', style: 'font-size:12.5px;padding:6px 11px;transition:.35s' });
      e.textContent = txt; return e;
    };
    const src = wrap.querySelector('#src'), tgt = wrap.querySelector('#tgt');
    const S = [mk('x_0 : tensor<8xf32>'), mk('x_1 : tensor<4xf32>')];
    const T = [mk('y_0 : tensor<8xf32>'), mk('y_1 : tensor<4xf32>')];
    S.forEach(e => src.appendChild(e));
    T.forEach(e => tgt.appendChild(e));

    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { msg.innerHTML = '两个输入、两个输出 —— 但它们不是"操作数→结果"的维度映射关系。'; });
    tl.at(2800, () => {
      msg.innerHTML = '第 0 条数据流边把 <span class="mono">x_0</span>、<span class="mono">return_value_0</span>、<span class="mono">body_arg_0</span>、<span class="mono">y_0</span> 绑在一起';
      S[0].className = 'chip c0'; S[0].style.fontSize = '12.5px';
    });
    tl.at(5200, () => {
      msg.innerHTML = '给 <span class="mono">x_0</span> 标上分片 → <span class="hl-a">同一条边上的其它值立刻同步</span>（不需要因子规则）';
      T[0].className = 'chip c0'; T[0].style.fontSize = '12.5px';
      T[0].classList.add('pulse');
    });
    tl.at(8000, () => {
      msg.innerHTML = '第 1 条边独立处理：<span class="mono">x_1</span> 与 <span class="mono">y_1</span> 是另一组，互不影响';
      S[1].className = 'chip c1'; S[1].style.fontSize = '12.5px';
      T[1].className = 'chip c1'; T[1].style.fontSize = '12.5px';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<span class="mono">sdy.manual_computation</span> 也是数据流算子 —— 所以传播能<b>穿过它</b>，在内部的"自由轴"上继续工作。';
    });
    tl.at(13600, () => {
      msg.innerHTML = '这就是方言无关的接口：<span class="mono">ShardableDataFlowOpInterface</span>。';
    });
  }
},

/* -------------------------------------------------- 13 冲突消解层级 */
{
  kicker: '第 5 步 · 当分片打架时',
  title: '冲突消解：<span class="hl-a">四层</span>优先级嵌套',
  sub: '两个张量对同一维度的分片要求不一致怎么办？Shardy 用一个四层金字塔来裁决，越往上越"强势"。',
  caption: '可以把它读成嵌套的 for 循环：对每个用户优先级，完整跑一遍算子优先级传播；算子优先级的每一轮里，又完整跑一遍激进传播。',
  code: `// 从下往上，一层比一层"强势"：

// 4. 用户优先级 User Priority
//      用户标的 p0/p1/p2，决定传播轮次
// 3. 算子优先级 Op Priority
//      按算子类型分批决定传播方向
//      （如逐元素/reshape 优先于 dot）
// 2. 激进传播 Aggressive
//      主动消解冲突，可能多出通信但省显存
// 1. 基础传播 Basic
//      只传播"所有人都同意"的轴，绝不制造冲突

// 对应命令行：
//   -sdy-basic-propagate
//   -sdy-aggressive-propagate
//   -sdy-op-priority-propagate
//   -sdy-user-priority-propagate
//   -sdy-propagation-pipeline      ← 完整流程`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'row', style: 'gap:26px;align-items:center;justify-content:center;width:100%' });
    wrap.innerHTML = `<div class="pyr" id="pyr" style="flex:0 0 auto"></div>
      <div class="col" style="gap:10px;width:340px;min-width:0;flex:0 0 auto;overflow-wrap:anywhere" id="detail"></div>`;
    root.appendChild(wrap);

    const defs = [
      { t: '① 基础传播 Basic', d: '只传播所有操作数/结果都兼容的轴；不消解任何冲突', c: '#4ade80', w: 230,
        x: '最保守：永远不引入额外通信，但也最"推不动"。' },
      { t: '② 激进传播 Aggressive', d: '主动消解冲突，把一个轴强加给不兼容的张量（可能引入通信）', c: '#38bdf8', w: 272,
        x: '用通信换显存：越激进 = 越省显存，但可能多出通信。' },
      { t: '③ 算子优先级 Op Priority', d: '按算子类型分批：逐元素 / reshape 等"直通"算子优先，dot / reduce 次之', c: '#c084fc', w: 314,
        x: '每一轮都完整跑一遍激进传播；方向取最表达者：BOTH &gt; FWD/BWD &gt; NONE。' },
      { t: '④ 用户优先级 User Priority', d: '按用户标注的 p0/p1/p2 分轮传播，低优先级不会被高优先级覆盖', c: '#fbbf24', w: 356,
        x: '最外层。让你能精确控制"先批并行，再张量并行，最后 ZeRO"。' },
    ];
    const pyr = wrap.querySelector('#pyr');
    const detail = wrap.querySelector('#detail');
    const els = [];
    [...defs].reverse().forEach(d => {
      const e = U.el('div', { class: 'pyr-lvl' });
      e.style.width = d.w + 'px';
      e.innerHTML = `<div class="pt" style="color:${d.c}">${d.t}</div><div class="pd">${d.d}</div>`;
      pyr.appendChild(e); els.unshift(e);
    });

    defs.forEach((d, i) => {
      tl.at(800 + i * 3200, () => {
        els.forEach((e, k) => e.classList.toggle('on', k === i));
        detail.innerHTML = `<div class="card" style="border-color:${d.c}55">
            <div class="card-t" style="color:${d.c}">${d.t}</div>
            <div class="card-d" style="font-size:14px;line-height:1.65">${d.x}</div></div>`;
      });
    });
    tl.at(14200, () => {
      els.forEach(e => e.classList.add('on'));
      detail.innerHTML = `<div class="formula" style="font-size:13.5px;line-height:1.9">
        嵌套执行：<br>对每个<b>用户优先级</b> p<br>&nbsp;&nbsp;完整跑一遍<b>算子优先级</b>传播<br>
        &nbsp;&nbsp;&nbsp;&nbsp;每一轮都完整跑一遍<b>激进</b>传播<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其底层是<b>基础</b>传播</div>`;
    });
  }
},

/* ------------------------------------------------------ 14 编译器 API */
{
  kicker: '第 6 步 · 你能怎么干预',
  title: '编译器 API：<span class="hl-a">五种</span>影响传播的手段',
  sub: '你不需要把每个张量都标注清楚。只要在关键位置"钉"几个约束，剩下的让传播自己算。',
  caption: '注意生命周期：sharding_constraint 在传播之前由用户插入，传播会消费掉它；如果最终需要换分片，传播会插入 sdy.reshard，再由分区器换成真正的集合通信。',
  code: `// ① 输入/输出分片：直接标在函数参数 / 返回值上
func.func @main(%arg0: tensor<8x8xf32>
    {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {}]>})
    -> (tensor<8x16xf32>
    {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {"y"}]>})

// ② 分片约束：钉住中间张量（可开可闭，可悬空）
%1 = sdy.sharding_constraint %0 <@mesh_xy, [{"x"}, {?}]>
     : tensor<8x8xf32>

// ③ 分片组：让没有数据依赖的张量强行同分片
%0 = sdy.sharding_group %arg0 group_id=0 : tensor<8x2xi64>

// ④ 手动计算：这块我自己来（局部形状 + 显式通信）
%1 = sdy.manual_computation(%0)
     in_shardings=[<@mesh, [{"data"}, {"model", ?}]>]
     out_shardings=[<@mesh, [{"data"}, {?}]>]
     manual_axes={"data"}
     (%a: tensor<8x32xf32>) { ... }

// ⑤ 传播屏障：只允许分片单向流过
%2 = sdy.propagation_barrier %1 allowed_direction=FORWARD
     : tensor<8x8xf32>`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch" id="cards"></div>
      <div class="formula" id="msg" style="min-height:46px;display:flex;align-items:center;text-align:center"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 输入/输出分片', d: '钉住主函数的参数与返回值。用户指定的 in_shardings 通常是<b>闭维</b>，传播不能改。', c: 0 },
      { t: '② sharding_constraint', d: '钉住<b>中间</b>张量。悬空（无使用者）表示"这个张量本身就该这样切"；有使用者表示"用它的地方这样切"。', c: 1 },
      { t: '③ sharding_group', d: '把没有依赖关系的张量<b>绑成一组</b>：一旦其中一个被分片，其余成员立刻跟随。', c: 2 },
      { t: '④ manual_computation', d: '指定 <b>manual_axes</b>，区域内改用局部形状并手写通信。其余"自由轴"仍由传播处理，还可嵌套。', c: 3 },
      { t: '⑤ propagation_barrier', d: '像恒等算子，但只允许分片沿指定方向流过：FORWARD / BACKWARD / NONE。', c: 4 },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => {
      const e = U.el('div', { class: 'card', style: 'flex:1;opacity:.35;transition:.4s;min-width:0;padding:10px' });
      e.innerHTML = `<div class="card-t" style="font-size:12.5px;color:var(--ax${d.c})">${d.t}</div>
        <div class="card-d" style="font-size:11.5px;line-height:1.6">${d.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    const notes = [
      '最常见的用法：告诉编译器"输入按数据并行切，输出按模型并行切"。',
      '关键区别：<b>悬空</b>的约束直接约束张量本身；<b>有使用者</b>的约束只约束这些使用者，别的使用者可以有不同的分片。',
      '典型场景：输入和输出之间<b>没有数据依赖</b>，传播推不过去，用 sharding_group 把它们绑起来。',
      'manual_axes 里的轴：区域内是局部形状，通信由你手写。不在 manual_axes 里的"自由轴"：仍由传播自动处理。',
      '用于阻止不希望发生的传播，例如切断"常量被多个使用者拉成同一种分片"的假依赖。',
    ];
    defs.forEach((d, i) => {
      tl.at(700 + i * 3300, () => {
        els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.35'; e.style.borderColor = k === i ? `var(--ax${defs[k].c})` : ''; });
        msg.innerHTML = notes[i];
      });
    });
    tl.at(17400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '这些 API 都是<b>可选的</b>：一个都不写，传播也能靠算子的分片规则跑完全程。';
    });
  }
},

/* ------------------------------------------------------- 15 三条流水线 */
{
  kicker: '第 7 步 · 端到端',
  title: '三条流水线：<span class="hl-a">导入 → 传播 → 导出</span>',
  sub: 'Shardy 不是一个"一键"编译器，而是三段可自由组合的 pass 流水线。你可以只跑一段来调试，也可以全跑完拿到设备本地代码。',
  caption: '实际使用中：前端先把程序降到 StableHLO 并带上分片标注，然后依次跑这三段。设计上刻意拆开，就是为了让传播过程可观察、可定位。',
  code: `// ① 导入：清理 + 准备（-sdy-import-pipeline）
//     -sdy-constant-or-scalar-splitter   拆常量，避免假依赖
//     -sdy-import-func-calls             call -> named_computation
//     -sdy-lift-inlined-meshes           内联 mesh 提升为 sdy.mesh
//     -sdy-apply-sharding-constraints    落实约束
//     -sdy-sharding-group-import         分片组规范化
//     -sdy-remove-size-one-axes          去掉 size=1 的轴

// ② 传播：推导所有张量的分片（-sdy-propagation-pipeline）
//     内部 = 用户优先级 -> 算子优先级 -> 激进 -> 基础

// ③ 导出：变成真正能跑的 SPMD 代码（-sdy-export-pipeline）
//     -sdy-insert-explicit-reshards      补上必要的 reshard
//     -sdy-reshard-to-collectives        reshard -> 集合通信
//     -sdy-convert-global-to-local       全局形状 -> 局部形状
//     -sdy-close-shardings               关闭分片
//     -sdy-drop-sharding-and-mesh        擦除 mesh`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:16px;width:100%' });
    wrap.innerHTML = `<div class="pipe" id="pipe"></div>
      <div class="row" style="gap:14px;align-items:stretch" id="detail"></div>
      <div class="formula" id="msg" style="min-height:44px;display:flex;align-items:center;text-align:center"></div>`;
    root.appendChild(wrap);

    const segs = [
      { n: '① IMPORT', t: '导入', l: '· 拆分常量<br>· call→named_computation<br>· 提升内联 mesh<br>· 落实约束 / 分组<br>· 去掉 size=1 轴', d: '把前端产出的程序收拾干净：<b>消除假依赖</b>（共用同一个常量的两个使用者不该被强行切成一样）、把内联的 mesh 提成符号、把分片约束真正落实到张量上。' },
      { n: '② PROPAGATE', t: '传播', l: '· 用户优先级<br>· 算子优先级<br>· 激进传播<br>· 基础传播<br>· 直到不动点', d: '核心一步：从少数几个已知分片出发，推导出<b>每一个张量</b>的分片方案。结果里可能出现 sdy.reshard，表示"这里需要换分片"。' },
      { n: '③ EXPORT', t: '导出', l: '· 插入显式 reshard<br>· reshard→集合通信<br>· 全局→局部形状<br>· 关闭 / 擦除分片<br>· 去重常量', d: '把"逻辑分片方案"变成"物理上可执行的 SPMD 程序"：补通信、把 <span class="mono">tensor&lt;8x8&gt;</span> 换成每设备真实的 <span class="mono">tensor&lt;4x4&gt;</span>。' },
    ];
    const pipe = wrap.querySelector('#pipe');
    const els = segs.map(s => {
      const e = U.el('div', { class: 'pipe-seg' });
      e.innerHTML = `<div class="pn">${s.n}</div><div class="pt">${s.t}</div><div class="pl">${s.l}</div>`;
      pipe.appendChild(e); return e;
    });
    const detail = wrap.querySelector('#detail'), msg = wrap.querySelector('#msg');
    segs.forEach((s, i) => {
      tl.at(700 + i * 4200, () => {
        els.forEach((e, k) => e.classList.toggle('on', k === i));
        detail.innerHTML = `<div class="card" style="flex:1"><div class="card-t">${s.t}</div><div class="card-d" style="font-size:13.5px;line-height:1.7">${s.d}</div></div>`;
      });
    });
    tl.at(13400, () => {
      els.forEach(e => e.classList.add('on'));
      msg.innerHTML = '三段可以独立运行 —— 调试传播时通常只跑 ① 和 ②，再用 <span class="mono">-sdy-basic-propagate=debug-sharding-origins=true</span> 看每个分片是哪来的。';
    });
    tl.at(16200, () => {
      msg.innerHTML = '命令行：<span class="mono hl-a">sdy_opt 输入.mlir -sdy-propagation-pipeline</span> 就是完整的 ②。';
    });
  }
},

/* ------------------------------------------------------ 16 集合通信 */
{
  kicker: '第 7 步 · 通信从哪来',
  title: '六种集合通信：<span class="hl-a">分片变化的代价</span>',
  sub: '每当相邻算子需要的数据布局不一致，就必须移动数据。Shardy 把这些移动统一表示成 6 种集合通信算子。',
  caption: '记忆诀窍：all-gather 是"聚"，all-slice 是"分"，all-reduce 是"合"，reduce-scatter 是"先合再分"，all-to-all 是"互换"，collective-permute 是"重排"。',
  code: `// ① 分片 -> 复制（把别人的分片收过来）
%1 = sdy.all_gather [{"x"}] %0 out_sharding=<@m, [{}]>
     : tensor<16xf32>

// ② 归约（把各设备的部分和加起来）
%1 = sdy.all_reduce {"x"} %0 out_sharding=<@m, [{}]>
     : tensor<16xf32>
%1 = sdy.all_reduce max {"x"} %0 ...   // 也可 max / min

// ③ 复制 -> 分片（从整份里取自己那片）
%1 = sdy.all_slice [{"x"}] %0 out_sharding=<@m, [{"y"}]>

// ④ 先归约再分片
%1 = sdy.reduce_scatter {"x"} %0 out_sharding=<@m, [{"y"}]>

// ⑤ 设备间互换切片
%1 = sdy.all_to_all [{"b"}: 0->2] %0 out_sharding=...

// ⑥ 重排 / 替换轴（保持每维分片大小不变）
%1 = sdy.collective_permute %0 out_sharding=<@m, [...]>`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col' });
    wrap.style.gap = '14px'; wrap.style.width = '100%'; wrap.style.alignItems = 'center';
    wrap.innerHTML = `<div id="coll-stage" class="coll"></div>
      <div class="row" style="gap:10px;justify-content:center;flex-wrap:wrap;max-width:860px" id="names"></div>
      <div class="formula" id="msg" style="min-height:44px;display:flex;align-items:center;text-align:center;max-width:900px"></div>`;
    root.appendChild(wrap);

    const stage = wrap.querySelector('#coll-stage');
    const names = wrap.querySelector('#names');
    const msg = wrap.querySelector('#msg');

    const C = [
      { n: '① all_gather', ir: '{"x"} 分片 → 复制',
        before: [['A0'], ['A1'], ['A2'], ['A3']],
        after: [['A0', 'A1', 'A2', 'A3'], ['A0', 'A1', 'A2', 'A3'], ['A0', 'A1', 'A2', 'A3'], ['A0', 'A1', 'A2', 'A3']],
        d: '<b>all-gather</b>：每台设备都拿到全部切片。分片 → 复制，通信量最大。' },
      { n: '② all_reduce', ir: '{"x"} 部分和 → 全和',
        before: [['p0'], ['p1'], ['p2'], ['p3']],
        after: [['Σ'], ['Σ'], ['Σ'], ['Σ']],
        d: '<b>all-reduce</b>：把各设备的部分和加起来，结果复制到所有设备。matmul 切了收缩维之后必用。' },
      { n: '③ all_slice', ir: '复制 → {"y"} 分片',
        before: [['A0', 'A1', 'A2', 'A3'], ['A0', 'A1', 'A2', 'A3'], ['A0', 'A1', 'A2', 'A3'], ['A0', 'A1', 'A2', 'A3']],
        after: [['A0'], ['A1'], ['A2'], ['A3']],
        d: '<b>all-slice</b>：从完整数据里各取一片。复制 → 分片，与 all-gather 互为逆操作。' },
      { n: '④ reduce_scatter', ir: '归约 + 分片',
        before: [['p0'], ['p1'], ['p2'], ['p3']],
        after: [['Σ0'], ['Σ1'], ['Σ2'], ['Σ3']],
        d: '<b>reduce-scatter</b>：先归约再切片 = all-reduce + all-slice。常用于"结果还要继续切"的场合，省一次通信。' },
      { n: '⑤ all_to_all', ir: '切片互换',
        before: [['A00'], ['A10'], ['A20'], ['A30']],
        after: [['A01'], ['A11'], ['A21'], ['A31']],
        d: '<b>all-to-all</b>：设备之间互换不同的切片，把"切分"从一个维度搬到另一个维度。' },
      { n: '⑥ collective_permute', ir: '重排 / 替换轴',
        before: [['X'], ['Y'], ['Z'], ['W']],
        after: [['Y'], ['X'], ['W'], ['Z']],
        d: '<b>collective-permute</b>：点对点重排，每维分片大小保持不变，用于换轴或与复制轴互换。' },
    ];

    const devRow = U.el('div', { class: 'row', style: 'gap:22px' });
    stage.appendChild(devRow);
    const devBoxes = [];
    for (let i = 0; i < 4; i++) {
      const d = U.el('div', { class: 'col', style: 'gap:6px;align-items:center' });
      const lbl = U.el('div', { class: 'small faint mono' });
      lbl.textContent = 'dev' + i;
      const box = U.el('div', { class: 'dev', style: 'width:112px;height:74px;flex-wrap:wrap;gap:4px;padding:6px;align-content:center' });
      box.style.setProperty('--c', `var(--ax${i})`);
      d.appendChild(lbl); d.appendChild(box);
      devRow.appendChild(d);
      devBoxes.push(box);
    }

    const paint = (rows, delay = 0) => rows.forEach((chunks, i) => {
      setTimeout(() => {
        devBoxes[i].innerHTML = '';
        chunks.forEach(c => {
          const e = U.el('div', { class: 'chunk' });
          e.textContent = c;
          e.style.background = `var(--ax${i})`;
          devBoxes[i].appendChild(e);
        });
      }, delay + i * 90);
    });

    const btns = C.map(c => {
      const b = U.el('div', { class: 'chip mut', style: 'font-size:12px' });
      b.textContent = c.n;
      names.appendChild(b); return b;
    });

    C.forEach((c, i) => {
      tl.at(600 + i * 3400, () => {
        btns.forEach((b, k) => b.className = 'chip ' + (k === i ? `c${i % 6}` : 'mut'));
        btns[i].style.fontSize = '12px';
        paint(c.before);
        msg.innerHTML = `<span class="mono hl-a">${c.n}</span> &nbsp; ${c.ir}`;
      });
      tl.at(600 + i * 3400 + 1500, () => {
        paint(c.after, 0);
        msg.innerHTML = c.d;
      });
    });
    tl.at(600 + 6 * 3400, () => {
      msg.innerHTML = '导出阶段的 <span class="mono">-sdy-reshard-to-collectives</span> 负责把 <span class="mono">sdy.reshard</span> 翻译成上面这些算子。';
      paint([['?'], ['?'], ['?'], ['?']]);
    });
  }
},

/* ------------------------------------------------------ 17 SPMD 落地 */
{
  kicker: '第 7 步 · 最后一公里',
  title: '全局形状 → <span class="hl-a">局部形状</span>：变成 SPMD',
  sub: '传播阶段讨论的都是"逻辑上的全局张量"。真正下发到设备之前，必须把每个类型换成该设备上真实持有的那一块。',
  caption: '这一步之后，程序里不再有"分片"这个概念 —— 每个设备拿到的就是一段普通的、单机的、可直接执行的代码（SPMD：同一份程序，不同数据）。',
  code: `// 转换前（全局视角，带分片标注）
func.func @main(%arg0: tensor<8x8xf32>
      {sdy.sharding = #sdy.sharding<@mesh_xy,
                          [{"x"}, {"y"}]>})
      -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// 转换后（局部视角，每设备一份）
func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
// 分片属性、mesh 全部消失，
// 剩下的就是每台设备要执行的普通代码`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:16px;align-items:center;width:100%' });
    wrap.innerHTML = `
      <div class="row" style="gap:24px;align-items:center">
        <div class="col" style="gap:8px;align-items:center">
          <div class="mid faint">全局视角 · 1 个逻辑张量</div>
          <div id="g"></div>
          <div class="mono small dim">tensor&lt;8x8xf32&gt; + sharding</div>
        </div>
        <div class="col" style="gap:8px;align-items:center">
          <div class="arrow anim" style="font-size:30px">⟹</div>
          <div class="small mono hl-a">-sdy-convert-global-to-local</div>
        </div>
        <div class="col" style="gap:8px;align-items:center">
          <div class="mid faint">局部视角 · 4 份真实数据</div>
          <div id="l"></div>
          <div class="mono small dim">tensor&lt;4x4xf32&gt; ×4</div>
        </div>
      </div>
      <div class="formula" id="msg" style="min-height:46px;display:flex;align-items:center;text-align:center;max-width:900px"></div>`;
    root.appendChild(wrap);

    const gHost = wrap.querySelector('#g');
    const gg = U.tensorGrid(8, 8, D.owner22, { cell: 13, gap: 2, text: (r, c) => (r % 4 === 1 && c % 4 === 1) ? 'D' + D.owner22(r, c) : '' });
    gHost.appendChild(gg);

    const lHost = wrap.querySelector('#l');
    const lm = U.el('div', { class: 'mesh' });
    lm.style.gridTemplateColumns = 'repeat(2, 76px)';
    lm.style.gridTemplateRows = 'repeat(2, 76px)';
    for (let id = 0; id < 4; id++) {
      const d = U.el('div', { class: 'dev' });
      d.style.setProperty('--c', `var(--ax${id})`);
      d.style.opacity = '.15'; d.style.transition = 'opacity .6s';
      d.appendChild(U.tensorGrid(6, 6, () => id, { cell: 6, gap: 1 }));
      lm.appendChild(d);
    }
    lHost.appendChild(lm);
    const msg = wrap.querySelector('#msg');

    tl.at(700, () => { msg.innerHTML = '传播结束后，我们已知每个张量该怎么切 —— 但类型还是<b>全局</b>的 8x8。'; });
    tl.at(3000, () => {
      msg.innerHTML = '按分片把逻辑维度除以轴大小：<span class="mono hl-a">8/2 = 4</span>，于是类型变为 <span class="mono">tensor&lt;4x4xf32&gt;</span>';
      lm.querySelectorAll('.dev').forEach((d, i) => setTimeout(() => d.style.opacity = '1', i * 180));
    });
    tl.at(6200, () => {
      msg.innerHTML = '如果相邻算子要求的分片不一致，中间会被插入 <span class="mono">sdy.reshard</span>，再被翻译成集合通信。';
      gHost.style.transition = 'opacity .4s'; gHost.style.opacity = '.35';
    });
    tl.at(9000, () => { gHost.style.opacity = '1'; msg.innerHTML = '最终：<b>每个设备执行同一份代码</b>，只是各自的数据不同 —— 这就是 SPMD。'; });
    tl.at(12000, () => { msg.innerHTML = '分片标注（<span class="mono">sdy.sharding</span>）与 mesh 都被擦除：<span class="mono">-sdy-close-shardings</span> / <span class="mono">-sdy-drop-sharding-and-mesh</span>。'; });
  }
},

/* --------------------------------------------------------- 18 MPMD */
{
  kicker: '第 8 步 · 另一种并行',
  title: 'MPMD：把程序切成<span class="hl-a">多个子程序</span>',
  sub: 'SDY 是"一份程序、所有设备跑同一个 SPMD 程序"。MPMD 则允许<b>不同设备组跑不同的程序片段</b>（fragment），片段之间显式搬运数据。',
  caption: '适用场景：异构设备（GPU 段 + CPU 段）、流水线并行（不同 stage 放不同设备）、或把一部分计算交给完全不同的后端。',
  code: `// 拓扑：把设备划分成多个命名 mesh
//   注意：并没有 mpmd.mesh 这个 op —— mesh 只存在于
//   topology 属性和 mesh_tensor 类型里
#topology = #mpmd.topology<
    "m0": <["x"=4]>>, <"cpu": <["cpu_x"=2]>>>

func.func @main(%arg0: tensor<8x8xf32>) -> ... {
  // 片段的输入输出是"带 mesh 的类型"
  //   !mpmd.mesh_tensor<"m0", tensor<8x8xf32>>

  // 片段：一段计算绑定到某个 mesh 与 stage
  %1 = mpmd.fragment<mesh="m0", origin=["f1"], stage=0> (%0)
       (%a: tensor<8x8xf32>) {
    ...
    mpmd.return %r : tensor<8x8xf32>
  } : (!mpmd.mesh_tensor<"m0", tensor<8x8xf32>>)
   -> (!mpmd.mesh_tensor<"m0", tensor<8x8xf32>>)

  // 跨 mesh 搬运：类型里带着 mesh 名
  %2 = mpmd.transfer %1
       : (!mpmd.mesh_tensor<"m0",  tensor<8x8xf32>>)
      -> (!mpmd.mesh_tensor<"cpu", tensor<8x8xf32>>)
}`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:16px;align-items:center;width:100%' });
    wrap.innerHTML = `
      <div class="row" style="gap:16px;align-items:center" id="row"></div>
      <div class="formula" id="msg" style="min-height:48px;display:flex;align-items:center;text-align:center;max-width:940px"></div>`;
    root.appendChild(wrap);

    const r = wrap.querySelector('#row');
    const frag = (title, meshLabel, n, colorBase, sub) => {
      const box = U.el('div', { class: 'card', style: 'min-width:210px;text-align:center;transition:border-color .4s' });
      box.innerHTML = `<div class="card-t" style="font-size:13px">${title}</div>
        <div class="mono small dim" style="margin-bottom:8px">mesh = "${meshLabel}"</div>`;
      const g = U.el('div', { class: 'row', style: 'gap:6px;justify-content:center' });
      for (let i = 0; i < n; i++) {
        const d = U.el('div', { class: 'dev', style: 'width:36px;height:36px' });
        d.style.setProperty('--c', `var(--ax${(colorBase + i) % 6})`);
        g.appendChild(d);
      }
      box.appendChild(g);
      box.insertAdjacentHTML('beforeend', `<div class="small faint" style="margin-top:8px">${sub}</div>`);
      return box;
    };

    const A = frag('片段 ①', 'm0', 4, 0, '4 台设备 · 内部仍是 SPMD');
    const AR = U.el('div', { class: 'col', style: 'gap:4px;align-items:center' });
    AR.innerHTML = `<div class="mono small hl-w">mpmd.transfer</div><div class="arrow anim" style="font-size:26px">⟹</div>`;
    const B = frag('片段 ②', 'cpu', 2, 2, '2 台设备 · 内部仍是 SPMD');

    [A, AR, B].forEach(e => r.appendChild(e));

    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { msg.innerHTML = '<b>拓扑</b>：先把设备划分成多个命名 mesh，每个 mesh 是一个独立的 SPMD 世界。'; A.style.borderColor = 'var(--accent)'; });
    tl.at(3200, () => {
      msg.innerHTML = '<b>片段 (fragment)</b>：一段计算绑定到一个 mesh，内部仍然是普通的 SPMD 程序，照样可以用 SDY 的分片。';
      A.style.borderColor = '';
      A.querySelectorAll('.dev').forEach((d, i) => setTimeout(() => { d.classList.add('lit'); setTimeout(() => d.classList.remove('lit'), 700); }, i * 130));
    });
    tl.at(6200, () => { msg.innerHTML = '<b>stage</b> 标记片段在流水线中的第几棒；同一 mesh 上的多个片段还可以被合并优化。'; });
    tl.at(9000, () => {
      msg.innerHTML = '<b>mpmd.transfer</b>：把张量从 m0 搬到 cpu。类型里带着 mesh 名，编译器据此知道数据现在在哪。';
      AR.querySelector('.arrow').classList.add('pulse');
    });
    tl.at(12200, () => {
      msg.innerHTML = '片段 ② 在另外 2 台设备上执行 —— 输入已经由 transfer 送过来了。';
      B.querySelectorAll('.dev').forEach((d, i) => setTimeout(() => { d.classList.add('lit'); setTimeout(() => d.classList.remove('lit'), 700); }, i * 130));
      B.style.borderColor = 'var(--accent)';
    });
    tl.at(15400, () => {
      msg.innerHTML = '<b>核心差异</b>：SPMD = 一份程序 / 所有设备；MPMD = 多份程序 / 各自设备组，用 transfer 连接。';
    });
  }
},

/* ------------------------------------------------------- 19 速查表 */
{
  kicker: '收尾',
  title: '速查表 &amp; 上手命令',
  sub: '把这一路学到的语法浓缩成一页。真正的下一步：拿一个带分片的 StableHLO 程序，跑一遍传播亲眼看看。',
  caption: '本动画的全部 IR 片段均取自 openxla/shardy 仓库的文档与测试用例。',
  code: `# 构建（本机已完成）
/data/bin/bzl build -c opt //shardy/...

# 跑单个 pass 做实验
bazel-bin/shardy/tools/sdy_opt in.mlir -sdy-basic-propagate
bazel-bin/shardy/tools/sdy_opt in.mlir -sdy-propagation-pipeline
bazel-bin/shardy/tools/sdy_opt in.mlir -sdy-export-pipeline

# 看每个分片是怎么传播来的
bazel-bin/shardy/tools/sdy_opt in.mlir \\
  -sdy-basic-propagate=debug-sharding-origins=true

# 打印所有算子的分片规则
bazel-bin/shardy/tools/sdy_opt in.mlir \\
  -sdy-populate-op-sharding-rules

# MPMD 四段流水线
bazel-bin/shardy/tools/mpmd_opt in.mlir \\
  -mpmd-import-pipeline -mpmd-optimize-pipeline \\
  -mpmd-sharding-propagation-pipeline -mpmd-export-pipeline`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'row', style: 'gap:18px;align-items:flex-start;width:100%' });
    wrap.innerHTML = `
      <table class="ftable" style="font-size:12.5px;flex:1">
        <tr><th>概念</th><th>写法</th></tr>
        <tr><td>网格</td><td class="mono">sdy.mesh @m = &lt;["x"=2, "y"=2]&gt;</td></tr>
        <tr><td>分片</td><td class="mono">#sdy.sharding&lt;@m, [{"x"},{"y"}]&gt;</td></tr>
        <tr><td>开维</td><td class="mono">[{"x", ?}, {}]</td></tr>
        <tr><td>显式复制</td><td class="mono">replicated={"y"}</td></tr>
        <tr><td>子轴</td><td class="mono">"x":(1)2</td></tr>
        <tr><td>优先级</td><td class="mono">[{"x"}p1, {"y"}]</td></tr>
        <tr><td>未归约轴</td><td class="mono">unreduced={"z"}</td></tr>
        <tr><td>分片规则</td><td class="mono">([i,k],[k,j])-&gt;([i,j])</td></tr>
        <tr><td>约束</td><td class="mono">sdy.sharding_constraint</td></tr>
        <tr><td>分组</td><td class="mono">sdy.sharding_group ... group_id=0</td></tr>
        <tr><td>手动</td><td class="mono">sdy.manual_computation</td></tr>
        <tr><td>屏障</td><td class="mono">sdy.propagation_barrier</td></tr>
        <tr><td>换分片</td><td class="mono">sdy.reshard</td></tr>
      </table>
      <table class="ftable" style="font-size:12.5px;flex:1">
        <tr><th>集合通信</th><th>含义</th></tr>
        <tr><td class="mono">sdy.all_gather</td><td>分片 → 复制</td></tr>
        <tr><td class="mono">sdy.all_reduce</td><td>部分和 → 全和（复制）</td></tr>
        <tr><td class="mono">sdy.all_slice</td><td>复制 → 分片</td></tr>
        <tr><td class="mono">sdy.reduce_scatter</td><td>先归约再分片</td></tr>
        <tr><td class="mono">sdy.all_to_all</td><td>设备间互换切片</td></tr>
        <tr><td class="mono">sdy.collective_permute</td><td>重排 / 替换轴</td></tr>
        <tr><th>关键 pass</th><th>作用</th></tr>
        <tr><td class="mono">-sdy-import-pipeline</td><td>导入清理</td></tr>
        <tr><td class="mono">-sdy-basic-propagate</td><td>最保守的传播</td></tr>
        <tr><td class="mono">-sdy-user-priority-propagate</td><td>最完整的传播</td></tr>
        <tr><td class="mono">-sdy-insert-explicit-reshards</td><td>补 reshard</td></tr>
        <tr><td class="mono">-sdy-reshard-to-collectives</td><td>reshard → 通信</td></tr>
        <tr><td class="mono">-sdy-convert-global-to-local</td><td>全局 → 局部形状</td></tr>
      </table>`;
    root.appendChild(wrap);

    const rows = wrap.querySelectorAll('tr');
    rows.forEach(r => { r.style.opacity = '0'; r.style.transition = 'opacity .3s'; });
    rows.forEach((r, i) => tl.at(200 + i * 190, () => r.style.opacity = '1'));
  }
},

];
