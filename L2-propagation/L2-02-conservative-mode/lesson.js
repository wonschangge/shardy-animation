/* ==========================================================================
   L2-02 · conservative-mode
   --------------------------------------------------------------------------
   覆盖：transforms/propagation/test/basic_propagation_conservative.mlir (51)
   目标：讲透 -conservative-propagation 选项 —— 禁止分裂轴与不可整除轴。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 开关对比 */
{
  kicker: 'L2-02 · 保守模式',
  title: '同一个 pass，多一个<span class="hl-a">开关</span>',
  sub: '保守模式不是另一个算法，而是给基础传播加了<b>两条禁令</b>：不许拆分子轴、不许产生不可整除的分片。',
  caption: '动机：分裂轴与不可整除都会让后续的导出更复杂（需要 padding 或额外通信）。保守模式用"少推一点"换取"落地更简单"。',
  code: `// 默认：基础传播
// RUN: sdy_opt %s -sdy-basic-propagate

// 保守模式：同一个 pass，多一个选项
// RUN: sdy_opt %s -sdy-basic-propagate="conservative-propagation=true"

// 两条禁令：
//   ① 禁止【分裂轴】：不允许用子轴表达分片
//   ② 禁止【不可整除】：维大小必须能被分片轴大小整除
//
// 后果：某些本该传过去的分片，保守模式下会【直接不传】
//       -> 更多的"没有 sharding"，更少的通信与 padding`,
  duration: 12000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:22px;justify-content:center;align-items:stretch">
        <div class="card" style="width:300px;border-color:rgba(56,189,248,.5)">
          <div class="card-t" style="color:var(--ax0)">默认基础传播</div>
          <div class="card-d">允许拆子轴、允许不可整除。<br>
            <b>推得更多</b>，但导出时可能需要 padding / 额外通信。</div>
        </div>
        <div class="card" style="width:300px;border-color:rgba(251,191,36,.5)">
          <div class="card-t" style="color:var(--warn)">保守模式</div>
          <div class="card-d">两条禁令。<br>
            <b>推得更少</b>，但落地更简单、更可预测。</div>
        </div>
      </div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:760px"></div>`;
    root.appendChild(wrap);
    const msg = wrap.querySelector('#msg');
    tl.at(900, () => { msg.innerHTML = '注意 RUN 行的写法：<span class="mono">-sdy-basic-propagate="conservative-propagation=true"</span> —— 选项跟在 pass 后面。'; });
    tl.at(3600, () => { msg.innerHTML = '禁令 ① <b>禁止分裂轴</b>：不能出现 <span class="mono">"a":(1)2</span> 这类子轴。'; });
    tl.at(6400, () => { msg.innerHTML = '禁令 ② <b>禁止不可整除</b>：维大小必须能被分片轴大小整除。'; });
    tl.at(9200, () => { msg.innerHTML = '<b>代价</b>：本该传过去的分片会直接放弃 —— 结果里会出现更多"没有 sharding"的张量。'; });
  }
},

/* ------------------------------------------------ 2 禁令①：分裂轴 */
{
  kicker: 'L2-02 · 保守模式',
  title: '禁令 ①：需要<span class="hl-a">拆子轴</span>就完全不传',
  sub: '最典型的例子：`8` reshape 成 `2x4`，而轴 `a=4` 切的是原来那个 8。',
  caption: '注意测试用的是 <span class="mono">CHECK-NOT: sdy.sharding</span> —— 断言结果上<b>完全没有</b>分片属性，而不是"传得少一点"。',
  code: `sdy.mesh @mesh_a_4_b_2 = <["a"=4, "b"=2]>

func.func @reshape_split_dim(%arg0: tensor<8xf32>
      {sdy.sharding = #sdy.sharding<@mesh_a_4_b_2, [{"a"}]>})
      -> tensor<2x4xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// 期望：
//   stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
//   CHECK-NOT: sdy.sharding        <- 结果上没有任何分片

// 为什么：8 沿 a=4 切成 4 份；reshape 成 2x4 后
//   第 0 维只有 2，装不下 a（大小 4）
//   必须写成 "a":(1)2 与 "a":(2)2 -> 拆子轴
//   保守模式禁止 -> 放弃传播`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:22px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const box = (title, cells, color) => {
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center' });
      c.innerHTML = `<div class="small mono" style="color:${color}">${title}</div>
        <div class="row" style="gap:5px" data-d></div>`;
      const h = c.querySelector('[data-d]');
      cells.forEach(([t, cls]) => {
        const e = U.el('div', { class: 'chip ' + cls, style: 'padding:7px 12px;font-size:13px' });
        e.textContent = t; h.appendChild(e);
      });
      demo.appendChild(c); return c;
    };

    tl.at(700, () => {
      box('输入 tensor<8>，沿 a=4 切', [['"a"', 'c0']], 'var(--ax0)');
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹' }));
      box('reshape 后 tensor<2x4>', [['2', 'mut'], ['4', 'c0']], 'var(--ok)');
      msg.innerHTML = '第 0 维是 <b>2</b>，但轴 <span class="mono">a</span> 大小是 <b>4</b> —— 装不下。';
    });
    tl.at(4200, () => {
      msg.innerHTML = '<b>在默认模式下</b>：拆成 <span class="mono">"a":(1)2</span>（切第 0 维）与 <span class="mono">"a":(2)2</span>（切第 1 维）即可。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>在保守模式下</b>：分裂轴被禁止 → <span class="hl-w">整个传播放弃</span>，结果上没有任何分片。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '测试用 <span class="mono">CHECK-NOT: sdy.sharding</span> 精确断言了这一点 —— <b>不是"少传"，是"不传"</b>。';
    });
  }
},

/* -------------------------------------------- 3 禁令②：不可整除 */
{
  kicker: 'L2-02 · 保守模式',
  title: '禁令 ②：<span class="hl-a">不可整除</span>的轴被丢掉',
  sub: '与上一幕不同：这里不是"完全不传"，而是<b>把不整除的那个轴丢掉，其余的照传</b>。',
  caption: '对比很关键：<b>分裂轴是全局禁令</b>（触发就整条放弃），<b>不可整除是逐轴过滤</b>（只丢那一个轴）。',
  code: `sdy.mesh @mesh_a_2_b_8 = <["a"=2, "b"=8]>

// 输入第 0 维 = {"a", "b"}（2 x 8 = 16）
// reshape 后第 0 维大小是 4
func.func @multi_axis_major_dim_not_fully_sharded(
    %arg0: tensor<16xf32>
      {sdy.sharding = #sdy.sharding<@mesh_a_2_b_8, [{"a", "b"}]>})
    -> tensor<4x4xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<16xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// 结果：只留下 "a"
//   a=2 -> 4/2 = 2，整除 ✓
//   b=8 -> 4/8 < 1，不整除 ✗ 被丢掉
//   => [{"a", ?}, {?}]`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%;align-items:center' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'a = 2', r: '4 ÷ 2 = 2', ok: true, d: '整除 → <b>保留</b>', c: '#4ade80' },
      { t: 'b = 8', r: '4 ÷ 8 < 1', ok: false, d: '不整除 → <b>丢掉</b>', c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:300px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c}">轴 ${x.t}</div>
        <div class="mono" style="margin:6px 0;font-size:13px;color:#cfe0ff">${U.esc(x.r)}</b></div>
        <div class="card-d">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = 'reshape 后第 0 维大小是 <b>4</b>。逐个检查每个轴能否整除它。'; });
    tl.at(3800, () => { els[1].style.opacity = '1'; msg.innerHTML = '<span class="mono">b=8</span> 要求把 4 分成 8 份 —— 做不到，被过滤掉。'; });
    tl.at(7200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '最终结果 <span class="mono">[{"a", ?}, {?}]</span>：只留住能整除的那个轴。';
    });
    tl.at(10400, () => {
      msg.innerHTML = '<b>与上一幕的区别</b>：这里是<b>逐轴过滤</b>，不是整条放弃 —— 因为丢掉 <span class="mono">"b"</span> 后剩下的分片仍然合法。';
    });
  }
},

/* ------------------------------------------ 4 三种 reshape 结果对比 */
{
  kicker: 'L2-02 · 保守模式',
  title: '多轴 reshape 的<span class="hl-a">三种结局</span>',
  sub: '同一个"多轴切一维"的场景，结果取决于<b>轴的组合能否正好落在结果的各维上</b>。',
  caption: '判据只有两条：<b>每个轴是否整除它要落的维度</b>、<b>是否需要拆子轴</b>。',
  code: `// ① 完全对应 -> 全传
//   网格 a=16, b=2；32 -> 16x2
//   a=16 正好是第 0 维，b=2 正好是第 1 维
//   => [{"a", ?}, {"b", ?}]
@mesh_a_16_b_2 = <["a"=16, "b"=2]>

// ② 主维只用到部分轴 -> 部分传
//   网格 a=4, b=2；32 -> 16x2
//   a=4 整除 16 ✓；b=2 也整除 16，但两个轴都落在第 0 维
//   => [{"a", "b", ?}, {?}]
@mesh_a_4_b_2 = <["a"=4, "b"=2]>

// ③ 遇到不可整除 -> 丢掉那个轴
//   网格 a=2, b=8；16 -> 4x4
//   a=2 整除 4 ✓；b=8 不整除 4 ✗
//   => [{"a", ?}, {?}]`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 完全对应', m: 'a=16,b=2 → 16x2', r: '[{"a",?}, {"b",?}]', d: '每个轴正好落在一维上，<b>全部保留</b>。', c: '#4ade80' },
      { t: '② 落在同一维', m: 'a=4,b=2 → 16x2', r: '[{"a","b",?}, {?}]', d: '两轴都落在第 0 维且都整除，<b>叠加保留</b>。', c: '#38bdf8' },
      { t: '③ 遇到不整除', m: 'a=2,b=8 → 4x4', r: '[{"a",?}, {?}]', d: 'b 不整除 → <b>只丢 b</b>，a 仍然保留。', c: '#fbbf24' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:238px;opacity:.32;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="mono" style="margin:5px 0;font-size:10.5px;color:#93a3c4">${U.esc(x.m)}</div>
        <div class="mono" style="margin-bottom:5px;font-size:11px;color:#bdf7ec;overflow-wrap:anywhere">${U.esc(x.r)}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.32');
      msg.innerHTML = [
        '最理想的情形：每个轴的大小正好等于某一维 —— 无需拆分、无需 padding。',
        '两个轴切同一维是允许的（大小相乘）。只要都整除，就能共存。',
        '保守模式的核心价值：宁可少一个轴，也不要留下需要 padding 的分片。',
      ][i];
    }));
    tl.at(11800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '对照幕 2：那里<b>需要拆子轴</b>（全局禁令 → 全不传）；这里<b>只是不整除</b>（逐轴过滤 → 丢一个）。';
    });
  }
},

/* ------------------------------------------------------ 5 slice */
{
  kicker: 'L2-02 · 保守模式',
  title: 'slice：<span class="hl-a">切完就不整除了</span>',
  sub: '维度变小之后，原本整除的轴可能不再整除。保守模式会如实反映这一点。',
  caption: '注意第 2 维：<span class="mono">b=2</span> 其实<b>仍能整除</b>结果里的 2，但结果写的是 <span class="mono">{?}</span> —— 保守模式下传播更谨慎。',
  code: `sdy.mesh @mesh_a_16_b_2 = <["a"=16, "b"=2]>

// 32x4x8 --slice--> 32x1x2
func.func @slice(%arg0: tensor<32x4x8xf32>
      {sdy.sharding = #sdy.sharding<@mesh_a_16_b_2,
                        [{"a"}, {}, {"b"}]>})
      -> tensor<32x1x2xf32> {
  %0 = stablehlo.slice %arg0 [0:32, 1:2, 4:8:2]
       : (tensor<32x4x8xf32>) -> tensor<32x1x2xf32>
  return %0 : tensor<32x1x2xf32>
}

// 期望结果： [{"a", ?}, {?}, {?}]
//   dim0: 32 -> 32，a=16 整除 ✓  -> 保留 "a"
//   dim1: 4  -> 1，没有任何轴能整除 1 -> 丢
//   dim2: 8  -> 2，b=2 整除，但结果仍写 {?}
//          （保守模式下传播不主动把轴放回去）`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const dims = (title, vals, color) => {
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center' });
      c.innerHTML = `<div class="small mono" style="color:${color}">${title}</div>
        <div class="row" style="gap:5px" data-d></div>`;
      const h = c.querySelector('[data-d]');
      vals.forEach(([t, cls]) => {
        const e = U.el('div', { class: 'chip ' + cls, style: 'padding:6px 11px;font-size:12.5px' });
        e.textContent = t; h.appendChild(e);
      });
      demo.appendChild(c); return c;
    };

    tl.at(700, () => {
      dims('slice 前 32x4x8', [['{"a"}', 'c0'], ['{}', 'mut'], ['{"b"}', 'c0']], 'var(--ax0)');
      msg.innerHTML = '输入：dim0 沿 a 切、dim2 沿 b 切、dim1 不分片。';
    });
    tl.at(4000, () => {
      demo.innerHTML = '';
      dims('slice 后 32x1x2', [['{"a", ?}', 'c0'], ['{?}', 'mut'], ['{?}', 'mut']], 'var(--ok)');
      msg.innerHTML = '<b>dim0 保住了</b>：32 → 32，<span class="mono">a=16</span> 仍整除。';
    });
    tl.at(7400, () => {
      msg.innerHTML = '<b>dim1 丢掉了</b>：4 → <b>1</b>，没有任何轴能把 1 切成多份（1 不能被 16 或 2 整除）。';
    });
    tl.at(10600, () => {
      msg.innerHTML = '<b>dim2 也变成 {?}</b>：虽然 <span class="mono">b=2</span> 能整除新的 2，但保守模式<b>不主动把轴放回去</b>。';
    });
    tl.at(13200, () => {
      msg.innerHTML = '<b>读法</b>：<span class="mono">{?}</span> = "还没结论，且开放"；<span class="mono">{}</span> = "确定不分片"。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L2-02 · 练习',
  title: '练一练：<span class="hl-a">保守模式会怎么传？</span>',
  sub: '三道题分别考：分裂轴禁令、不可整除过滤、开关对比。',
  caption: '这一课很短，但"两条禁令的粒度不同"是关键 —— 一条全局、一条逐轴。',
  code: `// 题 1：保守模式下会传吗？
//   8 -> 2x4，轴 a=4 切原来的 8

// 题 2：保守模式下结果是什么？
//   mesh a=2, b=8；16 -> 4x4
//   输入第 0 维 = {"a", "b"}

// 题 3：默认模式与保守模式，哪个"推得更多"？`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">8</span> reshape 成 <span class="mono">2x4</span>，轴 <span class="mono">a=4</span> 切原来的 8。保守模式下会传播吗？',
        a: '<b class="badge bad">不会，完全不传</b><br>' +
           '第 0 维只有 2，装不下大小为 4 的轴 → 必须拆子轴 → 触发<b>全局禁令</b>。' +
           '<br><span class="dim">测试用 <span class="mono">CHECK-NOT: sdy.sharding</span> 断言结果上没有任何分片。</span>'
      },
      {
        q: '<span class="mono">mesh a=2, b=8</span>，<span class="mono">16</span> → <span class="mono">4x4</span>，输入第 0 维是 <span class="mono">{"a","b"}</span>。保守模式下结果？',
        a: '<span class="mono">[{"a", ?}, {?}]</span><br>' +
           '<b>逐轴检查</b>：<span class="mono">a=2</span> 整除 4 ✓ 保留；<span class="mono">b=8</span> 不整除 4 ✗ 丢掉。' +
           '<br><span class="dim">注意这里是<b>逐轴过滤</b>，不是整条放弃 —— 与题 1 的区别正在于此。</span>'
      },
      {
        q: '默认模式与保守模式，哪个「推得更多」？各自的代价是什么？',
        a: '<b>默认模式推得更多</b>：允许拆子轴、允许不可整除。' +
           '<br>代价：导出时可能需要 <b>padding</b>（L5-09）或额外通信。' +
           '<br><b>保守模式推得更少</b>，但落地更简单、更可预测 —— 适合"不想让编译器自作主张"的场景。' +
           '<br><span class="dim">两者的分片<b>都合法</b>，只是取舍不同。</span>'
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
