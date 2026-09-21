/* ==========================================================================
   L2-07 · sharding-group-propagation
   --------------------------------------------------------------------------
   覆盖：transforms/propagation/test/sharding_group_propagation.mlir (254 行 / 12 用例)
   目标：讲透分片组在传播中的行为 —— 一条绕开数据流的独立通道。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 独立通道 */
{
  kicker: 'L2-07 · 分片组传播',
  title: '分片组：传播里的<span class="hl-a">第二条通道</span>',
  sub: '正常传播沿着<b>数据流</b>走。分片组提供另一条路：把没有数据依赖的张量绑在一起，让它们分片一致。',
  caption: '注意 RUN 行：<span class="mono">-sdy-basic-propagate</span> —— 组<b>不是单独的 pass</b>，而是内建在传播里的一条通道。',
  code: `// RUN: sdy_opt %s -split-input-file -sdy-basic-propagate -verify-diagnostics

// 正常通道：沿数据流
%0 = stablehlo.tanh %arg0        // %0 从 %arg0 继承分片

// 组通道：没有数据依赖也能同步
sdy.sharding_group %0 group_id = 0 : tensor<8x8xf32>
sdy.sharding_group %1 group_id = 0 : tensor<8x8xf32>
// %0 与 %1 之间没有任何算子连接，但同组 -> 分片一致

// 组的实现要点（来自导入阶段 L3-09）：
//   组 id 会先做【传递闭包合并】并规范化为 0..N-1
//   然后在传播中作为一条独立通道生效`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:16px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:34px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:9px;align-items:center' });
      c.innerHTML = `
        <div class="row" style="gap:10px;align-items:center">
          <div class="chip c0" style="padding:8px 13px">%0</div>
          <div style="color:var(--ink-faint);font-size:20px">⇢</div>
          <div class="chip mut" style="padding:8px 13px">（无算子连接）</div>
          <div style="color:var(--ink-faint);font-size:20px">⇠</div>
          <div class="chip c1" style="padding:8px 13px">%1</div>
        </div>
        <div class="small faint">数据流上互不相干</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>问题</b>：两个张量之间没有数据依赖，传播<b>没有任何理由</b>把它们切成一样。';
    });
    tl.at(4400, () => {
      const c = U.el('div', { class: 'col', style: 'gap:9px;align-items:center' });
      c.innerHTML = `
        <div class="row" style="gap:10px;align-items:center">
          <div class="chip c0" style="padding:8px 13px">%0</div>
          <div style="color:var(--accent);font-size:22px">⟺</div>
          <div class="chip c1" style="padding:8px 13px">%1</div>
        </div>
        <div class="small" style="color:var(--accent)">同 group_id=0</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>组通道</b>：一旦其中一个拿到分片，另一个立刻跟随 —— 不需要数据依赖。';
    });
    tl.at(8200, () => {
      msg.innerHTML = '<b>典型场景</b>：输入与输出之间没有数据依赖（如常量与某个返回值），但业务上要求它们切法一致。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<b>与 L1-09 的关系</b>：那里讲 <span class="mono">sdy.sharding_group</span> 的语法，这里讲它在传播中的<b>行为</b>。';
    });
  }
},

/* ---------------------------------------------------- 2 穿透屏障 */
{
  kicker: 'L2-07 · 分片组传播',
  title: '最强例证：组能<span class="hl-a">穿透传播屏障</span>',
  sub: '即使路径上隔着 `allowed_direction=NONE` 的屏障，组这条通道依然把分片传了过去。',
  caption: '这直接证明了组<b>不是</b>"数据流传播的特例"，而是一条<b>独立通道</b>。',
  code: `// %0 (常量) 与 %4 (add) 同组
// 但从 %0 到 %4 的路径上隔着：
//   %3 = sdy.propagation_barrier %1 allowed_direction=NONE
// 屏障禁止一切传播 —— 可是组仍然生效

%0 = sdy.constant dense<0.000000e+00> : tensor<8x8xf32>
%1 = sdy.constant dense<0.000000e+00> : tensor<8x8xf32>
%2 = stablehlo.tanh %arg0          // 从 %arg0 拿到 [{"a",?},{?}]
%3 = sdy.propagation_barrier %1 allowed_direction=NONE
%4 = stablehlo.add %2, %3          // 从 %2 拿到分片
sdy.sharding_group %0 group_id=0
sdy.sharding_group %4 group_id=0

// 结果：%0 也被赋予 [{"a", ?}, {?}]
//   sdy.constant {sharding_per_value=[<@mesh, [{"a", ?}, {?}]>]} dense<0.…>

// 附带：屏障 %3 虽不在组里，但作为恒等算子也拿到了分片`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:12px;align-items:center;justify-content:center;flex-wrap:wrap" id="chain"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const chain = wrap.querySelector('#chain'), msg = wrap.querySelector('#msg');

    const node = (label, sub, color) => {
      const c = U.el('div', { class: 'col', style: 'gap:4px;align-items:center;transition:.4s;opacity:1' });
      c.innerHTML = `<div class="chip ${color}" style="padding:7px 12px;font-size:12.5px">${label}</div>
        <div class="small faint" style="font-size:10px">${sub}</div>`;
      chain.appendChild(c); return c;
    };
    const ar = (txt) => chain.insertAdjacentHTML('beforeend', `<div style="color:var(--ink-faint);font-size:16px">${txt}</div>`);

    tl.at(700, () => {
      node('%arg0', '有分片', 'c0'); ar('→');
      node('%2 tanh', '继承', 'mut'); ar('→');
      node('%4 add', '继承', 'mut'); ar('⤫');
      node('%3 屏障', 'NONE', 'c3');
      msg.innerHTML = '<span class="mono">%2 → %4</span> 是正常数据流；<span class="mono">%3</span> 是 <b>NONE 屏障</b>，禁止一切传播。';
    });
    tl.at(4600, () => {
      ar('⤫');
      node('%1 const', '无分片', 'mut'); ar('→');
      node('%0 const', '无分片', 'mut');
      msg.innerHTML = '<span class="mono">%0</span> 与 <span class="mono">%4</span> 同组，但它们之间<b>没有数据流路径</b> —— 中间还被屏障切断。';
    });
    tl.at(8600, () => {
      const chips = chain.querySelectorAll('.chip');
      chips.forEach(c => { if (c.textContent.includes('const') || c.textContent.includes('%4')) c.classList.add('pulse'); });
      msg.innerHTML = '<b>组通道生效</b>：<span class="mono">%0</span> 被赋予与 <span class="mono">%4</span> 相同的分片 <span class="mono">[{"a", ?}, {?}]</span>。';
    });
    tl.at(12200, () => {
      msg.innerHTML = '<b>结论</b>：组是一条<b>绕开数据流</b>的独立通道 —— 屏障拦不住它。';
    });
    tl.at(14600, () => {
      msg.innerHTML = '<span class="mono">%1</span> 与屏障 <span class="mono">%3</span> 也顺带拿到了分片（屏障是恒等算子）。';
    });
  }
},

/* ------------------------------------------------ 3 组内冲突裁决 */
{
   kicker: 'L2-07 · 分片组传播',
  title: '组内冲突：按<span class="hl-a">用户优先级</span>裁决',
  sub: '同一个组里的张量若已有不同的分片标注，传播必须选一个。裁决规则与 L2-06 一致：<b>优先级高的赢</b>。',
  caption: '这也是为什么 L1-09 说"组是强制同分片" —— 强制执行时就需要一条裁决规则，否则无从下手。',
  code: `// %arg0: 第 0 维 {"a", ?}p0，第 1 维 {"c", ?}
// %arg1: 第 0 维 {"c", "a",?}p1，第 1 维 {"b", ?}
// 两者同组 -> 必须统一

%0 = stablehlo.add %arg0, %arg0
%1 = stablehlo.add %arg1, %arg1
%2 = stablehlo.add %0, %1
sdy.sharding_group %0 group_id = 0
sdy.sharding_group %1 group_id = 0
sdy.sharding_group %2 group_id = 0

// 结果：三者统一为 [{"a", ?}, {"c", ?}]
//   即 %arg0 那一侧（p0）胜出
//   %arg1 的 {"c","a",?}p1 与 {"b",?} 被放弃`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:14px;justify-content:center;align-items:stretch" id="rows"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const rows = wrap.querySelector('#rows'), msg = wrap.querySelector('#msg');

    const mk = (title, body, color, op) => {
      const c = U.el('div', { class: 'card', style: `width:340px;border-color:${color}55;transition:.45s;opacity:${op}` });
      c.innerHTML = `<div class="card-t" style="color:${color};font-size:12px">${title}</div>
        <div class="card-d" style="font-size:11.5px;line-height:1.65">${body}</div>`;
      rows.appendChild(c); return c;
    };

    tl.at(700, () => {
      mk('%arg0 一侧（p0）', '<span class="mono">[{"a", ?}p0, {"c", ?}]</span>', 'var(--ok)', '.5');
      mk('%arg1 一侧（p1）', '<span class="mono">[{"c", "a",?}p1, {"b", ?}]</span>', 'var(--bad)', '.5');
      msg.innerHTML = '两者同组，但要求<b>不一致</b> —— 必须裁决。';
    });
    tl.at(4400, () => {
      rows.querySelectorAll('.card')[0].style.opacity = '1';
      msg.innerHTML = '<b>p0 胜出</b>：第 0 维用 <span class="mono">"a"</span>（不是 <span class="mono">"c","a"</span>），第 1 维用 <span class="mono">"c"</span>（不是 <span class="mono">"b"</span>）。';
    });
    tl.at(8000, () => {
      msg.innerHTML = '<b>结果</b>：组内三个张量统一为 <span class="mono">[{"a", ?}, {"c", ?}]</span>，与 <span class="mono">%arg0</span> 那一侧一致。';
    });
    tl.at(11200, () => {
      msg.innerHTML = '<b>裁决依据</b>：与 L2-06 同一条规则 —— 用户优先级。组本身不引入新的裁决方式。';
    });
    tl.at(13400, () => {
      msg.innerHTML = '实践建议：<b>同组张量尽量给一致或互补的标注</b>，否则结果取决于优先级，不易预测。';
    });
  }
},

/* ------------------------------------------------ 4 完全不兼容 */
{
  kicker: 'L2-07 · 分片组传播',
  title: '完全不兼容时：传播会<span class="hl-a">自动插入约束</span>',
  sub: '两个成员分别要求 `"a"` 和 `"b"`，谁也无法覆盖谁 —— 传播不会静默放弃，而是<b>插入分片约束来统一</b>。',
  caption: '这是很有信息量的一类行为：它说明分片组是<b>硬约束</b>，传播必须让它成立，必要时改动 IR。',
  code: `// %arg0 要求 "a"，%arg1 要求 "b"，同组
%arg0: [{"a"}, {}]
%arg1: [{"b"}, {}]
sdy.sharding_group %arg0 group_id = 0
sdy.sharding_group %arg1 group_id = 0

// 传播输出（自动插入两条约束）：
%[[WSC_0]] = sdy.sharding_constraint %arg1 <@mesh, [{"b", ?}, {?}]>
%[[WSC_1]] = sdy.sharding_constraint %arg0 <@mesh, [{"b", ?}, {?}]>
sdy.sharding_group %[[WSC_1]] group_id=0
sdy.sharding_group %[[WSC_0]] group_id=0

// 读法：
//   两个成员都被约束成 [{"b", ?}, {?}]
//   组被【重建】到新插入的约束上（而不是原来的参数上）
//
// 测试注释点明时机：
//   "... compatibility checks happend after unification +
//    canonicalization of group ids."`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%' });
    wrap.innerHTML = `
      <div class="row" style="gap:16px;justify-content:center;align-items:center">
        <div class="col" style="gap:8px;align-items:center">
          <div class="small mono faint">输入</div>
          <div class="chip c0" style="padding:7px 12px">%arg0: [{"a"},{}]</div>
          <div class="chip c1" style="padding:7px 12px">%arg1: [{"b"},{}]</div>
          <div class="small" style="color:var(--bad);font-size:11px">同组但不兼容</div>
        </div>
        <div class="arrow anim" style="font-size:26px">⟹</div>
        <div class="col" style="gap:8px;align-items:center">
          <div class="small mono faint">输出</div>
          <div class="chip c1" style="padding:7px 12px">%arg1 → constraint [{"b",?},{?}]</div>
          <div class="chip c1" style="padding:7px 12px">%arg0 → constraint [{"b",?},{?}]</div>
          <div class="small" style="color:var(--ok);font-size:11px">组重建到约束上</div>
        </div>
      </div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const msg = wrap.querySelector('#msg');

    tl.at(900, () => { msg.innerHTML = '<b>问题</b>：两个成员的分片标注互不相容，谁也无法"说服"谁。'; });
    tl.at(3800, () => { msg.innerHTML = '<b>传播的选择</b>：不是放弃，而是<b>插入 <span class="mono">sdy.sharding_constraint</span></b> 把两者统一。'; });
    tl.at(7400, () => {
      msg.innerHTML = '注意结果：<span class="mono">%arg0</span> 原本是 <span class="mono">"a"</span>，被统一到了 <span class="mono">"b"</span> —— 说明裁决取决于处理顺序。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '组被<b>重建</b>到新插入的约束上，而不是保留在原来的函数参数上。';
    });
    tl.at(13400, () => {
      msg.innerHTML = '<b>同族 5 个用例</b>覆盖：组内 / manual_computation 内 / 与已有约束冲突 / 部分闭的复制分片 / 全开的复制分片。';
    });
  }
},

/* ---------------------------------------------------- 5 其它两条 */
{
  kicker: 'L2-07 · 分片组传播',
  title: '两个补充：<span class="hl-a">已有分片直接赋值</span>、<span class="hl-a">跨数据流边</span>',
  sub: '组内已有成员带分片时，其余成员直接继承；组成员位于数据流边两侧时同样能同步。',
  caption: '这两个用例说明组的通道是"全局"的 —— 不限于相邻算子，也不限于同一条边。',
  code: `// ① 已有分片直接赋给组内成员
%arg0: [{"a"}, {"b"}]        // 已有分片
%arg1: [{"a"}, {"b"}]        // 已有分片
%0 = stablehlo.constant dense<0.0> : tensor<8x8xf32>   // 无分片
sdy.sharding_group %arg0 group_id = 0
sdy.sharding_group %arg1 group_id = 0
sdy.sharding_group %0 group_id = 0
// 结果：常量直接拿到 [{"a"}, {"b"}]

// ② 跨数据流边
//   用例 shard_as_across_dataflow_edge：
//   组成员位于数据流边的两侧时，组的分片仍能同步过去

// 另有两个用例覆盖组 id 的处理：
//   different_group_ids_greater_group_id_first
//   shard_as_reflects_user_priority（见第 4 幕）`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 已有分片直接赋值', c: '#4ade80',
        d: '组内两个参数已有 <span class="mono">[{"a"}, {"b"}]</span>，<br>常量成员<b>直接继承</b>这个分片。' },
      { t: '② 跨数据流边同步', c: '#38bdf8',
        d: '组成员位于数据流边的两侧时，<br>组的分片仍能跨过去 —— 通道是全局的。' },
      { t: '③ 组 id 的处理', c: '#c084fc',
        d: '导入阶段会做传递闭包合并与 id 规范化（L3-09）；<br>传播阶段按规范化后的 id 处理。' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:238px;opacity:.34;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.34');
      msg.innerHTML = [
        '常量本可以"自由决定"分片，但组的硬约束让它必须跟同伴一致。',
        '数据流边（L1-08）本身也是一条传播通道；组与它并不冲突，可以同时生效。',
        '记住顺序：<b>先合并组、规范化 id，再做兼容性检查</b>（测试注释里写明了这一点）。',
      ][i];
    }));
    tl.at(12000, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>小结</b>：组是硬约束 —— 传播必须让它成立，必要时插入约束、跨边同步、穿透屏障。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L2-07 · 练习',
  title: '练一练：<span class="hl-a">组会怎么起作用</span>',
  sub: '三道题分别考：独立通道、冲突裁决、不兼容时的行为。',
  caption: '一句话总结：<b>组是绕过数据流的硬约束通道</b>。',
  code: `// 题 1：两个张量之间没有数据依赖，中间还隔着
//       allowed_direction=NONE 的屏障，同组还能同步吗？

// 题 2：组内两个成员分别要求 p0 的 "a" 和 p1 的 "b"，
//       最终用哪个？

// 题 3：组内两个成员的分片完全不兼容时，传播会？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '两个张量没有数据依赖，中间还隔着 <span class="mono">allowed_direction=NONE</span> 的屏障，同组还能同步吗？',
        a: '<b class="badge ok">能</b>。组是一条<b>绕开数据流</b>的独立通道，屏障拦不住它。' +
           '<br>这正是用例 <span class="mono">shard_as_applies_despite_barrier</span> 展示的现象。' +
           '<br><span class="dim">这也是"组是独立通道"最有说服力的证据 —— 如果组只是数据流传播的特例，屏障必然会阻断它。</span>'
      },
      {
        q: '组内两个成员分别要求 p0 的 <span class="mono">"a"</span> 和 p1 的 <span class="mono">"b"</span>，最终用哪个？',
        a: '<b>用 p0 的 <span class="mono">"a"</span></b>。组内冲突按<b>用户优先级</b>裁决（与 L2-06 同一条规则）。' +
           '<br><span class="dim">组本身不引入新的裁决方式，它只是"强制统一"，统一到哪个由优先级决定。</span>'
      },
      {
        q: '组内两个成员的分片<b>完全不兼容</b>（一个要 "a"、一个要 "b"），传播会怎么做？',
        a: '<b>插入额外的 <span class="mono">sdy.sharding_constraint</span></b> 把两者统一，并把组重建到新约束上。' +
           '<br><b>不是放弃</b> —— 组是硬约束，传播必须让它成立，必要时改 IR。' +
           '<br><span class="dim">裁决结果取决于处理顺序：测试里 <span class="mono">%arg0</span> 的 "a" 被统一成了 "b"。</span>'
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
