/* ==========================================================================
   L2-08 · data-flow-edges
   --------------------------------------------------------------------------
   覆盖：transforms/propagation/test/basic_propagation_data_flow_edges.mlir (855)
         transforms/propagation/test/basic_propagation_token.mlir (28)
   目标：讲透分片如何跨区域边界流动，以及 token 为什么被跳过。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 问题 */
{
  kicker: 'L2-08 · 数据流边',
  title: '区域算子：分片<span class="hl-a">不能标在它身上</span>',
  sub: '`stablehlo.while`、`stablehlo.case` 这类算子包含<b>区域</b>。它们的"分片"该写在哪里？',
  caption: '测试里反复出现 <span class="mono">CHECK-NOT: sdy.sharding</span> —— 这就是答案：区域算子的<b>属性字典里没有分片</b>。',
  code: `// 区域算子（都含 region）：
//   stablehlo.while / stablehlo.case / stablehlo.if
//   stablehlo.optimization_barrier
//   sdy.manual_computation / sdy.named_computation

// 问题：
%0 = "stablehlo.case"(%arg0) ({
  stablehlo.return %arg1 : tensor<4xi64>
}, {
  stablehlo.return %arg2 : tensor<4xi64>
}) : (tensor<i32>) -> tensor<4xi64>
// 分片标在哪？标在 %0 上？

// 答案：标在后面的 sdy.data_flow_edge 上
%1 = sdy.data_flow_edge %0 : tensor<4xi64>
// -> sdy.data_flow_edge %0 sharding=<@mesh, [{"a", ?}]>

// 为什么不能直接标在 case 上：
//   区域算子的语义由【所有分支】共同决定
//   每个分支的返回值可能来自不同来源
//   用一条属性无法表达这种"多对一"的关系`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:flex-start" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:8px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--bad)">✗ 直接标在 region 算子上</div>
        <div class="chip c3" style="padding:8px 14px">stablehlo.case {sharding=…}</div>
        <div class="small faint">无法表达"哪个分支贡献了什么"</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>困难</b>：<span class="mono">case</span> 有多个分支，每个分支返回不同的值 —— 一条分片属性说不清。';
    });
    tl.at(4400, () => {
      const c = U.el('div', { class: 'col', style: 'gap:8px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ok)">✓ 标在 data_flow_edge 上</div>
        <div class="row" style="gap:8px;align-items:center">
          <div class="chip mut" style="padding:8px 14px">stablehlo.case</div>
          <div class="arrow" style="font-size:18px">→</div>
          <div class="chip c4" style="padding:8px 14px">data_flow_edge {sharding=…}</div>
        </div>
        <div class="small faint">边独立于算子，关系清晰</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>解法</b>：把分片放在一条<b>显式的边</b>上 —— 就是 L1-08 讲过的 <span class="mono">sdy.data_flow_edge</span>。';
    });
    tl.at(8600, () => {
      msg.innerHTML = '<b>边的作用</b>：它桥接"区域算子的结果"与"下游使用者"，分片写在边上。';
    });
    tl.at(11200, () => {
      msg.innerHTML = '这也解释了测试里为什么总在区域算子后面看到 <span class="mono">CHECK-NOT: sdy.sharding</span>。';
    });
  }
},

/* ------------------------------------------------------ 2 case */
{
  kicker: 'L2-08 · 数据流边',
  title: '<span class="mono hl-a">case</span>：从分支返回值传到边',
  sub: '`%arg1` 带分片 `[{"a"}]` 且是第 0 个分支的返回值 —— 传播把它送到 `case` 结果的边上。',
  caption: '注意最终是 <span class="mono">[{"a", ?}]</span>：轴的<b>闭维标记被打开</b>了，因为传播加上的部分默认开放（L2-01 讲过）。',
  code: `// %arg1 有分片，且是分支 0 的返回值
%0 = "stablehlo.case"(%arg0) ({
  stablehlo.return %arg1 : tensor<4xi64>
}, {
  stablehlo.return %arg2 : tensor<4xi64>
}) : (tensor<i32>) -> tensor<4xi64>

%1 = sdy.data_flow_edge %0 : tensor<4xi64>
return %1 : tensor<4xi64>

// 期望：
//   sdy.data_flow_edge %0 sharding=<@mesh, [{"a", ?}]>
//   case 算子上【没有】分片属性

// 读法：
//   %arg1 = [{"a"}]    闭维
//   边    = [{"a", ?}]  闭的部分保留，另加开维标记`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:8px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ax0)">分支 0</div>
        <div class="chip c0" style="padding:8px 14px">stablehlo.return %arg1</div>
        <div class="small faint">%arg1 = [{"a"}]</div>`;
      demo.appendChild(c);
      msg.innerHTML = '只有<b>分支 0</b> 的返回值带分片；分支 1 的 <span class="mono">%arg2</span> 没有任何标注。';
    });
    tl.at(4200, () => {
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹', style: 'font-size:24px' }));
      const c = U.el('div', { class: 'col', style: 'gap:8px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ok)">结果上的边</div>
        <div class="chip c4" style="padding:8px 14px">data_flow_edge</div>
        <div class="chip c4" style="padding:6px 12px;font-size:12px">sharding=[{"a", ?}]</div>`;
      demo.appendChild(c);
      msg.innerHTML = '分支 0 的要求被送上 <span class="mono">case</span> 结果的边 —— 变成 <span class="mono">[{"a", ?}]</span>。';
    });
    tl.at(8000, () => {
      msg.innerHTML = '<b>注意闭维变成了开维</b>：<span class="mono">{"a"}</span> → <span class="mono">{"a", ?}</span>。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<b>为什么</b>：边是"桥"，传播在桥上可以继续加轴 —— 所以默认带 <span class="mono">?</span>（L2-01 的规则）。';
    });
    tl.at(13400, () => {
      msg.innerHTML = '若多个分支都带分片，就按最长兼容前缀/优先级裁决（见后续用例）。';
    });
  }
},

/* ------------------------------------------------------ 3 while */
{
  kicker: 'L2-08 · 数据流边',
  title: '<span class="mono hl-a">while</span>：分片穿过循环体',
  sub: '函数结果的要求反向穿过 `while`，落到循环体内的算子；体内的要求也能反向推出去 —— <b>双向通道</b>。',
  caption: '循环是最需要数据流边的场景：迭代值在循环入口、体内、循环出口三处出现，必须保持一致。',
  code: `// 函数结果被锁成 [{"a"}, {"b"}]
func.func @while_func_return(%arg0: tensor<32x96xf32>)
    -> (tensor<32x96xf32>
        {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %0)
      : tensor<32x96xf32>, tensor<i32>
    cond { ... } do {
    %6 = stablehlo.add %iterArg_2, %1 : tensor<i32>
    %7 = stablehlo.add %iterArg, %iterArg : tensor<32x96xf32>
    stablehlo.return %7, %6 : tensor<32x96xf32>, tensor<i32>
  }
  %4 = sdy.data_flow_edge %3#0 : tensor<32x96xf32>

// 期望：
//   体内 add -> [{"a", ?}, {"b", ?}]   （从函数结果反向推入）
//   while 上 -> 无分片属性
//   边      -> [{"a", ?}, {"b", ?}]`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:20px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const box = (t, sub, color) => {
      const c = U.el('div', { class: 'col', style: 'gap:5px;align-items:center' });
      c.innerHTML = `<div class="chip ${color}" style="padding:8px 13px;font-size:12.5px">${t}</div>
        <div class="small faint" style="font-size:10.5px">${sub}</div>`;
      demo.appendChild(c); return c;
    };

    tl.at(700, () => {
      box('函数结果', '[{"a"},{"b"}] 锁定', 'c3');
      msg.innerHTML = '<span class="mono">return</span> 处被用户锁成 <span class="mono">[{"a"}, {"b"}]</span>。';
    });
    tl.at(3800, () => {
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟸', style: 'font-size:22px' }));
      box('data_flow_edge', 'while 结果', 'c4');
      box('while', '无分片属性', 'mut');
      box('体内 add', '[{"a",?},{"b",?}]', 'c0');
      msg.innerHTML = '<b>反向传播</b>：要求从函数结果一路推回，穿过边与 <span class="mono">while</span>，落到体内的 <span class="mono">add</span>。';
    });
    tl.at(8000, () => {
      msg.innerHTML = '<b>再前向回到边</b>：体内算子的分片又沿前向回传到 <span class="mono">data_flow_edge</span>。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<b>另一个用例</b> <span class="mono">while_body_propagate_block_arg</span> 展示反向：体内算子显式带 <span class="mono">[{"a"},{"b"}]</span>，推给块参数。';
    });
    tl.at(13600, () => {
      msg.innerHTML = '<b>结论</b>：数据流边是<b>双向</b>通道 —— 循环的入口/体内/出口三处分片保持一致。';
    });
  }
},

/* ------------------------------------------------------ 4 多结果 */
{
  kicker: 'L2-08 · 数据流边',
  title: '多结果：<span class="hl-a">每条边互相独立</span>',
  sub: '`optimization_barrier` 有 2 个结果，就有 2 条边 —— 它们各自拿到<b>不同</b>的分片，互不干扰。',
  caption: '这体现了"边"的粒度：<b>结果是几元组，边就是几条</b>。这比"给算子标一条属性"表达力强得多。',
  code: `// 两个结果 -> 两条边
%1:2 = stablehlo.optimization_barrier %0, %arg1
     : tensor<32x96xf32>, tensor<32x96xf32>
%2 = sdy.data_flow_edge %1#0 : tensor<32x96xf32>
%3 = sdy.data_flow_edge %1#1 : tensor<32x96xf32>
return %2, %3 : tensor<32x96xf32>, tensor<32x96xf32>

// 期望：
//   边 #0 -> [{"a", ?}, {"b", ?}]
//   边 #1 -> [{"b", ?}, {?}]
//
// 注意两条边【不同】—— 各自来自自己的下游要求
//   函数结果的第 0 个位置锁 [{?}, {"b"}]
//   函数结果的第 1 个位置锁 [{"b"}, {?}]
// 两边独立传播，不会互相污染`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:16px;justify-content:center;align-items:stretch" id="rows"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const rows = wrap.querySelector('#rows'), msg = wrap.querySelector('#msg');

    const mk = (n, res, color) => {
      const c = U.el('div', { class: 'card', style: 'width:340px;opacity:.35;transition:.4s;border-color:' + color + '55' });
      c.innerHTML = `<div class="card-t" style="color:${color};font-size:12px">结果 #${n} 的边</div>
        <div class="mono" style="font-size:12px;color:#cfe0ff;margin:6px 0">${U.esc(res)}</div>
        <div class="small faint" style="font-size:11px">来自第 ${n} 个返回位置的要求</div>`;
      rows.appendChild(c); return c;
    };

    tl.at(700, () => {
      mk(0, '[{"a", ?}, {"b", ?}]', 'var(--ok)').style.opacity = '1';
      msg.innerHTML = '边 <span class="mono">#0</span> 对应第 0 个结果 —— 拿到 <span class="mono">[{"a", ?}, {"b", ?}]</span>。';
    });
    tl.at(4000, () => {
      mk(1, '[{"b", ?}, {?}]', 'var(--ax2)').style.opacity = '1';
      msg.innerHTML = '边 <span class="mono">#1</span> 对应第 1 个结果 —— 拿到 <span class="mono">[{"b", ?}, {?}]</span>。';
    });
    tl.at(7600, () => {
      rows.querySelectorAll('.card').forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>两条边完全不同</b>：一条两维都有轴，另一条只有第 0 维。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>为什么重要</b>：如果只能在算子上标一条属性，就无法表达"两个结果切法不同"。';
    });
    tl.at(13200, () => {
      msg.innerHTML = '<b>边的粒度</b>：结果是几元组，边就有几条；每条边独立参与传播。';
    });
  }
},

/* ------------------------------------------------ 5 manual + token */
{
  kicker: 'L2-08 · 数据流边',
  title: '两处补充：<span class="hl-a">手动计算内的边</span>、<span class="hl-a">token 被跳过</span>',
  sub: '`manual_computation` 的块参数上也能挂边；而 token 是 non-shaped 类型，传播<b>完全不处理</b>它。',
  caption: 'token 的规则与 L1-02 一致：无形状 → 分片无意义。测试里用 <span class="mono">_skipped</span> 后缀的用例专测这一点。',
  code: `// ① manual_computation 内的边
%0 = sdy.manual_computation(%arg0)
     in_shardings=[<@mesh, [{"b", ?}, {?}]>]
     out_shardings=[<@mesh, [{"b", ?}, {?}]>]
     manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
  %2 = sdy.data_flow_edge %arg1 sharding=<@mesh, [{?}, {?}]>
       : tensor<16x32xf32>
  %3 = stablehlo.add %2, %2
       {sdy.sharding = #sdy.sharding_per_value<
          [<@mesh, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>
  ...
}
// 体内边被更新为 [{"a", ?}, {"c", ?}]
//   "a" 是【自由轴】（传播处理）
//   "c" 来自 %arg0 的分片

// ② token 被跳过
func.func @func_token_arg_skipped(%arg0: !stablehlo.token)
    -> !stablehlo.token {
  return %arg0 : !stablehlo.token
}
// 期望：参数与返回值上【都没有】分片属性`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 手动计算内的边', c: '#c084fc',
        d: '块参数上的边从 <span class="mono">[{?}, {?}]</span> 更新为 <span class="mono">[{"a", ?}, {"c", ?}]</span>：<br>自由轴 <span class="mono">"a"</span> 由传播填充，<span class="mono">"c"</span> 继承自外层。' },
      { t: '② token 直接跳过', c: '#fbbf24',
        d: '<span class="mono">!stablehlo.token</span> 是 non-shaped 类型，<br>传播<b>不报错也不加属性</b>，静默跳过。' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:12px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '手动计算内部的<b>自由轴</b>仍归传播处理 —— 这与 L1-07 讲的一致。'; });
    tl.at(4400, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>同一次调用</b>里，token 结果被跳过、tensor 结果正常传播（用例 <span class="mono">@main</span>）。';
    });
    tl.at(8200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = 'token 文件用自己的<b>流水线</b>：<span class="mono">-sdy-add-data-flow-edges → -sdy-basic-propagate → -sdy-sink-data-flow-edges</span>。';
    });
    tl.at(11600, () => {
      msg.innerHTML = '即：<b>先插边 → 再传播 → 最后把边收回</b>（sink 会把边合并回算子）。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L2-08 · 练习',
  title: '练一练：<span class="hl-a">边在哪里发挥作用</span>',
  sub: '三道题分别考：边的位置、双向性、token 处理。',
  caption: '一句话总结：<b>区域算子的分片写在边（data_flow_edge）上，不在算子上</b>。',
  code: `// 题 1：为什么分片不能直接标在 stablehlo.while / case 上？

// 题 2：while 体内的算子带分片时，能影响到循环外的结果吗？

// 题 3：!stablehlo.token 在传播中会怎样？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '为什么分片不能直接标在 <span class="mono">stablehlo.while</span> / <span class="mono">case</span> 上？',
        a: '因为这些算子含<b>区域</b>，语义由<b>所有分支/块共同决定</b>，一条属性表达不了"哪个分支贡献了什么"。' +
           '<br>而且多结果时，每个结果的切法可能不同 —— 一条属性更说不清。' +
           '<br><span class="dim">解法：把分片放到显式的 <span class="mono">sdy.data_flow_edge</span> 上，一条结果一条边。</span>'
      },
      {
        q: '<span class="mono">while</span> 体内的算子带分片时，能影响到循环外的结果吗？',
        a: '<b class="badge ok">能</b>。数据流边是<b>双向</b>通道。' +
           '<br>用例 <span class="mono">while_body_propagate_block_arg</span> 就是反向传播：体内算子显式带 <span class="mono">[{"a"},{"b"}]</span>，推给块参数，再影响循环的输入与结果。' +
           '<br><span class="dim">而 <span class="mono">while_func_return</span> 是正向（其实是反向前推）：函数结果的锁定要求穿过 while 落到体内。</span>'
      },
      {
        q: '<span class="mono">!stablehlo.token</span> 在传播中会怎样？',
        a: '<b>被静默跳过</b>：不报错，也不添加任何分片属性。' +
           '<br><b>理由</b>：token 是 <b>non-shaped</b> 类型（L1-02），分片对它没有意义。' +
           '<br><span class="dim">同一次函数调用里，token 结果被跳过、tensor 结果正常传播 —— 两者互不影响。测试里 <span class="mono">*_skipped</span> 后缀的用例专测这一点。</span>'
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
