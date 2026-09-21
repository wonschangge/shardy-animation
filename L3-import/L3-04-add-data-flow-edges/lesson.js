/* ==========================================================================
   L3-04 · add-data-flow-edges
   --------------------------------------------------------------------------
   覆盖：transforms/import/test/add_data_flow_edges.mlir (253 行 / 16 用例)
   目标：讲透数据流边是【怎么被插进来的】（L2-08 讲的是它在传播中的行为）。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 回顾与问题 */
{
  kicker: 'L3-04 · 插入数据流边',
  title: '边是<span class="hl-a">怎么被插进来的</span>',
  sub: 'L2-08 讲的是数据流边在**传播中**的行为。这一课看它们在**导入期**是怎么被创建的。',
  caption: '回顾 L2-08 的结论：区域算子（<span class="mono">while</span> / <span class="mono">case</span> / <span class="mono">barrier</span> / <span class="mono">named_computation</span> / <span class="mono">manual_computation</span>）<b>不能携带分片</b>。',
  code: `// RUN: sdy_opt %s -sdy-add-data-flow-edges -split-input-file

// 回顾：区域算子上【没有】分片属性
//   分片必须写在紧随其后的 sdy.data_flow_edge 上

// 这一课回答：
//   ① 哪些位置需要插边？
//      - 区域算子的【每个结果】（区域外）
//      - 区域内的【每个块参数】
//   ② 什么情况【不】插？
//      - token（non-shaped 类型）
//      - 动态形状（tensor<?x?xf32>）
//   ③ 插出来的边带不带分片？
//      - 一般不带（等传播填）
//      - manual_computation 的边【带】in/out_shardings 的分片`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ax0)">区域外（结果）</div>
        <div class="row" style="gap:7px;align-items:center">
          <div class="chip mut" style="padding:6px 11px;font-size:12px">while</div>
          <div style="color:var(--ok)">→</div>
          <div class="chip c4" style="padding:6px 11px;font-size:12px">edge</div>
          <div style="color:var(--ok)">→</div>
          <div class="chip mut" style="padding:6px 11px;font-size:12px">下游</div>
        </div>
        <div class="small faint">每个结果一条</div>`;
      demo.appendChild(c);
      msg.innerHTML = '区域算子的<b>每个结果</b>后面插一条边，下游改为使用边。';
    });
    tl.at(4400, () => {
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ax1)">区域内（块参数）</div>
        <div class="row" style="gap:7px;align-items:center">
          <div class="chip mut" style="padding:6px 11px;font-size:12px">block arg</div>
          <div style="color:var(--ok)">→</div>
          <div class="chip c4" style="padding:6px 11px;font-size:12px">edge</div>
          <div style="color:var(--ok)">→</div>
          <div class="chip mut" style="padding:6px 11px;font-size:12px">区域内算子</div>
        </div>
        <div class="small faint">每个块参数一条</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>区域内</b>也要插：块参数是区域的入口，同样需要能表达分片。';
    });
    tl.at(8600, () => {
      msg.innerHTML = '<b>两边都插</b>正是 L2-08 说的"边是双向通道"的<b>结构基础</b>。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '本课 16 个用例覆盖了插边的各种情形与两条跳过规则。';
    });
  }
},

/* ---------------------------------------------------- 2 每个结果一条 */
{
  kicker: 'L3-04 · 插入数据流边',
  title: '★ 核心规则：<span class="hl-a">每个结果一条边</span>',
  sub: '判据是「结果的**个数**」，不是「有几个被使用」—— 没被使用的结果也插。',
  caption: '这保持了"每个结果恰好一条边"的不变量，避免结构因使用情况而变形。',
  code: `// case 有 2 个结果 -> 插 2 条边
%0:2 = "stablehlo.case"(%arg0) ({...}, {...})
       : (tensor<i32>) -> (tensor<8xi64>, tensor<8xi64>)
%1 = stablehlo.add %0#1, %0#1
return %0#0, %1

// 输出：
%[[CASE]]:2 = "stablehlo.case"(%arg0)
%[[EDGE_1]] = sdy.data_flow_edge %[[CASE]]#0 : tensor<8xi64>
%[[EDGE_2]] = sdy.data_flow_edge %[[CASE]]#1 : tensor<8xi64>
%[[ADD]] = stablehlo.add %[[EDGE_2]], %[[EDGE_2]]     // 改用边
return %[[EDGE_1]], %[[ADD]]                          // 改用边

// while：结果 #1 没人用，【照样】插边
%[[WHILE]]:2 = stablehlo.while
%[[EDGE_1]] = sdy.data_flow_edge %[[WHILE]]#0
%[[EDGE_2]] = sdy.data_flow_edge %[[WHILE]]#1         // 死边，但仍在
return %[[EDGE_1]]`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:14px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:8px;align-items:center' });
      c.innerHTML = `
        <div class="small mono faint">2 个结果</div>
        <div class="row" style="gap:8px;align-items:center">
          <div class="chip mut" style="padding:7px 12px;font-size:12px">#0 被 return</div>
          <div class="chip mut" style="padding:7px 12px;font-size:12px">#1 被 add</div>
        </div>`;
      demo.appendChild(c);
      msg.innerHTML = '<span class="mono">case</span> 的两个结果都被使用了。';
    });
    tl.at(3800, () => {
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹', style: 'font-size:24px' }));
      const c = U.el('div', { class: 'col', style: 'gap:8px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ok)">2 条边</div>
        <div class="row" style="gap:8px;align-items:center">
          <div class="chip c4" style="padding:7px 12px;font-size:12px">edge %CASE#0</div>
          <div class="chip c4" style="padding:7px 12px;font-size:12px">edge %CASE#1</div>
        </div>
        <div class="small faint">下游全部改用边</div>`;
      demo.appendChild(c);
      msg.innerHTML = '每个结果一条边，且<b>下游全部改用边</b> —— 边成为结果的唯一消费者。';
    });
    tl.at(7800, () => {
      msg.innerHTML = '<b>为什么让边独占结果</b>：保证"一个结果 ↔ 一条边"的对应关系，不会歧义。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>while 的情形更说明问题</b>：结果 <span class="mono">#1</span>（循环计数器）没人用，但<b>照样插边</b>。';
    });
    tl.at(13400, () => {
      msg.innerHTML = '判据是「结果的<b>个数</b>」，不是「有几个被使用」—— 结构不因使用情况而变形。';
    });
  }
},

/* ---------------------------------------------------- 3 两条跳过规则 */
{
  kicker: 'L3-04 · 插入数据流边',
  title: '两条<span class="hl-a">跳过</span>规则',
  sub: 'token 与动态形状都**不插边** —— 理由不同，但结论一致。',
  caption: '两条规则都能在 L1-08 找到依据：边要求<b>静态形状</b>，token 是 <b>non-shaped</b> 类型。',
  code: `// ① 跳过 token
func.func @named_computation_skip_tokens(%arg0: tensor<8x2xi32>,
    %arg1: !stablehlo.token) -> (tensor<8x2xi32>, !stablehlo.token) {
  %0:2 = sdy.named_computation<"foo">(%arg0, %arg1)
         (%arg2: tensor<8x2xi32>, %arg3: !stablehlo.token) {
    sdy.return %arg2, %arg3 : tensor<8x2xi32>, !stablehlo.token
  } : ...
  return %0#0, %0#1 : tensor<8x2xi32>, !stablehlo.token
}
// 输出：%arg2 有边，%arg3（token）没有
//   %[[EDGE_1]] = sdy.data_flow_edge %arg2 : tensor<8x2xi32>
//   sdy.return %[[EDGE_1]], %arg3              <- token 原样返回
//   return %[[EDGE_2]], %[[NC]]#1              <- token 不走边

// ② 跳过动态形状
%0:2 = stablehlo.optimization_barrier %arg0, %arg1
       : tensor<32x96xf32>, tensor<?x?xf32>
return %0#0, %0#1
// 输出：只有 #0（静态）有边
//   %[[EDGE_1]] = sdy.data_flow_edge %[[OPT_BARRIER]]#0
//   return %[[EDGE_1]], %[[OPT_BARRIER]]#1     <- 动态形状直接返回`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① token 跳过', c: '#fbbf24', tag: 'non-shaped',
        d: '<span class="mono">!stablehlo.token</span> 没有形状<br>→ 分片对它没有意义<br>→ 不插边，原样传递' },
      { t: '② 动态形状跳过', c: '#fb7185', tag: 'tensor&lt;?x?xf32&gt;',
        d: '边<b>要求静态形状</b>（L1-08）<br>→ 动态形状无法确定分片<br>→ 不插边，直接返回' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="small mono" style="font-size:10.5px;color:${x.c};margin:4px 0">${x.tag}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>token 的情形</b>：同一次调用的两个参数，一个插边、一个不插。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>动态形状的情形</b>：同一个算子的两个结果，静态的插边、动态的不插。';
    });
    tl.at(8400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>共同点</b>：都是"分片无从表达"的情形 —— 一个是没有形状，一个是形状不固定。';
    });
    tl.at(11600, () => {
      msg.innerHTML = '<b>注意</b>：跳过的值<b>直接连到下游</b>（如 <span class="mono">return %[[NC]]#1</span>），不经过边。';
    });
  }
},

/* ------------------------------------------------ 4 manual 的边带分片 */
{
  kicker: 'L3-04 · 插入数据流边',
  title: '★ <span class="mono hl-a">manual_computation</span> 的边<span class="hl-a">带分片</span>',
  sub: '其它算子的边是"空的"（等传播填）；但手动计算的边**携带 in/out_shardings 的分片**。',
  caption: '原因：区域算子本身不带分片属性，<span class="mono">in/out_shardings</span> 的信息<b>只能</b>通过边来表达。',
  code: `%0:2 = sdy.manual_computation(%arg0, %arg1)
    in_shardings=[<@mesh, [{"a", ?}, {?}], replicated={"b"}>,
                  <@mesh, [{"b", ?}, {?}]>]
    out_shardings=[<@mesh, [{"a", ?}, {?}], replicated={"b"}>,
                   <@mesh, [{"b", ?}, {?}]>]
    manual_axes={"b"} (%arg2: tensor<8x2xi32>, %arg3: tensor<2x2xi32>) {
  sdy.return %arg2, %arg3 : tensor<8x2xi32>, tensor<2x2xi32>
} : ...

// 输出（区域内）：
%[[EDGE_1]] = sdy.data_flow_edge %arg2 sharding=<@mesh, [{"a", ?}, {?}]>
%[[EDGE_2]] = sdy.data_flow_edge %arg3 sharding=<@mesh, [{?}, {?}]>
// 输出（区域外）：
%[[EDGE_3]] = sdy.data_flow_edge %[[MC]]#0
              sharding=<@mesh, [{"a", ?}, {?}], replicated={"b"}>
%[[EDGE_4]] = sdy.data_flow_edge %[[MC]]#1 sharding=<@mesh, [{"b", ?}, {?}]>

// 逐条读：
//   区域内 %arg2 <- in_shardings[0]，但【去掉】了 replicated={"b"}
//   区域内 %arg3 <- in_shardings[1] 的【局部形状】版本（tensor<2x2xi32>）
//   区域外 #0   <- out_shardings[0] 完整（含 replicated）
//   区域外 #1   <- out_shardings[1]`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '区域内 %arg2', c: '#38bdf8',
        d: '来自 <span class="mono">in_shardings[0]</span><br><b>去掉</b> <span class="mono">replicated={"b"}</span><br>（区域内不再有 manual 轴标注）' },
      { t: '区域内 %arg3', c: '#c084fc',
        d: '来自 <span class="mono">in_shardings[1]</span><br>类型是 <span class="mono">tensor&lt;2x2xi32&gt;</span><br>（切过 manual 轴后的<b>局部形状</b>）' },
      { t: '区域外 #0', c: '#4ade80',
        d: '来自 <span class="mono">out_shardings[0]</span><br><b>完整</b>包含 <span class="mono">replicated={"b"}</span>' },
      { t: '区域外 #1', c: '#fbbf24',
        d: '来自 <span class="mono">out_shardings[1]</span><br>与声明一致' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:180px;opacity:.33;transition:.35s;border-color:' + x.c + '55;padding:9px' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:11.5px">${x.t}</div>
        <div class="card-d" style="font-size:10.5px;line-height:1.5">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>注意这个细节</b>：区域内的值不再带 manual 轴的复制标注 —— 因为区域内已经是"局部世界"。',
        '局部形状与 L1-07 的规则一致：<span class="mono">4</span> 沿 manual 轴 <span class="mono">b=2</span> 切成 <span class="mono">2</span>。',
        '区域外恢复成完整形状，所以 <span class="mono">replicated={"b"}</span> 又出现了。',
        '<b>核心原因</b>：区域算子不能带分片属性，<span class="mono">in/out_shardings</span> 的信息<b>只能</b>通过边表达。',
      ][i];
    }));
    tl.at(15200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>对比</b>：其它算子（case / while / barrier）的边是<b>空的</b>，等传播去填。';
    });
  }
},

/* ---------------------------------------------------- 5 族谱 */
{
  kicker: 'L3-04 · 插入数据流边',
  title: '16 个用例的<span class="hl-a">族谱</span>',
  sub: '按区域算子类型分组 —— 每个算子族都有自己的边界情形。',
  caption: '读法：抓住"每个结果一条边 + 两条跳过规则 + manual 带分片"三点即可。',
  code: `// 【case】3 个
//   case / case_existing_sharding / case_token_result_skipped

// 【optimization_barrier】2 个
//   optimization_barrier
//   optimization_barrier_dynamic_shaped_tensor_skipped

// 【while】1 个
//   while_unused_result                     未使用的结果也插边

// 【named_computation】5 个
//   named_computation_multiple_inputs_outputs
//   named_computation_with_shardings
//   named_computation_ops_inside_and_outside
//   named_computation_unused_result
//   named_computation_skip_tokens

// 【manual_computation】3 个
//   manual_computation_skip_tokens
//   manual_computation_multiple_inputs_outputs
//   manual_computation_user_priority

// 【综合】1 个
//   main`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: 'case', n: 3, c: '#38bdf8' }, { t: 'barrier', n: 2, c: '#fbbf24' },
      { t: 'while', n: 1, c: '#4ade80' }, { t: 'named_computation', n: 5, c: '#c084fc' },
      { t: 'manual_computation', n: 3, c: '#fb7185' }, { t: '综合', n: 1, c: '#94a3b8' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:132px;opacity:.4;transition:.35s;border-color:${f.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="card-t mono" style="color:${f.c};font-size:10.5px;overflow-wrap:anywhere">${f.t}</div>
        <div class="big" style="font-size:20px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 120)); msg.innerHTML = '<b>16 个用例</b>按区域算子分 6 组 —— named_computation 最多（5 个）。'; });
    tl.at(4000, () => {
      els.forEach((e, i) => { if (i !== 3 && i !== 4) e.style.opacity = '.25'; });
      msg.innerHTML = '这两个是<b>带区域且带参数</b>的算子 —— 边界最复杂（内外两侧都要插边）。';
    });
    tl.at(7400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>三点总结</b>：每个结果一条边 · token/动态形状跳过 · manual 的边带分片。';
    });
    tl.at(10600, () => {
      msg.innerHTML = '<b>下一课</b> L3-05 讲函数级的数据流边（<span class="mono">func_data_flow_edge</span>）。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L3-04 · 练习',
  title: '练一练：<span class="hl-a">插几条边</span>',
  sub: '三道题分别考：结果个数、跳过规则、manual 的边。',
  caption: '一句话总结：<b>每个结果一条边；token 与动态形状跳过；manual 的边带分片</b>。',
  code: `// 题 1：一个有 3 个结果的 case，插几条边？
//       其中 1 个结果没被使用，有影响吗？

// 题 2：stablehlo.optimization_barrier 的两个结果分别是
//       tensor<32x96xf32> 与 tensor<?x?xf32>，各插边吗？

// 题 3：manual_computation 的边为什么带分片？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '一个有 <b>3 个结果</b>的 <span class="mono">case</span>，插几条边？其中 1 个结果没被使用，有影响吗？',
        a: '<b>3 条</b>。没被使用<b>不影响</b> —— 判据是「结果的<b>个数</b>」，不是「有几个被使用」。' +
           '<br>用例 <span class="mono">while_unused_result</span> 就是这种情形：<span class="mono">while</span> 的第 1 个结果（循环计数器）没人用，但<b>照样插边</b>。' +
           '<br><span class="dim">这保持了"每个结果恰好一条边"的不变量，避免结构因使用情况而变形。</span>'
      },
      {
        q: '<span class="mono">optimization_barrier</span> 的两个结果分别是 <span class="mono">tensor&lt;32x96xf32&gt;</span> 与 <span class="mono">tensor&lt;?x?xf32&gt;</span>，各插边吗？',
        a: '<b>静态的插，动态的不插</b>。' +
           '<br><span class="mono">#0</span>（<span class="mono">tensor&lt;32x96xf32&gt;</span>）→ 有边；<span class="mono">#1</span>（<span class="mono">tensor&lt;?x?xf32&gt;</span>）→ 直接返回，无边。' +
           '<br><span class="dim">依据来自 L1-08：数据流边<b>要求静态形状</b>（<span class="mono">expected sdy.data_flow_edge to have a static-shaped result</span>）。</span>'
      },
      {
        q: '<span class="mono">manual_computation</span> 的边为什么<b>带分片</b>？',
        a: '因为区域算子本身<b>不能携带分片属性</b>，<span class="mono">in/out_shardings</span> 的信息<b>只能</b>通过边来表达。' +
           '<br><b>两个细节</b>：区域内的边来自 <span class="mono">in_shardings</span> 且<b>去掉</b> <span class="mono">replicated</span>（区域内是局部世界）；区域外的边完整来自 <span class="mono">out_shardings</span>。' +
           '<br><span class="dim">对比：其它算子（case / while / barrier）的边是<b>空的</b>，等传播去填。</span>'
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
