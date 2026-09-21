/* ==========================================================================
   L3-05 · add-func-data-flow-edges
   --------------------------------------------------------------------------
   覆盖：transforms/import/test/add_func_data_flow_edges.mlir (471 行 / 17 用例)
   目标：讲透【函数级】数据流边是怎么被插进来的。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 两个位置 */
{
  kicker: 'L3-05 · 函数级数据流边',
  title: '函数级边：<span class="hl-a">两个插入位置</span>',
  sub: 'L3-04 插的是**算子级**边。这一课插 **函数级**边（`sdy.func_data_flow_edge`）—— 桥接调用点与函数体。',
  caption: '回顾 L1-08 的模型：<b>调用点的实参 ↔ 函数体的形参</b>、<b>函数的 return ↔ 调用结果</b>。这两对都需要边。',
  code: `// RUN: sdy_opt %s -split-input-file -sdy-add-func-data-flow-edges

// 【位置 ①】函数参数（函数体内）
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.negate %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// 输出：
%[[EDGE]] = sdy.func_data_flow_edge %arg0
stablehlo.negate %[[EDGE]]          // 使用者改用边

// 【位置 ②】调用结果（调用点）
%[[CALL]] = call @bar(%[[ABS]])
%[[EDGE]] = sdy.func_data_flow_edge %[[CALL]]
stablehlo.abs %[[EDGE]]             // 下游改用边

// 两处合起来 = L1-08 说的"桥接实参与形参"`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:24px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ax0)">① 函数体内</div>
        <div class="row" style="gap:7px;align-items:center">
          <div class="chip mut" style="padding:6px 11px;font-size:11.5px">%arg0</div>
          <div style="color:var(--ok)">→</div>
          <div class="chip c4" style="padding:6px 11px;font-size:11.5px">edge</div>
          <div style="color:var(--ok)">→</div>
          <div class="chip mut" style="padding:6px 11px;font-size:11.5px">negate</div>
        </div>
        <div class="small faint">每个函数参数一条</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>函数参数</b>后面插边 —— 函数体的使用者改用边。';
    });
    tl.at(4400, () => {
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ax1)">② 调用点</div>
        <div class="row" style="gap:7px;align-items:center">
          <div class="chip mut" style="padding:6px 11px;font-size:11.5px">call</div>
          <div style="color:var(--ok)">→</div>
          <div class="chip c4" style="padding:6px 11px;font-size:11.5px">edge</div>
          <div style="color:var(--ok)">→</div>
          <div class="chip mut" style="padding:6px 11px;font-size:11.5px">abs</div>
        </div>
        <div class="small faint">每个调用结果一条</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>调用结果</b>后面插边 —— 下游改用边。';
    });
    tl.at(8600, () => {
      msg.innerHTML = '<b>两处合起来</b>才完整桥接了"调用点"与"函数体"两端（L1-08 的模型）。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>为什么函数级需要单独的边类型</b>：函数边界跨越了"编译单元"，普通算子级边表达不了。';
    });
  }
},

/* ------------------------------------------------ 2 多结果与多使用者 */
{
  kicker: 'L3-05 · 函数级数据流边',
  title: '多结果 / 多使用者：<span class="hl-a">与算子级边同一套规则</span>',
  sub: '每个结果一条边；多个使用者共用同一条边 —— 与 L3-04 的不变量完全一致。',
  caption: '这说明"边"的设计是<b>统一</b>的：无论算子级还是函数级，都是"一个值 ↔ 一条边"。',
  code: `// 多结果：call 有 2 个结果 -> 插 2 条边
%[[CALL]]:2 = call @bar(%[[ABS]])
%[[EDGE0]] = sdy.func_data_flow_edge %[[CALL]]#0
%[[EDGE1]] = sdy.func_data_flow_edge %[[CALL]]#1
%[[ADD]] = stablehlo.add %[[EDGE0]], %[[EDGE1]]
%[[ABS]] = stablehlo.abs %[[EDGE1]]
//                                    ^^^^^^^^^^ 同一个边被两处使用

// 函数体内：多结果函数只有一个参数
func.func @bar(%arg0: tensor<8xf32>) ->(tensor<8xf32>, tensor<8xf32>) {
  %[[EDGE]] = sdy.func_data_flow_edge %arg0
  stablehlo.negate %[[EDGE]]
  ...
}
// 注意：参数只有 1 个 -> 只插 1 条边
//       结果有 2 个 -> 调用点插 2 条边`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '多结果', c: '#38bdf8',
        d: '<span class="mono">call</span> 有 2 个结果<br>→ 插 <b>2 条边</b><br>（与 L3-04 同规则）' },
      { t: '多使用者', c: '#4ade80',
        d: '<span class="mono">%[[EDGE1]]</span> 被 <span class="mono">add</span> 与 <span class="mono">abs</span> 共用<br>→ <b>只插一条</b>，共享' },
      { t: '参数 vs 结果', c: '#fbbf24',
        d: '函数体插边的数量取决于<b>参数个数</b>；<br>调用点取决于<b>结果个数</b> —— 两者独立' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '判据是「值的个数」，不是「有几个被使用」（L3-04 已讲过这条）。',
        '<b>边独占结果</b>：所有使用者都通过同一条边访问 —— 保证"一个值 ↔ 一条边"。',
        '同一个函数里，参数边与结果边是<b>两组独立</b>的边，各管一段数据流。',
      ][i];
    }));
    tl.at(12200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>统一的设计</b>：无论算子级还是函数级，"边"的规则完全一致。';
    });
  }
},

/* ---------------------------------------------------- 3 链式调用 */
{
  kicker: 'L3-05 · 函数级数据流边',
  title: '链式调用：<span class="hl-a">两种边共存</span>',
  sub: '`main → bar → foo` 时，中间的 `@bar` 里**同时**有参数边与调用结果边。',
  caption: '它们互相独立 —— 各自服务自己那一段数据流。',
  code: `// @bar 内部
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @foo(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// 输出：
%[[EDGE0]] = sdy.func_data_flow_edge %arg0        // 参数边
%[[ABS0]] = stablehlo.abs %[[EDGE0]]
%[[CALL]] = call @foo(%[[ABS0]])
%[[EDGE1]] = sdy.func_data_flow_edge %[[CALL]]    // 调用结果边
%[[ABS1]] = stablehlo.abs %[[EDGE1]]

// 读法：
//   EDGE0 服务 "%arg0 进入函数体" 这一段
//   EDGE1 服务 "call 结果离开调用点" 这一段
//   两者没有关系，各自独立

// 完整调用图 main -> bar -> foo：
//   @main  里有 call @bar 的结果边
//   @bar   里有 %arg0 的参数边 + call @foo 的结果边
//   @foo   里有 %arg0 的参数边`,
  duration: 15000,
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
        <div class="small mono" style="color:var(--ax0)">@bar 内部的两段</div>
        <div class="row" style="gap:7px;align-items:center">
          <div class="chip mut" style="padding:6px 11px;font-size:11.5px">%arg0</div>
          <div style="color:var(--ok)">→</div>
          <div class="chip c4" style="padding:6px 11px;font-size:11.5px">EDGE0</div>
          <div style="color:var(--ink-faint)">→</div>
          <div class="chip mut" style="padding:6px 11px;font-size:11.5px">abs</div>
          <div style="color:var(--ink-faint)">→</div>
          <div class="chip mut" style="padding:6px 11px;font-size:11.5px">call</div>
          <div style="color:var(--ok)">→</div>
          <div class="chip c4" style="padding:6px 11px;font-size:11.5px">EDGE1</div>
        </div>
        <div class="small faint">参数边在前，调用结果边在后</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<span class="mono">@bar</span> 里有两段数据流，各有一条边。';
    });
    tl.at(4400, () => {
      msg.innerHTML = '<b>EDGE0</b> 服务"<span class="mono">%arg0</span> 进入函数体"；<b>EDGE1</b> 服务"<span class="mono">call</span> 结果离开调用点"。';
    });
    tl.at(7800, () => {
      msg.innerHTML = '<b>两者独立</b>：修改其中一条不影响另一条 —— 它们桥接的是不同的边界。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>整个调用图</b>：每一条"调用边"两端都各有一条边，形成完整的桥接链。';
    });
    tl.at(13200, () => {
      msg.innerHTML = '这也解释了为什么需要 <span class="mono">-sdy-import-func-calls</span>（L3-06）—— 桥接链最终要被内联掉。';
    });
  }
},

/* ---------------------------------------------------- 4 token 跳过 */
{
   kicker: 'L3-05 · 函数级数据流边',
  title: 'token <span class="hl-a">跳过</span>（测试注释写明了理由）',
  sub: '两个用例分别覆盖"token 作为参数"与"token 作为调用结果"。',
  caption: '理由与 L3-04 完全一致：token 不是静态形状类型，边对它没有意义。',
  code: `// ① token 作为函数参数
// 测试注释原文：
//   Tokens are not static-shaped types, so no func_data_flow_edge should be
//   created for them. Only the tensor argument gets an edge op.
func.func @bar(%arg0: !stablehlo.token, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %arg1
  // CHECK-NEXT:                stablehlo.negate %[[EDGE]]
  // CHECK-NOT:  sdy.func_data_flow_edge %arg0
  %0 = stablehlo.negate %arg1 : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// ② token 作为调用结果
%[[CALL]]:2 = call @bar(%arg0)
%[[EDGE]] = sdy.func_data_flow_edge %[[CALL]]#1
// CHECK-NOT:  sdy.func_data_flow_edge %[[CALL]]#0
stablehlo.abs %[[EDGE]]

// 注意两处都用 CHECK-NOT 精确断言"没有边"`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① token 参数', c: '#fbbf24',
        d: '<span class="mono">@bar(%arg0: token, %arg1: tensor)</span><br>→ <span class="mono">%arg1</span> 有边<br>→ <span class="mono">%arg0</span> <b>没有</b>边' },
      { t: '② token 结果', c: '#fb7185',
        d: '<span class="mono">call</span> 返回 <span class="mono">(token, tensor)</span><br>→ <span class="mono">#1</span> 有边<br>→ <span class="mono">#0</span> <b>没有</b>边' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>同一个函数的两个参数</b>：一个插边、一个不插。'; });
    tl.at(4400, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>同一次调用的两个结果</b>：一个插边、一个不插。';
    });
    tl.at(8000, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>测试用 <span class="mono">CHECK-NOT</span> 精确断言"没有边"</b> —— 这类"不该发生"的行为同样需要锁定。';
    });
    tl.at(11200, () => {
      msg.innerHTML = '<b>理由与 L3-04 一致</b>：token 不是静态形状类型（L1-02 / L1-08）。';
    });
  }
},

/* ---------------------------------------------------- 5 三种边对照 */
{
  kicker: 'L3-05 · 函数级数据流边',
  title: '三种边的<span class="hl-a">对照</span>',
  sub: '到这里 SDY 的三种"桥接结构"都见过了 —— 它们解决的问题不同。',
  caption: '这张表可以作为后面课程的索引：遇到"跨边界"的问题，先想是哪一种边。',
  code: `// ① sdy.data_flow_edge       —— 算子级
//   出现在：区域算子（while / case / barrier / named / manual）旁
//   桥接：  区域算子的结果 <-> 下游
//   讲在：  L1-08 / L2-08 / L3-04

// ② sdy.func_data_flow_edge  —— 函数级
//   出现在：函数参数 / 调用结果
//   桥接：  调用点的实参 <-> 函数体的形参
//   讲在：  L1-08 / 本课

// ③ sdy.named_computation    —— 整个函数体
//   出现在：函数调用点（内联后）
//   桥接：  把函数体搬进调用者（消除边界）
//   讲在：  L1-08 / L3-06

// 三者的关系：
//   ① 处理"区域"边界（算子内部的 region）
//   ② 处理"函数"边界（调用点与函数体之间）
//   ③ 干脆【消除】函数边界 —— 把调用内联掉

// 为什么 ③ 之后还需要 ②？
//   因为内联是导入期的选择；不内联时（或内联前）仍需 ② 来表达跨函数分片`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① data_flow_edge', c: '#38bdf8', tag: '算子级',
        d: '区域算子旁<br>桥接<b>区域</b>边界' },
      { t: '② func_data_flow_edge', c: '#c084fc', tag: '函数级',
        d: '函数参数 / 调用结果<br>桥接<b>函数</b>边界' },
      { t: '③ named_computation', c: '#4ade80', tag: '消除边界',
        d: '把函数体搬进调用者<br><b>消除</b>函数边界' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:11px;overflow-wrap:anywhere">${x.t}</div>
        <div class="small mono" style="font-size:10px;color:${x.c};margin:4px 0">${x.tag}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        'L3-04 讲的。区域是"算子内部的小世界"，它的入口出口需要边。',
        '<b>本课讲的</b>。函数边界跨越了编译单元，普通算子级边表达不了。',
        'L3-06 讲的。<b>干脆不跨边界</b> —— 把被调函数搬进调用者。',
      ][i];
    }));
    tl.at(12200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>为什么内联之后还需要 ②</b>：内联是导入期的<b>选择</b>；不内联时仍需 ② 表达跨函数分片。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L3-05 · 练习',
  title: '练一练：<span class="hl-a">插几条边、插在哪</span>',
  sub: '三道题分别考：插入位置、数量判据、token 规则。',
  caption: '一句话总结：<b>函数参数一条、调用结果一条；token 跳过</b>。',
  code: `// 题 1：函数级边插在哪两个位置？

// 题 2：一个函数有 2 个参数、返回 3 个结果，
//       函数体内插几条边？调用点插几条？

// 题 3：token 参数 / token 结果会被插边吗？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '函数级边插在哪<b>两个位置</b>？',
        a: '① <b>函数参数</b>（函数体内）—— 每个参数一条边，函数体的使用者改用边。' +
           '<br>② <b>调用结果</b>（调用点）—— 每个结果一条边，下游改用边。' +
           '<br><span class="dim">两处合起来才完整桥接了 L1-08 说的两对关系：<b>调用实参 ↔ 函数形参</b>、<b>函数 return ↔ 调用结果</b>。</span>'
      },
      {
        q: '一个函数有 <b>2 个参数</b>、返回 <b>3 个结果</b>，函数体内插几条边？调用点插几条？',
        a: '<b>函数体内 2 条</b>（每个参数一条）；<b>调用点 3 条</b>（每个结果一条）。' +
           '<br><b>两者独立</b>：函数体插边的数量取决于<b>参数个数</b>，调用点取决于<b>结果个数</b>。' +
           '<br><span class="dim">判据始终是「值的个数」，不是「有几个被使用」—— 与 L3-04 的不变量一致。</span>'
      },
      {
        q: 'token 参数 / token 结果会被插边吗？',
        a: '<b class="badge bad">都不会</b>。' +
           '<br><b>测试注释原文</b>：<span class="mono">Tokens are not static-shaped types, so no func_data_flow_edge should be created for them.</span>' +
           '<br><span class="dim">两个用例分别覆盖 token 作参数、token 作结果，并用 <span class="mono">CHECK-NOT</span> 精确断言"没有边"。理由与 L3-04 的 token 跳过规则完全一致。</span>'
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
