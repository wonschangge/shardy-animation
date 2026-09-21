/* ==========================================================================
   L3-02 · constant-splitter   （全计划最大文件：1782 行 / 95 用例）
   --------------------------------------------------------------------------
   覆盖：transforms/import/test/constant_or_scalar_splitter.mlir
   目标：讲透"为什么要拆常量"——消除假依赖，让每个使用独立分片。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 为什么拆 */
{
  kicker: 'L3-02 · 常量拆分',
  title: '为什么要拆？<span class="hl-a">假依赖</span>',
  sub: '一个常量被多处使用时，这些使用方**被迫**采用同一种分片 —— 哪怕它们本来毫无关系。',
  caption: '这就是 L1-10 讲的「<span class="mono">sdy.constant</span> 刻意不带 folder」要配合的动作：<b>不合并，才能拆开</b>。',
  code: `// 一个常量，三个使用
%0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
%1 = stablehlo.dot_general %0, %arg0, contracting_dims = [1] x [0]
%2 = stablehlo.add %0, %1
return %0, %2

// 问题：三个使用对分片的需求可能完全不同
//   dot_general  -> 希望沿收缩维切
//   add          -> 希望沿另一维切
//   return       -> 可能要求复制（不切）
//
// 但它们是【同一个值】-> 传播必须给一个统一答案
//   -> 至少两处需要 reshard（额外通信）
//   -> 这就是"假依赖"：数据上没有依赖，分片上被强行绑定

// 解法：复制成三份
%[[CONST_0]] = sdy.constant dense<1.…>   // 给 dot
%[[CONST_1]] = sdy.constant dense<1.…>   // 给 add
%[[CONST_2]] = sdy.constant dense<1.…>   // 给 return
// 每份可以独立分片 -> 假依赖消失`,
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
        <div class="small mono" style="color:var(--bad)">✗ 共用一份</div>
        <div class="chip mut" style="padding:8px 14px">1 个 constant</div>
        <div class="row" style="gap:8px;margin-top:3px">
          <div class="chip c0" style="padding:5px 10px;font-size:11.5px">dot</div>
          <div class="chip c1" style="padding:5px 10px;font-size:11.5px">add</div>
          <div class="chip c2" style="padding:5px 10px;font-size:11.5px">return</div>
        </div>
        <div class="small" style="color:var(--bad);font-size:11px">被迫同分片 → reshard</div>`;
      demo.appendChild(c);
      msg.innerHTML = '三个使用方<b>数据上毫无关系</b>，但因为共用同一个值，分片被强行绑定。';
    });
    tl.at(4600, () => {
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹', style: 'font-size:26px' }));
      const c = U.el('div', { class: 'col', style: 'gap:8px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ok)">✓ 复制三份</div>
        <div class="row" style="gap:6px">
          <div class="chip c0" style="padding:5px 10px;font-size:11px">const → dot</div>
          <div class="chip c1" style="padding:5px 10px;font-size:11px">const → add</div>
          <div class="chip c2" style="padding:5px 10px;font-size:11px">const → ret</div>
        </div>
        <div class="small" style="color:var(--ok);font-size:11px">各自分片 → 无假依赖</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>复制三份</b>后，每份可以按自己使用方的需要分片。';
    });
    tl.at(8800, () => {
      msg.innerHTML = '<b>代价</b>：IR 变长（三份常量）。<b>收益</b>：消除不必要的通信 —— 后者重要得多。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>与 L1-10 的呼应</b>：正因为 <span class="mono">sdy.constant</span> 不带 folder，复制出来的三份才不会被重写器合并回去。';
    });
    tl.at(14200, () => {
      msg.innerHTML = '这个文件有 <b>95 个用例</b> —— 因为"什么情况该拆"的组合非常多。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 核心规则 */
{
  kicker: 'L3-02 · 常量拆分',
  title: '★ 核心规则：<span class="hl-a">N 个使用 → N 份</span>',
  sub: '判据是「**使用次数**」，不是「使用者个数」—— 同一个算子用了两次也照样拆。',
  caption: '为什么？因为 <span class="mono">dot</span> 的左右两侧本来就需要不同的切法 —— 共用一份会互相牵制。',
  code: `// 输入：1 个常量，3 个使用
%0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
%1 = stablehlo.dot_general %0, %arg0, contracting_dims = [1] x [0]
%2 = stablehlo.add %0, %1
return %0, %2

// 输出：3 份
%[[CONST_0]] = sdy.constant dense<1.000000e+00>
%[[CONST_1]] = sdy.constant dense<1.000000e+00>
%[[CONST_2]] = sdy.constant dense<1.000000e+00>
%[[DOT]] = stablehlo.dot_general %[[CONST_0]], %arg0
%[[ADD]] = stablehlo.add %[[CONST_1]], %[[DOT]]
return %[[CONST_2]], %[[ADD]]

// 同一个算子用两次（dot_general %0, %0）-> 仍然拆两份
%[[DOT]] = stablehlo.dot_general %[[CONST_0]], %[[CONST_1]]
//                                         ^^^^^^^^^^  ^^^^^^^^^^
//                                         左右各一份

// 顺带：stablehlo.constant 被换成 sdy.constant`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'CONST_0', d: '给 <span class="mono">dot_general</span> 的<b>左</b>操作数', c: '#38bdf8' },
      { t: 'CONST_1', d: '给 <span class="mono">add</span>（同一算子的右侧情形也一样）', c: '#c084fc' },
      { t: 'CONST_2', d: '给 <span class="mono">return</span> —— 可能要求复制', c: '#fbbf24' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:242px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '同一个常量在不同位置，分片需求可以完全不同。',
        '<b>关键细节</b>：<span class="mono">dot_general %0, %0</span> 这种"同算子用两次"也拆 —— 因为左右两侧本就该分开决策。',
        '<span class="mono">return</span> 处往往要求"不切"（复制），与算子里"切得越细越好"正好相反。',
      ][i];
    }));
    tl.at(11600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>一句话</b>：有多少个"使用位置"，就复制多少份 —— 让每个位置独立决策。';
    });
  }
},

/* ------------------------------------------------ 3 什么不拆 */
{
  kicker: 'L3-02 · 常量拆分',
  title: '什么情况<span class="hl-a">不拆</span>',
  sub: '复制有代价（IR 变长）。没有收益时就不拆 —— 判据是「**复制后能否获得分片自由度**」。',
  caption: '三种不拆：单使用者（无假依赖）、标量（无维度可分）、非标量输入的 broadcast（复制不划算）。',
  code: `// ① 单使用者 -> 不拆
%0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
%1 = stablehlo.add %arg0, %0
// 输出：仍然只有 1 份 CONST
//   没有别的使用者 -> 不存在假依赖 -> 复制无收益

// ② 标量（rank 0）-> 不拆
%0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
%1 = stablehlo.add %0, %arg0
return %0, %1
// 输出：仍然只有 1 份 CONST
//   rank-0 没有维度可分片 -> 复制不带来任何分片自由度

// ③ 非标量输入的 broadcast -> 不拆
%0 = stablehlo.broadcast_in_dim %arg0, dims = [0]
     : (tensor<2xf32>) -> tensor<2x64xf32>
%1 = stablehlo.negate %0
%2 = stablehlo.abs %0
// 输出：broadcast 只有一条`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 单使用者', c: '#4ade80',
        d: '只有一个使用位置<br>→ 不存在假依赖<br>→ <b>复制无收益</b>' },
      { t: '② 标量 rank-0', c: '#38bdf8',
        d: '没有维度可分片<br>→ 复制不带来分片自由度<br>→ <b>只是让 IR 变长</b>' },
      { t: '③ 非标量 broadcast', c: '#fbbf24',
        d: '输入 <span class="mono">tensor&lt;2xf32&gt;</span> 非标量<br>→ 不套用标量 broadcast 规则' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:242px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '最直观的一条：只有一个使用者时，没有"两个使用方互相牵制"的问题。',
        '<b>核心洞察</b>：拆分的收益来自「每份可以有<b>不同分片</b>」。没有维度就没有不同分片可言。',
        '标量 broadcast 的规则有<b>前提条件</b>：输入必须是 rank 0。',
      ][i];
    }));
    tl.at(12200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>统一判据</b>：复制后能否获得分片自由度？能就拆，不能就不拆。';
    });
  }
},

/* ------------------------------------------------ 4 标量 broadcast */
{
  kicker: 'L3-02 · 常量拆分',
  title: '★ 最巧妙的一处：拆 <span class="hl-a">broadcast</span> 而不是标量',
  sub: '标量本身不分片（不拆），但它的**broadcast 结果**可以分片 —— 于是复制 broadcast 算子。',
  caption: '这就是文件名的由来：<span class="mono">constant_<b>or_scalar</b>_splitter</span> —— 拆常量，<b>或者</b>拆标量的 broadcast。',
  code: `// 输入：标量 broadcast 的结果被用了两次
%0 = stablehlo.broadcast_in_dim %arg0, dims = []
     : (tensor<f32>) -> tensor<2x64xf32>
%1 = stablehlo.negate %0 : tensor<2x64xf32>
%2 = stablehlo.abs %0 : tensor<2x64xf32>
%3 = stablehlo.multiply %1, %2

// 输出：broadcast 被复制成两条
%0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
%1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
%2 = stablehlo.negate %0 : tensor<2x64xf32>
%3 = stablehlo.abs %1 : tensor<2x64xf32>
%4 = stablehlo.multiply %2, %3 : tensor<2x64xf32>

// 读法：
//   标量 %arg0 本身不分片（rank 0）-> 复制它没意义
//   但 broadcast 的结果 tensor<2x64xf32> 可以分片
//   -> 复制 broadcast 算子，让 negate / abs 各自拥有一份`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:28px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--bad)">✗ 复制标量（无意义）</div>
        <div class="row" style="gap:7px;align-items:center">
          <div class="chip mut" style="padding:6px 11px;font-size:11.5px">scalar</div>
          <div style="color:var(--ink-faint)">→</div>
          <div class="chip mut" style="padding:6px 11px;font-size:11.5px">scalar</div>
        </div>
        <div class="small faint">rank-0 没有维度可分</div>`;
      demo.appendChild(c);
      msg.innerHTML = '标量是 <span class="mono">tensor&lt;f32&gt;</span>（rank 0）—— 复制它<b>不会带来任何分片自由度</b>。';
    });
    tl.at(4600, () => {
      demo.innerHTML = `
        <div class="col" style="gap:7px;align-items:center">
          <div class="small mono" style="color:var(--ok)">✓ 复制 broadcast（有意义）</div>
          <div class="row" style="gap:6px;align-items:center">
            <div class="chip mut" style="padding:5px 9px;font-size:11px">scalar</div>
            <div style="color:var(--ok)">→</div>
            <div class="chip c0" style="padding:5px 9px;font-size:11px">bc</div>
            <div style="color:var(--ok)">→</div>
            <div class="chip c0" style="padding:5px 9px;font-size:11px">negate</div>
          </div>
          <div class="row" style="gap:6px;align-items:center">
            <div class="chip mut" style="padding:5px 9px;font-size:11px">scalar</div>
            <div style="color:var(--ok)">→</div>
            <div class="chip c1" style="padding:5px 9px;font-size:11px">bc</div>
            <div style="color:var(--ok)">→</div>
            <div class="chip c1" style="padding:5px 9px;font-size:11px">abs</div>
          </div>
        </div>`;
      msg.innerHTML = '<b>复制 broadcast</b>：结果的 <span class="mono">tensor&lt;2x64xf32&gt;</span> 可以分片，于是两个使用者各得一份。';
    });
    tl.at(9000, () => {
      msg.innerHTML = '<b>成本很低</b>：broadcast 只是"把一个值铺开"，复制它几乎没有额外计算。';
    });
    tl.at(12000, () => {
      msg.innerHTML = '<b>对比</b>：非标量输入的 broadcast <b>不</b>套用这条规则（输入本身有维度，语义不同）。';
    });
    tl.at(14200, () => {
      msg.innerHTML = '这一族有约 <b>15 个</b>用例，覆盖 diamond 依赖、函数返回值、同一算子多次使用等各种形态。';
    });
  }
},

/* ------------------------------------------------ 5 族谱 */
{
  kicker: 'L3-02 · 常量拆分',
  title: '95 个用例的<span class="hl-a">族谱</span>',
  sub: '这是全计划最大的文件 —— 因为「什么情况该拆」的组合非常多。',
  caption: '读法建议：抓住"使用次数 / 有无维度 / 是否标量 broadcast"三个判据，绝大多数用例都能推出来。',
  code: `// 【基础拆分】constant_multiple_users / constant_multiple_uses_by_same_op
//             splits_sdy_constants

// 【不拆的情形】func_arg_is_not_constant / scalar_constant_multiple_users_simple
//               non_scalar_constant_multiple_users_simple

// 【常量子表达式】constant_sub_computation_multiple_users
//                splits_parts_of_const_sub_computation

// 【形状算子】constant_broadcast/reshape/slice_multiple_users

// 【标量 broadcast】splits_broadcast_on_scalar_*（约 15 个）
//                  does_not_split_broadcast_*（约 4 个）

// 【分片组】splits_sharding_groups / splits_const_subexpr_with_sharding_group

// 【区域内】constant_*_within_named_computation*（约 12 个）

// 【函数调用】constant_*_within_call* / constant_to_call_*（约 18 个）

// 【循环与嵌套】constant_to_while / constant_to_and_inside_nested_*

// 【复杂签名】constant_to_*_with_multiple_arguments_* / _multi_result_*

// 【综合】split_constants_different_sharding / simple_non_flat / single_call`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '基础拆分', n: 3, c: '#4ade80' }, { t: '不拆情形', n: 3, c: '#fb7185' },
      { t: '常量子表达式', n: 2, c: '#38bdf8' }, { t: '形状算子', n: 3, c: '#fbbf24' },
      { t: '标量 broadcast', n: 19, c: '#c084fc' }, { t: '分片组', n: 2, c: '#f472b6' },
      { t: '区域内', n: 12, c: '#93c5fd' }, { t: '函数调用', n: 18, c: '#5eead4' },
      { t: '循环嵌套', n: 5, c: '#a78bfa' }, { t: '复杂签名', n: 6, c: '#fdba74' },
      { t: '综合', n: 3, c: '#94a3b8' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:100px;opacity:.4;transition:.35s;border-color:${f.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div class="card-t" style="color:${f.c};font-size:10.5px">${f.t}</div>
        <div class="big" style="font-size:19px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>95 个用例</b>归成 11 族 —— 最大的两族是"标量 broadcast"与"函数调用"。'; });
    tl.at(4200, () => {
      els.forEach((e, i) => { if (i !== 4 && i !== 7) e.style.opacity = '.25'; });
      msg.innerHTML = '这两族合计近四成 —— 说明<b>区域内/调用边界上的常量</b>是最容易出问题的场景。';
    });
    tl.at(7800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>三个判据</b>：① 使用次数 &gt; 1？② 有维度可分？③ 是标量 broadcast？';
    });
    tl.at(11000, () => {
      msg.innerHTML = '抓住这三条，95 个用例里的绝大多数都能自己推出来 —— 不必逐条读。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L3-02 · 练习',
  title: '练一练：<span class="hl-a">该不该拆</span>',
  sub: '三道题分别考：核心判据、标量规则、broadcast 规则。',
  caption: '一句话总结：<b>复制后能获得分片自由度就拆，否则不拆</b>。',
  code: `// 题 1：一个常量有 3 个使用，拆成几份？

// 题 2：tensor<f32> 的常量有 2 个使用，拆吗？

// 题 3：标量 broadcast 的结果有 2 个使用，拆什么？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '一个常量有 <b>3 个使用</b>，拆成几份？',
        a: '<b>3 份</b> —— 每个使用位置各一份。' +
           '<br><b>注意判据是「使用次数」</b>：即使 <span class="mono">dot_general %0, %0</span> 这种同一算子用两次，也要拆成两份。' +
           '<br><span class="dim">因为 <span class="mono">dot</span> 的左右两侧本来就该分开决策。</span>'
      },
      {
        q: '<span class="mono">tensor&lt;f32&gt;</span> 的常量有 2 个使用，拆吗？',
        a: '<b class="badge bad">不拆</b>。' +
           '<br><b>理由</b>：rank-0 张量<b>没有维度可分片</b> —— 复制它不会带来任何分片自由度，只会让 IR 变长。' +
           '<br><span class="dim">核心洞察：拆分的收益来自「每份可以有<b>不同分片</b>」。没有维度就没有不同分片可言。</span>'
      },
      {
        q: '标量 broadcast 的结果有 2 个使用，拆什么？',
        a: '<b>拆 broadcast 算子</b>（复制成两条），而不是拆标量。' +
           '<br><b>理由</b>：标量本身 rank-0 不可分片；但 broadcast 的结果 <span class="mono">tensor&lt;2x64xf32&gt;</span> <b>可以</b>分片。' +
           '<br><span class="dim">这就是文件名的由来：<span class="mono">constant_<b>or_scalar</b>_splitter</span>。注意前提：输入必须是 rank 0 —— 非标量输入的 broadcast 不套用这条规则。</span>'
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
