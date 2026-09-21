/* ==========================================================================
   L3-03 · apply-sharding-constraints
   --------------------------------------------------------------------------
   覆盖：transforms/import/test/apply_sharding_constraints.mlir (716 行 / 39 用例)
         transforms/import/test/apply_sharding_constraints_preinlined.mlir (68 / 5)
   目标：讲透"约束该贴在哪里"——能否下沉到操作数上。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 问题 */
{
  kicker: 'L3-03 · 应用分片约束',
  title: '约束该<span class="hl-a">贴在哪里</span>？',
  sub: '`sdy.sharding_constraint` 是一个**独立的算子**，它把"我要求这样切"写成了一条中间值。但能不能把它"挪"到操作数上？',
  caption: '挪过去的好处：分片信息直接落在算子上，传播时少一层间接。这个动作叫<b>下沉</b>（sink）。',
  code: `// 输入
%0 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
%1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> : tensor<8x8xf32>
return %1

// 如果 %0 只被这个约束使用、且自己没有分片
//   -> 可以把 [{}, {"b"}] 直接写到 %0 上

// 输出（下沉）
stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}

// 什么时候【不能】下沉：
//   ① 操作数已有自己的分片标注
//   ② 约束是【开维】的（带 ?）
//   ③ 操作数还有【其它不同的】约束使用者`,
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
        <div class="small mono faint">下沉前</div>
        <div class="row" style="gap:8px;align-items:center">
          <div class="chip mut" style="padding:7px 12px;font-size:12px">add %0</div>
          <div style="color:var(--ink-faint)">→</div>
          <div class="chip c2" style="padding:7px 12px;font-size:12px">constraint</div>
          <div style="color:var(--ink-faint)">→</div>
          <div class="chip mut" style="padding:7px 12px;font-size:12px">return</div>
        </div>
        <div class="small faint">约束是一条独立的中间值</div>`;
      demo.appendChild(c);
      msg.innerHTML = '约束算子把"我要求这样切"写成了一条<b>中间值</b> —— 多了一层。';
    });
    tl.at(4600, () => {
      demo.innerHTML = `
        <div class="col" style="gap:7px;align-items:center">
          <div class="small mono faint">下沉后</div>
          <div class="row" style="gap:8px;align-items:center">
            <div class="chip c4" style="padding:7px 12px;font-size:12px">add {sharding=[{}, {"b"}]}</div>
          </div>
          <div class="small faint">分片直接落在算子上</div>
        </div>`;
      msg.innerHTML = '<b>下沉</b>：把约束的分片直接写到 <span class="mono">add</span> 上，约束算子消失。';
    });
    tl.at(8600, () => {
      msg.innerHTML = '<b>好处</b>：传播时不必"穿过"一条约束算子 —— 分片信息就在产生它的地方。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>但不是总能下沉</b>：本课大部分用例（39 个）都在锁定"什么情况不能下沉"。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 三个条件 */
{
  kicker: 'L3-03 · 应用分片约束',
  title: '★ 下沉的<span class="hl-a">三个条件</span>',
  sub: '三个条件必须**同时**满足，缺一不可。',
  caption: '记住这三条，39 个用例里绝大多数都能推出来。',
  code: `// ① 操作数【没有】自己的分片标注
%0 = stablehlo.add %arg0, %arg0 {sharding=[{"a", ?}, {?}]}   // 已有！
%1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]>
// -> 不下沉（覆盖已有分片会改变语义）

// ② 约束必须是【闭维】的（不带 ?）
%1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b", ?}]>      // 开维！
// -> 不下沉（开维 = "至少这样切，还可再加"，下沉会过度约束）

// ③ 操作数【没有其它不同的】约束使用者
%1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]>
%2 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]>         // 不同！
// -> 不下沉（一个分片满足不了两个要求）
//
// 注意：要求【相同】的两个约束不影响下沉
// 注意：普通使用者（非约束）也不影响下沉`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 操作数无分片', c: '#4ade80',
        d: '已有分片说明用户明确表态过<br>→ 不能被约束顶掉' },
      { t: '② 约束是闭维', c: '#38bdf8',
        d: '开维 <span class="mono">{"b", ?}</span> 表示"至少这样切"<br>→ 下沉会过度约束' },
      { t: '③ 无其它不同约束', c: '#fbbf24',
        d: '一个分片满足不了两个要求<br>→ 都不下沉（相同的可以）' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '这一条最直观：用户写了的不能被编译器"优化掉"。',
        '<b>开维的含义</b>：约束给出的是<b>下界</b>，不是最终答案。下沉会把下界当成定论。',
        '<b>关键区分</b>：妨碍下沉的是「其它<b>约束</b>使用者」，不是「其它使用者」。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>第三条的细节</b>：普通使用者（如另一个 <span class="mono">add</span>）<b>不</b>妨碍下沉 —— 它们的分片由传播去协调。';
    });
  }
},

/* ------------------------------------------------ 3 边界情形 */
{
  kicker: 'L3-03 · 应用分片约束',
  title: '两类容易搞错的边界',
  sub: '「其它使用者」与「悬空」—— 这两个概念看起来都像"不该下沉"，实际**都下沉**。',
  caption: '判据始终是那三条，不要被"看起来复杂"干扰。',
  code: `// ① 有其它【普通】使用者 -> 照样下沉
%0 = stablehlo.add %arg0, %arg0
%1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]>
%2 = stablehlo.add %0, %0            // 另一个使用者
return %0, %1, %2                    // 还被 return 了
// 输出：add %0 拿到 [{}, {"b"}]  ✓ 下沉
// 理由：妨碍下沉的是"其它【约束】使用者"，不是"其它使用者"

// ② 悬空约束 -> 照样下沉
%0 = stablehlo.add %arg0, %arg0
%1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]>
%2 = stablehlo.add %0, %0
return %0, %2                        // %1 没被任何算子使用（悬空）
// 输出：add %0 拿到 [{}, {"b"}]  ✓ 下沉
// 理由：悬空约束表达的正是"%0 本身应该这样切"（L1-09）`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 有其它普通使用者', c: '#4ade80', v: '✓ 下沉',
        d: '<span class="mono">%0</span> 被 <span class="mono">add</span> 与 <span class="mono">return</span> 用了<br>但都不是<b>约束</b>使用者。' },
      { t: '② 约束悬空', c: '#38bdf8', v: '✓ 下沉',
        d: '约束的结果没人用<br>但它表达的正是"<span class="mono">%0</span> 本身该怎么切"。' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t} <span class="badge ok">${x.v}</span></div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>反直觉</b>：<span class="mono">%0</span> 有一堆别的使用者，还是下沉了。'; });
    tl.at(4400, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>也反直觉</b>：约束自己没人用（悬空），照样下沉 —— 因为语义上它就是在描述 <span class="mono">%0</span>。';
    });
    tl.at(8200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>回到三条判据</b>：操作数无分片 ✓、约束闭维 ✓、无其它<b>不同约束</b>使用者 ✓ → 下沉。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>学习建议</b>：遇到不确定的情形，先套这三条，再看用例验证。';
    });
  }
},

/* ------------------------------------------------ 4 约束链 */
{
  kicker: 'L3-03 · 应用分片约束',
  title: '约束<span class="hl-a">链</span>：逐个下沉',
  sub: '多个约束串成链时，只有**链头**会下沉 —— 一次处理一条。',
  caption: '注意两条约束算子**都还在** —— 下沉不等于删除约束，只是把信息"抄"一份到操作数上。',
  code: `%0 = stablehlo.add %arg0, %arg0
%1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]>     // 链头
%2 = sdy.sharding_constraint %1 <@mesh, [{}, {"b"}]>     // 链尾
%3 = stablehlo.add %0, %0
return %2, %3

// 输出：
%[[ADD_0]] = stablehlo.add %arg0, %arg0
             {sharding_per_value=[<@mesh, [{"a"}, {}]>]}   // 链头下沉到这里
%[[WSC_0]] = sdy.sharding_constraint %[[ADD_0]] <@mesh, [{"a"}, {}]>
%[[WSC_1]] = sdy.sharding_constraint %[[WSC_0]] <@mesh, [{}, {"b"}]>
%[[ADD_1]] = stablehlo.add %[[WSC_1]], %[[WSC_1]]
return %[[WSC_1]], %[[ADD_1]]

// 读法：
//   链头 %1 的操作数是 %0（无分片、闭维、无其它约束）
//     -> 下沉，%0 拿到 [{"a"}, {}]
//   链尾 %2 的操作数是 %1（本身是约束）
//     -> 不下沉
//   两条约束算子都保留`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:14px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="small mono faint">输入链</div>
        <div class="row" style="gap:6px;align-items:center">
          <div class="chip mut" style="padding:6px 11px;font-size:11.5px">%0 add</div>
          <div style="color:var(--ink-faint)">→</div>
          <div class="chip c2" style="padding:6px 11px;font-size:11.5px">%1 [a]</div>
          <div style="color:var(--ink-faint)">→</div>
          <div class="chip c3" style="padding:6px 11px;font-size:11.5px">%2 [b]</div>
        </div>`;
      demo.appendChild(c);
      msg.innerHTML = '两个约束串成链：<span class="mono">%1</span> 要求 <span class="mono">[{"a"},{}]</span>，<span class="mono">%2</span> 要求 <span class="mono">[{},{"b"}]</span>。';
    });
    tl.at(4400, () => {
      msg.innerHTML = '<b>链头 <span class="mono">%1</span> 下沉</b>：它的操作数 <span class="mono">%0</span> 满足三条判据 → <span class="mono">%0</span> 拿到 <span class="mono">[{"a"}, {}]</span>。';
      const chips = demo.querySelectorAll('.chip');
      if (chips[0]) chips[0].classList.add('pulse');
    });
    tl.at(8200, () => {
      msg.innerHTML = '<b>链尾 <span class="mono">%2</span> 不下沉</b>：它的操作数 <span class="mono">%1</span> 本身就是一条约束 —— 不属于"普通算子"。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>两条约束算子都还在</b> —— 下沉只是把信息"抄"到操作数上，不删除约束本身。';
    });
    tl.at(13800, () => {
      msg.innerHTML = '<b>逐个处理</b>：一次只处理链上一条，不会递归穿透整条链。';
    });
  }
},

/* ------------------------------------------------ 5 preinlined */
{
  kicker: 'L3-03 · 应用分片约束',
  title: '<span class="mono hl-a">preinlined</span>：为什么要先内联',
  sub: '第二个文件用**两步**流水线：先内联函数调用，再应用约束。',
  caption: '原因很实际：<b>函数调用边界会挡住下沉分析</b> —— 约束的输入是跨函数的值，规则无从判断。',
  code: `// apply_sharding_constraints_preinlined.mlir 的 RUN 行：
// RUN: sdy_opt %s -split-input-file \\
//        -sdy-import-func-calls -sdy-apply-sharding-constraints

// 对比主文件的 RUN 行（只有一步）：
// RUN: sdy_opt %s -split-input-file -sdy-apply-sharding-constraints

// 为什么需要这个变体：
//   函数调用边界会挡住约束的下沉分析
//   约束的输入可能是"调用点的实参"，规则无从判断
//   先内联成 sdy.named_computation（L1-08 / L3-06）
//   -> 约束的输入变成区域内的值
//   -> 下沉规则才能在区域内部正常生效

// 该文件 5 个用例（foo / main / bar）都涉及跨函数结构`,
  duration: 14000,
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
        <div class="small mono" style="color:var(--bad)">✗ 直接应用约束</div>
        <div class="row" style="gap:7px;align-items:center">
          <div class="chip mut" style="padding:6px 11px;font-size:11.5px">call</div>
          <div style="color:var(--bad);font-size:15px">⤫</div>
          <div class="chip c2" style="padding:6px 11px;font-size:11.5px">constraint</div>
        </div>
        <div class="small faint">跨函数边界，规则无从判断</div>`;
      demo.appendChild(c);
      msg.innerHTML = '约束的输入可能来自<b>另一个函数</b> —— 下沉分析在这里断了。';
    });
    tl.at(4400, () => {
      demo.innerHTML = `
        <div class="col" style="gap:7px;align-items:center">
          <div class="small mono" style="color:var(--ok)">✓ 先内联，再应用</div>
          <div class="row" style="gap:7px;align-items:center">
            <div class="chip c4" style="padding:6px 11px;font-size:11.5px">named_computation</div>
            <div style="color:var(--ok);font-size:15px">→</div>
            <div class="chip c2" style="padding:6px 11px;font-size:11.5px">constraint</div>
          </div>
          <div class="small faint">约束的输入变成区域内的值</div>
        </div>`;
      msg.innerHTML = '<b>两步流水线</b>：先把调用内联成 <span class="mono">sdy.named_computation</span>（L1-08），再应用约束。';
    });
    tl.at(8600, () => {
      msg.innerHTML = '<b>内联后</b>：约束的输入是区域内的普通值，三条下沉判据可以正常套用。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '这也解释了 L3-01 讲的"<b>pass 顺序是隐式契约</b>"—— 顺序换了，分析就做不了。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L3-03 · 练习',
  title: '练一练：<span class="hl-a">会不会下沉</span>',
  sub: '三道题分别考：三个条件、其它使用者、约束链。',
  caption: '一句话总结：<b>操作数无分片 + 约束闭维 + 无其它不同约束 → 下沉</b>。',
  code: `// 题 1：下面会下沉吗？
%0 = stablehlo.add %arg0, %arg0 {sharding=[{"a",?},{?}]}
%1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]>

// 题 2：%0 还被另一个 add 使用，会影响下沉吗？

// 题 3：两个约束串成链，哪个会下沉？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">%0</span> 已有分片 <span class="mono">[{"a",?},{?}]</span>，约束要求 <span class="mono">[{},{"b"}]</span>，会下沉吗？',
        a: '<b class="badge bad">不会</b> —— 违反条件 ①（操作数已有自己的分片标注）。' +
           '<br><b>理由</b>：覆盖已有分片会<b>改变算子语义</b> —— 用户明确写了 <span class="mono">{"a", ?}</span>，不能被约束顶掉。' +
           '<br><span class="dim">输出里 add 保留原分片，约束仍留在结果上。</span>'
      },
      {
        q: '<span class="mono">%0</span> 还被另一个 <span class="mono">add</span> 使用，会影响下沉吗？',
        a: '<b class="badge ok">不影响，照样下沉</b>。' +
           '<br><b>关键区分</b>：妨碍下沉的是「其它<b>约束</b>使用者」（且要求不同），不是「其它使用者」。' +
           '<br><span class="dim">普通使用者的分片由传播去协调 —— 这正是传播该干的活。用例 <span class="mono">no_other_sharding_constraint_users</span> 验证了这一点。</span>'
      },
      {
        q: '两个约束串成链（<span class="mono">%0 → %1 → %2</span>），哪个会下沉？',
        a: '<b>只有链头 <span class="mono">%1</span></b> 下沉 —— 它的操作数 <span class="mono">%0</span> 是普通算子且满足三条判据。' +
           '<br>链尾 <span class="mono">%2</span> 的操作数 <span class="mono">%1</span> 本身就是约束 → 不下沉。' +
           '<br><span class="dim">注意：<b>两条约束算子都还在</b> —— 下沉只是把信息"抄"一份到操作数上，不删除约束本身。处理是<b>逐个</b>的，不会递归穿透整条链。</span>'
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
