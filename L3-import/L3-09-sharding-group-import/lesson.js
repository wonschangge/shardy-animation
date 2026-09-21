/* ==========================================================================
   L3-09 · sharding-group-import
   --------------------------------------------------------------------------
   覆盖：transforms/import/test/sharding_group_import.mlir (74 行 / 6 用例)
         transforms/import/test/sharding_group_constraints.mlir (185 / 8)
   目标：讲透组 id 的【合并】与【重编号】，以及组的校验约束。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 两个动作 */
{
  kicker: 'L3-09 · 分片组导入',
  title: '两个动作：<span class="hl-a">合并重叠</span> + <span class="hl-a">重编号</span>',
  sub: '用户在 IR 里随手写 `group_id`（可以是任意 i64）。导入期把它规范化成**紧凑的 `0..N-1`**。',
  caption: 'L2-07 说"组 id 先规范化再检查兼容性" —— 本课就是那个规范化的完整规则。',
  code: `// RUN: sdy_opt -split-input-file %s -sdy-sharding-group-import

// 用户的写法（任意数字，可能重叠）
sdy.sharding_group %arg0 group_id = 12 : tensor<4xf32>
sdy.sharding_group %arg1 group_id = 89 : tensor<4xf32>

// 规范化后
sdy.sharding_group %arg0 group_id=0 : tensor<4xf32>
sdy.sharding_group %arg1 group_id=1 : tensor<4xf32>

// 【动作 ①】合并重叠的组（传递闭包）
//   两个张量若共享任意一个 group_id -> 属于同一组
//   间接关联（通过中间组）也算
//   合并后采用【最小的 id】

// 【动作 ②】重编号为 0..N-1
//   紧凑编号让后续"按组遍历"可以用数组而不是哈希表

// ⚠ 注意 RUN 行的 -split-input-file：
//   组 id 计数器是【按段重置】的
//   漏掉它 -> 所有函数共用一个计数器 -> 编号完全不同`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 合并重叠', c: '#4ade80',
        d: '共享任意 group_id → 同组<br>间接关联也算<br>合并后取<b>最小 id</b>' },
      { t: '② 重编号', c: '#38bdf8',
        d: '任意 i64 → <b>0..N-1</b><br>紧凑编号便于<br>按组遍历' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>合并是传递闭包</b>：A 与 B 共享一个 id、B 与 C 共享另一个 id → A、B、C 同组。'; });
    tl.at(4400, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>重编号</b>让组 id 变成 <span class="mono">0, 1, 2, ...</span> 的连续整数。';
    });
    tl.at(8200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>为什么必须做</b>：不合并的话，同一组被拆成多个 id，传播时会各自为政（L2-07）。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>⚠ 复现提示</b>：RUN 行里的 <span class="mono">-split-input-file</span> 不能漏 —— 计数器是按段重置的。';
    });
  }
},

/* ------------------------------------------------ 2 合并规则 */
{
  kicker: 'L3-09 · 分片组导入',
  title: '★ 合并规则：<span class="hl-a">传递闭包</span>',
  sub: '判据是"是否共享任意一个 `group_id`" —— 直接共享或间接关联都算。',
  caption: '合并后采用该组**最小的 id** 作为编号（用例名 <span class="mono">overlap_min_id_used</span> 点明了）。',
  code: `// 【不重叠】-> 保持不变
%arg0 group_id = 0
%arg1 group_id = 1
// 两个张量分属不同组 -> 无可合并

// 【全部重叠】-> 合并成一个
%arg0 group_id = 0
%arg0 group_id = 1
%arg1 group_id = 0      // <- 通过组 0 与 %arg0 关联
%arg1 group_id = 2
// 传递闭包把 0/1/2 合成一个组 -> 都变成 0

// 【间接关联】-> 也合并
%arg0 group_id = 0
%arg0 group_id = 1
%arg0 group_id = 2
%arg1 group_id = 2      // <- 只通过组 2 关联
// -> 合并，采用最小 id 0

// 【部分重叠】-> 各自合并
%arg0 group_id = 0 / 1
%arg1 group_id = 2 / 3  // <- 与 %arg0 无交集
// -> 两个组：%arg0->0，%arg1->1`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '不重叠', c: '#94a3b8', r: '0, 1', d: '无可合并<br>编号保持' },
      { t: '全部重叠', c: '#4ade80', r: '0, 0', d: '共享组 0<br>→ 合成一个' },
      { t: '间接关联', c: '#38bdf8', r: '0, 0', d: '只通过组 2 关联<br>→ 传递闭包也合并' },
      { t: '部分重叠', c: '#fbbf24', r: '0, 1', d: '两组无交集<br>→ 各自合并' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:180px;opacity:.33;transition:.35s;border-color:' + x.c + '55;padding:9px' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:11.5px">${x.t}</div>
        <div class="mono" style="font-size:13px;color:#bdf7ec;margin:5px 0">${U.esc(x.r)}</div>
        <div class="card-d" style="font-size:10.5px;line-height:1.45">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '最简情形：两个张量没有任何共享 id → 两个独立的组。',
        '<b>直接共享</b>：<span class="mono">%arg0</span> 与 <span class="mono">%arg1</span> 都在组 0 → 关联。',
        '<b>间接关联</b>：两者只共享组 2，但组 2 同时属于 <span class="mono">%arg0</span> 的 {0,1,2} —— 传递闭包照样合并。',
        '<b>无交集就不合并</b>：<span class="mono">%arg0</span> 的 {0,1} 与 <span class="mono">%arg1</span> 的 {2,3} 各成一组。',
      ][i];
    }));
    tl.at(14400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>算法本质</b>：并查集（union-find）—— 把每个 group_id 当成一个节点，共享张量就把它们连起来。';
    });
  }
},

/* ------------------------------------------------ 3 重编号 */
{
  kicker: 'L3-09 · 分片组导入',
  title: '★ 重编号：任意 i64 → <span class="mono hl-a">0..N-1</span>',
  sub: '用户写 12、89、123456 都行；导入后一律变成紧凑的连续整数。',
  caption: '用例名点明了排序依据：<span class="mono">reindex_ordering_matches_min_element_ordering</span> —— 与最小元素顺序一致。',
  code: `// 【基本重编号】
%arg0 group_id = 12       ->  group_id=0
%arg1 group_id = 89       ->  group_id=1

// 【排序依据】三个组，各自的 id 集合：
//   %arg0: {567, 23}     最小 23
//   %arg1: {2}           最小 2
//   %arg2: {123456}      最小 123456
//
// 用例名：reindex_ordering_matches_min_element_ordering
// 期望输出：
//   %arg0 group_id=0
//   %arg1 group_id=1
//   %arg2 group_id=2

// ⚠ 实测提示：必须按 RUN 行加 -split-input-file
//   否则所有函数共用一个计数器，编号会变成 3、0、4 之类
//   与 CHECK 不符 —— 我实际踩过这个坑`,
  duration: 15000,
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
        <div class="small mono faint">用户写的（任意）</div>
        <div class="row" style="gap:6px">
          <div class="chip mut" style="padding:6px 11px;font-size:11.5px">12</div>
          <div class="chip mut" style="padding:6px 11px;font-size:11.5px">89</div>
          <div class="chip mut" style="padding:6px 11px;font-size:11.5px">123456</div>
        </div>`;
      demo.appendChild(c);
      msg.innerHTML = '<span class="mono">group_id</span> 是 <span class="mono">i64</span> —— 用户想写多大都行。';
    });
    tl.at(4400, () => {
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹', style: 'font-size:26px' }));
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ok)">规范化后（紧凑）</div>
        <div class="row" style="gap:6px">
          <div class="chip c4" style="padding:6px 11px;font-size:11.5px">0</div>
          <div class="chip c4" style="padding:6px 11px;font-size:11.5px">1</div>
          <div class="chip c4" style="padding:6px 11px;font-size:11.5px">2</div>
        </div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>重编号</b>：变成 <span class="mono">0..N-1</span> 的连续整数。';
    });
    tl.at(8600, () => {
      msg.innerHTML = '<b>为什么紧凑</b>：后续"按组遍历"可以用<b>数组</b>而不是哈希表 —— 这是性能与实现简洁性的考虑。';
    });
    tl.at(11600, () => {
      msg.innerHTML = '<b>排序依据</b>：用例名说是"与最小元素顺序一致"（每组取最小 id 参与排序）。';
    });
    tl.at(14000, () => {
      msg.innerHTML = '<b>⚠ 复现提示</b>：我第一次跑时漏了 <span class="mono">-split-input-file</span>，编号变成了 3、0、4 —— 与 CHECK 完全不符。';
    });
  }
},

/* ------------------------------------------------ 4 两条约束 */
{
  kicker: 'L3-09 · 分片组导入',
  title: '两条<span class="hl-a">校验约束</span>',
  sub: '组是"强制同分片"的承诺 —— 承诺不成立时必须报错，而不是静默忽略。',
  caption: '8 个用例里 6 条报错、2 个合法 —— 合法用例同样重要（防止误报）。',
  code: `// 【约束 ①】组的值不能跨越 manual_computation 边界
// expected-error: ShardingGroupOps values cannot cross
//                 ManualComputationOp boundaries for groupId: 90210
sdy.sharding_group %0 group_id = 90210 : tensor<8x8xf32>
// 同组的值一个在 manual_computation 内、一个在外 -> 报错
// 为什么：内部是"局部世界"（L1-07），内外分片语义不可比
// 4 个用例覆盖不同跨越方向（外->内、内->外、跨多层）

// 【约束 ②】同组的值形状必须相同
// expected-error: ShardingGroupOps values must have the same shape
//                 for groupId: 23
sdy.sharding_group %arg0 group_id = 23 : tensor<8x8xf32>
sdy.sharding_group %0 group_id = 23 : tensor<8x8xf32>
sdy.sharding_group %1 group_id = 23 : tensor<8x8x1xf32>   // 形状不同
// 为什么：形状不同 -> rank 不同 -> 分片根本无法比较

// 【合法的 2 个用例】
//   组的值全在 manual_computation 内部（不跨界）
//   组的值全在外部`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 不能跨界', c: '#fb7185', n: '4 个用例',
        d: '组的值不能横跨<br><span class="mono">manual_computation</span> 边界<br><span class="dim">内部是局部世界</span>' },
      { t: '② 形状相同', c: '#fbbf24', n: '1 个用例',
        d: '同组的值<br>形状必须一致<br><span class="dim">否则分片无法比较</span>' },
      { t: '✓ 合法情形', c: '#4ade80', n: '2 个用例',
        d: '全在内部 / 全在外部<br><b>不报错</b><br><span class="dim">防止误报</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>理由与 L2-09 一致</b>：manual 轴冻结、区域内是局部形状 —— 内外的值在分片语义上不可比。',
        'rank 不同 → 维度分片的<b>项数</b>都不同 → "强制同分片"无从谈起。',
        '<b>为什么要测合法情形</b>：校验器容易写得太严 —— 这两个用例锁定了"不该报错的不报"。',
      ][i];
    }));
    tl.at(13000, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>设计原则</b>：组是硬约束 —— 承诺不成立时<b>必须报错</b>，不能静默忽略。';
    });
  }
},

/* ------------------------------------------------ 5 生命周期 */
{
  kicker: 'L3-09 · 分片组导入',
  title: '组的<span class="hl-a">完整生命周期</span>',
  sub: '把 L1-09、L2-07、L3-01 与本课串起来 —— 一个 group_id 从写入到生效的全过程。',
  caption: '这是本层的最后一课（除 L3-10/11），正好可以做一次小结。',
  code: `// 【L1-09】用户写语法
sdy.sharding_group %arg0 group_id = 8675 : tensor<8x8xf32>

// 【本课 L3-09】导入期规范化
//   ① 传递闭包合并重叠的组
//   ② 重编号为 0..N-1
//   ③ 校验：不跨界 / 形状相同
sdy.sharding_group %arg0 group_id=0 : tensor<8x8xf32>

// 【L2-07】传播中作为【独立通道】生效
//   绕过数据流、穿透 propagation_barrier
//   组内冲突按用户优先级裁决
//   不兼容时插入额外约束

// 【L3-01】见过这个规范化的实际效果
//   4 条 sharding_group、3 个 id -> 2 条、1 个 id

// 一句话：用户随手写 -> 规范化 -> 传播中强制执行`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const steps = [
      { n: 'L1-09', t: '写语法', d: '任意 group_id', c: '#38bdf8' },
      { n: 'L3-09', t: '规范化', d: '合并 + 重编号<br>+ 校验', c: '#4ade80' },
      { n: 'L2-07', t: '传播生效', d: '独立通道<br>穿透屏障', c: '#fbbf24' },
      { n: 'L3-01', t: '实际效果', d: '4 条 → 2 条<br>3 个 id → 1 个', c: '#c084fc' },
    ];
    const host = wrap.querySelector('#cards');
    const els = steps.map(s => {
      const e = U.el('div', { class: 'card', style: `width:178px;opacity:.35;transition:.35s;border-color:${s.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="small mono" style="font-size:10px;color:${s.c}">${s.n}</div>
        <div style="font-size:12.5px;font-weight:600;margin:4px 0">${s.t}</div>
        <div class="small faint" style="font-size:10.5px;line-height:1.4">${s.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    steps.forEach((s, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.35');
      msg.innerHTML = [
        'L1-09 讲语法时提过 <span class="mono">group_id</span> 是任意整数 —— 这里看到了"为什么可以任意"。',
        '<b>本课</b>：合并、重编号、校验。这是"用户随意"到"内部紧凑"的转换点。',
        '规范化之后的组才是传播看到的样子 —— L2-07 讲的"独立通道"建立在这个基础上。',
        'L3-01 的那个例子（4 条 → 2 条）现在完全说得通了。',
      ][i];
    }));
    tl.at(14400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>一句话</b>：用户随手写 → 规范化 → 传播中强制执行。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L3-09 · 练习',
  title: '练一练：<span class="hl-a">合并后会是什么</span>',
  sub: '三道题分别考：合并判据、重编号、校验约束。',
  caption: '一句话总结：<b>传递闭包合并 + 重编号为 0..N-1 + 两条校验</b>。',
  code: `// 题 1：%a 在组 {0,1}，%b 在组 {1,2}，会合并吗？变成几个组？

// 题 2：%a 在组 12，%b 在组 89，重编号后是什么？

// 题 3：同组的值一个在 manual_computation 内、一个在外，
//       会怎样？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">%a</span> 在组 <span class="mono">{0,1}</span>，<span class="mono">%b</span> 在组 <span class="mono">{1,2}</span>，会合并吗？变成几个组？',
        a: '<b class="badge ok">会合并</b>，变成 <b>1 个组</b>。' +
           '<br><b>理由</b>：两者<b>共享组 1</b> → 传递闭包把它们连起来。合并后采用<b>最小的 id</b>（0）。' +
           '<br><span class="dim">这就是 <span class="mono">sharding_groups_all_overlap</span> / <span class="mono">overlap_min_id_used</span> 两个用例验证的。</span>'
      },
      {
        q: '<span class="mono">%a</span> 在组 12，<span class="mono">%b</span> 在组 89，重编号后是什么？',
        a: '<span class="mono">%a → 0</span>，<span class="mono">%b → 1</span>。' +
           '<br><b>理由</b>：两组不重叠 → 各自成组；重编号为紧凑的 <span class="mono">0..N-1</span>。' +
           '<br><span class="dim">紧凑编号让后续"按组遍历"可以用数组而不是哈希表。</span>'
      },
      {
        q: '同组的值一个在 <span class="mono">manual_computation</span> 内、一个在外，会怎样？',
        a: '<b class="badge bad">报错</b>：<span class="mono">ShardingGroupOps values cannot cross ManualComputationOp boundaries for groupId: N</span>' +
           '<br><b>理由</b>：手动计算内部是"局部世界"（L1-07），内外的值在分片语义上<b>不可比</b> —— 强制同分片无从谈起。' +
           '<br><span class="dim">4 个用例覆盖不同跨越方向。另有 2 个<b>合法</b>用例（全在内部 / 全在外部）防止误报。</span>'
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
