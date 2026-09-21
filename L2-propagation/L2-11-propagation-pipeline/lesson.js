/* ==========================================================================
   L2-11 · propagation-pipeline   （L2 完整流水线）
   --------------------------------------------------------------------------
   覆盖：transforms/propagation/test/propagation_pipeline.mlir (1200 行 / 58 用例)
   目标：看前面各课的效果叠加起来长什么样。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 全貌 */
{
  kicker: 'L2-11 · 完整流水线',
  title: '完整流水线：<span class="hl-a">各层叠加</span>后的样子',
  sub: '前面十课分别讲了单层策略。这一课跑的是 <span class="mono">-sdy-propagation-pipeline</span> —— <b>整个文件只有这一条 RUN 行</b>。',
  caption: '所以本课的重点不是新概念，而是<b>看组合效果</b>：用户写一个标注，流水线补出多少东西。',
  code: `// RUN: sdy_opt %s -split-input-file -sdy-propagation-pipeline

// 流水线做了什么（按顺序）：
//   ① 拆常量 / 拆标量        （-sdy-constant-or-scalar-splitter）
//   ② 规范化网格（内联 -> 命名）
//   ③ 组 id 合并与规范化
//   ④ 逐层传播（用户优先级 -> 算子优先级 -> 激进 -> 基础）
//   ⑤ 为未归约轴插入集合通信（all_reduce）
//   ⑥ 把 sharding_constraint 换成 reshard
//   ⑦ 清理（删掉临时属性）

// 用户写的：一个 unreduced 标注
// 流水线补的：all_reduce + reshard 两条通信算子`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const steps = ['拆常量', '规范化网格', '组合并', '逐层传播', '插集合通信', '约束→reshard', '清理'];
    const host = wrap.querySelector('#cards');
    const els = steps.map((t, i) => {
      const c = `var(--ax${i % 6})`;
      const e = U.el('div', { class: 'card', style: `width:96px;opacity:.35;transition:.3s;border-color:${c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div class="small mono" style="font-size:10px;color:${c}">${i + 1}</div>
        <div style="font-size:11px;margin-top:3px;line-height:1.35">${t}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { msg.innerHTML = '流水线共 7 步 —— 前四步在前面各课都见过，后三步是<b>收尾</b>。'; });
    tl.at(3600, () => { els.forEach((e, i) => e.style.opacity = i < 4 ? '1' : '.35'); msg.innerHTML = '<b>前四步</b>：拆常量、规范化网格、组合并、逐层传播。'; });
    tl.at(7200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = '<b>后三步</b>：为未归约轴插通信、把约束换成 reshard、清理临时属性。'; });
    tl.at(10600, () => { msg.innerHTML = '结果就是「用户写一点，流水线补一片」—— 本课用三个例子展示这一点。'; });
  }
},

/* ---------------------------------------------------- 2 拆常量 */
{
  kicker: 'L2-11 · 完整流水线',
  title: '★ 一个常量变<span class="hl-a">三个</span>',
  sub: '一个常量被两处使用、且两处需要不同分片 —— 流水线把它<b>复制成三份</b>，各自分片。',
  caption: '这是 L1-10 讲的「<span class="mono">sdy.constant</span> 刻意不实现 ConstantLike」的<b>实际收益</b>。',
  code: `// 输入：一个常量，两处使用
%0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
%1 = stablehlo.dot_general %0, %0, contracting_dims = [1] x [1]
%2 = stablehlo.add %1, %arg0
return %0, %2

// 输出：三个 sdy.constant，各自分片不同
%[[CONST_0]] = sdy.constant {sharding_per_value=[<@mesh, [{"a"}, {}]>]} dense<1.…>
%[[CONST_1]] = sdy.constant {sharding_per_value=[<@mesh, [{"b"}, {}]>]} dense<1.…>
%[[CONST_2]] = sdy.constant dense<1.000000e+00>          // 无分片

%[[DOT]] = stablehlo.dot_general %[[CONST_0]], %[[CONST_1]], contracting_dims = [1] x [1]
           {sharding_per_value=[<@mesh, [{"a"}, {"b"}]>]}
%[[ADD]] = stablehlo.add %[[DOT]], %arg0 {sharding_per_value=[<@mesh, [{"a"}, {"b"}]>]}
return %[[CONST_2]], %[[ADD]]

// 三个副本的分工：
//   CONST_0 -> dot 左操作数，沿 "a" 切
//   CONST_1 -> dot 右操作数，沿 "b" 切
//   CONST_2 -> 直接返回，不分片（复制）`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:30px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:8px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ax0)">输入</div>
        <div class="chip mut" style="padding:9px 15px">1 个 constant</div>
        <div class="small faint">被 dot 用了两次</div>`;
      demo.appendChild(c);
      msg.innerHTML = '一个常量，被 <span class="mono">dot_general</span> 当作<b>左右两个操作数</b>使用 —— 两处需要不同分片。';
    });
    tl.at(4400, () => {
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹', style: 'font-size:26px' }));
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ok)">输出</div>
        <div class="chip c0" style="padding:6px 12px;font-size:12px">CONST_0 → 沿 "a"</div>
        <div class="chip c1" style="padding:6px 12px;font-size:12px">CONST_1 → 沿 "b"</div>
        <div class="chip mut" style="padding:6px 12px;font-size:12px">CONST_2 → 不分片</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>一分为三</b>：两份给 dot（各切一维），一份给 return（复制）。';
    });
    tl.at(8600, () => {
      msg.innerHTML = '<b>为什么能拆</b>：<span class="mono">sdy.constant</span> 不实现 <span class="mono">ConstantLike</span>、不带 folder —— 重写器不会把它们合并回去（L1-10）。';
    });
    tl.at(12000, () => {
      msg.innerHTML = '<b>收益</b>：<span class="mono">dot</span> 的两个操作数可以各自按最优方式切，无需为对方妥协。';
    });
    tl.at(14200, () => {
      msg.innerHTML = '对应 pass：导入期的 <span class="mono">-sdy-constant-or-scalar-splitter</span>（L3-02）。';
    });
  }
},

/* ------------------------------------------------ 3 约束→reshard */
{
  kicker: 'L2-11 · 完整流水线',
  title: '约束 <span class="hl-a">→</span> reshard：用户意图被翻译成通信',
  sub: '`sdy.sharding_constraint` 在流水线结束时**不复存在** —— 它被换成了 `sdy.reshard`。',
  caption: '注意 <span class="mono">%arg0</span> 本身<b>没被改</b> —— 约束只作用于它那个使用者（L1-09 的"有使用者"语义）。',
  code: `// 输入
%0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
return %arg0, %0 : tensor<8x8xf32>, tensor<8x8xf32>

// 输出
// CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
// CHECK-NEXT: return %arg0, %0

// 读法：
//   %arg0 被【原样】返回 —— 没有被改成 {"a"},{"b"}
//   只有 %0 这一路变成了 reshard
//   -> 约束描述的是"这个使用者看到的分片"

// 完整链条（L1-10 讲过）：
//   constraint（用户写） -> 传播消费 -> reshard（流水线插）
//   -> collective（导出阶段替换）`,
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
        <div class="row" style="gap:8px;align-items:center">
          <div class="chip mut" style="padding:8px 13px">%arg0</div>
          <div style="color:var(--ax2);font-size:18px">⟹</div>
          <div class="chip c2" style="padding:8px 13px">constraint</div>
          <div style="color:var(--ink-faint);font-size:18px">→</div>
          <div class="chip c4" style="padding:8px 13px">return %0</div>
        </div>
        <div class="row" style="gap:8px;align-items:center">
          <div class="chip mut" style="padding:8px 13px">%arg0</div>
          <div style="color:var(--ink-faint);font-size:18px">──────────→</div>
          <div class="chip c4" style="padding:8px 13px">return %arg0</div>
        </div>
        <div class="small faint">两条独立的返回路径</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<span class="mono">%arg0</span> 被返回两次：一次<b>原样</b>，一次<b>经过约束</b>。';
    });
    tl.at(4600, () => {
      msg.innerHTML = '<b>输出</b>：只有经过约束的那一路变成 <span class="mono">sdy.reshard</span>；原样那一路完全没变。';
    });
    tl.at(8000, () => {
      msg.innerHTML = '<b>为什么</b>：约束描述的是"<b>这个使用者</b>看到的分片"，不是"张量本身的分片"（L1-09）。';
    });
    tl.at(11200, () => {
      msg.innerHTML = '<b>链条</b>：<span class="mono">constraint</span> → 传播消费 → <span class="mono">reshard</span> → 导出阶段替换成 collective。';
    });
    tl.at(13600, () => {
      msg.innerHTML = '测试注释直说了这一点：<span class="mono">// This test verifies that there is no sharding_constraint in the result.</span>';
    });
  }
},

/* ------------------------------------------------ 4 unreduced */
{
  kicker: 'L2-11 · 完整流水线',
  title: '★ 一个标注，<span class="hl-a">两条通信</span>',
  sub: '用户只写了 `unreduced={"b"}`，流水线自动补出 `all_reduce` 与 `reshard` 两条算子。',
  caption: '这是本课最能体现「流水线在做什么」的例子 —— 也解释了为什么手动写 SDY 很省事。',
  code: `// 输入：用户只写了 unreduced 标注
%0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
     {sdy.sharding = #sdy.sharding_per_value<
        [<@mesh, [{?}, {?}], unreduced={"b"}>]>} : …

// 输出：三步链条
// CHECK-NEXT: %[[DOT]] = stablehlo.dot_general %arg0, %arg1
// CHECK-SAME:   {sharding_per_value=[<@mesh, [{"a"}, {}], unreduced={"b"}>]}
// CHECK-NEXT: %[[ALL_REDUCE]] = sdy.all_reduce {"b"} %[[DOT]]
//                               out_sharding=<@mesh, [{"a"}, {}]>
// CHECK-NEXT: %[[RESHARD]] = sdy.reshard %[[ALL_REDUCE]]
//                            <@mesh, [{"a"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: %[[ADD]] = stablehlo.add %[[RESHARD]], %[[RESHARD]]
//                        {sharding_per_value=[<@mesh, [{"a"}, {"b"}]>]}

// 逐步：
//   ① dot 沿 "b" 切 -> 产生部分和（unreduced={"b"}）
//   ② all_reduce {"b"} -> 把部分和归约成完整值
//   ③ reshard -> 调整到下游 add 需要的分片`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:14px;justify-content:center;align-items:center" id="chain"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const chain = wrap.querySelector('#chain'), msg = wrap.querySelector('#msg');

    const node = (label, sub, color) => {
      const c = U.el('div', { class: 'col', style: 'gap:4px;align-items:center;opacity:0;transition:.45s' });
      c.innerHTML = `<div class="chip ${color}" style="padding:7px 12px;font-size:12px">${label}</div>
        <div class="small faint" style="font-size:10px;text-align:center">${sub}</div>`;
      chain.appendChild(c); return c;
    };
    const ar = () => chain.insertAdjacentHTML('beforeend', '<div class="arrow" style="font-size:18px">→</div>');

    tl.at(700, () => {
      node('dot_general', 'unreduced={"b"}', 'c0').style.opacity = '1';
      msg.innerHTML = '<b>用户写的</b>：只有 <span class="mono">unreduced={"b"}</span> 这一个标注。';
    });
    tl.at(4000, () => {
      ar();
      node('all_reduce {"b"}', '流水线插入', 'c2').style.opacity = '1';
      msg.innerHTML = '<b>第 1 条</b>：<span class="mono">all_reduce</span> —— 把沿 "b" 切出来的部分和归约成完整值。';
    });
    tl.at(7600, () => {
      ar();
      node('reshard', '流水线插入', 'c3').style.opacity = '1';
      msg.innerHTML = '<b>第 2 条</b>：<span class="mono">reshard</span> —— 把分片调整到下游需要的形状。';
    });
    tl.at(11200, () => {
      ar();
      node('add', '下游算子', 'c4').style.opacity = '1';
      msg.innerHTML = '<b>最终</b>：<span class="mono">add</span> 拿到 <span class="mono">[{"a"}, {"b"}]</span> —— 两维都切好了。';
    });
    tl.at(14000, () => {
      msg.innerHTML = '<b>对比</b>：手写这些通信很容易出错（少一条就语义错误）—— 这正是 SDY 的价值。';
    });
  }
},

/* ------------------------------------------------ 5 复用与其它 */
{
  kicker: 'L2-11 · 完整流水线',
  title: '同一批用例，<span class="hl-a">换个流水线</span>再跑一遍',
  sub: '本文件里的 `case_*` / `while_*` / `manual_computation_*` 与 L2-08、L2-09 <b>同名同形</b>，只是换了 RUN 行。',
  caption: '这是很好的<b>对照实验</b>：同一输入，单层策略 vs 完整流水线，差异一目了然。',
  code: `// L2-08 跑的是：-sdy-basic-propagate              （单层策略）
// L2-09 跑的是：-sdy-basic-propagate              （单层策略）
// 本课 跑的是：-sdy-propagation-pipeline          （完整流水线）

// 同名用例（可直接对照）：
//   case_* 13 个          （L2-08）
//   while_* 8 个          （L2-08）
//   optimization_barrier  （L2-08）
//   manual_computation_*  （L2-09）

// 其它值得看的用例：
//   user_priorities                       用户优先级（L2-06）
//   sharding_group_on_while_result        组作用在循环结果上（L2-07）
//   add_extra_sharding_constraint_...     组内不兼容插约束（L2-07）
//   do_not_propagate_manual_axes_...      manual 轴不外传（L2-09）
//   size_zero_dim_sharded                 大小为 0 的维度
//   dot_lhs_from_broadcast_and_large_rhs  真实模型形态
//   main × 8                              综合场景`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '同名对照', c: '#38bdf8',
        d: '<span class="mono">case_*</span>（13）、<span class="mono">while_*</span>（8）、<br><span class="mono">manual_computation_*</span> —— 与 L2-08/09 同名同形。' },
      { t: '跨课复用', c: '#c084fc',
        d: '<span class="mono">user_priorities</span>（L2-06）、<span class="mono">sharding_group_*</span>（L2-07）、<br><span class="mono">do_not_propagate_manual_axes_*</span>（L2-09）。' },
      { t: '边界与综合', c: '#4ade80',
        d: '<span class="mono">size_zero_dim_sharded</span>、<span class="mono">maximal/replicated_sharding_no_results</span>、<br><span class="mono">dot_lhs_from_broadcast_and_large_rhs</span>、<span class="mono">main</span> × 8。' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.34;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.34');
      msg.innerHTML = [
        '<b>对照实验</b>：同输入、不同 RUN 行 —— 直接看出流水线比单层策略多做了什么。',
        '这些用例证明：<b>前面各课的机制在流水线里同样生效</b>，不是被替换掉。',
        '综合用例（<span class="mono">main</span> × 8）最接近真实模型 —— 值得挑一两个细读。',
      ][i];
    }));
    tl.at(12400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>下一课</b> L2-12 讲传播调试 —— 当流水线结果不合预期时怎么排查。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L2-11 · 练习',
  title: '练一练：<span class="hl-a">流水线会补出什么</span>',
  sub: '三道题分别考：常量拆分、约束的归宿、未归约轴的处理。',
  caption: '一句话总结：<b>用户写一点，流水线补一片</b>。',
  code: `// 题 1：一个常量被两处使用、需要不同分片，流水线会怎么做？

// 题 2：sdy.sharding_constraint 在流水线结束时还在吗？

// 题 3：用户写了 unreduced={"b"}，流水线会补出什么？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '一个常量被两处使用、且需要不同分片，流水线会怎么做？',
        a: '<b>复制成多份</b>，每份按各自的需要分片。' +
           '<br>用例里一个常量变成了<b>三个</b> <span class="mono">sdy.constant</span>：两份给 <span class="mono">dot</span>（各切一维），一份给 <span class="mono">return</span>（不分片）。' +
           '<br><span class="dim">能这样做的前提是 <span class="mono">sdy.constant</span> 不实现 <span class="mono">ConstantLike</span>、不带 folder（L1-10）——否则会被合并回去。</span>'
      },
      {
        q: '<span class="mono">sdy.sharding_constraint</span> 在流水线结束时还在吗？',
        a: '<b class="badge bad">不在了</b>，被换成了 <span class="mono">sdy.reshard</span>。' +
           '<br>测试注释直说了：<span class="mono">// This test verifies that there is no sharding_constraint in the result.</span>' +
           '<br><span class="dim">完整链条：constraint（用户写）→ 传播消费 → reshard → 导出阶段换成 collective。</span>'
      },
      {
        q: '用户写了 <span class="mono">unreduced={"b"}</span>，流水线会补出什么？',
        a: '<b>两条通信算子</b>：<span class="mono">sdy.all_reduce {"b"}</span> 与 <span class="mono">sdy.reshard</span>。' +
           '<br><b>顺序</b>：dot 沿 "b" 切产生部分和 → all_reduce 归约成完整值 → reshard 调整到下游需要的分片。' +
           '<br><span class="dim">用户只写了一个标注，流水线补出整条通信链 —— 这正是 SDY 的价值所在。</span>'
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
