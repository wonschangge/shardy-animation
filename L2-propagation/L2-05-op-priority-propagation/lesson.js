/* ==========================================================================
   L2-05 · op-priority-propagation
   --------------------------------------------------------------------------
   覆盖：transforms/propagation/test/op_priority_propagation.mlir (313 行 / 15 用例)
   目标：讲透传播金字塔第 3 层 —— 按算子类型分批、按方向优先级传播。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 层级定位 */
{
  kicker: 'L2-05 · 算子优先级',
  title: '传播金字塔第 3 层：<span class="hl-a">按算子类型分批</span>',
  sub: '基础传播一次推完；激进传播多一组消解规则；算子优先级再往前一步：<b>按算子类型分批、按方向优先级逐轮传播</b>。',
  caption: '直白地说：<b>先让"简单"的算子（逐元素、广播）把分片铺开，再让"复杂"的算子（dot、reduce）去适配</b>。',
  code: `// RUN: sdy_opt %s -split-input-file -sdy-op-priority-propagate

// 机制（来自 pass 文档）：
//   从 op-priority 0 开始，逐级递增直到不动点
//   对优先级 p，考虑所有 i < p 的算子启发式
//   每个算子取【最表达】的方向：
//       BOTH > BACKWARD == FORWARD > NONE
//   若一轮中先见 FORWARD 又见 BACKWARD -> 变成 BOTH
//
// 每一轮内部都完整跑一遍【激进传播】

// 为什么需要分批？
//   逐元素算子是"直通"的：各维一一对应，分片能无损传过去
//   dot / reduce 会改变维度关系：先让直通算子铺开，冲突更少`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '第 1 批', d: '逐元素 / 广播等<b>直通</b>算子<br>—— 维度关系最简单', c: '#4ade80' },
      { t: '第 2 批', d: 'reshape / slice 等<b>形状变换</b><br>—— 需要按因子对应', c: '#38bdf8' },
      { t: '第 3 批', d: 'dot / reduce 等<b>改变维度</b>的算子<br>—— 冲突最多的放最后', c: '#c084fc' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:238px;opacity:.34;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.34');
      msg.innerHTML = [
        '最常见、也最"听话"的一批。它们的分片能无损地在操作数与结果之间传递。',
        '形状变了，但因子对应关系还在（L2-01 讲过），仍可零通信地传。',
        '这几类最容易产生冲突（收缩维、维度变化），放到最后处理，前面的铺开已经减少了歧义。',
      ][i];
    }));
    tl.at(11200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>方向也有优先级</b>：<span class="mono">BOTH &gt; BACKWARD == FORWARD &gt; NONE</span> —— 能双向就双向。';
    });
  }
},

/* ------------------------------------------------ 2 逐元素优先 */
{
  kicker: 'L2-05 · 算子优先级',
  title: '典型效果：<span class="hl-a">逐元素赢过 dot</span>',
  sub: '当 `%arg0` 与函数结果对同一个轴有不同要求时，谁说了算？答案是<b>逐元素算子那一侧</b>。',
  caption: '这是本课最直观的一个例子：结果被锁成 <span class="mono">[{?}, {"a", ?}]</span>，逐元素链路把它反向推到了 <span class="mono">dot_general</span>。',
  code: `// %arg0 要求 "a" 切第 0 维
// 函数结果 要求 "a" 切第 1 维  <- 两者冲突
%arg0: tensor<8x8xf32> [{"a", ?}, {?}]
返回 : tensor<8x8xf32> [{?}, {"a", ?}]

%0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
%1 = stablehlo.add %0, %0
%2 = stablehlo.add %1, %1
return %2

// 结果：dot_general 也变成 [{?}, {"a", ?}]
//   逐元素(add)优先 -> 先把结果的要求反向推到 dot
//   %arg0 自己的要求【没有赢】`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:15px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:20px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const node = (label, sub, cls) => {
      const c = U.el('div', { class: 'card', style: 'padding:8px 12px;text-align:center;transition:.4s' });
      c.innerHTML = `<div class="mono" style="font-size:12px">${label}</div>
        <div class="small faint" style="font-size:10.5px">${sub}</div>`;
      demo.appendChild(c); return c;
    };
    const ar = () => demo.insertAdjacentHTML('beforeend', '<div class="arrow" style="font-size:19px">→</div>');

    tl.at(700, () => {
      const a = node('%arg0', '[{"a",?},{?}]'); a.style.borderColor = 'var(--ax0)';
      ar();
      node('dot_general', '空');
      ar();
      node('add', '空');
      ar();
      node('add', '空');
      ar();
      const r = node('return', '[{?},{"a",?}]'); r.style.borderColor = 'var(--ax3)';
      msg.innerHTML = '两端各有一个要求，<b>方向相反</b> —— 谁先传播，谁就赢。';
    });
    tl.at(4400, () => {
      const cards = demo.querySelectorAll('.card');
      [2, 3].forEach(i => { if (cards[i]) { cards[i].style.borderColor = 'var(--ok)'; cards[i].querySelector('.small').textContent = '[{?},{"a",?}]'; } });
      msg.innerHTML = '<b>第 1 批（逐元素）</b>：把结果的要求反向推到两个 <span class="mono">add</span>。';
    });
    tl.at(8000, () => {
      const cards = demo.querySelectorAll('.card');
      if (cards[1]) { cards[1].style.borderColor = 'var(--ok)'; cards[1].querySelector('.small').textContent = '[{?},{"a",?}]'; }
      msg.innerHTML = '<b>第 3 批（dot）</b>：轮到时，<span class="mono">dot_general</span> 已经能从下游拿到明确要求了。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>注意 <span class="mono">%arg0</span> 的要求没有赢</b> —— 它在第 1 批时无法越过 dot 传过去（dot 属于后面的批次）。';
    });
    tl.at(13800, () => {
      msg.innerHTML = '这就是"<b>分批</b>"的实质：它决定了冲突时<b>谁先发声</b>。';
    });
  }
},

/* ------------------------------------------------ 3 多使用者推迟 */
{
  kicker: 'L2-05 · 算子优先级',
  title: '多使用者：<span class="hl-a">推迟</span>前向传播',
  sub: '一个算子的结果若被多个下游使用，过早前向传播会让"第一个使用者的要求"绑架其它使用者。于是它<b>被推迟</b>。',
  caption: '推迟不等于不传：`sine` 最终仍拿到 `%arg0` 的分片，只是等更明确的信息出现之后。',
  code: `// %1 = sine %arg0 被【两个】add 使用
%arg0: tensor<8x8xf32> [{"a"}, {}]
%1 = stablehlo.sine %arg0 : tensor<8x8xf32>
%2 = stablehlo.add %1, %1
%3 = stablehlo.add %1, %1
// 两个返回位置都被锁成 [{}, {"a"}]

// 期望：
//   sine  -> [{"a", ?}, {?}]      （来自 %arg0）
//   add_1 -> [{?}, {"a", ?}]      （来自锁定的返回位置）
//   add_2 -> [{?}, {"a", ?}]

// 关键：sine 的前向传播被推迟
//   否则第一个 add 的要求会先影响 sine
//   进而错误地影响第二个 add`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:16px;justify-content:center;align-items:stretch" id="rows"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const rows = wrap.querySelector('#rows'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'card', style: 'width:340px;opacity:1' });
      c.innerHTML = `<div class="card-t" style="font-size:12px">传播顺序</div>
        <div class="col" style="gap:5px;margin-top:6px;font-size:11.5px">
          <div class="mono faint">%1 = sine %arg0   ← 2 个使用者</div>
          <div class="mono faint">%2 = add %1, %1</div>
          <div class="mono faint">%3 = add %1, %1</div>
        </div>`;
      rows.appendChild(c);
      msg.innerHTML = '<span class="mono">sine</span> 的结果有 <b>2 个使用者</b> —— 这是触发"推迟"的条件。';
    });
    tl.at(4200, () => {
      msg.innerHTML = '如果立刻前向传播：第一个 <span class="mono">add</span> 的要求会先传到 <span class="mono">sine</span>，再被第二个 <span class="mono">add</span> 继承 —— <b>两个使用者被强行统一</b>。';
    });
    tl.at(7800, () => {
      msg.innerHTML = '<b>推迟后</b>：两个 <span class="mono">add</span> 各自从自己锁定的返回位置拿到要求，互不干扰。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '最终 <span class="mono">sine</span> 从 <span class="mono">%arg0</span> 拿到 <span class="mono">[{"a", ?}, {?}]</span> —— 它<b>也传到了</b>，只是晚了一步。';
    });
    tl.at(13400, () => {
      msg.innerHTML = '<b>对称的情形</b>：一个操作数被多个算子使用时，<b>反向</b>传播同样会被推迟。';
    });
  }
},

/* ------------------------------------------------ 4 因子类型优先级 */
{
  kicker: 'L2-05 · 算子优先级',
  title: 'pass-through 因子<span class="hl-a">优先于</span> reduction 因子',
  sub: '冲突发生在不同类型因子上时，算子优先级规则也给出了取舍方向。',
  caption: '这与"先让直通算子铺开"是同一个思路：<b>直通的（pass-through）优先</b>，需要归约的往后排。',
  code: `// 因子映射：([i, k], [k, j]) -> ([i, j])
//   %arg1 = (k, j)，它要求 "a" 切 k（【收缩/归约】因子）
//   函数结果 = (i, j)，它要求 "a" 切 i（【直通】因子）
%arg1: tensor<1024x16xf32> [{"a", ?}, {"b", ?}]
返回 : tensor<32x16xf32>   [{"a", ?}, {"b", ?}]

%0 = stablehlo.dot %arg0, %arg1, precision = [DEFAULT, DEFAULT]

// 结果：采用函数结果那一侧
//   dot -> [{"a", ?}, {"b", ?}]
// 即 pass-through 因子的要求赢了`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:15px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:28px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:10px;align-items:center' });
      c.innerHTML = `
        <div class="row" style="gap:16px;align-items:center">
          <div class="card" style="width:220px;border-color:rgba(251,191,36,.5)">
            <div class="card-t" style="color:var(--warn);font-size:12px">因子上：reduction</div>
            <div class="card-d" style="font-size:11.5px">%arg1 的 "a" 切收缩因子 k</div>
          </div>
          <div style="color:var(--ok);font-size:22px">VS</div>
          <div class="card" style="width:220px;border-color:rgba(74,222,128,.5)">
            <div class="card-t" style="color:var(--ok);font-size:12px">因子上：pass-through</div>
            <div class="card-d" style="font-size:11.5px">结果的 "a" 切直通因子 i</div>
          </div>
        </div>`;
      demo.appendChild(c);
      msg.innerHTML = '同一个轴 <span class="mono">"a"</span>，落在<b>两种不同性质</b>的因子上。';
    });
    tl.at(4400, () => {
      msg.innerHTML = '<b>pass-through 因子赢</b>：结果采用 <span class="mono">[{"a", ?}, {"b", ?}]</span> —— 函数结果那一侧。';
    });
    tl.at(7800, () => {
      msg.innerHTML = '<b>直觉</b>：直通因子的分片可以无损传递；归约因子的分片会带来通信（需要 all-reduce）。优先满足前者成本更低。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '这与"逐元素优先"是同一条设计思想：<b>把便宜的、无损的先定下来</b>。';
    });
  }
},

/* ------------------------------------------------ 5 方向优先级 */
{
  kicker: 'L2-05 · 算子优先级',
  title: '方向优先级：<span class="mono hl-a">BOTH &gt; FWD == BWD &gt; NONE</span>',
  sub: '同一个算子在不同轮次可能被要求前向或反向传播。规则是：<b>能双向就双向</b>，取最表达的方向。',
  caption: '实现上的细节：一轮中若先见 <span class="mono">FORWARD</span> 又见 <span class="mono">BACKWARD</span>（或反之），方向就升级为 <span class="mono">BOTH</span>。',
  code: `// broadcast 的例子：前向优先级高于反向
%arg0: tensor<32xf32> [{"a"}]
返回 : tensor<32x16x8xf32> [{}, {"a"}, {}]

%0 = stablehlo.broadcast_in_dim %arg0, dims = [0]
     : (tensor<32xf32>) -> tensor<32x16xf32>
%1 = stablehlo.broadcast_in_dim %0, dims = [0, 1]
     : (tensor<32x16xf32>) -> tensor<32x16x8xf32>

// 期望：
//   broadcast_1 -> [{?}, {"a", ?}]
//   broadcast_2 -> [{?}, {"a", ?}, {?}]
// 即结果的要求沿【前向】逐级传递，
// 而不是把 %arg0 的 "a" 沿反向推过去`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:8px;align-items:center' });
      c.innerHTML = `
        <div class="row" style="gap:10px;align-items:center">
          <div class="chip c0" style="padding:7px 12px">%arg0 {"a"}</div>
          <div style="color:var(--ok);font-size:20px">→</div>
          <div class="chip mut" style="padding:7px 12px">bc1</div>
          <div style="color:var(--ok);font-size:20px">→</div>
          <div class="chip mut" style="padding:7px 12px">bc2</div>
          <div style="color:var(--ok);font-size:20px">→</div>
          <div class="chip c3" style="padding:7px 12px">return [{},{"a"},{}]</div>
        </div>
        <div class="small faint">前向优先级更高 → 沿这个方向逐级传</div>`;
      demo.appendChild(c);
      msg.innerHTML = '结果被锁成 <span class="mono">[{}, {"a"}, {}]</span>，<span class="mono">%arg0</span> 也要求 <span class="mono">"a"</span> —— 两边不矛盾。';
    });
    tl.at(4400, () => {
      msg.innerHTML = 'broadcast 的<b>前向优先级更高</b>，所以要求沿前向逐级流动：<span class="mono">[{?}, {"a", ?}]</span> → <span class="mono">[{?}, {"a", ?}, {?}]</span>。';
    });
    tl.at(7800, () => {
      msg.innerHTML = '结果 <span class="mono">%arg0</span> 的 <span class="mono">"a"</span> 也保住了（落在第 1 维）—— 两个方向恰好一致。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>规则速记</b>：<span class="mono">BOTH &gt; BACKWARD == FORWARD &gt; NONE</span>；一轮内同时出现前向与反向就升级为 BOTH。';
    });
  }
},

/* ------------------------------------------------ 6 约束与串联 */
{
  kicker: 'L2-05 · 算子优先级',
  title: '两个补充观察：<span class="hl-a">约束会被传播</span>、<span class="hl-a">串联 dot 逐级生效</span>',
  sub: '分片约束算子本身也是传播链上的一环；串联的同类算子会按批次依次拿到分片。',
  caption: '这两个用例说明：算子优先级不是"只挑一类算子传"，而是<b>给所有算子排了个先后</b>，谁都要轮到。',
  code: `// ① sharding_constraint 也会被传播（用例名即结论）
//   %[[WSC]] = sdy.sharding_constraint %[[ADD_2]]
//              <@mesh, [{?}, {"a", ?}]> : tensor<8x8xf32>
//   return %[[WSC]]

// ② 串联的 dot_general：批次逐级生效
%0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
%1 = stablehlo.dot_general %0, %0, contracting_dims = [1] x [0]
%2 = stablehlo.add %1, %1

// 期望：
//   DOT_2 -> [{?}, {"a", ?}]     （逐元素先推过来）
//   DOT_1 -> [{"a", ?}, {?}]     （轮到 dot 批次时才定）
//   ADD_1 -> [{?}, {"a", ?}]`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 约束算子也参与', c: '#38bdf8',
        d: '<span class="mono">sdy.sharding_constraint</span> 会拿到分片并继续向后传。<br>它不是"透明的"，而是链上的一环。' },
      { t: '② 串联 dot 逐级生效', c: '#c084fc',
        d: '先由逐元素批次把要求推到离结果最近的 <span class="mono">DOT_2</span>；<br>轮到 dot 批次时 <span class="mono">DOT_1</span> 才拿到分片。' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:12px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '传播不区分"用户标注的算子"和"编译器内部算子" —— 都按同一套优先级处理。'; });
    tl.at(4200, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>为什么 DOT_1 拿到的是 <span class="mono">[{"a", ?}, {?}]</span></b>：它在第 1 批时还轮不到，等信息更充分后才定。';
    });
    tl.at(7800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '另一个用例 <span class="mono">unknown_op</span>（<span class="mono">concatenate</span>）验证：<b>没有优先级信息的算子也能被处理</b>。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<b>实践含义</b>：算子优先级是"内置启发式"，你不需要为每个算子配置 —— 除非要写自定义规则（L2-10）。';
    });
  }
},

/* ------------------------------------------------------------ 7 练习 */
{
  kicker: 'L2-05 · 练习',
  title: '练一练：<span class="hl-a">谁先谁后</span>',
  sub: '三道题分别考：分批顺序、多使用者推迟、因子类型优先级。',
  caption: '这一课的核心是"批次决定谁先发声" —— 冲突结果往往取决于此，而不是取决于哪边"更合理"。',
  code: `// 题 1：逐元素与 dot 冲突时，谁先传播？

// 题 2：一个结果被 3 个下游使用，前向传播会怎样？

// 题 3：同一个轴，一侧落在收缩因子上、另一侧落在
//       直通因子上，谁的要求会被采纳？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '逐元素算子与 <span class="mono">dot</span> 对同一个轴有不同要求时，谁先传播？',
        a: '<b>逐元素算子先</b>。它属于更早的批次（直通算子），会先把分片铺开。' +
           '<br>等到 <span class="mono">dot</span> 批次时，往往已经能从下游拿到明确要求了。' +
           '<br><span class="dim">这就是 <span class="mono">element_wise_over_dot_general</span> 用例展示的现象。</span>'
      },
      {
        q: '一个算子的结果被 <b>3 个</b>下游使用时，它的前向传播会怎样？',
        a: '<b>被推迟</b>。避免"第一个使用者的要求"过早绑架其余使用者。' +
           '<br>推迟后，三个使用者各自从自己锁定的位置拿要求，互不干扰。' +
           '<br><span class="dim">对称地，一个<b>操作数</b>被多个算子使用时，<b>反向</b>传播也会被推迟。</span>'
      },
      {
        q: '同一个轴，一侧落在<b>收缩因子</b>上、另一侧落在<b>直通因子</b>上，谁的要求会被采纳？',
        a: '<b>直通因子那一侧</b>。这与"逐元素优先"是同一条设计思想。' +
           '<br><b>理由</b>：直通因子的分片可以无损传递；归约因子的分片会带来 all-reduce 等通信成本。' +
           '<br><span class="dim">先把便宜、无损的定下来，剩下的再谈取舍。</span>'
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
