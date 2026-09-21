/* ==========================================================================
   L2-09 · manual-computation-propagation
   --------------------------------------------------------------------------
   覆盖：transforms/propagation/test/basic_propagation_manual_computation.mlir
         (415 行 / 27 用例)
   目标：讲透 manual_computation 在传播中的完整行为 —— 追加、移除、闭维保护。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 边界 */
{
  kicker: 'L2-09 · 手动计算的传播',
  title: '手动计算：<span class="hl-a">冻结 manual 轴</span>，但自由轴照传',
  sub: 'L1-07 讲了手动计算的语法。这一课看它在传播中的行为：<b>不是一堵墙，而是一层带过滤的膜</b>。',
  caption: '关键区分：<span class="mono">manual_axes</span> 里的轴被冻结；其余是<b>自由轴</b>，传播会继续处理，并写回 in/out_shardings。',
  code: `// RUN: sdy_opt %s -split-input-file -sdy-add-data-flow-edges \\
//        -sdy-basic-propagate -sdy-sink-data-flow-edges

// 手动计算的三个可被传播修改的位置：
//   ① in_shardings    <- 体内或外部的自由轴会被【追加】
//   ② out_shardings   <- 同上
//   ③ 体内的算子分片   <- 由 out_sharding 反向传入

// 一个不能碰的位置：
//   manual_axes 里那些【闭维】—— 传播不往里加任何轴

// 追加规则（与 L1-07 的顺序约束一致）：
//   manual 轴在前，自由轴在后
//   {"b", ?}  +  自由 {"a"}  ->  {"b", "a", ?}`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:15px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:22px;justify-content:center;align-items:stretch">
        <div class="card" style="width:300px;border-color:rgba(74,222,128,.5)">
          <div class="card-t" style="color:var(--ok)">自由轴（可传）</div>
          <div class="card-d">不在 <span class="mono">manual_axes</span> 里的轴。<br>
            传播会把它<b>追加</b>到 in/out_shardings 的 manual 轴之后。</div>
        </div>
        <div class="card" style="width:300px;border-color:rgba(251,113,133,.5)">
          <div class="card-t" style="color:var(--bad)">manual 轴（冻结）</div>
          <div class="card-d">在 <span class="mono">manual_axes</span> 里。<br>
            传播不能改它；<b>闭维</b>时连追加都不允许。</div>
        </div>
      </div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const msg = wrap.querySelector('#msg');
    tl.at(900, () => { msg.innerHTML = '所以手动计算<b>不是</b>"传播到此为止" —— 它只是把一部分轴的保护级别提高了。'; });
    tl.at(3800, () => { msg.innerHTML = 'in/out_shardings 是<b>可写</b>的：传播会把自由轴拼进去。'; });
    tl.at(6800, () => { msg.innerHTML = '但<b>闭维</b>（<span class="mono">{"b"}</span> 不带 <span class="mono">?</span>）是硬承诺，传播不会碰。'; });
    tl.at(9800, () => { msg.innerHTML = '本课 27 个用例，就是把这套规则的各种组合穷举了一遍。'; });
  }
},

/* ---------------------------------------------------- 2 传进体内 */
{
  kicker: 'L2-09 · 手动计算的传播',
  title: '没有 manual 轴时：<span class="hl-a">out_sharding 反向传进体内</span>',
  sub: '`manual_axes={}` 表示一个轴都没冻结 —— 此时整个区域就是普通传播区域。',
  caption: '这是理解"手动计算不阻断传播"最直接的例子。',
  code: `// manual_axes={} -> 没有轴被冻结
%0 = sdy.manual_computation(%arg0)
     in_shardings=[<@mesh, [{?}, {?}]>]
     out_shardings=[<@mesh, [{"a", ?}, {?}]>]      // <- 这里写了 "a"
     manual_axes={} (%arg1: tensor<32x32xf32>) {
  %1 = stablehlo.custom_call @sdy_testonly(%arg1)
  %2 = stablehlo.add %1, %1 : tensor<32x32xf32>
  sdy.return %2 : tensor<32x32xf32>
}

// 期望：体内的 add 拿到 [{"a", ?}, {?}]
//   stablehlo.add %1, %1 {sharding_per_value=[<@mesh, [{"a", ?}, {?}]>]}

// 读法：
//   out_shardings 是"结果应该怎么切"
//   传播把它【反向】推入体内，落到产生这个结果的算子上`,
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
        <div class="chip c3" style="padding:8px 14px">out_shardings [{"a",?},{?}]</div>
        <div style="font-size:20px;color:var(--ok)">↓ 反向</div>
        <div class="chip mut" style="padding:8px 14px">体内 add</div>
        <div class="chip c4" style="padding:6px 12px;font-size:12px">得到 [{"a",?},{?}]</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<span class="mono">manual_axes={}</span> → 没有任何轴被冻结 → 传播<b>自由进出</b>。';
    });
    tl.at(4400, () => {
      msg.innerHTML = '<b>in_shardings 是空的开维</b> <span class="mono">[{?}, {?}]</span> —— 也没有限制。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>结果</b>：<span class="mono">out_shardings</span> 的要求一路传到产生结果的 <span class="mono">add</span> 上。';
    });
    tl.at(10600, () => {
      msg.innerHTML = '<b>对比 L1-07</b>：那里有 <span class="mono">manual_axes={"a"}</span>，区域内是局部形状；这里没有，所以是全局形状。';
    });
  }
},

/* ------------------------------------------------ 3 ★ 追加机制 */
{
  kicker: 'L2-09 · 手动计算的传播',
  title: '★ 核心：自由轴被<span class="hl-a">追加</span>到 in/out_shardings',
  sub: '体内/体外的自由轴会被拼接到 manual 轴<b>之后</b>，写回 `in_shardings` 或 `out_shardings`。',
  caption: '顺序规则与 L1-07 完全一致：<b>manual 轴在前，自由轴在后</b>。',
  code: `// 初始：in/out 都只有 manual 轴 "b"
in_shardings=[<@mesh, [{"b", ?}, {?}]>]
out_shardings=[<@mesh, [{"b", ?}, {?}]>]
manual_axes={"b"}

// 【从体内追加】体内 add 的自由轴是 {"a", ?}
%1 = stablehlo.add %arg1, %arg1
     {sdy.sharding = #sdy.sharding_per_value<
        [<@mesh, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>

// 期望：in_shardings 变成 [{"b", "a", ?}, {?}]
//       out_shardings 保持 [{"b", ?}, {?}]

// 【从外部追加】外部 add 要求 [{"b", "a", ?}, {?}]
%3 = stablehlo.add %0, %0
     {sdy.sharding = #sdy.sharding_per_value<
        [<@mesh, [{"b", "a", ?}, {?}]>]>} : tensor<32x32xf32>
// 期望：out_shardings 变成 [{"b", "a", ?}, {?}]`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:14px;justify-content:center;align-items:stretch" id="rows"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const rows = wrap.querySelector('#rows'), msg = wrap.querySelector('#msg');

    const mk = (title, from, to, color) => {
      const c = U.el('div', { class: 'card', style: 'width:360px;opacity:.35;transition:.45s;border-color:' + color + '55' });
      c.innerHTML = `<div class="card-t" style="color:${color};font-size:12px">${title}</div>
        <div class="col" style="gap:6px;margin-top:7px">
          <div class="mono" style="font-size:11px;color:#93a3c4">${U.esc(from)}</div>
          <div style="color:${color};font-size:15px">↓ 追加自由轴</div>
          <div class="mono" style="font-size:11px;color:#bdf7ec">${U.esc(to)}</div>
        </div>`;
      rows.appendChild(c); return c;
    };

    tl.at(700, () => {
      mk('从体内追加', '[{"b", ?}, {?}]', '[{"b", "a", ?}, {?}]', 'var(--ax0)').style.opacity = '1';
      msg.innerHTML = '体内算子带自由轴 <span class="mono">"a"</span> → 追加到 <span class="mono">in_shardings</span>。';
    });
    tl.at(4800, () => {
      mk('从外部追加', '[{"b", ?}, {?}]', '[{"b", "a", ?}, {?}]', 'var(--ax1)').style.opacity = '1';
      msg.innerHTML = '外部算子要求 <span class="mono">{"b", "a", ?}</span> → 追加到 <span class="mono">out_shardings</span>。';
    });
    tl.at(8800, () => {
      rows.querySelectorAll('.card').forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>追加位置</b>：manual 轴<b>始终在前</b>，自由轴拼在后面 —— 这与 L1-07 的顺序约束一致。';
    });
    tl.at(12200, () => {
      msg.innerHTML = '<b>为什么这样设计</b>：局部形状由 manual 轴前缀决定（L1-07）。自由轴只能排在后面，否则局部形状就无从定义。';
    });
    tl.at(14600, () => {
      msg.innerHTML = '同族用例还覆盖多个操作数、多个结果的追加（<span class="mono">append_*_multiple_*</span>）。';
    });
  }
},

/* ------------------------------------------------ 4 移除已有轴 */
{
  kicker: 'L2-09 · 手动计算的传播',
  title: '追加前会<span class="hl-a">先移除</span>已有的自由轴',
  sub: '如果 `in_shardings` 里已经有自由轴，追加时不会简单拼接 —— 会<b>先删掉旧的</b>，避免重复。',
  caption: '用例名直说了：<span class="mono">remove_existing_free_axes_in_shardings</span>。',
  code: `// 初始 in_shardings 第 0 维 = {"b", "a", ?}
//   "b" 是 manual 轴，"a" 是【已有的自由轴】

// 体内算子的自由部分是 {"a", "c", ?}

// 结果： in_shardings = [{"b", "a", "c", ?}, {?}]
//        而不是 [{"b", "a", "a", "c", ?}, {?}]

// 读法：
//   先剥离 "b"（manual）
//   再把旧的自由轴 "a" 整个丢掉
//   最后拼接新的自由轴 "a", "c"

// 同族：remove_existing_free_axes_out_shardings
//   （同样的逻辑作用在 out_shardings 上）`,
  duration: 14000,
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
        <div class="small mono" style="color:var(--ax0)">已有 in_shardings 第 0 维</div>
        <div class="row" style="gap:5px">
          <div class="chip c1" style="padding:6px 11px">"b"</div>
          <div class="chip mut" style="padding:6px 11px">"a"</div>
          <div class="chip mut" style="padding:6px 11px">?</div>
        </div>
        <div class="small faint">manual · 旧的自由轴 · 开维</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<span class="mono">"b"</span> 是 manual 轴，<span class="mono">"a"</span> 是已经写在里面的自由轴。';
    });
    tl.at(4400, () => {
      msg.innerHTML = '<b>体内要求</b>自由部分是 <span class="mono">{"a", "c", ?}</span> —— 注意它<b>也含 "a"</b>。';
    });
    tl.at(7800, () => {
      const c = U.el('div', { class: 'col', style: 'gap:8px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ok)">结果</div>
        <div class="row" style="gap:5px">
          <div class="chip c1" style="padding:6px 11px">"b"</div>
          <div class="chip c0" style="padding:6px 11px">"a"</div>
          <div class="chip c0" style="padding:6px 11px">"c"</div>
          <div class="chip mut" style="padding:6px 11px">?</div>
        </div>
        <div class="small faint">没有重复的 "a"</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>结果</b>：<span class="mono">{"b", "a", "c", ?}</span> —— 旧的自由轴被<b>替换</b>，不是叠加。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>为什么必须移除</b>：否则会出现 <span class="mono">{"b", "a", "a", "c", ?}</span> —— 同一维上重复切两次，语义错误。';
    });
    tl.at(13600, () => {
      msg.innerHTML = '<b>规则总结</b>：manual 轴保留 + 自由部分<b>整体替换</b>。';
    });
  }
},

/* ------------------------------------------------ 5 replicated */
{
  kicker: 'L2-09 · 手动计算的传播',
  title: '`replicated` 里的 manual 轴与自由轴',
  sub: '不是所有 manual 轴都用来切维度 —— 有些只是"被冻结的复制轴"。',
  caption: '这是 L1-07 "manual 轴必须在所有分片里显式出现（切维<b>或</b>进 replicated）"的传播侧体现。',
  code: `// manual_axes 有两个轴，但只有 "b" 切了维度
%1 = sdy.manual_computation(%0)
     in_shardings=[<@mesh, [{"b", ?}, {?}], replicated={"c"}>]
     out_shardings=[<@mesh, [{"b", ?}, {?}], replicated={"c"}>]
     manual_axes={"b", "c"} (%arg1: tensor<16x32xf32>) {
  %2 = stablehlo.add %arg1, %arg1
       {sdy.sharding = #sdy.sharding_per_value<
          [<@mesh, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>
  sdy.return %2 : tensor<16x32xf32>
}

// 期望：in/out 都变成
//   [<@mesh, [{"b", "a", ?}, {?}], replicated={"c"}>]
// 即：自由轴 "a" 被追加，replicated 里的 "c" 保持不变

// 同族用例：
//   replicated_free_axes          自由轴进 replicated
//   replicated_inside_body        体内要求 replicated
//   replicated_manual_axes_directly_used_returned_from_func
//   replicated_free_axes_directly_used_returned_from_func`,
  duration: 15000,
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
        <div class="small mono" style="color:var(--ax1)">manual_axes = {"b", "c"}</div>
        <div class="row" style="gap:14px;align-items:center">
          <div class="col" style="gap:5px;align-items:center">
            <div class="chip c1" style="padding:7px 12px">"b"</div>
            <div class="small faint" style="font-size:10.5px">切维度</div>
          </div>
          <div class="col" style="gap:5px;align-items:center">
            <div class="chip c2" style="padding:7px 12px">"c"</div>
            <div class="small faint" style="font-size:10.5px">replicated</div>
          </div>
        </div>`;
      demo.appendChild(c);
      msg.innerHTML = '两个都是 manual 轴（都被冻结），但角色不同：一个<b>切维</b>，一个<b>复制</b>。';
    });
    tl.at(4600, () => {
      msg.innerHTML = '<b>为什么 "c" 也要写进 manual_axes</b>：L1-07 要求 manual 轴必须在所有分片里显式出现 —— 要么切维，要么进 <span class="mono">replicated</span>。';
    });
    tl.at(8400, () => {
      msg.innerHTML = '<b>传播结果</b>：自由轴 <span class="mono">"a"</span> 被追加进维度分片，<span class="mono">replicated={"c"}</span> 保持不变。';
    });
    tl.at(11600, () => {
      msg.innerHTML = '<b>自由轴也可以进 replicated</b>（用例 <span class="mono">replicated_free_axes</span>）—— 那表示"这个轴在区域内被复制"。';
    });
    tl.at(14000, () => {
      msg.innerHTML = '注意 <span class="mono">replicated</span> 里的轴<b>不影响局部形状</b>（L1-07 的规则）。';
    });
  }
},

/* ------------------------------------------------ 6 闭维保护 */
{
  kicker: 'L2-09 · 手动计算的传播',
  title: '闭维保护：<span class="hl-a">不往里追加任何轴</span>',
  sub: '如果那一维是闭维（不带 `?`），传播<b>完全不会碰它</b> —— 即使体内算子带了自由轴。',
  caption: '这是"用户硬承诺"优先于"传播想推"的又一例（与 L2-01 的闭维规则一脉相承）。',
  code: `// 注意第 0 维是 {"b"} —— 闭维，没有 ?
in_shardings=[<@mesh, [{"b"}, {?}]>]
out_shardings=[<@mesh, [{"b"}, {?}]>]
manual_axes={"b"}

// 体内算子带自由轴 "a"
%1 = stablehlo.add %arg1, %arg1
     {sdy.sharding = #sdy.sharding_per_value<
        [<@mesh, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>

// 期望：in/out 完全没有变化
//   in_shardings=[<@mesh, [{"b"}, {?}]>]
//   out_shardings=[<@mesh, [{"b"}, {?}]>]

// 三个方向的尝试都被拦住：
//   dont_propagate_into_closed_dim_from_inside        体内 -> in_sharding
//   dont_propagate_into_out_sharding_closed_dim_from_outside
//   dont_propagate_into_in_sharding_closed_dim_from_outside
// 以及 preserve_untouched_closed_dim`,
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
        <div class="row" style="gap:6px;align-items:center">
          <div class="chip c3" style="padding:7px 12px">{"b"}</div>
          <div class="chip mut" style="padding:7px 12px">{?}</div>
        </div>
        <div class="small faint">第 0 维【闭】 · 第 1 维【开】</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>闭维</b> <span class="mono">{"b"}</span> 表示"这一维就按 b 切，不许再加"。';
    });
    tl.at(4200, () => {
      demo.appendChild(U.el('div', { class: 'arrow', html: '⤫', style: 'font-size:26px;color:var(--bad)' }));
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="chip mut" style="padding:7px 12px">体内要求 {"a", ?}</div>
        <div class="small" style="color:var(--bad);font-size:11.5px">被拦住</div>`;
      demo.appendChild(c);
      msg.innerHTML = '体内算子虽然带了自由轴 <span class="mono">"a"</span>，但<b>推不进去</b>。';
    });
    tl.at(7800, () => {
      msg.innerHTML = '<b>结果</b>：<span class="mono">in_shardings</span> / <span class="mono">out_shardings</span> <b>完全没变</b>。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>对比第 1 幕</b>：开维 <span class="mono">{"b", ?}</span> 会被追加成 <span class="mono">{"b", "a", ?}</span>；闭维 <span class="mono">{"b"}</span> 一点都不动。';
    });
    tl.at(13200, () => {
      msg.innerHTML = '<b>三个用例</b>分别从体内、out_sharding 外部、in_sharding 外部尝试 —— 全被拦住。';
    });
  }
},

/* ------------------------------------------------------------ 7 练习 */
{
  kicker: 'L2-09 · 练习',
  title: '练一练：<span class="hl-a">预测 in/out_shardings</span>',
  sub: '三道题分别考：追加规则、移除规则、闭维保护。',
  caption: '一句话总结：<b>manual 轴保留 + 自由部分整体替换；闭维什么都不动</b>。',
  code: `// 题 1：in_shardings 初始 [{"b", ?}, {?}]，
//       体内自由轴 {"a", ?}，结果是什么？

// 题 2：in_shardings 初始 [{"b", "a", ?}, {?}]，
//       体内自由轴 {"a", "c", ?}，结果是什么？

// 题 3：in_shardings 初始 [{"b"}, {?}]（闭维），
//       体内自由轴 {"a", ?}，结果是什么？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">in_shardings = [{"b", ?}, {?}]</span>，体内自由轴是 <span class="mono">{"a", ?}</span>，结果是什么？',
        a: '<span class="mono">[{"b", "a", ?}, {?}]</span><br>' +
           '<b>规则</b>：manual 轴 <span class="mono">"b"</span> 保留在前，自由轴 <span class="mono">"a"</span> 追加在后。' +
           '<br><span class="dim">这正是用例 <span class="mono">append_in_sharding_from_inside</span>。</span>'
      },
      {
        q: '<span class="mono">in_shardings = [{"b", "a", ?}, {?}]</span>，体内自由轴是 <span class="mono">{"a", "c", ?}</span>，结果是什么？',
        a: '<span class="mono">[{"b", "a", "c", ?}, {?}]</span> —— <b>不是</b> <span class="mono">{"b", "a", "a", "c", ?}</span>。' +
           '<br><b>规则</b>：追加前会<b>先移除已有的自由轴</b>，自由部分整体替换。' +
           '<br><span class="dim">用例名就叫 <span class="mono">remove_existing_free_axes_in_shardings</span>。</span>'
      },
      {
        q: '<span class="mono">in_shardings = [{"b"}, {?}]</span>（第 0 维是闭维），体内自由轴是 <span class="mono">{"a", ?}</span>，结果是什么？',
        a: '<b class="badge bad">完全不变</b>：<span class="mono">[{"b"}, {?}]</span>。' +
           '<br><b>规则</b>：闭维是用户的硬承诺，传播<b>不往里追加任何轴</b>。' +
           '<br><span class="dim">对照：开维 <span class="mono">{"b", ?}</span> 会被追加成 <span class="mono">{"b", "a", ?}</span>。三个 <span class="mono">dont_propagate_into_*_closed_dim_from_*</span> 用例覆盖了三个方向的尝试。</span>'
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
