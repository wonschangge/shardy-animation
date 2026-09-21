/* ==========================================================================
   L2-06 · user-priority-propagation
   --------------------------------------------------------------------------
   覆盖：transforms/propagation/test/user_priority_propagation.mlir (451 行 / 20 用例)
   目标：讲透传播金字塔最高层 —— 用户标注的 pN 如何决定传播轮次与冲突结果。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 层级定位 */
{
  kicker: 'L2-06 · 用户优先级',
  title: '传播金字塔<span class="hl-a">最高层</span>：你说了算',
  sub: '前三层都是编译器的启发式。这一层是<b>用户标注</b>的 <span class="mono">pN</span> —— 数字越小越优先。',
  caption: '它嵌套在最外层：对每个用户优先级 <span class="mono">p</span>，完整跑一遍<b>算子优先级传播</b>。',
  code: `// RUN: sdy_opt %s -split-input-file -sdy-user-priority-propagate

// 标注方式：写在维度分片的花括号外
#sdy.sharding<@mesh, [{"a", ?}p1, {"b"}p1]>     // 两维都在 p1
#sdy.sharding<@mesh, [{"c"}p0, {?}]>            // 第 0 维在 p0
#sdy.sharding<@mesh, [{?}, {?}p1]>              // 空的【开】维在 p1

// 传播顺序：
//   第 0 轮：所有 p0 的分片先传播到全程序
//   第 1 轮：再加入 p1
//   第 2 轮：...
//   （允许跳号：p0 与 p4 共存，中间的 p1~p3 可以缺失）

// 每一轮内部都完整跑一遍【算子优先级传播】

// 与 intro 第 10 幕呼应：那里是概念，这里是完整用例`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:14px;align-items:flex-end;justify-content:center" id="chart"></div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const chart = wrap.querySelector('#chart'), msg = wrap.querySelector('#msg');
    const bars = [0, 1, 2, 4].map(p => {
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center;opacity:.3;transition:.4s' });
      c.innerHTML = `<div class="small mono">p${p}</div>
        <div style="width:78px;height:${38 + p * 14}px;border-radius:7px;background:var(--ax${p % 6});opacity:.85"></div>
        <div class="small faint" style="font-size:10.5px">第 ${p + 1} 轮</div>`;
      chart.appendChild(c); return c;
    });

    tl.at(800, () => { msg.innerHTML = '没有标注 <span class="mono">pN</span> 时，全部按 <b>p0</b> 处理（用例 <span class="mono">no_priorities</span>）。'; });
    [0, 1, 2, 3].forEach((k, i) => tl.at(3000 + i * 2400, () => {
      bars.forEach((b, j) => b.style.opacity = j <= k ? '1' : '.3');
      msg.innerHTML = [
        '第 0 轮：只传播 <b>p0</b> 的分片，铺满整张图。',
        '第 1 轮：p0 的结果保持不变，再加入 <b>p1</b>。',
        '第 2 轮：加入 <b>p2</b>（本课用例里没有，但机制上存在）。',
        '注意最后一根是 <b>p4</b> —— <b>允许跳号</b>，不影响语义。',
      ][i];
    }));
    tl.at(13200, () => {
      bars.forEach(b => b.style.opacity = '1');
      msg.innerHTML = '<b>为什么这样设计</b>：让你能精确控制"先批并行、再张量并行、最后 ZeRO"，且每一步都可预测。';
    });
  }
},

/* ------------------------------------------------ 2 逐维比较（核心） */
{
  kicker: 'L2-06 · 用户优先级',
  title: '★ 核心：优先级是<span class="hl-a">逐维</span>比较的',
  sub: '不是"整个分片二选一"，而是<b>每个维度各自比优先级</b> —— 这是最容易误解的一点。',
  caption: '结果可能是"第 0 维听你的，第 1 维听我的" <span class="mono">——</span> 两个来源各赢一维。',
  code: `// %arg0 两维都在 p1
%arg0: tensor<8x8xf32> [{"a", ?}p1, {"b"}p1]

// divide 第 0 维在 p0，第 1 维空着
divide: [{"c"}p0, {?}]

// 传播结果：[{"c", ?}, {"b", ?}]

// 逐维看：
//   第 0 维： "a"(p1)  vs  "c"(p0)   -> p0 赢 -> "c"
//   第 1 维： "b"(p1)  vs  无竞争者   -> 保留   -> "b"
//
// 所以结果既不是全用 %arg0 的，也不是全用 divide 的`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="rows"></div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const rows = wrap.querySelector('#rows'), msg = wrap.querySelector('#msg');

    const mk = (title, chips, note, color) => {
      const c = U.el('div', { class: 'card', style: 'width:238px;opacity:.35;transition:.4s;border-color:' + color + '55;text-align:center' });
      c.innerHTML = `<div class="card-t" style="font-size:12px;color:${color}">${title}</div>
        <div class="row" style="gap:8px;margin:8px 0;justify-content:center">
          ${chips.map(x => `<div class="chip ${x.cls}" style="padding:6px 11px;font-size:12.5px">${x.v}</div>`).join('')}
        </div>
        <div class="small faint" style="font-size:11.5px">${note}</div>`;
      rows.appendChild(c); return c;
    };

    tl.at(700, () => {
      mk('第 0 维', [{ v: '"a" p1', cls: 'c1' }, { v: 'vs', cls: 'mut' }, { v: '"c" p0', cls: 'c2' }],
        'p0 更优先 → <b>采用 "c"</b>', 'var(--ax2)').style.opacity = '1';
      msg.innerHTML = '第 0 维上两个来源都有要求，比较优先级：<b>p0 &lt; p1</b>。';
    });
    tl.at(4400, () => {
      mk('第 1 维', [{ v: '"b" p1', cls: 'c1' }, { v: '无竞争者', cls: 'mut' }],
        '没人争 → <b>保留 "b"</b>', 'var(--ax0)').style.opacity = '1';
      msg.innerHTML = '<span class="mono">divide</span> 的第 1 维是空的，没有竞争者。';
    });
    tl.at(8200, () => {
      rows.querySelectorAll('.card').forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>最终结果</b>：<span class="mono">[{"c", ?}, {"b", ?}]</span> —— 两维各来自不同的来源。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>实践含义</b>：你可以只对"关心的那一维"标优先级，其余维度让编译器自由发挥。';
    });
    tl.at(13800, () => {
      msg.innerHTML = '这也解释了为什么优先级写在<b>花括号外</b>（L1-02）—— 它是<b>维度</b>的属性，不是整条分片的属性。';
    });
  }
},

/* ---------------------------------------------------- 3 谁赢谁输 */
{
  kicker: 'L2-06 · 用户优先级',
  title: '两种情形：<span class="hl-a">参数更高</span> vs <span class="hl-a">结果更高</span>',
  sub: '把上一幕反过来看：当参数是 p0、下游是 p1 时，参数赢 —— 但下游自己的标注仍在它自己身上。',
  caption: '一个重要细节：<b>用户标注不会被删除</b>，只是"不向外传播"。它仍然描述着那个张量自己的分片。',
  code: `// 情形 A：参数 p1、结果 p0  -> 结果赢
%arg0: [{"a", ?}p1, {"b"}p1]
divide: [{"c"}p0, {?}]
//   -> [{"c", ?}, {"b", ?}]        第 0 维用 "c"

// 情形 B：参数 p0、结果 p1  -> 参数赢
%arg1: [{"a"}p0, {"b"}p0]
divide: [{"c", ?}p1, {?}]
//   add 链 -> [{"a", ?}, {"b", ?}]   第 0 维用 "a"
//   divide 自己 -> [{"c", ?}, {"b", ?}]
//                   ^^^ 用户标注的 "c" 仍在这里
//                       第 1 维的 "b" 从 p0 传了过来`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="rows"></div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const rows = wrap.querySelector('#rows'), msg = wrap.querySelector('#msg');

    const mk = (title, body, color) => {
      const c = U.el('div', { class: 'card', style: 'width:360px;opacity:.35;transition:.4s;border-color:' + color + '55' });
      c.innerHTML = `<div class="card-t" style="color:${color};font-size:12.5px">${title}</div>
        <div class="card-d" style="font-size:11.5px;line-height:1.65">${body}</div>`;
      rows.appendChild(c); return c;
    };

    tl.at(700, () => {
      mk('情形 A：结果优先级更高',
        '参数 <span class="mono">p1</span>，结果 <span class="mono">p0</span><br>' +
        '→ <b>结果赢</b>：<span class="mono">[{"c", ?}, {"b", ?}]</span><br>' +
        '<span class="dim">第 0 维采用 "c"</span>', 'var(--ax2)').style.opacity = '1';
      msg.innerHTML = '<span class="mono">divide</span> 是 p0，比 <span class="mono">%arg0</span> 的 p1 优先。';
    });
    tl.at(4400, () => {
      mk('情形 B：参数优先级更高',
        '参数 <span class="mono">p0</span>，结果 <span class="mono">p1</span><br>' +
        '→ <b>参数赢</b>：<span class="mono">add</span> 链得到 <span class="mono">[{"a", ?}, {"b", ?}]</span><br>' +
        '<span class="dim">但 divide 自己仍是 [{"c", ?}, {"b", ?}]</span>', 'var(--ax0)').style.opacity = '1';
      msg.innerHTML = '注意 <span class="mono">divide</span> 上的 <span class="mono">"c"</span> —— <b>它没有被删掉</b>。';
    });
    tl.at(8600, () => {
      rows.querySelectorAll('.card').forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>关键区分</b>：用户标注描述的是"这个张量自己的分片"，不是"要向外传播什么"。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '所以在情形 B 里：<span class="mono">divide</span> 保留自己的 <span class="mono">"c"</span>，同时从 p0 补上了空着的第 1 维 <span class="mono">"b"</span>。';
    });
    tl.at(14000, () => {
      msg.innerHTML = '<b>推论</b>：优先级冲突时，输的一方往往需要一次 <span class="mono">reshard</span> 来对齐 —— 这是通信的来源之一。';
    });
  }
},

/* ------------------------------------------- 4 低优先级维被继续切 */
{
  kicker: 'L2-06 · 用户优先级',
  title: '高优先级可以在低优先级的开维上<span class="hl-a">继续加轴</span>',
  sub: '`{"b", ?}p1` 表示"已按 b 切，且开放"。更高优先级的要求可以在这基础上<b>接着切</b>。',
  caption: '这体现了"开维"的价值：它不是"还没切"，而是"切了但欢迎继续"。',
  code: `// %arg0 第 0 维 = {"b", ?}p1   （已按 b 切，开维，p1）
// %1 要求    第 0 维 = {"b", "a", ?}p0
//                        ^^^^^^^ 在 b 的基础上又加了 a

// 结果：%1 -> [{"b", "a", ?}, {}]
//   轴序保持（b 在前、a 在后），只是变长了

// 对比：如果是"闭维" {"b"}p1
//   高优先级不能往里塞轴（闭维锁定）

// 同族用例：
//   different_priorities_with_closed_empty_dim
//   open_empty_dim_with_priority`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:15px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="row" style="gap:6px;align-items:center">
          <div class="chip c1" style="padding:7px 12px">"b"</div>
          <div class="chip mut" style="padding:7px 12px">?</div>
          <span class="small mono faint">p1（%arg0）</span>
        </div>
        <div style="font-size:18px;color:var(--ink-faint)">↓ 高优先级 p0 继续加轴</div>
        <div class="row" style="gap:6px;align-items:center">
          <div class="chip c1" style="padding:7px 12px">"b"</div>
          <div class="chip c0" style="padding:7px 12px">"a"</div>
          <div class="chip mut" style="padding:7px 12px">?</div>
          <span class="small mono faint">结果</span>
        </div>`;
      demo.appendChild(c);
      msg.innerHTML = '<span class="mono">{"b", ?}p1</span> 是<b>开维</b> —— 已经按 b 切了，但允许继续加轴。';
    });
    tl.at(4600, () => {
      msg.innerHTML = '<b>高优先级 p0</b> 的要求是 <span class="mono">{"b", "a", ?}</span> —— 在 b 的基础上追加 a，<b>轴序保持</b>。';
    });
    tl.at(8000, () => {
      msg.innerHTML = '如果 <span class="mono">%arg0</span> 写的是<b>闭维</b> <span class="mono">{"b"}p1</span>，就没有 <span class="mono">?</span>，高优先级也塞不进新轴。';
    });
    tl.at(11200, () => {
      msg.innerHTML = '<b>设计意图</b>：开维 = "我给了下限，上限交给编译器"。';
    });
    tl.at(13400, () => {
      msg.innerHTML = '同族还有一个用例 <span class="mono">different_priorities_with_closed_empty_dim</span>，验证闭空维的情形。';
    });
  }
},

/* ------------------------------------------------ 5 空开维的优先级 */
{
  kicker: 'L2-06 · 用户优先级',
  title: '空的<span class="hl-a">开</span>维也能标优先级：<span class="mono">{"{"}?{"}"}p1</span>',
  sub: '这一维目前没有分片轴，但标注了 p1 —— 意思是"轮到 p1 时，这一维可以被切"。',
  caption: '注意与 <span class="mono">{}</span> 的区别：<b>闭的空维不能有优先级</b>（L1-02 的不变量），因为"就是不切"这个决定没有可排序的东西。',
  code: `// %0 的两维：第 0 维空的【开】维 p1，第 1 维也是空的【开】维 p1
%0 = stablehlo.add %arg0, %arg1
     {sdy.sharding = #sdy.sharding_per_value<
        [<@mesh, [{?}, {?}p1]>]>} : tensor<8x8xf32>

// 注意第 0 维是 {?} —— 空的【开】维，【没有】标优先级
//      第 1 维是 {?}p1 —— 空的【开】维，标了 p1

// %arg0 = [{"a"}p0, {"b"}p0]   （p0 的两维）
// %arg3 = [{?}, {"c"}p0]

// 传播结果：
//   ADD_0 -> [{"a", ?}, {"b", ?}]     p0 的两维先铺开
//   ADD_1 -> [{"a", ?}, {"c", ?}]     p0 的 "c" 覆盖了第 1 维
//   divide-> [{"a", ?}, {"c", ?}]

// 对照（L1-02 不变量）：
//   {?}     空的开维  -> 可以有优先级
//   {}      空的闭维  -> 不能有优先级`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="rows"></div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const rows = wrap.querySelector('#rows'), msg = wrap.querySelector('#msg');

    const mk = (code, note, color) => {
      const c = U.el('div', { class: 'card', style: 'width:300px;opacity:.35;transition:.4s;border-color:' + color + '55' });
      c.innerHTML = `<div class="mono" style="font-size:15px;color:${color};text-align:center;margin:4px 0">${U.esc(code)}</div>
        <div class="card-d" style="font-size:11.5px;text-align:center">${note}</div>`;
      rows.appendChild(c); return c;
    };

    tl.at(700, () => {
      mk('{?}p1', '空的<b>开</b>维 + 优先级<br>→ <b class="badge ok">合法</b>：轮到 p1 时可切', 'var(--ok)').style.opacity = '1';
      msg.innerHTML = '空的<b>开</b>维表示"目前没切，但欢迎来切" —— 加上 <span class="mono">p1</span> 就是"轮到 p1 时再切"。';
    });
    tl.at(4400, () => {
      mk('{}p1', '空的<b>闭</b>维 + 优先级<br>→ <b class="badge bad">非法</b>：没有可排序的东西', 'var(--bad)').style.opacity = '1';
      msg.innerHTML = '闭维表示"这一维就是不切" —— 这个决定没有任何可排序的内容，所以不能标优先级（L1-02 的不变量）。';
    });
    tl.at(8200, () => {
      rows.querySelectorAll('.card').forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>记忆法</b>：优先级描述的是"<b>切分决策</b>的先后"。没有切分决策（空闭维）就无从排序。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '测试里还有 <span class="mono">different_priorities_with_closed_empty_dim</span>，专门验证闭空维的边界行为。';
    });
  }
},

/* ---------------------------------------------------- 6 组合与边界 */
{
  kicker: 'L2-06 · 用户优先级',
  title: '20 个用例的<span class="hl-a">族谱</span>',
  sub: '用户优先级是传播金字塔的最外层，所以它的用例大多在验证"与内层如何组合"。',
  caption: '规律：<b>用户优先级定轮次，内层策略定细节</b>。两层叠加，行为完全可预测。',
  code: `// 【基础】3 个
//   no_priorities / skipped_priorities / open_empty_dim_with_priority

// 【参数 vs 结果】6 个
//   arg_lower/higher_priority_than_return_value(+_with_replicated)
//   result_lower/higher_priority_than_arg
//   different_priorities_with_closed_empty_dim

// 【维度层面】1 个
//   dim_with_lower_priority_gets_further_sharded_by_higher

// 【来源不同的优先级】4 个
//   different_priorities_from_args / _from_ops
//   different_sharding_constraint_priorities
//   propagate_to/from_multi_result_op_with_priorities

// 【与其它构件组合】4 个
//   manual_computation_shardings_with_priority
//   manual_computation_sharding_with_low_priority
//   user_based_and_op_based
//   maximal_sharding_no_results
//   unreduced_custom_reduction_op`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '基础', n: 3, c: '#4ade80' },
      { t: '参数 vs 结果', n: 6, c: '#38bdf8' },
      { t: '维度层面', n: 1, c: '#fbbf24' },
      { t: '优先级来源', n: 4, c: '#c084fc' },
      { t: '与其它构件组合', n: 6, c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: 'width:140px;opacity:.4;transition:.35s;border-color:' + f.c + '55;padding:9px;text-align:center' });
      e.innerHTML = `<div class="card-t" style="color:${f.c};font-size:11.5px">${f.t}</div>
        <div class="big" style="font-size:22px;color:${f.c};margin-top:3px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 130)); msg.innerHTML = '<b>20 个用例</b>按主题归成 5 组 —— 最大的两组是"参数 vs 结果"与"与其它构件组合"。'; });
    tl.at(4200, () => {
      els.forEach((e, i) => { if (i !== 1 && i !== 4) e.style.opacity = '.25'; });
      msg.innerHTML = '这两组说明：<b>用户优先级几乎总是与别的机制一起出现</b> —— 单独用它反而不常见。';
    });
    tl.at(7800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>组合规律</b>：用户优先级决定"哪一轮"，内层策略（算子优先级 / 激进 / 基础）决定"这一轮怎么传"。';
    });
    tl.at(11200, () => {
      msg.innerHTML = '<b>下一课</b> L2-07 讲分片组传播 —— 那是一种"没有数据依赖也要同分片"的机制。';
    });
  }
},

/* ------------------------------------------------------------ 7 练习 */
{
  kicker: 'L2-06 · 练习',
  title: '练一练：<span class="hl-a">谁赢哪一维</span>',
  sub: '三道题分别考：逐维比较、跳号、开闭维与优先级。',
  caption: '核心只有一句：<b>优先级是维度的属性，每个维度各自比较</b>。',
  code: `// 题 1：结果第 0/1 维分别是什么？
//   %arg0: [{"a", ?}p1, {"b"}p1]
//   divide: [{"c"}p0, {?}]

// 题 2：p0 与 p4 能共存吗？

// 题 3：{}p1 合法吗？`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">%arg0 = [{"a", ?}p1, {"b"}p1]</span>，<span class="mono">divide = [{"c"}p0, {?}]</span>，结果的第 0 / 1 维分别是什么？',
        a: '<b>第 0 维 = <span class="mono">{"c", ?}</span></b>：<span class="mono">"a"</span>(p1) 与 <span class="mono">"c"</span>(p0) 竞争，<b>p0 赢</b>。<br>' +
           '<b>第 1 维 = <span class="mono">{"b", ?}</span></b>：只有 <span class="mono">"b"</span>(p1) 一个要求，没有竞争者，<b>保留</b>。' +
           '<br><span class="dim">结果 <span class="mono">[{"c", ?}, {"b", ?}]</span> —— 两维来自不同来源，这正是"逐维比较"的体现。</span>'
      },
      {
        q: '<span class="mono">p0</span> 与 <span class="mono">p4</span> 能共存吗？中间的 <span class="mono">p1~p3</span> 缺失有影响吗？',
        a: '<b class="badge ok">能共存，无影响</b>。优先级<b>允许跳号</b>（L1-02 的不变量）。' +
           '<br>传播时按"存在的优先级从小到大"逐轮进行，缺号的轮次直接跳过。' +
           '<br><span class="dim">用例 <span class="mono">skipped_priorities</span> 就是这个场景：p1 的要求赢了 p4 的要求。</span>'
      },
      {
        q: '<span class="mono">{}p1</span>（空的闭维 + 优先级）合法吗？那 <span class="mono">{?}p1</span> 呢？',
        a: '<span class="mono">{}p1</span> <b class="badge bad">非法</b>：闭维表示"这一维就是不切"，没有可排序的切分决策。' +
           '<br><span class="mono">{?}p1</span> <b class="badge ok">合法</b>：开维表示"目前没切，但欢迎来切"，p1 说明"轮到 p1 时再切"。' +
           '<br><span class="dim">记忆法：优先级描述的是<b>切分决策</b>的先后；没有决策就无从排序。</span>'
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
