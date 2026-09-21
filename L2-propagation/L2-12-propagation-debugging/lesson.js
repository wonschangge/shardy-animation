/* ==========================================================================
   L2-12 · propagation-debugging   （L2 收官课）
   --------------------------------------------------------------------------
   覆盖：transforms/propagation/debugging/test/sharding_origins.mlir (758)
         transforms/propagation/debugging/test/edge_shardings.mlir (402)
   目标：讲透两个调试属性，并做成可点击的"分片来源追溯"。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 总览 */
{
  kicker: 'L2-12 · 传播调试',
  title: '两个调试属性：<span class="hl-a">来源</span>与<span class="hl-a">路径</span>',
  sub: '传播结果不合预期时，最想知道的是：<b>这个轴是谁决定的？它是怎么传过来的？</b>',
  caption: '两个开关要<b>在流水线的每一步都打开</b> —— 因为调试属性是各 pass 逐步累积的。',
  code: `// ① 分片来源：这个轴的最终值来自哪里
//   -sdy-aggressive-propagate=debug-sharding-origins=true
sdy.sharding_origins = {a = "self", c = "input: 1"}

// ② 传播路径：分片沿哪条边、第几轮流动过
//   -sdy-aggressive-propagate=debug-propagation-edge-sharding=true
sdy.propagation_edges = #sdy.propagation_edges<[
  {step-1 = [{"a" = operand-0 -> [operand-1, result-0]},
             {"b" = result-0 -> [operand-0, operand-1]}]}]>

// 注意：两个文件都用了【三步流水线】，每步都打开同一个开关：
//   -sdy-add-data-flow-edges
//   -sdy-apply-sharding-constraints=<debug>=true
//   -sdy-aggressive-propagate=<debug>=true
//   -sdy-sink-data-flow-edges="sink-<debug>=true"
// 因为属性要一路传递到最终 IR 上`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:22px;justify-content:center;align-items:stretch">
        <div class="card" style="width:320px;border-color:rgba(56,189,248,.5)">
          <div class="card-t mono" style="color:var(--ax0);font-size:12px">sdy.sharding_origins</div>
          <div class="card-d">回答：这个轴<b>来自哪里</b><br>
            <span class="mono small">self</span> / <span class="mono small">input: N</span> / <span class="mono small">output: N</span><br>
            <span class="dim small">结论 · 逐轴一个来源</span></div>
        </div>
        <div class="card" style="width:320px;border-color:rgba(192,132,252,.5)">
          <div class="card-t mono" style="color:var(--ax1);font-size:12px">sdy.propagation_edges</div>
          <div class="card-d">回答：分片<b>沿哪条路走</b><br>
            <span class="mono small">step-N</span> + <span class="mono small">来源 → [目标]</span><br>
            <span class="dim small">过程 · 可多条边 + 轮次</span></div>
        </div>
      </div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const msg = wrap.querySelector('#msg');
    tl.at(900, () => { msg.innerHTML = '一个看<b>结论</b>（快照），一个看<b>过程</b>（轨迹）—— 排查时通常两个都要。'; });
    tl.at(3800, () => { msg.innerHTML = '打开方式：把调试开关加在<b>每一步</b> pass 上（见 RUN 行）。'; });
    tl.at(7000, () => { msg.innerHTML = '这些属性只在调试时存在，正常编译不会产生 —— 所以<b>没有运行时开销</b>。'; });
    tl.at(10000, () => { msg.innerHTML = '本课最后会给一套<b>排查流程</b>，把这两个属性用在正确的位置上。'; });
  }
},

/* ------------------------------------------------ 2 origins 格式 */
{
  kicker: 'L2-12 · 传播调试',
  title: '<span class="mono hl-a">sharding_origins</span> 的三种来源',
  sub: '格式是 `{轴名 = "来源"}`，<b>逐轴</b>记录。来源只有三种取值。',
  caption: '算子上是<b>列表</b>（每个结果一项）；函数参数/结果上是<b>字典</b>（直接逐轴）。',
  code: `// 输入
%arg0: [{"a", ?}, {?}]           // 自己写了 "a"
%arg1: [{?}, {"c", ?}]           // 自己写了 "c"
%arg2: 无标注
返回 : [{?}, {"b", ?}]           // 结果写了 "b"

// 传播后（只看 sharding_origins）
%arg0: {a = "self",     c = "input: 1"}    // a 自己的；c 从 %arg1 来
%arg1: {a = "input: 0", c = "self"}        // a 从 %arg0 来；c 自己的
%arg2: {c = "input: 1", b = "output: 0"}   // c 从 %arg1；b 从结果反向来
返回 : {a = "input: 0", b = "self"}        // a 从 %arg0；b 自己的

// 算子上是列表（多结果时每项一个）：
%0 = stablehlo.add %arg0, %arg1
     {sdy.sharding_origins = [{a = "input: 0", c = "input: 1"}]}
%1 = stablehlo.dot_general %0, %arg2, ...
     {sdy.sharding_origins = [{a = "input: 0", b = "output: 0"}]}`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '"self"', c: '#4ade80',
        d: '来自<b>这个张量自己</b>的标注。<br>用户写的，或它作为参数/结果被锁定的。' },
      { t: '"input: N"', c: '#38bdf8',
        d: '来自<b>第 N 个输入/操作数</b>。<br>前向传播的痕迹。' },
      { t: '"output: N"', c: '#c084fc',
        d: '来自<b>第 N 个输出/结果</b>。<br>反向传播的痕迹。' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:242px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:13px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.35');
      msg.innerHTML = [
        '<span class="mono">"self"</span> 说明这个轴是<b>用户意图</b> —— 排查时通常不用怀疑它。',
        '<span class="mono">"input: N"</span> 说明是<b>前向</b>传来的 —— 沿着 N 号操作数往上找源头。',
        '<span class="mono">"output: N"</span> 说明是<b>反向</b>传来的 —— 沿着 N 号结果往下找源头。',
      ][i];
    }));
    tl.at(11600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>格式差异</b>：函数参数/结果是字典；算子上是列表（多结果时每项一个）。';
    });
  }
},

/* ------------------------------------------------ 3 ★ 交互追溯 */
{
  kicker: 'L2-12 · 传播调试',
  title: '★ 动手追溯：<span class="hl-a">点开每个张量看它的分片从哪来</span>',
  sub: '点击下面的张量，看它的每个轴分别来自 <span class="mono">self</span> / <span class="mono">input</span> / <span class="mono">output</span>。',
  caption: '这是排查的核心动作：<b>从一个可疑的轴出发，沿着来源一路回溯到用户标注</b>。',
  code: `// 同一个用例，四个张量的来源表：
%arg0: {a = "self",     c = "input: 1"}
%arg1: {a = "input: 0", c = "self"}
%arg2: {c = "input: 1", b = "output: 0"}
返回 : {a = "input: 0", b = "self"}

// 追溯示例：返回值的 "a" 从哪来？
//   output:0 的 a = "input: 0"
//   -> 沿函数第 0 个操作数往上
//   实际上 a 最初是 %arg0 写的（self）
//
// 追溯示例：%arg2 的 "b" 从哪来？
//   b = "output: 0" -> 来自函数结果
//   结果里 b = "self" -> 是用户在结果上写的
//   结论：%arg2 的 "b" 是【反向】推过来的`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:9px;justify-content:center;align-items:center;flex-wrap:wrap" id="nodes"></div>
      <div class="detail" id="detail" style="width:100%;min-height:132px;border:1px solid var(--line);border-radius:10px;padding:11px 14px;background:var(--bg-soft)"></div>
      <div class="formula" id="msg" style="min-height:44px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const nodes = wrap.querySelector('#nodes'), detail = wrap.querySelector('#detail'), msg = wrap.querySelector('#msg');

    const DATA = [
      { name: '%arg0', color: 'c0', axes: [['a', 'self', '自己写的'], ['c', 'input: 1', '从 %arg1 前向传来']] },
      { name: '%arg1', color: 'c1', axes: [['a', 'input: 0', '从 %arg0 前向传来'], ['c', 'self', '自己写的']] },
      { name: '%arg2', color: 'c2', axes: [['c', 'input: 1', '从 %arg1 前向传来'], ['b', 'output: 0', '从函数结果【反向】传来']] },
      { name: 'return', color: 'c3', axes: [['a', 'input: 0', '从 %arg0 前向传来'], ['b', 'self', '结果上写的']] },
    ];
    const colorOf = v => v === 'self' ? 'var(--ok)' : (v.startsWith('input') ? 'var(--ax0)' : 'var(--ax1)');

    const show = (i) => {
      const d = DATA[i];
      detail.innerHTML = `<div class="row" style="gap:8px;align-items:center;margin-bottom:7px">
          <span class="mono" style="font-size:13px">${d.name}</span>
          <span class="small faint">的 sharding_origins</span></div>
        <div class="row" style="gap:10px;flex-wrap:wrap">
          ${d.axes.map(([ax, src, note]) => `
            <div class="col" style="gap:3px;align-items:center;padding:6px 10px;border:1px solid ${colorOf(src)}44;border-radius:8px">
              <div class="mono" style="font-size:12px;color:${colorOf(src)}">${ax} = "${src}"</div>
              <div class="small faint" style="font-size:10.5px">${note}</div>
            </div>`).join('')}
        </div>`;
    };

    const btns = DATA.map((d, i) => {
      const b = U.el('div', { class: 'chip ' + d.color, style: 'padding:9px 15px;cursor:pointer;font-size:13px' });
      b.textContent = d.name;
      b.onclick = () => show(i);
      nodes.appendChild(b); return b;
    });

    tl.at(800, () => {
      msg.innerHTML = '四个张量各有一张来源表 —— <b>点击任一</b>查看（第 3 个 <span class="mono">%arg2</span> 最有意思）。';
    });
    tl.at(3000, () => { show(0); msg.innerHTML = '<span class="mono">%arg0</span>：<span class="mono">a</span> 是自己写的，<span class="mono">c</span> 是前向传来的。'; });
    tl.at(6200, () => { show(2); msg.innerHTML = '<span class="mono">%arg2</span>：<span class="mono">b</span> 标着 <span class="mono">output: 0</span> —— 它是<b>反向</b>从函数结果推过来的。'; });
    tl.at(9800, () => { show(3); msg.innerHTML = '<span class="mono">return</span>：<span class="mono">b</span> 是 <span class="mono">self</span> —— 源头在这里，是用户写的。'; });
    tl.at(13200, () => {
      show(1);
      msg.innerHTML = '<b>追溯完成</b>：<span class="mono">%arg2</span> 的 <span class="mono">b</span> ← 函数结果 ← 用户标注。这就是"顺藤摸瓜"。';
    });
  }
},

/* ------------------------------------------------ 4 edges 格式 */
{
  kicker: 'L2-12 · 传播调试',
  title: '<span class="mono hl-a">propagation_edges</span>：路径与轮次',
  sub: '格式是 `#sdy.propagation_edges<[{step-N = [{"轴" = 来源 -> [目标]}]}]>` —— 记录<b>第几轮</b>、从哪到哪。',
  caption: '注意它是<b>一对多</b>的：一次传播可以同时影响多个目标。',
  code: `%0 = stablehlo.add %arg0, %arg1 : tensor<8x8x8xf32>

// 期望：
sdy.propagation_edges = #sdy.propagation_edges<[
  {step-1 = [
    {"a" = operand-0 -> [operand-1, result-0]},
    {"b" = result-0  -> [operand-0, operand-1]},
    {"c" = operand-1 -> [operand-0, result-0]}]}]>

// 逐条读：
//   "a" = operand-0 -> [operand-1, result-0]
//       轴 a 从 %arg0 出发，同时传给了 %arg1 和结果（一对多）
//   "b" = result-0 -> [operand-0, operand-1]
//       轴 b 从【函数结果】反向传给两个操作数
//   "c" = operand-1 -> [operand-0, result-0]
//       与 "a" 对称

// 函数结果上还记录了多轮：
//   {step-0 = [{"b" = result-0 -> [operand-0]}]},
//   {step-2 = [{"a" = operand-0 -> [result-0]},
//              {"c" = operand-0 -> [result-0]}]}`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="col" style="gap:8px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const arrow = (ax, from, to, color) => {
      const c = U.el('div', { class: 'row', style: 'gap:9px;align-items:center;opacity:0;transition:.45s' });
      c.innerHTML = `
        <div class="mono" style="font-size:11.5px;color:${color};width:24px;text-align:right">"${ax}"</div>
        <div class="chip mut" style="padding:5px 10px;font-size:11.5px">${from}</div>
        <div style="color:${color};font-size:16px">→</div>
        <div class="chip mut" style="padding:5px 10px;font-size:11.5px">${to}</div>`;
      demo.appendChild(c); return c;
    };

    tl.at(700, () => {
      arrow('a', 'operand-0', '[operand-1, result-0]', 'var(--ax0)').style.opacity = '1';
      msg.innerHTML = '轴 <span class="mono">a</span> 从 <span class="mono">%arg0</span> 出发，<b>同时</b>传给了操作数 1 与结果。';
    });
    tl.at(4400, () => {
      arrow('b', 'result-0', '[operand-0, operand-1]', 'var(--ax1)').style.opacity = '1';
      msg.innerHTML = '轴 <span class="mono">b</span> 从<b>函数结果</b>反向出发 —— 注意源是 <span class="mono">result-0</span>。';
    });
    tl.at(8000, () => {
      arrow('c', 'operand-1', '[operand-0, result-0]', 'var(--ax2)').style.opacity = '1';
      msg.innerHTML = '轴 <span class="mono">c</span> 与 <span class="mono">a</span> 完全对称。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>一对多</b>是重点：传播不是"一条边传一次"，一次可以影响多个目标。';
    });
    tl.at(13800, () => {
      msg.innerHTML = '<span class="mono">step-N</span> 记录<b>轮次</b> —— 同一个张量上可能有第 0 轮和第 2 轮的记录。';
    });
  }
},

/* ------------------------------------------------ 5 对比与流程 */
{
  kicker: 'L2-12 · 传播调试',
  title: '实战：<span class="hl-a">排查流程</span>',
  sub: '两个属性各有用处 —— 先用 origins 定位可疑轴，再用 edges 看它是怎么传过来的。',
  caption: '这套流程把前面几课的工具串起来了：L2-03 看规则、L2-10 打印规则、本课看来源与路径。',
  code: `// 【第 1 步】结果不对，先看规则（L2-03 / L2-10）
//   -sdy-basic-propagate='keep-sharding-rules=true'
//   -sdy-populate-op-sharding-rules
//   规则错了，后面全错 -> 先排除

// 【第 2 步】规则没问题，看分片来源
//   -sdy-aggressive-propagate=debug-sharding-origins=true
//   找到那个"不该有分片却有"或"该有却没有"的轴
//   看它的来源是 self / input:N / output:N

// 【第 3 步】来源是 input:N，往前追
//   沿第 N 个操作数往上找 -> 直到找到 self（用户标注）
//   如果一路都是 self，说明是【用户标注本身】需要调整

// 【第 4 步】路径不清楚，看 propagation_edges
//   -sdy-aggressive-propagate=debug-propagation-edge-sharding=true
//   看这个轴从哪来、经过哪些目标、第几轮

// 【第 5 步】还不行 -> 换传播策略试
//   basic / aggressive / op-priority / user-priority 各跑一遍对比`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const steps = [
      { n: '1', t: '看规则', d: '规则错了后面全错<br>先排除', c: '#38bdf8' },
      { n: '2', t: '看来源', d: '<span class="mono">sharding_origins</span><br>定位可疑轴', c: '#4ade80' },
      { n: '3', t: '往前追', d: '沿 <span class="mono">input:N</span><br>回溯到 self', c: '#fbbf24' },
      { n: '4', t: '看路径', d: '<span class="mono">propagation_edges</span><br>轮次与目标', c: '#c084fc' },
      { n: '5', t: '换策略', d: '四种策略<br>各跑一遍对比', c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = steps.map(s => {
      const e = U.el('div', { class: 'card', style: `width:138px;opacity:.32;transition:.35s;border-color:${s.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="small mono" style="font-size:10px;color:${s.c}">第 ${s.n} 步</div>
        <div style="font-size:12px;font-weight:600;margin:4px 0">${s.t}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4">${s.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    steps.forEach((s, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.32');
      msg.innerHTML = [
        '分片是规则的推论。<b>先确认规则对</b>，否则后面全白忙（L2-03 讲过这条）。',
        '找到那个不对劲的轴，看它标着 <span class="mono">self</span> 还是 <span class="mono">input/output</span>。',
        '如果是 <span class="mono">input: N</span>，就沿第 N 个操作数往上找 —— 直到找到 <span class="mono">self</span>。',
        '路径与轮次能解释"为什么是这个轴赢了" —— 往往与传播策略的批次顺序有关（L2-05）。',
        '<b>最后一招</b>：四种策略各跑一遍对比 —— 差异本身就是线索。',
      ][i];
    }));
    tl.at(16400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>L2 完结</b>：从基础传播到调试，传播算法的完整图景已经建立。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L2-12 · 练习',
  title: '练一练：<span class="hl-a">读懂来源表</span>',
  sub: '三道题分别考：三种来源的含义、追溯方法、两个属性的分工。',
  caption: '掌握这套工具，就能独立排查传播问题了。',
  code: `// 题 1：{a = "self", c = "input: 1"} 分别是什么意思？

// 题 2：返回值的 a = "input: 0"，怎么找它的源头？

// 题 3：sharding_origins 与 propagation_edges 各回答什么问题？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">{a = "self", c = "input: 1"}</span> 分别是什么意思？',
        a: '<span class="mono">a = "self"</span>：轴 <span class="mono">a</span> 来自<b>这个张量自己</b>的标注（用户写的）。' +
           '<br><span class="mono">c = "input: 1"</span>：轴 <span class="mono">c</span> 是从<b>第 1 个输入/操作数</b>前向传过来的。' +
           '<br><span class="dim">三种取值：<span class="mono">self</span>（自己）/ <span class="mono">input: N</span>（前向）/ <span class="mono">output: N</span>（反向）。</span>'
      },
      {
        q: '返回值的 <span class="mono">a = "input: 0"</span>，怎么找它的源头？',
        a: '<b>沿第 0 个操作数往上找</b>。' +
           '<br>在这个例子里，第 0 个操作数是 <span class="mono">%arg0</span>，而 <span class="mono">%arg0</span> 的 <span class="mono">a = "self"</span> —— 源头找到了：用户写的。' +
           '<br><span class="dim">如果一路往上都是 <span class="mono">input: N</span> 而没有 <span class="mono">self</span>，说明这条链可能有问题。</span>'
      },
      {
        q: '<span class="mono">sharding_origins</span> 与 <span class="mono">propagation_edges</span> 各回答什么问题？',
        a: '<span class="mono">sharding_origins</span> 回答「这个轴的最终值<b>来自哪里</b>」—— 是<b>结论/快照</b>，逐轴一个来源。' +
           '<br><span class="mono">propagation_edges</span> 回答「分片<b>沿哪条路径</b>流动过」—— 是<b>过程/轨迹</b>，可有多条边与多个轮次。' +
           '<br><span class="dim">排查时先用前者定位可疑轴，再用后者看它是怎么传过来的。</span>'
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
