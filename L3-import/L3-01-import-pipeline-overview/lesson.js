/* ==========================================================================
   L3-01 · import-pipeline-overview   （L3 层开篇）
   --------------------------------------------------------------------------
   覆盖：transforms/import/test/import_pipeline.mlir (165 行 / 8 用例)
   目标：看传播【之前】程序被做了哪些规范化。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 总览 */
{
  kicker: 'L3-01 · 导入流水线',
  title: 'L3 的主题：<span class="hl-a">传播之前</span>做了什么',
  sub: 'L1 讲构件、L2 讲传播。L3 回答一个问题：<b>用户写的 IR 在进入传播前，被改成了什么样？</b>',
  caption: '目的只有一个：把 IR 规范化成<b>传播友好</b>的形态 —— 消除假依赖、补齐不变量、统一表示。',
  code: `// RUN: sdy_opt %s -split-input-file -sdy-import-pipeline='dedup-functions-fully=true'

// 导入流水线的主要工作（L3 各课展开）：
//   ① 拆常量 / 拆标量      （L3-02，全计划最大文件）
//   ② 应用分片约束          （L3-03）
//   ③ 插数据流边            （L3-04 / L3-05）
//   ④ 内联函数调用          （L3-06）
//   ⑤ 提升内联 mesh         （L3-07）
//   ⑥ 清理 manual 轴        （L3-08）
//   ⑦ 规范化分片组          （L3-09）
//   ⑧ 其它清理              （L3-10）
//   ⑨ gather/scatter 批维   （L3-11）

// 本课用 8 个用例先看个大概：
//   pass 顺序 / 组 id 规范化 / manual 轴清理（3 种）/ 函数调用（3 种）`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const items = [
      { t: '拆常量', c: '#4ade80', n: 'L3-02' }, { t: '应用约束', c: '#38bdf8', n: 'L3-03' },
      { t: '插数据流边', c: '#fbbf24', n: 'L3-04/05' }, { t: '内联调用', c: '#c084fc', n: 'L3-06' },
      { t: '提升 mesh', c: '#f472b6', n: 'L3-07' }, { t: '清理 manual 轴', c: '#fb7185', n: 'L3-08' },
      { t: '规范化分片组', c: '#93c5fd', n: 'L3-09' }, { t: '其它清理', c: '#5eead4', n: 'L3-10' },
      { t: 'gather/scatter', c: '#94a3b8', n: 'L3-11' },
    ];
    const host = wrap.querySelector('#cards');
    const els = items.map(x => {
      const e = U.el('div', { class: 'card', style: `width:104px;opacity:.4;transition:.35s;border-color:${x.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:11px">${x.t}</div>
        <div class="small mono faint" style="font-size:9.5px;margin-top:3px">${x.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = 'L3 共 <b>11 课</b>，把导入期的每一步拆开讲。'; });
    tl.at(4000, () => {
      els.forEach((e, i) => { if (i !== 0 && i !== 5) e.style.opacity = '.25'; });
      msg.innerHTML = '其中 <b>拆常量</b>（1782 行）与 <b>清理 manual 轴</b>最值得细看 —— 后者本课就会看到三种情形。';
    });
    tl.at(7600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>一句话</b>：导入流水线把"用户怎么写都行"变成"传播只需处理一种形态"。';
    });
    tl.at(10600, () => {
      msg.innerHTML = '这解释了一个常见疑问：<b>为什么我的 IR 进传播后多了一堆东西</b> —— 是导入期加的。';
    });
  }
},

/* ---------------------------------------------------- 2 pass 顺序 */
{
  kicker: 'L3-01 · 导入流水线',
  title: '先插边，<span class="hl-a">再</span>应用约束',
  sub: 'pass 的顺序不是随意的 —— 顺序错了，约束就可能"贴"在错误的位置上。',
  caption: '测试开头的注释直说了意图：<span class="mono">Verifies that -apply-sharding-constraints pass is applied after -add-data_flow_edges pass</span>。',
  code: `// 输入
%0 = stablehlo.optimization_barrier %arg0 : tensor<32x96xf32>
%1 = sdy.sharding_constraint %0 <@mesh, [{}, {"a"}]> : tensor<32x96xf32>
return %1

// 期望：
// CHECK-NEXT: %[[OPT_BARRIER:.*]] = stablehlo.optimization_barrier %arg0
// CHECK-NEXT: sdy.data_flow_edge %[[OPT_BARRIER]] : tensor<32x96xf32>
// CHECK-NOT: sdy.sharding
// CHECK-NEXT: sdy.sharding_constraint

// 顺序：
//   ① optimization_barrier 是区域算子 -> 后面插 data_flow_edge（L2-08）
//   ② sharding_constraint 出现在【边之后】

// 如果反过来（先应用约束、再插边）：
//   约束可能落到边的前面 -> 传播时看不到它`,
  duration: 14000,
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
        <div class="small mono" style="color:var(--bad)">✗ 先约束、后插边</div>
        <div class="row" style="gap:7px;align-items:center">
          <div class="chip mut" style="padding:6px 11px;font-size:12px">barrier</div>
          <div style="color:var(--bad);font-size:16px">→</div>
          <div class="chip c2" style="padding:6px 11px;font-size:12px">constraint</div>
          <div style="color:var(--ink-faint);font-size:16px">→</div>
          <div class="chip c4" style="padding:6px 11px;font-size:12px">edge</div>
        </div>
        <div class="small faint">约束跑到了边的前面</div>`;
      demo.appendChild(c);
      msg.innerHTML = '如果先应用约束，它会被"贴"在 <b>边之前</b>的位置上。';
    });
    tl.at(4600, () => {
      demo.innerHTML = `
        <div class="col" style="gap:7px;align-items:center">
          <div class="small mono" style="color:var(--ok)">✓ 先插边、后约束</div>
          <div class="row" style="gap:7px;align-items:center">
            <div class="chip mut" style="padding:6px 11px;font-size:12px">barrier</div>
            <div style="color:var(--ok);font-size:16px">→</div>
            <div class="chip c4" style="padding:6px 11px;font-size:12px">edge</div>
            <div style="color:var(--ok);font-size:16px">→</div>
            <div class="chip c2" style="padding:6px 11px;font-size:12px">constraint</div>
          </div>
          <div class="small faint">约束在边之后，传播能看到它</div>
        </div>`;
      msg.innerHTML = '<b>正确顺序</b>：约束落在边上，传播时能正确读到（L2-08 讲过的结构）。';
    });
    tl.at(8800, () => {
      msg.innerHTML = '<b>为什么重要</b>：约束描述的是"某个使用者看到的分片"（L1-09）—— 位置错了，语义就变了。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>这也是测试的价值</b>：pass 顺序是<b>隐式契约</b>，只能靠这类用例锁定。';
    });
  }
},

/* ---------------------------------------------------- 3 组规范化 */
{
  kicker: 'L3-01 · 导入流水线',
  title: '分片组 id：<span class="hl-a">传递闭包合并</span> + 重编号',
  sub: '4 条 `sharding_group`、3 个不同 id，输出只剩 2 条且 id 统一为 0。',
  caption: '这就是 L2-07 讲的"组 id 先做合并与规范化，再做兼容性检查"的<b>具体形态</b>。',
  code: `// 输入：4 条 sharding_group，3 个 id
sdy.sharding_group %arg0 group_id = 1234 : tensor<8x8xf32>
sdy.sharding_group %arg0 group_id = 2345 : tensor<8x8xf32>
sdy.sharding_group %arg1 group_id = 1234 : tensor<8x8xf32>
sdy.sharding_group %arg1 group_id = 3456 : tensor<8x8xf32>

// 期望：只剩 2 条，id 都是 0
// CHECK-DAG: sdy.sharding_group %arg0 group_id=0 : tensor<8x8xf32>
// CHECK-DAG: sdy.sharding_group %arg1 group_id=0 : tensor<8x8xf32>

// 读法：
//   %arg0 与 %arg1 通过 id 1234 间接关联 -> 属于同一组
//   2345 / 3456 是各自独有的 -> 也被合并进来
//   （同一张量上的多个组 id 会合并成一个组）
//   最后把所有组 id 重编号为 0..N-1`,
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
        <div class="small mono" style="color:var(--ax0)">输入：3 个 id</div>
        <div class="row" style="gap:6px">
          <div class="chip c0" style="padding:5px 10px;font-size:11.5px">1234</div>
          <div class="chip c1" style="padding:5px 10px;font-size:11.5px">2345</div>
          <div class="chip c2" style="padding:5px 10px;font-size:11.5px">3456</div>
        </div>
        <div class="small faint">%arg0 与 %arg1 共享 1234</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<span class="mono">1234</span> 同时出现在两个张量上 —— 它们因此被<b>传递闭包</b>关联起来。';
    });
    tl.at(4600, () => {
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹', style: 'font-size:26px' }));
      const c = U.el('div', { class: 'col', style: 'gap:8px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ok)">输出：1 个组，id = 0</div>
        <div class="row" style="gap:6px">
          <div class="chip c4" style="padding:6px 12px;font-size:12px">%arg0 : group_id=0</div>
          <div class="chip c4" style="padding:6px 12px;font-size:12px">%arg1 : group_id=0</div>
        </div>
        <div class="small faint">4 条 -> 2 条</div>`;
      demo.appendChild(c);
      msg.innerHTML = '合并后只剩一个组，id 重编号为 <b>0</b>。';
    });
    tl.at(8800, () => {
      msg.innerHTML = '<b>两个动作</b>：① 传递闭包合并（谁和谁间接相关）；② 重编号为 <span class="mono">0..N-1</span>。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>为什么必须做</b>：不合并的话，同一组被拆成多个 id，传播时会各自为政（L2-07）。';
    });
    tl.at(14000, () => {
      msg.innerHTML = 'L3-09 会展开这一步的完整规则。';
    });
  }
},

/* ---------------------------------------------------- 4 manual 清理 */
{
  kicker: 'L3-01 · 导入流水线',
  title: 'manual 轴清理：<span class="hl-a">三种情形</span>',
  sub: '`manual_axes` 里列了但没切任何维度的轴，会被移进 `replicated` —— 保证 L1-07 的不变量成立。',
  caption: '不变量的要求是：<b>manual 轴必须在所有分片里显式出现</b>（切维<b>或</b>进 replicated）。导入流水线负责补齐。',
  code: `// 三种情形，规则相同（"a" 没切维度 -> 进 replicated）：

// ① in/out 是开维 {"c", ?}
in_shardings=[<@mesh, [{"c", ?}]>]  manual_axes={"c", "a"}
-> in_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>]
   并插入两条 data_flow_edge（块参数上、结果上）

// ② in/out 是【全闭】的 {"c"}
in_shardings=[<@mesh, [{"c"}]>]  manual_axes={"c", "a"}
-> 提升到函数参数上：
   %arg0: {sdy.sharding = #sdy.sharding<@mesh, [{"c"}, {}], replicated={"a"}>}
   （测试注释：Due to the in_sharding being fully closed, the in_sharding
     is added to the func arg but with the manual axis added as replicated.）

// ③ 清理发生在【插边之前】
in_shardings=[<@mesh, [{?}]>]  manual_axes={"a"}
-> in_shardings=[<@mesh, [{?}], replicated={"a"}>]
   （测试注释：This test verifies that the manual axes are cleaned up
     before adding data flow edges.）`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 开维分片', c: '#4ade80',
        d: '把 <span class="mono">"a"</span> 加进 <span class="mono">replicated</span>，<br>并插入两条 data flow edge。' },
      { t: '② 全闭分片', c: '#38bdf8',
        d: '除加 replicated 外，还把分片<br><b>提升到函数参数</b>上。' },
      { t: '③ 顺序约束', c: '#fbbf24',
        d: '清理必须发生在<b>插边之前</b>——<br>否则边上的分片会不完整。' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.34;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.34');
      msg.innerHTML = [
        '最常见的情形。<span class="mono">"c"</span> 切维度保留，<span class="mono">"a"</span> 没切所以进 replicated。',
        '<b>为什么提升到参数</b>：分片是全闭的，说明它就是"这个张量本身的分片"，可以直接写回参数。',
        '顺序很关键 —— 与第 2 幕的"先插边后约束"是同一类<b>隐式契约</b>。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>统一规则</b>：manual 轴要么切维、要么进 <span class="mono">replicated</span> —— 导入期负责补齐（L1-07 的不变量）。';
    });
  }
},

/* ---------------------------------------------------- 5 函数调用 */
{
  kicker: 'L3-01 · 导入流水线',
  title: '函数调用：<span class="hl-a">三种调用图形态</span>',
  sub: '测试用 `dedup-functions-fully=true` 跑，并用三个用例锁定不同调用图下函数**保持原样**的行为。',
  caption: '注意注释里的关键词 <span class="mono">test: non_flat</span> —— 指的是"调用图不是扁平的一层"。',
  code: `// ① 单次调用
%0 = call @foo(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
// @foo 的参数上带分片标注，被保留

// ② 两次调用同一个函数（注释：test: non_flat. two calls on foo.）
%[[CALL0]] = call @foo(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
%[[CALL1]] = call @foo(%arg0) : (tensor<8xf32>) -> tensor<8xf32>

// ③ 链式调用（注释：main calls foo and bar. foo calls bar.）
func.func @main(...) {
  %[[CALL0]] = call @foo(
  %[[CALL1]] = call @bar(
}
func.func private @foo(%arg0: tensor<8xf32>
    {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) -> tensor<8xf32> {
  %[[CALL]] = call @bar(
}
func.func private @bar(%arg0: tensor<8xf32>
    {sdy.sharding = #sdy.sharding<@mesh, [{"b"}]>}) -> tensor<8xf32> { ... }

// 三个用例的共同点：函数【没有被展平或内联】
//   参数上的分片标注原样保留
//   -> dedup-functions-fully 影响的是去重策略，不是"一定要展平"`,
  duration: 16000,
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
        <div class="small mono" style="color:var(--ax0)">① 单次</div>
        <div class="row" style="gap:7px;align-items:center">
          <div class="chip c0" style="padding:6px 12px;font-size:12px">main</div>
          <div style="color:var(--ink-faint)">→</div>
          <div class="chip mut" style="padding:6px 12px;font-size:12px">foo</div>
        </div>`;
      demo.appendChild(c);
      msg.innerHTML = '最简单的形态：<span class="mono">main</span> 调用 <span class="mono">foo</span> 一次。';
    });
    tl.at(4000, () => {
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ax1)">② 两次</div>
        <div class="row" style="gap:7px;align-items:center">
          <div class="chip c0" style="padding:6px 12px;font-size:12px">main</div>
          <div style="color:var(--ax1);font-size:15px">⇉</div>
          <div class="chip mut" style="padding:6px 12px;font-size:12px">foo</div>
        </div>
        <div class="small faint">同一个函数被调用两次</div>`;
      demo.appendChild(c);
      msg.innerHTML = '同一函数被调两次 —— 这时"要不要去重"就成了问题。';
    });
    tl.at(7600, () => {
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ax2)">③ 链式</div>
        <div class="row" style="gap:6px;align-items:center">
          <div class="chip c0" style="padding:6px 12px;font-size:12px">main</div>
          <div style="color:var(--ink-faint)">→</div>
          <div class="chip mut" style="padding:6px 12px;font-size:12px">foo</div>
          <div style="color:var(--ink-faint)">→</div>
          <div class="chip mut" style="padding:6px 12px;font-size:12px">bar</div>
        </div>
        <div class="small faint">调用图不是扁平的</div>`;
      demo.appendChild(c);
      msg.innerHTML = '链式调用 —— 注释里叫 <span class="mono">non_flat</span>（调用图有多层）。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>三个用例的共同结论</b>：函数<b>没有被展平或内联</b>，参数上的分片标注原样保留。';
    });
    tl.at(13800, () => {
      msg.innerHTML = '完整的函数调用处理在 <b>L3-06</b>（<span class="mono">-sdy-import-func-calls</span>）展开。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L3-01 · 练习',
  title: '练一练：<span class="hl-a">导入期做了什么</span>',
  sub: '三道题分别考：pass 顺序、组规范化、manual 轴清理。',
  caption: '记住一句话：导入流水线把"用户怎么写都行"变成"传播只需处理一种形态"。',
  code: `// 题 1：为什么必须"先插数据流边、再应用分片约束"？

// 题 2：4 条 sharding_group、3 个 id，输出为什么只剩 2 条且 id=0？

// 题 3：manual_axes={"c","a"} 但只有 "c" 切了维度，
//       导入流水线会怎么处理 "a"？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '为什么必须"先插数据流边、再应用分片约束"？',
        a: '因为约束描述的是"<b>某个使用者看到的分片</b>"（L1-09）—— 它需要贴在<b>正确的值</b>上。' +
           '<br>区域算子（如 <span class="mono">optimization_barrier</span>）后面要先插出 <span class="mono">data_flow_edge</span>，约束才能落在边之后。' +
           '<br><span class="dim">顺序反了，约束可能落到边的前面，传播时就看不到它了。</span>'
      },
      {
        q: '4 条 <span class="mono">sharding_group</span>、3 个 id，输出为什么只剩 2 条且 id=0？',
        a: '因为做了<b>两件事</b>：① <b>传递闭包合并</b> —— <span class="mono">%arg0</span> 与 <span class="mono">%arg1</span> 通过共享的 id 1234 间接关联，加上各自独有的 2345/3456，最终合成一个组；② <b>重编号</b>为 <span class="mono">0..N-1</span>。' +
           '<br><span class="dim">不合并的话，同一组会被拆成多个 id，传播时各自为政（L2-07）。L3-09 会展开完整规则。</span>'
      },
      {
        q: '<span class="mono">manual_axes={"c","a"}</span> 但只有 <span class="mono">"c"</span> 切了维度，导入流水线会怎么处理 <span class="mono">"a"</span>？',
        a: '把 <span class="mono">"a"</span> 移进 <span class="mono">replicated</span>。' +
           '<br><b>理由</b>：L1-07 的不变量要求 manual 轴必须在所有分片里<b>显式出现</b> —— 要么切维、要么进 <span class="mono">replicated</span>。' +
           '<br><span class="dim">若分片是全闭的，还会把分片<b>提升到函数参数</b>上；且这一步必须在<b>插边之前</b>完成。</span>'
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
