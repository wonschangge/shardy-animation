/* ==========================================================================
   L4-01 · export-pipeline-overview   （L4 层开篇）
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/export_pipeline.mlir (146 行 / 12 用例)
         transforms/export/test/export_pipeline_explicit_collectives.mlir (182 / 11)
   目标：讲导出流水线全貌与两条分支。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 定位 */
{
  kicker: 'L4-01 · 导出流水线',
  title: 'L4 的主题：<span class="hl-a">分片怎么落地</span>',
  sub: 'L3 讲"传播之前做了什么"。L4 讲**传播之后**：那些 `sdy.reshard` 最终变成什么？',
  caption: '回顾 L1-10 的链条：<span class="mono">constraint</span> → 传播消费 → <span class="mono">reshard</span> → <b>collective</b>。L4 就是最后一步。',
  code: `// 文件 1：默认导出流水线
// RUN: sdy_opt %s -split-input-file -sdy-add-data-flow-edges -sdy-export-pipeline

// 文件 2：显式插入集合通信
// RUN: sdy_opt %s -split-input-file -sdy-export-pipeline='\\
//        enable-insert-explicit-collectives=true \\
//        remove-all-gather-reduce-scatter-for-cmv1=true \\
//        mark-partial-result-with-unreduced-axes=true'

// 导出流水线做什么：
//   ① 对齐函数边界上的分片
//   ② 为每个操作数计算它真正需要的分片
//   ③ 处理自由轴（含不可整除 -> 子轴）
//   ④ 把 reshard 翻译成集合通信
//   ⑤ 【融合】：reduce + reshard -> reduce_scatter

// ⚠ pass 顺序契约（文件 1 的 NOTE）：
//   必须先跑 -sdy-add-data-flow-edges（L3-04）
//   因为导出流水线内部的 -sdy-sink-data-flow-edges 需要边已存在`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 对齐边界', c: '#38bdf8', d: '函数参数与结果<br>的分片必须自洽' },
      { t: '② 计算分片', c: '#4ade80', d: '为每个操作数<br>算出真正需要的分片' },
      { t: '③ 处理自由轴', c: '#fbbf24', d: '含<b>不可整除</b><br>→ 拆子轴' },
      { t: '④ 翻译', c: '#c084fc', d: 'reshard<br>→ 集合通信' },
      { t: '⑤ 融合', c: '#fb7185', d: 'reduce + reshard<br>→ reduce_scatter' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:138px;opacity:.33;transition:.35s;border-color:' + x.c + '55;padding:9px;text-align:center' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:11px">${x.t}</div>
        <div class="card-d" style="font-size:10.5px;line-height:1.45">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '与 L3-01 的"先插边、后约束"同类 —— <b>pass 顺序是隐式契约</b>。',
        'L3-06 的 <span class="mono">out_shardings</span> 来源问题在这里收尾。',
        'L2-01 讲的"不可整除不是错误"在这里看到<b>代价</b>：需要拆子轴。',
        '这是 L1-10 链条的最后一环。',
        '<b>本课的重点</b>：融合让通信步数变少，是导出期最重要的优化。',
      ][i];
    }));
    tl.at(15400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>两个文件 = 两条分支</b>：默认分支先"整理干净"，显式分支再"翻译并融合"。';
    });
  }
},

/* ------------------------------------------------ 2 对齐与计算 */
{
  kicker: 'L4-01 · 导出流水线',
  title: '文件 1：<span class="hl-a">对齐边界</span>与<span class="hl-a">计算分片</span>',
  sub: '默认分支做的是"整理"：让分片自洽、为每个操作数算出它真正需要的分片。',
  caption: '这一课不深入细节 —— 后续 L4-02～L4-07 会按算子族展开。',
  code: `// 【用例 ③】对齐函数边界
// 输入：参数 [{"a", ?}, {"b"}]，结果 [{"a":(1)2, ?}, {"b", "c"}]  <- 不一致
%arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{"a", ?}, {"b"}]>}
返回 : tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{"a":(1)2, ?}, {"b", "c"}]>}

// 输出：两者都变成 [{}, {"b"}]
// CHECK-SAME: %arg0: ... {sdy.sharding = ...<mesh<["a"=4, "b"=4, "c"=4]>, [{}, {"b"}]>}
// CHECK-SAME: -> ... {sdy.sharding = ...<mesh<["a"=4, "b"=4, "c"=4]>, [{}, {"b"}]>}
// 为什么：函数边界上的分片必须【自洽】—— 否则调用者与被调者对不上

// 【用例 ①】为每个操作数计算分片
// 输入：6 个操作数的 in/out_shardings 全都是 [{"a", "b", "c"}]
// 输出：变成【六种不同】的分片
//   [{"a"}]  [{"a"}]  [{"a", "b":(1)2}]  [{"a", "b"}]  [{"a", "b", "c":(1)2}]  [{"a", "b", "c"}]
// 注意 "b":(1)2 / "c":(1)2 —— 【子轴】！
// 用例名点明了挑战：free_axes_non_divisible（自由轴不可整除）`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '对齐边界', c: '#38bdf8', n: '用例 ③',
        d: '参数与结果的分片<br>不一致 → 都改成<br>同一个（<span class="mono">[{}, {"b"}]</span>）' },
      { t: '计算分片', c: '#4ade80', n: '用例 ①',
        d: '6 个操作数从"全一样"<br>变成<b>六种不同</b>的分片' },
      { t: '拆子轴', c: '#fbbf24', n: '不可整除',
        d: '<span class="mono">"b":(1)2</span> / <span class="mono">"c":(1)2</span><br>因为自由轴<b>不可整除</b>' },
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
        '<b>为什么要自洽</b>：否则调用者传进来的分片与被调者期望的对不上。',
        '输入里它们被"一刀切"写成一样，导出期才<b>逐个算准</b>。',
        'L2-01 说"不可整除不是错误"—— 这里看到它的<b>代价</b>：必须用子轴表达。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>注意</b>：输出里网格是<b>内联写法</b> —— 导出阶段不像 L3-07 那样做提升。';
    });
  }
},

/* ------------------------------------------------ 3 ★ 融合 */
{
  kicker: 'L4-01 · 导出流水线',
  title: '★ 文件 2 的核心：<span class="hl-a">融合</span>',
  sub: '`reduce` + `sharding_constraint` 被翻译并**融合**成一条 `reduce_scatter` —— 而不是"先 all-reduce 再 slice"两步。',
  caption: '这是导出期<b>最重要的优化</b>：通信步数从 2 降到 1。',
  code: `// 输入
%1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1]
     : (tensor<16x8x8xf32>, tensor<f32>) -> tensor<16x8xf32>
%2 = sdy.sharding_constraint %1 <@mesh, [{"x"}, {"y"}]> : tensor<16x8xf32>

// 输出（三步）
%0 = stablehlo.reduce(...) {sharding_per_value=[<mesh<…>, [{}, {}], unreduced={"x"}>]}
//                          ^^^^^^^^^^^^^^^^^^ 归约后 "x" 上是【部分和】
%1 = sdy.reduce_scatter [{"x"}, {}] %0 out_sharding=<mesh<…>, [{"x"}, {}]>
//   ^^^^^^^^^^^^^^^^^ 一边归约一边重新切分（融合！）
%2 = sdy.all_slice [{}, {"y"}] %1 out_sharding=<mesh<…>, [{"x"}, {"y"}]>
//   ^^^^^^^^^^^ 再调整到目标分片

// 关键：
//   reduce 标上 unreduced={"x"} 表示"x 上是部分和"
//   reduce_scatter 把"all-reduce + slice"【合成一步】
//   最后 all_slice 补齐剩余的分片差异

// 对比不融合的做法：
//   all_reduce {"x"} -> 完整值 -> all_slice 调整分片    （2 步通信）
// 融合后：
//   reduce_scatter {"x"} -> all_slice                  （1 步通信 + 1 步本地调整）`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:14px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const node = (label, sub, color) => {
      const c = U.el('div', { class: 'col', style: 'gap:4px;align-items:center;opacity:0;transition:.45s' });
      c.innerHTML = `<div class="chip ${color}" style="padding:7px 12px;font-size:11.5px">${label}</div>
        <div class="small faint" style="font-size:10px;text-align:center">${sub}</div>`;
      demo.appendChild(c); return c;
    };
    const ar = () => demo.insertAdjacentHTML('beforeend', '<div class="arrow" style="font-size:18px">→</div>');

    tl.at(700, () => {
      node('reduce', 'unreduced={"x"}<br>部分是部分和', 'c0').style.opacity = '1';
      msg.innerHTML = '<b>第 1 步</b>：<span class="mono">reduce</span> 沿第 1 维归约，但 <span class="mono">"x"</span> 上留下<b>部分和</b>（标 <span class="mono">unreduced</span>）。';
    });
    tl.at(4600, () => {
      ar();
      node('reduce_scatter [{"x"},{}]', '融合：归约 + 切分', 'c2').style.opacity = '1';
      msg.innerHTML = '<b>第 2 步</b>：<span class="mono">reduce_scatter</span> 把"<b>all-reduce + slice</b>"<b>合成一步</b>。';
    });
    tl.at(8600, () => {
      ar();
      node('all_slice [{},{"y"}]', '补齐剩余差异', 'c4').style.opacity = '1';
      msg.innerHTML = '<b>第 3 步</b>：<span class="mono">all_slice</span> 把分片调整到目标 <span class="mono">[{"x"}, {"y"}]</span>。';
    });
    tl.at(12400, () => {
      msg.innerHTML = '<b>对比不融合</b>：<span class="mono">all_reduce</span> → <span class="mono">all_slice</span> 需要<b>两步通信</b>；融合后只需一步 + 一步本地调整。';
    });
    tl.at(15000, () => {
      msg.innerHTML = '<b>这就是导出期最重要的优化</b> —— 减少的是<b>跨设备通信</b>，代价最高的一环。';
    });
  }
},

/* ------------------------------------------------ 4 配对与其余 */
{
  kicker: 'L4-01 · 导出流水线',
  title: '其余机制：<span class="hl-a">配对算子</span>与<span class="hl-a">未归约</span>',
  sub: '`all_slice` 与 `all_gather` 互逆；`replicated_to_unreduced` 处理"复制 → 未归约"的转换。',
  caption: '这些都是 L1-06 讲的八个集合通信算子在导出期的<b>实际用法</b>。',
  code: `// 【配对】all_slice 与 all_gather 互逆
func.func @all_slice_all_gather(
    %arg0 : tensor<16x2xf32> {sdy.sharding = ...<@mesh, [{"y"}, {}]>})
    -> (tensor<16x2xf32> {sdy.sharding = ...<@mesh, [{}, {"x"}]>}) {
// 输出：%0 = sdy.all_slice [{}, {"x"}] %arg0 out_sharding=<mesh<…>, [{"y"}, {"x"}]>
//       ^^^^^^^^^^^ 把 "x" 加到第 1 维（从无到有）

// 【未归约】replicated_to_unreduced
// 输入：%arg0 是 [{"c"}, {}], unreduced={"b"}
//       约束要求 unreduced={"a", "b"}   <- "a" 也要变成未归约
// 输出：%0 = sdy.replicated_to_unreduced
//       return %arg0                    <- 直接返回，不经约束
// 含义："a" 原本是【复制】状态 -> 现在要变成【未归约】

// 【其它用例】
//   reshard_of_reshard              连续 reshard 的合并（L1-10 讲过）
//   all_to_all_fusion               all_to_all 的融合
//   dot_general_with_unreduced_result(+2)  未归约结果的三种延迟程度
//   reduce_unreduced_to_sharded_max / _min  归约算子为 max/min 的情形`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'all_slice ↔ all_gather', c: '#38bdf8',
        d: '互逆的一对<br><span class="mono">all_slice</span> 把轴<b>加进</b>分片<br><span class="mono">all_gather</span> 把轴<b>取出</b>' },
      { t: 'replicated_to_unreduced', c: '#fbbf24',
        d: '把<b>复制</b>状态<br>转成<b>未归约</b>状态<br><span class="dim">三种状态转换之一</span>' },
      { t: '融合系列', c: '#c084fc',
        d: '<span class="mono">reduce_scatter</span> / <span class="mono">all_to_all</span><br>把 reshard 与相邻算子<br><b>合并</b>成一步' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:11px;overflow-wrap:anywhere">${x.t}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        'L1-06 讲过八个集合通信算子 —— 这里是它们<b>在导出期的实际用法</b>。',
        '回顾 L1-02 的三种状态：<b>分片 / 复制 / 未归约</b>。它们之间可以互相转换。',
        '<b>融合的共同思路</b>：如果一条 reshard 紧跟着一个能"顺便完成它"的算子，就合并。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>后续课程</b>：L4-02 讲 reshard 插入总纲，L4-03～07 按算子族展开，L4-08 起讲各类 collective。';
    });
  }
},

/* ------------------------------------------------ 5 两条分支 */
{
  kicker: 'L4-01 · 导出流水线',
  title: '两条分支的<span class="hl-a">对照</span>',
  sub: '同一个导出流水线，两个文件只差几个选项 —— 但输出形态差别很大。',
  caption: '这也解释了为什么这个 pass 有这么多开关：<b>不同后端需要不同的输出形态</b>。',
  code: `// 【文件 1】默认分支
// RUN: -sdy-add-data-flow-edges -sdy-export-pipeline
// 输出特点：
//   多为 reshard / 分片更新 / replicated_to_unreduced
//   集合通信算子较少
// 关注点：把 IR "整理干净" —— 分片自洽、自由轴处理好

// 【文件 2】显式 collective 分支
// RUN: -sdy-export-pipeline='enable-insert-explicit-collectives=true
//        remove-all-gather-reduce-scatter-for-cmv1=true
//        mark-partial-result-with-unreduced-axes=true'
// 输出特点：
//   显式的 sdy.reduce_scatter / all_slice / all_gather
// 关注点：翻译 + 【融合】

// 三个选项的作用：
//   enable-insert-explicit-collectives      显式插入集合通信算子
//   remove-all-gather-reduce-scatter-for-cmv1  为 CMV1 移除 all-gather/reduce-scatter
//   mark-partial-result-with-unreduced-axes 用 unreduced 轴标记部分结果

// 一句话：
//   默认分支先"整理干净"，显式分支再"翻译并融合"`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '默认分支', c: '#38bdf8', tag: '文件 1',
        d: '多为 <span class="mono">reshard</span> / 分片更新<br>集合通信较少<br><b>整理干净</b>' },
      { t: '显式 collective', c: '#fb7185', tag: '文件 2',
        d: '显式的 <span class="mono">reduce_scatter</span> /<br><span class="mono">all_slice</span> / <span class="mono">all_gather</span><br><b>翻译 + 融合</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t} <span class="faint small">${x.tag}</span></div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>文件 1</b>的输出里还能看到 <span class="mono">reshard</span> —— 说明它还没走到"翻译"那一步。'; });
    tl.at(4400, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>文件 2</b>把 reshard 翻译并融合成了具体的集合通信。';
    });
    tl.at(8200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>为什么有这么多开关</b>：不同后端需要不同的输出形态 —— 有的能自己处理 all-gather，有的需要显式算子。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>学习建议</b>：先掌握文件 2（显式形态更直观），再回头看文件 1 的"整理"动作。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L4-01 · 练习',
  title: '练一练：<span class="hl-a">导出流水线在做什么</span>',
  sub: '三道题分别考：与 L3 的分工、融合、两条分支。',
  caption: '一句话总结：<b>默认分支整理，显式分支翻译并融合</b>。',
  code: `// 题 1：L3 与 L4 的分工分别是什么？

// 题 2：reduce + sharding_constraint 被融合成了什么？
//       相比不融合省了什么？

// 题 3：为什么导出流水线有这么多开关？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col compact-ex', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: 'L3 与 L4 的分工分别是什么？',
        a: '<b>L3（导入）</b>：传播<b>之前</b>的规范化 —— 消除假依赖、补齐不变量、统一表示。<br>' +
           '<b>L4（导出）</b>：传播<b>之后</b>的落地 —— 对齐边界、计算分片、把 <span class="mono">reshard</span> 翻译成集合通信。' +
           '<br><span class="dim">一句话：L3 把"用户怎么写都行"变成"传播只需处理一种形态"；L4 把"分片意图"变成"具体通信"。</span>'
      },
      {
        q: '<span class="mono">reduce</span> + <span class="mono">sharding_constraint</span> 被融合成了什么？相比不融合省了什么？',
        a: '融合成 <b><span class="mono">sdy.reduce_scatter</span></b>（再补一条 <span class="mono">all_slice</span>）。<br>' +
           '<b>不融合</b>：<span class="mono">all_reduce</span>（归约）→ <span class="mono">all_slice</span>（调整分片）= <b>两步通信</b>。<br>' +
           '<b>融合后</b>：<span class="mono">reduce_scatter</span> 把"归约 + 切分"<b>合成一步</b>。' +
           '<br><span class="dim">省下的是<b>跨设备通信</b> —— 代价最高的一环。</span>'
      },
      {
        q: '为什么导出流水线有这么多开关？',
        a: '因为<b>不同后端需要不同的输出形态</b>。' +
           '<br>例如 <span class="mono">enable-insert-explicit-collectives</span> 决定是否显式插入集合通信算子；' +
           '<span class="mono">remove-all-gather-reduce-scatter-for-cmv1</span> 针对特定后端移除某些算子。' +
           '<br><span class="dim">有的后端能自己处理 all-gather，有的需要显式算子 —— 开关让同一条流水线适配不同目标。</span>'
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
