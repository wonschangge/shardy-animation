/* ==========================================================================
   L3-06 · import-func-calls
   --------------------------------------------------------------------------
   覆盖：transforms/import/test/import_func_calls.mlir (634 行 / 21 用例)
         transforms/import/test/import_func_calls_add_data_flow_edges_on_
           named_computations_false.mlir (572 / 21)
         transforms/import/test/flatten_call_graph.mlir (415 / 13)
   目标：讲透"把函数调用内联成 named_computation"这个动作。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 核心变换 */
{
  kicker: 'L3-06 · 内联函数调用',
  title: '核心变换：<span class="mono hl-a">call</span> → <span class="mono hl-a">named_computation</span>',
  sub: '这是前面几课反复提到的那个动作。一次变换同时做**四件事**。',
  caption: '回顾 L1-08：<span class="mono">named_computation</span> 的语义就是"把函数体内联进一个命名区域"，让传播看不见调用边界。',
  code: `// 输入
%0 = call @foo(%arg0) {random_attr = "random_value",
                        mhlo.frontend_attributes = {backend_config = "…"}}
     : (tensor<8x2xi32>) -> tensor<8x2xi32>
%1 = stablehlo.custom_call @MoveToHost(%0) {backend_config = ""}

func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  %0 = stablehlo.multiply %arg0, %arg0
       {mhlo.frontend_attributes = {_xla_compute_type = "host"}}
  return %0
}

// 输出（四件事同时发生）：
//   ① call 变成 named_computation，名字取自【被调函数名】
%[[NC]] = sdy.named_computation<"foo">(%arg0) (%arg1: tensor<8x2xi32>) {
//   ② 函数体被搬进区域，并插入 data_flow_edge
  %[[EDGE_1]] = sdy.data_flow_edge %arg1 : tensor<8x2xi32>
  %[[MULT]] = stablehlo.multiply %[[EDGE_1]], %[[EDGE_1]]
  sdy.return %[[MULT]]
//   ③ 调用点的属性被搬到 named_computation 上
} {mhlo.frontend_attributes = {backend_config = "…"}, random_attr = "random_value"}
  : (tensor<8x2xi32>) -> tensor<8x2xi32>
//   ④ 原来的私有函数被删除
// CHECK-NOT: func private @foo`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 换名字', c: '#38bdf8', d: '<span class="mono">call @foo</span> →<br><span class="mono">named_computation&lt;"foo"&gt;</span><br>名字取自<b>被调函数名</b>' },
      { t: '② 搬函数体', c: '#4ade80', d: '函数体进入区域<br>并插入 <span class="mono">data_flow_edge</span><br>（L3-04 的规则）' },
      { t: '③ 搬属性', c: '#fbbf24', d: '调用点的属性<br>（<span class="mono">backend_config</span> / <span class="mono">random_attr</span>）<br>原样搬到新算子上' },
      { t: '④ 删函数', c: '#fb7185', d: '原私有函数被删除<br><span class="mono">CHECK-NOT: func private @foo</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:180px;opacity:.32;transition:.35s;border-color:' + x.c + '55;padding:9px' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:11.5px">${x.t}</div>
        <div class="card-d" style="font-size:10.5px;line-height:1.5">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.32');
      msg.innerHTML = [
        '名字很重要 —— 它让 outline 回函数时能还原（L4-12）。',
        '函数体里的参数变成块参数，按 L3-04 的规则插边。',
        '属性搬运保证语义不变（<span class="mono">backend_config</span> 影响后端行为）。',
        '<b>内联的代价</b>：函数消失，代码"长大"了 —— 但传播不再受调用边界阻隔。',
      ][i];
    }));
    tl.at(14400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>为什么值得</b>：调用边界会挡住传播分析（L3-03 见过这个问题）。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 两个来源 */
{
  kicker: 'L3-06 · 内联函数调用',
  title: '★ <span class="mono hl-a">in_shardings</span> 与 <span class="mono hl-a">out_shardings</span> 的<span class="hl-a">来源不同</span>',
  sub: '这是本课最容易搞错的一点：一个来自**被调函数**，一个来自**调用点**。',
  caption: '测试里的 NOTE 点明了：<span class="mono">we ignore any arg/result shardings on the function</span>（指结果侧）。',
  code: `// 【例子 A】两者都有
%0 = call @foo(%arg0, %arg0)
     {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>, …}
// 被调函数 @foo 的参数上有 [{"x"}, {}] 与 [{}, {"y"}]

// 输出：
in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]   // <- 来自【函数参数】
out_shardings=[<@mesh, [{"y"}, {"x"}]>]                     // <- 来自【调用点】

// 【例子 B】函数上有、调用点没有 -> 忽略函数上的
//   函数结果有 [{"x","y"}, {}]，调用点无 sharding
//   -> named_computation 【没有】out_shardings

// 【例子 C】函数上没有、调用点有 -> 采用调用点的
//   调用点有 [{"y"},{"x"}]，函数结果无标注
//   -> out_shardings=[<@mesh, [{"y"}, {"x"}]>]`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:30px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'card', style: 'width:300px;border-color:rgba(56,189,248,.5)' });
      c.innerHTML = `<div class="card-t mono" style="color:var(--ax0);font-size:12px">in_shardings</div>
        <div class="card-d">来自<b>被调函数</b>的<br>参数分片标注<br>
          <span class="dim small">有 N 个参数就有 N 项</span></div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>输入侧</b>：函数定义里参数写了什么，就搬什么过来。';
    });
    tl.at(4400, () => {
      const c = U.el('div', { class: 'card', style: 'width:300px;border-color:rgba(251,191,36,.5)' });
      c.innerHTML = `<div class="card-t mono" style="color:var(--warn);font-size:12px">out_shardings</div>
        <div class="card-d">来自<b>调用点</b>的<br><span class="mono">sdy.sharding</span> 属性<br>
          <span class="dim small">有 N 个结果就有 N 项</span></div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>输出侧</b>：调用点要求什么，就搬什么过来 —— 函数结果上的标注<b>被忽略</b>。';
    });
    tl.at(8600, () => {
      msg.innerHTML = '<b>为什么这样设计</b>：同一个函数可能被多个调用点调用，各调用点可以要求不同分片 —— 所以以<b>调用点</b>为准。';
    });
    tl.at(12000, () => {
      msg.innerHTML = '<b>注意</b>：区域内的边<b>继承 in_shardings</b>（L3-04 的规则），结果上的边继承 out_shardings。';
    });
    tl.at(14400, () => {
      msg.innerHTML = '这也解释了为什么 L3-05 的 <span class="mono">func_data_flow_edge</span> 需要<b>两端</b>都插边。';
    });
  }
},

/* ------------------------------------------------ 3 属性与多调用 */
{
  kicker: 'L3-06 · 内联函数调用',
  title: '属性搬运与多个调用点',
  sub: '调用点的属性原样搬到新算子上；同一函数被调多次时，每个调用点各生成一个 `named_computation`。',
  caption: '注意多个 <span class="mono">named_computation</span> 会<b>同名</b> —— 名字只是可读性，不要求唯一。',
  code: `// 属性搬运（用例覆盖）
//   backend_config_no_out_shardings       backend_config + random_attr
//   backend_config_out_shardings          同上 + 分片
//   inlineable_false / inlineable_true    inlineable 属性
//   no_backend_config_or_inlineable_attr  两者都没有

// 输出里的属性：
} {mhlo.frontend_attributes = {inlineable = "false"}}
} {mhlo.frontend_attributes = {inlineable = "true"}}
// 原样保留

// 多个调用点（用例）
//   multiple_call_ops_same_name
//     同一函数被调多次 -> 每个调用点各生成一个 named_computation（同名）
//   multiple_call_ops_same_name_func_no_input_output_shardings
//     同上，函数无 in/out 分片

// 非扁平调用图
//   non_flat_call_graph_all_uninlineable
//   non_flat_call_graph_all_inlineable`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '属性搬运', c: '#fbbf24',
        d: '调用点的属性（<span class="mono">backend_config</span> / <span class="mono">inlineable</span> / 自定义）<br>原样搬到 <span class="mono">named_computation</span> 上' },
      { t: '多个调用点', c: '#38bdf8',
        d: '同一函数被调 N 次 → 生成 N 个<br><span class="mono">named_computation</span>，<b>同名</b>' },
      { t: '非扁平调用图', c: '#c084fc',
        d: '多层调用图<br>分"全可内联"与"全不可内联"两种情形' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>为什么必须搬</b>：<span class="mono">backend_config</span> 会影响后端行为，丢了就改变语义。',
        '同名是允许的 —— 名字只用于<b>可读性</b>与 outline 时复用（L4-12）。',
        '<b>非扁平</b>指的是调用图有多层（<span class="mono">main → foo → bar</span>），不是扁平的星形结构。',
      ][i];
    }));
    tl.at(12200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>关键区分</b>：<span class="mono">named_computation</span> 的名字取自被调函数名，与调用点所在函数无关。';
    });
  }
},

/* ------------------------------------------------ 4 开关与展平 */
{
  kicker: 'L3-06 · 内联函数调用',
  title: '一个开关与一个配套 pass',
  sub: '第二个文件用 `add-data-flow-edges-on-named-computations=false`；第三个文件是 `-sdy-flatten-call-graph`。',
  caption: '两者都是<b>避免重复工作</b>或<b>改变结构</b>的工程手段。',
  code: `// ① add-data-flow-edges-on-named-computations=false
// RUN: sdy_opt --split-input-file %s \\
//        -sdy-import-func-calls='add-data-flow-edges-on-named-computations=false'

// 作用：生成 named_computation 时【不】在区域内插数据流边
// 为什么需要：插边有代价（IR 变大）
//   若后续有单独的 -sdy-add-data-flow-edges 统一插边（L3-04）
//   这里就可以关掉，避免重复工作
// 该文件与主文件用例一一对应（都是 21 个），只是少了边

// ② -sdy-flatten-call-graph
// RUN: sdy_opt %s -split-input-file -sdy-flatten-call-graph
// 415 行 / 13 个用例，处理【调用图结构】
//   singleton / simple_call_graph / main_calls_foo_twice
//   main_calls_foo_calls_bar(_twice) / simple_non_flat
//   simple_non_flat_sharding_on_func_arguments / _on_func_results
//   simple_non_flat_non_matching_sharding_*  <- 分片不匹配怎么办
//   simple_non_flat_with_manual_computations`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 关掉自动插边', c: '#38bdf8', tag: '选项',
        d: '避免与 L3-04 的 <span class="mono">-sdy-add-data-flow-edges</span><br><b>重复工作</b>。<br>用例与主文件一一对应。' },
      { t: '② 展平调用图', c: '#c084fc', tag: '独立 pass',
        d: '处理调用图<b>结构</b>（多层 → 扁平）。<br>13 个用例，重点是<br><b>分片不匹配</b>时怎么办。' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t} <span class="faint small">${x.tag}</span></div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>开关的意义</b>：让流水线的组装更灵活 —— 插边这件事可以只做一次。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>展平的价值</b>：扁平调用图让后续分析更简单（没有嵌套的调用层次）。';
    });
    tl.at(8400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>难点在分片</b>：展平时函数上的标注与各调用点的标注可能<b>不一致</b> —— 三个用例专门覆盖这种情况。';
    });
    tl.at(11600, () => {
      msg.innerHTML = '这与第 2 幕的"以调用点为准"呼应：多个调用点要求不同时，需要逐个处理。';
    });
  }
},

/* ---------------------------------------------------- 5 族谱 */
{
  kicker: 'L3-06 · 内联函数调用',
  title: '21 个用例的<span class="hl-a">族谱</span>',
  sub: '主文件的用例按主题分 6 组 —— 最大的一组是"多参数的分片组合"。',
  caption: '读法：抓住"四件事 + 两个来源"即可，其余用例都是边界组合。',
  code: `// 【属性搬运】5 个
//   backend_config_no_out_shardings / _out_shardings
//   inlineable_false / _true / no_backend_config_or_inlineable_attr

// 【多调用点】2 个
//   multiple_call_ops_same_name(_func_no_input_output_shardings)

// 【非扁平调用图】2 个
//   non_flat_call_graph_all_uninlineable / _all_inlineable

// 【多参数的分片组合】6 个
//   single_call_multiple_args_func_no_input_sharding
//   _all_arguments_with_input_sharding
//   _some_arguments_with_input_sharding_some_arguments_without
//   _with_different_ranks_...
//   _on_maximal_mesh_...
//   _on_maximal_and_on_non_maximal_mesh_...

// 【函数结果 vs 调用结果】5 个
//   func_has_out_sharding_call_no_out_sharding
//   func_has_out_sharding_call_has_different_out_sharding
//   func_no_out_sharding_call_has_out_sharding
//   func_has_out_sharding_on_one_result_call_has_out_sharding_on_both_results
//   func_has_out_sharding_on_one_result_call_has_no_out_sharding

// 【命名】1 个
//   func_name_and_original_names_different`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '属性搬运', n: 5, c: '#fbbf24' }, { t: '多调用点', n: 2, c: '#38bdf8' },
      { t: '非扁平调用图', n: 2, c: '#c084fc' }, { t: '多参数分片', n: 6, c: '#4ade80' },
      { t: '结果分片对比', n: 5, c: '#fb7185' }, { t: '命名', n: 1, c: '#94a3b8' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:132px;opacity:.4;transition:.35s;border-color:${f.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="card-t" style="color:${f.c};font-size:11px">${f.t}</div>
        <div class="big" style="font-size:20px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 120)); msg.innerHTML = '<b>21 个用例</b>归成 6 组 —— 最大的两组是"多参数分片"与"结果分片对比"。'; });
    tl.at(4200, () => {
      els.forEach((e, i) => { if (i !== 3 && i !== 4) e.style.opacity = '.25'; });
      msg.innerHTML = '这两组合计过半 —— 说明<b>分片的来源与组合</b>是这个 pass 最复杂的地方。';
    });
    tl.at(7600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>两个来源</b>：<span class="mono">in_shardings</span> ← 函数参数；<span class="mono">out_shardings</span> ← 调用点。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L3-07 讲提升内联 mesh —— 把内联写的网格定义提升为顶层声明。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L3-06 · 练习',
  title: '练一练：<span class="hl-a">变换后会是什么样</span>',
  sub: '三道题分别考：四件事、两个来源、开关的作用。',
  caption: '一句话总结：<b>call 变 named_computation，函数体搬进来，属性搬过来，原函数删掉</b>。',
  code: `// 题 1：call -> named_computation 时，同时发生了哪四件事？

// 题 2：in_shardings 和 out_shardings 分别来自哪里？
//       如果函数结果上有标注、调用点没有，会怎样？

// 题 3：add-data-flow-edges-on-named-computations=false
//       这个选项解决什么问题？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">call</span> → <span class="mono">named_computation</span> 时，同时发生了哪<b>四件事</b>？',
        a: '① <b>换名字</b>：<span class="mono">call @foo</span> → <span class="mono">named_computation&lt;"foo"&gt;</span>，名字取自被调函数名。' +
           '<br>② <b>搬函数体</b>：函数体进入区域，并插入 <span class="mono">data_flow_edge</span>。' +
           '<br>③ <b>搬属性</b>：调用点的属性（<span class="mono">backend_config</span> / <span class="mono">inlineable</span> 等）原样搬过来。' +
           '<br>④ <b>删函数</b>：原私有函数被删除。'
      },
      {
        q: '<span class="mono">in_shardings</span> 和 <span class="mono">out_shardings</span> 分别来自哪里？函数结果上有标注、调用点没有会怎样？',
        a: '<b><span class="mono">in_shardings</span> ← 被调函数的参数标注</b>；<b><span class="mono">out_shardings</span> ← 调用点的 <span class="mono">sdy.sharding</span></b>。' +
           '<br>函数结果上有、调用点没有 → <b>忽略函数上的</b>，<span class="mono">named_computation</span> 没有 <span class="mono">out_shardings</span>。' +
           '<br><span class="dim">为什么以调用点为准？同一函数可能被多个调用点调用，各调用点可要求不同分片。</span>'
      },
      {
        q: '<span class="mono">add-data-flow-edges-on-named-computations=false</span> 这个选项解决什么问题？',
        a: '让生成 <span class="mono">named_computation</span> 时<b>不</b>在区域内插数据流边。' +
           '<br><b>为什么需要</b>：插边有代价（IR 变大）。若后续有单独的 <span class="mono">-sdy-add-data-flow-edges</span>（L3-04）统一插边，这里关掉就能<b>避免重复工作</b>。' +
           '<br><span class="dim">该文件与主文件用例一一对应（都是 21 个），只是输出里少了边 —— 便于对照。</span>'
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
