/* ==========================================================================
   L3-07 · lift-inlined-meshes
   --------------------------------------------------------------------------
   覆盖：transforms/import/test/lift_inlined_meshes.mlir (286 行 / 15 用例)
   目标：讲透"内联 mesh 提升为顶层声明"与"按内容去重"两条规则。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 问题 */
{
  kicker: 'L3-07 · 提升内联网格',
  title: '问题：内联 mesh <span class="hl-a">无法被引用</span>',
  sub: 'SDY 允许把网格定义**内联**写在分片属性里。但内联定义没有名字 —— 别处想用同一个网格就没法引用。',
  caption: 'L1-01 讲过 <span class="mono">sdy.mesh</span> 的语法（顶层声明）。这一课讲内联写法怎么被<b>规范化</b>成那种形式。',
  code: `// RUN: sdy_opt -split-input-file %s -sdy-lift-inlined-meshes

// 内联写法（合法，但无法被别处引用）
%0 = stablehlo.add %arg0, %arg1
     {sdy.sharding = #sdy.sharding_per_value<
        [<mesh<["x"=2, "y"=2]>, [{"x"}, {}]>]>} : tensor<8x8xf32>

// 规范化后：多一条顶层声明 + 改成命名引用
sdy.mesh @mesh_0 = <["x"=2, "y"=2]>          // <- 新增
%0 = stablehlo.add %arg0, %arg1
     {sdy.sharding = #sdy.sharding_per_value<
        [<@mesh_0, [{"x"}, {}]>]>} : tensor<8x8xf32>
//       ^^^^^^^ 命名引用

// 这个 pass 做两件事：
//   ① 提升（lift）：内联定义 -> 顶层声明 + 起名
//   ② 去重（dedup）：内容相同的声明合并成一个`,
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
        <div class="small mono" style="color:var(--bad)">✗ 内联写法</div>
        <div class="chip c3" style="padding:7px 12px;font-size:11.5px">mesh&lt;["x"=2, "y"=2]&gt;</div>
        <div class="small faint">没有名字 → 别处引用不了</div>`;
      demo.appendChild(c);
      msg.innerHTML = '内联定义<b>没有名字</b> —— 同一个网格想复用就只能再写一遍。';
    });
    tl.at(4400, () => {
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹', style: 'font-size:26px' }));
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ok)">✓ 顶层声明</div>
        <div class="chip c4" style="padding:7px 12px;font-size:11.5px">sdy.mesh @mesh_0 = &lt;["x"=2, "y"=2]&gt;</div>
        <div class="small faint">有名字 → 可被多处引用</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>提升</b>：生成一条顶层声明并起名，原处改成命名引用。';
    });
    tl.at(8600, () => {
      msg.innerHTML = '<b>为什么必须做</b>：有了名字，才能判断"两处用的是不是同一个网格"，从而<b>去重</b>。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '这个现象在 L2-11 的 <span class="mono">inlined_mesh</span> 用例里见过 —— 本课讲它的完整规则。';
    });
  }
},

/* ------------------------------------------------ 2 命名规则 */
{
  kicker: 'L3-07 · 提升内联网格',
  title: '命名规则：<span class="hl-a">三种基础名</span>',
  sub: '普通网格用 `mesh_N`，空网格用 `empty_mesh`，maximal 网格用 `maximal_mesh_N`（**设备号进名字**）。',
  caption: '编号会<b>避开已占用的名字</b> —— 所以同一个文件里可能看到 <span class="mono">@mesh_0</span>、<span class="mono">@mesh_1</span>、<span class="mono">@mesh_2</span>。',
  code: `// ① 普通网格 -> @mesh_N
mesh<["x"=2, "y"=2]>        ->  @mesh_0
//   若 @mesh 已被占用（内容不同）

// ② 空网格 -> @empty_mesh（特殊名）
mesh<[]>                     ->  @empty_mesh

// ③ maximal 网格 -> @maximal_mesh_N（设备号进名字！）
mesh<[], device_ids=[3]>     ->  @maximal_mesh_3
mesh<[], device_ids=[7]>     ->  @maximal_mesh_7
//   两者【不会】被去重 —— 设备号不同就是不同网格

// 注意：轴名不同不影响命名
mesh<["b"=2]>                ->  @mesh_0
//   即使 @mesh 是 <["a"=2]>，新网格仍编号为 @mesh_0`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 普通', c: '#38bdf8', r: '@mesh_N',
        d: '基础名 <span class="mono">mesh</span> + 编号<br>编号避开已占用的名字' },
      { t: '② 空网格', c: '#4ade80', r: '@empty_mesh',
        d: '<span class="mono">mesh&lt;[]&gt;</span><br>用<b>特殊名</b>，不编号' },
      { t: '③ maximal', c: '#fbbf24', r: '@maximal_mesh_N',
        d: '<span class="mono">mesh&lt;[], device_ids=[N]&gt;</span><br><b>设备号进名字</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="mono" style="font-size:12px;color:#bdf7ec;margin:5px 0">${U.esc(x.r)}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '编号是<b>按需递增</b>的 —— 已占用的名字会被跳过。',
        '<b>为什么特殊对待</b>：空网格是"占位符"（L2-01 讲过它在传播中的特殊规则），给个固定的名字更可读。',
        '<b>关键细节</b>：设备号不同 → 名字不同 → <b>不会被去重</b>。因为它们语义上就是不同网格。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>注意</b>：轴名不同<b>不影响</b>命名 —— 只看名字是否已被占用。';
    });
  }
},

/* ------------------------------------------------ 3 去重规则 */
{
  kicker: 'L3-07 · 提升内联网格',
  title: '★ 去重：<span class="hl-a">按内容</span>，不按名字',
  sub: '内容完全相同的声明会被**合并**成一个，多余的声明被删除、引用被改写。',
  caption: '去重与名字无关 —— 两个名字不同的声明，只要内容一样就会合并。',
  code: `// 输入：@mesh3 与 @mesh1 内容相同
sdy.mesh @mesh1 = <["x"=2, "y"=2]>
sdy.mesh @mesh2 = <["x"=2, "y"=4]>
sdy.mesh @mesh3 = <["x"=2, "y"=2]>          // 与 mesh1 重复

%0 = stablehlo.add %arg0, %arg1
     {sdy.sharding = #sdy.sharding_per_value<
        [<@mesh3, [{}, {"y"}]>]>} : tensor<8x8xf32>

// 输出：
// CHECK-NOT: sdy.mesh @mesh3               <- @mesh3 被删除
// CHECK-NEXT: stablehlo.add %arg0, %arg1
//   {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{}, {"y"}]>]>
//                                              ^^^^^^^ 引用改成 @mesh1

// 更复杂的一例（many_inlined_meshes_and_duplicates）：
//   @mesh_1        = <["x"=2, "y"=4]>      与 @mesh 重复 -> 删除
//   @copy_of_mesh  = <["x"=2, "y"=4]>      与 @mesh 重复 -> 删除
//   @copy_of_mesh_0= <["x"=2], device_ids=[1,0]>  与 @mesh_0 重复 -> 删除`,
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
        <div class="small mono faint">输入</div>
        <div class="chip c0" style="padding:6px 11px;font-size:11px">@mesh1 = &lt;["x"=2, "y"=2]&gt;</div>
        <div class="chip c1" style="padding:6px 11px;font-size:11px">@mesh2 = &lt;["x"=2, "y"=4]&gt;</div>
        <div class="chip c3" style="padding:6px 11px;font-size:11px">@mesh3 = &lt;["x"=2, "y"=2]&gt;</div>
        <div class="small" style="color:var(--bad);font-size:11px">mesh3 与 mesh1 内容相同</div>`;
      demo.appendChild(c);
      msg.innerHTML = '两个名字不同、但<b>内容完全一样</b>的声明 —— 它们是同一个网格。';
    });
    tl.at(4600, () => {
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹', style: 'font-size:26px' }));
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ok)">输出</div>
        <div class="chip c0" style="padding:6px 11px;font-size:11px">@mesh1 = &lt;["x"=2, "y"=2]&gt;</div>
        <div class="chip c1" style="padding:6px 11px;font-size:11px">@mesh2 = &lt;["x"=2, "y"=4]&gt;</div>
        <div class="small" style="color:var(--ok);font-size:11px">@mesh3 被删除，引用改成 @mesh1</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>合并</b>：保留先出现的那个名字，其余删除、引用改写。';
    });
    tl.at(8800, () => {
      msg.innerHTML = '<b>判据是内容</b>（轴名、大小、设备序），<b>不是名字</b>。';
    });
    tl.at(11600, () => {
      msg.innerHTML = '<b>收益</b>：IR 更短，且"这两处是同一个网格"这个事实变得<b>显式可见</b>。';
    });
    tl.at(14000, () => {
      msg.innerHTML = '复杂用例里一次删掉了 <b>3 条</b>重复声明（<span class="mono">@mesh_1</span> / <span class="mono">@copy_of_mesh</span> / <span class="mono">@copy_of_mesh_0</span>）。';
    });
  }
},

/* ------------------------------------------------ 4 覆盖范围 */
{
  kicker: 'L3-07 · 提升内联网格',
  title: '覆盖范围：<span class="hl-a">所有带分片的位置</span>',
  sub: '内联 mesh 可以出现在任何有 `sdy.sharding` 的地方，这个 pass 全部处理。',
  caption: '包括 <span class="mono">manual_computation</span> 的 in/out_shardings、集合通信算子、函数调用结果等。',
  code: `// 用例覆盖的位置：
//   single_sharding_sdy_ops          单个 SDY 算子上的分片
//   manual_computation               manual_computation 的 in/out_shardings
//   single_call                      函数调用
//   all_reduce_inlined_mesh          sdy.all_reduce
//   inlined_mesh_on_call_result      调用结果上
//   non_flat_graph_inlined_mesh      非扁平调用图
//   inlined_mesh_with_reduction_op   带归约的算子
//   tagged_stablehlo_mesh_attribute  带 tag 的 stablehlo mesh 属性

// tagged 用例特别值得一提：
//   @mesh = <["a"=2]>  已存在
//   内联 mesh<["b"=2]> 轴名【不同】
//   -> 仍编号为 @mesh_0（只看名字是否被占用）

// 也就是说：命名与"内容像不像"无关
//   只有【去重】才比较内容`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const items = [
      { t: 'SDY 算子分片', c: '#38bdf8' }, { t: 'manual_computation', c: '#4ade80' },
      { t: '函数调用', c: '#fbbf24' }, { t: 'all_reduce', c: '#c084fc' },
      { t: '调用结果', c: '#f472b6' }, { t: '非扁平调用图', c: '#fb7185' },
      { t: '归约算子', c: '#93c5fd' }, { t: 'tagged mesh 属性', c: '#5eead4' },
    ];
    const host = wrap.querySelector('#cards');
    const els = items.map(x => {
      const e = U.el('div', { class: 'card', style: `width:138px;opacity:.38;transition:.35s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:10.5px;color:${x.c};overflow-wrap:anywhere">${x.t}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 120)); msg.innerHTML = '<b>8 类位置</b>都覆盖到了 —— 内联 mesh 出现在哪都能被提升。'; });
    tl.at(4200, () => {
      els.forEach((e, i) => { if (i !== 7) e.style.opacity = '.25'; });
      msg.innerHTML = '<b>tagged 用例</b>特别值得一提：内联 mesh 与已有 <span class="mono">@mesh</span> 的<b>轴名不同</b>，但仍编号为 <span class="mono">@mesh_0</span>。';
    });
    tl.at(7800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>关键区分</b>：命名只看"名字是否被占用"；只有<b>去重</b>才比较内容。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '这也解释了为什么会有 <span class="mono">@mesh_0</span> 这种"看起来没必要"的编号。';
    });
  }
},

/* ---------------------------------------------------- 5 族谱 */
{
  kicker: 'L3-07 · 提升内联网格',
  title: '15 个用例的<span class="hl-a">族谱</span>',
  sub: '按"做什么"分四组 —— 只有第一组是纯去重。',
  caption: '读法：抓住"提升 + 去重"两条规则，再看命名与覆盖范围。',
  code: `// 【纯去重（无内联 mesh）】2 个
//   no_inlined_meshes_or_duplicates
//   no_inlined_meshes_with_duplicates

// 【基础提升 + 三种命名】3 个
//   inlined_mesh                -> @mesh_0
//   inlined_empty_mesh          -> @empty_mesh
//   inlined_maximal_mesh        -> @maximal_mesh_N

// 【多网格与去重混合】2 个
//   many_inlined_meshes_and_duplicates
//   another_func_in_module

// 【各类位置】8 个
//   single_sharding_sdy_ops / manual_computation / single_call
//   all_reduce_inlined_mesh / inlined_mesh_on_result
//   non_flat_graph_inlined_mesh / inlined_mesh_with_reduction_op
//   tagged_stablehlo_mesh_attribute`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '纯去重', n: 2, c: '#4ade80' }, { t: '基础提升', n: 3, c: '#38bdf8' },
      { t: '多网格混合', n: 2, c: '#fbbf24' }, { t: '各类位置', n: 8, c: '#c084fc' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:150px;opacity:.4;transition:.35s;border-color:${f.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="card-t" style="color:${f.c};font-size:11.5px">${f.t}</div>
        <div class="big" style="font-size:22px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 130)); msg.innerHTML = '<b>15 个用例</b>分四组 —— 最大的一组是"各类位置"（8 个）。'; });
    tl.at(4000, () => {
      els.forEach((e, i) => { if (i !== 3) e.style.opacity = '.25'; });
      msg.innerHTML = '过半用例在验证"<b>内联 mesh 出现在哪里都能被处理</b>" —— 因为出现的位置太多了。';
    });
    tl.at(7400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>两条规则</b>：提升（内联 → 顶层声明）+ 去重（按内容合并）。';
    });
    tl.at(10600, () => {
      msg.innerHTML = '<b>下一课</b> L3-08 讲清理 manual 轴 —— 把"没切维度的 manual 轴"移进 replicated。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L3-07 · 练习',
  title: '练一练：<span class="hl-a">会生成什么名字</span>',
  sub: '三道题分别考：命名规则、去重判据、特殊网格。',
  caption: '一句话总结：<b>提升为顶层声明，按内容去重</b>。',
  code: `// 题 1：文件里已有 @mesh = <["x"=2, "y"=4]>，
//       现在遇到内联 mesh<["x"=2, "y"=2]>，会生成什么名字？

// 题 2：@a = <["x"=2]> 与 @b = <["x"=2]>，会怎样？

// 题 3：mesh<[], device_ids=[3]> 与 mesh<[], device_ids=[7]>
//       会生成什么？会被去重吗？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '文件里已有 <span class="mono">@mesh = &lt;["x"=2, "y"=4]&gt;</span>，现在遇到内联 <span class="mono">mesh&lt;["x"=2, "y"=2]&gt;</span>，会生成什么名字？',
        a: '<span class="mono">@mesh_0</span>。' +
           '<br><b>规则</b>：基础名是 <span class="mono">mesh</span>，但 <span class="mono">@mesh</span> 已被占用 → 编号成 <span class="mono">@mesh_0</span>。' +
           '<br><span class="dim">注意：命名<b>只看名字是否被占用</b>，与内容像不像无关。所以即使轴名完全不同（如 <span class="mono">mesh&lt;["b"=2]&gt;</span>）也会是 <span class="mono">@mesh_0</span>。</span>'
      },
      {
        q: '<span class="mono">@a = &lt;["x"=2]&gt;</span> 与 <span class="mono">@b = &lt;["x"=2]&gt;</span>，会怎样？',
        a: '<b>合并成一个</b>：保留先出现的名字（<span class="mono">@a</span>），删除 <span class="mono">@b</span>，把对 <span class="mono">@b</span> 的引用改成 <span class="mono">@a</span>。' +
           '<br><b>判据是内容</b>（轴名、大小、设备序），<b>不是名字</b>。' +
           '<br><span class="dim">测试用 <span class="mono">CHECK-NOT: sdy.mesh @b</span> 精确断言删除。</span>'
      },
      {
        q: '<span class="mono">mesh&lt;[], device_ids=[3]&gt;</span> 与 <span class="mono">mesh&lt;[], device_ids=[7]&gt;</span> 会生成什么？会被去重吗？',
        a: '生成 <span class="mono">@maximal_mesh_3</span> 与 <span class="mono">@maximal_mesh_7</span>。' +
           '<br><b>不会被去重</b> —— <b>设备号进入了名字</b>，而且它们语义上就是不同的网格（maximal 网格表示"某台设备独占"）。' +
           '<br><span class="dim">对比：空网格 <span class="mono">mesh&lt;[]&gt;</span> → <span class="mono">@empty_mesh</span>（固定名，不带编号）。</span>'
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
