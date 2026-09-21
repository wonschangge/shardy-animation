/* ==========================================================================
   L4-12 · export-named-computations
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/export_named_computations.mlir (1263 行 / 93 用例)
   目标：讲透 named_computation 反向 outline 成函数 + call 的规则。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 基本 outline */
{
  kicker: 'L4-12 · 导出命名计算',
  title: '★ 基本 outline：<span class="hl-a">区域变函数，调用点变 call</span>',
  sub: '`sdy.named_computation` 被**反向 outline** 成 `func.func private` + `call`。',
  caption: 'L1-08 讲过 <span class="mono">named_computation</span> 是"内联的函数" —— 本课做<b>反向操作</b>。',
  code: `// RUN: sdy_opt %s -sdy-export-named-computations -split-input-file | FileCheck %s

// 【输入】named_computation
func.func @vanilla_named_computation(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  %0 = sdy.named_computation<"bar">(%arg0) (%arg1: tensor<8x2xi32>) {
    %1 = stablehlo.multiply %arg1, %arg1 : tensor<8x2xi32>
    sdy.return %1 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// 【输出】调用点变成 call
// CHECK-LABEL: func @vanilla_named_computation(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
// CHECK-NEXT: %[[CALL:.*]] = call @bar(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
// CHECK-NEXT: return %[[CALL]] : tensor<8x2xi32>

// 【输出】区域变成 private 函数
// CHECK-LABEL: func private @bar(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32>
//               attributes {sdy.original_func_name = "bar"} {
//                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 保留原名

// 三步：
//   ① named_computation<"bar"> 的区域 -> func.func private @bar
//   ② 调用点 -> call @bar(%arg0)
//   ③ 函数带 sdy.original_func_name 属性

// 为什么需要 sdy.original_func_name：
//   函数名在 IR 里必须【唯一】
//   同名多次出现时要加后缀区分（见第 3 幕）
//   但"它原本叫什么"这个信息不能丢 -> 用属性记下来`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 区域', c: '#38bdf8', d: '<span class="mono">named_computation</span><br>→ <span class="mono">func.func private</span>' },
      { t: '② 调用点', c: '#4ade80', d: '<span class="mono">named_computation</span> 调用<br>→ <span class="mono">call</span>' },
      { t: '③ 属性', c: '#fbbf24', d: '<span class="mono">sdy.original_func_name</span><br>保留<b>原名</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '区域里的算子（<span class="mono">multiply</span>）成为函数体。',
        '调用点的实参直接成为 <span class="mono">call</span> 的实参。',
        '<b>为什么要保留原名</b>：函数名要唯一（会加后缀），但"来源"信息不能丢。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>与 L3-06 正好相反</b>：那里把 <span class="mono">call</span> 内联成 <span class="mono">named_computation</span>，本课 outline 回去。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 分片来源 */
{
  kicker: 'L4-12 · 导出命名计算',
  title: '★ 分片的<span class="hl-a">两个来源</span>',
  sub: '函数参数的分片来自 **block argument**；call 的 sharding 来自 **`out_shardings`**。',
  caption: '测试开头的注释直接点明了这一点：<span class="mono">we don\'t override the block argument shardings</span>。',
  code: `// Note we don't override the block argument shardings of the function
// @ignore_operand_shardings, but we set the argument shardings on the call
// to @foo.
//   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 测试注释点明规则

func.func @ignore_operand_shardings(
    %arg0: tensor<8x2xi32> {...<@mesh, [{}, {"y"}]>})
    -> (tensor<8x2xi32> {...<@mesh, [{"x"}, {"y"}]>}) {
  %0 = sdy.named_computation<"foo">(%arg0)
       in_shardings=[<@mesh, [{}, {"y"}]>]         // <- 描述"调用者怎么传"
       out_shardings=[<@mesh, [{"x"}, {}]>]        // <- 描述"返回什么"
       (%arg1: tensor<8x2xi32>) {                  // <- block argument
    %2 = stablehlo.multiply %arg1, %arg1
         {mhlo.frontend_attributes = {_xla_compute_type = "host"},
          sdy.sharding = ...<@mesh, [{"x", ?}, {"y", ?}]>} : tensor<8x2xi32>
    sdy.return %2 : tensor<8x2xi32>
  } {random_attr = "random_value",
     mhlo.frontend_attributes = {backend_config = "{...}"}} : ...

// 【函数参数】来自 block argument
// CHECK-LABEL: func private @foo
// CHECK-SAME:    (%arg0: tensor<8x2xi32> {sdy.sharding = ...<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> tensor<8x2xi32> attributes {sdy.original_func_name = "foo"} {

// 【call 的 sharding】来自 out_shardings
// CHECK-NEXT: %[[CALL:.*]] = call @foo(%arg0)
// CHECK-SAME:   {mhlo.frontend_attributes = {backend_config = "{...}"},
// CHECK-SAME:    random_attr = "random_value",
// CHECK-SAME:    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
//                                  ^^^^^^^^^^^^^^ = out_shardings
// CHECK-SAME:   : (tensor<8x2xi32>) -> tensor<8x2xi32>

// 【属性也搬到 call 上】
//   named_computation 上的 random_attr / mhlo.frontend_attributes
//   -> 全部移到 call 上
//   因为 named_computation 是"内联的调用点"，属性语义上属于调用点`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '函数参数', c: '#38bdf8', src: 'block argument',
        d: '取 <span class="mono">%arg1</span> 那一侧的分片<br><b>不是</b> <span class="mono">in_shardings</span>' },
      { t: 'call 的 sharding', c: '#4ade80', src: 'out_shardings',
        d: '描述"这个调用返回什么"<br><span class="mono">[{"x"}, {}]</span>' },
      { t: '属性', c: '#fbbf24', src: 'named_computation',
        d: '<span class="mono">random_attr</span> 等<br>全部移到 <b>call</b> 上' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="mono" style="font-size:10px;color:#bdf7ec;margin:3px 0">${U.esc(x.src)}</div>
        <div class="card-d" style="font-size:10.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>为什么</b>：outline 后的函数<b>就是那个函数体</b>，它看到的是 block argument。',
        '<span class="mono">in_shardings</span> 描述"调用者怎么传"，<span class="mono">out_shardings</span> 描述"返回什么"。',
        '<b>为什么属性在 call 上</b>：<span class="mono">named_computation</span> 是"内联的调用点"，属性语义上属于调用点。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>注意</b>：本例中两者恰好相同（都是 <span class="mono">[{}, {"y"}]</span>），但<b>规则是取 block argument</b>。';
    });
  }
},

/* ------------------------------------------------ 3 ★ 命名与去重 */
{
  kicker: 'L4-12 · 导出命名计算',
  title: '★ 命名：<span class="mono hl-a">@baz</span> 与 <span class="mono hl-a">@baz_0</span>',
  sub: '同一个名字出现两次 → **每个实例一个函数**，名字冲突时加 `_0` 后缀。',
  caption: '两个函数都带 <span class="mono">sdy.original_func_name = "baz"</span> —— 说明它们<b>同源</b>。',
  code: `// 【相同分片】两个 named_computation<"baz">，in/out_shardings 完全一样
// CHECK-NEXT: %0 = call @baz(%arg0)
//    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : ...
// CHECK-NEXT: %1 = call @baz_0(%arg0)
//    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : ...
//                              ^^^^^^ 同一个 sharding
// CHECK-NEXT: return %1 : tensor<8x2xi32>
// CHECK-LABEL: func private @baz(
// CHECK-SAME:  attributes {sdy.original_func_name = "baz"}

// 【不同分片】两个 named_computation<"baz">，out_shardings 不同
// CHECK-NEXT: %0 = call @baz(%arg0)
//    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : ...
// CHECK-NEXT: %1 = call @baz_0(%arg0)
//    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : ...
//                              ^^^^^^^^^^^^^^^ 不同的 sharding
// CHECK-NEXT: return %1 : tensor<8x2xi32>
// CHECK-LABEL: func private @baz_0(
// CHECK-SAME:    (%arg0: tensor<8x2xi32> {sdy.sharding = ...<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> tensor<8x2xi32>
// CHECK-SAME:  attributes {sdy.original_func_name = "baz"}

// 【关键观察】
//   两种情形下函数命名【一样】：@baz 与 @baz_0
//   差别只在 call 的 sharding
//   两个函数【都】带 sdy.original_func_name = "baz" -> 它们【同源】
//   函数参数的分片在两例中【都是】[{}, {"y"}]
//     -> 再次印证"函数参数分片来自 block argument"（差异只在 out_shardings）

// 测试里还有一族 three_named_computations_same_origin_func_*
//   用例名里的 same_origin_func 进一步印证了"同源"这个概念`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '相同分片', c: '#4ade80',
        d: '两个 call 的 sharding<br><b>都是</b> <span class="mono">[{"x"}, {}]</span><br>函数名仍是 <span class="mono">@baz</span> / <span class="mono">@baz_0</span>' },
      { t: '不同分片', c: '#fbbf24',
        d: '第二个 call 是<br><span class="mono">[{"x"}, {"y"}]</span><br>函数名<b>一样</b>是 <span class="mono">@baz</span> / <span class="mono">@baz_0</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>每个实例一个函数</b>：同名时加 <span class="mono">_0</span> 后缀。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>分片不同也不影响命名</b> —— 差别只体现在 call 的 sharding 上。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>两个函数都带 <span class="mono">sdy.original_func_name = "baz"</span></b> —— 下游据此知道它们<b>同源</b>。';
    });
    tl.at(12000, () => {
      msg.innerHTML = '<b>再次印证</b>：两例中函数参数的分片<b>都是</b> <span class="mono">[{}, {"y"}]</span> —— 因为差异只在 <span class="mono">out_shardings</span>。';
    });
  }
},

/* ------------------------------------------------ 4 族谱 */
{
  kicker: 'L4-12 · 导出命名计算',
  title: '93 个用例的<span class="hl-a">族谱</span>',
  sub: '从用例名就能读出覆盖的场景 —— 最复杂的是与 `manual_computation` 的**交叉**。',
  caption: '93 个用例说明这个 pass 的规则组合非常多。',
  code: `// 【基础】vanilla_named_computation / single_call
// 【分片来源】ignore_operand_shardings
// 【同名去重】multiple_same_named_computations_{same,different}_shardings
// 【嵌套】non_flat_nested_named_computations_{same,different,mixed}_shardings
// 【与 manual_axes 交互】named_computations_with_manual_axes_*（多个）
// 【在 manual_computation 内】*_one_inside_manual_computation*（多个）
// 【嵌套 manual_computation】nested_manual_computations
// 【无 out_sharding】single_named_computation_no_out_sharding
//                    same_named_computations_one_with_no_out_sharding
// 【同源多实例】three_named_computations_same_origin_func_*
// 【其它】xla / multiple_same_named_computations_same_shardings_
//          named_computations_have_different_manual_computation_calls

// 【最复杂的一族：manual_axes 与嵌套】
//   named_computation 是"内联的函数"（L1-08）
//   manual_computation 是"区域内自己管分片"（L1-07）
//
//   当 named_computation 出现在 manual_computation 区域内时：
//     它看到的 block argument 是【局部形状】
//     outline 出的函数参数分片也应该是【局部的】
//     但 in_shardings / out_shardings 写的是【全局分片】-> 需要转换
//
//   用例名里 one_without_manual_axes / with_manual_axes 成对出现
//   -> 说明这个交叉有多个变体需要区分

// 【注意 no_out_sharding 一族】
//   缺 out_shardings 时怎么办？用例名说明这需要【单独处理】
//   这与 L4-02 讲的"空分片"情形呼应`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '基础', c: '#94a3b8', d: 'outline 三步' },
      { t: '同名去重', c: '#4ade80', d: '<span class="mono">@baz</span> / <span class="mono">@baz_0</span>' },
      { t: '嵌套', c: '#38bdf8', d: 'named_computation<br>里面还有一层' },
      { t: '与 manual_axes', c: '#fbbf24', d: '带 <span class="mono">manual_axes</span><br>的情形' },
      { t: '在 manual 区域内', c: '#c084fc', d: '交叉：局部 vs 全局<br>分片需要转换' },
      { t: '无 out_sharding', c: '#fb7185', d: '缺 <span class="mono">out_shardings</span><br>单独处理' },
      { t: '同源多实例', c: '#5eead4', d: '<span class="mono">same_origin_func</span><br>三个实例' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:126px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div style="font-size:10.5px;color:${x.c}">${x.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>7 类场景</b> —— 最复杂的是"在 manual_computation 区域内"。'; });
    tl.at(4200, () => {
      els.forEach((e, i) => { if (i !== 4) e.style.opacity = '.25'; });
      msg.innerHTML = '<b>交叉的难点</b>：区域内看到的是<b>局部形状</b>，但 <span class="mono">in/out_shardings</span> 写的是<b>全局分片</b>。';
    });
    tl.at(7600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>成对出现的用例名</b>（<span class="mono">with_manual_axes</span> / <span class="mono">without</span>）说明这个交叉有多个变体。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>注意 <span class="mono">no_out_sharding</span> 一族</b>：缺分片时需要单独处理 —— 与 L4-02 的"空分片"呼应。';
    });
  }
},

/* ------------------------------------------------ 5 与 L3-06 的关系 */
{
  kicker: 'L4-12 · 导出命名计算',
  title: '与 L3-06 的<span class="hl-a">镜像关系</span>',
  sub: 'L3-06 把 `call` **内联**成 `named_computation`；本课把它 **outline** 回函数。',
  caption: '两课合起来解释了"为什么 Shardy 需要 <span class="mono">named_computation</span> 这个中间形态"。',
  code: `// 【L3-06 导入期】call --内联--> named_computation
//   为什么：传播需要"看得见"函数体
//   否则分片传不进去（L1-08 讲过）

// 【L4-12 导出期】named_computation --outline--> func + call
//   为什么：IR 要交给后端，后端需要【真正的函数】

// 一条完整的链路：
//   用户写的 call
//     --L3-06--> sdy.named_computation      （传播期的形态）
//     --传播-->  分片被推导
//     --L4-12--> func + call                （交给后端的形态）

// 【为什么需要这个中间形态】
//   call 是"边界"：传播默认【不穿过】它（L1-08）
//   named_computation 是"内联的调用点"：传播可以穿进去
//   但又保留了"这是一个命名计算"的信息
//   -> 导出时才能准确 outline 回函数

// 【与 L3-10 的呼应】
//   L3-10 讲"把函数结果的分片搬到调用点"
//   本课讲"把 named_computation 变回函数"
//   两者都在处理【函数边界】上的分片

// 一句话总结：
//   named_computation 是"传播期的临时形态"
//   导入时把 call 内联成它，导出时再 outline 回函数`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:16px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const node = (label, sub, color) => {
      const c = U.el('div', { class: 'col', style: 'gap:5px;align-items:center;opacity:0;transition:.45s' });
      c.innerHTML = `<div class="chip ${color}" style="padding:7px 12px;font-size:11px">${label}</div>
        <div class="small faint" style="font-size:9.5px;text-align:center;max-width:130px;line-height:1.35">${sub}</div>`;
      demo.appendChild(c); return c;
    };

    tl.at(700, () => {
      node('用户写的 call', '普通的函数调用', 'c0').style.opacity = '1';
      msg.innerHTML = '起点：用户写的普通 <span class="mono">call</span>。';
    });
    tl.at(4000, () => {
      demo.insertAdjacentHTML('beforeend', '<div class="arrow" style="font-size:16px">→</div>');
      node('named_computation', 'L3-06 内联<br>传播期的形态', 'c2').style.opacity = '1';
      msg.innerHTML = '<b>L3-06</b>：内联成 <span class="mono">named_computation</span> —— 传播才能<b>看得见</b>函数体。';
    });
    tl.at(8000, () => {
      demo.insertAdjacentHTML('beforeend', '<div class="arrow" style="font-size:16px">→</div>');
      node('func + call', 'L4-12 outline<br>交给后端的形态', 'c4').style.opacity = '1';
      msg.innerHTML = '<b>L4-12</b>：outline 回函数 —— 后端需要<b>真正的函数</b>。';
    });
    tl.at(12000, () => {
      msg.innerHTML = '<b>为什么需要中间形态</b>：<span class="mono">call</span> 是边界（传播不穿过），<span class="mono">named_computation</span> 可穿进去但保留名字。';
    });
    tl.at(14800, () => {
      msg.innerHTML = '<b>一句话</b>：<span class="mono">named_computation</span> 是"传播期的临时形态"。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L4-12 · 练习',
  title: '练一练：<span class="hl-a">预测 outline 的结果</span>',
  sub: '三道题分别考：基本 outline、分片来源、命名规则。',
  caption: '一句话总结：<b>区域变函数，调用点变 call，参数分片取 block argument</b>。',
  code: `// 题 1：named_computation<"bar"> 会被 outline 成什么？

// 题 2：outline 后函数参数的分片从哪里来？call 的 sharding 呢？

// 题 3：同一个名字 baz 出现两次，会生成几个函数？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col compact-ex', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">named_computation&lt;"bar"&gt;</span> 会被 outline 成什么？',
        a: '<b>一个 <span class="mono">func.func private @bar</span></b>（带 <span class="mono">sdy.original_func_name = "bar"</span> 属性），调用点变成 <span class="mono">call @bar(...)</span>。' +
           '<br><b>属性也搬家</b>：<span class="mono">named_computation</span> 上的 <span class="mono">random_attr</span>、<span class="mono">mhlo.frontend_attributes</span> 全部移到 <b>call</b> 上。' +
           '<br><span class="dim">因为 <span class="mono">named_computation</span> 是"内联的调用点"，属性语义上属于调用点。</span>'
      },
      {
        q: 'outline 后函数参数的分片从哪里来？<span class="mono">call</span> 的 sharding 呢？',
        a: '<b>函数参数</b>：来自 <b>block argument</b> 的分片（<span class="mono">%arg1: tensor&lt;8x2xi32&gt;</span> 那一侧），<b>不是</b> <span class="mono">in_shardings</span>。' +
           '<br><b>call 的 sharding</b>：来自 <b><span class="mono">out_shardings</span></b>。' +
           '<br><span class="dim">测试注释点明了：<span class="mono">we don\'t override the block argument shardings of the function @ignore_operand_shardings, but we set the argument shardings on the call to @foo</span>。理由：outline 后的函数<b>就是那个函数体</b>，它看到的是 block argument。</span>'
      },
      {
        q: '同一个名字 <span class="mono">baz</span> 出现两次，会生成几个函数？',
        a: '<b>两个</b>：<span class="mono">@baz</span> 与 <span class="mono">@baz_0</span> —— <b>每个实例一个函数</b>，名字冲突时加 <span class="mono">_0</span> 后缀。' +
           '<br><b>两个函数都带 <span class="mono">sdy.original_func_name = "baz"</span></b> —— 下游据此知道它们<b>同源</b>。' +
           '<br><span class="dim">有趣的是：分片<b>相同</b>与<b>不同</b>两种情形下，函数命名<b>一样</b>（都是 <span class="mono">@baz</span> / <span class="mono">@baz_0</span>），差别只在 call 的 sharding 上。</span>'
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
