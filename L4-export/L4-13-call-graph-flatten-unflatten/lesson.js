/* ==========================================================================
   L4-13 · call-graph-flatten-unflatten
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/ 下 2 个文件
         (unflatten_call_graph 1646 行 / 115 用例
          unflatten_call_graph_dedup_functions_fully_true 1665 / 117)
   目标：讲透"按 in/out 分片去重函数"与 dedup-functions-fully 的取舍。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 去重 */
{
  kicker: 'L4-13 · 调用图还原',
  title: '★ 去重：<span class="hl-a">两个函数合并成一个</span>',
  sub: '按 **in/out 分片**判断能否合并 —— 分片相同就合并，调用点统一重定向。',
  caption: '本课是 <b>L4-12 的直接后续</b>：outline 产生的重复函数在这里被收敛回去。',
  code: `// RUN: sdy_opt %s -split-input-file -sdy-unflatten-call-graph | FileCheck %s

// 【输入】两个函数，in/out 分片【完全相同】
func.func private @baz(%arg0: tensor<8x2xi32> {...<@mesh, [{}, {"y"}]>})
    -> (tensor<8x2xi32> {...<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {...} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
func.func private @baz_0(%arg0: tensor<8x2xi32> {...<@mesh, [{}, {"y"}]>})
    -> (tensor<8x2xi32> {...<@mesh, [{"x"}, {}]>})
attributes { sdy.original_func_name = "baz" } {          // <- 同源标记！
  %0 = stablehlo.multiply %arg0, %arg0 {...} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// 【输入】两个调用点分别指向两个函数
func.func @multiple_same_calls_same_shardings(...) -> ... {
  %0 = call @baz(%arg0) {...} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz_0(%arg0) {...} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// 【输出】两个调用点【都指向 @baz】
// CHECK-NEXT: %0 = call @baz(%arg0) {sdy.sharding = ...<@mesh, [{"x"}, {}]>]>} : ...
// CHECK-NEXT: %1 = call @baz(%arg0) {sdy.sharding = ...<@mesh, [{"x"}, {}]>]>} : ...
//                          ^^^^^ @baz_0 的调用点被【重定向】到 @baz
// CHECK-NEXT: return %1 : tensor<8x2xi32>

// "unflatten" 的含义：
//   L4-12: named_computation × N  --outline-->  @baz, @baz_0, @baz_1, ...
//   L4-13: @baz, @baz_0, ...      --去重-->     @baz`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '输入', c: '#38bdf8', d: '<span class="mono">@baz</span> 与 <span class="mono">@baz_0</span><br>in/out 分片<b>完全相同</b><br>两个调用点各指一个' },
      { t: '判断依据', c: '#fbbf24', d: '看 <b>in/out 分片</b><br>相同 → 可合并<br><span class="dim">@baz_0 带 original_func_name 表明同源</span>' },
      { t: '输出', c: '#4ade80', d: '两个 call <b>都指向 @baz</b><br>调用点被<b>重定向</b>' },
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
        '<span class="mono">@baz_0</span> 正是 L4-12 outline 时产生的（带 <span class="mono">original_func_name</span>）。',
        '<b>为什么安全</b>：两个函数的签名（in/out 分片）一模一样 —— 互换调用点不会改变语义。',
        '<b>收益</b>：函数数量变少 → IR 更小、编译更快。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>与 L4-12 的关系</b>：outline 让函数<b>变多</b>（每个实例一个），本课让它<b>变少</b>。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 选项取舍 */
{
  kicker: 'L4-13 · 调用图还原',
  title: '★ <span class="mono hl-a">dedup-functions-fully</span> 的取舍',
  sub: '分片**不同**时：默认**不合并**；`true` 时**合并** —— 因为真正的分片要求由**调用点**决定。',
  caption: '这正是 TODOLIST 验收点问的："能判断两个函数在给定选项下是否会被合并"。',
  code: `// 两个函数的【结果分片不同】：
//   @baz    结果 = [{"x"}, {}]
//   @baz_0  结果 = [{"x"}, {"y"}]        <- 不同！

// 【默认】dedup-functions-fully=false
// CHECK-NEXT: %0 = call @baz(%arg0)
//    {sdy.sharding = ...<@mesh, [{"x"}, {}]>]>} : ...
// CHECK-NEXT: %1 = call @baz_0(%arg0)
//    {sdy.sharding = ...<@mesh, [{"x"}, {"y"}]>]>} : ...
//                    ^^^^^ 分别指向两个函数 -> 【不合并】

// 【true】dedup-functions-fully=true
// CHECK-NEXT: %0 = call @baz(%arg0)
//    {sdy.sharding = ...<@mesh, [{"x"}, {}]>]>} : ...
// CHECK-NEXT: %1 = call @baz(%arg0)
//    {sdy.sharding = ...<@mesh, [{"x"}, {"y"}]>]>} : ...
//                    ^^^^^ 都指向 @baz -> 【合并了】！
//    注意第二个 call 的 sharding 【仍然是】 [{"x"}, {"y"}]（与第一个不同）

// 【为什么 true 时合并是可行的】
//   函数的分片信息在【调用点】上 —— 看 call ... {sdy.sharding = ...}
//   所以即使两个调用点要求不同分片，也可以调用【同一个函数】
//   函数的签名只是一个"默认约定"，真正的分片要求由调用点决定
//
//   这正是 L3-06 / L3-10 讲过的：
//     out_shardings 只来自【调用点】
//     （L3-06 的 NOTE：we ignore any arg/result shardings on the function）

// 【取舍】
//   默认：保守 —— 每个函数的分片自洽，但函数更多
//   true：激进 —— 函数更少（IR 更小、编译更快），但依赖调用点的 sharding
//   收益：函数数量 ↓    代价：下游必须【总是读调用点的 sharding】`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '默认 false', c: '#fbbf24', tag: '按 in/out 分片',
        d: '分片不同 → <b>不合并</b><br><span class="mono">call @baz</span> + <span class="mono">call @baz_0</span><br><span class="dim">每个函数分片自洽</span>' },
      { t: 'true', c: '#4ade80', tag: '完全去重',
        d: '分片不同也 <b>合并</b><br><span class="mono">call @baz</span> × 2<br><span class="dim">依赖调用点的 sharding</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:12px">${x.t} <span class="faint small">${x.tag}</span></div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>默认很保守</b>：只要 in/out 分片有一处不同，就保留两个函数。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>true 很激进</b>：即使分片不同也合并 —— 但第二个 call 的 sharding <b>仍然是自己的</b>。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>为什么可行</b>：函数的分片信息在<b>调用点</b>上（<span class="mono">call ... {sdy.sharding = ...}</span>）。';
    });
    tl.at(12200, () => {
      msg.innerHTML = '<b>回到 L3-06</b>：那里的 NOTE 说 <span class="mono">we ignore any arg/result shardings on the function</span> —— 函数签名只是"默认约定"。';
    });
    tl.at(15200, () => {
      msg.innerHTML = '<b>验收点答案</b>：看 in/out 分片是否相同 —— 相同则<b>总是合并</b>；不同则<b>只有 true 才合并</b>。';
    });
  }
},

/* ------------------------------------------------ 3 用例差集 */
{
  kicker: 'L4-13 · 调用图还原',
  title: '两个文件的<span class="hl-a">用例差集</span>',
  sub: '115 / 117 个用例，绝大多数相同。差集正好揭示了 `true` 额外覆盖的场景。',
  caption: '差集分析是理解"选项改变了什么"的<b>捷径</b>。',
  code: `// 【仅 true 有的用例】
//   multiple_same_calls_different_shardings_different_number_of_call_sites_one_called_twice
//     调用点【数量不同】（一个被调两次）
//   multiple_same_calls_different_shardings_different_number_of_call_sites_multiple_func_origins
//     调用点数量不同 + 【多个来源】
//   single_call_func_result_empty_sharding_call_has_sharding
//     函数结果【空分片】，call 有分片
//   two_calls_same_origin_one_call_with_empty_sharding
//     一个 call 【空分片】
//   three_calls_same_origin_func_with_two_calls_results_no_sharding
//     三个调用点，两个结果【无分片】
//   three_calls_same_origin_func_with_one_call_results_no_sharding
//     一个结果无分片

// 【默认版本专有的用例】（节选）
//   single_call_func_result_no_sharding_call_has_sharding
//   single_call_func_arg_has_sharding_call_arg_no_sharding
//   single_call_func_arg_no_sharding_call_arg_has_sharding
//   non_flat_manual_computation_and_non_manual_computation_calls_same_original_func
//   simple_non_flat_with_manual_computation_on_inner_most_call
//   two_manual_computations_call_the_same_func
//     ^^^^ 覆盖"参数分片与 call 实参分片不一致"的情形
//          -> 保守模式下这些差异会导致【不合并】

// 读法：
//   true 需要额外覆盖"【空分片】"与"【调用点数量不同】"
//     因为完全去重时，这些差异【不再阻止合并】-> 必须有测试锁定行为
//   默认版本专有那些"分片不一致"的用例
//     因为保守模式下它们【会导致不合并】-> 必须验证行为正确

// 这体现了测试设计的原则：
//   每个选项都要有【针对该选项特有行为】的用例`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '仅 true 有', c: '#4ade80', n: '6+ 个',
        d: '<b>空分片</b>的情形<br><b>调用点数量不同</b><br>多个来源' },
      { t: '默认专有', c: '#fbbf24', n: '6+ 个',
        d: '参数分片与 call 实参<br><b>不一致</b>的情形<br>与 manual_computation 交叉' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>为什么 true 需要这些</b>：完全去重时，空分片/调用点数量差异<b>不再阻止合并</b>。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>为什么默认需要那些</b>：保守模式下，分片不一致<b>会导致不合并</b> —— 必须验证行为正确。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>测试设计原则</b>：每个选项都要有<b>针对该选项特有行为</b>的用例。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>方法论</b>：<b>差集分析</b>是理解"选项改变了什么"的捷径 —— 比逐个读用例快得多。';
    });
  }
},

/* ------------------------------------------------ 4 与 L4-12 衔接 */
{
  kicker: 'L4-13 · 调用图还原',
  title: '与 L4-12 的<span class="hl-a">衔接</span>：先变多，再变少',
  sub: 'L4-12 的 outline 必须**保守**（每个实例一个函数）；本课在全局视角下再做一次去重。',
  caption: '<span class="mono">sdy.original_func_name</span> 在这里发挥作用 —— 它是 outline 留下的<b>同源标记</b>。',
  code: `// 【两步走】
//   L4-12 outline:  named_computation × N  ->  @baz, @baz_0, @baz_1, ...
//                                              函数【变多】（每个实例一个）
//   L4-13 去重:     @baz, @baz_0, ...      ->  @baz
//                                              函数【变少】（收敛回去）

// 【为什么 L4-12 不能直接去重】
//   outline 时是【局部视角】—— 处理一个 named_computation 时
//   不知道别处是否还有同源且同分片的实例
//   -> 只能保守地每个实例生成一个函数
//   本课在【全局视角】下 —— 能看到所有函数，可以安全合并

// 【sdy.original_func_name 的作用】
//   它是 L4-12 留下的"同源标记"
//   让 L4-13 知道"@baz_0 原本也叫 baz"
//   没有这个属性，去重只能靠"函数体是否相同"来判断
//     -> 更贵且不可靠

// 【导入 vs 导出的镜像】
//   导入期 flatten_call_graph（415 行 / 54 用例，属 L3 覆盖）
//   导出期 unflatten_call_graph（本课）
//   两个 pass 方向相反，但都在处理"函数太多/太少"的问题

// 一句话总结：
//   unflatten_call_graph 按 in/out 分片去重函数
//   默认保守（分片不同就不合并）
//   dedup-functions-fully=true 激进（完全去重）
//   因为真正的分片要求由【调用点】决定`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:18px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const node = (label, sub, color) => {
      const c = U.el('div', { class: 'col', style: 'gap:5px;align-items:center;opacity:0;transition:.45s' });
      c.innerHTML = `<div class="chip ${color}" style="padding:7px 12px;font-size:11px">${label}</div>
        <div class="small faint" style="font-size:9.5px;text-align:center;max-width:150px;line-height:1.35">${sub}</div>`;
      demo.appendChild(c); return c;
    };

    tl.at(700, () => {
      node('named_computation × N', '源 IR 里的多个实例', 'c0').style.opacity = '1';
      msg.innerHTML = '起点：源 IR 里有 <b>N 个</b> <span class="mono">named_computation</span> 实例。';
    });
    tl.at(4200, () => {
      demo.insertAdjacentHTML('beforeend', '<div class="arrow" style="font-size:16px">→</div>');
      node('@baz, @baz_0, ...', 'L4-12 outline<br>函数【变多】', 'c2').style.opacity = '1';
      msg.innerHTML = '<b>L4-12</b>：每个实例生成一个函数 —— 因为 outline 时是<b>局部视角</b>，不知道别处有没有同分片的。';
    });
    tl.at(8600, () => {
      demo.insertAdjacentHTML('beforeend', '<div class="arrow" style="font-size:16px">→</div>');
      node('@baz', 'L4-13 去重<br>函数【变少】', 'c4').style.opacity = '1';
      msg.innerHTML = '<b>L4-13</b>：在<b>全局视角</b>下看到所有函数，可以安全合并。';
    });
    tl.at(12400, () => {
      msg.innerHTML = '<b><span class="mono">sdy.original_func_name</span> 的作用</b>：outline 留下的<b>同源标记</b> —— 让去重知道"<span class="mono">@baz_0</span> 原本也叫 <span class="mono">baz</span>"。';
    });
    tl.at(15200, () => {
      msg.innerHTML = '<b>一句话</b>：先变多（保守 outline），再变少（全局去重）。';
    });
  }
},

/* ------------------------------------------------ 5 族谱 */
{
  kicker: 'L4-13 · 调用图还原',
  title: '232 个用例的<span class="hl-a">族谱</span>与小结',
  sub: '两个文件几乎一样，差别只在去重的激进程度 —— 这个"对照实验"的设计本身就是教学材料。',
  caption: '同一个 pass 跑两次、用两个选项、两份期望输出 —— 与 L4-05 / L4-09 的"多 RUN 行"同类。',
  code: `// 【族谱】
//   unflatten_call_graph                        1646 行 / 115 用例   默认
//   unflatten_call_graph_dedup_functions_fully_true 1665 / 117       完全去重
//   合计 3311 行 / 232 用例

// 从用例名可见覆盖的场景：
//   singleton / vanilla_call                    基础
//   ignore_operand_shardings                    参数分片来源（与 L4-12 同名）
//   multiple_same_calls_{same,different}_shardings  去重核心
//   non_flat_nested_calls_same_shardings        嵌套调用
//   calls_same_func_with_manual_axes            与 manual_axes 交互
//   calls_same_funcs_*_one_inside_manual_computation   区域内
//   calls_same_funcs_on_the_same_manual_axes_different_shardings
//   two_manual_computations_call_the_same_func  manual_computation 交叉
//   ..._different_number_of_call_sites_*        调用点数量不同（仅 true）
//   ..._empty_sharding / ..._no_sharding        空分片情形

// 【注意与 L4-12 的同名用例】
//   ignore_operand_shardings 在两个文件里都出现
//   -> L4-12 讲"outline 时参数分片从哪来"
//   -> L4-13 讲"去重时怎么用这个分片"
//   同名用例，两个视角

// 【设计上的对照实验】
//   同一个 pass、两个选项、两份期望输出
//   -> 读者可以【逐用例对比】看选项改变了什么
//   与 L4-05（mark-partial-result）、L4-09（halo exchange）同类

// 一句话总结：
//   按 in/out 分片去重函数，把调用点统一到同一个函数上
//   默认保守；dedup-functions-fully=true 激进
//   因为真正的分片要求由【调用点】决定`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '默认', n: 115, c: '#fbbf24' }, { t: '完全去重', n: 117, c: '#4ade80' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.4;transition:.35s;border-color:${f.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="card-t" style="color:${f.c};font-size:12px">${f.t}</div>
        <div class="big" style="font-size:22px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 150)); msg.innerHTML = '<b>232 个用例</b> —— 两个文件几乎一样，只差 <b>2 个</b>用例。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>差集分析是捷径</b>：从 6+ 个独有用例就能看出选项改变了什么。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>设计上的对照实验</b>：同 pass、两选项、两份期望 —— 与 L4-05 / L4-09 的多 RUN 行同类。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L4-14 讲 <span class="mono">single-device-and-unreduced</span>。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L4-13 · 练习',
  title: '练一练：<span class="hl-a">这两个函数会合并吗</span>',
  sub: '三道题分别考：去重依据、选项差异、与 L4-12 的关系。',
  caption: '一句话总结：<b>按 in/out 分片去重；true 时完全去重</b>。',
  code: `// 题 1：两个函数在什么条件下会被合并？

// 题 2：dedup-functions-fully=true 时，为什么分片不同也能合并？

// 题 3：L4-12 与 L4-13 的函数数量变化分别是什么方向？为什么？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col compact-ex', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '两个函数在什么条件下会被合并？',
        a: '<b>默认</b>：它们的 <b>in/out 分片完全相同</b>时才合并。' +
           '<br><b><span class="mono">dedup-functions-fully=true</span></b>：<b>完全去重</b> —— 分片不同也合并。' +
           '<br><span class="dim">合并的表现是：调用点被<b>重定向</b>到同一个函数（<span class="mono">call @baz_0</span> → <span class="mono">call @baz</span>）。</span>'
      },
      {
        q: '<span class="mono">dedup-functions-fully=true</span> 时，为什么分片不同也能合并？',
        a: '因为<b>函数的分片信息在【调用点】上</b> —— 看 <span class="mono">call ... {sdy.sharding = ...}</span>。' +
           '<br>所以即使两个调用点要求不同的 out 分片，也可以调用<b>同一个函数</b>：函数签名只是一个"默认约定"，<b>真正的分片要求由调用点决定</b>。' +
           '<br><span class="dim">这正是 L3-06 的 NOTE 说的：<span class="mono">we ignore any arg/result shardings on the function</span>。<b>收益</b>是函数更少（IR 更小、编译更快）；<b>代价</b>是下游必须总是读调用点的 sharding。</span>'
      },
      {
        q: 'L4-12 与 L4-13 的函数数量变化分别是什么方向？为什么？',
        a: '<b>L4-12（outline）</b>：函数<b>变多</b> —— 每个 <span class="mono">named_computation</span> 实例生成一个（<span class="mono">@baz</span>、<span class="mono">@baz_0</span>…）。' +
           '<br><b>L4-13（去重）</b>：函数<b>变少</b> —— 同源且分片相同的被收敛回一个。' +
           '<br><b>为什么 L4-12 不直接去重</b>：outline 时是<b>局部视角</b>，处理一个实例时不知道别处有没有同分片的；本课在<b>全局视角</b>下才能安全合并。' +
           '<br><span class="dim"><span class="mono">sdy.original_func_name</span> 是 outline 留下的<b>同源标记</b>，让去重能识别"它们原本同名"。</span>'
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
