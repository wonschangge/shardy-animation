/* ==========================================================================
   L4-16 · sink-and-convert
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/ 下 3 个文件
         (sink_data_flow_edges 251 / 13, sink_func_data_flow_edges 198 / 15,
          sharding_constraint_to_reshard 19 / 2) = 468 行 / 30 用例
   目标：讲透"把标注型 op 消除掉"的三个 pass。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 共同点 */
{
  kicker: 'L4-16 · 边下沉与转换',
  title: '共同点：<span class="hl-a">消除"标注型"的 op</span>',
  sub: '传播结束后，那些**只为传播服务**的 op 就没有存在必要了。',
  caption: '三个 pass 都在做"<b>让分片直接落在值上</b>"这件事。',
  code: `// 【三个 pass】
//   -sdy-sink-data-flow-edges            251 行 / 13 用例
//     把 sdy.data_flow_edge 的分片【下沉】到输入值，删除边 op
//   -sdy-sink-func-data-flow-edges       198 行 / 15 用例
//     把 sdy.func_data_flow_edge 的分片【下沉】，删除边 op
//   -sdy-sharding-constraint-to-reshard   19 行 /  2 用例
//     把 sdy.sharding_constraint 转成 sdy.reshard

// 【共同点】都在消除"标注型"的 op
//   这些 op 是【传播期的辅助机制】：
//     data_flow_edge      区域算子的边（L1-08 定义，L3-04 插入）
//     func_data_flow_edge 函数调用的边（L3-05 插入，L3-10 搬分片）
//     sharding_constraint 用户意图（L1-10 定义，传播消费它）
//   传播结束后 -> 分片应该【直接落在张量上】-> 这些 op 没有存在必要

// 【分片去哪】
//   边     -> 落到边的【输入值】上
//   约束   -> 【转成】sdy.reshard

// 一句话：
//   传播结束后，所有"标注型"的 op 都要消失
//   边被下沉（分片直接写在值上），约束被转成真正的搬运`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'sink_data_flow_edges', c: '#38bdf8', n: '13 用例',
        d: '<span class="mono">sdy.data_flow_edge</span><br>分片<b>下沉到输入值</b><br>删除边 op' },
      { t: 'sink_func_data_flow_edges', c: '#4ade80', n: '15 用例',
        d: '<span class="mono">sdy.func_data_flow_edge</span><br>同样下沉<br><span class="dim">函数调用作用域</span>' },
      { t: 'constraint_to_reshard', c: '#fbbf24', n: '2 用例',
        d: '<span class="mono">sdy.sharding_constraint</span><br>→ <span class="mono">sdy.reshard</span><br><span class="dim">只改 op 名</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:10.5px;overflow-wrap:anywhere">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:10.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>回顾 L1-08</b>：区域算子本身不带分片 —— 分片写在边上。',
        '<b>回顾 L3-05 / L3-10</b>：函数级边桥接调用点与函数体。',
        '<b>回顾 L1-10</b>：<span class="mono">sharding_constraint</span> 是"用户意图"，传播会消费它。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>一句话</b>：边被下沉（分片直接写在值上），约束被转成真正的搬运。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 边的下沉 */
{
  kicker: 'L4-16 · 边下沉与转换',
  title: '★ 边的下沉：<span class="hl-a">分片落到输入值上</span>',
  sub: '边 op 被删除，后续算子**直接读原值** —— 分片则落到那个值所在的算子上。',
  caption: '注意两种情形：边在 <b>block argument</b> 上 vs 边在<b>算子结果</b>上 —— 分片落点不同。',
  code: `// 【输入】边作用在 block argument（while 的循环变量）上
func.func @data_flow_edge_on_block_arg(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.constant dense<1> : tensor<i32>
  %2 = stablehlo.constant dense<32> : tensor<i32>
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %0) : tensor<32x96xf32>, tensor<i32>
    cond { ... } do {
    %4 = sdy.data_flow_edge %iterArg sharding=<@mesh, [{"a"}, {}]> : tensor<32x96xf32>
    //   ^^^^^^^^^^^^^^^^^^ 边声明"循环变量 %iterArg 的分片"
    %5 = stablehlo.add %iterArg_2, %1 : tensor<i32>
    %6 = stablehlo.add %4, %4 : tensor<32x96xf32>     // 用【边的结果】%4
    stablehlo.return %6, %5 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}

// 【输出】边被删除，直接用 %iterArg
// CHECK:      } do {
// CHECK-NEXT:   %[[ADD_1:.*]] = stablehlo.add %iterArg_2, %[[C1]]
// CHECK-NEXT:   %[[ADD_2:.*]] = stablehlo.add %iterArg, %iterArg
//                                              ^^^^^^^^^ 原来是 %4（边的结果）
// CHECK-NEXT:   stablehlo.return %[[ADD_2]], %[[ADD_1]]
// CHECK-NEXT: }
// CHECK-NOT:  sdy.sharding
// CHECK-NEXT: return %[[WHILE]]#0
//   ^^^^ 两处变化：① 边 op 被删；② add 直接用 %iterArg

// 【边作用在【算子结果】上时】分片落在 while op 上
// CHECK:      %[[WHILE:.*]]:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %[[C0]])
// CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>, <@mesh, []>]>}
//   ^^^^^^^^^^ 分片落在 while op 上（因为边作用的是 while 的【结果】%3#0）

// 【下沉的目标】边 op 的【输入值】
//   分片从"边的属性"变成"那个值所在【算子】上的分片"`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '边在 block argument', c: '#38bdf8',
        d: '分片落到 <span class="mono">while</span> 的<br><b>操作数</b>（<span class="mono">%arg0</span>）上<br><span class="dim">while op 上无 sharding 属性</span>' },
      { t: '边在算子结果', c: '#4ade80',
        d: '分片落到 <b>while op 本身</b><br><span class="mono">{sdy.sharding = ...}</span><br><span class="dim">紧跟 while 行（CHECK-SAME）</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>两种情形落点不同</b> —— 取决于边作用的那个值<b>是谁定义的</b>。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '边作用在 <span class="mono">%3#0</span>（while 的结果）→ 分片就落在 <span class="mono">while</span> op 上。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>共同动作</b>：删除边 op，让后续算子<b>直接读原值</b>。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>下沉的目标</b>：边 op 的<b>输入值</b> —— 分片从"边的属性"变成"那个值所在算子上的分片"。';
    });
  }
},

/* ------------------------------------------------ 3 边界与空缺 */
{
  kicker: 'L4-16 · 边下沉与转换',
  title: '13 个用例的<span class="hl-a">边界</span>与一处<span class="hl-a">测试空缺</span>',
  sub: '边的数量与 iterArg 数量**可以不一致** —— 这带来了多种边界情形。',
  caption: '测试里有一处 <b>TODO 注释</b>，明确指出某个规则<b>还没有测试覆盖</b>。',
  code: `// 【13 个用例覆盖的边界】
//   data_flow_edge_on_block_arg   边在 block argument 上
//   data_flow_edge_on_op_result   边在算子结果上
//   no_shardings                  边【没有】sharding
//   some_edges_have_sharding      【部分】边有 sharding
//   all_edges_have_sharding       【全部】边有 sharding
//   missing_edge                  【缺】某条边
//   sharding_overrided            分片被【覆盖】
//   edge_missing_sharding         边缺 sharding

// 【注意 missing_edge】
//   while 有多个 iterArg，但只写了【部分】边
//   -> 说明边的数量与 iterArg 数量可以【不匹配】

// 【测试开头的 TODO 注释】（原文）
// TODO(tomnatan): once ops like while are allowed to have shardings with
// different meshes, add a test that verifies that the first mesh name is used
// for missing shardings.
//
// 读法：
//   目前 while 上的分片【必须用同一个 mesh】
//   "缺 sharding 时用第一个 mesh 名"这个规则【还没有测试覆盖】
//   -> 这是【已知的测试空缺】

// 【为什么值得在课件里点出】
//   上游测试文件里的 TODO 是【真实的信息】
//   它告诉我们：这条规则存在，但当前【不受测试保护】
//   读者如果自己写 IR 遇到这个情形，要特别小心
//
// 【回顾 AGENTS.md 的红线】
//   "测试文件里的注释不一定是真的"
//   -> TODO 这类注释描述的是【未来计划】，不是当前行为
//   -> 不能据此推断当前语义`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '边在 block arg', c: '#38bdf8' }, { t: '边在算子结果', c: '#4ade80' },
      { t: '无 sharding', c: '#94a3b8' }, { t: '部分有 sharding', c: '#fbbf24' },
      { t: '全部有 sharding', c: '#4ade80' }, { t: '缺某条边', c: '#fb7185' },
      { t: '分片被覆盖', c: '#c084fc' }, { t: '边缺 sharding', c: '#f472b6' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:150px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div style="font-size:10.5px;color:${x.c};line-height:1.35">${x.t}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 100)); msg.innerHTML = '<b>8 类边界</b>（13 个用例中有些重复覆盖）—— 因为边的数量与 iterArg 数量<b>可以不一致</b>。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>注意 <span class="mono">missing_edge</span></b>：while 有多个 iterArg，但只写了部分边。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>测试里的 TODO</b>：<span class="mono">once ops like while are allowed to have shardings with different meshes...</span> —— 说明某规则<b>还没有测试覆盖</b>。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<b>AGENTS.md 的红线</b>：<b>TODO 描述的是未来计划，不是当前行为</b> —— 不能据此推断当前语义。';
    });
  }
},

/* ------------------------------------------------ 4 函数级边与转换 */
{
  kicker: 'L4-16 · 边下沉与转换',
  title: '函数级边与 <span class="mono hl-a">constraint → reshard</span>',
  sub: '函数级边的作用域是**函数调用**；`constraint` 的转换则是**只改 op 名**。',
  caption: '两者都回指前面的课：L3-05 / L3-10（函数级边）、L1-09 / L1-10（约束）。',
  code: `// 【函数级边：函数内部】边被删，直接用原值
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xf32>
  %1 = stablehlo.negate %0: tensor<8xf32>
  return %1 : tensor<8xf32>
}
// CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %arg0
//                                            ^^^^^ 原来是 %0（边的结果）

// 【函数级边：调用点上】同样被删
func.func @simple_call_graph_on_func_with_single_argument(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = sdy.func_data_flow_edge %1 : tensor<8xf32>
  %3 = stablehlo.abs %2 : tensor<8xf32>
  return %3 : tensor<8xf32>
}
// CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[CALL]]
//                                            ^^^^^^^ 原来是 %2（边的结果）

// 【两种边的区别】
//   sdy.data_flow_edge       区域算子内部（while/case 的 block）  L1-08 / L3-04
//   sdy.func_data_flow_edge  函数调用（参数与结果）              L3-05 / L3-10
// 共同点：都是"桥接"op —— 把分片从一处传递到另一处
//         传播结束后桥接使命完成 -> 下沉并删除

// 【constraint -> reshard】只改 op 名
func.func @sharding_constraint_to_reshard(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {?}]> {foo} :  tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
// CHECK: %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {?}]> {foo} : tensor<8x8xf32>
//            ^^^^^^^^^^ op 名变了，分片参数与属性 {foo} 【完全不变】

// 【token 的特殊情形】0 维 reshard
func.func @token_sharding_constraint_to_reshard(%arg0: !stablehlo.token) -> !stablehlo.token {
  %0 = sdy.sharding_constraint %arg0 <@mesh, []> : !stablehlo.token
  return %0 : !stablehlo.token
}
// CHECK: %0 = sdy.reshard %arg0 <@mesh, []> : !stablehlo.token
// token 是【无维度】的值 -> 0 维 reshard

// 【为什么需要转换】
//   传播期：constraint 是【用户意图】—— 传播【消费】它（L1-10）
//           语义是"我希望这里是这个分片"（【声明性】）
//   传播后：如果它还在（如悬空约束，L1-09）
//           -> 就变成一条【真正的搬运】-> 转成 reshard
//   reshard 的语义是"请把它搬成这样"（【命令性】）
//   传播结束后，"我希望"已经没有意义 -> 统一转成 reshard`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '函数级边', c: '#4ade80', n: '15 用例',
        d: '函数内部 + 调用点<br>都<b>下沉并删除</b><br><span class="dim">L3-05 / L3-10</span>' },
      { t: 'constraint → reshard', c: '#fbbf24', n: '2 用例',
        d: '<b>只改 op 名</b><br>分片参数与属性保留<br><span class="dim">声明性 → 命令性</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>两种边的作用域不同</b>：一个管区域内部，一个管函数调用 —— 但处理方式一样（下沉 + 删除）。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>关键区分</b>：<span class="mono">constraint</span> 是<b>声明性</b>（"我希望"），<span class="mono">reshard</span> 是<b>命令性</b>（"请搬"）。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>为什么必须转</b>：传播结束后"我希望"已经没有意义 —— 要么已满足（冗余），要么必须真搬。';
    });
    tl.at(12000, () => {
      msg.innerHTML = '<b>与 L4-10 的关系</b>：L4-10 消除<b>冗余</b>的 permute；本 pass 把<b>约束</b>转成<b>搬运</b>。';
    });
  }
},

/* ------------------------------------------------ 5 族谱 */
{
  kicker: 'L4-16 · 边下沉与转换',
  title: '3 个 pass / 30 个用例的<span class="hl-a">族谱</span>',
  sub: '三个 pass 都很小，但都是"**让 IR 变干净**"的必要步骤。',
  caption: '本课是 L4 倒数第二课 —— 下一课是最后的杂项清理。',
  code: `// 【族谱】3 个文件 / 468 行 / 30 用例
//   sink_data_flow_edges           251 行 / 13 用例   区域算子的边
//   sink_func_data_flow_edges      198 行 / 15 用例   函数调用的边
//   sharding_constraint_to_reshard  19 行 /  2 用例   约束 -> 搬运
//
// 前两个较大（因为边的形态组合多），第三个极小（只改 op 名）

// 【跨课呼应】
//   sink_data_flow_edges         <- L1-08（边的定义）、L3-04（插入边）
//   sink_func_data_flow_edges    <- L3-05（插入函数级边）、L3-10（搬分片）
//   constraint_to_reshard        <- L1-09（悬空约束）、L1-10（生命周期）
//
// 【在导出流水线中的位置】
//   L4-01 的 NOTE 说过：
//     we apply sdy-add-data-flow-edges first, to make sure
//     sdy-sink-data-flow-edges is applied before any pass that operated on
//     ShardableDataFlowOpInterface rather than DataFlowEdgeOp.
//   ^^^^ 本课的 sink 就是那个"必须先跑"的 pass！
//   因为后续 pass 期望分片【已经落在张量上】，而不是还在边上

// 一句话总结：
//   传播结束后，所有"标注型"的 op 都要消失
//   边被下沉（分片直接写在值上）
//   约束被转成真正的搬运（reshard）`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: 'data_flow_edges', n: 13, c: '#38bdf8' },
      { t: 'func_data_flow_edges', n: 15, c: '#4ade80' },
      { t: 'constraint_to_reshard', n: 2, c: '#fbbf24' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:190px;opacity:.4;transition:.35s;border-color:${f.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="card-t mono" style="color:${f.c};font-size:10px;overflow-wrap:anywhere">${f.t}</div>
        <div class="big" style="font-size:20px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 130)); msg.innerHTML = '<b>30 个用例</b> —— 前两个较大（边的形态组合多），第三个极小（只改 op 名）。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>在流水线中的位置</b>：L4-01 的 NOTE 说 <span class="mono">sdy-sink-data-flow-edges</span> <b>必须先跑</b>。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>为什么必须先跑</b>：后续 pass 期望分片<b>已经落在张量上</b>，而不是还在边上。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L4-17 讲 <span class="mono">merger-and-debug-cleanup</span> —— L4 最后一课。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L4-16 · 练习',
  title: '练一练：<span class="hl-a">分片下沉到哪</span>',
  sub: '三道题分别考：边的下沉、两种边的区别、约束的转换。',
  caption: '一句话总结：<b>边下沉到输入值，约束转成 reshard</b>。',
  code: `// 题 1：边 op 被删除后，它的分片去了哪里？

// 题 2：data_flow_edge 与 func_data_flow_edge 有什么区别？

// 题 3：sharding_constraint 为什么不能直接写成 reshard？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col compact-ex', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '边 op 被删除后，它的分片去了哪里？',
        a: '<b>下沉到边的【输入值】上</b> —— 分片从"边的属性"变成"那个值所在<b>算子</b>上的分片"。' +
           '<br><b>具体落点取决于值是谁定义的</b>：' +
           '<br>• 边在 <b>block argument</b> 上 → 分片落到 <span class="mono">while</span> 的<b>操作数</b>上；' +
           '<br>• 边在<b>算子结果</b>上 → 分片落到 <b>while op 本身</b>上。' +
           '<br><span class="dim">同时后续算子改成<b>直接读原值</b>（如 <span class="mono">add %4, %4</span> → <span class="mono">add %iterArg, %iterArg</span>）。</span>'
      },
      {
        q: '<span class="mono">data_flow_edge</span> 与 <span class="mono">func_data_flow_edge</span> 有什么区别？',
        a: '<b>作用域不同</b>：' +
           '<br>• <span class="mono">sdy.data_flow_edge</span> —— <b>区域算子内部</b>（<span class="mono">while</span>/<span class="mono">case</span> 的 block），L1-08 定义、L3-04 插入；' +
           '<br>• <span class="mono">sdy.func_data_flow_edge</span> —— <b>函数调用</b>（参数与结果），L3-05 插入、L3-10 搬分片。' +
           '<br><b>共同点</b>：都是"<b>桥接</b>"op —— 把分片从一处传递到另一处；传播结束后桥接使命完成 → <b>下沉并删除</b>。' +
           '<br><span class="dim">处理方式完全一样，只是作用的 IR 结构不同。</span>'
      },
      {
        q: '<span class="mono">sharding_constraint</span> 为什么不能直接写成 <span class="mono">reshard</span>？',
        a: '因为两者<b>语义不同</b>：' +
           '<br>• <span class="mono">sharding_constraint</span> 是<b>声明性</b>的 —— "我<b>希望</b>这里是这个分片"，允许传播器<b>自由选择</b>如何满足（可能通过改变上游来满足）；' +
           '<br>• <span class="mono">reshard</span> 是<b>命令性</b>的 —— "请把它<b>搬</b>成这样"，是明确的搬运指令。' +
           '<br><b>传播期</b>需要前者（让传播有自由度）；<b>传播结束后</b>"我希望"已经没有意义 —— 要么已满足（冗余），要么必须真搬 → 所以统一转成 <span class="mono">reshard</span>。' +
           '<br><span class="dim">转换本身只改 op 名，分片参数与附加属性（如 <span class="mono">{foo}</span>）完全保留。</span>'
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
