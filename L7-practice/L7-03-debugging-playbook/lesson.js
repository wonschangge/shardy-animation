/* ==========================================================================
   L7-03 · debugging-playbook   （P0）
   --------------------------------------------------------------------------
   覆盖：transforms/propagation/test/ 下 4 个文件
         (data_flow_edges 820/40, dedup_functions_fully_true 304/21,
          registry_conservative 78/7, registry_failures 8/1)
   目标：给出"症状 → 检查项"的排查清单与三步法。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 四类线索 */
{
  kicker: 'L7-03 · 调试手册',
  title: '★ 四个文件 = <span class="hl-a">四类调试线索</span>',
  sub: '本课完成后<b>覆盖率将达 241/241 = 100%</b>。',
  caption: '这是 L7 的第三课 —— 从"策略"转到"<b>排错</b>"视角。',
  code: `// 【四个文件】
//   文件                                  行数   用例数   提供什么线索
//   propagation_pipeline_data_flow_edges   820    【40】   【区域算子】的传播
//   propagation_pipeline_dedup_functions_fully_true
//                                          304    21      【选项开关】
//   op_sharding_rule_registry_conservative  78     7      【保守规则】
//   op_sharding_rule_registry_failures       8     1      【诊断警告】

// 【★ 四类调试线索：症状 -> 该看哪个文件】
//   症状                        该看哪个文件
//   【分片没传播过去】          data_flow_edges（区域算子的数据流）
//   【出现意外的 all-gather】   registry_conservative（规则是否太保守）
//   【算子没有 sharding rule】  registry_failures（警告信息）
//   【想切换传播行为】          dedup_functions_fully_true（选项开关）

// 【★ 调试 Shardy 分片问题的三步法】
//   ① 先看【诊断警告】（算子有没有规则）
//   ② 再【切换选项】（定位到哪个 pass）
//   ③ 最后 【dump IR 逐步 diff】（精确定位）

// 【注意最小的文件最有用】
//   registry_failures 只有 8 行，但它给出的是【最直接的线索】——
//   一条明确的警告信息

// 一句话：
//   调试 = 先找线索（诊断）-> 再缩小范围（选项）-> 最后精确定位（dump）`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'data_flow_edges', c: '#38bdf8', n: '40 用例', d: '<b>区域算子</b><br>传播' },
      { t: 'dedup_functions', c: '#4ade80', n: '21 用例', d: '<b>选项开关</b>' },
      { t: 'conservative', c: '#fbbf24', n: '7 用例', d: '<b>保守规则</b>' },
      { t: 'registry_failures', c: '#fb7185', n: '1 用例', d: '<b>诊断警告</b><br><span class="dim">8 行但最有用</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:9px;color:${x.c};overflow-wrap:anywhere">${x.t}</div>
        <div class="small" style="font-size:9px;color:${x.c};margin:2px 0">${x.n}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>区域算子最容易出问题</b> —— 数据流要穿过 region 边界，所以有 40 个用例。',
        '<b>选项开关</b>：切换它看行为是否改变，是定位"问题在哪个 pass"的最快方法。',
        '<b>保守规则</b>：如果规则太保守（因子=1），传播不过去就会产生多余通信。',
        '<b>最直接的线索</b>：一条明确的警告信息 —— 只有 8 行，但价值最高。',
      ][i];
    }));
    tl.at(15200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 三步法</b>：先看诊断 → 再切选项 → 最后 dump IR 逐步 diff。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 诊断警告 */
{
  kicker: 'L7-03 · 调试手册',
  title: '★ 最直接的线索：<span class="mono hl-a">-verify-diagnostics</span>',
  sub: '**8 行**的文件，却给出最有用的信息 —— 一条明确的警告。',
  caption: '当算子"分片没传播过去"时，<b>第一件事</b>就是跑它。',
  code: `// RUN: sdy_opt %s -sdy-populate-op-sharding-rules -verify-diagnostics

// CHECK-LABEL: func @unknown_custom_op
func.func @unknown_custom_op(%arg0: tensor<8x2xui32>, %arg1: tensor<8x2xui32>) -> tensor<8x2xui64> {
  // expected-warning@+1 {{custom call @unknown_custom_op is unknown to SDY sharding rule registry}}
  %0 = stablehlo.custom_call @unknown_custom_op(%arg0, %arg1) : (tensor<8x2xui32>, tensor<8x2xui32>) -> tensor<8x2xui64>

// 【读法】本课最有价值的一处
//   -verify-diagnostics
//     —— 让 sdy_opt 【校验诊断信息】（而不只是输出 IR）
//   // expected-warning@+1 {{...}}
//     —— 断言【下一行】会产生一条【警告】，内容是
//        "custom call @unknown_custom_op is unknown to SDY sharding rule registry"

// 【★ 这条警告就是调试的入口】
//   当你的自定义算子"分片没传播过去"时，【第一件事】是跑
//     sdy_opt -sdy-populate-op-sharding-rules
//   看有没有这条警告
//   如果有 -> 说明【算子没有注册 sharding rule】（见 L7-04 的接入方法）

// 【★ -verify-diagnostics 的用法】
//   sdy_opt input.mlir -sdy-populate-op-sharding-rules -verify-diagnostics
//   它把"预期诊断"变成【可断言的测试】——
//   这是 MLIR 生态的【标准调试手法】

// 【注意是 expected-warning 而非 expected-error】
//   【缺少 sharding rule 不是错误】
//   只是"传播时会跳过这个算子"
//   -> 这也解释了为什么症状是"分片没传播过去"而不是"编译失败"`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '症状', c: '#fb7185', d: '自定义算子<br><b>分片没传播过去</b><br><span class="dim">但不报错</span>' },
      { t: '第一步', c: '#4ade80', d: '跑 <span class="mono">-sdy-populate-op-sharding-rules</span><br>看<b>警告</b>' },
      { t: '结论', c: '#38bdf8', d: '有警告 →<br><b>没注册 sharding rule</b><br>→ 见 L7-04' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>典型症状</b>：算子<b>不报错</b>，但分片就是传播不过去 —— 最难查的一类。',
        '<b>第一件事</b>：跑 <span class="mono">-sdy-populate-op-sharding-rules</span>，看有没有警告。',
        '<b>结论明确</b>：有这条警告 = 算子没注册规则 → 去 <b>L7-04</b> 看怎么接入。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>关键区分</b>：这是 <span class="mono">expected-warning</span> 而非 <span class="mono">error</span> —— <b>缺规则不报错</b>，只是跳过。';
    });
  }
},

/* ------------------------------------------------ 3 ★ 保守规则 */
{
  kicker: 'L7-03 · 调试手册',
  title: '★ <span class="mono hl-a">conservative-propagation</span>：规则太保守 → 多余通信',
  sub: '因子被设为 **1** 意味着"这一维不可分" —— 传播不过去就要 reshard。',
  caption: '如果你看到<b>意外的 all-gather</b>，先对比这个开关。',
  code: `// RUN: sdy_opt %s -sdy-populate-op-sharding-rules="conservative-propagation=true" 2>&1 | FileCheck %s

// CHECK-LABEL: func @concat
func.func @concat(%arg0: tensor<4x3x256xf32>, %arg1: tensor<4x5x256xf32>) -> tensor<4x8x256xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, k, j], [i, l, j])->([i, m, j]) {i=4, j=256, k=1, l=1, m=1}>
 %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<4x3x256xf32>, tensor<4x5x256xf32>) -> tensor<4x8x256xf32>
 return %0 : tensor<4x8x256xf32>
}

// 【读法】
//   conservative-propagation=true —— 打开【保守传播】模式
//   生成的规则：([i, k, j], [i, l, j])->([i, m, j]) {i=4, j=256, k=1, l=1, m=1}
//   注意 【k=1, l=1, m=1】—— 拼接维（k/l/m）的因子被设为 【1】！
//   因子为 1 意味着"【这一维不可分】"-> 【保守】（不冒险切拼接维）
//
// 【7 个用例】都是【形状类/窗口类】算子：
//   concat / conv / pad / pad_same_shape_permutation /
//   reduce_window / select_and_scatter / slice
//   —— 它们的规则容易出【歧义】
//
// 【★ 为什么这是调试线索】
//   如果你看到【意外的 all-gather】，可能是某个算子的规则【太保守】
//     （某维因子被设为 1 -> 传播不过去 -> 需要 reshard）
//   打开 conservative-propagation 对比一下
//     就能判断是不是规则的问题
//
// 【回顾 L2-10】
//   那里讲 op_sharding_rule_registry 与因子的含义
//   本课看到【保守模式】下规则长什么样`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '规则因子 = 1', c: '#fb7185', d: '该维<b>不可分</b><br>→ 传播<b>过不去</b><br>→ 需要 <span class="mono">reshard</span>' },
      { t: '症状', c: '#fbbf24', d: '<b>意外的 all-gather</b><br><span class="dim">本来不该有通信</span>' },
      { t: '排查', c: '#4ade80', d: '打开 <span class="mono">conservative-propagation</span> 对比<br>→ 判断是不是规则问题' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>保守规则的样子</b>：<span class="mono">k=1, l=1, m=1</span> —— 拼接维的因子被设为 1。',
        '<b>后果</b>：传播不过去 → 插入 reshard → 你看到<b>意外的 all-gather</b>。',
        '<b>排查手法</b>：切换 <span class="mono">conservative-propagation</span> 对比 —— 行为变了就说明是规则问题。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>为什么这 7 个算子</b>：形状类/窗口类算子的规则<b>容易出歧义</b>，所以需要保守模式兜底。';
    });
  }
},

/* ------------------------------------------------ 4 ★ 区域算子 */
{
  kicker: 'L7-03 · 调试手册',
  title: '★ <span class="mono hl-a">data_flow_edges</span>：区域算子最容易出问题',
  sub: '**820 行 / 40 个用例** —— 本课最大的文件，因为 `CaseOp`/`WhileOp` 最难。',
  caption: '数据流要<b>穿过 region 边界</b> —— 这是传播最脆弱的地方。',
  code: `// RUN: sdy_opt %s -sdy-propagation-pipeline -split-input-file 2>&1 | FileCheck %s

// Propagation tests for ops with data-flow edges like CaseOp and WhileOp

// 【读法】820 行 / 40 个用例 —— 本课最大的文件
//   注释直接点明："data-flow edges like CaseOp and WhileOp"
//   CaseOp / WhileOp 是【区域算子】—— 它们有【嵌套的 region】
//     数据流要【穿过 region 边界】
//   40 个用例说明这是【最容易出问题】的地方
//
// 【★ 为什么区域算子最容易出问题】（回顾 L2-07）
//   data-flow edges 描述"分片怎么从一个算子流到另一个算子"
//   区域算子的数据流要【穿过 region 边界】——
//     而 region 内的 block argument、sdy.return、嵌套结构
//     都可能【打断传播】
//
// 【★ 调试线索】
//   如果"分片没传播过去"发生在 while / case 里
//   先看 data_flow_edges 里【对应结构】的测试
//     —— 它展示了【正确】的传播结果
//
// 【回顾 L2-08】
//   那里讲 basic_propagation_data_flow_edges（基础情形）
//   本课的是【完整流水线】（-sdy-propagation-pipeline）的 40 个用例
//
// 【为什么用 -split-input-file】
//   40 个用例写在一个文件里，用 "//-----" 分隔
//   -split-input-file 让 sdy_opt 【逐个处理】并分别检查
//   -> 这是写大量小用例的标准手法（L6-00 讲过类似机制）`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '区域算子', c: '#fb7185', d: '<span class="mono">CaseOp</span> / <span class="mono">WhileOp</span><br>有<b>嵌套 region</b><br>数据流<b>穿过边界</b>' },
      { t: '易出问题', c: '#fbbf24', d: 'block argument<br><span class="mono">sdy.return</span><br>嵌套结构<b>打断传播</b>' },
      { t: '调试', c: '#4ade80', d: '看 <b>40 个用例</b><br>里对应结构的<br><b>正确</b>结果' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>为什么最难</b>：<span class="mono">CaseOp</span>/<span class="mono">WhileOp</span> 有<b>嵌套 region</b>，数据流要穿过边界。',
        '<b>三个易断点</b>：block argument、<span class="mono">sdy.return</span>、嵌套结构。',
        '<b>怎么用</b>：40 个用例覆盖了各种结构 —— 找到和你代码<b>对应</b>的那个，看正确结果。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>40 个用例 / 820 行</b> —— 这个体量本身就说明区域算子是<b>传播最脆弱</b>的地方。';
    });
  }
},

/* ------------------------------------------------ 5 ★ 两个调试工具 */
{
  kicker: 'L7-03 · 调试手册',
  title: '★ 两个调试工具（<span class="hl-a">已实测存在</span>）',
  sub: '`--debug-sharding-origins` 追溯来源；`--module-dump-directory` 定位步骤。',
  caption: '这两个 flag 我用 <span class="mono">sdy_opt --help</span> 实测确认过。',
  code: `// 【实测确认：两个 flag 都存在】
--debug-sharding-origins
    whether to save information about the origin of a sharding on the MLIR module.
    These would be the shardings on the function inputs, outputs, sharding
    constraints and manual computations before propagation.

--module-dump-directory=<string>
    where to dump any rewritten modules for debugging

// 【--debug-sharding-origins：追溯分片的来源】
//   用途：记录"【每个分片是从哪来的】"——
//     函数输入/输出、sharding_constraint、manual_computation 上的分片
//
//   【★ 调试场景】
//     "这个分片是谁塞进来的？"
//     打开它，就能看到传播【之前】的原始分片标注
//       -> 从而判断"意外的分片"是【用户写的】还是【传播产生的】
//
//   配套选项：--sink-debug-sharding-origins
//     （"Whether to sink the debug sharding origins info"）

// 【--module-dump-directory：dump 每一步的 IR】
//   用途：把【每个 pass 改写后的 module】都 dump 到指定目录
//
//   【★ 调试场景】
//     "分片是在哪一步消失的？"
//     指定一个目录，跑一遍，然后【逐步 diff】——
//       就能精确定位是【哪个 pass】改变了行为
//
// 【★ 两个工具的组合用法】
//   问题                    工具
//   分片【从哪来】          --debug-sharding-origins
//   分片【在哪一步变】      --module-dump-directory
//
// 【为什么"实测确认"很重要】
//   AGENTS.md §3.1 要求：课件中出现的每个 -sdy-* flag 必须与 passes.td 一致
//   这两个 flag 来自 TODOLIST 的讲解要点
//   【必须跑一遍 --help 确认】—— 不能只凭 TODOLIST 的表述
//   （这正是 §3.3 那条红线："凡测试文件里写着但没见过实际输出的，必须跑一遍确认"）`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '--debug-sharding-origins', c: '#38bdf8', d: '追溯分片的<b>来源</b><br>传播<b>之前</b>的标注<br><span class="dim">"这个分片是谁塞的？"</span>' },
      { t: '--module-dump-directory', c: '#4ade80', d: 'dump <b>每个 pass</b><br>之后的 IR<br><span class="dim">"分片在哪一步消失？"</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:10.5px;overflow-wrap:anywhere">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>追溯来源</b>：看到传播<b>之前</b>的原始标注 → 判断分片是<b>用户写的</b>还是<b>传播产生的</b>。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>定位步骤</b>：dump 每个 pass 后的 IR，<b>逐步 diff</b> → 精确定位到<b>哪个 pass</b>。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 组合用法</b>：一个回答"从哪来"、一个回答"在哪一步变"。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>我实测确认过这两个 flag</b>（<span class="mono">sdy_opt --help</span>）—— 不能只凭 TODOLIST 的表述（§3.3 红线）。';
    });
  }
},

/* ------------------------------------------------ 6 ★ 排查清单 */
{
  kicker: 'L7-03 · 调试手册',
  title: '★ <span class="hl-a">排查清单</span>与三步法',
  sub: '验收点：给定一个"坏"IR 能定位原因。',
  caption: '一句话：<b>先找线索 → 再缩小范围 → 最后精确定位</b>。',
  code: `// 【★ 排查清单：症状 -> 检查项】
//   症状                      第一步                          第二步
//   【分片没传播过去】        跑 -sdy-populate-op-sharding-rules
//                             -verify-diagnostics
//                             看警告                          看 data_flow_edges
//                                                             里对应结构
//   【出现意外的 all-gather】 对比 conservative-propagation=true  看是不是规则太保守
//                                                             （因子=1）
//   【不可整除报错】          看是不是 L5-09 的场景            检查网格轴大小
//                                                             与维度大小
//   【不确定分片从哪来】      --debug-sharding-origins         对比传播前后的标注
//   【不确定哪一步出问题】    --module-dump-directory          逐步 diff

// 【★ 三个通用手法】
//   ① 看【诊断信息】（-verify-diagnostics）—— 最直接
//   ② 【切换选项开关】（conservative-propagation /
//                       dedup-functions-fully）—— 定位到 pass
//   ③ 【dump 中间 IR】（--module-dump-directory）—— 逐步定位

// 【★ 一个重要的心态】
//   "分片没传播过去"【不报错】—— 这是最难查的一类问题
//   因为它不是编译错误，而是"静默地跳过了"
//   -> 所以【主动检查诊断信息】比等报错更重要
//
// 【★ 与前面课的呼应】
//   L2-07  data-flow edges 的概念（本课 40 个用例验证）
//   L2-08  基础 data-flow edges（本课是完整流水线版）
//   L2-10  op_sharding_rule_registry（本课看保守模式）
//   L3-07  flatten/unflatten call graph（本课看 dedup 选项）
//   L5-09  不可整除（本课排查清单里的一项）
//   L7-04  跨方言集成（"没注册规则"的解决方案）

// 【★ 本课在 L7 中的位置】
//   L7-01 端到端走查（【流水线】视角）
//   L7-02 并行策略（【策略】视角）
//   L7-03 调试手册（本课，【排错】视角）
//   L7-04 跨方言集成（【扩展】视角）

// 一句话总结：
//   调试 Shardy 分片问题的三步法
//   ① 先看诊断警告（算子有没有规则）
//   ② 再切换选项（定位到哪个 pass）
//   ③ 最后 dump IR 逐步 diff（精确定位）`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const steps = [
      { n: '①', t: '看诊断', c: '#fb7185', d: '<span class="mono">-verify-<br>diagnostics</span>' },
      { n: '②', t: '切选项', c: '#fbbf24', d: '<span class="mono">conservative-propagation</span>' },
      { n: '③', t: 'dump IR', c: '#4ade80', d: '<span class="mono">--module-dump-directory</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = steps.map(s => {
      const e = U.el('div', { class: 'card', style: `width:246px;opacity:.33;transition:.3s;border-color:${s.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="small mono" style="font-size:11px;color:${s.c}">${s.n}</div>
        <div style="font-size:11.5px;margin-top:3px">${s.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35;margin-top:3px">${s.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    steps.forEach((s, i) => tl.at(700 + i * 4200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>先找线索</b>：看诊断警告 —— 算子有没有注册 sharding rule。',
        '<b>缩小范围</b>：切换选项开关，看行为是否改变 → 定位到哪个 pass。',
        '<b>精确定位</b>：dump 每个 pass 的 IR，逐步 diff。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 一个重要的心态</b>：分片没传播过去<b>不报错</b> —— 所以要<b>主动检查</b>诊断信息。';
    });
  }
},

];
