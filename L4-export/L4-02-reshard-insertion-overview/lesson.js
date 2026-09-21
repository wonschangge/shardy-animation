/* ==========================================================================
   L4-02 · reshard-insertion-overview   （L4 本节总纲）
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/insert_explicit_reshards.mlir (682 行 / 53 用例)
         transforms/export/test/insert_func_call_reshards.mlir (488 / 42)
   目标：讲清"为什么需要插入 reshard"，为 L4-03~07 建立整体图景。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 中心思想 */
{
  kicker: 'L4-02 · reshard 插入总纲',
  title: '中心思想：<span class="hl-a">传播定分片，reshard 对齐冲突</span>',
  sub: '传播结束后每个张量都有分片了。但**算子自己的约束**可能仍不满足。',
  caption: '这是 L4 本节（L4-02～L4-07）的总纲 —— 后面几课按算子族展开。',
  code: `// RUN: sdy_opt %s -allow-unregistered-dialect \\
//        -sdy-insert-explicit-reshards='enable-full-version=false
//          mark-partial-result-with-unreduced-axes=true'

// 问题：传播是【各方向独立推导】的，可能出现算子约束不满足
//
// 以 dot 为例，它的要求是：
//   收缩维在【两个操作数】上必须同分片
//   非收缩维在【操作数与结果】之间必须同分片
//
// 但 lhs 与 rhs 可能被推导出冲突的分片
//
// 解法：插入显式的 sdy.reshard，让某一侧"搬"到另一侧的分片上

// 一句话：
//   传播决定"每个张量怎么切"
//   reshard 负责"把不兼容的地方对齐"

// 本文件 53 个用例的分布：
//   unreduced 处理     8 个
//   dot 的 reshard    24 个  <- 最大的一族
//   concatenate        3 个
//   状态转换          17 个`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'unreduced 处理', c: '#fbbf24', n: '8 个',
        d: '归约<b>时机</b>的三种程度<br>越晚越省计算' },
      { t: 'dot 的 reshard', c: '#38bdf8', n: '24 个',
        d: '<b>最大的一族</b><br>各种冲突类型的组合' },
      { t: 'concatenate', c: '#4ade80', n: '3 个',
        d: '拼接维的分片<br>必须一致' },
      { t: '状态转换', c: '#c084fc', n: '17 个',
        d: '分片 / 复制 / 未归约<br>之间的相互转换' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:180px;opacity:.33;transition:.35s;border-color:' + x.c + '55;padding:9px' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:11.5px">${x.t}</div>
        <div class="mono" style="font-size:16px;color:${x.c};margin:4px 0">${x.n}</div>
        <div class="card-d" style="font-size:10.5px;line-height:1.45">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>为什么延迟</b>：未归约时每个设备只算自己那一份部分和 —— 越晚归约越省计算。',
        '<b>为什么最多</b>：dot 有收缩维/非收缩维/批维三类维度，冲突组合最多。',
        '拼接维上各操作数必须同分片，否则无法直接拼。',
        'L1-02 讲过三种状态是<b>正交</b>的 —— 这里处理它们的相互转换。',
      ][i];
    }));
    tl.at(14400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>一句话</b>：传播决定"每个张量怎么切"，reshard 负责"把不兼容的地方对齐"。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 延迟归约 */
{
  kicker: 'L4-02 · reshard 插入总纲',
  title: '★ 最精彩的一族：<span class="hl-a">尽量延迟归约</span>',
  sub: '`unreduced` 轴上的 `all_reduce` 会被**尽量推迟** —— 越晚归约，中间能省的计算越多。',
  caption: '5 个用例覆盖"归约时机"的不同选择，从"立刻"到"完全不插"。',
  code: `// 【情形 ①】下游算子需要完整值 -> 立刻归约
func.func @all_reduce_on_func_input(
    %arg0: tensor<4x8xf32> {sdy.sharding = ...<@mesh, [{}, {}], unreduced={"y"}>},
    %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
// 输出：%[[ALL_REDUCE]] = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh, [{}, {}]>
//       %[[MUL]] = stablehlo.multiply %[[ALL_REDUCE]], %arg1
//       ^^^^^^^^^ 插在 multiply【之前】

// 【情形 ②】结果也接受未归约 -> 完全不插
func.func @unreduced_func_input_until_return(...)
    -> (tensor<4x8xf32> {...<@mesh, [{}, {}], unreduced={"y"}>}) {
  %0 = stablehlo.add %arg0, %arg0 {...unreduced={"y"}>} : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
// 输出：%[[ADD]] = stablehlo.add %arg0, %arg0
//       return %[[ADD]]
//       ^^^^^^^^^ 一条 all_reduce 都不插

// 【情形 ③】完全延迟到 return 前
// 输出：reduce / add 都在【未归约】状态下完成
//       %[[ALL_REDUCE]] = sdy.all_reduce {"x"} %[[ADD]] out_sharding=<@mesh, [{}, {}]>
//       return %[[ALL_REDUCE]]`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '立刻', c: '#fb7185', d: '下游需要完整值<br><b>插在最前</b>' },
      { t: '延迟到某算子前', c: '#fbbf24', d: '<span class="mono">all_reduce_delayed_until_op</span>' },
      { t: '部分延迟', c: '#38bdf8', d: '<span class="mono">all_reduce_partially_delayed_until_return</span>' },
      { t: '完全延迟', c: '#4ade80', d: '推迟到 <span class="mono">return</span> 前<br>中间全在未归约态' },
      { t: '完全不插', c: '#c084fc', d: '结果也接受未归约<br><b>一条都不插</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:140px;opacity:.33;transition:.35s;border-color:' + x.c + '55;padding:9px;text-align:center' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:11px">${x.t}</div>
        <div class="card-d" style="font-size:10px;line-height:1.4">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '最保守的选择：下游算子不能处理部分和，只能先归约。',
        '推迟到某个具体的算子之前 —— 中间的算子享受"未归约"的便宜。',
        '一部分算子未归约、一部分需要完整值 —— 分段处理。',
        '<b>最理想</b>：整条链都在未归约态，只在出口归约一次。',
        '函数结果也声明了 <span class="mono">unreduced</span> → 归约责任交给<b>调用者</b>。',
      ][i];
    }));
    tl.at(16400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>核心权衡</b>：未归约时每个设备只算<b>自己那一份部分和</b> —— 越晚归约，省下的重复计算越多。';
    });
  }
},

/* ------------------------------------------------ 3 ★ dot 的 reshard */
{
  kicker: 'L4-02 · reshard 插入总纲',
  title: '★ 最大的一族：<span class="mono hl-a">dot</span> 的 reshard',
  sub: '24 个用例覆盖 `dot` 的各种冲突类型 —— 因为它的维度关系最复杂（收缩 / 非收缩 / 批）。',
  caption: '规则：<b>算子自身要满足"对应维同分片"；冲突时选一侧，另一侧用 reshard 补齐。</b>',
  code: `// 【基础】结果分片与 lhs 冲突 -> 插入 reshard
func.func @reshard_dot_result_to_match_lhs(
    %arg0: tensor<4x32xf32> {...<@mesh, [{"y"}, {"x"}]>},   // lhs: dim0=y, dim1=x
    %arg1: tensor<32x8xf32> {...<@mesh, [{"x"}, {}]>})      // rhs: dim0=x
    -> tensor<4x8xf32> {
  %0 = stablehlo.dot %arg0, %arg1
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : ...
  return %0 : tensor<4x8xf32>
}
// 输出：
//   %[[DOT]] = stablehlo.dot %arg0, %arg1 {sharding_per_value=[<@mesh, [{"y"}, {}]>]}
//                                                            ^^^ 采用 lhs 那一侧
//   %[[RESHARD]] = sdy.reshard %[[DOT]] <@mesh, [{"x"}, {}]>
//                  ^^^^^^^^^^^^ 再搬到目标分片
//   return %[[RESHARD]]

// 逐项看：
//   收缩维：lhs 第1维 {"x"} vs rhs 第0维 {"x"} -> 一致 ✓
//   非收缩维：lhs 第0维 {"y"} vs 结果 {"x"}    -> 冲突 ✗
//   -> 让 dot 采用 lhs，再插 reshard 把结果搬到 [{"x"}, {}]

// 【冲突在收缩维上】用例 dot_lhs_and_rhs_conflicting_contracting_dim
//   lhs 收缩维 = {"x"}，rhs 收缩维 = {"y"} -> 冲突
//   输出【没有】插 reshard，dot_general 保持 [{}, {"y"}]
//   -> 并非所有冲突都靠 reshard 解决（后续课程展开）`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '结果 vs lhs/rhs', c: '#38bdf8', n: '2' }, { t: '多轴 / 多使用者', c: '#4ade80', n: '2' },
      { t: '批维', c: '#fbbf24', n: '1' }, { t: '多收缩/非收缩维', c: '#c084fc', n: '2' },
      { t: '缺分片 / 跨网格', c: '#f472b6', n: '2' }, { t: 'lhs vs rhs 冲突', c: '#fb7185', n: '4' },
      { t: '结果 vs 空分片', c: '#93c5fd', n: '1' }, { t: '多冲突', c: '#5eead4', n: '1' },
      { t: '结果"更大"', c: '#fdba74', n: '2' }, { t: '与归约轴冲突', c: '#a78bfa', n: '3' },
      { t: '与屏障', c: '#94a3b8', n: '1' }, { t: '其它', c: '#64748b', n: '3' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:104px;opacity:.4;transition:.3s;border-color:${x.c}55;padding:7px;text-align:center` });
      e.innerHTML = `<div style="font-size:10px;color:${x.c};line-height:1.3">${x.t}</div>
        <div class="mono" style="font-size:14px;color:${x.c};margin-top:2px">${x.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 90)); msg.innerHTML = '<b>24 个用例</b>按冲突类型分 12 组 —— 组合非常多。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>为什么最多</b>：dot 有收缩维 / 非收缩维 / 批维三类维度，每类都可能冲突。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>规则</b>：算子自身满足"对应维同分片"；冲突时选一侧，另一侧用 reshard 补齐。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>注意</b>：收缩维冲突的用例<b>没有</b>插 reshard —— 并非所有冲突都靠 reshard 解决。';
    });
  }
},

/* ------------------------------------------------ 4 其余三族 */
{
  kicker: 'L4-02 · reshard 插入总纲',
  title: '其余三族：<span class="hl-a">拼接</span>、<span class="hl-a">状态转换</span>、<span class="hl-a">函数边界</span>',
  sub: '`concatenate` 的拼接维、`sharded_to_unreduced` 的状态转换、以及函数调用边界的对齐。',
  caption: '第二、三族是 L1-02「三种状态正交」的实际运用；第四族与 L3-05 / L3-10 配套。',
  code: `// 【族 ①】concatenate：拼接维的分片必须一致
func.func @concatenate_different_shardings(
    %arg0: tensor<4x32x256xf32> {...<@mesh, [{"x"}, {}, {}]>},
    %arg1: tensor<4x48x256xf32> {...<@mesh, [{"y"}, {}, {}]>})
    -> tensor<4x80x256xf32> {
// 三个用例：分片不同 / 相同 / 结果分片与操作数不同
// 规则：拼接维上各操作数必须同分片，否则无法直接拼 -> 插 reshard

// 【族 ②】sharded_to_unreduced 等（17 个）
//   分片 -> 未归约：sharded_to_unreduced(_single_axis/_multiple_axes/_multiple_dims/_with_subaxis)
//   复制 -> 未归约：replicated_to_unreduced_result_without_reshard
//   混合：implicitly_and_explicitly_replicated_to_unreduced_{full,sub}_axis
//         replicated_and_sharded_to_unreduced_{full,sub}_axis
//   与 collective 组合：all_gather_and_... / all_slice_and_... / reshard_and_...
//   反向（未归约 -> 复制）：insert_reshard_unreduced_to_{partial,fully}_replicated
//                            （含 _max / _min 归约算子）
// 读法：unreduced 与"分片/复制"是【正交】的状态维度（L1-02）

// 【族 ③】insert_func_call_reshards（488 行 / 42 用例）
// RUN: -sdy-insert-func-call-reshards
// 处理【跨函数调用】的分片对齐：
//   调用点实参 vs 函数形参
//   函数 return vs 调用结果
// 为什么需要：L3-05 的 func_data_flow_edge 只做"桥接"，
//             两端要求不同时仍需在调用点插 reshard`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① concatenate', c: '#4ade80', n: '3 个',
        d: '拼接维的分片<br>必须一致<br><span class="dim">否则无法直接拼</span>' },
      { t: '② 状态转换', c: '#c084fc', n: '17 个',
        d: '分片 / 复制 / 未归约<br>之间的相互转换<br><span class="dim">L1-02 的三种状态</span>' },
      { t: '③ 函数边界', c: '#38bdf8', n: '42 个',
        d: '调用点实参 ↔ 形参<br>return ↔ 调用结果<br><span class="dim">另一个作用域</span>' },
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
        '最简单的规则：拼接维上各操作数必须同分片。',
        '<b>L1-02 讲过</b>：分片 / 复制 / 未归约是三个<b>正交</b>的维度 —— 这里处理它们的转换。',
        '<b>与 L3-05 / L3-10 配套</b>：桥接不等于对齐，两端要求不同时仍需 reshard。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>两个作用域</b>：<span class="mono">insert_explicit_reshards</span> 管算子内部，<span class="mono">insert_func_call_reshards</span> 管函数边界。';
    });
  }
},

/* ------------------------------------------------ 5 族谱 */
{
  kicker: 'L4-02 · reshard 插入总纲',
  title: '95 个用例的<span class="hl-a">族谱</span>',
  sub: '两个文件合计 95 个用例 —— 是 L4 层用例最多的课之一。',
  caption: '读法：抓住"延迟归约"与"对应维同分片"两条主线，其余都是组合变体。',
  code: `// 【insert_explicit_reshards.mlir】53 个
//   unreduced 处理        8    延迟归约的三种程度
//   dot 的 reshard       24    最大的一族（12 组冲突类型）
//   concatenate           3    拼接维一致性
//   状态转换             17    分片/复制/未归约互转
//   其它                  2    manual_computation / reduce 多结果

// 【insert_func_call_reshards.mlir】42 个
//   函数调用边界的对齐

// 合计 95 个用例

// 后续课程的分工（本课是总纲）：
//   L4-03 ~ L4-07  按【算子族】展开 reshard 插入
//                   元素级/形状类、dot、gather/scatter、卷积、规约...
//   L4-08 起       讲各类 collective 的具体形态

// 本课要建立的两个直觉：
//   ① 归约【越晚越好】（延迟归约）
//   ② 冲突时【选一侧，另一侧 reshard】`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: 'unreduced', n: 8, c: '#fbbf24' }, { t: 'dot', n: 24, c: '#38bdf8' },
      { t: 'concatenate', n: 3, c: '#4ade80' }, { t: '状态转换', n: 17, c: '#c084fc' },
      { t: '其它', n: 2, c: '#94a3b8' }, { t: '函数边界', n: 42, c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:130px;opacity:.4;transition:.35s;border-color:${f.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="card-t" style="color:${f.c};font-size:11px">${f.t}</div>
        <div class="big" style="font-size:20px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 120)); msg.innerHTML = '<b>95 个用例</b> —— 最大两族是 dot（24）与函数边界（42）。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>两条主线</b>：① 归约越晚越好；② 冲突时选一侧、另一侧 reshard。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>后续分工</b>：L4-03～07 按算子族展开，L4-08 起讲各类 collective。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L4-03 从元素级与形状类算子开始展开。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L4-02 · 练习',
  title: '练一练：<span class="hl-a">该不该插 reshard</span>',
  sub: '三道题分别考：中心思想、延迟归约、dot 的规则。',
  caption: '一句话总结：<b>传播定分片，reshard 对齐冲突；归约越晚越好</b>。',
  code: `// 题 1：为什么传播结束后还需要插入 reshard？

// 题 2：unreduced 轴上的 all_reduce 为什么尽量延迟？
//       什么情况下【完全不插】？

// 题 3：dot 的结果分片与 lhs 冲突时怎么处理？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col compact-ex', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '为什么传播结束后还需要插入 <span class="mono">reshard</span>？',
        a: '因为<b>传播是各方向独立推导的</b>，可能出现算子自己的约束不满足。' +
           '<br>以 <span class="mono">dot</span> 为例：它要求<b>收缩维在两操作数上同分片</b>、<b>非收缩维在操作数与结果间同分片</b> —— 而 lhs 与 rhs 可能被推出冲突的分片。' +
           '<br><span class="dim">一句话：传播决定"每个张量怎么切"，reshard 负责"把不兼容的地方对齐"。</span>'
      },
      {
        q: '<span class="mono">unreduced</span> 轴上的 <span class="mono">all_reduce</span> 为什么尽量延迟？什么情况下完全不插？',
        a: '<b>延迟的理由</b>：未归约时每个设备只算<b>自己那一份部分和</b>，计算量更小 —— 越晚归约，中间省下的重复计算越多。' +
           '<br><b>完全不插</b>：当<b>函数结果也声明了 <span class="mono">unreduced</span></b> 时（用例 <span class="mono">unreduced_func_input_until_return</span>）—— 归约责任交给调用者。' +
           '<br><span class="dim">5 个用例覆盖从"立刻"到"完全不插"的五种时机。</span>'
      },
      {
        q: '<span class="mono">dot</span> 的结果分片与 <span class="mono">lhs</span> 冲突时怎么处理？',
        a: '<b>让 dot 采用 lhs 那一侧，再插一条 <span class="mono">sdy.reshard</span> 把结果搬到目标分片。</b>' +
           '<br>例：lhs 第 0 维是 <span class="mono">{"y"}</span>、结果要求 <span class="mono">{"x"}</span> → dot 输出 <span class="mono">[{"y"}, {}]</span>，然后 <span class="mono">reshard</span> 到 <span class="mono">[{"x"}, {}]</span>。' +
           '<br><span class="dim">规则：算子自身要满足"对应维同分片"；冲突时选一侧，另一侧用 reshard 补齐。注意收缩维冲突的用例<b>没有</b>插 reshard —— 并非所有冲突都这样解决。</span>'
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
