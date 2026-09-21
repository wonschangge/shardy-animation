/* ==========================================================================
   L4-14 · single-device-and-unreduced
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/ 下 3 个文件
         (resolve_single_device_sharding 186 行 / 5 用例
          verify_unreduced_axes 302 / 0（全 expected-error）
          remove_ag_rs_for_cmv1 130 / 10) = 618 行
   目标：讲透导出期三类"特殊状态"的处理。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 三类特殊情况 */
{
  kicker: 'L4-14 · 单设备与未归约',
  title: '三类<span class="hl-a">特殊情况</span>',
  sub: 'L4-03～13 处理"常规算子"；本课处理三类**边界情形**。',
  caption: '这三个 pass 是导出期的<b>收尾工作</b> —— 把所有特殊状态处理干净。',
  code: `// 【① 单设备分片】resolve_single_device_sharding (186 行 / 5 用例)
// RUN: sdy_opt %s -sdy-resolve-single-device-sharding
//   问题：单设备网格 @single_dev_0 = <[], device_ids=[0]>
//         表示"这个算子只在设备 0 上执行"
//         但整个程序要在【所有设备】上运行 -> 其他设备怎么办？
//   解法：用 stablehlo.if 按【设备号】守卫

// 【② 未归约轴校验】verify_unreduced_axes (302 行 / 全部 expected-error)
// RUN: sdy_opt %s -sdy-verify-unreduced-axes -split-input-file -verify-diagnostics
//   校验"未归约轴"的转换是否合法
//   核心概念：blessed operation（受祝福的操作）

// 【③ 后端特定移除】remove_ag_rs_for_cmv1 (130 行 / 10 用例)
// RUN: sdy_opt %s -sdy-remove-all-gather-reduce-scatter-for-cmv1
//   CMV1 后端能【自己处理】all_gather / reduce_scatter
//   -> 导出时要把这些算子移除

// 【共同点】都是导出期的【边界情形处理】
//   前面几课处理常规算子，本课处理三类特殊状态：
//     只在一台设备上执行（单设备分片）
//     部分结果（未归约轴）
//     后端能自己做的事（CMV1）`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 单设备分片', c: '#38bdf8', n: '5 用例',
        d: '只在<b>一台设备</b>上执行<br>其他设备怎么办？<br>→ 加 <span class="mono">if</span> 守卫' },
      { t: '② 未归约轴', c: '#fbbf24', n: '全 expected-error',
        d: '<b>部分结果</b>的转换<br>必须由 blessed<br>operation 显式完成' },
      { t: '③ 后端特定', c: '#4ade80', n: '10 用例',
        d: 'CMV1 能<b>自己处理</b><br><span class="mono">all_gather</span> / <span class="mono">reduce_scatter</span><br>→ 移除' },
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
        '<b>核心思想</b>：单设备分片 = "只有一台设备算，其他设备拿到占位值"。',
        '<b>为什么必须校验</b>：未归约轴代表"部分结果"—— 悄悄丢掉就改变了语义。',
        '<b>与 L4-10 同类</b>：都是消除冗余通信，但本课是<b>后端特定</b>的。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>本课是 L4 的收尾</b>：把前面几课没覆盖的特殊状态一次处理完。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 单设备 if 守卫 */
{
  kicker: 'L4-14 · 单设备与未归约',
  title: '★ 单设备分片 → <span class="mono hl-a">if</span> 守卫',
  sub: '把"只在一台设备上执行"翻译成 `stablehlo.if` + 设备号比较。',
  caption: '输入先 reshard 成<b>全复制</b> —— 因为所有设备都要执行这条 <span class="mono">if</span>，都需要数据。',
  code: `// 输入：%arg0 在 @mesh 上切了 "x"，但算子要在【单设备网格】上执行
func.func @custom_call_single_device_0(
    %arg0: tensor<8x16xf32> {...<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {...<@mesh, [{}, {}]>}, ...) {

// 输出（五步）：
// ① 输入 reshard 成【全复制】
// CHECK-NEXT: %[[IN_REPL]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
// ② 取【当前设备号】
// CHECK-NEXT: %[[PART_ID]] = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT: %[[PART_ID_I64]] = stablehlo.convert %[[PART_ID]] : (tensor<ui32>) -> tensor<i64>
// CHECK-NEXT: %[[C0]] = stablehlo.constant dense<0> : tensor<i64>
// ③ 判断"我是不是设备 0"
// CHECK-NEXT: %[[IS_DEV0]] = stablehlo.compare EQ, %[[PART_ID_I64]], %[[C0]] : ...
// ④ if：是 -> 执行；否 -> 返回【零值】
// CHECK-NEXT: %[[IF_RES]]:3 = "stablehlo.if"(%[[IS_DEV0]]) ({
// CHECK-NEXT:   %[[EXEC]]:3 = stablehlo.custom_call @SomeCustomCall(%[[IN_REPL]]) : ...
// CHECK-NEXT:   stablehlo.return %[[EXEC]]#0, %[[EXEC]]#1, %[[EXEC]]#2 : ...
// CHECK-NEXT: }, {
// CHECK-NEXT:   %[[ZEROS0]] = stablehlo.constant dense<0.000000e+00> : tensor<8x16xf32>
// CHECK-NEXT:   %[[ZEROS1]] = stablehlo.constant dense<0> : tensor<4x32xi32>
// CHECK-NEXT:   %[[TOKEN]] = stablehlo.create_token
// CHECK-NEXT:   stablehlo.return %[[ZEROS0]], %[[ZEROS1]], %[[TOKEN]] : ...

// 【为什么先全复制】
//   所有设备都要执行这条 if -> 它们都需要【输入数据】
//   即使不参与计算，也要能"走个过场"
// 【为什么 else 返回零值】
//   非设备 0 的设备不参与计算，但 if 的结果类型必须一致 -> 零值占位
// 【注意 token 类型】
//   第三个结果 !stablehlo.token 在 else 分支用 create_token 生成
//   token 是【副作用标记】，即使不执行也要造一个`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const steps = [
      { n: '①', t: 'reshard 全复制', c: '#38bdf8' },
      { n: '②', t: 'partition_id', c: '#4ade80' },
      { n: '③', t: 'compare EQ 0', c: '#fbbf24' },
      { n: '④', t: 'if 守卫', c: '#c084fc' },
      { n: '⑤', t: 'else 返回零值', c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = steps.map(s => {
      const e = U.el('div', { class: 'card', style: `width:140px;opacity:.33;transition:.3s;border-color:${s.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="small mono" style="font-size:11px;color:${s.c}">${s.n}</div>
        <div style="font-size:11px;margin-top:3px;line-height:1.35">${s.t}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    steps.forEach((s, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>为什么全复制</b>：所有设备都要执行这条 <span class="mono">if</span>，都需要输入数据。',
        '<span class="mono">stablehlo.partition_id</span> 取当前设备号（还需 <span class="mono">convert</span> 对齐类型）。',
        '判断"我是不是设备 0"。',
        '<b>是</b> → 真正执行算子；<b>否</b> → 什么都不做。',
        '<b>但类型必须一致</b> → 用零值占位；token 用 <span class="mono">create_token</span> 造一个。',
      ][i];
    }));
    tl.at(16400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>核心思想</b>：单设备分片 = "只有一台设备算，其他设备拿到占位值"。';
    });
  }
},

/* ------------------------------------------------ 3 ★ 未归约校验 */
{
  kicker: 'L4-14 · 单设备与未归约',
  title: '★ <span class="mono hl-a">blessed operation</span>：未归约轴不能悄悄丢',
  sub: '从"未归约"变成"已归约"必须由**特定算子显式声明** —— 普通算子不能顺手完成。',
  caption: '这个文件<b>全部是 <span class="mono">expected-error</span></b> —— 一个正面用例都没有（除了 blessed 的反例）。',
  code: `// RUN: sdy_opt %s -sdy-verify-unreduced-axes -split-input-file -verify-diagnostics

// 【规则 ①】不能悄悄丢掉未归约轴
func.func @dropped_by_add(
    %arg0: tensor<8x8xf32> {...[{"x"}, {}], unreduced={"y"}>})
    -> tensor<8x8xf32> {
  // expected-error@+1 {{'stablehlo.add' op dropped unreduced axis 'y' without
  //   a blessed operation (e.g., sdy.reshard). This is an invalid transition
  //   from unreduced to reduced.}}
  %0 = stablehlo.add %arg0, %arg0 {...[<@mesh, [{"x"}, {}]>]} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
// 关键词：without a blessed operation (e.g., sdy.reshard)

// 【规则 ②】blessed 的正确做法
func.func @dropped_by_add_blessed(
    %arg0: tensor<8x8xf32> {...[{"x"}, {}], unreduced={"y"}>})
    -> tensor<8x8xf32> {
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x"}, {}]> : tensor<8x8xf32>
  //   ^^^^^^^^^^^^^^^^^^^^^ 显式声明"这里要把 y 归约掉"
  %1 = stablehlo.add %0, %0 {...} : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// 【规则 ③】函数调用两端必须【匹配】(4 个用例)
//   expected-error: 'func.call' op has unreduced axes mismatch for 'y'
//                   at call argument 0.
//   覆盖：实参丢掉 / 结果丢掉 / 结果多出 / 实参缺少

// 【规则 ④】func.return 与 manual_computation 同样受约束
//   'func.return' op has unreduced axes mismatch for 'y' at return value 0
//     without a blessed operation (e.g., sdy.reshard).
//   'sdy.manual_computation' op dropped unreduced axis 'y' without a blessed
//     operation (e.g., sdy.reshard).
//   -> 规则是【统一】的

// 【规则 ⑤】归约算子（sum/max/min）不能变
//   'sdy.reshard' op cannot change the reduction operator of kept unreduced
//     axes from max to sum.
//   cannot introduce 'max' unreduced axes. Expected 'sum'.
//   为什么：unreduced=max{"y"} 与 unreduced=sum{"y"} 是【完全不同】的部分结果`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const rules = [
      { t: '① 不能悄悄丢', c: '#fb7185', d: '普通算子不能<br>完成这个转换' },
      { t: '② blessed 可以', c: '#4ade80', d: '<span class="mono">reshard</span> /<br><span class="mono">sharding_constraint</span>' },
      { t: '③ 调用两端匹配', c: '#38bdf8', d: '4 个用例<br>实参/结果都要对' },
      { t: '④ return/manual', c: '#fbbf24', d: '规则<b>统一</b><br>没有例外' },
      { t: '⑤ 归约算子不变', c: '#c084fc', d: '<span class="mono">sum</span>/<span class="mono">max</span>/<span class="mono">min</span><br>改了就是改语义' },
    ];
    const host = wrap.querySelector('#cards');
    const els = rules.map(x => {
      const e = U.el('div', { class: 'card', style: `width:140px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div style="font-size:10.5px;color:${x.c}">${x.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    rules.forEach((r, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>为什么危险</b>：未归约轴代表"<b>部分结果</b>"—— 悄悄丢掉意味着少了跨设备求和。',
        '<b>blessed operation</b> 的作用就是"<b>显式声明分片状态的变化</b>"。',
        '调用点与函数体是<b>同一个值的两种视角</b>（L3-05 / L3-10）—— 状态不一致就语义错乱。',
        '<span class="mono">func.return</span>、<span class="mono">manual_computation</span> <b>也</b>受约束 —— 规则没有例外。',
        '<span class="mono">max</span> 与 <span class="mono">sum</span> 的部分结果<b>完全不同</b> —— 改了就改变了程序含义。',
      ][i];
    }));
    tl.at(16400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>验收点答案</b>：<span class="mono">unreduced</span> 轴由归约类算子<b>引入</b>，必须由 <b>blessed operation</b> <b>显式消除</b>。';
    });
  }
},

/* ------------------------------------------------ 4 CMV1 移除 */
{
  kicker: 'L4-14 · 单设备与未归约',
  title: '<span class="mono hl-a">remove_ag_rs_for_cmv1</span>：后端能做的就移除',
  sub: 'CMV1 后端能**自己处理** `all_gather` / `reduce_scatter` —— 所以导出时把这些算子移除。',
  caption: '关键对比：<b>独立的</b> <span class="mono">all_gather</span> 保留，<b>可融合的</b>被移除。',
  code: `// RUN: sdy_opt %s -split-input-file -sdy-remove-all-gather-reduce-scatter-for-cmv1

// 【独立 all_gather】保留（它是最终结果，移除就错了）
func.func @single_all_gather(
    %arg0: tensor<8x16xf32> {...<@mesh, [{"x"}, {"y"}]>})
    -> (tensor<8x16xf32> {...<@mesh, [{"x"}, {}]>}) {
  %0 = sdy.all_gather [{}, {"y"}] %arg0 out_sharding=<@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
// CHECK: %0 = sdy.all_gather [{}, {"y"}] %arg0 out_sharding=<@mesh, [{"x"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %0
//   ^^^^ 保留！

// 【可融合的 all_gather】被移除
func.func @all_gather_dot(
    %arg0: tensor<8x16xf32> {...<@mesh, [{"x"}, {}]>},
    %arg1: tensor<16x32xf32> {...<@mesh, [{"x"}, {}]>})
    -> (tensor<8x32xf32> {...<@mesh, [{"x"}, {}]>}) {
  %0 = sdy.all_gather [{"x"}, {}] %arg1 out_sharding=<@mesh, [{}, {}]> : tensor<16x32xf32>
  %1 = stablehlo.dot %arg0, %0 {...} : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %1 : tensor<8x32xf32>
}
// CHECK: %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = ...<@mesh, [{"x"}, {}]>]>}
// CHECK-NEXT: return %0
//   ^^^^ all_gather 【消失了】！dot 直接读 %arg1

// 【为什么可以移除】
//   all_gather 的【唯一使用者】是 dot
//   CMV1 能在 dot 内部【自己完成】这个聚合
//   -> 显式的 all_gather 是冗余的
//
// 【与 L4-10 的对比】
//   L4-10：消除 all_to_all 前的 collective_permute（【通用】优化）
//   L4-14：消除 CMV1 能自己处理的 all_gather/reduce_scatter（【后端特定】）
//   共同点：都是"消除冗余的通信算子"`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '独立的 all_gather', c: '#4ade80', tag: '保留',
        d: '它是函数的<b>最终结果</b><br>移除就错了<br><span class="dim">没有使用者可以"吸收"它</span>' },
      { t: '可融合的 all_gather', c: '#fbbf24', tag: '移除',
        d: '唯一使用者是 <span class="mono">dot</span><br>CMV1 能在 <span class="mono">dot</span> 内部<br>自己完成聚合' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:11.5px">${x.t} <span class="faint small">${x.tag}</span></div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>判断依据</b>：这个通信算子<b>有没有被"吸收"的可能</b>。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>可以移除</b>：<span class="mono">all_gather</span> 的唯一使用者是 <span class="mono">dot</span>，而 CMV1 能在 <span class="mono">dot</span> 内部完成聚合。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>与 L4-10 的对比</b>：L4-10 是<b>通用</b>优化；本课是<b>后端特定</b>的适配。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>共同点</b>：都是"消除冗余的通信算子" —— 只是判断依据不同。';
    });
  }
},

/* ------------------------------------------------ 5 族谱与小结 */
{
  kicker: 'L4-14 · 单设备与未归约',
  title: '三个 pass 的<span class="hl-a">共同点</span>与小结',
  sub: '都是导出期的**边界情形处理** —— 把前面几课没覆盖的特殊状态一次处理完。',
  caption: '本课是 L4 的<b>收尾</b>：L4-15 之后进入导出流水线的最终阶段。',
  code: `// 【族谱】3 个文件 / 618 行
//   resolve_single_device_sharding  186 行 /  5 用例   单设备分片 -> if 守卫
//   verify_unreduced_axes           302 行 /  0 用例   未归约轴校验（全 expected-error）
//   remove_ag_rs_for_cmv1           130 行 / 10 用例   后端特定移除
//
//   合计 618 行 / 15 个 CHECK-LABEL 用例 + 若干 expected-error

// 【注意 verify_unreduced_axes 的特殊性】
//   它【没有】CHECK-LABEL 用例 —— 全部是 expected-error
//   这说明它是个【纯校验】pass：不改变 IR，只报错
//   这类 pass 的测试形态与"转换类"pass 完全不同

// 【三个 pass 的分工】
//   ① resolve_single_device_sharding  -> 加守卫（【改写】IR）
//   ② verify_unreduced_axes           -> 报错（【不改】IR）
//   ③ remove_ag_rs_for_cmv1           -> 移除（【改写】IR，后端特定）

// 【共同点】都是导出期的【边界情形处理】
//   前面几课处理"常规算子"，本课处理三类特殊情况：
//     只在一台设备上执行
//     部分结果（未归约轴）
//     后端能自己做的事

// 一句话总结：
//   导出期的最后阶段要把所有"特殊状态"处理干净
//   单设备分片加守卫、未归约轴校验、后端特定优化`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '单设备分片', n: 5, c: '#38bdf8' }, { t: '未归约校验', n: 0, c: '#fbbf24' },
      { t: 'CMV1 移除', n: 10, c: '#4ade80' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:150px;opacity:.4;transition:.35s;border-color:${f.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="card-t" style="color:${f.c};font-size:11.5px">${f.t}</div>
        <div class="big" style="font-size:20px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 130)); msg.innerHTML = '<b>15 个 CHECK-LABEL 用例</b> + 若干 <span class="mono">expected-error</span>。'; });
    tl.at(4200, () => {
      els.forEach((e, i) => { if (i !== 1) e.style.opacity = '.25'; });
      msg.innerHTML = '<b>注意中间那个 0</b>：<span class="mono">verify_unreduced_axes</span> 是<b>纯校验</b> pass —— 不改变 IR，只报错。';
    });
    tl.at(7600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>三类分工</b>：加守卫（改写）/ 报错（不改）/ 移除（改写，后端特定）。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L4-15 讲 <span class="mono">export-finalize</span> —— 导出流水线的最终阶段。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L4-14 · 练习',
  title: '练一练：<span class="hl-a">三类特殊状态</span>',
  sub: '三道题分别考：单设备守卫、未归约的来去、CMV1 移除。',
  caption: '一句话总结：<b>单设备加守卫、未归约要显式消除、后端能做的就移除</b>。',
  code: `// 题 1：单设备分片在导出期怎么处理？

// 题 2：unreduced 轴从哪来？必须在哪被消除？

// 题 3：为什么"独立的 all_gather"保留，而"后面跟着 dot 的"被移除？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col compact-ex', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '单设备分片在导出期怎么处理？',
        a: '翻译成 <b><span class="mono">stablehlo.if</span> + 设备号比较</b>：' +
           '<br>① 输入 <b>reshard 成全复制</b>（所有设备都要执行这条 if，都需要数据）；' +
           '<br>② <span class="mono">stablehlo.partition_id</span> 取当前设备号；' +
           '<br>③ <span class="mono">compare EQ</span> 判断"我是不是设备 0"；' +
           '<br>④ <span class="mono">if</span>：<b>是</b> → 执行算子；<b>否</b> → 返回<b>零值</b>占位（token 用 <span class="mono">create_token</span>）。' +
           '<br><span class="dim">核心思想：单设备分片 = "只有一台设备算，其他设备拿到占位值"。</span>'
      },
      {
        q: '<span class="mono">unreduced</span> 轴从哪来？必须在哪被消除？',
        a: '<b>从哪来</b>：<span class="mono">dot</span> / <span class="mono">reduce</span> 等算子沿<b>归约维切分</b>时<b>引入</b>（L4-04 的 <span class="mono">dot</span>、L4-05 的 <span class="mono">reduce</span>）。' +
           '<br><b>必须在哪消除</b>：由 <b>blessed operation</b>（<span class="mono">sdy.reshard</span> / <span class="mono">sdy.sharding_constraint</span>）<b>显式</b>消除。' +
           '<br><b>普通算子不能</b>悄悄完成这个转换 —— <span class="mono">add</span> / <span class="mono">call</span> / <span class="mono">return</span> / <span class="mono">manual_computation</span> 都会报错：<span class="mono">without a blessed operation (e.g., sdy.reshard)</span>。' +
           '<br><span class="dim">另外归约算子（<span class="mono">sum</span>/<span class="mono">max</span>/<span class="mono">min</span>）<b>不能变</b> —— 它们代表完全不同的部分结果。</span>'
      },
      {
        q: '为什么"独立的 <span class="mono">all_gather</span>"保留，而"后面跟着 <span class="mono">dot</span> 的"被移除？',
        a: '<b>独立</b>的 <span class="mono">all_gather</span> 是函数的<b>最终结果</b> —— 没有使用者可以"吸收"它，移除就错了。' +
           '<br><b>后面跟着 <span class="mono">dot</span></b> 的那个，唯一使用者是 <span class="mono">dot</span>，而 <b>CMV1 能在 <span class="mono">dot</span> 内部自己完成聚合</b> → 显式的 <span class="mono">all_gather</span> 冗余。' +
           '<br><span class="dim">与 L4-10 的对比：L4-10 消除 <span class="mono">all_to_all</span> 前的 <span class="mono">collective_permute</span>（<b>通用</b>优化）；本课消除 CMV1 能自己处理的通信（<b>后端特定</b>）。</span>'
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
