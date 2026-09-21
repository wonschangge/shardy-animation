/* ==========================================================================
   L5-09 · pad-for-divisibility   （P0 · L5 收官）
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/pad_for_divisibility/ 下 12 个文件
         (1449 行 / 57 用例)
   目标：讲透不可整除分片如何在导出时被补齐。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 核心问题 */
{
  kicker: 'L5-09 · 为整除性补齐',
  title: '★ 核心问题：<span class="hl-a">除不尽怎么办</span>',
  sub: '分片要求"维度大小能被轴整除" —— 但现实中常有 `tensor<3>` 沿 `x=4` 切。',
  caption: '这是 L2-01「<b>不可整除不是错误</b>」的<b>最终答案</b>。',
  code: `// RUN: sdy_opt %s -sdy-pad-for-divisibility | FileCheck %s

// 【问题】
//   分片要求"维度大小能被轴整除"
//   但现实中常有 tensor<3> 沿 x=4 切、tensor<7> 沿 y=2 切 —— 除不尽
//
// 【L2-01 讲过】"不可整除【不是错误】"
//   但导出时【必须解决】—— 后端需要知道"每台设备拿多少"
//
// 【★ 两种方向】
//   输入不可整除：通信的【输入】大小除不尽
//     -> pad 补齐 -> 通信 -> slice 裁回
//   输出不可整除：通信的【输出】大小除不尽
//     -> slice 到【可整除】 -> 通信 -> slice 裁到目标
//
// 【统一模式】
//   让通信发生在【可整除的形状】上
//   用 pad / slice 在两端做适配
//
// 【12 个文件 / 1449 行 / 57 用例】
//   stablehlo_convolution     292 行 / 8 用例   <- 最大（permutation 因子）
//   stablehlo_while           201 行 / 4 用例
//   func_ops                  165 行 / 7 用例
//   stablehlo_reshape         145 行 / 6 用例
//   stablehlo_pad             142 行 / 8 用例
//   dot_general               130 行 / 4 用例
//   sdy_all_slice_all_gather  104 行 / 6 用例
//   all_to_all                 85 行 / 5 用例
//   reduce_scatter             72 行 / 4 用例
//   stablehlo_gather           68 行 / 2 用例
//   stablehlo_slice            31 行 / 2 用例
//   generic_ops                14 行 / 1 用例   <- 最小`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '问题', c: '#fb7185', d: '<span class="mono">tensor&lt;3&gt;</span> 沿 <span class="mono">x=4</span> 切<br><b>除不尽</b>' },
      { t: '输入不可整除', c: '#38bdf8', d: '<b>pad 补齐</b><br>→ 通信<br>→ slice 裁回' },
      { t: '输出不可整除', c: '#4ade80', d: 'slice 到<b>可整除</b><br>→ 通信<br>→ slice 裁到目标' },
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
        '<b>为什么必须解决</b>：后端需要知道"每台设备拿多少" —— 除不尽就<b>未定义</b>。',
        '输入端补几个元素，让大小能被整除 —— 补的部分之后会被裁掉。',
        '输出端反过来：先切到可整除的大小，通信后再裁到目标。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>统一模式</b>：让通信发生在<b>可整除的形状</b>上 —— 用 pad/slice 在两端适配。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 输出不可整除 */
{
  kicker: 'L5-09 · 为整除性补齐',
  title: '★ 输出不可整除：<span class="mono hl-a">slice</span> 到可整除',
  sub: '源 IR 想切 3，但 3 不能被 `x=4` 整除 → 导出时改成切 **4**，通信后再裁到 3。',
  caption: '注意 <span class="mono">all_gather</span> 的参数从 <span class="mono">tensor&lt;3x8&gt;</span> 变成了 <b><span class="mono">tensor&lt;4x8&gt;</span></b>。',
  code: `func.func @result_indivisible(
    %arg0: tensor<4x8xf32> {...<@mesh_4_2, [{"x"}, {}]>})
    -> tensor<3x8xf32> {
  // 源 IR 想切到 3 —— 但 3 不能被 x=4 整除！
  %0 = stablehlo.slice %arg0 [0:3, 0:8] {...} : (tensor<4x8xf32>) -> tensor<3x8xf32>
  %1 = sdy.all_gather [{"x"}, {}] %0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<3x8xf32>
  return %1 : tensor<3x8xf32>
}

// 【导出后】三步
// CHECK-LABEL: func @result_indivisible
// ① 改成切【可整除】的 4
// CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %arg0 [0:4, 0:8]
//               {...<@mesh_4_2, [{"x"}, {}]>}> : (tensor<4x8xf32>) -> tensor<4x8xf32>
//                                              ^^^^ 3 改成了 4
// ② 通信在 4x8 上做 —— 【可整除】✓
// CHECK-NEXT: %[[AG:.*]] = sdy.all_gather [{"x"}, {}] %[[SLICE]]
//               out_sharding=<@mesh_4_2, [{}, {}]> : tensor<4x8xf32>
//                                                   ^^^^^^^^^^^^ 参数变了！
// ③ 通信后裁到目标 3x8
// CHECK-NEXT: %[[TRIM:.*]] = stablehlo.slice %[[AG]] [0:3, 0:8]
//               {...<@mesh_4_2, [{}, {}]>}> : (tensor<4x8xf32>) -> tensor<3x8xf32>
// ④ 返回
// CHECK-NEXT: return %[[TRIM]] : tensor<3x8xf32>

// 【关键观察】
//   all_gather 的参数从 tensor<3x8xf32> 变成 tensor<4x8xf32>
//   -> 【通信在补齐后的形状上发生】
//
// 【为什么这样做】
//   x=4 要切第 0 维，而第 0 维大小是 4 -> 切完每台 1
//   如果只切 3 -> 3 不能被 4 整除 -> 无法确定"每台拿多少"
//   -> 先切到 4（可整除），通信后再裁到 3
//
// 【代价】多了一次 slice —— 但这是【本地操作】，无通信`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① slice 到 4', c: '#38bdf8', d: '从 <span class="mono">[0:3]</span><br>改成 <span class="mono">[0:4]</span><br><b>可整除</b>' },
      { t: '② all_gather', c: '#4ade80', d: '在 <span class="mono">4x8</span> 上做<br>参数<b>变了</b>' },
      { t: '③ slice 裁到 3', c: '#fbbf24', d: '裁到目标 <span class="mono">3x8</span><br><span class="dim">本地操作、无通信</span>' },
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
        '<b>关键改动</b>：源 IR 的 <span class="mono">[0:3]</span> 被改成 <span class="mono">[0:4]</span>。',
        '<b>通信的参数跟着变</b>：<span class="mono">tensor&lt;3x8&gt;</span> → <span class="mono">tensor&lt;4x8&gt;</span>。',
        '<b>多了一次 slice</b> —— 但它是本地操作，不产生通信。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>代价</b>：多一次本地 <span class="mono">slice</span>；<b>收益</b>：通信变成可定义的。';
    });
  }
},

/* ------------------------------------------------ 3 ★ 输入不可整除 */
{
   kicker: 'L5-09 · 为整除性补齐',
  title: '★ 输入不可整除：<span class="mono hl-a">pad</span> 补齐',
  sub: '`4x7` 的第 1 维 `7` 不能被 `y=2` 整除 → 补 **1** 个元素变成 `4x8`。',
  caption: '注意 <span class="mono">pad</span> 的参数：<span class="mono">low = [0, 0], high = [0, 1]</span> —— <b>只在尾部补</b>。',
  code: `func.func @input_indivisible(%arg0: tensor<4x7xi32> )
  -> (tensor<4x6xi32> {...<@mesh_4_2, [{}, {"y"}]>}) {
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]>
       : tensor<4x7xi32>
  %1 = stablehlo.slice %0 [0:4, 0:6] {...} : (tensor<4x7xi32>) -> tensor<4x6xi32>
  return %1 : tensor<4x6xi32>
}

// 【导出后】三步
// CHECK-LABEL: func @input_indivisible
// ① pad 补齐：7 -> 8（可被 y=2 整除）
// CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST]],
//               low = [0, 0], high = [0, 1], interior = [0, 0]
//               : (tensor<4x7xi32>, tensor<i32>) -> tensor<4x8xi32>
//                            ^^^^ 只在【尾部】补 1 个
// ② 通信在 4x8 上做 —— 【可整除】✓
// CHECK-NEXT: %[[ALL_SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]]
//               out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xi32>
// ③ 裁到目标 4x6（6 = 7 - 1，源 IR 本来就要切到 6）
// CHECK-NEXT: %[[RESULT:.*]] = stablehlo.slice %[[ALL_SLICE]] [0:4, 0:6]
//               {...} : (tensor<4x8xi32>) -> tensor<4x6xi32>
// CHECK-NEXT: return %[[RESULT]] : tensor<4x6xi32>

// 【pad 的三个参数】
//   low       前面补多少
//   high      后面补多少
//   interior  元素之间补多少（通常 0）
//
// 【补齐的方向】本用例只在 high 补
//   因为只需要"凑够"可整除的数量 —— 在尾部补最简单
//
// 【为什么填 0】
//   通信本身是【数据搬运】，不涉及算术 -> 填什么都不会影响"搬运"的正确性
//   搬运后的 slice 会把补的部分【裁掉】-> 补的值根本不会出现在最终结果里
//   -> 填 0 是最简单的选择
//
// 【但如果通信后紧跟归约】（如 reduce_scatter）
//   填的值就【必须不影响归约结果】：
//     加法 / 最大值：填 0 可能有问题（如果数据全为负）
//     乘法：填 0 会把结果变成 0 -> 必须填 1
//   本课的 12 个文件覆盖的正是这些情形`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① pad 补齐', c: '#38bdf8', d: '<span class="mono">7</span> → <span class="mono">8</span><br><span class="mono">high = [0, 1]</span><br><b>只在尾部补</b>' },
      { t: '② all_slice', c: '#4ade80', d: '在 <span class="mono">4x8</span> 上做<br><b>可整除</b>' },
      { t: '③ slice 裁回', c: '#fbbf24', d: '裁到 <span class="mono">4x6</span><br>补的部分被<b>裁掉</b>' },
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
        '<b>补多少</b>：补到<b>下一个能被轴整除的数</b> —— 7 补到 8。',
        '<b>通信的参数跟着变</b>：<span class="mono">tensor&lt;4x7&gt;</span> → <span class="mono">tensor&lt;4x8&gt;</span>。',
        '<b>补的值会被裁掉</b> —— 所以填 0 是安全的。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>但如果通信后紧跟归约</b>：填 0 可能有问题（乘法要填 1、max 要填最小值）。';
    });
  }
},

/* ------------------------------------------------ 4 padding 值 */
{
  kicker: 'L5-09 · 为整除性补齐',
  title: '<span class="mono hl-a">padding</span> 值的选择',
  sub: '12 个文件里 padding 值**都是零** —— 但这是一个**需要论证**的选择。',
  caption: '关键区分：通信后是<b>纯搬运</b>还是<b>跟了归约</b>。',
  code: `// 【统计 12 个文件的 padding 常量】
//   dense<0.000000e+00>（浮点零）   56 次
//   dense<0>（整数零）              14 次
//   -> padding 值【都是零】

// 【为什么可以填 0】（纯搬运的情形）
//   通信本身（all_gather / all_slice / all_to_all）是【数据搬运】
//   不涉及算术 -> 填什么值都不会影响"搬运"的正确性
//   搬运后的 slice 会把补的部分【裁掉】
//   -> 补的值【根本不会出现在最终结果里】
//   -> 填 0 是最简单的选择

// 【★ 但如果通信后紧跟归约】
//   填的值就【必须不影响归约结果】
//     sum   填 0  ✓（加法单位元）
//     max   填 0  ✗ 如果数据全为负 -> 0 会变成最大值！应填 -inf
//     min   填 0  ✗ 如果数据全为正 -> 0 会变成最小值！应填 +inf
//     mul   填 0  ✗ 会把结果变成 0 -> 必须填 1
//   本课的 12 个文件覆盖的正是这些情形
//   （reduce_scatter / dot_general / convolution）

// 【注意：dense<1> / dense<10> / dense<7> 不是 padding 值】
//   它们是 stablehlo_while 里的【循环边界常量】
//     %c1  = stablehlo.constant dense<1>   : tensor<i32>
//     %c10 = stablehlo.constant dense<10>  : tensor<i32>
//   -> 查 padding 值时要看 stablehlo.pad 的【第二个操作数】

// 【pad 的三个参数】
//   low       前面补多少
//   high      后面补多少
//   interior  元素之间补多少（通常 0）

// 【本课覆盖的 padding 参数样例】
//   low = [0, 0], high = [1, 0]                    15 次
//   low = [0, 0], high = [0, 1]                     7 次
//   low = [0, 0, 0, 0], high = [0, 0, 1, 0]         5 次
//   low = [0, 0, 0, 0], high = [0, 0, 1, 1]         4 次
//   low = [2, 0], high = [2, 0]                     2 次  <- 【两边都补】
//   low = [1, 0], high = [2, 0]                     2 次  <- 【不对称】
// 读法：补的位置与数量【取决于具体的不可整除情形】`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '纯搬运', c: '#4ade80', d: '填什么都会被裁掉<br>→ 填 <b>0</b>' },
      { t: 'sum', c: '#38bdf8', d: '填 <b>0</b> ✓<br>加法单位元' },
      { t: 'max', c: '#fbbf24', d: '填 0 <b>✗</b><br>数据全为负时出错<br>应填 <span class="mono">-inf</span>' },
      { t: 'mul', c: '#fb7185', d: '填 0 <b>✗</b><br>会把结果变成 0<br>必须填 <b>1</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:11px;color:${x.c}">${x.t}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>最常见的情形</b>：通信只是搬运，补的部分随后被裁掉 —— 填 0 无害。',
        '加法归约的单位元就是 0 —— 填 0 不影响和。',
        '<b>陷阱</b>：<span class="mono">max</span> 归约填 0，如果数据全为负，0 会变成"最大值"！',
        '<b>陷阱</b>：乘法填 0 会把整个结果变成 0 —— 必须填 <b>1</b>（乘法单位元）。',
      ][i];
    }));
    tl.at(15200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>核心原则</b>：padding 值必须是该运算的<b>单位元</b>（或不影响结果的值）。';
    });
  }
},

/* ------------------------------------------------ 5 族谱与呼应 */
{
  kicker: 'L5-09 · 为整除性补齐',
  title: '12 个文件的<span class="hl-a">族谱</span>与跨课呼应',
  sub: '从集合通信到结构算子 —— 不可整除**无处不在**。',
  caption: '本课是 L5 的<b>收官课</b>，也是 L2-01「不可整除不是错误」的最终答案。',
  code: `// 【族谱】12 个文件 / 1449 行 / 57 用例
//   族          文件                                        用例数
//   集合通信    all_to_all / reduce_scatter /
//               sdy_all_slice_all_gather                   15
//   计算类      dot_general / stablehlo_convolution        12
//   形状类      stablehlo_reshape / _slice / _pad          16
//   访存类      stablehlo_gather                            2
//   结构类      func_ops / stablehlo_while / generic_ops   12

// 【为什么 convolution 最大】（292 行 / 8 用例）
//   它的 permutation 因子（空间维不成整数倍，L5-06 讲过）
//   让不可整除的情形【最多】

// 【为什么 while 有 4 个用例】
//   循环体的分片要在【每次迭代】都保持可整除
//   需要把 padding 也【带进循环】（或每轮重新补）

// 【generic_ops 的 no_pad】
//   最简单的用例 —— 可整除时【什么都不做】
//   它验证了 pass 的【最小性】：能整除就不该有 pad

// 【★ 与前面几课的呼应】
//   L2-01  "不可整除不是错误"        <- 本课是它的【解决方案】
//   L2-02  子轴                       <- 不可整除的【另一种】处理
//   L4-02  update_non_divisible_...   <- 导出侧的【另一处理】
//   L5-04  pad 的语义                 <- 本课大量使用
//   L5-06  permutation 因子           <- 卷积不可整除的来源

// 【★ 注意与 L4-02 的区别】
//   L4-02（update_non_divisible_input_output_shardings）
//     把不可整除的【输入输出分片】用【子轴截断】（{"x":(1)2}）
//     -> 改变【分片】
//   L5-09（本课）
//     在【算子层面】插入 pad / slice
//     -> 改变【张量形状】
//   两者【互补】：L4-02 处理"分片怎么表达"，L5-09 处理"数据怎么补齐"

// 【★ 验收点答案】
//   给定 tensor<7x3x8> 沿 z=3 分片：
//     若沿【第 2 维】（大小 8）分片：8 不能被 3 整除
//       -> 补到下一个 3 的倍数 = 【9】
//       -> pad low = [0, 0, 0], high = [0, 0, 1]
//       -> 补齐后形状 【7x3x9】
//     通信后 slice 裁回 7x3x8

// 一句话总结：
//   不可整除时，让通信发生在【可整除的形状】上
//   输入端 pad 补齐、输出端 slice 到可整除，最后统一 slice 裁回目标`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '集合通信', n: 15, c: '#38bdf8' }, { t: '计算类', n: 12, c: '#4ade80' },
      { t: '形状类', n: 16, c: '#fbbf24' }, { t: '访存类', n: 2, c: '#c084fc' },
      { t: '结构类', n: 12, c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:140px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div style="font-size:10.5px;color:${f.c}">${f.t}</div>
        <div class="big" style="font-size:16px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>57 个用例</b>分五族 —— 不可整除<b>无处不在</b>。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>为什么 convolution 最大</b>：它的 <span class="mono">permutation</span> 因子让不可整除的情形最多。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>与 L4-02 互补</b>：L4-02 用<b>子轴</b>改分片；本课用 <b>pad/slice</b> 改形状。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<b>验收点</b>：<span class="mono">tensor&lt;7x3x8&gt;</span> 沿 <span class="mono">z=3</span> 切第 2 维 → 补到 <b>9</b> → <span class="mono">high = [0,0,1]</span>，形状 <b><span class="mono">7x3x9</span></b>。';
    });
  }
},

/* ------------------------------------------------ 6 小结 */
{
  kicker: 'L5-09 · 为整除性补齐',
  title: 'L5 <span class="hl-a">收官</span>：九课的主线',
  sub: 'L5 讲"全局张量怎么变成局部张量" —— 从总览到各类算子的降级。',
  caption: '完成后 L5 层 9 课全部结束，覆盖率超过 62%。',
  code: `// 【L5 的九课】
//   L5-01 全局转局部总览         102 行 /  7 用例
//   L5-02 集合通信降级           661 行 / 30 用例
//   L5-03 结构算子降级           314 行 / 14 用例
//   L5-04 形状类降级             300 行 / 18 用例
//   L5-05 矩阵乘降级（P0）       119 行 /  6 用例
//   L5-06 卷积降级               191 行 /  6 用例
//   L5-07 归约降级               148 行 /  5 用例
//   L5-08 访存类降级             956 行 / 22 用例
//   L5-09 为整除性补齐（本课）  1449 行 / 57 用例
//   ─────────────────────────────────────────────
//   合计 4240 行 / 165 用例

// 【一条主线】
//   全局张量 + 分片
//     -> 类型转换（每个维度按轴大小相除）
//     -> 算子改写（各类算子各自的规则）
//     -> 通信降级（sdy.* -> stablehlo.*）
//     -> 不可整除补齐（pad / slice）
//   = 每台设备实际持有的局部张量

// 【★ L5 的三条核心规律】
//   ① 只有【归约方向】上的分片需要通信
//      （L5-05 收缩维 / L5-06 归约因子 / L5-07 归约维 / L5-08 归约维）
//   ② 位置相关的算子要【补偿位置】
//      （L5-04 iota 偏移 / L5-08 索引重映射）
//   ③ 不可整除时让通信发生在【可整除的形状】上
//      （本课：pad 补齐 + slice 裁回）

// 【下一层 L6】执行与解释器 —— 分片代码怎么真正跑起来`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:8px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: 'L5-01 总览', n: 7, c: '#94a3b8' }, { t: 'L5-02 通信', n: 30, c: '#38bdf8' },
      { t: 'L5-03 结构', n: 14, c: '#0ea5e9' }, { t: 'L5-04 形状', n: 18, c: '#4ade80' },
      { t: 'L5-05 矩阵乘', n: 6, c: '#22c55e' }, { t: 'L5-06 卷积', n: 6, c: '#fbbf24' },
      { t: 'L5-07 归约', n: 5, c: '#f59e0b' }, { t: 'L5-08 访存', n: 22, c: '#c084fc' },
      { t: 'L5-09 补齐', n: 57, c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:104px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:7px;text-align:center` });
      e.innerHTML = `<div style="font-size:9px;color:${f.c};line-height:1.3">${f.t}</div>
        <div class="mono" style="font-size:14px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 90)); msg.innerHTML = '<b>165 个用例</b>覆盖 L5 九课 —— 本课最大（57 个）。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>★ 三条核心规律</b>：只有归约方向需要通信 / 位置相关要补偿 / 不可整除要补齐。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>L5 完成</b>：从"全局张量 + 分片"到"每台设备实际持有的局部张量"。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一层 L6</b>：执行与解释器 —— 分片代码怎么真正跑起来。';
    });
  }
},

];
