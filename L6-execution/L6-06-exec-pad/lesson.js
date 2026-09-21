/* ==========================================================================
   L6-06 · exec-pad   （P0 · ★ 本层最大族）
   --------------------------------------------------------------------------
   覆盖：19 个文件
         executable_convert_global_to_local/ 3 个（分片语义）
         executable_partitioner_pipeline/   16 个（halo exchange 边界情形）
   目标：讲清 pad 为何是最大族 + uniform/non-uniform 判据。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 为何最大族 */
{
  kicker: 'L6-06 · pad 的执行',
  title: '★ 为什么 <span class="mono hl-a">pad</span> 是"本层最大族"',
  sub: '19 个文件 —— 答案在 16 个 partitioner 文件的**命名**里。',
  caption: '它们是 <b>halo exchange</b> 的边界情形族（回顾 <b>L4-09</b>）。',
  code: `// 【19 个文件】
//   族                              文件数   位置
//   分片语义（uniform/non-uniform）  3       executable_convert_global_to_local/
//   halo exchange 边界情形          16      executable_partitioner_pipeline/

// 【★ 答案在 16 个文件的命名里】
//   命名关键词      含义
//   hop             相邻设备间的数据交换【跳数】（halo exchange 的步数）
//   shift           数据移动【方向】（left / right）
//   large_pad       padding【跨越多个 hop】
//   multidim        【多维】同时 padding
//   replica_id      用 replica_id 而非 partition_id
//   replicated_*    【全复制】情形
//   negative_*_padding  【负 padding】（裁剪）

// 【★ 这些全是 halo exchange 的边界情形】（回顾 L4-09）
//   permutation 因子（空间维不成整数倍）有两种消解方式：
//     REPL（全复制）与 HALO（halo exchange）
//   HALO 需要"和邻居交换边界数据"，而【边界情形极多】——
//     跳数 x 方向 x 维度 x 正负 padding 的每一种组合都要验证

// 【大部分 partitioner 文件有两个 RUN 行】
//   // RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
//   // RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false
//   -> 验证 HALO 与 REPL 两种模式的结果【一致】（L4-09 的核心结论）

// 一句话：
//   pad 是本层最大族，因为它是【halo exchange 的实现载体】`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'hop', c: '#38bdf8', d: '<b>跳数</b><br>交换的步数' },
      { t: 'shift', c: '#4ade80', d: '<b>方向</b><br>left / right' },
      { t: 'large_pad', c: '#fbbf24', d: '<b>跨多个 hop</b>' },
      { t: 'multidim', c: '#c084fc', d: '<b>多维</b><br>同时 padding' },
      { t: 'replicated_*', c: '#fb7185', d: '<b>全复制</b><br>情形' },
      { t: 'negative_*', c: '#f472b6', d: '<b>负 padding</b><br>裁剪' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:118px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:9.5px;color:${x.c};overflow-wrap:anywhere">${x.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>六个命名关键词</b> —— 每一个都对应一类边界情形。'; });
    tl.at(4400, () => {
      msg.innerHTML = '<b>★ 全是 halo exchange 的边界情形</b>：跳数 × 方向 × 维度 × 正负 padding。';
    });
    tl.at(7800, () => {
      msg.innerHTML = '<b>为什么这么多</b>：HALO 要"和邻居交换边界数据"，边界情形<b>极多</b>。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<b>大部分文件有两个 RUN 行</b> —— 验证 <b>HALO 与 REPL 等价</b>（L4-09）。';
    });
  }
},

/* ------------------------------------------------ 2 未分片维 */
{
  kicker: 'L6-06 · pad 的执行',
  title: 'padding 在<span class="hl-a">未分片</span>的维上 → 无通信',
  sub: '每台设备的"填充位置"相同 → **local pad 参数不变**。',
  caption: '这是 L5-04 的"作用维判据"在 pad 上的体现。',
  code: `sdy.mesh @mesh_2 = <["x"=2]>

// Padding on replicated Dim 1, sharded on Dim 0.
func.func @pad_uniform(
  %arg0: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
  -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c0, low = [0, 1], high = [0, 1], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} : (tensor<4x2xi32>, tensor<i32>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

func.func @main() {
  %input = stablehlo.constant dense<[[1, 2], [3, 4], [5, 6], [7, 8]]> : tensor<4x2xi32>

  %s0 = "stablehlo.slice"(%input) {start_indices=array<i64: 0,0>, limit_indices=array<i64: 2,2>, strides=array<i64: 1,1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices=array<i64: 2,0>, limit_indices=array<i64: 4,2>, strides=array<i64: 1,1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>

  %res:2 = "interpreter.run_parallel"(%s0, %s1) {
    programs = [[@pad_uniform, @pad_uniform]]
  } : (tensor<2x2xi32>, tensor<2x2xi32>) -> (tensor<2x4xi32>, tensor<2x4xi32>)

  %e0 = stablehlo.constant dense<[[0, 1, 2, 0], [0, 3, 4, 0]]> : tensor<2x4xi32>
  "check.expect_eq"(%res#0, %e0) : (tensor<2x4xi32>, tensor<2x4xi32>) -> ()
  return
}

// 【读法】
//   注释：Padding on replicated Dim 1, sharded on Dim 0.
//   low = [0, 1], high = [0, 1] —— 【只在第 1 维（未分片维）padding】
//   每台设备本地 2x2 -> 2x4，local pad 参数【相同】-> 【无通信】✓
//
// 【期望值】设备 0 的 [[1,2],[3,4]] -> [[0,1,2,0],[0,3,4,0]]
//   第 1 维两侧各补一个 0 ✓
//
// 【★ 这正是 L5-04 的"作用维判据"】
//   padding 的维【未分片】
//     -> 每台设备的"填充位置"相同
//     -> 参数不变、无通信
//
// 【注意只断言了 %res#0】
//   因为两台设备的 local pad 相同，且断言的是【局部结果】（不是全局）`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '分片维', c: '#38bdf8', d: '第 <b>0</b> 维<br><span class="mono">[{"x"}, {}]</span><br><span class="mono">4 / 2 = 2</span>' },
      { t: 'padding 维', c: '#4ade80', d: '第 <b>1</b> 维<br><b>未被分片</b><br><span class="mono">low=[0,1] high=[0,1]</span>' },
      { t: '→ 结论', c: '#fbbf24', d: '每台 local pad<br><b>参数相同</b><br><b>无通信</b> ✓' },
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
        '<b>分片在第 0 维</b> —— 每台拿 2 行。',
        '<b>padding 在第 1 维</b> —— 而这一维<b>没被切</b>，所以每台看到的都是完整的 2 列。',
        '<b>不冲突</b> → 每台独立 padding → <b>无通信</b>。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 这就是 L5-04 的"作用维判据"</b>在 pad 上的体现。';
    });
  }
},

/* ------------------------------------------------ 3 ★ uniform */
{
  kicker: 'L6-06 · pad 的执行',
  title: '★ <span class="hl-a">uniform</span>：<span class="mono">pLow + pHigh = pInt</span>',
  sub: '在**分片维**上 padding，但每台设备的 local pad 参数**相同**。',
  caption: '这解释了 L5-09 讲的"uniform vs non-uniform" —— 那里是<b>推断</b>，这里是<b>实际语义</b>。',
  code: `sdy.mesh @mesh_2 = <["x"=2]>

// Padding on sharded Dim 0, with pLow + pHigh = pInt, which is uniform.
func.func @pad_sharded_uniform(
  %arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>})
  -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c0, low = [1], high = [0], interior = [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}]>]>} : (tensor<4xi32>, tensor<i32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

func.func @main() {
  %input = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %s0 = "stablehlo.slice"(%input) {start_indices=array<i64: 0>, limit_indices=array<i64: 2>, strides=array<i64: 1>} : (tensor<4xi32>) -> tensor<2xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices=array<i64: 2>, limit_indices=array<i64: 4>, strides=array<i64: 1>} : (tensor<4xi32>) -> tensor<2xi32>

  %res:2 = "interpreter.run_parallel"(%s0, %s1) {
    programs = [[@pad_sharded_uniform, @pad_sharded_uniform]]
  } : (tensor<2xi32>, tensor<2xi32>) -> (tensor<4xi32>, tensor<4xi32>)

  %e0 = stablehlo.constant dense<[0, 1, 0, 2]> : tensor<4xi32>
  %e1 = stablehlo.constant dense<[0, 3, 0, 4]> : tensor<4xi32>

  "check.expect_eq"(%res#0, %e0) : (tensor<4xi32>, tensor<4xi32>) -> ()
  "check.expect_eq"(%res#1, %e1) : (tensor<4xi32>, tensor<4xi32>) -> ()
  return
}

// 【读法】本课的核心概念
//   注释：with pLow + pHigh = pInt, which is uniform.
//   pLow + pHigh = 1 + 0 = 1 = pInt  -> 【均匀】✓
//   在【分片维】上 padding —— 但每台设备的 local pad 参数【相同】
//     （都是 low=1, interior=1）

// 【逐设备推结果】（输入 [1,2,3,4]，x=2 -> 每台 2 个元素）
//   设备   输入       局部结果          全局位置
//   0      [1, 2]     【[0, 1, 0, 2]】   第 0~3 位
//   1      [3, 4]     【[0, 3, 0, 4]】   第 4~7 位
//   全局拼接 = [0, 1, 0, 2, 0, 3, 0, 4] ✓
//     low=1 在最前面补 0、interior=1 在每两个元素间补 0

// 【★ 为什么"均匀"】
//   每台设备都在【自己那段的开头】补 low=1 个 0
//   并在【元素之间】补 interior=1 个 0 —— 【参数完全相同】
//   这是因为 pLow + pHigh = pInt：
//     边界的 padding 恰好等于 interior 的 padding
//     -> 段与段之间的边界看起来和【段内】一样

// 【★ 这解释了 L5-09 讲的 "uniform vs non-uniform"】
//   L5-09 我从 pad_for_divisibility 的用例名【推断】出这个区分
//   这里看到了【实际语义】`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '设备 0', c: '#38bdf8', d: '输入 <span class="mono">[1, 2]</span><br>局部 → <b><span class="mono">[0, 1, 0, 2]</span></b>' },
      { t: '设备 1', c: '#4ade80', d: '输入 <span class="mono">[3, 4]</span><br>局部 → <b><span class="mono">[0, 3, 0, 4]</span></b>' },
      { t: '全局拼接', c: '#fbbf24', d: '<span class="mono">[0,1,0,2,0,3,0,4]</span><br><b>两台的 pad 参数相同</b>' },
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
        '设备 0 在自己那段开头补 <span class="mono">low=1</span> 个 0，元素间补 <span class="mono">interior=1</span> 个 0。',
        '设备 1 <b>用完全相同的参数</b> —— 所以是"均匀"。',
        '<b>全局拼接</b>起来正好是完整的 padding 结果 ✓',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 为什么均匀</b>：<span class="mono">pLow + pHigh = pInt</span> —— 边界 padding 恰好等于 interior。';
    });
  }
},

/* ------------------------------------------------ 4 ★ non-uniform */
{
  kicker: 'L6-06 · pad 的执行',
  title: '★ <span class="hl-a">non-uniform</span>：<span class="mono">pLow + pHigh &gt; pInt</span>',
  sub: '首/尾设备的 padding 分布与中间设备**不同**。',
  caption: '测试故意用 <b>9</b> 作为填充值 —— 这样能区分"填充值"与"真实数据"。',
  code: `// Padding on sharded Dim 0, with pLow + pHigh > pInt, which is not uniform.
  %arg0: tensor<2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>})
  -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) {
  %0 = stablehlo.pad %arg0, %c0, low = [2], high = [2], interior = [2]

// part2 的期望值：
  %e0 = stablehlo.constant dense<[9, 9, 1, 9]> : tensor<4xi32>
  %e1 = stablehlo.constant dense<[9, 2, 9, 9]> : tensor<4xi32>
  "check.expect_eq"(%res#0, %e0) : (tensor<4xi32>, tensor<4xi32>) -> ()

// 【读法】
//   注释：with pLow + pHigh > pInt, which is not uniform.
//   pLow + pHigh = 2 + 2 = 4 > pInt = 2  -> 【非均匀】✓
//   输入 tensor<2xi32> 切 {"x"} -> 【每台 1 个元素】
//   结果 8xi32

// 【逐设备推结果】（假设输入 [1, 2]）
//   设备   输入    局部结果
//   0      [1]     【[9, 9, 1, 9]】
//   1      [2]     【[9, 2, 9, 9]】
//
// 【★ 两台设备的填充分布【不同】】
//   设备 0：9 在【前两个】位置
//   设备 1：9 在【第 2 和第 4】个位置
//   -> 这就是"非均匀"

// 【★ 填充值是 9 而非 0】
//   测试【故意】用 9 作为填充值
//   -> 这样能【区分】"填充值"与"真实数据"
//      （如果填 0，就可能与真实数据的 0 混淆）
//   这是 L6-03 讲的"缩放技巧"的【同源手法】

// 【★ uniform vs non-uniform 的判据】
//   情形            判据                        结果
//   uniform         pLow + pHigh = pInt         每台 local pad【相同】
//   non-uniform     pLow + pHigh > pInt         首/尾设备与中间设备【不同】`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'uniform', c: '#4ade80', d: '<span class="mono">pLow + pHigh = pInt</span><br>每台 local pad <b>相同</b>' },
      { t: 'non-uniform', c: '#fb7185', d: '<span class="mono">pLow + pHigh &gt; pInt</span><br>首/尾设备<b>不同</b><br><span class="mono">[9,9,1,9]</span> vs <span class="mono">[9,2,9,9]</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>uniform</b>：边界 padding 恰好等于 interior padding → 每台参数相同。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>non-uniform</b>：<span class="mono">2+2=4 &gt; 2</span> → 两台设备的 9 出现在<b>不同位置</b>。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 填充值是 9 而非 0</b>：故意用 9 以便<b>区分</b>填充值与真实数据（L6-03 的同源手法）。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>★ 这解释了 L5-09 的 uniform/non-uniform</b> —— 那里是<b>推断</b>，这里是<b>实际语义</b>。';
    });
  }
},

/* ------------------------------------------------ 5 ★ indivisible */
{
  kicker: 'L6-06 · pad 的执行',
  title: '★ <span class="mono hl-a">indivisible</span>：L5-09 的<span class="hl-a">完整实例</span>',
  sub: '注释写得最详细 —— 逐句对应 L5-09 的流程。',
  caption: '注意两个 RUN 行用的是<b>位置参数</b> —— 正好印证 L6-00 讲的脚本参数解析。',
  code: `// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"

// The pad input is sliced from 8x2 to 7x2 (indivisible, padded to 8x2).
// The original pad result size (10) is also indivisible by mesh axis size (4).
// PadForDivisibility should adjust high padding of the pad op to 3 (instead of 2)
// to make the result 12x2 (divisible). The result is then trimmed to 10x2
// after the final reshard (all_gather).

// 【读法】逐句对应 L5-09
//   ① 输入 8x2 -> 7x2 —— 【不可整除】（7 不能被 mesh 轴大小整除）
//      -> 补到 8x2
//   ② pad 结果 10 也【不能被 mesh 轴大小 4 整除】—— 又一次不可整除！
//   ③ PadForDivisibility 把 high 从 2 调到 3
//      -> 结果变成 12x2（12 = 4 x 3 可整除）✓
//   ④ 最后 all_gather 后裁回 10x2

// 【★ 这就是 L5-09 讲的"pad 补齐 + slice 裁回"的完整流程】
//   7x2（不可整除）
//     -> pad 到 8x2
//     -> 结果 10 又不可整除 -> high 从 2 调到 3
//     -> 12x2（可整除）
//     -> all_gather
//     -> 裁回 10x2 ✓

// 【★ 注意两个 RUN 行用的是【位置参数】】
//   // RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
//   // RUN: %S/run_sdy_interpreter_test.sh %s %t "false"
//
//   这正好印证了 L6-00 讲的脚本参数解析：
//     脚本支持【位置参数】—— 第 1 个是 src、第 2 个是 temp_dir、
//     【第 3 个是 ENABLE_HALO_EXCHANGE】
//   所以这里的 "true" / "false" 就是设置 ENABLE_HALO_EXCHANGE
//
//   （大部分 partitioner 文件用的是 --enable_halo_exchange=true 的【命名参数】形式
//     本文件用的是【位置参数】形式 —— 两种都支持，L6-00 讲过）`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const steps = [
      { n: '①', t: '7x2', c: '#fb7185', d: '不可整除' },
      { n: '②', t: 'pad 到 8x2', c: '#fbbf24', d: '补到可整除' },
      { n: '③', t: '结果 10 又不可整除', c: '#f59e0b', d: '<span class="mono">high</span> 2→3' },
      { n: '④', t: '12x2', c: '#4ade80', d: '<span class="mono">12 = 4×3</span>' },
      { n: '⑤', t: 'all_gather', c: '#38bdf8', d: '通信' },
      { n: '⑥', t: '裁回 10x2', c: '#c084fc', d: 'slice 裁回' },
    ];
    const host = wrap.querySelector('#cards');
    const els = steps.map(s => {
      const e = U.el('div', { class: 'card', style: `width:118px;opacity:.33;transition:.3s;border-color:${s.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div class="small mono" style="font-size:10px;color:${s.c}">${s.n}</div>
        <div class="mono" style="font-size:9.5px;margin-top:2px;overflow-wrap:anywhere">${s.t}</div>
        <div class="small faint" style="font-size:9px;line-height:1.3;margin-top:2px">${s.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    steps.forEach((s, i) => tl.at(700 + i * 2800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>起点</b>：输入被切成 <span class="mono">7x2</span> —— <b>不可整除</b>。',
        '先补到 <span class="mono">8x2</span>（可被轴大小整除）。',
        '<b>又一次不可整除</b>：pad 的结果 <span class="mono">10</span> 也不能被 <span class="mono">4</span> 整除。',
        '把 <span class="mono">high</span> 从 <span class="mono">2</span> 调到 <span class="mono">3</span> → 结果 <b><span class="mono">12x2</span></b>（<span class="mono">4×3</span>）。',
        '在可整除的形状上做 <span class="mono">all_gather</span>。',
        '最后 <span class="mono">slice</span> 裁回目标 <span class="mono">10x2</span> ✓',
      ][i];
    }));
    tl.at(17800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 两个 RUN 行用位置参数</b>（<span class="mono">"true"</span>/<span class="mono">"false"</span>）—— 印证 L6-00 讲的脚本参数解析。';
    });
  }
},

/* ------------------------------------------------ 6 小结 */
{
  kicker: 'L6-06 · pad 的执行',
  title: '小结：19 个文件的<span class="hl-a">族谱</span>与三条结论',
  sub: '`pad` 是本层最大族，因为它是 **halo exchange 的实现载体**。',
  caption: '一句话：<b>跳数 × 方向 × 维度 × 正负 padding</b> —— 每种组合都要验证。',
  code: `// 【族谱】19 个文件
//
// 【executable_convert_global_to_local/】3 个 —— 分片语义
//   文件                    判据                      通信
//   pad_non_sharded         padding 在【未分片】维    无
//   pad_sharded_uniform     pLow + pHigh = pInt       无（local 参数相同）
//   pad_sharded_non_uniform pLow + pHigh > pInt       首尾设备不同
//
// 【executable_partitioner_pipeline/】16 个 —— halo exchange 边界情形
//   族          文件
//   跳数        single_left_hop / single_right_hop /
//               multiple_right_hops / multiple_hops_right_shift
//   方向        left_shift / right_shift
//   大 padding  large_pad / large_pad_within_one_hop
//   多维        multidim_mixed_shifts / multidim_with_hops
//   全复制      replicated_dual_slice_pad /
//               replicated_negative_high_padding / replicated_negative_low_padding
//   其它        indivisible / interior / replica_id

// 【★ 三条结论】
//   ① pad 之所以是最大族，是因为【halo exchange 的边界情形极多】
//      跳数 x 方向 x 维度 x 正负 padding 的每种组合都要验证
//   ② uniform / non-uniform 的判据是 pLow + pHigh 与 pInt 的关系
//      相等则每台 local pad 相同，大于则首尾设备不同
//   ③ indivisible 是 L5-09 的【完整实例】
//      7x2 补到 8x2、结果 10 又不可整除故 high 从 2 调到 3、
//      最终 12x2 再裁回 10x2

// 【★ 与前面课的呼应】
//   L4-09  permutation 因子的 REPL / HALO 消解（本课 16 个文件验证 HALO）
//   L5-04  作用维判据（pad_non_sharded）
//   L5-09  uniform / non-uniform + 不可整除补齐（本课 3 个文件 + indivisible）
//   L6-00  脚本的位置参数解析（indivisible 的两个 RUN 行）
//   L6-03  填充值用非零值以便辨识（non_uniform 用 9）

// 【L6 的进度】
//   L6-00~05 已做（机制 / sdy 通信 / stablehlo 通信 / 卷积 /
//                   矩阵乘-fft-iota / gather）
//   L6-06 pad（本课，★最大族）
//   L6-07 reshape     L6-08 reverse/slice     L6-09 scatter 与杂项

// 一句话总结：
//   pad 是本层最大族，因为它是 halo exchange 的实现载体
//   16 个 partitioner 文件覆盖跳数/方向/维度/正负 padding 的各种边界情形
//   3 个 convert_global_to_local 文件讲清了 uniform / non-uniform 的判据`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '分片语义', n: 3, c: '#38bdf8' }, { t: '跳数', n: 4, c: '#4ade80' },
      { t: '方向', n: 2, c: '#22c55e' }, { t: '大 padding', n: 2, c: '#fbbf24' },
      { t: '多维', n: 2, c: '#f59e0b' }, { t: '全复制', n: 3, c: '#c084fc' },
      { t: '其它', n: 3, c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:104px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div style="font-size:9.5px;color:${f.c};overflow-wrap:anywhere">${f.t}</div>
        <div class="big" style="font-size:16px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 100)); msg.innerHTML = '<b>19 个文件</b>分七族 —— 16 个是 halo exchange 的边界情形。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>★ 三条结论</b>：最大族的原因 / uniform 判据 / indivisible 是 L5-09 的完整实例。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>与前面课呼应密集</b>：L4-09（REPL/HALO）、L5-04（作用维）、L5-09（补齐）、L6-00（位置参数）。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L6-07 讲 reshape 的执行（P0）。';
    });
  }
},

];
