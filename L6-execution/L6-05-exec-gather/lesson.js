/* ==========================================================================
   L6-05 · exec-gather   （P0 · ★ 最复杂）
   --------------------------------------------------------------------------
   覆盖：executable_convert_global_to_local/ 下 6 个文件
         (two_reduction_dims 78, not_in_start_index_map 60, is_collapsed 59,
          _i32 59, _max 59, _min 59)
   目标：用数值验证 L5-08 的索引重映射；实测发现填充值 = 归约的单位元。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 六个文件 */
{
  kicker: 'L6-05 · gather 的执行',
  title: '★ 六个文件，验证 <span class="hl-a">L5-08 的索引重映射</span>',
  sub: 'L5-08 是全目录最复杂的机制 —— 本课看它跑出来的结果。',
  caption: '本课还<b>实测发现</b>了一件事，深化了 L5-08 与 L5-09 的表述。',
  code: `// 【六个文件】
//   文件                                        验证什么                    差异
//   ..._is_collapsed                            基准（sum 归约）            —
//   ..._is_collapsed_i32                        【索引类型无关】            i64 -> i32
//   ..._is_collapsed_min                        【min 归约】                填充值 0 -> +inf
//   ..._is_collapsed_max                        【max 归约】                填充值 0 -> -inf
//   ..._not_in_start_index_map                  collapsed 维不在 map        结果 2x2 -> 2x1
//   ..._two_reduction_dims                      【两个归约维】各切一个轴    网格 2x2

// 【网格】sdy.mesh @mesh_2 = <["x"=2]>

// 【gather 的规则】（基准文件的注释）
//   ([i, j], [k]) -> ([k, j]) reduction={i}
//   The sharded dim is a reduction dimensions and it is also a collapsed dim.

// 【★ 本课的三条结论】
//   ① L5-08 的索引重映射得到【数值验证】
//      mask / select 填充 / all_reduce 三步配合，结果与串行版一致
//   ② ★ 【填充值是归约的单位元，不是固定的 0】
//      sum -> 0、min -> +inf、max -> -inf
//      这深化了 L5-08 的表述，也印证了 L5-09 的"填充值必须是单位元"
//   ③ 索引类型（i32/i64）【不影响】降级逻辑

// 一句话：
//   L6-05 用数值验证了 L5-08 的索引重映射
//   并实测发现 mask 的填充值是【归约的单位元】`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'is_collapsed', c: '#94a3b8', d: '<b>基准</b><br>sum 归约' },
      { t: '_i32', c: '#38bdf8', d: '索引类型<br>i64→i32' },
      { t: '_min', c: '#4ade80', d: '<b>min</b><br>填充 <span class="mono">+∞</span>' },
      { t: '_max', c: '#fbbf24', d: '<b>max</b><br>填充 <span class="mono">−∞</span>' },
      { t: 'not_in_start_index_map', c: '#c084fc', d: 'collapsed 维<br>不在 map' },
      { t: 'two_reduction_dims', c: '#fb7185', d: '<b>两个归约维</b><br>网格 2x2' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:118px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:8.5px;color:${x.c};overflow-wrap:anywhere">${x.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>六个文件</b> —— 一个是基准，五个是变体。'; });
    tl.at(4400, () => {
      els.forEach((e, i) => { if (i !== 2 && i !== 3) e.style.opacity = '.25'; });
      msg.innerHTML = '<b>★ 最值得注意的是 min/max</b>：它们的填充值<b>不是 0</b>。';
    });
    tl.at(8000, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>规则</b>：<span class="mono">([i, j], [k]) -&gt; ([k, j]) reduction={i}</span> —— 分片的维既是<b>归约维</b>又是 <b>collapsed 维</b>。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>本课的核心</b>：用 <span class="mono">sdy_opt</span> 复现降级输出，看到填充值的<b>真实规律</b>。';
    });
  }
},

/* ------------------------------------------------ 2 基准场景 */
{
  kicker: 'L6-05 · gather 的执行',
  title: '基准场景：索引<span class="hl-a">跨两台设备</span>',
  sub: '索引 `[1, 3]` —— **1 属于设备 0、3 属于设备 1**，各自只有一个属于自己。',
  caption: '这正是 L5-08 讲的索引重映射场景。',
  code: `// ([i, j], [k]) -> ([k, j]) reduction={i}
// The sharded dim is a reduction dimensions and it is also a collapsed dim.
func.func @parallel_gather(
  %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg1: tensor<2xi64>) -> tensor<2x2xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2, [{}, {}], unreduced={"x"}>]>
  } : (tensor<4x2xf32>, tensor<2xi64>) -> tensor<2x2xf32>

  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2, [{}, {}]> : tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}

// 【验证方式】
func.func @main() {
  %input = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>
  %indices = stablehlo.constant dense<[1, 3]> : tensor<2xi64>

  %seq = func.call @sequential_gather(%input, %indices) : (tensor<4x2xf32>, tensor<2xi64>) -> tensor<2x2xf32>

  %shard0 = "stablehlo.slice"(%input) {
    start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>
  } : (tensor<4x2xf32>) -> tensor<2x2xf32>
  %shard1 = "stablehlo.slice"(%input) {
    start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>
  } : (tensor<4x2xf32>) -> tensor<2x2xf32>

  %pars:2 = "interpreter.run_parallel"(%shard0, %indices, %shard1, %indices) {
    programs = [[@parallel_gather, @parallel_gather]]
  } : (tensor<2x2xf32>, tensor<2xi64>, tensor<2x2xf32>, tensor<2xi64>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  "check.expect_eq"(%pars#0, %seq) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
  "check.expect_eq"(%pars#1, %seq) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
  return
}

// 【★ 逐设备读】
//   设备  持有的行      索引 1              索引 3
//   0     第 0~1 行     【属于我】-> [3,4]   不属于 -> 填填充值
//   1     第 2~3 行     不属于 -> 填填充值   【属于我】-> [7,8]
//
//   all_reduce 合并 -> 两台都得到 [[3,4],[7,8]] = %seq ✓
//
// 【★ 这验证了 L5-08 八步中的第 ⑤⑦⑧ 步】
//   ⑤ 生成 mask（判断索引是否属于我）
//   ⑦ select 填充（不属于我的位置填填充值）
//   ⑧ all_reduce 合并
//
// 【注意索引是【共享的】】
//   run_parallel(%shard0, %indices, %shard1, %indices)
//   两台设备拿【同一份索引】—— 但各自只能处理属于自己那部分`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '设备 0', c: '#38bdf8', d: '持有第 <b>0~1</b> 行<br>索引 <span class="mono">1</span> → <b>属于我</b><br>得 <span class="mono">[3,4]</span>' },
      { t: '设备 1', c: '#4ade80', d: '持有第 <b>2~3</b> 行<br>索引 <span class="mono">3</span> → <b>属于我</b><br>得 <span class="mono">[7,8]</span>' },
      { t: 'all_reduce', c: '#fbbf24', d: '合并<br>→ 两台都得到<br><span class="mono">[[3,4],[7,8]]</span>' },
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
        '<b>设备 0</b>：索引 <span class="mono">1</span> 在它的范围内 → gather 得 <span class="mono">[3,4]</span>；索引 <span class="mono">3</span> 不在 → <b>填填充值</b>。',
        '<b>设备 1</b>：对称 —— 索引 <span class="mono">3</span> 属于它，索引 <span class="mono">1</span> 填填充值。',
        '<b>all_reduce</b> 把两台的<b>部分结果</b>合并 → 两台都得到完整结果 ✓',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 这就是 L5-08 的第 ⑤⑦⑧ 步</b>：mask → select 填充 → all_reduce。';
    });
  }
},

/* ------------------------------------------------ 3 ★★ 实测发现 */
{
  kicker: 'L6-05 · gather 的执行',
  title: '★★ 实测发现：填充值是<span class="hl-a">归约的单位元</span>',
  sub: '**不是固定的 0** —— `sum`→`0`、`min`→`+∞`、`max`→`−∞`。',
  caption: '这是本课最有价值的发现，用 <span class="mono">sdy_opt</span> 复现降级输出才看到。',
  code: `// 【复现命令】（与 run_sdy_interpreter_test.sh 一致）
//   sdy_opt part1.mlir --sdy-convert-global-to-local \\
//     --sdy-inline-meshes --sdy-drop-sharding-and-mesh --allow-unregistered-dialect

// 【三个变体的输出只有两处不同】
=== 基准 vs min 的差异 ===
26c26
<     %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>
---
>     %cst = stablehlo.constant dense<0x7F800000> : tensor<2x2xf32>
30c30
<       %17 = stablehlo.add %arg2, %arg3 : tensor<f32>
---
>       %17 = stablehlo.minimum %arg2, %arg3 : tensor<f32>

=== 基准 vs max 的差异 ===
26c26
<     %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>
---
>     %cst = stablehlo.constant dense<0xFF800000> : tensor<2x2xf32>
30c30
<       %17 = stablehlo.add %arg2, %arg3 : tensor<f32>
---
>       %17 = stablehlo.maximum %arg2, %arg3 : tensor<f32>

// 【★ 汇总】
//   归约种类     select 的填充值              十六进制      含义
//   sum（基准）  dense<0.000000e+00>          0x00000000    【0】
//   min          dense<0x7F800000>            0x7F800000    【+inf】
//   max          dense<0xFF800000>            0xFF800000    【-inf】

// 【★ 填充值是该归约的"单位元"（identity element）】
//   sum -> 0      （加法单位元：0 + x = x）
//   min -> +inf   （min(+inf, x) = x）
//   max -> -inf   （max(-inf, x) = x）

// 【★ 这同时印证并深化了两课】
//   课        当时说的                        本课的实测
//   L5-08     gather 的 mask"【填 0】"       【只对 sum 成立】——那是基准文件的情形
//   L5-09     "填充值必须是该运算的单位元"   【完全证实】——gather 也遵守这条

// 【★ 为什么 min / max 必须单独测试】
//   如果用 sum 的 0 去配 min 归约，结果会【错】——
//   因为 min(0, 3) = 0 而不是 3（0 会变成"最小值"）

// 【★ 这是 AGENTS.md §3.3 那条红线的一次直接收获】
//   只看基准文件的 IR 形态，会以为"永远是填 0"
//   【跑三个变体才看到真实规律】
//   "凡测试文件里写着但没见过实际输出的，必须跑一遍确认"

// 【另注意】
//   %cst 的类型是 tensor<2x2xf32> —— 填充值是【张量常量】（形状同结果）
//   reducer 区域内的算子（add / minimum / maximum）也随种类变化`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'sum', c: '#94a3b8', hex: '0x00000000', v: '0', why: '加法单位元<br><span class="mono">0 + x = x</span>' },
      { t: 'min', c: '#4ade80', hex: '0x7F800000', v: '+∞', why: '<span class="mono">min(+∞, x) = x</span>' },
      { t: 'max', c: '#fbbf24', hex: '0xFF800000', v: '−∞', why: '<span class="mono">max(−∞, x) = x</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:246px;opacity:.33;transition:.35s;border-color:${x.c}55` });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:12px">${x.t} <span class="faint small">${x.hex}</span></div>
        <div class="big" style="font-size:19px;color:${x.c};margin:3px 0">${x.v}</div>
        <div class="card-d" style="font-size:10.5px">${x.why}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>基准（sum）</b>：填 <span class="mono">0</span> —— 这就是 L5-08 讲的"填 0"。',
        '<b>min 归约</b>：填 <b><span class="mono">+∞</span></b> —— 因为 <span class="mono">min(+∞, x) = x</span>。',
        '<b>max 归约</b>：填 <b><span class="mono">−∞</span></b> —— 因为 <span class="mono">max(−∞, x) = x</span>。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 规律</b>：填充值 = 该归约的<b>单位元</b>（identity element）。';
    });
    tl.at(16200, () => {
      msg.innerHTML = '<b>★ 为什么 min/max 必须单独测试</b>：用 <span class="mono">0</span> 配 <span class="mono">min</span> 会错 —— <span class="mono">min(0,3)=0</span> 而非 <span class="mono">3</span>。';
    });
  }
},

/* ------------------------------------------------ 4 两课闭环 */
{
  kicker: 'L6-05 · gather 的执行',
  title: '★ 一个<span class="hl-a">跨课闭环</span>：L5-08 → L5-09 → L6-05',
  sub: 'L5-08 说"填 0"，L5-09 提炼出通用原则，本课实测证实。',
  caption: '这正是"<b>跑一遍确认</b>"这条纪律的价值 —— 只看 IR 形态会得出不完整的结论。',
  code: `// 【三课的关系】
//   L5-08  讲 gather 的索引重映射八步
//          第 ⑦ 步说"【不属于我的填零】"
//          —— 那是基于【基准文件（sum 归约）】的观察
//   L5-09  讲不可整除补齐时提炼出通用原则
//          "padding 值必须是该运算的【单位元】"
//   L6-05  实测证明：gather 的填充值【确实遵守】那条通用原则
//          sum -> 0、min -> +inf、max -> -inf

// 【★ 所以 L5-08 的表述需要【深化】】
//   "填零"是 sum 归约的【特例】
//   通用表述应该是："填该归约的【单位元】"
//
//   这不是 L5-08 讲错了 —— 而是当时只看了基准文件
//   本课跑遍三个变体才看到完整规律
//   -> 这正是 AGENTS.md §3.3 那条红线的价值：
//      "凡测试文件里写着但没见过实际输出的，必须跑一遍确认"

// 【★ 一个可推广的读代码方法】
//   看到"多个变体文件"时，先【diff】它们
//     diff 会立刻告诉你"变化的维度是什么"
//   本课 diff 五个变体 -> 立刻看出：
//     变化的维度 = 归约种类 / 索引类型 / collapsed 维位置 / 归约维个数
//   然后【跑一遍】看变化的维度如何影响输出
//   -> 这比逐个读文件快得多

// 【本课 diff 的收获】
//   _i32   vs 基准：只差索引类型 -> 不影响降级逻辑
//   _min   vs 基准：差【填充值】与 reducer 算子
//   _max   vs 基准：同上
//   not_in_start_index_map vs 基准：差【结果形状】(2x2 -> 2x1)
//   two_reduction_dims 是独立文件（网格 2x2、两个归约维）`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'L5-08', c: '#38bdf8', d: '索引重映射八步<br>第 ⑦ 步说"<b>填零</b>"<br><span class="dim">基于 sum 基准文件</span>' },
      { t: 'L5-09', c: '#fbbf24', d: '提炼通用原则<br>"填充值必须是<br>该运算的<b>单位元</b>"' },
      { t: 'L6-05（本课）', c: '#4ade80', d: '<b>实测证实</b><br><span class="mono">sum→0 / min→+∞ / max→−∞</span><br>gather 也遵守那条原则' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>L5-08</b> 的"填零"是<b>基于基准文件</b>的观察 —— 那时只看了 sum 归约。',
        '<b>L5-09</b> 在讲补齐时提炼出通用原则：填充值必须是<b>单位元</b>。',
        '<b>本课实测</b>：gather 的填充值<b>确实遵守</b>那条原则 —— 三课闭环 ✓',
      ][i];
    }));
    tl.at(14800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 所以 L5-08 的表述需要深化</b>："填零"是 sum 的<b>特例</b>，通用表述是"填单位元"。';
    });
    tl.at(18000, () => {
      msg.innerHTML = '<b>一个可推广的方法</b>：看到多个变体文件时，先 <span class="mono">diff</span> —— 立刻看出"变化的维度是什么"。';
    });
  }
},

/* ------------------------------------------------ 5 其余变体 */
{
  kicker: 'L6-05 · gather 的执行',
  title: '其余三个变体：<span class="hl-a">索引类型 / 形状 / 多个归约维</span>',
  sub: '`_i32` 不影响逻辑；`not_in_start_index_map` 改变结果形状；`two_reduction_dims` 用两个轴。',
  caption: '这三个变体各自验证一个边界情形。',
  code: `// 【_i32：索引类型无关】
< func.func @sequential_gather(%arg0: tensor<4x2xf32>, %arg1: tensor<2xi64>) -> tensor<2x2xf32> {
> func.func @sequential_gather(%arg0: tensor<4x2xf32>, %arg1: tensor<2xi32>) -> tensor<2x2xf32> {
// 【读法】i64 -> i32 —— 索引类型【不影响】降级逻辑
//   测试目的是确认索引重映射对 i32 索引同样正确
//   （partition_id 的转换、查找表的下标运算都要适配）

// 【not_in_start_index_map：collapsed 维不在 start_index_map】
< func.func @sequential_gather(%arg0: tensor<4x2xf32>, %arg1: tensor<2xi64>) -> tensor<2x2xf32> {
> func.func @sequential_gather(%arg0: tensor<4x2xf32>, %arg1: tensor<2xi64>) -> tensor<2x1xf32> {
// 【读法】结果形状从 2x2 变成 2x1
//   基准的 start_index_map = [0] —— collapsed 维【在】start_index_map 里
//   本变体的 collapsed 维【不在】start_index_map 里
//     意味着这一维【被压掉但不参与索引】（slice_sizes 在该维为 1）
//   -> 【不需要重映射索引】（因为没有索引指向它）
// 【回顾 L2-10】gather 规则里的 blocked_propagation 涉及的正是 start_index_map
//   本变体验证了那个因子的一个边界情形

// 【two_reduction_dims：两个归约维】
// ([r1, b, r2], [b, i, v]) -> ([b, i]) reduction={r1, r2}
    %arg0: tensor<4x2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {}, {"y"}]>},
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_2, [{}, {}], unreduced={"x", "y"}>]>
  %1 = sdy.all_reduce {"x", "y"} %0 out_sharding=<@mesh_2_2, [{}, {}]> : tensor<2x1xf32>
// 【读法】
//   规则 ([r1, b, r2], [b, i, v]) -> ([b, i]) reduction={r1, r2}
//     -> 【两个归约维】r1、r2
//   %arg0 的【第 0 维（r1）切 {"x"}】、【第 2 维（r2）切 {"y"}】
//     -> 两个归约维【各切一个轴】
//   -> unreduced={"x", "y"} -> all_reduce {"x", "y"} ✓
// ★ 这验证了 L5-08 讲的"shard_two_of_three_reduction_dims"情形
//   多个归约维可以【分别用不同的轴】分片，all_reduce 一次归约所有轴`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '_i32', c: '#38bdf8', d: '索引 <span class="mono">i64</span>→<span class="mono">i32</span><br><b>不影响</b>降级逻辑' },
      { t: 'not_in_start_index_map', c: '#c084fc', d: 'collapsed 维不在 map<br>结果 <span class="mono">2x2</span>→<span class="mono">2x1</span><br>不需重映射索引' },
      { t: 'two_reduction_dims', c: '#fb7185', d: '<b>两个归约维</b><br>各切一个轴<br><span class="mono">unreduced={"x","y"}</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:10.5px;overflow-wrap:anywhere">${x.t}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>索引类型无关</b>：<span class="mono">i32</span> 与 <span class="mono">i64</span> 走同一套重映射逻辑。',
        '<b>collapsed 维不在 map</b>：没有索引指向它 → <b>不需要重映射</b> → 结果形状也不同。',
        '<b>两个归约维</b>：分别用 <span class="mono">x</span> 和 <span class="mono">y</span> 两个轴切 → 一次归约两个轴。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>三个变体</b>各自验证一个边界情形 —— 合起来覆盖了 gather 降级的主要分支。';
    });
  }
},

/* ------------------------------------------------ 6 小结 */
{
  kicker: 'L6-05 · gather 的执行',
  title: '小结：<span class="hl-a">最复杂一课</span>的三个收获',
  sub: '数值验证 + 实测发现 + 读代码方法。',
  caption: '一句话：<b>填充值是归约的单位元</b> —— 这修正了"填零"的简化说法。',
  code: `// 【族谱】6 个文件
//   文件                          验证什么                  与基准的差异
//   ..._is_collapsed              基准（sum 归约）          —
//   ..._is_collapsed_i32          索引类型无关              i64 -> i32
//   ..._is_collapsed_min          min 归约                  填充值 0 -> +inf
//   ..._is_collapsed_max          max 归约                  填充值 0 -> -inf
//   ..._not_in_start_index_map    collapsed 维不在 map      结果 2x2 -> 2x1
//   ..._two_reduction_dims        两个归约维各切一个轴      网格 2x2

// 【★ 三个收获】
//   ① 数值验证
//      L5-08 的索引重映射（mask / select 填充 / all_reduce）结果与串行版一致
//   ② ★ 实测发现
//      填充值【不是固定的 0】，而是【归约的单位元】：
//        sum -> 0、min -> +inf、max -> -inf
//      这深化了 L5-08 的表述，也印证了 L5-09 的通用原则
//   ③ 读代码方法
//      看到多个变体文件时，先 diff —— 立刻看出"变化的维度"
//      然后跑一遍看它如何影响输出

// 【L6 的进度】
//   L6-00 可执行测试机制（已做）    L6-01 sdy.* 集合通信（已做）
//   L6-02 stablehlo 集合通信（已做）L6-03 卷积（已做）
//   L6-04 矩阵乘/fft/iota（已做）   L6-05 gather（本课，★最复杂）
//   L6-06 pad（★最大族）            L6-07 reshape
//   L6-08 reverse/slice             L6-09 scatter 与杂项

// 一句话总结：
//   L6-05 用数值验证了 L5-08 的索引重映射
//   并实测发现 mask 的填充值是【归约的单位元】
//   这与 L5-09 提炼的"填充值必须是单位元"原则完全一致`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '基准 sum', n: 1, c: '#94a3b8' }, { t: 'i32', n: 1, c: '#38bdf8' },
      { t: 'min', n: 1, c: '#4ade80' }, { t: 'max', n: 1, c: '#fbbf24' },
      { t: 'not_in_map', n: 1, c: '#c084fc' }, { t: 'two_dims', n: 1, c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:118px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div style="font-size:9.5px;color:${f.c};overflow-wrap:anywhere">${f.t}</div>
        <div class="big" style="font-size:16px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 100)); msg.innerHTML = '<b>6 个文件</b>覆盖 gather 降级的主要分支。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>★ 三个收获</b>：数值验证 / 实测发现（填充值 = 单位元）/ 读代码方法（先 diff）。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>最重要的一条</b>：<b>填充值是归约的单位元</b> —— 这修正了"填零"的简化说法。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L6-06 讲 pad 的执行（★ 本层最大族）。';
    });
  }
},

];
