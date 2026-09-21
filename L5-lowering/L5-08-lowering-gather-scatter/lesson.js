/* ==========================================================================
   L5-08 · lowering-gather-scatter
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/convert_global_to_local/ 下 3 个文件
         (stablehlo_gather 463/11, stablehlo_scatter 416/9,
          stablehlo_select_and_scatter 77/2) = 956 行 / 22 用例
   目标：讲透访存类算子的降级 —— 本目录中最复杂的一族。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 核心问题 */
{
  kicker: 'L5-08 · 访存类降级',
  title: '★ 核心问题：<span class="hl-a">索引重映射</span>',
  sub: '`gather` / `scatter` 的索引是**全局坐标**，但每台设备只持有**局部数据**。',
  caption: '这是本目录中<b>最复杂</b>的一族（956 行 / 22 用例）。',
  code: `// 【三个文件】956 行 / 22 用例
//   stablehlo_gather               463 行 / 11 用例   <- 最大
//   stablehlo_scatter              416 行 /  9 用例
//   stablehlo_select_and_scatter    77 行 /  2 用例

// 【★ 为什么访存类最难】
//   gather / scatter 的索引是【全局坐标】
//   但分片后每台设备只持有【局部数据】
//   -> 索引必须【转换到本地坐标系】！

// 【与前面几课的对比】
//   L5-04 iota       序号起点不同    -> 本地 iota + 加偏移
//   L5-07 reduce     部分结果需合并  -> all_reduce / 收齐再算
//   L5-08 gather/scatter  索引指向别的设备 -> 【索引重映射 + mask】
//                                          ^^^^^^^^^^^^^^^^^^^^^^^ 最复杂

// 【为什么最复杂】
//   不仅要把索引【平移】到本地坐标系
//   还要处理"【这个索引根本不属于我】"的情况
//   -> 需要 mask + select 填零

// 【三个文件共同的机制】
//   clamp -> partition_id + 查表 -> 减偏移 -> mask
//         -> 操作 -> select 填零 -> all_reduce
//   这就是"索引重映射"的完整流程

// 一句话：
//   访存类算子的降级 = 索引重映射 + 部分结果合并`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'gather', c: '#38bdf8', n: '463 行', d: '读操作<br>归约维 collapsed' },
      { t: 'scatter', c: '#4ade80', n: '416 行', d: '写操作<br>被索引的维' },
      { t: 'select_and_scatter', c: '#fbbf24', n: '77 行', d: '<span class="mono">reduce_window</span><br>的反向操作' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:11px;overflow-wrap:anywhere">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>gather 的问题</b>：索引指向的行可能在<b>别的设备</b>上。',
        '<b>scatter 的问题</b>：要写的目标位置可能在<b>别的设备</b>上。',
        '<b>select_and_scatter</b> 与 <span class="mono">reduce_window</span> 同构 —— 判据都是"窗口是否跨设备"。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>统一机制</b>：<span class="mono">clamp</span> → 查表 → 减偏移 → <b>mask</b> → 操作 → 填零 → <span class="mono">all_reduce</span>。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 索引重映射八步 */
{
  kicker: 'L5-08 · 访存类降级',
  title: '★ 索引重映射的<span class="hl-a">八个步骤</span>',
  sub: '`gather` 的归约维被 `collapsed` 时，需要**十一行 IR** 来完成重映射。',
  caption: '这是全目录<b>最复杂</b>的一个用例 —— 值得逐步拆开看。',
  code: `// 规则：([i, j], [k]) -> ([k, j]) reduction={i}
func.func @shard_reduction_dim_is_collapsed(
  %arg0: tensor<8x10xf32> {...[{"x"}, {}]>},   // 第0维(归约维 i)切 x=2
  %arg1: tensor<2xi64>) -> tensor<2x10xf32> {  // 索引是【全局坐标】0~7
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 10>,
    sdy.sharding = ...<@mesh_2_4, [{}, {}], unreduced={"x"}>}
  : (tensor<8x10xf32>, tensor<2xi64>) -> tensor<2x10xf32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2_4, [{}, {}]> : tensor<2x10xf32>
  return %1 : tensor<2x10xf32>
}
// 【问题】%arg0 第0维切 x=2 -> 每台设备只有【4 行】（8/2）
//         %arg1 的值是【全局坐标】0~7
//         设备 1 只持有第 4~7 行 —— 它【无法处理索引 2】！

// 【八步解法】
// ① clamp 到全局范围 [0, 7]（越界索引先夹紧）
// CHECK-DAG: %[[C0]] = stablehlo.constant dense<0> : tensor<2xi64>
// CHECK-DAG: %[[C7]] = stablehlo.constant dense<7> : tensor<2xi64>
// CHECK: %[[CLAMPED]] = stablehlo.clamp %[[C0]], %[[ARG1]], %[[C7]] : tensor<2xi64>
// ② 查表得到本设备【偏移】
// CHECK: %[[PID]] = stablehlo.partition_id : tensor<ui32>
// CHECK: %[[CVT_PID]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
// CHECK: %[[TABLE]] = stablehlo.constant dense<[0, 0, 0, 0, 4, 4, 4, 4]> : tensor<8xi64>
// CHECK: %[[SLICE]] = stablehlo.dynamic_slice %[[TABLE]], %[[CVT_PID]], sizes = [1]
// CHECK: %[[RESHAPE]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi64>) -> tensor<i64>
// CHECK: %[[OFFSET]] = stablehlo.convert %[[RESHAPE]] : tensor<i64>
//   ^^^^ 设备 0~3 偏移 0、设备 4~7 偏移 4（每台 4 行）
// ③ 算出本地范围 [OFFSET, OFFSET+3]
// CHECK: %[[C3]] = stablehlo.constant dense<3> : tensor<i64>
// CHECK: %[[LIMIT]] = stablehlo.add %[[OFFSET]], %[[C3]] : tensor<i64>
// CHECK: %[[BCAST_OFF]] = stablehlo.broadcast_in_dim %[[OFFSET]], dims = []
// CHECK-SAME: {...<@mesh_2_4, [{}]>}> : (tensor<i64>) -> tensor<2xi64>
// CHECK: %[[BCAST_LIM]] = stablehlo.broadcast_in_dim %[[LIMIT]], dims = []
// CHECK-SAME: {...<@mesh_2_4, [{}]>}> : (tensor<i64>) -> tensor<2xi64>
//   ^^^^ OFFSET/LIMIT 是【标量】，索引是 tensor<2xi64> -> 需要广播
// ④ 【索引转换到本地坐标系】
// CHECK: %[[LOCAL_IDX]] = stablehlo.subtract %[[CLAMPED]], %[[BCAST_OFF]] : tensor<2xi64>
//   ^^^^ 设备 1 的索引 5 -> 本地 5 - 4 = 1 ✓
// ⑤ 生成 mask：判断"这个索引是否落在我的范围内"
// CHECK: %[[GE]] = stablehlo.compare GE, %[[CLAMPED]], %[[BCAST_OFF]]
// CHECK: %[[LE]] = stablehlo.compare LE, %[[CLAMPED]], %[[BCAST_LIM]]
// CHECK: %[[MASK]] = stablehlo.and %[[GE]], %[[LE]] : tensor<2xi1>
// ⑥ 用【本地索引】gather，结果标 unreduced
// CHECK: %[[GATHER]] = "stablehlo.gather"(%[[ARG0]], %[[LOCAL_IDX]])
// CHECK-SAME: {sdy.sharding = ...<@mesh_2_4, [{}, {}], unreduced={"x"}>]}
// ⑦ select 填零：不属于我的位置填 0
// CHECK: %[[MASK_BCAST]] = stablehlo.broadcast_in_dim %[[MASK]], dims = [0]
// CHECK: %[[ZERO]] = stablehlo.constant dense<0.000000e+00> : tensor<2x10xf32>
// CHECK: %[[SEL]] = stablehlo.select %[[MASK_BCAST]], %[[GATHER]], %[[ZERO]]
// ⑧ all_reduce 合并部分结果
// CHECK: %[[RES]] = "stablehlo.all_reduce"(%[[SEL]])
// CHECK-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<
//               mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "x">]>
// CHECK: return %[[RES]] : tensor<2x10xf32>

// 【为什么填零是安全的】
//   all_reduce 用【加法】—— 填 0 不影响和

// 【比 L5-04 的 iota 复杂在哪】
//   iota 只需【平移】（每台设备都有对应的序号）
//   gather 的索引【可能指向任何设备】-> 必须用 mask 排除`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:8px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const steps = [
      { n: '①', t: 'clamp', c: '#94a3b8' },
      { n: '②', t: '查表得偏移', c: '#38bdf8' },
      { n: '③', t: '广播', c: '#0ea5e9' },
      { n: '④', t: '减偏移', c: '#4ade80' },
      { n: '⑤', t: '生成 mask', c: '#fbbf24' },
      { n: '⑥', t: 'gather', c: '#f472b6' },
      { n: '⑦', t: 'select 填零', c: '#c084fc' },
      { n: '⑧', t: 'all_reduce', c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = steps.map(s => {
      const e = U.el('div', { class: 'card', style: `width:88px;opacity:.33;transition:.3s;border-color:${s.c}55;padding:7px;text-align:center` });
      e.innerHTML = `<div class="small mono" style="font-size:10px;color:${s.c}">${s.n}</div>
        <div style="font-size:9.5px;margin-top:2px;line-height:1.3">${s.t}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    steps.forEach((s, i) => tl.at(700 + i * 2600, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '索引先夹紧到<b>全局</b>范围 <span class="mono">[0, 7]</span>。',
        '查表 <span class="mono">[0,0,0,0,4,4,4,4]</span> 得到本设备的<b>偏移</b>。',
        '<span class="mono">OFFSET</span>/<span class="mono">LIMIT</span> 是<b>标量</b>，需要广播成张量。',
        '<b>关键</b>：<span class="mono">LOCAL_IDX = CLAMPED - OFFSET</span> —— 索引转换到<b>本地坐标系</b>。',
        '判断"这个索引是否<b>落在我的范围内</b>"。',
        '用<b>本地索引</b>做 gather，结果标 <span class="mono">unreduced</span>。',
        '<b>不属于我的填零</b> —— 因为 all_reduce 用加法，填 0 不影响和。',
        '合并各设备的部分结果。',
      ][i];
    }));
    tl.at(21800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>比 iota 复杂在哪</b>：iota 只需平移；gather 的索引<b>可能指向任何设备</b> → 必须用 mask 排除。';
    });
  }
},

/* ------------------------------------------------ 3 gather 的 11 个用例 */
{
  kicker: 'L5-08 · 访存类降级',
  title: '<span class="mono hl-a">gather</span> 的 11 个用例',
  sub: '关键区分：**归约维是否被 `collapsed`**。',
  caption: '回顾 <b>L2-10</b>：gather 的规则里有 <span class="mono">reduction={i, p}</span> 与 <span class="mono">blocked_propagation={k}</span> —— 本课看到它们的实际后果。',
  code: `// 【11 个用例】
//   operand_replicated                          操作数全复制
//   operand_sharded_pass_through_dim            分片在 pass-through 维（无通信）
//   shard_reduction_dim_is_collapsed            【归约维被 collapsed】（本课主例）
//   shard_reduction_dim_is_collapsed_i32        同上（i32 索引）
//   shard_reduction_dim_is_collapsed_not_in_start_index_map
//   shard_reduction_dim_is_collapsed_explicit_scalar_index_vector_dim
//   shard_reduction_dim_not_collapsed           【归约维未被 collapsed】
//   shard_reduction_dim_explicit_scalar_indices 标量索引
//   shard_two_of_three_reduction_dims           【三个归约维中切两个】
//   shard_two_of_three_reduction_dims_one_not_in_start_index_map
//   gather_unreduced                            结果未归约（延迟）

// 【关键区分】归约维是否被 collapsed
//   collapsed_slice_dims 里的维是"被压掉"的维（slice_sizes 里为 1）
//   它们对应 sharding_rule 的【reduction】因子
//   被 collapsed + 分片 -> 需要索引重映射 + all_reduce
//   未被 collapsed      -> 分片在输出维 -> 无通信

// 【基础用例的读法】operand_replicated
// CHECK-SAME: %[[ARG0]]: tensor<3x4x2x5xf32>,          <- 操作数全复制（无分片）
// CHECK-SAME: %[[ARG1]]: tensor<1x3x3xi64> {...[{"x"}, {}, {}]>})   <- 索引切 x
// CHECK-SAME: -> (tensor<1x3x2x2x1xf32> {...[{"x"}, {}, {}, {}, {}]>})
// CHECK: %[[GATHER]] = "stablehlo.gather"(%[[ARG0]], %[[ARG1]])
// CHECK-SAME: dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4],
//               collapsed_slice_dims = [0], start_index_map = [1, 0, 3],
//               index_vector_dim = 2>,
// 读法：操作数无分片 -> 索引怎么切都不影响 -> 无通信
//       索引的分片直接"传下去"到结果

// 【回顾 L2-10 的 gather 规则】
//   reduction={i, p}          <- 归约因子
//   blocked_propagation={k}   <- start_index_map 涉及的维不做传播
// 本课看到：归约因子分片 -> 索引重映射 + all_reduce
//           blocked_propagation 的维【不能分片】（否则无法传播）

// 【回顾 L3-11】那里讲"把隐式批维显式化"
//   因为隐式批维落在 blocked_propagation 上
//   本课的例子用的都是【已显式化】的 gather`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '全复制', c: '#94a3b8', d: '操作数无分片' },
      { t: 'pass-through 维', c: '#4ade80', d: '输出维分片<br><b>无通信</b>' },
      { t: '归约维 collapsed', c: '#fb7185', d: '<b>索引重映射</b><br>+ all_reduce' },
      { t: '归约维未 collapsed', c: '#38bdf8', d: '输出维分片<br><b>无通信</b>' },
      { t: '三个归约维切两个', c: '#fbbf24', d: '组合变体' },
      { t: '标量索引', c: '#c084fc', d: '边界情形' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div style="font-size:11px;color:${x.c}">${x.t}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>11 个用例</b> —— 关键区分是<b>归约维是否被 collapsed</b>。'; });
    tl.at(4400, () => {
      els.forEach((e, i) => { if (i !== 2 && i !== 3) e.style.opacity = '.25'; });
      msg.innerHTML = '<b>核心对比</b>：归约维被 collapsed + 分片 → 需要重映射；未被 collapsed → 只是输出维分片。';
    });
    tl.at(8000, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>回顾 L2-10</b>：<span class="mono">reduction={i, p}</span> 是归约因子，<span class="mono">blocked_propagation={k}</span> 禁止传播。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>回顾 L3-11</b>：隐式批维落在 <span class="mono">blocked_propagation</span> 上 —— 所以必须显式化。';
    });
  }
},

/* ------------------------------------------------ 4 scatter */
{
  kicker: 'L5-08 · 访存类降级',
  title: '<span class="mono hl-a">scatter</span>：被索引的维分片',
  sub: '与 `gather` **对称** —— 同样需要索引重映射，但 `scatter` 是**写操作**。',
  caption: '注意用例名里的 <span class="mono">inserted</span> / <span class="mono">non_inserted</span> 区分 —— 对应 <span class="mono">update_window_dims</span> 的不同情形。',
  code: `func.func @input_sharded_on_indexed_inserted__window_dim(
    %arg0: tensor<3x4x2xf32> {...[{}, {"x"}, {}]>},   // 第1维(被索引的维)切 x
    %arg1: tensor<2x3x2xi64>,                          // 索引
    %arg2: tensor<2x3x1xf32>)                          // 更新值
 -> (tensor<3x4x2xf32> {...[{}, {"x"}, {}]>}) {
  // CHECK: %[[C0]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK: %[[PID]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[CVT_PID]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK: %[[TABLE]] = stablehlo.constant dense<[0, 0, 0, 0, 2, 2, 2, 2]> : tensor<8xi64>
  // CHECK: %[[SLICE]] = stablehlo.dynamic_slice %[[TABLE]], %[[CVT_PID]], sizes = [1]
  //   ^^^^ 与 gather 完全相同的索引重映射开头
  //        表 [0,0,0,0,2,2,2,2]：设备 0~3 偏移 0、设备 4~7 偏移 2
  //        （tensor<3x4x2> 切第1维 -> 每台 4/2 = 2）
}

// 【与 gather 的对称性】
//               gather                    scatter
//   问题维      【归约维】(collapsed)     【被索引的维】
//   索引处理    平移 + mask               平移 + mask
//   通信        all_reduce                可能不需要

// 【为什么 scatter 可能不需要 all_reduce】
//   scatter 是【写操作】—— 多台设备写【不同的目标位置】时互不冲突
//   只有当它们可能写【同一位置】时才需要合并（用 update_computation）

// 【9 个用例】
//   input_not_sharded_scatter_indices_update_sharded_on_implicit_batch_dim
//   input_scalar_scatter_indices_update_sharded_on_implicit_batch_dim
//   input_sharded_not_on_indexed_dim                    分片【不在】被索引的维
//   input_sharded_on_indexed_inserted__window_dim       分片【在】被索引的维
//   input_sharded_on_indexed_but_non_inserted_window_dim 在被索引但【未插入】的维
//   shard_indexd_dim_scalar_scatter_indices             标量索引
//   scatter_replicated_bounds                           边界全复制
//   scatter_unreduced_axes_fallback_all_reduce          未归约 + all_reduce

// 【inserted vs non_inserted】
//   对应 update_window_dims 的不同情形 —— 决定"更新值怎么映射到窗口"
//   这是 scatter 特有的复杂度（gather 没有"更新值"）`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'gather', c: '#38bdf8', d: '问题维 = <b>归约维</b><br>(collapsed)<br>读操作' },
      { t: 'scatter', c: '#4ade80', d: '问题维 = <b>被索引的维</b><br>写操作<br><span class="dim">多一个"更新值"</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>对称的问题</b>：gather 是"读不到"，scatter 是"写不到"。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>索引重映射的开头完全相同</b> —— <span class="mono">partition_id</span> + 查表。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>scatter 多的复杂度</b>：它有<b>更新值</b>，所以要区分 <span class="mono">inserted</span> / <span class="mono">non_inserted</span>。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>为什么可能不需要 all_reduce</b>：多台设备写<b>不同位置</b>时互不冲突。';
    });
  }
},

/* ------------------------------------------------ 5 select_and_scatter 与小结 */
{
  kicker: 'L5-08 · 访存类降级',
  title: '<span class="mono hl-a">select_and_scatter</span> 与<span class="hl-a">统一规律</span>',
  sub: '它与 `reduce_window` **完全同构** —— 判据都是"窗口是否跨设备"。',
  caption: '本课把 L5-05～L5-08 的"归约方向"规律推广到了访存类算子。',
  code: `// 【select_and_scatter】2 个用例，与 L5-07 的 reduce_window 同构
//   select_and_scatter_batch_sharded                批维分片（窗口=1）  -> 无
//   select_and_scatter_stride_greater_than_window   步长 >= 窗口         -> 无

// 它是 reduce_window 的【反向操作】：
//   reduce_window        从窗口【归约出】输出
//   select_and_scatter   从输出【散射回】窗口

// 判据相同：【窗口是否跨设备边界】

// 【★ 三个文件的对照】
//   文件                    关键判据                      通信
//   gather                  【归约维】是否分片            unreduced + all_reduce
//   scatter                 【被索引的维】是否分片        索引重映射（可能无需通信）
//   select_and_scatter      【窗口是否跨设备】            与 reduce_window 同

// 【★ 统一规律】
//   归约方向上的分片需要通信；输出方向上的分片不需要
//   访存类算子的特殊之处是：【索引也要跟着重映射】

// 【索引重映射的完整流程】（三个文件共用）
//   clamp -> partition_id + 查表 -> 减偏移 -> mask
//         -> 操作 -> select 填零 -> all_reduce

// 【族谱】3 个文件 / 956 行 / 22 用例
//   stablehlo_gather               463 行 / 11 用例
//   stablehlo_scatter              416 行 /  9 用例
//   stablehlo_select_and_scatter    77 行 /  2 用例

// 【跨课呼应】
//   L2-10  gather 的规则（reduction / blocked_propagation）
//   L3-11  隐式批维显式化（为什么必须做）
//   L4-06  访存类的 reshard（导出侧）
//   L5-04  iota 的偏移补偿（索引重映射的简化版）
//   L5-07  reduce_window（select_and_scatter 的镜像）

// 一句话总结：
//   访存类算子的降级 = 索引重映射 + 部分结果合并
//   索引重映射是它独有的复杂度
//   因为索引是【全局坐标】，而每台设备只有【局部数据】`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: 'gather', n: 11, c: '#38bdf8' }, { t: 'scatter', n: 9, c: '#4ade80' },
      { t: 'select_and_scatter', n: 2, c: '#fbbf24' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:190px;opacity:.4;transition:.35s;border-color:${f.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="card-t mono" style="color:${f.c};font-size:10px;overflow-wrap:anywhere">${f.t}</div>
        <div class="big" style="font-size:20px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 130)); msg.innerHTML = '<b>22 个用例</b>分三个算子 —— <span class="mono">gather</span> 最大（463 行 / 11 个）。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>统一规律</b>：归约方向的分片需要通信；输出方向的<span style="color:#4ade80">不需要</span>。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>访存类的特殊之处</b>：<b>索引也要跟着重映射</b> —— 这是它独有的复杂度。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L5-09 讲 <span class="mono">pad-for-divisibility</span>（P0，12 个文件）。';
    });
  }
},

];
