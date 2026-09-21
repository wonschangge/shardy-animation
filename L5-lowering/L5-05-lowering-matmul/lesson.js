/* ==========================================================================
   L5-05 · lowering-matmul   （P0）
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/convert_global_to_local/ 下 2 个文件
         (stablehlo_dot 68/4, stablehlo_dot_general 51/2) = 119 行 / 6 用例
   目标：讲透矩阵乘分片降级 —— 经典并行策略的 IR 体现。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 六种情形 */
{
  kicker: 'L5-05 · 矩阵乘降级',
  title: '★ 六种情形 = <span class="hl-a">经典并行策略</span>的 IR 体现',
  sub: '`dot` 的 4 个用例 + `dot_general` 的 2 个用例，正好覆盖六种组合。',
  caption: '本课是 <b>L4-04</b> 的直接后续：那里讲 <span class="mono">dot</span> 的三类情形，这里看它们降级后的样子。',
  code: `// 【六种情形】2 个文件 / 119 行 / 6 用例
//   dot          fully_replicated                    无分片
//   dot          sharded_non_contracting_dims        【非收缩维】 -> 无通信
//   dot          sharded_contracting_dim             【收缩维】   -> all_reduce
//   dot          ..._unreduced_result                收缩维+结果未归约 -> 无
//   dot_general  not_shard_contracting_dims          【批维】     -> 无通信
//   dot_general  shard_contracting_dims              批维+收缩维  -> all_reduce

// 【对应关系】
//   非收缩维分片  -> 每个设备算输出的【一块】   -> 【模型并行】
//   收缩维分片    -> 每个设备算【部分和】       -> 【数据并行】
//   批维分片      -> 每个设备算【一批】         -> 【批并行】
//   两者都分片    -> 【2D 并行】

// 【★ 核心规律】
//   只有【收缩维】上的分片需要 all_reduce
//   非收缩维 / 批维上的分片都是"各算各的"，无需通信
//
//   为什么：收缩维是【归约的方向】
//           沿它分片 -> 每台设备只算了【部分和】
//           其他维分片 -> 每台设备算的是【不同的输出元素】，互不重叠

// 一句话：
//   矩阵乘的分片降级 = 经典并行策略的 IR 体现`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '全复制', c: '#94a3b8', d: '无通信' },
      { t: '非收缩维', c: '#4ade80', d: '<b>模型并行</b><br>无通信' },
      { t: '收缩维', c: '#fb7185', d: '<b>数据并行</b><br>all_reduce' },
      { t: '收缩维+未归约', c: '#fbbf24', d: '<b>延迟归约</b><br>无通信' },
      { t: '批维', c: '#38bdf8', d: '<b>批并行</b><br>无通信' },
      { t: '批维+收缩维', c: '#c084fc', d: '<b>2D 并行</b><br>all_reduce' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:118px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div style="font-size:10.5px;color:${x.c}">${x.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>六种情形</b> —— 只有两种需要 <span class="mono">all_reduce</span>。'; });
    tl.at(4400, () => {
      els.forEach((e, i) => { if (i !== 2 && i !== 5) e.style.opacity = '.25'; });
      msg.innerHTML = '<b>需要通信的只有这两种</b>：收缩维被分片的情形。';
    });
    tl.at(7800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 核心规律</b>：<b>只有收缩维上的分片需要 <span class="mono">all_reduce</span></b>。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<b>为什么</b>：收缩维是<b>归约的方向</b> —— 沿它分片，每台设备只算了<b>部分和</b>。';
    });
  }
},

/* ------------------------------------------------ 2 无通信的三种 */
{
  kicker: 'L5-05 · 矩阵乘降级',
  title: '无通信的三种：<span class="hl-a">各算各的</span>',
  sub: '全复制 / 非收缩维 / 批维 —— 这三种情况下每台设备算的都是**不同的输出元素**。',
  caption: '其中"非收缩维"就是<b>模型并行</b>。',
  code: `// 【① 全复制】直接转换，类型不变
func.func @fully_replicated(
  %arg0: tensor<8x16xf32> {...[{}, {}]>},
  %arg1: tensor<16x32xf32> {...[{}, {}]>}) -> tensor<8x32xf32> {
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}
// CHECK: %[[RES]] = stablehlo.dot %arg0, %arg1 : (tensor<8x16xf32>, ...) -> tensor<8x32xf32>
//   ^^^^ 类型不变、无通信

// 【② 非收缩维】= 模型并行
func.func @sharded_non_contracting_dims(
  %arg0: tensor<8x16xf32> {...[{"x"}, {}]>},
  %arg1: tensor<16x32xf32> {...[{}, {"y"}]>})
  -> (tensor<8x32xf32> {...[{"x"}, {"y"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {...[{"x"}, {"y"}]...} : ...
  return %0 : tensor<8x32xf32>
}
// CHECK-SAME: (%[[ARG0]]: tensor<4x16xf32> {...}, %[[ARG1]]: tensor<16x8xf32> {...})
// CHECK-SAME: -> (tensor<4x8xf32> {...})
// CHECK: %[[RES]] = stablehlo.dot %[[ARG0]], %[[ARG1]]
// CHECK-SAME: (tensor<4x16xf32>, tensor<16x8xf32>) -> tensor<4x8xf32>
// 【逐项】
//   张量     全局    分片              局部    算式
//   %arg0    8x16    [{"x"}, {}]       4x16    8 / 2
//   %arg1    16x32   [{}, {"y"}]       16x8    32 / 4
//   结果     8x32    [{"x"}, {"y"}]    4x8     8/2, 32/4
// 【关键】收缩维（arg0 第1维、arg1 第0维）都是【完整的 16】
//   每台设备算输出矩阵的【一块】（4x8）-> 不需要通信 ✓
// 这就是【模型并行】：把权重/输出切开，每台算一部分

// 【⑤ 批维】= 批并行（dot_general）
func.func @not_shard_contracting_dims(
  %arg0: tensor<4x16x8xf32> {...[{"x"}, {}, {}]>},
  %arg1: tensor<4x8x16xf32> {...[{"x"}, {}, {}]>})
  -> (tensor<4x16x16xf32> {...[{"x"}, {}, {}]>}) {
  %0 = stablehlo.dot_general %arg0, %arg1,
       batching_dims = [0] x [0], contracting_dims = [2] x [1] {...} : ...
  return %0 : tensor<4x16x16xf32>
}
// CHECK-SAME: %[[ARG0]]: tensor<2x16x8xf32> {...}, %[[ARG1]]: tensor<2x8x16xf32> {...}
// CHECK-SAME: -> (tensor<2x16x16xf32> {...})
// CHECK: %[[RES]] = stablehlo.dot_general %[[ARG0]], %[[ARG1]],
//         batching_dims = [0] x [0], contracting_dims = [2] x [1]
// 【关键】分片 [{"x"}, {}, {}] 正好在【批维】上 -> 每台设备算【一批】
//   收缩维 [2] x [1] 都是完整的（8）-> 无通信 ✓
//   局部类型：4x16x8 -> 2x16x8（批维 ÷2）`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 全复制', c: '#94a3b8', d: '类型不变<br>无通信' },
      { t: '② 非收缩维', c: '#4ade80', d: '<b>模型并行</b><br>每台算一块 4x8' },
      { t: '⑤ 批维', c: '#38bdf8', d: '<b>批并行</b><br>每台算一批' },
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
        '最简单：没有分片就没有局部化问题。',
        '<b>关键</b>：收缩维是<b>完整的 16</b> → 每台设备的 dot 结果就是<b>最终</b>那一块。',
        '<b>批维分片</b>：各批独立计算 —— 这是最自然的并行方式。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>共同点</b>：每台设备算的都是<b>不同的输出元素</b> —— 互不重叠，所以无需通信。';
    });
  }
},

/* ------------------------------------------------ 3 ★ 收缩维 = 数据并行 */
{
   kicker: 'L5-05 · 矩阵乘降级',
  title: '★ 收缩维分片 = <span class="hl-a">数据并行</span>',
  sub: '每台设备算的只是**部分和** → 结果标 `unreduced` → 需要 `all_reduce`。',
  caption: '这正是 <b>L4-04</b> 讲的"<span class="mono">dot</span> 情形 ②"降级后的样子。',
  code: `func.func @sharded_contracting_dim(
  %arg0: tensor<8x16xf32> {...[{}, {"x"}]>},     // 收缩维切 x
  %arg1: tensor<16x32xf32> {...[{"x"}, {}]>})    // 收缩维切 x（一致 ✓）
  -> tensor<8x32xf32> {
  %0 = stablehlo.dot %arg0, %arg1
       {...[{}, {}], unreduced={"x"}>} : ...
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2_4, [{}, {}]> : tensor<8x32xf32>
  return %1 : tensor<8x32xf32>
}
// CHECK-SAME: (%[[ARG0]]: tensor<8x8xf32> {...[{}, {"x"}]>},
// CHECK-SAME:  %[[ARG1]]: tensor<8x32xf32> {...[{"x"}, {}]>})
// CHECK: %[[DOT]] = stablehlo.dot %[[ARG0]], %[[ARG1]]
// CHECK-SAME: {sdy.sharding = ...<@mesh_2_4, [{}, {}], unreduced={"x"}>]}
// CHECK-SAME: (tensor<8x8xf32>, tensor<8x32xf32>) -> tensor<8x32xf32>
// CHECK: %[[RES]] = "stablehlo.all_reduce"(%[[DOT]])
// CHECK-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<
//               mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "x">]>
// CHECK: return %[[RES]] : tensor<8x32xf32>

// 【逐项】
//   张量       全局    分片                     局部    算式
//   %arg0      8x16    [{}, {"x"}]              8x8     16 / 2
//   %arg1      16x32   [{"x"}, {}]              8x32    16 / 2
//   dot 结果   8x32    [{}, {}], unreduced={"x"} 8x32   全复制

// 【关键】
//   收缩维（arg0 第1维、arg1 第0维）【都被切了 x】—— 且一致 ✓
//   每台设备算的只是【部分和】-> 结果标 unreduced={"x"}
//   -> 需要 all_reduce {"x"} 合并
// 这就是【数据并行】：把数据切开，每台算一部分，最后归约

// 【完整链路】
//   dot（收缩维分片）
//     --L4-04-->  unreduced + sdy.all_reduce
//     --L5-05-->  stablehlo.dot(unreduced) + stablehlo.all_reduce
//   三课连起来才是完整故事

// 【replica_groups】用 x 轴 —— 与 L5-02 的 V3 表示一致`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '为什么产生部分和', c: '#fbbf24', d: '收缩维被切 → 每台设备<br>只累加了<b>一部分</b>乘积' },
      { t: '为什么必须归约', c: '#fb7185', d: '结果是<b>部分和</b><br>必须跨设备相加<br>才能得到完整值' },
      { t: '对应策略', c: '#4ade80', d: '<b>数据并行</b><br>切开数据、各算一部分<br>最后归约' },
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
        '<b>收缩维的语义</b>：沿它做累加。切了之后每台设备只累加自己那段。',
        '<span class="mono">unreduced={"x"}</span> 就是"这个结果在 <span class="mono">x</span> 上还没归约"的标记。',
        '<b>经典数据并行</b>：切数据、各算各的、最后 all_reduce 合并梯度/部分和。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>完整链路</b>：<span class="mono">dot</span> --L4-04--&gt; <span class="mono">unreduced</span>+<span class="mono">all_reduce</span> --L5-05--&gt; <span class="mono">stablehlo</span>。';
    });
  }
},

/* ------------------------------------------------ 4 ★ 延迟归约 */
{
  kicker: 'L5-05 · 矩阵乘降级',
  title: '★ 情形 ④：<span class="hl-a">延迟归约</span>',
  sub: '与情形 ③ 的唯一区别：**函数结果也声明了 `unreduced`** → 不插 `all_reduce`。',
  caption: '这就是 <b>L4-05</b> 讲的"延迟归约"在矩阵乘上的体现。',
  code: `func.func @sharded_contracting_dim_unreduced_result(
  %arg0: tensor<8x16xf32> {...[{}, {"x"}]>},
  %arg1: tensor<16x32xf32> {...[{"x"}, {}]>})
  -> (tensor<8x32xf32> {...[{}, {}], unreduced={"x"}>}) {   // <- 结果【也】未归约
  %0 = stablehlo.dot %arg0, %arg1
       {...[{}, {}], unreduced={"x"}>} : ...
  return %0 : tensor<8x32xf32>          // <- 直接返回，【没有】all_reduce
}
// CHECK-SAME: -> (tensor<8x32xf32> {...[{}, {}], unreduced={"x"}>}) {
// CHECK: %[[DOT]] = stablehlo.dot %[[ARG0]], %[[ARG1]]
// CHECK-SAME: {sdy.sharding = ...<@mesh_2_4, [{}, {}], unreduced={"x"}>]}
// CHECK-SAME: (tensor<8x8xf32>, tensor<8x32xf32>) -> tensor<8x32xf32>
// CHECK: return %[[DOT]] : tensor<8x32xf32>
//   ^^^^ 直接返回！没有 all_reduce

// 【四种情形的完整对照】
//   #   收缩维    结果要求      all_reduce
//   ①   完整      完整          不需要
//   ②   完整      完整          不需要
//   ③   【分片】  【完整】      【需要】
//   ④   【分片】  【未归约】    【不需要】（延迟）

// 【判据】
//   收缩维是否被分片  +  结果是否接受未归约

// 【为什么延迟更好】
//   未归约时每台设备只算【自己那一份部分和】-> 计算量更小
//   越晚归约，中间省下的重复计算越多（L4-05 讲过）
//   这里把归约责任【交给调用者】

// 【回顾 L4-05 的五种时机】
//   立刻 / 延迟到某算子前 / 部分延迟 / 完全延迟到 return / 【完全不插】
//   本情形就是"完全不插"—— 因为结果本来就接受未归约`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '情形 ③', c: '#fb7185', d: '结果要求<b>完整</b><br>→ 插 <span class="mono">all_reduce</span>' },
      { t: '情形 ④', c: '#4ade80', d: '结果<b>也接受未归约</b><br>→ <b>不插</b><br><span class="dim">归约交给调用者</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>唯一的区别</b>就在函数结果的声明上 —— <span class="mono">unreduced={"x"}</span> 有没有。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>为什么可以省</b>：未归约时每台设备只算<b>自己那份部分和</b> —— 计算量更小。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>判据</b>：<b>收缩维是否被分片</b> + <b>结果是否接受未归约</b>。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>回顾 L4-05</b>：五种归约时机，本情形是"<b>完全不插</b>"—— 因为结果本来就接受未归约。';
    });
  }
},

/* ------------------------------------------------ 5 dot_general 的 2D 并行 */
{
  kicker: 'L5-05 · 矩阵乘降级',
  title: '<span class="mono hl-a">dot_general</span>：批维 + 收缩维 = <span class="hl-a">2D 并行</span>',
  sub: '批维分片**不需要**通信；收缩维分片**需要** —— 两者叠加就是 2D 并行。',
  caption: '这验证了核心规律：<b>只有收缩维上的分片需要 <span class="mono">all_reduce</span></b>。',
  code: `func.func @shard_contracting_dims(
  %arg0: tensor<4x16x8xf32> {...[{"x"}, {}, {"y"}]>},   // 批维切 x、收缩维切 y
  %arg1: tensor<4x8x16xf32> {...[{"x"}, {"y"}, {}]>})   // 批维切 x、收缩维切 y
  -> (tensor<4x16x16xf32> {...[{"x"}, {}, {}]>}) {
  %0 = stablehlo.dot_general %arg0, %arg1,
       batching_dims = [0] x [0], contracting_dims = [2] x [1]
       {...[{"x"}, {}, {}], unreduced={"y"}>} : ...
  %1 = sdy.all_reduce {"y"} %0 out_sharding=<@mesh_2_4, [{"x"}, {}, {}]> : ...
  return %1 : tensor<4x16x16xf32>
}
// CHECK-SAME: (%[[ARG0]]: tensor<2x16x2xf32> {...[{"x"}, {}, {"y"}]>},
// CHECK-SAME:  %[[ARG1]]: tensor<2x2x16xf32> {...[{"x"}, {"y"}, {}]>})
// CHECK-SAME: -> (tensor<2x16x16xf32> {...[{"x"}, {}, {}]>}) {
// CHECK: %[[DOT]] = stablehlo.dot_general %[[ARG0]], %[[ARG1]],
//         batching_dims = [0] x [0], contracting_dims = [2] x [1]
// CHECK-SAME: {sdy.sharding = ...<@mesh_2_4, [{"x"}, {}, {}], unreduced={"y"}>]}
// CHECK-SAME: : (tensor<2x16x2xf32>, tensor<2x2x16xf32>) -> tensor<2x16x16xf32>
// CHECK: %[[RES]] = "stablehlo.all_reduce"(%[[DOT]])
// CHECK-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<
//               mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "y">]>
// CHECK-SAME: use_global_device_ids
// CHECK: ^bb0(%[[ACC]]: tensor<f32>, %[[UPD]]: tensor<f32>):
// CHECK:   %[[ADD]] = stablehlo.add %[[ACC]], %[[UPD]] : tensor<f32>
// CHECK:   stablehlo.return %[[ADD]] : tensor<f32>
// CHECK: }) : (tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
// CHECK: return %[[RES]] : tensor<2x16x16xf32>

// 【逐项】
//   %arg0 的【收缩维（第 2 维）切 y】、%arg1 的【收缩维（第 1 维）切 y】—— 一致 ✓
//     -> 8 / 4 = 2
//   批维（第 0 维）切 x -> 4 / 2 = 2 ✓
//   结果标 unreduced={"y"} -> all_reduce {"y"}

// 【这是 2D 并行】
//   批维切 x（批并行）+ 收缩维切 y（数据并行）
//   【只有收缩维上的分片需要 all_reduce】—— 批维上的不需要 ✓
//   -> 这正是核心规律的验证

// 【reduction 区域】^bb0 里是 add —— 与 L5-02 的 all_reduce 一致`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '批维切 x', c: '#38bdf8', d: '各批独立<br><b>不需要</b>通信' },
      { t: '收缩维切 y', c: '#fb7185', d: '产生部分和<br><b>需要</b> <span class="mono">all_reduce</span>' },
      { t: '= 2D 并行', c: '#c084fc', d: '两者叠加<br>只有收缩维那部分<br>需要通信' },
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
        '批维分片是最自然的并行 —— 各批互不干扰。',
        '收缩维分片产生部分和 —— 必须归约。',
        '<b>2D 并行</b>：两个方向同时切。通信只发生在<b>收缩维</b>那个方向。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>核心规律的验证</b>：<b>只有收缩维上的分片需要 <span class="mono">all_reduce</span></b> —— 批维上的不需要。';
    });
  }
},

/* ------------------------------------------------ 6 小结 */
{
  kicker: 'L5-05 · 矩阵乘降级',
  title: '六种情形的小结与<span class="hl-a">核心规律</span>',
  sub: '一句话：**只有收缩维上的分片需要通信**。',
  caption: '这是矩阵乘降级的全部内容 —— 也是经典并行策略的完整图景。',
  code: `// 【六种情形总表】
//   文件          用例                            分片位置        通信        策略
//   dot           fully_replicated                无              无          —
//   dot           sharded_non_contracting_dims    非收缩维        无          模型并行
//   dot           sharded_contracting_dim         收缩维          all_reduce  数据并行
//   dot           ..._unreduced_result            收缩维(未归约)  无          延迟归约
//   dot_general   not_shard_contracting_dims      批维            无          批并行
//   dot_general   shard_contracting_dims          批维+收缩维     all_reduce  2D 并行

// 【★ 核心规律】
//   只有【收缩维】上的分片需要 all_reduce
//   非收缩维 / 批维上的分片都是"各算各的"，无需通信
//
//   为什么：
//     收缩维 = 【归约的方向】
//       沿它分片 -> 每台设备只算了【部分和】-> 必须合并
//     其他维 = 【输出元素的方向】
//       沿它分片 -> 每台设备算的是【不同的输出元素】-> 互不重叠

// 【族谱】2 个文件 / 119 行 / 6 用例
//   stablehlo_dot          68 行 / 4 用例
//   stablehlo_dot_general  51 行 / 2 用例

// 【跨课呼应】
//   L4-04  dot 的三类情形（本课是它们的降级形态）
//   L4-05  延迟归约的五种时机（本课的情形 ④ 是"完全不插"）
//   L5-02  all_reduce 的 reduction 区域与 replica_groups

// 一句话总结：
//   矩阵乘的分片降级 = 经典并行策略的 IR 体现
//   非收缩维 -> 模型并行；收缩维 -> 数据并行；两者叠加 -> 2D 并行
//   而通信只发生在【收缩维】上`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '全复制', n: 1, c: '#94a3b8' }, { t: '模型并行', n: 1, c: '#4ade80' },
      { t: '数据并行', n: 1, c: '#fb7185' }, { t: '延迟归约', n: 1, c: '#fbbf24' },
      { t: '批并行', n: 1, c: '#38bdf8' }, { t: '2D 并行', n: 1, c: '#c084fc' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:118px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div style="font-size:10.5px;color:${f.c}">${f.t}</div>
        <div class="big" style="font-size:16px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 100)); msg.innerHTML = '<b>6 个用例</b>，每个对应一种并行策略 —— 这是最"教科书式"的一课。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>核心规律</b>：只有<b>收缩维</b>上的分片需要 <span class="mono">all_reduce</span>。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>为什么</b>：收缩维是<b>归约方向</b>（产生部分和）；其他维是<b>输出方向</b>（产生不同元素）。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L5-06 讲卷积降级与 halo 处理。';
    });
  }
},

];
