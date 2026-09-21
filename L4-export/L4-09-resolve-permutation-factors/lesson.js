/* ==========================================================================
   L4-09 · resolve-permutation-factors
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/resolve_permutation_factors.mlir (1463 行 / 46 用例)
         + resolve_permutation_factors/resolve_permutation_factors_replica_id.mlir
   目标：讲透 permutation 因子的消解 —— REPL 与 HALO 两种模式的差异。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 问题 */
{
  kicker: 'L4-09 · 置换因子消解',
  title: '问题：<span class="hl-a">permutation 因子</span>上的分片无法对应',
  sub: 'L2-10 讲过四类因子 —— `permutation` 是第四类，含义是**尺寸不成整数倍**。',
  caption: '本课的 RUN 行有两个，正是本课的主线：<b>halo exchange 开关</b>。',
  code: `// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=false" \\
//        | FileCheck %s --check-prefixes=CHECK,REPL
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=true" \\
//        | FileCheck %s --check-prefixes=CHECK,HALO
//        ^^^^^^^^^^^^^^^^^ 同一文件跑两次，两组 CHECK 前缀

// 【permutation 从哪来】尺寸不成整数倍
//   convolution   窗口 + 步长让空间维变小（16 -> 14）
//   pad / slice   显式改变维度大小
//   reshape       维度合并/拆分
//   reduce_window 窗口滑动

// 【问题】
//   如果张量正好在 permutation 因子上【有分片】
//   -> 它的分片【无法直接对应】到另一侧
//   -> 因为"每台设备拿多少"变了

// 【两条出路】
//   ① 全复制（REPL）—— 简单但通信量大
//   ② halo exchange（HALO）—— 通信量小但受 hop 数限制

// 本课 46 个用例覆盖三类算子：
//   convolution (1) / pad (12) / reshape (14) / 其它 (19)`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'permutation 从哪来', c: '#38bdf8', d: '卷积窗口 / pad / reshape<br>共同点：<b>尺寸不成整数倍</b>' },
      { t: '为什么无法对应', c: '#fbbf24', d: '"每台设备拿多少"变了<br>分片边界对不上' },
      { t: '两条出路', c: '#4ade80', d: '① <b>全复制</b>（REPL）<br>② <b>halo exchange</b>（HALO）' },
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
        '<b>回顾 L2-10</b>：四类因子是 <span class="mono">pass-through</span> / <span class="mono">reduction</span> / <span class="mono">need_replication</span> / <span class="mono">permutation</span>。',
        '比如 16 个元素切 2 台设备、每台 8 个；卷积后变成 14 个 —— <b>14 切不成两个 8</b>。',
        '<b>本课就是讲这两条出路的选择</b>：什么时候用哪个、代价分别是什么。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>两个 RUN 行</b>是理解本课的钥匙 —— 同一个文件，两种期望输出。';
    });
  }
},

/* ------------------------------------------------ 2 ★ REPL vs HALO */
{
  kicker: 'L4-09 · 置换因子消解',
  title: '★ 两种模式：<span class="mono hl-a">REPL</span> vs <span class="mono hl-a">HALO</span>',
  sub: '同一个 `convolution`，两种模式下的 IR 差别很大 —— 这正是 TODOLIST 的验收点。',
  caption: '<b>REPL</b>：全复制，简单但通信大。<b>HALO</b>：包进 <span class="mono">manual_computation</span>，只交换边界。',
  code: `// 输入在第 2 维（空间维）切了 "a"，而空间维是 permutation（16 -> 14）
func.func @convolution_spatial_permutation(
    %arg0: tensor<1x1x16x16xf32> {...<@mesh, [{}, {}, {"a"}, {}]>},
    %arg1: tensor<3x3x1x1xf32>)
    -> (tensor<1x1x14x14xf32> {...<@mesh, [{}, {}, {"a"}, {}]>}) {

// 【REPL 模式】enable-halo-exchange=false
//   %[[RESHARD_IN]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}, {}]>
//   ^^^^^^^^^^^^^^ ① 输入 reshard 成【全复制】（去掉 a）
//   %[[CONV]] = stablehlo.convolution(%[[RESHARD_IN]], %arg1)
//                 {sdy.sharding = ...<@mesh, [{}, {}, {}, {}]>}
//   ^^^^^^^^^ ② 在全复制状态下算
//   %[[RES]] = sdy.reshard %[[CONV]] <@mesh, [{}, {}, {"a"}, {}]>
//   ^^^^^^^^^ ③ 结果 reshard 回 a
//   代价：全复制 -> 每台设备都要拿到【完整输入】-> 通信量很大

// 【HALO 模式】enable-halo-exchange=true
//   把算子包进 sdy.manual_computation，只交换【边界数据】
//   %[[MC]] = sdy.manual_computation(%[[ARG0]], %[[CST]])
//       in_shardings=[<@mesh_a4, [{"a":(2)2}, {"b"}]>, <@mesh_a4, []>]
//       out_shardings=[<@mesh_a4, [{"a":(2)2}, {"b"}]>]
//       manual_axes={"a", "b"} (%[[ARG1]]: tensor<2x4xi32>, ...) {
//       ^^^^^^^^^^^^^^^^^^^ 两个轴都"冻结"
//   区域内每台设备处理【自己那一块 + 边界】
//   代价：只交换边界（几个元素）-> 比全复制小得多`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'REPL', c: '#fbbf24', tag: 'halo=false',
        d: 'reshard 成<b>全复制</b><br>→ 算子 → reshard 回去<br><b>通信量大</b>，但 IR 简单' },
      { t: 'HALO', c: '#4ade80', tag: 'halo=true',
        d: '包进 <span class="mono">manual_computation</span><br>只交换<b>边界</b>（halo）<br><b>通信量小</b>，但 IR 复杂' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:12px">${x.t} <span class="faint small">${x.tag}</span></div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>REPL 的思路</b>：既然分片对不上，那就<b>不要分片</b> —— 全复制后随便算。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>HALO 的思路</b>：分片保留，但每台设备<b>多拿一点边界数据</b>。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>为什么用 <span class="mono">manual_computation</span></b>：它正是"区域内自己管分片"的机制（L1-07）—— 天然适合表达 halo。';
    });
    tl.at(12000, () => {
      msg.innerHTML = '<b>取舍</b>：HALO 通信少但需要 hop 数在限制内；否则退回 REPL。';
    });
    tl.at(14600, () => {
      msg.innerHTML = '<b>这就是验收点问的</b>："开关 halo exchange 时 IR 的差别" —— 一个是两条 reshard，一个是 <span class="mono">manual_computation</span> 区域。';
    });
  }
},

/* ------------------------------------------------ 3 hop 概念 */
{
  kicker: 'L4-09 · 置换因子消解',
  title: '<span class="mono hl-a">pad</span>：<span class="hl-a">hop</span> 的概念与 <span class="mono">comm_free</span>',
  sub: '12 个 `pad_*` 用例把 halo 的代价讲得很细 —— 关键概念是 **hop**（需要跨几台设备取边界）。',
  caption: '<span class="mono">comm_free</span> 的情形<b>两种模式都零通信</b> —— 因为 padding 量恰好是整块。',
  code: `// 【comm_free】padding 量恰好是"一整块" -> 零通信
func.func @pad_comm_free(
  %arg0: tensor<8x8xi32> {...<@mesh, [{"a"}, {"b"}]>})
  -> (tensor<16x8xi32> {...<@mesh, [{"a"}, {"b"}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NOT: sdy.manual_computation
  // CHECK:      %[[PAD:.*]] = stablehlo.pad %[[ARG0]], %[[CST]]
  %0 = stablehlo.pad %arg0, %c, low = [4, 0], high = [4, 0], interior = [0, 0]
    {...<@mesh, [{"a"}, {"b"}]>} : (tensor<8x8xi32>, tensor<i32>) -> tensor<16x8xi32>
}
// 读法：mesh a=2 切 8 -> 每台 4 个元素
//       padding 4 正好是【一整块】-> 每台自己补就行
//       CHECK-NOT: sdy.manual_computation -> 两种模式都零通信

// 【single_left_hop】单跳
func.func @pad_single_left_hop(
  %arg0: tensor<4x8xi32> {...<@mesh_a4, [{"a":(2)2}, {"b"}]>})
  -> tensor<7x8xi32> {
  // REPL: %[[RESHARD]] = sdy.reshard %[[ARG0]] <@mesh_a4, [{}, {"b"}]>
  // REPL: %[[PAD]] = stablehlo.pad %[[RESHARD]], %[[CST]], low =...
  // REPL: %[[RES]] = sdy.reshard %[[PAD]] <@mesh_a4, [{"a":(2)2}, {"b"}]>
  // ^^^^ 全复制
  // HALO: %[[MC]] = sdy.manual_computation(%[[ARG0]], %[[CST]])
  //       in_shardings=[<@mesh_a4, [{"a":(2)2}, {"b"}]>, <@mesh_a4, []>]
  //       out_shardings=[<@mesh_a4, [{"a":(2)2}, {"b"}]>] manual_axes={"a", "b"}
  // ^^^^ 只交换 1 个元素（单跳）

// 【hop 的含义】需要跨【几台设备】取边界数据：
//   single_left_hop / single_right_hop        单跳（只跟邻居要）
//   multiple_left_hops                        多跳
//   multiple_right_hops_beyond_halo_limit     超出 halo 限制！
//   two_direction_hops                        双向都要
//   large_low_pad_within_one_hop              padding 大但在一跳内
//
// beyond_halo_limit 很关键：hop 数有上限
//   超过上限 -> halo exchange 不可行 -> 只能退回 REPL`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'comm_free', c: '#4ade80', n: '零通信',
        d: 'padding 恰好是<b>一整块</b><br>每台自己补<br><span class="mono">CHECK-NOT: manual_computation</span>' },
      { t: '单跳 / 多跳', c: '#38bdf8', n: 'hop',
        d: '需要跨<b>几台设备</b><br>取边界数据<br><span class="dim">只跟邻居要 = 单跳</span>' },
      { t: '超出 halo 限制', c: '#fb7185', n: '退回 REPL',
        d: '<span class="mono">beyond_halo_limit</span><br>hop 数有上限<br>超了就<b>不可行</b>' },
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
        '<b>为什么零通信</b>：mesh <span class="mono">a=2</span> 切 8 → 每台 4 个；padding 4 <b>正好一整块</b>。',
        '<b>hop 是 halo 的代价单位</b>：跳得越远，通信开销越大。',
        '<b>这是关键的边界情形</b>：halo 不是万能的 —— 超出限制时必须退回全复制。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>设计启示</b>：<span class="mono">hop</span> 上限是一个<b>可调参数</b> —— 在通信量与 IR 复杂度之间权衡。';
    });
  }
},

/* ------------------------------------------------ 4 reshape 与其余 */
{
  kicker: 'L4-09 · 置换因子消解',
  title: '<span class="mono hl-a">reshape</span> 的 14 个用例与其余算子',
  sub: 'reshape 也会产生 permutation 因子 —— 但**某些情形仍然零通信**。',
  caption: '注意命名里的 <span class="mono">comm_free</span> 与 <span class="mono">halo_impossible</span> —— 两类边界情形。',
  code: `// 【reshape 的 14 个用例】
//   reshape_1d_to_2d_non_divisible_comm_free      不可整除但【零通信】
//   reshape_2d_to_1d_non_divisible_comm_free      反向
//   reshape_single_dim_split_comm_free            单维拆分，零通信
//   reshape_single_dim_combine_comm_free          单维合并，零通信
//   reshape_indivisible_cross_dims                跨维不可整除
//   reshape_2x3x5_to_30_group_padded_size_mismatch 尺寸不匹配
//   reshape_1d_to_2d_split                        基础拆分
//   reshape_2d_split_with_unrelated_axis          带无关轴
//   reshape_1d_to_2d_split_gap_2                  中间隔 2
//   reshape_1d_to_3d_split                        拆成 3 维
//   reshape_two_splitting_groups                  两组同时拆
//   reshape_1d_to_2d_split_custom_device_ids      【自定义设备号】
//   reshape_mix_split_combine_halo_impossible     混合 + halo 不可行

// 两类值得注意的边界：
//   comm_free       分片恰好落在"整齐"的位置 -> 零通信
//   halo_impossible halo 不可行 -> 必须退回 REPL

// 【其余用例】
//   convolution_spatial_permutation  卷积空间维（本课开篇的例子）
//   strided_slice                    带步长的切片
//   reduce_window_permutation        窗口规约
//   reverse_divisible                反转（可整除）

// reverse_divisible 与 L4-03 的 reverse.mlir 呼应：
//   反转【不改变维度大小】-> 可整除时分片能直接对应

// 【第二个文件】resolve_permutation_factors_replica_id.mlir
//   处理 replica_id 场景
//   stablehlo.replica_id 返回当前设备的副本编号
//   -> 它让程序能【根据设备编号做不同的事】
//   这与分片有微妙关系：
//     分片改变"哪台设备拿哪片数据"，也就改变了 replica_id 的语义
//   -> 所以有 replica_id 时必须【特别小心】
//   为什么单独一个文件：replica_id 是【依赖设备身份】的操作，
//     与"纯数据并行"的分片模型有本质冲突`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'comm_free', c: '#4ade80', n: '零通信',
        d: '分片恰好落在<br>"整齐"的位置<br><span class="dim">reshape / pad 都有</span>' },
      { t: 'halo_impossible', c: '#fb7185', n: '退回 REPL',
        d: '混合拆分合并时<br>halo 不可行' },
      { t: '自定义设备号', c: '#38bdf8', n: 'device_ids',
        d: '<span class="mono">reshape_1d_to_2d_split_<br>custom_device_ids</span>' },
      { t: 'replica_id', c: '#c084fc', n: '第二文件',
        d: '依赖<b>设备身份</b><br>与纯数据并行<br>有本质冲突' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px` });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:11.5px">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:10.5px;line-height:1.45">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>为什么能零通信</b>：分片边界恰好与 reshape 后的边界重合 —— 没有数据需要搬。',
        '另一类边界：<b>halo 不可行</b>时必须退回 REPL。',
        '设备号不同会影响"哪台设备拿哪片"—— 与 L4-07 的 mesh 切换呼应。',
        '<b>为什么单独一个文件</b>：<span class="mono">replica_id</span> 让程序依赖设备编号，而分片恰好改变设备与数据的对应。',
      ][i];
    }));
    tl.at(14400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>两类边界情形贯穿本课</b>：<span class="mono">comm_free</span>（不用通信）与 <span class="mono">halo_impossible</span>（不能用 halo）。';
    });
  }
},

/* ------------------------------------------------ 5 族谱 */
{
  kicker: 'L4-09 · 置换因子消解',
  title: '族谱与<span class="hl-a">一句话总结</span>',
  sub: 'permutation 因子上的分片无法直接对应 —— 两条出路各有代价。',
  caption: '本课与 L2-10（因子分类）、L4-03（reverse）、L4-08（collective 序列）都有呼应。',
  code: `// 【族谱】主文件 46 个用例
//   convolution                 1   空间维 permutation
//   pad_*                      12   hop 数与 comm_free 的边界
//   reshape_*                  14   合并/拆分的 permutation 处理
//   strided_slice / reduce_window / reverse   3   其它 permutation 来源
//   其它                       ~16

// 【两种模式对照】
//   ┌──────────┬──────────────────┬──────────┬────────────┐
//   │ 模式     │ 做法             │ 通信量   │ IR 复杂度  │
//   ├──────────┼──────────────────┼──────────┼────────────┤
//   │ REPL     │ 全复制 → 算子 →  │ 大       │ 简单       │
//   │ (halo=F) │ reshard 回去     │          │ (两条)     │
//   │ HALO     │ manual_computation│ 小       │ 复杂       │
//   │ (halo=T) │ 只交换边界       │          │ (一个区域) │
//   └──────────┴──────────────────┴──────────┴────────────┘

// 【三种特殊情形】
//   comm_free        分片恰好整齐 -> 两种模式都零通信
//   beyond_halo_limit hop 数超上限 -> 只能退回 REPL
//   halo_impossible  halo 不可行 -> 只能退回 REPL

// 一句话总结：
//   permutation 因子上的分片无法直接对应
//   两条出路：全复制（REPL，简单但通信大）
//             或 halo exchange（HALO，通信小但受 hop 限制）`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: 'convolution', n: 1, c: '#94a3b8' }, { t: 'pad', n: 12, c: '#4ade80' },
      { t: 'reshape', n: 14, c: '#38bdf8' }, { t: '其它算子', n: 3, c: '#fbbf24' },
      { t: '其它', n: 16, c: '#c084fc' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:140px;opacity:.4;transition:.35s;border-color:${f.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="card-t" style="color:${f.c};font-size:11px">${f.t}</div>
        <div class="big" style="font-size:20px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 120)); msg.innerHTML = '<b>46 个用例</b>分五族 —— 最大两族是 reshape（14）与 pad（12）。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>两种模式</b>：REPL 通信大但 IR 简单；HALO 通信小但 IR 复杂。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>三种特殊情形</b>：<span class="mono">comm_free</span>（零通信）、<span class="mono">beyond_halo_limit</span>、<span class="mono">halo_impossible</span>。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L4-10 讲 <span class="mono">optimize-collectives</span> —— 对已插入的通信做优化。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L4-09 · 练习',
  title: '练一练：<span class="hl-a">两种模式的取舍</span>',
  sub: '三道题分别考：permutation 的问题、两种模式、hop 限制。',
  caption: '一句话总结：<b>全复制简单但通信大，halo 通信小但受 hop 限制</b>。',
  code: `// 题 1：为什么 permutation 因子上的分片无法直接对应？

// 题 2：halo exchange 开关打开与关闭，IR 的差别是什么？

// 题 3：什么情况下 halo exchange 不可行？怎么办？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col compact-ex', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '为什么 <span class="mono">permutation</span> 因子上的分片无法直接对应？',
        a: '因为 permutation 意味着<b>尺寸不成整数倍</b> —— "每台设备拿多少"变了。' +
           '<br><b>例</b>：16 个元素切 2 台、每台 8 个；卷积后变成 14 个 —— <b>14 切不成两个 8</b>，分片边界对不上。' +
           '<br><span class="dim">典型来源：卷积窗口、<span class="mono">pad</span>/<span class="mono">slice</span>、<span class="mono">reshape</span>、<span class="mono">reduce_window</span>。</span>'
      },
      {
        q: '<span class="mono">halo exchange</span> 开关打开与关闭，IR 的差别是什么？',
        a: '<b>关闭（REPL）</b>：输入 reshard 成<b>全复制</b> → 在全复制上算子 → 结果 reshard 回去。<b>两条 reshard，IR 简单，但通信量大</b>。' +
           '<br><b>打开（HALO）</b>：把算子包进 <span class="mono">sdy.manual_computation</span>（<span class="mono">manual_axes</span> 冻结相关轴），区域内每台设备处理<b>自己那块 + 边界</b>。<b>一个区域，IR 复杂，但通信量小</b>。' +
           '<br><span class="dim">这就是本课两个 RUN 行（<span class="mono">REPL</span> / <span class="mono">HALO</span> 两组 CHECK 前缀）的差异。</span>'
      },
      {
        q: '什么情况下 <span class="mono">halo exchange</span> 不可行？怎么办？',
        a: '两种情形：' +
           '<br>① <b>hop 数超出上限</b>（用例 <span class="mono">pad_multiple_right_hops_beyond_halo_limit</span>）—— 边界数据跨太多台设备；' +
           '<br>② <b>halo 本身不可行</b>（用例 <span class="mono">reshape_mix_split_combine_halo_impossible</span>）—— 混合拆分合并时结构不允许。' +
           '<br><b>怎么办</b>：<b>退回 REPL</b>（全复制）。' +
           '<br><span class="dim">另有一个"不用通信"的幸运情形：<span class="mono">comm_free</span> —— padding 量恰好是整块（或分片边界恰好整齐），两种模式都零通信。</span>'
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
