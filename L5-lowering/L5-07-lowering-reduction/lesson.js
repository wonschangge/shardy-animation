/* ==========================================================================
   L5-07 · lowering-reduction
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/convert_global_to_local/ 下 2 个文件
         (stablehlo_reduce 88/3, stablehlo_reduce_window 60/2) = 148 行 / 5 用例
   目标：讲透归约类算子的降级 —— 局部归约 + 跨设备合并。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 五种情形 */
{
  kicker: 'L5-07 · 归约降级',
  title: '★ 五种情形：<span class="hl-a">归约是否跨设备</span>',
  sub: '归约是**跨元素**的操作 —— 归约维被分片时，每台设备只能归约自己那段。',
  caption: '本课的核心问题是：**怎么把各设备的部分归约结果合并起来**。',
  code: `// 【五种情形】2 个文件 / 148 行 / 5 用例
//   reduce        stablehlo_reduce                   无分片            -> 无通信
//   reduce        ..._fallback_all_reduce            归约维分片        -> all_reduce
//   reduce        ..._multi_result_sharded           归约维+多结果     -> all_gather+reduce
//   reduce_window batch_sharded                      批维(窗口=1)      -> 无
//   reduce_window ..._stride_greater_than_window     窗口维(步长>=窗口) -> 无

// 【核心问题】
//   归约是【跨元素】的操作
//   归约维被分片 -> 每台设备只能归约【自己那段】
//   -> 需要跨设备合并

// 【★ 两种合并方式】（本课重点）
//   单结果 -> all_reduce
//     归约【可结合】：reduce(reduce(a), reduce(b)) = reduce(a ++ b)
//     -> 直接合并部分结果即可
//   多结果 -> all_gather + 【重新 reduce】
//     多个结果之间【有依赖】（如 argmax 的"值+索引"）
//     -> 不能分别 all_reduce，必须【收齐数据再算】

// 【★ 统一的规律】
//   归约类算子的通信需求，取决于"归约是否跨设备"
//   reduce        归约维被分片       -> 跨设备 -> 需要通信
//   reduce_window 窗口跨设备边界     -> 需要通信；否则不需要

// 一句话：
//   归约类降级 = 局部归约 + 跨设备合并
//   合并方式取决于【结果个数】`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '无分片', c: '#94a3b8', d: '直接转换<br><b>无通信</b>' },
      { t: '归约维分片', c: '#fb7185', d: '局部归约<br>+ <b>all_reduce</b>' },
      { t: '多结果 + 分片', c: '#fbbf24', d: '<b>all_gather</b><br>+ 重新 reduce' },
      { t: '批维(窗口=1)', c: '#38bdf8', d: '窗口不跨批<br><b>无通信</b>' },
      { t: '步长≥窗口', c: '#4ade80', d: '窗口不跨界<br><b>无通信</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:140px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div style="font-size:10.5px;color:${x.c}">${x.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>五种情形</b> —— 只有两种需要通信。'; });
    tl.at(4400, () => {
      msg.innerHTML = '<b>判据</b>：<b>归约是否跨设备</b> —— 跨了就要合并，没跨就各算各的。';
    });
    tl.at(7800, () => {
      msg.innerHTML = '<b>★ 两种合并方式</b>：单结果用 <span class="mono">all_reduce</span>；多结果必须<b>收齐再算</b>。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<b>为什么多结果不行</b>：结果之间有<b>依赖</b>（如 argmax 的值与索引）—— 分别合并会不一致。';
    });
  }
},

/* ------------------------------------------------ 2 基础与 all_reduce */
{
  kicker: 'L5-07 · 归约降级',
  title: '基础情形与 <span class="mono hl-a">all_reduce</span>',
  sub: '无分片时直接转换（**连 reducer 区域都原样保留**）；归约维分片时"局部归约 + 全局归约"。',
  caption: '注意基础用例里那个<b>奇怪的 reducer</b> —— 它验证转换器<b>不分析 reducer 语义</b>。',
  code: `// 【情形 ①：无分片】直接转换
func.func @stablehlo_reduce(%arg0: tensor<32x8xi32>) -> (tensor<32xi32>) {
  %cst = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.reduce"(%arg0, %cst) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      %2 = stablehlo.add %1, %arg2 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
  }) { dimensions = array<i64: 1> }
  : (tensor<32x8xi32>, tensor<i32>) -> tensor<32xi32>
  return %0 : tensor<32xi32>
}
// CHECK-SAME: (%[[ARG0]]: tensor<32x8xi32>) -> tensor<32xi32> {
// CHECK-NEXT: %[[CST]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT: %[[RES]] = stablehlo.reduce(%[[ARG0]] init: %[[CST]]) across dimensions = [1]
// CHECK-NEXT:  reducer(%[[ARG1]]: tensor<i32>, %[[ARG2]]: tensor<i32>) {
// CHECK-NEXT:    %[[ADD]] = stablehlo.add %[[ARG1]], %[[ARG2]] : tensor<i32>
// CHECK-NEXT:    %[[ADD_2]] = stablehlo.add %[[ADD]], %[[ARG2]] : tensor<i32>
// CHECK-NEXT:    stablehlo.return %[[ADD_2]] : tensor<i32>
// CHECK-NEXT:  }
// CHECK-NEXT: return %[[RES]] : tensor<32xi32>
// 【读法】
//   无分片 -> 直接转换，类型不变（32x8 -> 32）
//   reducer 区域【原样保留】—— 注意它内部有两个 add（非标准的归约函数）
//   -> 这说明转换器【不关心 reducer 的内容】，只改类型

// 【情形 ②：归约维分片】局部归约 + all_reduce
func.func @stablehlo_reduce_unreduced_axes_fallback_all_reduce(
    %arg0: tensor<8x16xf32> {...[{"x"}, {"y"}]>})
    -> (tensor<8xf32> {...[{"x"}]>}) {
  %0 = stablehlo.reduce(%arg0 init: %init) across dimensions = [1]
      {...[{"x"}]...} : (tensor<8x16xf32>, tensor<f32>) -> tensor<8xf32>
    reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %mul = stablehlo.multiply %lhs, %rhs : tensor<f32>
      stablehlo.return %mul : tensor<f32>
    }
}
// CHECK-SAME: (%[[ARG0]]: tensor<4x4xf32> {...[{"x"}, {"y"}]>})
// CHECK-SAME: -> (tensor<4xf32> {...[{"x"}]>})
// CHECK-NEXT: %[[LOCAL_RED]] = stablehlo.reduce(%[[ARG0]] init: %[[INIT]])
//               applies stablehlo.multiply across dimensions = [1]
//               {...[{"x"}]...} : (tensor<4x4xf32>, tensor<f32>) -> tensor<4xf32>
// CHECK: %[[ALL_RED]] = "stablehlo.all_reduce"(%[[LOCAL_RED]])
// CHECK-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<
//               mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "y">]>
// CHECK: return %[[ALL_RED]] : tensor<4xf32>
// 【逐项】
//   张量        全局    分片            局部    说明
//   %arg0       8x16    [{"x"},{"y"}]   4x4     两维都切
//   局部归约后  8       [{"x"}]         4       归约第1维
// 【为什么 all_reduce 可行】
//   归约【可结合】：reduce(reduce(a), reduce(b)) = reduce(a ++ b)
//   -> "局部归约 + 全局归约" = "全局归约" ✓
// 【用例名里的 fallback】
//   说明这是【通用但可能不是最优】的做法
// 【reducer 是 multiply】—— 乘法也可结合，所以 all_reduce 仍然可行`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '无分片', c: '#94a3b8', d: '类型不变<br>reducer 区域<b>原样保留</b>' },
      { t: '归约维分片', c: '#fb7185', d: '局部 reduce（<span class="mono">4x4</span>→<span class="mono">4</span>）<br>+ <span class="mono">all_reduce</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>那个奇怪的 reducer</b>（两个 add）验证了转换器<b>不分析 reducer 语义</b>。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>为什么 <span class="mono">all_reduce</span> 可行</b>：归约<b>可结合</b> —— 局部归约再合并 = 全局归约。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>用例名里的 <span class="mono">fallback</span></b>：这是<b>通用但可能不是最优</b>的做法。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>注意 reducer 是 <span class="mono">multiply</span></b> —— 乘法也可结合，所以 <span class="mono">all_reduce</span> 仍然成立。';
    });
  }
},

/* ------------------------------------------------ 3 ★ 多结果 */
{
  kicker: 'L5-07 · 归约降级',
  title: '★ 多结果归约：<span class="mono hl-a">all_gather</span> + 重新 reduce',
  sub: '**不能**用 `all_reduce` —— 因为多个结果之间**有依赖**。',
  caption: '这是本课最精妙的一处，也是"多结果归约"的通用解法。',
  code: `func.func @stablehlo_reduce_multi_result_sharded(
    %arg0: tensor<8x16xf32> {...[{"x"}, {"y"}]>},
    %arg1: tensor<8x16xi32> {...[{"x"}, {"y"}]>})
    -> (tensor<8xf32> {...[{"x"}]>}, tensor<8xi32> {...[{"x"}]>}) {
  %0:2 = stablehlo.reduce(%arg0 init: %init0), (%arg1 init: %init1)
         across dimensions = [1] {...[{"x"}]...} : ...
    reducer(%lhs0: tensor<f32>, %rhs0: tensor<f32>)
           (%lhs1: tensor<i32>, %rhs1: tensor<i32>) {
      %cmp = stablehlo.compare GT, %lhs0, %rhs0 : ...
      %max_val = stablehlo.select %cmp, %lhs0, %rhs0 : ...
      %max_idx = stablehlo.select %cmp, %lhs1, %rhs1 : ...
      stablehlo.return %max_val, %max_idx : tensor<f32>, tensor<i32>
    }
  return %0#0, %0#1 : tensor<8xf32>, tensor<8xi32>
}
// 【五步】
// ① 局部 reduce -> 每台设备归约自己那段 -> 4xf32 + 4xi32
// CHECK-NEXT: %[[LOCAL_RED]]:2 = stablehlo.reduce(...) across dimensions = [1]
//               {...} : (tensor<4x4xf32>, tensor<4x4xi32>, ...) -> (tensor<4xf32>, tensor<4xi32>)
// ② reshape 为 all_gather 准备维度
// CHECK: %[[RESHAPE0]] = stablehlo.reshape %[[LOCAL_RED]]#0 : (tensor<4xf32>) -> tensor<4x1xf32>
// ③ all_gather 把各设备的结果【收齐】（两个各一次，channel 编号不同）
// CHECK-NEXT: %[[GATHER0]] = "stablehlo.all_gather"(%[[RESHAPE0]]) <{
//               all_gather_dim = 1 : i64,
//               channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>,
//               replica_groups = #stablehlo.replica_group_mesh_axes<
//                 mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "y">]>,
//               use_global_device_ids}> : (tensor<4x1xf32>) -> tensor<4x4xf32>
// CHECK-NEXT: %[[RESHAPE1]] = stablehlo.reshape %[[LOCAL_RED]]#1 : (tensor<4xi32>) -> tensor<4x1xi32>
// CHECK-NEXT: %[[GATHER1]] = "stablehlo.all_gather"(%[[RESHAPE1]]) <{
//               all_gather_dim = 1 : i64,
//               channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>,
//               ...}> : (tensor<4x1xi32>) -> tensor<4x4xi32>
// ④ 在【完整数据】上重新归约
// CHECK-NEXT: %[[FINAL_RED]]:2 = stablehlo.reduce(%[[GATHER0]] init: %[[INIT0]]),
//               (%[[GATHER1]] init: %[[INIT1]]) across dimensions = [1] : ...
// ⑤ 返回
// CHECK: return %[[FINAL_RED]]#0, %[[FINAL_RED]]#1 : tensor<4xf32>, tensor<4xi32>

// 【★ 为什么不能直接用 all_reduce】
//   reducer 是 argmax 模式：同时算出【最大值】与【最大值的位置】
//   两个结果【互相依赖】：%max_idx 的选择依赖 %cmp（由 %lhs0/%rhs0 决定）
//   如果分别对 %max_val 与 %max_idx 做 all_reduce
//     -> 两者的"最大值来自哪个设备"可能【不一致】-> 结果错误！
//   -> 必须【把数据收齐】，在完整数据上重新归约一次

// 【两个细节】
//   channel_handle 编号不同：GATHER0 用 2、GATHER1 用 3（避免冲突）
//   两个结果类型不同（f32/i32）-> 不能合并成一次 gather -> 各自做一次

// 【★ 这就是"多结果归约"的通用解法】
//   收齐数据 + 重新归约 —— 代价更高，但语义正确`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const steps = [
      { n: '①', t: '局部 reduce', c: '#38bdf8' },
      { n: '②', t: 'reshape', c: '#0ea5e9' },
      { n: '③', t: 'all_gather', c: '#fbbf24' },
      { n: '④', t: '重新 reduce', c: '#4ade80' },
      { n: '⑤', t: '返回', c: '#94a3b8' },
    ];
    const host = wrap.querySelector('#cards');
    const els = steps.map(s => {
      const e = U.el('div', { class: 'card', style: `width:140px;opacity:.33;transition:.3s;border-color:${s.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div class="small mono" style="font-size:10.5px;color:${s.c}">${s.n}</div>
        <div style="font-size:10.5px;margin-top:3px">${s.t}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    steps.forEach((s, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '每台设备先归约自己那段 —— 得到<b>部分结果</b>。',
        '为 <span class="mono">all_gather</span> 准备维度（<span class="mono">4</span> → <span class="mono">4x1</span>）。',
        '<b>把各设备的结果收齐</b> —— 两个结果各做一次（类型不同、通道不同）。',
        '在<b>完整数据</b>上重新归约 —— 这次能得到正确结果。',
        '两个结果一起返回。',
      ][i];
    }));
    tl.at(19000, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 为什么不能 all_reduce</b>：argmax 的"值"与"索引"<b>互相依赖</b> —— 分别合并会不一致。';
    });
  }
},

/* ------------------------------------------------ 4 reduce_window */
{
  kicker: 'L5-07 · 归约降级',
  title: '<span class="mono hl-a">reduce_window</span>：窗口是否<span class="hl-a">跨设备</span>',
  sub: '**唯一**的判据 —— 窗口跨设备边界就需要通信，否则不需要。',
  caption: '第二个用例特别巧妙：<b>分片维上有窗口</b>，但因为<b>步长 ≥ 窗口</b>，仍然无通信。',
  code: `// 【情形 ④：批维分片】窗口大小 = 1 -> 不跨批 -> 无通信
func.func @batch_sharded(
    %arg0: tensor<32x16x16x8xf32> {...[{"x"}, {}, {}, {}]>})
    -> (tensor<32x8x8x8xf32> {...[{"x"}, {}, {}, {}]>}) {
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{
    padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>,
    window_dimensions = array<i64: 1, 3, 3, 1>,      // <- 批维窗口 = 1！
    window_strides = array<i64: 1, 2, 2, 1>
  }> ({ ^bb0(%arg1, %arg2): %1 = stablehlo.maximum ... })
  {...[{"x"}, {}, {}, {}]...} : (tensor<32x16x16x8xf32>, tensor<f32>) -> tensor<32x8x8x8xf32>
}
// CHECK-SAME: (%[[ARG0]]: tensor<16x16x16x8xf32> -> (tensor<16x8x8x8xf32>
// CHECK-NEXT: %[[RES]] = "stablehlo.reduce_window"(%[[ARG0]], %[[CST]]) <{
// CHECK-SAME:   window_dimensions = array<i64: 1, 3, 3, 1>,
// CHECK-SAME:   window_strides = array<i64: 1, 2, 2, 1>}> ({
// 【读法】
//   批维（第0维）分片 -> 32/2 = 16
//   window_dimensions = [1, 3, 3, 1] —— 批维的窗口大小是【1】
//     -> 窗口【不跨批】-> 分片无通信 ✓
//   padding / window_dimensions / window_strides 【全部不变】

// 【情形 ⑤：步长 >= 窗口】仍无通信（本课最巧妙的用例）
func.func @reduce_window_stride_greater_than_window(
    %arg0: tensor<16x32xf32> {...[{}, {"x"}]>})
    -> (tensor<16x16xf32> {...[{}, {"x"}]>}) {
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{
    padding = dense<0> : tensor<2x2xi64>,
    window_dimensions = array<i64: 1, 2>,            // <- 第1维有窗口！
    window_strides = array<i64: 1, 2>                // <- 步长 = 窗口
  }> ({ ^bb0(%arg1, %arg2): %1 = stablehlo.maximum ... })
  {...[{}, {"x"}]...} : (tensor<16x32xf32>, tensor<f32>) -> tensor<16x16xf32>
}
// CHECK-SAME: (%[[ARG0]]: tensor<16x16xf32> -> (tensor<16x8xf32>
// CHECK-SAME:   window_dimensions = array<i64: 1, 2>,
// CHECK-SAME:   window_strides = array<i64: 1, 2>}>
// 【读法】表面上看应该需要通信 —— 因为【第1维被分片】且【第1维有窗口】！
//   但 window_strides = [1, 2] —— 【步长等于窗口大小】！
//   -> 每个窗口恰好落在一个设备的分片内，【不跨设备】✓
// 【逐项验证】
//   全局 16x32，第1维切 x=2 -> 每台 16x16
//   窗口大小 2、步长 2 -> 每台产生 16/2 = 8 个输出
//   设备0 处理第 0~15 列（输出第 0~7 个）
//   设备1 处理第 16~31 列（输出第 8~15 个）
//   窗口 [0,1]、[2,3]…… 【全部落在同一设备内】✓
// 【★ 规律】
//   reduce_window 需要通信的【唯一】情形：窗口跨越设备边界
//   当 stride >= window_dimensions 时，窗口不会跨界 -> 无需通信`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '批维分片', c: '#38bdf8', d: '窗口大小 = <b>1</b><br>窗口<b>不跨批</b><br>→ 无通信' },
      { t: '窗口维分片', c: '#4ade80', d: '但 <b>步长 ≥ 窗口</b><br>窗口恰好落在一台内<br>→ <b>仍无通信</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>批维最简单</b>：窗口大小是 1，窗口根本不跨批。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>第二个用例很巧妙</b>：分片维上<b>确实有窗口</b>，但<b>步长等于窗口</b> → 窗口不跨界。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>验证</b>：设备 0 处理第 0~15 列、窗口 <span class="mono">[0,1]</span>/<span class="mono">[2,3]</span>… <b>全在同一设备内</b>。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>★ 规律</b>：<span class="mono">reduce_window</span> 需要通信的<b>唯一</b>情形是<b>窗口跨设备边界</b>。';
    });
  }
},

/* ------------------------------------------------ 5 对照与小结 */
{
  kicker: 'L5-07 · 归约降级',
  title: '五种情形的<span class="hl-a">对照</span>与小结',
  sub: '一条判据 + 两种合并方式。',
  caption: '本课是 L5-05/L5-06 的姊妹课 —— 三课合起来覆盖了"归约方向"的所有情形。',
  code: `// 【五种情形总表】
//   文件           用例                            分片位置           通信
//   reduce         stablehlo_reduce                无                 无
//   reduce         ..._fallback_all_reduce         归约维             all_reduce
//   reduce         ..._multi_result_sharded        归约维 + 多结果    all_gather + reduce
//   reduce_window  batch_sharded                   批维（窗口=1）     无
//   reduce_window  ..._stride_greater_than_window  窗口维（步长>=窗口）无

// 【★ 一条判据】
//   归约类算子的通信需求 = "归约是否跨设备"
//     reduce        归约维被分片        -> 跨设备 -> 需要通信
//     reduce_window 窗口跨设备边界      -> 需要通信；否则不需要

// 【★ 两种合并方式】
//   单结果 -> all_reduce
//     归约【可结合】-> 直接合并部分结果
//   多结果 -> all_gather + 重新 reduce
//     结果间【有依赖】-> 必须收齐数据再算
//   选择依据：【结果个数】

// 【族谱】2 个文件 / 148 行 / 5 用例
//   stablehlo_reduce         88 行 / 3 用例
//   stablehlo_reduce_window  60 行 / 2 用例

// 【跨课呼应】
//   L4-05  延迟归约的五种时机（本课是它的降级形态）
//   L5-02  all_reduce / all_gather 的属性（本课复用）
//   L5-05  矩阵乘（归约维 = 收缩维）
//   L5-06  卷积（归约维 = 窗口 + 输入通道）

// 【L5 的进度】
//   L5-01 全局转局部    L5-02 集合通信    L5-03 结构算子    L5-04 形状类
//   L5-05 矩阵乘        L5-06 卷积        L5-07 归约（本课）
//   L5-08 gather/scatter    L5-09 pad for divisibility

// 一句话总结：
//   归约类降级 = 局部归约 + 跨设备合并
//   合并方式取决于【结果个数】—— 单结果 all_reduce、多结果收齐再算`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '无分片', n: 1, c: '#94a3b8' }, { t: 'all_reduce', n: 1, c: '#fb7185' },
      { t: '多结果', n: 1, c: '#fbbf24' }, { t: '批维', n: 1, c: '#38bdf8' },
      { t: '步长≥窗口', n: 1, c: '#4ade80' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:140px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div style="font-size:10.5px;color:${f.c}">${f.t}</div>
        <div class="big" style="font-size:16px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 100)); msg.innerHTML = '<b>5 个用例</b>，每个对应一种情形 —— 与 L5-05/L5-06 的组织方式一致。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>一条判据</b>：<b>归约是否跨设备</b>。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>两种合并方式</b>：单结果 <span class="mono">all_reduce</span>；多结果<b>收齐再算</b>。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L5-08 讲 gather/scatter 的降级（与 L3-11、L4-06 呼应）。';
    });
  }
},

];
