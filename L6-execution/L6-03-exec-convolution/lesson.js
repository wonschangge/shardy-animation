/* ==========================================================================
   L6-03 · exec-convolution
   --------------------------------------------------------------------------
   覆盖：executable_convert_global_to_local/ 下 4 个文件
         + executable_partitioner_pipeline/stablehlo_convolution_
           dual_semantics_factor_indivisible.mlir
   目标：用真实数值验证 L5-06 的卷积分片规则 + L4-09 的 REPL/HALO 等价。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 四个维度 */
{
  kicker: 'L6-03 · 卷积执行',
  title: '★ 四个文件 = <span class="hl-a">四种分片维度</span>',
  sub: '与 **L5-06 的六个用例完全对应** —— 那里讲 IR 形态，这里验证数值。',
  caption: '本课还多一个文件：验证 <b>L4-09</b> 的 REPL / HALO 等价性。',
  code: `// 【5 个文件】
//   文件                                  分片的维              group_count
//   stablehlo_convolution_shard_batch     批维 b                1
//   stablehlo_convolution_shard_batch_group 批维 + 权重输出维   【2】
//   stablehlo_convolution_shard_contracting_dim 输入通道 i（收缩维）  1
//   stablehlo_convolution_shard_feature_group   特征维 f          【2】
//   partitioner_pipeline/
//     stablehlo_convolution_dual_semantics_factor_indivisible
//                                          （验证 REPL vs HALO 等价）

// 【两课的分工】
//   L5-06  讲【IR 形态】（降级后长什么样）
//   L6-03（本课）验证【数值正确性】

// 【网格】sdy.mesh @mesh_2 = <["x"=2]>

// 【★ 本课最重要的价值：三个测试设计技巧】
//   ① 用 @sequential_conv 【现算期望值】  <- 卷积的期望值难手算
//   ② 【手动切分输入】                     <- 验证"分片 + 拼接"全过程
//   ③ 【缩放】让贡献可辨识                 <- 提高测试敏感度
//   这三个技巧在后面几课也会反复出现

// 一句话：
//   L6-03 用真实数值验证 L5-06 的卷积分片规则
//   四种分片维度各一个文件，外加一个验证 REPL/HALO 等价的流水线测试`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'shard_batch', c: '#38bdf8', d: '批维<br><span class="mono">group=1</span>' },
      { t: 'shard_batch_group', c: '#4ade80', d: '批维+权重输出维<br><span class="mono">group=2</span>' },
      { t: 'shard_contracting_dim', c: '#fb7185', d: '<b>收缩维</b><br>需 all_reduce' },
      { t: 'shard_feature_group', c: '#fbbf24', d: '特征维<br><span class="mono">group=2</span>' },
      { t: 'dual_semantics', c: '#c084fc', d: '<b>REPL vs HALO</b><br><span class="dim">L4-09</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:140px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:9px;color:${x.c};overflow-wrap:anywhere">${x.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>前四个验证 L5-06</b>（四种分片维度），<b>第五个验证 L4-09</b>（REPL/HALO）。'; });
    tl.at(4400, () => {
      msg.innerHTML = '<b>两课分工</b>：L5-06 讲 <b>IR 形态</b>；本课验证 <b>数值正确性</b>。';
    });
    tl.at(7800, () => {
      msg.innerHTML = '<b>★ 本课最重要的价值</b>：三个<b>测试设计技巧</b>（现算期望值 / 手动切分 / 缩放）。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<b>这三个技巧</b>在后面几课也会反复出现 —— 是读 <span class="mono">executable_*</span> 测试的通用工具。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 技巧一：现算期望值 */
{
  kicker: 'L6-03 · 卷积执行',
  title: '★ 技巧一：用 <span class="mono hl-a">@sequential_conv</span> 现算期望值',
  sub: '卷积的期望值**很难手算** —— 直接调串行版算，既准确又省力。',
  caption: '这是<b>最自然的期望值来源</b> —— 因为 L6-00 讲的机制就是"分片版 vs 串行版"。',
  code: `  %expected = func.call @sequential_conv(%lhs, %rhs) : (tensor<2x4x4x2xi32>, tensor<3x3x2x4xi32>) -> tensor<2x2x2x4xi32>

// 【读法】期望值【不是硬编码常量】，而是【调用串行版现算】
//
// 【@sequential_conv 从哪来？】
//   ① 脚本【自动生成】（如果 part1 里只有 @parallel_conv）—— L6-00 讲过
//   ② 【手写】（如果 part1 里显式写了 @sequential_conv）
//   脚本的 if 条件是"有 parallel 且【没有】sequential"时才自动生成
//   -> 手写时脚本【不覆盖】

// 【★ 对比 L6-01 / L6-02 的期望值来源】
//   课        期望值来源
//   L6-01    【硬编码常量】（dense<1111>）
//   L6-02    硬编码 / concatenate 现算
//   L6-03    【func.call @sequential_conv(...)】
//
// 【为什么 L6-03 要用现算】
//   卷积的期望值【很难手算】—— 窗口 + 步长 + padding 的组合
//   直接调串行版算 -> 【既准确又省力】

// 【★ 这是"最自然的标准答案"】
//   L6-00 讲的机制就是"分片版 vs 串行版"
//   串行版【本来就是】脚本要生成来当参考实现的
//   直接在 part2 里调用它 -> 最直接

// 【shard_contracting_dim 里 @sequential_conv 是【手写】的】
func.func @sequential_conv(%arg0: tensor<2x4x4x4xi32>, %arg1: tensor<3x3x4x4xi32>) -> tensor<2x2x2x4xi32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]}
    {
      feature_group_count = 1 : i64,
      batch_group_count = 1 : i64
    } : (tensor<2x4x4x4xi32>, tensor<3x3x4x4xi32>) -> tensor<2x2x2x4xi32>
  return %0 : tensor<2x2x2x4xi32>
}
// 【读法】显式写在 part1 里 —— 【没有分片属性】，是纯串行版
//   脚本【不会覆盖它】（if 条件：有 parallel 且没有 sequential 时才生成）`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'L6-01', c: '#38bdf8', d: '期望值<br><b>硬编码常量</b><br><span class="mono">dense&lt;1111&gt;</span>' },
      { t: 'L6-02', c: '#4ade80', d: '硬编码<br>或 <span class="mono">concatenate</span> 现算' },
      { t: 'L6-03（本课）', c: '#fbbf24', d: '<b><span class="mono">func.call @sequential_conv</span></b><br>最自然的"标准答案"' },
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
        '<b>L6-01</b> 的期望值最简单 —— 求和结果 <span class="mono">1111</span> 可以直接写。',
        '<b>L6-02</b> 的拼接类用 <span class="mono">concatenate</span> 现算 —— 因为语义就是拼接。',
        '<b>本课</b>：卷积的期望值<b>很难手算</b> → 直接调串行版算，<b>既准确又省力</b>。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>@sequential_conv 的来源</b>：脚本自动生成（若 part1 只有 parallel）或<b>手写</b>（脚本不覆盖）。';
    });
  }
},

/* ------------------------------------------------ 3 ★ 技巧二：手动切分 */
{
  kicker: 'L6-03 · 卷积执行',
  title: '★ 技巧二：<span class="hl-a">手动切分输入</span>',
  sub: '手工模拟"**分片 → 各算各的 → 拼接**"的全过程。',
  caption: '这验证了 L5-06 讲的"<b>批维分片无通信</b>"。',
  code: `  %l0 = "stablehlo.slice"(%lhs) {start_indices=array<i64: 0,0,0,0>, limit_indices=array<i64: 1,4,4,2>, strides=array<i64: 1,1,1,1>} : (tensor<2x4x4x2xi32>) -> tensor<1x4x4x2xi32>
  %l1 = "stablehlo.slice"(%lhs) {start_indices=array<i64: 1,0,0,0>, limit_indices=array<i64: 2,4,4,2>, strides=array<i64: 1,1,1,1>} : (tensor<2x4x4x2xi32>) -> tensor<1x4x4x2xi32>

  %res:2 = "interpreter.run_parallel"(%l0, %rhs, %l1, %rhs) {
    programs = [[@parallel_conv, @parallel_conv]]
  } : (tensor<1x4x4x2xi32>, tensor<3x3x2x4xi32>, tensor<1x4x4x2xi32>, tensor<3x3x2x4xi32>) -> (tensor<1x2x2x4xi32>, tensor<1x2x2x4xi32>)

  %actual = "stablehlo.concatenate"(%res#0, %res#1) {dimension = 0 : i64} : (tensor<1x2x2x4xi32>, tensor<1x2x2x4xi32>) -> tensor<2x2x2x4xi32>
  "check.expect_eq"(%actual, %expected) : (tensor<2x2x2x4xi32>, tensor<2x2x2x4xi32>) -> ()

// 【读法】手工模拟"分片 + 拼接"的全过程
//   ① %lhs 是完整的 2x4x4x2
//      按【批维】手动 slice 成 %l0（第 0 批）与 %l1（第 1 批）
//   ② 两台设备各拿一份 1x4x4x2
//      —— 这【正是】批维分片的效果（2/2 = 1）
//   ③ 各自卷积 -> 1x2x2x4
//   ④ concatenate 拼回 2x2x2x4
//      —— 这【就是】"分片计算 + 结果合并"
//   ⑤ 与 %expected（串行版结果）比较 ✓

// 【★ 这验证了什么】
//   批维分片 = "各算各的批次，再拼起来"
//   与串行结果一致
//   -> 这正是 L5-06 讲的"【批维分片无通信】"的【数值证据】

// 【为什么不用 run_parallel 自动切分】
//   run_parallel 的每个操作数就是一台设备的输入
//   【必须手动】构造出"每台设备该拿什么"
//   -> 这反而让测试【更透明】：读者能直接看到"设备 0 拿第 0 批"

// 【对比 shard_contracting_dim 的切法】
//   shard_batch            沿【批维】切（dim 0）
//   shard_contracting_dim  沿【通道维】切（dim 3 与 dim 2）
//   -> 切法不同，验证的分片语义也不同`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const steps = [
      { n: '①', t: 'slice 切分', c: '#38bdf8', d: '<span class="mono">2x4x4x2</span> → 两个<br><span class="mono">1x4x4x2</span>' },
      { n: '②', t: '各算各的', c: '#4ade80', d: '两台设备<br>各卷积自己的批次' },
      { n: '③', t: 'concatenate', c: '#fbbf24', d: '拼回<br><span class="mono">2x2x2x4</span>' },
      { n: '④', t: 'expect_eq', c: '#c084fc', d: '与串行结果<br>比较 ✓' },
    ];
    const host = wrap.querySelector('#cards');
    const els = steps.map(s => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${s.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="small mono" style="font-size:10.5px;color:${s.c}">${s.n}</div>
        <div class="mono" style="font-size:10.5px;margin-top:3px">${s.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35;margin-top:3px">${s.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    steps.forEach((s, i) => tl.at(700 + i * 3800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>手动 slice</b> —— 按批维把 <span class="mono">2x4x4x2</span> 切成两份。',
        '这【正是】批维分片的效果（<span class="mono">2/2 = 1</span>）—— 每台一个批次。',
        '<b>concatenate 拼回</b> —— 这就是"分片计算 + 结果合并"。',
        '与串行版结果比较 → 一致 ✓',
      ][i];
    }));
    tl.at(15800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 这验证了 L5-06 的"批维分片无通信"</b> —— 各算各的批次，再拼起来。';
    });
  }
},

/* ------------------------------------------------ 4 ★ 技巧三：缩放 */
{
  kicker: 'L6-03 · 卷积执行',
  title: '★ 技巧三：<span class="hl-a">缩放</span>让贡献可辨识',
  sub: '把 Shard 1 乘以 1000 —— 让"漏掉一个 shard"必然导致结果明显错误。',
  caption: '这是一个<b>通用</b>的测试设计技巧，值得单独记住。',
  code: `  %l1_unscaled = "stablehlo.slice"(%lhs_base) {start_indices=array<i64: 0,0,0,2>, limit_indices=array<i64: 2,4,4,4>, strides=array<i64: 1,1,1,1>} : (tensor<2x4x4x4xi32>) -> tensor<2x4x4x2xi32>
  // Scale Shard 1 by 1000 to make its contribution identifiable in the sum
  %c1000 = stablehlo.constant dense<1000> : tensor<2x4x4x2xi32>
  %l1 = stablehlo.multiply %l1_unscaled, %c1000 : tensor<2x4x4x2xi32>
  %lhs = "stablehlo.concatenate"(%l0, %l1) {dimension = 3 : i64} : (tensor<2x4x4x2xi32>, tensor<2x4x4x2xi32>) -> tensor<2x4x4x4xi32>

// 【注释直接点明了意图】
//   Scale Shard 1 by 1000 to make its contribution identifiable in the sum
//
// 【读法】把 Shard 1 乘以 1000
//   让它的贡献在【求和】中"【可辨识】"
//
// 【★ 为什么需要这一步】
//   shard_contracting_dim 的用例里，两个 shard 的贡献会被
//     all_reduce 【累加】
//   如果两个 shard 的数值【相近】：
//     "【正确累加】"与"【只算了一个 shard】"的结果可能【碰巧接近】
//     -> 测试就【区分不出来】！
//   缩放后，贡献差异被【放大】
//     -> 测试的【敏感度提高】 ✓
//
// 【★ 这是一个通用的测试设计技巧】
//   当要验证"多个部分的贡献都被正确合并"时
//   【给各部分不同的量级】
//   让"漏掉一个"必然导致结果明显错误
//
// 【在别处也能看到类似的技巧】
//   L6-01 的 all_reduce 用例：4 台设备给 1/10/100/1000
//     也是"不同的量级" —— 让"漏掉一台"必然算错
//   L6-02 同样用 1/10/100/1000
//   -> 这是 Shardy 测试的【惯例】
//
// 【反例：如果都给 1】
//   4 台设备都输入 1，all_reduce 得 4
//   如果只归约了 3 台，得 3 —— 也能发现
//   但如果输入是 1/2/3/4，正确得 10
//     "漏掉一个"可能得 6、7、8、9 —— 都不同，更容易定位
//   而 1/10/100/1000 的差距更【悬殊】，更安全`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '问题', c: '#fb7185', d: '两个 shard 数值<b>相近</b>时<br>"正确累加"与"只算一个"<br>结果可能<b>碰巧接近</b>' },
      { t: '解法', c: '#4ade80', d: '把 Shard 1 <b>乘以 1000</b><br>→ 贡献差异被<b>放大</b><br>→ 测试<b>敏感度提高</b>' },
      { t: '惯例', c: '#38bdf8', d: 'Shardy 测试常用<br><span class="mono">1 / 10 / 100 / 1000</span><br><span class="dim">L6-01 / L6-02 也是</span>' },
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
        '<b>隐患</b>：如果两个 shard 数值相近，"漏算一个"可能<b>看不出来</b>。',
        '<b>解法</b>：给不同量级 —— 让"漏掉一个"必然导致结果<b>明显错误</b>。',
        '<b>这是惯例</b>：Shardy 的测试普遍用 <span class="mono">1/10/100/1000</span> 这种悬殊的量级。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>通用技巧</b>：验证"多部分贡献都被合并"时，<b>给各部分不同的量级</b>。';
    });
  }
},

/* ------------------------------------------------ 5 contracting_dim */
{
  kicker: 'L6-03 · 卷积执行',
  title: '<span class="mono hl-a">shard_contracting_dim</span>：验证 L5-06 的核心规律',
  sub: '收缩维分片 → `unreduced` + `all_reduce` —— 这是 L5-06 讲的"归约因子"规律。',
  caption: '与 L5-06 的 <span class="mono">shard__reduction_factors</span> 用例<b>同构</b>。',
  code: `func.func @parallel_conv(
  %arg0: tensor<2x4x4x4xi32> {sdy.sharding = #sdy.sharding<@mesh<｜｜begin▁of▁sentence｜｜>陕西省_2, [{}, {}, {}, {"x"}]>},
  %arg1: tensor<3x3x4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}, {"x"}, {}]>}
) -> (tensor<2x2x2x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}, {}, {}]>}) {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]}
    {
      feature_group_count = 1 : i64,
      batch_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2, [{}, {}, {}, {}], unreduced={"x"}>]>
    } : (tensor<2x4x4x4xi32>, tensor<3x3x4x4xi32>) -> tensor<2x2x2x4xi32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2, [{}, {}, {}, {}]> : tensor<2x2x2x4xi32>
  return %1 : tensor<2x2x2x4xi32>
}

// 【读法】与 L5-06 的 shard__reduction_factors 用例【同构】
//   %arg0 的【第 3 维（输入通道 i，收缩维）】切 {"x"}
//   %arg1 的【第 2 维（输入通道 i）】切 {"x"}
//   -> 两个收缩维【都被切了 x】—— 且一致 ✓
//   -> convolution 标 unreduced={"x"} -> all_reduce {"x"}
//
// 【★ 这就是 L5-06 讲的"归约因子分片需要 all_reduce"的数值验证】

// 【验证方式】
  %seq = func.call @sequential_conv(%lhs, %rhs) : (tensor<2x4x4x4xi32>, tensor<3x3x4x4xi32>) -> tensor<2x2x2x4xi32>
  %pars:2 = "interpreter.run_parallel"(%l0, %r0, %l1, %r1) {
    programs = [[@parallel_conv, @parallel_conv]]
  } : (tensor<2x4x4x2xi32>, tensor<3x3x2x4xi32>, tensor<2x4x4x2xi32>, tensor<3x3x2x4xi32>) -> (tensor<2x2x2x4xi32>, tensor<2x2x2x4xi32>)
  "check.expect_eq"(%pars#0, %seq) : (tensor<2x2x2x4xi32>, tensor<2x2x2x4xi32>) -> ()
// 【读法】
//   两台设备各拿【输入通道的一半】（%l0/%l1 与 %r0/%r1 都沿通道维切）
//   @parallel_conv 内部有 all_reduce
//     -> 两台设备的结果【都应该是完整的卷积】
//   断言 %pars#0 == %seq —— 【只检查设备 0】
//     因为 all_reduce 后两台【相同】

// 【对比 shard_batch 的切法】
//   shard_batch            沿【批维】切（dim 0）-> 无通信
//   shard_contracting_dim  沿【通道维】切（dim 3 与 dim 2）-> all_reduce
//   -> 切法不同，验证的分片语义也不同`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'shard_batch', c: '#4ade80', d: '沿<b>批维</b>切<br><span class="mono">[{}, {}, {}, {"x"}]</span><br><b>无通信</b>' },
      { t: 'shard_contracting_dim', c: '#fb7185', d: '沿<b>通道维</b>切<br><span class="mono">[{"x"}...]</span> 与 <span class="mono">[{}, {"x"}...]</span><br><b>all_reduce</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:11px;overflow-wrap:anywhere">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>批维切</b>：各算各的批次，<b>无需通信</b> —— 这是 L5-06 讲的"输出维分片"。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>通道维切</b>：两个收缩维都切了 → 产生<b>部分结果</b> → 需要 <span class="mono">all_reduce</span>。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>断言只检查设备 0</b> —— 因为 <span class="mono">all_reduce</span> 后两台设备结果<b>相同</b>。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>★ 这是 L5-06 核心规律的数值验证</b>：<b>归约因子分片需要 all_reduce</b>。';
    });
  }
},

/* ------------------------------------------------ 6 小结 */
{
  kicker: 'L6-03 · 卷积执行',
  title: '小结：<span class="hl-a">五个文件</span>与三个技巧',
  sub: '前四个验证 L5-06，第五个验证 L4-09。',
  caption: '一句话：<b>卷积的期望值难手算，所以用串行版现算</b>。',
  code: `// 【族谱】5 个文件
//   文件                              验证什么                    对应课
//   shard_batch                       批维分片（无通信）           L5-06
//   shard_batch_group                 batch_group_count 改批维     L5-06
//   shard_contracting_dim             收缩维分片需 all_reduce      L5-06
//   shard_feature_group               feature_group_count + 特征维 L5-06
//   dual_semantics_factor_indivisible 【REPL vs HALO 等价】        L4-09

// 【★ 三个测试设计技巧】
//   ① 用 @sequential_conv 【现算期望值】
//      卷积的期望值难手算 -> 直接调串行版算
//   ② 【手动切分输入】
//      验证"分片 + 拼接"的全过程
//   ③ 【缩放】让贡献可辨识
//      提高测试敏感度，让"漏掉一个"必然出错
//   -> 这三个技巧在后面几课也会反复出现

// 【★ dual_semantics 的两个 RUN 行】
//   // RUN: ... %t --enable_halo_exchange=true
//   // RUN: ... %t --enable_halo_exchange=false
//   验证 L4-09 的 enable-halo-exchange 选项
//   -> 证明 REPL（全复制）与 HALO（halo exchange）【语义等价】
//   注意它是 f32（其他 4 个是 i32）—— 因为 halo 涉及浮点运算
//   而且它在【另一个目录】（executable_partitioner_pipeline/）
//     跑的是【完整分区器流水线】（L6-00 讲的第二个脚本）

// 【L6 的进度】
//   L6-00 可执行测试机制（已做）
//   L6-01 sdy.* 集合通信（已做）
//   L6-02 stablehlo 集合通信（已做）
//   L6-03 卷积（本课）
//   L6-04 矩阵乘/fft/iota     L6-05 gather（★最复杂）
//   L6-06 pad（★最大族）      L6-07 reshape
//   L6-08 reverse/slice       L6-09 scatter 与杂项

// 一句话总结：
//   L6-03 用真实数值验证 L5-06 的卷积分片规则
//   四种分片维度各一个文件，外加一个验证 REPL/HALO 等价的流水线测试
//   三个测试设计技巧（现算期望值 / 手动切分 / 缩放）值得单独记住`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: 'shard_batch', n: 1, c: '#38bdf8' }, { t: 'batch_group', n: 1, c: '#4ade80' },
      { t: 'contracting_dim', n: 1, c: '#fb7185' }, { t: 'feature_group', n: 1, c: '#fbbf24' },
      { t: 'dual_semantics', n: 1, c: '#c084fc' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:140px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:9px;color:${f.c};overflow-wrap:anywhere">${f.t}</div>
        <div class="big" style="font-size:16px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>5 个文件</b> —— 前四个对应 L5-06 的四种分片维度。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>★ 三个技巧</b>：现算期望值（卷积难手算）/ 手动切分（验证全过程）/ 缩放（提高敏感度）。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>第五个文件特殊</b>：在另一个目录、跑完整流水线、两个 RUN 行验证 <b>REPL vs HALO 等价</b>。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L6-04 讲矩阵乘 / fft / iota 的执行测试。';
    });
  }
},

];
