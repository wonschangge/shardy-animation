/* ==========================================================================
   L6-02 · exec-stablehlo-collectives
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/executable_convert_global_to_local/ 下 5 个文件
         (stablehlo_all_to_all 60, _all_reduce 51, _all_gather 48,
          _collective_permute 45, _reduce_scatter) —— 手写 stablehlo 通信的执行测试
   目标：讲透"直接写 stablehlo 集合通信"时的验证方式，并与 L6-01 对照。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 与 L6-01 对照 */
{
  kicker: 'L6-02 · stablehlo 通信执行',
  title: '★ 与 L6-01 的<span class="hl-a">关键差异</span>',
  sub: '同样验证集合通信，但 part1 里写的东西**完全不同**。',
  caption: '两课合起来证明：<b>L5-02 的降级是语义保持的</b>。',
  code: `// 【同样验证集合通信，但 part1 写的东西不同】
//              L6-01（sdy_*）            L6-02（stablehlo_*）
//   part1 写的  【sdy.all_reduce】       【manual_computation
//                                          + stablehlo.all_reduce】
//   抽象层次    【高层】（SDY 算子）     【低层】（直接写硬件通信）
//   replica_    由降级【生成】           【手写】
//     groups
//   对应课      L5-02（降级）            L4-11（逐指令分区）

// 【为什么两者都需要】
//   L6-01 验证"SDY 算子 -> 降级 -> 执行"这条路径
//   L6-02 验证"手写底层通信 -> 执行"这条路径
//   两者【语义应该等价】
//     —— 因为 L5-02 的降级产物【就是】 stablehlo.*

// 【★ 本课的核心价值】
//   同一个 all_reduce，两条路径的结果【都是 1111】
//   -> 证明 L5-02 的降级【语义保持】

// 【5 个文件】stablehlo_all_to_all 60 行 / _all_reduce 51 / _all_gather 48
//              / _collective_permute 45 / _reduce_scatter
// 网格是【单轴】的：sdy.mesh @mesh_4 = <["x"=4]>
//   -> replica_groups 都是 [[0, 1, 2, 3]]（全部一组，无需分组）`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'L6-01', c: '#38bdf8', tag: 'sdy_*',
        d: 'part1 写 <b><span class="mono">sdy.all_reduce</span></b><br>高层抽象<br><span class="dim">replica_groups 由降级生成</span>' },
      { t: 'L6-02（本课）', c: '#4ade80', tag: 'stablehlo_*',
        d: 'part1 写 <b><span class="mono">manual_computation</span></b><br>+ <span class="mono">stablehlo.all_reduce</span><br><span class="dim">replica_groups 手写</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:11.5px">${x.t} <span class="faint small">${x.tag}</span></div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>L6-01</b> 验证的是"<b>SDY 算子</b>"这条路径 —— 高层、简洁。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>本课</b> 验证"<b>手写底层通信</b>"这条路径 —— 低层、显式。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>两者语义应该等价</b> —— 因为 L5-02 的降级产物<b>就是</b> <span class="mono">stablehlo.*</span>。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>★ 核心价值</b>：同一个 <span class="mono">all_reduce</span>，两条路径的结果<b>都是 1111</b> → 降级语义保持 ✓';
    });
  }
},

/* ------------------------------------------------ 2 ★ 共同结构 */
{
  kicker: 'L6-02 · stablehlo 通信执行',
  title: '★ 共同结构：<span class="mono hl-a">manual_computation</span> 包裹',
  sub: '这正是 **L4-11 讲的"逐指令分区"形态** —— 区域内自己管分片。',
  caption: '5 个文件的 part1 <b>结构完全一致</b>，只是里面的通信算子不同。',
  code: `func.func @manual_all_reduce(
  %arg0: tensor<16x8xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>})
  -> (tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {}]>}) {
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_4, [{"x"}, {}]>]
    out_shardings=[<@mesh_4, [{}, {}]>]
    manual_axes={"x"} (%arg1: tensor<4x8xi32>) {
      %1 = "stablehlo.all_reduce"(%arg1) ({
        ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
          %2 = stablehlo.add %arg2, %arg3 : tensor<i32>
          stablehlo.return %2 : tensor<i32>
      }) {
        replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
        channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
        use_global_device_ids
      } : (tensor<4x8xi32>) -> tensor<4x8xi32>
      sdy.return %1 : tensor<4x8xi32>
  } : (tensor<16x8xi32>) -> (tensor<4x8xi32>)
  return %0 : tensor<4x8xi32>
}

// 【★ 这正是 L4-11 讲的"逐指令分区"形态】
//   sdy.manual_computation 包裹  -> 区域内【自己管分片】（L1-07）
//   manual_axes={"x"}            -> x 轴【冻结】，区域内不再分片
//   区域内【直接放 stablehlo.all_reduce】
//     -> 因为分片已经"手动处理"了
//
// 【类型读法】
//   in_shardings / out_shardings 是【全局分片】
//   区域内 %arg1 是【局部形状】：16x8 沿 x=4 切 -> 4x8 ✓
//
// 【replica_groups = dense<[[0, 1, 2, 3]]>】
//   网格是【单轴】@mesh_4 -> 所有 4 台一组，无需分组
//   （对比 L6-01 的 2x2 mesh：跨 x 时要分成 {0,2}/{1,3}）
//
// 【channel_handle = <handle = 1, type = 0>】
//   注意 type = 【0】—— 与 L5-02 降级产物（type = 1）【不同】
//   这是【手写】的特征（降级生成的用 type = 1）
//
// 【5 个文件的差异只在于区域内放哪个算子】
//   stablehlo.all_reduce / all_gather / all_to_all /
//   collective_permute / reduce_scatter`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'manual_computation', c: '#38bdf8', d: '包裹整个区域<br><b>自己管分片</b><br><span class="dim">L1-07 讲过</span>' },
      { t: 'manual_axes={"x"}', c: '#4ade80', d: '<b>冻结 x 轴</b><br>区域内不再分片' },
      { t: '区域内局部形状', c: '#fbbf24', d: '<span class="mono">16x8</span> → <span class="mono">4x8</span><br>（沿 <span class="mono">x=4</span> 切）' },
      { t: '手写 replica_groups', c: '#c084fc', d: '<span class="mono">[[0,1,2,3]]</span><br>单轴网格<br>全部一组' },
      { t: 'type = 0', c: '#fb7185', d: '与降级产物<br>（<span class="mono">type = 1</span>）<b>不同</b><br><span class="dim">手写的特征</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:140px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:9.5px;color:${x.c};overflow-wrap:anywhere">${x.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>这就是 L4-11 的形态</b>：区域内的分片由<b>自己</b>负责，不靠传播。',
        '冻结 <span class="mono">x</span> 轴 → 区域内每台设备只处理自己那部分。',
        '<b>类型读法</b>：<span class="mono">in_shardings</span> 是全局的，区域内 <span class="mono">%arg1</span> 是局部的。',
        '<b>单轴网格</b>所以不用分组 —— 对比 L6-01 的 2x2 mesh 要分成两组。',
        '<b>一个可辨识的细节</b>：<span class="mono">type = 0</span> 说明这是<b>手写</b>的（降级生成用 <span class="mono">type = 1</span>）。',
      ][i];
    }));
    tl.at(16400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>5 个文件的差异</b>只在于区域内放哪个算子 —— 结构完全一致。';
    });
  }
},

/* ------------------------------------------------ 3 all_reduce 与 all_gather */
{
  kicker: 'L6-02 · stablehlo 通信执行',
  title: '<span class="mono hl-a">all_reduce</span> 与 <span class="mono hl-a">all_gather</span>：期望值的两种写法',
  sub: '一个**硬编码常量**、一个用 `concatenate` **现算** —— 写法取决于算子语义。',
  caption: '这是一个值得学的<b>测试技巧</b>：期望值怎么算最不容易写错。',
  code: `// 【all_reduce：期望值【硬编码】】
func.func @main() {
  // Create distinct 4x8 local shards for the 4 virtual devices.
  %s0 = stablehlo.constant dense<1> : tensor<4x8xi32>
  %s1 = stablehlo.constant dense<10> : tensor<4x8xi32>
  %s2 = stablehlo.constant dense<100> : tensor<4x8xi32>
  %s3 = stablehlo.constant dense<1000> : tensor<4x8xi32>

  // Expected global result for a sum reduction: 1 + 10 + 100 + 1000 = 1111.
  %expected = stablehlo.constant dense<1111> : tensor<4x8xi32>

  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@manual_all_reduce, @manual_all_reduce, @manual_all_reduce, @manual_all_reduce]]
  } : (tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>) ->
      (tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>)

  "check.expect_eq"(%res#0, %expected) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res#1, %expected) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res#2, %expected) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res#3, %expected) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  return
}
// 【读法】与 L6-01 的 all_reduce 用例【完全同构】
//   4 台各拿 1/10/100/1000 -> replica_groups [[0,1,2,3]] -> 全部一组
//   1 + 10 + 100 + 1000 = 1111 -> 4 台【都得到 1111】✓
// 【与 L6-01 对比】
//   L6-01 part1 写 sdy.all_reduce {"x"}    期望值 dense<1111>
//   本课  part1 写 stablehlo.all_reduce   期望值 dense<1111>
//   -> 两者【结果相同】-> 降级语义保持 ✓

// 【all_gather：期望值【现算】】
func.func @main() {
  // Create distinct 4x8 local shards for the 4 devices.
  %s0 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
  %s1 = stablehlo.constant dense<10.0> : tensor<4x8xf32>
  %s2 = stablehlo.constant dense<100.0> : tensor<4x8xf32>
  %s3 = stablehlo.constant dense<1000.0> : tensor<4x8xf32>

  // Create the expected global tensor for verification.
  %input = "stablehlo.concatenate"(%s0, %s1, %s2, %s3) {dimension = 0 : i64}
    : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<16x8xf32>
  ...
  "check.expect_eq"(%res#0, %input) : (tensor<16x8xf32>, tensor<16x8xf32>) -> ()
// 【读法】期望值【不是硬编码】，而是用 stablehlo.concatenate 【现算】
//   因为 all_gather 的语义就是"把各设备的 shard 沿某维【拼接】"
//   -> 用 concatenate 表达期望值【最直接、最不容易写错】
// 【对比 all_reduce】
//   那里期望值是【硬编码的 1111】—— 因为"求和"的结果
//   无法用更简单的算子表达（除非也写个 reduce）

// 【★ 期望值的写法取决于算子语义】
//   算子                  期望值写法
//   all_reduce            【硬编码常量】（1111）—— 求和结果最简单
//   all_gather            【concatenate 现算】—— 语义就是拼接
//   all_to_all            【concatenate 现算】（用置换后的输入）`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'all_reduce', c: '#4ade80', d: '期望值<b>硬编码</b><br><span class="mono">dense&lt;1111&gt;</span><br><span class="dim">求和结果最简单</span>' },
      { t: 'all_gather', c: '#38bdf8', d: '期望值<b>现算</b><br><span class="mono">concatenate(s0..s3)</span><br><span class="dim">语义就是拼接</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>与 L6-01 完全同构</b>：4 台各拿 1/10/100/1000 → 都得到 1111。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>现算的好处</b>：<span class="mono">concatenate</span> 就是 <span class="mono">all_gather</span> 的语义 —— <b>最不容易写错</b>。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 期望值写法取决于算子语义</b>：求和 → 硬编码；拼接 → 现算。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>★ 核心验证</b>：L6-01 与 L6-02 的 <span class="mono">all_reduce</span> <b>结果相同</b> → L5-02 的降级<b>语义保持</b> ✓';
    });
  }
},

/* ------------------------------------------------ 4 ★ collective_permute */
{
  kicker: 'L6-02 · stablehlo 通信执行',
  title: '★ <span class="mono hl-a">collective_permute</span>：循环置换',
  sub: '`source_target_pairs` 是**收发对照表** —— 读它的关键是注意**方向**。',
  caption: '本课<b>最清晰</b>的一个例子：4 台设备的循环置换，每台结果都不同。',
  code: `      %1 = "stablehlo.collective_permute"(%arg1) {
        source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], [3, 0]]> : tensor<4x2xi64>,

// part2.mlir 的断言：
"check.expect_eq"(%res#0, %s3) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
"check.expect_eq"(%res#1, %s0) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
"check.expect_eq"(%res#2, %s1) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
"check.expect_eq"(%res#3, %s2) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()

// 【读法】source_target_pairs = [[0,1], [1,2], [2,3], [3,0]]
//   这是一个【循环置换】：
//     设备 0 -> 1、1 -> 2、2 -> 3、3 -> 【0】（回到起点）
//
// 【★ 逐设备推结果】（注意方向！）
//   设备    收到谁的          期望值
//   0       设备【3】发的      %s3
//   1       设备【0】发的      %s0
//   2       设备【1】发的      %s1
//   3       设备【2】发的      %s2
//   ✓ 与断言完全一致
//
// 【★ 这是"收发对照表"语义的最清晰展示】
//   [0, 1] 读作"【设备 0 发给设备 1】"
//     -> 所以【设备 1 收到 %s0】
//   断言写的是【接收方】的期望值 —— 注意【方向】
//
// 【回顾 L5-02】
//   那里看到 collective_permute 的 source_target_pairs
//   且 [1,4] / [4,1] 【成对出现】（交换语义）
//   本课的 [[0,1],[1,2],[2,3],[3,0]] 是【循环】（不全是成对的）
//   -> 说明 source_target_pairs 【不要求成对】，任意映射都行
//
// 【结果分布】每台设备【都不同】
//   —— 与 all_reduce 的"所有设备相同"形成鲜明对比（L6-01 讲过）`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const pairs = [
      { p: '[0, 1]', r: '设备 1 收到 %s0', c: '#38bdf8' },
      { p: '[1, 2]', r: '设备 2 收到 %s1', c: '#4ade80' },
      { p: '[2, 3]', r: '设备 3 收到 %s2', c: '#fbbf24' },
      { p: '[3, 0]', r: '设备 0 收到 %s3', c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = pairs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:11px;color:${x.c}">${x.p}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4;margin-top:3px">${x.r}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    pairs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<span class="mono">[0, 1]</span> 读作"<b>设备 0 发给设备 1</b>" → 所以设备 <b>1</b> 收到 <span class="mono">%s0</span>。',
        '依此类推 —— 关键是注意<b>方向</b>：断言写的是<b>接收方</b>的期望值。',
        '<span class="mono">[2, 3]</span> → 设备 3 收到 <span class="mono">%s2</span>。',
        '<span class="mono">[3, 0]</span> 让循环<b>闭合</b> —— 设备 0 收到设备 3 的 <span class="mono">%s3</span>。',
      ][i];
    }));
    tl.at(14400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>对比 L5-02</b>：那里 <span class="mono">[1,4]</span>/<span class="mono">[4,1]</span> <b>成对出现</b>（交换）；这里是<b>循环</b> → 说明不要求成对。';
    });
  }
},

/* ------------------------------------------------ 5 reduce_scatter 与族谱 */
{
  kicker: 'L6-02 · stablehlo 通信执行',
  title: '<span class="mono hl-a">reduce_scatter</span> 与 5 个文件的族谱',
  sub: '`reduce_scatter` 的区域内输入是**完整**的 —— 与其他 4 个文件不同。',
  caption: '这个差异揭示了 <span class="mono">in_shardings</span> 与区域内容的关系。',
  code: `func.func @manual_reduce_scatter(
    manual_axes={"x"} (%arg1: tensor<16x8xi32>) {     // <- 【完整大小】！
      %1 = "stablehlo.reduce_scatter"(%arg1) ({
        replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,

// part2.mlir：
%expected = stablehlo.constant dense<1111> : tensor<4x8xi32>
"check.expect_eq"(%res#0, %expected) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()

// 【★ 与其他 4 个文件的关键区别】
//   区域内 %arg1 的类型是 tensor<16x8xi32> —— 【完整大小】！
//   而 all_reduce 用例里区域内是 tensor<4x8xi32>（局部大小）
//
// 【为什么】
//   reduce_scatter 的 in_shardings 是 [{}, {}]（【全复制】）
//   -> 区域内每台设备都有【完整数据】
//   -> 然后 reduce_scatter 在区域内做"归约 + 切分"
//
// 【结果】期望值 1111 —— 归约后每台得到完整的和
//   ★ 这验证了 L5-02 讲的 reduce_scatter 语义："边归约边切分"
//
// 【★ 一个规律】
//   区域内 %arg1 的类型【取决于 in_shardings】：
//     in_shardings = [{"x"}, {}]  -> 局部 4x8（已分片）
//     in_shardings = [{}, {}]     -> 完整 16x8（全复制）
//   -> 读这类测试时，先看 in_shardings 就能知道区域内的形状

// 【族谱】5 个文件
//   文件                          通信算子              期望值写法        结果分布
//   stablehlo_all_reduce          all_reduce            硬编码 1111       所有设备相同
//   stablehlo_all_gather          all_gather            concatenate 现算  都得到完整拼接
//   stablehlo_all_to_all          all_to_all            concatenate 现算  置换后的拼接
//   stablehlo_collective_permute  collective_permute    各设备的输入      每台不同（循环）
//   stablehlo_reduce_scatter      reduce_scatter        硬编码 1111       都得到完整的和

// 【★ 与 L6-01 的总结对照】
//              L6-01（5 个 sdy_*）        L6-02（5 个 stablehlo_*）
//   抽象       【高层 SDY 算子】           【低层 stablehlo 算子】
//   容器       无（直接写）               【manual_computation】
//   分片处理   由 sdy.* 算子隐含          【manual_axes 显式冻结】
//   replica_   降级生成                   【手写】
//     groups
//   验证的路径 L5-02 的【降级】           【手写底层通信】
//   -> 两者语义等价 = L5-02 降级的正确性依据

// 一句话总结：
//   L6-01 验证"SDY 算子 -> 降级 -> 执行"
//   L6-02 验证"手写 stablehlo 通信 -> 执行"
//   两者结果相同，证明 L5-02 的降级是【语义保持】的`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: 'all_reduce', n: 1, c: '#4ade80' }, { t: 'all_gather', n: 1, c: '#38bdf8' },
      { t: 'all_to_all', n: 1, c: '#0ea5e9' }, { t: 'collective_permute', n: 1, c: '#fbbf24' },
      { t: 'reduce_scatter', n: 1, c: '#c084fc' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:140px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:9px;color:${f.c};overflow-wrap:anywhere">${f.t}</div>
        <div class="big" style="font-size:16px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>5 个文件各一个算子</b> —— 结构完全一致，只是区域内的算子不同。'; });
    tl.at(4200, () => {
      els.forEach((e, i) => { if (i !== 4) e.style.opacity = '.25'; });
      msg.innerHTML = '<b>reduce_scatter 特殊</b>：区域内输入是<b>完整大小</b>（<span class="mono">16x8</span>）—— 因为 <span class="mono">in_shardings</span> 是全复制。';
    });
    tl.at(7800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 一个规律</b>：区域内形状取决于 <span class="mono">in_shardings</span> —— 先看它就能知道区域内的类型。';
    });
    tl.at(11200, () => {
      msg.innerHTML = '<b>两课总结</b>：L6-01 验证降级路径，本课验证手写路径 —— <b>结果相同 = 降级语义保持</b>。';
    });
  }
},

/* ------------------------------------------------ 6 小结 */
{
  kicker: 'L6-02 · stablehlo 通信执行',
  title: '小结：<span class="hl-a">两条路径，同一个结果</span>',
  sub: '本课与 L6-01 合起来，构成了对 L5-02 降级的**完整验证**。',
  caption: '一句话：<b>L6-01 验证降级，L6-02 验证手写 —— 两者等价</b>。',
  code: `// 【本课的核心结论】
//   L6-01 验证"SDY 算子 -> 降级 -> 执行"
//   L6-02 验证"手写 stablehlo 通信 -> 执行"
//   两者结果【相同】-> 证明 L5-02 的降级是【语义保持】的

// 【三条可迁移的知识】
//   ① manual_computation 包裹 = L4-11 的"逐指令分区"形态
//      manual_axes 冻结轴 -> 区域内自己管分片
//   ② 期望值的写法取决于算子语义
//      求和 -> 硬编码；拼接 -> concatenate 现算；置换 -> 用各设备的输入
//   ③ 区域内形状取决于 in_shardings
//      [{"x"}, {}] -> 局部 4x8；[{}, {}] -> 完整 16x8

// 【一个可辨识的细节】
//   channel_handle 的 type：本课手写的是 type = 0
//   L5-02 降级生成的是 type = 1
//   -> 可以用来区分"手写"与"生成"的 IR

// 【L6 的进度】
//   L6-00 可执行测试机制（已做）
//   L6-01 sdy.* 集合通信的执行（已做）
//   L6-02 stablehlo 集合通信的执行（本课）
//   L6-03 卷积                  L6-04 矩阵乘/fft/iota
//   L6-05 gather（★最复杂）     L6-06 pad（★最大族）
//   L6-07 reshape               L6-08 reverse/slice
//   L6-09 scatter 与杂项

// 【后续课的模式】
//   它们都是"某个算子的 executable 测试"
//   读法都是：先看 part1 的算子与分片，再看 part2 的期望值
//   理解了 L6-00 的机制 + 本课的两条路径，后续课就只是"换个算子看结果"

// 一句话总结：
//   L6-01 与 L6-02 是同一件事的两条路径
//   前者走 SDY 算子的降级，后者直接手写 stablehlo
//   两者结果相同 —— 这就是降级正确性的【经验证据】`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① manual 包裹', c: '#38bdf8', d: 'L4-11 的<br>逐指令分区形态' },
      { t: '② 期望值写法', c: '#4ade80', d: '取决于<br>算子语义' },
      { t: '③ 区域内形状', c: '#fbbf24', d: '取决于<br><span class="mono">in_shardings</span>' },
      { t: 'type = 0', c: '#fb7185', d: '区分手写<br>与生成' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div style="font-size:11px;color:${x.c}">${x.t}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<span class="mono">manual_computation</span> + <span class="mono">manual_axes</span> —— 区域内自己管分片。',
        '求和 → 硬编码；拼接 → <span class="mono">concatenate</span> 现算；置换 → 用各设备的输入。',
        '<span class="mono">[{"x"}, {}]</span> → 局部 <span class="mono">4x8</span>；<span class="mono">[{}, {}]</span> → 完整 <span class="mono">16x8</span>。',
        '<span class="mono">channel_handle</span> 的 <span class="mono">type</span>：手写是 0、生成是 1。',
      ][i];
    }));
    tl.at(15200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>下一课</b> L6-03 讲卷积的执行测试。';
    });
  }
},

];
