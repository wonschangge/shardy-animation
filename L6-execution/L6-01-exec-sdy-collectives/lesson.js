/* ==========================================================================
   L6-01 · exec-sdy-collectives   （P0）
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/executable_convert_global_to_local/ 下 10 个文件
         (590 行) —— sdy_* 集合通信的执行测试
   目标：用 L6-00 的机制验证 L5-02/L5-03 的推断。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 测试结构 */
{
  kicker: 'L6-01 · 集合通信执行',
  title: '★ 可执行测试的<span class="hl-a">结构</span>：每台设备给不同输入',
  sub: '`interpreter.run_parallel` 在 4 台设备上并行运行，**每台设备的输入不同**。',
  caption: '本课用 L6-00 讲的机制，验证 <b>L5-02 / L5-03</b> 从 IR 推断出的规则。',
  code: `// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir
sdy.mesh @mesh_2_2 = <["x"=2, "y"=2]>      // 4 台设备

// All-reduce across the entire mesh (both "x" and "y" axes).
func.func @all_reduce_xy(...) -> ... {
  %0 = sdy.all_reduce {"x", "y"} %arg0 out_sharding=<@mesh_2_2, [{}, {}]> : ...
}

// All-reduce across only the "x" axis.
// In a 2x2 mesh, this creates two replica groups: {0, 2} and {1, 3}.
func.func @all_reduce_x(...) -> ... {
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh_2_2, [{}, {}]> : ...
}

//--- part2.mlir
func.func @main() {
  %c1    = stablehlo.constant dense<1>    : tensor<4x4xi32>
  %c10   = stablehlo.constant dense<10>   : tensor<4x4xi32>
  %c100  = stablehlo.constant dense<100>  : tensor<4x4xi32>
  %c1000 = stablehlo.constant dense<1000> : tensor<4x4xi32>

  // Sum = 1 + 10 + 100 + 1000 = 1111.
  %res_xy:4 = "interpreter.run_parallel"(%c1, %c10, %c100, %c1000) {
    programs = [[@all_reduce_xy, @all_reduce_xy, @all_reduce_xy, @all_reduce_xy]]
  } : (...) -> (...)

  %expected_xy = stablehlo.constant dense<1111> : tensor<4x4xi32>
  "check.expect_eq"(%res_xy#0, %expected_xy) : (...) -> ()
  "check.expect_eq"(%res_xy#3, %expected_xy) : (...) -> ()
  return
}

// 【★ 核心机制】
//   interpreter.run_parallel  在【4 台设备上并行运行】
//     programs = [[...]]      每台设备运行哪个函数
//     四个操作数就是【四台设备的输入】：
//       设备 0 拿 1、设备 1 拿 10、设备 2 拿 100、设备 3 拿 1000
//   check.expect_eq           断言结果等于期望值
//
// 【★ 为什么"每台设备给不同输入"很关键】
//   如果所有设备输入相同，就无法验证"数据分布"是否正确
//   给了不同的输入，才能看出：
//     哪些设备的结果【应该相同】（如 all_reduce）
//     哪些设备的结果【应该不同】（如 collective_permute）
//
// 【数值验证】1 + 10 + 100 + 1000 = 1111
//   所有设备都得到 1111 —— 因为 all_reduce 让每台都有完整的和 ✓`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'part1.mlir', c: '#38bdf8', d: '分片函数<br><b>被测对象</b>' },
      { t: 'run_parallel', c: '#4ade80', d: '4 台设备<b>并行运行</b><br>每台输入不同' },
      { t: 'expect_eq', c: '#fbbf24', d: '<b>断言</b>结果<br>等于期望值' },
      { t: '串行参考版', c: '#c084fc', d: '脚本自动生成<br><span class="dim">L6-00 讲过</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div style="font-size:11.5px;color:${x.c}">${x.t}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<span class="mono">part1.mlir</span> 里是<b>带分片</b>的函数 —— 它们会被导出流水线处理成局部代码。',
        '<b>关键</b>：四个操作数就是四台设备的输入（<span class="mono">1/10/100/1000</span>）。',
        '用 <span class="mono">check.expect_eq</span> 断言 —— 数值不对就会失败。',
        '分片版与串行版的结果<b>都要正确</b> —— 这是 L6-00 讲的验证思路。',
      ][i];
    }));
    tl.at(15200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>为什么给不同输入</b>：才能看出哪些设备的结果<b>应该相同</b>、哪些<b>应该不同</b>。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 逐设备数值 */
{
  kicker: 'L6-01 · 集合通信执行',
  title: '★ 逐设备数值：<span class="hl-a">哪台设备算出什么</span>',
  sub: '同一个 `all_reduce`，跨**两个轴**和跨**一个轴**的结果完全不同。',
  caption: '这是 TODOLIST 要的"逐设备数值动画" —— 数值<b>完全可预测</b>。',
  code: `// 【只跨 "x" 轴的情形】
// Device grid (x, y):
// (0,0): dev 0, input 1
// (0,1): dev 1, input 10
// (1,0): dev 2, input 100
// (1,1): dev 3, input 1000
// Replica groups for "x" (dim 0): {0, 2} and {1, 3}.
%res_x:4 = "interpreter.run_parallel"(%c1, %c10, %c100, %c1000) {
  programs = [[@all_reduce_x, @all_reduce_x, @all_reduce_x, @all_reduce_x]]
} : (...) -> (...)

%expected_x_02 = stablehlo.constant dense<101>  : tensor<4x4xi32> // 1 + 100
%expected_x_13 = stablehlo.constant dense<1010> : tensor<4x4xi32> // 10 + 1000

"check.expect_eq"(%res_x#0, %expected_x_02) : (...) -> ()
"check.expect_eq"(%res_x#2, %expected_x_02) : (...) -> ()
"check.expect_eq"(%res_x#1, %expected_x_13) : (...) -> ()
"check.expect_eq"(%res_x#3, %expected_x_13) : (...) -> ()

// 【★ 逐设备读】
//   设备  (x,y)    输入    归约组    结果
//   0     (0,0)    1       {0,2}     101  = 1 + 100
//   1     (0,1)    10      {1,3}     1010 = 10 + 1000
//   2     (1,0)    100     {0,2}     101
//   3     (1,1)    1000    {1,3}     1010

// 【★ 为什么分组是 {0,2} 与 {1,3}】
//   跨 x 归约 -> 【同一个 y 值】的设备归为一组
//     y=0 的设备是 0 和 2  -> {0, 2}
//     y=1 的设备是 1 和 3  -> {1, 3}
//   ✓ 与注释一致

// 【对比：跨两轴】
//   %expected_xy = dense<1111>    // 1 + 10 + 100 + 1000
//   -> 所有设备都得到 1111

// 【★ 设备号与网格坐标的对应】（注释直接给出）
//   设备 0 = (x=0, y=0)    设备 1 = (x=0, y=1)
//   设备 2 = (x=1, y=0)    设备 3 = (x=1, y=1)
//   即 x 【变化最慢】-> x 是最 major 的轴 ✓
//
// ★ 这【直接验证】了 L5-01 的推导！
//   L5-01 里我从查找表 [0,0,0,0,2,2,2,2] 【反推】出
//     "设备号按轴序【最 major 优先】"
//   这里测试文件的注释【直接确认】了这个规则
//   （设备 0、1 的 x=0 -> x 变化最慢）`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '跨两轴', c: '#4ade80', d: '1+10+100+1000<br>= <b>1111</b><br><b>所有设备相同</b>' },
      { t: '跨 x 轴', c: '#fbbf24', d: '分两组 <span class="mono">{0,2}</span> / <span class="mono">{1,3}</span><br>设备 0,2 → <b>101</b><br>设备 1,3 → <b>1010</b>' },
      { t: '设备号规则', c: '#38bdf8', d: '设备 0,1 的 <span class="mono">x=0</span><br>→ <b>x 变化最慢</b><br><span class="dim">验证 L5-01 的推导</span>' },
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
        '跨两轴 = 全部 4 台一起归约 → 每台都得到<b>完整的和</b>。',
        '跨 x 轴 = <b>同一个 y 值</b>的设备归为一组 → 两组各算各的。',
        '<b>★ 关键证据</b>：注释说设备 0、1 的 <span class="mono">x=0</span> → <span class="mono">x</span> 变化最慢 → <b>最 major</b>。',
      ][i];
    }));
    tl.at(15200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>这就是 L5-01 那条规则</b>：设备号按轴序<b>最 major 优先</b>展开 —— 这里得到了直接证据。';
    });
  }
},

/* ------------------------------------------------ 3 ★ 结果分布 */
{
  kicker: 'L6-01 · 集合通信执行',
  title: '★ 各算子的<span class="hl-a">结果分布</span>对比',
  sub: '不同算子的"结果分布方式"不同 —— 这是判断"验证方式对不对"的依据。',
  caption: '理解分布规律，才能判断一个测试的断言写得对不对。',
  code: `// 【四种分布方式】
//   算子                  结果分布                        例子
//   all_reduce            【所有设备相同】                都得到 1111
//   collective_permute    【每台设备不同】                %e0~%e3 各不同
//   all_gather            每台得到【拼接后的完整数据】
//   all_slice             每台得到【自己那片】

// 【collective_permute：每台设备结果不同】
// sdy_collective_permute_with_self_loops.mlir 的 part2.mlir：
"check.expect_eq"(%res#0, %e0) : (tensor<1xi32>, tensor<1xi32>) -> ()
"check.expect_eq"(%res#1, %e1) : (tensor<1xi32>, tensor<1xi32>) -> ()
"check.expect_eq"(%res#2, %e2) : (tensor<1xi32>, tensor<1xi32>) -> ()
"check.expect_eq"(%res#3, %e3) : (tensor<1xi32>, tensor<1xi32>) -> ()
// 【读法】四台设备各有【不同】的期望值
//   因为 collective_permute 是"设备间【交换数据】"的操作
//   每台设备拿到的是【别人发来的】数据
//   -> 与 all_reduce 的"所有设备相同"形成鲜明对比

// 【三个 collective_permute 文件的区分】
//   ..._with_self_loops        置换表里有"发给自己"的项
//   ..._without_self_loops     没有"发给自己"的项
//   ..._cross_replica          跨副本的置换
// 为什么有 3 个文件：source_target_pairs 的每一种形态都要验证
//
// 【回顾 L5-02】
//   那里看到 source_target_pairs 是一张【设备收发对照表】
//   且 [1,4] / [4,1] 【成对出现】（交换语义）
//   本课验证了它的【数值效果】

// 【all_gather 的特殊断言】
// sdy_all_gather.mlir 的 part2.mlir：
"check.expect_eq"(%res_comb#0, %res_pdim#0) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()
"check.expect_eq"(%res_comb#0, %cst) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()
// 【读法】
//   %res_comb  = 【合并模式】（combined，默认）
//   %res_pdim  = 【逐维模式】（per-dim-all-gather=true）
//   断言【两者结果相同】，且都等于期望常量 %cst
// ★ 这验证了 L5-02 讲的 per-dim-all-gather 选项
//   那里有【四个 RUN 行】（enable-rgv3 x per-dim-all-gather 的 2x2 组合）
//   本课证明：两种模式【语义等价】—— 只是 IR 形态不同`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'all_reduce', c: '#4ade80', d: '<b>所有设备相同</b><br>都得到完整的和' },
      { t: 'collective_permute', c: '#fbbf24', d: '<b>每台设备不同</b><br>拿到别人发来的数据' },
      { t: 'all_gather', c: '#38bdf8', d: '每台得到<br>拼接后的完整数据' },
      { t: 'all_slice', c: '#c084fc', d: '每台得到<br>自己那片' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:10.5px;color:${x.c};overflow-wrap:anywhere">${x.t}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>为什么所有设备相同</b>：<span class="mono">all_reduce</span> 让每台都有<b>完整的和</b>。',
        '<b>为什么每台不同</b>：<span class="mono">permute</span> 是<b>设备间交换数据</b> —— 每台拿到别人发来的。',
        '<b>all_gather</b>：每台都得到<b>所有设备的数据拼接</b>。',
        '<b>all_slice</b>：每台只留<b>自己那一片</b>。',
      ][i];
    }));
    tl.at(15200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>理解分布规律</b>才能判断一个测试的断言写得对不对 —— 这是本课的实际价值。';
    });
  }
},

/* ------------------------------------------------ 4 ★ splat vs dense */
{
  kicker: 'L6-01 · 集合通信执行',
  title: '★ <span class="mono hl-a">splat</span> vs <span class="mono">dense</span>：L5-03 洞察的<span class="hl-a">数值证据</span>',
  sub: 'dense 常量的两台设备期望值**不同**；splat 的**相同**。',
  caption: '这是本课最有价值的一处 —— <b>L5-03 的推断在这里得到直接验证</b>。',
  code: `// sdy_constant.mlir
func.func @sharded_dense_constant()
  %0 = sdy.constant {...<@mesh_2, [{"x"}, {}]>} dense<[...]>   // 元素各不相同
func.func @sharded_splat_constant()
  %0 = sdy.constant {...<@mesh_2, [{"x"}, {}]>} dense<7> : tensor<4x4xi32>  // 全是 7

// part2.mlir 的期望值：
%e_dense_0 = stablehlo.constant dense<[...]>      // 设备 0 的期望
%e_dense_1 = stablehlo.constant dense<[...]>      // 设备 1 的期望
"check.expect_eq"(%res_dense#0, %e_dense_0) : (tensor<2x4xi32>, tensor<2x4xi32>) -> ()
"check.expect_eq"(%res_dense#1, %e_dense_1) : (tensor<2x4xi32>, tensor<2x4xi32>) -> ()
%e_splat = stablehlo.constant dense<7> : tensor<2x4xi32>   // 两台设备【共用】一个期望值

// 【★ 逐项对比】这直接验证了 L5-03 的核心洞察
//   常量类型              每台设备的期望值
//   dense（元素各不同）   %e_dense_0 与 %e_dense_1 —— 【两台不同】！
//   splat（dense<7>）     %e_splat = dense<7> —— 【两台相同】

// 【读法】
//   dense 常量：每台设备得到【不同的切片】
//     -> 必须用 replica_id + 切片（L5-01 讲的六步）
//   splat 常量：每台设备得到【相同的值】（都是 dense<7>）
//     -> 【不需要切片】（L5-03 讲的"一步搞定"）

// 【★ 注意断言的数量】
//   dense 有【两个】expect_eq（#0 和 #1）-> 两个不同的期望值
//   splat 只有【一个】%e_splat -> 两台设备共用
//   -> 断言的结构本身就反映了"结果是否相同"

// 【同时注意类型】
//   局部类型是 tensor<2x4xi32>（4x4 沿 x=2 切第 0 维 -> 2x4）
//   splat 的期望值【也是】tensor<2x4>
//   -> 形状是局部的，但【内容处处相同】✓

// 【★ 这就是 L5-03 那个洞察的数值验证】
//   L5-03 里我从 IR 形态【推断】"splat 不需要切片"
//   这里测试文件用【两个不同的期望值】证明了那个推断：
//     dense 的两台设备期望值不同、splat 的相同`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'dense 常量', c: '#fb7185', d: '<b>两台设备期望值不同</b><br><span class="mono">%e_dense_0</span> ≠ <span class="mono">%e_dense_1</span><br>→ 必须切片（L5-01 六步）' },
      { t: 'splat 常量', c: '#4ade80', d: '<b>两台设备相同</b><br>共用 <span class="mono">%e_splat = dense&lt;7&gt;</span><br>→ <b>不需要切片</b>（L5-03）' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>断言的结构本身就是证据</b>：dense 有<b>两个</b>期望值，说明两台设备结果不同。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>splat 只有【一个】期望值</b> —— 两台设备共用，说明结果相同。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 这是 L5-03 洞察的直接证据</b>：那里从 IR 形态推断，这里用<b>数值</b>验证。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>注意类型</b>：局部类型是 <span class="mono">tensor&lt;2x4&gt;</span> —— 形状是局部的，但 splat 的<b>内容处处相同</b>。';
    });
  }
},

/* ------------------------------------------------ 5 族谱 */
{
  kicker: 'L6-01 · 集合通信执行',
  title: '10 个文件的<span class="hl-a">族谱</span>与跨课对应',
  sub: '每个文件验证一个算子的数值行为。',
  caption: '<span class="mono">collective_permute</span> 有 3 个文件 —— 因为它的语义最"细"。',
  code: `// 【族谱】10 个文件 / 590 行
//   算子                  文件                                        验证什么
//   all_gather            sdy_all_gather                              拼接结果 + per-dim 选项等价
//   all_reduce            sdy_all_reduce                              跨轴 vs 跨单轴的分组差异
//   all_slice             sdy_all_slice                               每台设备拿自己那片
//   all_to_all            sdy_all_to_all / _cross_replica             轴跨维移动的数值效果
//   collective_permute    ..._with_self_loops /
//                         ..._without_self_loops /
//                         ..._cross_replica                           收发对照表的效果（3 个变体）
//   reduce_scatter        sdy_reduce_scatter                          边归约边切分
//   constant              sdy_constant                                【splat vs dense 的差异】

// 【为什么 collective_permute 有 3 个文件】
//   它的语义最"细" —— source_target_pairs 的每一种形态都要验证
//     with_self_loops     置换表里有"发给自己"的项
//     without_self_loops  没有"自己发给自己"
//     cross_replica       跨副本的置换

// 【注意 sdy_constant 不是集合通信】
//   它验证的是【常量处理】（L5-01 / L5-03）
//   放在这里是因为它同样需要"逐设备不同输入"的验证方式

// 【★ 与前面课的对应】
//   本课验证                              对应课
//   all_reduce 的分组 {0,2} / {1,3}       L5-02（replica_groups 的构造）
//   设备号 <-> 网格坐标的对应             L5-01（查找表推导的规则）
//   per-dim-all-gather 两种模式等价       L5-02（四个 RUN 行的选项）
//   splat vs dense 的差异                 L5-03（"splat 不需要切片"）
//   collective_permute 的收发效果         L5-02（source_target_pairs）

// 【★ 本课的价值】
//   L5-02 / L5-03 是从【IR 形态】推断规则的
//   本课用【真正执行的数值】验证了那些推断
//   特别是 splat vs dense 那条 —— L5-03 的洞察在这里得到了【直接的数值证据】

// 【L6 的进度】
//   L6-00 可执行测试机制（已做）
//   L6-01 sdy.* 集合通信的执行（本课）
//   L6-02 stablehlo 集合通信    L6-03 卷积
//   L6-04 矩阵乘/fft/iota       L6-05 gather（★最复杂）
//   L6-06 pad（★最大族）        L6-07 reshape
//   L6-08 reverse/slice         L6-09 scatter 与杂项`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: 'all_gather', n: 1, c: '#38bdf8' }, { t: 'all_reduce', n: 1, c: '#4ade80' },
      { t: 'all_slice', n: 1, c: '#22c55e' }, { t: 'all_to_all', n: 2, c: '#fbbf24' },
      { t: 'collective_permute', n: 3, c: '#f59e0b' }, { t: 'reduce_scatter', n: 1, c: '#c084fc' },
      { t: 'constant', n: 1, c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:140px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:9px;color:${f.c};overflow-wrap:anywhere">${f.t}</div>
        <div class="big" style="font-size:16px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 100)); msg.innerHTML = '<b>10 个文件</b> —— <span class="mono">collective_permute</span> 最多（3 个变体）。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>为什么 permute 最多</b>：<span class="mono">source_target_pairs</span> 的每种形态都要验证。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>注意 <span class="mono">sdy_constant</span></b> 不是集合通信 —— 它验证常量处理（L5-01/L5-03）。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L6-02 讲 <span class="mono">stablehlo</span> 集合通信的执行。';
    });
  }
},

/* ------------------------------------------------ 6 练习 */
{
  kicker: 'L6-01 · 练习',
  title: '练一练：<span class="hl-a">预测每台设备的结果</span>',
  sub: '三道题分别考：分组、设备号规则、splat vs dense。',
  caption: '一句话总结：<b>不同算子的结果分布不同，这决定了断言该怎么写</b>。',
  code: `// 题 1：2x2 mesh、4 台设备输入 1/10/100/1000，
//       all_reduce {"x"} 时设备 0 和 2 得到什么？

// 题 2：为什么设备 0、1 的 x 都是 0？

// 题 3：dense 常量与 splat 常量在分片后，
//       两台设备的期望值分别是什么关系？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col compact-ex', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '2x2 mesh、4 台设备输入 <span class="mono">1/10/100/1000</span>，<span class="mono">all_reduce {"x"}</span> 时设备 0 和 2 得到什么？',
        a: '<b>都得到 <span class="mono">101</span></b>（= 1 + 100）。' +
           '<br><b>为什么</b>：跨 <span class="mono">x</span> 归约 → <b>同一个 <span class="mono">y</span> 值</b>的设备归为一组。设备 0 和 2 的 <span class="mono">y</span> 都是 0 → 同一组 <span class="mono">{0, 2}</span>。' +
           '<br>设备 0 输入 1、设备 2 输入 100 → 归约得 <b>101</b> ✓' +
           '<br><span class="dim">对比：跨<b>两轴</b>时所有设备都得到 <span class="mono">1111</span>（= 1+10+100+1000）。</span>'
      },
      {
        q: '为什么设备 0、1 的 <span class="mono">x</span> 都是 0？',
        a: '因为 <b><span class="mono">x</span> 是最 major 的轴</b> —— 设备号按轴序<b>最 major 优先</b>展开。' +
           '<br><b>证据</b>：测试文件的注释直接给出<br><span class="mono">// (0,0): dev 0 &nbsp; (0,1): dev 1 &nbsp; (1,0): dev 2 &nbsp; (1,1): dev 3</span>' +
           '<br>设备 0、1 的 <span class="mono">x=0</span>，设备 2、3 的 <span class="mono">x=1</span> → <span class="mono">x</span> 变化最慢 ✓' +
           '<br><span class="dim">★ 这<b>直接验证</b>了 L5-01 的推导 —— 那里我从查找表 <span class="mono">[0,0,0,0,2,2,2,2]</span> 反推出这条规则。</span>'
      },
      {
        q: '<span class="mono">dense</span> 常量与 <span class="mono">splat</span> 常量在分片后，两台设备的期望值分别是什么关系？',
        a: '<b>dense</b>：两台设备期望值<b>不同</b>（<span class="mono">%e_dense_0</span> ≠ <span class="mono">%e_dense_1</span>）→ 必须用 <span class="mono">replica_id</span> + 切片（L5-01 的六步）。' +
           '<br><b>splat</b>：两台设备期望值<b>相同</b>（共用 <span class="mono">%e_splat = dense&lt;7&gt;</span>）→ <b>不需要切片</b>（L5-03 的"一步搞定"）。' +
           '<br><span class="dim">★ 断言的结构本身就是证据：dense 有<b>两个</b> <span class="mono">expect_eq</span>，splat 只有<b>一个</b>共用期望值。这是 L5-03 那个洞察的<b>数值验证</b>。</span>'
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
