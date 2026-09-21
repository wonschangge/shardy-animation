/* ==========================================================================
   L6-10 · exec-single-device-and-alltoall   （P2 · L6 最后一课）
   --------------------------------------------------------------------------
   覆盖：executable_partitioner_pipeline/ 下 4 个文件
         (sdy_all_to_all_fully_scattered 95, ..._partially_scattered 69,
          single_device_switch 52, single_device_add 45)
   目标：讲清单设备分片如何被执行（partition_id + 单设备网格）。
   与 L7-01 引用同一批文件，但视角不同：本课讲【执行】，L7-01 讲【时间轴】。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 partition_id */
{
  kicker: 'L6-10 · 单设备与 all-to-all',
  title: '★ <span class="mono hl-a">partition_id</span>：让"哪台设备算了什么"可验证',
  sub: '把**设备号**加到结果上 —— 这样测试就能区分"正确"与"错误"的实现。',
  caption: 'L6 的<b>最后一课</b>。',
  code: `sdy.mesh @mesh = <["x"=2]>
sdy.mesh @single_dev_0 = <[], device_ids=[0]>
sdy.mesh @single_dev_1 = <[], device_ids=[1]>

func.func @single_device_add(
    %arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
    -> (tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
  %part_id = stablehlo.partition_id : tensor<ui32>
  %part_id_i32 = stablehlo.convert %part_id : (tensor<ui32>) -> tensor<i32>
  %part_id_bc = stablehlo.broadcast_in_dim %part_id_i32, dims=[] : (tensor<i32>) -> tensor<4xi32>
  %0 = stablehlo.add %arg0, %part_id_bc {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_1, []>]>
  } : tensor<4xi32>
  return %0 : tensor<4xi32>
}

// 【读法】本课的核心洞察
//   stablehlo.partition_id —— 取【当前设备的编号】
//   转成 i32 -> broadcast_in_dim 广播成 tensor<4xi32> -> 【加到 %arg0】
//
//   【★ 效果】每台设备的计算结果【不同】
//     设备 0 加 0、设备 1 加 1
//   -> 测试就能验证"【哪台设备算了什么】" ✓
//
// 【★ 与前面几课的手法对照】
//   课              手法                              目的
//   L6-01 / L6-04   给每台设备【不同的输入】          让结果可区分
//   L6-03           把某个 shard 【乘以 1000】        让贡献可辨识
//   L6-10（本课）   用 【partition_id】 加到结果上    让【设备号】体现在结果里
//
//   共同目的：让测试能区分"正确"与"错误"的实现

// 【★ 分片的变化：从 @mesh 到单设备网格】
//   位置          分片标注
//   %arg0         @mesh, [{"x"}]        【2 设备网格】
//   %0（结果）    【@single_dev_1, []】  【单设备网格】！
//
//   "单设备分片" = 算子被标注在【一个没有轴的单设备网格】上
//     意味着"这个算子的计算【只在一台设备上进行】"
//   -sdy-resolve-single-device-sharding（L4-14）就是处理这种标注的 pass

// 【★ 为什么要"单设备"】
//   有些算子（如某些 custom_call）【无法在设备间切分】
//     必须【收拢到一台设备】上算
//   用单设备网格标注它
//     -> 后续的 reshard 就会【自动插入把数据收拢的通信】`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'partition_id', c: '#38bdf8', d: '取<b>设备编号</b><br>加到结果上' },
      { t: '效果', c: '#4ade80', d: '设备 0 加 0<br>设备 1 加 1<br><b>结果可区分</b>' },
      { t: '@single_dev_1', c: '#fbbf24', d: '<b>单设备网格</b><br><span class="mono">&lt;[], device_ids=[1]&gt;</span><br>没有轴' },
      { t: '含义', c: '#fb7185', d: '计算<b>只在一台<br>设备上进行</b><br><span class="dim">pass 自动插收拢通信</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:9.5px;color:${x.c};overflow-wrap:anywhere">${x.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<span class="mono">partition_id</span> 取当前设备编号 —— 这是让"设备号"进入数据的<b>唯一方式</b>。',
        '<b>加到结果上</b> → 每台设备的输出不同 → 测试就能验证"<b>哪台设备算了什么</b>"。',
        '<b>单设备网格</b>：轴列表为空 —— 表达"这个算子只在一台设备上算"。',
        '当算子无法在设备间切分时，用单设备网格标注它，pass 会<b>自动插入收拢通信</b>。',
      ][i];
    }));
    tl.at(15200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 三种"让结果可区分"的手法</b>：不同输入（L6-01）/ 缩放（L6-03）/ <span class="mono">partition_id</span>（本课）。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 网格切换 */
{
  kicker: 'L6-10 · 单设备与 all-to-all',
  title: '★ <span class="mono hl-a">single_device_switch</span>：两次网格切换',
  sub: '数据流：`@mesh` → `@single_dev_2` → `@single_dev_1`。',
  caption: '每次切换都要<b>收拢/搬运数据</b>。',
  code: `sdy.mesh @mesh = <["x"=4]>
sdy.mesh @single_dev_1 = <[], device_ids=[1]>
sdy.mesh @single_dev_2 = <[], device_ids=[2]>

func.func @single_device_switch(
    %arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
    -> (tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
  %part_id = stablehlo.partition_id : tensor<ui32>
  %part_id_i32 = stablehlo.convert %part_id : (tensor<ui32>) -> tensor<i32>
  %part_id_bc = stablehlo.broadcast_in_dim %part_id_i32, dims=[] : (tensor<i32>) -> tensor<4xi32>

  %c10 = stablehlo.constant dense<10> : tensor<4xi32>
  %part_id_x10 = stablehlo.multiply %part_id_bc, %c10 : tensor<4xi32>

  %0 = stablehlo.add %arg0, %part_id_bc {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_2, []>]>
  } : tensor<4xi32>

  %1 = stablehlo.add %0, %part_id_x10 {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_1, []>]>
  } : tensor<4xi32>

// 【读法】比 single_device_add 多了一次切换
//   位置          网格                说明
//   %arg0         @mesh（x=4）        起点：4 台设备
//   【%0】        【@single_dev_2】   第一次加法在【物理设备 2】
//   【%1】        【@single_dev_1】   第二次加法在【物理设备 1】
//
//   【★ 数据流】@mesh -> @single_dev_2 -> @single_dev_1
//     —— 【两次网格切换】，每次都要【收拢/搬运数据】
//
// 【★ partition_id x 10 的作用】
//   第一次加 partition_id（设备 2 -> 加 2）
//   第二次加 partition_id x 10（设备 1 -> 加 10）
//   -> 【两次加法的影响可区分】✓
//
// 【★ 这验证了 L4-14 的 resolve-single-device-sharding】
//   从多设备网格切到单设备网格时
//     pass 要【插入 reshard】把数据【收拢到目标设备】
//   反向切换则要【广播】`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const steps = [
      { t: '@mesh', c: '#38bdf8', d: '起点<br><b>4 台设备</b>' },
      { t: '@single_dev_2', c: '#fbbf24', d: '第一次加法<br>在<b>物理设备 2</b>' },
      { t: '@single_dev_1', c: '#4ade80', d: '第二次加法<br>在<b>物理设备 1</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = steps.map(s => {
      const e = U.el('div', { class: 'card', style: `width:246px;opacity:.33;transition:.3s;border-color:${s.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:10.5px;color:${s.c}">${s.t}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4;margin-top:3px">${s.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    steps.forEach((s, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>起点</b>：数据分片在 4 台设备上（<span class="mono">x=4</span>）。',
        '<b>第一次切换</b>：收拢到<b>物理设备 2</b> —— 需要通信。',
        '<b>第二次切换</b>：再搬到<b>物理设备 1</b> —— 又一次通信。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b><span class="mono">partition_id × 10</span> 的作用</b>：让两次加法的影响<b>可区分</b>（设备 2 加 2、设备 1 加 10）。';
    });
  }
},

/* ------------------------------------------------ 3 注释的坦诚 */
{
  kicker: 'L6-10 · 单设备与 all-to-all',
  title: '上游注释里的<span class="hl-a">坦诚</span>',
  sub: '**"现实中不会这么做"** —— 但为了测试机制，用简单算子代替。',
  caption: '这是上游测试的一个<b>常见模式</b>，读懂它能省很多困惑。',
  code: `// It doesn't make much sense to run an element-wise op on single-device in
// in real practice. But since we can't hook up an arbitrary custom-call op
// to the StablleHLO interpreter straight-forwardly, we use a simple add op
// here to test the single-device sharding handling in the partitioner
// pipeline.

// 【读法】三段话，三个信息
//   ① "It doesn't make much sense to run an element-wise op on single-device
//      in real practice"
//      —— 承认【现实中不会这么做】（逐元素算子没必要放单设备）
//
//   ② "since we can't hook up an arbitrary custom-call op to the StablleHLO
//      interpreter straight-forwardly"
//      —— 【真正的原因】：无法把任意 custom_call 接到 StableHLO 解释器上
//
//   ③ "we use a simple add op here to test the single-device sharding
//      handling in the partitioner pipeline"
//      —— 所以用 add 【代替】 custom_call 来【测试单设备分片处理】
//
// 【★ 这是上游测试的一个常见模式】
//   用"【能跑起来的简单算子】"代替"【想测但跑不起来的算子】"
//   测试的是【机制】（单设备分片处理），而不是那个具体算子
//
// 【★ 与 L7-03 的呼应】
//   L7-03 的 registry_failures 用的【就是】custom_call
//     但它只验证【警告】，不执行
//   本课用 add 【代替】custom_call 来【执行】
//   -> 两个测试从不同角度覆盖了 custom_call 相关的问题
//
// 【★ 读测试的一个技巧】
//   上游注释【经常直接说明"为什么这样写"】
//   遇到看起来"奇怪"的测试，【先读注释】——
//   往往能省下大量猜测时间（L6-03、L6-04、L6-06 都有类似情况）`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '想测什么', c: '#38bdf8', d: '<span class="mono">custom_call</span> 的<br><b>单设备分片处理</b>' },
      { t: '为什么不能', c: '#fb7185', d: '无法把任意<br><span class="mono">custom_call</span> 接到<br>StableHLO 解释器' },
      { t: '怎么办', c: '#4ade80', d: '用 <span class="mono">add</span> <b>代替</b><br>测试的是<b>机制</b><br>而非具体算子' },
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
        '<b>真正想测的</b>：<span class="mono">custom_call</span> 在单设备上的分片处理。',
        '<b>障碍</b>：<span class="mono">custom_call</span> 无法直接接到 StableHLO 解释器上执行。',
        '<b>解法</b>：用 <span class="mono">add</span> 代替 —— <b>测的是机制，不是那个算子</b>。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 读测试的技巧</b>：遇到看起来"奇怪"的测试，<b>先读注释</b> —— 往往直接说明原因。';
    });
  }
},

/* ------------------------------------------------ 4 all-to-all */
{
  kicker: 'L6-10 · 单设备与 all-to-all',
  title: '两个 <span class="mono hl-a">all-to-all</span> 文件：本课看"执行"',
  sub: '与 L7-01 引用同一批文件 —— 但视角不同：本课讲**执行**，L7-01 讲**时间轴**。',
  caption: '执行层面的关键问题：<b>优化后的通信跑出来的数值对不对</b>。',
  code: `// 【fully_scattered】
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
// Resharding pattern:
// Initial lowering creates sdy.collective_permute (swapping "x" and "y" on
// OptimizeCollectivesPass eliminates the collective_permute by decomposing
// The permuted axes ("x" and "y") are fully scattered off dimension 0 to

// 【partially_scattered】
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
// Resharding pattern:
// Initial lowering creates sdy.collective_permute (swapping "x" and "y" on
// OptimizeCollectivesPass detects the collective_permute + all_to_all chain and
// The permuted axes ("x" and "y") are partially scattered off dimension 0 (only

// 【读法】本课聚焦"执行"视角
//   两者都有【两个 RUN 行】（"false" / "true"）
//     -> 验证 REPL 与 HALO 等价（L4-09）
//   【执行层面的意义】：collective_permute 被优化成 all_to_all 后
//     【执行结果必须不变】—— 这正是这两个文件验证的
//
// 【★ 本课与 L7-01 的分工】
//   课              视角
//   L6-10（本课）   【执行】：优化后的通信【跑出来的数值对不对】
//   L7-01           【时间轴】：collective_permute 在流水线里【怎么被消除】
//
//   同一批文件、两个角度 —— 这正是"一课可以多角度讲"的例子
//   （AGENTS.md §4 允许一个文件被多课引用，但需声明主课）
//
// 【★ 为什么"执行"视角重要】
//   优化 pass 改写的是【通信方式】（collective_permute -> all_to_all）
//   改错了 IR【依然能通过校验】，但【跑出来的数值会错】
//   -> 必须用可执行测试验证（L6-00 的核心思想）
//
// 【fully vs partially】
//   fully：OptimizeCollectivesPass【主动分解】collective_permute
//   partially：它【检测到】"collective_permute + all_to_all 链"
//     -> 优化【依赖于已有的 all_to_all】
//   两者【执行结果相同】—— 只是优化路径不同`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'L6-10（本课）', c: '#4ade80', d: '<b>执行</b>视角<br>优化后的通信<br><b>数值对不对</b>' },
      { t: 'L7-01', c: '#38bdf8', d: '<b>时间轴</b>视角<br><span class="mono">collective_permute</span><br><b>怎么被消除</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>本课的角度</b>：优化 pass 改的是<b>通信方式</b> —— 改错了 IR 仍能通过校验，但<b>数值会错</b>。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>L7-01 的角度</b>：<span class="mono">collective_permute</span> 在流水线里<b>怎么被消除</b>。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 同一批文件、两个角度</b> —— 一课可以多角度讲（§4 允许，需声明主课）。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>两个 RUN 行</b>验证 REPL 与 HALO 等价（L4-09）；<b>fully vs partially</b> 执行结果相同，只是优化路径不同。';
    });
  }
},

/* ------------------------------------------------ 5 小结 */
{
  kicker: 'L6-10 · 单设备与 all-to-all',
  title: '小结：<span class="hl-a">L6 的最后一课</span>',
  sub: '三条结论 —— 关于"怎么让测试可验证"。',
  caption: '一句话：<b>partition_id 让设备号进入数据，单设备网格表达"只在一台算"</b>。',
  code: `// 【族谱】4 个文件
//   族                文件                                本课聚焦
//   单设备分片        single_device_add                   partition_id + 单设备网格
//   网格切换          single_device_switch                @mesh -> @single_dev_2 -> @single_dev_1
//   all-to-all 优化   sdy_all_to_all_fully_scattered      优化后【执行结果不变】
//   all-to-all 优化   sdy_all_to_all_partially_scattered  同上（部分分解）

// 【★ 三条结论】
//   ① partition_id 是让"哪台设备算了什么"可验证的关键手法
//      加到结果上，设备号就体现在数值里
//   ② 单设备网格 <[], device_ids=[N]> 表达"这个算子只在一台设备上算"
//      从多设备网格切过去时，pass 会【自动插入收拢通信】
//   ③ 上游测试常用"简单算子代替想测但跑不起来的算子"
//      注释坦诚说明了这一点（用 add 代替 custom_call）

// 【★ L6 全层回顾】（本课是 L6 最后一课）
//   L6-00 可执行测试机制         L6-01 sdy.* 集合通信
//   L6-02 stablehlo 集合通信     L6-03 卷积
//   L6-04 矩阵乘/fft/iota        L6-05 gather
//   L6-06 pad（最大族）          L6-07 reshape
//   L6-08 reverse/slice          L6-09 scatter 与杂项
//   L6-10 单设备与 all-to-all（本课）
//   ────────────────────────────────────
//   11 课 / 78 个文件

// 【★ L6 的三条主线（复述）】
//   ① 可执行测试 = 分片版 vs 串行版的【数值对比】
//   ② 集合通信的两条路径：SDY 算子降级 vs 手写 stablehlo
//   ③ 各算子族的执行验证（归约/形状/访存/通信优化/单设备）

// 【★ 与前面课呼应】
//   L4-09  REPL / HALO（两个 all-to-all 文件的两个 RUN 行）
//   L4-14  resolve-single-device-sharding（单设备网格）
//   L6-00  可执行测试机制（本课全部文件都用它）
//   L7-01  端到端时间轴（引用同一批文件，角度不同）
//   L7-03  registry_failures 的 custom_call（与本课的 add 形成互补）

// 一句话总结：
//   L6-10 讲"单设备分片怎么被执行"
//   用 partition_id 让设备号体现在结果里
//   用单设备网格表达"只在一台设备上算"
//   并验证 all-to-all 优化后的执行结果不变`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: 'partition_id', n: 1, c: '#38bdf8' }, { t: '网格切换', n: 1, c: '#4ade80' },
      { t: 'all-to-all', n: 2, c: '#fbbf24' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:200px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:10px;color:${f.c};overflow-wrap:anywhere">${f.t}</div>
        <div class="big" style="font-size:17px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 130)); msg.innerHTML = '<b>4 个文件</b> —— 单设备分片 + 网格切换 + all-to-all 优化。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>★ 三条结论</b>：<span class="mono">partition_id</span> 让设备号进数据 / 单设备网格 / 简单算子代替。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>L6 共 11 课 / 78 个文件</b> —— 本课是最后一课。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>与 L7-01 同源不同角度</b>：本课讲执行，L7-01 讲时间轴。';
    });
  }
},

];
