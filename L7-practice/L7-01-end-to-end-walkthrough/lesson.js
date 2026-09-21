/* ==========================================================================
   L7-01 · end-to-end-walkthrough   （P0 · L7 开篇）
   --------------------------------------------------------------------------
   覆盖：executable_partitioner_pipeline/ 下 4 个端到端流水线测试
         (sdy_all_to_all_fully_scattered 95, ..._partially_scattered 69,
          single_device_switch 52, single_device_add 45)
   目标：用四个真实片段展示"从导入到设备代码"这条时间轴。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 端到端流水线 */
{
  kicker: 'L7-01 · 端到端走查',
  title: 'L7 开篇：<span class="hl-a">完整分区器流水线</span>',
  sub: '四个文件都跑 `executable_partitioner_pipeline` 的**完整流水线**（L6-00 讲的第二个脚本）。',
  caption: '本课是"从导入到设备代码"这条时间轴上的<b>四个真实片段</b>。',
  code: `// 【本课覆盖】4 个端到端流水线测试
//   sdy_all_to_all_fully_scattered        95 行   通信优化（全散射）
//   sdy_all_to_all_partially_scattered    69 行   通信优化（部分散射）
//   single_device_switch                  52 行   单设备切换
//   single_device_add                     45 行   单设备加法

// 【★ 它们跑的是【完整分区器流水线】（回顾 L6-00）】
//   pass                                        对应课
//   --sdy-insert-explicit-reshards              L4-02 ~ L4-07
//   --sdy-resolve-permutation-factors           L4-09
//   --sdy-reshard-to-collectives                L4-08
//   --sdy-optimize-collectives                  L4-10
//   --sdy-pad-for-divisibility                  L5-09
//   --sdy-resolve-single-device-sharding        L4-14

// 【★ 本课四个文件正好覆盖其中两个 pass】
//   single_device_*   -> --sdy-resolve-single-device-sharding（L4-14）
//   sdy_all_to_all_*  -> --sdy-optimize-collectives（L4-10）

// 【★ 本课的三条结论】
//   ① "空轴网格" sdy.mesh @m = <[], device_ids=[0]> 是单设备的表达方式
//      没有轴可分，但仍需一个合法 mesh
//   ② 网格可以在一个程序里切换
//      从多设备切到单设备意味着【收拢分片】（需要通信）
//   ③ collective_permute 可以被优化消除
//      fully 是主动分解、partially 是复用已有的 all_to_all

// 一句话：
//   L7-01 用四个端到端流水线测试展示了两个 pass 的实际效果`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'single_device_add', c: '#38bdf8', d: '<b>空轴网格</b><br><span class="mono">&lt;[], device_ids=[0]&gt;</span>' },
      { t: 'single_device_switch', c: '#0ea5e9', d: '多设备 ↔<br><b>单设备切换</b>' },
      { t: 'fully_scattered', c: '#4ade80', d: '<span class="mono">collective_permute</span><br><b>完全分解</b>' },
      { t: 'partially_scattered', c: '#fbbf24', d: '<b>部分</b>分解<br><span class="dim">复用已有 all_to_all</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:9.5px;color:${x.c};overflow-wrap:anywhere">${x.t}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>四个文件</b>覆盖两个 pass：单设备解析 + 通信优化。'; });
    tl.at(4400, () => {
      msg.innerHTML = '<b>★ 它们跑的是完整流水线</b>（L6-00 讲的第二个脚本），六个 pass 串成一条真实可跑的命令。';
    });
    tl.at(7800, () => {
      msg.innerHTML = '<b>本课覆盖其中两个</b>：<span class="mono">resolve-single-device-sharding</span>（L4-14）与 <span class="mono">optimize-collectives</span>（L4-10）。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<b>三个结论</b>：空轴网格 / 网格可切换 / <span class="mono">collective_permute</span> 可被消除。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 空轴网格 */
{
  kicker: 'L7-01 · 端到端走查',
  title: '★ <span class="hl-a">空轴网格</span>：单设备的表达方式',
  sub: '`sdy.mesh @m = <[], device_ids=[0]>` —— **轴列表是空的**！',
  caption: '这是本课最特别的一处语法。',
  code: `// No need to run this test without HALO export.
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"
// It doesn't make much sense to run an element-wise op on single-device in
sdy.mesh @mesh = <["x"=2]>
sdy.mesh @single_dev_0 = <[], device_ids=[0]>
sdy.mesh @single_dev_1 = <[], device_ids=[1]>

// 【读法】本课最特别的一处
//   sdy.mesh @single_dev_0 = <[], device_ids=[0]>
//     —— 【空轴列表 []】！这是一个【只有一个设备、没有任何轴】的网格
//   device_ids=[0] —— 它映射到【物理设备 0】
//   同文件还定义了 @mesh（x=2，两台设备）与 @single_dev_1（物理设备 1）

// 【★ 为什么需要"空轴网格"】
//   单设备情形下【没有"轴"可分】
//   但仍然需要一个【合法的 mesh】来表达"这个张量只在一台设备上"
//   -> 空轴网格就是这个表达方式

// 【注释里的两句话】
//   ① // No need to run this test without HALO export.
//      只用 "false"（不用 HALO）跑，因为【单设备不需要 halo exchange】
//   ② // It doesn't make much sense to run an element-wise op on single-device in
//      承认这个测试【本身意义不大】（逐元素算子在单设备上没什么可分的）
//      它存在是为了【覆盖这条代码路径】

// 【★ 这正是 L4-14 讲的 resolve-single-device-sharding 的场景】
//   把"单设备分片"解析成【不需要通信】的形式

// 【注意 RUN 行用位置参数 "false"】
//   第 3 个位置参数就是 ENABLE_HALO_EXCHANGE（L6-00 讲的）`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '普通网格', c: '#38bdf8', d: '<span class="mono">&lt;["x"=2]&gt;</span><br>两台设备<br><b>有轴可分</b>' },
      { t: '空轴网格', c: '#fbbf24', d: '<span class="mono">&lt;[], device_ids=[0]&gt;</span><br><b>一台设备</b><br><b>没有轴</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:11px;overflow-wrap:anywhere">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>普通网格</b>：有轴（<span class="mono">x=2</span>），可以沿轴切分。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>空轴网格</b>：<span class="mono">[]</span> —— <b>一台设备、没有轴</b>，但仍需一个合法 mesh。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>为什么需要它</b>：单设备时<b>没有轴可分</b>，但要用 mesh 表达"只在一台设备上"。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>注释很坦诚</b>：这个测试<b>本身意义不大</b> —— 它存在是为了<b>覆盖代码路径</b>。';
    });
  }
},

/* ------------------------------------------------ 3 ★ 网格切换 */
{
  kicker: 'L7-01 · 端到端走查',
  title: '★ <span class="mono hl-a">single_device_switch</span>：多设备 ↔ 单设备',
  sub: '同一程序里**多个网格并存** —— 切换意味着**收拢分片**（需要通信）。',
  caption: '与 L1-01 讲的 mesh 定义形成呼应。',
  code: `// No need to run this test without HALO export.
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"
sdy.mesh @mesh = <["x"=4]>
sdy.mesh @single_dev_1 = <[], device_ids=[1]>
sdy.mesh @single_dev_2 = <[], device_ids=[2]>
func.func @single_device_switch(

// 【读法】
//   @mesh 是 x=4（【4 台设备】），而 @single_dev_1/@single_dev_2 是【单设备网格】
//   用例名 switch —— 程序在【两种网格之间切换】：
//     一部分算子跑在 4 设备网格上、一部分跑在单设备网格上
//
// 【★ 这验证的是"网格切换时的 reshard"】
//   从 x=4 切到单设备 -> 把【所有分片收拢到一台设备】（需要通信）
//   反之从单设备切回 x=4 -> 【广播】
//
// 【★ 与 L1-01 的呼应】
//   那里讲 sdy.mesh 的定义与 device_ids
//   本课看到【同一程序里多个网格并存】的实际用法
//
// 【为什么只跑 "false"】
//   注释：// No need to run this test without HALO export.
//   单设备场景【不涉及 halo exchange】-> 只需要一种模式
//
// 【device_ids 的作用】
//   @single_dev_1 用【物理设备 1】、@single_dev_2 用【物理设备 2】
//   -> 验证"收拢到哪台设备"是可指定的
//   （回顾 L1-02 讲的 device_ids 语义）`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '多设备网格', c: '#38bdf8', d: '<span class="mono">&lt;["x"=4]&gt;</span><br>4 台设备<br>分片<b>散在 4 台</b>' },
      { t: '切换到单设备', c: '#fbbf24', d: '<span class="mono">&lt;[], device_ids=[1]&gt;</span><br><b>收拢到 1 台</b><br><b>需要通信</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:11px;overflow-wrap:anywhere">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>起点</b>：数据分片在 4 台设备上（<span class="mono">x=4</span>）。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>切换</b>：收拢到<b>单台设备</b> —— 这需要<b>把所有分片汇总</b>（通信）。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>反向切换</b>：从单设备切回 <span class="mono">x=4</span> 意味着<b>广播</b>。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b><span class="mono">device_ids</span> 的作用</b>：<span class="mono">@single_dev_1</span> 用物理设备 1 —— <b>收拢到哪台</b>是可指定的（L1-02）。';
    });
  }
},

/* ------------------------------------------------ 4 ★ 通信优化 */
{
  kicker: 'L7-01 · 端到端走查',
  title: '★ <span class="mono hl-a">collective_permute</span> 的完全/部分分解',
  sub: '注释**完整描述了优化过程** —— 这是 L4-10 的最佳实例。',
  caption: '两个文件只差几个词，但语义不同。',
  code: `// 【fully_scattered：完全分解】
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
// Resharding pattern:
// Initial lowering creates sdy.collective_permute (swapping "x" and "y" on
// OptimizeCollectivesPass eliminates the collective_permute by decomposing
// The permuted axes ("x" and "y") are fully scattered off dimension 0 to

// 【partially_scattered：部分分解】
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
// Resharding pattern:
// Initial lowering creates sdy.collective_permute (swapping "x" and "y" on
// OptimizeCollectivesPass detects the collective_permute + all_to_all chain and
// The permuted axes ("x" and "y") are partially scattered off dimension 0 (only

// 【★ 逐句对比】
//   行      fully 版本                                    partially 版本
//   第3句   eliminates ... by 【decomposing】             【detects】 the collective_permute
//                                                         + all_to_all 【chain】 and
//   第4句   are 【fully】 scattered off dimension 0 to    are 【partially】 scattered off
//                                                         dimension 0 (【only】
//
// 【★ 两个版本的差异】
//   fully：OptimizeCollectivesPass 【主动分解】 collective_permute
//   partially：它【检测到】"collective_permute + all_to_all 链"
//     -> 即优化【依赖于已有的 all_to_all】，而不是凭空分解
//   散射程度：fully 是两轴都散、partially 是【只有一部分】散开
//
// 【★ 这验证了 L4-10 讲的"优化模式的多样性"】
//   同一个 collective_permute，在不同上下文里可能被
//   【完全消除】或【部分消除】—— 取决于周围有没有可复用的通信
//
// 【优化过程三步】
//   ① Initial lowering 产生 sdy.collective_permute（交换 x 和 y 轴）
//   ② OptimizeCollectivesPass 消除它（分解 / 检测链）
//   ③ 被置换的轴散射到第 0 维之外
//
// 【两个文件都有两个 RUN 行】—— 都验证 REPL/HALO 等价（L4-09）`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'fully scattered', c: '#4ade80', d: '<b>主动分解</b><br><span class="mono">by decomposing</span><br>两轴<b>都</b>散开' },
      { t: 'partially scattered', c: '#fbbf24', d: '<b>检测到链</b><br><span class="mono">detects ... chain</span><br><b>只有一部分</b>散开' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:11.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>fully</b>：pass <b>主动分解</b> <span class="mono">collective_permute</span> —— 两轴都散开。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>partially</b>：pass <b>检测到</b>"<span class="mono">collective_permute</span> + <span class="mono">all_to_all</span> 链" —— <b>复用已有通信</b>。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 这验证了 L4-10 的"优化模式多样性"</b>：同一个算子可能被<b>完全</b>或<b>部分</b>消除。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>优化三步</b>：产生 <span class="mono">collective_permute</span> → pass 消除它 → 轴散射到第 0 维之外。';
    });
  }
},

/* ------------------------------------------------ 5 小结 */
{
  kicker: 'L7-01 · 端到端走查',
  title: '小结：<span class="hl-a">时间轴上的四个片段</span>',
  sub: '四个文件覆盖两个 pass —— 这是"从导入到设备代码"的真实切片。',
  caption: '一句话：<b>端到端流水线 = 六个 pass 串起来的一条命令</b>。',
  code: `// 【族谱】4 个文件
//   族          文件                                验证什么              对应 pass
//   单设备      single_device_add                   【空轴网格】[]        resolve-single-device-sharding
//   单设备      single_device_switch                多设备 <-> 单设备切换  （L4-14）
//   通信优化    sdy_all_to_all_fully_scattered      collective_permute
//                                                   【完全分解】          optimize-collectives
//   通信优化    sdy_all_to_all_partially_scattered  【部分】分解           （L4-10）

// 【★ 三条结论】
//   ① "空轴网格" sdy.mesh @m = <[], device_ids=[0]> 是单设备的表达方式
//      没有轴可分，但仍需一个合法 mesh
//   ② 网格可以在一个程序里切换
//      从多设备切到单设备意味着【收拢分片】（需要通信）
//   ③ collective_permute 可以被优化消除
//      fully 是主动分解、partially 是复用已有的 all_to_all

// 【★ 完整流水线的六个 pass】（L6-00 讲过）
//   insert-explicit-reshards     L4-02 ~ L4-07
//   resolve-permutation-factors  L4-09
//   reshard-to-collectives       L4-08
//   optimize-collectives         L4-10   <- 本课两个文件验证
//   pad-for-divisibility         L5-09
//   resolve-single-device-sharding L4-14 <- 本课两个文件验证

// 【★ 本课在 L7 中的位置】
//   L7 是【综合实战】层，四课：
//     L7-01 端到端走查（本课）      一条时间轴走完所有 pass
//     L7-02 并行策略                数据/张量/流水线并行 + ZeRO
//     L7-03 调试手册                常见问题排查
//     L7-04 跨方言集成              为自己的方言接入 Shardy

// 一句话总结：
//   L7-01 用四个端到端流水线测试展示了两个 pass 的实际效果
//   这是"从导入到设备代码"这条时间轴上的两个真实片段`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '空轴网格', n: 1, c: '#38bdf8' }, { t: '网格切换', n: 1, c: '#0ea5e9' },
      { t: '完全分解', n: 1, c: '#4ade80' }, { t: '部分分解', n: 1, c: '#fbbf24' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:170px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div style="font-size:10px;color:${f.c};line-height:1.3">${f.t}</div>
        <div class="big" style="font-size:16px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 120)); msg.innerHTML = '<b>4 个文件</b>覆盖两个 pass —— 单设备解析 + 通信优化。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>★ 三条结论</b>：空轴网格 / 网格可切换 / <span class="mono">collective_permute</span> 可被完全或部分消除。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>完整流水线有六个 pass</b>（L6-00 讲过），本课覆盖其中两个。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L7-02 讲并行策略（P0）—— 数据/张量/流水线并行 + ZeRO。';
    });
  }
},

];
