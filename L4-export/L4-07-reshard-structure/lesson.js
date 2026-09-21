/* ==========================================================================
   L4-07 · reshard-structure   （L4 按算子族展开的最后一课）
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/insert_explicit_reshards/ 下 9 个文件
         (1455 行 / 112 用例)
   目标：讲透"结构性"场景的 reshard —— 区域、函数、mesh、单设备四类边界。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 四类边界 */
{
  kicker: 'L4-07 · 结构性 reshard',
  title: '四类<span class="hl-a">边界</span>：reshard 就是跨边界的搬运',
  sub: 'L4-03～06 按算子族展开；这一课按**结构**展开 —— 四类边界问题。',
  caption: '这是 L4 按算子族展开的最后一课，也是把前面几课的"边界机制"汇总的一课。',
  code: `// 9 个文件按"边界类型"归成四族：

// 【区域的边界】data_flow_ops (7) + manual_computation (2)
//   reshard 可以插在区域【内部】（如 case 的每个分支里）
//   与 L3-04 配合：边负责跨边界，reshard 负责区域内部

// 【函数的边界】call_ops (11) + call_ops_...false (11) + func_inputs_outputs (14)
//   实参/形参、return/结果都要对齐
//   与 L3-05（桥接）+ L3-10（搬到调用点）配合

// 【mesh 的边界】meshes (19)
//   mesh 不同（含【设备序不同】）-> 传播根本不发生
//   必须靠 reshard 显式转换

// 【单设备与多设备】single_device_sharding (10) + _errors (0)
//   maximal 网格的导出侧（L1-01 / L2-01 讲过它的特殊规则）

// 加上 L4-02 总纲的完整展开：
// 【未归约】unreduced (38)

// 一句话：
//   前四族都是"边界"问题 —— 区域的、函数的、mesh 的、单设备的
//   reshard 就是跨越这些边界的搬运`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '区域边界', n: 9, c: '#38bdf8' }, { t: '函数边界', n: 36, c: '#4ade80' },
      { t: 'mesh 边界', n: 19, c: '#fbbf24' }, { t: '单设备', n: 10, c: '#c084fc' },
      { t: '未归约', n: 38, c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:138px;opacity:.4;transition:.35s;border-color:${f.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="card-t" style="color:${f.c};font-size:11px">${f.t}</div>
        <div class="big" style="font-size:20px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 120)); msg.innerHTML = '<b>112 个用例</b>分五族 —— 最大两族是未归约（38）与函数边界（36）。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>前四族的共同点</b>：都是"<b>边界</b>"问题 —— reshard 就是跨边界的搬运。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>第五族是特例</b>：<span class="mono">unreduced</span> 不是边界问题，而是 L4-02 总纲的<b>完整展开</b>。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>本课的价值</b>：把 L1-07/L1-08/L3-04/L3-05 的边界机制与 L4 的 reshard 插入串起来。';
    });
  }
},

/* ------------------------------------------------ 2 区域内部 */
{
  kicker: 'L4-07 · 结构性 reshard',
  title: '★ 区域的边界：reshard 插在<span class="hl-a">内部</span>',
  sub: '区域算子（`case` / `while` / `barrier`）内部**可以有 reshard** —— 而且常常必须插在里面。',
  caption: '与 L3-04 配合：<b>边负责跨边界，reshard 负责区域内部</b>。',
  code: `// 【case】分支内部插 reshard
func.func @case(%arg0: tensor<210xf32> {...<@mesh, [{"x":(1)2}]>}, %arg1: tensor<i32>)
    -> (tensor<210xf32> {...<@mesh, [{"y"}]>}) {
  %0 = "stablehlo.case"(%arg1) ({
    // 参数是 [{"x":(1)2}]（子轴），分支内 abs 要求 [{"x"}]（完整轴）
    // CHECK: %[[RESHARD]] = sdy.reshard %arg0 <@mesh, [{"x"}]>
    // CHECK-NEXT: stablehlo.abs %[[RESHARD]]
    %2 = stablehlo.abs %arg0 {...[{"x"}]...} : tensor<210xf32>
    // return 前再插一条到 [{"y"}]（函数结果的要求）
    // CHECK: %[[RESHARD]] = sdy.reshard %{{.*}} <@mesh, [{"y"}]>
    // CHECK-NEXT: stablehlo.return %[[RESHARD]]
    stablehlo.return %2 : tensor<210xf32>
  }, { ... }, { ... }) : ...
}

// 两处 reshard 都在【分支内部】：
//   ① abs 之前：把子轴 "x":(1)2 补齐成完整轴 "x"
//   ② return 之前：调整到函数结果要求的 "y"

// 【manual_computation】内外都要插
//   区域外：%arg0 是 [{"x":(1)2}]，in_shardings 要求 [{"x"}] -> 之前插
//   区域内：abs 产出 [{"x"}]，out_shardings 要求 [{"y"}] -> return 之前插
//   带 manual_axes 时，区域内的 reshard 只在【未冻结】的轴上切换

// 为什么内外都要：
//   区域外是【全局世界】（完整形状）
//   区域内是【局部世界】（L1-07），但 out_shardings 是区域对外的【承诺】`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 分支内 abs 前', c: '#38bdf8', d: '子轴 <span class="mono">"x":(1)2</span><br>→ 完整轴 <span class="mono">"x"</span>' },
      { t: '② 分支内 return 前', c: '#4ade80', d: '调整到<br>函数结果要求的 <span class="mono">"y"</span>' },
      { t: '③ manual 区域外', c: '#fbbf24', d: '满足 <span class="mono">in_shardings</span><br>插在算子【之前】' },
      { t: '④ manual 区域内', c: '#c084fc', d: '满足 <span class="mono">out_shardings</span><br>插在 <span class="mono">return</span>【之前】' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:180px;opacity:.33;transition:.35s;border-color:' + x.c + '55;padding:9px' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:11.5px">${x.t}</div>
        <div class="card-d" style="font-size:10.5px;line-height:1.5">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>为什么子轴要补齐</b>：分支内的算子要求完整轴，而外层给的是子轴。',
        '函数结果要求 <span class="mono">[{"y"}]</span> → 分支内就要把结果调整好。',
        '<b>区域外</b>是全局世界，要满足 <span class="mono">in_shardings</span> 的承诺。',
        '<b>区域内</b>虽然是局部世界，但 <span class="mono">out_shardings</span> 是对外的承诺 —— 也要满足。',
      ][i];
    }));
    tl.at(15200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>与 L3-04 的分工</b>：边负责<b>跨边界</b>，reshard 负责<b>区域内部</b>。';
    });
  }
},

/* ------------------------------------------------ 3 函数与 mesh */
{
  kicker: 'L4-07 · 结构性 reshard',
  title: '函数边界与 <span class="mono hl-a">mesh 切换</span>',
  sub: '**mesh 不同**是比"分片不同"更根本的差异 —— 传播**根本不发生**，只能靠 reshard 转换。',
  caption: '注意 <span class="mono">meshes_different_device_order</span>：轴名与大小都相同，但<b>设备顺序不同</b>，仍是两个网格。',
  code: `// 【函数边界】三个文件共 36 个用例
//   call_ops (11)                        enable-full-version=true
//   call_ops_enable_full_version_false (11)   默认（false）
//   func_inputs_outputs (14)
//
//   实参/形参、return/结果都要对齐
//   与前面几课配合：
//     L3-05 提供【桥接机制】（func_data_flow_edge）
//     L3-10 把分片【搬到调用点】（避免丢失）
//     本课  在两端分片【不一致】时插 reshard
//
//   func_inputs_outputs 的用例名暴露了场景：
//     funcop_result_sharding_does_not_match
//     funcop_result_unsharded_but_different_meshes_between_return_and_func_result
//     funcop_result_sharding_matches_but_different_meshes_between_return_and_func_result
//     ..._multiple_results
//     ^^^^^^^^ 反复出现 "different_meshes"

// 【mesh 边界】meshes (19)
//   optimization_barrier_different_meshes            两端用【不同 mesh】
//   optimization_barrier_meshes_different_device_order  轴名大小相同但【设备序不同】
//   negate_from_empty_sharding_to_iota_sharded
//   negate_from_empty_sharding_to_iota_unsharded
//   binary_op_from_empty_sharding_to_iota_unsharded
//
//   关键：L2-01 讲过"分片只在同一网格内传播"
//     -> 跨 mesh 时传播【根本不发生】
//     -> 必须靠 reshard 显式转换
//   判据（L2-01）：看【轴名与设备顺序】是否一致 —— 名字只是符号`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '函数边界', c: '#4ade80', n: '36 个',
        d: '实参/形参、return/结果<br>都要对齐<br><span class="dim">L3-05 桥接 + L3-10 搬运 + 本课 reshard</span>' },
      { t: 'mesh 边界', c: '#fbbf24', n: '19 个',
        d: '跨 mesh 时传播<b>不发生</b><br>只能靠 reshard 转换<br><span class="dim">含"设备序不同"的情形</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>函数边界</b>：三个文件共 36 个用例 —— 是数量第二多的一族。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>mesh 不同更根本</b>：分片引用的是<b>特定网格</b>的轴，跨网格的对应关系没有定义。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>最微妙的一条</b>：<span class="mono">meshes_different_device_order</span> —— 轴名与大小都相同，只是<b>设备顺序不同</b>，仍是两个网格。';
    });
    tl.at(12000, () => {
      msg.innerHTML = '<b>判据</b>（L2-01）：看<b>轴名与设备顺序</b>是否一致 —— 名字只是符号，内容才是语义。';
    });
  }
},

/* ------------------------------------------------ 4 单设备与未归约 */
{
  kicker: 'L4-07 · 结构性 reshard',
  title: '单设备分片与<span class="hl-a">未归约的完整展开</span>',
  sub: '`maximal` 网格的导出侧；以及 L4-02 总纲那 8 个用例的**完整版**（38 个）。',
  caption: '注意 <span class="mono">all_reduce_delayed_to_call_site</span> —— 归约能<b>跨函数</b>延迟到调用点。',
  code: `// 【单设备分片】single_device_sharding (241 行 / 10 用例) + _errors (30 行)
//   single_device_result_to_tiled_consumer       单设备结果 -> 分片消费者
//   single_device_result_to_replicated_consumer  单设备结果 -> 复制消费者
//   single_device_op0_to_single_device_op1       单设备 -> 单设备
//   single_device_result_directly_returned       单设备结果直接返回
//   tiled_operand_to_single_device_consumer      分片操作数 -> 单设备消费者
//
//   背景：
//     L1-01 讲过 @maximal_mesh = <[], device_ids=[0]> ——"单设备网格"
//     L2-01 讲过它的传播规则：【不被替换、也不沿它传播】
//   本课处理【导出侧】：算子在单设备网格上、消费者在别的网格上时，
//     如何插 reshard 完成转换
//   _errors 文件用 -verify-diagnostics 锁定【报错情形】
//     -> 说明有些单设备分片的组合是【非法】的

// 【未归约】unreduced (433 行 / 38 用例)  <- 本课最大一族
//   它是 L4-02 那 8 个用例的【完整版】，用例名高度重叠：
//     all_reduce_on_func_input
//     unreduced_func_input_until_return
//     lhs_and_result_unreduced_rhs_replicated
//     all_reduce_fully_delayed_until_return
//     all_reduce_delayed_to_call_site          <- 本课新增的维度！
//
//   all_reduce_delayed_to_call_site：
//     归约不仅能在函数内延迟，还能【跨函数延迟到调用点】
//     -> 与第三节的函数调用处理衔接
//
// 读法：L4-02 是总纲（8 个代表），本课是完整展开（38 个）
//       两者用【同名用例】，可以对照阅读`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '单设备分片', c: '#c084fc', n: '10 + 0 个',
        d: '<span class="mono">maximal</span> 网格的导出侧<br>五种转换方向<br><span class="dim">_errors 锁定非法组合</span>' },
      { t: '未归约', c: '#fb7185', n: '38 个',
        d: 'L4-02 总纲的<b>完整展开</b><br>新增"跨函数延迟"维度<br><span class="dim">同名用例可对照阅读</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>maximal 网格</b>是"某台设备独占"—— 它的分片与多设备网格上的分片需要显式转换。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>38 个用例</b>是本课最大一族 —— 说明未归约轴的处理组合非常多。';
    });
    tl.at(8400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>新增维度</b>：<span class="mono">all_reduce_delayed_to_call_site</span> —— 归约能<b>跨函数</b>延迟。';
    });
    tl.at(11600, () => {
      msg.innerHTML = '<b>学习建议</b>：先读 L4-02（总纲），再读本课的 <span class="mono">unreduced</span>（完整展开）—— 同名用例对照最有效。';
    });
  }
},

/* ------------------------------------------------ 5 族谱与收官 */
{
  kicker: 'L4-07 · 结构性 reshard',
  title: 'L4 收官：<span class="hl-a">按算子族展开的七课</span>',
  sub: 'L4-02 是总纲，L4-03～07 按算子族展开 —— 到这里全部讲完。',
  caption: '后面 L4-08～17 会转向各类 collective 的具体形态。',
  code: `// 【L4 本节回顾】
//   L4-02 reshard 插入总纲        95 用例  两条主线：延迟归约 / 对应维同分片
//   L4-03 逐元素与形状类         109 用例  "更大"的一侧优先；并集解法；因子对应
//   L4-04 矩阵与卷积类            95 用例  dot 三类情形；reduce-scatter 模式
//   L4-05 归约与排序类            22 用例  能不能靠归约补救
//   L4-06 访存与通信类            39 用例  三类算子都不是透明的
//   L4-07 结构性场景             112 用例  四类边界（区域/函数/mesh/单设备）
//   ─────────────────────────────────────
//   合计 472 个用例

// 【贯穿全节的判据】
//   ① 冲突在哪两侧？          -> 决定 reshard 插在【之前】还是【之后】
//   ② 有没有 reduction 因子？ -> 有则 unreduced + all_reduce
//   ③ 结果能不能靠归约合并？  -> 不能则【全复制】
//   ④ 是不是跨边界？          -> 区域/函数/mesh/单设备 -> 必须显式转换

// 【下一段】L4-08 ~ L4-17
//   从"插 reshard"转向"各类 collective 的具体形态"
//   包括 all_gather / reduce_scatter 的融合、reshard 的消除、导出边界等`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const items = [
      { t: 'L4-02 总纲', n: 95, c: '#94a3b8' }, { t: 'L4-03 逐元素/形状', n: 109, c: '#38bdf8' },
      { t: 'L4-04 矩阵/卷积', n: 95, c: '#4ade80' }, { t: 'L4-05 归约/排序', n: 22, c: '#fbbf24' },
      { t: 'L4-06 访存/通信', n: 39, c: '#c084fc' }, { t: 'L4-07 结构性', n: 112, c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = items.map(x => {
      const e = U.el('div', { class: 'card', style: `width:118px;opacity:.4;transition:.3s;border-color:${x.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div style="font-size:10px;color:${x.c};line-height:1.3">${x.t}</div>
        <div class="mono" style="font-size:15px;color:${x.c};margin-top:3px">${x.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>472 个用例</b>覆盖 L4 本节六课 —— 是本计划用例最多的部分。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>四条判据</b>：冲突在哪两侧 / 有无 reduction 因子 / 能否靠归约合并 / 是否跨边界。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>下一段</b>：L4-08 起从"插 reshard"转向"各类 collective 的具体形态"。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>L4 进度</b>：已完成 7 / 17 课 —— 本节（算子族展开）收官，进入 collective 专题。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L4-07 · 练习',
  title: '练一练：<span class="hl-a">边界在哪里</span>',
  sub: '三道题分别考：区域内部、mesh 切换、四条判据。',
  caption: '一句话总结：<b>前四族都是边界问题，reshard 就是跨边界的搬运</b>。',
  code: `// 题 1：case 分支内的算子要求与分支外不同，reshard 插在哪？

// 题 2：两个 mesh 的轴名与大小都相同，只是设备顺序不同，
//       需要 reshard 吗？

// 题 3：判断该插什么 reshard 的四条判据是什么？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col compact-ex', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">case</span> 分支内的算子要求与分支外不同，reshard 插在哪？',
        a: '<b>插在分支【内部】</b> —— 具体是两处：' +
           '<br>① 分支内算子**之前**（如把子轴 <span class="mono">"x":(1)2</span> 补齐成完整轴 <span class="mono">"x"</span>）；' +
           '<br>② <span class="mono">stablehlo.return</span> **之前**（调整到函数结果要求的分片）。' +
           '<br><span class="dim">与 L3-04 的分工：<b>边</b>负责跨边界，<b>reshard</b> 负责区域内部。</span>'
      },
      {
        q: '两个 mesh 的轴名与大小都相同，只是<b>设备顺序不同</b>，需要 reshard 吗？',
        a: '<b class="badge bad">需要</b>。它们仍然是<b>两个不同的网格</b>。' +
           '<br><b>判据</b>（L2-01）：看<b>轴名与设备顺序</b>是否一致 —— 名字只是符号，内容才是语义。' +
           '<br><span class="dim">用例 <span class="mono">optimization_barrier_meshes_different_device_order</span> 专门覆盖这种情形。跨 mesh 时传播<b>根本不发生</b>，只能靠 reshard 显式转换。</span>'
      },
      {
        q: '判断该插什么 reshard 的<b>四条判据</b>是什么？',
        a: '① <b>冲突在哪两侧？</b> → 决定 reshard 插在【之前】还是【之后】；' +
           '<br>② <b>有没有 <span class="mono">reduction</span> 因子？</b> → 有则 <span class="mono">unreduced</span> + <span class="mono">all_reduce</span>；' +
           '<br>③ <b>结果能不能靠归约合并？</b> → 不能（顺序、随机序列）则【全复制】；' +
           '<br>④ <b>是不是跨边界？</b> → 区域 / 函数 / mesh / 单设备 → 必须显式转换。' +
           '<br><span class="dim">这四条贯穿 L4-02～07 的 472 个用例。</span>'
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
