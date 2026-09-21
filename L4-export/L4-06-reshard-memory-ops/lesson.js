/* ==========================================================================
   L4-06 · reshard-memory-ops
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/insert_explicit_reshards/ 下 3 个文件
         (gather_scatter 288 / custom_call 318 / collective_ops 107)
         合计 713 行 / 39 用例
   目标：讲透访存、自定义调用、集合通信三类算子的分片约束。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 gather */
{
  kicker: 'L4-06 · 访存与通信类',
  title: '<span class="mono hl-a">gather</span>：与 <span class="mono hl-a">dot</span> 同构的四步',
  sub: '测试注释把这个用例称为 "the most expressive example" —— 它把四步链条完整展示了。',
  caption: '与 L4-04 的 <span class="mono">dot</span> 完全同构：规则里有 <span class="mono">reduction</span> 因子 → 部分结果 → 需要 <span class="mono">all_reduce</span>。',
  code: `// 规则（测试里的 COM 注释给出）：
// sharding_rule<([i, k, p, n, l], [q, l, m, n, o])->([j, k, l, m, n, o])
//   {i=2, j=1, k=6, l=22, m=12, n=26, o=14, p=4, q=2}
//   reduction={i, p} need_replication={j, q}
//              ^^^^^^^ 有 reduction 因子！

// 四步链条：
//   ① 两个操作数各 reshard（把冲突的轴去掉/对齐）
%[[RESHARD0]] = sdy.reshard %arg0 <@mesh_xyzt, [{}, {"x":(2)2}, ...]>
%[[RESHARD1]] = sdy.reshard %arg1 <@mesh_xyzt, [{}, {"z":(1)2}, ...]>
//   ② gather 执行，结果标 unreduced
%[[GATHER]] = "stablehlo.gather"(%[[RESHARD0]], %[[RESHARD1]])
    {...[{}, {"x":(2)2}, ...], unreduced={"y":(1)2}>}
//                              ^^^^^^^^^^^^^^^^^^^ 部分结果
//   ③ all_reduce 合并
%[[ALL_REDUCE]] = sdy.all_reduce {"y":(1)2} %[[GATHER]] out_sharding=<@mesh_xyzt, [...]>
//   ④ 再 reshard 到目标分片
%[[RESHARD_RET]] = sdy.reshard %[[ALL_REDUCE]] <@mesh_xyzt, [{"x":(1)2}, ...]>

// 注意：这个用例里几乎每个轴都是【子轴】（{"x":(1)2}、{"y":(2)2}…）
// 因为 @mesh_xyzt 有 4 个轴共 4x4x4x8 台设备，要用子轴精确切分`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const steps = [
      { n: '①', t: 'reshard 操作数', c: '#38bdf8' },
      { n: '②', t: 'gather（unreduced）', c: '#4ade80' },
      { n: '③', t: 'all_reduce', c: '#fbbf24' },
      { n: '④', t: 'reshard 到目标', c: '#c084fc' },
    ];
    const host = wrap.querySelector('#cards');
    const els = steps.map(s => {
      const e = U.el('div', { class: 'card', style: `width:178px;opacity:.35;transition:.3s;border-color:${s.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="small mono" style="font-size:11px;color:${s.c}">${s.n}</div>
        <div style="font-size:12px;margin-top:3px">${s.t}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    steps.forEach((s, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.35');
      msg.innerHTML = [
        '两个操作数分片冲突 → 各自 reshard 对齐（把 <span class="mono">reduction</span> 轴去掉）。',
        '<b>关键</b>：结果标 <span class="mono">unreduced={"y":(1)2}</span> —— 表示这一轴上是部分结果。',
        '<b>与 dot 同构</b>：有 <span class="mono">reduction</span> 因子就必须归约。',
        '归约后的分片还不是目标 → 再 reshard 一次。',
      ][i];
    }));
    tl.at(14400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>四步 = L4-04 的 dot 模式</b>：reshard → 算子(unreduced) → all_reduce → reshard。';
    });
  }
},

/* ------------------------------------------------ 2 与 L3-11 呼应 */
{
  kicker: 'L4-06 · 访存与通信类',
  title: '与 L3-11 的<span class="hl-a">呼应</span>：先"可表达"，再"可落地"',
  sub: 'L3-11 讲 `gather` 的**批维显式化**；这一课看到显式化**之后**的样子。',
  caption: '两课合起来才是 gather 的完整故事。',
  code: `// 【L3-11】为什么必须显式化批维
//   JAX 的 arr.at[jnp.arange(B), offset] 会生成 iota + concat
//   把批维写进 start_index_map
//   -> 落在 blocked_propagation 因子上
//   -> 传播【无法处理】这一维
//   解法：移到 operand_batching_dims / start_indices_batching_dims

// 【本课】显式化之后
//   本用例的 gather 已经带了 operand_batching_dims：
//     operand_batching_dims = [3, 4]
//     start_indices_batching_dims = [3, 1]
//   -> 批维不再落在 blocked_propagation 里
//   -> 分片能被正常处理（插 reshard + all_reduce）

// 两课合起来：
//   L3-11 让 gather 的分片【可表达】
//   L4-06 让 gather 的分片【可落地】

// 一句话：没有 L3-11 的显式化，L4-06 的这些 reshard 根本无从谈起`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'card', style: 'width:300px;border-color:rgba(56,189,248,.5)' });
      c.innerHTML = `<div class="card-t" style="color:var(--ax0);font-size:12px">L3-11 导入期</div>
        <div class="card-d" style="font-size:11.5px">把 iota 从 <span class="mono">start_index_map</span><br>
          移到 <span class="mono">operand_batching_dims</span><br>
          → 分片<b>可表达</b></div>`;
      demo.appendChild(c);
      msg.innerHTML = 'L3-11 解决的问题：批维落在 <span class="mono">blocked_propagation</span> 因子上，<b>传播无法处理</b>。';
    });
    tl.at(4600, () => {
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹', style: 'font-size:26px' }));
      const c = U.el('div', { class: 'card', style: 'width:300px;border-color:rgba(74,222,128,.5)' });
      c.innerHTML = `<div class="card-t" style="color:var(--ok);font-size:12px">L4-06 导出期</div>
        <div class="card-d" style="font-size:11.5px">按规则插 reshard + all_reduce<br>
          → 分片<b>可落地</b></div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>本课</b>：显式化之后，gather 的分片能被正常处理 —— 插 reshard 与 <span class="mono">all_reduce</span>。';
    });
    tl.at(9000, () => {
      msg.innerHTML = '<b>两课合起来</b>：L3-11 让分片<b>可表达</b>，L4-06 让分片<b>可落地</b>。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>顺序不能反</b>：没有 L3-11 的显式化，L4-06 的这些 reshard <b>根本无从谈起</b>。';
    });
  }
},

/* ------------------------------------------------ 3 custom_call */
{
  kicker: 'L4-06 · 访存与通信类',
  title: '<span class="mono hl-a">custom_call</span>：内置注册 vs 用户规则，<span class="hl-a">同等对待</span>',
  sub: '24 个用例与 L2-10 的 `custom_call_*` **同名** —— 那里验证规则生成，这里验证 reshard 插入。',
  caption: '最关键的发现：用户为<b>未注册</b>算子写的规则，与内置注册的规则<b>受到同等对待</b>。',
  code: `// 【内置注册】按注册表的规则处理
func.func @custom_call_compact_wy_helper(
    %arg0: tensor<128x128xf32> {...<@mesh, [{"x"}, {"y"}]>})
    -> (tensor<128x128xf32> {...<@mesh, [{"y"}, {"x"}]>}) {   // 轴序相反
  %0 = stablehlo.custom_call @CompactWyHelper(%arg0)
       {...<@mesh, [{"y"}, {"x"}]>} : ...
}
// NOTE: sdy.sharding_rule = ([i, j])->([i, j]) {i=4, j=8}   <- 逐元素式
// 输出：
//   %[[CUSTOM_CALL]] = stablehlo.custom_call @CompactWyHelper(%arg0)
//       {...<@mesh, [{"x"}, {"y"}]>}    <- 保持输入的分片（与规则兼容）
//   %[[RESHARD]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x"}]>
//   ^^^^^^^^^^^ 结果轴序相反 -> 之后 reshard

// 【未注册 + 用户规则】同样处理！
func.func @unregistered_custom_call_with_existing_rule(
    %arg0: tensor<4x2xf32> {...[{"x"}, {"y"}]})
    -> (tensor<2x4xf32> {...[{"x":(1)2}, {"y"}]}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.custom_call @foo(%arg0)
       {sdy.sharding_rule = #sdy.op_sharding_rule<
          ([i, j])->([j, i]) {i=4, j=2}, custom>,      // <- 转置规则
        sdy.sharding = ...<@mesh, [{"x":(1)2}, {"y"}]>} : ...
}
// 输入 [{"x"}, {"y"}]、结果 [{"x":(1)2}, {"y"}]
// 按【转置规则】完全兼容 -> CHECK-NOT: sdy.reshard（一条都不插）！

// 22 个内置算子（与 L2-10 一致）：
//   CompactWyHelper / Eigh / Qr / QrDecompositionBlock / HouseholderProduct
//   X64Combine / X64SplitHigh / X64SplitLow
//   TopK(1d/2d) / Top2 / ApproxTopK / PartialReduce
//   MoveToDevice / MoveToHost / InspectSharding / LayoutConstraint / Erf
//   XlaMegascaleProvideMetadata`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '内置注册', c: '#38bdf8', n: '22 个',
        d: '注册表里写好了规则<br>按规则判定兼容性<br>不兼容就插 reshard' },
      { t: '未注册 + 用户规则', c: '#4ade80', n: '带 custom',
        d: '用户写 <span class="mono">sdy.sharding_rule</span><br><b>与内置同等对待</b><br>兼容时一条都不插' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>内置注册</b>覆盖了 XLA 常用的自定义算子 —— 这些算子没有标准语义，只能靠注册表。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>关键发现</b>：<span class="mono">@foo</span> 未注册，但用户的转置规则让它<b>完全兼容</b> → <span class="mono">CHECK-NOT: sdy.reshard</span>。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>这说明 <span class="mono">custom</span> 规则是"一等公民"</b> —— 与内置注册的规则走同一套兼容性判定。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>与 L2-10 的分工</b>：那里验证<b>规则生成</b>，这里验证<b>reshard 插入</b> —— 同名用例，两个视角。';
    });
  }
},

/* ------------------------------------------------ 4 ★ collective */
{
  kicker: 'L4-06 · 访存与通信类',
  title: '★ 集合通信算子<span class="hl-a">自身</span>也有分片规则',
  sub: '这是本课最反直觉的一点：`all_gather` 不是"分片的消费者"，而是**有自己的约束**的普通算子。',
  caption: '注意规则里的 <b><span class="mono">need_replication={j, k}</span></b> —— 这两个因子必须复制。',
  code: `func.func @all_gather(
    %arg0: tensor<2x2xi64> {...<@mesh, [{"y"}, {}]>},
    %arg1: tensor<2x2xi64> {...<@mesh, [{}, {"y"}]>})    // <- 与 %arg0 不同！
    -> (tensor<2x4xi64> {...<@mesh, [{"y"}, {}]>},
        tensor<2x4xi64> {...<@mesh, [{}, {"y"}]>}) {
  %0:2 = "stablehlo.all_gather"(%arg0, %arg1) {
    all_gather_dim = 1 : i64, ...
    sdy.sharding = ...<@mesh, [{"y"}, {}]>, <@mesh, [{}, {"y"}]>]>
  } : ...
}

// 规则（测试注释给出）：
// #sdy.op_sharding_rule<([i, j], [i, j])->([i, k], [i, k])
//   {i=2, j=2, k=4} need_replication={j, k}>
//                    ^^^^^^^^^^^^^^^^^^^^^^^^ 必须复制！

// 输出（三步）：
%0 = sdy.reshard %arg1 <@mesh, [{"y"}, {}]>
//   ^^^^^^^^^^^ ① 把 %arg1 拉到与 %arg0 同分片
%1:2 = "stablehlo.all_gather"(%arg0, %0)
       {...<@mesh, [{"y"}, {}]>, <@mesh, [{"y"}, {}]>]>}   // ② 两个结果都是 [{"y"}, {}]
%2 = sdy.reshard %1#1 <@mesh, [{}, {"y"}]>
//   ^^^^^^^^^^^ ③ %1#1 要求别的分片 -> 再 reshard 回去

// 【为什么 need_replication】
//   all_gather 在 all_gather_dim 上把各设备的数据【拼起来】
//   这要求每台设备都能看到【完整】的输入（否则拼不出完整结果）
//   j 是输入的非聚合维、k 是输出新增的聚合维

// 【对比 all_reduce】规则是 ([i,j],[i,j])->([i,j],[i,j])
//   输出形状与输入【完全相同】-> 没有 need_replication`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 对齐操作数', c: '#38bdf8', d: '两操作数必须<b>同分片</b><br><span class="mono">%arg1</span> reshard 到 <span class="mono">[{"y"}, {}]</span>' },
      { t: '② 执行', c: '#4ade80', d: '两结果都是 <span class="mono">[{"y"}, {}]</span><br>由 <span class="mono">need_replication</span> 决定' },
      { t: '③ 结果对齐', c: '#fbbf24', d: '<span class="mono">%1#1</span> 要求 <span class="mono">[{}, {"y"}]</span><br>再 reshard 回去' },
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
        '<b>反直觉</b>：两个<b>操作数</b>的分片也要一致 —— 不只是结果。',
        '<span class="mono">all_gather</span> 把各设备的数据<b>拼起来</b>，所以要看到完整输入。',
        '结果分片由规则决定，不一定等于目标 → 再 reshard。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>对比 <span class="mono">all_reduce</span></b>：它的规则是逐元素式（输出形状 = 输入形状），所以<b>没有</b> <span class="mono">need_replication</span>。';
    });
  }
},

/* ------------------------------------------------ 5 族谱 */
{
  kicker: 'L4-06 · 访存与通信类',
  title: '3 个文件 / 39 个用例的<span class="hl-a">族谱</span>',
  sub: '共同点：这三类算子都**不是透明的** —— 都有自己的分片约束。',
  caption: '这与 L4-03～05 的元素级/形状类形成对比：那些算子的约束更"局部"。',
  code: `// 【访存】gather_scatter        288 行 / 9 用例
//   与 dot 同构：reduction 因子 -> unreduced + all_reduce + reshard
//   与 L3-11 呼应：L3-11 让分片【可表达】，本课让分片【可落地】
//   大量使用【子轴】（mesh 有 4 个轴共 512 台设备）

// 【自定义调用】custom_call     318 行 / 24 用例
//   22 个内置注册 + 2 个未注册
//   关键：custom 规则与内置规则【同等对待】
//   与 L2-10 同名用例：那里看规则生成，这里看 reshard 插入

// 【集合通信】collective_ops    107 行 / 6 用例
//   all_gather / all_reduce / all_to_all_same_dimension
//   collective_broadcast / collective_permute / reduce_scatter
//   关键：集合通信算子【自身有分片规则】
//         all_gather 的 need_replication={j, k}
//         all_reduce 没有（输出形状 = 输入形状）

// 一句话：
//   这三类算子都【不是透明的】—— 都有自己的分片约束，
//   都需要显式的 reshard 来对齐`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '访存', n: 9, c: '#38bdf8' }, { t: '自定义调用', n: 24, c: '#4ade80' },
      { t: '集合通信', n: 6, c: '#fbbf24' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:150px;opacity:.4;transition:.35s;border-color:${f.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="card-t" style="color:${f.c};font-size:11.5px">${f.t}</div>
        <div class="big" style="font-size:20px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 130)); msg.innerHTML = '<b>39 个用例</b>分三族 —— 最大的一族是自定义调用（24 个）。'; });
    tl.at(4000, () => {
      msg.innerHTML = '<b>为什么 custom_call 最多</b>：内置注册了 22 个算子，每个都要验证 reshard 行为。';
    });
    tl.at(7400, () => {
      msg.innerHTML = '<b>共同点</b>：这三类算子都<b>不是透明的</b> —— 都有自己的分片约束。';
    });
    tl.at(10600, () => {
      msg.innerHTML = '<b>下一课</b> L4-07 讲结构类（数据流算子、manual_computation、call）—— L4 按算子族展开的最后一课。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L4-06 · 练习',
  title: '练一练：<span class="hl-a">三类算子的约束</span>',
  sub: '三道题分别考：gather 的四步、custom 规则的地位、集合通信的自身规则。',
  caption: '一句话总结：<b>这三类算子都不是透明的</b>。',
  code: `// 题 1：gather 的处理与哪个算子同构？为什么？

// 题 2：未注册的 custom_call 写了用户规则，会被怎样对待？

// 题 3：all_gather 的规则里为什么有 need_replication，
//       而 all_reduce 没有？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col compact-ex', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">gather</span> 的处理与哪个算子同构？为什么？',
        a: '与 <b><span class="mono">dot</span></b> 同构（L4-04）。' +
           '<br><b>理由</b>：两者的规则里都有 <b><span class="mono">reduction</span> 因子</b>（gather 是 <span class="mono">reduction={i, p}</span>）—— 沿归约因子切会产生<b>部分结果</b>，需要 <span class="mono">all_reduce</span>。' +
           '<br><span class="dim">四步链条：reshard 操作数 → 算子（标 <span class="mono">unreduced</span>）→ <span class="mono">all_reduce</span> → reshard 到目标。</span>'
      },
      {
        q: '未注册的 <span class="mono">custom_call</span> 写了用户规则，会被怎样对待？',
        a: '<b>与内置注册的规则同等对待</b> —— 都走同一套兼容性判定。' +
           '<br><b>证据</b>：用例 <span class="mono">unregistered_custom_call_with_existing_rule</span> 里，<span class="mono">@foo</span> 未注册但用户写了转置规则 <span class="mono">([i,j])-&gt;([j,i])</span>，按该规则输入输出<b>完全兼容</b> → <span class="mono">CHECK-NOT: sdy.reshard</span>（一条都不插）。' +
           '<br><span class="dim">这说明 <span class="mono">custom</span> 规则是"一等公民"，不是权宜之计。</span>'
      },
      {
        q: '<span class="mono">all_gather</span> 的规则里为什么有 <span class="mono">need_replication</span>，而 <span class="mono">all_reduce</span> 没有？',
        a: '<b>因为输出形状不同</b>。' +
           '<br><span class="mono">all_gather</span> 在 <span class="mono">all_gather_dim</span> 上把各设备的数据<b>拼起来</b> → 输出在聚合维上<b>变大</b>（<span class="mono">j=2</span> → <span class="mono">k=4</span>），这要求每台设备能看到<b>完整输入</b> → <span class="mono">need_replication={j, k}</span>。' +
           '<br><span class="mono">all_reduce</span> 的规则是 <span class="mono">([i,j],[i,j])-&gt;([i,j],[i,j])</span> —— 输出形状与输入<b>完全相同</b>，不需要拼接。' +
           '<br><span class="dim">反直觉点：集合通信算子<b>自身也有分片规则</b>，不是"分片的消费者"。</span>'
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
