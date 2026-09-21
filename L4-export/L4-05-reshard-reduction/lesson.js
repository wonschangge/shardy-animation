/* ==========================================================================
   L4-05 · reshard-reduction
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/insert_explicit_reshards/ 下 4 个文件
         (reduce 112 / reduce_window 77 / sort 140 / rng 14) = 343 行 / 22 用例
   目标：讲透归约与排序类的受限维，以及"能不能靠归约补救"这个判据。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 reduce */
{
  kicker: 'L4-05 · 归约与排序类',
  title: '<span class="mono hl-a">reduce</span>：归约维切没切决定一切',
  sub: '规则与 `dot` 的收缩维**完全同构** —— 只是这里叫"归约维"。',
  caption: '判据：<b>被归约的维度上有没有分片</b>。有 → 产生部分和 → 需要 `all_reduce`。',
  code: `// 【情形 ①】归约维【未切】-> 保留其他维，之后 reshard
func.func @reduce_single_result_reduction_dim_not_sharded(
    %arg0: tensor<2x64x13xf32> {...<@mesh, [{"x"}, {}, {}]>}) -> tensor<2x13xf32> {
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1]
       : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
}
// 归约第 1 维，而 x 切第 0 维 -> 不冲突
// 输出：%[[REDUCE]] = stablehlo.reduce(...) {...[{"x"}, {}]...}
//       %[[RESHARD]] = sdy.reshard %[[REDUCE]] <@mesh, [{}, {}]>
//       ^^^^^^^^^^^ 结果无分片要求 -> 之后 reshard 掉

// 【情形 ②】归约维【被切】-> unreduced + all_reduce
// 输入 [{}, {"x"}, {}]  <- x 切的正是被归约的第 1 维
// 输出：%[[REDUCE]] = stablehlo.reduce(...) {...[{}, {}], unreduced={"x"}>}
//       %[[ALL_REDUCE]] = sdy.all_reduce {"x"} %[[REDUCE]] out_sharding=<@mesh, [{}, {}]>
//       ^^^^^^^^^^^^^ 每台设备只有【部分和】-> 需要归约

// 【多个归约维】
// across dimensions = [0, 1]，两维分别切 y 与 x
// 输出：unreduced={"x", "y"} + all_reduce {"x", "y"}   （一次归约两个轴）`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 归约维未切', c: '#4ade80', d: '保留其他维<br>结果不要就 reshard 掉<br><b>无需 all_reduce</b>' },
      { t: '② 归约维被切', c: '#fbbf24', d: '产生<b>部分和</b><br>标 <span class="mono">unreduced</span><br>+ <span class="mono">all_reduce</span>' },
      { t: '③ 多个归约维', c: '#38bdf8', d: '<span class="mono">unreduced={"x","y"}</span><br>一次归约两个轴' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>为什么不用归约</b>：归约维没被切，每台设备算的是<b>完整</b>的归约结果。',
        '沿归约维切 → 每台设备只算了一部分 → 必须 <span class="mono">all_reduce</span> 合并。',
        '两个轴一起归约 —— 因为它们都是被切过的归约维。',
      ][i];
    }));
    tl.at(12200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>与 dot 同构</b>：<span class="mono">dot</span> 的"收缩维"就是这里的"归约维"。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 选项对比 */
{
  kicker: 'L4-05 · 归约与排序类',
  title: '★ 两个 RUN 行的<span class="hl-a">对比</span>',
  sub: '`reduce.mlir` **跑两次**，分别用 `mark-partial-result-with-unreduced-axes=true/false`。',
  caption: '这是"<b>可观测性 vs 简洁性</b>"的取舍 —— 与 L2-03 的 <span class="mono">keep-sharding-rules</span> 同类。',
  code: `// 两个 RUN 行：
// RUN: ...='enable-full-version=true mark-partial-result-with-unreduced-axes=true' \\
//        | FileCheck %s --check-prefixes=CHECK,UNREDUCED
// RUN: ...='enable-full-version=true mark-partial-result-with-unreduced-axes=false' \\
//        | FileCheck %s --check-prefixes=CHECK,NOUNREDUCED

// 同一个用例，两组期望：
//   UNREDUCED:        %[[REDUCE]] = stablehlo.reduce(...)
//                     {...[{}, {}], unreduced={"x"}>}
//   NOUNREDUCED:      %[[REDUCE]] = stablehlo.reduce(...)
//   NOUNREDUCED-NOT:  sdy.sharding
//   CHECK-NEXT: %[[ALL_REDUCE]] = sdy.all_reduce {"x"} %[[REDUCE]] out_sharding=<@mesh, [{}, {}]>
//   CHECK-NEXT: return %[[ALL_REDUCE]]

// 对比：
//   选项         reduce 上的分片              all_reduce
//   ─────────────────────────────────────────────────────
//   true         [{}, {}], unreduced={"x"}    仍然插
//   false        完全没有分片属性              仍然插
//
// 读法：
//   两者【都会插 all_reduce】—— 归约这件事跑不掉
//   区别只在要不要在 IR 里显式标出"这个结果是部分和"
//   true  -> 信息完整（后续 pass 知道这里是部分和）
//   false -> IR 更干净`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'mark = true', c: '#4ade80', tag: 'UNREDUCED',
        d: 'reduce 标 <span class="mono">unreduced={"x"}</span><br><b>信息完整</b><br>后续 pass 知道是部分和' },
      { t: 'mark = false', c: '#94a3b8', tag: 'NOUNREDUCED',
        d: 'reduce 上<b>无分片属性</b><br>IR 更干净<br><span class="dim">但丢了"这是部分和"的信息</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:11.5px">${x.t} <span class="faint small">${x.tag}</span></div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>相同点</b>：两者<b>都会插 <span class="mono">all_reduce</span></b> —— 归约跑不掉。'; });
    tl.at(4400, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>唯一区别</b>：要不要在 <span class="mono">reduce</span> 上标出"这个结果是部分和"。';
    });
    tl.at(8200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>为什么需要两个选项</b>：不同后端/调试场景对"信息量"与"IR 简洁度"的偏好不同。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>同类设计</b>：L2-03 的 <span class="mono">keep-sharding-rules</span> —— 都是纯观测性选项，<b>不改变语义</b>。';
    });
  }
},

/* ------------------------------------------------ 3 ★ sort */
{
   kicker: 'L4-05 · 归约与排序类',
  title: '★ <span class="mono hl-a">sort</span>：被排序维<span class="hl-a">必须全复制</span>',
  sub: '这是本课最重要的规则 —— 因为排序**不能靠通信补救**。',
  caption: '这正是 L2-10 规则里 `sort` 的 <span class="mono">need_replication</span> 因子的来源。',
  code: `// 输入：tensor<4x32x8xi32>，第 0 维切 x
// sort 的 dimension = 0   <- 【排序维就是第 0 维】！
func.func @sort(%arg0: tensor<4x32x8xi32> {...<@mesh, [{"x"}, {}, {}]>}, ...) {
  %0:2 = "stablehlo.sort"(%arg0, %arg1) ({...})
       {dimension = 0 : i64, is_stable = true} : ...
}

// 输出（先搬走、再排序、再搬回）：
%[[RESHARD0]] = sdy.reshard %arg0 <@mesh, [{}, {"x"}, {}]>
%[[RESHARD1]] = sdy.reshard %arg1 <@mesh, [{}, {"x"}, {}]>
//              ^^^^^^^^^^^^ 把 x 从【排序维】搬到第 1 维
%[[SORT]]:2 = "stablehlo.sort"(%[[RESHARD0]], %[[RESHARD1]])
              {...[{}, {"x"}, {}]...}       // 排序维上无分片 ✓
%[[RESHARD2]] = sdy.reshard %2#0 <@mesh, [{}, {}, {}]>
%[[RESHARD3]] = sdy.reshard %2#1 <@mesh, [{?}, {?}, {?}]>
//              ^^^^^^^^^^^^ 排完再搬回（两个结果分片不同）

// 【为什么排序维不能分片】
//   排序是【全局操作】：元素位置取决于整条维上的所有元素
//   沿排序维分片 -> 每台设备只看到自己那一段 -> 无法确定排在哪里
//
// 【与 reduce 的关键区别】
//   reduce 沿归约维分片还能靠 all_reduce 补救（部分和可归约）
//   sort 【不能】靠通信补救（顺序不是可归约的量）-> 必须全复制`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 搬走分片', c: '#38bdf8', d: '把 <span class="mono">x</span> 从排序维<br>搬到别的维<br><span class="mono">[{"x"},{},{}]</span> → <span class="mono">[{},{"x"},{}]</span>' },
      { t: '② 排序', c: '#4ade80', d: '排序维上<b>无分片</b><br>每台设备看到完整维<br>顺序正确' },
      { t: '③ 搬回', c: '#fbbf24', d: '排完再 reshard<br>两个结果各一条<br>（分片可以不同）' },
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
        '<b>关键动作</b>：不是"让 sort 支持分片"，而是<b>把分片挪出排序维</b>。',
        '现在每台设备都能看到<b>完整的排序维</b>，可以正确确定顺序。',
        '排完再把分片搬回目标位置 —— 这次是两个结果各一条（它们的期望分片不同）。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>为什么不能像 reduce 那样补救</b>：顺序<b>不是可归约的量</b> —— 无法用 all_reduce 合并。';
    });
  }
},

/* ------------------------------------------------ 4 无处可搬 */
{
  kicker: 'L4-05 · 归约与排序类',
  title: '<span class="hl-a">无处可搬</span>时：只能全复制',
  sub: '如果其它维的大小都是 1，分片就**没有地方可挪** —— 唯一的选择是复制。',
  caption: '这就回答了 TODOLIST 的那个问题：<b>sort 为什么需要 need_replication 因子</b>。',
  code: `// 输入：tensor<1x4x1xi32>，第 1 维切 x
// 其它两维大小都是【1】！
func.func @sort_all_other_dims_size_one(
    %arg0: tensor<1x4x1xi32> {...<@mesh, [{}, {"x"}, {}]>})
    -> tensor<1x4x1xi32> {
  %0 = "stablehlo.sort"(%arg0) ({...}) {dimension = 1 : i64, ...} : ...
}

// 输出：
%[[RESHARD]] = sdy.reshard %arg0 <@mesh, [{}, {}, {}]> : tensor<1x4x1xi32>
//              ^^^^^^^^^^^^ 只能【全复制】
"stablehlo.sort"(%[[RESHARD]])

// 读法：
//   排序维是第 1 维（dimension = 1），它切了 x
//   第 0、2 维大小都是 1 -> 搬过去也没有意义（size-1 维上的分片会被删掉，
//                                      见 L3-10 的 remove_size_one_axes）
//   -> 只能 reshard 成 [{}, {}, {}]（全复制）

// 这就是 sort 的 need_replication 因子的来源：
//   分片无处安放时，唯一的选择就是【复制】
//
// 10 个用例里还有两个值得注意：
//   sort_incompatible_on_nonsort_dimensions   非排序维上不兼容
//   sort_compatible_on_nonsort_dimension      非排序维上兼容
//   -> 说明规则是【逐维】的：非排序维可以各切各的`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="small mono faint">tensor&lt;1x4x1&gt;</div>
        <div class="row" style="gap:5px">
          <div class="chip c3" style="padding:5px 9px;font-size:11px">1</div>
          <div class="chip c0" style="padding:5px 9px;font-size:11px">4 {"x"}</div>
          <div class="chip c3" style="padding:5px 9px;font-size:11px">1</div>
        </div>
        <div class="small faint">排序维是第 1 维（切了 x）</div>`;
      demo.appendChild(c);
      msg.innerHTML = '排序维切了 <span class="mono">x</span>，但<b>其它两维大小都是 1</b>。';
    });
    tl.at(4600, () => {
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹', style: 'font-size:24px' }));
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ok)">只能全复制</div>
        <div class="chip c4" style="padding:6px 12px;font-size:11.5px">[{}, {}, {}]</div>
        <div class="small faint">没有地方可以挪分片</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>搬到 size-1 维没有意义</b> —— 那里的分片会被删掉（L3-10 的 <span class="mono">remove_size_one_axes</span>）。';
    });
    tl.at(9000, () => {
      msg.innerHTML = '<b>唯一的选择</b>：reshard 成<b>全复制</b> <span class="mono">[{}, {}, {}]</span>。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>这就是答案</b>：sort 的 <span class="mono">need_replication</span> 因子 —— 分片无处安放时必须复制。';
    });
    tl.at(14200, () => {
      msg.innerHTML = '<b>注意规则是逐维的</b>：非排序维上可以各切各的（有专门的两个用例验证）。';
    });
  }
},

/* ------------------------------------------------ 5 其余两个 */
{
  kicker: 'L4-05 · 归约与排序类',
  title: '其余两个：<span class="mono hl-a">reduce_window</span> 与 <span class="mono hl-a">rng</span>',
  sub: '窗口规约的尺寸会变（permutation）；随机数生成器的**状态必须全复制**。',
  caption: 'RNG 的理由与 sort 同源：<b>状态是一条连续序列，不能靠归约合并</b>。',
  code: `// 【reduce_window】窗口会改变维度大小
func.func @reduce_window(
    %arg0: tensor<48x48x3xf32> {...<@mesh, [{"x"}, {}, {}]>},
    %arg1: tensor<48x48x3xi32>, ...)
    -> (tensor<16x48x1xf32> {...}, tensor<16x48x1xi32> {...}) {
// 测试注释给出规则：
//   sdy.sharding_rule = ([i,j,k],[i,j,k],[],[])->([i,j,k],[i,j,k])
//     {i=16, j=48, k=1} permutation={i, j, k}
// 读法：48x48x3 -> 16x48x1（窗口 + 步长 + padding）
//       【所有三个因子都是 permutation】—— 尺寸全变了
//       与 L2-10 的 conv 同理
// 输出：
//   %0 = sdy.reshard %arg1 <@mesh, [{"x"}, {}, {}]>   <- 第二个操作数先对齐
//   %1:2 = "stablehlo.reduce_window"(%arg0, %0, ...)
//   ^^^^ reduce_window 的两个操作数必须【同分片】

// 【rng_bit_generator】状态必须全复制
func.func @rng_bit_generator(
    %arg0: tensor<2xui64> {...<@mesh, [{"y"}]>}) -> tensor<2xui64> {
  %0, %output = stablehlo.rng_bit_generator %arg0, algorithm = DEFAULT
      {...<@mesh, [{"x":(1)2}]>, <@mesh, [{"y"}, {"x":(2)2}]>}>} : ...
}
// 输出：
//   %[[RESHARD1]] = sdy.reshard %arg0 <@mesh, [{}]>        <- 输入状态：全复制
//   %output_state, %output = stablehlo.rng_bit_generator %[[RESHARD1]], ...
//       {...<@mesh, [{}]>, <@mesh, [{"y"}, {"x":(2)2}]>}>}
//          ^^^^^^^^^ 输出状态全复制      ^^^^^^^^^^^^^^^^^^ 输出数据【可自由分片】
//   %[[RESHARD2]] = sdy.reshard %output_state <@mesh, [{"x":(1)2}]>
//   ^^^^^^^^^^^^ 下游 negate 要别的分片 -> 再 reshard

// 【为什么状态必须全复制】
//   随机数生成器的状态是一条【连续的序列】
//   若分片，每台设备会生成【不同的序列】-> 破坏"同一份随机数"的语义
//   -> 必须让每台设备都持有【完整的生成器状态】
// 输出数据可以分片：它是"从序列里取出的元素"，各设备取自己那份即可`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'reduce_window', c: '#38bdf8', n: '3 个',
        d: '窗口改变尺寸<br><b>全 permutation</b><br>两操作数必须同分片' },
      { t: 'rng_bit_generator', c: '#c084fc', n: '1 个',
        d: '<b>状态必须全复制</b><br>输出数据可自由分片<br><span class="dim">与 sort 同源</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:11.5px">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>与 reduce 的区别</b>：<span class="mono">reduce</span> 的规则是纯因子对应；<span class="mono">reduce_window</span> 因窗口而尺寸全变。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>RNG 的理由</b>：状态是一条<b>连续序列</b>，分片会让每台设备生成不同的序列。';
    });
    tl.at(8400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>注意区分</b>：<span class="mono">rng</span> 的<b>状态</b>要复制，但<b>输出数据</b>可以自由分片。';
    });
    tl.at(11600, () => {
      msg.innerHTML = '<b>与 sort 同源</b>：都是"结果不能靠归约合并"→ 只能复制。';
    });
  }
},

/* ------------------------------------------------ 6 族谱与判据 */
{
  kicker: 'L4-05 · 归约与排序类',
  title: '★ 判据：<span class="hl-a">能不能靠归约补救</span>',
  sub: '本课四个算子分成两类 —— 分开它们的是一条**统一的判据**。',
  caption: '这是 L4-05 最值得带走的一句话。',
  code: `// 【族谱】4 个文件 / 343 行 / 22 个用例
//   规约        reduce                          8 个
//   窗口规约    reduce_window_select_and_scatter  3 个
//   排序        sort                           10 个
//   随机数      rng_bit_generator               1 个

// 【两类算子】
//   ┌─────────────┬──────────┬──────────────────────┐
//   │ 算子        │ 受限维   │ 补救方式             │
//   ├─────────────┼──────────┼──────────────────────┤
//   │ reduce      │ 归约维   │ all_reduce（可补救） │
//   │ reduce_window│ 窗口维  │ permutation（尺寸变）│
//   │ sort        │ 排序维   │ 全复制（不可补救）   │
//   │ rng         │ 状态     │ 全复制（不可补救）   │
//   └─────────────┴──────────┴──────────────────────┘

// 【统一判据】
//   这个操作的结果【能不能靠"归约"合并】？
//     能（求和、取最大）        -> all_reduce
//     不能（顺序、随机序列）    -> 必须全复制
//
// 【为什么顺序不能归约】
//   all_reduce 的语义是"把多个值合并成一个"（加、乘、max、min...）
//   但"排序结果"不是可合并的量：
//     设备 A 排 [3,1] -> [1,3]
//     设备 B 排 [4,2] -> [2,4]
//   合并 [1,3] 与 [2,4] 得不到 [1,2,3,4] —— 因为缺少【跨设备比较】
//
// 【为什么随机序列不能归约】
//   同理：把两段不同的随机序列"合并"不出同一份随机数
//
// 一句话：
//   归约维可以"部分和 + all_reduce"补救；
//   排序维和 RNG 状态只能全复制`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: 'reduce', n: 8, c: '#4ade80', tag: 'all_reduce' },
      { t: 'reduce_window', n: 3, c: '#38bdf8', tag: 'permutation' },
      { t: 'sort', n: 10, c: '#fbbf24', tag: '全复制' },
      { t: 'rng', n: 1, c: '#c084fc', tag: '全复制' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:150px;opacity:.4;transition:.35s;border-color:${f.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="card-t mono" style="color:${f.c};font-size:10.5px;overflow-wrap:anywhere">${f.t}</div>
        <div class="big" style="font-size:18px;color:${f.c};margin-top:2px">${f.n}</div>
        <div class="small" style="font-size:9.5px;color:${f.c};margin-top:2px">${f.tag}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 130)); msg.innerHTML = '<b>22 个用例</b>分四族 —— 后两族都只能"全复制"。'; });
    tl.at(4200, () => {
      els.forEach((e, i) => { if (i < 2) e.style.opacity = '.25'; });
      msg.innerHTML = '<b>关键区别</b>：<span class="mono">sort</span> 与 <span class="mono">rng</span> <b>不能</b>靠通信补救。';
    });
    tl.at(7600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>统一判据</b>：结果<b>能不能靠"归约"合并</b>？能 → all_reduce；不能 → 全复制。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<b>为什么顺序不能归约</b>：<span class="mono">all_reduce</span> 合并的是"值"，而排序缺少<b>跨设备比较</b>。';
    });
    tl.at(13800, () => {
      msg.innerHTML = '<b>下一课</b> L4-06 讲访存类与自定义调用（gather/scatter、custom_call、collective）。';
    });
  }
},

/* ------------------------------------------------------------ 7 练习 */
{
  kicker: 'L4-05 · 练习',
  title: '练一练：<span class="hl-a">该插什么</span>',
  sub: '三道题分别考：reduce 的两类情形、sort 的规则、统一判据。',
  caption: '一句话总结：<b>能归约就 all_reduce，不能就全复制</b>。',
  code: `// 题 1：reduce 归约的是第 1 维，而 x 切第 0 维，需要 all_reduce 吗？

// 题 2：sort 的排序维上切了 x，怎么办？

// 题 3：为什么 sort 和 rng 不能用 all_reduce 补救？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col compact-ex', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">reduce</span> 归约的是第 1 维，而 <span class="mono">x</span> 切第 0 维，需要 <span class="mono">all_reduce</span> 吗？',
        a: '<b class="badge ok">不需要</b>。归约维（第 1 维）没被切 → 每台设备算的是<b>完整</b>的归约结果。' +
           '<br><span class="mono">reduce</span> 保留第 0 维的 <span class="mono">x</span>；若结果不要分片，<b>之后</b>插一条 <span class="mono">reshard</span> 即可。' +
           '<br><span class="dim">对比：若 <span class="mono">x</span> 切的正是被归约的那一维 → 产生部分和 → 必须 <span class="mono">all_reduce</span>。</span>'
      },
      {
        q: '<span class="mono">sort</span> 的排序维上切了 <span class="mono">x</span>，怎么办？',
        a: '<b>把 <span class="mono">x</span> 搬到别的维上</b>，在"排序维无分片"的状态下排序，排完再搬回。' +
           '<br><b>为什么</b>：排序是<b>全局操作</b>，元素位置取决于整条维上的所有元素 —— 沿排序维分片就无法确定顺序。' +
           '<br><span class="dim">若其它维大小都是 1（如 <span class="mono">tensor&lt;1x4x1&gt;</span>），无处可搬 → 只能<b>全复制</b>。这就是 <span class="mono">need_replication</span> 因子的来源。</span>'
      },
      {
        q: '为什么 <span class="mono">sort</span> 和 <span class="mono">rng</span> 不能用 <span class="mono">all_reduce</span> 补救？',
        a: '<b>统一判据</b>：结果<b>能不能靠"归约"合并</b>？' +
           '<br><span class="mono">all_reduce</span> 的语义是"把多个值合并成一个"（加、乘、max、min）—— 但：' +
           '<br>• <b>排序结果</b>不是可合并的量：设备 A 排 <span class="mono">[3,1]</span>、设备 B 排 <span class="mono">[4,2]</span>，合并得不到 <span class="mono">[1,2,3,4]</span>（缺少<b>跨设备比较</b>）。' +
           '<br>• <b>随机序列</b>同理：两段不同的序列合不出同一份随机数。' +
           '<br><span class="dim">所以只能<b>全复制</b> —— 让每台设备都持有完整的数据/状态。</span>'
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
