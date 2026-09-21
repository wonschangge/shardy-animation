/* ==========================================================================
   L6-09 · exec-scatter-and-misc   （L6 收官）
   --------------------------------------------------------------------------
   覆盖：executable_convert_global_to_local/ 下 7 个文件
         (scatter ×3: 77/76/70 行；misc ×4: select_and_scatter 70,
          reduce_window 51, concatenate 49, sort 43)
   目标：讲透 scatter 的三种分片形态 + 四个 misc 算子的作用维判据。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 scatter 三形态 */
{
  kicker: 'L6-09 · scatter 与杂项',
  title: '★ <span class="mono hl-a">scatter</span> 的三种分片形态',
  sub: '三个文件的 `scatter_dimension_numbers` **几乎相同** —— 区别在**谁被分片**。',
  caption: 'L6 收官课 —— 讲完这课，L6 全层 10 课结束。',
  code: `// 【7 个文件】全在 executable_convert_global_to_local/
//   scatter ×3:  scatter_shard_indexed_inserted_dim  77 行
//                scatter_shard_implicit_dim           76 行
//                scatter_replicated_bounds            70 行
//   misc ×4:     stablehlo_select_and_scatter         70 行
//                stablehlo_reduce_window              51 行
//                stablehlo_concatenate                49 行
//                stablehlo_sort                       43 行

// 【三个 scatter 文件的 scatter_dimension_numbers 【几乎相同】】
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0], // Row dimension is collapsed/inserted
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
// -> 区别在【谁被分片】！

// 【★ 三种形态】
//   文件                          %arg0(bounds)   %arg1(索引)   %arg2(更新值)
//   shard_implicit_dim            【全复制】      切 {"x"}      切 {"x"}
//   shard_indexed_inserted_dim    【切 {"x"}】    无分片        无分片
//   replicated_bounds             【全复制】      切 {"x"}      切 {"x"}

// 【★ scatter 的三个操作数】（回顾 L5-08）
//   %arg0 = 【被更新的张量】（bounds）
//   %arg1 = 【索引】
//   %arg2 = 【更新值】
//
// 【scatter_dimension_numbers 的关键字段】
//   update_window_dims = [1]             更新值的第 1 维是【窗口维】
//   inserted_window_dims = [0]           第 0 维是【被插入(collapsed)的维】
//   scatter_dims_to_operand_dims = [0]   索引映射到操作数的第 0 维
//   index_vector_dim = 1                 索引向量的第 1 维是"索引分量维"

// 一句话：
//   scatter 的降级取决于"分片落在哪个操作数的哪个维上"`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'shard_implicit_dim', c: '#38bdf8', d: '分片在<br><b>索引数维</b><br><span class="dim">隐式批维</span>' },
      { t: 'shard_indexed_inserted_dim', c: '#fb7185', d: '分片在<br><b>被索引的维</b><br><span class="dim">最复杂</span>' },
      { t: 'replicated_bounds', c: '#4ade80', d: 'bounds<br><b>全复制</b>' },
      { t: 'misc ×4', c: '#fbbf24', d: '作用维<br><b>都未分片</b><br><span class="dim">全部无通信</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:9px;color:${x.c};overflow-wrap:anywhere">${x.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>七种情形</b> —— 三种 scatter 形态 + 四个 misc 算子。'; });
    tl.at(4400, () => {
      els.forEach((e, i) => { if (i !== 0 && i !== 1) e.style.opacity = '.25'; });
      msg.innerHTML = '<b>★ 核心对照</b>：<span class="mono">implicit_dim</span>（索引数维）vs <span class="mono">indexed_inserted_dim</span>（被索引的维）。';
    });
    tl.at(8000, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>三个 scatter 文件的 dimension_numbers 几乎相同</b> —— 区别只在<b>谁被分片</b>。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>L6 收官</b>：讲完这课，L6 全层 10 课结束。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 核心对照 */
{
  kicker: 'L6-09 · scatter 与杂项',
  title: '★ 隐式维 vs 显式维分片：<span class="hl-a">验收点</span>',
  sub: '**隐式维**各写各的；**显式维**需要**索引重映射**。',
  caption: '这是 TODOLIST 的验收点 —— 也是本课最重要的一处对照。',
  code: `// 【隐式维分片】shard_implicit_dim
  %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>},
  %arg1: tensor<2x1xi64> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg2: tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
// 【读法】
//   %arg0（4x2）【全复制】—— 每台设备都有【完整的】被更新张量
//   %arg1（2x1 索引）切 {"x"} -> 每台 1 个索引
//   %arg2（2x2 更新值）切 {"x"} -> 每台 1 行更新值
//   每台设备处理"自己那个索引"的写入
//     但写的是【同一份】%arg0 的副本
//   -> 结果需要合并（因为每台只做了一部分写入）
//
// 【★ "隐式维"的含义】
//   %arg1 的第 0 维（大小 2）是【索引的个数】
//   它在 scatter_dimension_numbers 里【没有对应的 operand 维】
//     （scatter_dims_to_operand_dims = [0] 指的是 %arg1 的【第 1 维】
//       映射到 %arg0 的第 0 维）
//   -> 所以第 0 维是【隐式的批维】

// 【显式维分片】shard_indexed_inserted_dim
  %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg1: tensor<2x1xi64>,
  %arg2: tensor<2x2xf32>)
  -> (tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
      inserted_window_dims = [0], // Row dimension is collapsed/inserted
// 【读法】与 shard_implicit_dim 【正好相反】
//   %arg0（4x2）【切 {"x"}】—— 【行维被切开】，每台 2 行
//   %arg1（索引）、%arg2（更新值）【都无分片】
//     —— 每台都有【完整的】索引与更新值
//   注释直接点明：// Row dimension is collapsed/inserted
//   【问题】索引可能指向【别的设备持有的行】-> 【需要索引重映射】！
//     设备 0 有第 0~1 行 -> 索引 0 或 1 属于它
//     设备 1 有第 2~3 行 -> 索引 2 或 3 属于它
//     索引 3 在设备 0 上【无法处理】-> 必须重映射或通信
//
// 【★ 这正是 L5-08 讲的"索引重映射"场景】
//   本课是它在 scatter 上的【完整执行验证】
//
// 【★ 两种形态的区别】
//   形态          分片位置                     含义
//   隐式维        %arg1/%arg2 的第 0 维        分片在"索引数"维
//   显式维        %arg0 的第 0 维              分片在被索引的 inserted 维
//
// 【为什么这个区分重要】
//   隐式维分片：每个设备处理【不同的索引/更新值】—— 各写各的位置
//   显式维分片：%arg0 的【行维被切开】—— 每台只持有【部分行】
//     而索引可能指向【别的设备持有的行】-> 需要索引重映射！`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '隐式维分片', c: '#38bdf8', d: '<b>索引数维</b>被切<br>bounds <b>全复制</b><br>各写各的位置' },
      { t: '显式维分片', c: '#fb7185', d: '<b>被索引的维</b>被切<br>索引<b>无分片</b><br>→ <b>需索引重映射</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>隐式维</b>：<span class="mono">%arg1</span> 的第 0 维是<b>索引的个数</b> —— 它没有对应的 operand 维。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>显式维</b>：<span class="mono">%arg0</span> 的行维被切开 —— 索引可能指向<b>别的设备</b>持有的行。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 所以显式维分片最复杂</b> —— 需要 L5-08 讲的<b>索引重映射</b>（八步）。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>注释直接点明</b>：<span class="mono">// Row dimension is collapsed/inserted</span> —— 上游测试的注释又一次给了答案。';
    });
  }
},

/* ------------------------------------------------ 3 replicated_bounds */
{
  kicker: 'L6-09 · scatter 与杂项',
  title: '<span class="mono hl-a">replicated_bounds</span>：边界全复制',
  sub: '分片位置与 `shard_implicit_dim` **相同**，但**用例意图不同**。',
  caption: '它强调「bounds 全复制」这个前提对**合并策略**的影响。',
  code: `  %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>},
  %arg1: tensor<2x1xi64> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg2: tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({

// 【读法】
//   %arg0（【bounds】，被更新张量）【全复制】[{}, {}] —— 用例名直接点明
//   %arg1/%arg2 切 {"x"}
//
// 【"replicated bounds" 的意义】
//   每台设备都有【完整的】被更新张量副本
//   各自写入自己那部分
//   -> 最后【需要合并】（因为写入是"叠加"语义）
//
// 【与 shard_implicit_dim 的区别】
//   分片位置【相同】，但【用例意图不同】——
//   本用例强调"bounds 全复制"这个前提对【合并策略】的影响
//
// 【回顾 L5-08 的 scatter 语义】
//   scatter 是【写操作】—— 多台设备可能写同一位置
//   所以它的"归约"语义用 update_computation（而非 all_reduce）
//   bounds 全复制时，每台都在【自己的副本】上写
//     -> 最终要把各副本的写入【合并】起来`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'bounds 全复制', c: '#4ade80', d: '每台都有<b>完整副本</b><br>各写自己那部分<br>→ <b>需要合并</b>' },
      { t: 'scatter 是写操作', c: '#fbbf24', d: '多台可能写<b>同一位置</b><br>用 <span class="mono">update_computation</span><br><span class="dim">而非 all_reduce</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>用例名直接点明</b>：<span class="mono">replicated_bounds</span> —— 被更新张量<b>全复制</b>。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>scatter 是写操作</b> —— 所以它的合并用 <span class="mono">update_computation</span>，而不是 <span class="mono">all_reduce</span>。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>与 <span class="mono">shard_implicit_dim</span> 的区别</b>：分片位置相同，但本用例强调<b>合并策略</b>。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>回顾 L5-08</b>：那里讲 scatter 的"写"语义与 <span class="mono">update_computation</span>。';
    });
  }
},

/* ------------------------------------------------ 4 ★ 四个 misc */
{
  kicker: 'L6-09 · scatter 与杂项',
  title: '★ 四个 misc 算子：作用维<span class="hl-a">都未分片</span>',
  sub: '拼接维 / 排序维 / 窗口维都在第 1 维，而分片在第 0 维 → **全部无通信**。',
  caption: '这是 <b>L5-04 作用维判据</b>在更多算子上的验证。',
  code: `// 【四个文件的共同点】
//   文件                          作用维                        分片维    通信
//   stablehlo_concatenate         拼接维 dim = 1                第 0 维   无
//   stablehlo_sort                排序维 dimension = 1          第 0 维   无
//   stablehlo_reduce_window       窗口维 window_dimensions=[1,2] 第 0 维   无
//   stablehlo_select_and_scatter  窗口维 window_dimensions=[1,2] 第 0 维   无

// 【concatenate：拼接维未分片】
func.func @parallel_concat(
  %arg0: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg1: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
  -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1
// 【读法】
//   dim = 1 —— 【拼接维是第 1 维】；而分片在【第 0 维】-> 【不冲突】✓
//   每个操作数 4x2 -> 局部 2x2；拼接后 2x4（全局 4x4）
//   【无通信】—— 每台设备在自己那段里做同样的拼接
// ★ 这正是 L5-04 的"作用维判据"
//   （那里讲 concatenate 的 sharded_non_concat_dim）

// 【sort：排序维未分片】
func.func @parallel_sort(
  %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
  -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = "stablehlo.sort"(%arg0) ({
  }) {dimension = 1 : i64, is_stable = true,
// 【读法】
//   dimension = 1 —— 【排序维是第 1 维】；分片在第 0 维 -> 【不冲突】✓
//   is_stable = true —— 稳定排序（相等元素的相对顺序保持）
//   【无通信】—— 每台设备独立排序自己那些行的第 1 维
// ★ 如果排序维被分片：排序需要【看到整行】才能确定顺序
//     跨设备时就要通信（类似 reverse 的情形）

// 【reduce_window：窗口维未分片】
func.func @parallel_reduce_window(%arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
    -> (tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{
    window_dimensions = array<i64: 1, 2>,
// 【读法】
//   window_dimensions = [1, 2] —— 【窗口在第 1 维】（大小 2）；分片在第 0 维
//   输入 2x4 -> 输出 2x2（窗口 2、步长 2）
//   【无通信】—— 窗口不跨设备（L5-07 讲的判据）

// 【select_and_scatter：窗口维未分片】
func.func @parallel_select_and_scatter(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
                                       %arg1: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
    -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %init) <{
    window_dimensions = array<i64: 1, 2>,
// 【读法】
//   window_dimensions = [1, 2] —— 窗口在第 1 维；分片在第 0 维
//   【无通信】
// ★ 回顾 L5-08：select_and_scatter 是 reduce_window 的【反向操作】
//   （从输出散射回窗口）`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'concatenate', c: '#38bdf8', d: '拼接维 <span class="mono">dim=1</span>' },
      { t: 'sort', c: '#4ade80', d: '排序维 <span class="mono">dimension=1</span>' },
      { t: 'reduce_window', c: '#fbbf24', d: '窗口维 <span class="mono">[1,2]</span>' },
      { t: 'select_and_scatter', c: '#c084fc', d: '窗口维 <span class="mono">[1,2]</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:10px;color:${x.c};overflow-wrap:anywhere">${x.t}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>拼接维在第 1 维</b>、分片在第 0 维 → 不冲突 → 每台做同样的拼接。',
        '<b>排序维在第 1 维</b> → 每台独立排序自己那些行。<b>若排序维被分片</b>，就需要看到整行 → 要通信。',
        '<b>窗口在第 1 维</b>、大小 2 → 窗口不跨设备（L5-07 的判据）。',
        '<b>同上</b> —— 且它是 <span class="mono">reduce_window</span> 的<b>反向操作</b>（L5-08）。',
      ][i];
    }));
    tl.at(15800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 四个算子的作用维都在第 1 维、分片都在第 0 维</b> → 全部无通信（L5-04 判据）。';
    });
  }
},

/* ------------------------------------------------ 5 小结 */
{
  kicker: 'L6-09 · scatter 与杂项',
  title: 'L6 <span class="hl-a">收官</span>：十课的主线',
  sub: 'L6 讲"分片代码怎么真正跑起来" —— 从机制到各类算子的执行验证。',
  caption: '完成后 L6 层 10 课全部结束，只剩 L7 四课。',
  code: `// 【L6 的十课】
//   L6-00 可执行测试机制（P0）        2 个 .sh（机制说明）
//   L6-01 sdy.* 集合通信的执行（P0）  10 个文件
//   L6-02 stablehlo 集合通信的执行     5 个文件
//   L6-03 卷积的执行                  5 个文件
//   L6-04 矩阵乘/fft/iota 的执行      4 个文件
//   L6-05 gather 的执行（P0 ★）       6 个文件
//   L6-06 pad 的执行（P0 ★最大族）   19 个文件
//   L6-07 reshape 的执行（P0）        7 个文件
//   L6-08 reverse/slice 的执行        9 个文件
//   L6-09 scatter 与杂项（本课）      7 个文件
//   ─────────────────────────────────────────
//   合计 74 个测试文件

// 【★ L6 的三条主线】
//   ① 可执行测试 = 分片版 vs 串行版的【数值对比】（L6-00）
//   ② 集合通信的两条路径：SDY 算子降级 vs 手写 stablehlo（L6-01/02）
//   ③ 各算子族的执行验证：
//      · 归约方向分片 -> 通信（L6-03/04/05）
//      · pad 是 halo exchange 的载体（L6-06）
//      · reshape 触发分片重排（L6-07）
//      · slice/reverse 看是否改变数据归属（L6-08）
//      · scatter 看分片落在哪个操作数的哪个维（L6-09）

// 【★ L6 的两个"实测发现"】
//   ① gather 的 mask 填充值是【归约的单位元】（L6-05）
//      sum -> 0、min -> +inf、max -> -inf
//      这修正了 L5-08 的"填零"表述（已单独提交 fix）
//   ② pad 的 uniform 判据是 pLow + pHigh 与 pInt 的关系（L6-06）

// 【★ L6 与前面层的关系】
//   L6 是【验证层】—— 它用真实数值验证 L4/L5 讲的规则
//     L4-02~10  reshard 与通信优化  -> L6-01/02 验证
//     L5-04     作用维判据          -> L6-04/08/09 验证
//     L5-05     数据并行            -> L6-04 验证
//     L5-06     卷积分片规则        -> L6-03 验证
//     L5-07     归约判据            -> L6-06/09 验证
//     L5-08     索引重映射          -> L6-05/09 验证
//     L5-09     整除性补齐          -> L6-06/07 验证

// 【下一层 L7】综合实战 —— 端到端 / 并行策略 / 调试 / 跨方言集成

// 一句话总结：
//   L6 用真实数值验证了 L4/L5 的全部规则
//   并有两个实测发现（填充值 = 单位元、pad 的 uniform 判据）`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:8px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: 'L6-00', n: 2, c: '#94a3b8' }, { t: 'L6-01', n: 10, c: '#38bdf8' },
      { t: 'L6-02', n: 5, c: '#0ea5e9' }, { t: 'L6-03', n: 5, c: '#22c55e' },
      { t: 'L6-04', n: 4, c: '#4ade80' }, { t: 'L6-05', n: 6, c: '#fbbf24' },
      { t: 'L6-06', n: 19, c: '#f59e0b' }, { t: 'L6-07', n: 7, c: '#c084fc' },
      { t: 'L6-08', n: 9, c: '#fb7185' }, { t: 'L6-09', n: 7, c: '#f472b6' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:96px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:7px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:9px;color:${f.c}">${f.t}</div>
        <div class="big" style="font-size:15px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 90)); msg.innerHTML = '<b>74 个测试文件</b>覆盖 L6 十课 —— <span class="mono">pad</span> 最大（19 个）。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>★ 三条主线</b>：可执行测试机制 / 通信的两条路径 / 各算子族的执行验证。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>★ 两个实测发现</b>：gather 填充值 = 单位元（修正了 L5-08）；pad 的 uniform 判据。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<b>下一层 L7</b>：综合实战 —— 端到端 / 并行策略 / 调试 / 跨方言集成（4 课）。';
    });
  }
},

];
