/* ==========================================================================
   L4-08 · reshard-to-collectives   （L4 ★ 核心课）
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/reshard_to_collectives.mlir (960 行 / 98 用例)
         + reshard_to_collectives_keep_redundant_reshards_true.mlir (27 / 3)
   目标：讲透"一条 reshard 如何分解成集合通信"的核心对照表。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 ★ 对照表 */
{
  kicker: 'L4-08 · reshard 转集合通信',
  title: '★ 核心对照表：<span class="hl-a">分片变化 → 通信</span>',
  sub: '这是本课最重要的产出。**每一维上的分片变化，对应一种通信。**',
  caption: '五条规则覆盖了 98 个用例里的绝大多数情形。',
  code: `// RUN: sdy_opt %s -sdy-reshard-to-collectives

// ① 去掉某维的轴 -> all_gather
%arg0: [{"y"}, {"x"}]  ->  [{"y"}, {}]
sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh2d, [{"y"}, {}]>
//               ^^^^^^^^^ 参数是"要聚合掉的轴的位置"

// ② 加上某维的轴 -> all_slice
%arg0: [{}, {}]  ->  [{"x"}, {"y", "z"}]
sdy.all_slice [{"x"}, {"y", "z"}] %arg0 out_sharding=<@mesh3d, [{"x"}, {"y", "z"}]>
//             ^^^^^^^^^^^^^^^^^^ 参数是【目标分片】

// ③ 轴从一个维【移到】另一个维 -> all_to_all
%arg0: [{"x"}, {"y"}, {}]  ->  [{}, {"y"}, {"x"}]
sdy.all_to_all [{"x"}: 0->2] %arg0 out_sharding=<@mesh3d, [{}, {"y"}, {"x"}]>
//              ^^^^^^^^^^^^ 参数是"哪个轴：从哪维到哪维"

// ④ 轴在两个维之间【交换】 -> collective_permute
%arg0: [{"x"}, {"y"}]  ->  [{"y"}, {"x"}]      （尺寸相同！）
sdy.collective_permute %arg0 out_sharding=<@mesh2d, [{"y"}, {"x"}]>

// ⑤ 前后分片【相同】 -> 无（直接删除）
%0 = sdy.reshard %arg0 <@mesh2d, [{}, {}]> : ...
return %0
// 输出：return %arg0                            <- reshard 整条消失`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const rules = [
      { t: '去掉轴', c: '#4ade80', op: 'all_gather' },
      { t: '加上轴', c: '#38bdf8', op: 'all_slice' },
      { t: '轴跨维移动', c: '#fbbf24', op: 'all_to_all' },
      { t: '轴互换', c: '#c084fc', op: 'collective_permute' },
      { t: '无变化', c: '#94a3b8', op: '删除' },
    ];
    const host = wrap.querySelector('#cards');
    const els = rules.map(r => {
      const e = U.el('div', { class: 'card', style: `width:140px;opacity:.35;transition:.3s;border-color:${r.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div style="font-size:11.5px;color:${r.c}">${r.t}</div>
        <div class="mono" style="font-size:9.5px;color:#bdf7ec;margin-top:4px;overflow-wrap:anywhere">${U.esc(r.op)}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    rules.forEach((r, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.35');
      msg.innerHTML = [
        '<span class="mono">all_gather</span> 把轴"<b>收掉</b>"—— 参数是<b>要聚合的轴的位置</b>。',
        '<span class="mono">all_slice</span> 把轴"<b>切开</b>"—— 参数是<b>目标分片</b>。两者<b>互逆</b>。',
        '<b>注意轴没有消失</b>，只是换了位置 —— 参数写成 <span class="mono">[{"x"}: 0->2]</span>。',
        '<b>交换</b>是双向的移动，用 permute 更直接。前提是<b>尺寸相同</b>。',
        '<b>冗余消除</b>：前后分片一致时整条 reshard 被删掉，不产生任何通信。',
      ][i];
    }));
    tl.at(16400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>一句话</b>：每一维上的分片变化，对应一种通信。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 冗余消除 */
{
  kicker: 'L4-08 · reshard 转集合通信',
  title: '★ 冗余 reshard 的<span class="hl-a">消除</span>',
  sub: '前后分片相同 → **整条 reshard 被删掉**，不产生任何通信。',
  caption: '最有意思的是：<b>跨 mesh</b>、甚至<b>设备序不同</b>，只要都是全复制就仍然冗余。',
  code: `// 【基础】前后都是 [{}, {}]
func.func @redundant_reshard_fully_replicated(
    %arg0 : tensor<16x8xf32> {...<@mesh2d, [{}, {}]>}) -> tensor<16x8xf32> {
  %0 = sdy.reshard %arg0 <@mesh2d, [{}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
// 输出：// CHECK-NEXT: return %arg0        <- reshard 整条消失

// 【跨 mesh】输入 @mesh2d_2x3（6 台），reshard 到 @mesh1d_6（也是 6 台）
func.func @redundant_reshard_fully_replicated_different_meshes(
    %arg0 : tensor<16x8xf32> {...<@mesh2d_2x3, [{}, {}]>}) -> tensor<16x8xf32> {
  %0 = sdy.reshard %arg0 <@mesh1d_6, [{}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
// 输出：// CHECK-NEXT: return %arg0        <- 仍然冗余！

// 【设备序不同】@mesh2d vs @mesh2d_non_iota（device_ids=[3,2,1,0]）
func.func @redundant_reshard_fully_replicated_same_mesh_different_device_ids(
    %arg0 : tensor<16x8xf32> {...<@mesh2d, [{}, {}]>}) -> tensor<16x8xf32> {
  %0 = sdy.reshard %arg0 <@mesh2d_non_iota, [{}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
// 输出：// CHECK-NEXT: return %arg0        <- 仍然冗余！

// 关键洞察：
//   全复制状态下，网格的具体形状【无关紧要】
//   数据在每台设备上都有一份完整副本 -> 换网格不产生任何通信
//   所以 L4-07 说的"设备序不同需要 reshard"，在全复制状态下【不适用】`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 基础', c: '#4ade80', d: '同 mesh 同分片<br>→ 删除' },
      { t: '② 跨 mesh', c: '#38bdf8', d: '<span class="mono">@mesh2d_2x3</span> → <span class="mono">@mesh1d_6</span><br>都是全复制 → 删除' },
      { t: '③ 设备序不同', c: '#fbbf24', d: '<span class="mono">@mesh2d</span> → <span class="mono">@mesh2d_non_iota</span><br>仍然 → 删除' },
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
        '最简单的情形：前后分片一模一样，reshard 是纯粹的冗余。',
        '<b>为什么跨 mesh 也冗余</b>：全复制意味着每台设备都有<b>完整副本</b> —— 换网格不需要搬任何数据。',
        '<b>这修正了 L4-07 的结论</b>：设备序不同<b>通常</b>需要 reshard，但<b>全复制状态下不需要</b>。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>关键洞察</b>：全复制状态下，网格的具体形状<b>无关紧要</b>。';
    });
  }
},

/* ------------------------------------------------ 3 复杂组合 */
{
  kicker: 'L4-08 · reshard 转集合通信',
  title: '复杂情形：<span class="hl-a">先 slice 再 all_to_all</span>',
  sub: '`all_to_all` 要求源维与目标维的轴**尺寸匹配**。不匹配时要先 `all_slice` 调整。',
  caption: '这一族有 <b>10 个用例</b>，还有 <b>9 个"不能 slice"</b>的用例锁定失败条件。',
  code: `// 【模式】先 slice 再 all_to_all（10 个用例）
//   slice_on_src_dim_then_all_to_all
//   slice_on_src_dim_then_all_to_all_multiple_axes
//   slice_on_src_dim_then_two_all_to_alls
//   slice_on_src_dim_then_two_all_to_alls_diff_tgts
//   slice_on_src_dim_then_all_to_all_and_all_gather
//   slice_on_multiple_src_dims
//   slice_on_one_src_dim_but_not_other
//   slice_on_src_dim_considering_existing_axes_on_src_dim
//   slice_on_src_dim_and_replace_axis_in_another_dim

// 【9 个"不能 slice"的用例】锁定失败条件：
//   cannot_slice_on_src_dim_output_sharded          输出维上已有分片
//   cannot_slice_on_src_dim_tgt_dim_sharded         目标维上已有分片
//   cannot_slice_on_src_dim_axes_out_of_order       轴序不对
//   cannot_slice_on_src_dim_axes_non_contiguous     轴不连续
//   cannot_slice_on_src_dim_size_too_small（×2）    尺寸太小
//   cannot_slice_on_src_dim_considering_existing_axes_on_src_dim
//   cannot_slice_on_src_dim_size_non_divisible      尺寸不可整除

// 为什么这些用例重要：
//   "先 slice"【不是万能的】—— 条件不满足时要换别的策略（或放弃）
//   这类"负面用例"与正面用例【同等重要】`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '正面：先 slice', c: '#4ade80', n: '10 个',
        d: '尺寸不匹配时<br>先 <span class="mono">all_slice</span> 调整<br>再 <span class="mono">all_to_all</span>' },
      { t: '负面：不能 slice', c: '#fb7185', n: '9 个',
        d: '输出维已有分片 / 目标维已有分片<br>轴序不对 / 轴不连续<br>尺寸太小 / 不可整除' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>为什么要先 slice</b>：<span class="mono">all_to_all</span> 要求源维与目标维的轴<b>尺寸匹配</b>。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>9 个负面用例</b>说明"先 slice"<b>不是万能的</b> —— 条件不满足时要换策略。';
    });
    tl.at(8400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>测试设计的原则</b>：<b>负面用例与正面用例同等重要</b> —— 它们锁定了"什么时候不能用这个策略"。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>读法建议</b>：<span class="mono">cannot_*</span> 开头的用例名本身就是条件清单。';
    });
  }
},

/* ------------------------------------------------ 4 三类轴操作 */
{
  kicker: 'L4-08 · reshard 转集合通信',
  title: '三类<span class="hl-a">轴操作</span>：replace / swap / reorder',
  sub: '这三个词很容易混 —— 它们的区别在于**轴的来源与去向**。',
  caption: '还有一类特殊的：<b>设备号重排</b>（<span class="mono">reorder_device_ids</span>）。',
  code: `// 【replace】用一个轴【替换】另一个（26+ 个用例）
//   replace_same_size_axes_same_dim        同尺寸、同维
//   replace_smaller_axis_with_bigger_same_dim   小换大
//   replace_bigger_axis_with_smaller_same_dim   大换小
//   replace_major_most_axis_in_dim         替换最 major 的轴
//   replace_major_most_axis_then_all_gather     替换后再 all_gather
//   replace_same_size_axes_diff_dims       同尺寸、跨维
//   replace_multiple_axes_diff_dims        多轴跨维

// 【swap】两个维的轴【互换】（6 个用例）
//   swap_same_size_axes_between_dims       同尺寸跨维互换 -> collective_permute
//   swap_diff_size_axes_between_dims       不同尺寸
//   slice_and_swap_axes_between_dims       先 slice 再 swap
//   swap_axes_between_dims_then_all_to_all 先 swap 再 all_to_all
//   slice_sub_axes_then_swap_between_dims
//   swap_sub_axes_then_all_to_all_and_all_gather

// 【reorder】同一个维内轴的【顺序】变化（8 个用例）
//   reorder_axes_single_dim                单维内重排
//   reorder_axes_across_dims               跨维重排
//   slice_then_reorder_axes
//   reorder_axes_for_all_gather
//   reorder_axes_for_all_to_all_then_all_gather_single_axis
//   reorder_axes_for_all_to_all_then_all_gather_remaining_axes
//   all_to_all_axes_at_src_out_of_order
//   all_to_all_axes_at_src_and_tgt_out_of_order

// 【设备号重排】4 个用例
//   reorder_device_ids
//   reorder_axes_and_device_ids
//   reorder_device_ids_then_all_gather
//   slice_then_reorder_axes_and_device_ids
//   slice_then_reorder_device_ids_then_all_to_all
//   reorder_device_ids_then_two_all_to_alls

// 【关键区分】
//   swap   是【跨维】的（两个维之间）
//   reorder 可以是【同维内】的（一个维里轴的顺序）
//   replace 是"一个轴被另一个【取代】"

// 【gcd > 1】4 个用例
//   replace_smaller_axis_with_bigger_gcd_greater_than_one
//   replace_bigger_axis_with_smaller_gcd_greater_than_one
//   replace_axes_and_all_gather_gcd_greater_than_one（×3）
//   含义：两轴大小的最大公约数 > 1 时可以【部分匹配】
//         （用子轴切出公共部分），不必整体 slice —— 更优的策略`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'replace', c: '#4ade80', n: '26+', d: '一个轴被<b>另一个取代</b><br>同维 / 跨维<br>同尺寸 / 更大 / 更小' },
      { t: 'swap', c: '#38bdf8', n: '6', d: '两个维的轴<b>互换</b><br><b>跨维</b>操作' },
      { t: 'reorder', c: '#fbbf24', n: '8', d: '同维内轴的<b>顺序</b>变化<br>也可以是跨维的' },
      { t: '设备号重排', c: '#c084fc', n: '6', d: '<span class="mono">reorder_device_ids</span><br>配合 all_gather / all_to_all' },
      { t: 'gcd > 1', c: '#fb7185', n: '4', d: '最大公约数 > 1 时可<b>部分匹配</b><br>不必整体 slice' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:140px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div style="font-size:11.5px;color:${x.c}">${x.t}</div>
        <div class="mono" style="font-size:14px;color:${x.c};margin:2px 0">${x.n}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '最大的一族（26+）—— 因为"替换"的组合最多（尺寸关系 × 维度关系）。',
        '<b>swap 是跨维的</b>：两个维之间互换，通常对应 <span class="mono">collective_permute</span>。',
        '<b>reorder 可以是同维内的</b>：一个维里轴的顺序变化 —— 这是它与 swap 的关键区别。',
        '<b>设备号重排</b>是另一类：轴没变，但设备与数据的对应关系变了。',
        '<b>gcd > 1 是优化</b>：能部分匹配就不必整体 slice —— 通信量更小。',
      ][i];
    }));
    tl.at(16400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>记忆法</b>：<span class="mono">swap</span> 跨维、<span class="mono">reorder</span> 同维内、<span class="mono">replace</span> 是取代。';
    });
  }
},

/* ------------------------------------------------ 5 族谱与选项 */
{
  kicker: 'L4-08 · reshard 转集合通信',
  title: '101 个用例的<span class="hl-a">族谱</span>与一个选项',
  sub: '第二个文件用 `keep-redundant-reshards=true` —— **不删除**冗余的 reshard。',
  caption: '这与 L2-03 的 <span class="mono">keep-sharding-rules</span>、L4-05 的 <span class="mono">mark-partial-result</span> 同类：<b>观测性选项</b>。',
  code: `// 【族谱】101 个用例
//   冗余消除              7    前后分片相同（含跨 mesh / 设备序不同）
//   分片 -> 全复制         3    跨 mesh 的版本
//   all_gather            4    去掉轴
//   all_slice             4    加上轴
//   all_to_all            8    轴跨维移动
//   slice + all_to_all   10    尺寸不匹配时的组合
//   不能 slice            9    锁定失败条件
//   replace/swap/reorder 26+   三类轴操作
//   设备号重排            6    配合通信算子
//   gcd > 1               4    部分匹配的优化
//   边界                  4    unreduced / barrier / 单设备

// 【第二个文件】reshard_to_collectives_keep_redundant_reshards_true.mlir
// RUN: sdy_opt %s -sdy-reshard-to-collectives='keep-redundant-reshards=true'
// 27 行 / 3 个用例
// 作用：【不删除】冗余的 reshard（默认会删）
//
// 为什么需要：
//   冗余 reshard 通常被删除（本课第二节）
//   但调试时你可能想看到"传播到底插了哪些 reshard"
//   保留它们可以看清全貌
//
// 同类选项（都是【观测性】的，不改变语义）：
//   L2-03  keep-sharding-rules              保留分片规则
//   L4-05  mark-partial-result-with-unreduced-axes  标记部分和
//   本课   keep-redundant-reshards          保留冗余 reshard`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '冗余消除', n: 7, c: '#94a3b8' }, { t: '→ 全复制', n: 3, c: '#64748b' },
      { t: 'all_gather', n: 4, c: '#4ade80' }, { t: 'all_slice', n: 4, c: '#38bdf8' },
      { t: 'all_to_all', n: 8, c: '#fbbf24' }, { t: 'slice+all_to_all', n: 10, c: '#c084fc' },
      { t: '不能 slice', n: 9, c: '#fb7185' }, { t: '轴操作', n: 26, c: '#f472b6' },
      { t: '设备号重排', n: 6, c: '#93c5fd' }, { t: 'gcd>1', n: 4, c: '#5eead4' },
      { t: '边界', n: 4, c: '#fdba74' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:100px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:7px;text-align:center` });
      e.innerHTML = `<div style="font-size:9.5px;color:${f.c};line-height:1.3">${f.t}</div>
        <div class="mono" style="font-size:15px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 90)); msg.innerHTML = '<b>101 个用例</b>分 11 族 —— 最大的是"轴操作"（26+）。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>第二个文件</b>用 <span class="mono">keep-redundant-reshards=true</span> —— 不删除冗余 reshard。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>用途</b>：调试时看清"传播到底插了哪些 reshard"。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>同类选项</b>：L2-03 的 <span class="mono">keep-sharding-rules</span>、L4-05 的 <span class="mono">mark-partial-result</span> —— 都是<b>观测性</b>的，不改变语义。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L4-08 · 练习',
  title: '练一练：<span class="hl-a">写出 collective 序列</span>',
  sub: '三道题分别考：对照表、冗余消除、复杂组合。',
  caption: '一句话总结：<b>每一维上的分片变化，对应一种通信</b>。',
  code: `// 题 1：[{"y"},{"x"}] -> [{"y"},{}] 对应什么通信？

// 题 2：[{"x"},{"y"},{}] -> [{},{"y"},{"x"}] 对应什么？
//       与 [{"x"},{"y"}] -> [{"y"},{"x"}] 有什么区别？

// 题 3：输入在 @mesh2d_2x3 全复制，reshard 到 @mesh1d_6 全复制，
//       会产生通信吗？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col compact-ex', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">[{"y"},{"x"}]</span> → <span class="mono">[{"y"},{}]</span> 对应什么通信？',
        a: '<b><span class="mono">sdy.all_gather [{}, {"x"}]</span></b>。' +
           '<br><b>规则</b>：第 1 维从 <span class="mono">{"x"}</span> 变成 <span class="mono">{}</span> —— <b>去掉了轴</b> → <span class="mono">all_gather</span>。' +
           '<br><span class="dim">参数是"<b>要聚合掉的轴的位置</b>"。反过来（加上轴）用 <span class="mono">all_slice</span>，参数是<b>目标分片</b>。两者<b>互逆</b>。</span>'
      },
      {
        q: '<span class="mono">[{"x"},{"y"},{}]</span> → <span class="mono">[{},{"y"},{"x"}]</span> 对应什么？与 <span class="mono">[{"x"},{"y"}]</span> → <span class="mono">[{"y"},{"x"}]</span> 有什么区别？',
        a: '前者用 <b><span class="mono">sdy.all_to_all [{"x"}: 0->2]</span></b> —— 轴<b>从第 0 维移到第 2 维</b>（轴没消失）。' +
           '<br>后者用 <b><span class="mono">sdy.collective_permute</span></b> —— 轴在两个维之间<b>互换</b>（尺寸相同）。' +
           '<br><span class="dim">区别：<span class="mono">all_to_all</span> 是单向的移动（参数写明 <span class="mono">0->2</span>）；<span class="mono">permute</span> 是双向的交换。交换的前提是<b>尺寸相同</b>。</span>'
      },
      {
        q: '输入在 <span class="mono">@mesh2d_2x3</span> 全复制，reshard 到 <span class="mono">@mesh1d_6</span> 全复制，会产生通信吗？',
        a: '<b class="badge ok">不会</b> —— 这是<b>冗余 reshard</b>，会被<b>整条删除</b>（输出里直接 <span class="mono">return %arg0</span>）。' +
           '<br><b>理由</b>：全复制意味着每台设备都有<b>完整副本</b> —— 换网格不需要搬任何数据。两个网格都是 6 台设备，形状不同但都是全复制。' +
           '<br><span class="dim">同理：即使<b>设备序不同</b>（如 <span class="mono">device_ids=[3,2,1,0]</span>），只要都是全复制就仍然冗余。这修正了 L4-07 的结论——"设备序不同需要 reshard"在全复制状态下<b>不适用</b>。</span>'
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
