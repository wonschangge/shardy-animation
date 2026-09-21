/* ==========================================================================
   L4-15 · export-finalize   （P0）
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/ 下 7 个文件
         (close_shardings 92 / update_non_divisible 182 / remove_sub_axes 108
          drop_sharding_rules 35 / remove_sharding_groups 12
          inline_meshes 138 / drop_sharding_and_mesh 26) = 593 行 / 58 用例
   目标：讲透收尾 7 个 pass 的目的与顺序。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 全景 */
{
  kicker: 'L4-15 · 导出收尾',
  title: '收尾阶段：<span class="hl-a">全都在"删东西"</span>',
  sub: '7 个 pass 把 IR 清理成交给后端的最终形态 —— 目标是**干净且自包含**。',
  caption: '这是 L4 导出流水线的<b>最后阶段</b>（P0 优先级）。',
  code: `// 7 个收尾 pass，合计 593 行 / 58 用例：

// 【① 分片本身】三个 pass
//   -sdy-close-shardings                         92 行 / 15 用例
//     闭合【开维】(?) —— 传播期保留 ? 便于调整，导出时必须定下来
//   -sdy-update-non-divisible-input-output-shardings  182 行 / 21 用例
//     不可整除的输入输出分片 -> 截断到可整除前缀
//   -sdy-remove-sub-axes-in-input-output-shardings    108 行 /  6 用例
//     边界上只保留最粗的轴（移除子轴）

// 【② 辅助信息】两个 pass
//   -sdy-drop-sharding-rules                     35 行 /  4 用例
//     删 sdy.sharding_rule（传播已结束）
//   -sdy-remove-sharding-groups                  12 行 /  1 用例
//     删 sdy.sharding_group（承诺已兑现）

// 【③ 网格】两个 pass
//   -sdy-inline-meshes                          138 行 /  9 用例
//     命名网格 -> 内联（自包含）
//   -sdy-drop-sharding-and-mesh                  26 行 /  2 用例
//     全部删掉（纯 StableHLO，可选）

// 【共同点】这 7 个 pass 全都在【删东西】
//   把传播/导出期用的辅助信息清理掉
//   交给后端一个【干净】的 IR`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 分片本身', c: '#38bdf8', n: '3 个',
        d: '<span class="mono">close_shardings</span><br><span class="mono">update_non_divisible</span><br><span class="mono">remove_sub_axes</span>' },
      { t: '② 辅助信息', c: '#fbbf24', n: '2 个',
        d: '删 <span class="mono">sharding_rule</span><br>删 <span class="mono">sharding_group</span><br><span class="dim">传播已结束</span>' },
      { t: '③ 网格', c: '#4ade80', n: '2 个',
        d: '<span class="mono">inline_meshes</span><br><span class="mono">drop_sharding_and_mesh</span><br><span class="dim">二选一</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:10.5px;line-height:1.5">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>为什么先做这组</b>：后续步骤都依赖"<b>分片已经确定</b>"。',
        '这些信息<b>只在传播中有用</b> —— 传播结束就没有意义了。',
        '<b>为什么最后做</b>：分片<b>引用</b>网格 —— 必须先定下分片，才能安全地内联或删除网格。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>一句话</b>：收尾阶段全都在删东西 —— 先定分片、再删辅助、最后处理网格。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 三个分片 pass */
{
  kicker: 'L4-15 · 导出收尾',
  title: '★ 分片的收尾：<span class="hl-a">闭合</span>、<span class="hl-a">截断</span>、<span class="hl-a">只动边界</span>',
  sub: '三个 pass 各解决一个问题 —— 其中"不可整除"的规则最实用。',
  caption: '注意 <span class="mono">remove_sub_axes</span> 的注释：<b>只动输入输出，不动中间张量</b>。',
  code: `// 【① close_shardings】把开维 (?) 闭合
sdy.mesh @mesh = <["x"=4, "y"=2]>
func.func @func_input_sharding_is_open(
    %arg0: tensor<8x16xf32> {...<@mesh, [{"x", ?}, {?}]>})       // 有开维
    -> (tensor<8x16xf32> {...<@mesh, [{}, {"x"}]>}) {
// 输出：
// CHECK-LABEL: func @func_input_sharding_is_open(
//   %arg0: tensor<8x16xf32> {...<@mesh, [{"x"}, {}]>}) -> ...
//                                            ^^^^^^^ 开维被"闭合"
// 读法：
//   {"x", ?} -> {"x"}   保留 x，丢掉开维
//   {?}      -> {}      只有开维 -> 变成无分片
// 为什么：传播期保留 ? 让后续 pass 有调整余地
//         但导出时【必须定下来】—— 后端需要知道"到底怎么切"

// 【② update_non_divisible】截断到可整除前缀
sdy.mesh @mesh_x_4_y_2 = <["x"=4, "y"=2]>
func.func @only_one_dim_modified(
    %arg0: tensor<2x2xf32> {...<@mesh_x_4_y_2, [{"x"}, {"y"}]>})
    -> tensor<2x2xf32> { return %arg0 : tensor<2x2xf32> }
// CHECK-SAME: %arg0: tensor<2x2xf32> {...<@mesh_x_4_y_2, [{"x":(1)2}, {"y"}]>}
//                                                       ^^^^^^^^^ 子轴！
// 读法：第 0 维大小 2，但 x = 4 -> 【除不尽】
//       -> 用子轴 {"x":(1)2} 表示"只用 x 轴的前 2 个设备"
//       第 1 维大小 2、y = 2 -> 整除，保持不变

// 【两种处理方式】（第二个用例 tensor<2x3>）
//   第 0 维 2 vs x=4  -> 【部分整除】-> 子轴截断 {"x":(1)2}
//   第 1 维 3 vs y=2  -> 【完全不能整除】-> 整个轴去掉 {}
// CHECK-SAME: %arg0: tensor<2x3xf32> {...<@mesh_x_4_y_2, [{"x":(1)2}, {}]>}
//                                                                ^^ 直接去掉

// 【③ remove_sub_axes】只动边界
// 测试注释：
//   1. We remove sub-axes and the trailing axes in input and output shardings.
//   2. We do not modify the shardings for intermediate tensors.
// 例：
//   {"x":(1)2, "y", ?}  ->  {?}        删子轴 + 它前面的轴
//   {"y", "x":(1)2, ?}  ->  {"y", ?}   非子轴的 "y" 保留
// 规律：移除子轴及其【前面】的轴，保留【后面】的 -> 只保留最粗的一段
// 为什么只动输入输出：
//   中间张量的分片是导出期【精心算出来的】-> 不能动
//   输入输出是【函数对外的接口】-> 调用者只需知道"大概怎么切"`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① close_shardings', c: '#38bdf8', d: '把<b>开维</b> <span class="mono">?</span> 闭合<br><span class="mono">{"x",?}</span> → <span class="mono">{"x"}</span><br><span class="mono">{?}</span> → <span class="mono">{}</span>' },
      { t: '② update_non_divisible', c: '#fbbf24', d: '<b>部分整除</b> → 子轴截断<br><b>完全不能整除</b> → 去掉<br><span class="dim">最实用的规则</span>' },
      { t: '③ remove_sub_axes', c: '#4ade80', d: '<b>只动输入输出</b><br>只保留最粗的一段<br><span class="dim">中间张量不动</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:11px;overflow-wrap:anywhere">${x.t}</div>
        <div class="card-d" style="font-size:10.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>为什么开维要闭合</b>：后端需要知道"<b>到底怎么切</b>"—— <span class="mono">?</span> 是"还可以切"，不确定。',
        '<b>为什么必须做</b>：<span class="mono">tensor&lt;2&gt;</span> 分给 4 台设备是<b>未定义</b>的 —— 必须明确成"前 2 台各拿 1 个"。',
        '<b>为什么中间张量不动</b>：它们的分片是导出期<b>精心算出来的</b>（L4-02～14）—— 动了就前功尽弃。',
      ][i];
    }));
    tl.at(15200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>关键区分</b>：<b>部分整除</b>（2 与 4）用子轴截断；<b>完全不能整除</b>（3 与 2）直接去掉轴。';
    });
  }
},

/* ------------------------------------------------ 3 辅助信息清理 */
{
  kicker: 'L4-15 · 导出收尾',
  title: '删辅助信息：<span class="mono hl-a">sharding_rule</span> 与 <span class="mono">sharding_group</span>',
  sub: '两者都**只在传播中有用** —— 传播结束就没有意义了。',
  caption: '注意 <span class="mono">drop_sharding_rules</span> <b>只删规则</b>，<span class="mono">sdy.sharding</span> 保留。',
  code: `// 【drop_sharding_rules】删分片规则
func.func @dot(%arg0: ... [{}, {"y"}]>, %arg1: ... [{"y"}, {"x"}]>)
    -> (tensor<8x16xf32> {...[{"x"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>,
      sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j])
                                                  {i=8, j=16, k=32}>}
      : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
// CHECK: %0 = stablehlo.dot %arg0, %arg1
//           {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : ...
//   ^^^^ sdy.sharding_rule 【被删】，sdy.sharding 【保留】
// 为什么可以删：分片规则是【传播期】用的（L2-01 讲过"沿规则传播"）
//               导出后传播已经结束 -> 规则不再需要
// 回顾 L2-03：那里有 keep-sharding-rules 选项（保留规则便于调试）
//             本 pass 就是"最终真的删掉"的那一步

// 【remove_sharding_groups】删分片组
func.func @sharding_group_ops(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<32x96xf32>
  %1 = stablehlo.add %0, %arg0 : tensor<32x96xf32>
  // CHECK-NOT:   sdy.sharding_group
  sdy.sharding_group %arg0 group_id = 747 : tensor<32x96xf32>
  sdy.sharding_group %0 group_id = 747 : tensor<32x96xf32>
  sdy.sharding_group %1 group_id = 747 : tensor<32x96xf32>
  return %1 : tensor<32x96xf32>
}
// 三条 sdy.sharding_group 全部【被删】（CHECK-NOT）
// 回顾 L3-09：分片组是"强制同分片"的【承诺】
//             它在传播期作为独立通道生效（L2-07）
//             传播结束后承诺已经"兑现" -> 组本身没有意义了`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'drop_sharding_rules', c: '#fbbf24', n: '4 用例',
        d: '删 <span class="mono">sdy.sharding_rule</span><br><b>保留</b> <span class="mono">sdy.sharding</span><br><span class="dim">规则只在传播期用</span>' },
      { t: 'remove_sharding_groups', c: '#4ade80', n: '1 用例',
        d: '删 <span class="mono">sdy.sharding_group</span><br><b>承诺已兑现</b><br><span class="dim">传播期作为独立通道</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:11.5px;overflow-wrap:anywhere">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>注意区别</b>：只删<b>规则</b>，分片本身（<span class="mono">sdy.sharding</span>）要保留。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>回顾 L3-09</b>：分片组是"<b>强制同分片</b>"的承诺 —— 传播结束后承诺已经兑现。';
    });
    tl.at(8400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>共同点</b>：两者都<b>只在传播中有用</b> —— 传播结束就没有意义了。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>L2-03 的呼应</b>：那里有 <span class="mono">keep-sharding-rules</span>（保留规则便于调试），本 pass 是"最终真的删掉"。';
    });
  }
},

/* ------------------------------------------------ 4 ★ 网格与顺序 */
{
  kicker: 'L4-15 · 导出收尾',
  title: '★ 网格处理与<span class="hl-a">收尾顺序</span>',
  sub: '`inline_meshes` 与 L3-07 的"提升"**正好相反**；③ 的两个 pass 是**二选一**。',
  caption: '顺序的逻辑：先定分片 → 再删辅助 → 最后处理网格（因为分片<b>引用</b>网格）。',
  code: `// 【inline_meshes】把命名网格内联
// CHECK-LABEL: func @no_lifted_meshes(
// CHECK-SAME: %arg0: tensor<8x8xf32>
//    {sdy.sharding = #sdy.sharding<mesh<["x"=2, "y"=2]>, [{"x"}, {}]>}
//   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 内联网格保持不变
// CHECK-NOT: sdy.mesh @empty_mesh = <[]>
sdy.mesh @empty_mesh = <[]>
//   ^^^^^^^^ 顶层的 sdy.mesh 声明【被删除】
//
// 9 个用例含 manual_computation / named_computation
//   -> 说明内联要【递归】处理区域算子内部

// 【与 L3-07 的镜像关系】
//   L3-07  -sdy-lift-inlined-meshes   内联网格 -> 顶层声明（+ 按内容去重）
//   L4-15  -sdy-inline-meshes         顶层声明 -> 内联
// 为什么导入要提升、导出要内联：
//   导入期：多个内联网格可能【内容相同】-> 提升后可以去重（L3-07）
//   导出期：后端要看到【自包含】的 IR -> 每个分片直接写出它用的网格

// 【drop_sharding_and_mesh】全部删掉
// CHECK-NOT: sdy.mesh
sdy.mesh @mesh_2 = <["x"=2]>
func.func @drop_sharding(%arg0: tensor<2x4xf32> {...<@mesh_2, [{"x"}, {}]>})
    -> (tensor<2x4xf32> {...}) {
  // CHECK-NEXT: stablehlo.add
  // CHECK-NOT:  sdy.sharding
  %0 = stablehlo.add %arg0, %arg0 {...} : tensor<2x4xf32>
}
// sdy.mesh 与所有 sdy.sharding 全部消失 -> IR 变回【纯 StableHLO】
// 这是"完全不用 Shardy"的后端的选项
//   分片信息已经被后端消化掉了（如转换成了集合通信，见 L4-08）

// 【★ 收尾顺序】
//   ① 分片本身：close_shardings
//                update_non_divisible_input_output_shardings
//                remove_sub_axes_in_input_output_shardings
//   ② 辅助信息：drop_sharding_rules
//                remove_sharding_groups
//   ③ 网格：    inline_meshes
//                drop_sharding_and_mesh      <- 与上一步【二选一】
//
// 顺序的逻辑：
//   先定分片（①）—— 后续步骤都依赖"分片已经确定"
//   再删辅助（②）—— 它们只在传播中有用
//   最后处理网格（③）—— 因为分片【引用】网格
//                        必须先保证分片定下来了，才能安全内联或删除网格
//   ③ 的两步是"二选一"：
//     后端【需要】网格信息 -> 只做 inline_meshes
//     后端【不需要】       -> 再做 drop_sharding_and_mesh`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const steps = [
      { n: '①', t: '定分片', c: '#38bdf8', d: 'close<br>update_non_divisible<br>remove_sub_axes' },
      { n: '②', t: '删辅助', c: '#fbbf24', d: 'drop_sharding_rules<br>remove_sharding_groups' },
      { n: '③', t: '处理网格', c: '#4ade80', d: 'inline_meshes<br>drop_sharding_and_mesh<br><b>二选一</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = steps.map(s => {
      const e = U.el('div', { class: 'card', style: `width:240px;opacity:.33;transition:.3s;border-color:${s.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="small mono" style="font-size:11px;color:${s.c}">${s.n}</div>
        <div style="font-size:12.5px;font-weight:600;margin:3px 0">${s.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.4">${s.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    steps.forEach((s, i) => tl.at(700 + i * 4400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>为什么先做</b>：后续步骤都依赖"<b>分片已经确定</b>"。',
        '这些信息<b>只在传播中有用</b> —— 删掉让 IR 更干净。',
        '<b>为什么最后做</b>：分片<b>引用</b>网格 —— 必须先定下分片，才能安全地内联或删除网格。',
      ][i];
    }));
    tl.at(15200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>验收点答案</b>：三阶段顺序 —— 先定分片、再删辅助、最后处理网格（③ 内二选一）。';
    });
  }
},

/* ------------------------------------------------ 5 族谱 */
{
  kicker: 'L4-15 · 导出收尾',
  title: '7 个 pass 的<span class="hl-a">族谱</span>与小结',
  sub: '本课是 L4 导出流水线的**最后阶段** —— 之后只剩 L4-16/17 的杂项。',
  caption: '593 行 / 58 用例 —— 每个 pass 都很小，但合起来决定了交给后端的形态。',
  code: `// 【族谱】7 个文件 / 593 行 / 58 用例
//   ① close_shardings                          92 行 / 15 用例
//   ① update_non_divisible_input_output_shardings 182 / 21  <- 最大
//   ① remove_sub_axes_in_input_output_shardings 108 /  6
//   ② drop_sharding_rules                       35 /  4
//   ② remove_sharding_groups                    12 /  1  <- 最小
//   ③ inline_meshes                            138 /  9
//   ③ drop_sharding_and_mesh                    26 /  2
//
// 最大的两个（update_non_divisible 21、close_shardings 15）
//   都是"分片本身"的处理 -> 规则组合最多

// 【跨课呼应】
//   close_shardings           <- L2-01（开维 ? 的含义）
//   update_non_divisible      <- L2-01（不可整除不是错误）、L4-02（子轴）
//   remove_sub_axes           <- L4-02（子轴的产生）
//   drop_sharding_rules       <- L2-01（沿规则传播）、L2-03（keep 选项）
//   remove_sharding_groups    <- L3-09（组的生命周期）、L2-07（独立通道）
//   inline_meshes             <- L3-07（提升，镜像操作）
//   drop_sharding_and_mesh    <- L4-08（分片已转成集合通信）

// 一句话总结：
//   收尾阶段全都在"删东西"
//   先定分片、再删辅助信息、最后处理网格
//   目标是给后端一个【干净且自包含】的 IR`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: 'close', n: 15, c: '#38bdf8' }, { t: 'update_nd', n: 21, c: '#0ea5e9' },
      { t: 'rm_subaxes', n: 6, c: '#4ade80' }, { t: 'drop_rules', n: 4, c: '#fbbf24' },
      { t: 'rm_groups', n: 1, c: '#c084fc' }, { t: 'inline_mesh', n: 9, c: '#f472b6' },
      { t: 'drop_all', n: 2, c: '#94a3b8' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:104px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:7px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:9.5px;color:${f.c};overflow-wrap:anywhere">${f.t}</div>
        <div class="big" style="font-size:16px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 90)); msg.innerHTML = '<b>58 个用例</b>分 7 个 pass —— 最大的两个都在"分片本身"组。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>为什么最大</b>：<span class="mono">update_non_divisible</span> 与 <span class="mono">close_shardings</span> 的<b>规则组合最多</b>。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>跨课呼应密集</b>：7 个 pass 分别回指 L2-01 / L2-03 / L2-07 / L3-07 / L3-09 / L4-02 / L4-08。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L4-16 讲 <span class="mono">sink-and-convert</span>。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L4-15 · 练习',
  title: '练一练：<span class="hl-a">该跑哪几个收尾 pass</span>',
  sub: '三道题分别考：不可整除、只动边界、收尾顺序。',
  caption: '一句话总结：<b>先定分片、再删辅助、最后处理网格</b>。',
  code: `// 题 1：tensor<2x3> 在 mesh <["x"=4, "y"=2]> 上，
//       两个维的分片分别怎么处理？

// 题 2：remove_sub_axes 为什么不修改中间张量的分片？

// 题 3：收尾的三个阶段是什么？为什么是这个顺序？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col compact-ex', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">tensor&lt;2x3&gt;</span> 在 mesh <span class="mono">&lt;["x"=4, "y"=2]&gt;</span> 上，两个维的分片分别怎么处理？',
        a: '<b>第 0 维</b>（大小 2 vs <span class="mono">x=4</span>）：<b>部分整除</b> → 用<b>子轴截断</b> <span class="mono">{"x":(1)2}</span>（只用前 2 个设备）。' +
           '<br><b>第 1 维</b>（大小 3 vs <span class="mono">y=2</span>）：<b>完全不能整除</b> → <b>整个轴去掉</b> <span class="mono">{}</span>。' +
           '<br><b>结果</b>：<span class="mono">[{"x":(1)2}, {}]</span>。' +
           '<br><span class="dim">为什么必须做：<span class="mono">tensor&lt;2&gt;</span> 分给 4 台设备是<b>未定义</b>的 —— 必须明确成"前 2 台各拿 1 个，后 2 台空闲"。</span>'
      },
      {
        q: '<span class="mono">remove_sub_axes</span> 为什么不修改中间张量的分片？',
        a: '因为<b>中间张量的分片是导出期精心算出来的</b>（L4-02～L4-14 的全部工作）—— 动了就前功尽弃。' +
           '<br>而<b>输入输出是函数对外的接口</b> —— 调用者只需要知道"大概怎么切"，不需要知道子轴这种细节。' +
           '<br><span class="dim">测试注释点明了：<span class="mono">We remove sub-axes and the trailing axes in input and output shardings. We do not modify the shardings for intermediate tensors.</span></span>'
      },
      {
        q: '收尾的三个阶段是什么？为什么是这个顺序？',
        a: '<b>① 分片本身</b>：<span class="mono">close_shardings</span> / <span class="mono">update_non_divisible</span> / <span class="mono">remove_sub_axes</span>；' +
           '<br><b>② 辅助信息</b>：<span class="mono">drop_sharding_rules</span> / <span class="mono">remove_sharding_groups</span>；' +
           '<br><b>③ 网格</b>：<span class="mono">inline_meshes</span> / <span class="mono">drop_sharding_and_mesh</span>（<b>二选一</b>）。' +
           '<br><b>为什么这个顺序</b>：① 后续步骤都依赖"分片已经确定"；② 辅助信息只在传播中有用；③ 分片<b>引用</b>网格 —— 必须先定下分片，才能安全地内联或删除网格。' +
           '<br><span class="dim">③ 的两步二选一：后端<b>需要</b>网格信息就只做 <span class="mono">inline_meshes</span>；<b>不需要</b>就再做 <span class="mono">drop_sharding_and_mesh</span>。</span>'
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
