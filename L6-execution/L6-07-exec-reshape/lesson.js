/* ==========================================================================
   L6-07 · exec-reshape   （P0）
   --------------------------------------------------------------------------
   覆盖：executable_partitioner_pipeline/ 下 7 个 reshape 测试
         (1d_to_2d_split_2groups 69, ..._passthrough_indivisible 56,
          ..._gap_2 48, axis_shift 48, 1d_to_2d_split 39,
          ..._custom_device_ids 39, 2d_split_unrelated_axis 39)
   目标：讲透 reshape 的三步降级模式与分片重排的各种情形。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 三步模式 */
{
  kicker: 'L6-07 · reshape 的执行',
  title: '★ <span class="mono hl-a">reshape</span> 降级的三步模式',
  sub: '`slice`（切到可 reshape）→ `reshape`（重排分片）→ `reshard`（可能通信）。',
  caption: '数值验证 <b>L4-03</b> 的 reshape reshard 规则。',
  code: `// 【7 个文件】全部在 executable_partitioner_pipeline/
//   文件                                        行数
//   stablehlo_reshape_1d_to_2d_split_2groups    69
//   ..._passthrough_indivisible                 56
//   ..._gap_2                                   48
//   stablehlo_reshape_axis_shift                48
//   stablehlo_reshape_1d_to_2d_split            39
//   ..._custom_device_ids                       39
//   stablehlo_reshape_2d_split_unrelated_axis   39

// 【★ 共同结构：三步模式】
//   %0 = stablehlo.slice %arg0 [...]   // ① 切到可 reshape 的大小
//   %1 = stablehlo.reshape %0 {...}    // ② reshape 并【重排分片】
//   %2 = sdy.reshard %1 <...>          // ③ reshard（可能通信）
//
// 【为什么需要 slice】
//   reshape 要求【元素总数不变】
//   但原形状往往拆不成目标形状（如 8 拆不成 2x3）
//   -> 先 slice 到能拆的大小（6）
//
// 【为什么需要 reshard】
//   reshape 后分片可能【不再匹配】（如 c 要切 3/2 除不尽）
//   -> 需要通信

// 【★ 全部 7 个文件都有两个 RUN 行】
//   --enable_halo_exchange=false / true
//   -> 验证 REPL 与 HALO 两种模式的结果一致（L4-09）

// 【★ 与 L4-03 的呼应】
//   L4-03 讲 reshape 的 reshard 插入规则（什么时候需要通信）
//   本课用【真实数值】验证那些规则的正确性`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const steps = [
      { n: '①', t: 'slice', c: '#38bdf8', d: '切到<br><b>可 reshape</b><br>的大小' },
      { n: '②', t: 'reshape', c: '#4ade80', d: '<b>重排分片</b><br>一维 → 多维' },
      { n: '③', t: 'reshard', c: '#fbbf24', d: '<b>可能通信</b><br>分片不匹配时' },
    ];
    const host = wrap.querySelector('#cards');
    const els = steps.map(s => {
      const e = U.el('div', { class: 'card', style: `width:246px;opacity:.33;transition:.3s;border-color:${s.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="small mono" style="font-size:11px;color:${s.c}">${s.n}</div>
        <div class="mono" style="font-size:11.5px;margin-top:3px">${s.t}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4;margin-top:3px">${s.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    steps.forEach((s, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>为什么需要 slice</b>：<span class="mono">reshape</span> 要求元素总数不变，但 <span class="mono">8</span> 拆不成 <span class="mono">2x3</span> → 先切到 <span class="mono">6</span>。',
        '<b>核心动作</b>：一维的复合分片（如 <span class="mono">{"b","c"}</span>）被<b>拆到两个维度</b>。',
        '<b>为什么需要 reshard</b>：重排后分片可能<b>不再匹配</b>（除不尽）→ 需要通信。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 7 个文件都有两个 RUN 行</b> —— 验证 REPL 与 HALO 两种模式等价（L4-09）。';
    });
  }
},

/* ------------------------------------------------ 2 基准 */
{
  kicker: 'L6-07 · reshape 的执行',
  title: '基准：<span class="hl-a">一维复合分片拆到两维</span>',
  sub: '`[{"b","c"}]` → `[{"b"},{"c"}]` —— 这是 `reshape` 的核心能力。',
  caption: '注意 <span class="mono">iota + 1</span> 这个小技巧。',
  code: `//--- part1.mlir
sdy.mesh @mesh_a_4 = <["b"=2, "c"=2]>

func.func @parallel_reshape_1d_to_2d_split(%arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"b", "c"}]>}) -> (tensor<2x3xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:6] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a_4, [{"b", "c"}]>]>} : (tensor<8xi32>) -> tensor<6xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a_4, [{"b"}, {"c"}]>]>} : (tensor<6xi32>) -> tensor<2x3xi32>
  %2 = sdy.reshard %1 <@mesh_a_4, [{}, {}]> : tensor<2x3xi32>
  return %2 : tensor<2x3xi32>
}

// 【读法】本课的核心机制
//   网格 b=2, c=2 —— 【4 台设备】
//   %arg0 切 [{"b", "c"}] —— 【两个轴都切在第 0 维】（复合分片）
//     所以 8 / (2x2) = 2 个元素/台
//   slice [0:6] —— 切到 6（因为 8 拆不成 2x3；2x3 = 6）
//   ★ reshape 到 2x3，分片变成 [{"b"}, {"c"}]
//       —— 【一维的复合分片被"拆"到两个维度】！
//       b 切第 0 维（2 / 2 = 1）
//       c 切第 1 维（3 / 2 —— 【除不尽】！）
//   -> sdy.reshard 到全复制 [{}, {}] —— 【需要通信】

// 【验证方式：逐设备推数值】
func.func @main() {
  %input_seq = stablehlo.iota dim = 0 : tensor<8xi32>
  %c1 = stablehlo.constant dense<1> : tensor<8xi32>
  %input = stablehlo.add %input_seq, %c1 : tensor<8xi32>
  %seq = func.call @sequential_reshape_1d_to_2d_split(%input) : (tensor<8xi32>) -> tensor<2x3xi32>
  ...
  "check.expect_eq"(%res#0, %seq) : (tensor<2x3xi32>, tensor<2x3xi32>) -> ()
// 【逐设备推数值】
//   %input = iota(8) + 1 = 【[1, 2, 3, 4, 5, 6, 7, 8]】
//   slice [0:6] -> [1, 2, 3, 4, 5, 6]
//   reshape -> 【[[1, 2, 3], [4, 5, 6]]】（2x3）
//   %seq 就是它（串行版的结果）
//
//   4 台设备的输入（8/4 = 2 个元素/台）：
//     设备 0 -> [1, 2]    设备 1 -> [3, 4]
//     设备 2 -> [5, 6]    设备 3 -> [7, 8]
//   【4 台设备的结果都等于 %seq】
//     因为最后 reshard 到全复制，每台都拿到了完整的结果 ✓
//
// 【★ 注意 iota + 1 这个技巧】
//   iota 从 0 开始，直接用它第 0 个元素是 0
//     -> 可能与"填充值 0"混淆
//   +1 让所有元素都【非零】，便于辨识
//   （与 L6-03 的"缩放技巧"、L6-06 的"填充值用 9"同源）`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'reshape 前', c: '#38bdf8', d: '<span class="mono">tensor&lt;6xi32&gt;</span><br>分片 <b><span class="mono">[{"b","c"}]</span></b><br>两轴都在第 0 维' },
      { t: 'reshape 后', c: '#4ade80', d: '<span class="mono">tensor&lt;2x3xi32&gt;</span><br>分片 <b><span class="mono">[{"b"},{"c"}]</span></b><br>拆到两个维度' },
      { t: 'reshard', c: '#fbbf24', d: '全复制 <span class="mono">[{},{}]</span><br><span class="mono">3/2</span> <b>除不尽</b><br>→ 需要通信' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '原本 <span class="mono">b</span> 和 <span class="mono">c</span> <b>都切在第 0 维</b> —— 这是"复合分片"。',
        '<b>reshape 把它们拆开</b>：<span class="mono">b</span> 留在第 0 维、<span class="mono">c</span> 去第 1 维。',
        '但 <span class="mono">3/2</span> 除不尽 → 分片不匹配 → <b>reshard 到全复制</b>。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 这就是 reshape 的核心</b>：改变维度结构 → 分片必须<b>重排</b> → 可能触发通信。';
    });
  }
},

/* ------------------------------------------------ 3 七种情形 */
{
  kicker: 'L6-07 · reshape 的执行',
  title: '★ 7 个文件的<span class="hl-a">分片重排</span>情形',
  sub: '子轴 / 设备号 / 奇数轴 / pass-through / 无关轴 / 轴移位。',
  caption: '每一种都验证一个特定的重排场景。',
  code: `// 【7 个文件的差异】
//   文件                          网格                  分片变化                        特点
//   1d_to_2d_split                b=2,c=2               [{"b","c"}]->[{"b"},{"c"}]      基准
//   _2groups                      a=4,b=4               [{"a"},{"b"}]->四个【子轴】     【子轴】
//   _custom_device_ids            b=2,c=2 + device_ids=[3,2,1,0]  同基准              【设备号倒序】
//   _gap_2                        b=2,c=3               [{"b","c"}]->[{"b"},{"c"}]      【c=3 奇数】
//   _passthrough_indivisible      x=2,b=2,c=2           [{"x"},{"b","c"}]->[{"x"},{"b"},{"c"}]  【pass-through】
//   2d_split_unrelated_axis       a=2,b=2               [{"a"},{"b"}]->[{"a"},{},{"b"}]  【中间维未分片】
//   axis_shift                    a=4                   单轴 -> [{},{}]                 【轴移位】

// 【_2groups：用子轴拆两组】
sdy.mesh @mesh_ab_16 = <["a"=4, "b"=4]>
func.func @parallel_reshape_1d_to_2d_split_2groups(%arg0: tensor<8x16xi32> {sdy.sharding = #sdy.sharding<@mesh_ab_16, [{"a"}, {"b"}]>}) -> (tensor<2x3x2x7xi32> {sdy.sharding = #sdy.sharding<@mesh_ab_16, [{}, {}, {}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:6, 0:14] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_ab_16, [{"a"}, {"b"}]>]>} : (tensor<8x16xi32>) -> tensor<6x14xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_ab_16, [{"a":(1)2}, {"a":(2)2}, {"b":(1)2}, {"b":(2)2}]>]>} : (tensor<6x14xi32>) -> tensor<2x3x2x7xi32>
// 【读法】
//   网格 a=4, b=4（【16 台设备】）；输入 8x16 切 [{"a"}, {"b"}]
//   slice [0:6, 0:14] -> 6x14
//   reshape 到 2x3x2x7 —— 【两个维度拆成四个】！
//   分片用【子轴】：[{"a":(1)2}, {"a":(2)2}, {"b":(1)2}, {"b":(2)2}]
//     a=4 被拆成两组 (1)2 和 (2)2 —— a 的后半给了第 1 维
//     b=4 同理
//   -> 6/2=3（第1维）、2/2=1（第0维）、14/2=7（第3维）、2/2=1（第2维）
// ★ 这验证了 L2-02 讲的子轴：
//   当一个轴需要【同时切多个维】时，必须用子轴把轴的大小分配下去

// 【_custom_device_ids：设备号可以自定义】
sdy.mesh @mesh_custom = <["b"=2, "c"=2], device_ids=[3, 2, 1, 0]>
// 【读法】device_ids=[3, 2, 1, 0] —— 【设备号倒序】！（回顾 L1-02）
//   除设备号外，其余与基准【完全相同】
//   目的：验证【设备号的映射顺序不影响结果】
//     reshape 后的 reshard 通信依赖"谁和谁通信"
//     而 device_ids 决定了【物理设备号】的排列
// ★ 这验证了 L1-02 讲的 device_ids 不影响语义`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:8px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '基准', c: '#94a3b8', d: '一维拆两维' },
      { t: '子轴', c: '#38bdf8', d: '一轴拆多组<br><span class="dim">L2-02</span>' },
      { t: '设备号倒序', c: '#4ade80', d: '<span class="mono">device_ids</span><br><span class="dim">L1-02</span>' },
      { t: '奇数轴', c: '#fbbf24', d: '<span class="mono">c=3</span><br>间隙处理' },
      { t: 'pass-through', c: '#c084fc', d: '未涉及的轴<br>原样传递' },
      { t: '无关轴', c: '#fb7185', d: '插入大小为 1<br>的未分片维' },
      { t: '轴移位', c: '#f472b6', d: '单轴→全复制' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:104px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div style="font-size:10px;color:${x.c}">${x.t}</div>
        <div class="small faint" style="font-size:9px;line-height:1.3;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 100)); msg.innerHTML = '<b>七种情形</b> —— 每种验证一个特定的分片重排场景。'; });
    tl.at(4400, () => {
      els.forEach((e, i) => { if (i !== 1) e.style.opacity = '.25'; });
      msg.innerHTML = '<b>★ 子轴</b>：网格 <span class="mono">a=4, b=4</span>，一个轴要拆给<b>多个维</b> → 必须用子轴分配。';
    });
    tl.at(8000, () => {
      els.forEach((e, i) => { if (i !== 2) e.style.opacity = '.25'; });
      msg.innerHTML = '<b>★ 设备号倒序</b>：<span class="mono">device_ids=[3,2,1,0]</span> —— 除设备号外与基准相同，验证它<b>不影响语义</b>。';
    });
    tl.at(11600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>其余五种</b>：奇数轴间隙 / pass-through / 无关轴 / 轴移位 / 基准。';
    });
  }
},

/* ------------------------------------------------ 4 其余三种 */
{
  kicker: 'L6-07 · reshape 的执行',
  title: '其余三种：<span class="hl-a">pass-through / 无关轴 / 轴移位</span>',
  sub: '这三种展示了 `reshape` 如何**部分**影响分片。',
  caption: '共同点是：分片的变化<b>只发生在被 reshape 的维度上</b>。',
  code: `// 【_passthrough_indivisible：pass-through 轴】
sdy.mesh @mesh_xbc_8 = <["x"=2, "b"=2, "c"=2]>
func.func @parallel_reshape_1d_to_2d_split_passthrough_indivisible(%arg0: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_xbc_8, [{"x"}, {"b", "c"}]>}) -> (tensor<3x2x3xi32> {sdy.sharding = #sdy.sharding<@mesh_xbc_8, [{}, {}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:3, 0:6] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_xbc_8, [{"x"}, {"b", "c"}]>]>} : (tensor<4x8xi32>) -> tensor<3x6xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_xbc_8, [{"x"}, {"b"}, {"c"}]>]>} : (tensor<3x6xi32>) -> tensor<3x2x3xi32>
// 【读法】
//   网格 x=2, b=2, c=2（【8 台设备】）
//   输入 4x8 切 [{"x"}, {"b","c"}] —— x 切第 0 维、{"b","c"} 切第 1 维
//   slice [0:3, 0:6] -> 3x6
//   reshape 到 3x2x3 —— 第 1 维（6）拆成两维（2x3）
//   ★ x 轴【原样"传递"】（pass-through）：[{"x"}, {"b"}, {"c"}]
//       —— x 仍在第 0 维，【未被 reshape 影响】✓
//   indivisible：3 不能被 x=2 整除 -> 需要补齐/裁回

// 【2d_split_unrelated_axis：中间维未分片】
sdy.mesh @mesh_a2_b2 = <["a"=2, "b"=2]>
func.func @parallel_reshape_2d_split_unrelated_axis(%arg0: tensor<6x4xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{"a"}, {"b"}]>}) -> (tensor<1x5x4xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:5, 0:4] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a2_b2, [{"a"}, {"b"}]>]>} : (tensor<6x4xi32>) -> tensor<5x4xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a2_b2, [{"a"}, {}, {"b"}]>]>} : (tensor<5x4xi32>) -> tensor<1x5x4xi32>
// 【读法】
//   输入 6x4 切 [{"a"}, {"b"}]；slice [0:5, 0:4] -> 5x4
//   ★ reshape 到 1x5x4 —— 【在最前面插入一个大小为 1 的维】！
//   分片变成 [{"a"}, {}, {"b"}] —— 【新插入的中间维未分片】✓
//   "unrelated axis"：插入的维（1）与分片【无关】
//     因为它的大小是 1，切不了

// 【axis_shift：轴移位】
sdy.mesh @mesh = <["a"=4]>
func.func @parallel_reshape_axis_shift(
  %arg0: tensor<24xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
  -> (tensor<3x6xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
// 【读法】
//   网格 a=4（4 台设备），输入 24 切 {"a"}；结果 3x6 全复制
//   "axis shift"：分片的轴从"第 0 维"变成"没有维"（全复制）
//     —— 轴的位置发生了【根本变化】`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'pass-through', c: '#c084fc', d: '<span class="mono">x</span> 轴<b>原样传递</b><br><span class="mono">[{"x"},{"b","c"}]</span><br>→ <span class="mono">[{"x"},{"b"},{"c"}]</span>' },
      { t: '无关轴', c: '#fb7185', d: '插入大小为 <b>1</b> 的维<br><span class="mono">[{"a"},{"b"}]</span><br>→ <span class="mono">[{"a"},{},{"b"}]</span>' },
      { t: '轴移位', c: '#f472b6', d: '单轴 → <b>全复制</b><br><span class="mono">[{"a"}]</span><br>→ <span class="mono">[{},{}]</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>pass-through</b>：<span class="mono">x</span> 轴没被 reshape 涉及 → 分片<b>原样保留</b>。',
        '<b>无关轴</b>：插入的维大小是 <span class="mono">1</span> → 切不了 → <b>未分片</b>。',
        '<b>轴移位</b>：分片的轴<b>彻底消失</b>（变成全复制）。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>共同点</b>：分片的变化<b>只发生在被 reshape 涉及的维度上</b> —— 其余原样保留。';
    });
  }
},

/* ------------------------------------------------ 5 小结 */
{
  kicker: 'L6-07 · reshape 的执行',
  title: '小结：<span class="hl-a">三步模式 + 七种重排</span>',
  sub: '`reshape` 的降级规律清晰：改变维度 → 重排分片 → 可能通信。',
  caption: '一句话：<b>reshape 是"分片重排"的触发器</b>。',
  code: `// 【族谱】7 个文件
//   族              文件                        验证什么
//   基准            1d_to_2d_split              一维复合分片拆到两维
//   子轴            _2groups                    一个轴拆给多个维（L2-02）
//   设备号          _custom_device_ids          device_ids 倒序不影响语义（L1-02）
//   奇数轴          _gap_2                      c=3 的间隙处理
//   pass-through    _passthrough_indivisible    未涉及的轴原样传递
//   无关轴          2d_split_unrelated_axis     插入大小为 1 的未分片维
//   轴移位          axis_shift                  单轴 -> 全复制

// 【★ 三条结论】
//   ① reshape 的降级是【三步模式】
//      slice（切到可 reshape）-> reshape（重排分片）-> reshard（通信）
//   ② 分片的重排可以很复杂
//      子轴（_2groups）、pass-through（_passthrough）、
//      插入未分片维（unrelated_axis）—— 每种都要验证
//   ③ 两个 RUN 行验证 REPL/HALO 等价
//      全部 7 个文件都是，因为 reshape 的 reshard 可能触发 halo exchange

// 【★ 与前面课的呼应】
//   L1-02  device_ids 的语义（_custom_device_ids 验证不影响结果）
//   L2-02  子轴（_2groups 的实际运用）
//   L4-03  reshape 的 reshard 插入规则（本课是数值验证）
//   L4-09  REPL / HALO 两种模式（7 个文件都有两个 RUN 行）
//   L5-04  作用维判据（pass-through 轴的保留）

// 【L6 的进度】
//   L6-00~06 已做（机制 / sdy 通信 / stablehlo 通信 / 卷积 /
//                   矩阵乘-fft-iota / gather / pad）
//   L6-07 reshape（本课）    L6-08 reverse/slice    L6-09 scatter 与杂项

// 一句话总结：
//   reshape 改变维度结构 -> 分片必须重排 -> 可能触发通信
//   7 个文件覆盖了分片重排的各种情形
//   并用真实数值验证结果与串行版一致`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '基准', n: 1, c: '#94a3b8' }, { t: '子轴', n: 1, c: '#38bdf8' },
      { t: '设备号', n: 1, c: '#4ade80' }, { t: '奇数轴', n: 1, c: '#fbbf24' },
      { t: 'pass-through', n: 1, c: '#c084fc' }, { t: '无关轴', n: 1, c: '#fb7185' },
      { t: '轴移位', n: 1, c: '#f472b6' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:104px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div style="font-size:9.5px;color:${f.c};overflow-wrap:anywhere">${f.t}</div>
        <div class="big" style="font-size:16px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 100)); msg.innerHTML = '<b>7 个文件</b>覆盖 reshape 的分片重排情形。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>★ 三条结论</b>：三步模式 / 重排可以很复杂 / 两个 RUN 行验证 REPL-HALO 等价。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>与前面课呼应密集</b>：L1-02（device_ids）、L2-02（子轴）、L4-03（reshard 规则）、L4-09（REPL/HALO）。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L6-08 讲 reverse / slice 的执行。';
    });
  }
},

];
