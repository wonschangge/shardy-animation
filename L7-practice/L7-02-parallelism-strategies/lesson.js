/* ==========================================================================
   L7-02 · parallelism-strategies   （P0）
   --------------------------------------------------------------------------
   综合课：用同一份矩阵乘程序演示四种并行策略的分片标注差异与通信代价。
   引用来源：L5-05 的 stablehlo_dot.mlir + L6-04 的 executable dot_general。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 四种策略 */
{
  kicker: 'L7-02 · 并行策略',
  title: '★ 四种策略的差异 = <span class="hl-a">切哪个维</span>的差异',
  sub: '同一份 `dot(A, B)` —— 策略的差异**本质上是分片标注的差异**。',
  caption: '而「切哪个维」直接决定了<b>需不需要通信</b>。',
  code: `// 【同一份程序】dot(A, B)：A: 8x16、B: 16x32、C: 8x32
//   网格 sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>

// 【★ 四种策略的分片标注差异】
//   策略          分片标注                    通信              对应用例
//   数据并行(DP)  收缩维切                    【all_reduce】     sharded_contracting_dim
//   张量并行(TP)  非收缩维切                  【无】             sharded_non_contracting_dims
//   ZeRO          收缩维切 + 【结果未归约】    【无】（延迟）     ..._unreduced_result
//   流水线并行    按【层】切（跨算子）        只在【层边界】     （无单一用例）

// 【★ 核心洞察】
//   策略的差异【本质上是"切哪个维"的差异】
//   而"切哪个维"直接决定了【需不需要通信】

// 【★ 判据仍是 L5-05 那条】
//   只有【收缩维】上的分片需要 all_reduce
//   非收缩维分片在【算子内】无通信

// 【本课的 IR 来源】
//   L5-05  convert_global_to_local/stablehlo_dot.mlir
//   L6-04  executable_convert_global_to_local/stablehlo_dot_general.mlir
//   （本课不引入新文件，而是重新引用这两个文件）

// 一句话：
//   并行策略 = 在"切哪个维"上做选择`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '数据并行', c: '#fb7185', d: '切<b>收缩维</b><br><span class="mono">all_reduce</span>' },
      { t: '张量并行', c: '#4ade80', d: '切<b>非收缩维</b><br><b>无通信</b>' },
      { t: 'ZeRO', c: '#fbbf24', d: '切收缩维<br>+ <b>未归约</b>' },
      { t: '流水线并行', c: '#38bdf8', d: '按<b>层</b>切<br>层边界通信' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div style="font-size:11px;color:${x.c}">${x.t}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>四种策略</b> —— 前三种切的是<b>同一个算子的不同维</b>。'; });
    tl.at(4400, () => {
      els.forEach((e, i) => { if (i < 2) e.style.opacity = '1'; else e.style.opacity = '.25'; });
      msg.innerHTML = '<b>★ 最核心的对照</b>：数据并行切<b>收缩维</b>（要 all_reduce）vs 张量并行切<b>非收缩维</b>（无通信）。';
    });
    tl.at(8000, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>ZeRO</b> 与数据并行的唯一区别是<b>结果未归约</b> —— 每台只保存部分和（省显存）。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>流水线并行是另一个层次</b>：它切的是<b>算子之间</b>（按层），不是算子内部的维。';
    });
  }
},

/* ------------------------------------------------ 2 数据并行 vs 张量并行 */
{
  kicker: 'L7-02 · 并行策略',
  title: '★ <span class="hl-a">数据并行</span> vs <span class="hl-a">张量并行</span>',
  sub: '切**收缩维**要 `all_reduce`；切**非收缩维**在算子内**无通信**。',
  caption: '这是本课最重要的一组对照 —— 也是 L5-05 那条判据的直接应用。',
  code: `// 【数据并行：切收缩维】sharded_contracting_dim
func.func @sharded_contracting_dim(
  %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x"}]>},
  %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) -> tensor<8x32xf32> {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{}, {}], unreduced={"x"}>]>}
   : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2_4, [{}, {}]> : tensor<8x32xf32>
  return %1 : tensor<8x32xf32>
}
// 【读法】
//   %arg0 的【第 1 维（收缩维）】切 {"x"}
//   %arg1 的【第 0 维（收缩维）】切 {"x"}
//   每台设备算一个【部分和】-> unreduced={"x"} -> 【all_reduce】
//
// 【★ 为什么这是"数据并行"】
//   收缩维对应【输入数据的方向】（A 的列、B 的行）
//   切开它等于【把输入数据分给不同设备】，每台算一部分和，最后归约
//
// 【通信代价】一次 all_reduce，通信量 ∝ 【结果大小】（8x32）

// 【张量并行：切非收缩维】sharded_non_contracting_dims
func.func @sharded_non_contracting_dims(
  %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
  %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>})
  -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>}
   : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}
// 【读法】
//   %arg0 的【第 0 维（非收缩维 M）】切 {"x"}
//   %arg1 的【第 1 维（非收缩维 N）】切 {"y"}
//   收缩维（K）【都是完整的 16】
//     -> 每台算输出矩阵的【一块】-> 【无通信】✓
//
// 【★ 为什么这是"张量并行"】
//   非收缩维对应【权重/输出的方向】
//   切开它等于【把权重矩阵分给不同设备】，每台算输出的一部分
//
// 【通信代价】零（在这个算子内）
//   但注意：如果后续算子需要【完整的输入】，就要在【算子边界】插入通信

// 【★ 对照表】
//             数据并行        张量并行
//   切的维    【收缩维】      【非收缩维】
//   每台算什么 【部分和】     【输出的一块】
//   通信       【all_reduce】  【无】
//   通信量     ∝ 结果大小      0（本算子内）`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '数据并行', c: '#fb7185', d: '切 <b>K</b>（收缩维）<br>每台算<b>部分和</b><br><span class="mono">all_reduce</span>' },
      { t: '张量并行', c: '#4ade80', d: '切 <b>M</b>/<b>N</b>（非收缩）<br>每台算<b>一块</b><br><b>无通信</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>切收缩维</b> = 把<b>输入数据</b>分给不同设备 → 每台算部分和 → 必须归约。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>切非收缩维</b> = 把<b>权重矩阵</b>分给不同设备 → 每台算输出的一块 → <b>互不重叠</b>。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 判据仍是 L5-05 那条</b>：<b>只有收缩维上的分片需要 <span class="mono">all_reduce</span></b>。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>注意"无通信"是有条件的</b>：若下游算子需要完整输入，就要在<b>算子边界</b>插通信。';
    });
  }
},

/* ------------------------------------------------ 3 ★ ZeRO */
{
  kicker: 'L7-02 · 并行策略',
  title: '★ <span class="hl-a">ZeRO</span>：切收缩维但<span class="hl-a">不归约</span>',
  sub: '与数据并行的**唯一区别**是函数结果也声明了 `unreduced`。',
  caption: '核心思想：<b>不要每台设备都保存完整状态</b>。',
  code: `func.func @sharded_contracting_dim_unreduced_result(
  %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x"}]>},
  %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
  -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}], unreduced={"x"}>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{}, {}], unreduced={"x"}>]>}
   : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// 【读法】与数据并行的【唯一区别】
//   函数结果【也声明了 unreduced={"x"}】
//     -> 归约责任交给【调用者】
//   -> 【不插 all_reduce】，直接 return %0 ✓
//
// 【★ 为什么这对应 ZeRO 的思想】
//   ZeRO 的核心是【"不要每台设备都保存完整的状态"】
//   未归约时每台设备【只保存自己那份部分和】，显存占用【更小】
//   归约推迟到【真正需要完整结果时】才做（L4-05 讲的"延迟归约"）
//
// 【★ 对照表】
//             数据并行        ZeRO（延迟归约）
//   收缩维    【切】          【切】
//   结果要求  【完整】        【未归约】
//   all_reduce【插】          【不插】
//   每台保存  【完整结果】    【只有部分和】（省显存）
//
// 【★ 回顾 L4-05 的五种归约时机】
//   立刻 / 延迟到某算子前 / 部分延迟 / 完全延迟到 return / 【完全不插】
//   本策略是"完全不插"—— 因为结果本来就接受未归约
//
// 【与数据并行的关系】
//   它们是【同一个分片标注】，只差【结果的声明】
//   -> 这是一个【纯声明层面】的优化：改一个属性就能省显存
//   -> 非常优雅`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '数据并行', c: '#fb7185', d: '结果要求<b>完整</b><br>→ 插 <span class="mono">all_reduce</span><br>每台存<b>完整结果</b>' },
      { t: 'ZeRO', c: '#fbbf24', d: '结果<b>也接受未归约</b><br>→ <b>不插</b><br>每台只存<b>部分和</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>数据并行</b>：结果必须完整 → 必须 <span class="mono">all_reduce</span> → 每台都存完整结果。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>ZeRO</b>：结果接受未归约 → 不插通信 → <b>每台只存部分和</b>（省显存）。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 两者是同一个分片标注</b>，只差<b>结果的声明</b> —— 改一个属性就能省显存。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>回顾 L4-05 的五种归约时机</b>：本策略是"<b>完全不插</b>"—— 因为结果本来就接受未归约。';
    });
  }
},

/* ------------------------------------------------ 4 ★ 流水线并行 */
{
  kicker: 'L7-02 · 并行策略',
  title: '★ <span class="hl-a">流水线并行</span>：另一个层次的切分',
  sub: '它切的是**算子之间**（按层），而不是单个算子内部的维。',
  caption: '这是四种策略里<b>唯一不通过"算子分片标注"表达</b>的一种。',
  code: `// 【流水线并行与前三种的层次不同】
//   策略                  切分粒度
//   数据 / 张量 / ZeRO    【单个算子内部】的维
//   流水线并行            【算子之间】（按层分组）

// 【★ 在 Shardy 里如何表达】
//   流水线并行【不是】通过"某个算子的分片标注"实现的
//   而是【把不同层放到不同的网格/设备上】
//   这可以用：
//     ① 【多个 mesh】（L7-01 讲的"网格切换"）
//     ② 【sdy.manual_computation】（L4-11 讲的"逐指令分区"）
//   来表达

// 【通信代价】
//   只在【层边界】通信（把上一层的输出传给下一层）
//   【层内部无通信】—— 这是流水线并行的优势
//   代价是【流水线气泡】（bubble）—— 前几层在算时后几层空闲

// 【★ 与其他三种的本质区别】
//   数据/张量/ZeRO 是【空间上的切分】（同一个算子的不同维）
//   流水线并行是【时间/层次上的切分】（不同算子在不同设备）
//
// 【为什么 Shardy 能表达它】
//   Shardy 的分片标注是【逐算子】的
//   而 mesh 是【程序级】的
//   -> 用"算子属于哪个 mesh"就能表达"层属于哪组设备"
//   这正是 L7-01 讲的 single_device_switch 的思想：
//     同一程序里多个网格并存、数据在网格间流动
//
// 【一个实际考虑】
//   流水线并行通常与张量并行【组合】使用：
//     层内用张量并行（切权重、无通信）
//     层间用流水线并行（层边界通信）
//   -> 这就是"3D 并行"的两个维度`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '空间切分', c: '#4ade80', d: '数据 / 张量 / ZeRO<br>切<b>同一个算子的维</b><br>用<b>分片标注</b>表达' },
      { t: '层次切分', c: '#38bdf8', d: '<b>流水线并行</b><br>切<b>算子之间</b><br>用<b>多个 mesh</b> 表达' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>前三种</b>都是<b>空间切分</b> —— 用算子上的分片标注表达。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>流水线并行</b>是<b>层次切分</b> —— 用"算子属于哪个 mesh"表达。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ Shardy 为什么能表达它</b>：分片标注是<b>逐算子</b>的，而 mesh 是<b>程序级</b>的。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>实际用法</b>：层内用张量并行、层间用流水线并行 —— 这是 <b>3D 并行</b>的两个维度。';
    });
  }
},

/* ------------------------------------------------ 5 ★ 选型 */
{
  kicker: 'L7-02 · 并行策略',
  title: '★ <span class="hl-a">通信代价</span>对比与选型',
  sub: '验收点：给定模型规模与设备数，能选合理策略并说明代价。',
  caption: '把四种策略的代价放在一张表里，选型就清晰了。',
  code: `// 【通信代价总表】
//   策略          切的维              通信模式        通信量          显存收益
//   数据并行      收缩维              all_reduce      ∝ 【结果】大小   中
//   张量并行      非收缩维            【无】（算子内） 0               【大】
//   ZeRO          收缩维 + 未归约     【无】（延迟）   0（推迟）        【大】
//   流水线并行    按层                层边界          ∝ 【层输出】大小  【大】

// 【★ 选型判据】
//   情形                              推荐          理由
//   模型能放进单卡，但数据量大        【数据并行】  切数据、all_reduce 梯度，最成熟
//   模型放不进单卡，层内可切          【张量并行】  切权重、【无通信】（算子内）
//   模型放不进单卡，且要省显存        【ZeRO】      延迟归约，每台只存部分和
//   层数多、层间依赖弱                【流水线并行】层边界通信，层内无通信
//   超大规模                          【组合】      例如 TP x PP x DP（3D 并行）

// 【★ 组合的例子】（回顾 L5-05 的 dot_general）
//   批维切 x（批并行）+ 收缩维切 y（数据并行）= 【2D 并行】
//   那里【只有收缩维】上的分片需要 all_reduce —— 批维上的不需要

// 【★ 一个反直觉的点】
//   张量并行"无通信"是【有条件的】：
//     它指的是【单个算子内部】无通信
//     但如果下游算子需要【完整的】输入
//       就要在【边界】插入 all_gather 之类的通信
//   判据始终是 L5-05 那条：只有收缩维上的分片需要 all_reduce
//     但"算子之间"的通信由【数据流】决定（L2-07 的 data-flow edges）

// 【一句话选型】
//   先问"模型放得下吗" -> 放得下就数据并行
//   放不下 -> 层内切用张量并行、层间切用流水线并行、省显存用 ZeRO`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:8px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '数据并行', c: '#fb7185', d: 'all_reduce<br>∝ 结果大小' },
      { t: '张量并行', c: '#4ade80', d: '<b>无</b>（算子内）<br>0' },
      { t: 'ZeRO', c: '#fbbf24', d: '<b>无</b>（延迟）<br>0（推迟）' },
      { t: '流水线并行', c: '#38bdf8', d: '层边界<br>∝ 层输出' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div style="font-size:11px;color:${x.c}">${x.t}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>数据并行</b>：唯一需要显式 <span class="mono">all_reduce</span> 的策略 —— 通信量正比于结果大小。',
        '<b>张量并行</b>：算子内<b>零通信</b> —— 但边界可能仍需通信。',
        '<b>ZeRO</b>：把归约<b>推迟</b>，换来显存收益。',
        '<b>流水线并行</b>：只在层边界通信，代价是流水线气泡。',
      ][i];
    }));
    tl.at(15200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 一句话选型</b>：先问"模型放得下吗" → 放得下就数据并行；放不下则层内 TP、层间 PP、省显存用 ZeRO。';
    });
  }
},

/* ------------------------------------------------ 6 小结 */
{
  kicker: 'L7-02 · 并行策略',
  title: '小结：<span class="hl-a">一条判据串起四种策略</span>',
  sub: '所有策略都回到 **L5-05** 那条：只有收缩维上的分片需要 `all_reduce`。',
  caption: '一句话：<b>并行策略 = 在"切哪个维"上做选择</b>。',
  code: `// 【四种策略总表】
//   策略          分片标注                    通信              对应用例
//   数据并行      收缩维切                    all_reduce        sharded_contracting_dim
//   张量并行      非收缩维切                  无                sharded_non_contracting_dims
//   ZeRO          收缩维切 + 结果未归约        无（延迟）        ..._unreduced_result
//   流水线并行    按层切（跨算子）            只在层边界        （无单一用例）

// 【★ 三条结论】
//   ① 四种策略的差异【本质上是"切哪个维"的差异】
//      而"切哪个维"直接决定了【需不需要通信】
//   ② 判据仍是 L5-05 那条：只有【收缩维】上的分片需要 all_reduce
//      非收缩维分片在【算子内】无通信
//   ③ 流水线并行是【另一个层次的切分】
//      它切的是【算子之间】（按层），而不是单个算子内部的维

// 【★ 与前面课的呼应】
//   L5-05  矩阵乘的四类情形（本课的策略分类依据）
//   L5-06  卷积的归约因子（同样是"切哪个维"的问题）
//   L4-05  延迟归约的五种时机（ZeRO 用的是"完全不插"）
//   L4-11  逐指令分区（表达流水线并行的一种方式）
//   L2-07  data-flow edges（决定算子之间的通信）
//   L6-04  收缩维分片的数值验证（数据并行真的算对了）
//   L7-01  网格切换（表达流水线并行的另一种方式）

// 【★ 本课在 L7 中的位置】
//   L7-01 端到端走查（【流水线】视角）
//   L7-02 并行策略（本课，【策略】视角）
//   L7-03 调试手册（【排错】视角）
//   L7-04 跨方言集成（【扩展】视角）

// 一句话总结：
//   并行策略 = 在"切哪个维"上做选择
//   切收缩维是数据并行（要 all_reduce）
//   切非收缩维是张量并行（算子内无通信）
//   不归约是 ZeRO（省显存）
//   按层切是流水线并行（层边界通信）`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '数据并行', n: 1, c: '#fb7185' }, { t: '张量并行', n: 1, c: '#4ade80' },
      { t: 'ZeRO', n: 1, c: '#fbbf24' }, { t: '流水线并行', n: 1, c: '#38bdf8' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:170px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div style="font-size:10px;color:${f.c};line-height:1.3">${f.t}</div>
        <div class="big" style="font-size:16px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 120)); msg.innerHTML = '<b>四种策略</b> —— 前三种切算子的维，第四种切算子之间。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>★ 三条结论</b>：差异 = 切哪个维 / 判据仍是 L5-05 / 流水线并行是另一个层次。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>与前面课呼应密集</b>：L5-05（判据）、L4-05（延迟归约）、L4-11（逐指令分区）、L2-07（数据流）。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L7-03 讲调试手册（P0）—— 常见问题排查与定位。';
    });
  }
},

];
