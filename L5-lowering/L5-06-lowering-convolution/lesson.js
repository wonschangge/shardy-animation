/* ==========================================================================
   L5-06 · lowering-convolution
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/convert_global_to_local/stablehlo_convolution.mlir
         (191 行 / 6 用例)
   目标：讲透卷积分片降级 —— 与矩阵乘同构的规律 + 卷积特有的复杂度。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 六个用例 */
{
  kicker: 'L5-06 · 卷积降级',
  title: '★ 六个用例按<span class="hl-a">分片的维度</span>组织',
  sub: '与 L5-05 的矩阵乘**同构** —— 只有归约因子上的分片需要通信。',
  caption: '本课是 <b>L5-05</b> 的姊妹课：那里是矩阵乘，这里是卷积。',
  code: `// 【六个用例】1 个文件 / 191 行 / 6 用例
//   shard_batch                             批维分片           -> 无通信
//   shard_batch_group                       batch_group 维     -> 无通信
//   shard_feature                           特征维分片         -> 无通信
//   shard_feature_group                     feature_group 维   -> 无通信
//   shard__reduction_factors                【归约因子】       -> all_reduce
//   shard_reduction_factors_unreduced_result 归约因子(未归约)  -> 无（延迟）

// 【★ 核心规律】（与 L5-05 一致）
//   只有【归约因子】上的分片需要 all_reduce
//   批维 / 特征维上的分片都是"各算各的输出元素"，无需通信

// 【卷积的 sharding_rule】（测试文件开头的注释）
//   ([i, jk, mn, o], [k, n, o, p])->([i, j, m, p])
//   {i=2, j=112, k=2, m=112, n=2, o=3, p=64}
//   reduction={k, n, o} permutation={j, m}
//   ^^^^^^^^^^^^^^^^^^^ 【窗口 + 输入通道】是归约因子！
//                       ^^^^^^^^^^^^^^^^ 空间维因步长不成整数倍
//
// 这就解释了为什么"归约因子分片需要通信"：
//   k / n（窗口）与 o（输入通道）都是【被累加】的方向
//   沿它们分片 -> 每台设备只累加了【一部分】

// 一句话：
//   卷积分片降级与矩阵乘同构
//   输出维分片无需通信、归约因子分片需要 all_reduce`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'shard_batch', c: '#94a3b8', d: '批维<br><b>无通信</b>' },
      { t: 'shard_batch_group', c: '#a3a3a3', d: 'batch_group<br><b>无通信</b>' },
      { t: 'shard_feature', c: '#38bdf8', d: '特征维<br><b>无通信</b>' },
      { t: 'shard_feature_group', c: '#0ea5e9', d: 'feature_group<br><b>无通信</b>' },
      { t: 'shard__reduction_factors', c: '#fb7185', d: '<b>归约因子</b><br><b>all_reduce</b>' },
      { t: '..._unreduced_result', c: '#4ade80', d: '归约因子<br><b>无（延迟）</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:118px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:8.5px;color:${x.c};overflow-wrap:anywhere">${x.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>六个用例</b> —— 前四个无通信，只有后两个涉及归约。'; });
    tl.at(4400, () => {
      els.forEach((e, i) => { if (i !== 4 && i !== 5) e.style.opacity = '.25'; });
      msg.innerHTML = '<b>关键区分</b>：<b>输出维</b>（批/特征）分片 vs <b>归约因子</b>（窗口/输入通道）分片。';
    });
    tl.at(8000, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>sharding_rule 里的 <span class="mono">reduction={k, n, o}</span></b> 就是答案 —— 窗口与输入通道是被累加的方向。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>与 L5-05 一致</b>：<b>只有归约方向上的分片需要 <span class="mono">all_reduce</span></b>。';
    });
  }
},

/* ------------------------------------------------ 2 无通信的两个 */
{
  kicker: 'L5-06 · 卷积降级',
  title: '无通信的两个：<span class="hl-a">批维</span>与<span class="hl-a">特征维</span>',
  sub: '每台设备算**不同的输出元素** —— 互不重叠，所以无需通信。',
  caption: '注意两者的 <b>dim_numbers / window / group_count 全部不变</b> —— 只改类型。',
  code: `// 【① 批维分片】shard_batch
func.func @shard_batch(
  %arg0: tensor<2x224x224x3xf32> {...[{"x"}, {}, {}, {}]>},
  %arg1: tensor<3x3x3x64xf32>)
  -> (tensor<2x112x112x64xf32> {...[{"x"}, {}, {}, {}]>}) {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]}
    { feature_group_count = 1 : i64, batch_group_count = 1 : i64,
      sdy.sharding = ...<@mesh_2_4, [{"x"}, {}, {}, {}]> }
    : (tensor<2x224x224x3xf32>, tensor<3x3x3x64xf32>) -> tensor<2x112x112x64xf32>
  return %0 : tensor<2x112x112x64xf32>
}
// CHECK-SAME: (%arg0: tensor<1x224x224x3xf32> {...}, %arg1: tensor<3x3x3x64xf32>)
// CHECK-SAME: -> (tensor<1x112x112x64xf32> {...})
// CHECK: %[[CONV]] = stablehlo.convolution(%arg0, %arg1)
// CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
// CHECK-SAME{LITERAL}: window = {stride = [2, 2], pad = [[0, 1], [0, 1]]}
// CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
// CHECK-SAME: : (tensor<1x224x224x3xf32>, tensor<3x3x3x64xf32>) -> tensor<1x112x112x64xf32>
// 【读法】
//   批维（第0维）切 x=2 -> 2/2 = 1（局部批大小为 1）
//   权重【无分片】
//   dim_numbers / window / group_count 【全部不变】—— 只改类型！
//   -> 【无通信】✓
// 为什么：批维是【输出维】—— 每个样本独立卷积，互不干扰

// 【② 特征维分片】shard_feature
func.func @shard_feature(
  %arg0: tensor<2x224x224x4xf32>,
  %arg1: tensor<3x3x4x64xf32> {...[{}, {}, {}, {"x"}]>})
  -> (tensor<2x112x112x64xf32> {...[{}, {}, {}, {"x"}]>}) {
// CHECK-SAME: (%arg0: tensor<2x224x224x4xf32>,
// CHECK-SAME:  %arg1: tensor<3x3x4x32xf32> {...[{}, {}, {}, {"x"}]>})
// CHECK-SAME: -> (tensor<2x112x112x32xf32> {...[{}, {}, {}, {"x"}]>})
// 【读法】
//   输出通道（第3维 p）切 x=2 -> 64/2 = 32
//   %arg0（输入）【无分片】
//   结果也在第3维切 {"x"}
//   -> 【无通信】✓
// 为什么：输出通道是【输出维】—— 每台算【不同的输出通道】，互不重叠
// 【注意】这里 %arg1 的 p 维切了，但 o 维（输入通道，归约因子）【没切】
//         -> 所以不需要通信`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '批维分片', c: '#94a3b8', d: '每个<b>样本</b>独立卷积<br><span class="mono">2x224x224x3</span> → <span class="mono">1x224x224x3</span><br><b>无通信</b>' },
      { t: '特征维分片', c: '#38bdf8', d: '每个设备算<b>不同的输出通道</b><br><span class="mono">3x3x4x64</span> → <span class="mono">3x3x4x32</span><br><b>无通信</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>批维是最自然的并行方向</b>：样本之间完全独立。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>特征维同理</b>：每个输出通道是独立算出来的 —— 互不依赖。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>共同点</b>：<span class="mono">dim_numbers</span> / <span class="mono">window</span> / <span class="mono">group_count</span> <b>全部不变</b> —— 只改类型。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>关键细节</b>：<span class="mono">shard_feature</span> 里 <span class="mono">p</span> 维切了，但 <span class="mono">o</span> 维（<b>归约因子</b>）<b>没切</b> → 所以无通信。';
    });
  }
},

/* ------------------------------------------------ 3 ★ 归约因子 */
{
  kicker: 'L5-06 · 卷积降级',
  title: '★ 归约因子分片：<span class="mono hl-a">unreduced</span> + <span class="mono">all_reduce</span>',
  sub: '窗口与输入通道都是**被累加**的方向 —— 沿它们分片就产生**部分结果**。',
  caption: '这是本课的核心用例，也是"为什么需要通信"的完整答案。',
  code: `func.func @shard__reduction_factors(
  %arg0: tensor<2x224x224x4xf32> {...[{}, {}, {}, {"y":(2)2}]>},
  %arg1: tensor<2x2x4x64xf32> {...[{"x"}, {"y":(1)2}, {"y":(2)2}, {}]>})
    -> (tensor<1x112x112x64xf32> {...[{}, {}, {}, {}]>}) {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 0], [0, 0]]}
    { feature_group_count = 1 : i64, batch_group_count = 2 : i64,
      sdy.sharding = ...<@mesh_2_4, [{}, {}, {}, {}], unreduced={"x", "y"}> }
    : (tensor<2x224x224x4xf32>, tensor<2x2x4x64xf32>) -> tensor<1x112x112x64xf32>
}
// CHECK-SAME: (%arg0: tensor<2x224x224x2xf32> {...[{}, {}, {}, {"y":(2)2}]>},
// CHECK-SAME:  %arg1: tensor<1x1x2x64xf32> {...[{"x"}, {"y":(1)2}, {"y":(2)2}, {}]>})
// CHECK-SAME: -> (tensor<1x112x112x64xf32> {...[{}, {}, {}, {}]>})
// CHECK: %[[CONV]] = stablehlo.convolution(%arg0, %arg1)
// CHECK-SAME{LITERAL}: window = {stride = [2, 2], pad = [[0, 0], [0, 0]]}
// CHECK-SAME: {batch_group_count = 2 : i64, feature_group_count = 1 : i64}
// CHECK-SAME: : (tensor<2x224x224x2xf32>, tensor<1x1x2x64xf32>) -> tensor<1x112x112x64xf32>

// 【逐项读】本课最关键的一处
//   张量      全局           分片                        局部            说明
//   %arg0     2x224x224x4    [{}, {}, {}, {"y":(2)2}]    2x224x224x2     输入通道 o 切
//   %arg1     2x2x4x64       [{"x"}, {"y":(1)2},
//                             {"y":(2)2}, {}]             1x1x2x64        k/n/o 都切
//   结果      1x112x112x64   [{}, {}, {}, {}]            1x112x112x64    全复制
//   conv 的 sharding         【unreduced={"x", "y"}】                    【部分结果】！

// 【归约因子被切了】
//   %arg0 的【输入通道 o】（第3维）切 {"y":(2)2}
//   %arg1 的【窗口 k/n】（第0/1维）切 {"y":(1)2}
//   %arg1 的【输入通道 o】（第2维）切 {"y":(2)2}
//   而 %arg1 的【批维】（第0维）切 {"x"}
// -> 结果标 unreduced={"x", "y"} -> 后续需要 all_reduce {"x", "y"}

// 【为什么 x 也在 unreduced 里】
//   %arg1 的批维切了 x，而 batch_group_count = 2
//   意味着【批维参与了分组卷积】-> 分组的结果也需要归约

// 【子轴的用法】{"y":(1)2} / {"y":(2)2}
//   因为 y=4 要【同时】切窗口维和输入通道维
//   -> 必须用【子轴】分配（L2-02 讲过子轴）

// 【★ 这就是本课的核心】
//   归约因子（窗口 / 输入通道）上的分片
//   -> 产生部分结果 -> 需要 all_reduce`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '输入通道 o 分片', c: '#fb7185', d: '%arg0 第3维<br>%arg1 第2维' },
      { t: '窗口 k/n 分片', c: '#fbbf24', d: '%arg1 第0/1维<br><b>被累加</b>的方向' },
      { t: '批维分片 + group', c: '#c084fc', d: '%arg1 第0维<br><span class="mono">batch_group_count=2</span>' },
      { t: '→ unreduced', c: '#4ade80', d: '<b>部分结果</b><br>需 <span class="mono">all_reduce</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div style="font-size:11px;color:${x.c}">${x.t}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>输入通道</b>是被累加的方向之一（`reduction={k,n,o}` 里的 `o`）。',
        '<b>窗口</b>也是被累加的方向 —— 每个输出元素要累加整个窗口的贡献。',
        '<b>批维参与分组</b>：<span class="mono">batch_group_count=2</span> 让批维也进了归约。',
        '三个归约方向都被切了 → 结果标 <span class="mono">unreduced={"x","y"}</span> → 需要 <span class="mono">all_reduce</span>。',
      ][i];
    }));
    tl.at(16400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 核心</b>：<b>归约因子上的分片 → 部分结果 → <span class="mono">all_reduce</span></b>。';
    });
  }
},

/* ------------------------------------------------ 4 延迟归约 */
{
  kicker: 'L5-06 · 卷积降级',
  title: '情形 ⑥：<span class="hl-a">延迟归约</span>',
  sub: '与情形 ⑤ **同构** —— 唯一区别是函数结果也声明了 `unreduced`。',
  caption: '与 <b>L5-05</b> 的情形 ④、<b>L4-05</b> 的五种归约时机一致。',
  code: `// 【两个用例的对照】
//   shard__reduction_factors                 结果要求【完整】  -> 插 all_reduce
//   shard_reduction_factors_unreduced_result 结果【也】未归约  -> 【不插】

// 两个用例的输入与 conv 完全相同：
//   conv 的结果都标 unreduced={"x", "y"}
//   区别只在【函数结果的声明】上

// 【为什么可以省】
//   未归约时每台设备只算【自己那份部分和】-> 计算量更小
//   越晚归约，中间省下的重复计算越多（L4-05 讲过）
//   这里把归约责任【交给调用者】

// 【三课的一致规律】
//   L4-05  延迟归约的五种时机（总纲）
//   L5-05  矩阵乘的情形 ④（"完全不插"）
//   L5-06  卷积的情形 ⑥（同样"完全不插"）
//   -> 判据统一：【结果是否接受未归约】

// 【判据总结】
//   归约因子是否被分片  +  结果是否接受未归约
//     分片 + 要完整      -> 需要 all_reduce
//     分片 + 接受未归约  -> 不插（延迟）
//     未分片             -> 不需要（无论结果如何）`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '情形 ⑤', c: '#fb7185', d: '结果要求<b>完整</b><br>→ 插 <span class="mono">all_reduce</span>' },
      { t: '情形 ⑥', c: '#4ade80', d: '结果<b>也接受未归约</b><br>→ <b>不插</b><br><span class="dim">归约交给调用者</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>两个用例的输入与 conv 完全相同</b> —— 区别只在函数结果的声明上。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>为什么可以省</b>：未归约时每台设备只算<b>自己那份部分和</b> —— 计算量更小。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>三课一致</b>：L4-05 总纲 / L5-05 矩阵乘 / L5-06 卷积 —— 判据都是<b>结果是否接受未归约</b>。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>判据总结</b>：归约因子<b>是否被分片</b> + 结果<b>是否接受未归约</b>。';
    });
  }
},

/* ------------------------------------------------ 5 与矩阵乘对照 */
{
  kicker: 'L5-06 · 卷积降级',
  title: '与 <span class="mono hl-a">L5-05</span>（矩阵乘）的<span class="hl-a">对照</span>',
  sub: '规律完全一致；卷积的**额外复杂度**来自三个地方。',
  caption: '把两课并排看，规律会更清楚。',
  code: `// 【对照表】
//                   矩阵乘（dot）        卷积（convolution）
//   输出维          非收缩维             批维 / 特征维
//   归约维          收缩维               【窗口 + 输入通道】
//   无通信          输出维分片           批维 / 特征维分片
//   需通信          收缩维分片           【归约因子分片】
//   延迟归约        结果标 unreduced     结果标 unreduced

// 【★ 共同的规律】
//   只有【归约方向】上的分片需要 all_reduce
//   输出方向上的分片都是"各算各的输出元素"，互不重叠

// 【卷积比矩阵乘多的三个地方】
//   ① 归约因子有【三个】（k/n 窗口 + o 输入通道）
//      而矩阵乘只有一个收缩维
//   ② 【permutation 因子】（空间维因步长不成整数倍）
//      这带来 halo 问题（L4-09 讲过）
//   ③ 【batch_group_count / feature_group_count】
//      会改变批维/特征维的【大小】
//      所以分片要用【子轴】精细分配

// 【子轴的实际用法】
//   %arg1: [{"x"}, {"y":(1)2}, {"y":(2)2}, {}]
//                          ^^^^^^^^^^  ^^^^^^^^^^
//   y=4 要【同时】切窗口维和输入通道维
//   -> 用子轴 (1)2 与 (2)2 把 y 分成两份，各切一个维
//   这是 L2-02 讲的子轴语义的实际运用

// 一句话总结：
//   卷积分片降级与矩阵乘同构
//   输出维分片无需通信、归约因子分片需要 all_reduce
//   额外复杂度来自【多个归约因子】与【permutation 因子】`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '归约因子有 3 个', c: '#fb7185', d: '<span class="mono">k</span>/<span class="mono">n</span> 窗口<br>+ <span class="mono">o</span> 输入通道<br><span class="dim">矩阵乘只有 1 个收缩维</span>' },
      { t: 'permutation 因子', c: '#fbbf24', d: '空间维因步长<br><b>不成整数倍</b><br><span class="dim">带来 halo 问题（L4-09）</span>' },
      { t: 'group_count', c: '#c084fc', d: '<span class="mono">batch_group_count</span><br>改变批维<b>大小</b><br>→ 分片要用<b>子轴</b>' },
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
        '<b>三个归约因子</b>意味着分片的组合更多 —— 这是卷积比矩阵乘复杂的主因。',
        '<b>permutation</b> 让空间维的分片"对不齐" —— 需要 halo 或全复制（L4-09）。',
        '<b>group_count</b> 改变维度大小 → 普通的轴不够用 → 必须用<b>子轴</b>精细分配。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>子轴的实际用法</b>：<span class="mono">{"y":(1)2}</span> 与 <span class="mono">{"y":(2)2}</span> 把 <span class="mono">y</span> 分成两份，各切一个维。';
    });
  }
},

/* ------------------------------------------------ 6 小结 */
{
  kicker: 'L5-06 · 卷积降级',
  title: '六个用例的<span class="hl-a">小结</span>',
  sub: '一条规律 + 三个额外复杂度。',
  caption: '本课是 L5-05 的姊妹课，两课合起来覆盖了"计算密集算子"的降级。',
  code: `// 【族谱】1 个文件 / 191 行 / 6 用例
//   shard_batch                               批维             无通信
//   shard_batch_group                         batch_group      无通信
//   shard_feature                             特征维           无通信
//   shard_feature_group                       feature_group    无通信
//   shard__reduction_factors                  归约因子         all_reduce
//   shard_reduction_factors_unreduced_result  归约因子(未归约) 无（延迟）

// 【一条规律】
//   只有【归约方向】上的分片需要 all_reduce
//   输出方向（批 / 特征）上的分片都是"各算各的"

// 【三个额外复杂度】
//   ① 三个归约因子（k/n/o）-> 组合更多
//   ② permutation 因子 -> halo 问题（L4-09）
//   ③ group_count 改变维度大小 -> 用子轴精细分配

// 【跨课呼应】
//   L2-02  子轴语义（本课的实际用法）
//   L2-10  卷积的 sharding_rule（本课开头的注释）
//   L4-09  permutation 因子的 halo 处理
//   L5-05  矩阵乘降级（同构的规律）

// 【L5 的进度】
//   L5-01 全局转局部    L5-02 集合通信    L5-03 结构算子
//   L5-04 形状类        L5-05 矩阵乘      L5-06 卷积（本课）
//   L5-07 归约          L5-08 gather/scatter   L5-09 pad for divisibility

// 一句话总结：
//   卷积分片降级与矩阵乘同构 —— 输出维分片无需通信、归约因子分片需要 all_reduce
//   额外复杂度来自多个归约因子与 permutation 因子`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '批维', n: 1, c: '#94a3b8' }, { t: 'batch_group', n: 1, c: '#a3a3a3' },
      { t: '特征维', n: 1, c: '#38bdf8' }, { t: 'feature_group', n: 1, c: '#0ea5e9' },
      { t: '归约因子', n: 1, c: '#fb7185' }, { t: '延迟归约', n: 1, c: '#4ade80' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:118px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div style="font-size:10px;color:${f.c};overflow-wrap:anywhere">${f.t}</div>
        <div class="big" style="font-size:16px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 100)); msg.innerHTML = '<b>6 个用例</b>，每个对应一种分片位置 —— 与 L5-05 的组织方式完全一致。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>一条规律</b>：只有<b>归约方向</b>上的分片需要 <span class="mono">all_reduce</span>。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>三个额外复杂度</b>：多个归约因子 / permutation 因子 / group_count 改变维度大小。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L5-07 讲归约类算子（reduce / reduce_window）的降级。';
    });
  }
},

];
