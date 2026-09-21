/* ==========================================================================
   L3-11 · gather-scatter-batching   （L3 收官课）
   --------------------------------------------------------------------------
   覆盖：transforms/import/test/explicit_gather_scatter_batching.mlir (341 行)
         + executable_explicit_gather_scatter_batching/ 下 5 个可执行用例
   目标：回答"为什么隐式批维会导致分片规则无法表达"。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 问题 */
{
  kicker: 'L3-11 · gather/scatter 批维',
  title: '问题：<span class="hl-a">隐式批维</span>让分片规则无法表达',
  sub: 'JAX 里写 `arr.at[jnp.arange(B), offset]` 这种**逐行索引**，会生成「`iota` + `concat`」来构造索引。',
  caption: '这种写法把"可并行的批维"**伪装**成了"不可传播的索引维" —— 本课讲怎么纠正它。',
  code: `// RUN: sdy_opt %s -sdy-explicit-gather-scatter-batching

// 测试注释点明了这个模式的来源：
//   This pattern typically arises from row-wise indexing in JAX using a
//   batch iota, such as: arr.at[jnp.arange(B), offset] or
//   arr.at[jax.lax.iota(jnp.int32, B), offset]
//   Both emit stablehlo.iota concatenated with the column index.

// 转换前：iota 被写进 start_index_map
%iota = stablehlo.iota dim = 0 : tensor<4x1xi32>       // [0,1,2,3]
%indices = stablehlo.concatenate %iota, %offset, dim = 1
%result = "stablehlo.gather"(%operand, %indices) {
  dimension_numbers = #stablehlo.gather<
    offset_dims = [1],
    collapsed_slice_dims = [0],
    start_index_map = [0, 1],       // <- iota 在这里，当成普通索引维
    index_vector_dim = 1>,
  ...}

// 转换后：改成显式批维
//   operand_batching_dims = [0]
//   start_indices_batching_dims = [0]
//   start_index_map = [1]           // <- iota 移出去了`,
  duration: 13000,
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
        <div class="small mono" style="color:var(--bad)">✗ 隐式批维</div>
        <div class="chip c3" style="padding:7px 12px;font-size:11.5px">start_index_map = [0, 1]</div>
        <div class="small faint">iota 混在索引维里</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<span class="mono">iota</span> 生成的 <span class="mono">[0,1,2,3]</span> 被当成"普通的索引列"写进了 <span class="mono">start_index_map</span>。';
    });
    tl.at(4400, () => {
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹', style: 'font-size:26px' }));
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ok)">✓ 显式批维</div>
        <div class="chip c4" style="padding:5px 10px;font-size:11px">operand_batching_dims = [0]</div>
        <div class="chip c4" style="padding:5px 10px;font-size:11px">start_indices_batching_dims = [0]</div>
        <div class="chip c4" style="padding:5px 10px;font-size:11px">start_index_map = [1]</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>显式声明</b>：把这一维标成<b>批维</b>，从索引维里移出去。';
    });
    tl.at(8600, () => {
      msg.innerHTML = '<b>语义完全不变</b> —— 只是换了一种<b>表达方式</b>。';
    });
    tl.at(11200, () => {
      msg.innerHTML = '<b>但表达方式决定了能不能分片</b> —— 这正是下一幕要讲的。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 为什么 */
{
  kicker: 'L3-11 · gather/scatter 批维',
  title: '★ 为什么隐式批维<span class="hl-a">无法分片</span>',
  sub: '答案在 L2-10 见过的 `gather` 分片规则里 —— 关键在最后那段 `blocked_propagation`。',
  caption: '一句话：<b>隐式批维把"可并行的批维"伪装成了"不可传播的索引维"</b>。',
  code: `// L2-10 见过 gather 的规则（逐字见 op_sharding_rule_registry.mlir）：
//   因子分类：reduction={m, o}  need_replication={k, n, p}
//   另有：    blocked_propagation={k}

// 推理链（四步）：
//   ① 隐式批维写在 start_index_map 里
//      -> 落在 blocked_propagation 的因子上
//   ② blocked_propagation 意味着【传播不能沿它推导分片】
//   ③ 但这一维在语义上【明明可以分片】
//      每个输出元素独立地取一行 -> 切批维是天然的并行方式
//   ④ 结果：想切却切不了 —— 分片规则"无法表达"这个意图

// 改成显式批维后：
//   批维有了专门的属性（operand_batching_dims / start_indices_batching_dims）
//   不再落在 blocked_propagation 里
//   -> 分片规则就能正常处理它

// 一句话：
//   隐式批维把"可并行的批维"伪装成了"不可传播的索引维"`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const steps = [
      { n: '①', t: '落在 blocked', d: '批维写在<br><span class="mono">start_index_map</span> 里', c: '#fb7185' },
      { n: '②', t: '不能传播', d: '<span class="mono">blocked_propagation</span><br>禁止沿它推导', c: '#fbbf24' },
      { n: '③', t: '明明可切', d: '每个输出元素<br>独立取一行', c: '#38bdf8' },
      { n: '④', t: '想切切不了', d: '规则<b>无法表达</b><br>这个意图', c: '#c084fc' },
    ];
    const host = wrap.querySelector('#cards');
    const els = steps.map(s => {
      const e = U.el('div', { class: 'card', style: `width:178px;opacity:.33;transition:.35s;border-color:${s.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="small mono" style="font-size:11px;color:${s.c}">${s.n}</div>
        <div style="font-size:12px;font-weight:600;margin:4px 0">${s.t}</div>
        <div class="small faint" style="font-size:10.5px;line-height:1.4">${s.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    steps.forEach((s, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<span class="mono">start_index_map</span> 涉及的因子会进 <span class="mono">blocked_propagation</span>。',
        '<b>这是关键</b>：<span class="mono">blocked_propagation</span> 是 L1-05 讲的<b>正交标注</b>，直接禁止传播。',
        '但从语义看，批维是<b>最自然的分片对象</b> —— 每个输出元素独立地取一行。',
        '<b>矛盾</b>：想切却切不了。改成显式批维后，它不再落在 <span class="mono">blocked_propagation</span> 里。',
      ][i];
    }));
    tl.at(14400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>一句话</b>：隐式批维把"可并行的批维"伪装成了"不可传播的索引维"。';
    });
  }
},

/* ------------------------------------------------ 3 五个不转换 */
{
  kicker: 'L3-11 · gather/scatter 批维',
  title: '五个<span class="hl-a">不转换</span>的情形（同样重要）',
  sub: '主文件 10 个用例里**一半**是"不转换"。这个比例本身就说明了问题。',
  caption: '这个变换会<b>改写算子的语义属性</b>（<span class="mono">start_index_map</span> 变了）—— 条件不满足时贸然改写会<b>改变程序行为</b>。',
  code: `// 五个不转换的用例及其原因：

// ① gather_no_iota_no_transform
//    indices 里【没有 iota】 -> 没有可识别的批维

// ② gather_mismatched_batch_size_no_transform
//    【批大小不匹配】 -> 不能认定为批维

// ③ gather_already_explicit_batching
//    【已经是显式批维】 -> 无需转换（no-op）

// ④ gather_single_index_dim_no_transform
//    【index_vector_dim 大小为 1】 -> 退化成普通索引

// ⑤ gather_dim0_not_collapsed_no_transform
//    【dim 0 不在 collapsed_slice_dims 里】 -> 形状不满足前提

// 为什么必须严格：
//   变换改写语义属性 -> 改错了 IR 仍能通过校验，但【跑出来数值会错】
//   所以前提条件与转换本身同等重要，且必须有测试锁定`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '没有 iota', c: '#94a3b8' }, { t: '批大小不匹配', c: '#fbbf24' },
      { t: '已是显式批维', c: '#4ade80' }, { t: 'index_vector_dim 为 1', c: '#38bdf8' },
      { t: 'dim 0 未 collapsed', c: '#c084fc' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:146px;opacity:.38;transition:.35s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div style="font-size:11px;color:${x.c};line-height:1.4">${x.t}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 130)); msg.innerHTML = '<b>5 转换 + 5 不转换</b> —— 一半用例在锁定"不该转的不转"。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>为什么这个比例合理</b>：变换改写的是<b>语义属性</b>，改错不会报错但会算错。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>① 与 ④</b> 是"特征不明显"；<b>② 与 ⑤</b> 是"形状前提不满足"；<b>③</b> 是幂等性。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>通用原则</b>：改写语义的变换，<b>前提条件与转换本身同等重要</b>。';
    });
  }
},

/* ------------------------------------------------ 4 可执行验证 */
{
  kicker: 'L3-11 · gather/scatter 批维',
  title: '<span class="hl-a">可执行验证</span>：为什么要跑一遍',
  sub: '目录 `executable_.../` 下的 5 个用例用**另一种方式**验证 —— 真正执行一遍看数值对不对。',
  caption: '因为属性改错了，IR <b>依然能通过校验</b>，但跑出来的<b>数值会错</b>。',
  code: `// RUN: %S/run_sdy_interpreter_test.sh %s %t

// 每个文件包含两部分：
//--- part1.mlir
func.func @transformed_gather(%operand: tensor<4x8xi32>,
    %col_idx: tensor<4x1xi32>) -> tensor<4x1xi32> {
  %iota = stablehlo.iota dim = 0 : tensor<4x1xi32>
  %indices = stablehlo.concatenate %col_idx, %iota, dim = 1
      : (tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x2xi32>
  %result = "stablehlo.gather"(%operand, %indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1>,
    indices_are_sorted = false
  } : (tensor<4x8xi32>, tensor<4x2xi32>) -> tensor<4x1xi32>
  return %result : tensor<4x1xi32>
}

//--- part2.mlir
// （含具体常量数据与期望输出）

// 验证思路：
//   part1 是【待转换的 IR】
//   part2 是【转换后应得到的结果】（含常量数据）
//   脚本把变换应用到 part1，用 SDY 解释器执行，与 part2 对比

// 注意 part1 里 start_index_map = [1, 0] —— iota 不在第一列
//   这验证了变换不依赖"iota 必须在首位"`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'part1.mlir', c: '#38bdf8', d: '待转换的 IR' },
      { t: '变换', c: '#fbbf24', d: '<span class="mono">-sdy-explicit-gather-scatter-batching</span>' },
      { t: '解释器执行', c: '#c084fc', d: '用 SDY 解释器<b>真正跑一遍</b>' },
      { t: 'part2.mlir', c: '#4ade80', d: '期望结果（含常量数据）<br>对比数值' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:178px;opacity:.35;transition:.35s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:10.5px;overflow-wrap:anywhere">${x.t}</div>
        <div class="card-d" style="font-size:10.5px;line-height:1.45">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.35');
      msg.innerHTML = [
        '<span class="mono">part1</span> 与 <span class="mono">part2</span> 写在<b>同一个文件</b>里，用 <span class="mono">//---</span> 分隔。',
        '这就是被测的那个 pass。',
        '<b>关键</b>：不是比对文本，而是<b>执行</b> —— 因为改错属性时 IR 仍能通过校验。',
        '数值一致才证明<b>语义保持</b>。',
      ][i];
    }));
    tl.at(13600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>5 个可执行用例</b>覆盖：iota 在首位 / 末位 / 中间、reshape 后 concat、broadcast 后 concat、以及 scatter。';
    });
  }
},

/* ------------------------------------------------ 5 族谱与收官 */
{
  kicker: 'L3-11 · gather/scatter 批维',
  title: 'L3 收官：<span class="hl-a">导入流水线全图</span>',
  sub: '11 课讲完了。这一层回答的是"传播之前做了什么"。',
  caption: '把 11 课串起来，就是一份<b>导入流水线的操作清单</b>。',
  code: `// L3 十一课一览：
//   L3-01 导入流水线总览      pass 顺序、组规范化、manual 轴清理
//   L3-02 常量与标量拆分      N 个使用 -> N 份（全计划最大文件）
//   L3-03 应用分片约束        下沉的三个条件
//   L3-04 插入数据流边        每个结果一条边
//   L3-05 函数级数据流边      函数参数 + 调用结果
//   L3-06 内联函数调用        call -> named_computation
//   L3-07 提升内联网格        内联 -> 顶层声明 + 按内容去重
//   L3-08 清理 manual 轴      补 replicated + 按网格序排序
//   L3-09 分片组导入          传递闭包合并 + 重编号
//   L3-10 杂项清理            删 size-1 轴 / 排函数序 / 搬分片
//   L3-11 gather/scatter 批维 隐式批维 -> 显式批维（本课）

// 一条主线：把"用户怎么写都行"变成"传播只需处理一种形态"
//   消除假依赖（L3-02）
//   补齐不变量（L3-03 / L3-08 / L3-09）
//   统一表示（L3-04 / L3-05 / L3-06 / L3-07 / L3-11）
//   减少噪声（L3-10）`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const items = [
      { t: '总览', c: '#94a3b8' }, { t: '拆常量', c: '#4ade80' }, { t: '应用约束', c: '#38bdf8' },
      { t: '插数据流边', c: '#fbbf24' }, { t: '函数级边', c: '#c084fc' }, { t: '内联调用', c: '#f472b6' },
      { t: '提升网格', c: '#fb7185' }, { t: '清理 manual', c: '#93c5fd' }, { t: '分片组', c: '#5eead4' },
      { t: '杂项清理', c: '#a78bfa' }, { t: '批维', c: '#fdba74' },
    ];
    const host = wrap.querySelector('#cards');
    const els = items.map(x => {
      const e = U.el('div', { class: 'card', style: `width:104px;opacity:.4;transition:.3s;border-color:${x.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div style="font-size:10.5px;color:${x.c};line-height:1.35">${x.t}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>L3 共 11 课</b>，覆盖导入期每一步。'; });
    tl.at(4000, () => {
      msg.innerHTML = '<b>主线</b>：把"用户怎么写都行"变成"传播只需处理<b>一种形态</b>"。';
    });
    tl.at(7400, () => {
      msg.innerHTML = '<b>四类动作</b>：消除假依赖（L3-02）/ 补齐不变量（L3-03、08、09）/ 统一表示（L3-04~07、11）/ 减少噪声（L3-10）。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一层 L4</b> 讲导出流水线 —— 传播之后，分片如何变成真正的集合通信与切片。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L3-11 · 练习',
  title: '练一练：<span class="hl-a">为什么要显式化批维</span>',
  sub: '三道题分别考：变换内容、失败原因、验证方式。',
  caption: '一句话总结：<b>隐式批维让分片规则无法表达，显式化后才能分片</b>。',
  code: `// 题 1：iota + concat 构造的索引，变换做了什么？

// 题 2：为什么隐式批维会导致分片规则无法表达？

// 题 3：为什么需要"可执行验证"而不只是比对 IR 文本？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col compact-ex', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">iota</span> + <span class="mono">concat</span> 的索引，变换做了什么？',
        a: '把 <b>iota 那一维从 <span class="mono">start_index_map</span> 里移出</b>，改成显式的批维：' +
           '<br>• 新增 <span class="mono">operand_batching_dims = [0]</span>' +
           '<br>• 新增 <span class="mono">start_indices_batching_dims = [0]</span>' +
           '<br>• <span class="mono">start_index_map</span> 从 <span class="mono">[0, 1]</span> 变成 <span class="mono">[1]</span>' +
           '<br><span class="dim">语义不变，只是换了表达方式。</span>'
      },
      {
        q: '为什么隐式批维会让分片规则无法表达？',
        a: '四步推理：' +
           '<br>① 隐式批维写在 <span class="mono">start_index_map</span> 里 → 落在 <span class="mono">blocked_propagation</span> 的因子上；' +
           '<br>② 该标注意味着<b>传播不能沿它推导分片</b>；' +
           '<br>③ 但这一维在语义上<b>明明可以分片</b> → <b>想切却切不了</b>。' +
           '<br><span class="dim">一句话：它把"可并行的批维"伪装成了"不可传播的索引维"。</span>'
      },
      {
        q: '为什么需要"可执行验证"，而不只是比对 IR 文本？',
        a: '因为这个变换<b>改写的是算子的语义属性</b>（<span class="mono">start_index_map</span> 等）。' +
           '<br>属性改错了，IR <b>依然能通过校验</b>，但<b>数值会错</b> —— 所以必须真正执行一遍。'
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
