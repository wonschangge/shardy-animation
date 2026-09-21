/* ==========================================================================
   L2-10 · op-sharding-rule-registry   （本计划第二大文件）
   --------------------------------------------------------------------------
   覆盖：transforms/propagation/test/op_sharding_rule_registry.mlir
         (1213 行 / 125 用例)
   目标：看真实的注册表为各类算子推导出了什么规则。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 注册表 */
{
  kicker: 'L2-10 · 规则注册表',
  title: '注册表：为<span class="hl-a">每个算子</span>推导规则',
  sub: 'L1-05 讲了规则的语法。这一课看真实的注册表：<b>各类算子分别被推导出什么规则</b>。',
  caption: '这个 pass 的作用就是把规则<b>打印出来</b> —— 是学习规则写法最好的参考。',
  code: `// RUN: sdy_opt %s -sdy-populate-op-sharding-rules -verify-diagnostics

// 每个用例的 CHECK 行就是【生成的规则】：
//   stablehlo.add          -> ([i, j, k], [i, j, k])->([i, j, k]) {i=2, j=1, k=4}
//   stablehlo.broadcast_in_dim -> ([i, k, l])->([i, j, k, l]) {i=2, j=64, k=13, l=1}
//   stablehlo.dot_general  -> ([i, k], [k, j])->([i, j]) {i=8, j=16, k=32} reduction={k}
//   stablehlo.reduce       -> ([i, j, k], [])->([i, k]) {i=2, j=64, k=13} reduction={j}
//   stablehlo.reshape      -> ([i, j])->([ij]) {i=2, j=4}
//   stablehlo.convolution  -> ([i, jk, lm, n], [k, m, n, o])->([i, j, l, o])
//                             {...} reduction={k, m, n} permutation={j, l}
//   stablehlo.gather       -> (...) reduction={m, o}
//                             need_replication={k, n, p} blocked_propagation={k}

// 125 个用例 = 全部算子族的一次系统枚举`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '逐元素', d: '维度一一对应<br><b>无标注</b>', c: '#4ade80' },
      { t: '广播', d: '结果多出因子<br><b>无标注</b>', c: '#38bdf8' },
      { t: 'dot / reduce', d: '收缩维 → <b>reduction</b>', c: '#fbbf24' },
      { t: 'conv / pad', d: '尺寸不成比例 → <b>permutation</b>', c: '#c084fc' },
      { t: 'gather', d: '四类 + <b>blocked_propagation</b>', c: '#fb7185' },
      { t: 'custom_call', d: '内置注册 / 用户 <b>custom</b>', c: '#5eead4' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:118px;opacity:.4;transition:.35s;border-color:' + x.c + '55;padding:8px;text-align:center' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:11px">${x.t}</div>
        <div class="card-d" style="font-size:10.5px;line-height:1.4">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 130)); msg.innerHTML = '<b>125 个用例</b>把全部算子族系统枚举了一遍 —— 本课按这 6 组来讲。'; });
    tl.at(4200, () => {
      els.forEach((e, i) => { if (i !== 2 && i !== 3) e.style.opacity = '.25'; });
      msg.innerHTML = '最值得先掌握的是 <b>reduction</b> 与 <b>permutation</b> —— 它们决定了"切了要不要通信"。';
    });
    tl.at(7600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>用法</b>：想知道某个算子的规则？写个最小用例跑一遍这个 pass 就行。';
    });
    tl.at(10600, () => {
      msg.innerHTML = '这也是 L2-03 说的"排查先看规则"的<b>具体操作方式</b>。';
    });
  }
},

/* ------------------------------------------------ 2 基础三族 */
{
  kicker: 'L2-10 · 规则注册表',
  title: '基础三族：<span class="hl-a">逐元素 / 广播 / 形状变换</span>',
  sub: '这三类的规则最直观 —— 看懂它们，就掌握了规则写法的基本形态。',
  caption: '注意 <span class="mono">broadcast</span> 的因子 <span class="mono">j</span> <b>只出现在结果侧</b> —— 它在输入里根本不存在。',
  code: `// ① 逐元素：因子一一对应，无标注
stablehlo.add : tensor<2x1x4xf32>
//   ([i, j, k], [i, j, k])->([i, j, k]) {i=2, j=1, k=4}

// ② 广播：结果多出因子 j=64
stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 3]
     : (tensor<2x13x1xf32>) -> tensor<2x64x13x1xf32>
//   ([i, k, l])->([i, j, k, l]) {i=2, j=64, k=13, l=1}
//   输入侧没有 j —— 它是广播出来的

// ③ reshape 合并：结果侧写复合因子 [ij]
stablehlo.reshape : (tensor<2x4xf32>) -> tensor<8xf32>
//   ([i, j])->([ij]) {i=2, j=4}

// ④ 转置：只重排维度，【不需要】permutation
stablehlo.transpose %arg0, dims = [3, 1, 0, 2]
     : (tensor<256x32x64x100xf32>) -> tensor<100x32x256x64xf32>
//   ([k, j, l, i])->([i, j, k, l]) {i=100, j=32, k=256, l=64}`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 逐元素', r: '([i,j,k],[i,j,k])->([i,j,k])', c: '#4ade80',
        d: '每个因子位置相同。<br><b>无任何标注</b> —— 分片可无损传递。' },
      { t: '② 广播', r: '([i,k,l])->([i,j,k,l])', c: '#38bdf8',
        d: '结果多出 <span class="mono">j</span>。<br>它只存在于结果侧，输入侧没有。' },
      { t: '③ reshape 合并', r: '([i,j])->([ij])', c: '#c084fc',
        d: '结果侧写<b>复合因子</b> <span class="mono">[ij]</span>。<br>这正是 L2-01 讲的"沿因子传播"。' },
      { t: '④ 转置', r: '([k,j,l,i])->([i,j,k,l])', c: '#fbbf24',
        d: '只重排维序，因子大小不变<br>→ <b>不需要</b> <span class="mono">permutation</span>。' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:182px;opacity:.33;transition:.35s;border-color:' + x.c + '55;padding:9px' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:11.5px">${x.t}</div>
        <div class="mono" style="font-size:9.5px;color:#bdf7ec;margin:5px 0;overflow-wrap:anywhere">${U.esc(x.r)}</div>
        <div class="card-d" style="font-size:10.5px;line-height:1.45">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '最简单的形态。大小是 1 的维度（<span class="mono">j=1</span>）也照常参与，没有特殊处理。',
        '广播维在输入侧不存在 —— 所以输入的分片规则里没有它，传播时也不会"往输入传"。',
        '合并后的 <span class="mono">[ij]</span> 让传播知道"i 和 j 现在住在同一维里"，这是零通信 reshape 的基础。',
        '<b>对比下一幕的 pad</b>：pad 改变了尺寸且不成整数倍，就必须标 <span class="mono">permutation</span>。',
      ][i];
    }));
    tl.at(14600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>小结</b>：这三族的规则"看起来就是维度本身" —— 没有任何特殊标注。';
    });
  }
},

/* ------------------------------------------------ 3 reduction */
{
  kicker: 'L2-10 · 规则注册表',
  title: '★ <span class="mono hl-a">reduction</span>：切了就要通信的因子',
  sub: '被收缩 / 归约掉的因子必须标注 —— 它告诉传播"沿这里切会产生部分和"。',
  caption: '这是规则里<b>最有实际影响</b>的标注：<span class="mono">reduction</span> 因子上的分片会引入 all-reduce。',
  code: `// ① dot_general：k 是收缩因子
stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
     : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
//   ([i, k], [k, j])->([i, j]) {i=8, j=16, k=32} reduction={k}
//                                              ^^^^^^^^^^^^^
//   k 在两个操作数里都有，结果里没有 -> 收缩维

// ② reduce：j 被归约掉
stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1]
     : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
//   ([i, j, k], [])->([i, k]) {i=2, j=64, k=13} reduction={j}
//   注意初始化常量是 rank 0 -> 写 []

// 为什么必须标：
//   沿 k 切 -> 每台设备只有部分和 -> 需要 all-reduce
//   沿 i 切 -> 各设备结果独立   -> 零通信`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:30px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:8px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ax0)">dot_general 因子映射</div>
        <div class="row" style="gap:8px;align-items:center">
          <div class="chip c0" style="padding:6px 11px">i</div>
          <div class="chip c3" style="padding:6px 11px">k</div>
          <div class="faint">→</div>
          <div class="chip c0" style="padding:6px 11px">i</div>
          <div class="chip c1" style="padding:6px 11px">j</div>
        </div>
        <div class="small faint">k 在两侧都有、结果里没有</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<span class="mono">k</span> 是两个操作数共有的维度，且不在结果里 —— 它是<b>收缩因子</b>。';
    });
    tl.at(4600, () => {
      const c = U.el('div', { class: 'col', style: 'gap:9px;align-items:center' });
      c.innerHTML = `
        <div class="row" style="gap:14px;align-items:center">
          <div class="col" style="gap:5px;align-items:center">
            <div class="chip c0" style="padding:7px 12px">沿 i 切</div>
            <div class="small" style="color:var(--ok);font-size:11px">各设备独立 ✓</div>
          </div>
          <div class="col" style="gap:5px;align-items:center">
            <div class="chip c3" style="padding:7px 12px">沿 k 切</div>
            <div class="small" style="color:var(--warn);font-size:11px">部分和 → all-reduce</div>
          </div>
        </div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>这就是标注的意义</b>：让传播知道哪些因子上的分片会引入通信。';
    });
    tl.at(8600, () => {
      msg.innerHTML = '<span class="mono">reduce</span> 同理：<span class="mono">j=64</span> 被归约掉 → <span class="mono">reduction={j}</span>。';
    });
    tl.at(11600, () => {
      msg.innerHTML = '<b>实战含义</b>：想避免 all-reduce 就别切 <span class="mono">reduction</span> 因子 —— 但要权衡显存。';
    });
    tl.at(14000, () => {
      msg.innerHTML = '这也是 L2-04 讲的"用通信换显存"在规则层面的依据。';
    });
  }
},

/* ------------------------------------------------ 4 permutation */
{
  kicker: 'L2-10 · 规则注册表',
  title: '<span class="mono hl-a">permutation</span>：尺寸<span class="hl-a">不成整数倍</span>的因子',
  sub: '卷积的步长、pad 的边界都会让因子在两侧尺寸不同。这时必须标 `permutation`。',
  caption: '判据很清晰：<b>因子在操作数侧与结果侧的尺寸是否成整数倍</b>。转置不成问题（只重排），pad 就有问题（改变尺寸）。',
  code: `// ① 卷积：reduction 与 permutation 同时出现
stablehlo.convolution(%arg0, %arg1)
     : (tensor<2x224x224x192xf32>, tensor<3x3x192x64xf32>)
       -> tensor<2x112x112x64xf32>
//   ([i, jk, lm, n], [k, m, n, o])->([i, j, l, o])
//   {i=2, j=112, k=2, l=112, m=2, n=192, o=64}
//   reduction={k, m, n} permutation={j, l}
//
//   输入用【复合因子】jk / lm：空间维 224 = 输出 112 × 窗口 2
//   reduction：窗口 k、m 与输入通道 n
//   permutation：j/l 是 112，与输入的 224 差 2 倍（步长）

// ② pad：尺寸变了 -> permutation
stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0]
     : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
//   ([i, j, k], [])->([i, j, k]) {i=28, j=28, k=16} permutation={i, j}
//   28 -> 30 与 28 -> 26 都不是整数倍 -> i, j 标 permutation
//   k=16 没变 -> 不标

// ③ 转置：只重排 -> 不标
//   ([k, j, l, i])->([i, j, k, l]) {i=100, j=32, k=256, l=64}`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '卷积', c: '#c084fc', tag: 'reduction + permutation',
        d: '输入复合因子 <span class="mono">jk</span>/<span class="mono">lm</span>；<br>窗口与输入通道归约；<br>空间维因步长而"不成比例"。' },
      { t: 'pad', c: '#fbbf24', tag: 'permutation',
        d: '28→30、28→26 都不是整数倍<br>→ <span class="mono">i</span>、<span class="mono">j</span> 标 permutation；<br><span class="mono">k</span> 没变不标。' },
      { t: 'transpose', c: '#4ade80', tag: '无标注',
        d: '只重排维度顺序，<br>因子大小完全不变<br>→ <b>不需要</b> permutation。' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:242px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="small mono" style="font-size:10px;color:${x.c};margin:4px 0">${x.tag}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '最复杂的一条规则：复合因子 + 两类标注同时出现。值得逐段拆开看。',
        '<span class="mono">pad</span> 的判据与 <span class="mono">conv</span> 相同 —— 都是"尺寸不成整数倍"。',
        '<b>对比出真知</b>：转置看起来"动了很多"，但因子大小没变，所以规则很干净。',
      ][i];
    }));
    tl.at(14200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>判据一句话</b>：因子在两侧尺寸成整数倍 → 不标；不成 → 标 <span class="mono">permutation</span>。';
    });
  }
},

/* ------------------------------------------------ 5 gather */
{
  kicker: 'L2-10 · 规则注册表',
  title: '<span class="mono hl-a">gather</span>：四类标注<span class="hl-a">同时出现</span>',
  sub: '这是全表最复杂的一条规则，也是理解 `need_replication` 与 `blocked_propagation` 的最佳样本。',
  caption: '<span class="mono">blocked_propagation</span> 是<b>正交标注</b>（L1-05）—— 它与四类因子分类并列，不是第五类。',
  code: `stablehlo.gather(%arg0, %arg1) {
  dimension_numbers = #stablehlo.gather<
    offset_dims = [2, 3, 4],
    collapsed_slice_dims = [0],
    start_index_map = [1, 0, 3],
    index_vector_dim = 2>,
  ...
} : (tensor<3x4x2x5xf32>, tensor<2x3x3xi64>) -> tensor<2x3x2x2x1xf32>

// 生成的规则：
//   ([o, k, l, m], [i, j, p])->([i, j, k, l, n])
//   {i=2, j=3, k=4, l=2, m=5, n=1, o=3, p=3}
//   reduction={m, o} need_replication={k, n, p} blocked_propagation={k}

// 逐段读：
//   reduction={m, o}          m 是切片维；o 被 collapsed_slice_dims 折叠掉
//   need_replication={k,n,p}  这几个因子要求【复制】才能正确分片
//   blocked_propagation={k}   k 上【不做传播】

// 为什么 gather 这么复杂：
//   索引张量（%arg1）决定了从哪里取数
//   分片方式会改变"每台设备需要哪些索引" -> 可能需要复制`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'reduction', v: '{m, o}', c: '#fbbf24',
        d: '<span class="mono">m</span> 是切片维；<span class="mono">o</span> 被 <span class="mono">collapsed_slice_dims</span> 折叠。' },
      { t: 'need_replication', v: '{k, n, p}', c: '#38bdf8',
        d: '这几个因子要求<b>复制</b>才能正确分片。<br><span class="mono">p</span> 是索引维。' },
      { t: 'blocked_propagation', v: '{k}', c: '#fb7185',
        d: '<b>正交标注</b>：<span class="mono">k</span> 上不做传播。<br>与四类因子分类并列。' },
      { t: '普通因子', v: 'i, j, l', c: '#4ade80',
        d: '没有任何标注 —— 可以自由分片、无损传递。' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:180px;opacity:.33;transition:.35s;border-color:' + x.c + '55;padding:9px' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:11.5px">${x.t}</div>
        <div class="mono" style="font-size:12px;color:#bdf7ec;margin:4px 0">${U.esc(x.v)}</div>
        <div class="card-d" style="font-size:10.5px;line-height:1.45">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '与 dot / reduce 的 <span class="mono">reduction</span> 同类，但这里有两个因子。',
        '<b>need_replication 的含义</b>：这些因子上的分片会迫使数据被复制到多台设备。',
        '<b>注意它是正交的</b>：<span class="mono">k</span> 既在 <span class="mono">need_replication</span> 里，也在 <span class="mono">blocked_propagation</span> 里 —— 两者描述不同维度的事。',
        '所以一条规则可以同时携带"因子分类"和"传播约束"两组信息。',
      ][i];
    }));
    tl.at(15400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>为什么 gather 这么复杂</b>：索引张量决定了"从哪里取数"，分片会改变每台设备需要哪些索引。';
    });
  }
},

/* ------------------------------------------------ 6 custom_call */
{
  kicker: 'L2-10 · 规则注册表',
  title: '<span class="mono hl-a">custom_call</span>：内置注册 vs 用户自定义',
  sub: '注册表为 22 个已知的 `custom_call` 内置了规则；未注册的则需要用户自己写，并带上 `custom` 标记。',
  caption: '注意用户规则的末尾有 <b><span class="mono">, custom</span></b> —— 这是"规则来源"的标记，L1-05 讲语法时提到过。',
  code: `// ① 内置注册：用户什么都不用写
stablehlo.custom_call @CompactWyHelper(%arg0)
     : (tensor<128x128xf32>) -> tensor<128x128xf32>
//   规则：([i, j])->([i, j]) {i=128, j=128}

// 内置的 22 个包括：
//   CompactWyHelper / X64Combine / X64SplitHigh / X64SplitLow
//   TopK / Top2 / ApproxTopK / PartialReduce
//   Eigh / Qr / QrDecompositionBlock / HouseholderProduct
//   InspectSharding / MoveToDevice / MoveToHost / LayoutConstraint
//   Erf / XlaMegascaleProvideMetadata ...

// ② 未注册 + 用户自定义规则
stablehlo.custom_call @foo(%arg0)
     {sdy.sharding_rule = #sdy.op_sharding_rule<
        ([i, j])->([j, i]) {i=4, j=2}, custom>}
     : (tensor<4x2xf32>) -> tensor<2x4xf32>
//   ^^^^^^ 末尾的 ", custom" 表示来自用户`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 内置注册', c: '#4ade80', n: '22 个',
        d: '注册表里写好了规则，用户<b>无需标注</b>。<br>如 <span class="mono">@CompactWyHelper</span>、<span class="mono">@TopK</span>、<span class="mono">@Eigh</span>。' },
      { t: '② 用户自定义', c: '#fbbf24', n: '带 custom',
        d: '未注册的算子，用户自己写规则。<br>末尾带 <span class="mono">, custom</span> 标记来源。' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:12px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>内置注册</b>覆盖了 XLA 常用的自定义算子 —— 这些算子本身没有标准语义，只能靠注册表。'; });
    tl.at(4400, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>用户自定义</b>是扩展点：写自己的 <span class="mono">sdy.sharding_rule</span> 属性即可。';
    });
    tl.at(8200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>为什么必须显式标 <span class="mono">custom</span></b>：让后续 pass 知道这条规则是"用户意图"，不能随意覆盖或删除。';
    });
    tl.at(11600, () => {
      msg.innerHTML = '这也与 L2-03 呼应：打开 <span class="mono">keep-sharding-rules</span> 时，你就能看到这些规则留在 IR 里。';
    });
  }
},

/* ------------------------------------------------------------ 7 练习 */
{
  kicker: 'L2-10 · 练习',
  title: '练一练：<span class="hl-a">读懂一条规则</span>',
  sub: '三道题分别考：reduction 判据、permutation 判据、custom 标记。',
  caption: '掌握这三条，就能读懂注册表里绝大多数规则。',
  code: `// 题 1：([i, k], [k, j])->([i, j]) 里，哪个因子是 reduction？

// 题 2：为什么 transpose 不需要 permutation，
//       而 pad 需要？

// 题 3：规则末尾的 ", custom" 是什么意思？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">([i, k], [k, j])-&gt;([i, j])</span> 里，哪个因子是 <span class="mono">reduction</span>？',
        a: '<b><span class="mono">k</span></b>。判据：它在<b>两个操作数里都出现</b>，但<b>结果里没有</b>。' +
           '<br><b>含义</b>：沿 <span class="mono">k</span> 切会产生部分和 → 需要 all-reduce。' +
           '<br><span class="dim">对比 <span class="mono">i</span> 与 <span class="mono">j</span> —— 它们是 pass-through。</span>'
      },
      {
        q: '为什么 <span class="mono">transpose</span> 不需要 <span class="mono">permutation</span>，而 <span class="mono">pad</span> 需要？',
        a: '<b>判据是"因子在两侧的尺寸是否成整数倍"</b>。' +
           '<br><span class="mono">transpose</span>：只重排维序，因子大小完全不变 → 不标。' +
           '<br><span class="mono">pad</span>：28→30、28→26，都不成整数倍 → 标 <span class="mono">permutation={i, j}</span>。' +
           '<br><span class="dim">卷积同理：224→112 差 2 倍（步长）。</span>'
      },
      {
        q: '规则末尾的 <span class="mono">, custom</span> 是什么意思？',
        a: '表示这条规则<b>来自用户</b>，而不是注册表推导的。' +
           '<br><b>用途</b>：让后续 pass 知道这是"用户意图"，不能随意覆盖或删除。' +
           '<br><span class="dim">这是为未注册的 <span class="mono">custom_call</span> 提供规则的扩展点。</span>'
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
