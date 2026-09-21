/* ==========================================================================
   L1-05 · op-sharding-rule
   --------------------------------------------------------------------------
   覆盖：ir/test/sharding_rule_parse_print.mlir (69)
         ir/test/sharding_rule_parsing_failure.mlir (223)
         ir/test/sharding_rule_verification.mlir (151)
   目标：讲透 #sdy.op_sharding_rule —— 因子映射、复合因子、符号命名、四分类。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------------ 1 动机 */
{
  kicker: 'L1-05 · 算子分片规则',
  title: '为什么需要<span class="hl-a">分片规则</span>？',
  sub: '传播算法不该认识几百个算子。它只需要知道一件事：这个算子的<b>哪些维度是同一个东西</b>。',
  caption: '把"算子语义"和"传播算法"解耦，是 Shardy 的核心设计。新增算子只要给一条规则，传播自动就能工作。',
  code: `// 没有规则：传播算法得为每个算子写一遍逻辑
//   if (op 是 add)  { 各维一一对应 }
//   if (op 是 dot)  { 收缩维要特殊处理 }
//   if (op 是 reshape) { 维度会合并/拆分 }
//   ... 几百个算子，几百段逻辑

// 有了规则：算子自己声明"维度 ↔ 因子"的映射
%0 = stablehlo.dot_general %arg0, %arg1 {
  sdy.sharding_rule = #sdy.op_sharding_rule<
      ([i, k], [k, j])->([i, j])   // ← 就这一张表
      {i=8, j=16, k=8}>
} : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>`,
  duration: 12000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:20px;justify-content:center;align-items:center">
        <div class="card" style="width:250px;border-color:rgba(251,113,133,.45)">
          <div class="card-t" style="color:var(--bad)">没有规则</div>
          <div class="card-d">传播算法里 <span class="mono">switch(op)</span> 几百个分支<br>
            <span class="dim">新增算子 = 改传播算法</span></div>
        </div>
        <div class="arrow anim" style="font-size:26px">⟹</div>
        <div class="card" style="width:250px;border-color:rgba(94,234,212,.5)">
          <div class="card-t" style="color:var(--accent)">有规则</div>
          <div class="card-d">算子自带一张<b>因子映射表</b><br>
            <span class="dim">新增算子 = 写一张表</span></div>
        </div>
      </div>
      <div class="formula" id="msg" style="min-height:46px;display:flex;align-items:center;text-align:center;max-width:720px"></div>`;
    root.appendChild(wrap);
    const msg = wrap.querySelector('#msg');
    tl.at(900, () => { msg.innerHTML = '传播算法只做一件事：<b>沿因子把分片推过去</b>。'; });
    tl.at(3400, () => { msg.innerHTML = '算子告诉它"哪些维度共享同一个因子"，剩下的算法统一处理。'; });
    tl.at(6200, () => { msg.innerHTML = '这就是为什么 SDY 的传播代码能对 <b>stablehlo + sdy + 自定义方言</b>同时生效。'; });
    tl.at(9000, () => { msg.innerHTML = '本课讲规则本身；<b>L2-10</b> 会把所有内置算子的规则做成速查表。'; });
  }
},

/* ------------------------------------------------------------ 2 语法 */
{
  kicker: 'L1-05 · 算子分片规则',
  title: '一条规则由三部分组成',
  sub: '操作数/结果的<b>维度映射</b>、每个<b>因子的大小</b>、以及可选的<b>特殊因子标注</b>。',
  caption: '注意映射里的每个符号代表一个<b>因子</b>，不是维度本身。维度与因子是<b>多对多</b>关系。',
  code: `#sdy.op_sharding_rule<
  ([i, k], [k, j]) -> ([i, j])          // ① 维度映射
  {i = 8, j = 16, k = 8}                // ② 因子大小
  reduction={k} need_replication={} ... // ③ 特殊因子（可选）
  , custom                              //    标记为用户自定义（可选）
>

// ① 的读法：operand0 的两个维度依次映射到因子 i、k
//            operand1 的两个维度依次映射到因子 k、j
//            result   的两个维度依次映射到因子 i、j

// ② 每个因子必须给出大小，且**按 iota 顺序**书写

// 映射个数 = 操作数/结果个数；每个映射的 rank = 该张量的 rank`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="mono" style="font-size:14.5px;line-height:1.95;padding:12px 18px;border-radius:10px;
           background:rgba(0,0,0,.3);border:1px solid var(--panel-brd)">
        <div><span style="color:#c084fc">#sdy.op_sharding_rule&lt;</span></div>
        <div>&nbsp;&nbsp;<span id="p1">([i, k], [k, j]) -&gt; ([i, j])</span></div>
        <div>&nbsp;&nbsp;<span id="p2">{i = 8, j = 16, k = 8}</span></div>
        <div>&nbsp;&nbsp;<span id="p3">reduction={k} need_replication={}</span></div>
        <div>&nbsp;&nbsp;<span id="p4">, custom</span></div>
        <div><span style="color:#c084fc">&gt;</span></div>
      </div>
      <div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:42px;display:flex;align-items:center;text-align:center"></div>`;
    root.appendChild(wrap);

    const ids = ['p1', 'p2', 'p3', 'p4'].map(i => wrap.querySelector('#' + i));
    const host = wrap.querySelector('#cards');
    const infos = [
      { t: '① 维度映射', d: '每个操作数/结果一个 <span class="mono">[...]</span>，<br>里面按维度顺序写因子符号。' },
      { t: '② 因子大小', d: '每个因子一个 <span class="mono">名字=大小</span>，<br>必须按 <span class="mono">i, j, k…</span> 的 iota 顺序。' },
      { t: '③ 特殊因子', d: '四类标注：<span class="mono">reduction</span> /<br><span class="mono">need_replication</span> / <span class="mono">permutation</span> / <span class="mono">blocked_propagation</span>。' },
      { t: '④ custom', d: '标记"用户自定义"，<br>传播不会删除它。' },
    ];
    const cards = infos.map(i => {
      const e = U.el('div', { class: 'card', style: 'width:170px;opacity:.3;transition:.3s;padding:9px' });
      e.innerHTML = `<div class="card-t" style="font-size:11.5px">${i.t}</div>
        <div class="card-d" style="font-size:11px;line-height:1.5">${i.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    infos.forEach((_, i) => tl.at(700 + i * 2900, () => {
      cards.forEach((c, k) => c.style.opacity = k === i ? '1' : '.3');
      ids.forEach((e, k) => { e.style.color = k === i ? 'var(--accent)' : ''; e.style.fontWeight = k === i ? '700' : ''; });
      msg.innerHTML = [
        '映射里每个符号是一个<b>因子</b>。同一个符号出现在不同张量上 = 它们共享这个因子。',
        '规则里出现的每个因子都必须给大小；反过来，给了大小的因子必须<b>被用到</b>。',
        '不写特殊标注时，所有因子都是 <b>pass-through</b>（直通）。',
        '<span class="mono">custom</span> 只能对自定义规则使用；内置规则不会带它。',
      ][i];
    }));
    tl.at(12400, () => {
      cards.forEach(c => c.style.opacity = '1');
      ids.forEach(e => { e.style.color = ''; e.style.fontWeight = ''; });
      msg.innerHTML = '三部分都可省略到只剩映射 —— 但映射本身必须存在。';
    });
  }
},

/* -------------------------------------------------------- 3 因子直觉 */
{
  kicker: 'L1-05 · 算子分片规则',
  title: '因子就是 <span class="mono hl-a">einsum</span> 的下标',
  sub: '把算子写成 einsum，因子立刻一目了然。传播做的事：<b>同一个因子在所有张量上必须同样分片</b>。',
  caption: '这就是传播算法的全部输入 —— 它不需要知道这是 matmul 还是卷积。',
  code: `// matmul：C = dot(A, B)
%0 = stablehlo.dot_general %arg0, %arg1 {
  sdy.sharding_rule = #sdy.op_sharding_rule<
      ([i, k], [k, j])->([i, j]) {i=8, j=16, k=8}>
}

// 逐元素：三个张量维度一一对应
#sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=8}>

// 归约：operand 有 j，result 没有 → j 是 reduction 因子
#sdy.op_sharding_rule<([i, j])->([i]) {i=8, j=4} reduction={j}

// rank-0：三个空列表
#sdy.op_sharding_rule<([], [])->([])>`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:14px;align-items:center;justify-content:center" id="ops"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:730px"></div>`;
    root.appendChild(wrap);
    const ops = wrap.querySelector('#ops');
    const msg = wrap.querySelector('#msg');

    const tensor = (label, dims, color) => {
      const c = U.el('div', { class: 'col', style: 'gap:5px;align-items:center' });
      c.innerHTML = `<div class="small mono faint">${label}</div>
        <div class="row" style="gap:5px" data-d></div>`;
      const h = c.querySelector('[data-d]');
      dims.forEach(d => {
        const e = U.el('div', { class: 'chip mut', style: 'padding:6px 11px;font-size:14px' });
        e.textContent = d; e.dataset.f = d; h.appendChild(e);
      });
      if (color) c.querySelector('.small').style.color = color;
      ops.appendChild(c);
      return c;
    };
    const t0 = tensor('operand0', ['i', 'k']);
    ops.appendChild(U.el('div', { class: 'arrow', html: '<span class="mono small dim">dot</span> ⟹' }));
    const t1 = tensor('operand1', ['k', 'j']);
    ops.appendChild(U.el('div', { class: 'arrow', html: '⟹' }));
    const t2 = tensor('result', ['i', 'j']);

    const hl = (box, f, cls) => {
      const e = box.querySelector(`[data-f="${f}"]`);
      if (e) { e.className = 'chip ' + cls; e.style.padding = '6px 11px'; e.style.fontSize = '14px'; }
    };
    tl.at(900, () => { msg.innerHTML = 'einsum 写法：<span class="mono">(i,k),(k,j)-&gt;(i,j)</span> —— 直接把下标抄进规则。'; });
    tl.at(3400, () => {
      msg.innerHTML = '把 <span class="mono">i</span> 沿轴 "x" 分片：operand0 的第 0 维、result 的第 0 维<b>一起</b>被切';
      hl(t0, 'i', 'c0'); hl(t2, 'i', 'c0');
    });
    tl.at(6400, () => {
      msg.innerHTML = 'operand1 里<b>没有</b> i → 它沿 "x" <span class="hl-w">复制</span>，因为它的每一份都要参与所有 i 的计算';
    });
    tl.at(9200, () => {
      msg.innerHTML = '把 <span class="mono">k</span> 沿轴 "y" 分片：两个操作数<b>都要切</b>，但 result 里没有 k';
      hl(t0, 'k', 'c1'); hl(t1, 'k', 'c1');
    });
    tl.at(12400, () => { msg.innerHTML = '<span class="mono">k</span> 是 <b>reduction 因子</b>：结果是部分和，需要 all-reduce 才能用。'; });
  }
},

/* ------------------------------------------------------- 4 复合因子 */
{
  kicker: 'L1-05 · 算子分片规则',
  title: '复合因子：reshape 怎么表达',
  sub: 'reshape 会合并或拆分维度，一个维度可能对应<b>多个因子</b>。写法是把符号拼在一起：<span class="mono">[ij]</span>。',
  caption: '注意<b>不能有空格</b>：<span class="mono">[ij]</span> 是一个复合因子，<span class="mono">[i j]</span> 会被解析成两个维度而报错。',
  code: `// 合并：2x4 -> 8，两个因子并成一个维度
#sdy.op_sharding_rule<([i, j])->([ij]) {i=2, j=4}>

// 拆分：8 -> 2x4，一个维度拆成两个因子
#sdy.op_sharding_rule<([ij])->([i, j]) {i=2, j=4}>

// 部分重叠：8x4 -> 2x16
//   ((ij), k) -> (i, (jk))   i=2, j=4, k=4
#sdy.op_sharding_rule<([ij, k])->([i, jk]) {i=2, j=4, k=4}>

// 为什么需要它？因为传播是沿**因子**进行的：
//   分片信息在因子空间里传递，再投影回维度。`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:22px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:730px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const mk = (label, cells, cls) => {
      const c = U.el('div', { class: 'col', style: 'gap:5px;align-items:center' });
      c.innerHTML = `<div class="small mono faint">${label}</div><div class="row" style="gap:5px" data-d></div>`;
      const h = c.querySelector('[data-d]');
      cells.forEach(t => {
        const e = U.el('div', { class: 'chip ' + cls, style: 'padding:7px 13px;font-size:15px' });
        e.textContent = t; e.dataset.f = t; h.appendChild(e);
      });
      demo.appendChild(c); return c;
    };

    tl.at(700, () => {
      const a = mk('reshape 前 tensor<2x4>', ['i', 'j'], 'c0');
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹' }));
      const b = mk('reshape 后 tensor<8>', ['ij'], 'c2');
      msg.innerHTML = '两个维度合并 → 写成一个复合因子 <span class="mono">[ij]</span>（i 与 j 紧挨着，无空格）';
    });
    tl.at(4200, () => {
      msg.innerHTML = '<b>关键性质</b>：如果 <span class="mono">i</span> 已经沿某轴分片，合并后这个分片信息仍在 <span class="mono">ij</span> 里 —— 传播能对上。';
      demo.querySelectorAll('.chip').forEach(e => e.classList.add('pulse'));
    });
    tl.at(7600, () => {
      demo.innerHTML = '';
      const a = mk('reshape 前 tensor<8x4>', ['ij', 'k'], 'c2');
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹' }));
      const b = mk('reshape 后 tensor<2x16>', ['i', 'jk'], 'c2');
      msg.innerHTML = '部分重叠：<span class="mono">((ij), k) -&gt; (i, (jk))</span> —— 因子 j 横跨两个张量的不同维度。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '这正是 <b>L1-02 子轴</b>能解决 reshape 通信问题的原因：<span class="mono">i</span> 与 <span class="mono">j</span> 各自对应一段设备，合并后数据不用动。';
    });
  }
},

/* ----------------------------------------------------- 5 因子符号命名 */
{
  kicker: 'L1-05 · 算子分片规则',
  title: '因子符号：<span class="mono hl-a">i</span>–<span class="mono hl-a">z</span> 用完了怎么办',
  sub: '符号表只有 18 个（i 到 z）。第 19 个起用 <span class="mono">z_1</span>、<span class="mono">z_2</span>… 依次续接。',
  caption: '复合因子里把相邻符号拼起来，于是会出现 <span class="mono">zz_1</span>（z 与 z_1）甚至 <span class="mono">z_8z_9z_10</span> 这种写法。',
  code: `// 18 个基本符号：i j k l m n o p q r s t u v w x y z
// 之后按 z_1, z_2, z_3, ... 续接

// rank-20 张量，映射的最后两位：
([... , z, z_1, z_2])            // 声明时逐个写
    -> ([... , zz_1, z_2])       // 复合时拼在一起：z + z_1 = zz_1

// rank-28 张量，最后一维由三个符号复合：
-> ([..., z, z_1, ..., z_7, z_8z_9z_10])

// 校验：符号索引必须 <= 2^63-1，且 z_ 后必须是
//       正整数、无前导零`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:4px;flex-wrap:wrap;justify-content:center;max-width:760px" id="syms"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:740px"></div>`;
    root.appendChild(wrap);
    const host = wrap.querySelector('#syms'), msg = wrap.querySelector('#msg');
    const syms = 'i j k l m n o p q r s t u v w x y z'.split(' ');
    const els = syms.map((s, i) => {
      const e = U.el('div', { class: 'chip c' + (i % 6), style: 'padding:4px 9px;font-size:13px;opacity:0;transition:.3s' });
      e.textContent = s; host.appendChild(e); return e;
    });
    const extra = ['z_1', 'z_2', 'z_3', '…'].map((s, i) => {
      const e = U.el('div', { class: 'chip mut', style: 'padding:4px 9px;font-size:13px;opacity:0;transition:.3s' });
      e.textContent = s; e.dataset.extra = '1'; host.appendChild(e); return e;
    });

    tl.at(700, () => {
      els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 55));
      msg.innerHTML = '18 个基本符号 <span class="mono">i</span> 到 <span class="mono">z</span> —— 覆盖 rank ≤ 18 的张量。';
    });
    tl.at(3000, () => {
      extra.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 120));
      msg.innerHTML = '第 19 个起改用 <span class="mono">z_1</span>、<span class="mono">z_2</span>…（<b>不是</b> aa、ab）';
    });
    tl.at(5800, () => { msg.innerHTML = '规则：<span class="mono">z_</span> 后面必须是<b>正整数</b>，不能有前导零（<span class="mono">z_0</span> ✗、<span class="mono">z_01</span> ✗）。'; });
    tl.at(8600, () => {
      msg.innerHTML = '声明时逐个写 <span class="mono">z_1</span>、<span class="mono">z_2</span>；<b>复合时直接拼接</b> → <span class="mono">zz_1</span>（= z + z_1）';
    });
    tl.at(11600, () => {
      msg.innerHTML = 'rank-28 的最后一维可以是 <span class="mono">z_8z_9z_10</span> —— 这就是<b>复合因子</b>在长符号上的样子。';
    });
  }
},

/* --------------------------------------------------------- 6 四分类 */
{
  kicker: 'L1-05 · 算子分片规则',
  title: '因子分类：<span class="hl-a">三类特殊</span> + 一类正交',
  sub: '官方定义是<b>四类</b>：不标注即为 <b>pass-through</b>（直通），另有三类特殊因子。' +
       '<span class="mono">blocked_propagation</span> 则与它们<b>正交</b> —— 不改变因子种类，只禁止传播。',
  caption: '三类特殊因子<b>互斥</b>：一个因子只能属于其中一类（或都不属于 = pass-through）。<span class="mono">blocked_propagation</span> 可叠加。',
  code: `#sdy.op_sharding_rule<
  ([i, j, k, l])->([i, k, l]) {i=2, j=3, k=5, l=7}
  reduction={j}             // 归约因子：结果里没有它 -> all-reduce
  need_replication={i, l}   // 必须全复制 -> 不能分片，如 sort 的被排序维
  permutation={k}           // 需要重排 -> collective-permute / halo exchange
  blocked_propagation={l}   // 禁止沿它传播
>

// pass-through：不在以上任何集合中
//   只要所有张量上该因子同样分片，就不需要额外通信`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%;align-items:center' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:740px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'pass-through', m: '（默认）', d: '同因子在所有张量上同样分片即可。<br><b>无额外通信</b>。', c: '#4ade80' },
      { t: 'reduction', m: 'reduction={j}', d: '结果里没有该因子 → 各设备只有部分和。<br>需 <b>all-reduce</b>。', c: '#fbbf24' },
      { t: 'need_replication', m: 'need_replication={i}', d: '该因子分片时所有操作数都要全复制。<br>如 sort 的被排序维。', c: '#38bdf8' },
      { t: 'permutation', m: 'permutation={k}', d: '分片会打乱数据对应关系。<br>需 <b>collective-permute</b> / halo exchange。', c: '#c084fc' },
      { t: 'blocked_propagation', m: 'blocked_propagation={l}', d: '禁止沿该因子传播。<br>与前三类<b>正交</b>，可叠加。', c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => {
      const e = U.el('div', { class: 'card', style: 'width:143px;opacity:.32;transition:.3s;padding:9px' });
      e.innerHTML = `<div class="card-t" style="font-size:11.5px;color:${d.c}">${d.t}</div>
        <div class="mono" style="margin:5px 0;font-size:9.5px;color:#bdf7ec;overflow-wrap:anywhere">${U.esc(d.m)}</div>
        <div class="card-d" style="font-size:10.5px;line-height:1.45">${d.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 2900, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.32');
      msg.innerHTML = [
        '绝大多数因子都是这一类。传播只需要"对齐分片"，不需要移动数据。',
        '典型：matmul 的收缩维。切了它，结果就是未归约的（L1-02 的 <span class="mono">unreduced</span>）。',
        '典型：<span class="mono">sort</span> 的被排序维 —— 排序需要看到全部数据，不能切。',
        '典型：<span class="mono">pad</span> 的补齐维、<span class="mono">reverse</span> 的被翻转维。',
        '<b>正交</b>：它不属于那三类中的任何一类，而是对任意因子附加"禁止传播"。',
      ][i];
    }));
    tl.at(700 + defs.length * 2900, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '记忆法：<b>三类特殊因子决定"要不要通信"，blocked_propagation 决定"能不能传播"</b>。';
    });
  }
},

/* ------------------------------------------------------- 7 解析错误 */
{
  kicker: 'L1-05 · 规则校验',
  title: '解析错误组 1：<span class="hl-a">符号与写法</span>',
  sub: '47 条解析错误里，一大半在符号命名和书写形式上。归成 4 组看。',
  caption: '这些都属于"规则本身没写成一个合法 attribute"，报文里会出现 <span class="mono">failed to parse</span>。',
  code: `// ① custom 标记写错
... {i=16, j=32},>                 // 逗号后没有 custom
... {i=16, j=32}, custom_rule>     // 只能是 custom
//   expected 'custom'

// ② 复合因子不能有空格
([i, j])->([i j])                  // ✗ 应为 [ij]
//   expected ']'

// ③ 因子大小必须按 iota 顺序
{j=4, i=2}                         // ✗ 应为 {i=2, j=4}
{i=2, k=4}                         // ✗ 跳过了 j
//   expecting factor indices to be ordered like an iota

// ④ 符号必须是 i-z 或 z_N
([a, j])                           // ✗ 'a' 不在 i..z
([z_, j])                          // ✗ z_ 后缺整数
([i_1, j])                         // ✗ 只有 z 可以带下标`,
  duration: 18000,
  build(root, tl) {
    const wrap = W.errLayout(root);
    W.errScenes(wrap, tl, [
      {
        ir: '%0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=16, j=32},>} : (tensor<16x32xf32>) -> tensor<16x32xf32>',
        err: "expected 'custom'",
        why: '逗号后面必须紧跟关键字 <span class="mono">custom</span>，不能空着。'
      },
      {
        ir: '%0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=16, j=32}, custom_rule>} : (tensor<16x32xf32>) -> tensor<16x32xf32>',
        err: "expected 'custom'",
        why: '关键字只能是 <span class="mono">custom</span>，<span class="mono">custom_rule</span> 不认识。'
      },
      {
        ir: '%0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i j]) {i=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>',
        err: "failed to parse Sdy_OpShardingRule parameter 'result_mappings' … expected ']'",
        why: '复合因子必须写成 <span class="mono">[ij]</span>。<span class="mono">[i j]</span> 会被当成"两个维度"，而结果只有 1 维。'
      },
      {
        ir: '%0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([ij]) {j=4, i=2}>} : (tensor<2x4xf32>) -> tensor<8xf32>',
        err: 'expecting factor indices to be ordered like an iota ([0,1,2,...], e.g. {i=#, j=#, ...})',
        why: '因子大小必须<b>按符号顺序</b>写：<span class="mono">{i=2, j=4}</span>。写成 j 在前会被拒绝。'
      },
      {
        ir: '%0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, k])->([ik]) {i=2, k=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>',
        err: 'expecting factor indices to be ordered like an iota ([0,1,2,...], e.g. {i=#, j=#, ...})',
        why: '也不能<b>跳过</b>符号：用了 <span class="mono">i</span> 和 <span class="mono">k</span> 就必须也有 <span class="mono">j</span>。'
      },
      {
        ir: '%0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([a, j])->([j]) {a=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>',
        err: "expecting symbol from 'i' to 'z'. Received: 'a'",
        why: '基本符号只有 <span class="mono">i</span>–<span class="mono">z</span>。<span class="mono">a</span> 不在其中。'
      },
      {
        ir: '%0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([z_0z_, j])->([i]) {z_0=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>',
        err: 'expecting positive integer without leading zeros. Received: \'0\'',
        why: '<span class="mono">z_0</span> 非法：下标必须从 <b>1</b> 开始，且不能有前导零。'
      },
      {
        ir: '%0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([z_-1, j])->([i]) {i=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>',
        err: "expecting integer after 'z_'",
        why: '<span class="mono">z_</span> 后面必须是整数，<span class="mono">-1</span> 不是正整数。'
      },
    ], {
      finalIr: '// 符号命名速记：\n//   i..z 共 18 个\n//   z_1, z_2, ... 续接（从 1 起，无前导零）\n//   复合 = 拼接（zz_1, z_8z_9z_10）',
      finalErr: "expected 'custom' / expected ']' / ordered like an iota / symbol from 'i' to 'z'",
      finalWhy: '这些错误在报文里都能直接看到"期望什么"，照着改即可。'
    });
  }
},

/* ------------------------------------------------- 8 语义校验错误 */
{
  kicker: 'L1-05 · 规则校验',
  title: '校验错误：<span class="hl-a">映射与张量必须对得上</span>',
  sub: '语法过关后，校验器检查规则与<b>宿主算子</b>是否自洽：数量、rank、因子使用情况。',
  caption: '共同点：规则是"为这个算子量身定制"的，所以必须与它的操作数/结果结构完全吻合。',
  code: `// ① 属性类型
{sdy.sharding_rule = 1 : i64}
//   should have a sharding rule attribute of type OpShardingRuleAttr

// ② 只支持静态形状的 ranked tensor
rule ... : tensor<*xf32>            // ✗ 无 rank
rule ... : tensor<?x?xf32>          // ✗ 动态

// ③ 映射个数 = 操作数/结果个数
([i, j])->([i, j]) 用在 2 个操作数上  // ✗ 少了一个
([i, j], [i, j])->([i, j], [i, j]) 用在 1 个结果上 // ✗ 多了一个

// ④ 每个映射的 rank = 该张量 rank
([i, j], [i])->([i, j])  : tensor<2x4>   // ✗ operand1 是 rank 2

// ⑤ 因子使用：不能重复、不能空、不能越界、不能未使用
([i, i])->([ij]) {i=2, j=2}          // ✗ 同一张量里 i 用了两次
([i,], [i,])->([i, j]) {i=2, j=8}    // ✗ 空了
([i, u], [i, j])->([i, j]) {i=2, j=8} // ✗ u 越界（只有 2 个因子）
{i=2, j=8, k=2} 但 k 没被用到        // ✗

// ⑥ 特殊因子集合：有序、唯一、在范围内、互斥、不能在结果里
need_replication={k, i}    // ✗ 无序
need_replication={i, i}    // ✗ 重复
need_replication={z}       // ✗ 越界（只有 i,j,k）
reduction={j} need_replication={j}  // ✗ 一个因子只能属于一类`,
  duration: 20000,
  build(root, tl) {
    const wrap = W.errLayout(root);
    W.errScenes(wrap, tl, [
      {
        ir: '%0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = 1 : i64} : (tensor<8xf32>) -> tensor<8xf32>',
        err: 'should have a sharding rule attribute of type OpShardingRuleAttr',
        why: '必须挂 <b>OpShardingRuleAttr</b>，给整数不行。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=2, j=4}>} : tensor<?x?xf32>',
        err: 'operand 0 - expected a ranked tensor with a static shape',
        why: '规则要按维度算局部形状，所以<b>动态形状不行</b>（<span class="mono">tensor&lt;*xf32&gt;</span> 无 rank 同样不行）。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=2, j=4}>} : tensor<2x4xf32>',
        err: 'number of operands and mappings must match: 2 != 1',
        why: '<span class="mono">stablehlo.add</span> 有 2 个操作数，规则只给了 1 个映射。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i])->([i, j]) {i=2, j=4}>} : tensor<2x4xf32>',
        err: 'operand 1 - mapping rank must match: 1 != 2',
        why: 'operand1 是 rank 2（2x4），映射 <span class="mono">[i]</span> 只有 1 项。'
      },
      {
        ir: '%0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, i])->([ij]) {i=2, j=2}>} : (tensor<2x2xf32>) -> tensor<4xf32>',
        err: 'op operand - cannot reuse factors for the same tensor value',
        why: '同一个张量里因子 <span class="mono">i</span> 出现两次 → 两个维度会互相关联，语义不明。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=2, j=8, k=2}>} : tensor<2x8xf32>',
        err: "has factor k=2 that isn't used in operand and result mappings",
        why: '声明了因子 <span class="mono">k</span> 但没有任何维度映射到它 → 冗余。'
      },
      {
        ir: '%0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, k]) {i=2, j=4, k=8} need_replication={k, i}>} : (tensor<2x4x8xf32>) -> tensor<2x8xf32>',
        err: 'indices of special factors must be sorted',
        why: '特殊因子集合必须<b>按因子索引升序</b>：应写成 <span class="mono">{i, k}</span>。'
      },
      {
        ir: '%0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, k]) {i=2, j=4, k=8} need_replication={z}>} : (tensor<2x4x8xf32>) -> tensor<2x8xf32>',
        err: 'index must be less than 3, got: 17',
        why: '<span class="mono">z</span> 是第 18 个符号（索引 17），但这条规则只有 3 个因子。'
      },
    ], {
      finalIr: '// 规则与算子必须自洽：\n//   映射个数 == 操作数/结果个数\n//   每个映射 rank == 该张量 rank\n//   因子：用到、不重复、不越界\n//   特殊集合：有序、唯一、互斥',
      finalErr: 'OpShardingRuleAttr / ranked tensor with a static shape / must match / cannot reuse',
      finalWhy: '本质：规则是<b>为这个算子量身定制</b>的，必须与它的结构完全吻合。'
    });
  }
},

/* ------------------------------------------------------------ 9 练习 */
{
  kicker: 'L1-05 · 练习',
  title: '练一练：<span class="hl-a">读规则、写规则</span>',
  sub: '三道题分别考：读映射、判断特殊因子、找错误。',
  caption: '能独立写出规则，就能在 L7-04 里为自己的方言接入 Shardy。',
  code: `// 题 1：这条规则描述的是什么算子？
#sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=8}>

// 题 2：sum 沿第 1 维归约，规则该怎么写？
//   tensor<8x4xf32> -> tensor<8xf32>

// 题 3：错在哪？
#sdy.op_sharding_rule<([i, j])->([ij]) {i=2, j=4}>`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">([i, j], [i, j])-&gt;([i, j]) {i=8, j=8}</span> 描述的是什么算子？',
        a: '<b>逐元素算子</b>（如 <span class="mono">add</span> / <span class="mono">multiply</span>）：两个操作数与结果维度一一对应，' +
           '共享同一对因子 <span class="mono">i</span>、<span class="mono">j</span>。<br>' +
           '<span class="dim">没有任何特殊因子 → 全直通，不需要额外通信。</span>'
      },
      {
        q: '<span class="mono">sum</span> 沿第 1 维归约：<span class="mono">tensor&lt;8x4xf32&gt; -&gt; tensor&lt;8xf32&gt;</span>，规则怎么写？',
        a: '<span class="mono">#sdy.op_sharding_rule&lt;([i, j])-&gt;([i]) {i=8, j=4} reduction={j}&gt;</span><br>' +
           '<b>关键</b>：因子 <span class="mono">j</span> 出现在 operand 但不在 result → 必须标 <span class="mono">reduction={j}</span>，' +
           '这样传播才知道切开 <span class="mono">j</span> 会得到部分和。'
      },
      {
        q: '<span class="mono">#sdy.op_sharding_rule&lt;([i, j])-&gt;([ij]) {i=2, j=4}&gt;</span> 用在 <span class="mono">reshape (tensor&lt;2x4xf32&gt;) -&gt; tensor&lt;8xf32&gt;</span> 上，错在哪？',
        a: '<b class="badge ok">其实是对的</b> —— 这正是 <span class="mono">@nested</span> 用例的规则：两个维度合并成复合因子 <span class="mono">[ij]</span>。<br>' +
           '<span class="dim">检查要点：① 因子大小按 iota 顺序 ✓ ② 每张量内因子不重复 ✓ ③ 映射 rank 与张量一致 ✓ ④ 因子都被用到 ✓</span>'
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
