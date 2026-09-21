/* ==========================================================================
   L2-03 · keep-sharding-rules
   --------------------------------------------------------------------------
   覆盖：transforms/propagation/test/basic_propagation_keep_sharding_rules.mlir (17)
   目标：讲透 keep-sharding-rules 选项 —— 把传播用过的规则留在 IR 里。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 选项 */
{
  kicker: 'L2-03 · 保留分片规则',
  title: '把传播用过的规则<span class="hl-a">留下来</span>',
  sub: '传播靠 <span class="mono">sharding_rule</span> 推导分片。默认情况下用完就删（保持 IR 干净）。这个选项让它们留下来。',
  caption: '这是**调试工具**：当传播结果不符合预期时，第一件事就是看"它当时用的是什么规则"。为 L2-12 做铺垫。',
  code: `// 默认：规则是临时的，用完就删
// RUN: sdy_opt %s -sdy-basic-propagate

// 保留：把规则留在 IR 里
// RUN: sdy_opt %s -sdy-basic-propagate='keep-sharding-rules=true'

// 留下的规则分两类：
//   ① 算子原本就有的（用户自定义规则，如 custom_call）
//   ② 传播【推导】出来的（为没有规则的算子补上）

// 用途：
//   - 确认传播用的是哪条规则
//   - 检查推导出的规则是否符合预期
//   - 排查"为什么这个算子的分片和我设想的不一样"`,
  duration: 12000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:22px;justify-content:center;align-items:stretch">
        <div class="card" style="width:300px;border-color:rgba(148,163,184,.45)">
          <div class="card-t" style="color:var(--ink-dim)">默认</div>
          <div class="card-d">算子上<b>没有</b> <span class="mono">sharding_rule</span>。<br>
            IR 更干净，但你看不到传播依据。</div>
        </div>
        <div class="card" style="width:300px;border-color:rgba(94,234,212,.5)">
          <div class="card-t" style="color:var(--accent)">keep-sharding-rules=true</div>
          <div class="card-d">规则留在算子上。<br>
            <b>可追溯</b>，代价是 IR 更啰嗦。</div>
        </div>
      </div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:760px"></div>`;
    root.appendChild(wrap);
    const msg = wrap.querySelector('#msg');
    tl.at(900, () => { msg.innerHTML = '运维/调试时打开它，生产构建保持默认即可。'; });
    tl.at(3600, () => { msg.innerHTML = '这是个<b>观测性</b>选项：不改变传播结果，只改变"留下什么痕迹"。'; });
    tl.at(6400, () => { msg.innerHTML = '与 L2-12 的 <span class="mono">debug-sharding-origins</span> 配合，可以完整还原传播过程。'; });
    tl.at(9200, () => { msg.innerHTML = '测试文件只有 <b>1 个用例</b>、17 行 —— 因为选项的行为很单纯。'; });
  }
},

/* ---------------------------------------------------- 2 用例解读 */
{
  kicker: 'L2-03 · 保留分片规则',
  title: '用例解读：<span class="hl-a">两类规则</span>都留下',
  sub: '测试用的两个 <span class="mono">add</span>：一个原本没规则、一个原本有规则。加上选项后，两条规则都在。',
  caption: '注意第二个 <span class="mono">add</span> 的规则里有<b>复合因子</b> <span class="mono">[ij, k]</span> —— 这是用户自定义规则的典型形态。',
  code: `sdy.mesh @mesh = <["a"=2, "b"=2]>

func.func @existing_and_created_rules_remain(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // ① 原本【没有】规则 -> 传播为它推导出一条
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>

  // ② 原本【已有】用户自定义规则 -> 原样保留
  %1 = stablehlo.add %0, %0
       {sdy.sharding_rule = #sdy.op_sharding_rule<
          ([ij, k], [ij, k])->([ij, k]) {i=4, j=2, k=8}>}
       : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// 期望的两个算子都带 sharding + sharding_rule：
//   ① sdy.sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=8}>
//   ② sdy.sharding_rule<([ij, k], [ij, k])->([ij, k]) {i=4, j=2, k=8}>`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 推导出的规则', r: '([i, j], [i, j])->([i, j]) {i=8, j=8}', c: '#38bdf8',
        d: '逐元素算子的标准形态：<b>各张量维度一一对应</b>。<br>传播为没有规则的算子补出来的。' },
      { t: '② 已有的自定义规则', r: '([ij, k], [ij, k])->([ij, k]) {i=4, j=2, k=8}', c: '#c084fc',
        d: '用户写的，含<b>复合因子</b> <span class="mono">[ij]</span>。<br>传播不会覆盖或删除它。' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:360px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="mono" style="margin:6px 0;font-size:10.5px;color:#bdf7ec;overflow-wrap:anywhere">${U.esc(x.r)}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '第一个 <span class="mono">add</span> 原本干干净净 —— 规则是传播<b>推导</b>出来的。'; });
    tl.at(4200, () => { els[1].style.opacity = '1'; msg.innerHTML = '第二个 <span class="mono">add</span> 带着用户规则 —— 传播<b>沿用</b>它，没有另起炉灶。'; });
    tl.at(7800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '加上 <span class="mono">keep-sharding-rules=true</span> 后，<b>两类都留在 IR 里</b>。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>为什么这条信息有价值</b>：分片是规则的<b>推论</b>。看到分片不合预期时，要先确认规则对不对。';
    });
    tl.at(13400, () => {
      msg.innerHTML = '读的时候注意：第一个算子的分片是 <span class="mono">[{"a", ?}, {"b", ?}]</span> —— 两维都成了<b>开维</b>。';
    });
  }
},

/* ---------------------------------------------------- 3 什么时候用 */
{
  kicker: 'L2-03 · 保留分片规则',
  title: '什么时候该打开它',
  sub: '它不是给生产构建用的，而是<b>排查问题</b>时的第一手证据。',
  caption: '记住这条排查路径：<b>先看规则对不对 → 再看传播策略 → 最后才怀疑分片本身</b>。',
  code: `// 【场景 1】分片没传过去
//   先确认：这个算子有规则吗？规则长什么样？
//   如果规则本身就说"这两维无关" -> 不传播是正确行为

// 【场景 2】传成了意想不到的轴序
//   规则里的因子映射决定了轴的对应关系
//   打印出来一眼就能看出是规则的锅还是传播的锅

// 【场景 3】自定义算子（custom_call）的行为与预期不符
//   它带的是你自己的规则 -> 直接对照检查

// 配合使用的其它工具：
//   -sdy-populate-op-sharding-rules   把所有内置算子的规则打出来（L2-10）
//   -debug-sharding-origins           每个分片是从哪来的（L2-12）
//   -debug-propagation-edge-sharding  传播边的来源（L2-12）`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 分片没传过去', d: '先看规则是否存在、是否声明了维度对应关系。', c: '#38bdf8' },
      { t: '② 轴序不符预期', d: '因子映射直接决定轴序对应，打印出来即可对照。', c: '#c084fc' },
      { t: '③ 自定义算子异常', d: '它带的是你自己的规则，直接检查最快。', c: '#fbbf24' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:238px;opacity:.34;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:12px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.34');
      msg.innerHTML = [
        '最常见的情形：你以为该传，但规则说这两维本来就无关。',
        '轴序问题几乎都能在规则里找到答案 —— 复合因子的写法尤其关键。',
        '自定义规则里常有 <span class="mono">reduction</span> / <span class="mono">permutation</span> 标注，写错会直接改变传播行为。',
      ][i];
    }));
    tl.at(11200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>排查顺序</b>：规则 → 传播策略 → 分片本身。跳过第一步常常会白忙。';
    });
  }
},

/* ------------------------------------------------------------ 4 练习 */
{
  kicker: 'L2-03 · 练习',
  title: '练一练：<span class="hl-a">读懂留下的规则</span>',
  sub: '三道题分别考：选项作用、两类规则、调试顺序。',
  caption: '这一课很短，但它是后面 L2-10（规则注册表）与 L2-12（调试）的入口。',
  code: `// 题 1：keep-sharding-rules=true 会改变传播结果吗？

// 题 2：留下的规则有哪两类？

// 题 3：分片不合预期时，先查什么？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">keep-sharding-rules=true</span> 会改变传播的<b>结果</b>吗？',
        a: '<b class="badge ok">不会</b> 它是纯粹的<b>观测性</b>选项 —— 只决定"用过的规则要不要留在 IR 里"。' +
           '<br>传播算法本身走的分支、推出的分片<b>完全一致</b>。' +
           '<br><span class="dim">所以可以放心在生产构建上临时打开它来排查问题。</span>'
      },
      {
        q: '打开选项后，IR 里留下的规则有哪两类？',
        a: '<b>① 算子原本就有的</b>：如 <span class="mono">custom_call</span> 上用户写的规则（常含复合因子）。' +
           '<br><b>② 传播推导出来的</b>：为没有规则的算子补的，如逐元素算子的 <span class="mono">([i,j],[i,j])->([i,j])</span>。' +
           '<br><span class="dim">默认情况下两类用完都会被删掉，保持 IR 干净。</span>'
      },
      {
        q: '分片结果不符合预期时，排查的第一步是什么？',
        a: '先看<b>规则</b>对不对 —— 分片是规则的推论，规则错了后面全错。' +
           '<br><b>顺序</b>：① 该算子的规则 → ② 传播策略（basic / aggressive / 优先级）→ ③ 分片本身。' +
           '<br><span class="dim">跳过第一步常常会白忙一场。</span>'
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
