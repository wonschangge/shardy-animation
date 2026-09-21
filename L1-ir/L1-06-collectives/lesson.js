/* ==========================================================================
   L1-06 · collectives
   --------------------------------------------------------------------------
   覆盖：ir/test/collective_parse_print.mlir (508)
         ir/test/collective_verification.mlir (1160)
         ir/test/collective_canonicalization.mlir (399)
   目标：讲透 8 个集合通信算子 —— 语法、out_sharding 推导、子轴语义、校验、规范化。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 算子地图 */
{
  kicker: 'L1-06 · 集合通信',
  title: '八个集合通信算子：<span class="hl-a">一张地图</span>',
  sub: 'SDY 把所有跨设备数据移动收敛成 8 个算子。它们分成三组：<b>搬数据</b>、<b>做归约</b>、<b>管未归约状态</b>。',
  caption: '记住这张地图，L4-08「reshard → collective」与 L5-02「降级到 StableHLO」都会回指本课。',
  code: `// ① 搬数据（不改变数值，只改变分布在哪些设备上）
sdy.all_gather           // 分片 -> 复制
sdy.all_slice            // 复制 -> 分片
sdy.all_to_all           // 把某个维度上的分片搬到另一个维度
sdy.collective_permute   // 重排 / 替换轴（每维分片大小不变）

// ② 做归约（会改变数值：把部分和加起来）
sdy.all_reduce           // 部分和 -> 全和（结果是复制）
sdy.reduce_scatter       // 先归约再切片 = all_reduce + all_slice

// ③ 管理"未归约"状态（不通信，只改标记）
sdy.sharded_to_unreduced      // 已分片的轴 -> 未归约
sdy.replicated_to_unreduced   // 复制的轴   -> 未归约`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%;align-items:center' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="groups"></div>
      <div class="formula" id="msg" style="min-height:48px;display:flex;align-items:center;text-align:center;max-width:740px"></div>`;
    root.appendChild(wrap);
    const groups = [
      { t: '① 搬数据', c: '#38bdf8', ops: ['sdy.all_gather', 'sdy.all_slice', 'sdy.all_to_all', 'sdy.collective_permute'],
        d: '数值不变，只换设备上的分布。<br><span class="dim">all_gather / all_slice 互为逆操作</span>' },
      { t: '② 做归约', c: '#fbbf24', ops: ['sdy.all_reduce', 'sdy.reduce_scatter'],
        d: '把各设备的部分和合起来。<br><span class="dim">会改变数值语义</span>' },
      { t: '③ 管状态', c: '#4ade80', ops: ['sdy.sharded_to_unreduced', 'sdy.replicated_to_unreduced'],
        d: '<b>不通信</b>，只把轴标记成"未归约"。<br><span class="dim">为后续 all_reduce 做准备</span>' },
    ];
    const host = wrap.querySelector('#groups');
    const els = groups.map(g => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.35;transition:.3s;border-color:' + g.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${g.c}">${g.t}</div>
        <div class="mono" style="margin:6px 0;font-size:10.5px;line-height:1.7;color:#cfe0ff">${g.ops.join('<br>')}</div>
        <div class="card-d" style="font-size:11.5px">${g.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    groups.forEach((g, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.35');
      msg.innerHTML = [
        '这组的共同点：<b>数值完全不变</b>，变的只是"哪块数据在哪台设备上"。',
        '这组会<b>改变数值</b>：部分和只有加起来才是正确的完整结果。',
        '这组最容易误解：<span class="mono">sharded_to_unreduced</span> 本身<b>不发任何消息</b>，只是改标记。',
      ][i];
    }));
    tl.at(11000, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '一条主线：<b>分片 ↔ 复制 ↔ 未归约</b> 三种状态之间的转换，各自有对应算子。';
    });
  }
},

/* ------------------------------------------------ 2 gather / slice */
{
  kicker: 'L1-06 · 集合通信',
  title: '<span class="mono hl-a">all_gather</span> / <span class="mono hl-a">all_slice</span>：一对逆操作',
  sub: '参数是<b>逐维的轴列表</b>（外层项数 = 张量 rank）：第 i 项说明"第 i 维要沿哪些轴通信"。',
  caption: '关键性质：<b><span class="mono">out_sharding</span> 不是输入，是推导结果</b> —— 它由"操作数分片 + 通信轴"唯一确定，写错会被校验器拒绝。',
  code: `sdy.mesh @mesh1 = <["x"=2, "y"=2]>

// 输入分片 [{"y"}, {"x"}]，即 dim0 沿 y、dim1 沿 x
// gather dim1 上的 "x"：把 x 收回来 → dim1 变成复制
%0 = sdy.all_gather [{}, {"x"}] %arg0
     out_sharding=<@mesh1, [{"y"}, {}]> : tensor<16x8xf32>

// 逆操作：把 dim1 上的 "x" 切回去
%0 = sdy.all_slice [{}, {"x"}] %arg0
     out_sharding=<@mesh1, [{"y"}, {"x"}]> : tensor<16x8xf32>

// 可以一次 gather 多个维度
%0 = sdy.all_gather [{"x"}, {"z"}] %arg0 ...`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:20px;justify-content:center;align-items:center" id="row"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:740px"></div>`;
    root.appendChild(wrap);
    const row = wrap.querySelector('#row'), msg = wrap.querySelector('#msg');
    const devs = W.devRow(4, { size: 76, label: i => 'dev' + i });
    row.appendChild(devs.el);
    const A = [['A0'], ['A1'], ['A2'], ['A3']];
    const ALL = [['A0', 'A1', 'A2', 'A3'], ['A0', 'A1', 'A2', 'A3'], ['A0', 'A1', 'A2', 'A3'], ['A0', 'A1', 'A2', 'A3']];
    const paint = rows => rows.forEach((c, i) => W.fillBox(devs.boxes[i], c, i, { cw: 22, ch: 22 }));

    tl.at(700, () => { paint(A); msg.innerHTML = '初始：每个设备只有自己那一片（dim1 沿 "x" 切成 4 份）'; });
    tl.at(3400, () => {
      paint(ALL);
      msg.innerHTML = '<span class="mono">all_gather [{}, {"x"}]</span> → 每台设备都拿到<b>全部 4 片</b>，dim1 变成复制';
    });
    tl.at(7000, () => {
      paint(A);
      msg.innerHTML = '<span class="mono">all_slice [{}, {"x"}]</span> → 各取一片，回到分片状态';
    });
    tl.at(10200, () => {
      msg.innerHTML = '<b>out_sharding 是推导的</b>：gather 后 dim1 必然无轴（<span class="mono">{}</span>）；写错会报 <span class="mono">doesn\'t match expected sharding</span>';
    });
    tl.at(13000, () => { msg.innerHTML = '记忆：<b>gather 把轴"吃掉"变成复制；slice 把轴"切回来"</b>。'; });
  }
},

/* ------------------------------------------------------ 3 all_to_all */
{
  kicker: 'L1-06 · 集合通信',
  title: '<span class="mono hl-a">all_to_all</span>：把分片从一个维度搬到另一个维度',
  sub: '参数语法是 <span class="mono">{轴}: 源维 -&gt; 目标维</span>。一个参数 = 一次搬运；多个参数用逗号分隔，源维必须<b>升序</b>。',
  caption: '用途：某些算子要求"批维分片"，而数据现在是"特征维分片" —— 用 all_to_all 把分片搬过去，而不是先 gather 再 slice。',
  code: `// 单个参数：把 dim0 上的 "x" 搬到 dim1
%0 = sdy.all_to_all [{"x"}: 0->1] %arg0
     out_sharding=<@mesh1, [{}, {"x"}]> : tensor<16x8xf32>

// 一次搬多个轴
%0 = sdy.all_to_all [{"y", "x"}: 0->1] %arg0
     out_sharding=<@mesh2, [{"z"}, {"y", "x"}]> : tensor<16x8xf32>

// 子轴也能搬
%0 = sdy.all_to_all [{"x":(4)2, "z"}: 1->0] %arg0
     out_sharding=<@mesh4, [{"y", "x":(4)2, "z"}, {"x":(1)4}]>
     : tensor<16x8xf32>

// 约束：参数列表非空；源/目标维在范围内；
//       同一维不能既是源又是目标；源维升序`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:740px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const mk = (label, cells, cls) => {
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center' });
      c.innerHTML = `<div class="small mono faint">${label}</div><div class="row" style="gap:6px" data-d></div>`;
      const h = c.querySelector('[data-d]');
      cells.forEach(t => {
        const e = U.el('div', { class: 'chip ' + cls, style: 'padding:6px 12px;font-size:13px' });
        e.textContent = t; h.appendChild(e);
      });
      demo.appendChild(c); return c;
    };

    tl.at(700, () => {
      mk('dim0', ['x'], 'c0'); mk('dim1', ['—'], 'mut');
      msg.innerHTML = '初始：轴 <span class="mono">"x"</span> 切在 dim0 上，dim1 没有分片（<span class="mono">{}</span>）';
    });
    tl.at(3800, () => {
      demo.innerHTML = '';
      mk('dim0', ['—'], 'mut'); mk('dim1', ['x'], 'c0');
      msg.innerHTML = '<span class="mono">[{"x"}: 0-&gt;1]</span> → 轴 "x" 从 dim0 <b>搬到</b> dim1；分片总数不变，只是换了维度';
    });
    tl.at(7200, () => {
      demo.innerHTML = '';
      mk('dim0', ['y', 'x'], 'c2'); mk('dim1', ['—'], 'mut');
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹' }));
      mk('dim0', ['—'], 'mut'); mk('dim1', ['y', 'x'], 'c2');
      msg.innerHTML = '<span class="mono">[{"y", "x"}: 0-&gt;1]</span> → 一次搬两个轴，顺序保持';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>为什么需要它</b>：dims 之间的分片"搬家"，比 gather 再 slice 少一次通信量。';
    });
    tl.at(13200, () => {
      msg.innerHTML = '校验要点：源维升序（<span class="mono">2-&gt;1, 0-&gt;3</span> ✗）、源≠目标（<span class="mono">0-&gt;0</span> ✗）。';
    });
  }
},

/* ---------------------------------------------- 4 reduce / scatter */
{
  kicker: 'L1-06 · 集合通信',
  title: '归约：<span class="mono hl-a">all_reduce</span> 与 <span class="mono hl-a">reduce_scatter</span>',
  sub: '<span class="mono">all_reduce</span> 把部分和加起来并复制到所有设备；<span class="mono">reduce_scatter</span> 则是"加完只留自己那一片"。',
  caption: '归约算子可以指定 <span class="mono">sum</span>（默认）/ <span class="mono">max</span> / <span class="mono">min</span>。归约掉一个未归约轴后，该轴变成<b>显式复制</b>。',
  code: `// ① 基本归约（默认 sum）
%0 = sdy.all_reduce {"y"} %arg0
     out_sharding=<@mesh1, [{}, {"x"}]> : tensor<16x2xf32>

// ② 指定 max
%0 = sdy.all_reduce max {"y"} %arg0
     out_sharding=<@mesh1, [{}, {"x"}]> : tensor<16x2xf32>

// ③ 归约掉未归约轴 "y" -> 结果里 "y" 变成显式复制
%0 = sdy.all_reduce {"y"} %arg0
     out_sharding=<@mesh2, [{}, {"x"}], replicated={"y"}, unreduced={"z"}>

// ④ reduce_scatter = all_reduce + all_slice
%0 = sdy.reduce_scatter [{"y"}] %arg0
     out_sharding=<@mesh1, [{"x", "y"}]> : tensor<16xf32>

// ⑤ 轴列表全空 = 什么都不做（会被规范化消除）
%0 = sdy.reduce_scatter [{}, {}] %arg0
     out_sharding=<@mesh1, [{"x"}, {}]> : tensor<16x8xf32>`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:20px;justify-content:center;align-items:center" id="row"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:740px"></div>`;
    root.appendChild(wrap);
    const row = wrap.querySelector('#row'), msg = wrap.querySelector('#msg');
    const devs = W.devRow(4, { size: 76, label: i => 'dev' + i });
    row.appendChild(devs.el);
    const paint = (rows, boxes) => rows.forEach((c, i) => W.fillBox((boxes || devs.boxes)[i], c, i, { cw: 22, ch: 22 }));

    tl.at(700, () => { paint([['p0'], ['p1'], ['p2'], ['p3']]); msg.innerHTML = '初始：每台设备各有一个<b>部分和</b> p0..p3（沿 "y" 有未归约轴）'; });
    tl.at(3600, () => { paint([['Σ'], ['Σ'], ['Σ'], ['Σ']]); msg.innerHTML = '<span class="mono">all_reduce {"y"}</span> → 加起来，结果<b>复制</b>到所有设备'; });
    tl.at(7000, () => {
      paint([['Σ0'], ['Σ1'], ['Σ2'], ['Σ3']]);
      msg.innerHTML = '<span class="mono">reduce_scatter [{"y"}]</span> → 先加、再各留一片。等于 <span class="mono">all_reduce + all_slice</span>，但<b>只走一次通信</b>';
    });
    tl.at(10600, () => { msg.innerHTML = '归约语义可选 <span class="mono">sum</span> / <span class="mono">max</span> / <span class="mono">min</span>：<span class="mono">all_reduce max {"y"}</span>'; });
    tl.at(13200, () => { msg.innerHTML = '归约掉未归约轴后，该轴在结果里变为 <b>显式复制</b> —— 状态从"未归约"变成"完整"。'; });
  }
},

/* -------------------------------------------------- 5 permute */
{
  kicker: 'L1-06 · 集合通信',
  title: '<span class="mono hl-a">collective_permute</span>：重排与替换轴',
  sub: '约束很特别：<b>每一维的分片大小必须保持不变</b>，只是换了哪些轴、甚至换了哪个网格。',
  caption: '它是"最灵活但最贵"的通信：点对点任意重排。可以跨网格（设备顺序不同），也可以用复制轴换掉分片轴。',
  code: `// ① 在两个维度之间互换轴
//    输入 [{"x"}, {"y"}] -> 输出 [{"y"}, {"x"}]
%0 = sdy.collective_permute %arg0
     out_sharding=<@mesh2, [{"y"}, {"x"}]> : tensor<16x8xf32>

// ② 换到另一个网格（设备顺序不同）
%0 = sdy.collective_permute %arg0
     out_sharding=<@mesh1_non_iota, [{"x", "y"}, {}]>
     : tensor<16x8xf32>

// ③ 用复制轴换掉分片轴（每维大小仍然是 2）
//    [{"x", "y"}, {}] -> [{"z"}, {}]   （mesh5 的 z=4）

// 校验：每一维"分片轴大小之积"必须前后一致`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:740px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const mk = (cells) => {
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center' });
      c.innerHTML = `<div class="row" style="gap:6px" data-d></div>`;
      const h = c.querySelector('[data-d]');
      cells.forEach(t => {
        const e = U.el('div', { class: 'chip c0', style: 'padding:6px 12px;font-size:13px' });
        e.textContent = t; h.appendChild(e);
      });
      demo.appendChild(c); return c;
    };

    tl.at(700, () => { mk(['x']); mk(['y']); msg.innerHTML = 'dim0 沿 "x" 切，dim1 沿 "y" 切 —— 两维各有 2 份。'; });
    tl.at(3600, () => {
      demo.innerHTML = ''; mk(['y']); mk(['x']);
      msg.innerHTML = '<span class="mono">[{}, {}] → [{"y"}, {"x"}]</span>：两维的轴<b>互换</b>，每维仍是 2 份 ✓';
    });
    tl.at(7000, () => {
      msg.innerHTML = '还可以<b>跨网格</b>：从 <span class="mono">@mesh1</span> 换到设备顺序不同的 <span class="mono">@mesh1_non_iota</span>。';
    });
    tl.at(10000, () => {
      msg.innerHTML = '也可以<b>用复制轴换分片轴</b>：只要每维的"分片轴大小之积"不变，怎么换都行。';
    });
    tl.at(12800, () => {
      msg.innerHTML = '<b>校验核心</b>：逐维比较前后"分片轴大小之积"，不等就报 <span class="mono">sharded size of result doesn\'t match operand</span>。';
    });
  }
},

/* ------------------------------------------------ 6 未归约状态 */
{
  kicker: 'L1-06 · 集合通信',
  title: '<span class="mono hl-a">sharded</span> / <span class="mono hl-a">replicated</span> → unreduced',
  sub: '这两个算子<b>不发任何消息</b>，只把轴的状态改成"未归约"，为后续归约做准备。',
  caption: '数学关系：<span class="mono">all-gather(x, axes) = all-reduce(sharded-to-unreduced(x, axes), axes)</span>。',
  code: `// ① 已分片的轴 -> 未归约
//    输入 [{"y"}, {"x"}]
%0 = sdy.sharded_to_unreduced [{}, {"x"}] %arg0
     out_sharding=<@mesh1, [{"y"}, {}], unreduced={"x"}>
     : tensor<16x8xf32>

// ② 复制的轴 -> 未归约（可指定 max / min）
%0 = sdy.replicated_to_unreduced {"x", "y"} %arg0
     out_sharding=<@mesh1, [{}, {}], unreduced=min{"x", "y"}>
     : tensor<16x8xf32>

// ③ all_reduce 可以把它们"消费掉"
//    all_reduce(sharded_to_unreduced(x, a), a) == all_gather(x, a)
//    all_reduce(replicated_to_unreduced(x, a), a) == x`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:750px"></div>`;
    root.appendChild(wrap);
    const host = wrap.querySelector('#cards'), msg = wrap.querySelector('#msg');
    const defs = [
      { t: 'sharded → unreduced', from: '分片（每台一份数据）', to: '未归约（每台一份部分和）', c: '#fbbf24',
        d: '等价于"把别人的分片加起来"的<b>中间态</b>。<br>之后 all_reduce 就得到完整值。' },
      { t: 'replicated → unreduced', from: '复制（每台都有全量）', to: '未归约（每台都是一份）', c: '#4ade80',
        d: '把"完整值"重新标记为"部分和"。<br>之后 all_reduce 会把它<b>乘上设备数</b>。' },
    ];
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:330px;opacity:.35;transition:.3s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="small" style="margin:6px 0;line-height:1.7"><span class="faint">从</span> ${x.from}<br>
          <span class="faint">到</span> ${x.to}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    defs.forEach((_, i) => tl.at(700 + i * 4200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.35');
      msg.innerHTML = [
        '注意：这一步<b>不通信</b>。它只是把"这份数据是部分的"这个事实<b>记下来</b>。',
        '反向操作：把"确定完整"的值标记成"其实只是一份"，让后续归约去做正确的加法。',
      ][i];
    }));
    tl.at(9200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '恒等式：<span class="mono">all_reduce(sharded_to_unreduced(x, a), a) == all_gather(x, a)</span>';
    });
    tl.at(12200, () => { msg.innerHTML = '这两个算子是<b>规范化</b>的产物：融合与消除经常需要它们在中间"表示状态"。'; });
  }
},

/* ---------------------------------------------------- 7 子轴语义 */
{
  kicker: 'L1-06 · 集合通信',
  title: '子轴在通信里的<span class="hl-a">四种语义</span>',
  sub: '当通信涉及的轴是子轴时，行为取决于它与操作数分片里已有轴的关系。',
  caption: '这是本课最细的一组用例（parse_print 里每个算子都有对应的 4 个函数），也是最容易在实际调试中撞到的。',
  code: `// ① exact_match：通信轴与分片里完全一致
//    分片 dim1 = {"x":(1)2}，gather 的也是 {"x":(1)2}
%0 = sdy.all_gather [{}, {"x":(1)2}] %arg0
     out_sharding=<@mesh3, [{"y"}, {}]> : tensor<16x8xf32>

// ② ignored：通信轴与分片里的轴不相干，各自保留
//    分片 dim1 = {"x":(1)2}，但 gather 的是 dim0 的 {"y"}
%0 = sdy.all_gather [{"y"}, {}] %arg0
     out_sharding=<@mesh3, [{}, {"x":(1)2}]> : tensor<16x8xf32>

// ③ suffix_of_full：子轴是完整轴的后缀
//    分片 dim1 = {"x", "z"}，gather {"x":(4)2, "z"}（x 的后半段 + z）
%0 = sdy.all_gather [{}, {"x":(4)2, "z"}] %arg0
     out_sharding=<@mesh4, [{"y"}, {"x":(1)4}]> : tensor<16x8xf32>

// ④ suffix_of_subaxis：子轴是另一个子轴的后缀
//    分片 dim1 = {"z", "x":(1)4}，gather {"x":(2)2}（x 的后半段）`,
  duration: 18000,
  build(root, tl) {
    const wrap = W.errLayout(root, {
      irTitle: '算子写法（合法）', errTitle: '子轴语义', whyTitle: '解读',
      errColor: 'var(--accent)', errBorder: 'rgba(94,234,212,.45)', errFg: '#bdf7ec'
    });
    W.errScenes(wrap, tl, [
      {
        ir: '%0 = sdy.all_gather [{}, {"x":(1)2}] %arg0 out_sharding=<@mesh3, [{"y"}, {}]> : tensor<16x8xf32>',
        err: '① exact_match',
        why: '通信轴与分片里的轴<b>完全一致</b>：gather 后 dim1 变空。这是最直观的情形。'
      },
      {
        ir: '%0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh3, [{}, {"x":(1)2}]> : tensor<16x8xf32>',
        err: '② ignored',
        why: '通信动的是 dim0 的 <span class="mono">"y"</span>，与 dim1 上的 <span class="mono">"x":(1)2</span> <b>互不相干</b> —— 后者原样保留。'
      },
      {
        ir: '%0 = sdy.all_gather [{}, {"x":(4)2, "z"}] %arg0 out_sharding=<@mesh4, [{"y"}, {"x":(1)4}]> : tensor<16x8xf32',
        err: '③ suffix_of_full',
        why: '分片是完整轴 <span class="mono">"x"</span>，通信只动它的<b>后半段</b> <span class="mono">"x":(4)2</span>（连同 "z"）→ 结果剩 <span class="mono">"x":(1)4</span>。'
      },
      {
        ir: '%0 = sdy.all_gather [{}, {"x":(2)2}] %arg0 out_sharding=<@mesh4, [{"y"}, {"z", "x":(1)2}]> : tensor<16x8xf32>',
        err: '④ suffix_of_subaxis',
        why: '分片里已是子轴 <span class="mono">"x":(1)4</span>，通信动它的后半段 <span class="mono">"x":(2)2</span> → 结果子轴缩小。'
      },
    ], {
      stepMs: 3400,
      finalIr: '// 判断顺序：\n//   通信轴在分片里出现吗？\n//     是 -> 完全一致(exact) 还是只动后缀？\n//     否 -> ignored（互不相干）',
      finalErr: 'exact_match / ignored / suffix_of_full / suffix_of_subaxis',
      finalWhy: '每个集合通信算子都有这 4 个用例，说明这四种组合是<b>正交且完备</b>的。'
    });
  }
},

/* ---------------------------------------------------- 8 校验错误 */
{
  kicker: 'L1-06 · 校验',
  title: '校验错误：<span class="hl-a">out_sharding 必须与推导一致</span>',
  sub: '112 条校验错误里，最主要的一类就是"你写的 out_sharding 与算法推导出来的不符"。',
  caption: '其余错误都是老面孔：轴重复、子轴可合并、rank 不符、集合通信轴无法应用 —— 与 L1-02/L1-03 一脉相承。',
  code: `// ① 操作数没有分片
%0 = sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {}]>
//   collective on operand without sharding

// ② 结果网格与操作数不一致
%0 = sdy.all_gather [{}, {"b"}] %arg0 out_sharding=<@mesh1, [{"y"}, {"x"}]>
//   result mesh does not match operand mesh

// ③ 轴重复 / 子轴可合并（与 L1-03 同源）
%0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh, [{}, {"x", "x"}]>
%0 = sdy.all_gather [{}, {"x":(1)2, "x":(2)2}] %arg0 out_sharding=<@mesh, [{"y"}, {}]>

// ④ 通信轴列表的 rank 必须等于张量 rank
%0 = sdy.all_gather [{}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]>
//   result sharding has rank 2 but collective axes has rank 1

// ⑤ 轴必须能"应用"到操作数分片上
%0 = sdy.all_gather [{}, {"x", "y"}] %arg0 out_sharding=<@mesh, [{"y"}, {}]>
//   can't apply gathering axis "y" to operand sharding on dimension 1

// ⑥ all_to_all 的参数约束
%0 = sdy.all_to_all [] %arg0 ...                            // parameter list is empty
%0 = sdy.all_to_all [{"y"}: 2->1] %arg0 ...                 // source dimension 2 is out of range
%0 = sdy.all_to_all [{"y"}: 0->0] %arg0 ...                 // overlapping source/target dimensions
%0 = sdy.all_to_all [{"y"}: 2->1, {"x"}: 0->3] %arg0 ...    // source dimensions are not sorted`,
  duration: 20000,
  build(root, tl) {
    const wrap = W.errLayout(root);
    W.errScenes(wrap, tl, [
      {
        ir: '%0 = sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x8xf32>',
        err: 'collective on operand without sharding',
        why: '集合通信必须知道操作数<b>当前怎么分片</b>，否则无法推导结果。'
      },
      {
        ir: '%0 = sdy.all_gather [{}, {"b"}] %arg0 out_sharding=<@mesh1, [{"y"}, {"x"}]> :  tensor<16x8xf32>',
        err: 'result mesh does not match operand mesh',
        why: '通信前后必须用同一个网格（换网格要用 <span class="mono">collective_permute</span>）。'
      },
      {
        ir: '%0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh, [{}, {"x", "x"}]> :  tensor<16x8xf32>',
        err: 'duplicate axis ref: "x"',
        why: '与 L1-03 同源：同一个轴在一条分片属性里只能出现一次。'
      },
      {
        ir: '%0 = sdy.all_gather [{}, {"x":(1)2, "x":(2)2}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x8xf32>',
        err: 'two consecutive sub-axes can be merged: "x":(1)2, "x":(2)2',
        why: '相邻子轴必须合并 —— 集合通信的轴列表同样受这条约束。'
      },
      {
        ir: '%0 = sdy.all_gather [{}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x8xf32>',
        err: 'result sharding has rank 2 but collective axes has rank 1',
        why: '张量是 rank 2，通信轴列表也只有 <b>2 项</b>（这里只写了 1 项）。'
      },
      {
        ir: '%0 = sdy.all_gather [{}, {"x", "y"}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x8xf32>',
        err: 'can\'t apply gathering axis "y" to operand sharding on dimension 1',
        why: 'dim1 的分片里没有 <span class="mono">"y"</span>（它切在 dim0 上），无法"收集"一个不在那里的轴。'
      },
      {
        ir: '%0 = sdy.all_gather [{"x", "y"}, {}] %arg0 out_sharding=<@mesh, [{}, {}]> :  tensor<16x8xf32>',
        err: 'can\'t apply gathering axis "x" to operand sharding on dimension 0',
        why: 'dim0 上并没有 <span class="mono">"x"</span> —— 轴的"位置"必须与分片一致。'
      },
      {
        ir: '%0 = sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x8xf32>',
        err: 'result sharding doesn\'t match expected sharding [] on dimension 1',
        why: '<b>最核心的一类</b>：gather 了 dim1 的 <span class="mono">"x"</span>，结果 dim1 就<b>必须</b>是 <span class="mono">{}</span>，你写了 <span class="mono">{"x"}</span> 就矛盾。'
      },
      {
        ir: '%0 = sdy.all_to_all [] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x8xf32>',
        err: 'parameter list is empty',
        why: 'all_to_all 至少要有一个搬运参数，空列表没有意义。'
      },
      {
        ir: '%0 = sdy.all_to_all [{"y"}: 2->1] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x8xf32>',
        err: 'source dimension 2 is out of range [0, 2)',
        why: '张量只有 dim0、dim1，源维 2 不存在。'
      },
      {
        ir: '%0 = sdy.all_to_all [{"y"}: 0->0] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x8xf32>',
        err: 'overlapping source/target dimensions in all-to-all params: 0',
        why: '源维与目标维相同 = 自己搬给自己，不产生任何效果。'
      },
      {
        ir: '%0 = sdy.all_to_all [{"y"}: 2->1, {"x"}: 0->3] %arg0 out_sharding=<@mesh, [{}, {"y"}, {}, {"x"}]> :  tensor<16x8x8x8xf32>',
        err: 'source dimensions are not sorted in ascending order: 0 appears after 2',
        why: '多个参数时源维必须<b>升序</b>排列，保证参数的规范形式唯一。'
      },
    ], {
      stepMs: 2000,
      finalIr: '// 三类根因：\n//   1. 操作数/结果本身不合法（无分片、网格不符、轴重复）\n//   2. 通信轴列表不合法（rank 不符、轴无法应用）\n//   3. out_sharding 与推导结果不符  <- 本课特有',
      finalErr: 'without sharding / mesh does not match / rank / doesn\'t match expected sharding',
      finalWhy: '第 3 类是本课独有的：<b>out_sharding 是输出，不是输入</b>，写错就是自相矛盾。'
    });
  }
},

/* ---------------------------------------------------- 9 规范化 */
{
  kicker: 'L1-06 · 规范化',
  title: '规范化：<span class="hl-a">消除空通信</span>与<span class="hl-a">算子融合</span>',
  sub: '测试文件里 42 个规范化用例，主要做两件事：把"什么都没做"的通信删掉，把能合并的通信合并。',
  caption: '这些优化在导入阶段就会跑，所以你在实际 IR 里很少看到"全零轴列表"的集合通信。',
  code: `// ① 消除：全零的 collective_permute 若无使用者 -> 直接删除
//    （null_collective_permute / null_all_gather / null_all_slice
//      / null_all_reduce / null_reduce_scatter 等用例）

// ② 融合：all_reduce + all_slice -> reduce_scatter
//    输入：
%0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{"y"}, {}]> : tensor<16x2xf32>
%1 = sdy.all_slice [{}, {"x"}] %0 out_sharding=<@mesh, [{"y"}, {"x"}]> : tensor<16x2xf32>

//    输出（一次通信搞定）：
%0 = sdy.reduce_scatter [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> : tensor<16x2xf32>

// ③ 抵消：all_gather 之后紧跟 all_slice 同一批轴 -> 互相抵消
//    （all_slice_of_all_gather 系列用例）

// ④ 链式 all_to_all 合并成一个
//    （all_to_all_fusion_chained_ops 等用例）`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:12px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:48px;display:flex;align-items:center;text-align:center;max-width:750px"></div>`;
    root.appendChild(wrap);
    const host = wrap.querySelector('#cards'), msg = wrap.querySelector('#msg');
    const defs = [
      { t: '① 消除 null', d: '轴列表全空 = 什么都没做 → 删掉。<br><span class="dim">例子：<span class="mono">null_collective_permute</span></span>', c: '#4ade80' },
      { t: '② 融合 all_reduce + all_slice', d: '合并成 <span class="mono">reduce_scatter</span>。<br><b>省一次通信</b>。', c: '#fbbf24' },
      { t: '③ 抵消 gather + slice', d: '同一批轴的 gather 后接 slice<br>→ 互相抵消，两个都删。', c: '#38bdf8' },
      { t: '④ 合并链式 all_to_all', d: '多个连续的 all_to_all<br>→ 合成一个（含排序规范化）。', c: '#c084fc' },
    ];
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:178px;opacity:.32;transition:.3s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.32');
      msg.innerHTML = [
        '<b>零成本优化</b>：删掉一个"什么都没做"的通信，不改变语义。',
        '<span class="mono">all_reduce</span> 把结果复制到所有设备，紧接着 <span class="mono">all_slice</span> 又各取一片 —— 中间的"复制"是白做的。',
        'gather 把轴收回来，紧接着又 slice 出去 —— 数据绕了一圈回到原处，直接删掉两个算子。',
        'all_to_all 是"把一个维度的分片搬到另一个维度"，连续两次搬运可以合成一次。',
      ][i];
    }));
    tl.at(700 + defs.length * 3400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '共同前提：<b>不改变语义</b>。规范化只做"数学上等价且更省"的改写。';
    });
  }
},

/* ------------------------------------------------------------ 10 练习 */
{
  kicker: 'L1-06 · 练习',
  title: '练一练：<span class="hl-a">选算子、判合法性</span>',
  sub: '三道题分别考：选对算子、推导 out_sharding、判断错误类型。',
  caption: '能答对这三题，L4-08「reshard → collective」就有了基础。',
  code: `// 题 1：dim1 从分片变成复制，用哪个算子？
//   输入 [{"y"}, {"x"}] -> 输出 [{"y"}, {}]

// 题 2：out_sharding 应该是什么？
%0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh, [?, ?]>
//   输入分片 [{"x"}, {"y"}] : tensor<16x8xf32>

// 题 3：错在哪？
%0 = sdy.all_gather [{}, {"x", "y"}] %arg0
     out_sharding=<@mesh, [{"y"}, {}]> : tensor<16x8xf32>
//   输入分片 [{"y"}, {}]`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '把 dim1 从分片变成复制，用哪个算子？',
        a: '<span class="mono">sdy.all_gather [{}, {"x"}] %arg0 out_sharding=&lt;@mesh, [{"y"}, {}]&gt;</span><br>' +
           '<b>理由</b>：gather 是"分片 → 复制"；参数第 1 项 <span class="mono">{"x"}</span> 指定收集 dim1 上的 "x"。'
      },
      {
        q: '输入 <span class="mono">[{"x"}, {"y"}]</span>，执行 <span class="mono">all_gather [{"y"}, {}]</span> 后 <span class="mono">out_sharding</span> 是什么？',
        a: '<span class="mono">[{}, {"y"}]</span><br>' +
           '<b>理由</b>：gather 动的是 dim0 上的 <span class="mono">"y"</span> → dim0 变空；dim1 的 <span class="mono">"y"</span> 与本次通信<b>无关</b>（ignored 语义），原样保留。'
      },
      {
        q: '输入分片 <span class="mono">[{"y"}, {}]</span>，却写 <span class="mono">all_gather [{}, {"x", "y"}]</span>，错在哪？',
        a: '<b class="badge bad">can\'t apply gathering axis "y" to operand sharding on dimension 1</b><br>' +
           'dim1 上<b>没有</b> <span class="mono">"y"</span>（它在 dim0 上）；而且 dim1 本身是空的，没有轴可收集。'
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
