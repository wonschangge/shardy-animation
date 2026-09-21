/* ==========================================================================
   L1-02 · tensor-sharding-syntax
   --------------------------------------------------------------------------
   覆盖：ir/test/tensor_sharding_parse_print.mlir (235)
         ir/test/tensor_sharding_parsing_failure.mlir (177)
   目标：讲透 #sdy.sharding / sharding_per_value 的完整语法与 15 类解析错误。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------- 1 两种属性形式 */
{
  kicker: 'L1-02 · 分片语法',
  title: '两种写法：<span class="mono hl-a">sharding</span> 与 <span class="mono hl-a">sharding_per_value</span>',
  sub: '同一个分片属性，挂在<b>函数参数/结果</b>上时叫 <span class="mono">sdy.sharding</span>，挂在<b>算子</b>上时叫 <span class="mono">sdy.sharding_per_value</span>（因为一个算子可能有多个操作数/结果）。',
  caption: '记住这条区分，你在读任何 SDY IR 时就不会困惑"为什么这里多一层方括号"。',
  code: `// ① 函数参数 / 结果：单个 sharding
func.func @f(%arg0: tensor<8x8xf32>
    {sdy.sharding = #sdy.sharding<@foo, [{"a"}p0, {"b"}]>})

// ② 算子：per_value 是一个列表，每个元素对应一个 操作数/结果
%0 = stablehlo.add %arg0, %arg1
     {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {}]>]>}

// ③ 多结果算子：列表里就有多项
%0:2 = stablehlo.optimization_barrier
     {sdy.sharding = #sdy.sharding_per_value<
        [<mesh<["a"=2, "b"=4]>, [{"a"}, {}]>, <@foo, [{"a"}, {}]>]>}

// ④ 空列表 = 没有分片信息
return {sdy.sharding = #sdy.sharding_per_value<[]>} %arg0`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `<div class="row" style="gap:16px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:46px;display:flex;align-items:center;text-align:center"></div>`;
    root.appendChild(wrap);

    const defs = [
      { t: '挂在函数上', mono: '#sdy.sharding<@foo, …>', d: '直接就是<b>一个</b> TensorShardingAttr。<br>用于函数参数与返回值。', c: '#38bdf8' },
      { t: '挂在算子上', mono: '#sdy.sharding_per_value<[ … ]>', d: '<b>列表</b>，每项对应一个 operand/result。<br>位置 i 的 sharding 属于第 i 个值。', c: '#c084fc' },
      { t: '空列表', mono: '#sdy.sharding_per_value<[]>', d: '一个值都没有分片信息 —— 常见于 <span class="mono">return</span>。', c: '#4ade80' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => {
      const e = U.el('div', { class: 'card', style: 'width:236px;opacity:.35;transition:.35s' });
      e.innerHTML = `<div class="card-t" style="color:${d.c};font-size:13px">${d.t}</div>
        <div class="mono small" style="margin:6px 0;color:#cfe0ff;font-size:11.5px">${U.esc(d.mono)}</div>
        <div class="card-d">${d.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    const tips = [
      '函数上的写法：<span class="mono">%arg0 ... {sdy.sharding = #sdy.sharding&lt;…&gt;}</span>',
      '算子上的写法：注意<b>多了一层 <span class="mono">[ ]</span></b> —— 那是"每个值一项"的列表。',
      '<span class="mono">optimization_barrier</span> 有两个结果 → 列表里就有两项，第二项用内联网格。',
    ];
    defs.forEach((d, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.35'; e.style.borderColor = k === i ? d.c + '88' : ''; });
      msg.innerHTML = tips[i];
    }));
    tl.at(11200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '一句话：<b>函数用 <span class="mono">sharding</span>，算子用 <span class="mono">sharding_per_value</span></b> —— 后者是前者组成的列表。';
    });
  }
},

/* --------------------------------------------------------- 2 语法全貌 */
{
  kicker: 'L1-02 · 分片语法',
  title: '一个分片属性里有什么',
  sub: '去掉简写后，一条 <span class="mono">#sdy.sharding</span> 由四部分组成：网格引用、每维分片、复制轴、未归约轴。',
  caption: '网格有两种写法：<span class="mono">@符号名</span>（引用 sdy.mesh）或 <span class="mono">mesh&lt;…&gt;</span>（内联）。',
  code: `#sdy.sharding<
    @foo,                      // ① 网格：符号引用
    [{"a"}p0, {"b"}]           // ② 每维分片（rank 个）：轴 + 开闭 + 优先级
  , replicated={"c"}           // ③ 显式复制轴
  , unreduced={"d"}            // ④ 未归约轴
>

// 网格也可以内联：
#sdy.sharding<mesh<["x"=2, "y"=2]>, [{"x"}, {}]>

// 简写：只写前两部分
#sdy.sharding<@foo, [{"a"}, {}]>`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="mono" style="font-size:15px;line-height:2.1;padding:12px 18px;border-radius:10px;
           background:rgba(0,0,0,.3);border:1px solid var(--panel-brd)">
        <div class="row" style="gap:8px"><span style="color:#c084fc">#sdy.sharding&lt;</span></div>
        <div class="row" style="gap:8px">&nbsp;&nbsp;<span id="p1">@foo</span><span class="faint">,</span></div>
        <div class="row" style="gap:8px">&nbsp;&nbsp;<span id="p2">[{"a"}p0, {"b"}]</span><span class="faint">,</span></div>
        <div class="row" style="gap:8px">&nbsp;&nbsp;<span id="p3">replicated={"c"}</span><span class="faint">,</span></div>
        <div class="row" style="gap:8px">&nbsp;&nbsp;<span id="p4">unreduced={"d"}</span></div>
        <div class="row" style="gap:8px"><span style="color:#c084fc">&gt;</span></div>
      </div>
      <div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:42px;display:flex;align-items:center;text-align:center"></div>`;
    root.appendChild(wrap);

    const ids = ['p1', 'p2', 'p3', 'p4'].map(i => wrap.querySelector('#' + i));
    const host = wrap.querySelector('#cards');
    const infos = [
      { t: '① 网格', d: '符号名 <span class="mono">@foo</span> 或内联 <span class="mono">mesh&lt;…&gt;</span>。<br>所有轴名必须来自这个网格。' },
      { t: '② 每维分片', d: 'rank 个 <span class="mono">{轴…}</span>。轴序 major→minor；<br><span class="mono">?</span> 表示开维，<span class="mono">pN</span> 表示优先级。' },
      { t: '③ 复制轴', d: '可选。<b>显式</b>声明这些轴保持复制，<br>禁止传播拿它们去切维度。' },
      { t: '④ 未归约轴', d: '可选。该轴上只有部分和。<br>默认 <span class="mono">sum</span>，可写 <span class="mono">max{"a"}</span>。' },
    ];
    const cards = infos.map(i => {
      const e = U.el('div', { class: 'card', style: 'width:178px;opacity:.3;transition:.3s' });
      e.innerHTML = `<div class="card-t" style="font-size:12px">${i.t}</div><div class="card-d" style="font-size:11.5px">${i.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    infos.forEach((_, i) => tl.at(700 + i * 2800, () => {
      cards.forEach((c, k) => c.style.opacity = k === i ? '1' : '.3');
      ids.forEach((e, k) => { e.style.color = k === i ? 'var(--accent)' : ''; e.style.fontWeight = k === i ? '700' : ''; });
      msg.innerHTML = [
        '轴名必须在网格里存在，否则报 <span class="mono">axis not found</span>。',
        '维数必须等于张量 rank —— 这是最常见的报错来源。',
        '复制轴的顺序会按网格声明顺序规范化。',
        '未归约轴与复制轴<b>不能重叠</b>，且排序也按网格规范化。',
      ][i];
    }));
    tl.at(12200, () => {
      cards.forEach(c => c.style.opacity = '1');
      ids.forEach(e => { e.style.color = ''; e.style.fontWeight = ''; });
      msg.innerHTML = '后三部分都可省略。省略 ≠ 不存在：<b>没被用到的轴 = 隐式复制</b>。';
    });
  }
},

/* ---------------------------------------------------- 3 开维与闭维 */
{
  kicker: 'L1-02 · 分片语法',
  title: '开维 <span class="mono hl-a">?</span> 与闭维：一行代码的差别',
  sub: '维度末尾加 <span class="mono">?</span> = 开维（传播可以继续加轴）；不加 = 闭维（锁死）。',
  caption: '这是 SDY 相对 GSPMD <span class="mono">unspecified_dims</span> 更精确的地方：开/闭是<b>每个维度</b>独立控制的。',
  code: `// 两维都开
#sdy.sharding_per_value<[<@foo, [{"a", ?}, {?}]>]>

// 两维都闭
#sdy.sharding_per_value<[<@foo, [{"a"}, {}]>]>

// 一开一闭：第 0 维锁在 "a" 上，第 1 维还能继续切
#sdy.sharding_per_value<[<@foo, [{"a", ?}, {}]>]>

// 开维上可以有多个轴：已按 a、b 切，还能再加
#sdy.sharding_per_value<[<@foo, [{"a", "b", ?}, {}]>]>`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:34px;justify-content:center">
        <div class="col" style="gap:10px;align-items:center">
          <div class="mid" style="font-weight:700">闭维 <span class="mono">{"a"}</span></div>
          <div class="row" style="gap:8px" id="closed"></div>
          <div class="small dim" id="cm" style="height:20px"></div>
        </div>
        <div class="col" style="gap:10px;align-items:center">
          <div class="mid" style="font-weight:700">开维 <span class="mono">{"a", ?}</span></div>
          <div class="row" style="gap:8px" id="open"></div>
          <div class="small dim" id="om" style="height:20px"></div>
        </div>
      </div>
      <div class="formula" id="msg" style="min-height:60px;display:flex;align-items:center;text-align:center;max-width:700px"></div>`;
    root.appendChild(wrap);

    wrap.querySelector('#closed').innerHTML = '<span class="chip c0">"a"</span><span class="chip mut">🔒</span>';
    wrap.querySelector('#open').innerHTML = '<span class="chip c0">"a"</span><span class="chip mut">?</span>';
    const cm = wrap.querySelector('#cm'), om = wrap.querySelector('#om'), msg = wrap.querySelector('#msg');

    tl.at(700, () => { msg.innerHTML = '假设传播想把轴 <span class="mono">"b"</span> 也加到第 0 维上…'; });
    tl.at(2600, () => {
      cm.innerHTML = '<span class="badge bad">✕ 拒绝</span>';
      om.innerHTML = '<span class="badge ok">✓ 接受</span>';
      wrap.querySelector('#open').innerHTML = '<span class="chip c0">"a"</span><span class="chip c1">"b"</span><span class="chip mut">?</span>';
    });
    tl.at(4600, () => {
      msg.innerHTML = '闭维的结果：<span class="mono">[{"a"}, …]</span> —— 第 0 维的分片被锁死，传播<b>不能改</b>';
    });
    tl.at(7000, () => {
      msg.innerHTML = '开维的结果：<span class="mono">[{"a", "b", ?}, …]</span> —— 第 0 维现在沿 a 再沿 b 切成 4 份，且仍可继续加';
    });
    tl.at(9800, () => {
      msg.innerHTML = '常见用法：<b>JAX 里用户指定的 <span class="mono">in_shardings</span> 是闭维</b>（不能变），其余维度留开。';
    });
    tl.at(11600, () => { msg.innerHTML = '不写 sharding 属性 &nbsp;≡&nbsp; 全开（编译器完全自由）。'; });
  }
},

/* ------------------------------------------- 4 replicated / unreduced */
{
  kicker: 'L1-02 · 分片语法',
  title: '<span class="hl-a">replicated</span> 与 <span class="hl-a">unreduced</span>',
  sub: '前者锁死"这些轴保持复制"；后者表示"这些轴上只有部分和"。两者都能写成子轴，也都能与维度分片共存。',
  caption: '注意第 4 组：输入把 <span class="mono">unreduced</span> 写在前面，打印时被规范化到 <span class="mono">replicated</span> 之后 —— <b>顺序由打印器决定，不由输入决定</b>。',
  code: `// ① 无复制轴（默认空）
#sdy.sharding_per_value<[<@foo, [{"a"}, {}]>]>

// ② 显式复制两个轴
#sdy.sharding_per_value<[<@foo, [{}, {}], replicated={"a", "b"}>]>

// ③ 未归约轴（默认 sum）
#sdy.sharding_per_value<[<@foo, [{}, {}], unreduced={"a", "b"}>]>

// ④ 未归约轴 + max 归约
#sdy.sharding_per_value<[<@foo, [{}, {}], unreduced=max{"a"}>]>

// ⑤ 两者共存（打印顺序固定为 replicated 在前）
#sdy.sharding_per_value<[<@foo, [{}, {}], replicated={"a"}, unreduced={"b"}>]>
// 即使输入写成 unreduced={"b"}, replicated={"a"}，输出也是上面这个顺序`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:10px;justify-content:center;align-items:center">
        <span class="chip c0">"a"=2</span><span class="chip c1">"b"=4</span>
        <span class="faint small">网格 @foo</span>
      </div>
      <div class="row" style="gap:12px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:44px;display:flex;align-items:center;text-align:center;max-width:720px"></div>`;
    root.appendChild(wrap);

    const infos = [
      { t: '① 无复制轴', m: '[{"a"}, {}]', d: '后两项都省略。<br>"b"、"c"、"d" 是<b>隐式复制</b>。' },
      { t: '② 显式复制', m: 'replicated={"a", "b"}', d: '锁死 a、b 保持复制，<br>传播不能拿它们切维度。' },
      { t: '③ 未归约 (sum)', m: 'unreduced={"a", "b"}', d: '这两个轴上只有部分和，<br>需要 all-reduce 才完整。' },
      { t: '④ 未归约 (max)', m: 'unreduced=max{"a"}', d: '归约语义是 <b>max</b> 而非 sum。<br>min 同理。' },
    ];
    const host = wrap.querySelector('#cards');
    const els = infos.map(i => {
      const e = U.el('div', { class: 'card', style: 'width:182px;opacity:.35;transition:.3s' });
      e.innerHTML = `<div class="card-t" style="font-size:12px">${i.t}</div>
        <div class="mono" style="margin:5px 0;color:#bdf7ec;font-size:10.5px;overflow-wrap:anywhere">${U.esc(i.m)}</div>
        <div class="card-d" style="font-size:11.5px">${i.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    infos.forEach((_, i) => tl.at(700 + i * 3000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.35');
      msg.innerHTML = [
        '不写 <span class="mono">replicated=</span> 就是"暂时没用这些轴"，传播<b>仍可</b>拿去切维度。',
        '显式复制 = 主动放弃这些轴的分片能力，用于表达"这里必须复制"。',
        '未归约轴的典型来源：matmul 切了收缩维，结果就是未归约的。',
        '归约语义会影响导出时生成哪种 all-reduce。',
      ][i];
    }));
    tl.at(13000, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '两者共存时，打印顺序恒为 <span class="mono hl-a">replicated</span> 在前、<span class="mono hl-a">unreduced</span> 在后。';
    });
    tl.at(14600, () => {
      msg.innerHTML = '<b>输入顺序不影响语义</b>，但会影响你 diff 时的判断 —— 以打印结果为准。';
    });
  }
},

/* ------------------------------------------------ 5 内联网格 vs 符号 */
{
  kicker: 'L1-02 · 分片语法',
  title: '网格的两种写法：<span class="mono hl-a">符号引用</span> vs <span class="mono hl-a">内联</span>',
  sub: '分片属性里的网格既可以引用 <span class="mono">sdy.mesh</span> 符号（<span class="mono">@foo</span>），也可以直接内联（<span class="mono">mesh&lt;…&gt;</span>）。',
  caption: '内联网格同样遵守 L1-01 的 iota 规则：设备顺序恰好等于默认时会被省略。',
  code: `// ① 符号引用（最常见）
#sdy.sharding_per_value<[<@foo, [{"a"}, {}]>]>

// ② 内联网格
#sdy.sharding_per_value<[<mesh<["x"=2, "y"=2]>, [{"x"}, {}]>]>

// ③ 内联网格 + 非 iota 设备顺序（保留）
#sdy.sharding_per_value<[<mesh<["x"=2], device_ids=[1, 0]>, [{"x"}, {}]>]>

// ④ 内联网格 + iota 设备顺序（打印时省略 device_ids）
//   输入 <mesh<["x"=2], device_ids=[0, 1]>>
//   输出 <mesh<["x"=2]>>

// ⑤ 同一个算子的两个结果可以用不同网格
#sdy.sharding_per_value<
  [<mesh<["a"=2, "b"=4]>, [{"a"}, {}]>, <@foo, [{"a"}, {}]>]>`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div id="demo" style="width:100%"></div>
      <div class="formula" id="msg" style="min-height:44px;display:flex;align-items:center;text-align:center;max-width:700px"></div>`;
    root.appendChild(wrap);

    const host = wrap.querySelector('#cards');
    const defs = [
      { t: '符号引用', m: '@foo', d: '引用 module 里 <span class="mono">sdy.mesh @foo</span> 声明的网格。', c: '#38bdf8' },
      { t: '内联网格', m: 'mesh<["x"=2, "y"=2]>', d: '直接把网格写进属性。<br>导入阶段会<b>提升</b>为符号（见 L3-07）。', c: '#c084fc' },
    ];
    const els = defs.map(d => {
      const e = U.el('div', { class: 'card', style: 'width:320px;opacity:.4;transition:.3s' });
      e.innerHTML = `<div class="card-t" style="color:${d.c};font-size:12.5px">${d.t}</div>
        <div class="mono small" style="margin:5px 0;color:#cfe0ff;overflow-wrap:anywhere">${U.esc(d.m)}</div>
        <div class="card-d">${d.d}</div>`;
      host.appendChild(e); return e;
    });
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '符号引用：网格在别处声明，属性里只写名字。'; });
    tl.at(3400, () => {
      els[1].style.opacity = '1'; els[0].style.opacity = '.4';
      msg.innerHTML = '内联网格：属性自包含，不依赖外部符号。';
    });
    tl.at(6000, () => {
      demo.innerHTML = W.irPair(
        `#sdy.sharding_per_value<[<mesh<["x"=2], device_ids=[0, 1]>, [{"x"}, {}]>]>`,
        `#sdy.sharding_per_value<[<mesh<["x"=2]>, [{"x"}, {}]>]>`,
        { inTitle: '输入', outTitle: '打印结果' });
      msg.innerHTML = '内联网格也遵守 iota 省略规则：<span class="mono">device_ids=[0, 1]</span> 等于默认 → 被省略';
    });
    tl.at(10000, () => {
      demo.innerHTML = W.irPair(
        `#sdy.sharding_per_value<[<mesh<["x"=2], device_ids=[1, 0]>, [{"x"}, {}]>]>`,
        null,
        { inTitle: '非 iota：原样保留' });
      msg.innerHTML = '设备顺序被真正改过（<span class="mono">[1, 0]</span>）时，<span class="mono">device_ids</span> 会保留。';
    });
    tl.at(13200, () => {
      msg.innerHTML = '同一个算子的不同结果<b>可以用不同网格</b> —— 跨网格使用时需要重新分片。';
    });
  }
},

/* ------------------------------------------------------------ 6 子轴 */
{
  kicker: 'L1-02 · 分片语法',
  title: '子轴：<span class="mono hl-a">"b":(2)2</span> 怎么读',
  sub: '记号是 <span class="mono">"轴名":(pre-size)size</span>。把一个大小为 n 的轴按 <span class="mono">[m, k, n/(m·k)]</span> 重排后，取中间那一段。',
  caption: '<b>pre-size</b> 是在它左边（更 major）的所有子轴大小之积；<b>size</b> 是它自身的大小。子轴可以出现在维度分片、replicated、unreduced 三个位置。',
  code: `sdy.mesh @bar = <["a"=4, "b"=2]>

// ① 维度分片里的子轴
#sdy.sharding_per_value<[<@foo, [{"a"}, {"b":(2)2}]>]>

// ② 显式复制子轴
#sdy.sharding_per_value<[<@foo, [{"a"}, {}], replicated={"b":(2)2}>]>

// ③ 未归约子轴
#sdy.sharding_per_value<[<@foo, [{"a"}, {}], unreduced={"b":(2)2}>]>

// ④ 完整轴夹在两个子轴中间（合法：它们不连续）
#sdy.sharding_per_value<[<@bar, [{"a":(1)2, "b", "a":(2)2}, {}]>]>

// ⑤ 逆序的连续子轴：解析合法，但校验会报
//    two consecutive sub-axes can be merged
#sdy.sharding_per_value<[<@bar, [{"a":(2)2, "a":(1)2}, {}]>]>`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:center">
        <div class="col" style="gap:6px;align-items:center">
          <div class="small faint">轴 "a"=4，拆成 4 个子轴</div>
          <div class="row" style="gap:5px" id="axis"></div>
        </div>
      </div>
      <div class="formula" id="decomp" style="min-height:44px;display:flex;align-items:center;text-align:center"></div>
      <div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:44px;display:flex;align-items:center;text-align:center;max-width:720px"></div>`;
    root.appendChild(wrap);

    const ax = wrap.querySelector('#axis');
    const labels = ['a:(1)1', 'a:(1)2', 'a:(2)1', 'a:(2)2'];
    labels.forEach((t, i) => {
      const e = U.el('div', { class: 'dev', style: 'width:62px;height:44px' });
      e.style.setProperty('--c', `var(--ax${i % 2 ? 0 : 1})`);
      e.innerHTML = `<span class="small mono" style="font-size:10px">${t}</span>`;
      ax.appendChild(e);
    });
    const decomp = wrap.querySelector('#decomp'), msg = wrap.querySelector('#msg');

    tl.at(700, () => { decomp.innerHTML = '把大小 n 的轴按 <span class="mono">[m, k, n/(m·k)]</span> 重排，子轴 <span class="mono">"a":(pre)size</span> 就是中间那段'; });
    tl.at(3000, () => {
      decomp.innerHTML = '<span class="mono">"a":(1)2</span> &nbsp;=&nbsp; pre=1, size=2 &nbsp;→&nbsp; 取第 1、2 个位置（下标 0..1）';
      ax.querySelectorAll('.dev').forEach((d, i) => d.classList.toggle('lit', i < 2));
    });
    tl.at(5600, () => {
      decomp.innerHTML = '<span class="mono">"a":(2)2</span> &nbsp;=&nbsp; pre=2, size=2 &nbsp;→&nbsp; 取第 3、4 个位置（下标 2..3）';
      ax.querySelectorAll('.dev').forEach((d, i) => d.classList.toggle('lit', i >= 2));
    });

    const host = wrap.querySelector('#cards');
    const defs = [
      { t: '④ 完整轴夹在中间', m: '[{"a":(1)2, "b", "a":(2)2}, {}]', d: '<b class="badge ok">合法</b>：a 的两个子轴被 "b" 隔开，不连续。' },
      { t: '⑤ 逆序连续子轴', m: '[{"a":(2)2, "a":(1)2}, {}]', d: '<b class="badge bad">报错</b>：相邻且可合并，必须写成 <span class="mono">"a"</span>。' },
    ];
    const els = defs.map(d => {
      const e = U.el('div', { class: 'card', style: 'width:320px;opacity:.35;transition:.3s' });
      e.innerHTML = `<div class="card-t" style="font-size:12px">${d.t}</div>
        <div class="mono" style="margin:5px 0;color:#bdf7ec;font-size:10.5px;overflow-wrap:anywhere">${U.esc(d.m)}</div>
        <div class="card-d" style="font-size:11.5px">${d.d}</div>`;
      host.appendChild(e); return e;
    });
    tl.at(8400, () => { els[0].style.opacity = '1'; msg.innerHTML = '不连续 → 无法合并成单个子轴，所以必须保持拆开写。'; });
    tl.at(11200, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '相邻且能合并 → 校验器要求你写成最简形式（L1-03 会细讲这条不变量）。';
    });
    tl.at(14200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = '用户手写的分片只能引用<b>完整轴</b>；子轴是编译器内部拆出来的。'; });
  }
},

/* --------------------------------------------------------- 7 优先级 */
{
  kicker: 'L1-02 · 分片语法',
  title: '优先级：<span class="mono hl-a">[{"a"}p0, {"b"}p1]</span>',
  sub: '写在<b>维度分片的花括号外</b>，紧跟轴列表。数字越小优先级越高；不写默认 p0。',
  caption: '优先级只影响<b>传播顺序</b>，不影响分片本身的含义。同一个 sharding 里不同维度可以有不同优先级。',
  code: `// ① 函数参数上：第 0 维 p0（可省略），第 1 维无优先级
func.func @f(%arg0: tensor<8x8xf32>
    {sdy.sharding = #sdy.sharding<@foo, [{"a"}p0, {"b"}]>})

// ② 算子结果上：第 1 维 p1
#sdy.sharding_per_value<[<@foo, [{"a"}, {"b"}p1]>]>

// ③ 多个维度各有优先级
#sdy.sharding_per_value<[<@foo, [{"a"}p0, {"b"}p1]>]>

// 注意：花括号内是轴，pN 在花括号<b>外</b>
//   {"a"}p0     正确
//   {"a"p0}     错误（见解析失败用例）`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:16px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="formula" style="font-size:15px">
        <span class="mono">[<span style="color:var(--ax1)">{"a"}</span><span style="color:var(--warn)">p0</span>, <span style="color:var(--ax1)">{"b"}</span>]</span>
        &nbsp;&nbsp;<span class="small faint">pN 在花括号外</span>
      </div>
      <div class="row" style="gap:14px;align-items:flex-end;justify-content:center" id="chart"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:720px"></div>`;
    root.appendChild(wrap);

    const chart = wrap.querySelector('#chart');
    const msg = wrap.querySelector('#msg');
    const bars = [0, 1, 2].map(p => {
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center;opacity:.3;transition:.4s' });
      c.innerHTML = `<div class="small mono">p${p}</div>
        <div style="width:74px;height:${40 + p * 26}px;border-radius:7px;background:var(--ax${p});opacity:.85"></div>
        <div class="small faint" style="font-size:11px">第 ${p + 1} 轮</div>`;
      chart.appendChild(c); return c;
    });

    tl.at(700, () => { msg.innerHTML = '传播按优先级<b>分批</b>进行：先所有 p0，再 p1，再 p2。'; });
    [0, 1, 2].forEach((p, i) => tl.at(2600 + i * 2600, () => {
      bars.forEach((b, k) => b.style.opacity = k <= i ? '1' : '.3');
      msg.innerHTML = [
        '第 0 轮：只传播 <b>p0</b> 的分片，铺满整张图。',
        '第 1 轮：p0 的结果保持不变，再加入 <b>p1</b>。',
        '第 2 轮：最后加入 <b>p2</b>，填充仍然开放的维度。',
      ][i];
    }));
    tl.at(10800, () => {
      msg.innerHTML = '不写 <span class="mono">pN</span> = <b>p0</b>；<span class="mono">[{"a"}p0, {"b"}]</span> 与 <span class="mono">[{"a"}, {"b"}]</span> 等价。';
    });
  }
},

/* ------------------------------------------------------ 8 特殊类型 */
{
  kicker: 'L1-02 · 分片语法',
  title: '四种边角情况：<span class="hl-a">rank-0 / 动态 / tuple / 无结果</span>',
  sub: '这些写法不常见，但一旦遇到就会卡住。测试文件专门覆盖了它们。',
  caption: '共同规律：<b>有多少个"值"，列表里就有多少项</b>；维数必须等于该值的 rank（rank-0 就是空列表）。',
  code: `// ① rank-0 张量：维分片列表为空，仍可显式复制
#sdy.sharding_per_value<[<@foo, [], replicated={"b"}>]>
     : tensor<f32>

// ② 动态形状：维数照样要写满
func.func @f(%arg0: tensor<?x?xf32>
    {sdy.sharding = #sdy.sharding<@foo, [{}, {"a"}]>})

// ③ tuple 类型：按 tuple 元素给分片
%0 = stablehlo.custom_call @sdy_testonly(%arg0)
     {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {}]>]>}
     : (tensor<8x8xf32>) -> tuple<tensor<8x8xf32>>

// ④ 无结果的算子：列表项对应"操作数"
stablehlo.custom_call @foo(%arg0)
     {has_side_effect = true,
      sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh, []>]>}
     : (tensor<8x8xf32>) -> ()`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%;align-items:center' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:740px"></div>`;
    root.appendChild(wrap);

    const defs = [
      { t: '① rank-0', m: '<@foo, [], replicated={"b"}>', d: '<span class="mono">tensor&lt;f32&gt;</span> 没有维度 →<br>维分片列表写成 <span class="mono">[]</span>，但仍可复制。' },
      { t: '② 动态形状', m: '<@foo, [{}, {"a"}]>', d: '<span class="mono">tensor&lt;?x?xf32&gt;</span> 也是 rank 2 →<br>照样要写两个维分片。' },
      { t: '③ tuple', m: '<@foo, [{"a"}, {}]>', d: '结果是 <span class="mono">tuple&lt;tensor&lt;8x8&gt;&gt;</span> →<br>列表<b>一项</b>，对应 tuple 里的张量。' },
      { t: '④ 无结果', m: '<@maximal_mesh, []>', d: '没有结果 → 列表项对应<b>操作数</b>。<br><span class="mono">[]</span> = 该操作数 rank-0。' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => {
      const e = U.el('div', { class: 'card', style: 'width:178px;opacity:.32;transition:.3s' });
      e.innerHTML = `<div class="card-t" style="font-size:12px">${d.t}</div>
        <div class="mono" style="margin:5px 0;color:#bdf7ec;font-size:10px;overflow-wrap:anywhere">${U.esc(d.m)}</div>
        <div class="card-d" style="font-size:11.5px">${d.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    const tips = [
      'rank-0 不是"没有分片"：它可以是复制状态，也可以有未归约轴。',
      '动态维也是维，维数必须写满 —— 与静态形状规则一致。',
      'tuple 内部的每个张量算作一个"值"。',
      '无结果算子的分片描述的是<b>输入</b>该怎么切，用于带副作用的调用。',
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.32');
      msg.innerHTML = tips[i];
    }));
    tl.at(14800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '统一规律：<b>列表项数 = 值的个数；每项的维分片数 = 该值的 rank</b>。';
    });
  }
},

/* --------------------------------------------------- 9 十五类解析错误 */
{
  kicker: 'L1-02 · 分片语法',
  title: '15 类解析错误：<span class="hl-a">语法</span>层面的坑',
  sub: '解析器在<b>语法</b>阶段就能拦下这些写法。它们与 L1-03 的"校验错误"不同 —— 那些是语法对但语义非法。',
  caption: '区分两级错误很重要：<b>解析错误</b>说"我看不懂"，<b>校验错误</b>说"我看懂了但不合法"。',
  code: `// ①~③ 优先级写法错
[{"b"}phigh]                  // 不是数字
[{"b"}p9999999999999999999]   // 整数溢出
[{"a"}p01, {"b"}]             // 前导零

// ④~⑨ 关键字与集合写法错
replicated{"a", "b"}          // 漏了 =
unreduced=["a", "b"]          // 用了方括号
unreduced={"a", "b"[]}        // 混入 []
replicated={"a", b}           // 轴名没引号
unreduced={"a", b}            // 轴名没引号
replicated unreduced={"a"}    // 两关键字连写

// ⑩~⑮ 逗号与重复字段
[{}, {},]                     // 逗号后没内容
unknown={"a"}                 // 未知字段名
replicated={"a"}, >           // 逗号后直接结束
replicated={"a"}, unreduced={"b"}, >   // 都多逗号
replicated={"a"}, replicated={"b"}     // 重复 replicated
replicated={"a"}, unreduced={"b"}, unreduced={"b"}  // 重复 unreduced`,
  duration: 20000,
  build(root, tl) {
    const GROUPS = [
      {
        name: '① 优先级写法错（3 类）',
        items: [
          ['[{"b"}phigh]', 'expecting priority in format \'p<number>\', got: phigh'],
          ['[{"b"}p9999999999999999999]', 'expecting integer priority, got: p9999999999999999999'],
          ['[{"a"}p01, {"b"}]', 'priorities with leading zeros are not allowed, got: p01'],
        ],
        why: '优先级必须是<b>合法十进制整数</b>：不能有字母、不能溢出 int64、不能有前导零（否则 p01 与 p1 会歧义）。',
      },
      {
        name: '② 关键字与集合写法错（6 类）',
        items: [
          ['replicated{"a", "b"}', "expected '='"],
          ['unreduced=["a", "b"]', "expected '{'"],
          ['unreduced={"a", "b"[]}', "expected '}'"],
          ['replicated={"a", b}', 'expected string'],
          ['unreduced={"a", b}', 'expected string'],
          ['replicated unreduced={"a", "b"}', "expected '='"],
        ],
        why: '两个关键字都必须写成 <span class="mono">name={...}</span>：<b>等号不能省</b>，集合用<b>花括号</b>，轴名必须<b>带引号</b>。',
      },
      {
        name: '③ 逗号与重复字段（6 类）',
        items: [
          ['[{}, {},]', 'expected valid named axis list after comma'],
          ['unknown={"a"}', 'expected valid named axis list after comma'],
          ['replicated={"a"}, >', 'expected valid named axis list after comma'],
          ['replicated={"a"}, unreduced={"b"}, >', 'expected valid named axis list after comma'],
          ['replicated={"a"}, replicated={"b"}', 'expected valid named axis list after comma'],
          ['replicated={"a"}, unreduced={"b"}, unreduced={"b"}', 'expected valid named axis list after comma'],
        ],
        why: '属性里<b>不允许尾随逗号</b>，字段名<b>只能出现一次</b>，且只认 <span class="mono">replicated</span> / <span class="mono">unreduced</span> 两个名字。',
      },
    ];

    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `
      <div class="row" style="gap:12px;align-items:center;justify-content:center" id="stepper"></div>
      <div class="mid" id="gname" style="text-align:center;font-weight:700"></div>
      <div class="row" style="gap:16px;align-items:stretch;width:100%">
        <div class="irbox in" style="flex:1.25"><div class="irh">错误写法</div>
          <pre id="bad" style="min-height:150px;font-size:11.5px"></pre></div>
        <div class="col" style="flex:1;gap:10px">
          <div class="card" style="border-color:rgba(251,113,133,.45)">
            <div class="card-t" style="color:var(--bad);font-size:12px">解析器报错</div>
            <div class="card-d mono" id="err" style="font-size:11px;line-height:1.6;color:#ffc9d0"></div></div>
          <div class="card"><div class="card-t" style="font-size:12px">共同根因</div>
            <div class="card-d" id="why" style="font-size:12.5px"></div></div>
        </div>
      </div>`;
    root.appendChild(wrap);

    const st = W.stepper(GROUPS.length);
    wrap.querySelector('#stepper').appendChild(st.el);
    const gname = wrap.querySelector('#gname'), bad = wrap.querySelector('#bad'),
      err = wrap.querySelector('#err'), why = wrap.querySelector('#why');

    GROUPS.forEach((g, i) => tl.at(800 + i * 5600, () => {
      st.set(i);
      gname.textContent = g.name;
      bad.innerHTML = U.hl(g.items.map(it => it[0]).join('\n'));
      err.innerHTML = g.items.map((it, k) => `${k + 1}. ${U.esc(it[1])}`).join('<br>');
      why.innerHTML = g.why;
    }));
    tl.at(800 + GROUPS.length * 5600, () => {
      st.set(-1); gname.textContent = '共性';
      bad.innerHTML = U.hl('// 解析错误的共同特征：\n// 属性本身就没写成一个合法的 attribute');
      err.innerHTML = 'failed to parse Sdy_TensorSharding …';
      why.innerHTML = '报文里出现 <span class="mono">failed to parse</span> 就是解析层。<br>下一课 L1-03 讲的是<b>语法正确但语义非法</b>的校验错误。';
    });
  }
},

/* ------------------------------------------------------------ 10 练习 */
{
  kicker: 'L1-02 · 练习',
  title: '练一练：<span class="hl-a">读属性、判错误</span>',
  sub: '三道题分别考：属性解读、语法纠错、规则判断。',
  caption: '能答对这三题，L1-01 与 L1-02 的语法部分就扎实了。',
  code: `// 题 1：这条属性说了什么？
#sdy.sharding<@foo, [{"a"}p0, {"b", ?}], replicated={"c"}>

// 题 2：错在哪？
#sdy.sharding<@foo, [{"a"}, {"b"}phigh]>

// 题 3：合法吗？
sdy.mesh @bar = <["a"=4, "b"=2]>
#sdy.sharding<@bar, [{"a":(1)2, "b", "a":(2)2}, {}]>`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">#sdy.sharding&lt;@foo, [{"a"}p0, {"b", ?}], replicated={"c"}&gt;</span> 说了什么？',
        a: '第 0 维沿 <b>"a"</b> 分片，优先级 <b>p0</b>，<b>闭维</b>；第 1 维沿 <b>"b"</b> 分片且是<b>开维</b>（还能加轴）；' +
           '<b>"c"</b> 被显式复制，禁止用于分片；其余轴（如 <span class="mono">"d"</span>）隐式复制。'
      },
      {
        q: '<span class="mono">#sdy.sharding&lt;@foo, [{"a"}, {"b"}phigh]&gt;</span> 错在哪？',
        a: '<b class="badge bad">解析错误</b> 优先级必须是数字：<span class="mono">phigh</span> 触发 ' +
           '<span class="mono">expecting priority in format \'p&lt;number&gt;\', got: phigh</span>。<br>' +
           '<span class="dim">正确写法：<span class="mono">{"b"}p1</span> 或干脆不写（默认 p0）。</span>'
      },
      {
        q: '<span class="mono">#sdy.sharding&lt;@bar, [{"a":(1)2, "b", "a":(2)2}, {}]&gt;</span> 合法吗？',
        a: '<b class="badge ok">合法</b> 两个 "a" 的子轴被完整轴 "b" 隔开，不连续，因此<b>不能</b>合并成单个子轴，' +
           '必须保持拆开写。<br><span class="dim">若写成 <span class="mono">[{"a":(2)2, "a":(1)2}, …]</span>（相邻且逆序）则会被校验器拒绝。</span>'
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
