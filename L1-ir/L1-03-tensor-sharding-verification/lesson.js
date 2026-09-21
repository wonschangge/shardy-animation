/* ==========================================================================
   L1-03 · tensor-sharding-verification
   --------------------------------------------------------------------------
   覆盖：ir/test/tensor_sharding_verification.mlir (556 行，L1 最大单文件)
   目标：讲透分片属性的语义不变量 —— 语法正确但非法的那 33 类错误。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

/* 复用的"错误用例"渲染：stepper + 错误 IR + 报错 + 根因 */
function errScenes(wrap, tl, cases, opts = {}) {
  const st = W.stepper(cases.length);
  wrap.querySelector('#stepper').appendChild(st.el);
  const bad = wrap.querySelector('#bad'), err = wrap.querySelector('#err'),
    why = wrap.querySelector('#why'), cnt = wrap.querySelector('#cnt');
  cases.forEach((c, i) => tl.at(700 + i * 2400, () => {
    st.set(i);
    if (cnt) cnt.textContent = `${i + 1} / ${cases.length}`;
    bad.innerHTML = U.hl(c.ir);
    err.innerHTML = c.err;
    why.innerHTML = c.why;
  }));
  tl.at(700 + cases.length * 2400, () => {
    st.set(-1);
    bad.innerHTML = U.hl(opts.finalIr || '// 完成后回顾：这些写法的共同问题是什么？');
    if (cnt) cnt.textContent = '小结';
    err.innerHTML = opts.finalErr || '';
    why.innerHTML = opts.finalWhy || '';
  });
}

/* 错误用例场景的通用骨架 */
function errLayout(root, title) {
  const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
  wrap.innerHTML = `
    <div class="row" style="gap:12px;align-items:center;justify-content:center" id="stepper"></div>
    <div class="row" style="gap:14px;align-items:stretch;width:100%">
      <div class="irbox in" style="flex:1.2"><div class="irh">违规 IR<span class="faint" id="cnt" style="float:right"></span></div>
        <pre id="bad" style="min-height:120px;font-size:11.5px"></pre></div>
      <div class="col" style="flex:1;gap:9px">
        <div class="card" style="border-color:rgba(251,113,133,.45)">
          <div class="card-t" style="color:var(--bad);font-size:12px">校验器报错</div>
          <div class="card-d mono" id="err" style="font-size:11px;line-height:1.55;color:#ffc9d0"></div></div>
        <div class="card"><div class="card-t" style="font-size:12px">为什么</div>
          <div class="card-d" id="why" style="font-size:12.5px"></div></div>
      </div>
    </div>`;
  root.appendChild(wrap);
  return wrap;
}

const SCENES = [

/* ------------------------------------------------------ 1 两级检查 */
{
  kicker: 'L1-03 · 分片校验',
  title: '语法对 ≠ <span class="hl-a">合法</span>：两级检查',
  sub: 'L1-02 讲的是<b>解析</b>（能不能读懂语法）。这一课讲<b>校验</b>：语法完全正确，但违反了分片的不变量。',
  caption: '报文里出现 <span class="mono">failed to parse</span> 是解析层；出现具体的语义描述（如 <span class="mono">duplicate axis ref</span>）就是校验层。',
  code: `// 解析层（L1-02）：属性根本没写成一个合法 attribute
//   failed to parse Sdy_TensorSharding …

// 校验层（本课）：属性合法，但语义不成立
//   duplicate axis ref: "a"
//   unknown axis name: "c"
//   sharding doesn't match tensor rank: 2 != 1
//   two consecutive sub-axes can be merged: "a":(2)2, "a":(4)4

// 测试方式：
// RUN: sdy_opt %s -split-input-file -verify-diagnostics`,
  duration: 12000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:16px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:20px;justify-content:center;align-items:stretch">
        <div class="card" style="width:300px;border-color:rgba(56,189,248,.5)">
          <div class="card-t" style="color:var(--ax0)">① 解析 parse</div>
          <div class="card-d">能不能读成一个 attribute？<br>
            <span class="mono small">failed to parse …</span><br>
            <span class="dim small">→ L1-02 覆盖 15 类</span></div>
        </div>
        <div class="arrow" style="font-size:26px;align-self:center">⟹</div>
        <div class="card" style="width:300px;border-color:rgba(251,113,133,.5)">
          <div class="card-t" style="color:var(--bad)">② 校验 verify</div>
          <div class="card-d">读懂了，但违反不变量？<br>
            <span class="mono small">duplicate axis ref …</span><br>
            <span class="dim small">→ 本课覆盖 33 类</span></div>
        </div>
      </div>
      <div class="formula" id="msg" style="min-height:46px;display:flex;align-items:center;text-align:center;max-width:720px"></div>`;
    root.appendChild(wrap);
    const msg = wrap.querySelector('#msg');
    tl.at(900, () => { msg.innerHTML = '解析器只关心"形状"：括号配对、引号闭合、字段名认识。'; });
    tl.at(3400, () => { msg.innerHTML = '校验器才关心"含义"：轴存在吗？重复了吗？和 rank 对得上吗？'; });
    tl.at(6000, () => { msg.innerHTML = '这一课把 33 类校验错误按<b>不变量</b>归成 7 组 —— 记住不变量，报错就能自己推。'; });
    tl.at(8800, () => { msg.innerHTML = '<span class="hl-a">读报错的诀窍</span>：先看它说哪条不变量，再看自己违反了哪个前提。'; });
  }
},

/* ------------------------------------------------------ 2 不变量总览 */
{
  kicker: 'L1-03 · 分片校验',
  title: '七条不变量：校验器在守什么',
  sub: '所有 33 类错误都能归到下面七条上。它们共同保证"分片属性<b>无歧义、可计算</b>"。',
  caption: '其中第 1、2 条最常见（占全部错误的一半以上）；第 7 条最容易被忽略 —— 它涉及<b>上下文</b>而非属性本身。',
  code: `// 1. 轴引用必须存在且唯一
//    unknown axis name / duplicate axis ref
//    both sub-axis and full-axis / overlapping sub-axes

// 2. 子轴参数必须合法
//    pre-size >= 1 / size > 1 / size != 整轴
//    next pre-size 必须整除整轴 / 相邻子轴必须可合并

// 3. 复制轴与未归约轴必须按网格顺序

// 4. 维分片数必须等于张量 rank

// 5. 空的闭维不能带优先级

// 6. 属性类型与数量必须匹配
//    算子用 per_value / 函数用 sharding / 项数 = 值数

// 7. 上下文约束：轴不能已被父级 manual_computation 绑定`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;flex-wrap:wrap;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:42px;display:flex;align-items:center;text-align:center"></div>`;
    root.appendChild(wrap);

    const defs = [
      { t: '1 · 轴引用唯一', d: '存在、不重复、<br>不与子轴混用、不重叠', c: '#fb7185', n: '14 类' },
      { t: '2 · 子轴参数合法', d: 'pre-size / size / 整除 /<br>可合并', c: '#fbbf24', n: '8 类' },
      { t: '3 · 排序正确', d: 'replicated 与 unreduced<br>按网格序', c: '#4ade80', n: '2 类' },
      { t: '4 · rank 一致', d: '维分片数 = 张量 rank', c: '#38bdf8', n: '2 类' },
      { t: '5 · 空闭维无优先级', d: '{} 不能带 pN', c: '#c084fc', n: '2 类' },
      { t: '6 · 属性类型/数量', d: 'per_value vs sharding；<br>项数 = 值数', c: '#f472b6', n: '4 类' },
      { t: '7 · 上下文约束', d: '轴被父级 manual<br>绑定后不可再切', c: '#93c5fd', n: '1 类' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => {
      const e = U.el('div', { class: 'card', style: 'width:102px;opacity:.3;transition:.3s;padding:8px' });
      e.innerHTML = `<div class="card-t" style="font-size:11px;color:${d.c}">${d.t}</div>
        <div class="card-d" style="font-size:10.5px;line-height:1.45">${d.d}</div>
        <div class="small faint" style="margin-top:5px;font-size:10px">${d.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((d, i) => tl.at(600 + i * 1900, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.3');
      msg.innerHTML = [
        '最基础也最常犯：轴必须来自网格，且每个轴只能出现一次。',
        '子轴的三个参数都有取值范围，且要"写满"（相邻能合并就必须合并）。',
        '排序不是语义要求，而是<b>规范化</b>要求 —— 保证同一分片只有一种写法。',
        '维分片列表的长度必须正好等于该张量的 rank。',
        '空的闭维 <span class="mono">{}</span> 没有任何可优先的东西。',
        '属性挂错地方、项数对不上，都属于结构性问题。',
        '唯一一条"依赖外部上下文"的不变量，只在 manual_computation 内出现。',
      ][i];
    }));
    tl.at(600 + defs.length * 1900, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '接下来按这七条逐一展开典型用例。';
    });
  }
},

/* ------------------------------------------------ 3 不变量 1：轴引用唯一 */
{
  kicker: 'L1-03 · 不变量 1',
  title: '轴引用：<span class="hl-a">存在</span>且<span class="hl-a">唯一</span>',
  sub: '每个轴（或子轴）在整条分片属性里<b>最多出现一次</b> —— 无论是切维度、显式复制还是未归约。',
  caption: '网格 <span class="mono">@mesh = &lt;["a"=2]&gt;</span> 只有一个轴，所以下面所有用 <span class="mono">"b"</span>/<span class="mono">"c"</span> 的例子都会报"未知轴"。',
  code: `// 网格：sdy.mesh @mesh = <["a"=2]>
// ① 维度分片用了 "a"，replicated 又用了 "a"
%0 = stablehlo.add %arg0, %arg1
     {sdy.sharding=#sdy.sharding_per_value<
        [<@mesh, [{}, {"a"}], replicated={"a"}>]>}

// ② 子轴重复
[<@mesh, [{}, {"a":(2)2}], replicated={"a":(2)2}>]

// ③ 完整轴与子轴混用
[<@mesh, [{}, {"a"}], replicated={"a":(2)2}>]

// ④ 子轴重叠（网格 a=8）
[<@mesh, [{"a":(2)4}, {"b":(2)2}], replicated={"a":(1)4}>]

// ⑤ 未知轴名
[<@mesh, [{}, {"c"}]>]

// ⑥ 未知网格
[<@other_mesh, [{}, {"a"}]>]`,
  duration: 20000,
  build(root, tl) {
    const wrap = errLayout(root);
    errScenes(wrap, tl, [
      {
        ir: '%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a"}], replicated={"a"}>]>} : tensor<8x8xf32>',
        err: 'duplicate axis ref: "a"',
        why: '轴 "a" 已经在第 1 维的分片里了，不能在 <span class="mono">replicated</span> 里再出现 —— 一个轴只能承担一种角色。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(2)2}], replicated={"a":(2)2}>]>} : tensor<8x8xf32>',
        err: 'duplicate axis ref: "a":(2)2',
        why: '子轴与完整轴<b>指向同一段设备</b>，重复使用同样违反唯一性。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a"}], replicated={"a":(2)2}>]>} : tensor<8x8xf32>',
        err: 'both sub-axis and full-axis are used for axis name: "a"',
        why: '完整轴已经覆盖了全部子轴，再写子轴就是重复占用。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"a":(2)4}, {"b":(2)2}], replicated={"a":(1)4}>]>} : tensor<8x8xf32>',
        err: 'overlapping sub-axes: "a":(1)4, "a":(2)4',
        why: '网格 a=8：<span class="mono">"a":(2)4</span> 覆盖下标 2–5，<span class="mono">"a":(1)4</span> 覆盖 0–3，<b>区间重叠</b>。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"c"}]>]>} : tensor<8x8xf32>',
        err: 'op result - unknown axis name: "c"',
        why: '网格是 <span class="mono">&lt;["a"=2]&gt;</span>，根本没有 "c"。轴名必须来自被引用的网格。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@other_mesh, [{}, {"a"}]>]>} : tensor<8x8xf32>',
        err: 'op result - unknown mesh: @other_mesh',
        why: '符号 <span class="mono">@other_mesh</span> 在 module 里不存在。'
      },
    ], {
      finalIr: '// 一句话：轴是"独占资源"\n// 切维度 / 显式复制 / 未归约 —— 三选一，且只能选一次',
      finalErr: 'duplicate axis ref / unknown axis name / overlapping sub-axes',
      finalWhy: '<b>14 类错误</b>都属于这一条不变量。修复方式：检查这个轴是不是已经在别处用过了。'
    });
  }
},

/* ------------------------------------------------ 4 不变量 2：子轴参数 */
{
  kicker: 'L1-03 · 不变量 2',
  title: '子轴参数：<span class="mono hl-a">(pre)size</span> 的取值约束',
  sub: '四个数都有要求：pre-size ≥ 1、size &gt; 1、size ≠ 整轴大小、next pre-size 必须整除整轴。',
  caption: '还有一条"写法最简"要求：相邻的子轴必须合并到最大，否则报 <span class="mono">can be merged</span>。',
  code: `// 网格 a=8
// ① pre-size 必须 >= 1
[<@mesh, [{}, {"a":(-1)2}]>]
//   sub-axis pre-size must be at least 1

// ② size 必须 > 1
[<@mesh, [{}, {"a":(2)1}]>]
//   sub-axis sizes must be greater than 1

// ③ size 不能等于整轴大小
[<@mesh, [{}, {"a":(1)8}]>]
//   sub-axis size is equal to the full axis size

// ④ next pre-size 必须整除整轴 (pre+size 越界)
[<@mesh, [{}, {"a":(4)4}]>]
//   sub-axis next pre-size 16 doesn't divide the size of the full axis 8

// ⑤ 相邻子轴必须可合并 -> 应写成更大的子轴
[<@mesh, [{}, {"a":(2)2, "a":(4)4}]>]
//   two consecutive sub-axes can be merged: "a":(2)2, "a":(4)4`,
  duration: 17000,
  build(root, tl) {
    const wrap = errLayout(root);
    errScenes(wrap, tl, [
      {
        ir: '%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(-1)2}]>]>} : tensor<8x8xf32>',
        err: 'sub-axis pre-size must be at least 1: "a":(-1)2',
        why: 'pre-size 是"左边子轴大小之积"，最小是 1（表示左边没有子轴）。负数无意义。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(2)1}]>]>} : tensor<8x8xf32>',
        err: 'sub-axis sizes must be greater than 1: "a":(2)1',
        why: '大小为 1 的子轴<b>不切任何东西</b>，等价于不存在，因此不允许。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(1)8}]>]>} : tensor<8x8xf32>',
        err: 'sub-axis size is equal to the full axis size: "a":(1)8',
        why: '既然覆盖了整个轴，就应该直接写完整轴 <span class="mono">"a"</span>，不该用子轴形式。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(4)4}]>]>} : tensor<8x8xf32>',
        err: "sub-axis next pre-size 16 doesn't divide the size of the full axis 8: \"a\":(4)4",
        why: 'next pre-size = pre + size = 4 + 4 = 8… 这里 pre=4、size=4 意味着起点在 4、宽 4，正好到 8；但校验报 16，说明 <span class="mono">(pre)size</span> 的 pre 与 size 需满足整除关系 —— 关键约束是<b>子轴不能越过整轴</b>。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(2)2, "a":(4)4}]>]>} : tensor<8x8xf32>',
        err: 'two consecutive sub-axes can be merged: "a":(2)2, "a":(4)4',
        why: '<span class="mono">"a":(2)2</span> 覆盖 2–3、<span class="mono">"a":(4)4</span> 覆盖 4–7，<b>相邻</b> → 必须合并写成 <span class="mono">"a":(2)6</span>。'
      },
    ], {
      finalIr: '// 子轴的"最简写法"要求：\n//   相邻子轴 -> 合并\n//   覆盖整轴 -> 用完整轴\n//   大小为 1 -> 删掉',
      finalErr: 'sub-axis pre-size / size / next pre-size / can be merged',
      finalWhy: '这四条保证：同一个分片方案<b>只有一种写法</b>（规范化），diff 与相等判断才可靠。'
    });
  }
},

/* --------------------------------------------- 5 不变量 3/4/5：三条快查 */
{
  kicker: 'L1-03 · 不变量 3/4/5',
  title: '三条"一看就懂"的不变量',
  sub: '排序、rank 一致、空闭维不带优先级 —— 这三条不需要推导，记住即可。',
  caption: '排序那条容易被忽略：<b>不是语义要求，而是规范化要求</b>，保证同一分片只有一种写法。',
  code: `// 网格：sdy.mesh @mesh = <["c"=2, "a"=2, "b"=2]>
// ① replicated 必须按网格声明顺序 c, a, b
replicated={"a", "b", "c"}       // ✗
//   replicated axes are not ordered w.r.t. mesh

// ② unreduced 同理
unreduced={"a", "b", "c"}        // ✗
//   unreduced axes are not ordered w.r.t. mesh

// ③ 维分片数必须等于 rank：tensor<8xf32> 是 rank 1
[<@mesh, [{}, {"b"}], replicated={"a"}>] : tensor<8xf32>   // ✗
//   sharding doesn't match tensor rank: 2 != 1

// ④ 空的闭维不能有优先级
[<@mesh, [{"a"}, {}p3]>]         // ✗
//   dim 1 is empty and closed but has a priority`,
  duration: 15000,
  build(root, tl) {
    const wrap = errLayout(root);
    errScenes(wrap, tl, [
      {
        ir: '%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {}], replicated={"a", "b", "c"}>]>} : tensor<8x8xf32>',
        err: 'replicated axes are not ordered w.r.t. mesh',
        why: '网格声明顺序是 <span class="mono">c, a, b</span>，所以必须写成 <span class="mono">replicated={"c", "a", "b"}</span>。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"a", "b", "c"}>]>} : tensor<8x8xf32>',
        err: 'unreduced axes are not ordered w.r.t. mesh',
        why: '<span class="mono">unreduced</span> 与 <span class="mono">replicated</span> 使用<b>同一套排序规则</b>。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"b"}], replicated={"a"}>]>} : tensor<8xf32>',
        err: "op result - sharding doesn't match tensor rank: 2 != 1",
        why: '给了 2 个维分片，但 <span class="mono">tensor&lt;8xf32&gt;</span> 只有 1 维 → 报 <span class="mono">2 != 1</span>。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}p3]>]>} : tensor<8x8xf32>',
        err: 'dim 1 is empty and closed but has a priority',
        why: '空的闭维 <span class="mono">{}</span> 表示"这一维不切"，没有可排序的分片决策，优先级无处生效。'
      },
    ], {
      finalIr: '// 记忆口诀：\n//   轴按网格排（replicated / unreduced）\n//   维数等于 rank\n//   空闭维别写 pN',
      finalErr: 'not ordered w.r.t. mesh / rank mismatch / empty and closed',
      finalWhy: '这三条都是"看一眼就能判断"的，实践中报错频率很高。'
    });
  }
},

/* -------------------------------------------- 6 不变量 6：类型与数量 */
{
  kicker: 'L1-03 · 不变量 6',
  title: '属性类型与数量：<span class="hl-a">挂对地方、数对数目</span>',
  sub: '算子必须用 <span class="mono">sharding_per_value</span>，函数参数必须用 <span class="mono">sharding</span>；项数必须等于值的个数。',
  caption: '这类错误的报文会明确告诉你"期望哪种属性"，是最容易自我修复的一类。',
  code: `// ① 算子用了 sharding（应为 per_value）
%0 = stablehlo.add %arg0, %arg1
     {sdy.sharding=#sdy.sharding<@mesh, [{}, {"a"}]>} : tensor<8x8xf32>
//   op should have a sharding attribute of type
//   TensorShardingPerValueAttr

// ② 函数参数用了 per_value（应为 sharding）
func.func @f(%arg0: tensor<8x8xf32>
    {sdy.sharding=#sdy.sharding_per_value<[...]>}) ...
//   should have a sharding attribute of type TensorShardingAttr

// ③ 项数必须等于值的个数：2 个结果给了 3 项
%1:2 = stablehlo.reduce(...) ...
//   op result shardings don't match number of values:
//   3 shardings vs 2 values

// ④ tuple 只支持大小为 1
... -> tuple<tensor<8x8xf32>, tensor<8x8xf32>>
//   ops can only have a sharding for a tuple of size 1`,
  duration: 16000,
  build(root, tl) {
    const wrap = errLayout(root);
    errScenes(wrap, tl, [
      {
        ir: '%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding<@mesh, [{}, {"a"}]>} : tensor<8x8xf32>',
        err: 'op should have a sharding attribute of type TensorShardingPerValueAttr',
        why: '算子上必须用<b>列表</b>形式（<span class="mono">per_value</span>），因为一个算子可能有多个操作数/结果。'
      },
      {
        ir: 'func.func @func_arg_with_tensor_sharding_per_value_attr(',
        err: "'func.func' op arg 1 - should have a sharding attribute of type TensorShardingAttr",
        why: '函数参数只有一个值，用<b>单值</b>形式 <span class="mono">sdy.sharding</span>，不需要列表。'
      },
      {
        ir: '%1:2 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %0) across dimensions = [1]',
        err: "op result shardings don't match number of values: 3 shardings vs 2 values",
        why: '算子有 2 个结果，列表却有 3 项 → 第 3 项<b>没有对应的值</b>。项数必须严格相等。'
      },
      {
        ir: '%0 = stablehlo.custom_call @sdy_testonly(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x8xf32>) -> tuple<tensor<8x8xf32>, tensor<8x8xf32>>',
        err: 'ops can only have a sharding for a tuple of size 1',
        why: '多元素 tuple 目前不支持逐元素分片；只有<b>单元素 tuple</b> 能挂分片。'
      },
    ], {
      finalIr: '// 速记：\n//   算子 -> per_value（列表）\n//   函数 -> sharding（单值）\n//   项数 == 值数\n//   tuple 仅支持 size 1',
      finalErr: 'TensorShardingPerValueAttr / TensorShardingAttr / don\'t match number of values / tuple of size 1',
      finalWhy: '报错信息本身就给出了正确答案，照着改即可。'
    });
  }
},

/* --------------------------------------------- 7 形状与网格的特殊限制 */
{
  kicker: 'L1-03 · 不变量 4（续）',
  title: '特殊类型：<span class="hl-a">token / 无 rank / maximal 网格</span>',
  sub: '非张量类型（token）、无 rank 张量（<span class="mono">tensor&lt;*xf32&gt;</span>）、以及 maximal-sharding 网格都有额外限制。',
  caption: '共同点：<b>分片需要一个确定的 rank</b>。rank 不确定或不存在时，只能"整体复制"这一种状态。',
  code: `// ① token 是 non-shaped，只能 rank 0 且无 replicated/unreduced
func.func @f(%arg0: !stablehlo.token
    {sdy.sharding=#sdy.sharding<@mesh, [{}]>}) ...
//   non-shaped tensors can only have a sharding with
//   rank 0 and no replicated or unreduced axes

// ② token 带 replicated / unreduced 同样非法
#sdy.sharding<@mesh, [], replicated={"a"}>
#sdy.sharding<@mesh, [], unreduced={"a"}>

// ③ 无 rank 张量不能有分片
func.func @f(%arg0: tensor<*xf32>
    {sdy.sharding=#sdy.sharding<@mesh, []>}) ...
//   only ranked tensors can have a sharding

// ④ maximal-sharding 网格只能配 rank 0
#sdy.sharding_per_value<[<@maximal_mesh, [{}, {}]>]>
#sdy.sharding_per_value<[<@maximal_mesh, [{}]>]>
//   a maximal sharding can only have a sharding with
//   rank 0 and no replicated or unreduced axes`,
  duration: 16000,
  build(root, tl) {
    const wrap = errLayout(root);
    errScenes(wrap, tl, [
      {
        ir: 'func.func @token_sharding_rank_non_zero(%arg0: !stablehlo.token {sdy.sharding=#sdy.sharding<@mesh, [{}]>}) -> !stablehlo.token {',
        err: "'func.func' op arg 0 - non-shaped tensors can only have a sharding with rank 0 and no replicated or unreduced axes",
        why: '<span class="mono">!stablehlo.token</span> 没有形状，维分片列表必须是<b>空</b>。'
      },
      {
        ir: 'func.func @token_sharding_with_replicated_axes(%arg0: !stablehlo.token {sdy.sharding=#sdy.sharding<@mesh, [], replicated={"a"}>}) -> !stablehlo.token {',
        err: "'func.func' op arg 0 - non-shaped tensors can only have a sharding with rank 0 and no replicated or unreduced axes",
        why: 'token 是<b>逐设备</b>的，谈不上"沿某个轴复制"。'
      },
      {
        ir: 'func.func @unranked_tensor_with_sharding(%arg0: tensor<*xf32> {sdy.sharding=#sdy.sharding<@mesh, []>}) -> tensor<*xf32> {',
        err: "'func.func' op arg 0 - only ranked tensors can have a sharding",
        why: '<span class="mono">tensor&lt;*xf32&gt;</span> 的 rank 未知，无法判断该写几个维分片。'
      },
      {
        ir: '%0 = stablehlo.custom_call @sdy_testonly(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh, [{}, {}]>]>} : (tensor<8x8xf32>) -> tuple<tensor<8x8xf32>>',
        err: 'a maximal sharding can only have a sharding with rank 0 and no replicated or unreduced axes',
        why: 'maximal 网格只有一台设备，"切"这件事没有意义，只能整体 rank 0。'
      },
      {
        ir: 'stablehlo.custom_call @foo(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh, [{}]>]>} : (tensor<8x8xf32>) -> ()',
        err: 'a maximal sharding can only have a sharding with rank 0 and no replicated or unreduced axes',
        why: '即使是空维分片 <span class="mono">[{}]</span>，它的长度 1 也代表 rank 1 ≠ 0。'
      },
    ], {
      finalIr: '// 判据：这个类型有确定的 rank 吗？\n//   没有 -> 只能 []（rank 0），且无 replicated/unreduced\n//   maximal 网格同理：它只有 1 台设备',
      finalErr: 'non-shaped tensors / only ranked tensors / a maximal sharding',
      finalWhy: '分片的前提是"能算出局部形状"，rank 不确定就无从算起。'
    });
  }
},

/* ------------------------------------------ 8 不变量 7：上下文约束 */
{
  kicker: 'L1-03 · 不变量 7',
  title: '唯一一条<span class="hl-a">上下文</span>约束：manual 轴已被绑定',
  sub: '前面七组都在检查属性本身。这一条要结合<b>外层算子</b>才能判断：轴已经被 <span class="mono">manual_computation</span> 绑定了。',
  caption: '<span class="mono">manual_axes={"a"}</span> 的意思是"轴 a 上的一切由我手动控制"，区域内自然不能再拿 a 去切维度。',
  code: `sdy.mesh @mesh = <["a"=2]>

func.func @sharding_bound_manual_computation(
    %arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = sdy.manual_computation(%arg0)
       in_shardings=[<@mesh, [{"a",?}, {?}]>]
       out_shardings=[<@mesh, [{"a",?}, {?}]>]
       manual_axes={"a"}
       (%arg1: tensor<8x32xf32>) {
    // ✗ 轴 "a" 已被父级绑定为 manual
    %0 = stablehlo.add %arg1, %arg1
         {sdy.sharding=#sdy.sharding_per_value<
            [ <@mesh, [{"a"}, {}]>]>} : tensor<8x32xf32>
    sdy.return %0 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `
      <div class="irbox in"><div class="irh">违规 IR</div>
        <pre style="font-size:11px;max-height:170px">${U.hl(`  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a",?}, {?}]>] out_shardings=[<@mesh, [{"a",?}, {?}]>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) {
    %0 = stablehlo.add %arg1, %arg1 {sdy.sharding=#sdy.sharding_per_value<[ <@mesh, [{"a"}, {}]>]>} : tensor<8x32xf32>
    sdy.return %0 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>`)}</pre></div>
      <div class="row" style="gap:14px;align-items:stretch">
        <div class="card" style="flex:1;border-color:rgba(251,113,133,.45)">
          <div class="card-t" style="color:var(--bad);font-size:12px">校验器报错</div>
          <div class="card-d mono" style="font-size:11px;color:#ffc9d0">'stablehlo.add' op result - operates on axis "a" which is already bound by a parent sdy.manual_computation op</div>
        </div>
        <div class="card" style="flex:1"><div class="card-t" style="font-size:12px">为什么</div>
          <div class="card-d" id="why" style="font-size:12.5px">等待…</div></div>
      </div>
      <div class="formula" id="msg" style="min-height:42px;display:flex;align-items:center;text-align:center"></div>`;
    root.appendChild(wrap);
    const why = wrap.querySelector('#why'), msg = wrap.querySelector('#msg');

    tl.at(900, () => { why.innerHTML = '区域内张量已经是<b>局部形状</b> <span class="mono">tensor&lt;8x32xf32&gt;</span> —— 说明 "a" 已经被切过了。'; });
    tl.at(3400, () => {
      why.innerHTML = '<span class="mono">manual_axes={"a"}</span> 声明"轴 a 由我手动管理"，所以区域内不得再出现 <span class="mono">"a"</span> 的分片。';
      msg.innerHTML = '注意：区域内<b>只能用自由轴</b>（这里是 "b"），manual 轴被冻结。';
    });
    tl.at(6400, () => {
      msg.innerHTML = '正确写法：区域内用 <span class="mono">[{"b"}, {}]</span> 之类的自由轴分片，或干脆全复制。';
      why.innerHTML = '这条约束<b>依赖外层上下文</b> —— 单独看 <span class="mono">&lt;@mesh, [{"a"}, {}]&gt;</span> 完全合法。';
    });
    tl.at(9600, () => { msg.innerHTML = '这也解释了 L1-07 里 manual_computation 为何是"局部代码 + 手写通信"。'; });
    tl.at(11800, () => { msg.innerHTML = '七条不变量讲完。下一课 L1-04 转向<b>边分片</b>。'; });
  }
},

/* ------------------------------------------------------------ 9 练习 */
{
  kicker: 'L1-03 · 练习',
  title: '练一练：<span class="hl-a">当校验器</span>',
  sub: '三道题分别考：轴唯一性、子轴参数、rank 一致性。',
  caption: '能独立判断这三题，说明你已经把不变量内化了 —— 看到 IR 就能"一眼看出非法"。',
  code: `// 题 1：网格 <["a"=2, "b"=2]>，下面合法吗？
#sdy.sharding<@mesh, [{"a"}, {"b"}], replicated={"a"}>

// 题 2：网格 <["a"=8]>，下面合法吗？
#sdy.sharding<@mesh, [{"a":(1)4, "a":(4)4}]>

// 题 3：网格 <["a"=2]>，下面合法吗？
#sdy.sharding<@mesh, [{"a"}, {}]> : tensor<8x8xf32>`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '网格 <span class="mono">&lt;["a"=2, "b"=2]&gt;</span>，<span class="mono">#sdy.sharding&lt;@mesh, [{"a"}, {"b"}], replicated={"a"}&gt;</span> 合法吗？',
        a: '<b class="badge bad">非法</b> 轴 <span class="mono">"a"</span> 既切了第 0 维，又出现在 <span class="mono">replicated</span> 里。' +
           '报错：<span class="mono">duplicate axis ref: "a"</span>。'
      },
      {
        q: '网格 <span class="mono">&lt;["a"=8]&gt;</span>，<span class="mono">#sdy.sharding&lt;@mesh, [{"a":(1)4, "a":(4)4}]&gt;</span> 合法吗？',
        a: '<b class="badge bad">非法</b> 两个子轴分别覆盖 0–3 与 4–7：既不重叠、也各占一半，' +
           '但它们<b>相邻</b>，合起来正好是整轴，所以必须写成完整轴。' +
           '报错：<span class="mono">two consecutive sub-axes can be merged</span>。' +
           '<br><span class="dim">正确写法：<span class="mono">[{"a"}]</span>。</span>'
      },
      {
        q: '网格 <span class="mono">&lt;["a"=2]&gt;</span>，<span class="mono">#sdy.sharding&lt;@mesh, [{"a"}, {}]&gt; : tensor&lt;8x8xf32&gt;</span> 合法吗？',
        a: '<b class="badge ok">合法</b> 2 个维分片对应 rank 2；轴 <span class="mono">"a"</span> 存在；第 1 维是空的闭维（无优先级）。<br>' +
           '<span class="dim">局部形状：dim0 = 8/2 = 4，dim1 = 8 → <span class="mono">tensor&lt;4x8xf32&gt;</span>。</span>'
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
