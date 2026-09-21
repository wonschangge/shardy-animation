/* ==========================================================================
   L3-10 · misc-import-cleanup
   --------------------------------------------------------------------------
   覆盖：transforms/import/test/remove_size_one_axes.mlir (166 行 / 10 用例)
         transforms/import/test/pre_order_funcs.mlir (19 / 3)
         transforms/import/test/propagate_sharding_from_func_to_call.mlir (151 / 20)
   目标：讲透三个收尾清理类 pass。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 总览 */
{
  kicker: 'L3-10 · 杂项导入清理',
  title: '三个收尾 pass：<span class="hl-a">删无用轴</span>、<span class="hl-a">排函数序</span>、<span class="hl-a">搬分片</span>',
  sub: 'L3 层的收尾工作 —— 三个互不相关的小 pass，各自解决一个具体问题。',
  caption: '它们都在导入期运行，为后面的传播与导出<b>减少噪声</b>。',
  code: `// ① 删掉大小为 1 的轴（-sdy-remove-size-one-axes）
//   大小为 1 的轴上只有 1 台设备
//   -> 沿它分片不会真的切开任何东西
//   -> 保留只会让分片变长、可能产生无意义通信
//   166 行 / 10 个用例

// ② 按调用序重排函数（-sdy-pre-order-funcs）
//   输入顺序 func2, func1, main（被调者在前）
//   -> 输出 main, func1, func2（调用者在前，前序遍历）
//   19 行 / 3 个用例（本层最小的文件）

// ③ 函数结果分片 -> 调用点（-sdy-propagate-sharding-from-func-to-call）
//   因为 L3-06 的 import-func-calls 只认【调用点】的分片
//   -> 函数结果上写的分片必须先搬到调用点，否则会丢
//   151 行 / 20 个用例`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 删 size-1 轴', c: '#4ade80', n: '10 用例',
        d: '网格定义不变<br>只改<b>分片</b>里对 size-1 轴的引用' },
      { t: '② 排函数序', c: '#38bdf8', n: '3 用例',
        d: '按<b>前序遍历</b>重排<br>调用者在被调者之前' },
      { t: '③ 搬分片', c: '#fbbf24', n: '20 用例',
        d: '函数结果的分片<br><b>抄到调用点</b><br>（不覆盖已有的）' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>关键词</b>：<b>网格定义不变</b> —— 只改引用它的分片。',
        '<b>为什么要排序</b>：让后续 pass 能按顺序<b>单遍处理</b>，不必来回跳转。',
        '<b>③ 与 L3-06 强相关</b>：L3-06 的 NOTE 说"忽略函数上的分片"，这个 pass 就是补救。',
      ][i];
    }));
    tl.at(12200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>共同点</b>：都是"<b>消除噪声</b>"—— 让后面的 pass 面对更干净、更规整的 IR。';
    });
  }
},

/* ------------------------------------------------ 2 为什么删 size-1 */
{
  kicker: 'L3-10 · 杂项导入清理',
  title: '★ 为什么删<span class="hl-a">大小为 1</span>的轴',
  sub: '大小为 1 的轴意味着"这条轴上只有 1 台设备" —— 沿它分片**不会真的切开任何东西**。',
  caption: '这是本课最值得理解的一条<b>设计理由</b>。',
  code: `// 网格：@mesh1 = <["a"=1, "b"=2, "c"=1, "d"=4, "e"=1]>
//                              ^^^^^^^^        ^^^^^^^^        ^^^^^^^^
//                              a=1            c=1            e=1

// 分片：%arg0 = [{"a", "b"}, {"c", ?}]
//                 ^^^^ "a" 大小 1，切了等于没切
//                              ^^^^ "c" 同样
// 清理后：      [{"b"}, {?}]
//                 ^^^^ 只留下真正起作用的 "b"

// 保留 size-1 轴的代价：
//   ① 分片表达式变长、难读
//   ② 可能让传播/导出产生【无意义的通信】
//      例如对一个宽度为 1 的维度做 all-gather
//   ③ 干扰"两个分片是否等价"的判断
//      [{"a","b"}] 与 [{"b"}] 实际等价，但要额外推理才知道

// 注意：网格定义本身【不变】
sdy.mesh @mesh1 = <["a"=1, "b"=2, "c"=1, "d"=4, "e"=1], device_ids=[…]>
// 原样保留 —— 改动只在【引用它的分片】上`,
  duration: 15000,
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
        <div class="small mono faint">清理前</div>
        <div class="row" style="gap:5px">
          <div class="chip c3" style="padding:6px 11px;font-size:12px">"a"(1)</div>
          <div class="chip c0" style="padding:6px 11px;font-size:12px">"b"(2)</div>
        </div>
        <div class="small faint">"a" 大小 1 —— 切了等于没切</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<span class="mono">"a"</span> 大小为 <b>1</b> —— 这条轴上只有 1 台设备。';
    });
    tl.at(4400, () => {
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹', style: 'font-size:26px' }));
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ok)">清理后</div>
        <div class="row" style="gap:5px">
          <div class="chip c0" style="padding:6px 11px;font-size:12px">"b"(2)</div>
        </div>
        <div class="small faint">只留下真正起作用的轴</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>删掉</b> <span class="mono">"a"</span> —— 分片表达式变短，语义<b>完全不变</b>。';
    });
    tl.at(8600, () => {
      msg.innerHTML = '<b>代价 ①</b>：分片变长、难读。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>代价 ②</b>：可能让传播/导出产生<b>无意义的通信</b> —— 比如对一个宽度 1 的维度做 all-gather。';
    });
    tl.at(13000, () => {
      msg.innerHTML = '<b>代价 ③</b>：干扰"两个分片是否等价"的判断 —— <span class="mono">[{"a","b"}]</span> 与 <span class="mono">[{"b"}]</span> 实际等价，但要额外推理。';
    });
    tl.at(15200, () => {
      msg.innerHTML = '<b>注意</b>：网格定义本身<b>不变</b> —— 改动只在引用它的分片上。';
    });
  }
},

/* ------------------------------------------------ 3 四种位置 */
{
  kicker: 'L3-10 · 杂项导入清理',
  title: '四种位置都会被清理',
  sub: '维分片、`replicated`、`unreduced`、算子上的分片 —— 只要引用了 size-1 轴就会被删。',
  caption: '注意 <span class="mono">@mesh2</span> 上的 <span class="mono">%arg2</span> <b>完全没变</b> —— 规则是逐轴判断，不是整体重写。',
  code: `// @mesh1 的 a/c/e 大小为 1；@mesh2 无 size-1 轴

// ① 函数参数的维分片
%arg0: [{"a", "b"}, {"c", ?}]        ->  [{"b"}, {?}]
//       ^^^^        ^^^^ 删掉

// ② replicated 里的 size-1 轴
%arg1: [{"d", "e", ?}, {}], replicated={"b", "c"}
    -> [{"d", ?}, {}],      replicated={"b"}
//                             ^^^ "c" 被删

// ③ unreduced 里的 size-1 轴
结果1: [{"e"}, {"c", ?}], unreduced={"a", "d"}
    -> [{}, {?}],        unreduced={"d"}
结果2: [{"a","b","c"}, {}], unreduced={"e"}
    -> [{"b"}, {}]      // unreduced 整个消失（清空后不再输出）

// ④ 算子上的分片
%0: [{"d", ?}, {"e", ?}]             ->  [{"d", ?}, {?}]
%2: [{"c"}, {}], replicated={"d"}    ->  [{}, {}], replicated={"d"}

// ⑤ 对照：@mesh2 上没有 size-1 轴 -> 完全不变
%arg2: [{"a"}, {"b"}]                ->  [{"a"}, {"b"}]`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 维分片', c: '#38bdf8', d: '最常见的<br>直接删掉引用' },
      { t: '② replicated', c: '#4ade80', d: '复制集合里的<br>size-1 轴也删' },
      { t: '③ unreduced', c: '#fbbf24', d: '未归约集合<br>清空后属性消失' },
      { t: '④ 算子分片', c: '#c084fc', d: '算子上的分片<br>同样处理' },
      { t: '⑤ 无 size-1', c: '#94a3b8', d: '<b>完全不变</b><br>逐轴判断' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:146px;opacity:.33;transition:.35s;border-color:' + x.c + '55;padding:9px' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:11px">${x.t}</div>
        <div class="card-d" style="font-size:10.5px;line-height:1.45">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<span class="mono">"a"</span> 与 <span class="mono">"c"</span> 都大小为 1 → 都删。',
        '<b>不只是维分片</b>：<span class="mono">replicated</span> 里的 size-1 轴同样没意义 —— 复制一个只有 1 台设备的轴等于没复制。',
        '<b>细节</b>：清空后属性<b>不再输出</b>（结果 2 的 <span class="mono">unreduced</span> 整个消失了）。',
        '算子级分片与函数签名上的分片<b>处理规则一致</b>。',
        '<b>关键</b>：<span class="mono">@mesh2</span> 上没有 size-1 轴，所以那一项<b>一个字符都没变</b>。',
      ][i];
    }));
    tl.at(15400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>规则本质</b>：<b>逐轴</b>检查"这个轴的大小是不是 1"，是就删。';
    });
  }
},

/* ------------------------------------------------ 4 pre_order_funcs */
{
  kicker: 'L3-10 · 杂项导入清理',
  title: '<span class="mono hl-a">pre-order-funcs</span>：按调用序重排函数',
  sub: '把"被调者在前"的书写顺序，改成"**调用者在前**"的前序遍历顺序。',
  caption: '这是编译器里常见的<b>调用图线性化</b>手法 —— 让后续 pass 能单遍处理。',
  code: `// 输入顺序（被调者在前）
func.func @func2() { return }
func.func @func1() { func.call @func2() ... return }
func.func @main()  { func.call @func1() ... return }

// 输出顺序（调用者在前）
// CHECK: func.func @main
// CHECK: func.func @func1
// CHECK: func.func @func2

// 读法：从入口 @main 出发做【前序遍历】
//   main  -> 它调用的 func1
//         -> func1 调用的 func2
//   访问顺序：main, func1, func2

// 为什么要重排：
//   后续 pass 常需要"处理一个函数时，它调用/被调用的函数已就绪"
//   线性化之后可以【按顺序单遍处理】，不必来回跳转
//   也便于做基于调用图的批量分析`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center' });
      c.innerHTML = `
        <div class="small mono faint">输入顺序</div>
        <div class="chip mut" style="padding:5px 10px;font-size:11.5px">func2</div>
        <div class="chip mut" style="padding:5px 10px;font-size:11.5px">func1</div>
        <div class="chip mut" style="padding:5px 10px;font-size:11.5px">main</div>
        <div class="small faint">被调者在前</div>`;
      demo.appendChild(c);
      msg.innerHTML = '源文件的书写顺序是 <span class="mono">func2, func1, main</span> —— <b>被调者在前</b>（C 语言的习惯）。';
    });
    tl.at(4400, () => {
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹', style: 'font-size:26px' }));
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ok)">输出顺序</div>
        <div class="chip c4" style="padding:5px 10px;font-size:11.5px">main</div>
        <div class="chip c4" style="padding:5px 10px;font-size:11.5px">func1</div>
        <div class="chip c4" style="padding:5px 10px;font-size:11.5px">func2</div>
        <div class="small faint">调用者在前（前序遍历）</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>重排</b>为从入口出发的<b>前序遍历</b>：<span class="mono">main → func1 → func2</span>。';
    });
    tl.at(8600, () => {
      msg.innerHTML = '<b>为什么要重排</b>：后续 pass 常需要"处理一个函数时，调用关系已就绪"。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>收益</b>：可以<b>按顺序单遍处理</b>，不必来回跳转 —— 这是调用图线性化的常见手法。';
    });
  }
},

/* ------------------------------------------------ 5 搬分片 */
{
  kicker: 'L3-10 · 杂项导入清理',
  title: '★ 把函数结果的分片<span class="hl-a">搬到调用点</span>',
  sub: 'L3-06 的 `import-func-calls` **只认调用点的分片**。如果函数结果上写了分片却不管，就会**丢失**。',
  caption: '这是三个 pass 里与其它课<b>关联最强</b>的一个。',
  code: `// 【情形 ①】调用点没有分片 -> 从函数抄过来
%0 = call @foo(%arg0) : ...              // 调用点无分片
func.func private @foo(...) -> (tensor<8x2xi32>
    {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) { ... }
// 输出：
%0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<
                        [<@mesh, [{"x"}, {"y"}]>]>} : ...
//                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 抄过来了

// 【情形 ②】调用点已有分片 -> 不覆盖
%0 = call @foo(%arg0) {sdy.sharding = ...[{"y"}, {"x"}]...} : ...
func.func private @foo(...) -> (... [{"x"}, {"y"}] ...)
// 输出：保持调用点的 [{"y"}, {"x"}]
// 用例名：do_not_overwrite_call_sharding

// 【情形 ③】两者都没有 -> 什么都不加

// 【与 L3-06 的衔接】
// L3-06 的 NOTE：we ignore any arg/result shardings on the function
//   -> out_shardings 只来自调用点
//   -> 函数结果上的分片若不先搬走就会【丢】
// 顺序：-sdy-propagate-sharding-from-func-to-call -> -sdy-import-func-calls`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 调用点无分片', c: '#4ade80', r: '抄过来',
        d: '从函数结果<br>复制到调用点' },
      { t: '② 调用点有分片', c: '#fbbf24', r: '不覆盖',
        d: '保持调用点的<br>（用例名点明了）' },
      { t: '③ 都没有', c: '#94a3b8', r: '不加',
        d: '什么都不做' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="mono" style="font-size:13px;color:#bdf7ec;margin:5px 0">${U.esc(x.r)}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '函数结果上写了分片，但调用点没有 —— 这时才需要搬。',
        '<b>为什么以调用点为准</b>：同一函数可能被多个调用点调用，各调用点可要求不同分片（L3-06 讲过）。',
        '没有信息可搬，也不需要凭空造一个。',
      ][i];
    }));
    tl.at(12200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>与 L3-06 的衔接</b>：这个 pass 是 <span class="mono">-sdy-import-func-calls</span> 的<b>前置补救</b>。';
    });
    tl.at(14400, () => {
      msg.innerHTML = '<b>顺序</b>：<span class="mono">-sdy-propagate-sharding-from-func-to-call</span> → <span class="mono">-sdy-import-func-calls</span>。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L3-10 · 练习',
  title: '练一练：<span class="hl-a">三个 pass 各解决什么</span>',
  sub: '三道题分别考：size-1 轴、函数排序、分片搬迁。',
  caption: '一句话总结：<b>删无用轴、排函数序、搬分片</b>。',
  code: `// 题 1：@mesh = <["a"=1, "b"=2]>，分片 [{"a","b"}]，
//       清理后是什么？网格定义会变吗？

// 题 2：函数顺序 func2, func1, main（main->func1->func2），
//       pre-order-funcs 后是什么顺序？

// 题 3：函数结果上有分片、调用点没有，会怎样？
//       如果调用点也有（且不同）呢？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">@mesh = &lt;["a"=1, "b"=2]&gt;</span>，分片 <span class="mono">[{"a","b"}]</span>，清理后是什么？网格定义会变吗？',
        a: '清理后是 <span class="mono">[{"b"}]</span> —— <span class="mono">"a"</span> 大小为 1，被删掉。' +
           '<br><b>网格定义不变</b>：<span class="mono">@mesh</span> 仍原样保留 <span class="mono">&lt;["a"=1, "b"=2]&gt;</span>。' +
           '<br><span class="dim">改动只在<b>引用它的分片</b>上 —— 因为大小为 1 的轴上只有 1 台设备，切了等于没切。</span>'
      },
      {
        q: '函数顺序 <span class="mono">func2, func1, main</span>（<span class="mono">main→func1→func2</span>），<span class="mono">pre-order-funcs</span> 后是什么顺序？',
        a: '<span class="mono">main, func1, func2</span> —— 从入口出发的<b>前序遍历</b>（调用者在前）。' +
           '<br><b>为什么重排</b>：让后续 pass 能<b>按顺序单遍处理</b>，不必来回跳转。' +
           '<br><span class="dim">这是编译器里常见的<b>调用图线性化</b>手法。</span>'
      },
      {
        q: '函数结果上有分片、调用点没有，会怎样？如果调用点也有（且不同）呢？',
        a: '<b>调用点没有 → 抄过来</b>：把函数结果的分片复制到调用点上。' +
           '<br><b>调用点也有 → 不覆盖</b>：保持调用点的（用例名 <span class="mono">do_not_overwrite_call_sharding</span>）。' +
           '<br><b>为什么需要这个 pass</b>：L3-06 的 NOTE 说 <span class="mono">we ignore any arg/result shardings on the function</span> —— <span class="mono">out_shardings</span> 只来自调用点。若不先搬走，函数结果上的分片就会<b>丢失</b>。' +
           '<br><span class="dim">顺序：<span class="mono">-sdy-propagate-sharding-from-func-to-call</span> → <span class="mono">-sdy-import-func-calls</span>。</span>'
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
