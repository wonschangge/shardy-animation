/* ==========================================================================
   L1-08 · named-computation-and-dataflow
   --------------------------------------------------------------------------
   覆盖：ir/test/named_computation_parse_print.mlir (49)
         ir/test/named_computation_verification.mlir (89)
         ir/test/data_flow_edge_verification.mlir (42)
         ir/test/func_data_flow_edge_verification.mlir (58)
   目标：讲透"传播如何穿过函数调用"，以及数据流边的 source/target/owner 模型。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------------ 1 动机 */
{
  kicker: 'L1-08 · 命名计算与数据流边',
  title: '传播怎么穿过<span class="hl-a">函数调用</span>？',
  sub: '如果一段计算被抽成了函数，传播就需要跨越调用边界。SDY 的答案是 <span class="mono">sdy.named_computation</span>：把一个函数体<b>内联</b>进一个带名字的区域。',
  caption: '导入阶段（L3-06）会把 <span class="mono">func.call</span> 转换成它。这样传播就"看不见"调用边界了 —— 一切都在同一个区域里。',
  code: `// 问题：函数调用把数据流切断了
func.func @callee(%x: tensor<8x2xi32>) -> tensor<8x2xi32> { ... }

func.func @main(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  %0 = func.call @callee(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
// 传播要跨过 call，就得理解函数调用语义

// 解决：把函数体内联进命名区域
%0 = sdy.named_computation<"foo">(%arg0) (%arg1: tensor<8x2xi32>) {
  sdy.return %arg1 : tensor<8x2xi32>
} : (tensor<8x2xi32>) -> tensor<8x2xi32>
// 现在区域内是普通算子，传播照常穿过`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:16px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:20px;justify-content:center;align-items:stretch">
        <div class="card" style="width:290px;border-color:rgba(251,113,133,.45)">
          <div class="card-t" style="color:var(--bad)">用 func.call</div>
          <div class="card-d">调用边界切断了数据流。<br>
            传播需要额外的跨函数机制。<br>
            <span class="dim small">调用图还可能被展平/还原（L4-13）</span></div>
        </div>
        <div class="card" style="width:290px;border-color:rgba(94,234,212,.5)">
          <div class="card-t" style="color:var(--accent)">用 named_computation</div>
          <div class="card-d">函数体<b>内联</b>进命名区域。<br>
            传播在区域内部照常工作。<br>
            <span class="dim small">导出时再 outline 回函数（L4-12）</span></div>
        </div>
      </div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:740px"></div>`;
    root.appendChild(wrap);
    const msg = wrap.querySelector('#msg');
    tl.at(900, () => { msg.innerHTML = '名字 <span class="mono">&lt;"foo"&gt;</span> 只是为了可读性与 outline 时复用；<b>对传播没有语义影响</b>。'; });
    tl.at(3600, () => { msg.innerHTML = '区域是 <span class="mono">IsolatedFromAbove</span>：只能通过块参数与 return 交互 —— 与真正的函数体一致。'; });
    tl.at(6400, () => { msg.innerHTML = '它同时也是一个<b>数据流算子</b>：传播视作"源与目标同分片"的恒等桥；只是这次的"目标"在另一个 level。'; });
    tl.at(9400, () => { msg.innerHTML = '区分三种边：<span class="mono">named_computation</span>（区域）、<span class="mono">data_flow_edge</span>（算子级）、<span class="mono">func_data_flow_edge</span>（函数级）。'; });
  }
},

/* ------------------------------------------------------------ 2 语法 */
{
  kicker: 'L1-08 · 命名计算与数据流边',
  title: '<span class="mono hl-a">sdy.named_computation</span> 的四种形态',
  sub: '它可以带 <span class="mono">in_shardings</span> / <span class="mono">out_shardings</span>（与 manual_computation 类似，但<b>不</b>冻结任何轴）。',
  caption: '与 manual_computation 的关键区别：named_computation <b>没有</b> <span class="mono">manual_axes</span>，区域内仍然是<b>全局形状</b>，传播照常进行。',
  code: `sdy.mesh @mesh = <["a"=2, "b"=2]>

// ① 单入单出：类型必须一一对应
%0 = sdy.named_computation<"foo">(%arg0) (%arg1: tensor<8x2xi32>) {
  sdy.return %arg1 : tensor<8x2xi32>
} : (tensor<8x2xi32>) -> tensor<8x2xi32>

// ② 多入多出
%0:2 = sdy.named_computation<"named_computation">(%arg0, %arg1)
       (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>) {
  sdy.return %arg2, %arg3 : tensor<8x2xi32>, tensor<4x2xi32>
} : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)

// ③ 带 in/out 分片：输入输出可以不同
%0 = sdy.named_computation<"foo">(%arg0)
     in_shardings=[<@mesh, [{"b"}, {}]>]
     out_shardings=[<@mesh, [{"a"}, {}]>] (%arg1: tensor<8x2xi32>) { ... }

// ④ token：分片必须是 rank 0
in_shardings=[<@mesh, [{"b"}, {}]>, <@mesh, []>]`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%;align-items:center' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:760px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 单入单出', d: '块参数类型 = 操作数类型；<br>return 类型 = 结果类型。', c: '#38bdf8' },
      { t: '② 多入多出', d: '个数必须一一对应。<br>与函数签名同构。', c: '#c084fc' },
      { t: '③ 带 in/out 分片', d: '输入输出分片<b>可以不同</b>；<br>相当于给传播一个提示。', c: '#fbbf24' },
      { t: '④ token', d: '非 shaped 类型 → 分片必须 <span class="mono">[]</span>（rank 0）。', c: '#4ade80' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:182px;opacity:.32;transition:.3s;border-color:' + x.c + '55;padding:9px' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:11.5px">${x.t}</div>
        <div class="card-d" style="font-size:11px;line-height:1.5">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.32');
      msg.innerHTML = [
        '名字可以让多个 named_computation 重名（会被 outline 到不同函数）。',
        '块参数与 return 一起决定区域的"接口"，校验器逐一核对。',
        '这里的分片是<b>建议</b>：传入时按 in_sharding，传出时按 out_sharding，中间由传播决定。',
        '与 L1-02 的 token 规则一致：无形状的类型只能 rank 0 且无 replicated/unreduced。',
      ][i];
    }));
    tl.at(13000, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>关键区别</b>：named_computation 不冻结任何轴，区域内是<b>全局形状</b>。';
    });
  }
},

/* -------------------------------------------------- 3 数据流边模型 */
{
  kicker: 'L1-08 · 命名计算与数据流边',
  title: '<span class="mono hl-a">data_flow_edge</span>：一条边上的值必须同分片',
  sub: '一条数据流边由三部分组成：<b>sources</b>（源）、<b>targets</b>（目标）、<b>owner</b>（其中一个目标，用来承载分片属性）。',
  caption: '传播把它当作"源与目标是同一个张量"（恒等规则）来处理 —— 所以它不需要 sharding rule。',
  code: `// 一条边 = 一组 source + 一组 target
//   owner 是其中一个 target，分片属性挂在它身上

// 例：stablehlo.while 有 n 条边，第 i 条：
//   sources: x_i（初始值）、return_value_i（循环体返回值）
//   targets: y_i（结果）、pred_arg_i、body_arg_i

// 例：sdy.data_flow_edge 显式表示这类边
%0 = sdy.data_flow_edge %arg0 sharding=<@mesh, [{}, {"a"}]>
     : tensor<8xf32>

// 传播规则：边上所有值分片一致
//   从任一 source 出发 -> 推到所有 targets
//   从任一 target 出发 -> 推到所有 sources`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:16px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:center" id="model"></div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:750px"></div>`;
    root.appendChild(wrap);
    const model = wrap.querySelector('#model'), msg = wrap.querySelector('#msg');

    const col = (title, items, cls, color) => {
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center' });
      c.innerHTML = `<div class="small mono" style="color:${color}">${title}</div>
        <div class="col" style="gap:5px" data-d></div>`;
      const h = c.querySelector('[data-d]');
      items.forEach(t => {
        const e = U.el('div', { class: 'chip ' + cls, style: 'padding:5px 11px;font-size:12.5px' });
        e.textContent = t; h.appendChild(e);
      });
      model.appendChild(c); return c;
    };

    tl.at(700, () => {
      col('sources（源）', ['x_i', 'return_value_i'], 'c0', 'var(--ax0)');
      model.appendChild(U.el('div', { class: 'arrow anim', html: '⟺', style: 'font-size:26px' }));
      col('targets（目标）', ['y_i', 'pred_arg_i', 'body_arg_i'], 'c1', 'var(--ax1)');
      msg.innerHTML = '一条边把两组值绑在一起：<b>任一处拿到分片，其它全部同步</b>。';
    });
    tl.at(4200, () => {
      msg.innerHTML = '<b>owner</b> 是 targets 里指定的一个（例子里是 <span class="mono">y_i</span>），分片属性存在它身上，保证表示唯一。';
      const chips = model.querySelectorAll('.chip.c1');
      if (chips[0]) chips[0].classList.add('pulse');
    });
    tl.at(7600, () => {
      msg.innerHTML = '为什么需要 owner：一条边有多个 target，若每处都存分片就会有<b>冗余与不一致</b>。';
    });
    tl.at(10600, () => {
      msg.innerHTML = '传播时把它当作<b>恒等规则</b>的普通算子：源→目标（前向）、目标→源（反向）。';
    });
    tl.at(13200, () => {
      msg.innerHTML = '所以数据流算子<b>不需要 sharding rule</b> —— 它没有"维度映射"，只有"同分片"这一个约束。';
    });
  }
},

/* ------------------------------------------------ 4 三条约束 */
{
  kicker: 'L1-08 · 命名计算与数据流边',
  title: '数据流边的<span class="hl-a">三条约束</span>',
  sub: '这些约束保证"边"是<b>良定义</b>的：形状可算、归属唯一、不会把分片传播搞乱。',
  caption: '注意第 4 条（输入不能由 SDY 算子定义）—— 它防止"SDY 算子 → 边 → SDY 算子"形成回环，让传播的实现无法确定处理顺序。',
  code: `// ① 结果必须是静态形状
%0 = sdy.data_flow_edge %arg0 : tensor<?x?xf32>            // ✗
//   expected sdy.data_flow_edge to have a static-shaped result

// ② 输入只能有一个使用者
func.func @f(%arg0: tensor<32x96xf32>) -> (tensor<32x96xf32>, tensor<32x96xf32>) {
  %0 = sdy.data_flow_edge %arg0 : tensor<32x96xf32>          // ✗ %arg0 被用了两次
  return %arg0, %0 : tensor<32x96xf32>, tensor<32x96xf32>
}
//   expected input of sdy.data_flow_edge to have a single user

// ③ 输入不能由 SDY 方言的算子定义
%0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {}]> : tensor<32x96xf32>
%1 = sdy.data_flow_edge %0 : tensor<32x96xf32>               // ✗
//   expected input of sdy.data_flow_edge to not be defined by an SdyDialect op

// ④ 分片本身的校验与普通张量分片完全一致
%0 = sdy.data_flow_edge %arg0 sharding=<@mesh, [{}, {"a"}]> : tensor<8xf32>
//   sharding doesn't match tensor rank: 2 != 1`,
  duration: 17000,
  build(root, tl) {
    const wrap = W.errLayout(root);
    W.errScenes(wrap, tl, [
      {
        ir: '%0 = sdy.data_flow_edge %arg0 : tensor<?x?xf32>',
        err: 'expected sdy.data_flow_edge to have a static-shaped result',
        why: '边的分片需要能算出局部形状，动态形状做不到。'
      },
      {
        ir: '%0 = sdy.data_flow_edge %arg0 : tensor<32x96xf32>',
        err: 'expected input of sdy.data_flow_edge to have a single user',
        why: '该用例里 <span class="mono">%arg0</span> 同时被 <span class="mono">return</span> 与边使用 —— 边必须是它的<b>唯一</b>使用者。'
      },
      {
        ir: '%1 = sdy.data_flow_edge %0 : tensor<32x96xf32>',
        err: 'expected input of sdy.data_flow_edge to not be defined by an SdyDialect op',
        why: '<span class="mono">%0</span> 是 <span class="mono">sdy.sharding_constraint</span> 的结果。SDY 算子已经能携带分片，再加一层边会形成<b>回环</b>。'
      },
      {
        ir: '%0 = sdy.data_flow_edge %arg0 sharding=<@mesh, [{}, {"a"}]> : tensor<8xf32>',
        err: "sharding doesn't match tensor rank: 2 != 1",
        why: '测试注释明确说明：<b>边上的分片校验与普通张量分片完全相同</b>，所以不必重复测其它失败类型。'
      },
    ], {
      stepMs: 3200,
      finalIr: '// 三条约束的共同目的：让"边"良定义\n//   静态形状 -> 能算局部形状\n//   单一使用者 -> 归属唯一，不歧义\n//   非 SDY 定义 -> 不形成处理顺序回环',
      finalErr: 'static-shaped result / single user / not be defined by an SdyDialect op',
      finalWhy: '第 4 条尤其重要：它保证了传播实现可以<b>确定</b>处理的先后关系。'
    });
  }
},

/* ------------------------------------------------ 5 函数级的边 */
{
  kicker: 'L1-08 · 命名计算与数据流边',
  title: '<span class="mono hl-a">func_data_flow_edge</span>：函数边界上的边',
  sub: '函数参数与调用结果也需要边。它是 <span class="mono">data_flow_edge</span> 的"函数版"：桥接<b>调用点的实参</b>与<b>函数体的形参</b>。',
  caption: '规则更少（只有两条）：函数级没有"宿主算子"可以查 operand 索引范围，这与 L1-04 里"func 边允许越界 operand"是同一件事。',
  code: `// 当操作数是块参数（函数形参）时：
//   桥接"调用点的实参" -> "函数体内的使用者"
// 当操作数是算子结果（call 的返回值）时：
//   桥接"被调用函数的 return" -> "调用结果的使用者"

// ① 结果必须是静态形状
%0 = sdy.func_data_flow_edge %arg0 : tensor<?xf32>     // ✗
//   expected sdy.func_data_flow_edge to have a static-shaped result

// ② 操作数只能有一个使用者
%0 = sdy.func_data_flow_edge %arg0 : tensor<8xf32>     // ✗ 若 %arg0 还有别的使用者
//   expected operand of sdy.func_data_flow_edge to have a single user`,
  duration: 14000,
  build(root, tl) {
    const wrap = W.errLayout(root);
    W.errScenes(wrap, tl, [
      {
        ir: '%0 = sdy.func_data_flow_edge %arg0 : tensor<?xf32>',
        err: 'expected sdy.func_data_flow_edge to have a static-shaped result',
        why: '与 <span class="mono">data_flow_edge</span> 同一条理由：静态形状才能算局部形状。'
      },
      {
        ir: '%0 = sdy.func_data_flow_edge %arg0 : tensor<8xf32>',
        err: 'expected operand of sdy.func_data_flow_edge to have a single user',
        why: '这里的 <span class="mono">%arg0</span> 还有其它使用者 —— 边必须是唯一使用者，否则归属不唯一。'
      },
    ], {
      stepMs: 3200,
      finalIr: '// 对比：func 版只有【两条】约束\n//   没有"输入不能由 SDY 算子定义"这一条？\n//   有的 —— 但函数级边不做 operand 索引范围检查\n//   （见 L1-04 的反例：func 边允许越界 operand）',
      finalErr: 'static-shaped result / single user',
      finalWhy: '规则更少不等于更松：<b>函数级边没有宿主算子</b>，所以无法做索引范围检查，只能省掉。'
    });
  }
},

/* ------------------------------------------------ 6 校验错误汇总 */
{
  kicker: 'L1-08 · 校验',
  title: '三类校验错误<span class="hl-a">速查</span>',
  sub: '把四个文件的校验错误并排看，能发现它们遵循同一套模式：<b>结构对齐</b> + <b>分片合法</b> + <b>边良定义</b>。',
  caption: '这些规则在 L2-08（数据流传播）里会用到 —— 那时看的是"传播如何沿这些边流动"。',
  code: `// 【named_computation】结构对齐（与函数签名同构）
//   块参数类型 != 操作数类型
//   number of block arguments must match the number of operands: 2 != 1
//   failed to infer returned types
//   inferred type(s) ... are incompatible with return type(s) of operation

// 【named_computation】分片合法（复用 L1-02/L1-03 规则）
//   op in_shardings shardings don't match number of values: 2 shardings vs 1 values
//   op out_shardings shardings don't match number of values
//   op in_shardings - sharding doesn't match tensor rank: 3 != 2
//   op in_shardings - unknown mesh: @unknown_mesh

// 【data_flow_edge】边良定义（3 条）
//   expected sdy.data_flow_edge to have a static-shaped result
//   expected input of sdy.data_flow_edge to have a single user
//   expected input of sdy.data_flow_edge to not be defined by an SdyDialect op

// 【func_data_flow_edge】边良定义（2 条）
//   expected sdy.func_data_flow_edge to have a static-shaped result
//   expected operand of sdy.func_data_flow_edge to have a single user`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%;align-items:center' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="groups"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:760px"></div>`;
    root.appendChild(wrap);
    const groups = [
      { t: '结构对齐', n: '4 条', c: '#38bdf8',
        d: 'named_computation：块参数 ↔ 操作数、return ↔ 结果，个数与类型都必须一致。' },
      { t: '分片合法', n: '4 条', c: '#c084fc',
        d: 'named_computation 的 in/out 分片：项数、rank、网格 —— 完全是 L1-02/L1-03 的老规则。' },
      { t: '边良定义', n: '5 条', c: '#fbbf24',
        d: '两类边：静态形状、单一使用者（data_flow_edge 另有"非 SDY 定义"）。' },
    ];
    const host = wrap.querySelector('#groups');
    const els = groups.map(g => {
      const e = U.el('div', { class: 'card', style: 'width:244px;opacity:.34;transition:.3s;border-color:' + g.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${g.c}">${g.t} <span class="faint small">${g.n}</span></div>
        <div class="card-d" style="font-size:12px">${g.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    groups.forEach((g, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.34');
      msg.innerHTML = [
        '本质：<b>区域接口必须与算子签名同构</b>。这与 manual_computation 的"个数匹配"是同一类检查。',
        '没有新规则 —— 分片属性的校验逻辑在所有算子上<b>完全复用</b>。',
        '这三条是边独有的：它要保证"这条边"能被确定地识别与处理。',
      ][i];
    }));
    tl.at(11600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = 'L1 到这里已经覆盖了<b>所有 SDY 构件</b>；剩下两课讲约束类算子与 reshard/constant。';
    });
  }
},

/* ------------------------------------------------------------ 7 练习 */
{
  kicker: 'L1-08 · 练习',
  title: '练一练：<span class="hl-a">判断边是否良定义</span>',
  sub: '三道题分别考 named_computation 与 manual_computation 的区别、边的约束、owner 的作用。',
  caption: '能答对这三题，L2-08「数据流传播」就有了完整前置。',
  code: `// 题 1：named_computation 与 manual_computation 最关键的区别？

// 题 2：下面合法吗？
func.func @f(%arg0: tensor<32x96xf32>) -> (tensor<32x96xf32>, tensor<32x96xf32>) {
  %0 = sdy.data_flow_edge %arg0 : tensor<32x96xf32>
  return %arg0, %0 : tensor<32x96xf32>, tensor<32x96xf32>
}

// 题 3：为什么一条边需要 owner？`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">sdy.named_computation</span> 与 <span class="mono">sdy.manual_computation</span> 最关键的区别是什么？',
        a: '<b>是否冻结轴</b>。<br>' +
           '<span class="mono">manual_computation</span> 有 <span class="mono">manual_axes</span>：指定轴被冻结，区域内是<b>局部形状</b>，通信自己写。<br>' +
           '<span class="mono">named_computation</span> <b>没有</b> <span class="mono">manual_axes</span>：区域内仍是<b>全局形状</b>，传播照常工作。<br>' +
           '<span class="dim">前者是"我自己来"，后者只是"给这段计算起个名字并内联函数体"。</span>'
      },
      {
        q: '<span class="mono">%arg0</span> 同时被 <span class="mono">return</span> 和 <span class="mono">sdy.data_flow_edge</span> 使用，合法吗？',
        a: '<b class="badge bad">非法</b> 报错：<span class="mono">expected input of sdy.data_flow_edge to have a single user</span>。' +
           '<br><b>理由</b>：边的输入若有多个使用者，"这个分片归属于谁"就不唯一 —— 边的语义是"这一条数据流上的值同分片"。' +
           '<br><span class="dim">修法：让边成为唯一使用者（把另一个使用者改成用边的结果）。</span>'
      },
      {
        q: '为什么一条数据流边需要指定 <b>owner</b>？',
        a: '因为一条边有<b>多个 target</b>（如 while 的第 i 条边有 <span class="mono">y_i</span>、<span class="mono">pred_arg_i</span>、<span class="mono">body_arg_i</span>）。' +
           '<br>分片属性只能存在一处，否则会<b>冗余且可能不一致</b>。owner 就是"把分片存在哪个 target 上"的约定。' +
           '<br><span class="dim">owner 由用户在实现接口时指定，但必须<b>静态确定</b>。</span>'
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
