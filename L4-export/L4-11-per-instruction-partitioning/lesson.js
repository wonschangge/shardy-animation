/* ==========================================================================
   L4-11 · per-instruction-partitioning
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/ 下 3 个文件
         (per_instruction_partitioning 578 行 / 23 用例
          _range 49 / 2   _subroutine 38 / 2) = 665 行 / 27 用例
   目标：讲透"只对指定指令跑分区器"的机制与三种 filter 语法。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 核心机制 */
{
  kicker: 'L4-11 · 逐指令分区',
  title: '★ 核心机制：只对<span class="hl-a">指定的指令</span>跑分区器',
  sub: '选中的指令被包进 `sdy.manual_computation`，未选中的**保持原样**。',
  caption: '这个 pass 的用途是 <b>bisect</b> —— 二分定位导出流水线的问题。',
  code: `// RUN: sdy_opt %s -split-input-file -sdy-per-instruction-partitioning=\\
//        "filter=dot,constant,reshard,all_gather,all_slice,concatenate,convolution,while,call,if"
//         ^^^^^^ filter 是一个【算子名列表】

// 【输入】dot 与 add 都在
func.func @selective_dot(%lhs: tensor<8x32xf32> {...[{"x"}, {}]>},
                         %rhs: tensor<32x16xf32> {...[{}, {"y"}]>})
    -> (tensor<8x16xf32> {...[{"x"}, {"y"}]>}) {
  %dot = stablehlo.dot %lhs, %rhs {...[{"x"}, {"y"}]...} : ...
  %add = stablehlo.add %dot, %dot {...[{"x"}, {"y"}]...} : ...
  return %add : tensor<8x16xf32>
}

// 【输出】dot 被包，add 不被包
// CHECK: %[[MANUAL]] = sdy.manual_computation(%[[LHS]], %[[RHS]])
// CHECK-SAME:   in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]
// CHECK-SAME:   out_shardings=[<@mesh, [{"x"}, {"y"}]>]
// CHECK-SAME:   manual_axes={"x", "y"} (%arg2: tensor<4x32xf32>, %arg3: tensor<32x8xf32>) {
// CHECK-NEXT:   %[[LOCAL_DOT]] = stablehlo.dot %arg2, %arg3
//                              : (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
//                              ^^^^^^^^^^^^^^^^^^^ 【局部形状】！
// CHECK-NEXT:   sdy.return %[[LOCAL_DOT]] : tensor<4x8xf32>
// CHECK-NEXT: } : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK: %[[ADD]] = stablehlo.add %[[MANUAL]], %[[MANUAL]] {...}
//   ^^^^ add 不在 filter 里 -> 【保持原样】，直接读 %[[MANUAL]]

// 读法：
//   manual_axes={"x","y"} -> 两个轴都【冻结】
//   区域内 block argument 是【局部形状】（全局 8x32 沿 x=2 切 -> 局部 4x32）
//   区域内的 dot 【不带任何分片属性】—— 它已经是"本地算子"`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '选中的指令', c: '#4ade80', d: '包进 <span class="mono">manual_computation</span><br>区域内用<b>局部形状</b><br>不带分片属性' },
      { t: '未选中的指令', c: '#94a3b8', d: '<b>保持原样</b><br>仍然读全局张量<br><span class="dim">完全不受影响</span>' },
      { t: '用途', c: '#38bdf8', d: '<b>bisect</b><br>二分定位<br>导出流水线的问题' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>为什么用 <span class="mono">manual_computation</span></b>：它正是"区域内自己管分片"的机制（L1-07）。',
        '<b>这就是 filter 的价值</b>：可以精确控制"只对谁动手"。',
        '<b>场景</b>：整个流水线在某段 IR 上失败，但不知道是哪个算子的问题。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>与 L4-09 的呼应</b>：HALO 模式也用 <span class="mono">manual_computation</span> 承载 —— 同一个机制，不同的目的。';
    });
  }
},

/* ------------------------------------------------ 2 三种语法 */
{
  kicker: 'L4-11 · 逐指令分区',
  title: '★ 三种 <span class="mono hl-a">filter</span> 语法',
  sub: '按**算子名**、按**指令位置**、按**所在函数** —— 三种选择方式各有用处。',
  caption: '三个文件正好各覆盖一种语法。',
  code: `// 【语法 ①】算子名子串（主文件 578 行 / 23 用例）
// RUN: ...="filter=dot,constant,reshard,all_gather,all_slice,concatenate,convolution,while,call,if"
//   匹配规则：算子名【子串】匹配
//   写 dot 会匹配 stablehlo.dot
//   注意 while / call / if 是【区域算子】—— filter 也能选中它们

// 【语法 ②】指令位置（range 文件 49 行 / 2 用例）
// RUN: ...="filter='selectLow=0, selectHigh=0'"
// RUN: ...="filter='selectHigh=0, selectLow=0'"      <- 顺序相反
//   selectLow=N / selectHigh=N 按【指令位置】选出一个区间
//   selectLow=0, selectHigh=0 -> 只选第 0 条指令
//   两个 RUN 行顺序相反 -> 验证【参数顺序不影响结果】

// 【语法 ③】限制在函数内（subroutine 文件 38 行 / 2 用例）
// RUN: ...="filter=func=subroutine,add"
//   func=<名字> 把选择范围【限制在指定函数内】
//   @subroutine 里的 add  -> 被包
//   @main 里的 add        -> 【不被包】

// 三种语法的适用场景：
//   ① 怀疑某个【算子】有问题 -> filter=dot
//   ② 怀疑某个【位置】的指令有问题 -> selectLow/selectHigh
//   ③ 同名算子在多处出现，只想测【某一个函数】-> func=名字`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 算子名', c: '#4ade80', n: '23 用例',
        d: '<span class="mono">filter=dot,pad</span><br>子串匹配<br><span class="dim">可含区域算子</span>' },
      { t: '② 指令位置', c: '#38bdf8', n: '2 用例',
        d: '<span class="mono">selectLow=0,<br>selectHigh=0</span><br>按<b>位置区间</b>选' },
      { t: '③ 所在函数', c: '#fbbf24', n: '2 用例',
        d: '<span class="mono">func=subroutine,add</span><br>限制在<b>函数内</b><br><span class="dim">同名算子别处不受影响</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>最常用的一种</b>。写 <span class="mono">filter=dot,pad</span> 就能只分区这两个算子。',
        '<b>为什么需要位置选择</b>：同一个算子名可能在一段 IR 里出现多次，位置能精确定位。',
        '<b>最精确的一种</b>：把算子名与函数名组合起来定位。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>验收点</b>：只分区 <span class="mono">dot</span> 与 <span class="mono">pad</span> 的 filter 串是 <span class="mono">filter=dot,pad</span>。';
    });
  }
},

/* ------------------------------------------------ 3 位置选择 */
{
  kicker: 'L4-11 · 逐指令分区',
  title: '位置选择：<span class="hl-a">同一个函数里两个 add</span>',
  sub: '`range` 文件用两个 `add` 证明了 filter 是**按指令位置**而非"按算子名"选择的。',
  caption: '两个 RUN 行的参数顺序<b>相反</b> —— 验证顺序不影响结果。',
  code: `// RUN: ...="filter='selectLow=0, selectHigh=0'"
// RUN: ...="filter='selectHigh=0, selectLow=0'"     <- 顺序相反，结果相同

func.func @selective_add_range(
    %arg0: tensor<4x8xf32> {...<@mesh, [{"x"}, {}]>})
    -> (tensor<4x8xf32> {...<@mesh, [{"x"}, {}]>}) {
  %add1 = stablehlo.add %arg0, %arg0 {...}: tensor<4x8xf32>
  %add2 = stablehlo.add %add1, %add1 {...}: tensor<4x8xf32>
  return %add2 : tensor<4x8xf32>
}

// 输出：只有【第一个】add 被包
// CHECK: %[[ADD1]] = sdy.manual_computation(%[[ARG0]], %[[ARG0]])
// CHECK-SAME:   in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{"x"}, {}]>]
// CHECK-SAME:   out_shardings=[<@mesh, [{"x"}, {}]>]
// CHECK-SAME:   manual_axes={"x"} (%arg1: tensor<2x8xf32>, %arg2: tensor<2x8xf32>) {
// CHECK-NEXT:   %[[LOCAL1]] = stablehlo.add %arg1, %arg2 : tensor<2x8xf32>
// CHECK-NEXT:   sdy.return %[[LOCAL1]] : tensor<2x8xf32>
// CHECK-NEXT: } : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK: %[[ADD2]] = stablehlo.add %[[ADD1]], %[[ADD1]] {...} : tensor<4x8xf32>
//   ^^^^ 第二个 add 【保持原样】

// 读法：
//   两个 add 的【算子名相同】，但只有第 0 条被选中
//   -> 证明 filter 是按【位置】选择的
//   另：%add1 被包后，%add2 读的是 %[[ADD1]]（manual_computation 的结果）`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '第 0 条 add', c: '#4ade80', d: '<b>被选中</b><br>包进 <span class="mono">manual_computation</span><br>区域内是局部形状 <span class="mono">2x8</span>' },
      { t: '第 1 条 add', c: '#94a3b8', d: '<b>未选中</b><br>保持原样<br>读 <span class="mono">%[[ADD1]]</span>' },
      { t: '两个 RUN 行', c: '#38bdf8', d: '参数顺序<b>相反</b><br><span class="mono">selectLow</span> 在前 / <span class="mono">selectHigh</span> 在前<br>→ <b>结果相同</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '第一个 <span class="mono">add</span> 被包 —— 因为 <span class="mono">selectLow=0, selectHigh=0</span> 选中了第 0 条。',
        '第二个 <span class="mono">add</span> 完全不受影响 —— 它读的是被包后的结果。',
        '<b>这个测试设计很细致</b>：用两个 RUN 行锁定"参数顺序无关"这一性质。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>关键结论</b>：两个 <span class="mono">add</span> 算子名相同，但只有第 0 条被选中 → filter 按<b>位置</b>选择。';
    });
  }
},

/* ------------------------------------------------ 4 函数限定 */
{
  kicker: 'L4-11 · 逐指令分区',
  title: '函数限定：<span class="mono hl-a">func=</span> 语法',
  sub: '同名算子在多个函数里出现时，用 `func=<名字>` **精确定位到某一个函数**。',
  caption: '测试注释把这个设计意图写得很清楚（两句英文注释）。',
  code: `// RUN: ...="filter=func=subroutine,add"

// @subroutine 与 @main 里【各有一个 add】
func.func private @subroutine(
    %arg0: tensor<8x16xf32> {...[{"x"}, {}]>},
    %arg1: tensor<8x16xf32> {...[{"x"}, {}]>})
    -> (tensor<8x16xf32> {...[{"x"}, {}]>}) {
  %0 = stablehlo.add %arg0, %arg1 {...} : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// 【@subroutine 里的 add】被包
// The add inside @subroutine is selected and wrapped in sdy.manual_computation.
// CHECK: %[[MANUAL_ADD]] = sdy.manual_computation(%[[ARG0]], %[[ARG1]])
// CHECK-SAME:   in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{"x"}, {}]>]
// CHECK-SAME:   out_shardings=[<@mesh, [{"x"}, {}]>]
// CHECK-SAME:   manual_axes={"x"} (%[[LOCAL_ARG0]]: tensor<4x16xf32>, %[[LOCAL_ARG1]]: tensor<4x16xf32>) {
// CHECK-NEXT:   %[[LOCAL_ADD]] = stablehlo.add %[[LOCAL_ARG0]], %[[LOCAL_ARG1]]
// CHECK-NEXT:   sdy.return %[[LOCAL_ADD]] : tensor<4x16xf32>
// CHECK-NEXT: } : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: return %[[MANUAL_ADD]] : tensor<8x16xf32>

// 【@main 里的 add】不被包
// The add in @main is not partitioned because filter restricts to func=subroutine.
// CHECK-NOT: sdy.manual_computation
// CHECK: %[[ADD]] = stablehlo.add %arg0, %arg1
// CHECK: %[[CALL]] = call @subroutine(%[[ADD]], %arg1)
// CHECK: return %[[CALL]] : tensor<8x16xf32>

// 读法：
//   两个函数的 add 算子名完全相同
//   但 func=subroutine 把范围【限定】在 @subroutine 内
//   -> @main 里的 add 完全不受影响（CHECK-NOT 锁定）`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '@subroutine', c: '#4ade80', tag: '在 filter 内',
        d: '里面的 <span class="mono">add</span><br><b>被包</b>进 <span class="mono">manual_computation</span><br>区域内局部形状 <span class="mono">4x16</span>' },
      { t: '@main', c: '#94a3b8', tag: '不在 filter 内',
        d: '里面的 <span class="mono">add</span><br><b>保持原样</b><br><span class="mono">CHECK-NOT: manual_computation</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:12px">${x.t} <span class="faint small">${x.tag}</span></div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>关键</b>：两个函数的 <span class="mono">add</span> <b>算子名完全相同</b>。'; });
    tl.at(4400, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<span class="mono">func=subroutine</span> 把范围<b>限定</b>在 <span class="mono">@subroutine</span> 内 → <span class="mono">@main</span> 完全不受影响。';
    });
    tl.at(8200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>测试设计</b>：用 <span class="mono">CHECK-NOT</span> 锁定"不该被包的没被包" —— 与正面断言同等重要。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>三种语法小结</b>：算子名（最常用）/ 指令位置（最精确到条）/ 函数名（限定作用域）。';
    });
  }
},

/* ------------------------------------------------ 5 bisect 与族谱 */
{
  kicker: 'L4-11 · 逐指令分区',
  title: '<span class="hl-a">bisect</span>：这个 pass 为什么存在',
  sub: '整个导出流水线在某段 IR 上失败时，用 filter **逐个/分组**测试，缩小问题范围。',
  caption: '这是一个<b>调试工具</b>性质的 pass —— 不参与正常编译流程。',
  code: `// 【场景】导出流水线在某段 IR 上失败（崩溃 / 报错 / 结果不对）
//   但 IR 很长、算子很多 -> 不知道是【哪个算子】的问题

// 【用法】用 filter 逐个/分组只对怀疑的算子跑分区器
//   看问题是否复现 -> 这就是【二分定位】（bisect）

// filter 写法与用途对照：
//   filter=dot                      只测 dot
//   filter=dot,pad                  只测 dot 与 pad     <- 验收点
//   filter='selectLow=0, selectHigh=0'   只测第 0 条指令
//   filter=func=subroutine,add      只测 @subroutine 里的 add

// 【族谱】3 个文件 / 665 行 / 27 个用例
//   per_instruction_partitioning          578 行 / 23 用例   算子名子串
//   per_instruction_partitioning_range     49 行 /  2 用例   selectLow/selectHigh
//   per_instruction_partitioning_subroutine 38 行 / 2 用例   func=名字

// 【与 L1-07 / L4-09 的呼应】
//   本 pass 把选中的指令包进 sdy.manual_computation
//   L4-09 的 HALO 模式也用同一个机制
//   ┌──────────┬──────────────────┬──────────────────────┐
//   │ 课       │ 谁包             │ 为什么               │
//   ├──────────┼──────────────────┼──────────────────────┤
//   │ L4-09    │ permutation 消解 │ 表达 halo            │
//   │ L4-11    │ 指定的指令       │ 表达"我已手动分好了" │
//   └──────────┴──────────────────┴──────────────────────┘
//   共同点：manual_computation 是表达"区域内自己管分片"的【通用机制】

// 一句话总结：
//   只对指定的指令跑分区器，结果包进 sdy.manual_computation
//   三种 filter 语法：算子名 / 指令位置 / 所在函数
//   目的是【二分定位】导出流水线的问题`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '算子名', n: 23, c: '#4ade80' }, { t: '指令位置', n: 2, c: '#38bdf8' },
      { t: '所在函数', n: 2, c: '#fbbf24' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:150px;opacity:.4;transition:.35s;border-color:${f.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="card-t" style="color:${f.c};font-size:11.5px">${f.t}</div>
        <div class="big" style="font-size:20px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 130)); msg.innerHTML = '<b>27 个用例</b>分三种语法 —— 算子名占绝大多数（23）。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>bisect 的用法</b>：把怀疑的算子分组，逐个用 filter 测试，看问题在哪一组复现。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>与 L4-09 的呼应</b>：两者都用 <span class="mono">manual_computation</span> 承载 —— 同一个机制，不同目的。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L4-12 讲 <span class="mono">export-named-computations</span>。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L4-11 · 练习',
  title: '练一练：<span class="hl-a">写出 filter 串</span>',
  sub: '三道题分别考：核心机制、三种语法、bisect 用途。',
  caption: '一句话总结：<b>只对指定指令跑分区器，包进 manual_computation</b>。',
  code: `// 题 1：被选中的指令会被怎样处理？未选中的呢？

// 题 2：写出"只分区 dot 与 pad"的 filter 串。
//       如果只想分区 @foo 里的 dot 呢？

// 题 3：这个 pass 的用途是什么？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col compact-ex', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '被选中的指令会被怎样处理？未选中的呢？',
        a: '<b>选中的</b>：包进 <span class="mono">sdy.manual_computation</span> —— 区域内用<b>局部形状</b>、不带分片属性，<span class="mono">manual_axes</span> 列出被冻结的轴。' +
           '<br><b>未选中的</b>：<b>保持原样</b>，完全不受影响（用例用 <span class="mono">CHECK-NOT</span> 锁定）。' +
           '<br><span class="dim">为什么用 <span class="mono">manual_computation</span> 承载：它正是"区域内自己管分片"的机制（L1-07），与 L4-09 的 HALO 模式同一个机制。</span>'
      },
      {
        q: '写出"只分区 <span class="mono">dot</span> 与 <span class="mono">pad</span>"的 filter 串。如果只想分区 <span class="mono">@foo</span> 里的 <span class="mono">dot</span> 呢？',
        a: '① 只分区 <span class="mono">dot</span> 与 <span class="mono">pad</span>：<b><span class="mono">filter=dot,pad</span></b>（逗号分隔的算子名列表，子串匹配）。' +
           '<br>② 只分区 <span class="mono">@foo</span> 里的 <span class="mono">dot</span>：<b><span class="mono">filter=func=foo,dot</span></b>。' +
           '<br><span class="dim">第三种语法是 <span class="mono">selectLow=N, selectHigh=N</span> —— 按<b>指令位置</b>选区间。</span>'
      },
      {
        q: '这个 pass 的用途是什么？',
        a: '<b>bisect</b>（二分定位）。' +
           '<br><b>场景</b>：整个导出流水线在某段 IR 上失败（崩溃 / 报错 / 结果不对），但 IR 很长、算子很多，<b>不知道是哪个算子的问题</b>。' +
           '<br><b>用法</b>：用 <span class="mono">filter</span> 逐个/分组只对怀疑的算子跑分区器，看问题在哪一组复现。' +
           '<br><span class="dim">这是一个<b>调试工具</b>性质的 pass —— 不参与正常编译流程。</span>'
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
