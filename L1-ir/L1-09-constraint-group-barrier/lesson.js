/* ==========================================================================
   L1-09 · constraint-group-barrier
   --------------------------------------------------------------------------
   覆盖：ir/test/sharding_constraint_verification.mlir (60)
         ir/test/sharding_group_parse_print.mlir (8)
         ir/test/propagation_barrier_parse_print.mlir (32)
         ir/test/propagation_barrier_verification.mlir (7)
   目标：讲透三个"影响传播"的约束类算子，并示范如何验证测试文件里的一句话。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 三者定位 */
{
  kicker: 'L1-09 · 约束类算子',
  title: '三个"影响传播"的算子',
  sub: '它们都不做计算，只改变<b>传播的行为</b>：一个钉住分片、一个强制同步、一个截断方向。',
  caption: '与 L1-07/L1-08 的区域类算子不同，这三个都是<b>单值进出</b>的轻量算子（barrier 与 constraint）。',
  code: `// ① 钉住某个张量的分片（可悬空、可有使用者）
%1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]>
     : tensor<8x8xf32>

// ② 把若干张量绑成一组，强制同分片
sdy.sharding_group %arg0 group_id=21 : tensor<8xf32>

// ③ 截断传播方向
%0 = sdy.propagation_barrier %arg0 allowed_direction=FORWARD
     : tensor<8xf32>

// 三者都是"编译器 API"的一部分（TODOLIST L1-09 / L2 会反复用到）`,
  duration: 12000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%;align-items:center' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:750px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'sdy.sharding_constraint', d: '钉住一个张量（或它的一部分使用者）的分片。<br><b>传播会消费掉它</b>，换成 reshard。', c: '#38bdf8' },
      { t: 'sdy.sharding_group', d: '把没有数据依赖的张量绑成一组，<br>一旦其中一个被分片，<b>全体跟随</b>。', c: '#c084fc' },
      { t: 'sdy.propagation_barrier', d: '像恒等算子，但只允许分片沿<b>指定方向</b>流过。<br>FORWARD / BACKWARD / NONE。', c: '#fbbf24' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:238px;opacity:.34;transition:.3s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:11.5px;overflow-wrap:anywhere">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.34');
      msg.innerHTML = [
        '生命周期：<b>用户插入 → 传播消费 → 变成 sdy.reshard</b>（L1-10 讲 reshard）。',
        '用途：输入与输出之间没有数据依赖，传播推不过去，就用组把它们绑起来。',
        '用途：阻止不想要的传播 —— 例如切断"共用常量被拉成同一种分片"的假依赖。',
      ][i];
    }));
    tl.at(11000, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '共同点：<b>它们都不改变数值</b>，只改变编译器如何决定分片。';
    });
  }
},

/* ------------------------------------------------ 2 constraint 语义 */
{
  kicker: 'L1-09 · 约束类算子',
  title: '<span class="mono hl-a">sharding_constraint</span>：悬空 vs 有使用者',
  sub: '同一个算子，取决于<b>有没有使用者</b>，语义完全不同 —— 这是最容易搞错的一点。',
  caption: '悬空：约束的是<b>输入张量本身</b>。有使用者：只约束<b>这些使用者</b>，别的使用者可以有不同分片。',
  code: `// ① 悬空（dangling）：没有使用者
%0 = ... : tensor<8x8xf32>
%1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> : tensor<8x8xf32>
// 没有其它地方用 %1 -> 表示"%0 本身就应该这样切"

// ② 有使用者：只约束这些使用者
%1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> : tensor<8x8xf32>
%2 = stablehlo.dot %1, %w : ...
// %1 只被 dot 用 -> 表示"dot 看到的 %1 应该是这样切的"
// 而 %0 的其它使用者仍可要求别的分片

// ③ 维度可以是开的：还能继续被传播细化
%1 = sdy.sharding_constraint %0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:30px;justify-content:center;align-items:flex-start">
        <div class="col" style="gap:8px;align-items:center">
          <div class="mid" style="font-weight:700;color:var(--ax0)">① 悬空</div>
          <div class="row" style="gap:8px">
            <div class="chip c0" style="padding:9px 14px">%0</div>
            <div class="arrow" style="font-size:20px">⟹</div>
            <div class="chip mut" style="padding:9px 14px">constraint</div>
          </div>
          <div class="small faint" id="d1" style="height:20px"></div>
        </div>
        <div class="col" style="gap:8px;align-items:center">
          <div class="mid" style="font-weight:700;color:var(--ax1)">② 有使用者</div>
          <div class="row" style="gap:8px">
            <div class="chip c1" style="padding:9px 14px">%0</div>
            <div class="arrow" style="font-size:20px">⟹</div>
            <div class="chip mut" style="padding:9px 14px">constraint</div>
            <div class="arrow" style="font-size:20px">⟹</div>
            <div class="chip c2" style="padding:9px 14px">dot</div>
          </div>
          <div class="small faint" id="d2" style="height:20px"></div>
        </div>
      </div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:750px"></div>`;
    root.appendChild(wrap);
    const msg = wrap.querySelector('#msg'), d1 = wrap.querySelector('#d1'), d2 = wrap.querySelector('#d2');

    tl.at(800, () => { d1.textContent = '没有别的使用者'; d2.textContent = '只被 dot 使用'; msg.innerHTML = '形状一样，但下游不同 → 语义不同。'; });
    tl.at(3800, () => {
      msg.innerHTML = '<b>悬空</b>：约束 <span class="mono">%0</span> 本身。传播会尽量让 <span class="mono">%0</span> 就按这个分片。';
    });
    tl.at(6800, () => {
      msg.innerHTML = '<b>有使用者</b>：只约束 <span class="mono">dot</span> 看到的这一份。<span class="mono">%0</span> 若还有别处使用，那些地方可以有别的分片。';
    });
    tl.at(10000, () => {
      msg.innerHTML = '所以传播可能插入 <span class="mono">reshard</span>：把 <span class="mono">%0</span> 的某一份转成约束要求的分片。';
    });
    tl.at(12800, () => {
      msg.innerHTML = '维度带 <span class="mono">?</span> 时是<b>开维</b>：约束给出下界，传播仍可继续往上加轴。';
    });
  }
},

/* ------------------------------------------- 3 constraint 特殊校验 */
{
  kicker: 'L1-09 · 校验',
  title: 'constraint 的三条<span class="hl-a">特殊校验</span>',
  sub: '除了分片本身要合法（与普通张量分片同一套），还有三条只属于 constraint 的检查。',
  caption: '测试注释说得很直接：<span class="mono">Since ShardingConstraintOp::verify has the same verification as any TensorShardingAttr, there is no need to check different types of failures.</span>',
  code: `// ① 分片本身要合法（复用 L1-02/L1-03 规则）
%0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {"b"}], replicated={"a"}>
     : tensor<8xf32>
//   sharding doesn't match tensor rank: 2 != 1

// ② 区域内不能用已被父级 manual 绑定的轴（两种写法都会报）
%1 = sdy.sharding_constraint %arg1 <@mesh, [{"a"}, {}]> : tensor<8x32xf32>
%1 = sdy.sharding_constraint %arg1 <@mesh, [{}, {}], replicated={"a"}> : tensor<8x32xf32>
//   op operates on axis "a" which is already bound by a parent
//   sdy.manual_computation op

// ③ 不能改已有未归约轴的归约算子
//   输入是 unreduced=max{"x"}，约束却写默认（sum）
%0 = sdy.sharding_constraint %arg0 <@mesh, [{}], unreduced={"x"}> : tensor<8xf32>
//   cannot change the reduction operator of kept unreduced axes
//   from max to sum.

// ④ token 只能 rank 0
%0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}]> : !stablehlo.token`,
  duration: 18000,
  build(root, tl) {
    const wrap = W.errLayout(root);
    W.errScenes(wrap, tl, [
      {
        ir: '  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {"b"}], replicated={"a"}> : tensor<8xf32>',
        err: "sharding doesn't match tensor rank: 2 != 1",
        why: '<span class="mono">tensor&lt;8xf32&gt;</span> 是 rank 1，分片给了 2 个维分片 —— 这条校验与 L1-03 <b>完全一样</b>。'
      },
      {
        ir: '    %1 = sdy.sharding_constraint %arg1 <@mesh, [{"a"}, {}]> : tensor<8x32xf32>',
        err: 'op operates on axis "a" which is already bound by a parent sdy.manual_computation op',
        why: '<span class="mono">manual_axes={"a"}</span> 已经冻结了轴 "a"，区域内不得再用它切维度。'
      },
      {
        ir: '    %1 = sdy.sharding_constraint %arg1 <@mesh, [{}, {}], replicated={"a"}> : tensor<8x32xf32>',
        err: 'op operates on axis "a" which is already bound by a parent sdy.manual_computation op',
        why: '换个写法（放进 <span class="mono">replicated</span>）同样不行 —— <b>只要提到这个轴就违规</b>，因为 manual 轴由外层全权管理。'
      },
      {
        ir: '  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}], unreduced={"x"}> : tensor<8xf32>',
        err: 'cannot change the reduction operator of kept unreduced axes from max to sum.',
        why: '输入已有 <span class="mono">unreduced=max{"x"}</span>，约束写成 <span class="mono">unreduced={"x"}</span>（默认 sum）—— <b>归约语义被改了</b>，数值会不同。'
      },
      {
        ir: '  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}]> : !stablehlo.token',
        err: 'non-shaped tensors can only have a sharding with rank 0 and no replicated or unreduced axes',
        why: 'token 无形状 → 只能 rank 0。与 L1-02 的规则一致。'
      },
    ], {
      stepMs: 2800,
      finalIr: '// 四类里的三类都是【复用】：\n//   rank 校验、token 规则、manual 绑定\n// 只有一条是 constraint 独有：\n//   不能改未归约轴的归约算子',
      finalErr: 'rank mismatch / bound by a parent / cannot change the reduction operator',
      finalWhy: '这说明 SDY 的校验是<b>按关注点分层</b>的：通用规则 + 各算子特有规则。'
    });
  }
},

/* ------------------------------------------------------ 4 group */
{
  kicker: 'L1-09 · 约束类算子',
  title: '<span class="mono hl-a">sharding_group</span>：强制一组张量同分片',
  sub: '给张量打上 <span class="mono">group_id</span>，同组张量在传播中<b>一荣俱荣</b>：任何一个被分片，全体立刻跟随。',
  caption: '语法极简：<span class="mono">$input group_id=N</span>，没有结果（悬空算子）。导入阶段还会做<b>传递闭包合并</b>（L3-09）。',
  code: `// 语法
sdy.sharding_group %arg0 group_id=21 : tensor<8xf32>

// 形状：$input group_id=N，无结果
// 语义：把 %arg0 加入 21 号组

// 典型用法：输入与输出之间没有数据依赖
func.func @main(%arg0: tensor<8x2xi64> {sdy.sharding = ...}) -> ... {
  %0 = sdy.sharding_group %arg0 group_id=0 : tensor<8x2xi64>
  %1 = stablehlo.constant dense<0> : tensor<8x2xi64>
  %2 = sdy.sharding_group %1 group_id=0 : tensor<8x2xi64>
  return %2 : tensor<8x2xi64>
}
// 没有组时，传播推不到 %2，它会退化成全复制
// 有了组，%arg0 的分片会传给它`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:center" id="grp"></div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:750px"></div>`;
    root.appendChild(wrap);
    const grp = wrap.querySelector('#grp'), msg = wrap.querySelector('#msg');

    const node = (label, cls) => {
      const c = U.el('div', { class: 'col', style: 'gap:5px;align-items:center' });
      c.innerHTML = `<div class="chip ${cls}" style="padding:9px 14px;font-size:13.5px">${label}</div>`;
      grp.appendChild(c); return c;
    };

    tl.at(700, () => {
      node('%arg0', 'c0'); node('%1 (constant)', 'mut');
      msg.innerHTML = '两个张量之间<b>没有数据依赖</b> —— 传播无法从一边推到另一边。';
    });
    tl.at(3800, () => {
      msg.innerHTML = '结果：传播各自决定，<span class="mono">%1</span> 很可能退化成<b>全复制</b>，与 <span class="mono">%arg0</span> 的分片不一致。';
    });
    tl.at(7000, () => {
      grp.innerHTML = '';
      const a = node('%arg0', 'c0'); node('%1 (constant)', 'c0');
      a.insertAdjacentHTML('beforeend', '<div class="small mono" style="color:var(--accent)">group_id=0</div>');
      grp.lastChild.insertAdjacentHTML('beforeend', '<div class="small mono" style="color:var(--accent)">group_id=0</div>');
      msg.innerHTML = '加上 <span class="mono">sharding_group group_id=0</span> 后，两者被绑成一组。';
    });
    tl.at(10400, () => {
      msg.innerHTML = '<b>效果</b>：<span class="mono">%arg0</span> 的分片会立刻同步给 <span class="mono">%1</span>，无需数据依赖。';
      grp.querySelectorAll('.chip').forEach(e => e.classList.add('pulse'));
    });
    tl.at(12800, () => {
      msg.innerHTML = '导入阶段会把重叠的组做<b>传递闭包合并</b>，并把组号规范化为 0..N-1（L3-09）。';
    });
  }
},

/* ---------------------------------------------------- 5 barrier */
{
  kicker: 'L1-09 · 约束类算子',
  title: '<span class="mono hl-a">propagation_barrier</span>：截断传播方向',
  sub: '它像恒等算子一样把输入原样输出，但<b>只允许分片沿指定方向流过</b>。',
  caption: '三种方向：<span class="mono">FORWARD</span>（只能从操作数到结果）、<span class="mono">BACKWARD</span>（反向）、<span class="mono">NONE</span>（完全阻断）。',
  code: `sdy.mesh @mesh = <["a"=2,"b"=2]>

// 三种合法方向
%0 = sdy.propagation_barrier %arg0 allowed_direction=BACKWARD : tensor<8xf32>
%0 = sdy.propagation_barrier %arg0 allowed_direction=FORWARD  : tensor<8xf32>
%0 = sdy.propagation_barrier %arg0 allowed_direction=NONE     : tensor<8xf32>

// 屏障自身也可以带分片
%0 = sdy.propagation_barrier %arg0 allowed_direction=BACKWARD
     {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"a",?}]>]>}
     : tensor<8xf32>

// BOTH 被拒绝：允许双向 = 没有屏障
%0 = sdy.propagation_barrier %arg0 allowed_direction=BOTH : tensor<8xf32>
//   cannot specify \`BOTH\` as the direction`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:40px;justify-content:center;align-items:center" id="dirs"></div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:750px"></div>`;
    root.appendChild(wrap);
    const dirs = wrap.querySelector('#dirs'), msg = wrap.querySelector('#msg');

    const pair = (title, fwd, bwd, cls) => {
      const box = U.el('div', { class: 'col', style: 'gap:6px;align-items:center;opacity:.35;transition:.35s' });
      box.innerHTML = `<div class="small mono" style="color:${cls}">${title}</div>
        <div class="row" style="gap:8px;align-items:center">
          <div class="chip mut" style="padding:6px 11px">operand</div>
          <div style="font-size:20px;color:${fwd ? 'var(--ok)' : 'var(--ink-faint)'}">${fwd ? '→' : '⤫'}</div>
          <div class="chip mut" style="padding:6px 11px">result</div>
          <div style="font-size:20px;color:${bwd ? 'var(--ok)' : 'var(--ink-faint)'}">${bwd ? '←' : '⤫'}</div>
        </div>`;
      dirs.appendChild(box); return box;
    };

    tl.at(700, () => { pair('FORWARD', true, false, 'var(--ax0)').style.opacity = '1'; msg.innerHTML = '分片只能<b>从操作数流向结果</b>（前向），不能回推。'; });
    tl.at(3800, () => { pair('BACKWARD', false, true, 'var(--ax1)').style.opacity = '1'; msg.innerHTML = '分片只能<b>从结果回推到操作数</b>。'; });
    tl.at(6800, () => { pair('NONE', false, false, 'var(--ax3)').style.opacity = '1'; msg.innerHTML = '完全阻断：两边<b>互不影响</b>，各自独立决定分片。'; });
    tl.at(9800, () => {
      msg.innerHTML = '<b>为什么禁止 BOTH</b>：允许双向就等于一个恒等算子，屏障毫无作用，编译器直接拒绝这种写法。';
    });
    tl.at(12200, () => {
      msg.innerHTML = '屏障自身也能带分片属性 —— 它是个"有立场的恒等算子"。';
    });
  }
},

/* ------------------------------------------------ 6 如何验证 */
{
  kicker: 'L1-09 · 方法论',
  title: '★ 陷阱：测试文件里的注释<span class="hl-a">不一定是真的</span>',
  sub: '这一课覆盖的 <span class="mono">sharding_group_parse_print.mlir</span> 里有一行<b>看起来像期望输出、实际已过时</b>的注释。',
  caption: '方法：<b>跑一遍</b>。测试文件是"人写的"，会有残留；<span class="mono">sdy_opt</span> 的实际输出才是事实。',
  code: `// 测试文件第 5 行（原样）：
// CHECK sdy.sharding_group %arg0 group_id=21 type=AS  : tensor<8xf32>

// 注意：这里是 "// CHECK " —— 冒号没了！
//   "// CHECK: xxx"  = FileCheck 指令
//   "// CHECK xxx"   = 普通注释，FileCheck 完全忽略

// 实测 sdy_opt 的输出：
sdy.sharding_group %arg0 group_id=21 : tensor<8xf32>
// 并没有 type=AS

// 查 op 定义（ops.td）：
//   let arguments = (ins AnyRankedTensor:$input, I64Attr:$group_id);
// 根本没有 type 属性

// 验证方法：
// ① 原样跑        -> sdy_opt ... | FileCheck ...  => 退出码 0
// ② 补上冒号再跑  -> 报 "expected string not found" => 退出码 1
// 结论：那行不是生效的指令，type=AS 也不存在`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `
      <div class="irpair">
        <div class="irbox in"><div class="irh">测试文件里的那行</div>
          <pre style="font-size:11.5px">${U.hl('// CHECK sdy.sharding_group %arg0 group_id=21 type=AS  : tensor<8xf32>')}</pre></div>
        <div class="irbox out"><div class="irh">sdy_opt 的实际输出</div>
          <pre style="font-size:11.5px">${U.hl('sdy.sharding_group %arg0 group_id=21 : tensor<8xf32>')}</pre></div>
      </div>
      <div class="row" style="gap:12px;align-items:stretch">
        <div class="card" style="flex:1;border-color:rgba(74,222,128,.45)">
          <div class="card-t" style="color:var(--ok);font-size:12px">原样跑</div>
          <div class="card-d mono" style="font-size:11px">exit = 0 &nbsp;<span class="badge ok">通过</span></div></div>
        <div class="card" style="flex:1;border-color:rgba(251,113,133,.45)">
          <div class="card-t" style="color:var(--bad);font-size:12px">补上冒号改成真指令</div>
          <div class="card-d mono" style="font-size:11px">exit = 1 &nbsp;<span class="badge bad">expected string not found</span></div></div>
      </div>
      <div class="formula" id="msg" style="min-height:48px;display:flex;align-items:center;text-align:center"></div>`;
    root.appendChild(wrap);
    const msg = wrap.querySelector('#msg');

    tl.at(900, () => { msg.innerHTML = '差别只在一个<b>冒号</b>：<span class="mono">// CHECK:</span> 是指令，<span class="mono">// CHECK</span> 只是注释。'; });
    tl.at(3600, () => { msg.innerHTML = '所以那行声称的 <span class="mono">type=AS</span> <b>从来没被检查过</b>，测试照样通过。'; });
    tl.at(6600, () => { msg.innerHTML = '查 <span class="mono">ops.td</span>：<span class="mono">arguments = (ins AnyRankedTensor:$input, I64Attr:$group_id)</span> —— 没有 <span class="mono">type</span> 属性。'; });
    tl.at(9800, () => { msg.innerHTML = '结论：这是<b>过时的残留注释</b>，不能据此推断语法。'; });
    tl.at(12600, () => {
      msg.innerHTML = '<b>通用方法</b>：凡是"测试文件里写着但你没见过实际输出"的语法，<b>跑一遍 sdy_opt 确认</b>。';
    });
    tl.at(15200, () => {
      msg.innerHTML = '本动画的全部 IR 都遵循这条：逐字取自测试文件，且用 <span class="mono">sdy_opt</span> 复现过关键结论。';
    });
  }
},

/* ------------------------------------------------------------ 7 练习 */
{
  kicker: 'L1-09 · 练习',
  title: '练一练：<span class="hl-a">选算子、判方向</span>',
  sub: '三道题分别考：三个算子的选择、constraint 语义、屏障方向。',
  caption: 'L1 到这里只剩最后一课（reshard 与 constant）。',
  code: `// 题 1：输入与输出无数据依赖，想让它们同分片，用哪个算子？

// 题 2：下面这条 constraint 约束的是谁？
%1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> : tensor<8x8xf32>
%2 = stablehlo.dot %1, %w : ...
%3 = stablehlo.multiply %0, %0 : ...

// 题 3：allowed_direction=NONE 的屏障，两边会怎样？`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '输入与输出之间<b>没有数据依赖</b>，想让它们同分片，用哪个算子？',
        a: '<span class="mono">sdy.sharding_group</span>。把两者打上同一个 <span class="mono">group_id</span> 即可。' +
           '<br><span class="dim">用 <span class="mono">sharding_constraint</span> 也能达到类似效果（在输出上钉一个分片），但前提是你<b>已经知道</b>要什么分片；' +
           '组的用法是"我不管具体怎么切，反正这两个要一样"。</span>'
      },
      {
        q: '<span class="mono">%1 = sdy.sharding_constraint %0 ...</span> 后面既有 <span class="mono">dot %1</span> 又有 <span class="mono">multiply %0</span>，这条约束约束的是谁？',
        a: '约束的是 <b><span class="mono">%1</span> 的使用者（dot）</b>，而不是 <span class="mono">%0</span> 本身。' +
           '<br><b>理由</b>：constraint 有使用者 → 只描述"这些使用者看到的分片"。' +
           '<br><span class="mono">%0</span> 的另一个使用者 <span class="mono">multiply</span> 完全可以要求不同分片；' +
           '传播会在需要时插入 <span class="mono">reshard</span>。'
      },
      {
        q: '<span class="mono">allowed_direction=NONE</span> 的屏障，两边会怎样？',
        a: '两边<b>完全互不影响</b>，各自独立决定分片。<br>' +
           '<b>用途</b>：切断不想要的传播路径。典型场景是"共用常量被多个使用者拉成同一种分片"这类<b>假依赖</b>。' +
           '<br><span class="dim">注意：屏障本身仍是恒等算子，数值不变 —— 它只影响编译期的分片决策。</span>'
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
