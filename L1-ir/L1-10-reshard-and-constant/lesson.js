/* ==========================================================================
   L1-10 · reshard-and-constant  （L1 收官课）
   --------------------------------------------------------------------------
   覆盖：ir/test/reshard_verification.mlir (66)
         ir/test/reshard_canonicalization.mlir (255)
         ir/test/constant_parse_print.mlir (15)
         ir/test/constant_verification.mlir (15)
   目标：讲透 sdy.reshard 的生命周期与规范化，以及 sdy.constant 的设计取舍。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 生命周期 */
{
  kicker: 'L1-10 · 重分片与常量',
  title: '<span class="mono hl-a">sdy.reshard</span> 的生命周期',
  sub: '它只在<b>传播之后、分区之前</b>存在。三个阶段各有一个算子承担"换分片"的职责。',
  caption: '这条链条是本课最重要的心智模型：<b>约束 → 传播 → reshard → collective</b>。',
  code: `// ① 传播前：用户用 constraint 表达"我要这样切"
%1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]>
     : tensor<8x8xf32>

// ② 传播中：constraint 被消费，需要换分片的地方插入 reshard
%2 = sdy.reshard %0 <@mesh, [{"a"}, {}]> : tensor<8x8xf32>

// ③ 分区（导出）时：reshard 被换成真正的集合通信
%2 = sdy.all_gather [{"b"}] %0 out_sharding=<@mesh, [{"a"}, {}]>
     : tensor<8x8xf32>

// 导出完成后，IR 里【不应该】再有 reshard`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:16px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:14px;justify-content:center;align-items:center" id="stages"></div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:760px"></div>`;
    root.appendChild(wrap);
    const stages = wrap.querySelector('#stages'), msg = wrap.querySelector('#msg');

    const stage = (n, title, op, color) => {
      const c = U.el('div', { class: 'card', style: `width:196px;border-color:${color}66;opacity:0;transition:.5s;text-align:center` });
      c.innerHTML = `<div class="card-t" style="color:${color};font-size:11.5px">${n} ${title}</div>
        <div class="mono" style="margin:6px 0;font-size:11px;color:#cfe0ff;overflow-wrap:anywhere">${U.esc(op)}</div>`;
      stages.appendChild(c); return c;
    };
    const a1 = () => stage('①', '传播前', 'sdy.sharding_constraint', '#38bdf8');
    const a2 = () => stage('②', '传播中', 'sdy.reshard', '#c084fc');
    const a3 = () => stage('③', '分区时', 'sdy.all_gather / …', '#4ade80');

    tl.at(700, () => { a1().style.opacity = '1'; msg.innerHTML = '用户用 <span class="mono">constraint</span> 表达意图；此时还没有 reshard。'; });
    tl.at(4000, () => {
      a2().style.opacity = '1';
      msg.innerHTML = '传播消费掉 constraint。如果某个使用者的分片要求与生产者不同，就插入 <span class="mono">reshard</span>。';
    });
    tl.at(7600, () => {
      a3().style.opacity = '1';
      msg.innerHTML = '分区器把每条 <span class="mono">reshard</span> 翻译成具体的集合通信（L4-08）。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<b>验收标准</b>：导出结束后 IR 里<b>不应再有</b> <span class="mono">sdy.reshard</span> —— 可以用这个检查自己的流水线。';
    });
  }
},

/* ------------------------------------------------------ 2 校验 */
{
  kicker: 'L1-10 · 校验',
  title: 'reshard 的校验：<span class="hl-a">复用 + 两条特有</span>',
  sub: '分片本身的校验与普通张量分片完全一样；另有两条只在"换分片"时才出现的检查。',
  caption: '测试注释同样写着：<span class="mono">Since ReshardOp::verify has the same verification as any TensorShardingAttr, there is no need to check different types of failures.</span>',
  code: `// ① 复用：分片本身要合法
%0 = sdy.reshard %arg0 <@mesh, [{}, {"b"}], replicated={"a"}> : tensor<8xf32>
//   sharding doesn't match tensor rank: 2 != 1

// ② 区域内不能用 manual 轴（与 constraint 完全一致）
%1 = sdy.reshard %arg1 <@mesh, [{"a"}, {}]> : tensor<8x32xf32>
//   op operates on axis "a" which is already bound by a parent
//   sdy.manual_computation op

// ③ 不能改"保留的未归约轴"的归约算子
//   输入 unreduced=max{"x"}，输出仍保留 "x" 但写成 sum -> ✗
%0 = sdy.reshard %arg0 <@mesh, [{}], unreduced={"x"}> : tensor<8xf32>
//   cannot change the reduction operator of kept unreduced axes
//   from max to sum.

// ④ 反例（合法）：丢掉 max 轴、引入新的未归约轴
//   输入 unreduced={"x","y"} -> 输出 unreduced={"x","z"}
%0 = sdy.reshard %arg0 <@mesh, [{}], unreduced={"x", "z"}> : tensor<8xf32>
//   保留的 "x" 归约算子没变 -> 合法`,
  duration: 18000,
  build(root, tl) {
    const wrap = W.errLayout(root);
    W.errScenes(wrap, tl, [
      {
        ir: '  %0 = sdy.reshard %arg0 <@mesh, [{}, {"b"}], replicated={"a"}> : tensor<8xf32>',
        err: "sharding doesn't match tensor rank: 2 != 1",
        why: '与 L1-03 完全同一条校验 —— reshard 不引入新的分片规则。'
      },
      {
        ir: '    %1 = sdy.reshard %arg1 <@mesh, [{"a"}, {}]> : tensor<8x32xf32>',
        err: 'op operates on axis "a" which is already bound by a parent sdy.manual_computation op',
        why: '与 constraint 一样：manual 轴由外层全权管理，区域内不得提及。'
      },
      {
        ir: '  %0 = sdy.reshard %arg0 <@mesh, [{}], unreduced={"x"}> : tensor<8xf32>',
        err: "'sdy.reshard' op cannot change the reduction operator of kept unreduced axes from max to sum.",
        why: '输入是 <span class="mono">unreduced=max{"x"}</span>，输出仍保留 <span class="mono">"x"</span> 却写成默认 sum —— <b>保留轴的归约语义被改了</b>，数值会不同。'
      },
      {
        ir: '  %0 = sdy.reshard %arg0 <@mesh, [{}], unreduced={"x", "y"}> : tensor<8xf32>',
        err: "'sdy.reshard' op cannot change the reduction operator of kept unreduced axes from max to sum.",
        why: '即使同时"引入了新轴 y"，只要保留的 <span class="mono">"x"</span> 归约算子变了就报错 —— 检查针对的是<b>保留轴</b>。'
      },
    ], {
      stepMs: 3000,
      finalIr: '// 反例（测试中【没有】expected-error）：\n//   输入 unreduced={"x","y"}  ->  输出 unreduced={"x","z"}\n//   保留的 "x" 仍是 sum  ->  合法\n//   丢掉了 "y"，新增了 "z"',
      finalErr: '（此用例不期望任何报错）',
      finalWhy: '<b>规则</b>：保留轴的归约算子不能改；<b>丢弃</b>与<b>新增</b>未归约轴则是允许的。'
    });
  }
},

/* ------------------------------------------------ 3 规范化：折叠 */
{
  kicker: 'L1-10 · 规范化',
  title: '链式折叠：<span class="hl-a">中间结果没人用就合并</span>',
  sub: '连续多条 reshard 可以合并成一条 —— 但前提是<b>中间结果没有别的使用者</b>。',
  caption: '这与通用的 DCE/CSE 思路一致：只有当中间值不被别处需要时，才能把它"跳过去"。',
  code: `// ① 中间结果无其它使用者 -> 只留最后一条
%0 = sdy.reshard %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
%1 = sdy.reshard %0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
return %1
//   => %0 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]>

// ② 中间结果被 return 了 -> 不能折叠
%0 = sdy.reshard %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
%1 = sdy.reshard %0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
return %0, %1
//   => 两条都保留

// ③ 三段链式同样折叠成一个
%0 = sdy.reshard %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
%1 = sdy.reshard %0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
%2 = sdy.reshard %1 <@mesh, [{?}, {"a", ?}]> : tensor<8x8xf32>
//   => %0 = sdy.reshard %arg0 <@mesh, [{?}, {"a", ?}]>`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:22px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:760px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const chain = (n, label, color, extra) => {
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center;opacity:0;transition:.45s' });
      c.innerHTML = `<div class="small mono" style="color:${color}">${label}</div>
        <div class="row" style="gap:5px" data-d></div>
        <div class="small faint" style="height:18px">${extra || ''}</div>`;
      const h = c.querySelector('[data-d]');
      for (let i = 0; i < n; i++) {
        const e = U.el('div', { class: 'chip c2', style: 'padding:6px 11px;font-size:12.5px' });
        e.textContent = 'reshard';
        h.appendChild(e);
        if (i < n - 1) h.insertAdjacentHTML('beforeend', '<span class="faint">→</span>');
      }
      demo.appendChild(c); return c;
    };

    tl.at(700, () => {
      chain(2, '① 链式两条', 'var(--ax0)', '中间值无其它使用者');
      chain(1, '折叠后', 'var(--ok)', '只留目标分片');
      msg.innerHTML = '<span class="mono">reshard(reshard(x, A), B)</span> 等价于 <span class="mono">reshard(x, B)</span> —— 中间的 A 是白做的。';
    });
    tl.at(4600, () => {
      demo.innerHTML = '';
      chain(2, '② 中间值被 return', 'var(--ax0)', '有其它使用者');
      msg.innerHTML = '这时<b>不能</b>折叠：中间结果的分片是程序的<b>可见输出</b>，跳过去就变了语义。';
    });
    tl.at(8200, () => {
      demo.innerHTML = '';
      chain(3, '③ 三段链式', 'var(--ax1)', '全部无其它使用者');
      chain(1, '折叠后', 'var(--ok)', '一个 reshard 直达');
      msg.innerHTML = '可以一次折叠到底 —— 中间两个 reshard 全部消失。';
    });
    tl.at(12000, () => {
      msg.innerHTML = '<b>判定条件只有一条</b>：中间结果除了下一条 reshard 之外<b>没有别的使用者</b>。';
    });
  }
},

/* -------------------------------------------------- 4 规范化：CSE */
{
  kicker: 'L1-10 · 规范化',
  title: 'CSE：<span class="hl-a">同一个输入 + 同一个目标分片 → 合并</span>',
  sub: '如果多条 reshard 的输入与目标分片都相同，它们就是同一个操作，可以共享结果。',
  caption: '这能显著减少通信：三条相同的 reshard 合并成一条，下游三个算子共享同一份重排后的数据。',
  code: `// ① 相同 -> 合并
%0 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
%1 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
return %0, %1
//   => %0 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]>
//      return %0, %0

// ② 目标分片不同 -> 不合并
%0 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
%1 = sdy.reshard %arg0 <@mesh, [{?}, {"a", ?}]> : tensor<8x8xf32>
return %0, %1
//   => 两条都保留

// ③ 合并后下游共享同一条通信
//   sin/cos/abs 三个使用者共用一个 reshard 结果

// 注意：只有当分片"完全等价"时才合并 ——
// 相同分片的不同写法（如显隐式复制的差异）需要先规范化。`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:760px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const fan = (n, label, color, shared) => {
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center;opacity:0;transition:.45s' });
      c.innerHTML = `<div class="small mono" style="color:${color}">${label}</div>
        <div class="row" style="gap:6px" data-d></div>`;
      const h = c.querySelector('[data-d]');
      const use = shared ? 1 : n;
      for (let i = 0; i < use; i++) {
        const e = U.el('div', { class: 'chip c2', style: 'padding:6px 11px;font-size:12.5px' });
        e.textContent = 'reshard';
        h.appendChild(e);
      }
      h.insertAdjacentHTML('beforeend', '<span class="faint">→</span>');
      for (let i = 0; i < n; i++) {
        const e = U.el('div', { class: 'chip mut', style: 'padding:6px 10px;font-size:12px' });
        e.textContent = ['sin', 'cos', 'abs'][i] || ('op' + i);
        h.appendChild(e);
      }
      demo.appendChild(c); return c;
    };

    tl.at(700, () => {
      fan(3, '① 合并前', 'var(--bad)', false);
      msg.innerHTML = '三个使用者各自写了一条 reshard —— 但输入与目标分片<b>完全相同</b>。';
    });
    tl.at(4200, () => {
      demo.innerHTML = '';
      fan(3, '② 合并后', 'var(--ok)', true);
      msg.innerHTML = 'CSE 后只剩一条 reshard，三个使用者共享结果 —— <b>通信量降到 1/3</b>。';
    });
    tl.at(7800, () => {
      msg.innerHTML = '<b>不合并的情形</b>：目标分片不同（如 <span class="mono">[{"a",?},{?}]</span> 与 <span class="mono">[{?},{"a",?}]</span>）—— 它们本来就是不同的重排。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '测试里还有"带 shape / transpose / 子轴"的变体，验证 CSE 在复杂分片下也正确合并。';
    });
    tl.at(13800, () => {
      msg.innerHTML = '前提：分片要"完全等价"。同一语义的不同写法需先由 canonicalize 统一。';
    });
  }
},

/* -------------------------------------------------- 5 sdy.constant */
{
  kicker: 'L1-10 · 常量',
  title: '<span class="mono hl-a">sdy.constant</span>：为什么不用现成的 ConstantLike？',
  sub: '官方说明很直接：它<b>故意不实现 ConstantLike、也不带 folder</b> —— 否则贪婪重写器会把常量合并回去，破坏"每个使用者可以有自己的分片"。',
  caption: '这是"设计取舍"的好例子：为了让子计算能被<b>复制并按使用处分别分片</b>，宁可放弃通用的常量折叠能力。',
  code: `// 官方定义里的说明（ops.td）：
//   NOTE: SDY defines its own constant op that isn't ConstantLike and
//   doesn't have a folder, so that we'll be able to duplicate constants
//   without any greedy pattern rewriter folding them back into a single
//   constant. In this way, constants can be sharded differently for
//   every use, and no propagation is done between constants
//   (or constant expressions).

// 结果：一个常量可以被复制成多份，各自分片不同
%a = sdy.constant dense<1.0> : tensor<8x16xf32>
     // 给 dot 用，沿 "a" 切
%b = sdy.constant dense<1.0> : tensor<8x16xf32>
     // 给 add 用，沿 "b" 切
// 如果是 ConstantLike，重写器会把它们合并成一个

// 导入阶段配套的 pass：
//   -sdy-constant-or-scalar-splitter   （L3-02，拆开）
//   -sdy-constant-or-scalar-merger     （L4-17，必要时再合回）`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:24px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:760px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const side = (title, n, color, note) => {
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center;opacity:0;transition:.45s' });
      c.innerHTML = `<div class="small mono" style="color:${color}">${title}</div>
        <div class="col" style="gap:5px" data-d></div>
        <div class="small faint" style="height:18px">${note}</div>`;
      const h = c.querySelector('[data-d]');
      for (let i = 0; i < n; i++) {
        const e = U.el('div', { class: 'chip ' + (i % 2 ? 'c1' : 'c0'), style: 'padding:6px 11px;font-size:12px' });
        e.textContent = 'constant @' + (i % 2 ? 'dot' : 'add');
        h.appendChild(e);
      }
      demo.appendChild(c); return c;
    };

    tl.at(700, () => {
      side('用 ConstantLike（会被合并）', 1, 'var(--bad)', '两个使用者被迫同分片');
      msg.innerHTML = '如果它实现了 ConstantLike，贪婪重写器会把两份常量<b>合并成一个</b>。';
    });
    tl.at(4200, () => {
      demo.innerHTML = '';
      side('用 sdy.constant（不合并）', 2, 'var(--ok)', '各自分片，互不影响');
      msg.innerHTML = '<span class="mono">sdy.constant</span> 刻意不带 folder → 两份常量<b>独立存在</b>，可以有不同的分片。';
    });
    tl.at(8000, () => {
      msg.innerHTML = '这就是"<b>假依赖</b>"问题的解法：共用常量不该让两个不相干的使用者被切成同一种分片。';
    });
    tl.at(11200, () => {
      msg.innerHTML = '导入时用 <span class="mono">-sdy-constant-or-scalar-splitter</span> 主动拆（L3-02）；必要时再用 <span class="mono">-sdy-constant-or-scalar-merger</span> 合回（L4-17）。';
    });
    tl.at(13800, () => {
      msg.innerHTML = '<b>代价</b>：放弃了通用的常量折叠能力 —— 这是为分片质量做的自觉取舍。';
    });
  }
},

/* -------------------------------------------- 6 constant 校验与打印 */
{
  kicker: 'L1-10 · 常量',
  title: '常量的打印与校验',
  sub: '语法与其它常量类似，但<b>形状必须静态</b>；量化类型仍由 <span class="mono">stablehlo.constant</span> 承载。',
  caption: '静态形状是硬要求：分片要能算出局部形状，动态/无 rank 的常量无从算起。',
  code: `// ① 基本打印
%0 = sdy.constant dense<1.000000e+00> : tensor<8x16xf32>

// ② 量化类型：仍用 stablehlo.constant
%0 = stablehlo.constant() {value = dense<[1, 512, 4]> : tensor<3xi32>}
     : () -> tensor<3x!quant.uniform<i32:f32, 2.000000e+00:15>>
// 打印为：
//   stablehlo.constant() <{value = dense<[1, 512, 4]> : tensor<3xi32>}>
//     : () -> tensor<3x!quant.uniform<i32:f32, 2.000000e+00:15>>

// ③ 校验：必须静态形状
%0 = sdy.constant dense<1.000000e+00> : tensor<8x?xf32>    // ✗ 动态
%0 = sdy.constant dense<1.000000e+00> : tensor<*xf32>      // ✗ 无 rank
//   elements literal type must have static shape`,
  duration: 15000,
  build(root, tl) {
    const wrap = W.errLayout(root, {
      irTitle: '写法', errTitle: '结果', whyTitle: '解读',
      errColor: 'var(--accent)', errBorder: 'rgba(94,234,212,.45)', errFg: '#bdf7ec'
    });
    W.errScenes(wrap, tl, [
      {
        ir: '  %0 = sdy.constant dense<1.000000e+00> : tensor<8x16xf32>',
        err: '✓ 合法',
        why: '最常见的形态：稠密元素字面量 + 静态形状。打印时浮点会补足精度（<span class="mono">1.000000e+00</span>）。'
      },
      {
        ir: '  %0 = stablehlo.constant() {value = dense<[1, 512, 4]> : tensor<3xi32>} : () -> tensor<3x!quant.uniform<i32:f32, 2.000000e+00:15>>',
        err: '✓ 合法（但用的是 stablehlo.constant）',
        why: '量化类型不由 <span class="mono">sdy.constant</span> 承载。测试用这个用例验证 <span class="mono">stablehlo.constant</span> 的打印形式。'
      },
      {
        ir: '  %0 = sdy.constant dense<1.000000e+00> : tensor<8x?xf32>',
        err: 'elements literal type must have static shape',
        why: '动态维度 <span class="mono">?</span> 无法确定元素个数，字面量放不下。'
      },
      {
        ir: '  %0 = sdy.constant dense<1.000000e+00> : tensor<*xf32>',
        err: 'elements literal type must have static shape',
        why: '无 rank 同样不行 —— 与 L1-03 对"无 rank 张量不能有分片"的理由一致：<b>算不出局部形状</b>。'
      },
    ], {
      stepMs: 2800,
      finalIr: '// 两条校验其实是同一条：\n//   "elements literal type must have static shape"\n// 动态与无 rank 都被这条覆盖。',
      finalErr: '（同一个报文）',
      finalWhy: '与分片规则一脉相承：<b>能算出静态形状才谈得上分片</b>。'
    });
  }
},

/* ------------------------------------------------------ 7 L1 总结 */
{
  kicker: 'L1-10 · 收官',
  title: '★ L1 总结：<span class="hl-a">十课的知识地图</span>',
  sub: 'L1 把 SDY 的<b>所有构件</b>讲完了。这张图把它们按"表示 / 区域 / 约束 / 辅助"分成四组。',
  caption: '接下来 L2 进入传播算法：会用到的每一个构件，都已经在 L1 出现过。',
  code: `// 【表示】描述"怎么切"
//   L1-01 sdy.mesh              逻辑网格与设备编号
//   L1-02 #sdy.sharding         维分片 / 开闭维 / 复制 / 子轴 / 优先级
//   L1-03 校验不变量            七条不变量
//   L1-04 sdy.propagation_edges 边分片（调试元数据）
//   L1-05 op_sharding_rule      算子分片规则（因子映射）

// 【区域】描述"哪里是局部/命名世界"
//   L1-07 sdy.manual_computation  手动计算（局部形状）
//   L1-08 sdy.named_computation   命名计算 + 数据流边

// 【约束】影响传播行为
//   L1-09 sdy.sharding_constraint / sharding_group / propagation_barrier

// 【辅助】表示中间状态
//   L1-06 八个集合通信算子
//   L1-10 sdy.reshard / sdy.constant

// 贯穿全程的一条主线：
//   分片 ↔ 复制 ↔ 未归约  三种状态的转换`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%;align-items:center' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="groups"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const groups = [
      { t: '表示', c: '#38bdf8', items: ['L1-01 mesh', 'L1-02 sharding', 'L1-03 校验', 'L1-04 边分片', 'L1-05 规则'] },
      { t: '区域', c: '#c084fc', items: ['L1-07 manual', 'L1-08 named + 边'] },
      { t: '约束', c: '#fbbf24', items: ['L1-09 constraint', 'group', 'barrier'] },
      { t: '辅助', c: '#4ade80', items: ['L1-06 集合通信', 'L1-10 reshard', 'constant'] },
    ];
    const host = wrap.querySelector('#groups');
    const els = groups.map(g => {
      const e = U.el('div', { class: 'card', style: 'width:180px;opacity:.32;transition:.35s;border-color:' + g.c + '55;padding:9px' });
      e.innerHTML = `<div class="card-t" style="color:${g.c};font-size:12px">${g.t}</div>
        <div class="col" style="gap:3px;margin-top:5px">${g.items.map(x => `<div class="mono" style="font-size:10.5px;color:#cfe0ff">${x}</div>`).join('')}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    groups.forEach((g, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.32');
      msg.innerHTML = [
        '五个构件回答同一个问题：<b>这个张量怎么切</b>。L1-03 的七条不变量约束它们必须自洽。',
        '两个区域算子回答：<b>哪一段是独立的世界</b> —— manual 是局部的，named 是命名的。',
        '三个约束算子回答：<b>如何干预传播</b> —— 钉住、绑定、截断方向。',
        '集合通信表示"分片正在变化"，reshard 表示"分片需要变化"，constant 是唯一带值的构件。',
      ][i];
    }));
    tl.at(14600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>L1 完结</b>：241 个测试文件里的 <b>28 个</b> ir/test 文件已全部覆盖（另有 L3/L4 层的文件在后面讲）。';
    });
  }
},

/* ------------------------------------------------------------ 8 练习 */
{
  kicker: 'L1-10 · 练习',
  title: '练一练：<span class="hl-a">判断折叠与合法性</span>',
  sub: '三道题分别考：链式折叠、CSE、未归约轴规则。',
  caption: '答完这三题，L1 的十课就完整闭环了。',
  code: `// 题 1：下面能折叠成一条 reshard 吗？
%0 = sdy.reshard %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
%1 = sdy.reshard %0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
return %0, %1

// 题 2：能 CSE 合并吗？
%0 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
%1 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>

// 题 3：合法吗？
//   输入 unreduced=max{"x"}
%0 = sdy.reshard %arg0 <@mesh, [{}], unreduced={"x", "y"}> : tensor<8xf32>`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">%0</span> 被 <span class="mono">return</span> 了，<span class="mono">%0→%1</span> 这条链能折叠吗？',
        a: '<b class="badge bad">不能</b> 中间结果 <span class="mono">%0</span> 有其它使用者（被 return），它的分片是程序的可见输出。' +
           '<br>折叠会改变 <span class="mono">%0</span> 的分片语义，因此两条都要保留。' +
           '<br><span class="dim">若 <span class="mono">%0</span> 只被 <span class="mono">%1</span> 使用，则可以折叠成 <span class="mono">sdy.reshard %arg0 &lt;@mesh, [{"a", ?}, {?}]&gt;</span>。</span>'
      },
      {
        q: '两条输入相同、目标分片也相同的 reshard，能 CSE 合并吗？',
        a: '<b class="badge ok">能</b> 它们的输入与目标分片<b>完全一致</b>，是同一个操作，合并成一条并让两个使用者共享结果。' +
           '<br><b>收益</b>：通信量减半。这也是规范化最有价值的一类优化。'
      },
      {
        q: '输入 <span class="mono">unreduced=max{"x"}</span>，reshard 成 <span class="mono">unreduced={"x", "y"}</span>，合法吗？',
        a: '<b class="badge bad">非法</b> 保留轴 <span class="mono">"x"</span> 的归约算子从 <span class="mono">max</span> 变成了默认的 <span class="mono">sum</span>。' +
           '<br>报错：<span class="mono">cannot change the reduction operator of kept unreduced axes from max to sum.</span>' +
           '<br><span class="dim">对比合法反例：输入 <span class="mono">unreduced={"x","y"}</span> → 输出 <span class="mono">unreduced={"x","z"}</span>（保留的 x 仍是 sum）。</span>'
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
