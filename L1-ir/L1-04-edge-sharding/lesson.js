/* ==========================================================================
   L1-04 · edge-sharding
   --------------------------------------------------------------------------
   覆盖：ir/test/edge_sharding_parse_print.mlir (14)
         ir/test/edge_sharding_verification.mlir (164)
   目标：讲透 sdy.propagation_edges —— 它记录什么、语法如何、15 类校验用例。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 为什么需要 */
{
  kicker: 'L1-04 · 边分片',
  title: '分片是<span class="hl-a">从哪来</span>的？边分片回答这个问题',
  sub: '传播过程中每个张量的分片都是"被别的值传染"的。<span class="mono">sdy.propagation_edges</span> 把这条传染链记录下来，用于调试与可解释性。',
  caption: '它是<b>调试元数据</b>，不参与语义 —— 导出前会被 <span class="mono">-sdy-remove-propagation-debug-info</span> 清掉（见 L4-17）。',
  code: `// 一个张量的分片：沿 "y" 和 "x" 切
%0 = stablehlo.add %arg0, %arg0
     {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"y"}, {"x"}]>]>}

// 但它是怎么变成这样的？边分片补充了来龙去脉：
//   第 2 步：operand-1 上的 "y" 传播到了 operand-0 和 result-0
//   第 12345 步：result-0 上的 "x" 反向传播到了 operand-0

// 有了它，就能回答"这个分片是谁引入的"`,
  duration: 12000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:16px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:22px;justify-content:center;align-items:center">
        <div class="col" style="gap:6px;align-items:center">
          <div class="small mono faint">operand-0</div>
          <div class="dev" style="width:66px;height:66px"><span class="small mono">a</span></div>
        </div>
        <div class="col" style="gap:6px;align-items:center">
          <div class="small mono faint">operand-1</div>
          <div class="dev" style="width:66px;height:66px;border-color:var(--accent)">
            <span class="small mono" style="color:var(--accent)">a</span></div>
        </div>
        <div class="arrow anim" style="font-size:26px">⟹</div>
        <div class="col" style="gap:6px;align-items:center">
          <div class="small mono faint">result-0</div>
          <div class="dev" style="width:66px;height:66px"><span class="small mono">a</span></div>
        </div>
      </div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:700px"></div>`;
    root.appendChild(wrap);
    const msg = wrap.querySelector('#msg');

    tl.at(900, () => { msg.innerHTML = '假设只有 <span class="mono">operand-1</span> 一开始带分片 —— 它是这条链的<b>源头</b>。'; });
    tl.at(3400, () => { msg.innerHTML = '传播把它推到同一条数据流边的其它值上：operand-0 与 result-0。'; });
    tl.at(6200, () => { msg.innerHTML = '边分片记录的就是：<b>第几步、哪条轴、从谁、到谁</b>。'; });
    tl.at(9000, () => { msg.innerHTML = '多个源头会形成多步记录，最终拼出完整的"分片来源图"。'; });
    tl.at(11000, () => { msg.innerHTML = '对应开关：<span class="mono">-sdy-aggressive-propagate=debug-propagation-edge-sharding=true</span>（L2-12 细讲）。'; });
  }
},

/* ---------------------------------------------------------- 2 语法 */
{
  kicker: 'L1-04 · 边分片',
  title: '拆开看：<span class="mono hl-a">#sdy.propagation_edges</span>',
  sub: '结构是层层嵌套的四层：属性 → 若干 step → 若干轴条目 → 一个 source 与若干 target。',
  caption: '注意 <span class="mono">-&gt;</span> 左边是 <b>source</b>，右边是 <b>targets</b> 列表 —— 一个源头可以流向多个目标。',
  code: `#sdy.propagation_edges<
  [                                            // ① step 列表
    {step-2 = [                                // ② 第 2 步
      {"y" = operand-1 -> [operand-0, result-0]} // ③ 轴 "y"，1 个 source → 2 个 targets
    ]},
    {step-12345 = [
      {"x" = result-0 -> [operand-0]}          //    轴 "x"，反向传播
    ]}
  ]
>

// 值引用的写法：
//   operand-0 / operand-1 / ...   操作数
//   result-0  / result-1  / ...   结果
//   block_arg-0 / ...             块参数`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="irbox" style="width:100%"><div class="irh">结构</div>
        <pre style="font-size:12px;max-height:190px">${U.hl(`#sdy.propagation_edges<
  [
    {step-2 = [
      {"y" = operand-1 -> [operand-0, result-0]}
    ]},
    {step-12345 = [
      {"x" = result-0 -> [operand-0]}
    ]}
  ]
>`)}</pre></div>
      <div class="row" style="gap:10px;justify-content:center" id="lvls"></div>
      <div class="formula" id="msg" style="min-height:44px;display:flex;align-items:center;text-align:center;max-width:730px"></div>`;
    root.appendChild(wrap);

    const host = wrap.querySelector('#lvls');
    const defs = [
      { t: '① step 列表', d: '按发生顺序排列的传播步骤', c: '#38bdf8' },
      { t: '② step-N', d: '第 N 步。N 必须唯一且非负', c: '#c084fc' },
      { t: '③ 轴条目', d: '"轴名" = source -> [targets]', c: '#4ade80' },
      { t: '④ 值引用', d: 'operand-i / result-i / block_arg-i', c: '#fbbf24' },
    ];
    const els = defs.map(d => {
      const e = U.el('div', { class: 'card', style: 'width:172px;opacity:.3;transition:.3s;padding:9px' });
      e.innerHTML = `<div class="card-t" style="font-size:12px;color:${d.c}">${d.t}</div>
        <div class="card-d" style="font-size:11.5px">${d.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 2800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.3');
      msg.innerHTML = [
        '一步 = 一次传播决策。步骤编号越大表示发生得越晚。',
        '编号必须<b>唯一</b>（不能有两个 step-1）且<b>非负</b>。',
        '一条轴条目描述"轴上的一次传播"，targets 可以有多个。',
        '<span class="mono">operand-1</span> 指第 2 个操作数；索引必须落在该算子的范围内。',
      ][i];
    }));
    tl.at(12200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '四个层次各自的约束，就是这一课要讲的 15 类校验用例。';
    });
  }
},

/* ------------------------------------------------------ 3 三种属性名 */
{
  kicker: 'L1-04 · 边分片',
  title: '三个属性名，各管一类值',
  sub: '按"分片挂在谁身上"分三种：算子本身、算子的结果、块的参数。它们的属性类型要求不同。',
  caption: '类型要求：<span class="mono">sdy.propagation_edges</span> 直接是 <span class="mono">PropagationEdgesAttr</span>；另两个必须是 <span class="mono">ArrayAttr&lt;PropagationEdgesAttr&gt;</span>。',
  code: `// ① 算子级：直接一个 PropagationEdgesAttr
%0 = stablehlo.add %arg0, %arg0
     {sdy.propagation_edges = #sdy.propagation_edges<[...]>}

// ② 结果级：必须是 ArrayAttr<PropagationEdgesAttr>
%0 = ... {sdy.result_propagation_edges = [...]}

// ③ 块参数级：同样是 ArrayAttr<PropagationEdgesAttr>
%0 = ... {sdy.block_arg_propagation_edges = [...]}

// 为什么后两个是数组？因为一个算子可能有多个
// 结果 / 多个块参数，每个值一份边信息。`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%;align-items:center' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:48px;display:flex;align-items:center;text-align:center;max-width:740px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'sdy.propagation_edges', ty: 'PropagationEdgesAttr', d: '挂在算子属性字典上，<br>描述整个算子的一次传播。', c: '#38bdf8' },
      { t: 'sdy.result_propagation_edges', ty: 'ArrayAttr<PropagationEdgesAttr>', d: '每个结果一份。<br><span class="mono">%0:2</span> 就有 2 项。', c: '#c084fc' },
      { t: 'sdy.block_arg_propagation_edges', ty: 'ArrayAttr<PropagationEdgesAttr>', d: '每个块参数一份。<br>用于区域内的传播记录。', c: '#4ade80' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => {
      const e = U.el('div', { class: 'card', style: 'width:236px;opacity:.35;transition:.3s' });
      e.innerHTML = `<div class="card-t mono" style="font-size:11.5px;color:${d.c};overflow-wrap:anywhere">${d.t}</div>
        <div class="mono small dim" style="margin:6px 0;font-size:10.5px;overflow-wrap:anywhere">${U.esc(d.ty)}</div>
        <div class="card-d" style="font-size:11.5px">${d.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.35');
      msg.innerHTML = [
        '最常见的形态。校验器会检查它是不是 <span class="mono">PropagationEdgesAttr</span>。',
        '数组的<b>元素</b>也必须是 <span class="mono">PropagationEdgesAttr</span>，写 <span class="mono">[1,2,3]</span> 会报错。',
        '与结果级用的是<b>同一套</b>类型校验，报文里两者一起出现。',
      ][i];
    }));
    tl.at(11000, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '记住差别：<b>算子级是单值，结果/块参数级是数组</b>。';
    });
  }
},

/* ------------------------------------------------ 4 错误：step 索引 */
{
  kicker: 'L1-04 · 边分片校验',
  title: '错误组 1：<span class="hl-a">step 索引</span>',
  sub: '<span class="mono">step-N</span> 必须唯一且非负 —— 它标识"第几步传播"，重复或负数都无法排序。',
  caption: '网格：<span class="mono">@mesh = &lt;["c"=8, "d"=8, "e"=8]&gt;</span>。这两条是最容易犯的错。',
  code: `// ① 两个 step-1（重复）
#sdy.propagation_edges<[
  {step-1 = [{"c":(1)4 = operand-0 -> [result-0]},
             {"e" = operand-0 -> [result-0]}]},
  {step-1 = [{"d" = operand-1 -> [result-0]}]}     // ✗ 重复
]>
//   propagation edges have duplicate step index: 1

// ② 负数 step
{step--3 = [...]}                                   // ✗
//   propagation edges have negative step index: -3`,
  duration: 12000,
  build(root, tl) {
    const wrap = W.errLayout(root);
    W.errScenes(wrap, tl, [
      {
        ir: '%0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-1 = [{"c":(1)4 = operand-0 -> [result-0]}, {"e" = operand-0 -> [result-0]}]},{step-1 = [{"d" = operand-1 -> [result-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"c":(1)4, ?}]>]>} : tensor<8x8xf32>',
        err: 'propagation edges have duplicate step index: 1',
        why: '两个 <span class="mono">{step-1 = …}</span> 条目。步骤号是<b>排序键</b>，重复后无法判断谁先谁后。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step--3 = [{"c":(1)4 = operand-0 -> [result-0]}, {"e" = operand-0 -> [result-0]}]},{step-1 = [{"d" = operand-1 -> [result-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"c":(1)4, ?}]>]>} : tensor<8x8xf32>',
        err: 'propagation edges have negative step index: -3',
        why: '<span class="mono">step--3</span> 解析成编号 −3。步骤编号从 0 开始递增，负数无意义。'
      },
    ], {
      finalIr: '// step-N 的两条约束：\n//   N >= 0\n//   同一属性内 N 不重复',
      finalErr: 'duplicate step index / negative step index',
      finalWhy: '本质是"步骤必须能唯一排序"。'
    });
  }
},

/* -------------------------------------------- 5 错误：source / target */
{
  kicker: 'L1-04 · 边分片校验',
  title: '错误组 2：<span class="hl-a">source 与 target</span>',
  sub: '四类错误：自环、重复目标、operand 索引越界、result 索引越界。',
  caption: '越界检查依赖<b>宿主算子</b>：<span class="mono">stablehlo.add</span> 有 2 个操作数、1 个结果，所以合法索引分别是 <span class="mono">[0,2)</span> 与 <span class="mono">[0,1)</span>。',
  code: `// ① source 与 target 相同（自环）
{"a" = operand-0 -> [operand-0]}                    // ✗
//   propagation edges have a source that is the same as a target

// ② 同一 step、同一轴下 target 重复
{"x" = result-0 -> [operand-0, operand-0]}          // ✗
//   propagation edges have duplicate targets for
//   step index: 123 and axis: x

// ③ operand 索引越界（add 只有 2 个操作数）
{"z" = result-0 -> [operand-3]}                     // ✗
//   expected a value ref to have an operand index
//   in range [0, 2), got: 3

// ④ result 索引越界（add 只有 1 个结果）
{"z" = operand-1 -> [operand-0, result-1]}          // ✗
//   expected a value ref to have a result index
//   in range [0, 1), got: 1`,
  duration: 15000,
  build(root, tl) {
    const wrap = W.errLayout(root);
    W.errScenes(wrap, tl, [
      {
        ir: '%0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-1 = [{"a" = operand-0 -> [operand-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a":(1)4, ?}]>]>} : tensor<8x8xf32>',
        err: 'propagation edges have a source that is the same as a target',
        why: '<span class="mono">operand-0 -> [operand-0]</span> 是<b>自环</b>：自己传播给自己，不携带任何信息。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-123 = [{"x" = result-0 -> [operand-0, operand-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"x":(1)4, ?}]>]>} : tensor<8x8xf32>',
        err: 'propagation edges have duplicate targets for step index: 123 and axis: x',
        why: '<span class="mono">operand-0</span> 在 targets 里出现两次。同一传播决策的目标应当<b>去重</b>。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-22 = [{"z" = result-0 -> [operand-3]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"z", ?}]>]>} : tensor<8x8xf32>',
        err: "'stablehlo.add' op expected a value ref to have an operand index in range [0, 2), got: 3",
        why: '<span class="mono">stablehlo.add</span> 只有 2 个操作数（索引 0、1），<span class="mono">operand-3</span> 不存在。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-22 = [{"z" = operand-1-> [operand-0,result-1]}]}]>} : tensor<8x8xf32>',
        err: "'stablehlo.add' op expected a value ref to have a result index in range [0, 1), got: 1",
        why: '<span class="mono">stablehlo.add</span> 只有 1 个结果，<span class="mono">result-1</span> 越界。'
      },
    ], {
      finalIr: '// 值引用 = 类型 + 索引\n//   operand-i   索引 < 操作数个数\n//   result-i    索引 < 结果个数\n//   block_arg-i 索引 < 块参数个数',
      finalErr: 'same as a target / duplicate targets / index in range',
      finalWhy: '范围检查由<b>宿主算子</b>决定，所以报文里带着算子名。'
    });
  }
},

/* ------------------------------------ 6 错误：类型、sharding、轴、特例 */
{
  kicker: 'L1-04 · 边分片校验',
  title: '错误组 3：<span class="hl-a">类型 / 引用 / 轴</span>（含一个反例）',
  sub: '最后一组：属性类型写错、没带 sharding、轴不在网格里。另外测试里还有一个<b>故意不报错</b>的反例。',
  caption: '反例很重要：它说明 <span class="mono">sdy.func_data_flow_edge</span> 上<b>越界的 operand 是允许的</b> —— 函数级边没有具体的宿主算子可以查范围。',
  code: `// ① 属性类型错：给了整数
{sdy.propagation_edges = 64}                        // ✗
//   should have a propagation edges attribute of type
//   PropagationEdgesAttr for attr named 'sdy.propagation_edges'

// ② 结果级给了整数
{sdy.result_propagation_edges = 64}                 // ✗
//   ... of type ArrayAttr<PropagationEdgesAttr> for attr named
//   'sdy.result_propagation_edges' or 'sdy.block_arg_propagation_edges'

// ③ 结果级给了错误的数组
{sdy.result_propagation_edges = [1,2,3]}            // ✗ 同上

// ④ block_arg 级给整数 / 错数组
{sdy.block_arg_propagation_edges = 64}              // ✗
{sdy.block_arg_propagation_edges = [0,2,4]}         // ✗

// ⑤ 只有 edges、没有 sharding
{sdy.propagation_edges = #sdy.propagation_edges<[...]>}   // ✗
//   expected propagation edges attr to reference a sharding

// ⑥ 轴不在网格里
{"z" = operand-1 -> [operand-0, result-1]}          // 网格里没有 "z" ✗
//   expected axis ref to be in one of the meshes

// ⑦ 反例：func_data_flow_edge 上越界 operand —— 允许，不报错
sdy.func_data_flow_edge %arg0 {sdy.propagation_edges = ...}`,
  duration: 20000,
  build(root, tl) {
    const wrap = W.errLayout(root);
    W.errScenes(wrap, tl, [
      {
        ir: '%0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = 64} : tensor<8x8xf32>',
        err: "should have a propagation edges attribute of type PropagationEdgesAttr for attr named 'sdy.propagation_edges'",
        why: '<span class="mono">sdy.propagation_edges</span> 必须是 <b>PropagationEdgesAttr</b>，不能是整数。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg0 {sdy.result_propagation_edges = 64} : tensor<8x8xf32>',
        err: "should have a propagation edges attribute of type ArrayAttr<PropagationEdgesAttr> for attr named 'sdy.result_propagation_edges' or 'sdy.block_arg_propagation_edges'",
        why: '结果级必须是<b>数组</b>。注意报文把两个属性名一起列出 —— 它们共用同一套校验。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg0 {sdy.result_propagation_edges = [1,2,3]} : tensor<8x8xf32>',
        err: "should have a propagation edges attribute of type ArrayAttr<PropagationEdgesAttr> for attr named 'sdy.result_propagation_edges' or 'sdy.block_arg_propagation_edges'",
        why: '数组<b>元素</b>也必须是 PropagationEdgesAttr，<span class="mono">1</span> 不是。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg0 {sdy.block_arg_propagation_edges = [0,2,4]} : tensor<8x8xf32>',
        err: "should have a propagation edges attribute of type ArrayAttr<PropagationEdgesAttr> for attr named 'sdy.result_propagation_edges' or 'sdy.block_arg_propagation_edges'",
        why: '<span class="mono">block_arg</span> 版本同样要求数组元素是边分片属性。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-93 = [{"z" = operand-1-> [operand-0,result-0]}]}]>} : tensor<8x8xf32>',
        err: 'expected propagation edges attr to reference a sharding',
        why: '边分片描述的是"<b>某个分片</b>的传播路径"，没有 sharding 就无所指。'
      },
      {
        ir: '%0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-93 = [{"z" = operand-1-> [operand-0,result-1]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"z":(1)4, ?}]>]>} : tensor<8x8xf32>',
        err: 'expected axis ref to be in one of the meshes',
        why: '轴 <span class="mono">"z"</span> 不在 <span class="mono">@mesh = &lt;["c"=8, "d"=8, "e"=8]&gt;</span> 里。'
      },
    ], {
      finalIr: '// 反例（测试中【没有】expected-error）：\n//   sdy.func_data_flow_edge 上的 operand-3 越界是允许的\n//   因为函数级边没有宿主算子可查范围',
      finalErr: '（此用例不期望任何报错）',
      finalWhy: '<b>反例也是测试</b>：它锁定了"函数级边不做范围检查"这一行为，防止被误改。'
    });
  }
},

/* ------------------------------------------------------------ 7 练习 */
{
  kicker: 'L1-04 · 练习',
  title: '练一练：<span class="hl-a">读边分片</span>',
  sub: '三道题分别考：语法解读、错误定位、范围判断。',
  caption: '边分片是调试信息，读懂它就能回答"这个分片是谁引入的"。',
  code: `// 题 1：这条记录说了什么？
#sdy.propagation_edges<[
  {step-2 = [{"y" = operand-1 -> [operand-0, result-0]}]}
]>

// 题 2：错在哪？
#sdy.propagation_edges<[
  {step-5 = [{"x" = operand-0 -> [operand-0]}]}
]>

// 题 3：stablehlo.dot_general 有 2 个操作数、1 个结果，
//        下面哪一项越界？
{"x" = result-0 -> [operand-1]}
{"x" = operand-0 -> [result-1]}`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">{step-2 = [{"y" = operand-1 -&gt; [operand-0, result-0]}]}</span> 说了什么？',
        a: '在第 <b>2</b> 步，轴 <span class="mono">"y"</span> 的分片从源 <span class="mono">operand-1</span>（第 2 个操作数）' +
           '传播到了 <span class="mono">operand-0</span>（第 1 个操作数）与 <span class="mono">result-0</span>（结果）。' +
           '<br><span class="dim">一个源可以同时流向多个目标。</span>'
      },
      {
        q: '<span class="mono">{step-5 = [{"x" = operand-0 -&gt; [operand-0]}]}</span> 错在哪？',
        a: '<b class="badge bad">自环</b> source 与 target 都是 <span class="mono">operand-0</span>。' +
           '报错：<span class="mono">propagation edges have a source that is the same as a target</span>。' +
           '<br><span class="dim">自己传播给自己不携带信息，因此被禁止。</span>'
      },
      {
        q: '<span class="mono">dot_general</span>（2 操作数 / 1 结果）中哪个值引用越界？<br>' +
           '<span class="mono">A</span> <span class="mono">result-0 → [operand-1]</span> &nbsp;·&nbsp; ' +
           '<span class="mono">B</span> <span class="mono">operand-0 → [result-1]</span>',
        a: '<b>B 越界</b>：<span class="mono">result-1</span> 超出结果范围 <span class="mono">[0,1)</span>；' +
           '<span class="mono">operand-1</span> 落在 <span class="mono">[0,2)</span> 内，合法。'
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
