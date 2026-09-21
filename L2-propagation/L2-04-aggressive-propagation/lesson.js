/* ==========================================================================
   L2-04 · aggressive-propagation
   --------------------------------------------------------------------------
   覆盖：transforms/propagation/test/aggressive_propagation.mlir (395 行 / 23 用例)
   目标：讲透"主动消解冲突"的那一层 —— 它能消解什么、不能消解什么。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 层级定位 */
{
  kicker: 'L2-04 · 激进传播',
  title: '激进传播：<span class="hl-a">用通信换显存</span>',
  sub: '基础传播推不动就放弃。激进传播多走一步：<b>主动消解冲突</b>，代价是可能多出通信。',
  caption: '它在传播金字塔的第 2 层（见 L2-01 的层级图）：基础 → <b>激进</b> → 算子优先级 → 用户优先级。',
  code: `// 基础传播：只传所有张量都同意的轴
%0 = stablehlo.add %arg0, %arg1 {sdy.sharding = ...}

// 激进传播：多一个策略选项
// RUN: sdy_opt %s -sdy-aggressive-propagate="propagation-strategy=aggressive"

// 它能做的事：
//   ① 识别【假冲突】—— 看似冲突，其实落在不同因子上
//   ② 真冲突时按启发式选一边（如"分片更多的那一侧"）
//   ③ 【侧向传播】—— 结果被锁死时，在操作数之间同步

// 它【不能】做的事：
//   两边都是用户标注、且要求不同的轴 -> 仍然放弃`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 识别假冲突', d: '同一个轴落在<b>不同因子</b>上 —— 各切各的，不是冲突。', c: '#4ade80' },
      { t: '② 真冲突选边', d: '按启发式挑一侧，如<b>优先分片更多</b>的那个因子。', c: '#38bdf8' },
      { t: '③ 侧向传播', d: '结果被锁死时，改为在<b>操作数之间</b>横向同步。', c: '#c084fc' },
      { t: '④ 消解不了就放弃', d: '两边都是用户标注且互不相容 → 不猜。', c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:182px;opacity:.32;transition:.35s;border-color:' + x.c + '55;padding:9px' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:11.5px">${x.t}</div>
        <div class="card-d" style="font-size:11px;line-height:1.5">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.32');
      msg.innerHTML = [
        '第 3 幕会看到这个例子：<span class="mono">"a"</span> 在操作数上切因子 <span class="mono">i</span>，在结果上切因子 <span class="mono">j</span>。',
        '对应用例 <span class="mono">prefer_most_sharded_factor_*</span>：分得越细，每台设备的数据越少。',
        '<span class="mono">sideways_propagation_*</span> 共 6 个用例，覆盖结果被各种方式锁死的情形。',
        '<b>这一点很重要</b>：激进 ≠ 万能。它有一组明确的启发式，不在其中的冲突就不动。',
      ][i];
    }));
    tl.at(13000, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '本课 23 个用例按这四类归纳 —— 前三类是"能消解"，第四类是"消解不了"。';
    });
  }
},

/* ------------------------------------------------------ 2 无冲突 */
{
  kicker: 'L2-04 · 激进传播',
  title: '无冲突时：与基础传播<span class="hl-a">逐字相同</span>',
  sub: '激进是<b>叠加</b>在基础之上的策略，不是替换。没有冲突时两者输出完全一致。',
  caption: '<span class="mono">@no_conflict</span> 用例与 <span class="mono">basic_propagation.mlir</span> 的 <span class="mono">@simple</span> 结果逐字相同 —— 这本身就是一种回归测试。',
  code: `// aggressive_propagation.mlir @no_conflict
%arg0: tensor<8x8xf32> [{"a"}, {"b"}]
%arg1: tensor<8x8xf32>        （无分片）
%arg2: tensor<8x16xf32>       （无分片）

%0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
%1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0]

// 结果与 basic_propagation.mlir 的 @simple 完全一致：
//   add -> [{"a", ?}, {"b", ?}]
//   dot -> [{"a", ?}, {?}]`,
  duration: 12000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:16px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:30px;justify-content:center;align-items:center">
        <div class="col" style="gap:7px;align-items:center">
          <div class="small mono" style="color:var(--ax0)">basic_propagation</div>
          <div class="chip c0" style="padding:8px 14px;font-size:13px">add → [{"a",?},{"b",?}]</div>
          <div class="chip c0" style="padding:8px 14px;font-size:13px">dot → [{"a",?},{?}]</div>
        </div>
        <div class="big" style="font-size:26px;color:var(--ok)">≡</div>
        <div class="col" style="gap:7px;align-items:center">
          <div class="small mono" style="color:var(--ok)">aggressive_propagation</div>
          <div class="chip c4" style="padding:8px 14px;font-size:13px">add → [{"a",?},{"b",?}]</div>
          <div class="chip c4" style="padding:8px 14px;font-size:13px">dot → [{"a",?},{?}]</div>
        </div>
      </div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const msg = wrap.querySelector('#msg');
    tl.at(900, () => { msg.innerHTML = '两种策略在这里给出<b>完全相同</b>的结果。'; });
    tl.at(3600, () => { msg.innerHTML = '因为 <span class="mono">%arg0</span> 的分片与两个算子都兼容，没有任何需要消解的东西。'; });
    tl.at(6600, () => { msg.innerHTML = '<b>理解方式</b>：激进传播 = 基础传播 + 一组额外的消解规则。没触发规则时两者等价。'; });
    tl.at(9600, () => { msg.innerHTML = '这也意味着：切换到激进模式<b>不会让原本正确的结果变差</b>，只会让更多张量拿到分片。'; });
  }
},

/* ---------------------------------------------------- 3 假冲突 */
{
  kicker: 'L2-04 · 激进传播',
  title: '假冲突：<span class="hl-a">同一个轴，不同因子</span>',
  sub: '看起来像冲突（同一个轴出现在两个张量上，位置不同），但在因子层面它们毫无关系。',
  caption: '<span class="mono">dot_general</span> 的 <span class="mono">contracting_dims = [1] x [1]</span> 是关键：它让因子映射变成 <span class="mono">([i,k],[j,k])-&gt;([i,j])</span>。',
  code: `// 因子映射：([i, k], [j, k]) -> ([i, j])
//   %arg0 (256x512) = (i, k)
//   %arg1 (128x512) = (j, k)
//   result(256x128) = (i, j)

// %arg0 第 0 维 = {"a", ?}   -> "a" 切因子 i
// result 第 1 维 = {"a", ?}  -> "a" 切因子 j

// 同一个轴 "a"，但一个切 i、一个切 j ——【不是冲突】

// 结果：照单全收
//   dot_general {sharding_per_value = [<@mesh, [{?}, {"a", ?}]>]}

// 对比：如果一个轴在同一个因子上要求两种切法 -> 那才是真冲突`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const box = (title, cells, color) => {
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center' });
      c.innerHTML = `<div class="small mono" style="color:${color}">${title}</div>
        <div class="row" style="gap:5px" data-d></div>`;
      const h = c.querySelector('[data-d]');
      cells.forEach(([t, cls]) => {
        const e = U.el('div', { class: 'chip ' + cls, style: 'padding:7px 12px;font-size:13px' });
        e.textContent = t; h.appendChild(e);
      });
      demo.appendChild(c); return c;
    };

    tl.at(700, () => {
      box('%arg0 (256x512) = (i, k)', [['i:{"a"}', 'c0'], ['k', 'mut']], 'var(--ax0)');
      msg.innerHTML = '<span class="mono">"a"</span> 切的是因子 <span class="mono">i</span>。';
    });
    tl.at(4000, () => {
      box('result (256x128) = (i, j)', [['i', 'mut'], ['j:{"a"}', 'c1']], 'var(--ax1)');
      msg.innerHTML = '结果里 <span class="mono">"a"</span> 切的是因子 <span class="mono">j</span>。';
    });
    tl.at(7400, () => {
      msg.innerHTML = '<b>i 与 j 是不同的因子</b> —— 各切各的，两个张量之间没有任何需要协调的地方。';
    });
    tl.at(10600, () => {
      msg.innerHTML = '所以激进传播<b>原样接受</b>：<span class="mono">[<@mesh, [{?}, {"a", ?}]>]</span>。';
    });
    tl.at(13200, () => {
      msg.innerHTML = '<b>判据</b>：冲突要看<b>因子</b>，不是看"轴上有没有同名"。同名不同因子 = 假冲突。';
    });
  }
},

/* ---------------------------------------------------- 4 真冲突 */
{
  kicker: 'L2-04 · 激进传播',
  title: '真冲突：同一因子上<span class="hl-a">两个不同的轴</span> → 不消解',
  sub: '两边都是用户标注、且要求不同的轴。激进传播<b>不会猜</b> —— 结果上直接没有分片。',
  caption: '这是本课最反直觉的一点：<b>激进 ≠ 消解一切冲突</b>。它只在启发式覆盖的情形下动手。',
  code: `// 逐元素算子：([i, j], [i, j]) -> ([i, j])
//   %arg0 第 0 维 = {"a", ?}   -> 因子 i 沿 "a" 切
//   %arg1 第 0 维 = {"b", ?}   -> 因子 i 沿 "b" 切

func.func @real_conflict_within_a_factor(
    %arg0: tensor<8x8xf32>
      {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {}]>},
    %arg1: tensor<8x8xf32>
      {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"b", ?}, {}]>})
    -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// 期望：
//   stablehlo.add %arg0, %arg1
//   CHECK-NOT: sdy.sharding      <- 没有任何分片

// 为什么不动手：两边都是【用户标注】，覆盖任何一边
// 都等于擅自修改用户的意图`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:15px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:8px;align-items:center' });
      c.innerHTML = `
        <div class="row" style="gap:12px;align-items:center">
          <div class="chip c0" style="padding:8px 13px">%arg0: i ← "a"</div>
          <div style="color:var(--bad);font-size:20px">⚡</div>
          <div class="chip c1" style="padding:8px 13px">%arg1: i ← "b"</div>
        </div>
        <div class="small faint">同一个因子 i，两个不同的轴</div>`;
      demo.appendChild(c);
      msg.innerHTML = '两个操作数都<b>用户标注</b>了第 0 维的分片，但用的轴不同。';
    });
    tl.at(4200, () => {
      msg.innerHTML = '<b>为什么不能选边</b>：无论采用 <span class="mono">"a"</span> 还是 <span class="mono">"b"</span>，都等于<b>覆盖用户的一个标注</b>。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '所以结果上<b>没有分片</b> —— 测试用 <span class="mono">CHECK-NOT: sdy.sharding</span> 精确断言。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>对比上一幕</b>：假冲突落在不同因子上，可以两者都保留；真冲突必须在同一因子上做选择，而这里没有可依据的启发式。';
    });
    tl.at(13400, () => {
      msg.innerHTML = '<b>实践建议</b>：遇到这种情形，通常说明用户的标注本身需要调整 —— 两个操作数本就该切成一样。';
    });
  }
},

/* ---------------------------------------------------- 5 侧向传播 */
{
  kicker: 'L2-04 · 激进传播',
  title: '侧向传播：结果被锁死时走<span class="hl-a">操作数之间</span>',
  sub: '正常传播是"操作数 → 结果"。若结果被显式锁死，激进传播会改为在操作数之间横向同步。',
  caption: '测试里共有 <b>6 个</b>侧向传播用例，覆盖结果被锁成闭空 / 开空 / 子轴 / 部分冲突等各种情形。',
  code: `// %arg0 有分片 {"a"}，结果被显式标注为空闭维 {}
func.func @sideways_propagation_if_result_is_closed_empty(
    %arg0: tensor<8xf32>
      {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}]>},
    %arg1: tensor<8xf32>)
    -> tensor<8xf32> {
  %0 = stablehlo.add %arg0, %arg1
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{}]>]>}
      : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// 结果：保留 [{}] —— 不把 "a" 强加到结果上

// "侧向"的意思：
//   分片不【穿过】结果继续传
//   而是在两个【操作数】之间横向对齐

// 相关用例：
//   allow_sideways_propagation_if_result_is_open_empty
//   sideways_propagation_if_result_is_closed_sub_axis
//   allow_partial_sideways_propagation_if_conflicting_with_result
//   allow_sideways_propagation_if_result_fully_matches
//   allow_sideways_propagation_if_no_conflicting_with_one_of_multiple_results`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:30px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:8px;align-items:center' });
      c.innerHTML = `
        <div class="row" style="gap:10px;align-items:center">
          <div class="chip c0" style="padding:8px 13px">%arg0: {"a"}</div>
          <div class="chip mut" style="padding:8px 13px">%arg1</div>
        </div>
        <div style="font-size:20px;color:var(--ink-faint)">↓ 正常传播方向</div>
        <div class="chip c3" style="padding:8px 13px">result: {} （被锁死）</div>`;
      demo.appendChild(c);
      msg.innerHTML = '结果被显式锁在 <span class="mono">{}</span>（空的闭维）—— 传播<b>不能改</b>它。';
    });
    tl.at(4400, () => {
      demo.innerHTML = `
        <div class="row" style="gap:10px;align-items:center">
          <div class="chip c0" style="padding:8px 13px">%arg0: {"a"}</div>
          <div style="font-size:20px;color:var(--ok)">⟷</div>
          <div class="chip mut" style="padding:8px 13px">%arg1</div>
        </div>
        <div class="small faint">侧向：在操作数之间同步</div>`;
      msg.innerHTML = '<b>改为侧向</b>：既然结果这条路走不通，就让操作数之间互相对齐。';
    });
    tl.at(8000, () => {
      msg.innerHTML = '在这个用例里 <span class="mono">%arg0</span> 已经带 <span class="mono">"a"</span>，结果保持 <span class="mono">{}</span> 不变。';
    });
    tl.at(11200, () => {
      msg.innerHTML = '<b>为什么需要它</b>：用户显式锁死结果时，仍希望输入侧的分片能发挥作用（至少让两个操作数一致）。';
    });
    tl.at(13600, () => {
      msg.innerHTML = '6 个用例的差别在于"结果被锁成什么样"以及"是否与某一侧完全匹配"。';
    });
  }
},

/* ------------------------------------------------ 6 优先分片最多 */
{
  kicker: 'L2-04 · 激进传播',
  title: '启发式：优先选<span class="hl-a">分片最多</span>的一侧',
  sub: '真冲突必须选边时，激进传播的默认倾向是"分得更细的那一侧" —— 因为分得越细，每台设备的数据越少。',
  caption: '这正是官方文档说的 "aggressive strategy ... can reduce the memory footprint at the cost of potential communication"。',
  code: `// %arg0 第 1 维 = {"a"}        —— 用 1 个轴
// %arg1 第 0 维 = {"b", "a"}   —— 用 2 个轴
func.func @prefer_most_sharded_factor_elementwise_op(
    %arg0: tensor<8x8xf32>
      {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{}, {"a"}]>},
    %arg1: tensor<8x8xf32>
      {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"b", "a"}, {}]>})
    -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// 结果采用分片更多的那一侧：
//   add -> [{"b", "a", ?}, {?}]

// 同族 4 个用例：
//   prefer_most_sharded_factor_elementwise_op / _op2
//   prefer_most_sharded_factor_on_operand_elementwise_op
//   prefer_lhs_factor_non_elementwise`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:14px;justify-content:center;align-items:stretch" id="rows"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const rows = wrap.querySelector('#rows'), msg = wrap.querySelector('#msg');

    const mk = (title, chips, note, color) => {
      const c = U.el('div', { class: 'card', style: 'width:330px;opacity:.35;transition:.4s;border-color:' + color + '55' });
      c.innerHTML = `<div class="card-t" style="font-size:12px;color:${color}">${title}</div>
        <div class="row" style="gap:5px;margin:7px 0">${chips.map(x => `<div class="chip ${x.cls}" style="padding:5px 10px;font-size:12px">${x.v}</div>`).join('')}</div>
        <div class="small faint" style="font-size:11.5px">${note}</div>`;
      rows.appendChild(c); return c;
    };

    tl.at(700, () => {
      mk('%arg0 第 1 维', [{ v: '"a"', cls: 'c0' }], '1 个轴', 'var(--ax0)').style.opacity = '1';
      msg.innerHTML = '<span class="mono">%arg0</span> 第 1 维只用了 <span class="mono">"a"</span> 一个轴。';
    });
    tl.at(4000, () => {
      mk('%arg1 第 0 维', [{ v: '"b"', cls: 'c1' }, { v: '"a"', cls: 'c0' }], '2 个轴（分得更细）', 'var(--ok)').style.opacity = '1';
      msg.innerHTML = '<span class="mono">%arg1</span> 第 0 维用了 <span class="mono">"b"</span> 与 <span class="mono">"a"</span> 两个轴。';
    });
    tl.at(7400, () => {
      msg.innerHTML = '<b>结果</b>：<span class="mono">[{"b", "a", ?}, {?}]</span> —— 采用了分片更多的一侧。';
    });
    tl.at(10600, () => {
      msg.innerHTML = '<b>代价</b>：另一侧（<span class="mono">%arg0</span>）可能需要通信来对齐 —— 这就是"用通信换显存"。';
    });
    tl.at(13000, () => {
      msg.innerHTML = '非逐元素算子用另一条启发式（<span class="mono">prefer_lhs_factor_non_elementwise</span>）—— 值得单独看一眼。';
    });
  }
},

/* ------------------------------------------ 7 未归约轴阻断反向 */
{
  kicker: 'L2-04 · 激进传播',
  title: '未归约轴会<span class="hl-a">阻断反向传播</span>',
  sub: '<span class="mono">unreduced={"b"}</span> 表示"这个轴上是部分和"。反向传播不能把它当完整值回推。',
  caption: '但<b>前向传播照常</b> —— 下游算子继承的是同一份未归约值，仍然可以继续往前推。',
  code: `%arg0: tensor<8x8xf32> [{"a"}, {"b"}]
%arg1: tensor<8x16xf32>

%0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
     {sdy.sharding = #sdy.sharding_per_value<
        [<@mesh_a_2_b_2, [{?}, {?}], unreduced={"b"}>]>}
%1 = stablehlo.add %0, %0 : tensor<8x16xf32>

// 结果：
//   dot_general -> [{"a", ?}, {?}], unreduced={"b"}
//   add         -> [{"a", ?}, {"b", ?}]

// 关键：dot 的结果带 unreduced={"b"}
//   反向推不动（"b" 是部分和，不是完整值）
//   前向照常（add 继承同一份未归约值）`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:22px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'row', style: 'gap:12px;align-items:center' });
      c.innerHTML = `
        <div class="chip mut" style="padding:8px 13px">%arg0</div>
        <div class="arrow" style="font-size:20px">→</div>
        <div class="chip c2" style="padding:8px 13px">dot_general</div>
        <div class="arrow" style="font-size:20px">→</div>
        <div class="chip c4" style="padding:8px 13px">add</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<span class="mono">dot_general</span> 切了收缩维 → 结果是<b>未归约</b>的（部分和）。';
    });
    tl.at(4200, () => {
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="row" style="gap:10px">
          <div class="chip c4" style="padding:6px 12px">add ← dot</div>
          <span style="color:var(--ok);font-size:18px">✓ 前向通</span>
        </div>
        <div class="row" style="gap:10px">
          <div class="chip c3" style="padding:6px 12px">dot ← add</div>
          <span style="color:var(--bad);font-size:18px">⤫ 反向阻断</span>
        </div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>未归约轴让反向传播失效</b>：不能把"部分和"当成完整值回推到操作数。';
    });
    tl.at(7800, () => {
      msg.innerHTML = '所以 <span class="mono">dot_general</span> 的分片只能由<b>前向</b>决定：<span class="mono">[{"a", ?}, {?}], unreduced={"b"}</span>。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<span class="mono">add</span> 继承同一份未归约值，于是拿到 <span class="mono">[{"a", ?}, {"b", ?}]</span> —— 注意它不含 <span class="mono">unreduced</span>。';
    });
    tl.at(13400, () => {
      msg.innerHTML = '<b>实践含义</b>：切收缩维能省显存，但在 all-reduce 之前这一路的分片决策更"单向"。';
    });
  }
},

/* ------------------------------------------------------------ 8 练习 */
{
  kicker: 'L2-04 · 练习',
  title: '练一练：<span class="hl-a">它会不会消解？</span>',
  sub: '三道题分别考：假冲突、真冲突、启发式选择。',
  caption: '核心判据只有一条：<b>先看落在哪个因子，再看有没有可依据的启发式</b>。',
  code: `// 题 1：这是假冲突还是真冲突？
//   %arg0 第 0 维 = {"a"}（因子 i）
//   结果  第 0 维 = {"a"}（同一个因子 i）

// 题 2：激进传播会怎么办？
//   %arg0 第 0 维 = {"a"}
//   %arg1 第 0 维 = {"b"}
//   逐元素算子

// 题 3：结果被锁成 {} 时，分片还能传吗？`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">%arg0</span> 第 0 维和结果第 0 维都要求 <span class="mono">{"a"}</span>（同一个因子 i），这是冲突吗？',
        a: '<b class="badge ok">不是冲突</b> —— 同一个因子上<b>同一个轴</b>，两边要求完全一致。' +
           '<br>传播会直接确认这个分片，连"消解"都不需要。' +
           '<br><span class="dim">冲突的定义是"同一因子上出现了不兼容的要求"，而不是"轴上出现了同名"。</span>'
      },
      {
        q: '<span class="mono">%arg0</span> 第 0 维 <span class="mono">{"a"}</span>、<span class="mono">%arg1</span> 第 0 维 <span class="mono">{"b"}</span>（逐元素算子），激进传播会怎么办？',
        a: '<b class="badge bad">不传播</b> —— 结果上没有分片。' +
           '<br><b>理由</b>：同一个因子 <span class="mono">i</span> 上两个<b>不同的轴</b>，且两边都是<b>用户标注</b>。' +
           '覆盖任何一边都等于擅自修改用户意图，而这里也没有"谁分片更多"可作为依据。' +
           '<br><span class="dim">这正是"激进 ≠ 消解一切冲突"的例子。</span>'
      },
      {
        q: '结果被显式锁成空的闭维 <span class="mono">{}</span> 时，输入侧的分片还能起作用吗？',
        a: '<b class="badge ok">能，但要换条路</b> —— 走<b>侧向传播</b>。' +
           '<br>结果锁死了，"操作数 → 结果"这条正向路走不通；激进传播改为在<b>操作数之间</b>横向对齐。' +
           '<br><span class="dim">侧向传播有 6 个测试用例，覆盖结果被锁成闭空 / 开空 / 子轴 / 部分冲突等情形。</span>'
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
