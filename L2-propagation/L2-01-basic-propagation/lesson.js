/* ==========================================================================
   L2-01 · basic-propagation   （L2 层开篇，本计划最大单课）
   --------------------------------------------------------------------------
   覆盖：transforms/propagation/test/basic_propagation.mlir (1106 行 / 79 用例)
   目标：讲透最保守的传播策略 —— 只传"所有人都同意"的轴，绝不消解冲突。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ---------------------------------------------------- 1 什么是传播 */
{
  kicker: 'L2-01 · 基础传播',
  title: '传播：从少数约束推出<span class="hl-a">全图分片</span>',
  sub: '程序里只有几个张量带分片标注，其余全是空的。传播沿着数据流反复推，直到不再变化（不动点）。',
  caption: '这是 L2 层的第一课。<b>基础传播</b>是最保守的策略：只传"所有操作数与结果都同意"的轴，绝不制造冲突。',
  code: `// RUN: sdy_opt %s -split-input-file -sdy-basic-propagate \\
//        -verify-diagnostics | FileCheck %s

// 只有 %arg0 带分片，其余都没有
func.func @simple(%arg0: tensor<8x8xf32>
      {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>},
      %arg1: tensor<8x8xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0] :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// 传播后：add 与 dot_general 都拿到了分片`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:15px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:10px;align-items:center;justify-content:center" id="chain"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const chain = wrap.querySelector('#chain'), msg = wrap.querySelector('#msg');

    const node = (label, sub) => {
      const c = U.el('div', { class: 'card', style: 'padding:8px 12px;text-align:center;transition:.4s' });
      c.innerHTML = `<div class="mono" style="font-size:12px">${label}</div>
        <div class="small faint" style="font-size:10.5px">${sub}</div>`;
      chain.appendChild(c); return c;
    };
    const arrow = () => chain.insertAdjacentHTML('beforeend', '<div class="arrow" style="font-size:20px">→</div>');

    tl.at(700, () => {
      const a = node('%arg0', '有分片 [a][b]');
      a.style.borderColor = 'var(--accent)';
      arrow();
      const b = node('%0 = add', '空');
      arrow();
      const c = node('%1 = dot_general', '空');
      arrow();
      node('return', '空');
      msg.innerHTML = '只有第一个张量有信息 —— 其余都要靠传播推出来。';
    });
    tl.at(4200, () => {
      const cards = chain.querySelectorAll('.card');
      if (cards[1]) { cards[1].style.borderColor = 'var(--ok)'; cards[1].querySelector('.small').textContent = '继承 [a][b]，都变成开维'; }
      msg.innerHTML = '<b>第 1 步</b>：<span class="mono">add</span> 是逐元素算子，两维一一对应 → 直接继承，但加上 <span class="mono">?</span>（变成开维）。';
    });
    tl.at(8000, () => {
      const cards = chain.querySelectorAll('.card');
      if (cards[2]) { cards[2].style.borderColor = 'var(--ok)'; cards[2].querySelector('.small').textContent = '继承 a，第 1 维仍开放'; }
      msg.innerHTML = '<b>第 2 步</b>：<span class="mono">dot_general</span> 的 <span class="mono">i</span> 因子对应结果的第 0 维 → 继续传下去。';
    });
    tl.at(11200, () => {
      msg.innerHTML = '重复这个过程（前向 + 反向）直到<b>不再变化</b> —— 这就是不动点，传播结束。';
    });
  }
},

/* -------------------------------------------------- 2 开维与闭维 */
{
  kicker: 'L2-01 · 基础传播',
  title: '传播中的<span class="hl-a">开维</span>与<span class="hl-a">闭维</span>',
  sub: '闭维（用户指定的那一维）传播不能改，但<b>传播自己加上的部分仍然开放</b> —— 这是最容易看错的一处。',
  caption: '结果形如 <span class="mono">{"b", ?}</span>：既有用户锁定的 <span class="mono">"b"</span>，又带着"还能继续加"的 <span class="mono">?</span>。',
  code: `// %arg1 第 1 维是闭的 {"b"}
func.func @closed_dim(%arg0: tensor<8x8xf32>
      {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>},
      %arg1: tensor<8x16xf32>
      {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{}, {"b"}]>})
      -> tensor<8x16xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{}, {?}]>]>} :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// 结果的第 1 维得到 {"b", ?}：
//   "b" 来自 %arg1（闭的部分）
//   ?   是传播加上的开放标记`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:16px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:30px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const col = (title, chips, color) => {
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center' });
      c.innerHTML = `<div class="small mono" style="color:${color}">${title}</div>
        <div class="row" style="gap:6px" data-d></div>`;
      const h = c.querySelector('[data-d]');
      chips.forEach(([t, cls]) => {
        const e = U.el('div', { class: 'chip ' + cls, style: 'padding:7px 12px;font-size:13px' });
        e.textContent = t; h.appendChild(e);
      });
      demo.appendChild(c); return c;
    };

    tl.at(700, () => {
      col('%arg1 第 1 维（输入）', [['"b"', 'c1']], 'var(--ax1)');
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹' }));
      col('结果第 1 维（输出）', [['"b"', 'c1'], ['?', 'mut']], 'var(--ok)');
      msg.innerHTML = '闭维 <span class="mono">"b"</span> 被继承，但结果里<b>多了一个 <span class="mono">?</span></b>。';
    });
    tl.at(4400, () => {
      msg.innerHTML = '<b>为什么加 ?</b>：传播加上的分片默认是<b>开放</b>的 —— 它还允许后续再叠加别的轴。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '对比：<span class="mono">{"b"}</span>（不带 ?）表示"到此为止，不许再加"；<span class="mono">{"b", ?}</span> 表示"已按 b 切，还可继续"。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>记忆</b>：闭维是<b>用户</b>锁的，<span class="mono">?</span> 是<b>传播</b>留的 —— 两者可以同时存在。';
    });
  }
},

/* ---------------------------------------------------- 3 冲突消解 */
{
  kicker: 'L2-01 · 基础传播',
  title: '冲突：只传<span class="hl-a">最长兼容前缀</span>',
  sub: '当多个操作数对同一维的要求不一致时，基础传播<b>不做任何消解</b> —— 能传多少传多少，传不了的放弃。',
  caption: '这正是"basic"与"aggressive"的分界：基础传播<b>绝不制造冲突</b>，代价是可能推不动。L2-04 会讲激进传播怎么处理。',
  code: `// 三个输入对第 0 维的要求：
//   %arg0: {"b", "a"}
//   %arg1: {"b", "c"}   -> 与 %arg0 的最长兼容前缀 = "b"
//   %arg2: {"a", "b"}   -> 与 %arg0 的轴序相反，一个都不兼容
func.func @multi_axes_conflict(...) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.add %arg0, %arg2 : tensor<8x8xf32>
  return %0, %1 : tensor<8x8xf32>, tensor<8x8xf32>
}

// 传播结果：
//   add %arg0, %arg1 -> [{"b", ?}, {?}]    只传 "b"
//   add %arg0, %arg2 -> 没有 sharding      一个都不传`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `
      <div class="row" style="gap:14px;justify-content:center;align-items:stretch" id="rows"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const rows = wrap.querySelector('#rows'), msg = wrap.querySelector('#msg');

    const mkRow = (title, seq, result, color) => {
      const c = U.el('div', { class: 'card', style: 'width:360px;opacity:.35;transition:.4s;border-color:' + color + '55' });
      c.innerHTML = `<div class="card-t" style="font-size:12px;color:${color}">${title}</div>
        <div class="row" style="gap:5px;flex-wrap:wrap;margin:6px 0">
          ${seq.map((t, i) => `<span class="chip ${t.cls}" style="padding:4px 9px;font-size:11.5px">${t.v}</span>${i < seq.length - 1 ? '<span class="faint">·</span>' : ''}`).join('')}
        </div>
        <div class="small" style="font-size:11.5px">${result}</div>`;
      rows.appendChild(c); return c;
    };

    tl.at(700, () => {
      mkRow('add %arg0, %arg1 → 兼容前缀 "b"',
        [{ v: '"b"', cls: 'c1' }, { v: '"a"', cls: 'mut' }],
        '公共前缀 <span class="mono">"b"</span> → 只传它', 'var(--ok)').style.opacity = '1';
      msg.innerHTML = '<span class="mono">%arg0 = {"b","a"}</span> 与 <span class="mono">%arg1 = {"b","c"}</span> 的公共前缀是 <span class="mono">"b"</span>。';
    });
    tl.at(4200, () => {
      msg.innerHTML = '于是两个操作数的这一维都变成 <span class="mono">{"b", ?}</span> —— <b>兼容的部分传下去，不兼容的丢掉</b>。';
    });
    tl.at(7600, () => {
      mkRow('add %arg0, %arg2 → 轴序相反，无公共前缀',
        [{ v: '"b"', cls: 'mut' }, { v: '"a"', cls: 'mut' }],
        '无公共前缀 → <b>一个都不传</b>', 'var(--bad)').style.opacity = '1';
      msg.innerHTML = '<span class="mono">{"b","a"}</span> 与 <span class="mono">{"a","b"}</span> 顺序相反 —— 逐位比较第一个就不等，交集为空。';
    });
    tl.at(11200, () => {
      msg.innerHTML = '这就是"<b>最长兼容主分片轴</b>"：从 major 开始逐位比，遇到不一致就停。';
    });
    tl.at(14000, () => {
      msg.innerHTML = '<b>基础传播的边界</b>：它不会为了推动而改变任何已有分片 —— 所以像上例第二行那样"推不动"是正常的。';
    });
  }
},

/* ------------------------------------------------------ 4 reshape */
{
  kicker: 'L2-01 · 基础传播',
  title: 'reshape：沿<span class="hl-a">因子</span>而不是维度传播',
  sub: '这是"为什么要有因子"最直观的例子：两个维度合并后，轴的对应关系只能在因子层面说清。',
  caption: '被切的如果是<b>最 major 的因子</b>，合并后仍能对应；若不是，就需要拆子轴（下面第 6 幕）。',
  code: `// 2x4 -> 8：两个维度合并成一个
//   因子映射：([i, j]) -> ([ij])
//   %arg0 第 0 维沿 "a" 切 —— "a" 切的是因子 i
func.func @reshape_merge_dim_only_major_most_factor_sharded(
    %arg0: tensor<2x4xf32>
      {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>})
    -> tensor<8xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// 结果：第 0 维得到 {"a", ?}
//   "a" 切 i，i 在合并后的 [ij] 里仍然是最 major
//   -> 对应关系保持，不需要通信

// 反例：如果 "a" 切的是 j（非最 major）
//   -> 合并后 j 的位置变了，无法直接对应
//   -> 需要拆子轴（见下一幕）`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:15px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:22px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const box = (title, dims, color) => {
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center' });
      c.innerHTML = `<div class="small mono" style="color:${color}">${title}</div>
        <div class="row" style="gap:6px" data-d></div>`;
      const h = c.querySelector('[data-d]');
      dims.forEach(([t, cls]) => {
        const e = U.el('div', { class: 'chip ' + cls, style: 'padding:7px 13px;font-size:14px' });
        e.textContent = t; h.appendChild(e);
      });
      demo.appendChild(c); return c;
    };

    tl.at(700, () => {
      box('reshape 前 tensor<2x4>（因子 i, j）', [['i', 'c0'], ['j', 'mut']], 'var(--ax0)');
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹' }));
      box('reshape 后 tensor<8>（因子 ij）', [['ij', 'c0']], 'var(--ok)');
      msg.innerHTML = '<span class="mono">"a"</span> 切的是因子 <span class="mono">i</span>（最 major）。';
    });
    tl.at(4400, () => {
      msg.innerHTML = '合并成 <span class="mono">[ij]</span> 后，<span class="mono">i</span> 仍在这个复合因子里的<b>最高位</b> —— 分片信息能对上。';
    });
    tl.at(7800, () => {
      msg.innerHTML = '所以结果第 0 维写成 <span class="mono">{"a", ?}</span>：<b>数据不用移动</b>，reshape 零通信。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<b>如果切的是 j</b>：合并后 j 落在复合因子的低位，无法用一个轴精确描述 → 需要拆成子轴（L1-02）。';
    });
    tl.at(13600, () => {
      msg.innerHTML = '测试文件里 reshape 有 <b>20 余个变体</b>，穷举了"切哪个因子 × 怎么合并/拆分"的组合。';
    });
  }
},

/* -------------------------------------------------------- 5 子轴 */
{
   kicker: 'L2-01 · 基础传播',
  title: '子轴：完整轴<span class="hl-a">落到</span>子轴上',
  sub: '当目标位置只能容纳轴的一部分时，传播会用<b>子轴</b>表达对应关系。',
  caption: '<span class="mono">"a":(1)2</span> 表示"取 a 的前半段"。完整轴 <span class="mono">"a"</span> 与它对应，因为前者包含后者。',
  code: `// 网格 a=4，张量第 0 维大小 32
//   %arg0 第 0 维 = {"a", ?}   （完整轴）
//   结果显式要求 = {"a":(1)2, ?}（只要前半段）
func.func @propagate_full_to_sub_axis(
    %arg0: tensor<32x8xf32>
      {sdy.sharding = #sdy.sharding<@mesh_a_4_b_2, [{"a", ?}, {}]>})
    -> tensor<32x8xf32> {
  %0 = stablehlo.add %arg0, %arg0
      {sdy.sharding = #sdy.sharding_per_value<
         [<@mesh_a_4_b_2, [{"a":(1)2, ?}, {}]>]>}
      : (tensor<32x8xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
  return %0 : tensor<32x8xf32>
}

// 为什么能对应：a 拆成 a:(1)2 与 a:(2)2
//   完整轴 a = a:(1)2 x a:(2)2
//   所以"用 a 切"包含"用 a:(1)2 切"

// 测试里还有 sub_axes_cannot_coexist：
//   两个不兼容的子轴同时出现 -> 不传播`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:15px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:22px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center' });
      c.innerHTML = `<div class="small mono" style="color:var(--ax0)">轴 "a" = 4，拆成两个子轴</div>
        <div class="row" style="gap:5px">
          <div class="chip c0" style="padding:7px 13px">"a":(1)2</div>
          <div class="chip mut" style="padding:7px 13px">"a":(2)2</div>
        </div>`;
      demo.appendChild(c);
      msg.innerHTML = '<span class="mono">"a"</span> 覆盖全部 4 台设备；<span class="mono">"a":(1)2</span> 只覆盖前 2 台。';
    });
    tl.at(4200, () => {
      msg.innerHTML = '<b>对应关系</b>：完整轴 <span class="mono">"a"</span> ⊇ 子轴 <span class="mono">"a":(1)2</span>，所以"用 a 切"蕴含"用 a:(1)2 切"。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '反过来<b>不成立</b>：已知用 <span class="mono">"a":(1)2</span> 切，推不出用完整 <span class="mono">"a"</span> 切。';
    });
    tl.at(10600, () => {
      msg.innerHTML = '所以传播是<b>有方向的</b>：完整轴 → 子轴可以，子轴 → 完整轴不行。';
    });
    tl.at(13200, () => {
      msg.innerHTML = '<span class="mono">sub_axes_cannot_coexist</span> 用例：两个不兼容的子轴（如 <span class="mono">(1)2</span> 与 <span class="mono">(2)2</span>）不能同时要求，冲突时不传播。';
    });
  }
},

/* ---------------------------------------------------- 6 不可整除 */
{
  kicker: 'L2-01 · 基础传播',
  title: '不可整除：<span class="hl-a">不是错误</span>，传播照常进行',
  sub: '维大小 4 沿大小为 3 的轴切，切不干净 —— 传播仍然继续，只是那一维会变成开维。',
  caption: '是否需要 padding 由<b>导出阶段</b>决定（L5-09）。传播阶段不做这个判断，也不报错。',
  code: `// 网格 a=3；张量第 1 维大小 4，已按 "a" 切（4 % 3 != 0）
func.func @single_factor_non_divisible(
    %arg0: tensor<2x4xf32>
      {sdy.sharding = #sdy.sharding<@mesh_a_3, [{}, {"a"}]>})
    -> tensor<2x4xf32> {
  %0 = stablehlo.add %arg0, %arg0
      : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// 结果：第 0 维变成 {?}（还没决定，且开放）
//       第 1 维保持 {"a", ?}
//   => [<@mesh_a_3, [{?}, {"a", ?}]>]

// 对比：如果能整除，第 0 维会是 {}（确定不分片）
// 这里写成 {?} 表示"还没有结论，等后续再看"`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:15px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:14px;justify-content:center;align-items:stretch" id="rows"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const rows = wrap.querySelector('#rows'), msg = wrap.querySelector('#msg');

    const mk = (title, dims, note, color) => {
      const c = U.el('div', { class: 'card', style: 'width:330px;opacity:.35;transition:.4s;border-color:' + color + '55' });
      c.innerHTML = `<div class="card-t" style="font-size:12px;color:${color}">${title}</div>
        <div class="row" style="gap:6px;margin:7px 0">
          ${dims.map(d => `<div class="chip ${d.cls}" style="padding:6px 12px;font-size:13px">${d.v}</div>`).join('')}
        </div>
        <div class="small faint" style="font-size:11.5px">${note}</div>`;
      rows.appendChild(c); return c;
    };

    tl.at(700, () => {
      mk('能整除的情形', [{ v: '{a}', cls: 'c0' }], '第 1 维确定沿 "a" 切', 'var(--ok)').style.opacity = '1';
      msg.innerHTML = '先看"正常"情形：维大小能被轴大小整除，那一维的分片是<b>确定的</b>。';
    });
    tl.at(4200, () => {
      mk('4 沿 a=3 切（不整除）', [{ v: '{?}', cls: 'mut' }, { v: '{"a", ?}', cls: 'c0' }],
        '第 0 维变成 <span class="mono">{?}</span>：还没结论', 'var(--warn)').style.opacity = '1';
      msg.innerHTML = '<b>不整除不是错误</b> —— L1-02 就说明了这一点。传播不会被阻断。';
    });
    tl.at(7800, () => {
      msg.innerHTML = '<span class="mono">{?}</span> 的意思：<b>这一维还没有分片结论，且保持开放</b>。后续可能被别的传播步骤填上。';
    });
    tl.at(11000, () => {
      msg.innerHTML = 'padding 是<b>导出阶段</b>的事（<span class="mono">-sdy-pad-for-divisibility</span>，见 L5-09）—— 传播阶段只负责推分片。';
    });
    tl.at(13400, () => {
      msg.innerHTML = '测试文件里"不可整除"有 <b>10 余个变体</b>，覆盖最 major / 非最 major / 多轴等组合。';
    });
  }
},

/* ------------------------------------------------------ 7 跨网格 */
{
  kicker: 'L2-01 · 基础传播',
  title: '跨网格：<span class="hl-a">不传播</span>（但同名不同实例会）',
  sub: '分片只在<b>同一个网格</b>内传播。若操作数与结果绑定了不同网格，传播就停在那里。',
  caption: '一个例外：<b>网格名不同但内容相同</b>（同样的轴与设备顺序）时，测试确认<b>可以</b>传播。',
  code: `// 输入用 @mesh_a_3，函数结果声明用 @mesh_a_6
func.func @source_result_different_meshes_not_propagated(
    %arg0: tensor<8x8xf32>
      {sdy.sharding = #sdy.sharding<@mesh_a_3, [{"a", ?}, {?}]>})
   -> (tensor<8x8xf32>
      {sdy.sharding = #sdy.sharding<@mesh_a_6, [{?}, {?}]>}) {
  %0 = stablehlo.tanh %arg0 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// 结果：tanh 只继承 @mesh_a_3 的分片
//   不会跨到 @mesh_a_6

// 相关的其它用例：
//   operands_different_meshes_not_propagated   操作数之间不同网格
//   different_meshes_and_empty_mesh_not_propagated
//   different_mesh_names_same_mesh_propagated  <- 名字不同但实际同一网格，会传播`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%' });
    wrap.innerHTML = `
      <div class="row" style="gap:16px;justify-content:center;align-items:stretch" id="rows"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const rows = wrap.querySelector('#rows'), msg = wrap.querySelector('#msg');

    const mk = (title, a, b, note, color) => {
      const c = U.el('div', { class: 'card', style: 'width:360px;opacity:.35;transition:.4s;border-color:' + color + '55' });
      c.innerHTML = `<div class="card-t" style="font-size:12px;color:${color}">${title}</div>
        <div class="row" style="gap:6px;margin:7px 0;align-items:center">
          <div class="chip mut" style="padding:5px 10px;font-size:11.5px">${a}</div>
          <span style="font-size:18px;color:${color}">${color === 'var(--ok)' ? '⟹' : '⤫'}</span>
          <div class="chip mut" style="padding:5px 10px;font-size:11.5px">${b}</div>
        </div>
        <div class="small faint" style="font-size:11.5px">${note}</div>`;
      rows.appendChild(c); return c;
    };

    tl.at(700, () => {
      mk('操作数 @mesh_a_3 → 结果 @mesh_a_6', '@mesh_a_3', '@mesh_a_6',
        '轴名/大小都不同 → <b>不传播</b>', 'var(--bad)').style.opacity = '1';
      msg.innerHTML = '<span class="mono">a=3</span> 与 <span class="mono">a=6</span> 是不同的网格，分片无法对应。';
    });
    tl.at(4200, () => {
      msg.innerHTML = '<b>为什么不传播</b>：分片里的轴名引用的是<b>特定网格</b>。跨网格的对应关系没有定义。';
    });
    tl.at(7600, () => {
      mk('两个不同名字、但内容相同', '@m1 = <["a"=3]>', '@m2 = <["a"=3]>',
        '轴与设备序都相同 → <b>会传播</b>', 'var(--ok)').style.opacity = '1';
      msg.innerHTML = '测试用例 <span class="mono">different_mesh_names_same_mesh_propagated</span> 确认了这个例外。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<b>判断标准</b>：看<b>轴名与设备顺序</b>是否一致 —— 名字只是符号，内容才是语义。';
    });
  }
},

/* ------------------------------------------ 8 空网格与 maximal 网格 */
{
  kicker: 'L2-01 · 基础传播',
  title: '空网格与 maximal 网格的<span class="hl-a">特殊规则</span>',
  sub: '这两种网格在传播中有专门的处理方式，测试文件用了 7 个用例来锁定行为。',
  caption: '记忆要点：<b>空网格可以被填上</b>（除非全闭）；<b>maximal 网格永不被替换，也不沿它传播</b>。',
  code: `// ① 空网格会被替换成真实网格，但闭维仍然被尊重
//    empty_mesh_replaced_closed_dim_respected

// ② 所有维都闭时，空网格不被替换
//    empty_mesh_all_dims_closed

// ③ maximal 网格（<[], device_ids=[0]>）不被替换
//    maximal_mesh_not_replaced

// ④ 不沿 maximal 网格传播
//    do_not_propagate_along_maximal_mesh

// ⑤ 空网格的开维可以被填充
//    does_propagate_to_empty_mesh

// ⑥ 空网格 + 闭维分片 -> 不填充
//    does_not_propagate_to_empty_mesh_with_closed_sharding

// ⑦ 部分开维的空网格 -> 可被填充
//    propagate_to_empty_mesh_with_partially_open_sharding

sdy.mesh @empty_mesh = <[]>
sdy.mesh @maximal_mesh = <[], device_ids=[0]>`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '空网格可被填充', d: '<span class="mono">empty_mesh_replaced_closed_dim_respected</span><br>开维填上，闭维不动。', c: '#4ade80' },
      { t: '全闭则不填', d: '<span class="mono">empty_mesh_all_dims_closed</span><br>没有可填的位置。', c: '#fbbf24' },
      { t: 'maximal 不替换', d: '<span class="mono">maximal_mesh_not_replaced</span><br>单设备网格是"确定的"。', c: '#fb7185' },
      { t: '不沿 maximal 传播', d: '<span class="mono">do_not_propagate_along_maximal_mesh</span><br>它没有轴，传不了。', c: '#c084fc' },
      { t: '闭维挡住填充', d: '<span class="mono">does_not_propagate_to_empty_mesh_with_closed_sharding</span>', c: '#38bdf8' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:146px;opacity:.32;transition:.35s;border-color:' + x.c + '55;padding:9px' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:11px">${x.t}</div>
        <div class="card-d" style="font-size:10.5px;line-height:1.45">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 2800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.32');
      msg.innerHTML = [
        '空网格是<b>占位符</b>：传播有真实网格可用时就填上。',
        '但<b>闭维</b>是用户的承诺，空网格也改不了它。',
        'maximal 网格是"这台设备独占"，语义已确定，不是占位符。',
        '<span class="mono">&lt;[], device_ids=[0]&gt;</span> 没有轴可引用，自然不传播。',
        '这与"闭维不可改"是同一条规则的不同表现。',
      ][i];
    }));
    tl.at(14800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '这 7 个用例是<b>边界行为的锁定测试</b> —— 防止后续改动无意中改变这些约定。';
    });
  }
},

/* ---------------------------------------------------- 9 用例族谱 */
{
  kicker: 'L2-01 · 基础传播',
  title: '79 个用例的<span class="hl-a">族谱</span>',
  sub: '这个文件是本计划最大的单文件。按算子族归一下，你会发现它其实很有结构。',
  caption: '读测试文件的方法：<b>先按族分组，再挑每族的代表读</b> —— 不必逐条啃 79 个用例。',
  code: `// 【入门】3 个
//   simple / pointwise_size_zero_dim / propagate_to_multi_result_op ...

// 【约束算子】8 个
//   propagate_from_sharding_constraint / propagation_barrier_* (7)

// 【多轴与冲突】6 个
//   multi_axes / multi_axes_conflict / multi_axes_*_incompatible ...

// 【形状变换】20+ 个
//   reshape_merge_* / reshape_split_* / reshape_size_zero_dim ...

// 【子轴】5 个
//   propagate_full_to_sub_axis / sub_axes_cannot_coexist ...

// 【不可整除】10+ 个
//   single_factor_non_divisible / *_overflows / merge_dim_*_non_divisible ...

// 【函数与返回】8 个
//   direct_arg_return_* / func_out_sharding / multiple_func_results ...

// 【网格边界】7 个
//   empty_mesh_* / maximal_mesh_* / different_meshes_* ...

// 【其它】若干
//   token_func_output_skipped / blocked_propagation_factor / main ...`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '入门', n: 3, c: '#4ade80' }, { t: '约束算子', n: 8, c: '#38bdf8' },
      { t: '多轴与冲突', n: 6, c: '#fbbf24' }, { t: '形状变换', n: 20, c: '#c084fc' },
      { t: '子轴', n: 5, c: '#f472b6' }, { t: '不可整除', n: 10, c: '#fb7185' },
      { t: '函数与返回', n: 8, c: '#93c5fd' }, { t: '网格边界', n: 7, c: '#5eead4' },
      { t: '其它', n: 12, c: '#94a3b8' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: 'width:104px;opacity:.4;transition:.35s;border-color:' + f.c + '55;padding:8px;text-align:center' });
      e.innerHTML = `<div class="card-t" style="color:${f.c};font-size:11.5px">${f.t}</div>
        <div class="big" style="font-size:22px;color:${f.c};margin-top:3px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 130)); msg.innerHTML = '<b>79 个用例</b>按算子族归成 9 组 —— 最大的两组是"形状变换"与"不可整除"。'; });
    tl.at(4200, () => {
      els.forEach((e, i) => { if (i !== 3 && i !== 5) e.style.opacity = '.25'; });
      msg.innerHTML = '<b>形状变换（20+）</b>与<b>不可整除（10+）</b>合计占了近四成 —— 因为这两类最容易出边界情况。';
    });
    tl.at(7600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '本课选了 12 个代表讲透；同类变体只要抓住<b>一个判据</b>就能自己推。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L2-02 讲保守模式：同一批用例在 <span class="mono">-conservative-propagation</span> 下的差异。';
    });
    tl.at(13400, () => {
      msg.innerHTML = 'L2 剩下的课程会依次叠加：激进 → 算子优先级 → 用户优先级 → 完整流水线。';
    });
  }
},

/* ------------------------------------------------------------ 10 练习 */
{
  kicker: 'L2-01 · 练习',
  title: '练一练：<span class="hl-a">预测传播结果</span>',
  sub: '三道题分别考：开维标记、兼容前缀、因子对应。',
  caption: '能答对，说明你已经开始用"因子"而不是"维度"来思考传播了。',
  code: `// 题 1：结果第 1 维会是什么？
//   %arg1 第 1 维 = {"b"}（闭维）
//   dot_general 结果的第 1 维

// 题 2：能传过去吗？
//   %arg0 第 0 维 = {"b", "a"}
//   %arg1 第 0 维 = {"a", "b"}

// 题 3：reshape 后为什么不需要通信？
//   2x4 -> 8，%arg0 第 0 维沿 "a" 切`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '输入第 1 维是闭维 <span class="mono">{"b"}</span>，传播后结果的第 1 维是什么？',
        a: '<span class="mono">{"b", ?}</span><br>' +
           '<b>理由</b>：闭维的 <span class="mono">"b"</span> 被继承，但<b>传播加上的部分保持开放</b>，因此带上 <span class="mono">?</span>。' +
           '<br><span class="dim">闭维锁的是"用户指定的那一维是什么"，不是"这个维度永远不能再加轴"。</span>'
      },
      {
        q: '<span class="mono">%arg0 = {"b","a"}</span> 与 <span class="mono">%arg1 = {"a","b"}</span>，能传播吗？',
        a: '<b class="badge bad">不能</b> 两者的轴序<b>完全相反</b>，逐位比较第一个就不同 → 最长兼容前缀为空。' +
           '<br><b>基础传播的行为</b>：一个轴都不传（不会尝试重排或消解冲突）。' +
           '<br><span class="dim">要处理这种冲突需要 <b>激进传播</b>（L2-04）。</span>'
      },
      {
        q: '把 <span class="mono">2x4</span> reshape 成 <span class="mono">8</span>，<span class="mono">%arg0</span> 第 0 维沿 <span class="mono">"a"</span> 切，为什么不需要通信？',
        a: '因为 <span class="mono">"a"</span> 切的是因子 <span class="mono">i</span>，而 <span class="mono">i</span> 在合并后的复合因子 <span class="mono">[ij]</span> 里仍是<b>最高位</b>。' +
           '<br>分片信息在因子层面能对上，数据<b>留在原地</b>即可。' +
           '<br><span class="dim">若切的是非最 major 因子 j，就需要拆子轴才能表达 —— 那才可能引入通信。</span>'
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
