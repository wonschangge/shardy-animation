/* ==========================================================================
   L4-03 · reshard-elementwise-shape
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/insert_explicit_reshards/ 下 9 个文件
         (1086 行 / 109 用例)
   目标：讲透逐元素与形状变换算子的 reshard 规则。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 位置规则 */
{
  kicker: 'L4-03 · 逐元素与形状类',
  title: '★ 逐元素算子：reshard 插在<span class="hl-a">哪一侧</span>',
  sub: '规则一句话：**算子采用"更大"的那一侧的分片；另一侧插 reshard。**',
  caption: '用例名直接点明了判据 —— <span class="mono">_input_sharding_is_larger</span> / <span class="mono">_output_sharding_is_larger</span>。',
  code: `// 网格：@mesh = <["x"=4, "y"=2, "z"=4]>

// 【规则 ①】输入无分片、结果要 -> reshard 插在【之前】
func.func @negate(
    %arg0: tensor<4x32xf32> {...<@mesh, [{}, {}]>})            // 输入无分片
    -> (tensor<4x32xf32> {...<@mesh, [{"x"}, {}]>}) {          // 结果要 x
  %0 = stablehlo.negate %arg0 {...<@mesh, [{"x"}, {}]>} : ...
}
// 输出：
//   %[[RESHARD]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}]>
//   stablehlo.negate %[[RESHARD]] {...}
//   ^^^^^^^^^ 在 negate【之前】reshard，让它在 x 上算

// 【规则 ②】输入侧更大 -> reshard 插在【之后】
//   输入 [{"x"}, {}] vs 结果 [{"y"}, {}]
//   -> negate 保持输入的分片 x，【之后】reshard 搬到 y
//   %[[NEGATE]] = stablehlo.negate %arg0 {...[{"x"}, {}]...}
//   %[[RESHARD]] = sdy.reshard %[[NEGATE]] <@mesh, [{"y"}, {}]>

// 【规则 ③】输出侧更大 -> reshard 插在【之前】
//   输入 [{"y"}, {}] vs 结果 [{"x"}, {}]
//   -> 【之前】把输入搬到 x
//   %[[RESHARD]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}]>
//   stablehlo.negate %[[RESHARD]] {...[{"x"}, {}]...}`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 输入无分片', c: '#4ade80', d: '结果要 x<br>→ <b>之前</b> reshard 到 x' },
      { t: '② 输入更大', c: '#38bdf8', d: '输入 x / 结果 y<br>→ 保持 x，<b>之后</b>搬到 y' },
      { t: '③ 输出更大', c: '#fbbf24', d: '输入 y / 结果 x<br>→ <b>之前</b>搬到 x' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '最直观的情形：输入什么都没写，结果有要求 → 先 reshard 再算。',
        '算子<b>沿用输入的分片</b>（不必白搬一次），输出后再对齐。',
        '算子<b>采用输出的分片</b>（因为它"更大"），输入前先搬。',
      ][i];
    }));
    tl.at(12200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>统一规则</b>：算子采用"更大"的那一侧的分片；另一侧插 reshard。';
    });
  }
},

/* ------------------------------------------------ 2 并集与配对 */
{
  kicker: 'L4-03 · 逐元素与形状类',
  title: '两个特例：<span class="hl-a">并集</span>与<span class="hl-a">双侧 reshard</span>',
  sub: '当两侧"维度都不同"时，算子被放在**两个分片的并集**上执行 —— 这样两侧的信息都不丢。',
  caption: '这个用例插了 <b>3 条</b> reshard —— 是本课最"重"的一个。',
  code: `// 【并集】两侧维度都不同
func.func @add_input_and_output_sharded_on_separate_dims(
    %arg0: ... [{"x"}, {}], %arg1: ... [{"x"}, {}])
    -> ... [{"x"}, {}] {
  %0 = stablehlo.add %arg0, %arg1 {...<@mesh, [{}, {"y"}]>} : ...   // 算子声明 [{},{"y"}]
}
// 输出（3 条 reshard）：
//   %[[RESHARD1]] = sdy.reshard %arg0 <@mesh, [{"x"}, {"y"}]>   // x + y
//   %[[RESHARD2]] = sdy.reshard %arg1 <@mesh, [{"x"}, {"y"}]>   // x + y
//   %[[ADD]] = stablehlo.add %[[RESHARD1]], %[[RESHARD2]]
//              {...<@mesh, [{"x"}, {"y"}]>}                     // 在【并集】上算
//   %[[RESHARD3]] = sdy.reshard %[[ADD]] <@mesh, [{}, {"y"}]>   // 再收回

// 洞察：算子被放在 [{"x"}, {"y"}] 上执行
//       两个方向的信息都不丢 -> 只多两条搬运

// 【输入相同、输出无分片】
//   两个输入都是 [{"x"}, {}]，输出无分片
//   -> 在 [{"x"}, {}] 上算 add，【之后】reshard 掉
//   %[[ADD]] = stablehlo.add %arg0, %arg1 {...[{"x"}, {}]...}
//   %[[RESHARD]] = sdy.reshard %[[ADD]] <@mesh, [{}, {}]>`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:22px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center' });
      c.innerHTML = `
        <div class="small mono faint">两侧维度都不同</div>
        <div class="row" style="gap:6px">
          <div class="chip c0" style="padding:5px 10px;font-size:11px">输入 [{"x"},{}]</div>
          <div class="chip c3" style="padding:5px 10px;font-size:11px">算子 [{},{"y"}]</div>
        </div>
        <div class="small faint">x 和 y 落在不同维度上</div>`;
      demo.appendChild(c);
      msg.innerHTML = '输入在 <b>dim0</b> 切 <span class="mono">x</span>，算子在 <b>dim1</b> 切 <span class="mono">y</span> —— 没法直接统一。';
    });
    tl.at(4600, () => {
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹', style: 'font-size:24px' }));
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ok)">并集</div>
        <div class="chip c4" style="padding:6px 12px;font-size:11.5px">[{"x"}, {"y"}]</div>
        <div class="small faint">两个方向都切</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>取并集</b> <span class="mono">[{"x"}, {"y"}]</span> —— 在它上面算 <span class="mono">add</span>，两侧信息都不丢。';
    });
    tl.at(8800, () => {
      msg.innerHTML = '<b>代价</b>：3 条 reshard（两个输入各一条、输出一条）。<b>收益</b>：只多搬运，不丢分片。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>对比</b>：如果两个输入分片相同、只是输出无分片，就只需 <b>1 条</b> reshard（在算子之后）。';
    });
  }
},

/* ------------------------------------------------ 3 ★ broadcast */
{
  kicker: 'L4-03 · 逐元素与形状类',
  title: '★ <span class="mono hl-a">broadcast_in_dim</span>：按 <span class="mono">dims</span> <span class="hl-a">反向映射</span>',
  sub: 'broadcast 会重排维度（`dims`）。所以分片要**反向映射回输入侧** —— 在输入上 reshard，让输出自然正确。',
  caption: '这是本课最巧妙的一处：不是"算完再搬"，而是"<b>先摆好位置再算</b>"。',
  code: `// 输入：tensor<2x3x5x1x7xf32>，第 0 维切 x
// dims = [0, 2, 1, 3, 4]  表示：输入第 0 维 -> 输出第 0 维
//                                输入第 1 维 -> 输出第 2 维
//                                输入第 2 维 -> 输出第 1 维
// 输出：tensor<2x5x3x11x7x13xf32>，要求 [{}, {"x"}, {}, {}, {}, {"y"}]
//                                 ^^^^^^^^^^ 第 1 维要 x

// 问题：输出第 1 维要 x，而输出第 1 维来自【输入第 2 维】
//       但输入现在是第 0 维有 x

// 解法：把 x 从输入第 0 维【搬到】输入第 2 维
%[[RESHARD]] = sdy.reshard %arg0 <@mesh, [{}, {}, {"x"}, {}, {}]>
//                                               ^^^^^^^^^^ 搬到第 2 维
%[[BC]] = stablehlo.broadcast_in_dim %[[RESHARD]], dims = [0, 2, 1, 3, 4] {...}

// 结果：broadcast 之后 x 自然落在输出第 1 维上 ✓

// 另一个用例（输出无分片）：
//   输入有 {"x"}、输出无 -> 输入侧 reshard 成 [{}, {}, {}, {}, {}]（把 x 去掉）`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:22px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="small mono faint">输入 dims 映射</div>
        <div class="row" style="gap:5px">
          <div class="chip c0" style="padding:5px 9px;font-size:10.5px">in0→out0</div>
          <div class="chip mut" style="padding:5px 9px;font-size:10.5px">in1→out2</div>
          <div class="chip c3" style="padding:5px 9px;font-size:10.5px">in2→out1</div>
        </div>
        <div class="small faint">dims = [0, 2, 1, 3, 4]</div>`;
      demo.appendChild(c);
      msg.innerHTML = '输出第 1 维要 <span class="mono">x</span>，而它来自<b>输入第 2 维</b>（<span class="mono">dims[2] = 1</span>）。';
    });
    tl.at(4800, () => {
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹', style: 'font-size:24px' }));
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ok)">输入侧 reshard</div>
        <div class="row" style="gap:5px">
          <div class="chip mut" style="padding:5px 9px;font-size:10.5px">{}</div>
          <div class="chip mut" style="padding:5px 9px;font-size:10.5px">{}</div>
          <div class="chip c4" style="padding:5px 9px;font-size:10.5px">{"x"}</div>
          <div class="chip mut" style="padding:5px 9px;font-size:10.5px">{}</div>
          <div class="chip mut" style="padding:5px 9px;font-size:10.5px">{}</div>
        </div>
        <div class="small faint">x 搬到第 2 维</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>把 x 搬到输入第 2 维</b> —— broadcast 之后它自然落在输出第 1 维。';
    });
    tl.at(9000, () => {
      msg.innerHTML = '<b>思路不同</b>：不是"算完再搬"，而是"<b>先摆好位置再算</b>"。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>为什么可以这样</b>：broadcast 是纯数据搬运（复制元素），<b>不改变元素值</b> —— 所以分片可以自由重排。';
    });
    tl.at(14200, () => {
      msg.innerHTML = '<b>另一用例</b>：输入有 <span class="mono">x</span>、输出无分片 → 输入侧 reshard 把 <span class="mono">x</span> 去掉。';
    });
  }
},

/* ------------------------------------------------ 4 reshape */
{
  kicker: 'L4-03 · 逐元素与形状类',
  title: '<span class="mono hl-a">reshape</span>：为什么<span class="hl-a">常需 reshard</span>',
  sub: '因为 reshape 的分片必须在**因子层面**对应（L2-01）。对不上就要 reshard —— 有时**两侧都插**。',
  caption: '<span class="mono">reshape.mlir</span> 有 <b>47 个用例</b>（本课最大），因为因子对应关系的组合非常多。',
  code: `// 【需要 reshard】两侧都插
func.func @reshape(
    %arg0: tensor<16x2x4xf32> {...<@mesh, [{"x"}, {}, {}]>})
    -> (tensor<16x8xf32> {...<@mesh, [{}, {"y", "x"}]>}) {
  %0 = stablehlo.reshape %arg0 {...<@mesh, [{}, {"y", "x"}]>} : ...
}
// 输出（两条 reshard）：
//   %[[RESHARD1]] = sdy.reshard %arg0 <@mesh, [{"x"}, {"y"}, {}]>
//   %[[RESHAPE]] = stablehlo.reshape %[[RESHARD1]] {...[{"x"}, {"y"}]...}
//   %[[RESHARD2]] = sdy.reshard %[[RESHAPE]] <@mesh, [{}, {"y", "x"}]>
// 为什么：16x2x4 -> 16x8 把第 1、2 维合并成 8
//         结果要求 {"y","x"} 落在【复合因子】上
//         输入的 x 在第 0 维（不参与合并）、y 又没被切 -> 对不上

// 【不需要 reshard】能对应时一条都不插
func.func @reshape_simple_merge_sharding_is_from_x_to_x_and_x_fits_exactly_to_first_dim(
    %arg0: tensor<4x8xf32> {...[{"x"}, {}]}) -> (tensor<32xf32> {...[{"x"}]}) {
  // CHECK-NOT: sdy.reshard
}
// 为什么：x 切的是【最 major 的因子】，合并后仍能对应（L2-01 的零通信情形）

// 47 个用例的命名规律（用例名本身就是答案）：
//   reshape_ij_k_to_i_jk          输入因子 [ij, k] -> 输出 [i, jk]
//   _and_x_to_x / _and_x_to_z     单轴切法不变 / 换成另一轴
//   _and_xy_to_yx / _and_yx_to_xy 轴序【交换】
//   _and_xy_to_x / _and_xy_to_y   复合轴【拆开】
//   _merged_dimensions_are_sharded 被合并的维上有分片
//   _singleton_dimensions_are_sharded 大小为 1 的维上有分片
//   _factor_j_is_sharded          具体哪个因子被切`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '需 reshard', c: '#fb7185', n: '两侧都插',
        d: '因子对应不上<br>先搬到"将被合并的维"<br>算完再搬回目标' },
      { t: '不需 reshard', c: '#4ade80', n: 'CHECK-NOT',
        d: '切的是<b>最 major 因子</b><br>合并后仍能对应<br><span class="dim">L2-01 的零通信情形</span>' },
      { t: '47 个用例', c: '#38bdf8', n: '命名即答案',
        d: '<span class="mono">ij_k_to_i_jk</span> 这类命名<br>把"输入因子 → 输出因子"<br>写得清清楚楚' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>为什么要先搬</b>：让 <span class="mono">y</span> 落在"将被合并的维度"上，合并后才能对应。',
        '<b>反例同样重要</b>：<span class="mono">CHECK-NOT: sdy.reshard</span> 锁定了"能对应时不插"。',
        '<b>读法建议</b>：不必逐个读 —— 抓住"因子对应"这一条，用例名本身就是答案。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>一句话</b>：reshape 看<b>因子</b>，不是看维度。';
    });
  }
},

/* ------------------------------------------------ 5 其余五族 */
{
  kicker: 'L4-03 · 逐元素与形状类',
  title: '其余五个文件的<span class="hl-a">规则速查</span>',
  sub: '9 个文件按规则归成五族 —— 抓住每族的一句话即可。',
  caption: '完整对照表在课件 README 里。',
  code: `// 【逐元素族】elementwise_ops (11) + clamp_select (4)
//   规则：对应维同分片；"更大"的一侧优先
//   clamp 是三操作数、select 是条件选择 —— 都是逐元素

// 【广播族】broadcast_in_dim (2)
//   规则：按 dims 反向映射到输入侧；不一致时在【输入侧】reshard

// 【形状变换族】reshape (47) + reverse (6) + bitcast_convert (4)
//   reshape：看【因子对应】，不对应就 reshard（可能两侧都插）
//   reverse：反转【不改变维度大小】-> 分片可直接对应
//   bitcast_convert：改变元素类型，覆盖 upcast / equal / downcast
//                    （与 L2-10 注册表里的三个用例对应）

// 【拼接族】concatenate (10)
//   规则：拼接维上各操作数的分片【必须一致】
//   覆盖：分片相同 / 不同 / 结果分片不同 / 多操作数

// 【切片填充族】pad_slice (19) + dynamic_slice_dynamic_update_slice (6)
//   pad / slice 会【改变维度大小】-> 可能不可整除（L2-02 讲过）
//   dynamic_slice：起始索引是【运行时值】，分片更受限
//                  （与 L2-10 里 gather 的 blocked_propagation 同理）

// 一句话总结：
//   逐元素类看"对应维"；形状类看"因子对应"；拼接类看"拼接维一致"`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '逐元素', n: 15, c: '#4ade80' }, { t: '广播', n: 2, c: '#38bdf8' },
      { t: '形状变换', n: 57, c: '#fbbf24' }, { t: '拼接', n: 10, c: '#c084fc' },
      { t: '切片填充', n: 25, c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:146px;opacity:.4;transition:.35s;border-color:${f.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="card-t" style="color:${f.c};font-size:11.5px">${f.t}</div>
        <div class="big" style="font-size:20px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 120)); msg.innerHTML = '<b>109 个用例</b>分五族 —— 最大的一族是"形状变换"（57 个）。'; });
    tl.at(4000, () => {
      els.forEach((e, i) => { if (i !== 2) e.style.opacity = '.25'; });
      msg.innerHTML = '形状变换占了<b>过半</b> —— 因为因子对应关系的组合最多。';
    });
    tl.at(7400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>三句话</b>：逐元素看"对应维"，形状类看"因子对应"，拼接类看"拼接维一致"。';
    });
    tl.at(10600, () => {
      msg.innerHTML = '<b>下一课</b> L4-04 讲矩阵/卷积类（dot、convolution、fft、cholesky）—— 冲突判定更复杂。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L4-03 · 练习',
  title: '练一练：<span class="hl-a">reshard 插在哪</span>',
  sub: '三道题分别考：位置规则、并集、reshape 的因子对应。',
  caption: '一句话总结：<b>算子采用"更大"的一侧，另一侧插 reshard</b>。',
  code: `// 题 1：输入 [{"x"},{}]，结果要 [{"y"},{}]，reshard 插在哪？

// 题 2：输入 [{"x"},{}]，算子声明 [{},{"y"}]，
//       两侧维度都不同，怎么办？

// 题 3：为什么 reshape 常需要 reshard？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col compact-ex', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '输入 <span class="mono">[{"x"},{}]</span>，结果要 <span class="mono">[{"y"},{}]</span>，reshard 插在哪？',
        a: '<b>插在算子【之后】</b>（用例 <span class="mono">negate_input_sharding_is_larger</span>）。' +
           '<br><b>理由</b>：输入侧的 <span class="mono">x</span> "更大" → 算子<b>沿用输入的分片</b>算，算完再搬到 <span class="mono">y</span>。' +
           '<br><span class="dim">统一规则：算子采用"更大"的那一侧的分片；另一侧插 reshard。反过来（输入 y、结果 x）就插在<b>之前</b>。</span>'
      },
      {
        q: '输入 <span class="mono">[{"x"},{}]</span>，算子声明 <span class="mono">[{},{"y"}]</span>，两侧维度都不同，怎么办？',
        a: '<b>取并集</b> <span class="mono">[{"x"}, {"y"}]</span>，在并集上执行算子，之后（如需）再收回。' +
           '<br>该用例插了 <b>3 条</b> reshard：两个输入各一条到并集，输出一条到目标。' +
           '<br><span class="dim">洞察：这样两个方向的信息<b>都不丢</b>，代价只是多两次搬运。</span>'
      },
      {
        q: '为什么 <span class="mono">reshape</span> 常需要 reshard？',
        a: '因为 reshape 的分片必须在<b>因子层面</b>对应（L2-01），而 reshape 会<b>合并/拆分维度</b>，因子对应关系经常对不上。' +
           '<br><b>例</b>：<span class="mono">16x2x4 → 16x8</span> 把第 1、2 维合并；若 <span class="mono">y</span> 没落在"将被合并的维"上，就要先 reshard 摆好位置，算完再搬回目标 —— <b>两侧都插</b>。' +
           '<br><span class="dim">反例：切的是<b>最 major 因子</b>时合并后仍能对应，用 <span class="mono">CHECK-NOT: sdy.reshard</span> 锁定"一条都不插"。</span>'
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
