/* ==========================================================================
   L4-04 · reshard-matmul-conv
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/insert_explicit_reshards/ 下 4 个文件
         (dot_dot_general 736 / convolution 60 / fft 74 / cholesky 202)
         合计 1072 行 / 95 用例
   目标：讲透矩阵/卷积/变换/分解类的冲突判定。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 dot 三类 */
{
  kicker: 'L4-04 · 矩阵与卷积类',
  title: '★ <span class="mono hl-a">dot</span> 的三类情形',
  sub: '66 个用例穷举了各种分片组合。归纳起来只有三类 —— 关键看**收缩维切没切**。',
  caption: '判据是：<b>收缩维在 lhs 与 rhs 上是否同分片</b>，以及<b>结果要不要在收缩维上有分片</b>。',
  code: `// 【情形 ①】收缩维【未切】-> 直接算，零通信
%arg0: [{"x"}, {}]      // lhs 第 0 维 x
%arg1: [{}, {"y"}]      // rhs 第 1 维 y
结果 : [{"x"}, {"y"}]   // 与操作数完全一致
// 输出：%[[DOT]] = stablehlo.dot %arg0, %arg1 {...[{"x"}, {"y"}]...}
//       return %[[DOT]]                     <- 无 reshard、无 collective

// 【情形 ②】收缩维【被切】-> unreduced + all_reduce
%arg0: [{"x"}, {"y"}]   // lhs 收缩维切 y
%arg1: [{"y"}, {}]      // rhs 收缩维切 y（一致 ✓）
结果 : [{"x"}, {}]      // 结果【不】要 y
// 输出：%[[DOT]] = stablehlo.dot %arg0, %arg1 {...[{"x"}, {}], unreduced={"y"}>}
//       %[[ALL_REDUCE]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{"x"}, {}]>
//       ^^^^^^^^^^^^^ 沿收缩维切产生【部分和】-> 需要归约

// 【情形 ③】收缩维被切 + 结果也要该轴 -> reduce-scatter 模式
结果 : [{"x"}, {"y"}]   // 结果也要 y
// 输出（三步）：
//   %[[DOT]] = stablehlo.dot ... {...[{"x"}, {}], unreduced={"y"}>}
//   %[[ALL_REDUCE]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{"x"}, {}]>
//   %[[RESHARD]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{"x"}, {"y"}]>
//   ^^^^^^^^^^^ 归约后再切回 y
// 测试注释直接点明：This is a reduce-scatter pattern.`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 收缩维未切', c: '#4ade80', r: '零通信',
        d: '结果与操作数一致<br><b>什么都不插</b>' },
      { t: '② 收缩维被切', c: '#fbbf24', r: 'unreduced + all_reduce',
        d: '产生<b>部分和</b><br>需要归约成完整值' },
      { t: '③ 结果也要该轴', c: '#fb7185', r: 'reduce-scatter 模式',
        d: '<b>归约 + 重新切分</b><br>三步合起来就是 reduce_scatter' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="mono" style="font-size:10.5px;color:#bdf7ec;margin:4px 0">${U.esc(x.r)}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '最理想：非收缩维各切各的、收缩维不切 —— <b>一次通信都不需要</b>。',
        '<b>为什么产生部分和</b>：收缩维被切后，每台设备只算了<b>一部分乘积</b>。',
        '<b>与 L4-01 的衔接</b>：导出流水线会把这三步<b>融合成一条 reduce_scatter</b>。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>本课看到的是融合【前】的形态</b>（all_reduce + reshard）；L4-01 看到的是融合后。';
    });
  }
},

/* ------------------------------------------------ 2 dot 用例命名 */
{
  kicker: 'L4-04 · 矩阵与卷积类',
  title: '66 个用例的<span class="hl-a">命名规律</span>',
  sub: '用例名把"冲突在哪里"写得很清楚 —— **不必逐个读代码**。',
  caption: '判据始终是「收缩维在两侧是否同分片」+「非收缩维在操作数与结果间是否对齐」。',
  code: `// 【兼容】dot_compatible_*
//   _contracting_unsharded      收缩维未切 -> 零通信
//   _contracting_dim_sharded    收缩维切了 -> all_reduce
//   _jk / _k / _empty           具体因子组合
//   _contracting_dim_empty      收缩维分片为空

// 【不兼容】dot_incompatible_*
//   _lhs_contracting_and_rhs_non_contracting_dims
//       lhs 的收缩维与 rhs 的非收缩维冲突
//   _subaxis_no_overlap         子轴无重叠
//   _a_times_a                  同一轴出现两次
//   _all_same_shardings         分片全相同（但语义不兼容！）
//   _same_factor_for_contracting_dim_and_output_i / _j
//       同一因子既做收缩又做输出
//   _same_non_contracting_dims_out_empty / _out_i / _out_j
//       非收缩维相同但输出分片不同

// 【重点】_contracting_dim_and_result_dim_sharded_same_axis
//   收缩维与结果维用【同一个轴】-> reduce-scatter 模式
//   _same_axis_incompatible_order  同轴但轴序不兼容

// 读法建议：抓住"哪两个位置的分片冲突"，用例名就是答案`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'compatible', c: '#4ade80', d: '收缩维未切<br>或只需 all_reduce' },
      { t: 'lhs vs rhs 冲突', c: '#fbbf24', d: 'lhs 收缩维<br>vs rhs 非收缩维' },
      { t: '子轴无重叠', c: '#38bdf8', d: '<span class="mono">subaxis_no_overlap</span>' },
      { t: '同轴两次', c: '#c084fc', d: '<span class="mono">a_times_a</span>' },
      { t: '全相同却不兼容', c: '#fb7185', d: '<span class="mono">all_same_shardings</span><br>语义层面冲突' },
      { t: '同因子两用', c: '#f472b6', d: '既做收缩<br>又做输出' },
      { t: 'reduce-scatter', c: '#5eead4', d: '收缩维与结果维<br><b>同轴</b>' },
      { t: '轴序不兼容', c: '#93c5fd', d: '<span class="mono">same_axis_<br>incompatible_order</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:126px;opacity:.35;transition:.3s;border-color:${x.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div style="font-size:10.5px;color:${x.c};line-height:1.3">${x.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>66 个用例</b>按冲突类型归成 8 组 —— 组合非常多。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>最反直觉的一条</b>：<span class="mono">all_same_shardings</span> —— 分片<b>全相同</b>却仍然不兼容（语义层面冲突）。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>判据</b>：收缩维在两侧是否同分片 + 非收缩维在操作数与结果间是否对齐。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>读法</b>：用例名把"哪两个位置冲突"写清楚了，不必逐个读代码。';
    });
  }
},

/* ------------------------------------------------ 3 convolution */
{
  kicker: 'L4-04 · 矩阵与卷积类',
  title: '<span class="mono hl-a">convolution</span>：批维分组会<span class="hl-a">改变大小</span>',
  sub: '3 个用例，但 `batch_group_count` 那条把 L2-10 的卷积规则完整展示了出来。',
  caption: '注意测试里的 NOTE 直接给出了 sharding rule —— 与 L2-10 注册表里看到的是同一条。',
  code: `// 【基础】输入在批维切 x，结果无分片
func.func @convolution(%arg0 : tensor<2x224x224x192xf32>
      {...<@mesh, [{"x"}, {}, {}, {}]>}, ...) -> tensor<2x112x112x64xf32> {
// 输出：
//   %[[CONVOLUTION]] = stablehlo.convolution(%arg0, %arg1)
//       {...<@mesh, [{"x"}, {}, {}, {}]>}
//   %[[RESHARD]] = sdy.reshard %[[CONVOLUTION]] <@mesh, [{}, {}, {}, {}]>
//   ^^^^^^^^^^^ 算完再 reshard 掉

// 【batch_group_count】批维大小改变
func.func @convolution_batch_group_count(
    %arg0: tensor<8x224x224x192xf32> {...<@mesh_xyz, [{"x"}, {"z"}, {}, {}]>}, ...)
    -> (tensor<2x112x112x256xf32> {...<@mesh_xyz, [{"y"}, {"z"}, {}, {}]>})
// 测试 NOTE 给出规则：
//   sdy.sharding_rule = ([ij, kl, mn, o], [l, n, o, ip])->([j, k, m, ip])
//     {i=4, j=2, k=112, l=2, m=112, n=2, o=192, p=64}
//     reduction={l, n, o} permutation={k, m}
// 读法：
//   输入批维是【复合因子 ij】：i=4（batch_group_count） × j=2（输出批维）
//   输入 8 = 4 × 2，输出批维只有 2 -> 批维【缩小了 4 倍】
//   reduction={l,n,o}：窗口与输入通道
//   permutation={k,m}：空间维因步长而"不成比例"

// 输出：
//   %[[RESHARD1]] = sdy.reshard %arg0 <@mesh_xyz, [{"x", "y"}, {"z"}, {}, {}]>
//   ^^^^^^^^^^^^ 取【并集】x+y（x 切 i、y 切 j）`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 基础情形', c: '#4ade80', d: '输入批维切 x<br>结果无分片<br>→ 算完 reshard 掉' },
      { t: '② 复合因子', c: '#fbbf24', d: '批维是 <span class="mono">ij</span><br><span class="mono">i=4</span>（分组）× <span class="mono">j=2</span>（输出批维）' },
      { t: '③ 并集解法', c: '#38bdf8', d: 'x 切 i、y 切 j<br>→ reshard 到 <span class="mono">[{"x","y"}, ...]</span><br><span class="dim">与 L4-03 的并集一致</span>' },
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
        '批维可以自由分片（各批独立卷积）—— 与 fft / cholesky 一样。',
        '<b>这就是 L2-10 那条卷积规则的来源</b>：<span class="mono">reduction={k,m,n} permutation={j,l}</span> 现在看到它在 reshard 中的作用。',
        '<b>为什么取并集</b>：<span class="mono">x</span> 与 <span class="mono">y</span> 切的是不同因子（<span class="mono">i</span> 与 <span class="mono">j</span>），一个分片满足不了两者。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>批维缩小 4 倍</b>是 <span class="mono">batch_group_count</span> 的直接后果 —— 输入 8 变成输出 2。';
    });
  }
},

/* ------------------------------------------------ 4 fft 与 cholesky */
{
  kicker: 'L4-04 · 矩阵与卷积类',
  title: '<span class="mono hl-a">fft</span> 与 <span class="mono hl-a">cholesky</span>：<span class="hl-a">受限维</span> vs <span class="hl-a">自由维</span>',
  sub: '这两类算子的共同结构：有一个**受限维**（变换维 / 分解维）和一个**自由维**（批维）。',
  caption: '这与 <span class="mono">dot</span> 的"收缩维 vs 非收缩维"是同一个模式 —— 只是叫法不同。',
  code: `// 【fft】7 个用例
//   fft / fft_inverse                          正向 / 逆变换
//   fft_real_truncated_result                  实数输入、结果【截断】
//   fft_inverse_real_expanded_result           逆变换、实数【扩展】结果
//   fft_small_batch_dimension                  批维较小
//   fft_single_fft_dimension                   单个 FFT 维
//   fft_single_fft_dimension_real_truncated_result
//
// 要点：
//   批维    -> 可自由分片（各批独立变换）
//   变换维  -> 受限（FFT 是【全局】操作，与 reduce 类似）
//   截断/扩展会【改变维度大小】-> 可能不可整除（与 L4-03 的 pad_slice 同理）

// 【cholesky_triangular_solve】19 个用例
//   cholesky_sharded_input_batch_dim_only       输入切【批维】
//   cholesky_sharded_output_batch_dim_only      输出切【批维】
//   cholesky_sharded_batch_dim_only_different   两者切【不同轴】
//   cholesky_sharded_input_cholesky_dim_only    输入切【cholesky 维】
//   cholesky_sharded_output_cholesky_dim_only   输出切 cholesky 维
//   cholesky_sharded_cholesky_dim_only_different / _same
//   cholesky_sharded_input_batch_dim_and_output_cholesky_dim_same
//   cholesky_sharded_output_batch_dim_and_input_cholesky_dim_same
//   cholesky_sharded_same
//
// 要点：
//   批维      -> 自由
//   cholesky 维 -> 受限（矩阵分解要看到【整个矩阵】，三角求解有依赖）
//
// 19 个用例就是"两个维度 × 切/不切/切哪个轴"的组合`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'fft', c: '#38bdf8', n: '7 个',
        d: '<b>变换维受限</b>（全局操作）<br>批维自由<br>截断/扩展改尺寸' },
      { t: 'cholesky_triangular_solve', c: '#c084fc', n: '19 个',
        d: '<b>分解维受限</b>（要看到整个矩阵）<br>批维自由<br>19 个组合' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:11px;overflow-wrap:anywhere">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>FFT 是全局操作</b>：一次变换要看到整条轴 —— 所以变换维上分片受限。'; });
    tl.at(4400, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>矩阵分解同理</b>：三角求解有<b>依赖关系</b>，要看到整个矩阵。';
    });
    tl.at(8200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>共同结构</b>：一个<b>受限维</b>（变换/分解）+ 一个<b>自由维</b>（批）。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '这与 <span class="mono">dot</span> 的"收缩维 vs 非收缩维"<b>是同一个模式</b> —— 只是叫法不同。';
    });
  }
},

/* ------------------------------------------------ 5 族谱 */
{
  kicker: 'L4-04 · 矩阵与卷积类',
  title: '4 个文件 / 95 个用例的<span class="hl-a">族谱</span>',
  sub: '一句话总结：**收缩/变换/分解维是"受限维"，批维是"自由维"。**',
  caption: '冲突判定就是看这两类维度上的分片是否兼容。',
  code: `// 【矩阵乘】dot_dot_general    736 行 / 66 用例  <- 主体
//   三类情形：收缩维未切（零通信）/ 被切（all_reduce）/
//             结果也要该轴（reduce-scatter 模式）
//   8 组冲突类型：lhs vs rhs、子轴无重叠、同轴两次、
//                 全相同却不兼容、同因子两用、轴序不兼容...

// 【卷积】convolution            60 行 / 3 用例
//   批维分组改变大小（复合因子 ij）
//   batch_group_count 把批维缩小 4 倍

// 【变换】fft                    74 行 / 7 用例
//   变换维受限、批维自由
//   截断/扩展改变维度大小

// 【分解】cholesky_triangular_solve  202 行 / 19 用例
//   分解维受限、批维自由
//   19 个"两维度 × 切/不切/切哪个轴"的组合

// 合计 1072 行 / 95 用例

// 一句话：
//   收缩/变换/分解维是【受限维】，批维是【自由维】
//   冲突判定 = 看这两类维度上的分片是否兼容`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '矩阵乘', n: 66, c: '#38bdf8' }, { t: '卷积', n: 3, c: '#4ade80' },
      { t: '变换', n: 7, c: '#fbbf24' }, { t: '分解', n: 19, c: '#c084fc' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:150px;opacity:.4;transition:.35s;border-color:${f.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="card-t" style="color:${f.c};font-size:11.5px">${f.t}</div>
        <div class="big" style="font-size:20px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 130)); msg.innerHTML = '<b>95 个用例</b> —— 矩阵乘占七成（66 个）。'; });
    tl.at(4000, () => {
      msg.innerHTML = '<b>为什么 dot 最多</b>：它有收缩维 / 非收缩维 / 批维三类维度，冲突组合最多。';
    });
    tl.at(7400, () => {
      msg.innerHTML = '<b>共同模式</b>：受限维（收缩/变换/分解）+ 自由维（批）。';
    });
    tl.at(10600, () => {
      msg.innerHTML = '<b>下一课</b> L4-05 讲归约类（reduce / sort / rng）—— 那里"被排序维必须全复制"。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L4-04 · 练习',
  title: '练一练：<span class="hl-a">dot 会插什么</span>',
  sub: '三道题分别考：三类情形、reduce-scatter 模式、受限维。',
  caption: '一句话总结：<b>收缩维切了就归约；结果也要该轴就再切回来</b>。',
  code: `// 题 1：收缩维未切时，dot 需要什么通信？

// 题 2：收缩维切了 y、结果也要 y，会插什么？

// 题 3：fft 与 cholesky 的共同结构是什么？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col compact-ex', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '收缩维<b>未切</b>时，<span class="mono">dot</span> 需要什么通信？',
        a: '<b>什么都不需要</b> —— 零通信。' +
           '<br><b>条件</b>：非收缩维各切各的（lhs 第 0 维、rhs 第 1 维），收缩维两侧都不切，结果分片与操作数完全一致。' +
           '<br><span class="dim">用例 <span class="mono">dot_compatible_contracting_unsharded</span>：输出里连 reshard 都没有。</span>'
      },
      {
        q: '收缩维切了 <span class="mono">y</span>、结果<b>也要</b> <span class="mono">y</span>，会插什么？',
        a: '<b>三步</b>：<span class="mono">dot</span>（标 <span class="mono">unreduced={"y"}</span>）→ <span class="mono">all_reduce</span> → <span class="mono">reshard</span>。' +
           '<br>测试注释直接点明：<span class="mono">This is a reduce-scatter pattern.</span>' +
           '<br><span class="dim">这三步合起来就是 <span class="mono">reduce_scatter</span> 的语义（一边归约一边重新切分）。L4-01 见过导出流水线会把它<b>融合成一条</b>。</span>'
      },
      {
        q: '<span class="mono">fft</span> 与 <span class="mono">cholesky</span> 的共同结构是什么？',
        a: '<b>一个受限维 + 一个自由维</b>。' +
           '<br><b>fft</b>：变换维受限（FFT 是<b>全局</b>操作）、批维自由。' +
           '<br><b>cholesky</b>：分解维受限（矩阵分解要看到<b>整个矩阵</b>、三角求解有依赖）、批维自由。' +
           '<br><span class="dim">这与 <span class="mono">dot</span> 的"收缩维 vs 非收缩维"是<b>同一个模式</b>，只是叫法不同。冲突判定就是看这两类维度上的分片是否兼容。</span>'
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
