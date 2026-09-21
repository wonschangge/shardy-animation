/* ==========================================================================
   L4-10 · optimize-collectives
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/optimize_collectives/ 下 2 个文件
         (all_to_all_fully_scattered 133 行 / 7 用例
          all_to_all_partially_scattered 98 行 / 6 用例) = 231 行 / 13 用例
   目标：讲透"消除 all_to_all 链前冗余 collective_permute"这个优化。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 优化内容 */
{
  kicker: 'L4-10 · 通信优化',
  title: '★ 优化内容：删掉<span class="mono hl-a">冗余的 permute</span>',
  sub: '`collective_permute` 后面紧跟着 `all_to_all` 时，permute 往往是**多余的**。',
  caption: '这是 L4 层唯一一个<b>纯粹的优化</b> pass —— 它不插入任何东西，只删除。',
  code: `// RUN: sdy_opt %s -sdy-optimize-collectives

// 【优化前】permute + 两条 all_to_all（三步，含一次真实通信）
func.func @two_axis_full_scatter(
    %arg0: tensor<16x8x8xf32> {...<@mesh_2d, [{"x", "y"}, {}, {}]>})
    -> (tensor<16x8x8xf32> {...<@mesh_2d, [{}, {"y"}, {"x"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{"y", "x"}, {}, {}]>
  //   ^^^^^^^^^^^^^^^^^^ 把 {"x","y"} 换成 {"y","x"} —— 只是【调整顺序】
  %1 = sdy.all_to_all [{"x"}: 0->2] %0 out_sharding=<@mesh_2d, [{"y"}, {}, {"x"}]>
  %2 = sdy.all_to_all [{"y"}: 0->1] %1 out_sharding=<@mesh_2d, [{}, {"y"}, {"x"}]>
  return %2 : tensor<16x8x8xf32>
}

// 【优化后】permute 消失，改用 reshape 拆复合轴
// CHECK-NOT:   sdy.collective_permute        <- permute 被删了！
// CHECK: %[[RESHAPE_IN]] = stablehlo.reshape %arg0
//            {...<@mesh_2d, [{"x"}, {"y"}, {}, {}, {}]>}
//            : (tensor<16x8x8xf32>) -> tensor<2x2x4x8x8xf32>
//        ^^^^^^^^^^^ 把复合轴 {"x","y"} 【拆成两个独立的维】
// CHECK: %[[A2A1]] = sdy.all_to_all [{"x"}: 0->4] %[[RESHAPE_IN]] ...
// CHECK: %[[A2A2]] = sdy.all_to_all [{"y"}: 1->3] %[[A2A1]] ...
// CHECK: %[[RESHAPE_OUT]] = stablehlo.reshape %[[A2A2]] ...
//            : (tensor<2x2x4x8x8xf32>) -> tensor<16x8x8xf32>
//        ^^^^^^^^^^^^ 再合回去

// 模式：
//   collective_permute -> reshape -> all_to_all × N -> reshape
//      （删除）           （拆复合轴）   （按轴搬）      （合回去）`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '优化前', c: '#fb7185', n: '3 步',
        d: '<span class="mono">permute</span>（真实通信）<br>+ 两条 <span class="mono">all_to_all</span>' },
      { t: '优化后', c: '#4ade80', n: '4 步但更省',
        d: '两条 <span class="mono">all_to_all</span><br>+ 两条 <span class="mono">reshape</span><br><span class="dim">reshape 是本地操作</span>' },
      { t: '省下的', c: '#38bdf8', n: '1 次通信',
        d: '跨设备通信是<br>整个流程里<b>最贵</b>的一环' },
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
        '<b>注意 permute 做了什么</b>：只是把 <span class="mono">{"x","y"}</span> 换成 <span class="mono">{"y","x"}</span> —— <b>纯顺序调整</b>。',
        '<b>reshape 无通信</b>：它只改变"怎么看待"数据，不搬任何东西。',
        '<b>这就是优化的价值</b>：用本地操作换掉一次跨设备通信。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>本课是 L4 唯一的纯优化 pass</b> —— 不插入任何东西，只删除冗余。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 为什么安全 */
{
  kicker: 'L4-10 · 通信优化',
  title: '★ 为什么删 permute <span class="hl-a">是安全的</span>',
  sub: '关键一句话：**`all_to_all` 按【轴名】搬运，轴在源维内的顺序不影响结果。**',
  caption: '这正是 TODOLIST 验收点问的："能指出哪条 permute 被消除以及<b>为什么安全</b>"。',
  code: `// 【all_to_all 的语义】把【某个轴】从某个维搬到某个维
sdy.all_to_all [{"x"}: 0->2] %0 out_sharding=...
//              ^^^^^ 参数写的是【轴名】，不是"第几个"

// 【permute 做的事】只是调整轴在维内的顺序
%0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{"y", "x"}, {}, {}]>
//                                                          ^^^^^^^^^^
//                            {"x","y"} -> {"y","x"}  仅此而已

// 【推理】
//   ① all_to_all 指定了要搬哪个【轴名】
//   ② 轴在源维内的【顺序】不影响搬运结果
//   ③ permute 只改顺序 -> 对后续 all_to_all【没有影响】
//   ④ 所以 permute 冗余，可以安全删除

// 【等价变换】
//   原来：permute(改顺序) + all_to_all(按名搬)
//   优化：reshape(拆成两维) + all_to_all(按名搬，目标维相应调整) + reshape
//   两者语义【完全等价】

// 【对比 L4-08 的冗余消除】
//   L4-08：消除的是【完全无通信】的 reshard（前后分片相同）
//   L4-10：消除的是【有通信但可被吸收】的 permute
//   共同点：都是"消除不产生必要通信的算子"`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:20px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const step = (n, t, c) => {
      const e = U.el('div', { class: 'col', style: 'gap:5px;align-items:center;opacity:0;transition:.45s' });
      e.innerHTML = `<div class="small mono" style="font-size:10.5px;color:${c}">${n}</div>
        <div style="font-size:11.5px;text-align:center;max-width:150px;line-height:1.4">${t}</div>`;
      demo.appendChild(e); return e;
    };
    let els = [];
    tl.at(700, () => { els.push(step('①', 'all_to_all 指定<b>轴名</b>', '#38bdf8')); els[0].style.opacity = '1'; msg.innerHTML = '<span class="mono">all_to_all [{"x"}: 0->2]</span> —— 参数写的是<b>轴名</b>，不是"第几个"。'; });
    tl.at(4200, () => { els.push(step('②', '轴在源维内的<b>顺序</b><br>不影响搬运', '#4ade80')); els[1].style.opacity = '1'; msg.innerHTML = '不管 <span class="mono">x</span> 排在 <span class="mono">y</span> 前面还是后面，<b>搬的都是同一个轴</b>。'; });
    tl.at(7800, () => { els.push(step('③', 'permute 只改顺序', '#fbbf24')); els[2].style.opacity = '1'; msg.innerHTML = '<span class="mono">permute</span> 做的事就是"把 <span class="mono">{"x","y"}</span> 换成 <span class="mono">{"y","x"}</span>" —— 仅此而已。'; });
    tl.at(11400, () => { els.push(step('④', '→ permute <b>冗余</b>', '#4ade80')); els[3].style.opacity = '1'; msg.innerHTML = '结论：permute 对后续 <span class="mono">all_to_all</span> <b>没有影响</b>，可以安全删除。'; });
    tl.at(14600, () => {
      msg.innerHTML = '<b>与 L4-08 的区别</b>：L4-08 消除的是<b>完全无通信</b>的 reshard；本课消除的是<b>有通信但可被吸收</b>的 permute。';
    });
  }
},

/* ------------------------------------------------ 3 两类散开 */
{
  kicker: 'L4-10 · 通信优化',
  title: '两类散开：<span class="mono hl-a">full</span> vs <span class="mono">partial</span>',
  sub: '两个文件的区别在于：复合轴里的轴是**全部**被搬走，还是**只有部分**。',
  caption: '<b>全散开</b>：所有轴都搬到别的维。<b>部分散开</b>：只有一个轴被搬，另一个留在原维。',
  code: `// 【全散开】all_to_all_fully_scattered (133 行 / 7 用例)
// 测试注释：Tests 2-axis full scatter on split dimension 0 where both axes
//           {"x", "y"} are permuted and communicated to separate target dimensions.
//   两个轴【都】被搬走
//   two_axis_full_scatter                    搬到【不同】的目标维
//   two_axis_scatter_to_same_target_dim      搬到【同一个】目标维
//   three_axis_full_scatter                  三个轴
//   two_axis_scatter_with_untouched_axis     带【未触及】的轴
//   non_major_split_dim                      切分维【不是最 major】的
//   sub_axis_full_scatter                    【子轴】
//   downstream_all_to_all_on_other_dim       下游还有 all_to_all（在别的维上）

// 【部分散开】all_to_all_partially_scattered (98 行 / 6 用例)
// 测试注释：2-axis permutation where only "x" is communicated, leaving
//           permuted "y" on dim 0.
//   只有【一个】轴被搬走
func.func @two_axis_permuted_one_scattered(
    %arg0: tensor<16x8xf32> {...<@mesh_2d, [{"x", "y"}, {}]>})
    -> (tensor<16x8xf32> {...<@mesh_2d, [{"y"}, {"x"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{"y", "x"}, {}]>
  %1 = sdy.all_to_all [{"x"}: 0->1] %0 out_sharding=<@mesh_2d, [{"y"}, {"x"}]>
  return %1 : tensor<16x8xf32>
}
// 优化后：
//   CHECK-NOT:   sdy.collective_permute
//   reshape: 16x8 -> 2x2x4x8
//   all_to_all [{"x"}: 0->3]   <- x 搬到第 3 维
//   all_to_all [{"y"}: 1->0]   <- y 从 reshape 出的第 1 维【搬回】第 0 维
//   reshape: 2x2x4x8 -> 16x8
//
// 为什么仍然两条 all_to_all：
//   reshape 拆开后 y 到了【新产生的第 1 维】
//   但目标要求 y 在第 0 维 -> 还要搬回去

// 其余 5 个用例：
//   three_axis_permuted_two_scattered      三个轴，两个被搬
//   cyclic_three_axis_permuted_one_scattered  【循环】置换
//   non_major_split_dim_partial_scatter    切分维不是最 major
//   sub_axis_partial_scatter               子轴
//   untouched_axis_communicated_permuted_remain  混合情形`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '全散开', c: '#38bdf8', n: '7 个',
        d: '复合轴里<b>所有</b>轴<br>都被搬到别的维<br><span class="dim">可搬到不同维或同一维</span>' },
      { t: '部分散开', c: '#4ade80', n: '6 个',
        d: '<b>只有一个</b>轴被搬走<br>另一个留在原维<br><span class="dim">但 reshape 后还要搬回去</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>全散开</b>：所有轴都离开原维 —— 最"彻底"的搬运。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>部分散开</b>：只有一个轴被搬走，另一个留在原维。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>一个反直觉点</b>：部分散开时<b>仍然有两条 all_to_all</b> —— 因为 reshape 把留下的轴挪到了新维，还要搬回去。';
    });
    tl.at(12000, () => {
      msg.innerHTML = '<b>共同模式</b>：<span class="mono">permute</span>（删）→ <span class="mono">reshape</span>（拆）→ <span class="mono">all_to_all × N</span> → <span class="mono">reshape</span>（合）。';
    });
  }
},

/* ------------------------------------------------ 4 边界情形 */
{
  kicker: 'L4-10 · 通信优化',
  title: '覆盖的<span class="hl-a">边界情形</span>',
  sub: '13 个用例不只覆盖"基本情形"，还包括若干**容易出错**的边界。',
  caption: '这些边界用例说明优化器必须<b>小心</b> —— 不是所有 permute 都能删。',
  code: `// 【子轴】sub_axis_full_scatter / sub_axis_partial_scatter
//   复合轴里含【子轴】（如 {"x":(1)2}）时的处理
//   子轴让"拆开复合轴"更复杂 —— 尺寸可能不可整除

// 【非最 major 的切分维】non_major_split_dim / non_major_split_dim_partial_scatter
//   被切分的维【不是第 0 维】时的处理
//   为什么重要：reshape 的语义与"哪个维是 major"有关

// 【未触及的轴】
//   two_axis_scatter_with_untouched_axis
//   untouched_axis_communicated_permuted_remain
//   分片里还有【不需要动】的轴 —— 优化不能误伤它们

// 【下游还有 all_to_all】downstream_all_to_all_on_other_dim
//   被优化的 all_to_all 链后面还跟着别的 all_to_all
//   （在【不同的维】上）—— 不能把它们一起误删

// 【循环置换】cyclic_three_axis_permuted_one_scattered
//   三个轴形成【循环】（x->y->z->x）时的处理
//   这类置换最容易出错

// 【搬到同一目标维】two_axis_scatter_to_same_target_dim
//   两个轴搬到【同一个】维 —— 顺序就变得重要了！
//   这时仍然能删 permute 吗？用例给出了答案

// 这些边界用例的价值：
//   证明优化是【有条件】的，不是"见到 permute + all_to_all 就删"
//   条件包括：轴的位置、是否子轴、是否有未触及的轴、下游是否还有通信`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '子轴', c: '#38bdf8', d: '<span class="mono">{"x":(1)2}</span><br>尺寸可能不可整除' },
      { t: '非 major 切分维', c: '#4ade80', d: '被切分的维<br>不是第 0 维' },
      { t: '未触及的轴', c: '#fbbf24', d: '分片里还有<br><b>不需要动</b>的轴' },
      { t: '下游还有通信', c: '#c084fc', d: '后面还跟着<br>别的 <span class="mono">all_to_all</span>' },
      { t: '循环置换', c: '#fb7185', d: '<span class="mono">x-&gt;y-&gt;z-&gt;x</span><br>最容易出错' },
      { t: '搬到同一目标维', c: '#f472b6', d: '两轴搬到同维<br>顺序就重要了' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:150px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div style="font-size:11px;color:${x.c}">${x.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>6 类边界情形</b> —— 每一类都可能让优化失效。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>最容易忽略的</b>：<span class="mono">搬到同一目标维</span> —— 两个轴到了同一个维，<b>顺序就重要了</b>。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>这些用例的价值</b>：证明优化是<b>有条件的</b>，不是"见到 permute + all_to_all 就删"。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>条件包括</b>：轴的位置、是否子轴、是否有未触及的轴、下游是否还有通信。';
    });
  }
},

/* ------------------------------------------------ 5 族谱 */
{
  kicker: 'L4-10 · 通信优化',
  title: '13 个用例的<span class="hl-a">族谱</span>与小结',
  sub: '这是 L4 层唯一一个**纯优化** pass —— 它不插入任何东西，只删除。',
  caption: '与 L4-08 的冗余消除同类，但消除的对象不同。',
  code: `// 【族谱】
//   all_to_all_fully_scattered    133 行 / 7 用例
//     two_axis_full_scatter
//     two_axis_scatter_to_same_target_dim
//     three_axis_full_scatter
//     two_axis_scatter_with_untouched_axis
//     non_major_split_dim
//     sub_axis_full_scatter
//     downstream_all_to_all_on_other_dim
//
//   all_to_all_partially_scattered 98 行 / 6 用例
//     two_axis_permuted_one_scattered
//     three_axis_permuted_two_scattered
//     cyclic_three_axis_permuted_one_scattered
//     non_major_split_dim_partial_scatter
//     sub_axis_partial_scatter
//     untouched_axis_communicated_permuted_remain

// 【与 L4-08 冗余消除的对比】
//   ┌──────────┬────────────────────────┬──────────────────┐
//   │          │ 消除对象               │ 为什么可以删     │
//   ├──────────┼────────────────────────┼──────────────────┤
//   │ L4-08    │ 前后分片相同的 reshard │ 本来就不产生通信 │
//   │ L4-10    │ all_to_all 前的 permute│ 通信可被 reshape │
//   │          │                        │ 吸收             │
//   └──────────┴────────────────────────┴──────────────────┘

// 一句话总结：
//   all_to_all 按【轴名】搬运
//   -> 只改轴序的 collective_permute 是冗余的
//   -> 用 reshape 拆开复合轴、调整 all_to_all 的目标维
//   -> 省下一次真实通信`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '全散开', n: 7, c: '#38bdf8' }, { t: '部分散开', n: 6, c: '#4ade80' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.4;transition:.35s;border-color:${f.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="card-t" style="color:${f.c};font-size:12px">${f.t}</div>
        <div class="big" style="font-size:22px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 150)); msg.innerHTML = '<b>13 个用例</b>分两类 —— 区别在于复合轴里的轴是<b>全部</b>被搬走还是<b>只有部分</b>。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>与 L4-08 的对比</b>：L4-08 消除"本来就不产生通信"的 reshard；本课消除"通信可被 reshape 吸收"的 permute。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>本课是 L4 唯一的纯优化 pass</b> —— 不插入，只删除。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L4-11 讲 <span class="mono">per-instruction-partitioning</span> —— 逐指令的分区。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L4-10 · 练习',
  title: '练一练：<span class="hl-a">这条 permute 能删吗</span>',
  sub: '三道题分别考：优化内容、安全性理由、边界条件。',
  caption: '一句话总结：<b>all_to_all 按轴名搬运，只改轴序的 permute 是冗余的</b>。',
  code: `// 题 1：这个优化做了什么？省下了什么？

// 题 2：为什么删 permute 是安全的？

// 题 3：为什么"部分散开"时仍然有两条 all_to_all？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col compact-ex', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '这个优化做了什么？省下了什么？',
        a: '把 <span class="mono">collective_permute</span> + <span class="mono">all_to_all</span> 链<br>改成 <b><span class="mono">reshape</span> → <span class="mono">all_to_all</span> × N → <span class="mono">reshape</span></b>。' +
           '<br><b>省下的是一次跨设备通信</b>（那条 permute）。<span class="mono">reshape</span> 是<b>本地操作</b>，不搬数据。' +
           '<br><span class="dim">本课是 L4 层唯一的<b>纯优化</b> pass —— 不插入任何东西，只删除。</span>'
      },
      {
        q: '为什么删 <span class="mono">permute</span> 是安全的？',
        a: '<b>关键</b>：<span class="mono">all_to_all</span> 按<b>轴名</b>搬运（<span class="mono">[{"x"}: 0->2]</span>），而<b>轴在源维内的顺序不影响搬运结果</b>。' +
           '<br><span class="mono">permute</span> 做的事只是把 <span class="mono">{"x","y"}</span> 换成 <span class="mono">{"y","x"}</span> —— <b>纯顺序调整</b>。' +
           '<br><span class="dim">推理链：① all_to_all 指定轴名 → ② 顺序无关 → ③ permute 只改顺序 → ④ 对后续 all_to_all 无影响 → 可安全删除。</span>'
      },
      {
        q: '为什么"部分散开"时<b>仍然</b>有两条 <span class="mono">all_to_all</span>？',
        a: '因为 <span class="mono">reshape</span> 拆开复合轴后，<b>留下的那个轴也换了位置</b>。' +
           '<br><b>例</b>：<span class="mono">16x8 → 2x2x4x8</span> 后，<span class="mono">y</span> 到了<b>新产生的第 1 维</b>，但目标要求它在<b>第 0 维</b> → 还要一条 <span class="mono">all_to_all [{"y"}: 1->0]</span> 搬回去。' +
           '<br><span class="dim">所以"部分散开"并不意味着"通信更少"—— 只是<b>散开的轴更少</b>。这类反直觉点正是边界用例存在的价值。</span>'
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
