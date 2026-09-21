/* ==========================================================================
   L5-03 · lowering-sdy-structural
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/convert_global_to_local/ 下 3 个文件
         (sdy_constant 48/4, sdy_named_computation 58/2,
          sdy_manual_computation 208/8) = 314 行 / 14 用例
   目标：讲透三个"结构算子"各不相同的降级方式。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 三种方式 */
{
  kicker: 'L5-03 · 结构算子降级',
  title: '三个结构算子，<span class="hl-a">三种降级方式</span>',
  sub: '它们的处理**各不相同** —— 判断依据是"这层结构在局部视图下还有没有意义"。',
  caption: '本课是 L5-02 的姊妹课：那里是<b>通信算子</b>，这里是<b>结构算子</b>。',
  code: `// 【三种降级方式】3 个文件 / 314 行 / 14 用例
//   sdy.constant           48 行 / 4 用例
//     -> 【按分片裁剪】：splat 直接换类型，dense 用切片
//   sdy.named_computation  58 行 / 2 用例
//     -> 【保留】，只把内部类型改成局部
//   sdy.manual_computation 208 行 / 8 用例   <- 最大
//     -> 【直接展开】：区域消失，内部算子内联到外层

// 【注意】TODOLIST 说 named_computation 是"内联"
//   但【实际测试显示它被保留了】—— 以实际 IR 为准

// 【判断依据】这层结构在局部视图下还有没有意义？
//   常量的"内容"需要按分片调整  -> 【要处理】
//   命名计算的"名字"仍然有意义  -> 【保留】
//   手动计算的"区域"已无区别    -> 【展开】

// 【一句话】
//   常量要"切"、命名计算要"留"、手动计算要"展开"`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'constant', c: '#4ade80', n: '4 用例', d: '<b>按分片裁剪</b><br>splat / dense<br>处理方式不同' },
      { t: 'named_computation', c: '#38bdf8', n: '2 用例', d: '<b>保留</b><br>只改内部类型<br><span class="dim">留给 L4-12 outline</span>' },
      { t: 'manual_computation', c: '#fbbf24', n: '8 用例', d: '<b>直接展开</b><br>区域消失<br>算子内联到外层' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:11px;overflow-wrap:anywhere">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:10.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>为什么常量要处理</b>：分片后每台设备该拿的<b>内容不同</b>。',
        '<b>为什么保留</b>：<span class="mono">named_computation</span> 是"内联的函数"（L1-08）—— 结构留给 L4-12。',
        '<b>为什么能展开</b>：区域内外的分片状态<b>已经相同</b> —— 这层"壳"没必要了。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>一句话</b>：常量要"切"、命名计算要"留"、手动计算要"展开"。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 常量三情形 */
{
  kicker: 'L5-03 · 结构算子降级',
  title: '★ <span class="mono hl-a">sdy.constant</span>：splat 不需要切片',
  sub: '同样是"有分片的常量"，**splat 一步搞定，dense 要六步**。',
  caption: '这解释了一个常见疑问：为什么有些分片常量降级后很"重"，有些却很"轻"。',
  code: `// 【情形 ①：splat + 有分片】-> 一步搞定
func.func @sharded_splat()
    -> (tensor<4x4xf32> {...<@mesh_2_4, [{"x"}, {"y"}]>}) {
  %0 = sdy.constant {...} dense<1.0> : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
// 输出：
// CHECK-NEXT: %[[CST]] = stablehlo.constant dense<1.000000e+00> : tensor<2x1xf32>
//                                                                ^^^^^^^^^
//   直接变成【局部形状】的常量！不需要切片
//   因为 dense<1.0> 是【广播常量】—— 所有元素都是 1.0
//   "切出来"和"原样"没区别

// 【情形 ②：dense + 有分片】-> 六步（与 L5-01/L5-02 同一套路）
func.func @sharded_dense()
    -> (tensor<4x2xf32> {...<@mesh_2_4, [{"x"}, {}]>}) {
  %0 = sdy.constant {...} dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]>
       : tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}
// 输出六步：
// CHECK-NEXT: %[[GLOBAL_CST]] = stablehlo.constant dense<[[1.0, 2.0], ...]>
//   ① 保留【全局常量】
// CHECK-NEXT: %[[PID]] = stablehlo.partition_id : tensor<ui32>
//   ② 取设备号（注意是 partition_id，不是 replica_id！）
// CHECK-NEXT: %[[PID_I64]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
// CHECK-NEXT: %[[TABLE]] = stablehlo.constant dense<[0, 0, 0, 0, 2, 2, 2, 2]> : tensor<8xi64>
//   ③ 查找表
// CHECK-NEXT: %[[OFFSET_SLICE]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID_I64]], sizes = [1]
// CHECK-NEXT: %[[START_0]] = stablehlo.reshape %[[OFFSET_SLICE]] : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT: %[[START_1]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[LOCAL_SLICE]] = stablehlo.dynamic_slice %[[GLOBAL_CST]], %[[START_0]], %[[START_1]], sizes = [2, 2]
//   ④⑤⑥ 查表 + 切出本地那份 -> tensor<2x2xf32>

// 【情形 ③④：无分片】-> 直接转换，类型不变
func.func @unsharded_splat() -> (tensor<4x4xf32> {...[{}, {}]>}) {
  %0 = sdy.constant {...} dense<1.0> : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
// CHECK-NEXT: %[[CST]] = stablehlo.constant dense<1.000000e+00> : tensor<4x4xf32>
//                                                                ^^^^^^^^^ 类型不变
func.func @unsharded_dense() -> (tensor<4x2xi32> {...[{}, {}]>}) {
  %0 = sdy.constant {...} dense<[[1, 2], [3, 4], [5, 6], [7, 8]]> : tensor<4x2xi32>
  return %0 : tensor<4x2xi32>
}
// CHECK: %[[CST]] = stablehlo.constant
// CHECK-SAME{LITERAL}: dense<[[1, 2], [3, 4], [5, 6], [7, 8]]> : tensor<4x2xi32>
//   ^^^^^^^^ {LITERAL} 因为方括号是 FileCheck 特殊字符

// 【★ 三种情形的对照】
//   情形  常量类型   分片   处理
//   ①     splat      有     直接变【局部形状】的常量（内容相同）
//   ②     dense      有     全局常量 + partition_id + 切片
//   ③④    任意       无     直接转换，【类型不变】

// 【核心洞察】
//   splat 常量【不需要切片】—— 因为所有元素相同
//   只有 dense + 有分片 才需要 partition_id + 切片`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'splat + 分片', c: '#4ade80', n: '1 步', d: '直接变局部形状<br><b>不需要切片</b>' },
      { t: 'dense + 分片', c: '#fbbf24', n: '6 步', d: '全局常量<br>+ <span class="mono">partition_id</span><br>+ 切片' },
      { t: '无分片', c: '#94a3b8', n: '1 步', d: '直接转换<br><b>类型不变</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>为什么不需要切片</b>：<span class="mono">dense&lt;1.0&gt;</span> 所有元素都是 1.0 —— 切出来还是全 1.0。',
        '<b>与 L5-01/L5-02 同一套路</b>：查找表 <span class="mono">[0,0,0,0,2,2,2,2]</span> + <span class="mono">dynamic_slice</span>。',
        '<b>为什么不变</b>：没有分片就没有"局部"的概念 —— 所有设备拿同一份。',
      ][i];
    }));
    tl.at(15800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>核心洞察</b>：<b>splat 常量不需要切片</b> —— 只有 dense + 有分片才需要。';
    });
  }
},

/* ------------------------------------------------ 3 named 保留 */
{
  kicker: 'L5-03 · 结构算子降级',
  title: '<span class="mono hl-a">named_computation</span>：保留，只改类型',
  sub: '与 TODOLIST 的"内联"表述不同 —— **实际测试显示它被保留了**。',
  caption: '本 pass 只把<b>类型</b>改成局部，结构留给 <b>L4-12</b> 的 outline 处理。',
  code: `func.func @flat(%arg0: tensor<16x32xf32> {...<@mesh_2_4, [{"x"}, {}]>})
  -> (tensor<16x32xf32> {...<@mesh_2_4, [{"x"}, {}]>}) {
  %0 = sdy.named_computation<"my_comp">(%arg0)
       in_shardings=[<@mesh_2_4, [{"x"}, {}]>]
       out_shardings=[<@mesh_2_4, [{"x"}, {}]>]
       (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.tanh %arg1 {...<@mesh_2_4, [{"x"}, {}]>} : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// 输出：op 还在，但【所有类型变成局部】
// CHECK-LABEL: func @flat
// CHECK-SAME: (%[[ARG0]]: tensor<8x32xf32> {...})      <- 函数签名 16x32 -> 8x32
// CHECK-SAME: -> (tensor<8x32xf32> {...}) {
// CHECK-NEXT: %[[RES]] = sdy.named_computation<"my_comp">(%[[ARG0]])
//   ^^^^^^^^ op 【被保留】！
// CHECK-SAME: in_shardings=[<@mesh_2_4, [{"x"}, {}]>]   <- 分片不变
// CHECK-SAME: out_shardings=[<@mesh_2_4, [{"x"}, {}]>]
// CHECK-SAME: (%[[INNER_ARG]]: tensor<8x32xf32>) {     <- block arg 也变局部
// CHECK-NEXT:   %[[TANH]] = stablehlo.tanh %[[INNER_ARG]] {...} : tensor<8x32xf32>
// CHECK-NEXT:   sdy.return %[[TANH]] : tensor<8x32xf32>
// CHECK-NEXT: } : (tensor<8x32xf32>) -> tensor<8x32xf32>
// CHECK-NEXT: return %[[RES]] : tensor<8x32xf32>

// 【变与不变】
//   变：函数签名 / block arg / 算子操作数 / sdy.return —— 全部 16x32 -> 8x32
//   不变：in_shardings / out_shardings —— 它们记录"怎么切的"

// 【为什么保留】
//   named_computation 是"【内联的函数】"（L1-08）
//   导出期它会被 L4-12 的 export-named-computations outline 成真正的函数
//   本 pass 只需把【类型】改成局部，结构留给后续 pass

// 【嵌套的情形】@two_nested
func.func @two_nested(%arg0: tensor<16x32xf32> {...<@mesh_2_4, [{"x", "y"}, {}]>})
    -> (tensor<16x32xf32> {...<@mesh_2_4, [{"x", "y"}, {}]>}) {
  %0 = sdy.named_computation<"outer">(%arg0)
      in_shardings=[<@mesh_2_4, [{"x", "y"}, {}]>]
      out_shardings=[<@mesh_2_4, [{"x", "y"}, {}]>]
      (%arg1: tensor<16x32xf32>) {
    %1 = sdy.named_computation<"inner">(%arg1)
        in_shardings=[<@mesh_2_4, [{"x", "y"}, {}]>]
        out_shardings=[<@mesh_2_4, [{"x", "y"}, {}]>]
        (%arg2: tensor<16x32xf32>) {
      %2 = stablehlo.tanh %arg2 {...<@mesh_2_4, [{"x", "y"}, {}]>} : tensor<16x32xf32>
      sdy.return %2 : tensor<16x32xf32>
    } : (tensor<16x32xf32>) -> tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}
// 两层嵌套【都保留】，且都改成局部类型（2x32）
//   16 / (2x4) = 2   （x=2, y=4 都切在第 0 维）
// CHECK-SAME: (%[[OUTER_ARG]]: tensor<2x32xf32>) {
// CHECK-SAME: (%[[INNER_ARG]]: tensor<2x32xf32>) {
// CHECK: %[[TANH]] = stablehlo.tanh %[[INNER_ARG]] {...} : tensor<2x32xf32>
// -> 递归处理，与 L5-01 讲的"区域算子内部同样处理"一致`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '变', c: '#4ade80', d: '函数签名 / block arg<br>算子操作数 / <span class="mono">sdy.return</span><br>全部 <span class="mono">16x32</span> → <span class="mono">8x32</span>' },
      { t: '不变', c: '#94a3b8', d: '<span class="mono">in_shardings</span><br><span class="mono">out_shardings</span><br><span class="dim">它们记录"怎么切的"</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>op 被保留</b> —— 只有类型变了。这与 TODOLIST 说的"内联"<b>不同</b>，以实际 IR 为准。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>分片属性不变</b>：它们记录"这个局部张量是怎么来的"。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>为什么保留</b>：结构留给 <b>L4-12</b> 的 outline 处理 —— 本 pass 只管类型。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>嵌套也递归处理</b>：两层 <span class="mono">named_computation</span> 都改成局部类型。';
    });
  }
},

/* ------------------------------------------------ 4 ★ manual 展开 */
{
  kicker: 'L5-03 · 结构算子降级',
  title: '★ <span class="mono hl-a">manual_computation</span>：直接展开',
  sub: '**区域完全消失** —— 内部算子内联到外层。这是本课最重要的动作。',
  caption: '这回答了 TODOLIST 验收点："能预测 <span class="mono">manual_computation</span> 展开后的形状"。',
  code: `func.func @no_free_axes_two_manual_axes(%arg0 : tensor<16x32xf32> {...<@mesh_2_4_2, [{"x"}, {"z"}]>})
  -> (tensor<16x32xf32> {...<@mesh_2_4_2, [{"x"}, {"z"}]>}) {
  %0 = stablehlo.abs %arg0 {...} : tensor<16x32xf32>
  %1 = sdy.manual_computation(%0)
    in_shardings=[<@mesh_2_4_2, [{"x"}, {"z"}]>]
    out_shardings=[<@mesh_2_4_2, [{"x"}, {"z"}]>]
    manual_axes={"x", "z"}
    (%arg1: tensor<8x16xf32>) {
    %2 = stablehlo.add %arg1, %arg1 : tensor<8x16xf32>
    %3 = stablehlo.tanh %2  : tensor<8x16xf32>
    sdy.return %3 : tensor<8x16xf32>
  } : (tensor<16x32xf32>) -> (tensor<16x32xf32>)
  func.return %1 : tensor<16x32xf32>
}

// 输出：区域【完全消失】，四个算子平铺
// CHECK-LABEL: func @no_free_axes_two_manual_axes
// CHECK-SAME: (%[[ARG0]]: tensor<8x16xf32> {...}) -> (tensor<8x16xf32> {...}) {
// CHECK-NEXT: %[[ABS]] = stablehlo.abs %[[ARG0]] {...} : tensor<8x16xf32>
// CHECK-NEXT: %[[ADD]] = stablehlo.add %[[ABS]], %[[ABS]] : tensor<8x16xf32>
//                                              ^^^^^^ 直接读 %[[ABS]]
// CHECK-NEXT: %[[TANH]] = stablehlo.tanh %[[ADD]] : tensor<8x16xf32>
// CHECK-NEXT: return %[[TANH]] : tensor<8x16xf32>
//   ^^^^ 四行线性代码：abs -> add -> tanh -> return

// 【为什么可以展开】
//   manual_axes={"x", "z"} 表示"区域内这两个轴【不再分片】"
//   但【导出后所有张量已经是局部的】——
//     "区域内不分片"和"外层"【是同一回事】
//   所以区域这层"壳"【没有存在的必要】-> 直接去掉

// 【验收点答案】
//   展开后的形状 = 区域内的【局部形状】（本例 8x16）
//   区域内外【一致】

// 【8 个用例覆盖的情形】
//   no_free_axes_two_manual_axes   无自由轴、两个 manual 轴
//   one_free_axis_one_manual_axis  一个自由轴、一个 manual 轴
//   nested_manual_computations     嵌套的 manual_computation
//   stablehlo_all_gather           区域内含 all_gather
//   stablehlo_all_reduce           区域内含 all_reduce
//   stablehlo_all_to_all           区域内含 all_to_all
//   stablehlo_collective_permute   区域内含 collective_permute
//   stablehlo_reduce_scatter       区域内含 reduce_scatter

// 【最后 5 个很重要】
//   manual_computation 里【可以包含集合通信】
//   展开后它们变成【局部的通信算子】（与 L5-02 的降级配合）

// 【one_free_axis_one_manual_axis 值得注意】
//   有【自由轴】时，区域内该轴【仍然可以分片】
//   -> 展开时要【保留自由轴的分片信息】
//   这与 L1-07 讲的"自由轴仍可传播"一致`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '无自由轴', c: '#38bdf8', n: '2 用例', d: '两个 manual 轴<br>区域直接展开' },
      { t: '有自由轴', c: '#4ade80', n: '1 用例', d: '自由轴<b>仍可分片</b><br>展开时保留分片' },
      { t: '嵌套', c: '#fbbf24', n: '1 用例', d: '多层区域<br>逐层展开' },
      { t: '含集合通信', c: '#c084fc', n: '5 用例', d: '区域内可含<br><b>5 种</b>集合通信' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div style="font-size:11.5px;color:${x.c}">${x.t}</div>
        <div class="mono" style="font-size:10px;color:${x.c};margin:2px 0">${x.n}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '最简单的情形：区域内所有轴都冻结 → 直接展开成线性代码。',
        '<b>L1-07 讲过</b>："自由轴仍可传播" —— 所以展开时要保留它的分片。',
        '嵌套区域要<b>逐层展开</b>，最终完全消失。',
        '<b>与 L5-02 配合</b>：展开后的通信算子再按 L5-02 的规则降级。',
      ][i];
    }));
    tl.at(15200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>验收点答案</b>：展开后的形状 = 区域内的<b>局部形状</b> —— 区域内外一致。';
    });
  }
},

/* ------------------------------------------------ 5 对照与小结 */
{
  kicker: 'L5-03 · 结构算子降级',
  title: '三个结构算子的<span class="hl-a">对照</span>与小结',
  sub: '降级方式**取决于语义** —— 判断依据是"这层结构在局部视图下还有没有意义"。',
  caption: '本课是 L5-02（通信算子）的姊妹课。',
  code: `// 【对照表】
//   ┌──────────────────────┬──────────────┬────────────────────┐
//   │ 结构算子             │ 降级方式     │ 为什么             │
//   ├──────────────────────┼──────────────┼────────────────────┤
//   │ sdy.constant         │ 按分片裁剪   │ 内容需要"切"       │
//   │ sdy.named_computation│ 保留，改类型 │ 留给 L4-12 outline │
//   │ sdy.manual_computation│ 展开（消失）│ 区域内外的分片相同 │
//   └──────────────────────┴──────────────┴────────────────────┘

// 【族谱】3 个文件 / 314 行 / 14 用例
//   sdy_manual_computation   208 行 / 8 用例   <- 最大
//   sdy_named_computation     58 行 / 2 用例
//   sdy_constant              48 行 / 4 用例

// 【跨课呼应】
//   sdy.constant           <- L5-01（replica_id + 切片）、L3-02（常量拆分）
//   sdy.named_computation  <- L1-08（内联的函数）、L4-12（outline）
//   sdy.manual_computation <- L1-07（区域语义）、L4-09（HALO 也用它）
//                              L4-11（逐指令分区也用它）

// 【L5 的进度】
//   L5-01 全局转局部总览
//   L5-02 集合通信降级
//   L5-03 结构算子降级（本课）
//   L5-04 逐元素与形状类降级
//   L5-05 矩阵乘降级
//   L5-06 卷积降级
//   L5-07 归约降级
//   L5-08 gather/scatter 降级
//   L5-09 为整除性加 padding

// 一句话总结：
//   结构算子的降级方式取决于它的语义
//   常量要"切"、命名计算要"留"、手动计算要"展开"`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: 'manual_computation', n: 8, c: '#fbbf24' },
      { t: 'constant', n: 4, c: '#4ade80' },
      { t: 'named_computation', n: 2, c: '#38bdf8' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:190px;opacity:.4;transition:.35s;border-color:${f.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="card-t mono" style="color:${f.c};font-size:9.5px;overflow-wrap:anywhere">${f.t}</div>
        <div class="big" style="font-size:20px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 130)); msg.innerHTML = '<b>14 个用例</b>分三个算子 —— <span class="mono">manual_computation</span> 最大（8 个）。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>为什么 manual 最多</b>：它有嵌套、自由轴、含集合通信等多种组合。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>跨课呼应密集</b>：三个算子分别回指 L5-01 / L1-08 / L1-07 等。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L5-04 讲逐元素与形状类算子的降级。';
    });
  }
},

];
