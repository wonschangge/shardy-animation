/* ==========================================================================
   L5-04 · lowering-elementwise-shape
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/convert_global_to_local/ 下 4 个文件
         (pad 139/8, iota 58/4, slice 51/4, concatenate 52/2) = 300 行 / 18 用例
   目标：讲透形状类算子在分片下的局部化，以及"作用维"这条判据。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 核心判据 */
{
  kicker: 'L5-04 · 形状类降级',
  title: '★ 核心判据：<span class="hl-a">作用维是不是分片维</span>',
  sub: '形状类算子的处理方式**只取决于这一个问题**。',
  caption: '本课是 L5-02/L5-03 的姊妹课：那里是通信与结构算子，这里是<b>形状算子</b>。',
  code: `// 【四个算子】4 个文件 / 300 行 / 18 用例
//   stablehlo_pad          139 行 / 8 用例   <- 最大（边界处理最复杂）
//   stablehlo_iota          58 行 / 4 用例
//   stablehlo_slice         51 行 / 4 用例
//   stablehlo_concatenate   52 行 / 2 用例

// 【★ 核心判据】
//   这个算子的"作用维"是不是被分片的那一维？
//     iota   的作用维 = dim（生成序号的方向）
//     slice  的作用维 = 切片维
//     pad    的作用维 = 填充维
//     concat 的作用维 = 拼接维

// 【两种情形】
//   作用维【未】被分片 -> 直接局部化（参数不变，只改类型）
//   作用维【被】分片   -> 需要特殊处理（补偿位置）

// 【为什么】
//   这些算子都是【位置相关】的 —— 输出依赖元素在全局中的位置
//   分片后每台设备只看到自己那段 -> 必须补上"我在全局中的位置"

// 【与前面几课的联系】
//   L5-01 的常量、L5-02 的 all_slice、L5-03 的 dense 常量
//   都用同一个套路：partition_id + 查找表
//   本课的 iota 偏移补偿也是这个套路`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'iota', c: '#38bdf8', d: '作用维 = <span class="mono">dim</span>' },
      { t: 'slice', c: '#4ade80', d: '作用维 = <b>切片维</b>' },
      { t: 'pad', c: '#fbbf24', d: '作用维 = <b>填充维</b>' },
      { t: 'concatenate', c: '#c084fc', d: '作用维 = <b>拼接维</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:11px;color:${x.c};overflow-wrap:anywhere">${x.t}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 130)); msg.innerHTML = '<b>四个形状算子</b>，各有自己的"作用维"。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>判据</b>：作用维<b>未</b>被分片 → 直接局部化；<b>被</b>分片 → 需要补偿位置。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>为什么</b>：这些都是<b>位置相关</b>的算子 —— 输出依赖元素在全局中的位置。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>统一套路</b>：<span class="mono">partition_id</span> + 查找表 —— 与 L5-01/L5-02/L5-03 一致。';
    });
  }
},

/* ------------------------------------------------ 2 ★ iota 偏移 */
{
  kicker: 'L5-04 · 形状类降级',
  title: '★ <span class="mono hl-a">iota</span>：分片维上要<span class="hl-a">补偿偏移</span>',
  sub: '`iota` 的输出**依赖元素位置** —— 分片后每台设备必须补上"我在全局中的起点"。',
  caption: '这是本课最精妙的一处，也是"位置相关算子"问题的典型代表。',
  code: `// 【情形 ①：作用维【未】分片】-> 值不需要调整
func.func @iota_on_non_sharded_dim()
    -> (tensor<8x16xi32> {...<@mesh_4, [{}, {"x"}]>}) {
  %0 = stablehlo.iota dim = 0 {...} : tensor<8x16xi32>
  return %0 : tensor<8x16xi32>
}
// CHECK-SAME: -> (tensor<8x4xi32> {...<@mesh_4, [{}, {"x"}]>}) {
// CHECK: %[[RES]] = stablehlo.iota dim = 0 : tensor<8x4xi32>
//   ^^^^ dim = 0 保持不变！只是类型 8x16 -> 8x4
// 读法：iota 沿第 0 维生成 [0..7]，而第 0 维【未被分片】
//       -> 每台设备的 iota 值完全相同 -> 不需要调整

// 【情形 ②：作用维【被】分片】-> 必须补偿
func.func @iota_on_sharded_dim()
    -> (tensor<8x16xi32> {...<@mesh_4, [{}, {"x"}]>}) {
// CHECK: %[[LOCAL_IOTA]] = stablehlo.iota dim = 1 : tensor<8x4xi32>
//   ^^^^ 本地 iota 只会生成 [0,1,2,3] —— 每台设备都从 0 开始！
//        但全局 iota 是 [0,1,...,15]
//        设备 0 该拿 [0,1,2,3]、设备 1 该拿 [4,5,6,7]...
//        -> 【错了】！必须加偏移
// CHECK: %[[PID]] = stablehlo.partition_id : tensor<ui32>
// CHECK: %[[PID_I64]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
// CHECK: %[[TABLE]] = stablehlo.constant dense<[0, 4, 8, 12]> : tensor<4xi64>
//   ^^^^^ 4 台设备各自的【起始序号】：设备 0->0, 1->4, 2->8, 3->12
//         （每台 4 个值，因为 16/4 = 4）
// -> 本地 iota + 偏移 = 正确的全局序号

// 【为什么不直接切全局 iota】
//   全局 iota 本身也要【算出来】（16 个值）
//   而"本地 iota（4 个值）+ 加偏移"【更省】

// 【与 L5-01 常量处理的对比】
//   常量：每台设备【内容】不同  -> 全局常量 + 切片
//   iota：每台设备【序号起点】不同 -> 本地 iota + 加偏移
//   相同点：都用 partition_id + 查找表`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '作用维未分片', c: '#4ade80', d: '<span class="mono">dim = 0</span> 不变<br>只是类型 <span class="mono">8x16</span>→<span class="mono">8x4</span><br><b>值不需要调整</b>' },
      { t: '作用维被分片', c: '#fb7185', d: '本地 iota 都从 0 开始<br><b>错了</b>！<br>必须加偏移' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>为什么不用调整</b>：第 0 维未被分片 → 每台设备的 iota <b>完全相同</b>。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>问题所在</b>：<span class="mono">iota dim=1</span> 在每台设备上都生成 <span class="mono">[0,1,2,3]</span> —— 但设备 1 该拿 <span class="mono">[4,5,6,7]</span>。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>解法</b>：<span class="mono">partition_id</span> + 查找表 <span class="mono">[0,4,8,12]</span> 得到起始序号，加到本地 iota 上。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>对比常量处理</b>：常量是"内容不同"→切片；iota 是"序号起点不同"→加偏移。';
    });
    tl.at(14400, () => {
      msg.innerHTML = '<b>统一套路</b>：都用 <span class="mono">partition_id</span> + 查找表 —— 这是 Shardy 处理"位置相关"问题的通用手法。';
    });
  }
},

/* ------------------------------------------------ 3 slice */
{
  kicker: 'L5-04 · 形状类降级',
  title: '<span class="mono hl-a">slice</span>：切片维未分片时<span class="hl-a">参数不变</span>',
  sub: '只有**分片维**的参数需要缩放 —— 切片维的参数原样保留。',
  caption: '这是"作用维"判据最清晰的一个例子。',
  code: `// 【情形 ①：全复制后切片】参数与全局一致
func.func @replicated_after_all_gather(
    %arg0: tensor<8x16xf32> {...<@mesh_4_2, [{"y"}, {}]>}) -> tensor<4x8xf32> {
  %0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh_4_2, [{}, {}]>
       : tensor<8x16xf32>
  %1 = stablehlo.slice %0 [0:4, 0:8] : (tensor<8x16xf32>) -> tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}
// CHECK: %[[GATHER]] = "stablehlo.all_gather"(%[[ARG0]]) <{all_gather_dim = 0 : i64,
// CHECK-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<
//               mesh = @mesh_4_2, axes = [#stablehlo.axis_ref<name = "y">]>
// CHECK-SAME: (tensor<4x16xf32>) -> tensor<8x16xf32>
// CHECK: %[[SLICE]] = stablehlo.slice %[[GATHER]] [0:4, 0:8] : (tensor<8x16xf32>) -> tensor<4x8xf32>
// 读法：all_gather 之后是【全复制】状态，每台设备都有完整数据
//       -> 切片是【本地操作】，参数与全局一致

// 【情形 ②：切片维【未】分片】-> 切片参数不变
func.func @slicing_dim_not_sharded(
    %arg0: tensor<16x32xf32> {...<@mesh_4_2, [{}, {"x"}]>})
    -> (tensor<4x32xf32> {...<@mesh_4_2, [{}, {"x"}]>}) {
  %0 = stablehlo.slice %arg0 [4:12:2, 0:32] {...} : (tensor<16x32xf32>) -> tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}
// CHECK-SAME: %arg0: tensor<16x8xf32> {...}) -> (tensor<4x8xf32> {...}) {
// CHECK: %[[RES]] = stablehlo.slice %arg0 [4:12:2, 0:8] : (tensor<16x8xf32>) -> tensor<4x8xf32>
//                                     ^^^^^^^^^^  ^^^^^
//                                     第0维不变    第1维 32->8
// 【逐项对比】
//   参数        全局        局部        变化
//   第0维(未分片) 4:12:2     4:12:2      【不变】✓
//   第1维(分片)   0:32       0:8         32/4 = 8
//   输入类型     16x32      16x8        第1维 ÷4
//   结果类型     4x32       4x8         第1维 ÷4

// 【为什么切片维参数不变】
//   切片是【本地操作】—— 每台设备对自己那份切一刀
//   如果切片维是【完整】的，那"切哪里"就是确定的
//   （第 0 维没被分片 -> 每台设备看到的第 0 维都是完整的 16）

// 【隐含的另一面】
//   如果【切片维被分片】，起始/结束就要按设备调整
//   （与 iota 的偏移补偿同类问题）`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '全复制后切片', c: '#4ade80', d: '<span class="mono">all_gather</span> 后每台都有<br>完整数据 → 切片是<b>本地操作</b><br>参数与全局一致' },
      { t: '切片维未分片', c: '#38bdf8', d: '切片维参数<b>不变</b><br>只缩放分片维<br><span class="dim">32→8 而 4:12:2 保留</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>全复制的好处</b>：数据完整 → 后续的切片/reshape 等本地操作都可以"按全局参数"做。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>关键对比</b>：<span class="mono">4:12:2</span>（第 0 维）<b>不变</b>，<span class="mono">0:32</span>（第 1 维）变成 <span class="mono">0:8</span>。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>为什么不变</b>：第 0 维没被分片 → 每台设备看到的都是完整的 16 → "切哪里"是确定的。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>隐含的另一面</b>：若<b>切片维被分片</b>，起始/结束就要按设备调整 —— 与 iota 同类问题。';
    });
  }
},

/* ------------------------------------------------ 4 concatenate 与 pad */
{
  kicker: 'L5-04 · 形状类降级',
  title: '<span class="mono hl-a">concatenate</span> 与 <span class="mono hl-a">pad</span>',
  sub: '前者在非拼接维分片时直接转换；后者的**边界处理**最复杂。',
  caption: '<span class="mono">pad</span> 有 8 个用例 —— 是本课最大的一族。',
  code: `// 【concatenate】分片在非拼接维 -> 直接转换
func.func @sharded_non_concat_dim(
  %arg0: tensor<16x4xf32> {...<@mesh_2, [{"x"}, {}]>},
  %arg1: tensor<16x2xf32> {...<@mesh_2, [{"x"}, {}]>},
  %arg2: tensor<16x1xf32> {...<@mesh_2, [{"x"}, {}]>})
// CHECK-SAME: (%[[ARG0]]: tensor<8x4xf32> {...},
// CHECK-SAME:  %[[ARG1]]: tensor<8x2xf32> {...},
// CHECK-SAME:  %[[ARG2]]: tensor<8x1xf32> {...})
// CHECK-SAME: -> (tensor<8x7xf32> {...})
// 读法：拼接维是第 1 维，分片在第 0 维 -> 不冲突
//   各操作数【各自局部化】：16x4->8x4、16x2->8x2、16x1->8x1
//   拼接结果：8x(4+2+1) = 8x7 ✓
//   【拼接维的大小不变】—— 每台设备都拼接了同样多的列
// 为什么可以直接转换：拼接是【逐元素对齐】的操作
//   每台设备在第 1 维上做同样的拼接，互不影响
// 隐含：若分片【在拼接维】上，情况就复杂了（本文件未覆盖）

// 【pad】边界处理最复杂（8 个用例）
// 【情形 ①：填充维未分片】规则与 slice 相同 -> 参数不变
// 【情形 ②：全复制后 pad】参数与全局一致
func.func @replicated_after_all_gather(
    %arg0: tensor<8x16xf32> {...<@mesh_2_4, [{"x"}, {}]>}) -> tensor<10x18xf32> {
  %pv = stablehlo.constant dense<0.0> : tensor<f32>
  %0 = sdy.all_gather [{"x"}, {}] %arg0 out_sharding=<@mesh_2_4, [{}, {}]>
       : tensor<8x16xf32>
  %1 = stablehlo.pad %0, %pv, low = [1, 1], high = [1, 1], interior = [0, 0]
       : (tensor<8x16xf32>, tensor<f32>) -> tensor<10x18xf32>
  return %1 : tensor<10x18xf32>
}
// CHECK: %[[PAD]] = stablehlo.pad %[[GATHER]], %[[CST]], low = [1, 1], high = [1, 1],
//                    interior = [0, 0] : (tensor<8x16xf32>, tensor<f32>) -> tensor<10x18xf32>
//   ^^^^ 参数与全局一致（因为 gather 后是全复制）

// 【★ 情形 ③：在【分片维】上 pad】-> uniform vs non-uniform
//   pad_sharded_dim_uniform_on_partitions_1     均匀情形 1
//   pad_sharded_dim_uniform_on_partitions_2     均匀情形 2
//   pad_sharded_dim_non_uniform_on_partitions   非均匀
// 为什么这个区分重要：
//   在分片维上 pad 时，【只有边界的那台设备】需要补数据
//   中间的设备完全不受影响
//   均匀   -> padding 量恰好让各分区"补齐" -> 本地 pad 参数【相同】
//   非均匀 -> 只影响首/尾设备 -> 本地参数【不同】，需按设备号判断

// 【情形 ④：负 padding】pad 也能【裁剪】
//   pad_negative_edges_non_sharded_dim     负 padding，填充维未分片
//   pad_negative_edges_sharded_dim         负 padding，填充维分片
//   pad_negative_edges_interior_sharded_dim 负 padding + interior + 分片维
// 负 padding 意味着"裁剪"—— low/high 可以是负数
// 所以 stablehlo.pad 实际上【同时承担了 pad 与 slice 的功能】`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'concat 非拼接维', c: '#c084fc', n: '直接转换', d: '各操作数各自局部化<br>拼接维大小不变' },
      { t: 'pad 未分片维', c: '#4ade80', n: '参数不变', d: '规则与 slice 相同' },
      { t: 'pad 分片维', c: '#fbbf24', n: 'uniform/non', d: '<b>边界处理</b><br>只有首尾设备受影响' },
      { t: 'pad 负 padding', c: '#fb7185', n: '裁剪', d: '<span class="mono">low/high</span> 可为负<br>兼具 slice 功能' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div style="font-size:11px;color:${x.c}">${x.t}</div>
        <div class="mono" style="font-size:9.5px;color:${x.c};margin:2px 0">${x.n}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>为什么可以直接转换</b>：拼接是<b>逐元素对齐</b>的操作 —— 每台设备做同样的拼接。',
        '填充维未分片 → 每台设备的"填充位置"相同 → 参数不变。',
        '<b>最复杂的情形</b>：在分片维上 pad 时，只有<b>首尾设备</b>需要补 —— 中间的不受影响。',
        '<b>负 padding</b> 让 <span class="mono">pad</span> 兼具裁剪功能 —— 所以 3 个负 padding 用例单独成组。',
      ][i];
    }));
    tl.at(16000, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>为什么 pad 用例最多</b>：它有<b>四种情形</b>（未分片 / 全复制 / 分片维 uniform-nonuniform / 负 padding）。';
    });
  }
},

/* ------------------------------------------------ 5 对照与小结 */
{
  kicker: 'L5-04 · 形状类降级',
  title: '四个算子的<span class="hl-a">对照</span>与小结',
  sub: '判据统一：**作用维是不是分片维**。',
  caption: '本课与 L5-01/L5-02/L5-03 共享同一个"位置补偿"套路。',
  code: `// 【对照表】
//   算子         作用维      未分片              被分片
//   iota         dim         直接转换            补偿偏移（partition_id + 表）
//   concatenate  拼接维      直接转换            需额外处理（本文件未覆盖）
//   slice        切片维      【参数不变】        起始/结束按设备调整
//   pad          填充维      参数不变            uniform / non-uniform

// 【族谱】4 个文件 / 300 行 / 18 用例
//   stablehlo_pad          139 行 / 8 用例   <- 最大
//   stablehlo_iota          58 行 / 4 用例
//   stablehlo_slice         51 行 / 4 用例
//   stablehlo_concatenate   52 行 / 2 用例

// 【跨课呼应】
//   iota        <- L5-01（常量切片）、L5-02（all_slice）
//   slice       <- L5-02（all_gather 后全复制）
//   pad         <- L4-09（permutation 因子的 halo 处理）
//   concatenate <- L4-03（concatenate 的 reshard 规则）

// 【L5 的进度】
//   L5-01 全局转局部总览      L5-02 集合通信降级
//   L5-03 结构算子降级        L5-04 形状类降级（本课）
//   L5-05 矩阵乘降级          L5-06 卷积降级
//   L5-07 归约降级            L5-08 gather/scatter 降级
//   L5-09 为整除性加 padding

// 一句话总结：
//   形状类算子的处理取决于"作用维是不是分片维"
//   不是 -> 直接局部化；是 -> 需要补偿位置
//   所有"位置相关"的算子都用同一个套路：partition_id + 查找表`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: 'pad', n: 8, c: '#fbbf24' }, { t: 'iota', n: 4, c: '#38bdf8' },
      { t: 'slice', n: 4, c: '#4ade80' }, { t: 'concatenate', n: 2, c: '#c084fc' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:150px;opacity:.4;transition:.35s;border-color:${f.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="card-t mono" style="font-size:10.5px;color:${f.c}">${f.t}</div>
        <div class="big" style="font-size:20px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 130)); msg.innerHTML = '<b>18 个用例</b>分四个算子 —— <span class="mono">pad</span> 最大（8 个，四种情形）。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>判据统一</b>：作用维是不是分片维 —— 不是就简单，是就要补偿位置。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>跨课呼应</b>：本课的"位置补偿"与 L5-01/L5-02/L5-03 是<b>同一个套路</b>。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L5-05 讲矩阵乘降级（P0）—— 经典并行策略的 IR 体现。';
    });
  }
},

];
