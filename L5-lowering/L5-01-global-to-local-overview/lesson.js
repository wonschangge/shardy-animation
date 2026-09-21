/* ==========================================================================
   L5-01 · global-to-local-overview   （L5 开篇）
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/convert_global_to_local/ 下 2 个文件
         (generic_ops 83 行 / 6 用例, replica_id 19 / 1) = 102 行 / 7 用例
   目标：讲透"全局张量 + 分片 -> 局部张量"的总体思路。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 转折点 */
{
  kicker: 'L5-01 · 全局转局部',
  title: 'L5 开篇：从"<span class="hl-a">描述分片</span>"到"<span class="hl-a">执行分片</span>"',
  sub: 'L1～L4 讲**分片怎么流动**；从本课开始讲**分片怎么变成每台设备实际持有的数据**。',
  caption: '这是整个动画的一个<b>转折点</b> —— 之前都是"逻辑层面"，从这里开始进入"物理层面"。',
  code: `// RUN: sdy_opt %s -sdy-convert-global-to-local -allow-unregistered-dialect

// 【核心动作】
//   输入：全局张量 + 分片
//   输出：局部张量（每台设备【实际持有的那份】）
//
//   例：全局 tensor<16xf32> 在 @mesh_2 = <["x"=2]> 上切 [{"x"}]
//       -> 局部 tensor<8xf32>          （16 / 2 = 8）
//       分片属性【保留不变】—— 记录"这个局部张量是怎么来的"

// 【三件事都要做】
//   ① 类型转换    每个维度按它对应的轴大小相除
//   ② 算子改写    所有算子的操作数/结果类型都换成局部
//   ③ 常量处理    【最精妙】用 replica_id + 运行时切片

// 【为什么这是转折点】
//   L1～L4：分片是【标注】，张量还是全局的（逻辑视图）
//   L5+  ：张量变成【局部的】，每台设备看到的是自己那份（物理视图）
//
// 一句话：
//   把"全局张量 + 分片"变成"每台设备实际持有的局部张量"`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 类型转换', c: '#38bdf8', d: '每维按<br>对应轴大小<b>相除</b>' },
      { t: '② 算子改写', c: '#4ade80', d: '操作数与结果<br>都换成<b>局部类型</b>' },
      { t: '③ 常量处理', c: '#fbbf24', d: '<b>最精妙</b><br><span class="mono">replica_id</span> + 切片' },
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
        '<b>规则很直观</b>：<span class="mono">x=2</span> 切 <span class="mono">tensor&lt;16&gt;</span> → 每台拿 8 个。',
        '连<b>未注册方言</b>的算子也改写 —— 因为转换是<b>按值的类型</b>驱动的。',
        '<b>问题</b>：常量在每台设备上内容<b>不同</b>，怎么表达？',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>转折点</b>：之前分片是<b>标注</b>（逻辑视图），现在张量是<b>局部的</b>（物理视图）。';
    });
  }
},

/* ------------------------------------------------ 2 类型转换规则 */
{
  kicker: 'L5-01 · 全局转局部',
  title: '类型转换：<span class="hl-a">每个维度独立相除</span>',
  sub: '`x` 只影响它切的那个维度，`y` 只影响它切的那个 —— 各算各的。',
  caption: '规则一句话：<b>维度大小 ÷ 该维度上轴的乘积</b>。',
  code: `// 网格：@mesh_2_4 = <["x"=2, "y"=4]>

// 【最简单的例子】
func.func @func_returning_sharded_arg(
    %arg0: tensor<16xf32> {...<@mesh_2, [{"x"}]>})       // x=2
    -> (tensor<16xf32> {...<@mesh_2, [{"x"}]>}) {
  return %arg0 : tensor<16xf32>
}
// CHECK-SAME: (%arg0: tensor<8xf32> {...}) -> (tensor<8xf32> {...})
//                              ^^^^^^^^ 16 / 2 = 8
// CHECK-NEXT:  return %arg0 : tensor<8xf32>

// 【多维多轴的例子】
func.func @func_with_dot_then_add(
  %arg0: tensor<8x16xf32> {...<@mesh_2_4, [{"x"}, {}]>},   // 第0维切 x=2
  %arg1: tensor<16x32xf32> {...<@mesh_2_4, [{}, {"y"}]>},  // 第1维切 y=4
  %arg2: tensor<8x32xf32> {...<@mesh_2_4, [{"x"}, {"y"}]>})// 两维都切
  -> (tensor<8x32xf32> {...<@mesh_2_4, [{"x"}, {"y"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {...} : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %1 = stablehlo.add %0, %arg2 {...} : tensor<8x32xf32>
  return %1 : tensor<8x32xf32>
}
// 逐项看（@mesh_2_4 是 x=2, y=4）：
//   张量        全局      分片              局部      算式
//   %arg0       8x16      [{"x"}, {}]       4x16      8 / 2 = 4
//   %arg1       16x32     [{}, {"y"}]       16x8      32 / 4 = 8
//   dot 结果    8x32      [{"x"}, {"y"}]    4x8       8/2=4, 32/4=8
//   %arg2       8x32      [{"x"}, {"y"}]    4x8       同上
// CHECK-NEXT:  %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {...} : (tensor<4x16xf32>, tensor<16x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:  %[[ADD:.*]] = stablehlo.add %[[DOT]], %arg2 {...} : tensor<4x8xf32>
// CHECK-NEXT:  return %[[ADD]] : tensor<4x8xf32>

// 【关键观察】每个维度【独立】按它对应的轴大小相除
//   x 只影响第 0 维（÷2），y 只影响第 1 维（÷4）

// 【无分片的类型不变】
//   while 的谓词 tensor<i1> 【不变】—— 它是无分片的（标量谓词不在网格上切）
//   规则：只有【带分片】的张量类型才转换

// 【未注册方言也转换】
//   "interpreter.print"(%arg0) : (tensor<16xf32>) -> ()
//   -> "interpreter.print"(%[[ARG0]]) : (tensor<8xf32>) -> ()
//   说明转换是"按值的类型"驱动的，与算子是否注册无关
//   （所以 RUN 行要加 -allow-unregistered-dialect）

// 【区域算子内部】
//   stablehlo.while(%iterArg = %arg0) : tensor<16xf32> -> tensor<8xf32>
//   区域内部的 add 也一样

// 【函数调用】
//   call @callee_sharded(%arg0) : (tensor<16xf32>) -> tensor<16xf32>
//   -> (tensor<8xf32>) -> tensor<8xf32>`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '规则', c: '#38bdf8', d: '维度大小 ÷<br>该维上轴的乘积' },
      { t: '无分片不变', c: '#94a3b8', d: '<span class="mono">tensor&lt;i1&gt;</span> 等<br><b>保持原样</b>' },
      { t: '未注册也转', c: '#fbbf24', d: '按<b>值的类型</b>驱动<br>与算子无关' },
      { t: '区域/调用', c: '#4ade80', d: '区域内部、<br>call 两端都改' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div style="font-size:11.5px;color:${x.c}">${x.t}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>每个维度独立</b>：<span class="mono">x</span> 只管第 0 维，<span class="mono">y</span> 只管第 1 维。',
        '<b>为什么不变</b>：<span class="mono">tensor&lt;i1&gt;</span> 是谓词，不在网格上切 —— 没有"局部"的概念。',
        '<b>这一点很重要</b>：转换是<b>类型驱动</b>的，所以能处理任意方言。',
        '<b>递归处理</b>：区域算子内部、函数调用两端都要改 —— 否则类型对不上。',
      ][i];
    }));
    tl.at(15800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>一句话规则</b>：<b>维度大小 ÷ 该维度上轴的乘积</b>。';
    });
  }
},

/* ------------------------------------------------ 3 ★ 常量处理 */
{
  kicker: 'L5-01 · 全局转局部',
  title: '★ 常量：<span class="mono hl-a">replica_id</span> + 运行时切片',
  sub: '分片常量在每台设备上**内容不同** —— 用"全局常量 + 查表切片"表达。',
  caption: '这是本课<b>最精妙</b>的部分，也是 <span class="mono">replica_id.mlir</span> 的全部内容。',
  code: `// RUN: sdy_opt %s -sdy-convert-global-to-local="replica-count=8 partition-count=1"

sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>      // 共 8 台设备

func.func @sharded_dense_replica_id()
    -> (tensor<4x2xf32> {...<@mesh_2_4, [{"x"}, {}]>}) {
  %0 = sdy.constant {...} dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]>
       : tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// 【问题】局部类型是 tensor<2x2xf32>
//   但【每台设备需要不同的 2 行】！
//     设备 0~3 需要第 0~1 行
//     设备 4~7 需要第 2~3 行

// 【解法】六步
// CHECK-NEXT: %[[GLOBAL_CST]] = stablehlo.constant dense<[[1.0, 2.0], ...]> : tensor<4x2xf32>
//   ① 保留【全局常量】—— 所有设备都有完整数据
// CHECK-NEXT: %[[RID]] = stablehlo.replica_id : tensor<ui32>
//   ② 取【当前设备号】（0~7）
// CHECK-NEXT: %[[RID_I64]] = stablehlo.convert %[[RID]] : (tensor<ui32>) -> tensor<i64>
//   ③ 类型对齐（索引用 i64）
// CHECK-NEXT: %[[TABLE]] = stablehlo.constant dense<[0, 0, 0, 0, 2, 2, 2, 2]> : tensor<8xi64>
//   ④ 【查找表】：8 台设备 -> 各自的起始行号
// CHECK-NEXT: %[[OFFSET_SLICE]] = stablehlo.dynamic_slice %[[TABLE]], %[[RID_I64]], sizes = [1]
//   ⑤ 查表取出"我的起始行"
// CHECK-NEXT: %[[START_0]] = stablehlo.reshape %[[OFFSET_SLICE]] : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT: %[[START_1]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[LOCAL_SLICE]] = stablehlo.dynamic_slice %[[GLOBAL_CST]], %[[START_0]], %[[START_1]], sizes = [2, 2]
//   ⑥ 从全局常量切出【本地那 2 行】-> tensor<2x2xf32>
// CHECK-NEXT: return %[[LOCAL_SLICE]] : tensor<2x2xf32>

// 【查找表为什么是 [0,0,0,0,2,2,2,2]】
//   轴序是 ["x"=2, "y"=4]，分片 [{"x"}, {}] 只切 x
//   设备号按 x 【最 major】排列：
//     x=0 -> 设备 0~3      x=1 -> 设备 4~7
//   x 有 2 个值，切 tensor<4> 的第 0 维 -> 每个 x 值拿 2 行
//   所以设备 0~3 起始行 0，设备 4~7 起始行 2

// 【代价与收益】
//   代价：每台设备都要持有【完整常量】+ 多做两次 dynamic_slice
//   收益：语义正确 —— 每台设备确实拿到自己该拿的那片`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const steps = [
      { n: '①', t: '保留全局常量', c: '#94a3b8' },
      { n: '②', t: 'replica_id', c: '#38bdf8' },
      { n: '③', t: 'convert', c: '#0ea5e9' },
      { n: '④', t: '查找表', c: '#fbbf24' },
      { n: '⑤', t: '查表取偏移', c: '#c084fc' },
      { n: '⑥', t: '切出本地片', c: '#4ade80' },
    ];
    const host = wrap.querySelector('#cards');
    const els = steps.map(s => {
      const e = U.el('div', { class: 'card', style: `width:118px;opacity:.33;transition:.3s;border-color:${s.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div class="small mono" style="font-size:10.5px;color:${s.c}">${s.n}</div>
        <div style="font-size:10px;margin-top:3px;line-height:1.35">${s.t}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    steps.forEach((s, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>关键决定</b>：不改成局部常量，而是<b>保留全局的</b> —— 所有设备都有完整数据。',
        '<span class="mono">replica_id</span> 取当前设备号（0~7）。',
        '索引用 <span class="mono">i64</span> —— 类型对齐。',
        '<b>核心</b>：<span class="mono">[0,0,0,0,2,2,2,2]</span> —— 8 台设备各自的<b>起始行号</b>。',
        '<span class="mono">dynamic_slice</span> 查表取出"我的起始行"。',
        '再 <span class="mono">dynamic_slice</span> 从全局常量切出<b>本地那 2 行</b>。',
      ][i];
    }));
    tl.at(19600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>这就是分片常量的表达方式</b>：全局常量 + 运行时按设备号切片。';
    });
  }
},

/* ------------------------------------------------ 4 查找表解读 */
{
  kicker: 'L5-01 · 全局转局部',
  title: '查找表 <span class="mono hl-a">[0,0,0,0,2,2,2,2]</span> 怎么来的',
  sub: '理解这个表，就理解了"**设备号 ↔ 数据位置**"的对应关系。',
  caption: '关键：设备号按轴序<b>最 major 优先</b>排列。',
  code: `// 网格：@mesh_2_4 = <["x"=2, "y"=4]>     共 8 台设备
// 分片：[{"x"}, {}]                       只切第 0 维（x）
// 张量：tensor<4x2xf32>                  第 0 维大小 4

// 【第一步：算出"每个 x 值拿几行"】
//   x 有 2 个值，要切第 0 维（大小 4）
//   -> 4 / 2 = 2 行
//   -> x=0 拿第 0~1 行，x=1 拿第 2~3 行
//   -> 起始行号：x=0 -> 0，x=1 -> 2

// 【第二步：算出"设备号 -> x 值"的对应】
//   设备号按轴序【最 major 优先】展开：
//     轴序 ["x"=2, "y"=4]，x 在前 -> x 是【最 major】
//     x 变化最慢、y 变化最快
//     设备 0: (x=0,y=0)   设备 1: (x=0,y=1)   设备 2: (x=0,y=2)   设备 3: (x=0,y=3)
//     设备 4: (x=1,y=0)   设备 5: (x=1,y=1)   设备 6: (x=1,y=2)   设备 7: (x=1,y=3)

// 【第三步：合并】
//   设备 0~3 (x=0) -> 起始行 0
//   设备 4~7 (x=1) -> 起始行 2
//   -> 查找表 = [0, 0, 0, 0, 2, 2, 2, 2]
//                    ^^^^^^^^^  ^^^^^^^^^
//                    设备 0~3   设备 4~7

// 【为什么是 4 个一组】
//   因为 y 有 4 个值 -> 每个 x 值对应【连续 4 台】设备
//   而 x 只有 2 个值 -> 表里只有两段

// 【验证】
//   设备 0 -> 表[0] = 0 -> 切第 0~1 行 ✓（x=0 该拿前半）
//   设备 7 -> 表[7] = 2 -> 切第 2~3 行 ✓（x=1 该拿后半）

// 【推广】若分片是 [{"x"}, {"y"}]（两维都切）
//   查找表就要对【每一维】各算一个偏移
//   表会变成 2 维（或两次查表）
//   本用例只切一维，所以只需一个偏移`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 每份多大', c: '#38bdf8', d: '<span class="mono">4 / 2 = 2</span> 行<br><span class="mono">x=0</span>→行 0<br><span class="mono">x=1</span>→行 2' },
      { t: '② 设备→x', c: '#fbbf24', d: '设备号按<br><b>最 major 优先</b><br><span class="mono">x</span> 变化最慢' },
      { t: '③ 合并成表', c: '#4ade80', d: '<span class="mono">[0,0,0,0,2,2,2,2]</span><br>设备 0~3 → 0<br>设备 4~7 → 2' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>先算"每个 x 值该拿哪一段"</b>：4 行分给 2 个 x 值，各 2 行。',
        '<b>关键</b>：轴序 <span class="mono">["x","y"]</span> 中 <span class="mono">x</span> 在前 → <b>最 major</b> → 变化最慢。',
        '<b>合并</b>：把"设备→x"与"x→起始行"两级映射串起来，就得到查找表。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>验证</b>：设备 0 → 表[0]=0 → 切行 0~1 ✓；设备 7 → 表[7]=2 → 切行 2~3 ✓。';
    });
  }
},

/* ------------------------------------------------ 5 四个选项 */
{
  kicker: 'L5-01 · 全局转局部',
  title: '四个<span class="hl-a">选项</span>',
  sub: 'TODOLIST 提到的四个选项 —— 说明均来自 `sdy_opt --help`（实测核对）。',
  caption: '前两个是<b>基本参数</b>；后两个影响<b>通信的表示与合并</b>（属后续课程）。',
  code: `// 【选项说明】（逐字来自 sdy_opt --help）
//   --replica-count=<long>    Number of replicas (data parallelism).
//   --partition-count=<long>  Number of partitions (model parallelism).
//   --enable-rgv3             Use StableHLO ReplicaGroupV3 (mesh-axes based)
//                             for collectives.
//   --per-dim-all-gather      Keep per-dimension all-gather without combining
//                             them into a single all-gather.

// 【replica-count / partition-count】基本参数
//   replica    = 副本数  -> 【data parallelism】（数据并行）
//   partition  = 分区数  -> 【model parallelism】（模型并行）
//   它们告诉转换器"总共多少台设备、怎么划分"
//   本课用例：replica-count=8 partition-count=1
//     -> 8 个副本、1 个分区 -> 共 8 台设备
//   这与 L1-01 的 @maximal_mesh 概念相关：
//     局部类型的大小取决于总设备数

// 【enable-rgv3】影响集合通信的【表示】
//   默认：用【设备号列表】表示 replica groups
//   rgv3：用【mesh 轴】表示（mesh-axes based）
//   -> 在 L5-02（集合通信降级）会看到实际效果

// 【per-dim-all-gather】影响通信的【合并策略】
//   默认：把多维 all-gather 【合并】成一个
//   打开：保持【逐维】all-gather
//   -> 与 L4-10 的通信优化相关

// 【注意】后两个选项不在本课的 2 个文件里
//   它们属于后续课程（L5-02 等）
//   本课只需知道它们【存在】且【作用范围】如上
//   —— 不要凭名字猜语义（这正是 check_flags.py 存在的意义）`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'replica-count', c: '#38bdf8', tag: '基本参数',
        d: '副本数<br><b>data parallelism</b>' },
      { t: 'partition-count', c: '#0ea5e9', tag: '基本参数',
        d: '分区数<br><b>model parallelism</b>' },
      { t: 'enable-rgv3', c: '#fbbf24', tag: '通信表示',
        d: '用 <b>mesh 轴</b><br>表示 replica groups' },
      { t: 'per-dim-all-gather', c: '#c084fc', tag: '通信合并',
        d: '保持<b>逐维</b><br>all-gather' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:10px;color:${x.c};overflow-wrap:anywhere">${x.t}</div>
        <div class="small" style="font-size:9px;color:${x.c};margin:2px 0">${x.tag}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>数据并行</b>：同一份模型、不同数据 —— 副本数决定"多少份"。',
        '<b>模型并行</b>：模型切开、放不同设备 —— 分区数决定"切几块"。',
        '<b>本课用例</b>：<span class="mono">replica-count=8 partition-count=1</span> → 8 个副本、1 个分区 → 共 8 台设备。',
        '<b>后续课程</b>：这两个选项的效果在 L5-02（集合通信降级）会看到。',
      ][i];
    }));
    tl.at(14400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>重要</b>：选项说明<b>逐字来自 <span class="mono">--help</span></b> —— 不要凭名字猜语义。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L5-01 · 练习',
  title: '练一练：<span class="hl-a">算出局部类型</span>',
  sub: '三道题分别考：类型转换、常量处理、查找表。',
  caption: '一句话总结：<b>维度大小 ÷ 该维上轴的乘积</b>。',
  code: `// 题 1：tensor<32x64xf32> 在 @mesh = <["x"=4, "y"=8]> 上切
//       [{"x"}, {"y"}]，局部类型是什么？

// 题 2：分片常量为什么不能直接写成局部常量？

// 题 3：查找表 [0,0,0,0,2,2,2,2] 里为什么是 4 个 0？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col compact-ex', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">tensor&lt;32x64xf32&gt;</span> 在 <span class="mono">@mesh = &lt;["x"=4, "y"=8]&gt;</span> 上切 <span class="mono">[{"x"}, {"y"}]</span>，局部类型是什么？',
        a: '<b><span class="mono">tensor&lt;8x8xf32&gt;</span></b>。' +
           '<br><b>算法</b>：<b>每个维度独立按它对应的轴大小相除</b> —— 第 0 维切 <span class="mono">x=4</span> → <span class="mono">32/4 = 8</span>；第 1 维切 <span class="mono">y=8</span> → <span class="mono">64/8 = 8</span>。' +
           '<br><span class="dim">分片属性<b>保留不变</b>（<span class="mono">[{"x"}, {"y"}]</span>）—— 它记录"这个局部张量是怎么来的"。无分片的类型（如 <span class="mono">tensor&lt;i1&gt;</span>）保持原样。</span>'
      },
      {
        q: '分片常量为什么不能直接写成局部常量？',
        a: '因为<b>每台设备需要的内容不同</b> —— 局部常量在所有设备上是<b>相同</b>的。' +
           '<br><b>例</b>：全局 <span class="mono">tensor&lt;4x2&gt;</span> 切 <span class="mono">[{"x"}, {}]</span>，局部是 <span class="mono">tensor&lt;2x2&gt;</span>，但设备 0~3 要第 0~1 行、设备 4~7 要第 2~3 行。' +
           '<br><b>解法</b>：<b>保留全局常量</b>（所有设备都有完整数据）+ 用 <span class="mono">replica_id</span> 在<b>运行时</b>切出自己那份。' +
           '<br><span class="dim">代价：每台设备都要持有完整常量 + 多做两次 <span class="mono">dynamic_slice</span>；收益：语义正确。</span>'
      },
      {
        q: '查找表 <span class="mono">[0,0,0,0,2,2,2,2]</span> 里为什么是 4 个 0？',
        a: '因为 <b><span class="mono">y</span> 有 4 个值</b> —— 每个 <span class="mono">x</span> 值对应<b>连续 4 台</b>设备。' +
           '<br><b>推导</b>：轴序 <span class="mono">["x"=2, "y"=4]</span> 中 <span class="mono">x</span> 在前 → <b>最 major</b> → 变化最慢。所以设备 0~3 的 <span class="mono">x=0</span>、设备 4~7 的 <span class="mono">x=1</span>。' +
           '<br>而 <span class="mono">tensor&lt;4&gt;</span> 分给 2 个 <span class="mono">x</span> 值 → 各 2 行 → <span class="mono">x=0</span> 起始行 0、<span class="mono">x=1</span> 起始行 2。' +
           '<br><span class="dim">验证：设备 0 → 表[0]=0 → 切行 0~1 ✓；设备 7 → 表[7]=2 → 切行 2~3 ✓。</span>'
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
