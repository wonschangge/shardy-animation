/* ==========================================================================
   L5-02 · lowering-sdy-collectives   （P0）
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/convert_global_to_local/ 下 6 个文件
         (sdy_all_gather 176/7, sdy_all_reduce 70/4, sdy_all_slice 77/4,
          sdy_all_to_all 96/4, sdy_reduce_scatter 138/5,
          sdy_collective_permute 104/6) = 661 行 / 30 用例
   目标：讲透六个 sdy.* 集合通信算子到 StableHLO 的映射。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 六条映射 */
{
  kicker: 'L5-02 · 集合通信降级',
  title: '★ 六条<span class="hl-a">映射总表</span>',
  sub: '`sdy.*` 集合通信 → StableHLO 的对应算子。**其中有一个例外。**',
  caption: '这是 L4-08 的<b>直接后续</b>：那里把 reshard 转成 <span class="mono">sdy.*</span>，这里把 <span class="mono">sdy.*</span> 降级成硬件指令。',
  code: `// 【六条映射】661 行 / 30 用例
//   sdy.all_gather         -> stablehlo.all_gather
//     属性：all_gather_dim / replica_groups / use_global_device_ids
//   sdy.all_reduce         -> stablehlo.all_reduce
//     属性：replica_groups / use_global_device_ids + 【reduction 区域】
//   sdy.all_slice          -> 【没有对应算子！】partition_id + dynamic_slice
//   sdy.all_to_all         -> stablehlo.all_to_all
//     属性：concat_dimension / split_count / replica_groups
//   sdy.reduce_scatter     -> stablehlo.reduce_scatter
//     属性：scatter_dimension / replica_groups + 【reduction 区域】
//   sdy.collective_permute -> stablehlo.collective_permute
//     属性：source_target_pairs

// 【两个值得注意的点】
//   ① all_slice 是【唯一的例外】
//      StableHLO 没有对应的"切片"通信算子
//      -> 它其实【不需要通信】！只是本地切片
//   ② all_reduce 与 reduce_scatter 需要【reduction 区域】
//      因为要指定"用什么算子归约"（如 add / max / min）

// 【与 L4-08 的完整链路】
//   reshard  --L4-08-->  sdy.* 通信  --L5-02-->  stablehlo 通信
//   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//   从"逻辑分片变化"一路降到"真正的硬件指令"`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'all_gather', c: '#4ade80' }, { t: 'all_reduce', c: '#38bdf8' },
      { t: 'all_slice', c: '#fb7185' }, { t: 'all_to_all', c: '#fbbf24' },
      { t: 'reduce_scatter', c: '#c084fc' }, { t: 'collective_permute', c: '#f472b6' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:170px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:10.5px;color:${x.c};overflow-wrap:anywhere">sdy.${x.t}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>六个算子</b>，其中五个有对应的 StableHLO 算子。'; });
    tl.at(4000, () => {
      els.forEach((e, i) => { if (i !== 2) e.style.opacity = '.25'; });
      msg.innerHTML = '<b>唯一的例外</b>：<span class="mono">sdy.all_slice</span> —— StableHLO <b>没有</b>对应的切片通信算子。';
    });
    tl.at(7400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>为什么</b>：<span class="mono">all_slice</span> 只是"每台设备留自己那片" —— <b>不需要跨设备通信</b>！';
    });
    tl.at(10600, () => {
      msg.innerHTML = '<b>完整链路</b>：<span class="mono">reshard</span> --L4-08--&gt; <span class="mono">sdy.*</span> --L5-02--&gt; <span class="mono">stablehlo</span>。';
    });
  }
},

/* ------------------------------------------------ 2 ★ all_gather 与两种表示 */
{
  kicker: 'L5-02 · 集合通信降级',
  title: '★ <span class="mono hl-a">replica_groups</span> 的两种表示',
  sub: 'V1 用**设备号列表**；V3 用 **mesh 轴名** —— 正是 L5-01 的 `enable-rgv3` 选项。',
  caption: '<span class="mono">all_gather.mlir</span> 有 <b>四个 RUN 行</b> —— 两个选项的 2×2 组合。',
  code: `// 【四个 RUN 行】enable-rgv3 × per-dim-all-gather = 2 × 2
// RUN: ...='enable-rgv3=false'                      | --check-prefixes=CHECK,COMBINED,V1,COMBINED-V1
// RUN: ...='per-dim-all-gather=true enable-rgv3=false' | --check-prefixes=CHECK,PER-DIM,V1,PER-DIM-V1
// RUN: ...（默认）                                   | --check-prefixes=CHECK,COMBINED,V3,COMBINED-V3
// RUN: ...='per-dim-all-gather=true'                 | --check-prefixes=CHECK,PER-DIM,V3,PER-DIM-V3
//          ^^^^^^^^^^^^^^^^^^^^ 两个开关        ^^^^^^^^^^^^^^^^^^^^^^ 四组前缀

// 【基础用例】
func.func @one_dim(%arg0 : tensor<8x16xf32> {...<@mesh_2_4, [{"x", "y"}, {}]>})
  -> (tensor<8x16xf32> {...<@mesh_2_4, [{"x"}, {}]>}){
  %0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh_2_4, [{"x"}, {}]> : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
// 局部类型：输入 1x16、输出 4x16
//   [{"x","y"}] -> [{"x"}]
//   第 0 维从"切 x×y（8 台）"变成"只切 x（2 台）"
//   局部大小从 8/(2×4)=1 变成 8/2=4

// 输出（关键属性）：
// CHECK: %[[GATHER]] = "stablehlo.all_gather"(%[[ARG0]]) <{
// CHECK-SAME:   all_gather_dim = 0 : i64,          <- 在第 0 维聚合
// CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
// V1-SAME{LITERAL}: replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]>
// V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<
//            mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "y">]>
// CHECK-SAME:   use_global_device_ids
// CHECK-SAME: }> : (tensor<1x16xf32>) -> tensor<4x16xf32>

// 【V1 解读】dense<[[0,1,2,3], [4,5,6,7]]>
//   两组，每组 4 台设备
//   [0,1,2,3] = x=0 的那 4 台（y 从 0 到 3）
//   [4,5,6,7] = x=1 的那 4 台
//   为什么这样分组：聚合的是 y 轴 -> 同一个 x 值下的 4 台要互相 gather

// 【V3 解读】replica_group_mesh_axes<mesh=@mesh_2_4, axes=["y"]>
//   用 mesh 【轴名】表示 —— 语义等价但更简洁
//   且与【设备号解耦】（改 mesh 大小时不用改这里）
//   这就是 L5-01 的 enable-rgv3 选项的效果
//   默认就是 V3（更可读、更稳定）

// 【注意 {LITERAL}】V1 那行带 {LITERAL} 修饰符
//   表示"按字面匹配，不做正则解释"
//   因为 [[0, 1, 2, 3], ...] 里的方括号会被 FileCheck 当特殊字符`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'V1（rgv3=false）', c: '#fbbf24', tag: '设备号列表',
        d: '<span class="mono">dense&lt;[[0,1,2,3],<br>[4,5,6,7]]&gt;</span><br>直观但<b>依赖设备编号</b>' },
      { t: 'V3（默认）', c: '#4ade80', tag: 'mesh 轴名',
        d: '<span class="mono">replica_group_mesh_axes<br>&lt;mesh=@mesh_2_4,<br>axes=["y"]&gt;</span><br><b>与设备号解耦</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:10.5px;overflow-wrap:anywhere">${x.t} <span class="faint small">${x.tag}</span></div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>V1 的分组逻辑</b>：聚合 <span class="mono">y</span> 轴 → 同一个 <span class="mono">x</span> 值下的 4 台设备互相 gather。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>V3 更简洁</b>：直接写"按 <span class="mono">y</span> 轴聚合" —— 不用手算设备号。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>为什么 V3 是默认</b>：更<b>可读</b>、更<b>稳定</b>（改 mesh 大小时不用改这里）。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>四个 RUN 行</b>说明这两个选项<b>正交</b> —— 可以自由组合。';
    });
  }
},

/* ------------------------------------------------ 3 ★ all_slice 例外 */
{
  kicker: 'L5-02 · 集合通信降级',
  title: '★ <span class="mono hl-a">all_slice</span>：唯一的例外',
  sub: '它**不需要通信** —— 只是"每台设备留自己那片"，用 `dynamic_slice` 就够了。',
  caption: '实现套路与 <b>L5-01 的常量处理</b>完全相同：<span class="mono">partition_id</span> + 查找表 + <span class="mono">dynamic_slice</span>。',
  code: `// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

func.func @one_dim_two_axes_xy(%arg0: tensor<16x32xf32>
      {...<@mesh_2_4_2, [{}, {}], replicated={"x", "y"}>})
    -> (tensor<16x32xf32> {...<@mesh_2_4_2, [{}, {"x", "y"}]>}) {
  %0 = sdy.all_slice [{}, {"x", "y"}] %arg0
       out_sharding=<@mesh_2_4_2, [{}, {"x", "y"}]> : tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// 输出（与 L5-01 的常量处理【同一个套路】）：
// CHECK-DAG: %[[PID]] = stablehlo.partition_id : tensor<ui32>
//   ^^^^ 注意是 partition_id，不是 replica_id！
// CHECK-DAG: %[[PIDI64]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
// CHECK-DAG: %[[OFF0]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK: %[[TABLE]] = stablehlo.constant
//           dense<[0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 28, 28]>
//           : tensor<16xi64>
//   ^^^^^ 查找表：16 台设备各自的【起始列】
// CHECK: %[[DS]] = stablehlo.dynamic_slice %[[TABLE]], %[[PIDI64]], sizes = [1]
// CHECK: %[[OFF1]] = stablehlo.reshape %[[DS]] : (tensor<1xi64>) -> tensor<i64>
// CHECK: %[[RESULT]] = stablehlo.dynamic_slice %[[ARG0]], %[[OFF0]], %[[OFF1]],
//           sizes = [16, 4] : (tensor<16x32xf32>, tensor<i64>, tensor<i64>) -> tensor<16x4xf32>
// CHECK: return %[[RESULT]] : tensor<16x4xf32>

// 【为什么 all_slice 不需要通信】
//   它的语义是"把数据切开，每台设备留自己那片"
//   这在【单机内】就是一次【本地切片】—— 不需要跨设备通信！
//   所以不需要通信算子，用 dynamic_slice 就够了

// 【与 all_gather 的鲜明对比】
//   all_gather  需要别人的数据  -> stablehlo.all_gather（真通信）
//   all_slice   只留自己的      -> dynamic_slice（无通信）

// 【查找表 [0,0,4,4,8,8,...] 的推导】
//   网格 @mesh_2_4_2 = <["x"=2, "y"=4, "z"=2]>  共 16 台
//   分片 [{"x","y"}] 切第 1 维（32 列）：
//     x=2, y=4 -> 共 8 份 -> 32/8 = 4 列一份
//   z 轴【不参与】第 1 维的切分（分片里没有 z）
//     -> z 变化时起始列【相同】-> 每个值【重复两次】
//   按 x 最 major：x=0 起始 0、x=1 起始 8...
//   最终：0,0,4,4,8,8,... （每个值重复 2 次 = z 的 2 个值）

// 【partition_id vs replica_id】
//   replica_id    副本编号（数据并行）  <- L5-01 的常量用它
//   partition_id  分区编号（模型并行）  <- 本课用它
//   因为 all_slice 是【模型并行】的操作`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'all_gather', c: '#4ade80', d: '<b>需要</b>别人的数据<br>→ <span class="mono">stablehlo.all_gather</span><br><b>真通信</b>' },
      { t: 'all_slice', c: '#fb7185', d: '<b>只留</b>自己的那片<br>→ <span class="mono">dynamic_slice</span><br><b>无通信</b>' },
      { t: '实现套路', c: '#38bdf8', d: '<span class="mono">partition_id</span><br>+ 查找表<br>+ <span class="mono">dynamic_slice</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:246px;opacity:.33;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:11px">${x.t}</div>
        <div class="card-d" style="font-size:11px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '聚合是"<b>把别人的数据拿过来</b>" —— 必须通信。',
        '切片是"<b>只保留自己那份</b>" —— 数据本来就在本地，切一刀就行。',
        '<b>与 L5-01 同一个套路</b>：查表得到偏移，再 <span class="mono">dynamic_slice</span>。',
      ][i];
    }));
    tl.at(15200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>注意用的是 <span class="mono">partition_id</span></b>（模型并行），而 L5-01 的常量用 <span class="mono">replica_id</span>（数据并行）。';
    });
  }
},

/* ------------------------------------------------ 4 归约类 */
{
  kicker: 'L5-02 · 集合通信降级',
  title: '归约类：为什么需要 <span class="hl-a">reduction 区域</span>',
  sub: '`all_reduce` 与 `reduce_scatter` 都要指定"**用什么算子归约**"。',
  caption: '两者都带 <span class="mono">use_global_device_ids</span> —— 与 <span class="mono">all_gather</span> 一致。',
  code: `// 【all_reduce】reduction 区域指定归约算子
%0 = sdy.all_reduce {"y":(1)2} %arg0 out_sharding=<@mesh_2_4, [{}, {"y":(2)2}]>
     : tensor<16x32xf32>
// 输出：
// CHECK: %[[RES]] = "stablehlo.all_reduce"(%[[ARG0]])
// V1-SAME{LITERAL}: replica_groups = dense<[[0, 2], [1, 3], [4, 6], [5, 7]]>
// V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<
//            mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "y",
//                                        sub_axis_info = (1)2>]>
//   ^^^^^^^^^^^^^^^ 注意【子轴】！因为 SDY 算子用的是 {"y":(1)2}
// CHECK-SAME: use_global_device_ids
// CHECK: ^bb0(%[[ACC]]: tensor<f32>, %[[UPD]]: tensor<f32>):
// CHECK: %[[ADD]] = stablehlo.add %[[ACC]], %[[UPD]] : tensor<f32>
//   ^^^^ 【reduction 区域】：指定"用加法归约"
// CHECK: stablehlo.return %[[ADD]] : tensor<f32>
// CHECK: }) : (tensor<16x16xf32>) -> tensor<16x16xf32>

// 【reduce_scatter】同样带 reduction 区域
// CHECK: %[[RES]] = "stablehlo.reduce_scatter"(%[[ARG0]])
// CHECK-SAME: channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>
// V1-SAME{LITERAL}: replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]>
// V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<
//            mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "x">]>
// CHECK-SAME: scatter_dimension = 1 : i64      <- 在第 1 维上边归约边切分
// CHECK-SAME: use_global_device_ids
// CHECK: (%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK:   %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
// CHECK:   stablehlo.return %1 : tensor<f32>

// 【为什么需要 reduction 区域】
//   all_reduce / reduce_scatter 都要【合并多个值】
//   但"怎么合并"不是唯一的：add / max / min / mul ...
//   -> 用一个【区域】来指定

// 【回顾 L4-01 的完整链路】
//   reduce + sharding_constraint
//     --L4-01 融合-->  sdy.reduce_scatter
//     --L5-02 降级-->  stablehlo.reduce_scatter
//   三课连起来才是完整的"归约类通信"故事`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'all_reduce', c: '#38bdf8',
        d: '归约成<b>一个值</b><br>每台设备都拿到结果<br><span class="mono">^bb0</span> 里是 <span class="mono">add</span>' },
      { t: 'reduce_scatter', c: '#c084fc',
        d: '<b>边归约边切分</b><br><span class="mono">scatter_dimension = 1</span><br>同样带 reduction 区域' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:11.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>为什么要有区域</b>：合并方式不唯一 —— <span class="mono">add</span> / <span class="mono">max</span> / <span class="mono">min</span>…'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>reduce_scatter 更复杂</b>：一边归约一边按 <span class="mono">scatter_dimension</span> 切分。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>注意子轴</b>：V3 的 <span class="mono">all_reduce</span> 里带 <span class="mono">sub_axis_info = (1)2</span> —— 因为 SDY 算子用的是 <span class="mono">{"y":(1)2}</span>。';
    });
    tl.at(12000, () => {
      msg.innerHTML = '<b>三课连起来</b>：L4-01 融合 → L4-08 转通信 → L5-02 降级 —— 归约类通信的完整故事。';
    });
  }
},

/* ------------------------------------------------ 5 其余两个 */
{
  kicker: 'L5-02 · 集合通信降级',
  title: '其余两个：<span class="mono hl-a">all_to_all</span> 与 <span class="mono hl-a">collective_permute</span>',
  sub: '前者用 `concat_dimension` + `split_count`；后者的 `source_target_pairs` 是一张**设备收发对照表**。',
  caption: '注意两者的 <b>channel_handle 只在 partition 模式下出现</b>。',
  code: `// 【all_to_all】
// RUN: ...='enable-rgv3=false' | --check-prefixes=CHECK,V1,PARTITION
// CHECK: %[[RESULT]] = "stablehlo.all_to_all"(%[[ARG0]]) <{
// PARTITION-SAME: channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>
// REPLICA-NOT: channel_handle
//   ^^^^^^^^^^^ 只在 partition 模式下有 channel_handle
// CHECK-SAME: concat_dimension = 0 : i64
// V1-SAME{LITERAL}: replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7],
//                                           [8, 10, 12, 14], [9, 11, 13, 15]]>
// V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<
//            mesh = @mesh_2_4_2, axes = [#stablehlo.axis_ref<name = "y">]>
// CHECK-SAME: split_count = 4 : i64
//   ^^^^^^^^^^^ 在维度 0 上拼接、分成 4 份

// 【collective_permute】source_target_pairs 是【设备收发对照表】
%0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2_4, [{"y":(2)2}, {"x"}]>
     : tensor<4x8xf32>
// CHECK: %[[RES]] = "stablehlo.collective_permute"(%[[ARG0]]) <{
// PARTITION-SAME: channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
// REPLICA-NOT: channel_handle
// CHECK-SAME{LITERAL}: source_target_pairs = dense<[[0, 0], [1, 4], [2, 2],
//   [3, 6], [4, 1], [5, 5], [6, 3], [7, 7]]> : tensor<8x2xi64>
// CHECK-SAME: }> : (tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK: return %[[RES]] : tensor<2x4xf32>

// 【读 source_target_pairs】
//   [1, 4] 表示"设备 1 发给设备 4"
//   [4, 1] 表示"设备 4 发给设备 1"
//   -> 成对出现，正是"交换"的语义！
//   [0,0] / [2,2] / [5,5] / [7,7] 是"发给自己"（不变）
//
// 局部类型从 4x8 变成 2x4
//   因为分片从 [{"x"},{"y"}] 变成 [{"y":(2)2},{"x"}]

// 【回顾 L4-08】
//   那里讲 collective_permute 是"轴在两个维之间交换"
//   现在看到它的【底层表示】—— 一张设备间的收发对照表

// 【channel_handle 的规律】
//   PARTITION 模式（模型并行）：有 channel_handle
//   REPLICA 模式（数据并行）：  没有
//   -> 同一副本内的通信 vs 跨副本的通信，机制不同`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'all_to_all', c: '#fbbf24',
        d: '<span class="mono">concat_dimension = 0</span><br><span class="mono">split_count = 4</span><br>拼接 + 分份' },
      { t: 'collective_permute', c: '#f472b6',
        d: '<span class="mono">source_target_pairs</span><br>设备收发<b>对照表</b><br>成对出现 = 交换' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:11.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>all_to_all</b>：每个设备把自己那份切成 N 份、分别发给 N 个设备，再拼起来。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>读对照表</b>：<span class="mono">[1,4]</span> 与 <span class="mono">[4,1]</span> <b>成对出现</b> —— 正是"交换"的语义。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>channel_handle 的规律</b>：partition 模式有、replica 模式没有 —— 两种并行的通信机制不同。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>回顾 L4-08</b>：那里讲 <span class="mono">collective_permute</span> 是"轴交换"，这里看到它的<b>底层设备对照表</b>。';
    });
  }
},

/* ------------------------------------------------ 6 族谱与小结 */
{
  kicker: 'L5-02 · 集合通信降级',
  title: '30 个用例的<span class="hl-a">族谱</span>与五个共同点',
  sub: '六个算子降级后的形态有**五条共同规律**。',
  caption: '本课把 L4-08 的"sdy 通信"一路降到了"硬件指令"。',
  code: `// 【族谱】6 个文件 / 661 行 / 30 用例
//   sdy_all_gather          176 行 / 7 用例   <- 最大（四个 RUN 行）
//   sdy_reduce_scatter      138 行 / 5 用例
//   sdy_collective_permute  104 行 / 6 用例
//   sdy_all_to_all           96 行 / 4 用例
//   sdy_all_slice            77 行 / 4 用例
//   sdy_all_reduce           70 行 / 4 用例

// 【五个共同点】
//   ① 局部类型都变小
//      因为分片更细了（如 8x16 -> 1x16）
//   ② replica_groups 两种表示
//      V1 设备号列表 / V3 mesh 轴（enable-rgv3 选项）
//   ③ 归约类带 reduction 区域
//      all_reduce / reduce_scatter 要指定算子
//   ④ channel_handle 视模式而定
//      partition 模式有、replica 模式没有
//   ⑤ all_slice 是例外
//      不需要通信 -> partition_id + dynamic_slice

// 【本课在 L5 中的位置】
//   L5-01 全局转局部总览
//   L5-02 集合通信降级（本课）
//   L5-03 结构性算子降级
//   L5-04 逐元素与形状类降级
//   L5-05 矩阵乘降级
//   L5-06 卷积降级
//   L5-07 归约降级
//   L5-08 gather/scatter 降级
//   L5-09 为整除性加 padding

// 一句话总结：
//   SDY 的集合通信算子降级为 StableHLO 的对应算子
//   唯一例外是 all_slice —— 它只是"本地切片"
//   与 L4-08 构成完整链路：reshard -> sdy.* -> stablehlo`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: 'all_gather', n: 7, c: '#4ade80' }, { t: 'reduce_scatter', n: 5, c: '#c084fc' },
      { t: 'collective_permute', n: 6, c: '#f472b6' }, { t: 'all_to_all', n: 4, c: '#fbbf24' },
      { t: 'all_slice', n: 4, c: '#fb7185' }, { t: 'all_reduce', n: 4, c: '#38bdf8' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:140px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:9px;color:${f.c};overflow-wrap:anywhere">${f.t}</div>
        <div class="big" style="font-size:16px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 100)); msg.innerHTML = '<b>30 个用例</b>分六个算子 —— <span class="mono">all_gather</span> 最大（四个 RUN 行）。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>五条共同规律</b>：局部变小 / 两种 replica_groups / 归约区域 / channel 视模式 / all_slice 例外。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>本课的价值</b>：把 L4-08 的 <span class="mono">sdy.*</span> 通信一路降到了<b>硬件指令</b>。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L5-03 讲结构性算子（constant / manual_computation / named_computation）的降级。';
    });
  }
},

];
