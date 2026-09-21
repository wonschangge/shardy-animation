/* ==========================================================================
   L4-17 · merger-and-debug-cleanup   （L4 最后一课）
   --------------------------------------------------------------------------
   覆盖：transforms/export/test/ 下 3 个文件
         (constant_or_scalar_merger 182 / 13, remove_propagation_debug_info 112 / 2,
          propagate_to_func_results 189 / 20) = 483 行 / 35 用例
   目标：讲透三个"最后的清理"pass，并收束 L4 全层。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 三个 pass */
{
  kicker: 'L4-17 · 合并与调试清理',
  title: 'L4 最后一课：<span class="hl-a">优化</span>、<span class="hl-a">补全</span>、<span class="hl-a">清理</span>',
  sub: '在 L4-15（收尾清理）与 L4-16（边下沉）之后，还有三件事要做。',
  caption: '一句话：<b>合并可以合并的、补全应该补全的、删掉不该留的</b>。',
  code: `// 【三个 pass】3 个文件 / 483 行 / 35 用例
//   --sdy-constant-or-scalar-merger          182 行 / 13 用例
//     合并重复的常量（sdy.constant 与 stablehlo.constant 都处理）
//   -sdy-propagate-to-func-results           189 行 / 20 用例
//     把函数体内终止符值的分片【传播到函数结果签名】
//   -sdy-remove-propagation-debug-info       112 行 /  2 用例
//     移除 sdy.propagation_edges 等调试信息

// 【三种类型】
//   ① 优化：合并重复常量        -> IR 变小
//   ② 补全：补全函数结果分片    -> 信息完整
//   ③ 清理：删调试信息          -> 去噪声

// 【它们都是"最后的收尾"】
//   L4-15 收尾清理（闭合开维、删辅助信息、处理网格）
//   L4-16 边下沉（分片落到值上）
//   L4-17 本课（优化 + 补全 + 清理）
//   -> 之后就是交给后端的最终 IR

// 一句话：
//   合并可以合并的、补全应该补全的、删掉不该留的`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 优化', c: '#4ade80', n: '13 用例',
        d: '合并重复常量<br><b>IR 变小</b>' },
      { t: '② 补全', c: '#38bdf8', n: '20 用例',
        d: '分片传播到<br><b>函数结果签名</b>' },
      { t: '③ 清理', c: '#fbbf24', n: '2 用例',
        d: '删 <span class="mono">propagation_edges</span><br><b>去噪声</b>' },
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
        '<b>与 L3-02 相反</b>：那里把常量<b>拆开</b>，这里<b>合并</b>回去。',
        '<b>与 L4-16 配合</b>：那里把边下沉到值上，这里把值的分片传到签名上。',
        '<b>同类设计</b>：与 L2-03 / L4-05 / L4-08 的观测性选项一样，最终都要清掉。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>本课是 L4 最后一课</b> —— 之后进入 L5 传播算法的深水区。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 常量合并 */
{
  kicker: 'L4-17 · 合并与调试清理',
  title: '★ 常量合并：与 <span class="mono hl-a">L3-02</span> 的<span class="hl-a">镜像关系</span>',
  sub: 'L3-02 把常量**拆开**（避免假依赖）；本课把它们**合并回去**。',
  caption: '这是"同一件事在两个阶段做相反操作"的又一个例子。',
  code: `// RUN: sdy_opt %s --sdy-constant-or-scalar-merger | FileCheck %s

// 【sdy.constant 的合并】
func.func @merge_constants_sdy() -> tensor<f32> {
  %0 = sdy.constant dense<1.000000e+00> : tensor<f32>
  %1 = sdy.constant dense<1.000000e+00> : tensor<f32>
  %2 = stablehlo.add %0, %1 : tensor<f32>
  return %2 : tensor<f32>
}
// 输出：
// CHECK: %[[C0:.*]] = sdy.constant dense<1.0{{.*}}> : tensor<f32>
// CHECK-NOT: sdy.constant                 <- 后面再没有第二个
// CHECK: stablehlo.add %[[C0]], %[[C0]] : tensor<f32>
//                       ^^^^^^^^^^^^^ 两个操作数指向【同一个】常量

// 【stablehlo.constant 同样处理】
func.func @merge_constants_constant_like() -> tensor<f32> {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
}
// CHECK: %[[C0:.*]] = stablehlo.constant dense<1.0{{.*}}> : tensor<f32>
// CHECK-NOT: stablehlo.constant
// CHECK: stablehlo.add %[[C0]], %[[C0]] : tensor<f32>
//   -> 所以 pass 名叫 constant 【or scalar】 merger

// 【★ 与 L3-02 的镜像关系】
//   L3-02  常量与标量【拆分】  N 个使用 -> N 份
//          为什么：避免传播期产生【假依赖】
//                 多个使用共享一个常量 -> 传播器以为它们【必须同分片】
//                 -> 限制了传播自由度
//   L4-17  常量与标量【合并】  相同内容 -> 一份
//          为什么：传播已结束 -> 假依赖不再是问题
//                 合并后 IR 更小、常量更少 -> 后端编译更快
//
// 【同类镜像】
//   L3-06 内联  <->  L4-12 outline
//   L3-07 提升  <->  L4-15 内联
//   L3-02 拆分  <->  L4-17 合并`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:16px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const node = (label, sub, color) => {
      const c = U.el('div', { class: 'col', style: 'gap:5px;align-items:center;opacity:0;transition:.45s' });
      c.innerHTML = `<div class="chip ${color}" style="padding:7px 12px;font-size:11px">${label}</div>
        <div class="small faint" style="font-size:9.5px;text-align:center;max-width:160px;line-height:1.35">${sub}</div>`;
      demo.appendChild(c); return c;
    };

    tl.at(700, () => {
      node('1 个常量 · N 个使用', '源 IR 里的形态', 'c0').style.opacity = '1';
      msg.innerHTML = '起点：一个常量被多处使用 —— 这在传播期会造成<b>假依赖</b>。';
    });
    tl.at(4200, () => {
      demo.insertAdjacentHTML('beforeend', '<div class="arrow" style="font-size:16px">→</div>');
      node('N 份（拆分）', 'L3-02 导入期<br>避免假依赖', 'c2').style.opacity = '1';
      msg.innerHTML = '<b>L3-02 拆开</b>：每个使用一份 —— 传播器就不会以为它们"必须同分片"。';
    });
    tl.at(8600, () => {
      demo.insertAdjacentHTML('beforeend', '<div class="arrow" style="font-size:16px">→</div>');
      node('1 份（合并）', 'L4-17 导出期<br>IR 更小', 'c4').style.opacity = '1';
      msg.innerHTML = '<b>本课合并回去</b>：传播已结束，假依赖不再是问题 —— 合并让 IR 更小。';
    });
    tl.at(12400, () => {
      msg.innerHTML = '<b>为什么现在安全</b>：分片<b>已经定下来</b>了，不再需要"传播自由度"。';
    });
    tl.at(15000, () => {
      msg.innerHTML = '<b>同类镜像</b>：L3-06 内联 ↔ L4-12 outline；L3-07 提升 ↔ L4-15 内联；L3-02 拆分 ↔ 本课合并。';
    });
  }
},

/* ------------------------------------------------ 3 补全与清理 */
{
  kicker: 'L4-17 · 合并与调试清理',
  title: '补全函数结果分片 & 删调试信息',
  sub: '`propagate_to_func_results` 把值的分片传到**签名**上；`remove_propagation_debug_info` 删掉调试记录。',
  caption: '前者与 L4-16 配合，后者与 L2-03 / L4-05 / L4-08 的观测性选项同类。',
  code: `// 【propagate_to_func_results】补全函数结果签名
// RUN: sdy_opt %s -allow-unregistered-dialect -sdy-propagate-to-func-results

// test: simple
sdy.mesh @mesh = <["x"=2]>

func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = call @foo(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// 输出：@foo 的【函数结果签名】被标上分片
// CHECK-LABEL: func private @foo
// CHECK-SAME:  -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
func.func private @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<
        [<@mesh, [{"x"}]>]>} :  tensor<8xf32>
  //   ^^^^^^^^^^^^ 算子结果【有】分片
  return %0 : tensor<8xf32>
}
// 读法：@foo 内部 %0 有分片，但函数结果签名原本【没有】
//       -> 把 %0 的分片【传播】到签名上
// 为什么：函数结果的分片必须【反映实际返回值的分片】
//         否则调用者看到的签名与真实情况不符

// 【与 L4-16 的配合】
//   L4-16 sink-func-data-flow-edges  把【边】的分片【下沉】到值上
//   本课  propagate-to-func-results  把【值】的分片【传播】到签名上
//   两步配合 -> 函数边界上的分片信息【完整】了
//               既在返回值上，也在签名上

// 【20 个用例】含各种边界：
//   // test: simple
//   // test: terminator value has no sharding, func result does not have either.
//   -> 终止符值无分片、函数结果也无 -> 什么都不做
//   其余覆盖：终止符有分片 / 结果已有分片 / 多结果 等

// 【remove_propagation_debug_info】删调试信息
// RUN: sdy_opt %s -split-input-file -sdy-remove-propagation-debug-info
// CHECK-NOT: sdy.propagation_edges
//   传播期会记录"分片是怎么传播过来的"（传播路径）
//   用于调试（如 dump propagation trace）
//   为什么要删：它【不是语义信息】—— 只描述"过程"，不描述"结果"
//               保留会让 IR 变大、干扰后续处理
//   同类：L2-03 keep-sharding-rules / L4-05 mark-partial-result /
//         L4-08 keep-redundant-reshards  -> 都是观测性信息，最终都要清掉
//   注意：文件只有 2 个用例但有 112 行 -> 一个用例里有很多 IR`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'propagate_to_func_results', c: '#38bdf8', n: '20 用例',
        d: '值 → <b>函数结果签名</b><br>与 L4-16 配合<br><span class="dim">信息完整</span>' },
      { t: 'remove_propagation_debug_info', c: '#fbbf24', n: '2 用例',
        d: '删 <span class="mono">sdy.propagation_edges</span><br><b>不是语义信息</b><br><span class="dim">只描述过程</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:10.5px;overflow-wrap:anywhere">${x.t} <span class="faint small">${x.n}</span></div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>为什么要补全</b>：函数结果的分片必须<b>反映实际返回值的分片</b>。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>为什么可以删</b>：它<b>不是语义信息</b> —— 只描述"传播过程"，不描述"结果是什么"。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>与 L4-16 的两步配合</b>：先把边下沉到值上，再把值的分片传到签名上 —— 函数边界信息就完整了。';
    });
    tl.at(12000, () => {
      msg.innerHTML = '<b>同类观测性选项</b>：L2-03 <span class="mono">keep-sharding-rules</span> / L4-05 <span class="mono">mark-partial-result</span> / L4-08 <span class="mono">keep-redundant-reshards</span> —— 最终都要清掉。';
    });
  }
},

/* ------------------------------------------------ 4 L4 收官 */
{
  kicker: 'L4-17 · 合并与调试清理',
  title: 'L4 收官：<span class="hl-a">17 课的主线</span>',
  sub: 'L4 讲"传播之后分片怎么落地" —— 从 reshard 插入到最终清理。',
  caption: '完成后 L4 层 17 课全部结束，覆盖率突破 50%。',
  code: `// 【L4 的 17 课】
//   L4-01 导出流水线总览              2 文件
//   L4-02 reshard 插入总纲（★）       2
//   L4-03 逐元素与形状类              9
//   L4-04 矩阵与卷积类                4
//   L4-05 归约与排序类                4
//   L4-06 访存与通信类                3
//   L4-07 结构性场景                  9
//   L4-08 reshard 转集合通信（★核心） 2
//   L4-09 置换因子消解                2
//   L4-10 通信优化                    2
//   L4-11 逐指令分区                  3
//   L4-12 导出命名计算                1
//   L4-13 调用图还原                  2
//   L4-14 单设备与未归约              3
//   L4-15 导出收尾（P0）              7
//   L4-16 边下沉与转换                3
//   L4-17 合并与调试清理（本课）      3

// 【一条主线】
//   传播后的 IR
//     -> 插入 reshard（L4-02～07，按算子族展开）
//     -> 转成集合通信（L4-08～10）
//     -> 处理特殊结构（L4-11～14）
//     -> 收尾清理（L4-15～17）
//   = 交给后端的 IR

// 【四条判据（L4-02～07 总结）】
//   ① 冲突在哪两侧？          -> reshard 插在【之前】还是【之后】
//   ② 有没有 reduction 因子？ -> 有则 unreduced + all_reduce
//   ③ 结果能不能靠归约合并？  -> 不能则【全复制】
//   ④ 是不是跨边界？          -> 区域/函数/mesh/单设备 -> 必须显式转换

// 【核心对照表（L4-08）】
//   去掉轴 -> all_gather    加上轴 -> all_slice
//   轴跨维移动 -> all_to_all  轴互换 -> collective_permute
//   前后分片相同 -> 删除

// 【下一层 L5】传播算法本身 —— 三级金字塔、优先级、冲突裁决`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:8px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const groups = [
      { t: '总纲', items: ['L4-01', 'L4-02'], c: '#94a3b8' },
      { t: '算子族', items: ['L4-03', 'L4-04', 'L4-05', 'L4-06', 'L4-07'], c: '#38bdf8' },
      { t: 'collective', items: ['L4-08', 'L4-09', 'L4-10'], c: '#4ade80' },
      { t: '特殊结构', items: ['L4-11', 'L4-12', 'L4-13', 'L4-14'], c: '#fbbf24' },
      { t: '收尾', items: ['L4-15', 'L4-16', 'L4-17'], c: '#c084fc' },
    ];
    const host = wrap.querySelector('#cards');
    const els = groups.map(g => {
      const e = U.el('div', { class: 'card', style: `width:146px;opacity:.35;transition:.3s;border-color:${g.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div style="font-size:11px;color:${g.c};font-weight:600">${g.t}</div>
        <div class="mono" style="font-size:9.5px;color:${g.c};line-height:1.5;margin-top:3px">${g.items.join('<br>')}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    groups.forEach((g, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.35');
      msg.innerHTML = [
        '<b>L4-01/02</b>：总览 + reshard 插入总纲 —— 后面所有课都回指这两课。',
        '<b>L4-03～07</b>：按<b>算子族</b>展开 —— 元素级/矩阵/归约/访存/结构性。',
        '<b>L4-08～10</b>：把 reshard 转成<b>集合通信</b>并优化。',
        '<b>L4-11～14</b>：处理<b>特殊结构</b> —— 逐指令/命名计算/调用图/单设备。',
        '<b>L4-15～17</b>：<b>收尾</b> —— 清理、下沉、合并与调试清理。',
      ][i];
    }));
    tl.at(16400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>L4 完成</b>：17 课覆盖导出流水线全流程 —— 下一层 L5 讲传播算法本身。';
    });
  }
},

/* ------------------------------------------------ 5 族谱 */
{
  kicker: 'L4-17 · 合并与调试清理',
  title: '3 个 pass / 35 个用例的<span class="hl-a">族谱</span>',
  sub: '三个 pass 都很小，但各自解决一个具体问题。',
  caption: 'L4 层的最后一课 —— 完成后覆盖率突破 50%。',
  code: `// 【族谱】3 个文件 / 483 行 / 35 用例
//   constant_or_scalar_merger        182 行 / 13 用例   优化（合并常量）
//   propagate_to_func_results        189 行 / 20 用例   补全（函数结果分片）
//   remove_propagation_debug_info    112 行 /  2 用例   清理（删调试信息）
//
// 注意 remove_propagation_debug_info 只有 2 个用例但有 112 行
//   -> 一个用例里有很多 IR（4 个 mesh 声明 + 大量传播信息）

// 【跨课呼应】
//   constant_or_scalar_merger      <- L3-02（拆分，镜像操作）
//   propagate_to_func_results      <- L4-16（边下沉）、L3-10（搬分片）
//   remove_propagation_debug_info  <- L2-03 / L4-05 / L4-08（观测性选项）

// 【L4 层的三类"镜像对"】
//   L3-02 拆分常量      <->  L4-17 合并常量
//   L3-06 内联 call     <->  L4-12 outline
//   L3-07 提升网格      <->  L4-15 内联网格
//   -> 导入期为了"传播自由度"做的事，导出期都要还原

// 一句话总结：
//   导出流水线的最后一课
//   合并可以合并的、补全应该补全的、删掉不该留的`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '常量合并', n: 13, c: '#4ade80' },
      { t: '函数结果传播', n: 20, c: '#38bdf8' },
      { t: '调试清理', n: 2, c: '#fbbf24' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:190px;opacity:.4;transition:.35s;border-color:${f.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="card-t" style="color:${f.c};font-size:11.5px">${f.t}</div>
        <div class="big" style="font-size:20px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 130)); msg.innerHTML = '<b>35 个用例</b>分三个 pass —— 各解决一个具体问题。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>L4 的三类镜像对</b>：拆分↔合并、内联↔outline、提升↔内联。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>共同的道理</b>：导入期为了"<b>传播自由度</b>"做的事，导出期都要还原。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>L4 完成</b>：17 课全部结束，覆盖率突破 50%。';
    });
  }
},

/* ------------------------------------------------------------ 6 练习 */
{
  kicker: 'L4-17 · 练习',
  title: '练一练：<span class="hl-a">为什么现在可以合并</span>',
  sub: '三道题分别考：常量合并、函数结果补全、调试清理。',
  caption: '一句话总结：<b>合并可以合并的、补全应该补全的、删掉不该留的</b>。',
  code: `// 题 1：L3-02 拆开常量，本课又合并回去，为什么两次都对？

// 题 2：propagate_to_func_results 补全的是什么？为什么需要？

// 题 3：为什么 propagation_edges 可以删？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col compact-ex', style: 'gap:6px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: 'L3-02 <b>拆开</b>常量，本课又<b>合并</b>回去，为什么两次都对？',
        a: '因为<b>两个阶段的目标不同</b>：' +
           '<br><b>L3-02（导入期）拆开</b>：避免传播期产生<b>假依赖</b> —— 多个使用共享一个常量，传播器会以为它们"<b>必须同分片</b>"，从而限制传播自由度。' +
           '<br><b>L4-17（导出期）合并</b>：传播<b>已经结束</b>，分片已经定下来 —— 假依赖不再是问题，合并让 <b>IR 更小、编译更快</b>。' +
           '<br><span class="dim">这是 L4 三类镜像对之一：<b>L3-02 拆分 ↔ L4-17 合并</b>、L3-06 内联 ↔ L4-12 outline、L3-07 提升 ↔ L4-15 内联。共同的道理是：<b>导入期为了"传播自由度"做的事，导出期都要还原</b>。</span>'
      },
      {
        q: '<span class="mono">propagate_to_func_results</span> 补全的是什么？为什么需要？',
        a: '补全的是<b>函数结果签名上的分片</b>。' +
           '<br><b>情形</b>：<span class="mono">@foo</span> 内部的终止符值（如 <span class="mono">%0 = stablehlo.abs ... {sdy.sharding = [{"x"}]}</span>）<b>有</b>分片，但函数结果签名 <span class="mono">-&gt; tensor&lt;8xf32&gt;</span> <b>没有</b> → 把值的分片<b>传播到签名</b>上。' +
           '<br><b>为什么需要</b>：函数结果的分片必须<b>反映实际返回值的分片</b> —— 否则调用者看到的签名与真实情况不符。' +
           '<br><span class="dim">与 L4-16 的两步配合：先把<b>边</b>的分片<b>下沉</b>到值上，再把<b>值</b>的分片<b>传播</b>到签名上 —— 函数边界信息就完整了。</span>'
      },
      {
        q: '为什么 <span class="mono">propagation_edges</span> 可以删？',
        a: '因为它<b>不是语义信息</b> —— 只描述"分片是<b>怎么传播过来的</b>"（传播路径），不描述"<b>结果是什么</b>"。' +
           '<br>它只在<b>调试</b>时有用（如 dump propagation trace）；保留会让 IR 变大、干扰后续处理。' +
           '<br><span class="dim">同类设计：L2-03 <span class="mono">keep-sharding-rules</span>、L4-05 <span class="mono">mark-partial-result</span>、L4-08 <span class="mono">keep-redundant-reshards</span> —— 都是<b>观测性</b>信息，最终都要清掉。</span>'
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
