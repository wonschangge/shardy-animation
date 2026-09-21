/* ==========================================================================
   L6-04 · exec-matmul-fft-iota
   --------------------------------------------------------------------------
   覆盖：executable_convert_global_to_local/ 下 3 个文件
         (stablehlo_dot_general 57, stablehlo_fft 43, stablehlo_iota 35)
         + executable_partitioner_pipeline/stablehlo_dot_general_indivisible.mlir
   目标：用数值验证 L5-04（位置补偿）、L5-05（数据并行）、L5-09（整除性补齐）。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 四个文件 */
{
  kicker: 'L6-04 · 矩阵乘/FFT/iota 执行',
  title: '★ 四个文件，<span class="hl-a">验证三条规律</span>',
  sub: 'L5-04 的位置补偿、L5-05 的数据并行、L5-09 的整除性补齐 —— 一网打尽。',
  caption: '其中 <span class="mono">dot_general_indivisible</span> 是 <b>L5-09 那条规则最直接的证据</b>。',
  code: `// 【四个文件与对应课】
//   文件                          验证什么                        对应课
//   stablehlo_iota                【iota 的偏移补偿】             L5-04
//   stablehlo_fft                 FFT 的【作用维未分片】          L5-04
//   stablehlo_dot_general         收缩维分片 = 【数据并行】       L5-05
//   dot_general_indivisible       【不可整除时补齐】（3->4）      L5-09

// 【网格】sdy.mesh @mesh_2 = <["x"=2]>

// 【★ 本课的一个特殊价值】
//   dot_general_indivisible 【直接印证】了 L5-09 的推导
//     L5-09 我从"补到下一个能被轴整除的数"这条规则出发
//       给出验收点 tensor<7x3x8> 沿 z=3 分片 -> 补到 9
//     本课的测试文件用一个【真实例子】证实了它：
//       收缩维 3 沿 y=2 分片 -> 补到 4
//   注释直接写着：
//     // Contracting dimension is sharded and indivisible (padded 3->4).

// 【四课验证一条链】
//   L5-04  iota 偏移补偿 / FFT 作用维判据
//   L5-05  收缩维分片 = 数据并行
//   L5-09  不可整除时补齐
//     ↓
//   L6-04  用真实数值全部验证 ✓

// 一句话：
//   L6-04 用数值验证了 L5-04 / L5-05 / L5-09 三条规律`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'stablehlo_iota', c: '#38bdf8', k: 'L5-04', d: '<b>偏移补偿</b><br>无补偿会得 <span class="mono">[0,0]</span>' },
      { t: 'stablehlo_fft', c: '#4ade80', k: 'L5-04', d: '<b>作用维未分片</b><br>每台独立算 4 点 FFT' },
      { t: 'stablehlo_dot_general', c: '#fbbf24', k: 'L5-05', d: '<b>数据并行</b><br>收缩维切 + all_reduce' },
      { t: 'dot_general_indivisible', c: '#fb7185', k: 'L5-09', d: '<b>补齐 3→4</b><br>收缩维 3 沿 y=2 除不尽' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:9px;color:${x.c};overflow-wrap:anywhere">${x.t}</div>
        <div class="small" style="font-size:9px;color:${x.c};margin:2px 0">${x.k}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>iota 的问题</b>：每台本地 iota 都从 0 开始 —— 设备 1 该拿 <span class="mono">[1]</span> 却拿到 <span class="mono">[0]</span>。',
        '<b>FFT 的问题</b>：分片在第 0 维、FFT 作用在第 1 维 —— <b>不冲突</b>。',
        '<b>矩阵乘</b>：两个收缩维都切了 → 产生部分和 → <span class="mono">all_reduce</span>。',
        '<b>不可整除</b>：收缩维 3 沿 <span class="mono">y=2</span> 除不尽 → <b>补到 4</b>。',
      ][i];
    }));
    tl.at(15200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 特殊价值</b>：第四个文件的注释直接写着 <span class="mono">padded 3-&gt;4</span> —— 印证了 L5-09 的推导。';
    });
  }
},

/* ------------------------------------------------ 2 ★ iota */
{
  kicker: 'L6-04 · 矩阵乘/FFT/iota 执行',
  title: '★ <span class="mono hl-a">iota</span>：验证 L5-04 的偏移补偿',
  sub: '如果没做偏移补偿，两台设备都会输出 `[0]` → 拼接得 `[0,0]` ≠ `[0,1]`。',
  caption: '这个测试<b>恰好能抓住那个 bug</b> —— 因为 <span class="mono">[0,0]</span> 与 <span class="mono">[0,1]</span> 不同。',
  code: `sdy.mesh @mesh_2 = <["x"=2]>

// Performs the same iota as in sequential_iota, but on 2 devices in parallel.
//
// We will use sdy-opt to convert this to a device local program that
// stablehlo interpreter can execute.
//
func.func @parallel_iota()
  -> (tensor<2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) {
  %0 = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}]>]>} : tensor<2xi32>
  return %0 : tensor<2xi32>
}

// 【读法】
//   @parallel_iota 【无参数】—— 因为 iota 不需要输入，只生成序号
//   全局 tensor<2xi32> 沿 {"x"}（x=2）切 -> 【每台设备 1 个元素】
//   全局 iota 是 [0, 1]；设备 0 该拿 [0]、设备 1 该拿 [1]

// 【验证方式】
// Main Orchestrator: executes the sequential and parallel iota and checks that
// they are equivalent.
func.func @main() {
  %seq = func.call @sequential_iota() : () -> tensor<2xi32>

  %pars:2 = "interpreter.run_parallel"() {
    programs = [[@parallel_iota, @parallel_iota]]
  } : () -> (tensor<1xi32>, tensor<1xi32>)
  %par = "stablehlo.concatenate"(%pars#0, %pars#1) {
    dimension = 0 : i64
  } : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>

  "check.expect_eq"(%seq, %par) : (tensor<2xi32>, tensor<2xi32>) -> ()

  return
}
// 【读法】
//   run_parallel() 【无输入】（iota 不需要输入）✓
//   每台设备各得 tensor<1xi32>
//   concatenate 拼回 tensor<2xi32> -> 与 %seq 比较

// 【★ 这验证了 L5-04 讲的 iota 偏移补偿】
//   问题：每台设备本地 iota 只会生成 [0] —— 但设备 1 该拿 [1]！
//   降级后的解法（L5-04 讲过）：
//     插入 partition_id + 查找表 [0, 1] + 加法
//   本课证明：补偿后 concatenate 的结果等于串行版 [0, 1] ✓

// 【★ 如果没做偏移补偿】
//   两台设备都会输出 [0] -> 拼接得 [0, 0] ≠ [0, 1]
//   这个测试【恰好能抓住那个 bug】
//   因为 [0,0] 与 [0,1] 明显不同`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '有偏移补偿（正确）', c: '#4ade80', d: '设备 0 → <span class="mono">[0]</span><br>设备 1 → <span class="mono">[1]</span><br>拼接 = <b><span class="mono">[0, 1]</span></b> ✓' },
      { t: '无偏移补偿（错误）', c: '#fb7185', d: '设备 0 → <span class="mono">[0]</span><br>设备 1 → <span class="mono">[0]</span><br>拼接 = <b><span class="mono">[0, 0]</span></b> ✗' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:12.5px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>正确的样子</b>：偏移补偿后每台拿到自己的序号 → 拼接得 <span class="mono">[0, 1]</span>。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>漏掉补偿的样子</b>：两台都从 0 开始 → 拼接得 <span class="mono">[0, 0]</span> —— 与串行版不符。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 这个测试恰好能抓住那个 bug</b>：<span class="mono">[0,0]</span> 与 <span class="mono">[0,1]</span> 明显不同。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>回顾 L5-04</b>：那里从 IR 形态推断"需要补偿偏移"，这里证明"补偿后结果正确"。';
    });
  }
},

/* ------------------------------------------------ 3 ★ fft */
{
  kicker: 'L6-04 · 矩阵乘/FFT/iota 执行',
  title: '★ <span class="mono hl-a">fft</span>：作用维<span class="hl-a">未</span>分片',
  sub: '分片在第 0 维（批维）、FFT 作用在第 1 维 —— **不冲突**。',
  caption: '注释直接给出了结论：<span class="mono">each device performs a 4-point FFT</span>。',
  code: `// This function computes the FFT of a 2x4 complex tensor.
// It is sharded along dimension 0 (batch), so each device performs a 4-point
// FFT.
func.func @parallel_fft(
  %arg0: tensor<2x4xcomplex<f64>> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
  -> (tensor<2x4xcomplex<f64>> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = "stablehlo.fft"(%arg0) {
    fft_length = array<i64: 4>,
    fft_type = #stablehlo<fft_type FFT>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>
  } : (tensor<2x4xcomplex<f64>>) -> tensor<2x4xcomplex<f64>>
  return %0 : tensor<2x4xcomplex<f64>>
}

// 【读法】注释直接给出结论
//   It is sharded along dimension 0 (batch), so each device performs a
//   4-point FFT.
//   分片在【第 0 维（批维）】；FFT 作用在【第 1 维】（fft_length = 4）
//   -> 作用维【未被分片】
//   -> 每台设备【独立做 4 点 FFT】
//   -> 【无通信】 ✓

// 【★ 这正是 L5-04 讲的"作用维判据"】
//   算子的"作用维"是不是被分片的那一维？
//     FFT 的作用维 = fft_length 对应的维（第 1 维）
//     分片在第 0 维 -> 【不冲突】-> 直接局部化

// 【注意类型是 complex<f64>】—— 本课特有的复数类型

// 【验证方式】
  // Input tensor with 8 unique complex values.
  %input = stablehlo.constant dense<[
    [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
    [(5.0, 0.0), (7.0, 0.0), (11.0, 0.0), (13.0, 0.0)]
  ]> : tensor<2x4xcomplex<f64>>

  %seq = func.call @sequential_fft(%input) : (tensor<2x4xcomplex<f64>>) -> tensor<2x4xcomplex<f64>>

  %s0 = "stablehlo.slice"(%input) {start_indices=array<i64: 0, 0>, limit_indices=array<i64: 1, 4>, strides=array<i64: 1, 1>} : (tensor<2x4xcomplex<f64>>) -> tensor<1x4xcomplex<f64>>
  %s1 = "stablehlo.slice"(%input) {start_indices=array<i64: 1, 0>, limit_indices=array<i64: 2, 4>, strides=array<i64: 1, 1>} : (tensor<2x4xcomplex<f64>>) -> tensor<1x4xcomplex<f64>>
  %pars:2 = "interpreter.run_parallel"(%s0, %s1) {
    programs = [[@parallel_fft, @parallel_fft]]
  } : (tensor<1x4xcomplex<f64>>, tensor<1x4xcomplex<f64>>) -> (tensor<1x4xcomplex<f64>>, tensor<1x4xcomplex<f64>>)
  %par = "stablehlo.concatenate"(%pars#0, %pars#1) {dimension = 0 : i64}
    : (tensor<1x4xcomplex<f64>>, tensor<1x4xcomplex<f64>>) -> tensor<2x4xcomplex<f64>>

  "check.expect_eq"(%seq, %par) : (tensor<2x4xcomplex<f64>>, tensor<2x4xcomplex<f64>>) -> ()
// 【读法】
//   输入是 8 个【各不相同】的复数值（1/2/3/4 与 5/7/11/13）
//     都是质数 —— 避免巧合相等（与 L6-03 的"缩放"技巧同源）
//   【手动 slice】成 %s0/%s1（按批维）—— L6-03 讲的技巧 ②
//   各自 FFT -> concatenate 拼回 -> 与串行版比较 ✓`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '分片维', c: '#38bdf8', d: '第 <b>0</b> 维（批维）<br><span class="mono">[{"x"}, {}]</span><br><span class="mono">2 / 2 = 1</span>' },
      { t: 'FFT 作用维', c: '#4ade80', d: '第 <b>1</b> 维<br><span class="mono">fft_length = 4</span><br><b>未被分片</b>' },
      { t: '→ 结论', c: '#fbbf24', d: '每台独立做<br><b>4 点 FFT</b><br><b>无通信</b> ✓' },
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
        '<b>分片在批维</b> —— 每台设备拿一批（<span class="mono">2/2 = 1</span>）。',
        '<b>FFT 作用在第 1 维</b>（长度 4）—— 而这一维<b>没被切</b>。',
        '<b>两者不冲突</b> → 每台独立算完整的 4 点 FFT → <b>无通信</b>。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 这就是 L5-04 的"作用维判据"</b>：作用维未分片 → 直接局部化。';
    });
  }
},

/* ------------------------------------------------ 4 ★ dot_general */
{
  kicker: 'L6-04 · 矩阵乘/FFT/iota 执行',
  title: '★ <span class="mono hl-a">dot_general</span>：收缩维分片 = 数据并行',
  sub: '两个收缩维都被切 → 产生部分和 → `all_reduce` → 结果正确。',
  caption: '注释直接标注了每台设备拿什么：<span class="mono">// Device 0: lhs[:, 0:2], rhs[0:2, :]</span>',
  code: `func.func @parallel_dot(
  %arg0: tensor<2x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {"x"}]>},
  %arg1: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}
) -> (tensor<2x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>}) {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2, [{}, {}], unreduced={"x"}>]>
  } : (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x2xi32>

  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2, [{}, {}]> : tensor<2x2xi32>
  return %1 : tensor<2x2xi32>
}

// 【读法】
//   lhs_contracting_dimensions = [1]、rhs_contracting_dimensions = [0]
//     -> 收缩维是 %arg0 的第 1 维、%arg1 的第 0 维
//   分片：%arg0 的第 1 维（收缩维）切 {"x"}
//         %arg1 的第 0 维（收缩维）切 {"x"}
//     -> 两个收缩维【都被切】-> 产生【部分和】
//     -> unreduced={"x"} + all_reduce ✓
// ★ 这正是 L5-05 讲的"收缩维分片 = 【数据并行】"

// 【验证方式】注释直接标注了分片方案
  %lhs = stablehlo.constant dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi32>
  %rhs = stablehlo.constant dense<[[1, 16], [2, 32], [4, 64], [8, 128]]> : tensor<4x2xi32>

  %seq = func.call @sequential_dot(%lhs, %rhs) : (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x2xi32>

  // Device 0: lhs[:, 0:2], rhs[0:2, :]
  %lhs0 = "stablehlo.slice"(%lhs) {start_indices=array<i64: 0, 0>, limit_indices=array<i64: 2, 2>, strides=array<i64: 1, 1>} : (tensor<2x4xi32>) -> tensor<2x2xi32>
  %rhs0 = "stablehlo.slice"(%rhs) {start_indices=array<i64: 0, 0>, limit_indices=array<i64: 2, 2>, strides=array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>
  // Device 1: lhs[:, 2:4], rhs[2:4, :]
  %lhs1 = "stablehlo.slice"(%lhs) {start_indices=array<i64: 0, 2>, limit_indices=array<i64: 2, 4>, strides=array<i64: 1, 1>} : (tensor<2x4xi32>) -> tensor<2x2xi32>
  %rhs1 = "stablehlo.slice"(%rhs) {start_indices=array<i64: 2, 0>, limit_indices=array<i64: 4, 2>, strides=array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>

  %pars:2 = "interpreter.run_parallel"(%lhs0, %rhs0, %lhs1, %rhs1) {
    programs = [[@parallel_dot, @parallel_dot]]
  } : (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) -> (tensor<2x2xi32>, tensor<2x2xi32>)

  "check.expect_eq"(%pars#0, %seq) : (tensor<2x2xi32>, tensor<2x2xi32>) -> ()
  "check.expect_eq"(%pars#1, %seq) : (tensor<2x2xi32>, tensor<2x2xi32>) -> ()

// 【读法】注释直接标注分片方案
//   Device 0 拿 lhs 的第 0~1 列 + rhs 的第 0~1 行（收缩维的前半）
//   Device 1 拿 lhs 的第 2~3 列 + rhs 的第 2~3 行（收缩维的后半）
//   各算一个【部分和】-> all_reduce 合并 -> 【两台都得到完整结果】
//   断言【两个设备】都等于 %seq ✓
//
// 【注意 %rhs 的取值】（1/16, 2/32, 4/64, 8/128）
//   每行是【倍数关系】—— 避免不同分片方式的乘积巧合相等`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '设备 0', c: '#38bdf8', d: '<span class="mono">lhs[:, 0:2]</span><br><span class="mono">rhs[0:2, :]</span><br>收缩维<b>前半</b>' },
      { t: '设备 1', c: '#4ade80', d: '<span class="mono">lhs[:, 2:4]</span><br><span class="mono">rhs[2:4, :]</span><br>收缩维<b>后半</b>' },
      { t: 'all_reduce', c: '#fbbf24', d: '两个<b>部分和</b><br>合并<br>→ 两台都完整' },
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
        '<b>设备 0</b> 拿收缩维的前半 —— 算一个<b>部分和</b>。',
        '<b>设备 1</b> 拿收缩维的后半 —— 算另一个<b>部分和</b>。',
        '<b>all_reduce 合并</b> → 两台设备都得到<b>完整</b>的矩阵乘结果 ✓',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 这就是 L5-05 的"数据并行"</b>：收缩维切分 → 各算部分和 → all_reduce。';
    });
  }
},

/* ------------------------------------------------ 5 ★ 不可整除 */
{
  kicker: 'L6-04 · 矩阵乘/FFT/iota 执行',
  title: '★★ <span class="mono hl-a">dot_general_indivisible</span>：验证 L5-09 的 3→4 补齐',
  sub: '收缩维大小是 **3**，沿 `y=2` 分片 → **除不尽** → **补到 4**。',
  caption: '这是 <b>L5-09 那条规则最直接的证据</b> —— 注释直接写着 <span class="mono">padded 3-&gt;4</span>。',
  code: `// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

// Contracting dimension is sharded and indivisible (padded 3->4).
func.func @parallel_dot_contracting_indivisible(
  %arg0: tensor<3x3xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
  %arg1: tensor<3x5xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>})
  -> (tensor<3x5xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>}) {
  %2 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_x2_y2, [{"x"}, {}], unreduced={"y"}>]>} : (tensor<3x3xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>

// 【读法】注释直接给出了答案
//   // Contracting dimension is sharded and indivisible (padded 3->4).
//   收缩维大小是【3】，沿 y=2 分片 -> 【3 不能被 2 整除】！
//   -> 【补齐 3 -> 4】 ✓
//   网格是 @mesh_x2_y2（2x2），结果标 unreduced={"y"}

// 【★★ 这正是 L5-09 验收点的答案】
//   L5-09 我讲过："补到【下一个能被轴整除的数】"
//     并给出验收点 tensor<7x3x8> 沿 z=3 分片 -> 补到 9
//   本课的测试文件【直接证实了这个规则】——
//     收缩维 3 沿 y=2 分片 -> 补到 4（下一个 2 的倍数）

// 【两个 RUN 行】
//   验证 enable-halo-exchange 的 true/false 【结果一致】
//   （与 L6-03 的 dual_semantics 同源，对应 L4-09）

// 【验证方式】
func.func @sequential_dot_contracting_indivisible(%arg0: tensor<3x3xf32>, %arg1: tensor<3x5xf32>) -> tensor<3x5xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<3x3xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>

  "check.expect_eq"(%res#0, %seq) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
// 【读法】
//   @sequential_dot_contracting_indivisible 是【手写】的串行版（无分片）
//   【4 台设备】（2x2 网格）的结果【都等于】%seq
//   注意结果类型仍是 tensor<3x5xf32> —— 【补齐是内部的】
//     最终结果【裁回了原始大小】（L5-09 讲的"pad 补齐 + slice 裁回"）✓

// 【★ 这完整验证了 L5-09 的流程】
//   收缩维 3 沿 y=2 分片（除不尽）
//     -> pad 补齐到 4
//     -> 通信 / 计算在可整除的形状上
//     -> slice 裁回 3
//     -> 结果与串行版一致 ✓`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const steps = [
      { n: '①', t: '收缩维 3', c: '#fb7185', d: '沿 <span class="mono">y=2</span> 分片<br><b>除不尽</b>' },
      { n: '②', t: 'pad 补齐', c: '#fbbf24', d: '<span class="mono">3</span> → <b><span class="mono">4</span></b><br>下一个 2 的倍数' },
      { n: '③', t: '通信/计算', c: '#38bdf8', d: '在<b>可整除</b>的<br>形状上做' },
      { n: '④', t: 'slice 裁回', c: '#4ade80', d: '裁回 <span class="mono">3</span><br>结果与串行一致 ✓' },
    ];
    const host = wrap.querySelector('#cards');
    const els = steps.map(s => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${s.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="small mono" style="font-size:10.5px;color:${s.c}">${s.n}</div>
        <div class="mono" style="font-size:10.5px;margin-top:3px">${s.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35;margin-top:3px">${s.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    steps.forEach((s, i) => tl.at(700 + i * 3800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>问题的来源</b>：收缩维大小 3，而 <span class="mono">y=2</span> 要求整除。',
        '<b>L5-09 的规则</b>：补到<b>下一个能被轴整除的数</b> —— 3 补到 <b>4</b>。',
        '补齐后通信与计算都在<b>可整除</b>的形状上进行。',
        '最后 <span class="mono">slice</span> 裁回 3 —— <b>结果类型仍是 <span class="mono">3x5</span></b>，与串行版一致 ✓',
      ][i];
    }));
    tl.at(15800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 这是 L5-09 那条规则最直接的证据</b>：注释写着 <span class="mono">padded 3-&gt;4</span>，与推导完全一致。';
    });
  }
},

/* ------------------------------------------------ 6 小结 */
{
  kicker: 'L6-04 · 矩阵乘/FFT/iota 执行',
  title: '小结：<span class="hl-a">四课验证一条链</span>',
  sub: 'L5-04 / L5-05 / L5-09 三条规律，在本课全部得到数值验证。',
  caption: '一句话：<b>本课是 L5 后半段规律的"实测报告"</b>。',
  code: `// 【族谱】4 个文件
//   文件                        验证什么                      对应课
//   stablehlo_iota              iota 的偏移补偿               L5-04
//   stablehlo_fft               FFT 作用维未分片              L5-04
//   stablehlo_dot_general       收缩维分片 = 数据并行         L5-05
//   dot_general_indivisible     不可整除补齐 3->4             L5-09

// 【★ 四课验证一条链】
//   L5-04  iota 偏移补偿 / FFT 作用维判据
//   L5-05  收缩维分片 = 数据并行
//   L5-09  不可整除时补齐
//     ↓
//   L6-04  用真实数值全部验证 ✓

// 【★ 三个可迁移的观察】
//   ① iota 的测试【恰好能抓住漏掉偏移补偿的 bug】
//      无补偿 -> 拼接得 [0,0] ≠ [0,1] -> 测试失败
//      -> 好的测试要能【区分正确与错误实现】
//   ② FFT 的注释【直接给出了结论】
//      "each device performs a 4-point FFT"
//      -> 上游测试的注释【常常直接点明意图】，值得先读
//   ③ dot_general 的注释【直接标注分片方案】
//      "// Device 0: lhs[:, 0:2], rhs[0:2, :]"
//      -> 读这类测试时，注释是最快的入口

// 【L6 的进度】
//   L6-00 可执行测试机制（已做）    L6-01 sdy.* 集合通信（已做）
//   L6-02 stablehlo 集合通信（已做）L6-03 卷积（已做）
//   L6-04 矩阵乘/fft/iota（本课）
//   L6-05 gather（★最复杂）        L6-06 pad（★最大族）
//   L6-07 reshape                   L6-08 reverse/slice
//   L6-09 scatter 与杂项

// 一句话总结：
//   L6-04 用数值验证了 L5-04（位置补偿）、L5-05（数据并行）、
//   L5-09（整除性补齐）三条规律
//   其中 dot_general_indivisible 的 3->4 补齐是 L5-09 最直接的证据`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: 'L5-04 iota', n: 1, c: '#38bdf8' }, { t: 'L5-04 fft', n: 1, c: '#0ea5e9' },
      { t: 'L5-05 dot', n: 1, c: '#fbbf24' }, { t: 'L5-09 补齐', n: 1, c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:170px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div style="font-size:10px;color:${f.c};line-height:1.3">${f.t}</div>
        <div class="big" style="font-size:16px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 120)); msg.innerHTML = '<b>4 个文件</b>覆盖三条规律（L5-04 占两个）。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>★ 三个可迁移观察</b>：好的测试能区分对错 / 注释常点明意图 / 注释标注分片方案。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>最有价值的证据</b>：<span class="mono">padded 3-&gt;4</span> 与 L5-09 推导的"补到下一个倍数"完全一致。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L6-05 讲 gather 的执行（★ 最复杂）。';
    });
  }
},

];
