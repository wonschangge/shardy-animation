/* ==========================================================================
   L6-08 · exec-reverse-slice
   --------------------------------------------------------------------------
   覆盖：9 个文件
         executable_convert_global_to_local/stablehlo_slice.mlir
         executable_partitioner_pipeline/ 下 6 个 slice + 2 个 reverse
   目标：讲透"切片/反转是否改变设备之间的数据归属"这条判据。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 核心判据 */
{
  kicker: 'L6-08 · slice/reverse 的执行',
  title: '★ 核心判据：<span class="hl-a">是否改变设备之间的数据归属</span>',
  sub: '`slice` 看**范围是否跨界**；`reverse` 看**是否反转了被分片的维**。',
  caption: '两个文件的命名直接点明了区分：<span class="mono">comm_free</span> vs <span class="mono">with_communication</span>。',
  code: `// 【9 个文件】
//   族              文件数   位置
//   slice           7        1 个在 convert_global_to_local/
//                            6 个在 partitioner_pipeline/
//   reverse         2        partitioner_pipeline/

// 【★ 核心判据】
//   slice   看【切片的范围是否跨越设备边界】
//   reverse 看【是否反转了被分片的维】

// 【两个文件的命名直接点明区分】
//   slice_comm_free             切片【不】跨界 -> 无通信
//   slice_with_communication    切片【跨】设备 -> 需通信

// 【★ 与 L5-04 的"作用维判据"的关系】
//   L5-04 讲"作用维【未分片】-> 参数不变"
//   本课更细一层：
//     即使作用维【就是分片维】，只要切片范围【覆盖完整】，也不跨设备

// 【9 个文件的族谱】
//   基准             convert/stablehlo_slice          切片维未分片
//   不跨界           slice_comm_free                  范围完整 -> 无通信
//   跨界             slice_with_communication         跨设备 -> 需通信
//   不可整除         slice_indivisible                3 不能被 4 整除
//   全复制           slice_replicated                 结果 [{}] + 网格有未用轴
//   全复制（单轴）   slice_replicated_mesh_2          同上，网格单轴
//   步长             slice_strided                    [1:7:2] 局部形状不同
//   reverse 多维     reverse_multi_dim_divisible      多维反转 + 可整除
//   reverse 单维     reverse_single_dim_indivisible   单维反转 + 不可整除 + 轴序不同

// 一句话：
//   slice / reverse 的降级取决于"是否改变设备之间的数据归属"`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'comm_free', c: '#4ade80', d: '切片<b>不跨界</b><br><b>无通信</b>' },
      { t: 'with_communication', c: '#fb7185', d: '切片<b>跨设备</b><br><b>需通信</b>' },
      { t: 'indivisible', c: '#fbbf24', d: '<span class="mono">3</span> 不能被 <span class="mono">4</span> 整除' },
      { t: 'replicated ×2', c: '#38bdf8', d: '结果 <span class="mono">[{}]</span><br>全复制' },
      { t: 'strided', c: '#c084fc', d: '<b>步长</b><br>局部形状不同' },
      { t: 'reverse ×2', c: '#f472b6', d: '<b>分片顺序</b><br>也要反转' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:118px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:9px;color:${x.c};overflow-wrap:anywhere">${x.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>六类情形</b> —— 前两类是核心判据的对照。'; });
    tl.at(4400, () => {
      els.forEach((e, i) => { if (i > 1) e.style.opacity = '.25'; });
      msg.innerHTML = '<b>★ 核心对照</b>：<span class="mono">comm_free</span>（不跨界）vs <span class="mono">with_communication</span>（跨界）。';
    });
    tl.at(8000, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>比 L5-04 更细一层</b>：作用维即使就是分片维，只要切片范围<b>覆盖完整</b>，也不跨设备。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>reverse 的难点</b>：反转被分片的维时，<b>分片的顺序也要反转</b>。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 核心对照 */
{
  kicker: 'L6-08 · slice/reverse 的执行',
  title: '★ <span class="mono hl-a">comm_free</span> vs <span class="mono">with_communication</span>',
  sub: '两个文件只差**切片范围** —— 一个不跨界、一个跨了两台设备。',
  caption: '这是本课最重要的一组对照。',
  code: `// 【comm_free：切片【不】跨界】
sdy.mesh @mesh = <["x"=2]>
func.func @parallel_slice_comm_free(
  %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
  -> (tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:4, 0:2]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"x"}, {}]>]>} : (tensor<4x4xi32>) -> tensor<4x2xi32>
// 【读法】
//   输入 4x4 切 [{"x"}, {}] —— 第 0 维分片（4/2 = 2 行/台）
//   slice [0:4, 0:2] —— 第 0 维是【0:4（完整）】、第 1 维切到 0:2
//   第 0 维（分片维）的切片范围是【完整的】
//     -> 切片本身【不跨设备】-> 【comm_free】✓

// 【with_communication：切片【跨】设备边界】
sdy.mesh @mesh = <["x"=2]>
func.func @parallel_slice_with_communication(
  %arg0: tensor<8x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
  -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [1:5, 0:4]
  %1 = sdy.reshard %0 <@mesh, [{}, {}]> : tensor<4x4xi32>
// 【读法】与 comm_free 的【唯一区别】就是切片范围
//   输入 8x4 切 {"x"} —— 设备 0 有第 【0~3】 行、设备 1 有第 【4~7】 行
//   slice [1:5, 0:4] —— 第 0 维取第 【1~4】 行
//     第 1、2、3 行在【设备 0】
//     第 4 行在【设备 1】
//     -> 切片范围【跨越了两台设备】！
//   -> 切片本身就需要通信 -> 显式的 sdy.reshard ✓

// 【★ 这就是两个文件的区别】
//   切片的范围【是否跨越设备边界】
//
// 【逐设备推演 comm_free】
//   设备 0 有第 0~1 行、设备 1 有第 2~3 行
//   slice [0:4, 0:2] 的第 0 维是完整的 -> 每台都能在自己那段内完成切片
//
// 【逐设备推演 with_communication】
//   设备 0 有第 0~3 行、设备 1 有第 4~7 行
//   slice [1:5] 要第 1~4 行 -> 设备 0 能给自己 1,2,3，但第 4 行【要向设备 1 要】
//   -> 必须通信`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'comm_free', c: '#4ade80', d: '切片 <span class="mono">[0:4, 0:2]</span><br>第 0 维<b>完整</b><br>→ 每台在自己段内完成 ✓' },
      { t: 'with_communication', c: '#fb7185', d: '切片 <span class="mono">[1:5, 0:4]</span><br>第 0 维取 <b>1~4</b><br>→ <b>跨了两台</b> ✗' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:11px;overflow-wrap:anywhere">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>不跨界</b>：切片的第 0 维是完整的 → 每台设备都能<b>在自己那段内</b>完成。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>跨界</b>：<span class="mono">[1:5]</span> 要第 1~4 行 —— 第 4 行在<b>设备 1</b> 手上 → 必须通信。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 唯一区别就是切片范围</b> —— 两个文件的输入形状与分片都不同，但判据相同。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>★ 比 L5-04 更细一层</b>：作用维<b>就是分片维</b>时，只要范围覆盖完整，仍不跨设备。';
    });
  }
},

/* ------------------------------------------------ 3 三个额外复杂度 */
{
  kicker: 'L6-08 · slice/reverse 的执行',
  title: '三个额外复杂度：<span class="hl-a">不可整除 / 步长 / 全复制</span>',
  sub: '这三个文件在核心判据之外，各加了一层复杂度。',
  caption: '其中 <b>步长</b> 让"每台设备拿多少"不再是简单除法。',
  code: `// 【不可整除】slice_indivisible
sdy.mesh @mesh_x4 = <["x"=4]>
  %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{"x"}, {}]>})
  -> (tensor<3x2xf32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:3, 0:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x4, [{"x"}, {}]>]>} : (tensor<4x2xf32>) -> tensor<3x2xf32>
  %0 = stablehlo.slice %arg0 [0:3, 0:2] : (tensor<4x2xf32>) -> tensor<3x2xf32>
// 【读法】
//   网格 x=4（【4 台设备】），输入 4x2 切 {"x"} -> 【每台 1 行】
//   slice [0:3, 0:2] —— 第 0 维取前 【3】 行
//   【3 不能被 4 整除】！-> 不可整除 ✓ -> 需要补齐（L5-09 的规则）
// 【注意第二行】是【串行版】的写法（无 sdy.sharding 属性）

// 【步长】slice_strided
sdy.mesh @mesh = <["x"=2]>
  %arg0: tensor<8x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
  -> (tensor<3x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [1:7:2, 0:4]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"x"}, {}]>]>} : (tensor<8x4xi32>) -> tensor<3x4xi32>
// 【读法】
//   slice [1:7:2, 0:4] —— 第 0 维：【起始 1、结束 7、步长 2】
//   取第 【1, 3, 5】 行（3 个元素）✓（结果 3x4）
//   【步长让切片更复杂】：每台设备要算"自己那段里哪些元素被选中"
//     设备 0 有第 0~3 行 -> 选中第 1、3 行
//     设备 1 有第 4~7 行 -> 选中第 5 行
//   -> 结果的【局部形状每台不同】（设备 0 得 2 行、设备 1 得 1 行）
// ★ 这是 slice 最复杂的情形 —— 步长让"每台拿多少"不再是简单除法

// 【全复制】slice_replicated / slice_replicated_mesh_2
sdy.mesh @mesh = <["a"=2, "b"=2]>
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}]>})
  -> (tensor<3xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
  %0 = stablehlo.slice %arg0 [0:3]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"b"}]>]>} : (tensor<8xi32>) -> tensor<3xi32>
// 【读法】两个文件几乎相同，只差网格
//   slice_replicated：网格 a=2, b=2（4 台），【只用 b 轴】分片
//     -> a 轴是【多余的】（结果 [{}] 无分片）
//   slice_replicated_mesh_2：网格 x=2（2 台），单轴
//   共同点：输入 8 切 {"b"}/{"x"}（8/2 = 4 个/台）
//     slice [0:3] 取前 3 个，结果声明 [{}]（【全复制】）
// ★ 这两个文件验证"结果全复制"的情形
//   输入有分片、结果无分片 -> 需要 reshard 到全复制
//   而 slice_replicated 的网格有【两个轴但只用一个】
//     -> 验证【未使用的轴】不影响`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '不可整除', c: '#fbbf24', d: '网格 <span class="mono">x=4</span><br>取 <b>3</b> 行<br><span class="mono">3 % 4 ≠ 0</span>' },
      { t: '步长', c: '#c084fc', d: '<span class="mono">[1:7:2]</span><br>取第 <b>1,3,5</b> 行<br>局部形状<b>每台不同</b>' },
      { t: '全复制', c: '#38bdf8', d: '结果 <span class="mono">[{}]</span><br>需 reshard<br><span class="dim">网格可有未用轴</span>' },
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
        '<b>不可整除</b>：<span class="mono">3</span> 行要分给 <span class="mono">4</span> 台设备 → 除不尽 → 补齐（L5-09）。',
        '<b>步长</b>：<span class="mono">[1:7:2]</span> 取第 1,3,5 行 —— <b>设备 0 得 2 行、设备 1 得 1 行</b>，局部形状不同。',
        '<b>全复制</b>：结果声明 <span class="mono">[{}]</span> → 需 reshard；网格可有<b>未使用的轴</b>（<span class="mono">a</span>）。',
      ][i];
    }));
    tl.at(13400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>三者叠加</b>在核心判据之上 —— 让 slice 的降级变得更复杂。';
    });
  }
},

/* ------------------------------------------------ 4 ★ reverse */
{
  kicker: 'L6-08 · slice/reverse 的执行',
  title: '★ <span class="mono hl-a">reverse</span>：分片顺序也要<span class="hl-a">反转</span>',
  sub: '反转一个**被分片**的维时，设备之间的数据归属会改变。',
  caption: '第二个文件还展示了<b>轴顺序与维度顺序不同</b>的情形。',
  code: `// 【reverse_multi_dim_divisible：多维 reverse】
sdy.mesh @mesh = <["a"=2, "b"=2]>
  %arg0: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
  -> (tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:4, 0:8]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>}
// 【读法】
//   网格 a=2, b=2，输入 4x8 切 [{"a"}, {"b"}]
//   slice [0:4, 0:8] —— 【完整范围】（不切）—— 这一步只是为了触发后续处理
//   用例名 multi_dim_divisible —— 【多维 reverse + 可整除】

// 【reverse_single_dim_indivisible：单维 reverse + 不可整除】
sdy.mesh @mesh_abc = <["a"=2, "b"=2, "c"=4]>
  %arg0: tensor<4x6x8xi32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {"c"}]>})
  %0 = stablehlo.slice %arg0 [0:4, 0:6, 0:5]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]>}
  %1 = stablehlo.reverse %0, dims = [0, 2]
// 【读法】
//   网格 a=2, b=2, c=4（【16 台设备】）
//   输入 4x6x8，分片是 【[{"b"}, {"a"}, {"c"}]】
//     —— 【注意轴顺序与维度顺序不同】：
//        第 0 维切 b、第 1 维切 a、第 2 维切 c
//   slice [0:4, 0:6, 0:5] —— 第 2 维切到 5（8 -> 5）
//   reverse %0, dims = [0, 2] —— 【反转第 0 维和第 2 维】！
//   indivisible：5 不能被 c=4 整除 -> 不可整除 ✓

// 【★ reverse 的关键问题】
//   反转一个【被分片】的维时，【分片的顺序也要反转】！
//     设备 0 原本拿第 0 段，反转后应该拿【最后一段】
//     -> 需要通信
//
// 【★ 为什么 reverse 比 slice 更"彻底"】
//   slice 只是"取一段"—— 段内的顺序不变
//   reverse 是"整体翻转"—— 【每台设备该拿的数据都变了】
//     -> 通常需要通信

// 【多维 vs 单维】
//   multi_dim_divisible      多维反转 + 可整除 -> 较简单
//   single_dim_indivisible   单维反转 + 不可整除 -> 两个复杂度叠加`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:14px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'slice', c: '#38bdf8', d: '"取一段"<br>段内顺序<b>不变</b><br>可能不需通信' },
      { t: 'reverse', c: '#fb7185', d: '"整体翻转"<br><b>每台该拿的数据都变了</b><br>通常需要通信' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:12px">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>slice</b> 只是"取一段" —— 段内顺序不变，所以可能不需要通信。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>reverse</b> 是"整体翻转" —— <b>设备 0 原本拿第 0 段，反转后该拿最后一段</b>。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 所以 reverse 通常需要通信</b> —— 因为它<b>彻底改变</b>了设备之间的数据归属。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>第二个文件还有个细节</b>：分片 <span class="mono">[{"b"},{"a"},{"c"}]</span> —— <b>轴顺序与维度顺序不同</b>。';
    });
  }
},

/* ------------------------------------------------ 5 小结 */
{
  kicker: 'L6-08 · slice/reverse 的执行',
  title: '小结：<span class="hl-a">九课的一条判据</span>',
  sub: '`slice` / `reverse` 的降级取决于**是否改变设备之间的数据归属**。',
  caption: '一句话：<b>不改变归属就不通信，改变了就要通信</b>。',
  code: `// 【族谱】9 个文件
//   族              文件                                  验证什么
//   基准            convert/stablehlo_slice               切片维未分片
//   不跨界          slice_comm_free                       范围完整 -> 无通信
//   跨界            slice_with_communication              跨设备 -> 需通信
//   不可整除        slice_indivisible                     3 不能被 4 整除
//   全复制          slice_replicated                      结果 [{}] + 网格有未用轴
//   全复制（单轴）  slice_replicated_mesh_2               同上，网格单轴
//   步长            slice_strided                         [1:7:2] 局部形状不同
//   reverse 多维    reverse_multi_dim_divisible           多维反转 + 可整除
//   reverse 单维    reverse_single_dim_indivisible        单维反转 + 不可整除 + 轴序不同

// 【★ 三条结论】
//   ① slice 的核心判据是"切片是否跨越设备边界"
//      不跨越则 comm_free，跨越则需通信
//   ② reverse 的难点是"分片顺序也要反转"
//      反转被分片的维时，设备之间的数据要重新分配
//   ③ 不可整除与步长是两个额外复杂度
//      前者要补齐（L5-09），后者让"每台拿多少"不再是简单除法

// 【★ 与前面课的呼应】
//   L5-04  作用维判据（本课更细一层：范围完整也不跨界）
//   L5-09  不可整除补齐（slice_indivisible）
//   L4-09  REPL / HALO（partitioner 文件的两个 RUN 行）
//   L1-02  网格的轴（slice_replicated 的未用轴 a）

// 【L6 的进度】
//   L6-00~07 已做（机制 / sdy 通信 / stablehlo 通信 / 卷积 /
//                   矩阵乘-fft-iota / gather / pad / reshape）
//   L6-08 reverse/slice（本课）    L6-09 scatter 与杂项

// 一句话总结：
//   slice / reverse 的降级取决于"是否改变设备之间的数据归属"
//   slice 看范围是否跨界、reverse 看是否反转了被分片的维`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: '基准', n: 1, c: '#94a3b8' }, { t: '不跨界', n: 1, c: '#4ade80' },
      { t: '跨界', n: 1, c: '#fb7185' }, { t: '不可整除', n: 1, c: '#fbbf24' },
      { t: '全复制', n: 2, c: '#38bdf8' }, { t: '步长', n: 1, c: '#c084fc' },
      { t: 'reverse', n: 2, c: '#f472b6' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:104px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div style="font-size:9.5px;color:${f.c};overflow-wrap:anywhere">${f.t}</div>
        <div class="big" style="font-size:16px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 100)); msg.innerHTML = '<b>9 个文件</b> —— 7 个 slice + 2 个 reverse。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>★ 三条结论</b>：slice 看是否跨界 / reverse 看是否反转分片维 / 不可整除与步长是额外复杂度。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>与前面课呼应</b>：L5-04（作用维）、L5-09（补齐）、L4-09（REPL/HALO）。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L6-09 讲 scatter 与杂项 —— <b>L6 的收官课</b>。';
    });
  }
},

];
