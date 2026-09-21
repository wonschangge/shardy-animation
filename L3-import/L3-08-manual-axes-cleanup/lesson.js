/* ==========================================================================
   L3-08 · manual-axes-cleanup
   --------------------------------------------------------------------------
   覆盖：transforms/import/test/manual_axes_cleanup.mlir (191 行 / 12 用例)
         transforms/import/test/manual_axes_cleanup_failures.mlir (10 / 1)
   目标：讲透 manual 轴清理的三个动作 —— 补 replicated、排序、替换空网格。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 三个动作 */
{
  kicker: 'L3-08 · 清理 manual 轴',
  title: '三个动作：<span class="hl-a">补 replicated</span>、<span class="hl-a">排序</span>、<span class="hl-a">换空网格</span>',
  sub: 'L1-07 立的不变量是：**manual 轴必须在所有分片里显式出现**（切维或进 `replicated`）。这个 pass 负责补齐。',
  caption: 'L3-01 见过它的三种情形；这一课是完整展开。',
  code: `// RUN: sdy_opt %s -split-input-file -sdy-manual-axes-cleanup

// 网格（注意轴序！）
sdy.mesh @mesh = <["c"=2, "a"=2, "b"=2]>      // c, a, b —— 不是字母序
sdy.mesh @mesh_xyz = <["x"=2, "y"=2, "z"=2]>

// 【动作 ①】把没切维度的 manual 轴补进 replicated
in_shardings=[<@mesh, [{"c", ?}]>]  manual_axes={"c", "a"}
-> in_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>]
//                              ^^^^^^^^^^^^^^^^ "a" 没切维度，补进来

// 【动作 ②】按【网格声明顺序】排序 manual_axes
manual_axes={"b", "a", "c"}   ->  manual_axes={"c", "a", "b"}
//   ^ 看起来像字母序              ^ 实际是 @mesh 的轴序

// 【动作 ③】空网格被替换成实际网格
in_shardings=[<@mesh_xyz, [{"x"}]>]  out_shardings=[<@empty_mesh, [{}]>]
-> out_shardings=[<@mesh_xyz, [{}], replicated={"x","y"}>]
//                  ^^^^^^^^^ 换成 in 用的那个网格`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 补 replicated', c: '#4ade80',
        d: 'manual_axes 里没切维度的轴<br>→ 加进 <span class="mono">replicated</span>' },
      { t: '② 排序', c: '#38bdf8',
        d: '<span class="mono">manual_axes</span> 按<br><b>网格声明顺序</b>排列<br>（不是字母序）' },
      { t: '③ 换空网格', c: '#fbbf24',
        d: '<span class="mono">@empty_mesh</span> 替换成<br>同一个手动计算里<br>实际使用的网格' },
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
        '这是三个动作里最核心的一个 —— 直接落实 L1-07 的不变量。',
        '<b>为什么要排序</b>：让输出<b>确定</b>，便于比较与测试。排序依据是网格本身，不是名字。',
        '替换后才有"完整的轴列表"可用 —— 动作 ①② 都依赖它。',
      ][i];
    }));
    tl.at(12200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>三个动作的顺序</b>：先换空网格（拿到确定的网格）→ 再补 replicated → 最后排序。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 补 replicated */
{
  kicker: 'L3-08 · 清理 manual 轴',
  title: '★ 动作 ①：补 <span class="mono hl-a">replicated</span>',
  sub: '`manual_axes` 里列了但没切任何维度的轴，会被加进 `in_shardings` / `out_shardings` 的 `replicated`。',
  caption: '三种情形：<b>新增</b>、<b>追加到已有</b>、<b>取并集</b>（in/out 各缺一个时）。',
  code: `// 【新增】replicated 原本为空
manual_axes={"c", "a"}
in_shardings=[<@mesh, [{"c", ?}]>]
-> in_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>]

// 【追加】已有 replicated={"a"}，还缺 "b"
manual_axes={"c", "a", "b"}
in_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>]
-> in_shardings=[<@mesh, [{"c", ?}], replicated={"a", "b"}>]

// 【取并集】in 有 {"b"}、out 有 {"a"}，两者都缺一个
manual_axes={"c", "a", "b"}
in_shardings=[<@mesh, [{"c", ?}], replicated={"b"}>]
out_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>]
-> in_shardings =[<@mesh, [{"c", ?}], replicated={"a", "b"}>]
-> out_shardings=[<@mesh, [{"c", ?}], replicated={"a", "b"}>]
//   两者都补成完整的并集`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 新增', c: '#4ade80', d: '<span class="mono">replicated</span> 原本为空<br>→ 直接加上' },
      { t: '② 追加', c: '#38bdf8', d: '已有部分<br>→ <b>追加</b>缺的那些' },
      { t: '③ 取并集', c: '#fbbf24', d: 'in 与 out 各缺一个<br>→ 都补成<b>完整并集</b>' },
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
        '最简单的情形。<span class="mono">"a"</span> 在 manual_axes 里但没切维度 → 进 replicated。',
        '注意是<b>追加</b>，不是替换 —— 已有的 <span class="mono">"a"</span> 保留。',
        '<b>为什么要取并集</b>：in 和 out 的分片必须覆盖<b>全部</b> manual 轴（L1-07 的不变量），缺一不可。',
      ][i];
    }));
    tl.at(12200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>本质</b>：让 in/out_shardings 各自"显式提到"每一个 manual 轴。';
    });
  }
},

/* ------------------------------------------------ 3 ★ 排序 */
{
  kicker: 'L3-08 · 清理 manual 轴',
  title: '★ 动作 ②：按<span class="hl-a">网格声明顺序</span>排序',
  sub: '这是本课最关键的一处 —— 排序依据**不是字母序**，而是网格里的轴序。',
  caption: '看 <span class="mono">@mesh = &lt;["c"=2, "a"=2, "b"=2]&gt;</span>：轴序是 c、a、b。',
  code: `// @mesh = <["c"=2, "a"=2, "b"=2]>      <- 轴序 c, a, b

// 输入
manual_axes={"b", "a", "c"}          // 看起来像字母序

// 输出
manual_axes={"c", "a", "b"}          // 实际是【网格序】

// 另一个网格确认这一点：
// @mesh_xyz = <["x"=2, "y"=2, "z"=2]>
manual_axes={"y", "x", "z"}  ->  manual_axes={"x", "y", "z"}

// replicated 也按同样规则排序：
replicated={"b"} + 补 "a"  ->  replicated={"a", "b"}
//                                        ^ a 在 b 前（网格序 c,a,b）

// 顺带：out_sharding 里的【内联 mesh】也被提升为命名引用
out_shardings=[<mesh<["x"=2, "y"=2, "z"=2]>, ...>]
-> out_shardings=[<@mesh_xyz, ...>]`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ax0)">网格的轴序</div>
        <div class="row" style="gap:6px">
          <div class="chip c0" style="padding:6px 12px">"c"</div>
          <div class="chip c1" style="padding:6px 12px">"a"</div>
          <div class="chip c2" style="padding:6px 12px">"b"</div>
        </div>
        <div class="small faint">@mesh = &lt;["c"=2, "a"=2, "b"=2]&gt;</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<span class="mono">@mesh</span> 的轴序是 <b>c、a、b</b> —— 注意<b>不是字母序</b>。';
    });
    tl.at(4600, () => {
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹', style: 'font-size:26px' }));
      const c = U.el('div', { class: 'col', style: 'gap:7px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ok)">排序后</div>
        <div class="row" style="gap:6px">
          <div class="chip c0" style="padding:6px 12px">"c"</div>
          <div class="chip c1" style="padding:6px 12px">"a"</div>
          <div class="chip c2" style="padding:6px 12px">"b"</div>
        </div>
        <div class="small faint">manual_axes={"c", "a", "b"}</div>`;
      demo.appendChild(c);
      msg.innerHTML = '输入 <span class="mono">{"b","a","c"}</span> 被排成 <span class="mono">{"c","a","b"}</span> —— 完全跟随网格序。';
    });
    tl.at(9000, () => {
      msg.innerHTML = '<b>为什么容易看错</b>：<span class="mono">{"b","a","c"}</span> 恰好像字母序倒过来，容易误以为是某种字母序规则。';
    });
    tl.at(12200, () => {
      msg.innerHTML = '<b>验证方法</b>：换个网格（<span class="mono">@mesh_xyz</span>）看结果 —— <span class="mono">{"y","x","z"}</span> → <span class="mono">{"x","y","z"}</span>，仍是网格序。';
    });
    tl.at(14600, () => {
      msg.innerHTML = '<b>为什么要排序</b>：让输出<b>确定</b>（与用户书写顺序无关），便于比较、测试与去重。';
    });
  }
},

/* ------------------------------------------------ 4 换空网格 */
{
  kicker: 'L3-08 · 清理 manual 轴',
  title: '动作 ③：空网格被<span class="hl-a">替换成实际网格</span>',
  sub: '`@empty_mesh` 是个占位符（L2-01 讲过它的传播语义）—— 这里要把它换成"这次手动计算真正用的网格"。',
  caption: '替换后才能按<b>那个网格的轴序</b>补 `replicated` 与排序。',
  code: `// in 用 @mesh_xyz，out 用 @empty_mesh
in_shardings=[<@mesh_xyz, [{"x"}]>]
out_shardings=[<@empty_mesh, [{}]>]
manual_axes={"y", "x"}

// 输出：out 也变成 @mesh_xyz
in_shardings=[<@mesh_xyz, [{"x"}], replicated={"y"}>]
out_shardings=[<@mesh_xyz, [{}], replicated={"x", "y"}>]
//              ^^^^^^^^^ 空网格被替换

// 对称情形：in 是空网格、out 是实际网格（用例 empty_mesh_operand）
in_shardings=[<@empty_mesh, [{}]>]     ->  <@mesh_xyz, [{}], replicated={"x","y"}>
out_shardings=[<@mesh_xyz, [{"x"}]>]   ->  <@mesh_xyz, [{"x"}], replicated={"y"}>

// 内联空网格同理（用例 inlined_empty_mesh_result）
//   这个 pass 不负责提升，只做替换 -> 输出仍是内联写法`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'col', style: 'gap:8px;align-items:center' });
      c.innerHTML = `
        <div class="small mono faint">输入</div>
        <div class="chip c0" style="padding:6px 11px;font-size:11.5px">in : @mesh_xyz</div>
        <div class="chip c3" style="padding:6px 11px;font-size:11.5px">out: @empty_mesh</div>
        <div class="small" style="color:var(--bad);font-size:11px">两边网格不一致</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<span class="mono">@empty_mesh</span> 是占位符 —— 表示"这里还没定用哪个网格"。';
    });
    tl.at(4400, () => {
      demo.appendChild(U.el('div', { class: 'arrow anim', html: '⟹', style: 'font-size:26px' }));
      const c = U.el('div', { class: 'col', style: 'gap:8px;align-items:center' });
      c.innerHTML = `
        <div class="small mono" style="color:var(--ok)">输出</div>
        <div class="chip c0" style="padding:6px 11px;font-size:11.5px">in : @mesh_xyz + replicated</div>
        <div class="chip c0" style="padding:6px 11px;font-size:11.5px">out: @mesh_xyz + replicated</div>
        <div class="small" style="color:var(--ok);font-size:11px">统一到实际网格</div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>替换</b>：空网格被换成 in 用的那个网格，然后两边都补 <span class="mono">replicated</span>。';
    });
    tl.at(8600, () => {
      msg.innerHTML = '<b>为什么必须先替换</b>：补 <span class="mono">replicated</span> 与排序都要知道<b>轴有哪些</b> —— 空网格给不出轴列表。';
    });
    tl.at(11600, () => {
      msg.innerHTML = '<b>对称情形</b>：<span class="mono">empty_mesh_operand</span> 用例里 in 是空网格、out 是实际网格，处理方向相反。';
    });
    tl.at(13800, () => {
      msg.innerHTML = '<b>注意分工</b>：这个 pass 只做替换，<b>不做提升</b> —— 内联 mesh 的提升是 L3-07 的活。';
    });
  }
},

/* ------------------------------------------------ 5 三个边界 */
{
  kicker: 'L3-08 · 清理 manual 轴',
  title: '三个<span class="hl-a">边界</span>：token / 子轴 / unreduced',
  sub: '这三种情形说明清理不是"无脑加一个 replicated 集合"。',
  caption: '尤其是<b>子轴</b>：会被展开成完整的子轴集合，而不是加一个笼统的轴名。',
  code: `// ① token 不加 replicated
in_shardings=[<@mesh, []>, <@mesh, [{?}]>]   manual_axes={"c"}
-> in_shardings=[<@mesh, []>, <@mesh, [{?}], replicated={"c"}>]
//                 ^^^^^^^^^^^ token 保持原样，不加
// 理由：rank-0 类型不能有 replicated（L1-02）

// ② 子轴被展开成完整集合
// @mesh_x_8_y_8 = <["x"=8, "y"=8]>，"y" 切成 4 个子轴
in_shardings=[<@mesh_x_8_y_8, [{"y":(2)2}]>]
-> replicated={"x", "y":(1)2, "y":(4)2}
//              ^^^^  ^^^^^^^^^^^^^^^^^^ 其余子轴全部补上
out_shardings=[<@mesh_x_8_y_8, [{"y":(1)2}], replicated={"x":(2)2}>]
-> replicated={"x", "y":(2)4}

// ③ unreduced 被保留（三种状态共存）
in_shardings=[<@mesh, [{"c", ?}], unreduced=max{"b"}>]
-> in_shardings=[<@mesh, [{"c", ?}], replicated={"a"}, unreduced=max{"b"}>]
//                分片部分     ^^^^^^^^^^^^^^^ 新增    ^^^^^^^^^^^^^^^^^^ 原样`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① token', c: '#fbbf24',
        d: 'token 的 <span class="mono">&lt;@mesh, []&gt;</span> <b>保持原样</b><br>replicated 只加在 tensor 那一项上' },
      { t: '② 子轴', c: '#c084fc',
        d: '子轴被<b>展开成完整集合</b><br>而不是加一个笼统的轴名' },
      { t: '③ unreduced', c: '#4ade80',
        d: '补 replicated 的同时<br><b>unreduced 原样保留</b><br>三种状态可共存' },
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
        '<b>依据</b>：L1-02 的不变量 —— rank-0 类型不能有 <span class="mono">replicated</span>。',
        '<b>为什么展开</b>：一个轴被切成子轴后，"用哪个子轴"是<b>逐子轴</b>决定的，必须逐个说明。',
        '<b>三种状态</b>：分片（维度分片）/ 复制（replicated）/ 未归约（unreduced）—— 它们<b>互不排斥</b>。',
      ][i];
    }));
    tl.at(12800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>共同点</b>：清理只动"该动的地方"，其余一律保持原样。';
    });
  }
},

/* ------------------------------------------------ 6 失败情形 */
{
  kicker: 'L3-08 · 清理 manual 轴',
  title: '唯一会<span class="hl-a">报错</span>的情形',
  sub: '没有分片可补，但 `manual_axes` 非空且函数体非空 —— 此时 manual 轴"无处安放"。',
  caption: '注意报文里的 <span class="mono">and the body is not empty</span>：如果<b>函数体是空的</b>，这种写法是允许的。',
  code: `func.func @manual_computation_no_inputs_or_outputs_with_manual_axes() {
  // expected-error @+1 {{op has manual_axes when there are no in/out shardings and the body is not empty}}
  sdy.manual_computation() in_shardings=[] out_shardings=[] manual_axes={"a"} () {
    %0 = sdy.constant dense<1.000000e+00> : tensor<8xf32>
    sdy.return
  } : () -> ()
  func.return
}

// 读法：
//   in_shardings=[] 且 out_shardings=[]  -> 没有地方可以写 replicated
//   manual_axes={"a"} 非空               -> 有个轴要安放
//   函数体非空                            -> 不是"空体内联"的情形
//   => 报错

// 为什么"函数体空"就允许：
//   L1-07 讲过 —— 空体 + 无 in/out 的 manual_computation
//   会被规范化直接内联掉（连算子都消失）
//   所以那个 manual_axes 根本没有机会生效`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:22px;justify-content:center;align-items:stretch" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    tl.at(700, () => {
      const c = U.el('div', { class: 'card', style: 'width:300px;border-color:rgba(251,113,133,.5)' });
      c.innerHTML = `<div class="card-t" style="color:var(--bad);font-size:12px">✗ 函数体非空</div>
        <div class="card-d" style="font-size:11.5px">
          <span class="mono">in/out_shardings=[]</span><br>
          <span class="mono">manual_axes={"a"}</span><br>
          函数体有算子<br>
          <b>→ 报错</b></div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>矛盾</b>：有个 manual 轴要安放，但没有任何分片可以写它。';
    });
    tl.at(4400, () => {
      const c = U.el('div', { class: 'card', style: 'width:300px;border-color:rgba(74,222,128,.5)' });
      c.innerHTML = `<div class="card-t" style="color:var(--ok);font-size:12px">✓ 函数体为空</div>
        <div class="card-d" style="font-size:11.5px">
          同样的签名<br>
          但函数体是空的<br>
          <b>→ 允许</b></div>`;
      demo.appendChild(c);
      msg.innerHTML = '<b>为什么允许</b>：空体 + 无 in/out 的 <span class="mono">manual_computation</span> 会被直接内联掉（L1-07）。';
    });
    tl.at(8600, () => {
      msg.innerHTML = '<b>报文的关键部分</b>：<span class="mono">... and the body is not empty</span> —— 条件里明确包含了这一点。';
    });
    tl.at(11400, () => {
      msg.innerHTML = '<b>实践建议</b>：看到这个报错，通常说明你写了 <span class="mono">manual_axes</span> 但忘了写 <span class="mono">in/out_shardings</span>。';
    });
  }
},

/* ------------------------------------------------------------ 7 练习 */
{
  kicker: 'L3-08 · 练习',
  title: '练一练：<span class="hl-a">清理后会是什么样</span>',
  sub: '三道题分别考：补 replicated、排序依据、失败条件。',
  caption: '一句话总结：<b>补 replicated + 按网格序排序 + 换掉空网格</b>。',
  code: `// 题 1：manual_axes={"c","a"}，in_sharding 只有 {"c",?}，
//       清理后 in_sharding 是什么？

// 题 2：manual_axes={"b","a","c"}，@mesh = <["c"=2,"a"=2,"b"=2]>，
//       排序后是什么？依据是什么？

// 题 3：什么情况下这个 pass 会报错？`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '<span class="mono">manual_axes={"c","a"}</span>，<span class="mono">in_sharding</span> 只有 <span class="mono">{"c",?}</span>，清理后是什么？',
        a: '<span class="mono">[&lt;@mesh, [{"c", ?}], replicated={"a"}&gt;]</span>' +
           '<br><b>理由</b>：<span class="mono">"a"</span> 在 <span class="mono">manual_axes</span> 里但没切任何维度 → 补进 <span class="mono">replicated</span>。' +
           '<br><span class="dim">这是落实 L1-07 的不变量：manual 轴必须在所有分片里显式出现（切维<b>或</b>进 replicated）。</span>'
      },
      {
        q: '<span class="mono">manual_axes={"b","a","c"}</span>，<span class="mono">@mesh = &lt;["c"=2,"a"=2,"b"=2]&gt;</span>，排序后是什么？依据是什么？',
        a: '<span class="mono">{"c", "a", "b"}</span>，依据是<b>网格的声明顺序</b>（<b>不是</b>字母序）。' +
           '<br><b>怎么验证</b>：换个网格 <span class="mono">@mesh_xyz = &lt;["x"=2,"y"=2,"z"=2]&gt;</span>，<span class="mono">{"y","x","z"}</span> → <span class="mono">{"x","y","z"}</span>，仍是网格序。' +
           '<br><span class="dim">容易看错：<span class="mono">{"b","a","c"}</span> 恰好像字母序倒过来。</span>'
      },
      {
        q: '什么情况下这个 pass 会<b>报错</b>？',
        a: '<b>没有分片可补，但 manual_axes 非空且函数体非空</b>。' +
           '<br>报文：<span class="mono">op has manual_axes when there are no in/out shardings and the body is not empty</span>' +
           '<br><b>注意</b>：如果<b>函数体是空的</b>，这种写法是<b>允许</b>的 —— 空体 + 无 in/out 的 <span class="mono">manual_computation</span> 会被直接内联掉（L1-07），manual_axes 没机会生效。'
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
