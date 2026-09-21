/* ==========================================================================
   L1-07 · manual-computation
   --------------------------------------------------------------------------
   覆盖：ir/test/manual_computation_parse_print.mlir (246)
         ir/test/manual_computation_verification.mlir (296)
         ir/test/manual_computation_canonicalization.mlir (89)
   目标：讲透 sdy.manual_computation —— 局部形状、manual/自由轴、不变量、嵌套。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------------ 1 动机 */
{
  kicker: 'L1-07 · 手动计算',
  title: '为什么要<span class="hl-a">手动计算</span>？',
  sub: '传播能自动决定分片、自动插通信。但有些用户想<b>自己控制</b>某一段计算怎么切、用哪些通信。',
  caption: '典型场景：想让 matmul 用特定通信模式、想复现论文里的并行策略、或想绕开编译器的某个决策。',
  code: `// 自动：传播决定一切
%0 = stablehlo.dot_general %a, %b : ...

// 手动：这段我自己来
%0 = sdy.manual_computation(%a)
     in_shardings=[<@mesh, [{"data"}, {"model", ?}]>]
     out_shardings=[<@mesh, [{"data"}, {?}]>]
     manual_axes={"data"}          // "data" 轴由我手动管理
     (%a_local: tensor<8x32xf32>) { // 区域内是【局部形状】
  // 这里写的是单设备上的代码，通信要自己写
  %1 = stablehlo.dot_general %a_local, ...
  sdy.return %1 : tensor<8x32xf32>
} : (tensor<16x32xf32>) -> tensor<16x32xf32>`,
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:16px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:22px;justify-content:center;align-items:stretch">
        <div class="card" style="width:290px;border-color:rgba(94,234,212,.5)">
          <div class="card-t" style="color:var(--accent)">自动（传播）</div>
          <div class="card-d">编译器决定分片与通信。<br>
            <b>省心</b>，但决策不完全可控。<br>
            <span class="dim small">覆盖你 95% 的需求</span></div>
        </div>
        <div class="card" style="width:290px;border-color:rgba(251,191,36,.5)">
          <div class="card-t" style="color:var(--warn)">手动（manual_computation）</div>
          <div class="card-d">指定 <span class="mono">manual_axes</span>，区域内写<b>局部代码</b>，通信自己写。<br>
            <b>完全可控</b>，但责任也归你。</div>
        </div>
      </div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:740px"></div>`;
    root.appendChild(wrap);
    const msg = wrap.querySelector('#msg');
    tl.at(900, () => { msg.innerHTML = '它<b>不是</b>替代传播的方案，而是给传播开了一个"我自己来"的口子。'; });
    tl.at(3600, () => { msg.innerHTML = '关键概念：<b>manual 轴被冻结</b>，区域内看到的是局部形状；<b>其余轴仍由传播处理</b>。'; });
    tl.at(6400, () => { msg.innerHTML = '这正好解释了 L1-03 的第 7 条不变量：区域内不能再用 manual 轴切维度。'; });
    tl.at(9400, () => { msg.innerHTML = '它还是<b>数据流算子</b>（L2-08），所以传播能穿过它，在自由轴上继续工作。'; });
    tl.at(11800, () => { msg.innerHTML = '规范化会把"没用的手动计算"消掉（空体、无 manual 轴），避免开销。'; });
  }
},

/* ------------------------------------------------------------ 2 语法 */
{
  kicker: 'L1-07 · 手动计算',
  title: '四个部分：<span class="mono hl-a">in_shardings</span> / <span class="mono hl-a">out_shardings</span> / <span class="mono hl-a">manual_axes</span> / 体',
  sub: '前两个描述<b>外层张量</b>怎么分片；<span class="mono">manual_axes</span> 指定哪些轴由你手动管；体里是<b>局部代码</b>。',
  caption: '注意：<span class="mono">manual_axes</span> 里的轴必须在<b>所有</b> in/out 分片里显式出现 —— 要么切维度，要么进 <span class="mono">replicated</span>。',
  code: `%0 = sdy.manual_computation(%arg0)
     in_shardings=[<@meshA, [{"a", ?}, {?}]>]    // ① 每个操作数的分片
     out_shardings=[<@meshA, [{"a", ?}, {?}]>]   // ② 每个结果的分片
     manual_axes={"a"}                            // ③ 我手动管的轴
     (%arg1: tensor<8x32xf32>) {                  // ④ 体：局部形状！
  %1 = stablehlo.add %arg1, %arg1 : tensor<8x32xf32>
  sdy.return %1 : tensor<8x32xf32>
} : (tensor<16x32xf32>) -> tensor<16x32xf32>

// 网格 sdy.mesh @meshA = <["a"=2, "b"=2]>
// "a" 是 manual 轴（大小 2）-> 区域内 16/2 = 8
// "b" 是自由轴            -> 不影响局部形状`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%;align-items:center' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:750px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① in_shardings', d: '每个操作数一项。<br>可用 manual 轴与<b>自由轴</b>。', c: '#38bdf8' },
      { t: '② out_shardings', d: '每个结果一项。<br>同样必须显式含所有 manual 轴。', c: '#c084fc' },
      { t: '③ manual_axes', d: '这些轴<b>被冻结</b>：区域内不能再切，<br>传播也不能改。', c: '#fbbf24' },
      { t: '④ 体（局部代码）', d: '块参数是<b>局部形状</b>；<br>通信要自己写。', c: '#4ade80' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:178px;opacity:.32;transition:.3s;border-color:' + x.c + '55;padding:9px' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:11.5px">${x.t}</div>
        <div class="card-d" style="font-size:11px;line-height:1.5">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.32');
      msg.innerHTML = [
        '分片里的 <span class="mono">?</span> 表示开维；manual 轴通常也带 <span class="mono">?</span>（因为还可以叠加自由轴）。',
        '输出的分片可以与输入不同，但<b>同一个网格</b>，且都必须覆盖 manual 轴。',
        '把轴交给 manual，就是把它的分片与通信责任<b>全部接手</b>。',
        '区域是 <span class="mono">IsolatedFromAbove</span>：不能引用外面的值，只能用块参数。',
      ][i];
    }));
    tl.at(13600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '一句话：<b>manual_axes 决定"哪里是局部世界"，其余全是全局世界</b>。';
    });
  }
},

/* -------------------------------------------------------- 3 局部形状 */
{
  kicker: 'L1-07 · 手动计算',
  title: '★ 核心：<span class="hl-a">局部形状</span>怎么算',
  sub: '区域内每个张量的形状，等于外层形状<b>除以 manual 轴大小</b>。自由轴<b>不影响</b>局部形状。',
  caption: '这条规则是本课最重要的：它解释了为什么体内写的是 <span class="mono">tensor&lt;8x32xf32&gt;</span> 而不是 <span class="mono">tensor&lt;16x32xf32&gt;</span>。',
  code: `sdy.mesh @meshA = <["a"=2, "b"=2]>

%0 = sdy.manual_computation(%arg0)
     in_shardings=[<@meshA, [{"a", ?}, {?}]>]
     out_shardings=[<@meshA, [{"a", ?}, {?}]>]
     manual_axes={"a"}
     (%arg1: tensor<8x32xf32>) {   // ← 16 / 2 = 8
  ...
} : (tensor<16x32xf32>) -> tensor<16x32xf32>

// 逐维计算：
//   dim0: 外层 16，被 manual 轴 "a"（大小 2）切 -> 16/2 = 8
//   dim1: 外层 32，没有被任何 manual 轴切   -> 32
//   => tensor<8x32xf32>

// 只切输出（输入不切）也合法：
//   "manual_computation_single_sharded_output" 用例：
//   体返回 tensor<16x32xf32>，但外层结果是 tensor<32x32xf32>

// 动态维同样适用：tensor<?x32xf32> 的局部仍是 tensor<?x32xf32>
//   （? 无法再除，但形状规则一致）`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:18px;justify-content:center;align-items:center" id="calc"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:760px"></div>`;
    root.appendChild(wrap);
    const calc = wrap.querySelector('#calc'), msg = wrap.querySelector('#msg');

    const box = (title, dims, cls, sub) => {
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center' });
      c.innerHTML = `<div class="small mono faint">${title}</div>
        <div class="row" style="gap:6px" data-d></div>
        <div class="small faint" style="height:18px">${sub || ''}</div>`;
      const h = c.querySelector('[data-d]');
      dims.forEach(t => {
        const e = U.el('div', { class: 'chip ' + cls, style: 'padding:7px 14px;font-size:15px' });
        e.textContent = t; h.appendChild(e);
      });
      calc.appendChild(c); return c;
    };

    tl.at(700, () => {
      box('外层（全局）', ['16', '32'], 'mut', 'tensor&lt;16x32xf32&gt;');
      calc.appendChild(U.el('div', { class: 'arrow anim', html: 'manual_axes={"a"}' }));
      box('体内（局部）', ['8', '32'], 'c4', 'tensor&lt;8x32xf32&gt;');
      msg.innerHTML = '<span class="mono">"a"</span> 大小 2 → 被它切的维度 <b>16 ÷ 2 = 8</b>；没被切的维度原样保留。';
    });
    tl.at(4400, () => {
      msg.innerHTML = '<b>自由轴不参与</b>：即使 in_sharding 写成 <span class="mono">{"a", "b"}</span>（含自由轴 "b"），局部形状仍只看 manual 轴。';
      const first = calc.querySelector('.chip');
      if (first) first.classList.add('pulse');
    });
    tl.at(7800, () => {
      msg.innerHTML = '所以校验器会核对：<b>外层形状 ÷ manual 轴大小 == 体内块参数形状</b>，不符就报 <span class="mono">Expected local shape …</span>。';
    });
    tl.at(11200, () => {
      msg.innerHTML = '推论：manual 轴必须<b>整除</b>维度大小 —— 否则会需要 padding，而 manual 计算不允许 padding。';
    });
    tl.at(14000, () => {
      msg.innerHTML = '<b>记法</b>：局部维度 = 外层维度 ÷ ∏(该维上属于 manual_axes 的轴大小)。';
    });
  }
},

/* ------------------------------------------------------------ 4 自由轴 */
{
  kicker: 'L1-07 · 手动计算',
  title: '自由轴：manual 之外的轴<span class="hl-a">仍归传播管</span>',
  sub: '只有 <span class="mono">manual_axes</span> 里列出的轴被冻结。其余轴是<b>自由轴</b>，可以出现在 in/out 分片里，也可以被传播继续切。',
  caption: '顺序约束：同一维的分片里，<b>manual 轴必须在自由轴之前</b>（更 major）。这条由校验器强制。',
  code: `// 输入 dim0 = {"a", "b"}：a 是 manual 轴，b 是自由轴
%0 = sdy.manual_computation(%arg0)
     in_shardings=[<@meshA, [{"a", "b"}, {?}]>]
     out_shardings=[<@meshA, [{"a", ?}, {?}]>]
     manual_axes={"a"}
     (%arg1: tensor<8x32xf32>) {   // 局部形状只看 "a"
  ...
}

// ✓ manual 在前、自由在后：{"a", "b"}
// ✗ 反过来：{"b", "a"}
//   -> op operand sharding ... must have all manual axes come before
//      free axes ... Saw manual axis "b" after free axis "a"

// 注意输入输出可以不一样：
//   输入 {"a", "b"}（已按自由轴 b 切）
//   输出 {"a", ?}（b 变成开维，传播可再决定）`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:26px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:750px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const mk = (cells, sub) => {
      const c = U.el('div', { class: 'col', style: 'gap:6px;align-items:center' });
      c.innerHTML = `<div class="row" style="gap:6px" data-d></div><div class="small faint" style="height:18px">${sub || ''}</div>`;
      const h = c.querySelector('[data-d]');
      cells.forEach(([t, cls]) => {
        const e = U.el('div', { class: 'chip ' + cls, style: 'padding:6px 12px;font-size:13px' });
        e.textContent = t; h.appendChild(e);
      });
      demo.appendChild(c); return c;
    };

    tl.at(700, () => {
      mk([['"a"', 'c2'], ['"b"', 'c0']], 'manual 在前 ✓');
      msg.innerHTML = '<span class="mono">{"a", "b"}</span>：manual 轴 "a" 更 major，自由轴 "b" 在后 —— 合法';
    });
    tl.at(4000, () => {
      demo.innerHTML = '';
      mk([['"b"', 'c0'], ['"a"', 'c2']], 'manual 在后 ✗');
      msg.innerHTML = '<span class="mono">{"b", "a"}</span>：自由轴排在 manual 轴前面 → 报 <span class="mono">must have all manual axes come before free axes</span>';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>为什么</b>：局部形状由 manual 轴前缀决定。若自由轴插在前面，就无法定义"切到第几层是局部"。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '自由轴的<b>用途</b>：让传播在这块手动区域之外继续工作 —— 例如把批维按数据并行再切一刀。';
    });
    tl.at(13200, () => {
      msg.innerHTML = '这也是 manual_computation 作为<b>数据流算子</b>的接口：内部冻结，外部照常传播。';
    });
  }
},

/* ------------------------------------------------------------ 5 嵌套 */
{
  kicker: 'L1-07 · 手动计算',
  title: '嵌套：每层绑定<span class="hl-a">不同的</span> manual 轴',
  sub: '可以嵌套多层手动计算，只要每层用<b>互不相同</b>的 manual 轴。外层先切，内层再在自己剩下的世界里切。',
  caption: '嵌套让"两级并行"这类策略能被精确表达：外层绑 <span class="mono">"a"</span>（如张量并行），内层绑 <span class="mono">"b"</span>（如专家并行）。',
  code: `// 网格 @meshA = <["a"=2, "b"=2]>
// 外层绑 "a"：16 -> 8
%0 = sdy.manual_computation(%arg0)
     in_shardings=[<@meshA, [{"a", ?}, {?}]>]
     out_shardings=[<@meshA, [{?}, {?}], replicated={"a"}>]
     manual_axes={"a"} (%arg1: tensor<8x32xf32>) {

  // 内层绑 "b"：8 -> 4
  %1 = sdy.manual_computation(%arg1)
       in_shardings=[<@meshA, [{"b", ?}, {?}]>]
       out_shardings=[<@meshA, [{"b", ?}, {?}]>]
       manual_axes={"b"} (%arg2: tensor<4x32xf32>) {
    %2 = stablehlo.add %arg2, %arg2 : tensor<4x32xf32>
    sdy.return %2 : tensor<4x32xf32>
  } : (tensor<8x32xf32>) -> tensor<8x32xf32>

  sdy.return %1 : tensor<8x32xf32>
} : (tensor<16x32xf32>) -> tensor<8x32xf32>

// 校验：内层不能用已被父级绑定的轴
//   operates on axis "a" which is already bound by a parent
//   sdy.manual_computation op`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:18px;justify-content:center;align-items:center" id="nest"></div>
      <div class="formula" id="msg" style="min-height:54px;display:flex;align-items:center;text-align:center;max-width:750px"></div>`;
    root.appendChild(wrap);
    const nest = wrap.querySelector('#nest'), msg = wrap.querySelector('#msg');

    const lvl = (title, shape, axis, color) => {
      const c = U.el('div', { class: 'card', style: `width:200px;border-color:${color}66;text-align:center;opacity:0;transition:.5s` });
      c.innerHTML = `<div class="card-t" style="color:${color};font-size:12px">${title}</div>
        <div class="mono" style="margin:5px 0;font-size:13px;color:#cfe0ff">${U.esc(shape)}</div>
        <div class="small faint">manual_axes={"${axis}"}</div>`;
      nest.appendChild(c); return c;
    };

    tl.at(700, () => {
      lvl('外层', 'tensor<16x32xf32>', 'a', '#38bdf8').style.opacity = '1';
      msg.innerHTML = '最外层：<span class="mono">tensor&lt;16x32xf32&gt;</span>，绑定轴 "a"（大小 2）';
    });
    tl.at(3400, () => {
      lvl('内层', 'tensor<8x32xf32>', 'b', '#c084fc').style.opacity = '1';
      msg.innerHTML = '外层的体内是 <span class="mono">tensor&lt;8x32xf32&gt;</span>（16÷2）；内层再绑定轴 "b"（大小 2）';
    });
    tl.at(6200, () => {
      lvl('最内层', 'tensor<4x32xf32>', '—', '#4ade80').style.opacity = '1';
      msg.innerHTML = '最内层看到 <span class="mono">tensor&lt;4x32xf32&gt;</span>（8÷2）—— 每进一层就再切一刀';
    });
    tl.at(9400, () => {
      msg.innerHTML = '<b>约束</b>：内层的 manual 轴<b>不能</b>与外层重复 —— 用已被绑定的轴会报错。';
    });
    tl.at(12200, () => {
      msg.innerHTML = '标准做法：给每个 mesh 轴分配一层。轴用完就不会冲突，也不存在嵌套过深的问题。';
    });
  }
},

/* ---------------------------------------------------- 6 校验错误 */
{
  kicker: 'L1-07 · 校验',
  title: '不变量校验：<span class="hl-a">四类根因</span>',
  sub: '21 条校验错误归成四类：<b>网格一致</b>、<b>结构与数量</b>、<b>局部形状</b>、<b>轴的角色</b>。',
  caption: '其中"局部形状"这一类是本课特有的 —— 它把"外层形状、分片、体内形状"三者绑在一起交叉验证。',
  code: `// ① 网格必须一致
in_shardings=[<@meshA, ...>] out_shardings=[<@meshB, ...>]
//   all in and out shardings must be bound to the same mesh

// ② 结构与数量：分片项数 = 值的个数；区域参数/返回值个数 = 操作数/结果个数
in_shardings=[<@mesh, [{}, {}]>, <@mesh, [{}, {}]>] 用在 1 个操作数上
//   op operand shardings don't match number of values: 2 shardings vs 1 values

// ③ 局部形状必须吻合
in_shardings=[<@mesh, [{}, {}]>] ... (%arg1: tensor<8x32xf32>)
//   Expected local shape 'tensor<16x32xf32>', actual local shape 'tensor<8x32xf32>'

// ④ 轴的角色
manual_axes={"b"} 但分片写成 [{"a", "b"}]     // manual 轴必须在前
manual_axes={"a", "b"} 但网格里只有 "a"        // unknown manual axis
内层 manual_axes={"a"} 而外层已绑定 "a"        // already bound by a parent
%0 = ... [{"a"}] ... 而维度大小 6 不能被 4 整除 // not divisible -> 会引入 padding

// ⑤ 区域隔离
%1 = stablehlo.add %arg0, %arg1     // 引用了区域外的 %arg0
//   using value defined outside the region`,
  duration: 20000,
  build(root, tl) {
    const wrap = W.errLayout(root);
    W.errScenes(wrap, tl, [
      {
        ir: '%0 = sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{}, {}]>] out_shardings=[<@meshB, [{}, {}]>] manual_axes={"a"} (%arg1: tensor<16x32xf32>) {',
        err: 'all in and out shardings must be bound to the same mesh or an empty mesh.',
        why: '输入用 <span class="mono">@meshA=&lt;["a"=2]&gt;</span>、输出用 <span class="mono">@meshB=&lt;["a"=4]&gt;</span>。同一次手动计算必须绑定同一个网格。'
      },
      {
        ir: '%0 = sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{"a"}, {}]>] out_shardings=[<@meshA, [{"a"}, {?}]>] manual_axes={"a", "b"} (%arg1: tensor<8x32xf32>) {',
        err: 'unknown manual axis: "b"',
        why: '<span class="mono">@meshA</span> 里没有轴 <span class="mono">"b"</span> —— manual_axes 必须来自绑定的网格。'
      },
      {
        ir: '%0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}]>, <@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {}]>] manual_axes={} (%arg1: tensor<16x32xf32>) {',
        err: "op operand shardings don't match number of values: 2 shardings vs 1 values",
        why: '只有 1 个操作数，却给了 2 项 in_sharding。'
      },
      {
        ir: '%0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>] manual_axes={} (%arg1: tensor<8x32xf32>) {',
        err: "op operand shape, corresponding sharding, and region operand shape at index 0 must match. Expected local shape 'tensor<16x32xf32>', actual local shape 'tensor<8x32xf32>'",
        why: '<span class="mono">manual_axes={}</span> → 没有任何轴被切 → 局部形状应等于外层 <span class="mono">16x32</span>，但块参数写成了 <span class="mono">8x32</span>。'
      },
      {
        ir: '%0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a"}]>] out_shardings=[<@mesh, [{"a"}]>] manual_axes={"a"} (%arg1: tensor<1xf32>) {',
        err: 'dimension size 6 is not divisible by the manual axes size 4',
        why: '维度大小 6 不能被 manual 轴大小 4 整除。手动计算<b>不允许 padding</b>，所以必须整除。'
      },
      {
        ir: '%0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"a", "b"}]>] out_shardings=[<@mesh, [{}, {}], replicated={"b"}>] manual_axes={"b"} (%arg1: tensor<16x16xf32>) {',
        err: 'op operand sharding at index 0 must have all manual axes come before free axes in its dimension sharding at index 1. Saw manual axis "b" after free axis "a"',
        why: 'dim1 写成 <span class="mono">{"a", "b"}</span>：自由轴 "a" 排在 manual 轴 "b" 前面。应为 <span class="mono">{"b", "a"}</span>。'
      },
      {
        ir: '    %1 = stablehlo.add %arg0, %arg1 : tensor<16x32xf32>',
        err: "'stablehlo.add' op using value defined outside the region",
        why: '区域是 <span class="mono">IsolatedFromAbove</span>：只能使用块参数，不能引用外层的 <span class="mono">%arg0</span>。'
      },
      {
        ir: '    %2 = sdy.manual_computation(%arg2) in_shardings=[<@foo, [{}, {}]>] out_shardings=[<@foo, [{}, {}]>] manual_axes={"a"} (%arg3: tensor<4x32xf32>) {',
        err: "op operates on axis \"a\" which is already bound by a parent sdy.manual_computation op",
        why: '外层已经绑定 <span class="mono">"a"</span>，内层再绑同一个轴 → 冲突。嵌套时每层要用<b>不同</b>的轴。'
      },
    ], {
      stepMs: 2400,
      finalIr: '// 四类根因：\n//   1. 网格不一致\n//   2. 结构/数量不匹配\n//   3. 局部形状算错（外层÷manual轴 ≠ 体内形状）\n//   4. 轴的角色错（顺序 / 存在性 / 重复绑定）',
      finalErr: 'same mesh / must match / Expected local shape / bound by a parent',
      finalWhy: '第 3 类最值得注意：它一次核对<b>三个</b>信息（外层形状、分片、体内形状），任一不符都报错。'
    });
  }
},

/* ---------------------------------------------------- 7 规范化 */
{
  kicker: 'L1-07 · 规范化',
  title: '规范化：把"没用的手动"<span class="hl-a">消掉</span>',
  sub: '手动计算有开销（阻断传播、可能阻碍优化）。如果它其实什么都没做，规范化会把它内联掉。',
  caption: '这也保证了"用户写的手动计算"不会成为无谓的负担 —— 只在真正需要控制的地方保留。',
  code: `// ① 删除未使用的参数
//    输入用了 3 个参数、只返回其中 1 个
%0 = sdy.manual_computation(%arg0, %arg1, %arg2)
     in_shardings=[<@mesh, [{"a"}]>, <@mesh, [{"a"}, {}]>,
                   <@mesh, [{"a"}]>]
     out_shardings=[<@mesh, [{"a"}, {}]>]
     manual_axes={"a"} (%arg3: tensor<4xf32>, %arg4: tensor<16x32xf32>,
                       %arg5: tensor<8xf32>) {
  sdy.return %arg4 : tensor<16x32xf32>
} : (tensor<8xf32>, tensor<32x32xf32>, tensor<16xf32>) -> tensor<32x32xf32>

//   -> 只剩 (%arg1)，in/out_shardings 也只留对应项

// ② 无输入输出且无 manual 轴 -> 直接内联区域内容
sdy.manual_computation() in_shardings=[] out_shardings=[]
    manual_axes={} () {
  stablehlo.custom_call @foo() {has_side_effect = true} : () -> ()
  sdy.return
} : () -> ()
//   -> 把 custom_call 提到外面，删掉 manual_computation

// ③ 冗余的 manual_axes：某个 manual 轴在所有分片里
//    都只是 replicated -> 该轴不必 manual`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%;align-items:center' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:750px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① 删除未使用参数', d: '<span class="mono">unused_args</span><br>只被传入、从未被使用或返回的参数会被剔除。', c: '#38bdf8' },
      { t: '② 空体直接内联', d: '<span class="mono">inline_no_inputs_outputs_and_no_manual_axes</span><br>无 in/out、无 manual 轴 → 区域内容提到外面。', c: '#4ade80' },
      { t: '③ 删除空体', d: '<span class="mono">erase_no_inputs_outputs_and_empty_body</span><br>彻底空的算子直接删掉。', c: '#c084fc' },
      { t: '④ 冗余 manual 轴', d: '<span class="mono">redundant_manual_axes</span><br>只被 replicated 的 manual 轴可以去掉。', c: '#fbbf24' },
      { t: '⑤ 嵌套全可内联', d: '<span class="mono">nested_two_inlinable</span><br>两层都能内联时，一起消除。', c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:146px;opacity:.32;transition:.3s;border-color:' + x.c + '55;padding:9px' });
      e.innerHTML = `<div class="card-t" style="color:${x.c};font-size:11px">${x.t}</div>
        <div class="card-d" style="font-size:10.5px;line-height:1.45">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 2800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.32');
      msg.innerHTML = [
        '手动计算常被用作"临时占位"：先框住一段，再逐步细化。没用的参数自然会被去掉。',
        '没有 manual 轴 = 没有局部世界 = 这个算子没有存在意义，内容是纯全局的。',
        '连区域内容都没有时，整个算子就是噪声。',
        '如果某个"手动轴"在所有分片里都只是 <span class="mono">replicated</span>，说明它根本没被手动管理。',
        '嵌套是逐层判定的：内层可内联后，外层可能也变得可内联。',
      ][i];
    }));
    tl.at(700 + defs.length * 2800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '规范化都是<b>保持语义</b>的：只是把"没有实际作用的手动"还原成普通代码。';
    });
  }
},

/* ------------------------------------------------------------ 8 练习 */
{
  kicker: 'L1-07 · 练习',
  title: '练一练：<span class="hl-a">算局部形状、判合法性</span>',
  sub: '三道题分别考：局部形状、轴顺序、嵌套约束。',
  caption: '这一课的概念在 L4-10（按指令分区）会再次出现 —— 那里分区器会自动生成 manual_computation。',
  code: `// 题 1：体内块参数是什么形状？
//   网格 <["a"=4, "b"=2]>
//   manual_axes={"a"}
//   外层 tensor<64x32xf32>
//   in_shardings=[<@mesh, [{"a", ?}, {?}]>]

// 题 2：合法吗？
//   manual_axes={"b"}
//   in_shardings=[<@mesh, [{"a", "b"}, {?}]>]

// 题 3：合法吗？
//   外层 manual_axes={"a"}
//   内层 manual_axes={"a", "b"}`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%;align-items:center' });
    root.appendChild(wrap);
    const qs = [
      {
        q: '网格 <span class="mono">&lt;["a"=4, "b"=2]&gt;</span>、<span class="mono">manual_axes={"a"}</span>、外层 <span class="mono">tensor&lt;64x32xf32&gt;</span>，体内块参数是什么形状？',
        a: '<span class="mono">tensor&lt;16x32xf32&gt;</span> &nbsp;<span class="dim">dim0 = 64÷4 = 16；dim1 = 32（无 manual 轴）</span><br>' +
           '<span class="dim">"b" 是自由轴，<b>不影响</b>局部形状。</span>'
      },
      {
        q: '<span class="mono">manual_axes={"b"}</span>，分片写成 <span class="mono">[{"a", "b"}, {?}]</span>，合法吗？',
        a: '<b class="badge bad">非法</b> 自由轴 <span class="mono">"a"</span> 排在 manual 轴 <span class="mono">"b"</span> 前面。' +
           '<br><b>正确写法</b>：<span class="mono">[{"b", "a"}, {?}]</span>。' +
           '<br><span class="dim">报错：must have all manual axes come before free axes</span>'
      },
      {
        q: '外层 <span class="mono">manual_axes={"a"}</span>，内层 <span class="mono">manual_axes={"a", "b"}</span>，合法吗？',
        a: '<b class="badge bad">非法</b> 内层重复绑定了外层已绑定的 <span class="mono">"a"</span>。' +
           '<br><b>正确写法</b>：内层只用 <span class="mono">{"b"}</span>。' +
           '<br><span class="dim">嵌套时每层必须用不同的轴。</span>'
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
