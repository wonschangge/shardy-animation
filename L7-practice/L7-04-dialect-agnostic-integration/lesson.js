/* ==========================================================================
   L7-04 · dialect-agnostic-integration   （P1 · 全部 75 课的最后一课）
   --------------------------------------------------------------------------
   引用：ir/op_interface.td（三个接口）+ ir/constants.h（常量拆分）
         —— 这两个文件不在 transforms/<dir>/test/ 下，不计入 241 个测试 IR
   目标：讲清"怎么让 Shardy 为你的方言工作"。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 三个接口 */
{
  kicker: 'L7-04 · 跨方言集成',
  title: '★ 最后一课：<span class="hl-a">三个接口</span>就是全部扩展点',
  sub: '前面 74 课讲「Shardy 怎么工作」，这一课讲「**怎么让 Shardy 为你的方言工作**」。',
  caption: '这是全部 <b>75 课</b>的收官。',
  code: `// 【本课引用】接口定义（.td / .h）而非测试 IR
//   ir/op_interface.td    三个接口的定义
//   ir/constants.h        常量拆分相关的辅助
// 注意：它们不在 transforms/<dir>/test/ 下，【不计入】241 个测试 IR

// 【★ op_interface.td 定义的三个接口】
//   接口                                  行号   作用
//   Sdy_ShardableDataFlowOpInterface       27    【数据流边】—— 让分片穿过算子传播
//   Sdy_ShardingRuleOpInterface           232    【自定义 sharding rule】
//   Sdy_CollectiveOpInterface             267    【集合通信算子】

// 【★ 接入 Shardy 就是实现其中的一个或多个】
//   算子类型                该实现什么
//   普通计算算子            ShardingRuleOpInterface（或注册规则）
//   区域算子（while/case）  + ShardableDataFlowOpInterface
//   集合通信算子            CollectiveOpInterface
//   逐元素算子              加 Elementwise trait（最轻）

// 【★ 本课的位置】
//   L7-04 是"向外"的一课 ——
//   前面 74 课讲"Shardy 怎么工作"
//   这一课讲"怎么让 Shardy 为你的方言工作"

// 【验收点】
//   能写出一条自定义算子的 sharding rule 并被传播正确使用

// 一句话：
//   三个接口 + 一个 trait，就是全部的扩展点`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'ShardingRuleOpInterface', c: '#38bdf8', d: '定义<b>分片规则</b><br><span class="dim">"我能怎么被切"</span>' },
      { t: 'ShardableDataFlowOpInterface', c: '#4ade80', d: '<b>数据流边</b><br>分片穿过算子<br><span class="dim">区域算子必需</span>' },
      { t: 'CollectiveOpInterface', c: '#fbbf24', d: '<b>通信算子</b><br>统一 <span class="mono">out_sharding</span>' },
      { t: 'Elementwise trait', c: '#c084fc', d: '<b>最轻</b>的方式<br>逐元素算子' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:8.5px;color:${x.c};overflow-wrap:anywhere">${x.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>三个接口 + 一个 trait</b> —— 这就是全部的扩展点。'; });
    tl.at(4400, () => {
      msg.innerHTML = '<b>怎么选</b>：普通算子实现规则接口；区域算子<b>还要</b>数据流接口；逐元素算子最轻。';
    });
    tl.at(7800, () => {
      msg.innerHTML = '<b>★ 本课是"向外"的一课</b>：前面讲 Shardy 怎么工作，这课讲怎么让 Shardy 为你的方言工作。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<b>验收点</b>：能写出一条自定义算子的 sharding rule 并被传播正确使用。';
    });
  }
},

/* ------------------------------------------------ 2 ★ 规则接口 */
{
  kicker: 'L7-04 · 跨方言集成',
  title: '★ <span class="mono hl-a">ShardingRuleOpInterface</span>：定义自己的规则',
  sub: '核心方法只有一个：**`getShardingRule()`**。',
  caption: '回顾 <b>L7-03</b> 的那条警告 —— 它就是"既没注册、也没实现接口"的结果。',
  code: `def Sdy_ShardingRuleOpInterface : OpInterface<"ShardingRuleOpInterface"> {
  let description = [{
    An op interface that allows the op to define its own sharding rule.
    A sharding rule specifies how an operation can be partitioned according to
    various properties on the op - any attributes, the shape of operands,
    the shape of the results, etc. See \`OpShardingRuleAttr\` for more
    details.
  }];
  let cppNamespace = "::mlir::sdy";
    let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the sharding rule of the op.
      }],
      /*retType=*/"mlir::sdy::OpShardingRuleAttr",
      /*methodName=*/"getShardingRule"
    >,

// 【读法】本课的核心
//   "allows the op to define its own sharding rule"
//     —— 算子可以【自己声明】"我能怎么被切分"
//   规则的依据：算子的属性、操作数的形状、结果的形状 等
//   核心方法：getShardingRule() —— 返回一个 OpShardingRuleAttr
//
// 【★ 回顾 L2-10】那里讲 op_sharding_rule 的【语法】
//   #sdy.op_sharding_rule<([i, k, j], [i, l, j])->([i, m, j]) {i=4, j=256, k=1, l=1, m=1}>
//   本课讲的是"【怎么生成这条规则】"—— 实现这个接口
//
// 【★ 两种生成方式】
//   方式                              适用
//   【注册规则】用【声明式】的规则表
//     （见 -sdy-populate-op-sharding-rules）
//   【实现接口】（本课）              规则的【逻辑复杂】、依赖运行时信息
//
// 【★ 回顾 L7-03 的那条警告】
//   "custom call @unknown_custom_op is unknown to SDY sharding rule registry"
//   就是"【既没注册、也没实现接口】"的结果
//   -> 本课就是那条警告的【解决方案】`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:13px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:stretch;justify-content:center" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '注册规则', c: '#38bdf8', d: '<b>声明式</b>规则表<br><span class="mono">-sdy-populate-op-sharding-rules</span><br><span class="dim">规则简单时用</span>' },
      { t: '实现接口', c: '#4ade80', d: '<b><span class="mono">getShardingRule()</span></b><br>逻辑复杂时用<br><span class="dim">依赖运行时信息</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: 'width:370px;opacity:.35;transition:.35s;border-color:' + x.c + '55' });
      e.innerHTML = `<div class="card-t mono" style="color:${x.c};font-size:10.5px;overflow-wrap:anywhere">${x.t}</div>
        <div class="card-d" style="font-size:11.5px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = '<b>注册规则</b>：用声明式的规则表 —— 规则简单时最省事。'; });
    tl.at(4600, () => {
      els[1].style.opacity = '1';
      msg.innerHTML = '<b>实现接口</b>：核心方法 <span class="mono">getShardingRule()</span> —— 规则依赖运行时信息时用它。';
    });
    tl.at(8600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 本课是 L7-03 那条警告的解决方案</b>：<b>没注册 + 没实现接口</b> = 那条警告。';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<b>回顾 L2-10</b>：那里讲规则的<b>语法</b>，本课讲<b>怎么生成</b>它。';
    });
  }
},

/* ------------------------------------------------ 3 ★ 数据流接口 */
{
  kicker: 'L7-04 · 跨方言集成',
  title: '★ <span class="mono hl-a">ShardableDataFlowOpInterface</span>：数据流边',
  sub: '定义「哪些东西**必须分片一致**」—— 区域算子的必需接口。',
  caption: '这段定义解释了<b>为什么区域算子需要它</b>。',
  code: `def Sdy_ShardableDataFlowOpInterface : OpInterface<"ShardableDataFlowOpInterface"> {
  let description = [{
    An op interface that allows shardy to propagate shardings through data flow
    edges of ops that extend this interface.

    A data flow edge of some op X defines a bridge between a set of sources
    (each is either an operand of X or an operand of X's block terminator) and
    a set of targets (each is either a result of X or a block argument of X),
    such that all sources and targets should be sharded in the same way.
    An op can have multiple data flow edges that are orthogonal to one another.

    An owner is a user specified target of the data flow edge used by shardy's
    propagation. The user can choose it arbitrarily but it needs to be static.

// 【读法】这段定义非常关键
//   "propagate shardings through data flow edges"
//     —— 让分片【穿过算子】传播
//
//   【数据流边（data flow edge）的定义】：
//     一端是 sources：算子 X 的【操作数】或【块终结符的操作数】
//     另一端是 targets：算子 X 的【结果】或【块参数】
//     要求：所有 sources 和 targets 【必须用同样的方式分片】
//   一个算子可以有【多个正交的】数据流边
//
//   【owner】：用户指定的 target
//     "can choose it arbitrarily but it needs to be static"
//       —— 可以任选，但必须是【静态的】
//
// 【★ 为什么需要这个接口】
//   回顾 L2-07：data-flow edges 描述"分片怎么从一个算子流到另一个算子"
//   对于【区域算子】（while/case，L7-03 讲的 40 个用例）
//     数据流要【穿过 region 边界】—— 通用规则【推导不出来】
//     必须由算子【自己告诉 Shardy】"哪些东西应该分片一致"
//
// 【★ sources / targets 的四种组合】
//   端          可以是
//   source      X 的操作数 / X 的【块终结符的操作数】
//   target      X 的结果   / X 的【块参数】
//
//   "块终结符的操作数"与"块参数"正是【区域算子】特有的
//     -> 这就是为什么区域算子需要这个接口`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'source', c: '#38bdf8', d: 'X 的<b>操作数</b><br>或 X 的<br><b>块终结符的操作数</b>' },
      { t: 'target', c: '#4ade80', d: 'X 的<b>结果</b><br>或 X 的<br><b>块参数</b>' },
      { t: 'owner', c: '#fbbf24', d: '用户指定的 target<br><b>必须静态</b>' },
      { t: '约束', c: '#fb7185', d: '所有 sources 与 targets<br><b>分片方式相同</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:10.5px;color:${x.c}">${x.t}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>source</b>：可以是操作数，也可以是<b>块终结符的操作数</b>（区域算子特有）。',
        '<b>target</b>：可以是结果，也可以是<b>块参数</b>（区域算子特有）。',
        '<b>owner</b>：用户指定的 target —— <b>可以任选，但必须静态</b>。',
        '<b>核心约束</b>：一条边上的所有 sources 与 targets 必须<b>分片方式相同</b>。',
      ][i];
    }));
    tl.at(15200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 为什么区域算子需要它</b>：数据流穿过 region 边界，<b>通用规则推导不出来</b>。';
    });
  }
},

/* ------------------------------------------------ 4 通信接口与 trait */
{
  kicker: 'L7-04 · 跨方言集成',
  title: '<span class="mono hl-a">CollectiveOpInterface</span> 与 <span class="hl-a">Elementwise trait</span>',
  sub: '前者统一 `out_sharding`，后者是**最轻**的接入方式。',
  caption: '四条约束很实用 —— 它们解释了 L5-02 见过的那些语法要求。',
  code: `def Sdy_CollectiveOpInterface : OpInterface<"CollectiveOpInterface"> {
  let description = [{
    Interface for all collective ops. Encapsulates common get/set for
    outSharding attribute.

    **Constraints:**
    - Operand must have a sharding or \`allowMissingInputSharding()\` returns
      true.
    - \`out_sharding\` is valid w.r.t the corresponding type.
    - Operand and result sharding must have the same mesh if
      \`allowDifferentMeshes()\` returns false.
    - Same rank for the operand and result sharding.
  }];

// 【读法】
//   "Encapsulates common get/set for outSharding attribute"
//     —— 统一管理 out_sharding（L5-02 反复强调的"out_sharding 是派生的"）
//
//   【四条约束】（很实用）
//     ① 操作数【必须有分片】（除非 allowMissingInputSharding() 返回 true）
//     ② out_sharding 必须与类型【匹配】
//     ③ 操作数与结果的 mesh 【必须相同】（除非 allowDifferentMeshes()）
//     ④ 操作数与结果的分片【秩必须相同】
//
// 【★ 回顾 L5-02】那里看到 8 种集合通信算子的 out_sharding 语法
//   本课看到它背后的【接口约束】
//
// 【★ Elementwise trait：最轻的接入方式】
//   逐元素算子（如 add、negate）的分片规则【完全一样】：
//     所有操作数与结果用【同样的分片】
//
//   为什么可以用 trait 而不是接口：
//     逐元素算子的规则【不需要任何计算】—— 它是【固定的】
//     用 trait（编译期标记）比实现接口（运行期调用）【更轻】
//
//   回顾 L5-04：那里讲 negate 是"逐元素算子，分片后无需通信"
//     这正是 Elementwise trait 的效果
//
// 【★ 常量拆分】（ir/constants.h）
//   一个【全局常量】（如 dense<[[1,2],[3,4]]>）分片后
//     每台设备需要【不同的切片】
//   "常量拆分"就是把常量按分片拆成每台设备的那一份
//   两种常量（L5-03 讲过）：
//     splat（dense<1.0>）  -> 【不需要拆分】（内容处处相同）
//     dense（元素各异）    -> 用 replica_id + 查找表 + 切片`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '必须有分片', c: '#38bdf8', d: '操作数<br>除非 <span class="mono">allowMissing-<br>InputSharding()</span>' },
      { t: '类型匹配', c: '#4ade80', d: '<span class="mono">out_sharding</span><br>必须与类型<b>匹配</b>' },
      { t: 'mesh 相同', c: '#fbbf24', d: '操作数与结果<br><b>mesh 必须相同</b><br><span class="dim">除非 allowDifferentMeshes</span>' },
      { t: '秩相同', c: '#c084fc', d: '操作数与结果<br>分片<b>秩相同</b>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div style="font-size:10.5px;color:${x.c}">${x.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.4;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>为什么要求必须有分片</b>：通信算子的输入必须有分片信息，否则无法确定通信方式。',
        '<b>为什么要求类型匹配</b>：<span class="mono">out_sharding</span> 的秩要与结果类型一致。',
        '<b>为什么要求 mesh 相同</b>：跨 mesh 的通信需要显式声明（<span class="mono">allowDifferentMeshes</span>）。',
        '<b>为什么要求秩相同</b>：通信不改变秩 —— 它只搬运数据。',
      ][i];
    }));
    tl.at(15000, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 最轻的方式</b>：逐元素算子只需加 <b>Elementwise trait</b> —— 规则固定，不需要计算。';
    });
  }
},

/* ------------------------------------------------ 5 ★ 接入步骤 */
{
  kicker: 'L7-04 · 跨方言集成',
  title: '★ <span class="hl-a">接入步骤</span>与验收点',
  sub: '验收点：能写出一条自定义算子的 sharding rule 并被传播正确使用。',
  caption: '规则写得对不对，<b>最终要靠数值验证</b>（L6-00 的做法）。',
  code: `// 【★ 三步接入】
//   步   做什么                        用什么
//   ①    让算子【能定义分片规则】      实现 ShardingRuleOpInterface 或【注册规则】
//   ②    让分片【能穿过算子】          实现 ShardableDataFlowOpInterface（区域算子必需）
//   ③    若是【逐元素】算子            加 Elementwise trait（最轻）

// 【一个最小例子：sharding rule 的写法】
// 回顾 L2-10 的语法，为一个假想的 my_op 写规则：
//   #sdy.op_sharding_rule<([i, j], [j, k])->([i, k]) {i=8, j=16, k=8}>
//
// 【逐项读】
//   ([i, j], [j, k])->([i, k])   两个操作数 [i,j]/[j,k]，结果 [i,k]
//   {i=8, j=16, k=8}             因子的【大小】
//   【j 出现在两个操作数里但不在结果里】
//     -> j 是【归约因子】（L2-10 的规则）
//     -> 如果 j 被分片，就需要 all_reduce（L5-05 的判据）

// 【★ 验证接入是否正确】
//   检查                命令
//   规则有没有生效      sdy_opt -sdy-populate-op-sharding-rules -verify-diagnostics
//                       （L7-03）
//   传播对不对          sdy_opt -sdy-propagation-pipeline 看结果
//   区域算子            参考 data_flow_edges 的 40 个用例（L7-03）
//   数值对不对          写一个【可执行测试】（L6-00 的三要素）

// 【★ 最后一条最重要】
//   规则写得对不对，【最终要靠数值验证】（L6-00 讲的"分片版 vs 串行版"）
//   这与 L6 全层的做法一致
//
// 【★ 一个完整的接入清单】
//   ☐ 算子需要被分片吗？        -> 不需要就不用做任何事
//   ☐ 是逐元素算子吗？          -> 加 Elementwise trait
//   ☐ 是集合通信算子吗？        -> 实现 CollectiveOpInterface
//   ☐ 是区域算子吗？            -> 实现 ShardableDataFlowOpInterface
//   ☐ 规则能声明式表达吗？      -> 注册规则
//   ☐ 规则依赖运行时信息吗？    -> 实现 ShardingRuleOpInterface
//   ☐ 写一个可执行测试验证数值`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const steps = [
      { n: '①', t: '定义规则', c: '#38bdf8', d: '接口或注册' },
      { n: '②', t: '穿过算子', c: '#4ade80', d: '数据流接口' },
      { n: '③', t: '逐元素', c: '#fbbf24', d: '加 trait' },
      { n: '④', t: '验证数值', c: '#fb7185', d: '可执行测试' },
    ];
    const host = wrap.querySelector('#cards');
    const els = steps.map(s => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${s.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="small mono" style="font-size:10.5px;color:${s.c}">${s.n}</div>
        <div style="font-size:11px;margin-top:3px">${s.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35;margin-top:3px">${s.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    steps.forEach((s, i) => tl.at(700 + i * 3800, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '<b>第一步</b>：让算子能定义规则 —— 规则简单就注册，依赖运行时信息就实现接口。',
        '<b>第二步</b>：区域算子必须实现数据流接口，否则分片穿不过 region 边界。',
        '<b>第三步</b>：逐元素算子只需加 trait —— 最轻。',
        '<b>第四步最重要</b>：写可执行测试验证<b>数值</b>（L6-00 的做法）。',
      ][i];
    }));
    tl.at(16200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>★ 关键洞察</b>：<span class="mono">j</span> 出现在两个操作数但不在结果里 → 它是<b>归约因子</b> → 分片就要 <span class="mono">all_reduce</span>。';
    });
  }
},

/* ------------------------------------------------ 6 ★ 全部 75 课回顾 */
{
  kicker: 'L7-04 · 跨方言集成',
  title: '★ <span class="hl-a">全部 75 课</span>回顾与三条主线',
  sub: '七层、241 个测试文件、75 课 —— 到这里全部结束。',
  caption: '这是最后一课。把七层串起来。',
  code: `// 【★ 七层的核心问题】
//   层    主题              核心问题
//   L1    SDY 方言基础      分片【怎么表达】？
//   L2    分片传播          分片【怎么流动】？
//   L3    导入              别的方言【怎么进来】？
//   L4    导出（reshard）   分片【怎么变成通信】？
//   L5    导出（降级）      全局张量【怎么变成局部】？
//   L6    执行与解释器      分片代码【怎么跑起来】？
//   L7    综合实战          【怎么用、怎么排错、怎么扩展】？

// 【★ 三条贯穿全课的主线】
//   ① 只有【归约方向】上的分片需要通信
//      L5-05 收缩维 / L5-06 归约因子 / L5-07 归约维 /
//      L5-08 collapsed 维 / L6 全层验证
//   ② 【位置相关】的算子要补偿位置
//      L5-04 iota 偏移 / L5-08 索引重映射 /
//      L6-05 填充值 = 单位元
//   ③ 【不可整除】时让通信发生在可整除的形状上
//      L5-09 pad 补齐 + slice 裁回 / L6-06 的 3->4

// 【★ 规模】
//   层    课数   测试文件数
//   intro  1     —（导览）
//   L1    10     28
//   L2    12     18
//   L3    11     23
//   L4    17     60
//   L5     9     60
//   L6    10     74
//   L7     4      8（其中 4 个为 L2 漂移补齐）
//   ────────────────────────
//   合计  74+1   241（100%）

// 【★ 本课的位置】
//   L7-04 是"向外"的一课 ——
//   前面 74 课讲"Shardy 怎么工作"
//   这一课讲"怎么让 Shardy 为你的方言工作"
//   三个接口 + 一个 trait，就是全部的扩展点

// 一句话总结：
//   接入 Shardy 就是实现三个接口之一
//   ShardingRuleOpInterface（定义规则）
//   ShardableDataFlowOpInterface（穿过算子）
//   CollectiveOpInterface（通信算子）
//   逐元素算子加 Elementwise trait 即可，常量走常量拆分`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:8px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: 'intro', n: 1, c: '#94a3b8' }, { t: 'L1 基础', n: 10, c: '#38bdf8' },
      { t: 'L2 传播', n: 12, c: '#0ea5e9' }, { t: 'L3 导入', n: 11, c: '#22c55e' },
      { t: 'L4 导出', n: 17, c: '#4ade80' }, { t: 'L5 降级', n: 9, c: '#fbbf24' },
      { t: 'L6 执行', n: 10, c: '#c084fc' }, { t: 'L7 实战', n: 4, c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:104px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:7px;text-align:center` });
      e.innerHTML = `<div style="font-size:9px;color:${f.c};line-height:1.3">${f.t}</div>
        <div class="big" style="font-size:15px;color:${f.c};margin-top:2px">${f.n}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 90)); msg.innerHTML = '<b>75 课</b>（含导览）覆盖 <b>241 个测试文件（100%）</b>。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>★ 三条主线</b>：只有归约方向要通信 / 位置相关要补偿 / 不可整除要补齐。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>★ 本课是"向外"的一课</b>：前面讲 Shardy 怎么工作，这课讲怎么让 Shardy 为你的方言工作。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<b>全部结束</b>：三个接口 + 一个 trait，就是全部的扩展点。';
    });
  }
},

];
