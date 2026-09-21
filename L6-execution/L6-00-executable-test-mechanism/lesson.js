/* ==========================================================================
   L6-00 · executable-test-mechanism   （P0 · L6 开篇）
   --------------------------------------------------------------------------
   覆盖：executable_convert_global_to_local/run_sdy_interpreter_test.sh
         executable_partitioner_pipeline/run_sdy_interpreter_test.sh
         （机制说明，非 IR —— 这两个是 shell 脚本，不计入 241 个 .mlir）
   目标：讲透"可执行测试"是怎么工作的。
   排版基准：#visual 可视区 = 780 x 521 舞台像素。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------------ 1 为什么需要 */
{
  kicker: 'L6-00 · 可执行测试机制',
  title: 'L6 开篇：<span class="hl-a">为什么 Shardy 的测试能"跑出数值"</span>',
  sub: 'L4/L5 的许多测试用 `// RUN: %S/run_sdy_interpreter_test.sh %s %t` —— 它们**不是比对 IR 文本**。',
  caption: '本课<b>不讲 IR，讲机制</b> —— 不理解它，就看不懂那一大批 <span class="mono">executable_*</span> 测试。',
  code: `// 【本课覆盖】2 个 shell 脚本（机制说明，非 IR）
//   executable_convert_global_to_local/run_sdy_interpreter_test.sh
//   executable_partitioner_pipeline/run_sdy_interpreter_test.sh
// 注意：它们不是 .mlir，【不计入】241 个测试 IR

// 【为什么需要这一课】
//   L3-11、L4-01 等课见过这样的 RUN 行：
//     // RUN: %S/run_sdy_interpreter_test.sh %s %t
//   这些测试【不比对 IR 文本】，而是【真的执行】
//   不理解这个机制，就看不懂 executable_* 那一大批测试

// 【★ 验证思路的完整链条】
//   part1.mlir（分片版）
//     --导出流水线--> 局部代码（@parallel_x）
//     --drop-sharding--> 串行代码（@sequential_x）
//                       ↓
//                 part2.mlir 的 main 同时调用两者、比较结果
//                       ↓
//                 --interpret 真正执行 -> 数值一致 = 语义保持 ✓

// 【为什么必须执行】（回顾 L3-11 / L4-01）
//   这些 pass 改写的是算子的【语义属性】（分片、通信、形状）
//   改错了 IR 【依然能通过校验】，但【跑出来的数值会错】
//   只有真正执行一遍才能确认语义保持

// 一句话：
//   Shardy 的可执行测试 = "分片版 vs 串行版"的【结果对比】`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: 'part1.mlir', c: '#38bdf8', d: '分片版<br><span class="mono">@parallel_x</span>' },
      { t: '导出流水线', c: '#fbbf24', d: 'L4 + L5<br>的全部 pass' },
      { t: '串行版', c: '#4ade80', d: '<span class="mono">@sequential_x</span><br>去分片 + 改名' },
      { t: 'part2.mlir', c: '#c084fc', d: '<span class="mono">main</span> 同时调用<br>两者并比较' },
      { t: '--interpret', c: '#fb7185', d: '<b>真正执行</b><br>数值一致 ✓' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:140px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:10px;color:${x.c};overflow-wrap:anywhere">${x.t}</div>
        <div class="small faint" style="font-size:9.5px;line-height:1.35;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>五个环节</b> —— 从分片版到"真正执行"的完整链条。'; });
    tl.at(4400, () => {
      msg.innerHTML = '<b>关键</b>：脚本会<b>自动生成</b>串行参考版 —— 从分片版去分片 + 改名。';
    });
    tl.at(7800, () => {
      msg.innerHTML = '<b>为什么必须执行</b>：改错语义属性时 IR <b>依然能通过校验</b>，但<b>数值会错</b>。';
    });
    tl.at(11000, () => {
      msg.innerHTML = '<b>一句话</b>：Shardy 的可执行测试 = "分片版 vs 串行版"的<b>结果对比</b>。';
    });
  }
},

/* ------------------------------------------------ 2 四步流程 */
{
  kicker: 'L6-00 · 可执行测试机制',
  title: '★ 脚本的<span class="hl-a">四步流程</span>',
  sub: '`split-file` 拆分 → `sdy_opt` 跑流水线 → `sed` 提取函数体 → `--interpret` 执行。',
  caption: '注意那个 <span class="mono">sed</span> —— 它用两行命令<b>提取函数体</b>，很巧妙。',
  code: `set -e
SRC=""; TMP=""
# When replica_count=partition_count=1, the test uses partition id.
REPLICA_COUNT=1
PARTITION_COUNT=1

for arg in "$@"; do
  case "$arg" in
    --src=*)                  SRC="\${arg#*=}" ;;
    --temp_dir=*)             TMP="\${arg#*=}" ;;
    --replica_count=*)        REPLICA_COUNT="\${arg#*=}" ;;
    --partition_count=*)      PARTITION_COUNT="\${arg#*=}" ;;
    --*)                      echo "Warning: Unknown flag '$arg'" >&2 ;;
    *)
      pos_idx=$((pos_idx + 1))
      case "$pos_idx" in
        1) SRC="$arg" ;;
        2) TMP="$arg" ;;
        3) REPLICA_COUNT="$arg" ;;
        4) PARTITION_COUNT="$arg" ;;
      esac ;;
  esac
done

SPLIT_FILE=\${SPLIT_FILE:-split-file}
SDY_OPT=\${SDY_OPT:-sdy_opt}
STABLEHLO_TRANSLATE=\${STABLEHLO_TRANSLATE:-stablehlo-translate}

"$SPLIT_FILE" "$SRC" "$TMP"
"$SDY_OPT" "$TMP/part1.mlir" \\
  --sdy-convert-global-to-local="replica-count=$REPLICA_COUNT partition-count=$PARTITION_COUNT" \\
  --sdy-inline-meshes \\
  --sdy-drop-sharding-and-mesh \\
  --allow-unregistered-dialect > "$TMP/part1_processed.mlir"
sed '1d; /^}/,$d' "$TMP/part1_processed.mlir" > "$TMP/combined.mlir"

// 【参数解析】
//   支持两种传参方式：--src=... 命名参数，或【位置参数】
//   RUN 行的 %s %t 用的是【位置参数】
//     %s = 源文件、%t = 临时目录（lit 的内置变量）
//   默认 REPLICA_COUNT=1 PARTITION_COUNT=1
//     注释："When replica_count=partition_count=1, the test uses partition id"
//     （单设备时用 partition_id 而非 replica_id —— 与 L5-01 呼应）

// 【工具可覆盖】
//   三个工具都用 \${VAR:-default} 形式 -> 可以用环境变量覆盖

// 【四步流程】
//   ① split-file 按 "//--- partN.mlir" 标记【拆分】源文件
//   ② sdy_opt 跑【导出流水线】（convert-global-to-local + 清理）
//   ③ sed '1d; /^}/,$d' 【提取函数体】
//   ④ 后面再接 stablehlo-translate --interpret

// 【★ 第 ③ 步的 sed 值得细看】
//   1d            删掉第 1 行
//   /^}/,$d       从第一个 } 开始【删到文件尾】
//   为什么能提取函数体：sdy_opt 输出是
//     module {
//       func.func @parallel_x(...) {
//         ...函数体...
//       }
//     }
//   删掉 module { 与末尾的两个 } -> 只剩函数体 ✓`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const steps = [
      { n: '①', t: 'split-file', c: '#38bdf8', d: '按 <span class="mono">//---</span> 拆分' },
      { n: '②', t: 'sdy_opt', c: '#fbbf24', d: '跑导出流水线' },
      { n: '③', t: 'sed', c: '#4ade80', d: '<b>提取函数体</b>' },
      { n: '④', t: '--interpret', c: '#fb7185', d: '真正执行' },
    ];
    const host = wrap.querySelector('#cards');
    const els = steps.map(s => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${s.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div class="small mono" style="font-size:11px;color:${s.c}">${s.n}</div>
        <div class="mono" style="font-size:11px;margin-top:3px">${s.t}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4;margin-top:3px">${s.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    steps.forEach((s, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '把源文件按 <span class="mono">//--- partN.mlir</span> 标记拆成多个文件。',
        '对 <span class="mono">part1.mlir</span> 跑导出流水线 —— <span class="mono">convert-global-to-local</span> + 清理。',
        '<b>很巧妙</b>：<span class="mono">sed \'1d; /^}/,$d\'</span> 两行就提取出函数体。',
        '把 <span class="mono">part2.mlir</span> 追加后<b>真正执行</b>。',
      ][i];
    }));
    tl.at(15000, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>参数细节</b>：默认 <span class="mono">replica_count=partition_count=1</span> → 单设备时用 <span class="mono">partition_id</span>（与 L5-01 呼应）。';
    });
  }
},

/* ------------------------------------------------ 3 ★ 串行参考实现 */
{
  kicker: 'L6-00 · 可执行测试机制',
  title: '★ 最关键的机制：<span class="hl-a">自动生成串行参考版</span>',
  sub: '脚本把分片版**去掉分片 + 改名**，得到语义等价的串行实现。',
  caption: '这是"为什么能验证正确性"的<b>核心答案</b>。',
  code: `# If part1.mlir contains @parallel_x but not @sequential_x, then remove sharding
# from @parallel_x and rename it to @sequential_x.
if (grep -q "@parallel_" "$TMP/part1.mlir") && (! grep -q "@sequential_" "$TMP/part1.mlir"); then
  "$SDY_OPT" "$TMP/part1.mlir" --sdy-drop-sharding-and-mesh --allow-unregistered-dialect | \\
  sed 's/parallel_/sequential_/g' > "$TMP/part1_sequential.mlir"
  sed '1d; /^}/,$d' "$TMP/part1_sequential.mlir" >> "$TMP/combined.mlir"
fi

cat "$TMP/part2.mlir" >> "$TMP/combined.mlir"
"$STABLEHLO_TRANSLATE" --interpret "$TMP/combined.mlir"

// 【这一步做了什么】（本课最关键）
//   如果 part1.mlir 里有 @parallel_ 但【没有】@sequential_：
//     ① 把 part1.mlir 跑一遍 --sdy-drop-sharding-and-mesh
//        -> 【去掉所有分片信息】-> 得到【串行版本】
//     ② sed 's/parallel_/sequential_/g' -> 【改名】
//        @parallel_x -> @sequential_x
//     ③ 提取函数体，【追加】到 combined.mlir

// 【为什么这就是"参考实现"】
//   @parallel_x   是【分片版】（经导出流水线 -> 局部代码）
//   @sequential_x 是【同一个函数去掉分片】-> 【串行计算】
//   两者【语义应该等价】—— 只是并行方式不同
//   -> 用串行版当作"标准答案"，比对分片版的结果

// 【★ 为什么 drop-sharding 就能得到串行版】
//   --sdy-drop-sharding-and-mesh 会删掉 sdy.mesh 与所有 sdy.sharding（L4-15 讲过）
//   删掉分片后，IR 变成【纯 StableHLO】—— 即"单设备上直接算"
//   这正是【串行语义】的定义

// 【一个细节：if 条件】
//   条件要求"有 parallel 且【没有】sequential"
//   -> 如果 part1.mlir 里【已经】有 @sequential_，脚本就【不会】自动生成
//   -> 这是给"手写参考实现"留的口子

// 【最后两步】
//   cat part2.mlir >> combined.mlir
//     part2.mlir 里有一个 main，【同时调用】@parallel_x 与 @sequential_x 并比较
//   stablehlo-translate --interpret combined.mlir
//     【真正执行】`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:14px;width:100%;align-items:center' });
    wrap.innerHTML = `
      <div class="row" style="gap:16px;justify-content:center;align-items:center" id="demo"></div>
      <div class="formula" id="msg" style="min-height:56px;display:flex;align-items:center;text-align:center;max-width:770px"></div>`;
    root.appendChild(wrap);
    const demo = wrap.querySelector('#demo'), msg = wrap.querySelector('#msg');

    const node = (label, sub, color) => {
      const c = U.el('div', { class: 'col', style: 'gap:5px;align-items:center;opacity:0;transition:.45s' });
      c.innerHTML = `<div class="chip ${color}" style="padding:7px 12px;font-size:10.5px">${label}</div>
        <div class="small faint" style="font-size:9.5px;text-align:center;max-width:150px;line-height:1.35">${sub}</div>`;
      demo.appendChild(c); return c;
    };

    tl.at(700, () => {
      node('@parallel_x', '分片版<br>经导出流水线 → 局部代码', 'c0').style.opacity = '1';
      msg.innerHTML = '<b>起点</b>：<span class="mono">part1.mlir</span> 里的分片版函数。';
    });
    tl.at(4200, () => {
      demo.insertAdjacentHTML('beforeend', '<div class="arrow" style="font-size:15px">→</div>');
      node('drop-sharding', '删掉 mesh 与<br>所有 sharding', 'c2').style.opacity = '1';
      msg.innerHTML = '<b>关键动作</b>：<span class="mono">--sdy-drop-sharding-and-mesh</span> 把分片全删掉（L4-15 讲过）。';
    });
    tl.at(8200, () => {
      demo.insertAdjacentHTML('beforeend', '<div class="arrow" style="font-size:15px">→</div>');
      node('@sequential_x', '串行参考版<br>改名后追加', 'c4').style.opacity = '1';
      msg.innerHTML = '<b>得到串行版</b>：删掉分片后就是"单设备上直接算"—— 即<b>串行语义</b>。';
    });
    tl.at(12400, () => {
      msg.innerHTML = '<b>为什么它是"标准答案"</b>：分片版与串行版<b>语义应该等价</b> —— 只是并行方式不同。';
    });
    tl.at(15200, () => {
      msg.innerHTML = '<b>最后</b>：<span class="mono">main</span> 同时调用两者、比较结果，<span class="mono">--interpret</span> 真正执行。';
    });
  }
},

/* ------------------------------------------------ 4 ★ 完整流水线 */
{
  kicker: 'L6-00 · 可执行测试机制',
  title: '★ 第二个脚本：<span class="hl-a">完整分区器流水线</span>',
  sub: '它把 **L4 + L5 的所有 pass** 串成了一条真实可跑的命令。',
  caption: '本课最有价值的一处 —— 它是前面 26 课的<b>集成视图</b>。',
  code: `ENABLE_HALO_EXCHANGE=true
# When replica_count=partition_count=1, the test use partition id.

# Run the partitioner pipeline passes.
  --sdy-insert-explicit-reshards="enable-full-version=true mark-partial-result-with-unreduced-axes=true" \\
  --sdy-resolve-permutation-factors="enable-halo-exchange=$ENABLE_HALO_EXCHANGE replica-count=$REPLICA_COUNT partition-count=$PARTITION_COUNT" \\
  --sdy-reshard-to-collectives \\
  --sdy-optimize-collectives \\
  --sdy-pad-for-divisibility \\
  --sdy-resolve-single-device-sharding="replica-count=$REPLICA_COUNT partition-count=$PARTITION_COUNT" \\

// 【★ 逐行对应到课】（本课最有价值的对照）
//   pass                                        对应课
//   --sdy-insert-explicit-reshards              L4-02 ~ L4-07（reshard 插入总纲与算子族）
//   --sdy-resolve-permutation-factors           L4-09（置换因子消解）
//   --sdy-reshard-to-collectives                L4-08（reshard 转集合通信）
//   --sdy-optimize-collectives                  L4-10（通信优化）
//   --sdy-pad-for-divisibility                  L5-09（为整除性补齐）
//   --sdy-resolve-single-device-sharding        L4-14（单设备分片）

// 【这就是 L4 + L5 的完整流水线】
//   把 17 + 9 = 26 课讲的 pass 串成了一条【真实可跑的命令】

// 【注意选项的传递】—— 全部在前面的课里出现过
//   enable-full-version=true                    L4-02 见过
//   mark-partial-result-with-unreduced-axes     L4-05 见过
//   enable-halo-exchange=...                    L4-09 见过
//   replica-count / partition-count             L5-01 见过
//   -> 本课是它们的【集成视图】

// 【与第一个脚本的区别】
//   第一个脚本：只跑 --sdy-convert-global-to-local（导出流水线的一部分）
//   第二个脚本：跑【完整的分区器流水线】（从 reshard 插入到整除性补齐）
//   第二个更接近"真实编译流程"

// 【ENABLE_HALO_EXCHANGE 默认 true】
//   对应 L4-09 的 enable-halo-exchange 选项
//   -> 默认用 halo exchange 而非全复制`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:8px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const passes = [
      { p: 'insert-explicit-reshards', k: 'L4-02~07', c: '#38bdf8' },
      { p: 'resolve-permutation-factors', k: 'L4-09', c: '#0ea5e9' },
      { p: 'reshard-to-collectives', k: 'L4-08', c: '#4ade80' },
      { p: 'optimize-collectives', k: 'L4-10', c: '#22c55e' },
      { p: 'pad-for-divisibility', k: 'L5-09', c: '#fbbf24' },
      { p: 'resolve-single-device-sharding', k: 'L4-14', c: '#fb7185' },
    ];
    const host = wrap.querySelector('#cards');
    const els = passes.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:8px;text-align:center` });
      e.innerHTML = `<div class="mono" style="font-size:9px;color:${x.c};overflow-wrap:anywhere">${x.p}</div>
        <div style="font-size:10px;color:${x.c};margin-top:3px">${x.k}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 110)); msg.innerHTML = '<b>六个 pass</b> —— 每个都能对应到前面某一课。'; });
    tl.at(4400, () => {
      msg.innerHTML = '<b>★ 这就是 L4 + L5 的完整流水线</b>：把 26 课讲的 pass 串成了一条真实可跑的命令。';
    });
    tl.at(7800, () => {
      msg.innerHTML = '<b>选项也都在前面出现过</b>：<span class="mono">enable-full-version</span>（L4-02）、<span class="mono">mark-partial-result</span>（L4-05）、<span class="mono">enable-halo-exchange</span>（L4-09）。';
    });
    tl.at(11200, () => {
      msg.innerHTML = '<b>两个脚本的区别</b>：第一个只跑导出流水线；第二个跑<b>完整分区器流水线</b>，更接近真实编译流程。';
    });
  }
},

/* ------------------------------------------------ 5 自己写一个 */
{
  kicker: 'L6-00 · 可执行测试机制',
  title: '如何<span class="hl-a">自己写一个</span>可执行测试',
  sub: '验收点：能独立写出一个最小可执行测试并通过。',
  caption: '三个必要元素 + 一个命名约定。',
  code: `// 【模板】三个必要元素
//
// // RUN: %S/run_sdy_interpreter_test.sh %s %t
//
// //--- part1.mlir
//
// func.func @parallel_<名字>(... 带 sdy.sharding ...) -> ... {
//   ...算子...
// }
//
// //--- part2.mlir
//
// func.func @main() {
//   // 构造输入、调用 @parallel_<名字> 与 @sequential_<名字>、比较结果
// }

// 【三个必要元素】
//   ① // RUN: 行   用 %S/run_sdy_interpreter_test.sh %s %t
//   ② part1.mlir  含 @parallel_* 函数（【带分片】），至少一个
//   ③ part2.mlir  含 @main，【同时调用】parallel 与 sequential 版本并比较

// 【★ 命名约定（关键）】
//   分片版函数名必须含 【@parallel_】 前缀
//   脚本会【自动生成】@sequential_ 版本（去分片 + 改名）
//   所以 part2.mlir 里要调用【两个】名字：
//     @parallel_x  与  @sequential_x

// 【注意】如果 part1.mlir 里【已经】有 @sequential_
//   脚本就【不会】自动生成（if 条件要求"有 parallel 且没有 sequential"）
//   这是给"手写参考实现"留的口子

// 【最小例子的骨架】
//   part1.mlir:
//     func.func @parallel_negate(%arg0: tensor<8xf32> {sdy.sharding = ...})
//         -> (tensor<8xf32> {...}) {
//       %0 = stablehlo.negate %arg0 {sdy.sharding = ...} : tensor<8xf32>
//       return %0 : tensor<8xf32>
//     }
//   part2.mlir:
//     func.func @main() {
//       %input = stablehlo.constant dense<[...]> : tensor<8xf32>
//       %p = call @parallel_negate(%input)   : (tensor<8xf32>) -> tensor<8xf32>
//       %s = call @sequential_negate(%input) : (tensor<8xf32>) -> tensor<8xf32>
//       // 比较 %p 与 %s ...
//       return
//     }

// 【执行流程】
//   ① split-file 拆出 part1.mlir / part2.mlir
//   ② 导出流水线处理 part1 -> @parallel_negate 变成【局部代码】
//   ③ drop-sharding + 改名 -> @sequential_negate
//   ④ main 同时调用两者
//   ⑤ --interpret 执行 -> 结果一致 ✓

// 【为什么用 negate 做例子】
//   它是【逐元素】算子 —— 分片后无需通信（L5-04 讲过）
//   -> 最容易写对、最容易理解
//   想练通信的话可以换成 all_gather / dot 等`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:12px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:52px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const defs = [
      { t: '① RUN 行', c: '#38bdf8', d: '<span class="mono">%S/run_sdy_<br>interpreter_test.sh %s %t</span>' },
      { t: '② part1.mlir', c: '#4ade80', d: '含 <span class="mono">@parallel_*</span><br><b>带分片</b>' },
      { t: '③ part2.mlir', c: '#fbbf24', d: '含 <span class="mono">@main</span><br>同时调两者<br>并比较' },
      { t: '命名约定', c: '#fb7185', d: '<b><span class="mono">@parallel_</span></b> 前缀<br>脚本自动生成<br><span class="mono">@sequential_</span>' },
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(x => {
      const e = U.el('div', { class: 'card', style: `width:180px;opacity:.33;transition:.3s;border-color:${x.c}55;padding:9px;text-align:center` });
      e.innerHTML = `<div style="font-size:11px;color:${x.c}">${x.t}</div>
        <div class="small faint" style="font-size:10px;line-height:1.4;margin-top:3px">${x.d}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.33');
      msg.innerHTML = [
        '用 lit 的内置变量 <span class="mono">%s</span>（源文件）与 <span class="mono">%t</span>（临时目录）。',
        '<b>至少一个</b> <span class="mono">@parallel_*</span> 函数 —— 它是被测对象。',
        '<span class="mono">main</span> 要<b>同时调用</b>两个版本并比较 —— 这是验证的核心。',
        '<b>最容易出错的地方</b>：函数名必须是 <span class="mono">@parallel_</span> 前缀，否则脚本不会生成串行版。',
      ][i];
    }));
    tl.at(16400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '<b>建议起点</b>：用 <span class="mono">negate</span> 这类<b>逐元素算子</b> —— 分片后无需通信，最容易写对。';
    });
  }
},

/* ------------------------------------------------ 6 小结 */
{
  kicker: 'L6-00 · 可执行测试机制',
  title: '小结与 <span class="hl-a">L6 全层预览</span>',
  sub: '本课是 L6 的**开篇** —— 后续所有课都用本课讲的机制。',
  caption: 'L6 是"执行与解释器"层：分片代码怎么<b>真正跑起来</b>。',
  code: `// 【本课覆盖】2 个 shell 脚本（机制说明，非 IR）
//   executable_convert_global_to_local/run_sdy_interpreter_test.sh
//   executable_partitioner_pipeline/run_sdy_interpreter_test.sh
// 注意：它们不是 .mlir，【不计入】241 个测试 IR
//   -> 本课对覆盖率的贡献是 0
//   -> 这是【诚实】的：TODOLIST 也说这是"机制说明，非 IR"

// 【★ 三条核心结论】
//   ① 可执行测试 = 【分片版 vs 串行版】的结果对比
//   ② 串行版是脚本【自动生成】的（drop-sharding + 改名）
//   ③ 必须【真正执行】（--interpret），因为改错语义属性时 IR 仍能通过校验

// 【★ 与前面课的关系】
//   本课把 L4 + L5 的 pass 串成了一条【真实可跑的命令】
//     insert-explicit-reshards     L4-02~07
//     resolve-permutation-factors  L4-09
//     reshard-to-collectives       L4-08
//     optimize-collectives         L4-10
//     pad-for-divisibility         L5-09
//     resolve-single-device-sharding  L4-14
//   -> 是前面 26 课的【集成视图】

// 【L6 全层预览】8 课
//   L6-00 可执行测试机制（本课）        <- 机制基础
//   L6-01 sdy.* 集合通信的执行          P0
//   L6-02 stablehlo 集合通信的执行      P1
//   L6-03 卷积的执行                    P1
//   L6-04 矩阵乘 / fft / iota 的执行    P1
//   L6-05 gather 的执行                 P0 ★ 最复杂
//   L6-06 pad 的执行                    P0 ★ 本层最大族
//   L6-07 reshape 的执行                P0
//   L6-08 reverse / slice 的执行        P1
//   L6-09 scatter 与杂项                P1

// 【后续课的共同点】
//   它们都是 executable_* 目录下的测试
//   都用 run_sdy_interpreter_test.sh 驱动
//   -> 理解了本课的机制，后续课就只是"换个算子看结果"

// 一句话总结：
//   Shardy 的可执行测试 = 分片版与串行版的数值对比
//   脚本自动生成串行版，用 --interpret 真正执行来验证语义保持`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px;align-items:stretch;justify-content:center;flex-wrap:wrap" id="cards"></div>
      <div class="formula" id="msg" style="min-height:50px;display:flex;align-items:center;text-align:center;max-width:780px"></div>`;
    root.appendChild(wrap);
    const fams = [
      { t: 'L6-00 机制', n: 0, c: '#94a3b8' }, { t: 'L6-01 sdy 通信', n: 0, c: '#38bdf8' },
      { t: 'L6-02 shlo 通信', n: 0, c: '#0ea5e9' }, { t: 'L6-03 卷积', n: 0, c: '#4ade80' },
      { t: 'L6-04 矩阵/fft', n: 0, c: '#22c55e' }, { t: 'L6-05 gather', n: 0, c: '#fbbf24' },
      { t: 'L6-06 pad', n: 0, c: '#f59e0b' }, { t: 'L6-07 reshape', n: 0, c: '#c084fc' },
      { t: 'L6-08 rev/slice', n: 0, c: '#fb7185' }, { t: 'L6-09 scatter', n: 0, c: '#f472b6' },
    ];
    const host = wrap.querySelector('#cards');
    const els = fams.map(f => {
      const e = U.el('div', { class: 'card', style: `width:104px;opacity:.4;transition:.3s;border-color:${f.c}55;padding:7px;text-align:center` });
      e.innerHTML = `<div style="font-size:9px;color:${f.c};line-height:1.3">${f.t}</div>`;
      host.appendChild(e); return e;
    });
    const msg = wrap.querySelector('#msg');
    tl.at(700, () => { els.forEach((e, i) => setTimeout(() => e.style.opacity = '1', i * 90)); msg.innerHTML = '<b>L6 共 10 课</b>（含本课）—— 后续九课都基于本课的机制。'; });
    tl.at(4200, () => {
      msg.innerHTML = '<b>★ 三条核心结论</b>：分片版 vs 串行版对比 / 串行版自动生成 / 必须真正执行。';
    });
    tl.at(7600, () => {
      msg.innerHTML = '<b>本课是集成视图</b>：把 L4 + L5 的六个 pass 串成了一条真实可跑的命令。';
    });
    tl.at(10800, () => {
      msg.innerHTML = '<b>下一课</b> L6-01 讲 <span class="mono">sdy.*</span> 集合通信的执行（P0）。';
    });
  }
},

];
