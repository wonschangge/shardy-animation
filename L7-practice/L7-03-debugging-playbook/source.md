<!-- sdy-coverage
transforms/propagation/test/op_sharding_rule_registry_conservative.mlir
transforms/propagation/test/op_sharding_rule_registry_failures.mlir
transforms/propagation/test/propagation_pipeline_data_flow_edges.mlir
transforms/propagation/test/propagation_pipeline_dedup_functions_fully_true.mlir
-->

# L7-03 · debugging-playbook — 源 IR

**调试手册**：四个文件提供**四类调试线索**。

| 文件 | 行数 | 用例数 | 提供什么线索 |
|---|---|---|---|
| `transforms/propagation/test/propagation_pipeline_data_flow_edges.mlir` | 820 | **40** | **区域算子**的传播 |
| `transforms/propagation/test/propagation_pipeline_dedup_functions_fully_true.mlir` | 304 | 21 | **选项开关** |
| `transforms/propagation/test/op_sharding_rule_registry_conservative.mlir` | 78 | 7 | **保守规则** |
| `transforms/propagation/test/op_sharding_rule_registry_failures.mlir` | 8 | 1 | **诊断警告** |

---

## 一、★ 四类调试线索

| 症状 | 该看哪个文件 |
|---|---|
| **分片没传播过去** | `data_flow_edges`（区域算子的数据流） |
| **出现意外的 all-gather** | `registry_conservative`（规则是否太保守） |
| **算子没有 sharding rule** | `registry_failures`（**警告信息**） |
| **想切换传播行为** | `dedup_functions_fully_true`（选项开关） |

---

## 二、★ 最直接的线索：`registry_failures` 的诊断警告

**8 行，1 个用例 —— 但它是最有用的调试素材。**

```mlir
// RUN: sdy_opt %s -sdy-populate-op-sharding-rules -verify-diagnostics

// CHECK-LABEL: func @unknown_custom_op
func.func @unknown_custom_op(%arg0: tensor<8x2xui32>, %arg1: tensor<8x2xui32>) -> tensor<8x2xui64> {
```

```mlir
  // expected-warning@+1 {{custom call @unknown_custom_op is unknown to SDY sharding rule registry}}
  %0 = stablehlo.custom_call @unknown_custom_op(%arg0, %arg1) : (tensor<8x2xui32>, tensor<8x2xui32>) -> tensor<8x2xui64>
```

**读法**（**本课最有价值的一处**）：
- **`-verify-diagnostics`** —— 让 `sdy_opt` **校验诊断信息**（而不只是输出 IR）。
- **`// expected-warning@+1 {{...}}`** —— 断言**下一行**会产生一条**警告**，
  内容是 `custom call @unknown_custom_op is unknown to SDY sharding rule registry`。

**★ 这条警告就是调试的入口**：
> 当你的自定义算子"分片没传播过去"时，**第一件事**是跑
> `sdy_opt -sdy-populate-op-sharding-rules` 看有没有这条警告 ——
> 如果有，说明**算子没有注册 sharding rule**（见 **L7-04** 的接入方法）。

**★ `-verify-diagnostics` 的用法**：
```sh
sdy_opt input.mlir -sdy-populate-op-sharding-rules -verify-diagnostics
```
它把"预期诊断"变成**可断言的测试** —— 这是 MLIR 生态的标准调试手法。

**注意这是 `expected-warning` 而非 `expected-error`** ——
**缺少 sharding rule 不是错误**，只是"传播时会跳过这个算子"。

---

## 三、`registry_conservative`：规则太保守会导致多余通信

```mlir
// RUN: sdy_opt %s -sdy-populate-op-sharding-rules="conservative-propagation=true" 2>&1 | FileCheck %s

// CHECK-LABEL: func @concat
func.func @concat(%arg0: tensor<4x3x256xf32>, %arg1: tensor<4x5x256xf32>) -> tensor<4x8x256xf32> {
```

```mlir
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, k, j], [i, l, j])->([i, m, j]) {i=4, j=256, k=1, l=1, m=1}>
 %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<4x3x256xf32>, tensor<4x5x256xf32>) -> tensor<4x8x256xf32>
 return %0 : tensor<4x8x256xf32>
}
```

**读法**：
- **`conservative-propagation=true`** —— 打开**保守传播**模式。
- 生成的规则：`([i, k, j], [i, l, j])->([i, m, j]) {i=4, j=256, k=1, l=1, m=1}`
  - 注意 **`k=1, l=1, m=1`** —— 拼接维（`k`/`l`/`m`）的因子被设为 **1**！
  - **因子为 1 意味着"这一维不可分"** → **保守**（不冒险切拼接维）。
- 7 个用例：`concat` `conv` `pad` `pad_same_shape_permutation` `reduce_window` `select_and_scatter` `slice`
  —— 都是**形状类/窗口类**算子（它们的规则容易出歧义）。

**★ 为什么这是调试线索**：
> 如果你看到**意外的 all-gather**，可能是某个算子的规则**太保守**
> （某维因子被设为 1 → 传播不过去 → 需要 reshard）。
> 打开 `conservative-propagation` 对比一下，就能判断是不是规则的问题。

**回顾 L2-10**：那里讲 `op_sharding_rule_registry` 与因子的含义 ——
本课看到**保守模式**下规则长什么样。

---

## 四、`data_flow_edges`：区域算子最容易出问题

```mlir
// RUN: sdy_opt %s -sdy-propagation-pipeline -split-input-file 2>&1 | FileCheck %s

// Propagation tests for ops with data-flow edges like CaseOp and WhileOp
```

**读法**（**820 行 / 40 个用例 —— 本课最大的文件**）：
- 注释直接点明：**"data-flow edges like `CaseOp` and `WhileOp`"**。
- **`CaseOp` / `WhileOp` 是区域算子** —— 它们有**嵌套的 region**，
  数据流要**穿过 region 边界**。
- **40 个用例**说明这是**最容易出问题**的地方。

**★ 为什么区域算子最容易出问题**（回顾 **L2-07**）：
> `data-flow edges` 描述"分片怎么从一个算子流到另一个算子"。
> 区域算子的数据流要**穿过 region 边界** ——
> 而 region 内的 block argument、`sdy.return`、嵌套结构都可能**打断传播**。

**★ 调试线索**：
> 如果"分片没传播过去"发生在 `while` / `case` 里，
> 先看 `data_flow_edges` 里**对应结构**的测试 —— 它展示了**正确**的传播结果。

**回顾 L2-08**：那里讲 `basic_propagation_data_flow_edges`（基础情形）；
本课的是**完整流水线**（`-sdy-propagation-pipeline`）的 40 个用例。

---

## 五、`dedup_functions_fully_true`：选项开关

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-propagation-pipeline='dedup-functions-fully=true' | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2]>
```

**读法**：
- **`dedup-functions-fully=true`** —— 一个**传播流水线的选项**。
- **21 个用例** —— 覆盖各种函数结构。
- **"dedup functions"** 指**去重函数**（多个相同签名的函数合并）。

**★ 为什么这是调试线索**：
> 当你怀疑"某个函数的分片被错误地合并/复用了"时，
> 切换 `dedup-functions-fully` 开关**对比结果** ——
> 如果行为变了，说明问题在**函数去重**这一步。

**★ 调试的通用手法**：
> **切换选项开关，看行为是否改变** ——
> 这是定位"问题在哪个 pass"的最快方法。

**回顾 L3-07**：那里讲 `flatten_call_graph` / `unflatten_call_graph` ——
函数结构的变换；本课的 `dedup` 是另一个函数级选项。

---

## 六、★ 两个调试工具（已实测存在）

`debugging-playbook` 的讲解要点提到的两个 flag，我实测确认它们存在：

```text
--debug-sharding-origins
    whether to save information about the origin of a sharding on the MLIR module.
    These would be the shardings on the function inputs, outputs, sharding
    constraints and manual computations before propagation.

--module-dump-directory=<string>
    where to dump any rewritten modules for debugging
```

### `--debug-sharding-origins`：追溯分片的来源

**用途**：记录"**每个分片是从哪来的**" ——
函数输入/输出、`sharding_constraint`、`manual_computation` 上的分片。

**★ 调试场景**：
> **"这个分片是谁塞进来的？"**
> 打开它，就能看到传播**之前**的原始分片标注 ——
> 从而判断"意外的分片"是**用户写的**还是**传播产生的**。

**注意**：还有配套的 `--sink-debug-sharding-origins`
（"Whether to sink the debug sharding origins info"）。

### `--module-dump-directory=<string>`：dump 每一步的 IR

**用途**：把**每个 pass 改写后的 module** 都 dump 到指定目录。

**★ 调试场景**：
> **"分片是在哪一步消失的？"**
> 指定一个目录，跑一遍，然后**逐步 diff** ——
> 就能精确定位是**哪个 pass** 改变了行为。

**★ 这两个工具的组合用法**：
| 问题 | 工具 |
|---|---|
| 分片**从哪来** | `--debug-sharding-origins` |
| 分片**在哪一步变** | `--module-dump-directory` |

---

## 七、★ 排查清单

按"症状 → 检查项"整理：

| 症状 | 第一步 | 第二步 |
|---|---|---|
| **分片没传播过去** | 跑 `-sdy-populate-op-sharding-rules -verify-diagnostics` 看警告 | 看 `data_flow_edges` 里对应结构 |
| **出现意外的 all-gather** | 对比 `conservative-propagation=true` | 看是不是规则太保守（因子=1） |
| **不可整除报错** | 看是不是**L5-09** 的场景 | 检查网格轴大小与维度大小 |
| **不确定分片从哪来** | `--debug-sharding-origins` | 对比传播前后的标注 |
| **不确定哪一步出问题** | `--module-dump-directory` | 逐步 diff |

**★ 三个通用手法**：
1. **看诊断信息**（`-verify-diagnostics`）—— 最直接
2. **切换选项开关**（`conservative-propagation` / `dedup-functions-fully`）—— 定位到 pass
3. **dump 中间 IR**（`--module-dump-directory`）—— 逐步定位

**一句话总结**：
> **调试 Shardy 分片问题的三步法**：
> ① 先看**诊断警告**（算子有没有规则）；
> ② 再**切换选项**（定位到哪个 pass）；
> ③ 最后 **dump IR 逐步 diff**（精确定位）。
