<!-- sdy-coverage
transforms/export/test/unflatten_call_graph.mlir
transforms/export/test/unflatten_call_graph_dedup_functions_fully_true.mlir
-->

# L4-13 · unflatten-call-graph — 源 IR

**核心机制**：按 **in/out 分片**去重函数，把调用点统一到同一个函数上。

| 文件 | 行数 | 用例数 | RUN 选项 |
|---|---|---|---|
| `transforms/export/test/unflatten_call_graph.mlir` | **1646** | **115** | 默认 |
| `transforms/export/test/unflatten_call_graph_dedup_functions_fully_true.mlir` | **1665** | **117** | `dedup-functions-fully=true` |

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-unflatten-call-graph | FileCheck %s
```

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-unflatten-call-graph='dedup-functions-fully=true' | FileCheck %s
```

**两个文件几乎相同** —— 差别只在去重的**激进程度**，这正是本课的主线。

---

## 一、★ 去重：分片相同的函数被合并

### 输入：两个函数，分片相同

```mlir
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
```

```mlir
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
```

```mlir
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
attributes { sdy.original_func_name = "baz" } {
```

```mlir
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
```

**读法**：
- `@baz` 与 `@baz_0` 的 **in/out 分片完全相同**
  （参数 `[{}, {"y"}]`、结果 `[{"x"}, {}]`）。
- `@baz_0` 带 **`sdy.original_func_name = "baz"`** —— 说明它**同源**
  （正是 **L4-12** outline 时产生的）。

### 输出：两个调用点都指向 `@baz`

```mlir
func.func @multiple_same_calls_same_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
```

```mlir
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}
```

```mlir
  // CHECK-NEXT: %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %1 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %1 : tensor<8x2xi32>
```

**读法**（本课的核心动作）：
- 输入：`call @baz` 与 **`call @baz_0`**（两个不同的函数）
- 输出：**两个都是 `call @baz`** —— `@baz_0` 的调用点被**重定向**到 `@baz`。
- 因为两个函数的 **in/out 分片相同**，所以合并是**安全的**。

**这就是 "unflatten" 的含义**：
L4-12 的 outline 会为**每个实例**生成一个函数（`@baz`、`@baz_0`…），
本课再把**同源且分片相同**的那些**收敛回一个**。

```
L4-12: named_computation × N  --outline-->  @baz, @baz_0, @baz_1, ...
L4-13: @baz, @baz_0, ...      --去重-->     @baz
```

---

## 二、★ `dedup-functions-fully` 的取舍

### 分片**不同**时的两种行为

```mlir
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
```

```mlir
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
```

**注意**：`@baz` 的结果分片是 `[{"x"}, {}]`，`@baz_0` 是 `[{"x"}, {"y"}]` —— **不同**！

**默认（`dedup-functions-fully=false`）**：

```mlir
  // CHECK-NEXT: %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
```

→ **两个调用点分别指向 `@baz` 与 `@baz_0`** —— **不合并**。

**`dedup-functions-fully=true`**：

```mlir
  // CHECK-NEXT: %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %1 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
```

→ **两个调用点都指向 `@baz`** —— **合并了**！
注意第二个 call 的 **sharding 仍然是 `[{"x"}, {"y"}]`**（与第一个不同）。

### ★ 为什么 `true` 时合并是可行的

**关键**：**函数的分片信息在【调用点】上** —— 看 `call ... {sdy.sharding = ...}`。

所以即使两个调用点要求**不同的 out 分片**，也可以调用**同一个函数**：
- 函数的签名（in/out 分片）只是一个"默认约定"
- **真正的分片要求由调用点决定**

**这正是 L3-06 / L3-10 讲过的**：
> `out_shardings` 只来自**调用点**（L3-06 的 NOTE：`we ignore any arg/result shardings on the function`）。

### 两种选项的取舍

| | 默认（`false`） | `dedup-functions-fully=true` |
|---|---|---|
| **去重依据** | **按 in/out 分片** | **完全去重** |
| **分片不同时** | **不合并**（`@baz` / `@baz_0`） | **合并**（都用 `@baz`） |
| **函数数量** | 多 | **少** |
| **保守程度** | 保守（每个函数分片自洽） | 激进（依赖调用点的 sharding） |

**收益**：函数更少 → IR 更小、编译更快、代码体积更小。
**代价**：函数签名上的分片可能与调用点不一致 —— 下游必须**总是读调用点的 sharding**。

**这回答了 TODOLIST 的验收点**："能判断两个函数在给定选项下是否会被合并" ——
**看它们的 in/out 分片是否相同**：相同则总是合并；不同则只有 `true` 才合并。

---

## 三、两个文件的**用例差集**

两个文件 115 / 117 个用例，绝大多数相同。差集揭示了 `true` 额外覆盖的场景：

| 仅 `true` 有的用例 | 含义 |
|---|---|
| `multiple_same_calls_different_shardings_different_number_of_call_sites_one_called_twice` | **调用点数量不同**（一个被调两次） |
| `multiple_same_calls_different_shardings_different_number_of_call_sites_multiple_func_origins` | 调用点数量不同 + **多个来源** |
| `single_call_func_result_empty_sharding_call_has_sharding` | 函数结果**空分片**，call 有分片 |
| `two_calls_same_origin_one_call_with_empty_sharding` | 一个 call **空分片** |
| `three_calls_same_origin_func_with_two_calls_results_no_sharding` | 三个调用点，两个结果**无分片** |
| `three_calls_same_origin_func_with_one_call_results_no_sharding` | 一个结果无分片 |

**读法**：`true` 需要额外覆盖"**空分片**"与"**调用点数量不同**"的情形 ——
因为完全去重时，这些差异**不再阻止合并**，所以必须有测试锁定行为。

**默认版本专有的用例**（如 `single_call_func_arg_has_sharding_call_arg_no_sharding`）
则覆盖"参数分片与 call 实参分片不一致"的情形 ——
保守模式下这些差异会导致**不合并**。

---

## 四、与 L4-12 的衔接

**本课是 L4-12 的直接后续**：

| 课 | 做什么 | 函数数量变化 |
|---|---|---|
| **L4-12** outline | `named_computation` → `func` + `call` | **变多**（每个实例一个） |
| **L4-13** unflatten | 按 in/out 分片去重 | **变少**（收敛回去） |

**为什么 L4-12 要多、L4-13 要少**：
- L4-12 的 outline 必须**保守** —— 每个 `named_computation` 实例单独生成函数，
  因为它们的 in/out 分片可能不同（L4-12 的用例名 `..._different_shardings` 就是这种）。
- L4-13 在**全局视角**下再做一次去重 —— 此时能看到所有函数，可以安全合并。

**`sdy.original_func_name` 属性在这里的作用**：
它是 L4-12 留下的"**同源标记**"，让 L4-13 知道"`@baz_0` 原本也叫 `baz`"。
没有这个属性，去重只能靠"函数体是否相同"来判断（更贵且不可靠）。

---

## 五、两个 pass 的分工（导入 vs 导出）

`flatten_call_graph.mlir`（**导入**期，415 行 / 54 用例）是本课的**镜像**：

| 时期 | pass | 方向 |
|---|---|---|
| **导入** | `-sdy-flatten-call-graph` | 函数**变少**（合并/扁平化） |
| **导出** | `-sdy-unflatten-call-graph` | 函数**变少**（去重） |

> 注意：本课**只覆盖导出侧的 2 个文件**（TODOLIST 的覆盖清单）。
> `flatten_call_graph.mlir` 属于导入侧，在 L3 层的覆盖范围内。

**一句话总结**：
> **`unflatten_call_graph` 按 in/out 分片去重函数，把调用点统一到同一个函数上。**
> 默认保守（分片不同就不合并）；`dedup-functions-fully=true` 激进（完全去重，
> 因为**真正的分片要求由调用点决定**）。
