<!-- sdy-coverage
transforms/propagation/debugging/test/edge_shardings.mlir
transforms/propagation/debugging/test/sharding_origins.mlir
-->

# L2-12 · propagation-debugging — 源 IR

本课覆盖两个调试文件（**L2 收官课**）：

| 文件 | 行数 | 调试属性 |
|---|---|---|
| `transforms/propagation/debugging/test/sharding_origins.mlir` | 758 | `sdy.sharding_origins` |
| `transforms/propagation/debugging/test/edge_shardings.mlir` | 402 | `sdy.propagation_edges` |

两个文件的 RUN 行都是**三步流水线**，且**每一步都打开同一个调试开关**：

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-add-data-flow-edges -sdy-apply-sharding-constraints=debug-sharding-origins=true -sdy-aggressive-propagate=debug-sharding-origins=true -sdy-sink-data-flow-edges="sink-debug-sharding-origins=true" 2>&1 | FileCheck %s
```

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-add-data-flow-edges -sdy-apply-sharding-constraints=debug-propagation-edge-sharding=true -sdy-aggressive-propagate=debug-propagation-edge-sharding=true -sdy-sink-data-flow-edges="sink-debug-propagation-edge-sharding=true" 2>&1 | FileCheck %s
```

网格：`sdy.mesh @mesh = <["a"=2, "b"=2, "c"=8]>`

---

## 一、`sdy.sharding_origins`：**每个轴从哪来**

### 函数签名上的形式

```mlir
func.func @input_output_source_sharding(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>},
  %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"c", ?}]>},
  %arg2: tensor<8x16xf32>
  ) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b", ?}]>}) {
```

传播后：

```mlir
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"c", ?}]>,
// CHECK-SAME:                            sdy.sharding_origins = {a = "self", c = "input: 1"}}
// CHECK-SAME:    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"c", ?}]>,
// CHECK-SAME:                            sdy.sharding_origins = {a = "input: 0", c = "self"}}
// CHECK-SAME:    %arg2: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c", ?}, {"b", ?}]>,
// CHECK-SAME:                             sdy.sharding_origins = {b = "output: 0", c = "input: 1"}}
// CHECK-SAME:    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}]>,
// CHECK-SAME:                          sdy.sharding_origins = {a = "input: 0", b = "self"}}) {
```

**格式**：`sdy.sharding_origins = {轴名 = "来源"}`，**逐轴**记录。

**三种来源取值**：

| 取值 | 含义 |
|---|---|
| `"self"` | 来自**这个张量自己**的标注（用户写的，或它作为函数参数/结果被锁定的） |
| `"input: N"` | 来自**第 N 个输入/操作数** |
| `"output: N"` | 来自**第 N 个输出/结果** |

**逐行读**：
- `%arg0`：`a` 是 `self`（它自己原本就写了 `a`）；`c` 是 `input: 1`（从 `%arg1` 传来的）。
- `%arg1`：`a` 是 `input: 0`（从 `%arg0` 传来）；`c` 是 `self`。
- `%arg2`：`c` 是 `input: 1`（从 `%arg1` 传来）；`b` 是 `output: 0`（从**函数结果**反向传来）。
- 函数结果：`a` 是 `input: 0`；`b` 是 `self`（结果自己写的）。

### 算子上的形式（是**列表**）

```mlir
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0] :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
```

```mlir
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"c", ?}]>]>,
  // CHECK-SAME:   sdy.sharding_origins = [{a = "input: 0", c = "input: 1"}]}
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[ADD]], %arg2, contracting_dims = [1] x [0] {
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>,
  // CHECK-SAME:   sdy.sharding_origins = [{a = "input: 0", b = "output: 0"}]}
  // CHECK-NEXT: return %[[DOT]]
```

**格式差异**：算子上是**列表**（`[{...}]`，每个结果一项），
因为一个算子可能有多个结果，每个结果各有自己的来源表。

**逐行读**：
- `add` 的 `a` 来自 `input: 0`（即 `%arg0`），`c` 来自 `input: 1`（即 `%arg1`）。
- `dot_general` 的 `a` 来自 `input: 0`（即 `%0`），`b` 来自 `output: 0`（函数结果反向推入）。

---

## 二、`sdy.propagation_edges`：**分片沿哪条路走**

同一个 `@input_output_source_sharding` 用例，换成 `edge_shardings` 开关后：

```mlir
func.func @input_output_source_sharding(
  %arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}, {?}]>},
  %arg1: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {?}, {"c", ?}]>}
) -> (tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b", ?}, {?}]>}) {
```

```mlir
  // CHECK-NEXT:  %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {
  // CHECK-SAME:    sdy.propagation_edges = #sdy.propagation_edges<[
  // CHECK-SAME:                                {step-1 = [
  // CHECK-SAME:                                  {"a" = operand-0 -> [operand-1, result-0]},
  // CHECK-SAME:                                  {"b" = result-0 -> [operand-0, operand-1]},
  // CHECK-SAME:                                  {"c" = operand-1 -> [operand-0, result-0]}]}]>,
  // CHECK-SAME:    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}, {"c", ?}]>]>
  // CHECK-SAME:  } : tensor<8x8x8xf32>
```

**格式**：`#sdy.propagation_edges<[{step-N = [{"轴" = 来源 -> [目标列表]}]}]>`

**逐条读**（这是本课信息量最大的一段）：
- `{"a" = operand-0 -> [operand-1, result-0]}`：
  轴 `"a"` 从 `%arg0`（operand-0）出发，传到了 `%arg1`（operand-1）**和**结果（result-0）。
  —— 注意它是**一对多**的：一次传播可以同时影响多个目标。
- `{"b" = result-0 -> [operand-0, operand-1]}`：
  轴 `"b"` 从**函数结果**反向传给了两个操作数。方向是 `result-0 -> operand-*`。
- `{"c" = operand-1 -> [operand-0, result-0]}`：与 `"a"` 对称。

**`step-N`**：传播的**轮次**。同一个算子上可能记录多轮的边
（函数结果上就有 `step-0` 与 `step-2` 两组）。

函数结果上的记录：

```mlir
// CHECK-SAME:  ) -> (tensor<8x8x8xf32> {sdy.propagation_edges = #sdy.propagation_edges<[
// CHECK-SAME:                                                       {step-0 = [{"b" = result-0 -> [operand-0]}]},
// CHECK-SAME:                                                       {step-2 = [{"a" = operand-0 -> [result-0]}, {"c" = operand-0 -> [result-0]}]}]>,
// CHECK-SAME:                           sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}, {"c", ?}]>}) {
```

---

## 三、两个属性的分工

| 属性 | 回答的问题 | 粒度 |
|---|---|---|
| `sdy.sharding_origins` | 这个轴的最终值**来自哪里** | 逐轴，一个来源 |
| `sdy.propagation_edges` | 分片**沿哪条路径**流动过 | 逐轴，可多条边 + 轮次 |

前者是**结论**（快照），后者是**过程**（轨迹）。

---

## 四、用例族谱

### `sharding_origins.mlir`（758 行 / 20 个用例）

| 族 | 用例 |
|---|---|
| 基础来源 | `input_output_source_sharding`、`partial_axes_match`、`larger_prefix_match` |
| 直接返回的参数 | `direct_returned_arg_new_axis_input` / `_output` |
| 约束 | `single_sharding_constraint`、`two_sharding_constraint`、`push_sharding_constraints_to_func_results` |
| 子轴 | `sub_axis_update`、`sub_axes_splitting_reshape`、`sub_axes_merging_reshape`、`already_split_sub_axis_result_reshape` |
| manual computation | `manual_computation_no_manual_axes`、`_manual_axes`、`_multiple_results` |
| 冲突与多结果 | `multiple_axes`、`tie_across_operands_results`、`real_conflict_across_factors_diff_tensors_size`、`while_loop_with_multiple_results` |

### `edge_shardings.mlir`（402 行 / 15 个用例）

| 族 | 用例 |
|---|---|
| 基础 | `input_output_source_sharding`、`duplicate_operands`、`multiple_axes` |
| 子轴 | `sub_axis_update`、`sub_axes_splitting_reshape`、`sub_axes_merging_reshape` |
| manual computation | `manual_computation_manual_axes`、`manual_computation_multiple_results` |
| 约束 | `two_open_sharding_constraint`、`open_sharding_constraint`、`push_sharding_constraints_to_func_results`、`has_other_identical_sharding_constraint_user`、`has_different_sharding_constraint_user` |
| 其它 | `input_already_has_sharding`、`chain_on_block_arg_after_other_user` |
