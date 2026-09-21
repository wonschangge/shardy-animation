<!-- sdy-coverage
transforms/export/test/insert_explicit_reshards.mlir
transforms/export/test/insert_func_call_reshards.mlir
-->

# L4-02 · reshard-insertion-overview — 源 IR

**L4 本节的总纲**。后续 L4-03～L4-07 会按算子族展开，本课先建立整体图景。

| 文件 | 行数 | 用例数 | RUN 行 |
|---|---|---|---|
| `transforms/export/test/insert_explicit_reshards.mlir` | 682 | 53 | `-allow-unregistered-dialect -sdy-insert-explicit-reshards='enable-full-version=false mark-partial-result-with-unreduced-axes=true'` |
| `transforms/export/test/insert_func_call_reshards.mlir` | 488 | 42 | `-sdy-insert-func-call-reshards` |

网格：

```mlir
sdy.mesh @mesh = <["x"=2, "y"=2, "z"=4]>
```

---

## 一、★ 为什么需要插入 reshard

**问题**：传播结束后，IR 里每个张量都拿到了分片。但**算子的约束**可能仍不满足。

以 `dot` 为例，它的要求是：

> **收缩维在两个操作数上必须同分片；非收缩维在操作数与结果之间必须同分片。**

而传播是从各个方向独立推导的，可能出现：

（具体例子见下面第二节的 `dot` 用例。）

**解法**：插入显式的 `sdy.reshard`，让某一侧"搬"到另一侧的分片上。

**这就是本节的中心思想**：
> **传播决定"每个张量怎么切"，reshard 负责"把不兼容的地方对齐"。**

---

## 二、`unreduced` 的处理：**尽量延迟归约**

这是本课最精彩的部分 —— 5 个用例覆盖"归约时机"的不同选择。

### 情形 ①：下游算子需要完整值 → 立刻归约

```mlir
func.func @all_reduce_on_func_input(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"y"}>}, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
```

```mlir
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: %[[MUL:.*]] = stablehlo.multiply %[[ALL_REDUCE]], %arg1
  // CHECK-NEXT: return %[[MUL]]
```

**读法**：`%arg0` 带 `unreduced={"y"}`（部分是部分和），而 `multiply` 需要完整值
→ 在 `multiply` **之前**插 `all_reduce`。

### 情形 ②：结果也接受未归约 → **完全不插**

```mlir
func.func @unreduced_func_input_until_return(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"y"}>}, %arg1: tensor<4x8xf32>)
    -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"y"}>}) {
```

```mlir
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"y"}>]>} : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: return %[[ADD]]
```

**读法**：**函数结果也声明了 `unreduced={"y"}`** → 整条链上都保持未归约
→ **一条 `all_reduce` 都不插**。用例名 `unreduced_func_input_until_return` 点明了"一直延迟到 return"。

### 情形 ③：**完全延迟**到 return 前

```mlir
func.func @all_reduce_fully_delayed_until_return(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) -> tensor<2x13xf32> {
```

```mlir
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>} : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  %2 = stablehlo.add %1, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>} : tensor<2x13xf32>
  return %2 : tensor<2x13xf32>
}
```

```mlir
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>}
  // CHECK:      %[[ADD:.*]] = stablehlo.add %[[REDUCE]], %[[REDUCE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x"} %[[ADD]] out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
```

**读法**：函数结果**没有**声明 `unreduced` → 最终必须归约。
但归约被**推迟到 `return` 之前**（`reduce` 与 `add` 都在未归约状态下完成）。

**为什么延迟**：
- 未归约状态下，每个设备只算**自己那一份部分和** —— 计算量更小。
- 越晚归约，中间能省下的计算越多。
- 反之，过早归约会让后续算子都做**重复的完整计算**。

**三种延迟程度**（用例名直接体现）：

| 用例 | 归约插在哪 |
|---|---|
| `all_reduce_on_func_input` | 立刻（下游需要完整值） |
| `all_reduce_delayed_until_op` | 延迟到某个算子前 |
| `all_reduce_partially_delayed_until_return` | 部分延迟 |
| `all_reduce_fully_delayed_until_return` | 完全延迟到 return 前 |
| `unreduced_func_input_until_return` | **完全不插**（结果也接受未归约） |

---

## 三、★ `dot` 的 reshard：让"对应维同分片"

这是用例数最多的一族（第 9～32 个，共 24 个）。

### 基础：结果分片与 lhs 冲突 → 插入 reshard

```mlir
func.func @reshard_dot_result_to_match_lhs(
    %arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>},
    %arg1: tensor<32x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<4x8xf32> {
```

```mlir
  %0 = stablehlo.dot %arg0, %arg1
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} :
      (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"x"}, {}]>
  // CHECK-NEXT: return %[[RESHARD]]
```

**逐项读**：
- **收缩维**：lhs 的第 1 维是 `{"x"}`，rhs 的第 0 维是 `{"x"}` → **一致** ✓
- **非收缩维**：lhs 的第 0 维是 `{"y"}`，而**结果要求** `{"x"}` → **冲突** ✗
- **解法**：让 `dot` 采用 **lhs 那一侧**（`[{"y"}, {}]`），再插一条
  `sdy.reshard` 把结果搬到 `[{"x"}, {}]`。

**规则**：
> **算子自身要满足"对应维同分片"；冲突时选一侧，另一侧用 reshard 补齐。**

### 冲突在**收缩维**上

```mlir
func.func @dot_lhs_and_rhs_conflicting_contracting_dim(
    %arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>},
    %arg1: tensor<32x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<4x8xf32> {
```

```mlir
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} :
      (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: return %[[DOT_GENERAL]]
```

**读法**：lhs 的收缩维（第 1 维）是 `{"x"}`，rhs 的收缩维（第 0 维）是 `{"y"}` —— **两者冲突**。
输出**没有插 reshard**，`dot_general` 保持 `[{}, {"y"}]`（与 rhs 的非收缩维一致）。

> **注意**：这个用例没有 `reshard` —— 说明并非所有冲突都靠 reshard 解决。
> 收缩维冲突的完整处理在后续课程展开。

### 这一族覆盖的冲突类型（24 个用例）

| 类型 | 代表用例 |
|---|---|
| 结果 vs lhs | `reshard_dot_result_to_match_lhs`、`reshard_dot_result_to_match_rhs` |
| 多轴 | `reshard_dot_result_with_multiple_axes` |
| 多使用者 | `reshard_dot_result_multiple_uses` |
| 批维 | `reshard_dot_general_batching_dim` |
| 多收缩维 / 多非收缩维 | `reshard_dot_general_with_multiple_sharded_contracting_dims` / `..._non_contracting_dims` |
| 缺分片 | `dot_result_missing_sharding` |
| 跨网格 | `dot_different_meshes` |
| lhs/rhs 冲突 | `dot_lhs_and_rhs_conflicting_{contracting,non_contracting,batching}_dim` |
| 子轴冲突 | `..._non_contracting_dim_sub_axis` |
| 结果 vs 空分片 | `dot_result_conflict_with_lhs_empty_lhs_sharding` |
| 多冲突 | `dot_multiple_conflicts_with_result` |
| 结果"更大" | `dot_result_bigger_than_conflicting_lhs` / `_rhs` |
| 与归约轴冲突 | `dot_result_conflicting_sharding_mismatch_with_reduction_axes`（3 个变体） |
| 与屏障 | `reshard_dot_result_to_match_lhs_with_barrier` |

---

## 四、`concatenate` 与 `sharded_to_unreduced`

### `concatenate`：拼接维的分片要一致

```mlir
func.func @concatenate_different_shardings(%arg0: tensor<4x32x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<4x48x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}, {}]>}) -> tensor<4x80x256xf32> {
```

```mlir
func.func @concatenate_same_shardings(%arg0: tensor<4x32x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<4x48x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> tensor<4x80x256xf32> {
```

三个用例覆盖：操作数分片不同 / 相同 / 结果分片与操作数不同。
**规则**：拼接维上各操作数的分片必须一致（否则无法直接拼），
不一致时插 reshard。

### `sharded_to_unreduced`：**分片 → 未归约**的转换

一族用例（第 36～52 个，共 17 个）：

| 用例 | 场景 |
|---|---|
| `replicated_to_unreduced_result_without_reshard` | 复制 → 未归约，无需 reshard |
| `sharded_to_unreduced_result_without_reshard` | 分片 → 未归约，无需 reshard |
| `sharded_to_unreduced` / `_single_axis` / `_multiple_axes` / `_multiple_dims` | 基本形态 |
| `sharded_to_unreduced_with_subaxis` | 子轴情形 |
| `implicitly_and_explicitly_replicated_to_unreduced_{full,sub}_axis` | 隐式 + 显式复制 |
| `replicated_and_sharded_to_unreduced_{full,sub}_axis` | 复制 + 分片混合 |
| `all_gather_and_replicated_to_unreduced_and_sharded_to_unreduced` | 与 all_gather 组合 |
| `all_slice_and_replicated_to_unreduced_and_sharded_to_unreduced` | 与 all_slice 组合 |
| `reshard_and_replicated_to_unreduced` | 与 reshard 组合 |
| `insert_reshard_unreduced_to_partial_replicated` / `_fully_replicated`（+`_max`/`_min`） | 未归约 → 复制（4 个，含 `max`/`min` 归约算子） |

**读法**：`unreduced` 与"分片/复制"是**正交**的状态维度（L1-02）。
这一族处理它们之间的相互转换。

---

## 五、`insert_func_call_reshards`：函数调用边界的 reshard

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-insert-func-call-reshards | FileCheck %s
```

488 行 / 42 个用例。这个 pass 处理**跨函数调用**的分片对齐：

- 调用点的实参分片 vs 函数形参的分片
- 函数的 return 分片 vs 调用结果的分片

**为什么需要**：L3-05 讲过 `sdy.func_data_flow_edge` 桥接调用点与函数体
（L3-10 又讲了"把函数结果的分片搬到调用点"）。但**桥接不等于对齐** ——
如果两端要求不同，就需要在调用点插入 reshard。

> 这与 `insert_explicit_reshards` 是**同一思路的两个作用域**：
> 前者处理算子内部，后者处理函数边界。

---

## 六、53 + 42 个用例的族谱

### `insert_explicit_reshards.mlir`（53）

| 族 | 数量 | 说明 |
|---|---|---|
| `unreduced` 处理 | 8 | 延迟归约的三种程度 |
| `dot` 的 reshard | 24 | **最大的一族** |
| `concatenate` | 3 | 拼接维分片一致性 |
| `sharded_to_unreduced` 等 | 17 | 状态转换 |
| `manual_computation` / `reduce` 多结果 | 2 | |

### `insert_func_call_reshards.mlir`（42）

函数调用边界的对齐 —— 与 L3-05 / L3-10 的函数级边配套。
