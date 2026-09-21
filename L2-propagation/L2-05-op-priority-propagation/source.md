<!-- sdy-coverage
transforms/propagation/test/op_priority_propagation.mlir
-->

# L2-05 · op-priority-propagation — 源 IR

覆盖：`shardy/dialect/sdy/transforms/propagation/test/op_priority_propagation.mlir`（313 行 / 15 个用例）。

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-op-priority-propagate 2>&1 | FileCheck %s
```

网格：`sdy.mesh @mesh = <["a"=2, "b"=2]>`

---

## 一、逐元素算子优先于 dot

`%arg0` 要求 `"a"` 切第 0 维；**函数结果**要求 `"a"` 切第 1 维。

```mlir
func.func @element_wise_over_dot_general(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}, %arg1: tensor<8x8xf32>) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"a", ?}]>}) {
```

```mlir
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = stablehlo.add %0, %0 : tensor<8x8xf32>
  %2 = stablehlo.add %1, %1 : tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}
```

```mlir
  // CHECK:      %[[DOT:.*]] = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>}
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[DOT]], %[[DOT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: %[[ADD_2:.*]] = stablehlo.add %[[ADD_1]], %[[ADD_1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: return %[[ADD_2]] : tensor<8x8xf32>
```

**读法**：结果被锁成 `[{?}, {"a", ?}]`。逐元素算子（`add`）**优先传播**，
把结果的要求反向推到 `dot_general`。最终 `%arg0` 自己的 `[{"a", ?}, {?}]` **没有赢** ——
`dot_general` 采用了结果那一侧。

---

## 二、多使用者的算子：推迟前向传播

`%1 = sine %arg0` 被用了**两次**：

```mlir
func.func @defer_forward_propagation_for_multi_use_ops(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>},
        tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>}) {
```

```mlir
  %1 = stablehlo.sine %arg0 : tensor<8x8xf32>
  %2 = stablehlo.add %1, %1 : tensor<8x8xf32>
  %3 = stablehlo.add %1, %1 : tensor<8x8xf32>
```

```mlir
  // CHECK-NEXT: %[[SINE:.*]] = stablehlo.sine %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>}
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[SINE]], %[[SINE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>}
  // CHECK-NEXT: %[[ADD_2:.*]] = stablehlo.add %[[SINE]], %[[SINE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>}
```

**要点**：
- `sine` 的结果有 2 个使用者 → 它的**前向传播被推迟**（避免一个使用者的要求过早"绑架"另一个）。
- 于是 `sine` 先拿到 `%arg0` 的分片 `[{"a", ?}, {?}]`。
- 而两个 `add` 则从**锁定的函数结果**反向拿到 `[{?}, {"a", ?}]`。

---

## 三、多使用者的操作数：推迟反向传播

`%arg0` 被两个 `add` 同时使用：

```mlir
func.func @defer_backwards_propagation_for_op_with_multi_use_operand(%arg0: tensor<8x8xf32>)
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>},
        tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
```

```mlir
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  %1 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  %2 = stablehlo.sine %0 : tensor<8x8xf32>
  return %1, %2 : tensor<8x8xf32>, tensor<8x8xf32>
```

```mlir
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>}
  // CHECK-NEXT: %[[ADD_2:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>}
  // CHECK-NEXT: %[[SINE:.*]] = stablehlo.sine %[[ADD_1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>}
  // CHECK-NEXT: return %[[ADD_2]], %[[SINE]]
```

**要点**：两个 `add` 各自拿到**不同**的分片（一个 `[{"a",?},{?}]`，一个 `[{?},{"a",?}]`）——
它们来自各自锁定的返回位置，互不干扰。`sine` 则跟随 `%0`。

---

## 四、pass-through 因子优先于 reduction 因子

```mlir
func.func @pass_through_factor_higher_priority_than_reduction_factor(
  %arg0: tensor<32x1024xf32>,
  %arg1: tensor<1024x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}]>}
) -> (tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}]>}) {
```

```mlir
  %0 = stablehlo.dot %arg0, %arg1, precision = [DEFAULT, DEFAULT] : (tensor<32x1024xf32>, tensor<1024x16xf32>) -> tensor<32x16xf32>
  return %0 : tensor<32x16xf32>
}
```

```mlir
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1, precision = [DEFAULT, DEFAULT] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>}
  // CHECK-NEXT: return %[[DOT]] : tensor<32x16xf32>
```

**因子映射**：`([i, k], [k, j]) -> ([i, j])`
- `%arg1` 的 `"a"` 切的是**收缩因子 `k`**（reduction）
- 结果的 `"a"` 切的是**非收缩因子 `i`**（pass-through）

结果采用了**函数结果那一侧** —— 用例名点明了原因：**pass-through 因子比 reduction 因子优先级更高**。

---

## 五、broadcast：前向优先于反向

两个串联的 `broadcast_in_dim`：

```mlir
func.func @broadcast_forward_higher_priority_than_backwards(
  %arg0: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}
) -> (tensor<32x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}, {}]>}) {
```

```mlir
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<32xf32>) -> tensor<32x16xf32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x8xf32>
  return %1 : tensor<32x16x8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[BROADCAST_1:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>}
  // CHECK-NEXT: %[[BROADCAST_2:.*]] = stablehlo.broadcast_in_dim %[[BROADCAST_1]], dims = [0, 1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}, {?}]>]>}
  // CHECK-NEXT: return %[[BROADCAST_2]]
```

**要点**：结果锁成 `[{}, {"a"}, {}]`。broadcast 的**前向传播优先级更高**，
于是要求沿前向逐级传递（`[{?}, {"a", ?}]` → `[{?}, {"a", ?}, {?}]`），
而不是把 `%arg0` 的 `"a"` 沿反向推过去。

---

## 六、串联的 dot：优先级逐级生效

```mlir
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = stablehlo.dot_general %0, %0, contracting_dims = [1] x [0] : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %2 = stablehlo.add %1, %1 : tensor<8x8xf32>
```

```mlir
  // CHECK:      %[[DOT_1:.*]] = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>}
  // CHECK:      %[[DOT_2:.*]] = stablehlo.dot_general %[[DOT_1]], %[[DOT_1]], contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>}
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[DOT_2]], %[[DOT_2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>} : tensor<8x8xf32>
```

**读法**：逐元素优先 → 先把结果的要求推到最近的 `DOT_2`；再轮到 dot 优先级 → `DOT_1` 得到 `[{"a", ?}, {?}]`。

---

## 七、分片约束也会被传播

```mlir
  // CHECK-NEXT: %[[WSC:.*]] = sdy.sharding_constraint %[[ADD_2]] <@mesh, [{?}, {"a", ?}]> : tensor<8x8xf32>
  // CHECK-NEXT: return %[[WSC]] : tensor<8x8xf32>
```

约束算子本身也会拿到分片，并继续向后传递 —— 见用例 `sharding_constraint_propagated`。
