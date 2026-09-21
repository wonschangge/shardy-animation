<!-- sdy-coverage
transforms/propagation/test/aggressive_propagation.mlir
-->

# L2-04 · aggressive-propagation — 源 IR

覆盖：`shardy/dialect/sdy/transforms/propagation/test/aggressive_propagation.mlir`（395 行 / 23 个用例）。

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-aggressive-propagate="propagation-strategy=aggressive" -verify-diagnostics 2>&1 | FileCheck %s
```

网格（节选）：

```mlir
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
```

```mlir
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>
```

```mlir
sdy.mesh @mesh_a_2_b_2_c_2_d_2 = <["a"=2, "b"=2, "c"=2, "d"=2]>
```

```mlir
sdy.mesh @empty_mesh = <[]>
```

---

## 一、无冲突时：与 basic 完全一致

```mlir
func.func @no_conflict(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>},
                       %arg1: tensor<8x8xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0] :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}
```

```mlir
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>}
  // CHECK-NEXT: stablehlo.dot_general %[[ADD]], %arg2
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {?}]>]>}
```

与 `basic_propagation.mlir` 的 `@simple` 用例结果**逐字相同** —— 没有冲突时两种策略不分家。

---

## 二、假冲突：同一个轴落在**不同因子**上

`dot_general` 的 `contracting_dims = [1] x [1]`，于是因子映射是
`([i, k], [j, k]) -> ([i, j])`。

```mlir
func.func @fake_conflict_between_two_non_contracting_dims(%arg0: tensor<256x512xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {?}]>},
                                                          %arg1: tensor<128x512xf32>)
          -> (tensor<256x128xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{?}, {"a", ?}]>}) {
```

```mlir
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{?}, {"a", ?}]>]>} :
    (tensor<256x512xf32>, tensor<128x512xf32>) -> tensor<256x128xf32>
  return %0 : tensor<256x128xf32>
}
```

```mlir
  // CHECK-NEXT: stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{?}, {"a", ?}]>]>}
```

**为什么是"假冲突"**：`"a"` 在 `%arg0` 上切的是因子 `i`，在结果上切的是因子 `j`
—— 两者是**不同的因子**，各切各的，没有冲突。所以传播照单全收。

---

## 三、真冲突：同一个因子上有**两个不同的轴**

逐元素算子，两维一一对应（`([i, j], [i, j]) -> ([i, j])`）：

```mlir
func.func @real_conflict_within_a_factor(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {}]>},
                                         %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"b", ?}, {}]>}) -> tensor<8x8xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: stablehlo.add %arg0, %arg1
  // CHECK-NOT:    sdy.sharding
```

**结果：不传播，结果上没有分片。** `%arg0` 要求因子 `i` 沿 `"a"` 切，`%arg1` 要求沿 `"b"` 切
—— 同一个因子上两个**不同的轴**，且两边都是用户标注，谁也不能覆盖谁。

> 这说明：**aggressive 并不是"消解一切冲突"**。它能消解的冲突有一组明确的启发式，
> 消解不了的会如实放弃。

---

## 四、侧向传播：结果被锁死时，走操作数之间

结果被显式标注为**空的闭维** `{}`：

```mlir
func.func @sideways_propagation_if_result_is_closed_empty(
    %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}]>},
    %arg1: tensor<8xf32>)
    -> tensor<8xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
```

```mlir
  // CHECK-NEXT: stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{}]>]>}
```

**"侧向"的含义**：分片不是从操作数"穿过"结果继续传，而是在**操作数之间**横向传播。
这里结果锁在 `{}`，于是 `"a"` 不会强加到结果上 —— 但两个操作数可以互相同步。

测试里共有 6 个侧向传播用例（`sideways_propagation_*` / `allow_sideways_*`），
覆盖"结果闭空 / 开空 / 子轴 / 部分冲突 / 完全匹配 / 多结果中只有一个冲突"等情形。

---

## 五、优先选"分片最多"的因子

```mlir
func.func @prefer_most_sharded_factor_elementwise_op(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{}, {"a"}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"b", "a"}, {}]>})
    -> tensor<8x8xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"b", "a", ?}, {?}]>]>}
```

**读法**：
- `%arg0` 第 1 维 `{"a"}` —— 用 1 个轴
- `%arg1` 第 0 维 `{"b", "a"}` —— 用 2 个轴
- 结果 `[{"b", "a", ?}, {?}]` —— 采用了**分片更多的那一侧**

这正是"减少内存占用"的直觉：分得越细，每台设备上的数据越少。
代价是另一侧可能需要通信来对齐。

> 同族的 4 个用例（`prefer_most_sharded_factor_*`）覆盖逐元素算子与
> 非逐元素算子（`prefer_lhs_factor_non_elementwise`）的不同取舍。

---

## 六、未归约轴会阻断反向传播

```mlir
func.func @unreduced_axes_block_bwd_propagation(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>},
    %arg1: tensor<8x16xf32>)
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{?}, {"b"}]>}) {
```

```mlir
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{?}, {?}], unreduced={"b"}>]>} :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = stablehlo.add %0, %0 : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}
```

```mlir
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {?}], unreduced={"b"}>]>}
  // CHECK-NEXT: stablehlo.add %[[DOT_GENERAL]], %[[DOT_GENERAL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>}
```

**`unreduced={"b"}` 的作用**：`dot_general` 的结果在 `"b"` 上是**部分和**。
反向传播不能把它当完整值回推到操作数，所以被阻断；但**前向传播照常**
（`add` 拿到的分片里仍带 `{"b", ?}`，因为它继承的是同一份未归约值）。

> 函数结果的声明 `[{?}, {"b"}]` 也不会覆盖这个判断 —— 未归约轴的语义优先。
