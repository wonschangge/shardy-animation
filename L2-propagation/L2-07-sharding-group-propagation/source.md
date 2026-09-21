<!-- sdy-coverage
transforms/propagation/test/sharding_group_propagation.mlir
-->

# L2-07 · sharding-group-propagation — 源 IR

覆盖：`shardy/dialect/sdy/transforms/propagation/test/sharding_group_propagation.mlir`（254 行 / 12 个用例）。

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-basic-propagate -verify-diagnostics | FileCheck %s
```

**注意 RUN 行**：用的是 `-sdy-basic-propagate` —— 分片组不是单独的 pass，
而是**内建在传播里**的一条独立通道。

网格：`sdy.mesh @mesh = <["a"=2, "b"=2]>`

---

## 一、最简形态：组里只有一个成员

```mlir
func.func @main(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {?}]>})
   -> (tensor<8x8xf32>) {
```

```mlir
  %0 = stablehlo.tanh %arg0 : tensor<8x8xf32>
  sdy.sharding_group %0 group_id = 0 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: stablehlo.tanh %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", ?}, {?}]>]>} : tensor<8x8xf32>
```

单个成员时组没有额外约束，`tanh` 正常从 `%arg0` 继承分片。

---

## 二、组的作用可以**穿透传播屏障**

这是最能说明"组是独立通道"的例子：

```mlir
func.func @shard_as_applies_despite_barrier(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>})
   -> (tensor<8x8xf32>) {
```

```mlir
  %0 = sdy.constant dense<0.000000e+00> : tensor<8x8xf32>
  %1 = sdy.constant dense<0.000000e+00> : tensor<8x8xf32>
```

```mlir
  %2 = stablehlo.tanh %arg0 : tensor<8x8xf32>
```

```mlir
  %3 = sdy.propagation_barrier %1 allowed_direction=NONE : tensor<8x8xf32>
```

```mlir
  %4 = stablehlo.add %2, %3 : tensor<8x8xf32>
```

```mlir
  sdy.sharding_group %0 group_id=0 : tensor<8x8xf32>
  sdy.sharding_group %4 group_id=0 : tensor<8x8xf32>
  return %4 : tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} dense<0.000000e+00> : tensor<8x8xf32>
```

```mlir
  // CHECK: stablehlo.tanh %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<8x8xf32>
```

```mlir
  // CHECK-NEXT: sdy.propagation_barrier %1 allowed_direction=NONE {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<8x8xf32>
```

```mlir
  // CHECK-NEXT: stablehlo.add %2, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<8x8xf32>
```

**读法**：
- `%4`（`add`）通过 `%2` 从 `%arg0` 拿到 `[{"a", ?}, {?}]`。
- `%0`（常量）与 `%4` **同组** → 也被赋予同样的分片。
- `%1` 与 `%3`（屏障）虽然**不在组里**，但因为屏障是恒等算子、且 `%3` 被 `%4` 使用，
  也一并拿到了分片。
- 关键：`%0` 到 `%4` 之间**没有数据依赖**，路径上还隔着 `allowed_direction=NONE` 的屏障 ——
  **组这条通道绕过了它**。

---

## 三、组内冲突：按**用户优先级**裁决

```mlir
func.func @shard_as_reflects_user_priority(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}p0, {"c", ?}]>},
  %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c", "a",?}p1, {"b", ?}]>})
   -> (tensor<8x8xf32>) {
```

```mlir
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  %1 = stablehlo.add %arg1, %arg1 : tensor<8x8xf32>
  %2 = stablehlo.add %0, %1 : tensor<8x8xf32>
  sdy.sharding_group %0 group_id = 0 : tensor<8x8xf32>
  sdy.sharding_group %1 group_id = 0 : tensor<8x8xf32>
  sdy.sharding_group %2 group_id = 0 : tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"c", ?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"c", ?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: stablehlo.add %0, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"c", ?}]>]>} : tensor<8x8xf32>
```

**读法**：组内三个张量最终统一为 `[{"a", ?}, {"c", ?}]` ——
即 `%arg0` 那一侧（p0）的分片，而不是 `%arg1` 的 `{"c", "a",?}p1`。

**结论**：组内出现不一致时，用**用户优先级**裁决（与 L2-06 的规则一致）。

---

## 四、完全不兼容时：插入额外的分片约束

两个成员的标注互不相容（一个要 `"a"`、一个要 `"b"`）：

```mlir
func.func @add_extra_sharding_constraint_for_incompatible_shardings_in_sharding_group(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>}) {
```

```mlir
  sdy.sharding_group %arg0 group_id = 0 : tensor<8x8xf32>
  sdy.sharding_group %arg1 group_id = 0 : tensor<8x8xf32>
  func.return
}
```

```mlir
  // CHECK-NEXT:  %[[WSC_0:.*]] = sdy.sharding_constraint %arg1 <@mesh, [{"b", ?}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  %[[WSC_1:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{"b", ?}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  sdy.sharding_group %[[WSC_1]] group_id=0 : tensor<8x8xf32>
  // CHECK-NEXT:  sdy.sharding_group %[[WSC_0]] group_id=0 : tensor<8x8xf32>
```

**读法**：
- 传播**自动插入**两条 `sdy.sharding_constraint`，把两个成员都约束成 `[{"b", ?}, {?}]`。
- 注意：`%arg0` 原本是 `"a"`，这里被统一到了 `"b"` —— 说明裁决结果取决于处理顺序。
- 组被**重建**到新插入的约束上（`sharding_group %[[WSC_1]]`），而不是原来的参数上。

> 测试里的注释点明了时机：
> `Sharding Group and Sharding Constraint compatibility checks happend after unification + canonicalization of group ids.`

### 同族的 5 个用例

| 用例 | 场景 |
|---|---|
| `..._in_sharding_group` | 组内两个成员标注不兼容 |
| `..._in_manual_computation` | 发生在 manual_computation 内部 |
| `..._with_sharding_constraint` | 与已有的分片约束冲突 |
| `..._for_partially_closed_replicated_sharding` | 部分闭的复制分片 |
| `..._for_fully_open_replicated_sharding` | 全开的复制分片 |

---

## 五、跨数据流边

用例 `shard_as_across_dataflow_edge` 验证：组成员位于数据流边的两侧时，
组的分片仍然能同步过去。

---

## 六、已有分片会直接赋给组内成员

```mlir
func.func @set_existing_shardings_for_sharding_group_members(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
```

```mlir
  %0 = stablehlo.constant dense<0.0> : tensor<8x8xf32>
  sdy.sharding_group %arg0 group_id = 0 : tensor<8x8xf32>
  sdy.sharding_group %arg1 group_id = 0 : tensor<8x8xf32>
  sdy.sharding_group %0 group_id = 0 : tensor<8x8xf32>
  func.return
}
```

```mlir
  // CHECK: %cst = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>} dense<0.000000e+00> : tensor<8x8xf32>
```

常量 `%0` 原本没有分片，但因为与两个已分片的参数同组，直接拿到了 `[{"a"}, {"b"}]`。
