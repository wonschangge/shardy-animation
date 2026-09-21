<!-- sdy-coverage
transforms/propagation/test/basic_propagation_manual_computation.mlir
-->

# L2-09 · manual-computation-propagation — 源 IR

覆盖：`shardy/dialect/sdy/transforms/propagation/test/basic_propagation_manual_computation.mlir`（415 行 / 27 个用例）。

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-add-data-flow-edges -sdy-basic-propagate -sdy-sink-data-flow-edges 2>&1 | FileCheck %s
```

网格：`sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2, "d"=2, "e"=2, "f"=2, "g"=2]>`

---

## 一、`manual_axes={}` 时：`out_shardings` 直接传进体内

```mlir
func.func @manual_computation_output_sharding_annotation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
```

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{?}, {?}]>] out_shardings=[<@mesh, [{"a", ?}, {?}]>] manual_axes={} (%arg1: tensor<32x32xf32>) {
```

```mlir
    %2 = stablehlo.add %1, %1 : tensor<32x32xf32>
```

```mlir
    // CHECK: stablehlo.add %1, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<32x32xf32>
```

**读法**：`manual_axes={}` 表示**没有轴被冻结**，所以整个区域就是普通的传播区域。
`out_shardings` 里写的 `[{"a", ?}, {?}]` 被**反向传进体内**，落到 `add` 上。

---

## 二、★ 核心机制：自由轴会被**追加**到 in/out_shardings

### 从体内向外追加

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", ?}, {?}]>] out_shardings=[<@mesh, [{"b", ?}, {?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
```

```mlir
    // CHECK:      sdy.manual_computation(%arg0)
    // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b", "a", ?}, {?}]>]
    // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", ?}, {?}]>]
    // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
```

**读法**：`in_shardings` 原本是 `[{"b", ?}, {?}]`（只有 manual 轴 `"b"`）。
体内算子的分片是 `[{"a", ?}, {?}]` —— 自由轴 `"a"` 被**追加到 manual 轴之后**，
变成 `[{"b", "a", ?}, {?}]`。

> 这与 L1-07 的顺序约束一致：**manual 轴必须在前，自由轴在后**。

### 从外部向内追加

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", ?}, {?}]>] out_shardings=[<@mesh, [{"b", ?}, {?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
```

```mlir
  %3 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", "a", ?}, {?}]>]>} : tensor<32x32xf32>
```

```mlir
  // CHECK:      %[[MC:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", "a", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
```

**读法**：这次是**外部**的 `add` 要求 `[{"b", "a", ?}, {?}]`，于是
`out_shardings` 从 `[{"b", ?}, {?}]` 被追加成 `[{"b", "a", ?}, {?}]`。

---

## 三、追加前会**先移除已有自由轴**（避免重复）

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", "a", ?}, {?}]>] out_shardings=[<@mesh, [{"b", ?}, {?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
```

```mlir
    %1 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", "c", ?}, {?}]>]>} : tensor<16x32xf32>
```

```mlir
  // CHECK:      sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b", "a", "c", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
```

**读法**：
- 原 `in_shardings` 第 0 维 = `{"b", "a", ?}` —— `"b"` 是 manual，`"a"` 是**已有的自由轴**。
- 体内要求自由部分为 `{"a", "c", ?}`。
- 结果 = `{"b", "a", "c", ?}` —— **不是** `{"b", "a", "a", "c", ?}`。

**结论**：追加前会**先删掉原有的自由轴**，再拼接新的。用例名 `remove_existing_free_axes_in_shardings`
说的就是这个（`out_shardings` 有对应的 `_out_shardings` 版本）。

---

## 四、`replicated` 里的 manual 轴与自由轴

```mlir
  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"b", ?}, {?}], replicated={"c"}>] out_shardings=[<@mesh, [{"b", ?}, {?}], replicated={"c"}>] manual_axes={"b", "c"} (%arg1: tensor<16x32xf32>) {
```

```mlir
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", "a", ?}, {?}]>]>} : tensor<32x32xf32>
```

```mlir
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%[[ADD]])
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b", "a", ?}, {?}], replicated={"c"}>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", "a", ?}, {?}], replicated={"c"}>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b", "c"} (%arg1: tensor<16x32xf32>) {
```

**读法**：`manual_axes={"b", "c"}` 但只有 `"b"` 真正切了维度，`"c"` 放在
`replicated` 里。两者都是 manual 轴（都被冻结），只是角色不同。

> 同族用例还覆盖：`replicated_free_axes`（自由轴进 replicated）、
> `replicated_inside_body`、以及"直接被返回"的变体。

---

## 五、闭维保护：**不**往闭维里追加自由轴

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b"}, {?}]>] out_shardings=[<@mesh, [{"b"}, {?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
```

```mlir
    %1 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>
```

```mlir
  // CHECK:      sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b"}, {?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b"}, {?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
```

**读法**：注意第 0 维是 `{"b"}`（**闭维**，没有 `?`）。
虽然体内算子带了自由轴 `"a"`，`in_shardings` / `out_shardings` **完全没有变化**。

**结论**：闭维是用户的硬承诺 —— 传播**不会**往里追加任何轴。

> 同族三个用例分别覆盖从体内 / 从 out_sharding 外部 / 从 in_sharding 外部
> 尝试向内传播的情形（`dont_propagate_into_*_closed_dim_from_*`）。

---

## 六、用例族谱

| 族 | 用例 |
|---|---|
| 基础 / 体内传播 | `manual_computation_output_sharding_annotation`、`..._sharding_inside_body_propagation`、`..._directly_returned_body_arg(_conflict)`、`..._inside_body_forward_propagation` |
| **追加**自由轴 | `append_in_sharding_from_inside` / `_from_outside(+multiple)`、`append_out_sharding_from_inside` / `_from_outside`、`append_out_sharding_from_inside_multiple_results` |
| **移除**已有自由轴 | `remove_existing_free_axes_in_shardings` / `_out_shardings` |
| manual 轴组合 | `multiple_manual_axes_same_dim` / `_different_dim` |
| 自由轴落位 | `free_axis_on_own_dim_from_inside` / `_from_outside` |
| replicated | `replicated_manual_axes(+2 变体)`、`replicated_free_axes(+2 变体)`、`replicated_inside_body` |
| 闭维保护 | `preserve_untouched_closed_dim`、`dont_propagate_into_*_closed_dim_from_*`（3 个） |
| token | `manual_computation_with_tokens` |
