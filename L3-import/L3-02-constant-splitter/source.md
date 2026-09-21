<!-- sdy-coverage
transforms/import/test/constant_or_scalar_splitter.mlir
-->

# L3-02 · constant-splitter — 源 IR

覆盖：`shardy/dialect/sdy/transforms/import/test/constant_or_scalar_splitter.mlir`
（**1782 行 / 95 个用例 —— 全计划最大的文件**）。

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-constant-or-scalar-splitter | FileCheck %s
```

**L3 层最大的一课**。核心问题：一个常量被多处使用时，该不该复制？

---

## 一、★ 核心规则：**N 个使用 → 复制 N 份**

```mlir
func.func @constant_multiple_users(%arg0: tensor<16x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
```

```mlir
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  %2 = stablehlo.add %0, %1 : tensor<8x16xf32>
  return %0, %2 : tensor<8x16xf32>, tensor<8x16xf32>
}
```

```mlir
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[CONST_0]], %arg0
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CONST_1]], %[[DOT_GENERAL]]
  // CHECK-NEXT: return %[[CONST_2]], %[[ADD]]
```

**读法**（本课的核心）：
- 原来**一个**常量有 **3 个使用**：`dot_general` 的操作数、`add` 的操作数、`return` 的返回值。
- 输出变成**三个** `sdy.constant`，每个使用各得一份：
  - `%[[CONST_0]]` → 给 `dot_general`
  - `%[[CONST_1]]` → 给 `add`
  - `%[[CONST_2]]` → 给 `return`

**为什么**：三处对分片的需求可能不同（`dot` 要沿收缩维切、`add` 要沿另一维、`return` 可能要求复制）。
共用一份常量会制造**假依赖** —— 三个使用被迫统一分片，可能引入不必要的通信。

**顺带**：`stablehlo.constant` 被换成了 `sdy.constant`（导入期的统一）。

---

## 二、同一个算子用了两次，**照样拆**

```mlir
func.func @constant_multiple_uses_by_same_op() -> (tensor<8x16xf32>, tensor<8x8xf32>) {
```

```mlir
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = stablehlo.dot_general %0, %0, contracting_dims = [1] x [1] : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x8xf32>
  return %0, %1 : tensor<8x16xf32>, tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[CONST_0]], %[[CONST_1]]
  // CHECK-NEXT: return %[[CONST_2]], %[[DOT_GENERAL]]
```

**读法**：`dot_general %0, %0` —— **同一个算子**把常量用了两次，
但仍然拆成两份（`%[[CONST_0]]` 与 `%[[CONST_1]]`），外加 `return` 的一份。

**结论**：判据是「**使用次数**」，不是「使用者个数」。
即使是左右操作数，也可能需要不同的分片（`dot` 的两侧切法本就不同）。

---

## 三、**不**拆的三种情形

### 情形 A：只有一个使用者

```mlir
func.func @func_arg_is_not_constant(%arg0: tensor<8x16xf32>, %arg1: tensor<16x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
```

```mlir
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = stablehlo.add %arg0, %0 : tensor<8x16xf32>
  %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  return %1, %2 : tensor<8x16xf32>, tensor<8x16xf32>
}
```

```mlir
  // CHECK-NEXT: %[[CONST:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %[[CONST]]
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[ADD]], %arg1
  // CHECK-NEXT: return %[[ADD]], %[[DOT_GENERAL]]
```

**读法**：常量只被 `add` 用了一次 → **只有一份**，没有复制。
复制没有收益（只有一个使用者，不存在假依赖）。

> 注意函数名 `func_arg_is_not_constant` 指的是"函数参数不是常量"，与拆不拆无关 ——
> 这个用例的重点是"单使用者不拆"。

### 情形 B：标量（rank 0）常量不拆

```mlir
func.func @scalar_constant_multiple_users_simple(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
```

```mlir
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = stablehlo.add %0, %arg0 : tensor<f32>
  return %0, %1 : tensor<f32>, tensor<f32>
}
```

```mlir
  // CHECK-NEXT: %[[CONST:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CONST]], %arg0
  // CHECK-NEXT: return %[[CONST]], %[[ADD]]
```

**读法**：虽然是 `tensor<f32>`（rank 0）且有 2 个使用，**只有一份**。

**为什么**：rank-0 张量**没有维度可分片** —— 复制它不会带来任何分片自由度，
只会让 IR 变长。**拆分的收益来自"每份可以有不同分片"，没有维度就没有收益。**

### 情形 C：非标量输入的 broadcast 不拆

```mlir
func.func @does_not_split_broadcast_on_non_scalar_input(%arg0: tensor<2xf32>) -> tensor<2x64xf32> {
```

```mlir
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<2xf32>) -> tensor<2x64xf32>
  %1 = stablehlo.negate %0 : tensor<2x64xf32>
  %2 = stablehlo.abs %0 : tensor<2x64xf32>
  %3 = stablehlo.multiply %1, %2 : tensor<2x64xf32>
  return %3 :  tensor<2x64xf32>
}
```

```mlir
  // CHECK-NEXT: %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<2xf32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %1 = stablehlo.negate %0 : tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.abs %0 : tensor<2x64xf32>
  // CHECK-NEXT: %3 = stablehlo.multiply %1, %2 : tensor<2x64xf32>
  // CHECK-NEXT: return %3 :  tensor<2x64xf32>
```

**读法**：`broadcast_in_dim` 的输入是 `tensor<2xf32>`（**非标量**），
且结果被用了 2 次 —— 但 broadcast **没有被复制**。

---

## 四、标量 broadcast：复制 **broadcast** 而不是标量

```mlir
func.func @splits_broadcast_on_scalar_simple(%arg0: tensor<f32>) -> tensor<2x64xf32> {
```

```mlir
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %1 = stablehlo.negate %0 : tensor<2x64xf32>
  %2 = stablehlo.abs %0 : tensor<2x64xf32>
  %3 = stablehlo.multiply %1, %2 : tensor<2x64xf32>
  return %3 :  tensor<2x64xf32>
}
```

```mlir
  // CHECK-NEXT: %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.negate %0 : tensor<2x64xf32>
  // CHECK-NEXT: %3 = stablehlo.abs %1 : tensor<2x64xf32>
  // CHECK-NEXT: %4 = stablehlo.multiply %2, %3 : tensor<2x64xf32>
  // CHECK-NEXT: return %4 :  tensor<2x64xf32>
```

**读法**（本课最巧妙的一处）：
- `broadcast_in_dim` 的输入是**标量** `tensor<f32>`（`dims = []`）。
- 标量本身不分片（情形 B）—— 但它的**结果** `tensor<2x64xf32>` 是**可以分片**的。
- 于是流水线复制的是 **broadcast 算子**（`%0` 与 `%1`），而不是标量。
- 这样 `negate` 与 `abs` 各自拥有一份 broadcast 结果，可以独立分片。

**这就是文件名的由来**：`constant_or_scalar_splitter` ——
拆**常量**，或者拆**标量的 broadcast**。

---

## 五、`sdy.constant` 同样处理

```mlir
func.func @splits_sdy_constants(%arg0: tensor<16x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
```

```mlir
  %0 = sdy.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  %2 = stablehlo.add %0, %1 : tensor<8x16xf32>
  return %0, %2 : tensor<8x16xf32>, tensor<8x16xf32>
}
```

```mlir
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
```

已经写成 `sdy.constant` 的常量**同样会被拆** —— 处理逻辑与 `stablehlo.constant` 一致。

---

## 六、95 个用例的族谱

| 族 | 代表用例 | 说明 |
|---|---|---|
| **基础拆分** | `constant_multiple_users`、`constant_multiple_uses_by_same_op`、`splits_sdy_constants` | N 个使用 → N 份 |
| **不拆的情形** | `func_arg_is_not_constant`、`scalar_constant_multiple_users_simple`、`non_scalar_constant_multiple_users_simple` | 单使用者 / 标量 |
| **常量子表达式** | `constant_sub_computation_multiple_users`、`splits_parts_of_const_sub_computation` | 由常量构成的子图 |
| **形状算子** | `constant_broadcast_multiple_users`、`constant_reshape_multiple_users`、`constant_slice_multiple_users` | 常量经过 reshape/slice |
| **标量 broadcast** | `splits_broadcast_on_scalar_*`（约 15 个）、`does_not_split_broadcast_*` | 拆 broadcast 而非标量 |
| **分片组** | `splits_sharding_groups`、`splits_const_subexpr_with_sharding_group` | 与组交互 |
| **named_computation** | `constant_*_within_named_computation*`（约 12 个） | 区域内的常量 |
| **call** | `constant_*_within_call*`、`constant_to_call_*`（约 18 个） | 函数调用相关 |
| **while** | `constant_to_while` | 循环相关 |
| **嵌套** | `constant_to_and_inside_nested_*` | 多层嵌套 |
| **多参数多结果** | `constant_to_*_with_multiple_arguments_*`、`constant_to_multi_result_*` | 复杂签名 |
| **综合** | `split_constants_different_sharding`、`simple_non_flat`、`single_call` | 真实形态 |
