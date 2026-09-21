<!-- sdy-coverage
transforms/propagation/test/basic_propagation_conservative.mlir
-->

# L2-02 · conservative-mode — 源 IR

覆盖：`shardy/dialect/sdy/transforms/propagation/test/basic_propagation_conservative.mlir`（51 行 / 5 个用例）。

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-basic-propagate="conservative-propagation=true" 2>&1 | FileCheck %s
```

注意 RUN 行：**同一个 pass**，只是多了一个选项 `conservative-propagation=true`。

---

## 一、需要拆分子轴 → 不传播

网格 `a=4`；`8` reshape 成 `2x4`。

```mlir
sdy.mesh @mesh_a_4_b_2 = <["a"=4, "b"=2]>
```

```mlir
func.func @reshape_split_dim(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_4_b_2, [{"a"}]>}) -> tensor<2x4xf32> {
```

```mlir
  %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
```

期望（CHECK 行）——**结果上完全没有 sharding**：

```mlir
  // CHECK-NEXT: stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
  // CHECK-NOT: sdy.sharding
```

**为什么**：`8` 沿 `a`（大小 4）切成 4 份；reshape 成 `2x4` 后，`a` 无法整段落在任一维上
（第 0 维只有 2），必须**拆成子轴** `"a":(1)2` 与 `"a":(2)2`。保守模式禁止分裂轴 → 不传播。

---

## 二、多轴 reshape 的三种结果

### 1. 主维用满了所有轴 → 完整传播

网格 `a=4, b=2`；`32` → `16x2`；输入第 0 维是 `{"a", "b"}`（4×2 = 8）。

```mlir
sdy.mesh @mesh_a_4_b_2 = <["a"=4, "b"=2]>
```

```mlir
func.func @multi_axis_major_dim_uses_all_axes(%arg0: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh_a_4_b_2, [{"a", "b"}]>}) -> tensor<16x2xf32> {
```

```mlir
  %0 = stablehlo.reshape %arg0 : (tensor<32xf32>) -> tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}
```

```mlir
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4_b_2, [{"a", "b", ?}, {?}]>]>}
```

### 2. 主维只用到部分轴 → 部分传播

网格 `a=2, b=8`；`16` → `4x4`；输入第 0 维是 `{"a", "b"}`（2×8 = 16）。

```mlir
sdy.mesh @mesh_a_2_b_8 = <["a"=2, "b"=8]>
```

```mlir
func.func @multi_axis_major_dim_not_fully_sharded(%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_8, [{"a", "b"}]>}) -> tensor<4x4xf32> {
```

```mlir
  %0 = stablehlo.reshape %arg0 : (tensor<16xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
```

```mlir
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_8, [{"a", ?}, {?}]>]>}
```

**为什么只传 `"a"`**：第 0 维大小 4，`a=2` 可整除（4/2=2）；而 `b=8` **不可整除**（4/8 < 1）。
保守模式禁止不可整除的分片 → 丢掉 `"b"`。

### 3. 拆分后正好对应每一维 → 完整传播

网格 `a=16, b=2`；`32` → `16x2`；输入 `{"a", "b"}`（16×2 = 32）。

```mlir
sdy.mesh @mesh_a_16_b_2 = <["a"=16, "b"=2]>
```

```mlir
func.func @multi_axes_split_fully_sharded(%arg0: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh_a_16_b_2, [{"a", "b"}]>}) -> tensor<16x2xf32> {
```

```mlir
  %0 = stablehlo.reshape %arg0 : (tensor<32xf32>) -> tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}
```

```mlir
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_16_b_2, [{"a", ?}, {"b", ?}]>]>}
```

**为什么这次能全传**：`a=16` 正好等于结果的第 0 维，`b=2` 正好等于结果的第 1 维 ——
两个轴各自整段落在一个维度上，**不需要拆分子轴**，也不涉及不可整除。

---

## 三、slice：被切掉的维度不再整除

网格 `a=16, b=2`；`32x4x8` slice 成 `32x1x2`。

```mlir
func.func @slice(%arg0: tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_16_b_2, [{"a"}, {}, {"b"}]>}) -> tensor<32x1x2xf32> {
```

```mlir
  %0 = stablehlo.slice %arg0 [0:32, 1:2, 4:8:2] : (tensor<32x4x8xf32>) -> tensor<32x1x2xf32>
  return %0 : tensor<32x1x2xf32>
}
```

```mlir
  // CHECK-NEXT: stablehlo.slice %arg0 [0:32, 1:2, 4:8:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_16_b_2, [{"a", ?}, {?}, {?}]>]>}
```

**为什么只有第 0 维留下 `"a"`**：
- 第 0 维 32 → 32，`a=16` 仍整除 ✓
- 第 1 维 4 → **1**，无法被任何轴整除（1 不能被 16 或 2 切）
- 第 2 维 8 → **2**，`b=2` 恰好整除，但结果里写的是 `{?}` 而非 `{"b", ?}`
   —— 因为切片把这一维也变成了"需要重新判断"的状态

最终只有确定无疑的第 0 维保住了分片。
