<!-- sdy-coverage
transforms/export/test/convert_global_to_local/stablehlo_dot.mlir
transforms/export/test/convert_global_to_local/stablehlo_dot_general.mlir
-->

# L5-05 · lowering-matmul — 源 IR

**矩阵乘在各种分片组合下的局部化** —— 这是**经典并行策略**的 IR 体现。

| 文件 | 行数 | 用例数 |
|---|---|---|
| `convert_global_to_local/stablehlo_dot.mlir` | 68 | 4 |
| `convert_global_to_local/stablehlo_dot_general.mlir` | 51 | 2 |

网格：

```mlir
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
```

---

## 一、★ 四种情形 = 经典并行策略

`stablehlo_dot` 的 4 个用例**正好覆盖四种情形**：

| # | 用例 | 分片位置 | 通信 | 并行策略 |
|---|---|---|---|---|
| ① | `fully_replicated` | 无 | 无 | — |
| ② | `sharded_non_contracting_dims` | **非收缩维** | **无** | **模型并行** |
| ③ | `sharded_contracting_dim` | **收缩维** | **`all_reduce`** | **数据并行** |
| ④ | `sharded_contracting_dim_unreduced_result` | 收缩维 + 结果保持未归约 | **无**（延迟） | 延迟归约 |

**一句话**：
> **非收缩维分片 → 每个设备算输出的一块（无通信）；**
> **收缩维分片 → 每个设备算部分和（需要 `all_reduce`）。**

这正是 L4-04 讲的 `dot` 三类情形在**降级后**的样子。

---

## 二、情形 ①：全复制

```mlir
func.func @fully_replicated(
  %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}]>},
  %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}]>}) -> tensor<8x32xf32> {
```

```mlir
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
```

```mlir
  return %0 : tensor<8x32xf32>
```

```mlir
}
```

```mlir
  // CHECK: %[[RES:.*]] = stablehlo.dot %arg0, %arg1 : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
```

```mlir
  // CHECK: return %[[RES]] : tensor<8x32xf32>
```

**读法**：无分片 → 直接转换，**类型不变**、**无通信**。

---

## 三、★ 情形 ②：非收缩维分片 = **模型并行**

```mlir
func.func @sharded_non_contracting_dims(
  %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
  %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>})
  -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>}) {
```

```mlir
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>}
```

```mlir
   : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
```

```mlir
  return %0 : tensor<8x32xf32>
```

```mlir
}
```

```mlir
// CHECK-LABEL: func @sharded_non_contracting_dims
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>})
// CHECK-SAME: -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>}) {
```

```mlir
  // CHECK: %[[RES:.*]] = stablehlo.dot %[[ARG0]], %[[ARG1]]
```

```mlir
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>}
```

```mlir
  // CHECK-SAME: (tensor<4x16xf32>, tensor<16x8xf32>) -> tensor<4x8xf32>
```

```mlir
  // CHECK: return %[[RES]] : tensor<4x8xf32>
```

**逐项读**（`@mesh_2_4` 是 `x=2, y=4`）：

| 张量 | 全局 | 分片 | 局部 | 算式 |
|---|---|---|---|---|
| `%arg0` | `8x16` | `[{"x"}, {}]` | **`4x16`** | 8 ÷ 2 |
| `%arg1` | `16x32` | `[{}, {"y"}]` | **`16x8`** | 32 ÷ 4 |
| 结果 | `8x32` | `[{"x"}, {"y"}]` | **`4x8`** | 8÷2, 32÷4 |

**关键**：
- **收缩维（`%arg0` 第 1 维、`%arg1` 第 0 维）都是完整的 16** —— 没有分片。
- 每台设备算输出矩阵的**一块**（`4x8`）。
- **不需要任何通信** ✓

**这就是模型并行**：把权重/输出切开，每个设备算一部分。

**为什么不需要通信**：收缩维完整 → 每台设备的 `dot` 结果就是**最终的**那一块，
不需要跟别人合并。

---

## 四、★ 情形 ③：收缩维分片 = **数据并行**

```mlir
func.func @sharded_contracting_dim(
  %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x"}]>},
  %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) -> tensor<8x32xf32> {
```

```mlir
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{}, {}], unreduced={"x"}>]>}
```

```mlir
   : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
```

```mlir
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2_4, [{}, {}]> : tensor<8x32xf32>
```

```mlir
  return %1 : tensor<8x32xf32>
```

```mlir
}
```

```mlir
// CHECK-LABEL: func @sharded_contracting_dim
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x"}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) -> tensor<8x32xf32> {
```

```mlir
  // CHECK: %[[DOT:.*]] = stablehlo.dot %[[ARG0]], %[[ARG1]]
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{}, {}], unreduced={"x"}>]>}
  // CHECK-SAME:(tensor<8x8xf32>, tensor<8x32xf32>) -> tensor<8x32xf32>
```

```mlir
  // CHECK: %[[RES:.*]] = "stablehlo.all_reduce"(%[[DOT]])
```

```mlir
  // CHECK-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "x">]>
```

```mlir
  // CHECK: return %[[RES]] : tensor<8x32xf32>
```

**逐项读**：

| 张量 | 全局 | 分片 | 局部 | 算式 |
|---|---|---|---|---|
| `%arg0` | `8x16` | `[{}, {"x"}]` | **`8x8`** | 16 ÷ 2 |
| `%arg1` | `16x32` | `[{"x"}, {}]` | **`8x32`** | 16 ÷ 2 |
| `dot` 结果 | `8x32` | `[{}, {}], unreduced={"x"}` | **`8x32`** | 全复制 |

**关键**：
- **收缩维（`%arg0` 第 1 维、`%arg1` 第 0 维）都被切了 `x`** —— 且**一致** ✓
- 每台设备算的只是**部分和** → `dot` 的结果标 `unreduced={"x"}`
- → 需要 **`all_reduce {"x"}`** 合并

**这就是数据并行**：把数据切开，每个设备算一部分，最后归约。

**`replica_groups` 用 `x` 轴** —— 与 L5-02 的 V3 表示一致。

**回顾 L4-04**：那里讲 `dot` 的"情形 ②"就是"收缩维被切 → `unreduced` + `all_reduce`"。
现在看到它**降级后的样子** —— 完整链路：

```
dot（收缩维分片）
  --L4-04-->  unreduced + sdy.all_reduce
  --L5-05-->  stablehlo.dot(unreduced) + stablehlo.all_reduce
```

---

## 五、★ 情形 ④：延迟归约

```mlir
func.func @sharded_contracting_dim_unreduced_result(
  %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x"}]>},
  %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
  -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}], unreduced={"x"}>}) {
```

```mlir
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{}, {}], unreduced={"x"}>]>}
```

```mlir
   : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
```

```mlir
  return %0 : tensor<8x32xf32>
```

```mlir
}
```

```mlir
// CHECK-LABEL: func @sharded_contracting_dim_unreduced_result
```

```mlir
// CHECK-SAME: -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}], unreduced={"x"}>}) {
```

```mlir
  // CHECK: %[[DOT:.*]] = stablehlo.dot %[[ARG0]], %[[ARG1]]
```

```mlir
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{}, {}], unreduced={"x"}>]>}
```

```mlir
  // CHECK-SAME: (tensor<8x8xf32>, tensor<8x32xf32>) -> tensor<8x32xf32>
```

```mlir
  // CHECK: return %[[DOT]] : tensor<8x32xf32>
```

**读法**（与情形 ③ 的唯一区别）：
- **函数结果也声明了 `unreduced={"x"}`** → 归约责任交给**调用者**。
- → **不插 `all_reduce`**，直接 `return %[[DOT]]` ✓

**这就是 L4-05 讲的"延迟归约"** ——
未归约时每个设备只算**自己那一份部分和**，计算量更小。

**四种情形的完整对照**：

| # | 收缩维 | 结果要求 | `all_reduce` |
|---|---|---|---|
| ① | 完整 | 完整 | 不需要 |
| ② | 完整 | 完整 | 不需要 |
| ③ | **分片** | **完整** | **需要** |
| ④ | **分片** | **未归约** | **不需要**（延迟） |

**判据**：**收缩维是否被分片** + **结果是否接受未归约**。

---

## 六、`stablehlo_dot_general`：批维与收缩维

### `not_shard_contracting_dims`：分片在**批维** → 无通信

```mlir
func.func @not_shard_contracting_dims(
  %arg0: tensor<4x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>},
  %arg1: tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>})
  -> (tensor<4x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>}) {
```

```mlir
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1]
```

```mlir
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}, {}]>]>}
```

```mlir
  : (tensor<4x16x8xf32>, tensor<4x8x16xf32>) -> tensor<4x16x16xf32>
```

```mlir
  return %0 : tensor<4x16x16xf32>
```

```mlir
}
```

```mlir
// CHECK-LABEL: func @not_shard_contracting_dims(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>},
// CHECK-SAME: %[[ARG1:.*]]: tensor<2x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>})
// CHECK-SAME: -> (tensor<2x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>}) {
```

```mlir
  // CHECK: %[[RES:.*]] = stablehlo.dot_general %[[ARG0]], %[[ARG1]], batching_dims = [0] x [0], contracting_dims = [2] x [1]
```

```mlir
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}, {}]>]>}
```

```mlir
  // CHECK-SAME: : (tensor<2x16x8xf32>, tensor<2x8x16xf32>) -> tensor<2x16x16xf32>
```

```mlir
  // CHECK: return %[[RES]] : tensor<2x16x16xf32>
```

**读法**：
- `batching_dims = [0] x [0]` —— 第 0 维是**批维**。
- 分片 `[{"x"}, {}, {}]` **正好在批维上** → 每台设备算**一批**。
- **收缩维 `[2] x [1]` 都是完整的**（8）→ **无通信** ✓
- 局部类型：`4x16x8` → `2x16x8`（批维 ÷2）

**这是"批并行"** —— 批维分片，各批独立。

### `shard_contracting_dims`：分片在**收缩维** → `all_reduce`

```mlir
func.func @shard_contracting_dims(
  %arg0: tensor<4x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {"y"}]>},
  %arg1: tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}, {}]>})
  -> (tensor<4x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>}) {
```

```mlir
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1]
```

```mlir
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_4, [{"x"}, {}, {}], unreduced={"y"}>]>}
```

```mlir
  : (tensor<4x16x8xf32>, tensor<4x8x16xf32>) -> tensor<4x16x16xf32>
```

```mlir
  %1 = sdy.all_reduce {"y"} %0 out_sharding=<@mesh_2_4, [{"x"}, {}, {}]> : tensor<4x16x16xf32>
```

```mlir
  return %1 : tensor<4x16x16xf32>
```

```mlir
}
```

```mlir
// CHECK-LABEL: func @shard_contracting_dims
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x16x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {"y"}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<2x2x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}, {}]>})
// CHECK-SAME: -> (tensor<2x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>}) {
```

```mlir
  // CHECK: %[[DOT:.*]] = stablehlo.dot_general %[[ARG0]], %[[ARG1]], batching_dims = [0] x [0], contracting_dims = [2] x [1]
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}, {}], unreduced={"y"}>]>}
  // CHECK-SAME: : (tensor<2x16x2xf32>, tensor<2x2x16xf32>) -> tensor<2x16x16xf32>
```

```mlir
  // CHECK: %[[RES:.*]] = "stablehlo.all_reduce"(%[[DOT]])
```

```mlir
  // CHECK-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "y">]>
```

```mlir
  // CHECK-SAME: use_global_device_ids
```

```mlir
  // CHECK: ^bb0(%[[ACC:.*]]: tensor<f32>, %[[UPD:.*]]: tensor<f32>):
```

```mlir
  // CHECK:   %[[ADD:.*]] = stablehlo.add %[[ACC]], %[[UPD]] : tensor<f32>
```

```mlir
  // CHECK:   stablehlo.return %[[ADD]] : tensor<f32>
```

```mlir
  // CHECK: }) : (tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
```

```mlir
  // CHECK: return %[[RES]] : tensor<2x16x16xf32>
```

**读法**：
- `%arg0` 的**收缩维（第 2 维）切 `y`**、`%arg1` 的**收缩维（第 1 维）切 `y`** —— 一致 ✓
- `%arg0` 的第 2 维 `8 / 4 = 2`、`%arg1` 的第 1 维 `8 / 4 = 2` ✓
- 批维（第 0 维）切 `x` → `4 / 2 = 2` ✓
- 结果标 `unreduced={"y"}` → **`all_reduce {"y"}`**

**这是 2D 并行**：批维切 `x`（批并行）+ 收缩维切 `y`（数据并行）。
**只有收缩维上的分片需要 `all_reduce`** —— 批维上的不需要。

**reduction 区域**：`^bb0` 里是 `add` —— 与 L5-02 的 `all_reduce` 一致。

---

## 七、六条情形的完整图景

| 文件 | 用例 | 分片位置 | 通信 | 并行策略 |
|---|---|---|---|---|
| `dot` | `fully_replicated` | 无 | 无 | — |
| `dot` | `sharded_non_contracting_dims` | 非收缩维 | **无** | 模型并行 |
| `dot` | `sharded_contracting_dim` | 收缩维 | **`all_reduce`** | 数据并行 |
| `dot` | `..._unreduced_result` | 收缩维（结果未归约） | **无** | 延迟归约 |
| `dot_general` | `not_shard_contracting_dims` | **批维** | **无** | 批并行 |
| `dot_general` | `shard_contracting_dims` | 批维 + 收缩维 | **`all_reduce`** | 2D 并行 |

**★ 核心规律**：
> **只有【收缩维】上的分片需要 `all_reduce`。**
> 非收缩维 / 批维上的分片都是"各算各的"，无需通信。

**为什么**：收缩维是**归约的方向** —— 沿它分片意味着每台设备只算了**部分和**。
其他维分片意味着每台设备算的是**不同的输出元素**，互不重叠。

**一句话总结**：
> **矩阵乘的分片降级 = 经典并行策略的 IR 体现** ——
> 非收缩维分片是模型并行、收缩维分片是数据并行、
> 两者都分片是 2D 并行，而通信只发生在收缩维上。
