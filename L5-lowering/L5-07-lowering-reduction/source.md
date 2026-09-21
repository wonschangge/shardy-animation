<!-- sdy-coverage
transforms/export/test/convert_global_to_local/stablehlo_reduce.mlir
transforms/export/test/convert_global_to_local/stablehlo_reduce_window.mlir
-->

# L5-07 · lowering-reduction — 源 IR

**归约类算子的降级**：局部归约 + 跨设备合并。

| 文件 | 行数 | 用例数 |
|---|---|---|
| `convert_global_to_local/stablehlo_reduce.mlir` | 88 | 3 |
| `convert_global_to_local/stablehlo_reduce_window.mlir` | 60 | 2 |

网格：

```mlir
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
```

---

## 一、★ `stablehlo_reduce` 的三种情形

| 用例 | 归约维 | 通信 | 方式 |
|---|---|---|---|
| `stablehlo_reduce` | 无分片 | **无** | 直接转换 |
| `..._unreduced_axes_fallback_all_reduce` | **被分片** | **`all_reduce`** | 局部归约 + 全局归约 |
| `..._multi_result_sharded` | **被分片** + 多结果 | **`all_gather` + 重新 reduce** | 收齐数据再归约 |

**核心问题**：
> 归约是**跨元素**的操作 —— 如果归约维被分片，
> 每台设备只能归约**自己那段** → 需要跨设备合并。

**两种合并方式**（本课的重点）：
- **单结果** → `all_reduce`（归约可结合，直接合并部分结果）
- **多结果** → **不能**用 `all_reduce` → 改用 `all_gather` + **重新 reduce**

---

## 二、情形 ①：无分片

```mlir
func.func @stablehlo_reduce(%arg0: tensor<32x8xi32>)
    -> (tensor<32xi32>) {
```

```mlir
  %cst = stablehlo.constant dense<0> : tensor<i32>
```

```mlir
  %0 = "stablehlo.reduce"(%arg0, %cst) ({
```

```mlir
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
```

```mlir
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
```

```mlir
      %2 = stablehlo.add %1, %arg2 : tensor<i32>
```

```mlir
      stablehlo.return %2 : tensor<i32>
```

```mlir
  }) {
```

```mlir
    dimensions = array<i64: 1>
```

```mlir
  }: (tensor<32x8xi32>, tensor<i32>) -> tensor<32xi32>
```

```mlir
  return %0 : tensor<32xi32>
```

```mlir
}
```

```mlir
// CHECK-LABEL: func @stablehlo_reduce
// CHECK-SAME: (%[[ARG0:.*]]: tensor<32x8xi32>)
// CHECK-SAME: -> tensor<32xi32> {
```

```mlir
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
```

```mlir
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.reduce(%[[ARG0]] init: %[[CST]]) across dimensions = [1] : (tensor<32x8xi32>, tensor<i32>) -> tensor<32xi32>
```

```mlir
  // CHECK-NEXT:  reducer(%[[ARG1:.*]]: tensor<i32>, %[[ARG2:.*]]: tensor<i32>) {
```

```mlir
  // CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %[[ARG1]], %[[ARG2]] : tensor<i32>
```

```mlir
  // CHECK-NEXT:    %[[ADD_2:.*]] = stablehlo.add %[[ADD]], %[[ARG2]] : tensor<i32>
```

```mlir
  // CHECK-NEXT:    stablehlo.return %[[ADD_2]] : tensor<i32>
```

```mlir
  // CHECK-NEXT:  }
```

```mlir
  // CHECK-NEXT: return %[[RES]] : tensor<32xi32>
```

**读法**：
- 无分片 → 直接转换，**类型不变**（`32x8` → `32`）。
- **reducer 区域原样保留** —— 注意它内部有两个 `add`（一个**非标准**的归约函数）。
- **这说明转换不关心 reducer 的内容** —— 只改类型。

**为什么 reducer 里有奇怪的加法**：这是测试故意写的 ——
验证转换器**不分析 reducer 语义**，只做类型/分片层面的处理。

---

## 三、★ 情形 ②：归约维分片 → `all_reduce`

```mlir
func.func @stablehlo_reduce_unreduced_axes_fallback_all_reduce(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>})
    -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}]>}) {
```

```mlir
  %init = stablehlo.constant dense<1.0> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %init) across dimensions = [1]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}]>]>}
      : (tensor<8x16xf32>, tensor<f32>) -> tensor<8xf32>
    reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %mul = stablehlo.multiply %lhs, %rhs : tensor<f32>
      stablehlo.return %mul : tensor<f32>
    }
  return %0 : tensor<8xf32>
}
```

```mlir
// CHECK-LABEL: func.func @stablehlo_reduce_unreduced_axes_fallback_all_reduce
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>})
// CHECK-SAME: -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}]>})
```

```mlir
  // CHECK-NEXT: %[[INIT:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
```

```mlir
  // CHECK-NEXT: %[[LOCAL_RED:.*]] = stablehlo.reduce(%[[ARG0]] init: %[[INIT]]) applies stablehlo.multiply across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}]>]>} : (tensor<4x4xf32>, tensor<f32>) -> tensor<4xf32>
```

```mlir
  // CHECK:      %[[ALL_RED:.*]] = "stablehlo.all_reduce"(%[[LOCAL_RED]])
```

```mlir
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>
```

```mlir
  // CHECK-SAME:   replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "y">]>
```

```mlir
  // CHECK-SAME:   use_global_device_ids
```

```mlir
  // CHECK:      return %[[ALL_RED]] : tensor<4xf32>
```

**逐项读**：

| 张量 | 全局 | 分片 | 局部 | 说明 |
|---|---|---|---|---|
| `%arg0` | `8x16` | `[{"x"}, {"y"}]` | **`4x4`** | 两维都切 |
| 局部归约结果 | `8` | `[{"x"}]` | **`4`** | 归约第 1 维后 |
| 最终结果 | `8` | `[{"x"}]` | **`4`** | `all_reduce` 后 |

**三步**：
1. **局部归约**：`reduce ... across dimensions = [1]` 在 `4x4` 上做 → `4`
2. **`all_reduce`**：把各设备的部分结果合并（`replica_groups` 用 `y` 轴）
3. **返回**

**为什么 `all_reduce` 可行**：
归约是**可结合**的 —— `reduce(reduce(a), reduce(b)) = reduce(a ++ b)`。
所以"局部归约 + 全局归约"= "全局归约" ✓

**用例名里的 `fallback`**（退化方案）：
说明这是**通用但可能不是最优**的做法 ——
对于 `max` / `sum` 这类可结合的归约，`all_reduce` 是对的；
但如果是**不可结合**的归约（如求中位数），就需要别的方式。

**reducer 是 `multiply`** —— 注意这里不是加法，但 `all_reduce` 仍然可行
（乘法也可结合）。

---

## 四、★ 情形 ③：多结果归约 → `all_gather` + 重新 reduce

**这是本课最精妙的一处。**

```mlir
func.func @stablehlo_reduce_multi_result_sharded(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>},
    %arg1: tensor<8x16xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>})
    -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}]>},
        tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}]>}) {
```

```mlir
  %init0 = stablehlo.constant dense<0.0> : tensor<f32>
  %init1 = stablehlo.constant dense<0> : tensor<i32>
  %0:2 = stablehlo.reduce(%arg0 init: %init0), (%arg1 init: %init1) across dimensions = [1]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}]>, <@mesh_2_4, [{"x"}]>]>}
      : (tensor<8x16xf32>, tensor<8x16xi32>, tensor<f32>, tensor<i32>) -> (tensor<8xf32>, tensor<8xi32>)
    reducer(%lhs0: tensor<f32>, %rhs0: tensor<f32>) (%lhs1: tensor<i32>, %rhs1: tensor<i32>) {
      %cmp = stablehlo.compare GT, %lhs0, %rhs0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %max_val = stablehlo.select %cmp, %lhs0, %rhs0 : tensor<i1>, tensor<f32>
      %max_idx = stablehlo.select %cmp, %lhs1, %rhs1 : tensor<i1>, tensor<i32>
      stablehlo.return %max_val, %max_idx : tensor<f32>, tensor<i32>
    }
  return %0#0, %0#1 : tensor<8xf32>, tensor<8xi32>
}
```

```mlir
// CHECK-LABEL: func.func @stablehlo_reduce_multi_result_sharded
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>})
// CHECK-SAME: -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}]>},
// CHECK-SAME:     tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}]>})
```

```mlir
  // CHECK-NEXT: %[[INIT0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[INIT1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[LOCAL_RED:.*]]:2 = stablehlo.reduce(%[[ARG0]] init: %[[INIT0]]), (%[[ARG1]] init: %[[INIT1]]) across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}]>, <@mesh_2_4, [{"x"}]>]>} : (tensor<4x4xf32>, tensor<4x4xi32>, tensor<f32>, tensor<i32>) -> (tensor<4xf32>, tensor<4xi32>)
  // CHECK:      %[[RESHAPE0:.*]] = stablehlo.reshape %[[LOCAL_RED]]#0 : (tensor<4xf32>) -> tensor<4x1xf32>
  // CHECK-NEXT: %[[GATHER0:.*]] = "stablehlo.all_gather"(%[[RESHAPE0]]) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "y">]>, use_global_device_ids}> : (tensor<4x1xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT: %[[RESHAPE1:.*]] = stablehlo.reshape %[[LOCAL_RED]]#1 : (tensor<4xi32>) -> tensor<4x1xi32>
  // CHECK-NEXT: %[[GATHER1:.*]] = "stablehlo.all_gather"(%[[RESHAPE1]]) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "y">]>, use_global_device_ids}> : (tensor<4x1xi32>) -> tensor<4x4xi32>
  // CHECK-NEXT: %[[FINAL_RED:.*]]:2 = stablehlo.reduce(%[[GATHER0]] init: %[[INIT0]]), (%[[GATHER1]] init: %[[INIT1]]) across dimensions = [1] : (tensor<4x4xf32>, tensor<4x4xi32>, tensor<f32>, tensor<i32>) -> (tensor<4xf32>, tensor<4xi32>)
  // CHECK:      return %[[FINAL_RED]]#0, %[[FINAL_RED]]#1 : tensor<4xf32>, tensor<4xi32>
```

### 逐项读（五步）

| 步 | 操作 | 作用 |
|---|---|---|
| ① | **局部 `reduce`** | 每台设备归约自己那段 → `4xf32` + `4xi32` |
| ② | **`reshape`** → `4x1` | 为 `all_gather` 准备维度 |
| ③ | **`all_gather`**（两个各一次） | 把各设备的结果**收齐** → `4x4` |
| ④ | **再次 `reduce`** | 在**完整数据**上重新归约 → `4` |
| ⑤ | **`return`** | 两个结果 |

**为什么不能直接用 `all_reduce`**（本课的核心问题）：
- reducer 是 **argmax 模式** —— 它同时算出**最大值**和**最大值的位置**。
- 两个结果**互相依赖**：`%max_idx` 的选择依赖 `%cmp`（由 `%lhs0`/`%rhs0` 决定）。
- 如果分别对 `%max_val` 和 `%max_idx` 做 `all_reduce`，
  **两者的"最大值来自哪个设备"可能不一致** → 结果错误！
- → 必须**把数据收齐**（`all_gather`），在**完整数据**上重新归约一次。

**注意 `channel_handle` 的编号不同**：
- `GATHER0` 用 `handle = 2`
- `GATHER1` 用 `handle = 3`
→ 两次 gather 用**不同的通道**（避免冲突）。

**注意 `handle = 1` 出现在情形 ② 的 `all_reduce`** ——
说明通道编号是**递增分配**的。

**两个结果分别 reshape + gather**：
`%LOCAL_RED#0`（f32）与 `%LOCAL_RED#1`（i32）**类型不同**，
不能合并成一次 gather → 各自做一次。

**★ 这就是"多结果归约"的通用解法**：
> **收齐数据 + 重新归约** —— 代价更高，但语义正确。

---

## 五、`stablehlo_reduce_window`：窗口是否**跨设备**

2 个用例：

### `batch_sharded`：批维分片 → **无通信**

```mlir
func.func @batch_sharded(
    %arg0: tensor<32x16x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>})
    -> (tensor<32x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}) {
```

```mlir
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
```

```mlir
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{
```

```mlir
    padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>,
```

```mlir
    window_dimensions = array<i64: 1, 3, 3, 1>,
```

```mlir
    window_strides = array<i64: 1, 2, 2, 1>
```

```mlir
  }> ({
```

```mlir
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
```

```mlir
    %1 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
```

```mlir
    stablehlo.return %1 : tensor<f32>
```

```mlir
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>}
```

```mlir
  : (tensor<32x16x16x8xf32>, tensor<f32>) -> tensor<32x8x8x8xf32>
```

```mlir
  return %0 : tensor<32x8x8x8xf32>
```

```mlir
}
```

```mlir
// CHECK-LABEL: func @batch_sharded
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x16x16x8xf32>
// CHECK-SAME: -> (tensor<16x8x8x8xf32>
```

```mlir
  // CHECK-NEXT: %[[RES:.*]] = "stablehlo.reduce_window"(%[[ARG0]], %[[CST]]) <{
  // CHECK-SAME:   padding = dense<{{\[\[}}0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>,
  // CHECK-SAME:   window_dimensions = array<i64: 1, 3, 3, 1>,
  // CHECK-SAME:   window_strides = array<i64: 1, 2, 2, 1>}> ({
  // CHECK-NEXT: ^bb0(%[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<f32>):
  // CHECK-NEXT:   %[[MAX:.*]] = stablehlo.maximum %[[ARG1]], %[[ARG2]] : tensor<f32>
  // CHECK-NEXT:   stablehlo.return %[[MAX]] : tensor<f32>
  // CHECK-NEXT: })
  // CHECK-SAME: (tensor<16x16x16x8xf32>, tensor<f32>) -> tensor<16x8x8x8xf32>
```

**读法**：
- **批维（第 0 维）分片** → `32/2 = 16`。
- **`window_dimensions = [1, 3, 3, 1]`** —— **批维的窗口大小是 1**！
  → 窗口**不跨批** → 分片无通信 ✓
- **`padding` / `window_dimensions` / `window_strides` 全部不变**。

### `reduce_window_stride_greater_than_window`：**步长 ≥ 窗口** → 仍无通信

```mlir
func.func @reduce_window_stride_greater_than_window(
    %arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>})
    -> (tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
```

```mlir
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
```

```mlir
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{
```

```mlir
    padding = dense<0> : tensor<2x2xi64>,
```

```mlir
    window_dimensions = array<i64: 1, 2>,
```

```mlir
    window_strides = array<i64: 1, 2>
```

```mlir
  }> ({
```

```mlir
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
```

```mlir
    %1 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
```

```mlir
    stablehlo.return %1 : tensor<f32>
```

```mlir
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>}
```

```mlir
  : (tensor<16x32xf32>, tensor<f32>) -> tensor<16x16xf32>
```

```mlir
  return %0 : tensor<16x16xf32>
```

```mlir
}
```

```mlir
// CHECK-LABEL: func @reduce_window_stride_greater_than_window
// CHECK-SAME: (%[[ARG0:[^ :]+]]: tensor<16x16xf32> {{.*}}) -> (tensor<16x8xf32> {{.*}})
```

```mlir
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[RES:.*]] = "stablehlo.reduce_window"(%[[ARG0]], %[[CST]]) <{
  // CHECK-SAME:   padding = dense<0> : tensor<2x2xi64>,
  // CHECK-SAME:   window_dimensions = array<i64: 1, 2>,
  // CHECK-SAME:   window_strides = array<i64: 1, 2>}>
```

**读法**（**本课最巧妙的用例**）：
- **第 1 维被分片**（`{"x"}`，`x=2`），而**第 1 维恰好有窗口**（`window_dimensions = [1, 2]`）！
- **表面上看应该需要通信** —— 因为窗口跨越了分片边界？
- **但 `window_strides = [1, 2]`** —— **步长等于窗口大小**！
- → **每个窗口恰好落在一个设备的分片内**，**不跨设备** ✓
- → **无通信** ✓

**逐项验证**：
- 全局 `16x32`，第 1 维切 `x=2` → 每台 `16x16`。
- 窗口大小 2、步长 2 → 每台设备产生 `16/2 = 8` 个输出。
- 设备 0 处理第 0~15 列（输出第 0~7 个），设备 1 处理第 16~31 列（输出第 8~15 个）。
- 窗口 `[0,1]`、`[2,3]`…… 全部落在同一设备内 ✓

**★ 规律**：
> **`reduce_window` 需要通信的唯一情形是：窗口跨越设备边界。**
> 当 **`stride ≥ window_dimensions`**（步长不小于窗口）时，
> 窗口不会跨界 → 无需通信。

---

## 六、两个文件的对照与规律

| 文件 | 用例 | 分片位置 | 通信 |
|---|---|---|---|
| `reduce` | `stablehlo_reduce` | 无 | 无 |
| `reduce` | `..._fallback_all_reduce` | **归约维** | **`all_reduce`** |
| `reduce` | `..._multi_result_sharded` | 归约维 + 多结果 | **`all_gather` + reduce** |
| `reduce_window` | `batch_sharded` | 批维（窗口=1） | 无 |
| `reduce_window` | `..._stride_greater_than_window` | 窗口维（**步长≥窗口**） | **无** |

**★ 与前面几课统一的规律**：
> **归约类算子的通信需求，取决于"归约是否跨设备"。**
> - `reduce`：归约维被分片 → **跨设备** → 需要通信
> - `reduce_window`：窗口**跨设备边界** → 需要通信；否则不需要

**两种合并方式的选择**：
| 情形 | 方式 | 为什么 |
|---|---|---|
| **单结果** | `all_reduce` | 归约可结合，直接合并部分结果 |
| **多结果** | `all_gather` + 重新 `reduce` | 结果间有依赖，不能分别合并 |

**一句话总结**：
> **归约类降级 = 局部归约 + 跨设备合并**；
> 合并方式取决于**结果个数**（单结果用 `all_reduce`，多结果必须收齐再算）。
