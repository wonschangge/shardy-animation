<!-- sdy-coverage
transforms/export/test/convert_global_to_local/stablehlo_gather.mlir
transforms/export/test/convert_global_to_local/stablehlo_scatter.mlir
transforms/export/test/convert_global_to_local/stablehlo_select_and_scatter.mlir
-->

# L5-08 · lowering-gather-scatter — 源 IR

**访存类算子的降级** —— 本目录中**最复杂**的一族。

| 文件 | 行数 | 用例数 |
|---|---|---|
| `convert_global_to_local/stablehlo_gather.mlir` | 463 | 11 |
| `convert_global_to_local/stablehlo_scatter.mlir` | 416 | 9 |
| `convert_global_to_local/stablehlo_select_and_scatter.mlir` | 77 | 2 |

**合计 956 行 / 22 用例。**

网格：

```mlir
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
```

---

## 一、★ 核心问题：**索引重映射**

**为什么访存类算子最难**：
> `gather` / `scatter` 的**索引是全局坐标**，
> 但分片后每台设备只持有**局部数据**。
> → 索引必须**转换到本地坐标系**！

**与前面几课的对比**：

| 课 | 问题 | 解法 |
|---|---|---|
| L5-04 `iota` | **序号**起点不同 | 本地 iota + 加偏移 |
| L5-07 `reduce` | **部分结果**需要合并 | `all_reduce` / 收齐再算 |
| **L5-08 `gather/scatter`** | **索引**指向别的设备 | **索引重映射 + mask** |

**为什么最复杂**：不仅要把索引**平移**到本地坐标系，
还要处理"**这个索引根本不属于我**"的情况 —— 需要 **mask + select 填零**。

---

## 二、★ `gather`：归约维被 `collapsed` 时的完整流程

### 用例：`shard_reduction_dim_is_collapsed`

规则（测试注释给出）：

```
// ([i, j], [k]) -> ([k, j]) reduction={i}
```

```mlir
func.func @shard_reduction_dim_is_collapsed(
  %arg0: tensor<8x10xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
  %arg1: tensor<2xi64>) -> tensor<2x10xf32> {
```

```mlir
  %0 = "stablehlo.gather"(%arg0, %arg1) {
```

```mlir
    dimension_numbers = #stablehlo.gather<
```

```mlir
      offset_dims = [1],
```

```mlir
      collapsed_slice_dims = [0],
```

```mlir
      start_index_map = [0],
```

```mlir
      index_vector_dim = 1>,
```

```mlir
    slice_sizes = array<i64: 1, 10>,
```

```mlir
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_4, [{}, {}], unreduced={"x"}>]>
```

```mlir
  } : (tensor<8x10xf32>, tensor<2xi64>) -> tensor<2x10xf32>
```

```mlir
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2_4, [{}, {}]> : tensor<2x10xf32>
```

```mlir
  return %1 : tensor<2x10xf32>
```

```mlir
}
```

**问题**：
- `%arg0` 的**第 0 维（`i`，归约维）切了 `x=2`** → 每台设备只有 **4 行**（`8/2`）。
- `%arg1` 是**索引**（`tensor<2xi64>`），值是**全局坐标**（0~7）。
- 但设备 1 只持有第 4~7 行 —— 它无法处理索引 `2`！

### 解法：**十一行 IR 的索引重映射**

```mlir
  // CHECK-DAG: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<2xi64>
  // CHECK-DAG: %[[C7:.*]] = stablehlo.constant dense<7> : tensor<2xi64>
  // CHECK: %[[CLAMPED:.*]] = stablehlo.clamp %[[C0]], %[[ARG1]], %[[C7]] : tensor<2xi64>
```

**第 ① 步：`clamp` 到全局范围 `[0, 7]`** —— 越界索引先夹紧。

```mlir
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[CVT_PID:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 0, 0, 0, 4, 4, 4, 4]> : tensor<8xi64>
  // CHECK: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[CVT_PID]], sizes = [1]
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[OFFSET:.*]] = stablehlo.convert %[[RESHAPE]] : tensor<i64>
```

**第 ② 步：查表得到本设备的偏移** ——
表 `[0,0,0,0,4,4,4,4]`：设备 0~3 偏移 0、设备 4~7 偏移 4。
（与 L5-01/L5-04/L5-07 同一套路，但这次是**索引的偏移**。）

```mlir
  // CHECK: %[[C3:.*]] = stablehlo.constant dense<3> : tensor<i64>
  // CHECK: %[[LIMIT:.*]] = stablehlo.add %[[OFFSET]], %[[C3]] : tensor<i64>
  // CHECK: %[[BCAST_OFF:.*]] = stablehlo.broadcast_in_dim %[[OFFSET]], dims = []
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{}]>]>} : (tensor<i64>) -> tensor<2xi64>
  // CHECK: %[[BCAST_LIM:.*]] = stablehlo.broadcast_in_dim %[[LIMIT]], dims = []
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{}]>]>} : (tensor<i64>) -> tensor<2xi64>
```

**第 ③ 步：算出本地范围 `[OFFSET, OFFSET+3]`** ——
每台 4 行，所以最大索引是 `OFFSET + 3`。

**注意**：`OFFSET`/`LIMIT` 是**标量**（`tensor<i64>`），
但索引是 `tensor<2xi64>` → 用 **`broadcast_in_dim` 广播**。

```mlir
  // CHECK: %[[LOCAL_IDX:.*]] = stablehlo.subtract %[[CLAMPED]], %[[BCAST_OFF]] : tensor<2xi64>
```

**第 ④ 步：索引转换到本地坐标系** —— `LOCAL_IDX = CLAMPED - OFFSET`。
设备 1 的索引 `5` → 本地 `5 - 4 = 1` ✓

```mlir
  // CHECK: %[[GE:.*]] = stablehlo.compare GE, %[[CLAMPED]], %[[BCAST_OFF]]
  // CHECK: %[[LE:.*]] = stablehlo.compare LE, %[[CLAMPED]], %[[BCAST_LIM]]
  // CHECK: %[[MASK:.*]] = stablehlo.and %[[GE]], %[[LE]] : tensor<2xi1>
```

**第 ⑤ 步：生成 mask** —— 判断"这个索引是否落在我的范围内"。
`GE`（≥ 下界）**且** `LE`（≤ 上界）→ 属于我。

```mlir
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[ARG0]], %[[LOCAL_IDX]])
  // CHECK-SAME: dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>
  // CHECK-SAME: slice_sizes = array<i64: 1, 10>
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{}, {}], unreduced={"x"}>]>}
```

**第 ⑥ 步：用本地索引 gather** —— 结果标 `unreduced={"x"}`（部分结果）。

```mlir
  // CHECK: %[[MASK_BCAST:.*]] = stablehlo.broadcast_in_dim %[[MASK]], dims = [0] : (tensor<2xi1>) -> tensor<2x10xi1>
  // CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<2x10xf32>
  // CHECK: %[[SEL:.*]] = stablehlo.select %[[MASK_BCAST]], %[[GATHER]], %[[ZERO]] : tensor<2x10xi1>, tensor<2x10xf32>
```

**第 ⑦ 步：`select` 填零** ——
**不属于我的位置填 0**（因为 `all_reduce` 用加法，填 0 不影响结果）。

```mlir
  // CHECK: %[[RES:.*]] = "stablehlo.all_reduce"(%[[SEL]])
```

```mlir
  // CHECK-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "x">]>
```

```mlir
  // CHECK: return %[[RES]] : tensor<2x10xf32>
```

**第 ⑧ 步：`all_reduce`** —— 把各设备的部分结果相加。

### ★ 完整流程小结

| 步 | 操作 | 作用 |
|---|---|---|
| ① | `clamp` | 索引夹紧到全局范围 |
| ② | `partition_id` + 查表 | 得到本设备**偏移** |
| ③ | `broadcast` | 标量 → 张量 |
| ④ | `subtract` | **索引转换到本地坐标系** |
| ⑤ | `compare` + `and` | 生成 **mask**（是否属于我） |
| ⑥ | `gather` | 用**本地索引** |
| ⑦ | `select` + 0 | **不属于我的填零** |
| ⑧ | `all_reduce` | 合并部分结果 |

**为什么填零是安全的**：`all_reduce` 用**加法** —— 填 0 不影响和。

**这比 L5-04 的 iota 复杂得多**：iota 只需平移（每台设备都有对应的序号），
而 gather 的索引**可能指向任何设备** → 必须用 mask 排除。

---

## 三、`gather` 的 11 个用例

| 用例 | 场景 |
|---|---|
| `operand_replicated` | 操作数全复制 |
| `operand_sharded_pass_through_dim` | 分片在 **pass-through 维**（无通信） |
| `shard_reduction_dim_is_collapsed` | **归约维被 collapsed**（本课主例） |
| `shard_reduction_dim_is_collapsed_i32` | 同上（i32 索引） |
| `shard_reduction_dim_is_collapsed_not_in_start_index_map` | collapsed 但不在 `start_index_map` |
| `shard_reduction_dim_is_collapsed_explicit_scalar_index_vector_dim` | 标量索引 |
| `shard_reduction_dim_not_collapsed` | **归约维未被 collapsed** |
| `shard_reduction_dim_explicit_scalar_indices` | 标量索引 |
| `shard_two_of_three_reduction_dims` | **三个归约维中切两个** |
| `shard_two_of_three_reduction_dims_one_not_in_start_index_map` | 同上变体 |
| `gather_unreduced` | 结果未归约（延迟） |

**关键区分**：**归约维是否被 `collapsed`** ——
`collapsed_slice_dims` 里的维是"被压掉"的维（`slice_sizes` 里为 1），
它们对应 `sharding_rule` 的 `reduction` 因子。

**回顾 L2-10**：`gather` 的规则里有 `reduction={i, p}` 与
**`blocked_propagation={k}`** —— 本课看到这些因子的**实际后果**。

**回顾 L3-11**：那里讲"把隐式批维显式化"，
因为隐式批维落在 `blocked_propagation` 上。
本课的例子用的都是**已显式化**的 `gather`。

---

## 四、`scatter`：**被索引的维**分片

### 用例：`input_sharded_on_indexed_inserted__window_dim`

```mlir
func.func @input_sharded_on_indexed_inserted__window_dim(
    %arg0: tensor<3x4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x"}, {}]>},
    %arg1: tensor<2x3x2xi64>,
    %arg2: tensor<2x3x1xf32>)
 -> (tensor<3x4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x"}, {}]>}) {
```

```mlir
  // CHECK: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[CVT_PID:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 0, 0, 0, 2, 2, 2, 2]> : tensor<8xi64>
  // CHECK: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[CVT_PID]], sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
```

**读法**：
- `%arg0` 的**第 1 维被分片**（`{"x"}`）—— 而这一维正是**被索引的维**！
- `tensor<3x4x2>` 切第 1 维 → `3x2x2`（`4/2 = 2`）。
- **同样的索引重映射流程**：`partition_id` + 查找表 `[0,0,0,0,2,2,2,2]`
  （设备 0~3 偏移 0、设备 4~7 偏移 2）。

**与 `gather` 的对称性**：
| | `gather` | `scatter` |
|---|---|---|
| 问题维 | **归约维**（collapsed） | **被索引的维** |
| 索引处理 | 平移 + mask | 平移 + mask |
| 通信 | `all_reduce` | 可能不需要（见下） |

### `scatter` 的 9 个用例

| 用例 | 场景 |
|---|---|
| `input_not_sharded_scatter_indices_update_sharded_on_implicit_batch_dim` | 输入未分片，**索引/更新在隐式批维分片** |
| `input_scalar_scatter_indices_update_sharded_on_implicit_batch_dim` | 输入是标量 |
| `input_sharded_not_on_indexed_dim` | 分片**不在**被索引的维上 |
| `input_sharded_on_indexed_inserted__window_dim` | 分片**在**被索引的维上 |
| `input_sharded_on_indexed_but_non_inserted_window_dim` | 在被索引但**未插入**的维上 |
| `shard_indexd_dim_scalar_scatter_indices` | 标量索引 |
| `scatter_replicated_bounds` | 边界全复制 |
| `scatter_unreduced_axes_fallback_all_reduce` | 未归约 + `all_reduce` |

**注意**：`scatter` 是**写操作** —— 多台设备可能写同一位置，
所以它的"归约"语义与 `gather` 不同（用 `update_computation`）。

---

## 五、`select_and_scatter`：窗口是否跨设备

2 个用例，与 L5-07 的 `reduce_window` **完全同构**：

| 用例 | 场景 | 通信 |
|---|---|---|
| `select_and_scatter_batch_sharded` | **批维分片**（窗口=1） | **无** |
| `select_and_scatter_stride_greater_than_window` | **步长 ≥ 窗口** | **无** |

**`select_and_scatter`** 是 `reduce_window` 的**反向操作**：
`reduce_window` 从窗口归约出输出，`select_and_scatter` 从输出散射回窗口。

**判据相同**：**窗口是否跨设备边界**。

---

## 六、三个文件的对照

| 文件 | 关键判据 | 通信 |
|---|---|---|
| `gather` | **归约维（collapsed）是否分片** | `unreduced` + `all_reduce` |
| `scatter` | **被索引的维是否分片** | 索引重映射（可能无需通信） |
| `select_and_scatter` | **窗口是否跨设备** | 与 `reduce_window` 同 |

**★ 三个文件共同的机制**：**索引重映射**
```
clamp → partition_id + 查表 → 减偏移 → mask → 操作 → select 填零 → all_reduce
```

**★ 与前面几课统一的规律**：
> **归约方向上的分片需要通信；输出方向上的分片不需要。**
> 访存类算子的特殊之处是：**索引也要跟着重映射**。

**一句话总结**：
> **访存类算子的降级 = 索引重映射 + 部分结果合并。**
> 索引重映射是它独有的复杂度 —— 因为索引是**全局坐标**，
> 而每台设备只有**局部数据**。
