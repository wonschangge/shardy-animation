<!-- sdy-coverage
transforms/export/test/executable_convert_global_to_local/stablehlo_scatter_replicated_bounds.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_scatter_shard_implicit_dim.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_scatter_shard_indexed_inserted_dim.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_select_and_scatter.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_concatenate.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_reduce_window.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_sort.mlir
-->

# L6-09 · exec-scatter-and-misc — 源 IR

**L6 收官课**：`scatter` 的三种分片形态 + 四个算子的收尾。

| 族 | 文件 | 行数 |
|---|---|---|
| **scatter** | `scatter_shard_indexed_inserted_dim.mlir` | 77 |
| **scatter** | `scatter_shard_implicit_dim.mlir` | 76 |
| **scatter** | `scatter_replicated_bounds.mlir` | 70 |
| **misc** | `stablehlo_select_and_scatter.mlir` | 70 |
| **misc** | `stablehlo_reduce_window.mlir` | 51 |
| **misc** | `stablehlo_concatenate.mlir` | 49 |
| **misc** | `stablehlo_sort.mlir` | 43 |

网格：

```mlir
sdy.mesh @mesh_2 = <["x"=2]>
```

---

## 一、★ `scatter` 的三种分片形态

三个文件的 `scatter_dimension_numbers` **几乎相同**：

```mlir
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0], // Row dimension is collapsed/inserted
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
```

**区别在"谁被分片"**：

| 文件 | `%arg0`（被更新张量） | `%arg1`（索引） | `%arg2`（更新值） |
|---|---|---|---|
| **`shard_implicit_dim`** | **全复制** `[{}, {}]` | 切 `{"x"}` | 切 `{"x"}` |
| **`shard_indexed_inserted_dim`** | **切 `{"x"}`** | **无分片** | **无分片** |
| **`replicated_bounds`** | **全复制** `[{}, {}]` | 切 `{"x"}` | 切 `{"x"}` |

### ★ 验收点：隐式维 vs 显式维分片的区别

**`scatter` 的三个操作数**（回顾 **L5-08** 的 `scatter` 语义）：
- `%arg0` = **被更新的张量**（"bounds"）
- `%arg1` = **索引**
- `%arg2` = **更新值**

**`scatter_dimension_numbers` 的关键字段**：
- `update_window_dims = [1]` —— 更新值的第 1 维是**窗口维**
- `inserted_window_dims = [0]` —— 第 0 维是**被插入（collapsed）的维**
- `scatter_dims_to_operand_dims = [0]` —— 索引映射到操作数的第 0 维
- `index_vector_dim = 1` —— 索引向量的第 1 维是"索引分量维"

**两种分片形态的区别**：

| 形态 | 分片位置 | 含义 |
|---|---|---|
| **隐式维分片**（`shard_implicit_dim`） | `%arg1`/`%arg2` 的第 0 维 | 分片在**"索引数"维**（`index_vector_dim` 之外的维）—— 即**隐式的批维** |
| **显式维分片**（`shard_indexed_inserted_dim`） | `%arg0` 的第 0 维 | 分片在**被索引的 inserted 维**（`inserted_window_dims = [0]`） |

**★ 为什么这个区分重要**：
- **隐式维分片**：每个设备处理**不同的索引/更新值** —— 各写各的位置。
- **显式维分片**：`%arg0` 的**行维被切开** —— 每台设备只持有**部分行**，
  而索引可能指向**别的设备持有的行** → **需要索引重映射**（L5-08 的八步）！

**回顾 L5-08**：那里讲过 `scatter` 的
`input_sharded_on_indexed_inserted__window_dim` 用例 —— 本课的
`shard_indexed_inserted_dim` 是**同一情形**的完整执行验证。

---

## 二、`shard_implicit_dim`：分片在**隐式维**

```mlir
  %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>},
```

```mlir
  %arg1: tensor<2x1xi64> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
```

```mlir
  %arg2: tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
```

```mlir
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
```

**读法**：
- `%arg0`（`4x2`）**全复制** —— 每台设备都有**完整的**被更新张量。
- `%arg1`（`2x1` 索引）切 `{"x"}` → 每台 1 个索引。
- `%arg2`（`2x2` 更新值）切 `{"x"}` → 每台 1 行更新值。
- **每台设备处理"自己那个索引"的写入** —— 但写的是**同一份** `%arg0` 的副本。
- → 结果需要合并（因为每台只做了一部分写入）。

**★ "隐式维"的含义**：`%arg1` 的第 0 维（大小 `2`）是**索引的个数** ——
它在 `scatter_dimension_numbers` 里**没有对应的 `operand` 维**
（`scatter_dims_to_operand_dims = [0]` 指的是 `%arg1` 的**第 1 维**映射到 `%arg0` 的第 0 维）。
所以第 0 维是**隐式的批维**。

---

## 三、`shard_indexed_inserted_dim`：分片在**被索引的 inserted 维**

```mlir
  %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg1: tensor<2x1xi64>,
  %arg2: tensor<2x2xf32>)
  -> (tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
```

```mlir
      inserted_window_dims = [0], // Row dimension is collapsed/inserted
```

**读法**（**与 `shard_implicit_dim` 正好相反**）：
- `%arg0`（`4x2`）**切 `{"x"}`** —— **行维被切开**，每台 2 行。
- `%arg1`（索引）、`%arg2`（更新值）**都无分片** —— 每台都有**完整的**索引与更新值。
- **注释直接点明**：`// Row dimension is collapsed/inserted`
- **问题**：索引可能指向**别的设备持有的行** → **需要索引重映射**！
  - 设备 0 有第 0~1 行 → 索引 `0` 或 `1` 属于它
  - 设备 1 有第 2~3 行 → 索引 `2` 或 `3` 属于它
  - 索引 `3` 在设备 0 上**无法处理** → 必须重映射或通信

**★ 这正是 L5-08 讲的"索引重映射"场景**：
本课是它在 `scatter` 上的**完整执行验证**。

---

## 四、`replicated_bounds`：边界全复制

```mlir
  %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>},
```

```mlir
  %arg1: tensor<2x1xi64> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
```

```mlir
  %arg2: tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
```

```mlir
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
```

**读法**：
- `%arg0`（**bounds**，被更新张量）**全复制** `[{}, {}]` —— 用例名直接点明。
- `%arg1`/`%arg2` 切 `{"x"}`。
- **"replicated bounds" 的意义**：每台设备都有**完整的**被更新张量副本，
  各自写入自己那部分 → 最后**需要合并**（因为写入是"叠加"语义）。
- **与 `shard_implicit_dim` 的区别**：分片位置相同，但**用例意图不同** ——
  本用例强调"bounds 全复制"这个前提对合并策略的影响。

---

## 五、四个 misc 算子：作用维都**未**分片

**四个文件的共同点**：

| 文件 | 作用维 | 分片维 | 通信 |
|---|---|---|---|
| `stablehlo_concatenate` | **拼接维** `dim = 1` | 第 0 维 | **无** |
| `stablehlo_sort` | **排序维** `dimension = 1` | 第 0 维 | **无** |
| `stablehlo_reduce_window` | **窗口维** `window_dimensions = [1, 2]` | 第 0 维 | **无** |
| `stablehlo_select_and_scatter` | **窗口维** `window_dimensions = [1, 2]` | 第 0 维 | **无** |

### `concatenate`：拼接维未分片

```mlir
func.func @parallel_concat(
  %arg0: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg1: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
  -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
```

```mlir
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1
```

**读法**：
- `dim = 1` —— **拼接维是第 1 维**；而分片在**第 0 维** → **不冲突** ✓
- 每个操作数 `4x2` → 局部 `2x2`；拼接后 `2x4`（全局 `4x4`）。
- **无通信** —— 每台设备在自己那段里做同样的拼接。

**★ 这正是 L5-04 的"作用维判据"**（那里讲 `concatenate` 的 `sharded_non_concat_dim`）。

### `sort`：排序维未分片

```mlir
func.func @parallel_sort(
  %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
  -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
```

```mlir
  %0 = "stablehlo.sort"(%arg0) ({
```

```mlir
  }) {dimension = 1 : i64, is_stable = true,
```

**读法**：
- `dimension = 1` —— **排序维是第 1 维**；分片在第 0 维 → **不冲突** ✓
- `is_stable = true` —— 稳定排序（相等元素的相对顺序保持）。
- **无通信** —— 每台设备独立排序自己那些行的第 1 维。

**★ 如果排序维被分片**：排序需要**看到整行**才能确定顺序 ——
跨设备时就要通信（类似 `reverse` 的情形）。

### `reduce_window`：窗口维未分片

```mlir
func.func @parallel_reduce_window(%arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
    -> (tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
```

```mlir
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{
    window_dimensions = array<i64: 1, 2>,
```

**读法**：
- `window_dimensions = [1, 2]` —— **窗口在第 1 维**（大小 2）；分片在第 0 维。
- 输入 `2x4` → 输出 `2x2`（窗口 2、步长 2）。
- **无通信** —— 窗口不跨设备（**L5-07** 讲的判据）。

### `select_and_scatter`：窗口维未分片

```mlir
func.func @parallel_select_and_scatter(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
                                       %arg1: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
    -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
```

```mlir
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %init) <{
    window_dimensions = array<i64: 1, 2>,
```

**读法**：
- `window_dimensions = [1, 2]` —— 窗口在第 1 维；分片在第 0 维。
- **无通信**。
- **回顾 L5-08**：`select_and_scatter` 是 `reduce_window` 的**反向操作**
  （从输出散射回窗口）。

---

## 六、7 个文件的族谱

| 族 | 文件 | 验证什么 |
|---|---|---|
| **scatter 隐式维** | `scatter_shard_implicit_dim` | 分片在**索引数**维 |
| **scatter 显式维** | `scatter_shard_indexed_inserted_dim` | 分片在**被索引的 inserted 维** |
| **scatter 边界** | `scatter_replicated_bounds` | **bounds 全复制** |
| **misc** | `concatenate` | 拼接维未分片 |
| **misc** | `sort` | 排序维未分片 |
| **misc** | `reduce_window` | 窗口维未分片 |
| **misc** | `select_and_scatter` | 窗口维未分片 |

**★ 本课的三条结论**：

1. **`scatter` 的三种形态由"谁被分片"决定** ——
   隐式维（索引数）/ 显式维（被索引的 inserted 维）/ bounds 全复制。
2. **显式维分片最复杂** —— 索引可能指向别的设备持有的行 → 需要索引重映射（L5-08）。
3. **四个 misc 算子的作用维都未分片** —— 拼接维 / 排序维 / 窗口维都在第 1 维，
   而分片在第 0 维 → 全部**无通信**。这是 L5-04「作用维判据」在更多算子上的验证。

**一句话总结**：
> **`scatter` 的降级取决于「分片落在哪个操作数的哪个维上」** ——
> 隐式维分片各写各的、显式维分片需要索引重映射、bounds 全复制需要合并。
>
> 而四个 misc 算子则统一验证了 **L5-04 的作用维判据**。
