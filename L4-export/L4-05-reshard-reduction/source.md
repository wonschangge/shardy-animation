<!-- sdy-coverage
transforms/export/test/insert_explicit_reshards/reduce.mlir
transforms/export/test/insert_explicit_reshards/reduce_window_select_and_scatter.mlir
transforms/export/test/insert_explicit_reshards/sort.mlir
transforms/export/test/insert_explicit_reshards/rng_bit_generator.mlir
-->

# L4-05 · reshard-reduction — 源 IR

**L4 按算子族展开的第三课**：归约与排序类。覆盖 4 个文件：

| 文件 | 行数 | 用例数 |
|---|---|---|
| `insert_explicit_reshards/reduce.mlir` | 112 | 8 |
| `insert_explicit_reshards/reduce_window_select_and_scatter.mlir` | 77 | 3 |
| `insert_explicit_reshards/sort.mlir` | 140 | 10 |
| `insert_explicit_reshards/rng_bit_generator.mlir` | 14 | 1 |

网格：`sdy.mesh @mesh = <["x"=4, "y"=2]>`

---

## 一、`reduce`：**归约维切没切**决定一切

### 情形 ①：归约维**未切** → 保留其他维，之后 reshard

```mlir
func.func @reduce_single_result_reduction_dim_not_sharded(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> tensor<2x13xf32> {
```

```mlir
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  return %1 : tensor<2x13xf32>
}
```

```mlir
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[REDUCE]] <@mesh, [{}, {}]> : tensor<2x13xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<2x13xf32>
```

**读法**：
- 归约的是**第 1 维**（`across dimensions = [1]`），而 `x` 切的是**第 0 维** —— 不冲突。
- `reduce` 保留 `[{"x"}, {}]`（第 0 维的 `x` 不受归约影响），
- 结果要求无分片 → **之后**插一条 reshard 把 `x` 去掉。

### 情形 ②：归约维**被切** → `unreduced` + `all_reduce`

```mlir
func.func @reduce_single_result_reduction_dim_sharded(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) -> tensor<2x13xf32> {
```

```mlir
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  return %1 : tensor<2x13xf32>
}
```

```mlir
  // UNREDUCED:        %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>}
  // NOUNREDUCED:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1]
  // NOUNREDUCED-NOT:  sdy.sharding
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x"} %[[REDUCE]] out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
```

**读法**：`x` 切的正是**被归约的第 1 维** → 每台设备只有**部分和**
→ `reduce` 标 `unreduced={"x"}`，再插 `all_reduce {"x"}`。

### ★ 两个 RUN 行的对比：`mark-partial-result-with-unreduced-axes`

```mlir
// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true mark-partial-result-with-unreduced-axes=true' | FileCheck %s --check-prefixes=CHECK,UNREDUCED
// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true mark-partial-result-with-unreduced-axes=false' | FileCheck %s --check-prefixes=CHECK,NOUNREDUCED
```

**这个文件跑了两次**，分别用 `UNREDUCED` 与 `NOUNREDUCED` 两组 CHECK 前缀：

| 选项 | `reduce` 上的分片 | `all_reduce` |
|---|---|---|
| `true` | `[{}, {}], unreduced={"x"}` | **仍然插** |
| `false` | **完全没有分片属性**（`NOUNREDUCED-NOT: sdy.sharding`） | **仍然插** |

**读法**（本课最有价值的对比）：
- **两者都会插 `all_reduce`** —— 归约这件事跑不掉。
- 区别只在**要不要在 IR 里显式标出"这个结果是部分和"**。
- `true` 时更**信息完整**（后续 pass 知道这里是部分和）；`false` 时 IR 更**干净**。

**这是一个"可观测性 vs 简洁性"的取舍** —— 与 L2-03 的 `keep-sharding-rules` 同类。

### 多个归约维

```mlir
func.func @reduce_single_result_multiple_reduction_dims_sharded(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}, {}]>}) -> tensor<13xf32> {
```

```mlir
  // UNREDUCED:        %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0, 1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}], unreduced={"x", "y"}>]>}
  // NOUNREDUCED:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0, 1]
  // NOUNREDUCED-NOT:  sdy.sharding
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x", "y"} %[[REDUCE]] out_sharding=<@mesh, [{}]>
```

**读法**：`across dimensions = [0, 1]` 归约**两维**，而这两维分别切了 `y` 与 `x`
→ `unreduced={"x", "y"}`，`all_reduce` 一次归约两个轴。

---

## 二、★ `sort`：**被排序维必须全复制**

这是本课最重要的规则。

```mlir
func.func @sort(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<4x32x8xf32>) -> (tensor<4x32x8xi32>, tensor<4x32x8xf32>) {
```

```mlir
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"x"}, {}]>
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{}, {"x"}, {}]>
  // CHECK-NEXT: %[[SORT:.*]]:2 = "stablehlo.sort"(%[[RESHARD0]], %[[RESHARD1]])
  // CHECK: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {}]>, <@mesh, [{}, {"x"}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %2#0 <@mesh, [{}, {}, {}]>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %2#1 <@mesh, [{?}, {?}, {?}]>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]]
```

**读法**（`dimension = 0`，即**排序第 0 维**）：
- 输入 `%arg0` 在**第 0 维**切了 `x` —— 而第 0 维**正是被排序的维**！
- **排序维不能有分片** → 先把 `x` **搬到第 1 维**（`[{}, {"x"}, {}]`）。
- 在"排序维无分片"的状态下执行 `sort`。
- 排完再 reshard 回去（两个结果各一条，且**分片不同**）。

### 为什么排序维不能分片

**排序是全局操作**：元素的相对顺序取决于**整条维上的所有元素**。
如果沿排序维分片，每台设备只看到自己那一段 —— 无法确定元素应该排在哪里。

**这与 `reduce` 的归约维是同一类"受限维"** —— 只是排序更严格：
- `reduce` 沿归约维分片还能靠 `all_reduce` 补救（部分和可归约）。
- `sort` **不能**靠通信补救（顺序不是可归约的量）→ 必须**全复制**。

这就是 L2-10 规则里 `sort` 的 `need_replication` 因子的来源。

### `sort_all_other_dims_size_one`：只能全复制

```mlir
func.func @sort_all_other_dims_size_one(%arg0: tensor<1x4x1xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) -> tensor<1x4x1xi32> {
```

```mlir
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}, {}]> : tensor<1x4x1xi32>
  // CHECK-NEXT: "stablehlo.sort"(%[[RESHARD]])
```

**读法**：`tensor<1x4x1>` 的其他两维大小都是 **1** —— **没有地方可以"挪"分片**！
→ 只能 reshard 成**全复制** `[{}, {}, {}]`。

**这解释了"为什么 sort 需要 `need_replication`"**：
分片无处安放时，唯一的选择就是复制。

### 10 个用例的命名规律

| 用例 | 场景 |
|---|---|
| `sort` | 基础（分片在排序维上） |
| `sort_all_other_dims_size_one` | 其他维大小为 1 → 只能全复制 |
| `sort_single_input_output` | 单输入输出 |
| `sort_compatible` | 兼容 |
| `sort_input_and_output_shardings_are_same_on_sorting_dimension` | 排序维上输入输出分片**相同** |
| `sort_input_and_output_shardings_are_different_on_sorting_dimension` | 排序维上**不同** |
| `sort_sorting_dim_shardings_has_common_prefix` | 排序维分片有**公共前缀** |
| `sort_sorting_dim_shardings_has_common_prefix_and_large` | 同上且更复杂 |
| `sort_incompatible_on_nonsort_dimensions` | **非排序维**上不兼容 |
| `sort_compatible_on_nonsort_dimension` | 非排序维上兼容 |

**注意最后两个**：非排序维上的分片**可以**不同（各切各的），
只要**排序维无分片**即可 —— 说明规则是**逐维**的。

---

## 三、`reduce_window`：窗口会**改变维度大小**

```mlir
func.func @reduce_window(%arg0: tensor<48x48x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>},
                         %arg1: tensor<48x48x3xi32>, %arg2: tensor<f32>, %arg3: tensor<i32>)
    -> (tensor<16x48x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, tensor<16x48x1xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) {
```

```mlir
  // sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, j, k], [], [])->([i, j, k], [i, j, k]) {i=16, j=48, k=1} permutation={i, j, k}>
  // CHECK-NEXT: %0 = sdy.reshard %arg1 <@mesh, [{"x"}, {}, {}]> : tensor<48x48x3xi32>
  // CHECK-NEXT: %1:2 = "stablehlo.reduce_window"(%arg0, %0, %arg2, %arg3)
  // CHECK: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>, <@mesh, [{"x"}, {}, {}]>]>
```

**读法**（测试里的注释给出了规则）：
- 规则是 `permutation={i, j, k}` —— **所有三个因子都是 permutation**！
- 因为窗口（`window_dimensions = [3, 1, 3]`、`window_strides = [3, 1, 3]`、padding）
  把 `48x48x3` 变成了 `16x48x1` —— **尺寸全变了**。
- 第二个操作数（`%arg1`，`i32` 的那个）**没有分片** → 先 reshard 到 `[{"x"}, {}, {}]`
  与第一个操作数**对齐**（`reduce_window` 的两个操作数必须同分片）。

**与 `reduce` 的区别**：`reduce` 的规则是纯因子对应（无 permutation）；
`reduce_window` 因窗口而尺寸变化 → 全标 `permutation`（与 L2-10 的 `conv` 同理）。

---

## 四、`rng_bit_generator`：**状态必须全复制**

```mlir
func.func @rng_bit_generator(%arg0: tensor<2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> tensor<2xui64> {
```

```mlir
  %0, %output = stablehlo.rng_bit_generator %arg0, algorithm =  DEFAULT {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}]>, <@mesh, [{"y"}, {"x":(2)2}]>]>} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4x1000xui32>)
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}]>]>} : tensor<2xui64>
  return %1 : tensor<2xui64>
}
```

```mlir
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}]> : tensor<2xui64>
  // CHECK-NEXT: %output_state, %output = stablehlo.rng_bit_generator %[[RESHARD1]], algorithm =  DEFAULT {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>, <@mesh, [{"y"}, {"x":(2)2}]>]>}
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %output_state <@mesh, [{"x":(1)2}]> : tensor<2xui64>
  // CHECK-NEXT: stablehlo.negate %[[RESHARD2]]
```

**读法**：
- **输入状态** `%arg0` 在切了 `y`，但 `rng_bit_generator` 要求状态**全复制** `[{}]`
  → 先 reshard 掉 `y`。
- **输出状态** 声明为 `[{}]`（全复制），而下游 `negate` 要 `[{"x":(1)2}]`
  → 再 reshard。
- **输出数据** `%output` 可以是 `[{"y"}, {"x":(2)2}]` —— **自由分片**。

**为什么状态必须全复制**：
随机数生成器的状态是**一条连续的序列**。如果分片，每台设备会生成**不同的序列** ——
而"同一份随机数"是语义要求（复现性）。所以状态必须复制，
让每台设备都持有完整的生成器状态。

**输出数据可以分片**：数据是"从序列里取出的元素"，各设备取自己那份即可。

---

## 五、4 个文件 / 22 个用例的族谱

| 族 | 文件 | 用例数 | 受限维 | 补救方式 |
|---|---|---|---|---|
| **规约** | `reduce` | 8 | 归约维 | `all_reduce`（可补救） |
| **窗口规约** | `reduce_window_select_and_scatter` | 3 | 窗口维 | 尺寸变化 → permutation |
| **排序** | `sort` | 10 | **排序维** | **全复制**（不可补救） |
| **随机数** | `rng_bit_generator` | 1 | **状态** | **全复制** |

**一句话总结**：
> **归约维可以"部分和 + all_reduce"补救；排序维和 RNG 状态只能全复制。**
>
> 判据是：**这个操作的结果能不能靠"归约"合并？**
> 能（求和、取最大）→ 用 `all_reduce`；不能（顺序、随机序列）→ 必须复制。
