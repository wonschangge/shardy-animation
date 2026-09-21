<!-- sdy-coverage
transforms/export/test/insert_explicit_reshards/elementwise_ops.mlir
transforms/export/test/insert_explicit_reshards/broadcast_in_dim.mlir
transforms/export/test/insert_explicit_reshards/bitcast_convert.mlir
transforms/export/test/insert_explicit_reshards/reshape.mlir
transforms/export/test/insert_explicit_reshards/reverse.mlir
transforms/export/test/insert_explicit_reshards/concatenate.mlir
transforms/export/test/insert_explicit_reshards/clamp_select.mlir
transforms/export/test/insert_explicit_reshards/pad_slice.mlir
transforms/export/test/insert_explicit_reshards/dynamic_slice_dynamic_update_slice.mlir
-->

# L4-03 · reshard-elementwise-shape — 源 IR

**L4 按算子族展开的第一课**。覆盖 `insert_explicit_reshards/` 下 **9 个文件**：

| 文件 | 行数 | 用例数 |
|---|---|---|
| `elementwise_ops.mlir` | 95 | 11 |
| `broadcast_in_dim.mlir` | 21 | 2 |
| `bitcast_convert.mlir` | 41 | 4 |
| `reshape.mlir` | **479** | **47** |
| `reverse.mlir` | 58 | 6 |
| `concatenate.mlir` | 109 | 10 |
| `clamp_select.mlir` | 45 | 4 |
| `pad_slice.mlir` | 177 | 19 |
| `dynamic_slice_dynamic_update_slice.mlir` | 61 | 6 |

RUN 行（`elementwise_ops` 等）：`-sdy-insert-explicit-reshards='enable-full-version=true'`

网格：`sdy.mesh @mesh = <["x"=4, "y"=2, "z"=4]>`

---

## 一、★ 逐元素算子：三条**位置规则**

`elementwise_ops.mlir` 用 11 个用例把"reshard 插在哪"讲得很完整。

### 规则 ①：算子采用**"更大"的那一侧**的分片

```mlir
func.func @negate(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
```

```mlir
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}
```

```mlir
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<4x32xf32>
  // CHECK-NEXT: stablehlo.negate %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
```

**读法**：输入 `[{}, {}]`（无分片），结果要求 `[{"x"}, {}]`。
→ **在算子【之前】**插 reshard，让 `negate` 直接在 `[{"x"}, {}]` 上算。

### 规则 ②：输入侧"更大" → reshard 插在**之后**

```mlir
func.func @negate_input_sharding_is_larger(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
```

```mlir
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}
```

```mlir
  // CHECK: %[[NEGATE:.*]] = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[NEGATE]] <@mesh, [{"y"}, {}]> : tensor<4x32xf32>
```

**读法**：输入 `[{"x"}, {}]`，结果要 `[{"y"}, {}]`。
→ `negate` **保持输入的分片**（`x`），在**之后**插 reshard 搬到 `y`。

### 规则 ③：输出侧"更大" → reshard 插在**之前**

```mlir
func.func @negate_output_sharding_is_larger(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
```

```mlir
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}
```

```mlir
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<4x32xf32>
  // CHECK-NEXT: stablehlo.negate %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
```

**读法**：输入 `y`，结果要 `x`。
→ **在算子【之前】**把输入搬到 `x`，让 `negate` 在 `x` 上算。

**规则 ①②③ 合起来**：
> **算子采用"更大"的那一侧的分片；另一侧插 reshard。**
> 用例名 `_input_sharding_is_larger` / `_output_sharding_is_larger` 直接点明了判据。

### 两侧都需要 reshard 的情形

```mlir
func.func @add_input_and_output_sharded_on_separate_dims(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
```

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}
```

```mlir
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {"y"}]> : tensor<4x32xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {"y"}]> : tensor<4x32xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : tensor<4x32xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[ADD]] <@mesh, [{}, {"y"}]> : tensor<4x32xf32>
```

**读法**（**3 条 reshard**）：
- 两个操作数都是 `[{"x"}, {}]`，而算子声明的分片是 `[{}, {"y"}]` —— 两侧**维度都不同**。
- 解法：把两个操作数都 reshard 到 **`[{"x"}, {"y"}]`**（**并集**！），
  在并集上算 `add`，再把结果 reshard 回 `[{}, {"y"}]`。

**洞察**：算子被放在**两个分片的并集**上执行 —— 这样两侧的"信息"都不丢。

### 输入相同、输出无分片

```mlir
func.func @add_inputs_are_sharded_the_same_way_output_is_unsharded(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<4x32xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}
```

```mlir
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ADD]] <@mesh, [{}, {}]> : tensor<4x32xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<4x32xf32>
```

**读法**：两个输入都是 `[{"x"}, {}]`，输出无分片 → 在 `[{"x"}, {}]` 上算，**之后** reshard 掉。

### `transpose`：维度重排后要 reshard

```mlir
func.func @transpose(%arg0: tensor<256x32x64x100xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {"y"}, {}]>}) -> (tensor<100x32x256x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"z"}, {}]>}) {
```

```mlir
  %0 = stablehlo.transpose %arg0, dims = [3, 1, 0, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"z"}, {}]>]>} : (tensor<256x32x64x100xf32>) -> tensor<100x32x256x64xf32>
  return %0 : tensor<100x32x256x64xf32>
}
```

```mlir
  // CHECK: %[[TRANSPOSE:.*]] = stablehlo.transpose %arg0, dims = [3, 1, 0, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {}, {"y"}]>]>} : (tensor<256x32x64x100xf32>) -> tensor<100x32x256x64xf32>
```

```mlir
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[TRANSPOSE]] <@mesh, [{}, {"y"}, {"z"}, {}]> : tensor<100x32x256x64xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<100x32x256x64xf32>
```

**读法**：
- 输入 `[{}, {"x"}, {"y"}, {}]`（dim1=x, dim2=y）
- `dims = [3, 1, 0, 2]` 表示输出第 0 维来自输入第 3 维、输出第 1 维来自输入第 1 维……
- 所以输入的 `x`（dim1）→ 输出 dim1，输入的 `y`（dim2）→ 输出 **dim3**
- 输出的 transpose 分片写成 `[{}, {"x"}, {}, {"y"}]` —— **正是按 dims 重排后的位置**！
- 但结果要求 `[{}, {"y"}, {"z"}, {}]` → 再插一条 reshard。

**规则**：**transpose 的分片要按 `dims` 重排；重排后与目标不符时，在算子之后 reshard。**

---

## 二、`broadcast_in_dim`：按 `dims` 映射**在输入侧**对齐

```mlir
func.func @broadcast_in_dim_input_output_different(%arg0: tensor<2x3x5x1x7xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}, {}]>}) -> (tensor<2x5x3x11x7x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}, {}, {}, {"y"}]>}) {
```

```mlir
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 1, 3, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {}, {}, {}, {"y"}]>]>} : (tensor<2x3x5x1x7xf32>) -> tensor<2x5x3x11x7x13xf32>
  return %0 :  tensor<2x5x3x11x7x13xf32>
}
```

```mlir
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}, {"x"}, {}, {}]> : tensor<2x3x5x1x7xf32>
  // CHECK-NEXT: %[[BROADCAST_IN_DIM:.*]] = stablehlo.broadcast_in_dim %[[RESHARD]], dims = [0, 2, 1, 3, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {}, {}, {}, {"y"}]>]>} : (tensor<2x3x5x1x7xf32>) -> tensor<2x5x3x11x7x13xf32>
```

**读法**（本课最巧妙的一处）：
- 输入第 0 维是 `{"x"}`；`dims = [0, 2, 1, 3, 4]` 表示输入第 0 维 → 输出第 0 维。
- 但**输出第 0 维没有分片**，而输出第 1 维有 `{"x"}` —— 输出第 1 维来自**输入第 2 维**（`dims[2] = 1`）。
- 所以 reshard 把 `x` 从输入第 0 维**搬到输入第 2 维**：`[{}, {}, {"x"}, {}, {}]`。
- 这样 broadcast 之后，`x` 自然落在输出第 1 维上 ✓

**规则**：**broadcast 的分片要按 `dims` 反向映射到输入侧；不一致时在输入侧 reshard。**

第一个用例（`broadcast_in_dim`）则相反：输入有 `{"x"}`、输出**无**分片
→ 输入侧 reshard 成 `[{}, {}, {}, {}, {}]`（把 `x` 去掉）。

---

## 三、`reshape`：为什么**常需 reshard**

`reshape.mlir` 有 **47 个用例**（本课最大），因为 reshape 的分片对应关系最复杂。

```mlir
func.func @reshape(%arg0: tensor<16x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y", "x"}]>}) {
```

```mlir
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", "x"}]>]>} : (tensor<16x2x4xf32>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
```

```mlir
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {"y"}, {}]> : tensor<16x2x4xf32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<16x2x4xf32>) -> tensor<16x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[RESHAPE]] <@mesh, [{}, {"y", "x"}]> : tensor<16x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<16x8xf32>
```

**读法**：**两侧都插** reshard（`RESHARD1` 在 reshape 前，`RESHARD2` 在后）。

**为什么 reshape 常需 reshard**：
- reshape 的分片必须在**因子层面**对应（L2-01 讲过"沿因子传播"）。
- `16x2x4 → 16x8` 把第 1、2 维合并成 8；而结果要求 `{"y","x"}` 落在这个复合因子上。
- 输入的 `{"x"}` 在第 0 维（**不参与合并**），`y` 又没被切 —— 因子对应关系对不上。
- 于是先 reshard 到 `[{"x"}, {"y"}, {}]`（让 `y` 落在将被合并的维度上），
  reshape 后再 reshard 到目标。

**反例：能对应时不需要 reshard**

```mlir
func.func @reshape_simple_merge_sharding_is_from_x_to_x_and_x_fits_exactly_to_first_dim(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
```

```mlir
  // CHECK-NOT: sdy.reshard
```

**读法**：`CHECK-NOT: sdy.reshard` —— **一条都不插**。
因为 `x` 切的是**最 major 的因子**，合并后仍能对应（L2-01 讲过的零通信情形）。

### 47 个用例的命名规律

用例名把"输入因子 → 输出因子"的对应写得非常清楚：

| 命名片段 | 含义 |
|---|---|
| `reshape_ij_k_to_i_jk` | 输入因子 `[ij, k]` → 输出 `[i, jk]` |
| `_and_x_to_x` / `_and_x_to_z` | 单轴切法不变 / 变成另一轴 |
| `_and_xy_to_yx` / `_and_yx_to_xy` | 轴序**交换** |
| `_and_xy_to_x` / `_and_xy_to_y` | 复合轴**拆开** |
| `_merged_dimensions_are_sharded` | 被合并的维上有分片 |
| `_singleton_dimensions_are_sharded` | 大小为 1 的维上有分片 |
| `_factor_j_is_sharded` | 具体哪个因子被切 |
| `reshape_size_1_dimensions_1` / `_2` | 大小为 1 的维 |
| `reshape_strided_view_on_both_operand_and_result` | 带步长的视图 |

**读法建议**：不必逐个读 —— 抓住"**因子对应**"这一条，用例名本身就是答案。

---

## 四、其余五个文件

### `bitcast_convert.mlir`（41 行 / 4 用例）

`bitcast_convert` 改变元素类型（如 `f32` → `i32`），形状可能变化。
4 个用例覆盖 upcast / equal / downcast 三种情形（与 L2-10 注册表里的
`bitcast_convert_upcast` / `_equal` / `_downcast` 对应）。

### `reverse.mlir`（58 行 / 6 用例）

反转某一维。**反转不改变维度大小**，所以分片可以**直接对应** ——
但要注意反转的维上分片语义（`{"x"}` 在反转后仍是 `{"x"}`）。

### `concatenate.mlir`（109 行 / 10 用例）

**拼接维上各操作数的分片必须一致**（L4-02 讲过）。
10 个用例覆盖：分片相同 / 不同 / 结果分片不同 / 多个操作数等。

### `clamp_select.mlir`（45 行 / 4 用例）

`clamp`（三操作数）与 `select`（条件选择）—— 都是逐元素族，
规则与 `elementwise_ops` 一致（对应维同分片）。

### `pad_slice.mlir`（177 行 / 19 用例）

`pad` 与 `slice`。**这两个算子会改变维度大小**，所以分片可能**不可整除**
（L2-02 讲过"slice 切完就不整除了"）—— 19 个用例覆盖各种尺寸变化组合。

### `dynamic_slice_dynamic_update_slice.mlir`（61 行 / 6 用例）

动态切片：**起始索引是运行时值**，分片处理更受限（L2-10 的规则里
`gather` 有 `blocked_propagation`，动态切片同理）。

---

## 五、9 个文件 / 109 个用例的族谱

| 族 | 文件 | 用例数 | 核心规则 |
|---|---|---|---|
| **逐元素** | `elementwise_ops`、`clamp_select` | 15 | 对应维同分片；"更大"的一侧优先 |
| **广播** | `broadcast_in_dim` | 2 | 按 `dims` 反向映射到输入侧 |
| **形状变换** | `reshape`、`reverse`、`bitcast_convert` | 57 | **因子对应**；不对应就 reshard |
| **拼接** | `concatenate` | 10 | 拼接维分片必须一致 |
| **切片/填充** | `pad_slice`、`dynamic_slice_...` | 25 | 尺寸变化 → 可能不可整除 |

**一句话总结**：
> **逐元素类看"对应维"；形状类看"因子对应"；拼接类看"拼接维一致"。**
