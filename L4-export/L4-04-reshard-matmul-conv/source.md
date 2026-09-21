<!-- sdy-coverage
transforms/export/test/insert_explicit_reshards/dot_dot_general.mlir
transforms/export/test/insert_explicit_reshards/convolution.mlir
transforms/export/test/insert_explicit_reshards/fft.mlir
transforms/export/test/insert_explicit_reshards/cholesky_triangular_solve.mlir
-->

# L4-04 · reshard-matmul-conv — 源 IR

**L4 按算子族展开的第二课**：矩阵 / 卷积 / 变换类。覆盖 4 个文件：

| 文件 | 行数 | 用例数 |
|---|---|---|
| `insert_explicit_reshards/dot_dot_general.mlir` | **736** | **66** |
| `insert_explicit_reshards/convolution.mlir` | 60 | 3 |
| `insert_explicit_reshards/fft.mlir` | 74 | 7 |
| `insert_explicit_reshards/cholesky_triangular_solve.mlir` | 202 | 19 |

RUN 行：`-sdy-insert-explicit-reshards='enable-full-version=true mark-partial-result-with-unreduced-axes=true'`

网格：`sdy.mesh @mesh = <["x"=4, "y"=2]>`

---

## 一、★ `dot` 的三类情形

`dot_dot_general.mlir` 用 **66 个用例**穷举了各种分片组合 —— 是本课的主体。

### 情形 ①：收缩维**未切** → 直接算，无 reshard

```mlir
func.func @dot_compatible_contracting_unsharded(
    %arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
```

```mlir
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
```

```mlir
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>
  // CHECK: return %[[DOT]]
```

**读法**（最理想的情形）：
- lhs 第 0 维 `{"x"}`、rhs 第 1 维 `{"y"}` —— **非收缩维**，各切各的
- 收缩维（lhs 第 1 维、rhs 第 0 维）**都没切**
- 结果 `[{"x"}, {"y"}]` 与操作数**完全一致** → **零通信**。

### 情形 ②：收缩维**被切** → `unreduced` + `all_reduce`

```mlir
func.func @dot_compatible_contracting_dim_sharded(
    %arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>},
    %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
```

```mlir
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
```

```mlir
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}], unreduced={"y"}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{"x"}, {}]>
  // CHECK: return %[[ALL_REDUCE]]
```

**读法**：
- 收缩维（lhs 第 1 维、rhs 第 0 维）**都被切了 `y`** —— 且**一致** ✓
- 但沿收缩维切会产生**部分和** → `dot` 的结果标 `unreduced={"y"}`
- 结果不要求 `y` 上的分片 → 插一条 `all_reduce` 归约掉。

**这就是 L4-01 讲过的机制**：`unreduced` + `all_reduce`。

### 情形 ③：**reduce-scatter 模式**（本课的重点）

```mlir
// This is a reduce-scatter pattern.
// CHECK-LABEL: func @dot_contracting_dim_and_result_dim_sharded_same_axis
func.func @dot_contracting_dim_and_result_dim_sharded_same_axis(
    %arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>},
    %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
```

```mlir
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
```

```mlir
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}], unreduced={"y"}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{"x"}, {}]>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{"x"}, {"y"}]>
  // CHECK: return %[[RESHARD]]
```

**读法**（测试注释直接点明：`This is a reduce-scatter pattern.`）：
- 收缩维切了 `y`，**结果也要在 `y` 上有分片** —— 这就是"**收缩维与结果维用同一个轴**"。
- 三步：`dot`（标 `unreduced`）→ `all_reduce`（归约）→ `reshard`（重新切回 `y`）。
- **这三步合起来就是 `reduce_scatter` 的语义** —— 一边归约一边重新切分。

**为什么这是重点**：在 L4-01 见过导出流水线会把它**融合成一条 `reduce_scatter`**。
本课的 `all_reduce` + `reshard` 是**融合前**的形态。

### 66 个用例的命名规律

| 命名片段 | 含义 |
|---|---|
| `dot_compatible_*` | **兼容**（无需 reshard 或只需 all_reduce） |
| `dot_incompatible_*` | **不兼容**（需要 reshard） |
| `_contracting_unsharded` / `_contracting_dim_sharded` | 收缩维是否被切 |
| `_contracting_dim_and_result_dim_sharded_same_axis` | **收缩维与结果维同轴**（reduce-scatter 模式） |
| `_same_axis_incompatible_order` | 同轴但**轴序不兼容** |
| `_incompatible_lhs_contracting_and_rhs_non_contracting_dims` | lhs 收缩维与 rhs 非收缩维冲突 |
| `_incompatible_subaxis_no_overlap` | **子轴无重叠** |
| `_compatible_jk` / `_k` / `_empty` | 具体的因子组合 |
| `_incompatible_a_times_a` | 同一轴出现两次 |
| `_incompatible_all_same_shardings` | 全部分片相同（但语义不兼容） |
| `_incompatible_same_factor_for_contracting_dim_and_output_i` / `_j` | 同一因子既做收缩又做输出 |

**读法建议**：用例名把"冲突在哪里"写得很清楚，不必逐个读代码。

---

## 二、`convolution`：批维的分组会**改变大小**

```mlir
// CHECK-LABEL: func @convolution
func.func @convolution(%arg0 : tensor<2x224x224x192xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}, %arg1 : tensor<3x3x192x64xf32>) -> tensor<2x112x112x64xf32> {
```

```mlir
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      lhs_dilations = dense<1> : tensor<2xi64>,
      rhs_dilations = dense<1> : tensor<2xi64>
    } : (tensor<2x224x224x192xf32>, tensor<3x3x192x64xf32>) -> tensor<2x112x112x64xf32>
  return %0 : tensor<2x112x112x64xf32>
}
```

```mlir
  // CHECK: %[[CONVOLUTION:.*]] = stablehlo.convolution(%arg0, %arg1)
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CONVOLUTION]] <@mesh, [{}, {}, {}, {}]> : tensor<2x112x112x64xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<2x112x112x64xf32>
```

**读法**：输入在**批维**（`b`）切了 `x`，结果无分片要求
→ `convolution` 保持 `[{"x"}, {}, {}, {}]`，算完 reshard 掉。

### `batch_group_count` 的复杂性

```mlir
// CHECK-LABEL: func @convolution_batch_group_count
func.func @convolution_batch_group_count(%arg0: tensor<8x224x224x192xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {"z"}, {}, {}]>}, %arg1: tensor<3x3x192x256xf32>) -> (tensor<2x112x112x256xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {"z"}, {}, {}]>}) {
```

```mlir
  // NOTE: sdy.sharding_rule = ([ij, kl, mn, o], [l, n, o, ip])->([j, k, m, ip]) {i=4, j=2, k=112, l=2, m=112, n=2, o=192, p=64} reduction={l, n, o} permutation={k, m}
```

**读法**（测试里的 NOTE 直接给出了规则）：
- 输入批维是 **`ij`**（复合因子）：`i=4`（batch_group_count） × `j=2`（输出批维）
- 输入大小 8 = 4 × 2，输出批维只有 2 —— **`batch_group_count` 把批维缩小了 4 倍**
- 规则里 `reduction={l, n, o}`（窗口与输入通道）、`permutation={k, m}`（空间维因步长变化）

**这就是 L2-10 讲过的卷积规则的来源** —— 现在看到它在 reshard 插入中的实际作用。

```mlir
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x", "y"}, {"z"}, {}, {}]> : tensor<8x224x224x192xf32>
```

**读法**：输入批维切了 `x`，但结果要 `y` → 先 reshard 到 **`[{"x", "y"}, ...]`**（并集，
与 L4-03 的并集解法一致），因为 `x` 切的是 `i`、`y` 切的是 `j`。

---

## 三、`fft`：7 个用例覆盖变换的各种形态

| 用例 | 场景 |
|---|---|
| `fft` | 基础正向 |
| `fft_inverse` | 逆变换 |
| `fft_real_truncated_result` | 实数输入、**截断**结果 |
| `fft_inverse_real_expanded_result` | 逆变换、实数**扩展**结果 |
| `fft_small_batch_dimension` | 批维较小 |
| `fft_single_fft_dimension` | **单个** FFT 维 |
| `fft_single_fft_dimension_real_truncated_result` | 单 FFT 维 + 截断 |

**要点**：`fft` 有"**变换维**"与"**批维**"之分。
- 批维可以自由分片（各批独立变换）。
- 变换维上的分片更受限（变换是**全局**操作 —— 与 `reduce` 类似）。

**截断/扩展**（`real_truncated` / `real_expanded`）会**改变维度大小**，
所以可能引入不可整除（与 L4-03 的 `pad_slice` 同理）。

---

## 四、`cholesky_triangular_solve`：19 个用例的维度组合

用例名把"输入/输出分别在哪个维上切"写得非常清楚：

| 用例 | 输入切在哪 | 输出切在哪 |
|---|---|---|
| `cholesky_sharded_input_batch_dim_only` | 批维 | — |
| `cholesky_sharded_output_batch_dim_only` | — | 批维 |
| `cholesky_sharded_batch_dim_only_different` | 批维 | 批维（**不同轴**） |
| `cholesky_sharded_input_cholesky_dim_only` | cholesky 维 | — |
| `cholesky_sharded_output_cholesky_dim_only` | — | cholesky 维 |
| `cholesky_sharded_cholesky_dim_only_different` / `_same` | cholesky 维 | cholesky 维 |
| `cholesky_sharded_input_batch_dim_and_output_cholesky_dim_same` | 批维 | cholesky 维 |
| `cholesky_sharded_output_batch_dim_and_input_cholesky_dim_same` | cholesky 维 | 批维 |
| `cholesky_sharded_same` | 相同 | 相同 |

**读法**：两个关键维度 —— **批维**（可自由分片）与 **cholesky 维**（矩阵分解维，
分片受限）。19 个用例就是这两个维度"切/不切/切哪个轴"的组合。

**为什么 cholesky 维受限**：矩阵分解需要看到**整个矩阵**（三角求解有依赖关系），
所以沿它分片通常需要复制或特殊处理 —— 与 `sort` 的"被排序维必须全复制"同类
（L4-05 会讲）。

---

## 五、4 个文件 / 95 个用例的族谱

| 族 | 文件 | 用例数 | 核心规则 |
|---|---|---|---|
| **矩阵乘** | `dot_dot_general` | **66** | 收缩维一致 + 非收缩维对齐；reduce-scatter 模式 |
| **卷积** | `convolution` | 3 | 批维分组改变大小（复合因子） |
| **变换** | `fft` | 7 | 变换维受限、批维自由；截断/扩展改尺寸 |
| **分解** | `cholesky_triangular_solve` | 19 | 批维自由、分解维受限 |

**一句话总结**：
> **收缩/变换/分解维是"受限维"，批维是"自由维"。**
> 冲突判定就是看这两类维度上的分片是否兼容。
