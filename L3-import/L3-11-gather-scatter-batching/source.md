<!-- sdy-coverage
transforms/import/test/explicit_gather_scatter_batching.mlir
transforms/import/test/executable_explicit_gather_scatter_batching/gather_iota_at_end_of_concat.mlir
transforms/import/test/executable_explicit_gather_scatter_batching/gather_iota_broadcast_concat.mlir
transforms/import/test/executable_explicit_gather_scatter_batching/gather_iota_concat_batch_dim.mlir
transforms/import/test/executable_explicit_gather_scatter_batching/gather_iota_reshaped_concat.mlir
transforms/import/test/executable_explicit_gather_scatter_batching/scatter_batch_dim.mlir
-->

# L3-11 · gather-scatter-batching — 源 IR

本课覆盖 6 个文件（**L3 收官课**）：

| 文件 | 行数 | 作用 |
|---|---|---|
| `transforms/import/test/explicit_gather_scatter_batching.mlir` | 341 | 主测试：10 个用例（5 转换 + 5 不转换） |
| `executable_.../gather_iota_concat_batch_dim.mlir` | — | 可执行验证 |
| `executable_.../gather_iota_reshaped_concat.mlir` | — | 可执行验证 |
| `executable_.../gather_iota_at_end_of_concat.mlir` | — | 可执行验证 |
| `executable_.../gather_iota_broadcast_concat.mlir` | — | 可执行验证 |
| `executable_.../scatter_batch_dim.mlir` | — | 可执行验证 |

```mlir
// RUN: sdy_opt %s -sdy-explicit-gather-scatter-batching | FileCheck %s
```

**这一课回答**：为什么隐式批维会导致**分片规则无法表达**？

---

## 一、★ 核心变换：隐式批维 → 显式批维

### 背景：这个模式从哪来

测试注释说得很清楚：

```
// This pattern typically arises from row-wise indexing in JAX using a batch iota,
// such as: arr.at[jnp.arange(B), offset] or arr.at[jax.lax.iota(jnp.int32, B), offset]
// Both emit stablehlo.iota concatenated with the column index.
```

也就是说：JAX 里写 `arr.at[jnp.arange(B), offset]` 这种**逐行索引**，
会生成「`iota` + `concat`」来构造 `start_indices`。

### 转换前

```mlir
func.func @gather_iota_concat_batch_dim(
    %operand: tensor<4x8xf32>, %offset: tensor<4x1xi32>)
    -> tensor<4x1xf32> {
```

```mlir
  // The iota generates [0, 1, 2, 3] for the batch dimension.
  %iota = stablehlo.iota dim = 0 : tensor<4x1xi32>
  // Concatenate iota with the actual offset to form start_indices.
  %indices = stablehlo.concatenate %iota, %offset, dim = 1
      : (tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x2xi32>
  // CHECK: "stablehlo.gather"
  // CHECK-SAME: offset_dims = [1]
  // CHECK-SAME: operand_batching_dims = [0]
  // CHECK-SAME: start_indices_batching_dims = [0]
  // CHECK-SAME: start_index_map = [1]
  // CHECK-SAME: index_vector_dim = 1
  %result = "stablehlo.gather"(%operand, %indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0, 1],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1>,
    indices_are_sorted = false
  } : (tensor<4x8xf32>, tensor<4x2xi32>) -> tensor<4x1xf32>
  return %result : tensor<4x1xf32>
}
```

### 转换后

```mlir
  // CHECK: "stablehlo.gather"
  // CHECK-SAME: offset_dims = [1]
  // CHECK-SAME: operand_batching_dims = [0]
  // CHECK-SAME: start_indices_batching_dims = [0]
  // CHECK-SAME: start_index_map = [1]
  // CHECK-SAME: index_vector_dim = 1
```

**逐项对比**：

| 属性 | 转换前 | 转换后 |
|---|---|---|
| `start_index_map` | `[0, 1]` | **`[1]`** ← 少了 `0` |
| `operand_batching_dims` | （无） | **`[0]`** ← 新增 |
| `start_indices_batching_dims` | （无） | **`[0]`** ← 新增 |
| `offset_dims` | `[1]` | `[1]`（不变） |
| `collapsed_slice_dims` | `[0]` | （未在 CHECK 中列出，保持不变） |

**读法**：
- 原来的 `start_index_map = [0, 1]` 表示"用 `start_indices` 的第 0、1 列分别索引 operand 的第 0、1 维"。
- 但第 0 列其实是 **`iota`**（`[0,1,2,3]`）—— 它不是在"选行"，而是在**遍历行**。
- 识别出这一点后，把它从 `start_index_map` **移出**，改成显式的
  `operand_batching_dims = [0]` + `start_indices_batching_dims = [0]`。

---

## 二、★ 为什么隐式批维让分片规则**无法表达**

这是本课的核心问题。答案在 L2-10 见过的 `gather` 规则里：

（该规则逐字见 **L2-10** 的 `op_sharding_rule_registry.mlir`：因子分类为
`reduction={m, o}`、`need_replication={k, n, p}`，另带
`blocked_propagation={k}`。）

关键在最后一段：**`blocked_propagation={k}`** —— `start_index_map` 涉及的维度上
**不做传播**。

**推理链**：

1. 隐式批维被写在 `start_index_map` 里 → 落在 `blocked_propagation` 的因子上。
2. `blocked_propagation` 意味着**传播不能沿它推导分片**。
3. 但这一维（批维）在语义上**明明是可以分片的** ——
   每个输出元素独立地取一行，切批维是天然的并行方式。
4. 结果：**想切却切不了** —— 分片规则"无法表达"这个意图。

**改成显式批维后**：批维有了专门的属性（`operand_batching_dims` /
`start_indices_batching_dims`），不再落在 `blocked_propagation` 里，
分片规则就能正常处理它了。

> 一句话：**隐式批维把"可并行的批维"伪装成了"不可传播的索引维"**。

---

## 三、五个**不转换**的情形（同样重要）

主文件 10 个用例里有 5 个是"不转换"。它们锁定了变换的**前提条件**：

| 用例 | 不转换的原因 |
|---|---|
| `gather_no_iota_no_transform` | indices 里**没有 iota** —— 没有可识别的批维 |
| `gather_mismatched_batch_size_no_transform` | **批大小不匹配** —— 不能认定为批维 |
| `gather_already_explicit_batching` | **已经是显式批维** —— 无需转换（no-op） |
| `gather_single_index_dim_no_transform` | **`index_vector_dim` 大小为 1** —— 退化成普通索引 |
| `gather_dim0_not_collapsed_no_transform` | **dim 0 不在 `collapsed_slice_dims` 里** —— 形状不满足前提 |

**为什么要测这些**：变换会**改写算子的语义属性**（`start_index_map` 变了）。
条件不满足时贸然改写会**改变程序行为** —— 所以前提条件必须严格，
且必须有测试锁定"不该转的不转"。

---

## 四、可执行验证：5 个 `executable_*` 用例

目录 `executable_explicit_gather_scatter_batching/` 下的 5 个文件用的是
**另一种验证方式**：

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t
```

它们各自包含两个部分：

```mlir
//--- part1.mlir

func.func @transformed_gather(%operand: tensor<4x8xi32>,
    %col_idx: tensor<4x1xi32>) -> tensor<4x1xi32> {
  %iota = stablehlo.iota dim = 0 : tensor<4x1xi32>
  %indices = stablehlo.concatenate %col_idx, %iota, dim = 1
      : (tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x2xi32>
  %result = "stablehlo.gather"(%operand, %indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1>,
    indices_are_sorted = false
  } : (tensor<4x8xi32>, tensor<4x2xi32>) -> tensor<4x1xi32>
  return %result : tensor<4x1xi32>
}

//--- part2.mlir
```

**验证思路**：
- `part1.mlir` 是**待转换的 IR**。
- `part2.mlir` 是**转换后应得到的结果**（含具体的常量数据与期望输出）。
- `run_sdy_interpreter_test.sh` 把变换应用到 part1，然后用 **SDY 解释器执行**，
  与 part2 的结果对比。

**为什么需要执行验证**：这个变换**改写的是算子的语义属性**
（`start_index_map`、`*_batching_dims`）。属性改错了，
IR 依然能通过校验，但**跑出来的数值会错**。
只有真正执行一遍才能确认语义保持。

> 注意 `gather_iota_at_end_of_concat` 里 `start_index_map = [1, 0]` ——
> **iota 不在第一列**。这验证了变换不依赖"iota 必须在首位"。

**5 个可执行用例覆盖的组合**：

| 文件 | 覆盖 |
|---|---|
| `gather_iota_concat_batch_dim` | 基础：iota 在 concat 首位 |
| `gather_iota_reshaped_concat` | iota 先 **reshape** 再 concat |
| `gather_iota_at_end_of_concat` | iota 在 concat **末位** |
| `gather_iota_broadcast_concat` | iota 先 **broadcast** 再 concat |
| `scatter_batch_dim` | **`scatter`** 的对应变换 |

**读法**：前四个覆盖 `gather` 的 indices 构造方式（直接 / reshape / broadcast / 位置），
第五个覆盖 `scatter`（同一个 pass 也处理 scatter）。

---

## 五、主文件 10 个用例速览

| # | 用例 | 类型 |
|---|---|---|
| 1 | `gather_iota_concat_batch_dim` | ✓ 转换 |
| 2 | `gather_iota_reshaped_concat` | ✓ 转换 |
| 3 | `gather_iota_at_end_of_concat` | ✓ 转换 |
| 4 | `gather_iota_in_middle_of_concat` | ✓ 转换 |
| 5 | `gather_iota_broadcast_concat` | ✓ 转换 |
| 6 | `gather_no_iota_no_transform` | ✗ 不转换 |
| 7 | `gather_mismatched_batch_size_no_transform` | ✗ 不转换 |
| 8 | `gather_already_explicit_batching` | ✗ no-op |
| 9 | `gather_single_index_dim_no_transform` | ✗ 不转换 |
| 10 | `gather_dim0_not_collapsed_no_transform` | ✗ 不转换 |

**5 转换 + 5 不转换** —— 这个比例本身就说明了问题：
**改写语义属性的变换，前提条件与转换本身同样重要。**
