<!-- sdy-coverage
transforms/export/test/executable_convert_global_to_local/stablehlo_convolution_shard_batch.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_convolution_shard_batch_group.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_convolution_shard_contracting_dim.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_convolution_shard_feature_group.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_convolution_dual_semantics_factor_indivisible.mlir
-->

# L6-03 · exec-convolution — 源 IR

**用真实数值验证 L5-06 讲的卷积分片规则。**

| 文件 | 行数 |
|---|---|
| `executable_convert_global_to_local/stablehlo_convolution_shard_contracting_dim.mlir` | 58 |
| `executable_convert_global_to_local/stablehlo_convolution_shard_batch_group.mlir` | 51 |
| `executable_convert_global_to_local/stablehlo_convolution_shard_feature_group.mlir` | 45 |
| `executable_convert_global_to_local/stablehlo_convolution_shard_batch.mlir` | 43 |
| `executable_partitioner_pipeline/stablehlo_convolution_dual_semantics_factor_indivisible.mlir` | — |

网格：

```mlir
sdy.mesh @mesh_2 = <["x"=2]>
```

---

## 一、★ 四个文件 = 四种分片维度

**与 L5-06 的六个用例完全对应**：

| 文件 | 分片的维 | `group_count` |
|---|---|---|
| `shard_batch` | **批维 `b`** | 1 |
| `shard_batch_group` | 批维 + 权重输出维 | **2** |
| `shard_contracting_dim` | **输入通道 `i`**（收缩维） | 1 |
| `shard_feature_group` | **特征维 `f`** | **2** |

**两课的分工**：
- **L5-06** 讲**IR 形态**（降级后长什么样）
- **L6-03（本课）** 验证**数值正确性**

---

## 二、★ 测试设计的三个技巧

### 技巧 ①：用 `@sequential_conv` **现算期望值**

```mlir
  %expected = func.call @sequential_conv(%lhs, %rhs) : (tensor<2x4x4x2xi32>, tensor<3x3x2x4xi32>) -> tensor<2x2x2x4xi32>
```

**读法**（**本课最重要的技巧**）：
- 期望值**不是硬编码常量**，而是**调用串行版现算**。
- `@sequential_conv` 从哪来？
  - **脚本自动生成**（如果 part1 里只有 `@parallel_conv`）—— L6-00 讲过
  - **手写**（如果 part1 里显式写了 `@sequential_conv`）
- **这是最自然的"标准答案"** —— 因为 L6-00 讲的机制就是"分片版 vs 串行版"。

**对比 L6-01 / L6-02**：
| 课 | 期望值来源 |
|---|---|
| L6-01 | **硬编码常量**（`dense<1111>`） |
| L6-02 | 硬编码 / `concatenate` 现算 |
| **L6-03** | **`func.call @sequential_conv(...)`** |

**为什么 L6-03 要用现算**：
卷积的期望值**很难手算**（窗口 + 步长 + padding 的组合）——
所以直接调串行版算，**既准确又省力**。

### 技巧 ②：**手动切分输入**

```mlir
  %l0 = "stablehlo.slice"(%lhs) {start_indices=array<i64: 0,0,0,0>, limit_indices=array<i64: 1,4,4,2>, strides=array<i64: 1,1,1,1>} : (tensor<2x4x4x2xi32>) -> tensor<1x4x4x2xi32>
  %l1 = "stablehlo.slice"(%lhs) {start_indices=array<i64: 1,0,0,0>, limit_indices=array<i64: 2,4,4,2>, strides=array<i64: 1,1,1,1>} : (tensor<2x4x4x2xi32>) -> tensor<1x4x4x2xi32>

  %res:2 = "interpreter.run_parallel"(%l0, %rhs, %l1, %rhs) {
    programs = [[@parallel_conv, @parallel_conv]]
  } : (tensor<1x4x4x2xi32>, tensor<3x3x2x4xi32>, tensor<1x4x4x2xi32>, tensor<3x3x2x4xi32>) -> (tensor<1x2x2x4xi32>, tensor<1x2x2x4xi32>)

  %actual = "stablehlo.concatenate"(%res#0, %res#1) {dimension = 0 : i64} : (tensor<1x2x2x4xi32>, tensor<1x2x2x4xi32>) -> tensor<2x2x2x4xi32>
  "check.expect_eq"(%actual, %expected) : (tensor<2x2x2x4xi32>, tensor<2x2x2x4xi32>) -> ()
```

**读法**（**手工模拟"分片 + 拼接"的全过程**）：
- `%lhs` 是完整的 `2x4x4x2`，按**批维**手动 `slice` 成 `%l0`（第 0 批）与 `%l1`（第 1 批）。
- 两台设备各拿一份 `1x4x4x2` —— **这正是批维分片的效果**（`2/2 = 1`）。
- 各自卷积 → `1x2x2x4`。
- **`concatenate` 拼回** `2x2x2x4` —— **这就是"分片计算 + 结果合并"**。
- 与 `%expected`（串行版的结果）比较 ✓

**★ 这验证了什么**：
> **批维分片 = "各算各的批次，再拼起来"** —— 与串行结果一致。
> 这正是 **L5-06 讲的"批维分片无通信"**的数值证据。

### 技巧 ③：**缩放**让贡献可辨识

```mlir
  %l1_unscaled = "stablehlo.slice"(%lhs_base) {start_indices=array<i64: 0,0,0,2>, limit_indices=array<i64: 2,4,4,4>, strides=array<i64: 1,1,1,1>} : (tensor<2x4x4x4xi32>) -> tensor<2x4x4x2xi32>
  // Scale Shard 1 by 1000 to make its contribution identifiable in the sum
  %c1000 = stablehlo.constant dense<1000> : tensor<2x4x4x2xi32>
  %l1 = stablehlo.multiply %l1_unscaled, %c1000 : tensor<2x4x4x2xi32>
  %lhs = "stablehlo.concatenate"(%l0, %l1) {dimension = 3 : i64} : (tensor<2x4x4x2xi32>, tensor<2x4x4x2xi32>) -> tensor<2x4x4x4xi32>
```

**读法**（**注释直接点明了意图**）：
```
// Scale Shard 1 by 1000 to make its contribution identifiable in the sum
```
- 把 **Shard 1 乘以 1000** —— 让它的贡献在**求和**中"**可辨识**"。
- **为什么需要**：`shard_contracting_dim` 的用例里，两个 shard 的贡献会被
  **`all_reduce` 累加**。如果两个 shard 的数值**相近**，
  那么"**正确累加**"与"**只算了一个 shard**"的结果可能**碰巧接近** ——
  测试就**区分不出来**。
- 缩放后，**贡献差异被放大** → 测试的**敏感度提高** ✓

**★ 这是一个通用的测试设计技巧**：
> 当要验证"多个部分的贡献都被正确合并"时，
> **给各部分不同的量级**，让"漏掉一个"必然导致结果明显错误。

---

## 三、★ `shard_contracting_dim`：完整流程

```mlir
func.func @sequential_conv(%arg0: tensor<2x4x4x4xi32>, %arg1: tensor<3x3x4x4xi32>) -> tensor<2x2x2x4xi32> {
```

```mlir
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]}
    {
      feature_group_count = 1 : i64,
      batch_group_count = 1 : i64
    } : (tensor<2x4x4x4xi32>, tensor<3x3x4x4xi32>) -> tensor<2x2x2x4xi32>
  return %0 : tensor<2x2x2x4xi32>
}
```

**读法**：`@sequential_conv` **显式写在 part1 里** ——
没有分片属性，是**纯串行版**。

**脚本不会覆盖它**（L6-00 讲过的 `if` 条件：有 `parallel` 且**没有** `sequential` 时才自动生成）。

```mlir
func.func @parallel_conv(
  %arg0: tensor<2x4x4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}, {}, {"x"}]>},
  %arg1: tensor<3x3x4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}, {"x"}, {}]>}
) -> (tensor<2x2x2x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}, {}, {}]>}) {
```

```mlir
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]}
    {
      feature_group_count = 1 : i64,
      batch_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2, [{}, {}, {}, {}], unreduced={"x"}>]>
    } : (tensor<2x4x4x4xi32>, tensor<3x3x4x4xi32>) -> tensor<2x2x2x4xi32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2, [{}, {}, {}, {}]> : tensor<2x2x2x4xi32>
  return %1 : tensor<2x2x2x4xi32>
}
```

**读法**（**与 L5-06 的 `shard__reduction_factors` 用例同构**）：
- `%arg0` 的**第 3 维（输入通道 `i`，收缩维）**切 `{"x"}`
- `%arg1` 的**第 2 维（输入通道 `i`）**切 `{"x"}`
- **两个收缩维都被切了 `x`** —— 且一致 ✓
- → `convolution` 标 `unreduced={"x"}` → **`all_reduce {"x"}`**

**★ 这就是 L5-06 讲的"归约因子分片需要 all_reduce"的数值验证。**

```mlir
  %seq = func.call @sequential_conv(%lhs, %rhs) : (tensor<2x4x4x4xi32>, tensor<3x3x4x4xi32>) -> tensor<2x2x2x4xi32>
```

```mlir
  %pars:2 = "interpreter.run_parallel"(%l0, %r0, %l1, %r1) {
    programs = [[@parallel_conv, @parallel_conv]]
  } : (tensor<2x4x4x2xi32>, tensor<3x3x2x4xi32>, tensor<2x4x4x2xi32>, tensor<3x3x2x4xi32>) -> (tensor<2x2x2x4xi32>, tensor<2x2x2x4xi32>)
  "check.expect_eq"(%pars#0, %seq) : (tensor<2x2x2x4xi32>, tensor<2x2x2x4xi32>) -> ()
```

**读法**：
- 两台设备各拿**输入通道的一半**（`%l0`/`%l1` 与 `%r0`/`%r1` 都是沿通道维切）。
- `@parallel_conv` 内部有 `all_reduce` → **两台设备的结果都应该是完整的卷积**。
- 断言 `%pars#0 == %seq` —— **只检查设备 0**（因为 `all_reduce` 后两台相同）。

---

## 四、`shard_batch_group` 与 `shard_feature_group`

### `shard_batch_group`：`batch_group_count = 2`

```mlir
  %arg0: tensor<2x4x4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}, {}, {}]>},
  %arg1: tensor<3x3x2x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}, {}, {"x"}]>}
) -> (tensor<1x2x2x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}, {}, {"x"}]>}) {
```

```mlir
      feature_group_count = 1 : i64,
      batch_group_count = 2 : i64,
```

**读法**：
- **输入批维切 `{"x"}`**、**权重输出维（`o`）也切 `{"x"}`**
- `batch_group_count = 2` → **批维缩小**（`2 → 1`）—— L5-06 讲过
- 结果在**输出维 `f`** 上切 `{"x"}`

**★ 这验证了 L5-06 讲的 `batch_group_count` 改变批维大小**。

### `shard_feature_group`：`feature_group_count = 2`

```mlir
  %arg0: tensor<2x4x4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}, {}, {"x"}]>},
  %arg1: tensor<3x3x1x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}, {}, {"x"}]>}
) -> (tensor<2x2x2x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}, {}, {"x"}]>}) {
```

```mlir
      feature_group_count = 2 : i64,
      batch_group_count = 1 : i64,
```

**读法**：
- **输入特征维（`f`）切 `{"x"}`**、**权重特征维也切 `{"x"}`**
- `feature_group_count = 2` → **特征分组**
- 结果也在特征维切 `{"x"}` → **无通信**（输出维分片）

---

## 五、`partitioner_pipeline` 的 `dual_semantics`

第五个文件在**另一个目录**（`executable_partitioner_pipeline/`），
它跑的是**完整分区器流水线**（L6-00 讲过的第二个脚本）。

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false
```

**★ 两个 RUN 行** —— 验证 **L4-09 的 `enable-halo-exchange` 选项**！

```mlir
func.func @parallel_conv(
  %arg0: tensor<2x7x3x1xf32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}, {}, {}]>},
  %arg1: tensor<1x7x3x1xf32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}, {}, {}]>})
  -> (tensor<2x1x1x1xf32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}, {}, {}]>}) {
```

**读法**：
- **用例名 `dual_semantics_factor_indivisible`** ——
  "**双重语义因子**" + "**不可整除**"。
- 输入**无分片**（`[{}, {}, {}, {}]`）—— 分片由**流水线自己插入**。
- **两个 RUN 行**分别跑 `halo_exchange=true/false` ——
  验证**两种模式的结果一致**（与 L4-09 讲的 REPL vs HALO 对照）。

**★ 这验证了 L4-09 的核心结论**：
> **REPL（全复制）与 HALO（halo exchange）语义等价** —— 只是通信量不同。

**注意是 `f32`**（其他 4 个文件是 `i32`）——
因为 halo exchange 涉及浮点运算（`pad` 的填充值等）。

---

## 六、5 个文件的族谱

| 文件 | 验证什么 | 对应课 |
|---|---|---|
| `shard_batch` | 批维分片（无通信） | L5-06 |
| `shard_batch_group` | `batch_group_count` 改变批维 | L5-06 |
| `shard_contracting_dim` | 收缩维分片需 `all_reduce` | L5-06 |
| `shard_feature_group` | `feature_group_count` + 特征维分片 | L5-06 |
| `dual_semantics_factor_indivisible` | **REPL vs HALO 等价** | **L4-09** |

**前四个验证 L5-06，第五个验证 L4-09** ——
这是本课与前面两课的直接对应关系。

**★ 三个测试设计技巧的总结**：
| 技巧 | 目的 |
|---|---|
| ① 用 `@sequential_conv` **现算期望值** | 卷积的期望值**难手算** |
| ② **手动切分输入** | 验证"分片 + 拼接"的全过程 |
| ③ **缩放**让贡献可辨识 | 提高测试**敏感度** |

**一句话总结**：
> **L6-03 用真实数值验证 L5-06 的卷积分片规则** ——
> 四种分片维度各一个文件，外加一个验证 REPL/HALO 等价的流水线测试。
>
> 三个测试设计技巧（现算期望值 / 手动切分 / 缩放）值得单独记住。
