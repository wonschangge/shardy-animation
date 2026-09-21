<!-- sdy-coverage
transforms/export/test/executable_convert_global_to_local/stablehlo_pad_non_sharded.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_pad_sharded_non_uniform.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_pad_sharded_uniform.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_pad_indivisible.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_pad_interior.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_pad_large_pad.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_pad_large_pad_within_one_hop.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_pad_left_shift.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_pad_multidim_mixed_shifts.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_pad_multidim_with_hops.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_pad_multiple_hops_right_shift.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_pad_multiple_right_hops.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_pad_replica_id.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_pad_replicated_dual_slice_pad.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_pad_replicated_negative_high_padding.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_pad_replicated_negative_low_padding.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_pad_right_shift.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_pad_single_left_hop.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_pad_single_right_hop.mlir
-->

# L6-06 · exec-pad — 源 IR

**本层最大族（19 个文件）** —— `pad` 是 Shardy 测试里最庞大的一族。

| 族 | 文件数 | 位置 |
|---|---|---|
| **分片语义**（uniform / non-uniform / non-sharded） | 3 | `executable_convert_global_to_local/` |
| **halo exchange 边界情形** | 16 | `executable_partitioner_pipeline/` |

网格：

```mlir
sdy.mesh @mesh_2 = <["x"=2]>
```

---

## 一、★ 为什么 `pad` 是"本层最大族"

**答案在 16 个 partitioner 文件的命名里**：

| 命名关键词 | 含义 |
|---|---|
| `hop` | 相邻设备间的数据交换**跳数**（halo exchange 的步数） |
| `shift` | 数据移动**方向**（`left` / `right`） |
| `large_pad` | padding **跨越多个 hop** |
| `multidim` | **多维**同时 padding |
| `replica_id` | 用 `replica_id` 而非 `partition_id` |
| `replicated_*` | **全复制**情形 |
| `negative_*_padding` | **负 padding**（裁剪） |

**★ 这些全是 `halo exchange` 的边界情形**（回顾 **L4-09**）：
> `permutation` 因子（空间维不成整数倍）有两种消解方式 ——
> **REPL**（全复制）与 **HALO**（halo exchange）。
> HALO 需要"和邻居交换边界数据"，而**边界情形极多** ——
> 跳数、方向、维度、正负 padding 的每一种组合都要验证。

**大部分 partitioner 文件有两个 RUN 行**：

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false
```

→ 验证 **HALO 与 REPL 两种模式的结果一致**（L4-09 的核心结论）。

---

## 二、`pad_non_sharded`：padding 在**未分片**的维上

```mlir
sdy.mesh @mesh_2 = <["x"=2]>

// Padding on replicated Dim 1, sharded on Dim 0.
func.func @pad_uniform(
  %arg0: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
  -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c0, low = [0, 1], high = [0, 1], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} : (tensor<4x2xi32>, tensor<i32>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}
```

```mlir
func.func @main() {
```

```mlir
  %input = stablehlo.constant dense<[[1, 2], [3, 4], [5, 6], [7, 8]]> : tensor<4x2xi32>

  %s0 = "stablehlo.slice"(%input) {start_indices=array<i64: 0,0>, limit_indices=array<i64: 2,2>, strides=array<i64: 1,1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices=array<i64: 2,0>, limit_indices=array<i64: 4,2>, strides=array<i64: 1,1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>

  %res:2 = "interpreter.run_parallel"(%s0, %s1) {
    programs = [[@pad_uniform, @pad_uniform]]
  } : (tensor<2x2xi32>, tensor<2x2xi32>) -> (tensor<2x4xi32>, tensor<2x4xi32>)

  %e0 = stablehlo.constant dense<[[0, 1, 2, 0], [0, 3, 4, 0]]> : tensor<2x4xi32>
  "check.expect_eq"(%res#0, %e0) : (tensor<2x4xi32>, tensor<2x4xi32>) -> ()
  return
}
```

**读法**：
- 注释：`Padding on replicated Dim 1, sharded on Dim 0.`
- `low = [0, 1], high = [0, 1]` —— **只在第 1 维（未分片维）padding**。
- 每台设备本地 `2x2` → `2x4`，**local pad 参数相同** → **无通信** ✓
- **期望值**：设备 0 的 `[[1,2],[3,4]]` → **`[[0,1,2,0],[0,3,4,0]]`**
  —— 第 1 维两侧各补一个 `0` ✓

**★ 这正是 L5-04 的"作用维判据"**：
> padding 的维**未分片** → 每台设备的"填充位置"相同 → **参数不变、无通信**。

**注意只断言了 `%res#0`** —— 因为两台设备的 local pad 相同，
且断言的是**局部结果**（不是全局）。

---

## 三、★ `pad_sharded_uniform`：`pLow + pHigh = pInt`

```mlir
sdy.mesh @mesh_2 = <["x"=2]>

// Padding on sharded Dim 0, with pLow + pHigh = pInt, which is uniform.
func.func @pad_sharded_uniform(
  %arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>})
  -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c0, low = [1], high = [0], interior = [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}]>]>} : (tensor<4xi32>, tensor<i32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}
```

```mlir
func.func @main() {
```

```mlir
  %input = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %s0 = "stablehlo.slice"(%input) {start_indices=array<i64: 0>, limit_indices=array<i64: 2>, strides=array<i64: 1>} : (tensor<4xi32>) -> tensor<2xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices=array<i64: 2>, limit_indices=array<i64: 4>, strides=array<i64: 1>} : (tensor<4xi32>) -> tensor<2xi32>

  %res:2 = "interpreter.run_parallel"(%s0, %s1) {
    programs = [[@pad_sharded_uniform, @pad_sharded_uniform]]
  } : (tensor<2xi32>, tensor<2xi32>) -> (tensor<4xi32>, tensor<4xi32>)

  %e0 = stablehlo.constant dense<[0, 1, 0, 2]> : tensor<4xi32>
  %e1 = stablehlo.constant dense<[0, 3, 0, 4]> : tensor<4xi32>

  "check.expect_eq"(%res#0, %e0) : (tensor<4xi32>, tensor<4xi32>) -> ()
  "check.expect_eq"(%res#1, %e1) : (tensor<4xi32>, tensor<4xi32>) -> ()
  return
}
```

**读法**（**本课的核心概念**）：
- 注释：`Padding on sharded Dim 0, with pLow + pHigh = pInt, which is uniform.`
- **`pLow + pHigh = 1 + 0 = 1 = pInt`** → **均匀** ✓
- **在分片维上 padding** —— 但每台设备的 local pad 参数**相同**（都是 `low=1, interior=1`）。

**逐设备推结果**（输入 `[1,2,3,4]`，`x=2` → 每台 2 个元素）：

| 设备 | 输入 | 局部结果 | 全局位置 |
|---|---|---|---|
| 0 | `[1, 2]` | **`[0, 1, 0, 2]`** | 第 0~3 位 |
| 1 | `[3, 4]` | **`[0, 3, 0, 4]`** | 第 4~7 位 |

**全局拼接** = `[0, 1, 0, 2, 0, 3, 0, 4]` ✓
—— `low=1` 在最前面补 `0`、`interior=1` 在每两个元素间补 `0`。

**★ 为什么"均匀"**：
> 每台设备都在**自己那段的开头**补 `low=1` 个 `0`，
> 并在**元素之间**补 `interior=1` 个 `0` —— **参数完全相同**。
>
> 这是因为 `pLow + pHigh = pInt`：边界的 padding 恰好等于 interior 的 padding，
> 所以**段与段之间的边界**看起来和**段内**一样。

**★ 这解释了 L5-09 讲的"uniform vs non-uniform"** ——
那里我从 `pad_for_divisibility` 的用例名**推断**出这个区分，这里看到了**实际语义**。

---

## 四、★ `pad_sharded_non_uniform`：`pLow + pHigh > pInt`

```mlir
// Padding on sharded Dim 0, with pLow + pHigh > pInt, which is not uniform.
```

```mlir
  %arg0: tensor<2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>})
```

```mlir
  -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) {
```

```mlir
  %0 = stablehlo.pad %arg0, %c0, low = [2], high = [2], interior = [2]
```

```mlir
  %e0 = stablehlo.constant dense<[9, 9, 1, 9]> : tensor<4xi32>
  %e1 = stablehlo.constant dense<[9, 2, 9, 9]> : tensor<4xi32>
  "check.expect_eq"(%res#0, %e0) : (tensor<4xi32>, tensor<4xi32>) -> ()
```

**读法**：
- 注释：`Padding on sharded Dim 0, with pLow + pHigh > pInt, which is not uniform.`
- **`pLow + pHigh = 2 + 2 = 4 > pInt = 2`** → **非均匀** ✓
- 输入 `tensor<2xi32>` 切 `{"x"}` → **每台 1 个元素**。
- 结果 `8xi32`。

**逐设备推结果**（假设输入 `[1, 2]`）：

| 设备 | 输入 | 局部结果 |
|---|---|---|
| 0 | `[1]` | **`[9, 9, 1, 9]`** |
| 1 | `[2]` | **`[9, 2, 9, 9]`** |

**★ 两台设备的填充分布不同**：
- 设备 0：`9` 在**前两个**位置
- 设备 1：`9` 在**第 2 和第 4** 个位置

**这就是"非均匀"** —— 首/尾设备的 padding 分布与中间设备**不同**。

**★ 填充值是 `9` 而非 `0`**：
测试**故意**用 `9` 作为填充值 —— 这样能**区分**"填充值"与"真实数据"
（如果填 `0`，就可能与真实数据的 `0` 混淆）。这是 L6-03 讲的"缩放技巧"的同源手法。

**★ uniform vs non-uniform 的判据**：

| 情形 | 判据 | 结果 |
|---|---|---|
| **uniform** | `pLow + pHigh = pInt` | 每台 local pad **相同** |
| **non-uniform** | `pLow + pHigh > pInt` | 首/尾设备与中间设备**不同** |

---

## 五、`indivisible`：L5-09 的完整实例

这个文件的注释**写得最详细**，是 L5-09 的完整实例：

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"
```

```mlir
// The pad input is sliced from 8x2 to 7x2 (indivisible, padded to 8x2).
// The original pad result size (10) is also indivisible by mesh axis size (4).
// PadForDivisibility should adjust high padding of the pad op to 3 (instead of 2)
// to make the result 12x2 (divisible). The result is then trimmed to 10x2
// after the final reshard (all_gather).
```

**读法**（**逐句对应 L5-09**）：
1. **输入 `8x2` → `7x2`** —— **不可整除**（`7` 不能被 mesh 轴大小整除）→ **补到 `8x2`**。
2. **pad 结果 `10` 也不能被 mesh 轴大小 `4` 整除** —— 又一次不可整除！
3. **`PadForDivisibility` 把 `high` 从 `2` 调到 `3`** → 结果变成 **`12x2`**（`12 = 4 × 3` 可整除）✓
4. **最后 `all_gather` 后裁回 `10x2`**。

**★ 这就是 L5-09 讲的"pad 补齐 + slice 裁回"的完整流程**：
```
7x2（不可整除）
  -> pad 到 8x2
  -> 结果 10 又不可整除 -> high 从 2 调到 3
  -> 12x2（可整除）
  -> all_gather
  -> 裁回 10x2 ✓
```

**★ 注意两个 RUN 行用的是位置参数**：

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"
```

**这正好印证了 L6-00 讲的脚本参数解析**：
> 脚本支持**位置参数** —— 第 1 个是 `src`、第 2 个是 `temp_dir`、
> **第 3 个是 `ENABLE_HALO_EXCHANGE`**。
> 所以这里的 `"true"` / `"false"` 就是设置 `ENABLE_HALO_EXCHANGE`。

---

## 六、19 个文件的族谱

### `executable_convert_global_to_local/`（3 个 —— **分片语义**）

| 文件 | 判据 | 通信 |
|---|---|---|
| `pad_non_sharded` | padding 在**未分片**维 | **无** |
| `pad_sharded_uniform` | `pLow + pHigh = pInt` | 无（local 参数相同） |
| `pad_sharded_non_uniform` | `pLow + pHigh > pInt` | 首尾设备不同 |

### `executable_partitioner_pipeline/`（16 个 —— **halo exchange 边界情形**）

| 族 | 文件 |
|---|---|
| **跳数** | `single_left_hop`、`single_right_hop`、`multiple_right_hops`、`multiple_hops_right_shift` |
| **方向** | `left_shift`、`right_shift` |
| **大 padding** | `large_pad`、`large_pad_within_one_hop` |
| **多维** | `multidim_mixed_shifts`、`multidim_with_hops` |
| **全复制** | `replicated_dual_slice_pad`、`replicated_negative_high_padding`、`replicated_negative_low_padding` |
| **其它** | `indivisible`、`interior`、`replica_id` |

**大部分有两个 RUN 行**（`--enable_halo_exchange=true/false`）
→ 验证 **HALO 与 REPL 等价**（L4-09）。

**★ 本课的三条结论**：

1. **`pad` 之所以是最大族，是因为 halo exchange 的边界情形极多** ——
   跳数 × 方向 × 维度 × 正负 padding 的每种组合都要验证。
2. **uniform / non-uniform 的判据是 `pLow + pHigh` 与 `pInt` 的关系** ——
   相等则每台 local pad 相同，大于则首尾设备不同。
3. **`indivisible` 是 L5-09 的完整实例** ——
   `7x2` 补到 `8x2`、结果 `10` 又不可整除故 `high` 从 `2` 调到 `3`、
   最终 `12x2` 再裁回 `10x2`。

**一句话总结**：
> **`pad` 是本层最大族，因为它是 halo exchange 的实现载体** ——
> 16 个 partitioner 文件覆盖了跳数/方向/维度/正负 padding 的各种边界情形，
> 而 3 个 `convert_global_to_local` 文件讲清了 uniform / non-uniform 的判据。
