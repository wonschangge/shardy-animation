<!-- sdy-coverage
transforms/export/test/executable_convert_global_to_local/stablehlo_gather_shard_reduction_dim_is_collapsed.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_gather_shard_reduction_dim_is_collapsed_i32.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_gather_shard_reduction_dim_is_collapsed_max.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_gather_shard_reduction_dim_is_collapsed_min.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_gather_shard_reduction_dim_is_collapsed_not_in_start_index_map.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_gather_shard_two_reduction_dims.mlir
-->

# L6-05 · exec-gather — 源 IR

**L5-08 索引重映射的数值验证** —— 本层 P0 的最复杂一课。

| 文件 | 行数 |
|---|---|
| `executable_convert_global_to_local/stablehlo_gather_shard_two_reduction_dims.mlir` | 78 |
| `executable_convert_global_to_local/stablehlo_gather_shard_reduction_dim_is_collapsed_not_in_start_index_map.mlir` | 60 |
| `executable_convert_global_to_local/stablehlo_gather_shard_reduction_dim_is_collapsed.mlir` | 59 |
| `executable_convert_global_to_local/stablehlo_gather_shard_reduction_dim_is_collapsed_i32.mlir` | 59 |
| `executable_convert_global_to_local/stablehlo_gather_shard_reduction_dim_is_collapsed_max.mlir` | 59 |
| `executable_convert_global_to_local/stablehlo_gather_shard_reduction_dim_is_collapsed_min.mlir` | 59 |

网格：

```mlir
sdy.mesh @mesh_2 = <["x"=2]>
```

---

## 一、基准文件：`is_collapsed`

### `part1.mlir`：串行版 + 分片版

```mlir
sdy.mesh @mesh_2 = <["x"=2]>

// Sequential Baseline: Explicitly written, no shardings.
func.func @sequential_gather(%arg0: tensor<4x2xf32>, %arg1: tensor<2xi64>) -> tensor<2x2xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>
  } : (tensor<4x2xf32>, tensor<2xi64>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// Parallel in global view, to be converted to a device function.
//
// ([i, j], [k]) -> ([k, j]) reduction={i}
// The sharded dim is a reduction dimensions and it is also a collapsed dim.
func.func @parallel_gather(
  %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg1: tensor<2xi64>) -> tensor<2x2xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2, [{}, {}], unreduced={"x"}>]>
  } : (tensor<4x2xf32>, tensor<2xi64>) -> tensor<2x2xf32>

  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2, [{}, {}]> : tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}
```

**读法**：
- **注释直接给出规则**：`([i, j], [k]) -> ([k, j]) reduction={i}`
- `collapsed_slice_dims = [0]` —— 第 0 维是**被压掉的维**，也是 `reduction` 因子 `i`。
- `%arg0` 的**第 0 维（归约维）**切 `{"x"}` → **归约因子分片** → `unreduced={"x"}` + `all_reduce` ✓

### `part2.mlir`：索引跨两台设备

```mlir
func.func @main() {
```

```mlir
  %input = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>
  %indices = stablehlo.constant dense<[1, 3]> : tensor<2xi64>

  %seq = func.call @sequential_gather(%input, %indices) : (tensor<4x2xf32>, tensor<2xi64>) -> tensor<2x2xf32>

  %shard0 = "stablehlo.slice"(%input) {
    start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>
  } : (tensor<4x2xf32>) -> tensor<2x2xf32>
  %shard1 = "stablehlo.slice"(%input) {
    start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>
  } : (tensor<4x2xf32>) -> tensor<2x2xf32>

  %pars:2 = "interpreter.run_parallel"(%shard0, %indices, %shard1, %indices) {
    programs = [[@parallel_gather, @parallel_gather]]
  } : (tensor<2x2xf32>, tensor<2xi64>, tensor<2x2xf32>, tensor<2xi64>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  "check.expect_eq"(%pars#0, %seq) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
  "check.expect_eq"(%pars#1, %seq) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
  return
}
```

**读法**（**本课的关键场景**）：
- `%input` 是 `4x2`，按第 0 维切成 `%shard0`（第 0~1 行）与 `%shard1`（第 2~3 行）。
- **`%indices = [1, 3]` 是共享的** —— 两台设备拿**同一份索引**
  （`run_parallel(%shard0, %indices, %shard1, %indices)`）。
- **索引 `1` 属于设备 0、索引 `3` 属于设备 1** —— **各自只有一个属于自己**！

**★ 这正是 L5-08 讲的索引重映射场景**：

| 设备 | 持有的行 | 索引 `1` | 索引 `3` |
|---|---|---|---|
| 0 | 第 0~1 行 | **属于我** → gather 得 `[3,4]` | 不属于 → **填填充值** |
| 1 | 第 2~3 行 | 不属于 → **填填充值** | **属于我** → gather 得 `[7,8]` |

**然后 `all_reduce` 合并** → 两台设备都得到 `[[3,4],[7,8]]` = `%seq` ✓

**★ 这验证了 L5-08 八步中的第 ⑤⑦⑧ 步**（mask / select 填充 / all_reduce）。

---

## 二、★★ 实测发现：填充值**不是固定的 0**

**这是本课最有价值的发现。**

我用 `sdy_opt` 复现了三个变体的降级输出（命令与 `run_sdy_interpreter_test.sh` 一致）：

```bash
sdy_opt part1.mlir --sdy-convert-global-to-local \
  --sdy-inline-meshes --sdy-drop-sharding-and-mesh --allow-unregistered-dialect
```

**三个变体的输出只有两处不同**：

```text
=== 基准 vs min 的差异 ===
26c26
<     %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>
---
>     %cst = stablehlo.constant dense<0x7F800000> : tensor<2x2xf32>
30c30
<       %17 = stablehlo.add %arg2, %arg3 : tensor<f32>
---
>       %17 = stablehlo.minimum %arg2, %arg3 : tensor<f32>
```

```text
=== 基准 vs max 的差异 ===
26c26
<     %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>
---
>     %cst = stablehlo.constant dense<0xFF800000> : tensor<2x2xf32>
30c30
<       %17 = stablehlo.add %arg2, %arg3 : tensor<f32>
---
>       %17 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
```

### ★ 汇总

| 归约种类 | `select` 的填充值 | 十六进制 | 含义 |
|---|---|---|---|
| **sum**（基准） | `dense<0.000000e+00>` | `0x00000000` | **`0`** |
| **min** | `dense<0x7F800000>` | `0x7F800000` | **`+∞`** |
| **max** | `dense<0xFF800000>` | `0xFF800000` | **`−∞`** |

**★ 填充值是该归约的「单位元」（identity element）**：
- `sum` → **`0`**（加法单位元：`0 + x = x`）
- `min` → **`+∞`**（`min(+∞, x) = x`）
- `max` → **`−∞`**（`max(−∞, x) = x`）

**★ 这同时印证并深化了两课**：

| 课 | 当时说的 | 本课的实测 |
|---|---|---|
| **L5-08** | gather 的 mask "**填 0**" | **只对 `sum` 成立** —— 那是基准文件的情形 |
| **L5-09** | "填充值必须是该运算的**单位元**" | **完全证实** —— gather 的填充值也遵守这条 |

**★ 为什么 `min` / `max` 必须单独测试**：
> 如果用 `sum` 的 `0` 去配 `min` 归约，结果会**错** ——
> 因为 `min(0, 3) = 0` 而不是 `3`（0 会变成"最小值"）。

**★ 这是 AGENTS.md §3.3 那条红线的一次直接收获**：
> 只看基准文件的 IR 形态，会以为"永远是填 0"。
> **跑三个变体才看到真实规律** —— "凡测试文件里写着但没见过实际输出的，必须跑一遍确认"。

**另注意**：`%cst` 的类型是 `tensor<2x2xf32>` ——
填充值是**张量常量**（形状同结果）；reducer 区域内的算子（`add`/`minimum`/`maximum`）也随种类变化。

---

## 三、`_i32` 变体：索引类型无关

`_i32` 与基准的差异**只在索引类型**：

```text
< func.func @sequential_gather(%arg0: tensor<4x2xf32>, %arg1: tensor<2xi64>) -> tensor<2x2xf32> {
> func.func @sequential_gather(%arg0: tensor<4x2xf32>, %arg1: tensor<2xi32>) -> tensor<2x2xf32> {
```

**读法**：`i64` → `i32` —— **索引类型不影响降级逻辑**。
测试它的目的是确认**索引重映射对 `i32` 索引同样正确**
（`partition_id` 的转换、查找表的下标运算都要适配）。

---

## 四、`not_in_start_index_map`：collapsed 维**不在** `start_index_map`

与基准的差异（结果形状从 `2x2` 变成 `2x1`）：

```text
< func.func @sequential_gather(%arg0: tensor<4x2xf32>, %arg1: tensor<2xi64>) -> tensor<2x2xf32> {
> func.func @sequential_gather(%arg0: tensor<4x2xf32>, %arg1: tensor<2xi64>) -> tensor<2x1xf32> {
```

**读法**：
- 基准的 `start_index_map = [0]` —— collapsed 维**在** `start_index_map` 里。
- 本变体的 collapsed 维**不在** `start_index_map` 里 ——
  意味着这一维**被压掉但不参与索引**（`slice_sizes` 在该维为 1）。
- **`start_index_map` 不含它** → **不需要重映射索引**（因为没有索引指向它）。

**回顾 L2-10**：`gather` 的规则里 `blocked_propagation` 涉及的正是 `start_index_map` ——
本变体验证了那个因子的一个边界情形。

---

## 五、`two_reduction_dims`：两个归约维

独立文件（78 行），网格是 `2x2`：

```mlir
// ([r1, b, r2], [b, i, v]) -> ([b, i]) reduction={r1, r2}
```

```mlir
    %arg0: tensor<4x2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {}, {"y"}]>},
```

```mlir
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_2, [{}, {}], unreduced={"x", "y"}>]>
```

```mlir
  %1 = sdy.all_reduce {"x", "y"} %0 out_sharding=<@mesh_2_2, [{}, {}]> : tensor<2x1xf32>
```

**读法**：
- 规则 `([r1, b, r2], [b, i, v]) -> ([b, i]) reduction={r1, r2}` ——
  **两个归约维** `r1`、`r2`。
- `%arg0` 的**第 0 维（`r1`）切 `{"x"}`**、**第 2 维（`r2`）切 `{"y"}`** ——
  **两个归约维各切一个轴**。
- → `unreduced={"x", "y"}` → **`all_reduce {"x", "y"}`** ✓

**★ 这验证了 L5-08 讲的"`shard_two_of_three_reduction_dims`"情形** ——
多个归约维可以**分别用不同的轴**分片，`all_reduce` 一次归约所有轴。

---

## 六、六个文件的族谱

| 文件 | 验证什么 | 与基准的差异 |
|---|---|---|
| `..._is_collapsed` | 基准（`sum` 归约） | — |
| `..._is_collapsed_i32` | **索引类型无关** | `i64` → `i32` |
| `..._is_collapsed_min` | **`min` 归约** | 填充值 `0` → **`+∞`** |
| `..._is_collapsed_max` | **`max` 归约** | 填充值 `0` → **`−∞`** |
| `..._not_in_start_index_map` | collapsed 维**不在** `start_index_map` | 结果 `2x2` → `2x1` |
| `..._two_reduction_dims` | **两个归约维**各切一个轴 | 网格 `2x2`、`unreduced={"x","y"}` |

**★ 本课的三条结论**：

1. **L5-08 的索引重映射得到数值验证** ——
   mask / select 填充 / `all_reduce` 三步配合，结果与串行版一致。
2. **★ 填充值是归约的单位元，不是固定的 0** ——
   `sum`→`0`、`min`→`+∞`、`max`→`−∞`。这深化了 L5-08 的表述，
   也印证了 **L5-09** 的"填充值必须是单位元"原则。
3. **索引类型（`i32`/`i64`）不影响降级逻辑**。

**一句话总结**：
> **L6-05 用数值验证了 L5-08 的索引重映射** ——
> 并且实测发现 **mask 的填充值是归约的单位元**（`min`→`+∞`、`max`→`−∞`），
> 这与 **L5-09** 提炼的"填充值必须是单位元"原则**完全一致**。
