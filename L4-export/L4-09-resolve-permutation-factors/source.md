<!-- sdy-coverage
transforms/export/test/resolve_permutation_factors.mlir
transforms/export/test/resolve_permutation_factors/resolve_permutation_factors_replica_id.mlir
-->

# L4-09 · resolve-permutation-factors — 源 IR

**permutation 因子**（L2-10 讲的第四类因子）在导出期如何被"消解"。

| 文件 | 行数 | 用例数 |
|---|---|---|
| `transforms/export/test/resolve_permutation_factors.mlir` | **1463** | **46** |
| `transforms/export/test/resolve_permutation_factors/resolve_permutation_factors_replica_id.mlir` | — | — |

### ★ 两个 RUN 行 = 两种模式

```mlir
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=false" | FileCheck %s --check-prefixes=CHECK,REPL
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=true" | FileCheck %s --check-prefixes=CHECK,HALO
```

同一个文件跑**两次**，用 `REPL` 与 `HALO` 两组 CHECK 前缀 ——
**本课的主线就是这个开关的差异**。

---

## 一、回顾：什么是 permutation 因子

L2-10 讲过四类因子：`pass-through` / `reduction` / `need_replication` / **`permutation`**。

**`permutation` 的含义**：这个因子在输入与输出之间**尺寸不成整数倍**。

典型来源（本课覆盖的三类算子）：

| 算子 | 为什么是 permutation |
|---|---|
| `convolution` | 窗口 + 步长让空间维变小（16 → 14） |
| `pad` / `slice` | 显式改变维度大小 |
| `reshape` | 维度合并/拆分 |
| `reduce_window` | 窗口滑动 |

**问题**：如果张量**正好在 permutation 因子上有分片**，
那它的分片**无法直接对应**到另一侧 —— 因为"每台设备拿多少"变了。

---

## 二、★ 两种模式：`REPL` 与 `HALO`

以 `convolution_spatial_permutation` 为例。

```mlir
func.func @convolution_spatial_permutation(
    %arg0: tensor<1x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>},
    %arg1: tensor<3x3x1x1xf32>)
    -> (tensor<1x1x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>}) {
```

```mlir
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"a"}, {}]>]>
    }
    : (tensor<1x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x14x14xf32>
```

**输入在第 2 维（空间维）切了 `a`**，而空间维是 permutation（16 → 14）。

### `REPL` 模式（`enable-halo-exchange=false`）

```mlir
  // REPL: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}, {}]> : tensor<1x1x16x16xf32>
  // REPL: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_IN]], %arg1)
  // REPL: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>
  // REPL: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh, [{}, {}, {"a"}, {}]> : tensor<1x1x14x14xf32>
```

**读法**（三步）：
1. 把输入 **reshard 成全复制** `[{}, {}, {}, {}]`（**去掉 `a`**）
2. 在全复制状态下执行 `convolution`
3. 把结果 **reshard 回** `[{"a"}]`

**代价**：全复制意味着**每台设备都要拿到完整的输入** —— 通信量**很大**。

### `HALO` 模式（`enable-halo-exchange=true`）

**做法**：把算子包进 `sdy.manual_computation`，只交换**边界数据**（halo）。

以 `pad_single_left_hop` 为例：

```mlir
  // HALO: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO: %[[MC:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh_a4, [{"a":(2)2}, {"b"}]>, <@mesh_a4, []>] out_shardings=[<@mesh_a4, [{"a":(2)2}, {"b"}]>] manual_axes={"a", "b"} (%[[ARG1:.*]]: tensor<2x4xi32>, %[[ARG2:.*]]: tensor<i32>) {
```

**读法**：
- 用 `manual_computation` 把 `pad` **包起来**。
- `manual_axes={"a", "b"}` —— 两个轴都"冻结"（区域内不再分片）。
- 区域内每台设备处理**自己那一块 + 边界**（halo）。

**代价**：只交换**边界**（几个元素），比全复制**小得多**。

### ★ 两种模式的对照

| | `REPL`（halo=false） | `HALO`（halo=true） |
|---|---|---|
| 做法 | reshard 成**全复制** → 算子 → reshard 回去 | 算子包进 `manual_computation`，交换**边界** |
| 通信量 | **大**（全量复制） | **小**（只交换 halo） |
| IR 复杂度 | 简单（两条 reshard） | 复杂（一个 `manual_computation` 区域） |
| 适用 | 兜底方案 | 有 halo 需求且 hop 数在限制内 |

**这就是 TODOLIST 验收点问的"开关 halo exchange 时 IR 的差别"**。

---

## 三、`pad`：**hop** 的概念

12 个 `pad_*` 用例把 halo 的代价讲得很细。

### `pad_comm_free`：**零通信**

```mlir
func.func @pad_comm_free(
  %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
  -> (tensor<16x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
```

```mlir
  %c = stablehlo.constant dense<0> : tensor<i32>
```

```mlir
  %0 = stablehlo.pad %arg0, %c, low = [4, 0], high = [4, 0], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>} : (tensor<8x8xi32>, tensor<i32>) -> tensor<16x8xi32>
```

**读法**：
- `pad` 的 `low=[4,0], high=[4,0]` —— 在两端各补 4。
- mesh `a=2`，切 8 → 每台设备 4 个元素。
- padding 4 正好是"**一整块**" → 每台设备自己补就行，**不需要任何通信**。
- `CHECK-NOT: sdy.manual_computation` —— **两种模式下都是零通信**。

**这解释了为什么叫 `comm_free`** —— padding 量恰好是整块。

### `pad_single_left_hop`：单跳

```mlir
func.func @pad_single_left_hop(
  %arg0: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a":(2)2}, {"b"}]>})
  -> tensor<7x8xi32> {
```

```mlir
   %c = stablehlo.constant dense<0> : tensor<i32>
```

```mlir
  // REPL: %[[RESHARD:.*]] = sdy.reshard %[[ARG0]] <@mesh_a4, [{}, {"b"}]> : tensor<4x8xi32>
  // REPL: %[[PAD:.*]] = stablehlo.pad %[[RESHARD]], %[[CST]], low ={{.*}}
  // REPL: %[[RES:.*]] = sdy.reshard %[[PAD]] <@mesh_a4, [{"a":(2)2}, {"b"}]> : tensor<7x8xi32>
```

**读法**：`tensor<4x8>` 在 `{"a":(2)2}`（**子轴**，每台 2 个元素）上切，pad 后是 `tensor<7x8>`。
- `REPL`：全复制（两条 reshard）。
- `HALO`：`manual_computation` 包起来，只交换**1 个元素**。

**"hop" 的含义**：需要**跨几台设备**取边界数据。
- `single_left_hop` / `single_right_hop` —— 单跳（只跟邻居要）
- `multiple_left_hops` / `multiple_right_hops_beyond_halo_limit` —— 多跳 / **超出 halo 限制**
- `two_direction_hops` —— 双向都要
- `large_low_pad_within_one_hop` —— padding 很大但**在一跳内**

**`beyond_halo_limit` 很关键**：hop 数有上限 ——
超过上限时 halo exchange **不可行**，只能退回 `REPL`。

### 其余 `pad_*` 用例

| 用例 | 场景 |
|---|---|
| `pad_replicated_negative_low_padding` | **负 padding** + 复制 |
| `pad_replicated_negative_low_and_positive_high` | 一负一正 |
| `pad_replicated_negative_high_padding` | 负 padding |
| `pad_sharded_indivisible_interior_pad` | **interior** padding + 不可整除 |
| `pad_sharded_indivisible_interior_low_and_high` | 三者叠加 |
| `pad_multidim_with_hops` | 多维 hop |

**负 padding** 意味着"裁剪"（`pad` 可以用于切片）。

---

## 四、`reshape`：14 个用例的 permutation 处理

`reshape` 也会产生 permutation 因子（维度合并/拆分）。

| 用例 | 场景 |
|---|---|
| `reshape_1d_to_2d_non_divisible_comm_free` | 不可整除但**零通信** |
| `reshape_2d_to_1d_non_divisible_comm_free` | 反向 |
| `reshape_single_dim_split_comm_free` / `_combine_comm_free` | 单维拆分/合并，零通信 |
| `reshape_indivisible_cross_dims` | **跨维**不可整除 |
| `reshape_2x3x5_to_30_group_padded_size_mismatch` | 尺寸不匹配 |
| `reshape_1d_to_2d_split` | 基础拆分 |
| `reshape_2d_split_with_unrelated_axis` | 带无关轴 |
| `reshape_1d_to_2d_split_gap_2` | 中间隔 2 |
| `reshape_1d_to_3d_split` | 拆成 3 维 |
| `reshape_two_splitting_groups` | 两组同时拆 |
| `reshape_1d_to_2d_split_custom_device_ids` | **自定义设备号** |
| `reshape_mix_split_combine_halo_impossible` | **混合 + halo 不可行** |

**注意命名里的 `comm_free`**：与 `pad_comm_free` 一样，
说明**某些 reshape 在 permutation 因子上仍然零通信**（当分片恰好落在"整齐"的位置）。

`reshape_mix_split_combine_halo_impossible` 则锁定 **halo 不可行**的情形 ——
与 `pad_multiple_right_hops_beyond_halo_limit` 同类。

---

## 五、其余用例与 `replica_id` 文件

### 主文件其余用例

| 用例 | 场景 |
|---|---|
| `convolution_spatial_permutation` | 卷积空间维（本课开篇的例子） |
| `strided_slice` | 带步长的切片 |
| `reduce_window_permutation` | 窗口规约 |
| `reverse_divisible` | 反转（可整除） |

**`reverse_divisible`** 与 `reverse.mlir`（L4-03）呼应：
反转不改变维度大小，所以**可整除时**分片能直接对应。

### `resolve_permutation_factors_replica_id.mlir`

第二个文件处理 **`replica_id`** 场景。

**背景**：`stablehlo.replica_id` 返回当前设备的副本编号 ——
它让程序能**根据设备编号做不同的事**。这与分片有微妙关系：
- 分片改变"哪台设备拿哪片数据"，也就改变了 `replica_id` 的语义。
- 所以在有 `replica_id` 的程序里消解 permutation 因子时，必须**特别小心**。

**为什么单独一个文件**：`replica_id` 是**依赖设备身份**的操作，
与"纯数据并行"的分片模型有本质冲突 —— 需要专门处理。

---

## 六、族谱

| 族 | 用例数 | 核心问题 |
|---|---|---|
| `convolution` | 1 | 空间维 permutation（本课开篇例子） |
| `pad_*` | 12 | **hop 数**与 `comm_free` 的边界 |
| `reshape_*` | 14 | 合并/拆分的 permutation 处理 |
| `strided_slice` / `reduce_window` / `reverse` | 3 | 其它 permutation 来源 |
| 其它 | ~16 | — |
| `replica_id`（第二文件） | — | 设备身份相关 |

**一句话总结**：
> **permutation 因子上的分片无法直接对应。**
> 两条出路：**全复制**（`REPL`，简单但通信大）或
> **halo exchange**（`HALO`，通信小但受 hop 数限制、且 halo 不可行时要退回）。
