<!-- sdy-coverage
transforms/export/test/reshard_to_collectives.mlir
transforms/export/test/reshard_to_collectives_keep_redundant_reshards_true.mlir
-->

# L4-08 · reshard-to-collectives — 源 IR

**L4 的 ★ 核心课**：一条 `sdy.reshard` 如何分解成集合通信序列。

| 文件 | 行数 | 用例数 | RUN 行 |
|---|---|---|---|
| `transforms/export/test/reshard_to_collectives.mlir` | **960** | **98** | `-sdy-reshard-to-collectives` |
| `transforms/export/test/reshard_to_collectives_keep_redundant_reshards_true.mlir` | 27 | 3 | 同上 + `keep-redundant-reshards=true` |

网格（节选）：

```mlir
sdy.mesh @mesh2d = <["x"=2, "y"=2]>
```

```mlir
sdy.mesh @mesh3d = <["x"=2, "y"=2, "z"=2]>
```

```mlir
sdy.mesh @mesh2d_non_iota = <["x"=2, "y"=2], device_ids=[3, 2, 1, 0]>
```

---

## 一、★ 核心对照表：分片变化 → 通信

这是本课最重要的产出。四条规则覆盖绝大多数情形：

| 分片变化 | 通信 | 例子 |
|---|---|---|
| **去掉**某维的轴 | `sdy.all_gather` | `[{"y"},{"x"}]` → `[{"y"},{}]` |
| **加上**某维的轴 | `sdy.all_slice` | `[{},{}]` → `[{"x"},{"y","z"}]` |
| 轴**从一个维移到**另一个维 | `sdy.all_to_all` | `[{"x"},{"y"},{}]` → `[{},{"y"},{"x"}]` |
| 轴在**两个维之间交换** | `sdy.collective_permute` | `[{"x"},{"y"}]` → `[{"y"},{"x"}]` |
| **前后分片相同** | **无（直接删除）** | 冗余 reshard |

### `all_gather`：去掉轴

```mlir
func.func @all_gather_single_axis(%arg0 : tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh2d, [{"y"}, {"x"}]>}) -> tensor<16x8xf32> {
```

```mlir
  %0 = sdy.reshard %arg0 <@mesh2d, [{"y"}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
```

```mlir
  // CHECK-NEXT: sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh2d, [{"y"}, {}]>
```

**读法**：第 1 维从 `{"x"}` 变成 `{}` —— **去掉了 `x`**
→ 用 `all_gather [{}, {"x"}]`（参数是"要聚合掉的轴的位置"）。

### `all_slice`：加上轴

```mlir
func.func @all_slice_multiple_axes(%arg0 : tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{}, {}]>}) -> tensor<16x8xf32> {
```

```mlir
  %0 = sdy.reshard %arg0 <@mesh3d, [{"x"}, {"y", "z"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
```

```mlir
  // CHECK-NEXT: sdy.all_slice [{"x"}, {"y", "z"}] %arg0 out_sharding=<@mesh3d, [{"x"}, {"y", "z"}]>
```

**读法**：从 `[{}, {}]` 变成 `[{"x"}, {"y","z"}]` —— **加上了三个轴**
→ 用 `all_slice [{"x"}, {"y", "z"}]`（参数是**目标分片**）。

> `all_gather` 与 `all_slice` **互逆**（L1-06 / L4-01 都提过）：
> `all_gather` 把轴"收掉"，`all_slice` 把轴"切开"。

### `all_to_all`：轴跨维移动

```mlir
func.func @all_to_all_single_axis(%arg0 : tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{"x"}, {"y"}, {}]>}) -> tensor<16x8x8xf32> {
```

```mlir
  %0 = sdy.reshard %arg0 <@mesh3d, [{}, {"y"}, {"x"}]> : tensor<16x8x8xf32>
  return %0 : tensor<16x8x8xf32>
}
```

```mlir
  // CHECK-NEXT: sdy.all_to_all [{"x"}: 0->2] %arg0 out_sharding=<@mesh3d, [{}, {"y"}, {"x"}]>
```

**读法**：`x` 从**第 0 维**搬到了**第 2 维** —— 注意它**没有消失**，只是换了位置。
→ 用 `all_to_all [{"x"}: 0->2]`（参数是"哪个轴：从哪维到哪维"）。

### `collective_permute`：轴在两维间交换

```mlir
func.func @swap_same_size_axes_between_dims(%arg0 : tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh2d, [{"x"}, {"y"}]>}) -> tensor<16x8xf32> {
```

```mlir
  %0 = sdy.reshard %arg0 <@mesh2d, [{"y"}, {"x"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[COLLECTIVE_PERMUTE:.*]] = sdy.collective_permute %arg0 out_sharding=<@mesh2d, [{"y"}, {"x"}]>
  // CHECK-NEXT: return %[[COLLECTIVE_PERMUTE]]
```

**读法**：第 0 维 `{"x"}` 与第 1 维 `{"y"}` **互换**（**尺寸相同**，都是 2）
→ 用 `collective_permute`。

**为什么不用 `all_to_all`**：交换是"双向的移动"，用 permute 更直接。
注意用例名 `swap_**same_size**_axes_between_dims` —— **尺寸相同**是前提；
尺寸不同的情形另有用例（`swap_diff_size_axes_between_dims`）。

---

## 二、★ 冗余 reshard 的**消除**

7 个用例覆盖"前后分片相同 → 直接删除"。

### 最基本的

```mlir
func.func @redundant_reshard_fully_replicated(%arg0 : tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh2d, [{}, {}]>}) -> tensor<16x8xf32> {
```

```mlir
  %0 = sdy.reshard %arg0 <@mesh2d, [{}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
```

```mlir
  // CHECK-NEXT: return %arg0
```

**读法**：`reshard` 前后都是 `[{}, {}]` → **整条 reshard 被删掉**，
直接 `return %arg0`。

### 跨 mesh 也是冗余

```mlir
func.func @redundant_reshard_fully_replicated_different_meshes(%arg0 : tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh2d_2x3, [{}, {}]>}) -> tensor<16x8xf32> {
```

```mlir
  %0 = sdy.reshard %arg0 <@mesh1d_6, [{}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
```

```mlir
  // CHECK-NEXT: return %arg0
```

**读法**：输入在 `@mesh2d_2x3`（6 台设备）、reshard 到 `@mesh1d_6`（也是 6 台设备），
**两者都是全复制** → **冗余**，删掉。

**关键洞察**：**全复制状态下，网格的具体形状无关紧要** ——
数据在每台设备上都有一份完整副本，换网格不产生任何通信。

### 设备序不同也冗余

```mlir
func.func @redundant_reshard_fully_replicated_same_mesh_different_device_ids(%arg0 : tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh2d, [{}, {}]>}) -> tensor<16x8xf32> {
```

```mlir
  %0 = sdy.reshard %arg0 <@mesh2d_non_iota, [{}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
```

```mlir
  // CHECK-NEXT: return %arg0
```

**读法**：`@mesh2d` 与 `@mesh2d_non_iota` 的**设备顺序不同**（后者是 `[3,2,1,0]`），
但因为都是**全复制** → 仍然冗余。

**这补充了 L4-07 的结论**：设备序不同**通常**需要 reshard（L4-07 的
`meshes_different_device_order` 用例），**但在全复制状态下不需要** ——
因为复制态下没有"哪个设备持有哪片"的问题。

### 其余冗余用例

| 用例 | 场景 |
|---|---|
| `redundant_reshard_fully_replicated_same_empty_meshes` | 相同的空网格 |
| `redundant_reshard_fully_replicated_different_empty_meshes` | 不同的空网格 |
| `redundant_reshard_fully_replicated_input_mesh_nonempty_output_mesh_empty` | 非空网格 → 空网格 |
| `redundant_reshard` | 基础情形 |

### 保留冗余的选项

第二个文件（27 行 / 3 用例）用 `keep-redundant-reshards=true` ——
**不删除**冗余的 reshard。这是**观测性/调试**类选项
（与 L2-03 的 `keep-sharding-rules`、L4-05 的 `mark-partial-result` 同类）。

---

## 三、复杂情形：**先 slice 再 all_to_all**

`slice_on_src_dim_then_all_to_all*` 一族（10 个用例）覆盖"轴跨维移动但尺寸不匹配"。

**问题**：`all_to_all` 要求源维与目标维上的轴**尺寸匹配**。
如果目标维已经有轴、或者尺寸不一致，就要**先 `all_slice` 调整**。

用例名把这个模式写得很清楚：

| 用例 | 场景 |
|---|---|
| `slice_on_src_dim_then_all_to_all` | 基础：先 slice 再 all_to_all |
| `slice_on_src_dim_then_all_to_all_multiple_axes` | 多轴 |
| `slice_on_src_dim_then_two_all_to_alls` | 一个 slice + 两个 all_to_all |
| `slice_on_src_dim_then_two_all_to_alls_diff_tgts` | 两个 all_to_all 目标不同 |
| `slice_on_src_dim_then_all_to_all_and_all_gather` | 再叠一个 all_gather |
| `slice_on_multiple_src_dims` | 多个源维 |
| `slice_on_one_src_dim_but_not_other` | 只处理一个源维 |
| `slice_on_src_dim_considering_existing_axes_on_src_dim` | 考虑源维上已有的轴 |
| `slice_on_src_dim_and_replace_axis_in_another_dim` | 同时替换另一个维的轴 |

### 9 个"不能 slice"的用例

`cannot_slice_on_src_dim_*` 一族锁定**失败条件**：

| 用例 | 不能 slice 的原因 |
|---|---|
| `output_sharded` | 输出维上已有分片 |
| `tgt_dim_sharded` | 目标维上已有分片 |
| `axes_out_of_order` | 轴序不对 |
| `axes_non_contiguous` | 轴不连续 |
| `size_too_small`（×2） | 尺寸太小 |
| `considering_existing_axes_on_src_dim` | 考虑已有轴后不满足 |
| `size_non_divisible` | 尺寸不可整除 |

**为什么这些用例重要**：它们说明"先 slice"**不是万能的** ——
条件不满足时要换别的策略（或放弃）。

---

## 四、其余族速览

| 族 | 用例数 | 场景 |
|---|---|---|
| `reshard_from_sharded_to_fully_replicated*` | 3 | 分片 → 全复制（跨 mesh 的版本） |
| `all_gather_*` | 4 | 单轴 / 多轴 / 多维 / **子轴** |
| `all_slice_*` | 4 | 多轴 / 子轴 / **minor axis** / **缺输入分片** |
| `all_to_all_*` | 8 | 单轴 / 多轴 / 部分轴移动 / 两个 all_to_all 的组合 |
| `replace_*` | 12+ | **替换轴**（同尺寸 / 更大 / 更小；同维 / 跨维） |
| `swap_*` | 6 | **交换轴**（同尺寸 / 不同尺寸；跨维） |
| `reorder_axes_*` | 8 | **重排轴**（单维 / 跨维 / 配合 all_gather / all_to_all） |
| `reorder_device_ids*` | 4 | **重排设备号**（+ 配合 all_gather / all_to_all） |
| `*_gcd_greater_than_one` | 4 | **gcd > 1** 时的特殊处理 |
| 边界 | 4 | `out_unreduced_axes_preserved` / `reshard_with_propagation_barrier` / `single_device_in/out_sharding` |

### 三类"轴操作"的区分

| 操作 | 含义 | 例子 |
|---|---|---|
| **replace** | 用一个轴**替换**另一个 | `{"x"}` → `{"y"}` |
| **swap** | 两个维的轴**互换** | `[{"x"},{"y"}]` → `[{"y"},{"x"}]` |
| **reorder** | 同一个维内轴的**顺序**变化 | `{"x","y"}` → `{"y","x"}` |

**注意 swap 与 reorder 的区别**：
- **swap** 是**跨维**的（两个维之间）
- **reorder** 可以是**同维内**的（一个维里轴的顺序）

### `gcd_greater_than_one` 的含义

4 个用例覆盖"两个轴大小的**最大公约数 > 1**"的情形。

**为什么特殊**：`all_to_all` 等通信要求尺寸匹配。
当 gcd > 1 时，可以**部分匹配**（用子轴切出公共部分），
而不必整体 slice —— 这是更优的策略。

---

## 五、98 + 3 个用例的族谱

| 族 | 用例数 | 核心规则 |
|---|---|---|
| **冗余消除** | 7 | 前后分片相同（含跨 mesh、设备序不同）→ 删除 |
| **分片 → 全复制** | 3 | 跨 mesh 的版本 |
| **`all_gather`** | 4 | 去掉轴 |
| **`all_slice`** | 4 | 加上轴 |
| **`all_to_all`** | 8 | 轴跨维移动 |
| **slice + all_to_all** | 10 | 尺寸不匹配时的组合 |
| **不能 slice** | 9 | 锁定失败条件 |
| **replace / swap / reorder** | 26+ | 三类轴操作 |
| **设备号重排** | 4 | 配合通信算子 |
| **gcd > 1** | 4 | 部分匹配的优化 |
| **边界** | 4 | unreduced / barrier / 单设备 |

**一句话总结**：
> **每一维上的分片变化，对应一种通信。**
> 去掉轴 → `all_gather`；加上轴 → `all_slice`；
> 轴跨维移动 → `all_to_all`；轴互换 → `collective_permute`；无变化 → 删除。
