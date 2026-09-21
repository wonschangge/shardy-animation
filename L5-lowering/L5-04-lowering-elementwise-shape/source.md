<!-- sdy-coverage
transforms/export/test/convert_global_to_local/stablehlo_iota.mlir
transforms/export/test/convert_global_to_local/stablehlo_concatenate.mlir
transforms/export/test/convert_global_to_local/stablehlo_slice.mlir
transforms/export/test/convert_global_to_local/stablehlo_pad.mlir
-->

# L5-04 · lowering-elementwise-shape — 源 IR

**形状类算子在分片下的局部化。**

| 文件 | 行数 | 用例数 |
|---|---|---|
| `convert_global_to_local/stablehlo_pad.mlir` | 139 | 8 |
| `convert_global_to_local/stablehlo_iota.mlir` | 58 | 4 |
| `convert_global_to_local/stablehlo_slice.mlir` | 51 | 4 |
| `convert_global_to_local/stablehlo_concatenate.mlir` | 52 | 2 |

**合计 300 行 / 18 用例。**

网格：

```mlir
sdy.mesh @mesh_4 = <["x"=4]>
```
```mlir
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
```

---

## 一、★ 核心判据：算子的"作用维"是不是分片维

形状类算子的处理方式**取决于一个判据**：

> **这个算子的"作用维"（iota 的 dim、slice 的切片维、pad 的填充维、concat 的拼接维）
> 是不是被分片的那一维？**

| 情形 | 处理 |
|---|---|
| **作用维未被分片** | **直接局部化** —— 参数不变，只改类型 |
| **作用维被分片** | **需要特殊处理** —— 补偿偏移 / 调整边界 |

---

## 二、`stablehlo_iota`：分片维上要**补偿偏移**

### `non_sharded`：无分片

```mlir
func.func @non_sharded() -> tensor<16xi32> {
```
```mlir
  %0 = stablehlo.iota dim = 0 : tensor<16xi32>
  return %0 : tensor<16xi32>
```



**读法**：无分片 → 直接转换，**类型不变**。

### `iota_on_non_sharded_dim`：**作用维未分片**

```mlir
func.func @iota_on_non_sharded_dim() -> (tensor<8x16xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {"x"}]>}) {
```
```mlir
  %0 = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {"x"}]>]>} : tensor<8x16xi32>
```

```mlir
  return %0 : tensor<8x16xi32>
```





**读法**：
- `iota dim = 0` 生成 `[0, 1, 2, ..., 7]`（沿第 0 维）。
- **第 0 维未被分片**（分片在 `{"x"}` 即第 1 维）→ iota 的**值不需要调整**。
- 只是类型从 `8x16` 变成 `8x4`（第 1 维 `16/4 = 4`）。
- **`dim = 0` 保持不变** ✓

### `iota_on_sharded_dim`：**作用维被分片** → **必须补偿**

```mlir
func.func @iota_on_sharded_dim() -> (tensor<8x16xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {"x"}]>}) {
```



**读法**（本课最精妙的一处）：
- `iota` 沿**第 1 维**（`dim = 1`）生成，而第 1 维**正是被 `x=4` 切分的**。
- 全局的 iota 是 `[0, 1, 2, ..., 15]`（16 个值）。
- 但设备 0 该拿 `[0,1,2,3]`、设备 1 该拿 `[4,5,6,7]`……
- **`stablehlo.iota dim = 1 : tensor<8x4xi32>` 只会生成 `[0,1,2,3]`** ——
  每台设备都从 0 开始！**错了**。
- → 必须**加上偏移**：用 `partition_id` + 查找表 `[0, 4, 8, 12]` 得到"我的起始值"，
  再加到本地 iota 上。

**查找表 `[0, 4, 8, 12]` 的含义**：4 台设备各自的**起始序号** ——
设备 0 → 0、设备 1 → 4、设备 2 → 8、设备 3 → 12（每台 4 个值）。

**这是"位置相关算子"的典型问题**：
> `iota` 的输出**依赖元素的位置**。分片后每台设备只看到自己那段，
> 必须**补上"我在全局中的位置"**才能得到正确的值。

**与 L5-01 常量处理的对比**：

| | 常量 | iota |
|---|---|---|
| 问题 | 每台设备内容不同 | 每台设备**序号起点**不同 |
| 解法 | 全局常量 + 切片 | **本地 iota + 加偏移** |
| 相同点 | 都用 `partition_id` + 查找表 | |

**一个更简单的选择**：为什么不直接用 `dynamic_slice` 切全局 iota？
—— 因为**全局 iota 本身也要算出来**（16 个值），
而"本地 iota（4 个值）+ 加偏移"**更省**。

---

## 三、`stablehlo_concatenate`：非拼接维分片时**直接转换**

```mlir
// RUN: sdy_opt %s -sdy-convert-global-to-local -split-input-file -verify-diagnostics | FileCheck %s
```

### `not_sharded`

```mlir
func.func @not_sharded(
  %arg0: tensor<16x4xf32>,
  %arg1: tensor<16x4xf32>)
  -> tensor<16x8xf32> {
```
```mlir
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<16x4xf32>, tensor<16x4xf32>) -> tensor<16x8xf32>
```

```mlir
  return %0 : tensor<16x8xf32>
```



**读法**：无分片 → 直接转换，类型不变。

### `sharded_non_concat_dim`：**分片在非拼接维**

```mlir
func.func @sharded_non_concat_dim(
  %arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg1: tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg2: tensor<16x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
```



**读法**：
- 拼接维是**第 1 维**，而分片在**第 0 维**（`{"x"}`）—— **不冲突**。
- 各操作数**各自局部化**：`16x4` → `8x4`、`16x2` → `8x2`、`16x1` → `8x1`。
- 拼接结果：`8x(4+2+1) = 8x7` ✓
- **拼接维的大小不变**（`4+2+1=7`）—— 因为每台设备都拼接了**同样多**的列。

**为什么可以直接转换**：拼接是**逐元素对齐**的操作 ——
每台设备在第 1 维上做同样的拼接，互不影响。

**隐含的另一面**：如果分片**在拼接维**上，情况就复杂了 ——
每台设备该拿哪几段？这需要额外的通信或切片。
（本文件只覆盖了非拼接维的情形。）

---

## 四、`stablehlo_slice`：切片维未分片时**参数不变**

### `replicated_after_all_gather`

```mlir
func.func @replicated_after_all_gather(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"y"}, {}]>}) -> tensor<4x8xf32> {
```
```mlir
  %0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<8x16xf32>
```

```mlir
  %1 = stablehlo.slice %0 [0:4, 0:8] : (tensor<8x16xf32>) -> tensor<4x8xf32>
```

```mlir
  return %1 : tensor<4x8xf32>
```



**读法**：
- `all_gather` 先把 `y` 收掉（`[{"y"}, {}]` → `[{}, {}]`），
  局部类型从 `4x16` 变成 `8x16`（第 0 维恢复完整）。
- 然后 `slice [0:4, 0:8]` —— **切片参数与全局一致**！
- 因为 `all_gather` 之后是**全复制**状态，每台设备都有完整数据 → 切片是**本地操作**。

### `slicing_dim_not_sharded`：**切片维未分片**

```mlir
func.func @slicing_dim_not_sharded(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"x"}]>})
    -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"x"}]>}) {
```
```mlir
  %0 = stablehlo.slice %arg0 [4:12:2, 0:32] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"x"}]>]>} : (tensor<16x32xf32>) -> tensor<4x32xf32>
```

```mlir
  return %0 : tensor<4x32xf32>
```

```mlir
}
```





**读法**（**逐项对比**）：

| 参数 | 全局 | 局部 | 变化 |
|---|---|---|---|
| 第 0 维（**未分片**） | `4:12:2` | **`4:12:2`** | **不变** ✓ |
| 第 1 维（**分片**） | `0:32` | **`0:8`** | `32/4 = 8` |
| 输入类型 | `16x32` | `16x8` | 第 1 维 ÷4 |
| 结果类型 | `4x32` | `4x8` | 第 1 维 ÷4 |

**关键**：
- **切片维（第 0 维）的参数完全不变** —— 因为那一维没被分片，
  每台设备看到的第 0 维是**完整**的。
- **只有分片维（第 1 维）的参数需要缩放**（`0:32` → `0:8`）。

**为什么切片维未分片时参数不变**：
切片是**本地操作** —— 每台设备对自己的那份数据切一刀。
如果切片维是完整的，那"切哪里"就是确定的。

**隐含的另一面**：如果**切片维被分片**，起始/结束就要按设备调整
（类似 iota 的偏移补偿）。

---

## 五、`stablehlo_pad`：**边界处理**最复杂

8 个用例，是本课最大的一族。

### `pad_non_sharded_dim`：填充维未分片

**规则与 slice 相同**：填充维未分片 → 参数不变，只缩放分片维。

### `replicated_after_all_gather`：先 gather 再 pad

```mlir
func.func @replicated_after_all_gather(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) -> tensor<10x18xf32> {
```
```mlir
  %pv = stablehlo.constant dense<0.0> : tensor<f32>
```

```mlir
  %0 = sdy.all_gather [{"x"}, {}] %arg0 out_sharding=<@mesh_2_4, [{}, {}]> : tensor<8x16xf32>
```

```mlir
  %1 = stablehlo.pad %0, %pv, low = [1, 1], high = [1, 1], interior = [0, 0] : (tensor<8x16xf32>, tensor<f32>) -> tensor<10x18xf32>
```

```mlir
  return %1 : tensor<10x18xf32>
```

```mlir
}
```



**读法**：`all_gather` 后是全复制 → `pad` 参数**与全局一致**。

### ★ 在**分片维**上 pad：`uniform` vs `non-uniform`

用例名直接点出了关键区分：

| 用例 | 场景 |
|---|---|
| `pad_sharded_dim_uniform_on_partitions_1` | **均匀**（各分区相同）情形 1 |
| `pad_sharded_dim_uniform_on_partitions_2` | **均匀**情形 2 |
| `pad_sharded_dim_non_uniform_on_partitions` | **非均匀**（各分区不同） |

**为什么这个区分重要**：
在分片维上 pad 时，**只有边界的那台设备**需要补数据 ——
中间的设备完全不受影响。

- **均匀**：padding 量恰好让各分区"补齐" → 每台设备的本地 pad 参数**相同**。
- **非均匀**：padding 只影响首/尾设备 → 每台设备的本地 pad 参数**不同**，
  需要**按设备号判断**（用 `partition_id` 之类）。

### 负 padding：`pad` 也能**裁剪**

| 用例 | 场景 |
|---|---|
| `pad_negative_edges_non_sharded_dim` | 负 padding，填充维未分片 |
| `pad_negative_edges_sharded_dim` | 负 padding，填充维分片 |
| `pad_negative_edges_interior_sharded_dim` | 负 padding + **interior** + 分片维 |

**负 padding 意味着"裁剪"** —— `stablehlo.pad` 的 `low`/`high` 可以是负数，
表示"从这里切掉多少"。所以 `pad` 实际上同时承担了 pad 与 slice 的功能。

---

## 六、四个算子的对照

| 算子 | 作用维 | 作用维**未**分片 | 作用维**被**分片 |
|---|---|---|---|
| `iota` | `dim` | 直接转换 | **补偿偏移**（`partition_id` + 表） |
| `concatenate` | 拼接维 | 直接转换（各操作数各自局部化） | 需额外处理（本文件未覆盖） |
| `slice` | 切片维 | **参数不变**，只缩放分片维 | 起始/结束需按设备调整 |
| `pad` | 填充维 | 参数不变 | **均匀/非均匀**两种情形 |

**一句话总结**：
> **形状类算子的处理取决于"作用维是不是分片维"。**
> 不是 → 直接局部化；是 → 需要补偿位置（偏移 / 边界 / 按设备判断）。
>
> **共同点**：所有"位置相关"的算子（iota / slice / pad / concat）
> 在分片维上都会遇到"**我在全局中的位置**"这个问题 ——
> 而解法都是 `partition_id` + 查找表（与 L5-01 的常量、L5-02 的 all_slice 同一套路）。
