<!-- sdy-coverage
transforms/export/test/convert_global_to_local/stablehlo_convolution.mlir
-->

# L5-06 · lowering-convolution — 源 IR

**卷积分片降级** —— 规则与 L5-05 的矩阵乘**完全一致**。

| 文件 | 行数 | 用例数 |
|---|---|---|
| `convert_global_to_local/stablehlo_convolution.mlir` | 191 | 6 |

网格：

```mlir
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
```

---

## 一、★ 六个用例按**分片的维度**组织

| 用例 | 分片的位置 | 通信 |
|---|---|---|
| `shard_batch` | **批维** | **无** |
| `shard_batch_group` | batch_group 维 | **无** |
| `shard_feature` | **特征维** | **无** |
| `shard_feature_group` | feature_group 维 | **无** |
| `shard__reduction_factors` | **归约因子** | **`all_reduce`** |
| `shard_reduction_factors_unreduced_result` | 归约因子（结果未归约） | **无**（延迟） |

**★ 核心规律**（与 L5-05 一致）：
> **只有【归约因子】上的分片需要 `all_reduce`。**
> 批维 / 特征维上的分片都是"各算各的输出元素"，无需通信。

---

## 二、卷积的 `sharding_rule`

测试文件开头的注释给出了规则：

```
// ([i, jk, mn, o], [k, n, o, p])->([i, j, m, p])
// {i=2, j=112, k=2, m=112, n=2, o=3, p=64}
// reduction={k, n, o} permutation={j, m}>
```

**逐项读**：
- **输入** `[i, jk, mn, o]` ——
  `i` 是批维（含 batch_group）、`jk` 是空间维（复合因子）、`mn` 是空间维、`o` 是输入通道。
- **权重** `[k, n, o, p]` ——
  `k`/`n` 是窗口、`o` 是输入通道、`p` 是输出通道。
- **输出** `[i, j, m, p]` —— 批维、两个空间维、输出通道。
- **`reduction={k, n, o}`** —— **窗口 + 输入通道**是归约因子！
- **`permutation={j, m}`** —— 空间维因步长而"不成整数倍"（L2-10 讲过）。

**这就解释了为什么"归约因子分片需要通信"**：
`k`/`n`（窗口）与 `o`（输入通道）都是**被累加**的方向 ——
沿它们分片，每台设备只累加了**一部分**。

---

## 三、`shard_batch`：批维分片 → **无通信**

```mlir
func.func @shard_batch(
  %arg0: tensor<2x224x224x3xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}, {}]>},
  %arg1: tensor<3x3x3x64xf32>)
  -> (tensor<2x112x112x64xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}, {}]>}) {
```

```mlir
  %0 = stablehlo.convolution(%arg0, %arg1)
```

```mlir
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
```

```mlir
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]}
```

```mlir
    {
```

```mlir
      feature_group_count = 1 : i64,
```

```mlir
      batch_group_count = 1 : i64,
```

```mlir
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}, {}, {}]>]>
```

```mlir
    } : (tensor<2x224x224x3xf32>, tensor<3x3x3x64xf32>) -> tensor<2x112x112x64xf32>
```

```mlir
  return %0 : tensor<2x112x112x64xf32>
```

```mlir
}
```

```mlir
// CHECK-LABEL: func @shard_batch
// CHECK-SAME: (%arg0: tensor<1x224x224x3xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}, {}]>},
// CHECK-SAME: %arg1: tensor<3x3x3x64xf32>)
// CHECK-SAME: -> (tensor<1x112x112x64xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}, {}]>})
```

```mlir
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
```

```mlir
  // CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
```

```mlir
  // CHECK-SAME{LITERAL}: window = {stride = [2, 2], pad = [[0, 1], [0, 1]]}
```

```mlir
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
```

```mlir
  // CHECK-SAME: : (tensor<1x224x224x3xf32>, tensor<3x3x3x64xf32>) -> tensor<1x112x112x64xf32>
```

```mlir
  // CHECK: return %[[CONV]] : tensor<1x112x112x64xf32>
```

**读法**：
- **批维（第 0 维）切 `x=2`** → `2/2 = 1`（局部批大小为 1）。
- **权重无分片**（`%arg1` 没有 sharding）。
- **`dim_numbers` / `window` / group_count 全部不变** —— 只改类型！
- **无通信** ✓

**为什么无通信**：批维是**输出维** —— 每个样本独立卷积，互不干扰。

---

## 四、`shard_feature`：特征维分片 → **无通信**

```mlir
func.func @shard_feature(
  %arg0: tensor<2x224x224x4xf32>,
  %arg1: tensor<3x3x4x64xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"x"}]>})
  -> (tensor<2x112x112x64xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"x"}]>}) {
```

```mlir
// CHECK-LABEL: func @shard_feature
// CHECK-SAME: (%arg0: tensor<2x224x224x4xf32>,
// CHECK-SAME:  %arg1: tensor<3x3x4x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"x"}]>})
// CHECK-SAME: -> (tensor<2x112x112x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"x"}]>})
```

**读法**：
- **输出通道（第 3 维，`p`）切 `x=2`** → `64/2 = 32`。
- `%arg0`（输入）**无分片**。
- 结果也在第 3 维切 `{"x"}`。
- **无通信** ✓

**为什么无通信**：输出通道是**输出维** ——
每台设备算**不同的输出通道**，互不重叠。

**注意**：这里 `%arg1` 的 `p` 维切了，但 `o` 维（输入通道，归约因子）**没切** ——
所以不需要通信。

---

## 五、★ `shard__reduction_factors`：归约因子分片 → **`all_reduce`**

```mlir
func.func @shard__reduction_factors(
  %arg0: tensor<2x224x224x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"y":(2)2}]>},
  %arg1: tensor<2x2x4x64xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y":(1)2}, {"y":(2)2}, {}]>})
    -> (tensor<1x112x112x64xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {}]>}) {
```

```mlir
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 0], [0, 0]]}
    {
      feature_group_count = 1 : i64,
      batch_group_count = 2 : i64,
      sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_4, [{}, {}, {}, {}], unreduced={"x", "y"}>]>
    } : (tensor<2x224x224x4xf32>, tensor<2x2x4x64xf32>) -> tensor<1x112x112x64xf32>
```

```mlir
// CHECK-LABEL: func @shard__reduction_factors
// CHECK-SAME: (%arg0: tensor<2x224x224x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"y":(2)2}]>},
// CHECK-SAME:  %arg1: tensor<1x1x2x64xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y":(1)2}, {"y":(2)2}, {}]>})
// CHECK-SAME: -> (tensor<1x112x112x64xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {}]>})
```

```mlir
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
  // CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME{LITERAL}: window = {stride = [2, 2], pad = [[0, 0], [0, 0]]}
  // CHECK-SAME: {batch_group_count = 2 : i64, feature_group_count = 1 : i64}
  // CHECK-SAME: : (tensor<2x224x224x2xf32>, tensor<1x1x2x64xf32>) -> tensor<1x112x112x64xf32>
```

**逐项读**（本课最关键的一处）：

| 张量 | 全局 | 分片 | 局部 | 说明 |
|---|---|---|---|---|
| `%arg0` | `2x224x224x4` | `[{}, {}, {}, {"y":(2)2}]` | **`2x224x224x2`** | 第 3 维（输入通道 `o`）切 |
| `%arg1` | `2x2x4x64` | `[{"x"}, {"y":(1)2}, {"y":(2)2}, {}]` | **`1x1x2x64`** | 第 0 维（`k`）切、第 1 维（`n`）切、第 2 维（`o`）切 |
| 结果 | `1x112x112x64` | `[{}, {}, {}, {}]` | **`1x112x112x64`** | 全复制 |
| **`conv` 的 sharding** | — | **`unreduced={"x", "y"}`** | — | **部分结果**！ |

**读法**：
- **归约因子被切了**：
  - `%arg0` 的**输入通道 `o`**（第 3 维）切 `{"y":(2)2}`
  - `%arg1` 的**窗口 `k`/`n`**（第 0/1 维）切 `{"y":(1)2}`、**输入通道 `o`**（第 2 维）切 `{"y":(2)2}`
  - 而 `%arg1` 的**批维**（第 0 维）切 `{"x"}`
- **`batch_group_count = 2`** —— 因为批维被切了（从 1 变成 2）。
- **结果标 `unreduced={"x", "y"}`** —— 表示"在 `x` 和 `y` 上都还没归约"。
- → 后续需要 **`all_reduce {"x", "y"}`**（测试的 CHECK 会显示）。

**为什么 `x` 也在 unreduced 里**：
`%arg1` 的批维切了 `x`，而 `batch_group_count = 2` 意味着
**批维参与了分组卷积** —— 分组的结果也需要归约。

**子轴的用法**：`{"y":(1)2}` / `{"y":(2)2}` 是**子轴**（L2-02）——
因为 `y=4` 要同时切窗口维和输入通道维，必须用子轴分配。

**这就是本课的核心**：
> **归约因子（窗口 / 输入通道）上的分片 → 产生部分结果 → 需要 `all_reduce`。**

---

## 六、`..._unreduced_result`：延迟归约

第六个用例 `shard_reduction_factors_unreduced_result` 与第五个**同构**，
唯一区别是**函数结果也声明了 `unreduced`** → **不插 `all_reduce`**。

**这与 L5-05 的情形 ④ 完全一致**：

| | 结果要求 | `all_reduce` |
|---|---|---|
| `shard__reduction_factors` | 完整 | **需要** |
| `..._unreduced_result` | 未归约 | **不需要**（延迟） |

---

## 七、与 L5-05（矩阵乘）的对照

| | 矩阵乘（`dot`） | 卷积（`convolution`） |
|---|---|---|
| **输出维** | 非收缩维 | 批维 / 特征维 |
| **归约维** | 收缩维 | **窗口 + 输入通道** |
| **无通信的情形** | 输出维分片 | 批维 / 特征维分片 |
| **需通信的情形** | 收缩维分片 | **归约因子分片** |
| **延迟归约** | 结果标 `unreduced` | 结果标 `unreduced` |

**★ 共同的规律**：
> **只有【归约方向】上的分片需要 `all_reduce`。**
> 输出方向上的分片都是"各算各的输出元素"，互不重叠。

**卷积比矩阵乘多的地方**：
1. **归约因子有三个**（`k`/`n` 窗口 + `o` 输入通道），而矩阵乘只有一个收缩维。
2. **`permutation` 因子**（空间维因步长不成整数倍）—— 这带来 halo 问题（L4-09 讲过）。
3. **`batch_group_count` / `feature_group_count`** 会改变批维/特征维的**大小**，
   所以分片要用**子轴**精细分配。

**一句话总结**：
> **卷积分片降级与矩阵乘同构** ——
> 输出维分片无需通信、归约因子分片需要 `all_reduce`。
> 卷积的额外复杂度来自**多个归约因子**与 **`permutation` 因子**。
