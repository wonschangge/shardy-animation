<!-- sdy-coverage
transforms/export/test/pad_for_divisibility/all_to_all.mlir
transforms/export/test/pad_for_divisibility/dot_general.mlir
transforms/export/test/pad_for_divisibility/func_ops.mlir
transforms/export/test/pad_for_divisibility/generic_ops.mlir
transforms/export/test/pad_for_divisibility/reduce_scatter.mlir
transforms/export/test/pad_for_divisibility/sdy_all_slice_all_gather.mlir
transforms/export/test/pad_for_divisibility/stablehlo_convolution.mlir
transforms/export/test/pad_for_divisibility/stablehlo_gather.mlir
transforms/export/test/pad_for_divisibility/stablehlo_pad.mlir
transforms/export/test/pad_for_divisibility/stablehlo_reshape.mlir
transforms/export/test/pad_for_divisibility/stablehlo_slice.mlir
transforms/export/test/pad_for_divisibility/stablehlo_while.mlir
-->

# L5-09 · pad-for-divisibility — 源 IR

**L5 收官课**：不可整除的分片如何在导出时被**补齐**。

| 文件 | 行数 | 用例数 |
|---|---|---|
| `pad_for_divisibility/stablehlo_convolution.mlir` | 292 | 8 |
| `pad_for_divisibility/stablehlo_while.mlir` | 201 | 4 |
| `pad_for_divisibility/func_ops.mlir` | 165 | 7 |
| `pad_for_divisibility/stablehlo_reshape.mlir` | 145 | 6 |
| `pad_for_divisibility/stablehlo_pad.mlir` | 142 | 8 |
| `pad_for_divisibility/dot_general.mlir` | 130 | 4 |
| `pad_for_divisibility/sdy_all_slice_all_gather.mlir` | 104 | 6 |
| `pad_for_divisibility/all_to_all.mlir` | 85 | 5 |
| `pad_for_divisibility/reduce_scatter.mlir` | 72 | 4 |
| `pad_for_divisibility/stablehlo_gather.mlir` | 68 | 2 |
| `pad_for_divisibility/stablehlo_slice.mlir` | 31 | 2 |
| `pad_for_divisibility/generic_ops.mlir` | 14 | 1 |

**合计 1449 行 / 57 用例。**

```mlir
// RUN: sdy_opt %s -sdy-pad-for-divisibility | FileCheck %s
```

网格：

```mlir
sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>
```

---

## 一、★ 核心问题与两种方向

**问题**：分片要求"维度大小能被轴整除"。
但现实中常有 `tensor<3>` 沿 `x=4` 切、`tensor<7>` 沿 `y=2` 切 —— **除不尽**。

**L2-01 讲过**："不可整除**不是错误**" —— 但导出时**必须解决**，
因为后端需要知道"每台设备拿多少"。

**两种方向**：

| 方向 | 情形 | 解法 |
|---|---|---|
| **输入不可整除** | 通信的**输入**大小除不尽 | **`pad` 补齐** → 通信 → `slice` 裁回 |
| **输出不可整除** | 通信的**输出**大小除不尽 | `slice` 到**可整除** → 通信 → `slice` 裁到目标 |

**统一模式**：
```
让通信发生在【可整除的形状】上，用 pad/slice 在两端做适配
```

---

## 二、★ 输出不可整除：`slice` 到可整除

```mlir
func.func @result_indivisible(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) -> tensor<3x8xf32> {
```

```mlir
  %0 = stablehlo.slice %arg0 [0:3, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<4x8xf32>) -> tensor<3x8xf32>
```

```mlir
  %1 = sdy.all_gather [{"x"}, {}] %0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<3x8xf32>
```

```mlir
  return %1 : tensor<3x8xf32>
```

```mlir
}
```

```mlir
// CHECK-LABEL: func @result_indivisible
```

```mlir
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %arg0 [0:4, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<4x8xf32>) -> tensor<4x8xf32>
```

```mlir
  // CHECK-NEXT: %[[AG:.*]] = sdy.all_gather [{"x"}, {}] %[[SLICE]] out_sharding=<@mesh_4_2, [{}, {}]> : tensor<4x8xf32>
```

```mlir
  // CHECK-NEXT: %[[TRIM:.*]] = stablehlo.slice %[[AG]] [0:3, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}]>]>} : (tensor<4x8xf32>) -> tensor<3x8xf32>
```

```mlir
  // CHECK-NEXT: return %[[TRIM]] : tensor<3x8xf32>
```

**读法**（三步）：
1. **`slice [0:4, 0:8]`** —— 输入是 `4x8`，而 `x=4` 要整除第 0 维。
   源 IR 里写的是 `[0:3, 0:8]`（想切 3），但 **3 不能被 4 整除**！
   → 导出时**改成 `[0:4, 0:8]`**（切**可整除**的 4）。
2. **`all_gather`** 在 `4x8` 上做 —— **可整除** ✓
3. **`slice [0:3, 0:8]`** —— 通信后**裁到目标** `3x8`。

**关键**：`all_gather` 的参数从 `tensor<3x8xf32>` 变成 **`tensor<4x8xf32>`** ——
**通信在补齐后的形状上发生**。

---

## 三、★ 输入不可整除：`pad` 补齐

```mlir
func.func @input_indivisible(%arg0: tensor<4x7xi32> )
  -> (tensor<4x6xi32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"y"}]>}) {
```

```mlir
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xi32>
```

```mlir
  %1 = stablehlo.slice %0 [0:4, 0:6] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>}
```

```mlir
    : (tensor<4x7xi32>) -> tensor<4x6xi32>
```

```mlir
  return %1 : tensor<4x6xi32>
```

```mlir
}
```

```mlir
// CHECK-LABEL: func @input_indivisible
```

```mlir
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
```

```mlir
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xi32>, tensor<i32>) -> tensor<4x8xi32>
```

```mlir
  // CHECK-NEXT: %[[ALL_SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xi32>
```

```mlir
  // CHECK-NEXT: %[[RESULT:.*]] = stablehlo.slice %[[ALL_SLICE]] [0:4, 0:6] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<4x8xi32>) -> tensor<4x6xi32>
```

```mlir
  // CHECK-NEXT: return %[[RESULT]] : tensor<4x6xi32>
```

**读法**（三步）：
1. **`pad high = [0, 1]`** —— `4x7` 的第 1 维 `7` **不能被 `y=2` 整除**
   → 补 **1** 个元素变成 **`4x8`**（可被 2 整除）✓
   - `low = [0, 0]`、`high = [0, 1]` —— **只在尾部补**（high）
2. **`all_slice`** 在 `4x8` 上做 —— **可整除** ✓
3. **`slice [0:4, 0:6]`** —— 裁到目标 `4x6`（`6 = 7 - 1`，源 IR 本来就要切到 6）

**`pad` 的参数**：
- `low` = 前面补多少
- `high` = 后面补多少
- `interior` = 元素之间补多少（通常 0）

**补齐的方向**：本用例只在 `high` 补 —— 因为只需要"凑够"可整除的数量。

---

## 四、`padding` 值的选择

**统计 12 个文件里的 padding 常量**：

| 值 | 出现次数 |
|---|---|
| `dense<0.000000e+00>`（浮点零） | 56 |
| `dense<0>`（整数零） | 14 |

**观察**：**padding 值都是零**（浮点 0 或整数 0）。

**为什么可以填 0**：
- 通信本身（`all_gather` / `all_slice` / `all_to_all`）是**数据搬运**，
  不涉及算术 → 填什么值都不会影响"搬运"的正确性。
- 搬运后的 **`slice` 会把补的部分裁掉** → 补的值**根本不会出现在最终结果里**。

**所以填 0 是最简单的选择** —— 反正要被裁掉。

**但如果通信后紧跟归约**（如 `reduce_scatter`）：
填的值就**必须不影响归约结果** ——
- 加法 / 最大值：填 **0** 可能有问题（如果数据全为负）
- 乘法：填 **0** 会把结果变成 0 → 必须填 **1**

**本课的 12 个文件覆盖的正是这些情形**（`reduce_scatter` / `dot_general` / `convolution`）。

---

## 五、12 个文件的族谱

| 族 | 文件 | 用例数 | 说明 |
|---|---|---|---|
| **集合通信** | `all_to_all`、`reduce_scatter`、`sdy_all_slice_all_gather` | 15 | 通信维的 padding |
| **计算类** | `dot_general`、`stablehlo_convolution` | 12 | 收缩维 / 窗口维的 padding |
| **形状类** | `stablehlo_reshape`、`stablehlo_slice`、`stablehlo_pad` | 16 | 形状变换的 padding |
| **访存类** | `stablehlo_gather` | 2 | 索引维的 padding |
| **结构类** | `func_ops`、`stablehlo_while`、`generic_ops` | 12 | 函数边界 / 循环 / 通用 |

**为什么 `convolution` 最大**（292 行 / 8 用例）：
它的 `permutation` 因子（空间维不成整数倍，L5-06 讲过）
让不可整除的情形**最多**。

**为什么 `stablehlo_while` 有 4 个用例**：
循环体的分片要在**每次迭代**都保持可整除 ——
需要把 padding 也**带进循环**（或每轮重新补）。

**`generic_ops` 的 `no_pad`**：
最简单的用例 —— 可整除时**什么都不做**。
它验证了 pass 的**幂等性/最小性**：能整除就不该有 pad。

---

## 六、与前面几课的呼应

| 课 | 联系 |
|---|---|
| **L2-01** | "不可整除不是错误" —— 本课是它的**解决方案** |
| **L2-02** | 子轴 —— 不可整除时的**另一种**处理（用子轴截断） |
| **L4-02** | `update_non_divisible_input_output_shardings` —— 导出侧的另一处理 |
| **L5-04** | `pad` 的语义（本课大量使用） |
| **L5-06** | `permutation` 因子 —— 卷积不可整除的来源 |

**注意与 L4-02 的区别**：
- **L4-02**（`update_non_divisible_input_output_shardings`）：
  把不可整除的**输入输出分片**用**子轴截断**（`{"x":(1)2}`）—— 改变**分片**。
- **L5-09**（本课）：在**算子层面**插入 `pad`/`slice` —— 改变**张量形状**。

两者是**互补**的：L4-02 处理"分片怎么表达"，L5-09 处理"数据怎么补齐"。

**★ 验收点答案**：
> 给定 `tensor<7x3x8>` 沿 `z=3` 分片：
> - 若沿**第 2 维**（大小 8）分片：`8` 不能被 `3` 整除
>   → 补到下一个 3 的倍数 = **9**
>   → **`pad low = [0, 0, 0], high = [0, 0, 1]`**，补齐后形状 **`7x3x9`**
> - 通信后 `slice` 裁回 `7x3x8`

**一句话总结**：
> **不可整除时，让通信发生在【可整除的形状】上** ——
> 输入端 `pad` 补齐、输出端 `slice` 到可整除，最后统一 `slice` 裁回目标。
>
> 这是 L2-01 "不可整除不是错误"的**最终答案**。
