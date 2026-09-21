<!-- sdy-coverage
transforms/export/test/constant_or_scalar_merger.mlir
transforms/export/test/remove_propagation_debug_info.mlir
transforms/export/test/propagate_to_func_results.mlir
-->

# L4-17 · merger-and-debug-cleanup — 源 IR

**L4 的最后一课**：三个"最后的清理"pass。

| pass | 文件 | 行数 | 用例数 |
|---|---|---|---|
| `--sdy-constant-or-scalar-merger` | `constant_or_scalar_merger.mlir` | 182 | 13 |
| `-sdy-remove-propagation-debug-info` | `remove_propagation_debug_info.mlir` | 112 | 2 |
| `-sdy-propagate-to-func-results` | `propagate_to_func_results.mlir` | 189 | 20 |

网格：

```mlir
sdy.mesh @mesh = <["x"=2, "y"=2]>
```

---

## 一、★ `constant_or_scalar_merger`：合并重复的常量

### `sdy.constant` 的合并

```mlir
// RUN: sdy_opt %s --sdy-constant-or-scalar-merger | FileCheck %s
```

```mlir
func.func @merge_constants_sdy() -> tensor<f32> {
```

```mlir
  %0 = sdy.constant dense<1.000000e+00> : tensor<f32>
  %1 = sdy.constant dense<1.000000e+00> : tensor<f32>
```

```mlir
  // CHECK: stablehlo.add %[[C0]], %[[C0]] : tensor<f32>
  %2 = stablehlo.add %0, %1 : tensor<f32>
  return %2 : tensor<f32>
}
```

```mlir
  // CHECK: %[[C0:.*]] = sdy.constant dense<1.0{{.*}}> : tensor<f32>
  // CHECK-NOT: sdy.constant
```

**读法**：
- 输入：**两个** `sdy.constant dense<1.0>`
- 输出：**只剩一个**（`CHECK-NOT: sdy.constant` 表示后面再没有第二个）
- `stablehlo.add %0, %1` → **`stablehlo.add %[[C0]], %[[C0]]`** —— 两个操作数都指向同一个常量。

### 也处理 `stablehlo.constant`

```mlir
func.func @merge_constants_constant_like() -> tensor<f32> {
```

```mlir
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
```

```mlir
  // CHECK: stablehlo.add %[[C0]], %[[C0]] : tensor<f32>
```

```mlir
  // CHECK: %[[C0:.*]] = stablehlo.constant dense<1.0{{.*}}> : tensor<f32>
  // CHECK-NOT: stablehlo.constant
```

**读法**：`stablehlo.constant` 同样被合并 —— 所以 pass 名叫 "constant **or scalar** merger"。

### ★ 与 L3-02 的**镜像关系**

| 课 | pass | 方向 | 为什么 |
|---|---|---|---|
| **L3-02** | 常量与标量**拆分** | N 个使用 → N 份 | **避免传播期产生假依赖** |
| **L4-17** | 常量与标量**合并** | 相同内容 → 一份 | 传播已结束，假依赖不再是问题 |

**回顾 L3-02**：那里讲过"**为什么必须拆**" ——
如果多个使用共享一个常量，传播器会以为它们**必须同分片**（假依赖），
从而限制了传播的自由度。

**为什么现在可以合并**：
- **传播已经结束** —— 分片已经定下来，不再需要"传播自由度"。
- 合并后 IR 更小、常量更少 → 后端编译更快。

**这是"同一件事在两个阶段做相反操作"的典型例子** ——
与 L3-06/L4-12（内联 vs outline）、L3-07/L4-15（提升 vs 内联）同类。

---

## 二、`propagate_to_func_results`：补全函数结果的分片

### 基础情形

```mlir
// RUN: sdy_opt %s -allow-unregistered-dialect -sdy-propagate-to-func-results -split-input-file | FileCheck %s
```

```mlir
// test: simple
sdy.mesh @mesh = <["x"=2]>
```

```mlir
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
```

```mlir
  %0 = call @foo(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
```

```mlir
// CHECK-LABEL: func private @foo
// CHECK-SAME:  -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
func.func private @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
```

```mlir
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} :  tensor<8xf32>
  return %0 : tensor<8xf32>
}
```

**读法**：
- `@foo` 内部：`%0 = stablehlo.abs %arg0 {sdy.sharding = ...<@mesh, [{"x"}]>}` ——
  **算子结果有分片**。
- `@foo` 的**函数结果签名**原本**没有**分片（`-> tensor<8xf32>`）。
- 输出：函数结果**被标上** `[{"x"}]` —— 从 `%0` 的分片**传播**过来。

**为什么需要**：函数结果的分片必须**反映实际返回值的分片** ——
否则调用者看到的签名与真实情况不符。

### 与 L4-16 的关系

| 课 | pass | 做什么 |
|---|---|---|
| **L4-16** | `sink-func-data-flow-edges` | 把**边**的分片**下沉**到值上 |
| **L4-17** | `propagate-to-func-results` | 把**值**的分片**传播**到函数结果签名 |

**两步配合**：
1. L4-16 先把边下沉 → 分片落到**算子结果**上。
2. 本课再把算子结果的分片传播到**函数结果签名**上。

**合起来**：函数边界上的分片信息**完整**了 ——
既在返回值上，也在签名上。

### 20 个用例覆盖的情形

用例注释直接标出了场景（从文件里可见的）：

| 注释 | 场景 |
|---|---|
| `// test: simple` | 基础情形 |
| `// test: terminator value has no sharding, func result does not have either.` | 终止符值**无分片**，函数结果也**无** |

**其余用例**覆盖：终止符有分片 / 函数结果已有分片 / 多结果 / 不同类型等组合。

**关键**：这是一个**保守**的 pass ——
只在"函数结果没有分片、但终止符值有"时才补上；
如果两边都有（可能不一致），行为需要看具体用例。

---

## 三、`remove_propagation_debug_info`：移除调试信息

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-remove-propagation-debug-info | FileCheck %s
```

```mlir
// CHECK-NOT: sdy.propagation_edges
```

**读法**：`sdy.propagation_edges` 被**全部移除**。

**这是什么**：传播期会记录"**分片是怎么传播过来的**"（传播路径 / 边的记录）——
用于**调试**（配合相应的调试/转储选项查看）。

**为什么要删**：
- 它**不是语义信息** —— 只描述"传播过程"，不描述"结果是什么"。
- 保留它会让 IR 变大、干扰后续处理。
- **与 L2-03 的 `keep-sharding-rules`、L4-05 的 `mark-partial-result`、L4-08 的 `keep-redundant-reshards` 同类** ——
  都是**观测性/调试性**的信息，最终都要清掉。

**注意这个文件只有 2 个用例** —— 但文件有 112 行，
说明**一个用例里有很多 IR**（4 个不同 mesh 的声明 + 大量传播信息）。

---

## 四、★ 三个 pass 的**共同点**

| pass | 做什么 | 类型 |
|---|---|---|
| `constant_or_scalar_merger` | 合并重复常量 | **优化**（IR 变小） |
| `propagate_to_func_results` | 补全函数结果分片 | **补全**（信息完整） |
| `remove_propagation_debug_info` | 删调试信息 | **清理**（去噪声） |

**它们都是"最后的收尾"** —— 在 L4-15（收尾清理）与 L4-16（边下沉）之后，
还有这三件事要做：

1. **优化**：把 L3-02 拆开的常量**合并回去**。
2. **补全**：把算子结果的分片**传播到函数签名**。
3. **清理**：把调试信息**删掉**。

**一句话总结**：
> **导出流水线的最后一课** ——
> 合并可以合并的、补全应该补全的、删掉不该留的。

---

## 五、L4 层收官：17 课的回顾

| 课 | 主题 | 覆盖文件 |
|---|---|---|
| L4-01 | 导出流水线总览 | 2 |
| L4-02 | reshard 插入总纲（★） | 2 |
| L4-03 | 逐元素与形状类 | 9 |
| L4-04 | 矩阵与卷积类 | 4 |
| L4-05 | 归约与排序类 | 4 |
| L4-06 | 访存与通信类 | 3 |
| L4-07 | 结构性场景 | 9 |
| L4-08 | reshard 转集合通信（★ 核心） | 2 |
| L4-09 | 置换因子消解 | 2 |
| L4-10 | 通信优化 | 2 |
| L4-11 | 逐指令分区 | 3 |
| L4-12 | 导出命名计算 | 1 |
| L4-13 | 调用图还原 | 2 |
| L4-14 | 单设备与未归约 | 3 |
| L4-15 | 导出收尾（P0） | 7 |
| L4-16 | 边下沉与转换 | 3 |
| L4-17 | 合并与调试清理（本课） | 3 |

**L4 的一条主线**：
```
传播后的 IR
  → 插入 reshard（L4-02～07，按算子族）
  → 转成集合通信（L4-08～10）
  → 处理特殊结构（L4-11～14）
  → 收尾清理（L4-15～17）
= 交给后端的 IR
```
