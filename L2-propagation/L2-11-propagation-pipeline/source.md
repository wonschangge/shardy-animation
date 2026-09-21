<!-- sdy-coverage
transforms/propagation/test/propagation_pipeline.mlir
-->

# L2-11 · propagation-pipeline — 源 IR

覆盖：`shardy/dialect/sdy/transforms/propagation/test/propagation_pipeline.mlir`
（**1200 行 / 58 个用例**）。

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-propagation-pipeline | FileCheck %s
```

**整个文件只有一条 RUN 行** —— 它跑的是**完整流水线**，而不是某一层策略。
所以这一课看的是「**前面各课的效果叠加起来长什么样**」。

网格：`sdy.mesh @mesh = <["a"=2, "b"=2]>`

---

## 一、★ 常量被**拆分**：一个变三个

一个常量被两处使用，且两处需要不同分片：

```mlir
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
```

```mlir
  %1 = stablehlo.dot_general %0, %0, contracting_dims = [1] x [1] : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x8xf32>
```

```mlir
  %2 = stablehlo.add %1, %arg0 : tensor<8x8xf32>
```

```mlir
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
```

```mlir
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[CONST_0]], %[[CONST_1]], contracting_dims = [1] x [1]
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>}
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[DOT_GENERAL]], %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>}
  // CHECK-NEXT: return %[[CONST_2]], %[[ADD]]
```

**读法**（本课最漂亮的一个例子）：
- 原来的**一个** `stablehlo.constant` 变成了**三个** `sdy.constant`：
  - `%[[CONST_0]]` 沿 `"a"` 切 —— 给 `dot_general` 的左操作数
  - `%[[CONST_1]]` 沿 `"b"` 切 —— 给右操作数
  - `%[[CONST_2]]` **完全不带分片** —— 给 `return`（复制）
- 这正是 L1-10 讲的：`sdy.constant` **刻意不实现 ConstantLike**，
  所以能被复制成多份、各自分片不同。
- 对应 pass：导入期的 `-sdy-constant-or-scalar-splitter`（L3-02）。

---

## 二、分片约束 → `sdy.reshard`

```mlir
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
```

```mlir
  return %arg0, %0 : tensor<8x8xf32>, tensor<8x8xf32>
```

```mlir
  // CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
```

```mlir
  // CHECK-NEXT: return %arg0, %0
```

**读法**：`sharding_constraint` 被**消费掉**，换成了 `sdy.reshard` ——
这正是 L1-10 讲的「约束 → 传播 → reshard → collective」链条中的**第二步**。

> 注意 `%arg0` 本身**没有**被改（仍以原样返回）——
> 约束只作用在它**那个使用者**上（L1-09 讲的"有使用者"语义）。

---

## 三、未归约轴 → `all_reduce` → `reshard`

```mlir
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {?}], unreduced={"b"}>]>} :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
```

```mlir
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}], unreduced={"b"}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"b"} %[[DOT_GENERAL]] out_sharding=<@mesh, [{"a"}, {}]>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{"a"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[RESHARD]], %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>}
```

**读法**（完整的三步链条）：
1. `dot_general` 的分片带 `unreduced={"b"}` —— 沿 `"b"` 切会产生**部分和**。
2. 流水线插入 `sdy.all_reduce {"b"}` —— 把部分和**归约成完整值**。
3. 再插一条 `sdy.reshard` —— 把分片调整到下游 `add` 需要的样子。

> 这是本课最能体现「流水线在做什么」的例子：
> **用户只写了一个 `unreduced` 标注，流水线自动补出两条通信算子。**

---

## 四、内联 mesh 被规范化

```mlir
func.func @inlined_mesh(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["a"=2, "b"=2]>, [{"a"}, {"b"}]>},
    %arg1: tensor<8x8xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
```

```mlir
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>}
```

**读法**：输入里写的是**内联的 mesh 定义** `mesh<["a"=2, "b"=2]>`，
输出里已经变成**命名引用** `@mesh`。流水线会把内联网格**提升为顶层声明**。

---

## 五、与前面各课的复用

这个文件里的 `case_*`（13 个）、`while_*`（8 个）、`optimization_barrier`、
`manual_computation_*` 等用例，**与 L2-08 / L2-09 的用例同名同形** ——
区别只在于：

- L2-08 / L2-09 跑的是 `-sdy-basic-propagate`（单层策略）；
- 本课跑的是 `-sdy-propagation-pipeline`（完整流水线）。

所以可以用**同一批用例做对照**：看流水线比单层策略多做了什么。

---

## 六、其余值得一看的用例

| 用例 | 看点 |
|---|---|
| `user_priorities` | 用户优先级在完整流水线下的效果（L2-06） |
| `sharding_group_on_while_result` | 组作用在循环结果上（L2-07） |
| `add_extra_sharding_constraint_for_incompatible_group_member_shardings` | 组内不兼容时插入约束（L2-07） |
| `do_not_propagate_manual_axes_to_manual_computation` | manual 轴不外传（L2-09） |
| `maximal_sharding_no_results` / `replicated_sharding_no_results` | 边界情形 |
| `size_zero_dim_sharded` | 大小为 0 的维度 |
| `dot_lhs_from_broadcast_and_large_rhs` | 真实模型的形态 |
| `manual_computation_with_tokens` | token 处理 |
| `main` × 8 | 综合场景 |
