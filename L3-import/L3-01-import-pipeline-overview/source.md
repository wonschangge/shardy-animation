<!-- sdy-coverage
transforms/import/test/import_pipeline.mlir
-->

# L3-01 · import-pipeline-overview — 源 IR

覆盖：`shardy/dialect/sdy/transforms/import/test/import_pipeline.mlir`（165 行 / 8 个用例）。

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-import-pipeline='dedup-functions-fully=true' 2>&1 | FileCheck %s
```

**L3 层的主题**：传播**之前**，程序被做了哪些规范化。
这个文件用 8 个用例展示了导入流水线最典型的几类改动。

网格（各用例不同）：`sdy.mesh @mesh = <["a"=2]>`、`<["c"=2, "a"=2, "b"=2]>`、`<["a"=2, "b"=2]>`

---

## 一、pass **顺序**：先插边，再应用约束

```mlir
func.func @main(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
```

```mlir
  %0 = stablehlo.optimization_barrier %arg0 : tensor<32x96xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"a"}]> :  tensor<32x96xf32>
  return %1 : tensor<32x96xf32>
}
```

```mlir
  // CHECK-NEXT: %[[OPT_BARRIER:.*]] = stablehlo.optimization_barrier %arg0
  // CHECK-NEXT: sdy.data_flow_edge %[[OPT_BARRIER]] : tensor<32x96xf32>
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: sdy.sharding_constraint
```

**读法**：`optimization_barrier` 是区域算子 → 后面插了一条 `data_flow_edge`（L2-08）。
而 `sharding_constraint` 出现在**边之后**。

测试开头的注释点明了意图：

```
// Verifies that `-apply-sharding-constraints` pass is applied after
// `-add-data_flow_edges` pass
```

**为什么要这个顺序**：约束需要"贴"在正确的值上。如果先应用约束再插边，
约束可能会落到边的前面，导致传播时看不到它。

---

## 二、分片组 id **规范化**：合并 + 重编号

```mlir
func.func @main(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) {
```

```mlir
  sdy.sharding_group %arg0 group_id = 1234 : tensor<8x8xf32>
  sdy.sharding_group %arg0 group_id = 2345 : tensor<8x8xf32>
  sdy.sharding_group %arg1 group_id = 1234 : tensor<8x8xf32>
  sdy.sharding_group %arg1 group_id = 3456 : tensor<8x8xf32>
  func.return
}
```

```mlir
  // CHECK-DAG: sdy.sharding_group %arg0 group_id=0 : tensor<8x8xf32>
  // CHECK-DAG: sdy.sharding_group %arg1 group_id=0 : tensor<8x8xf32>
```

**读法**：
- 原本 4 条 `sharding_group`、3 个不同的 id（1234 / 2345 / 3456）。
- `%arg0` 与 `%arg1` 通过 **id 1234** 间接关联（传递闭包）→ 它们属于同一组。
- 输出只剩 **2 条**，id 被规范化为 **0**。

**这是 L2-07 提到的"传递闭包合并 + id 规范化"的具体形态**（原计划在 L3-09 展开）。

---

## 三、manual 轴的**清理**：三种情形

### 情形 A：manual 轴没切任何维度 → 移进 `replicated`

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"c", ?}]>] out_shardings=[<@mesh, [{"c", ?}]>] manual_axes={"c", "a"} (%arg1: tensor<4xf32>) {
    sdy.return %arg1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
```

```mlir
  // CHECK-NEXT: %0 = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>]
  // CHECK-SAME{LITERAL}:  out_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>]
  // CHECK-SAME{LITERAL}: manual_axes={"c", "a"} (%arg1: tensor<4xf32>) {
  // CHECK-NEXT:   %2 = sdy.data_flow_edge %arg1 sharding=<@mesh, [{?}]>
  // CHECK-NEXT:   sdy.return %2
  // CHECK-NEXT: }
  // CHECK-NEXT: %1 = sdy.data_flow_edge %0 sharding=<@mesh, [{"c", ?}], replicated={"a"}>
```

**读法**：`manual_axes={"c","a"}` 里只有 `"c"` 切了维度，`"a"` 没有
→ 流水线把 `"a"` 放进 `replicated`。同时**插入两条 data flow edge**
（块参数上一条、结果上一条）—— 这正是 L2-08/L2-09 讲的结构。

### 情形 B：`in_sharding` 全闭 → manual 轴也进 `replicated`，并写回函数参数

```mlir
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
```

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"c"}]>] out_shardings=[<@mesh, [{"c"}]>] manual_axes={"c", "a"} (%arg1: tensor<4xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<4xf32>
    sdy.return %1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
```

```mlir
// CHECK-SAME     %arg0: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}, {}], replicated={"a"}>}
// CHECK-SAME     -> tensor<16x16xf32> {
```

测试注释说明了原因：

```
// Due to the in_sharding being fully closed, the in_sharding is added to the
// func arg but with the manual axis added as replicated.
```

**读法**：`in_sharding` 是**全闭**的 `[{"c"}]`（没有 `?`）→ 流水线把它
**提升到函数参数**上，并把 manual 轴 `"a"` 作为 `replicated` 一并写上。

### 情形 C：manual 轴清理发生在**插边之前**

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{?}]>] out_shardings=[<@mesh, [{?}]>] manual_axes={"a"} (%arg1: tensor<8xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8xf32>
    sdy.return %1 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
```

```mlir
  // CHECK-NEXT: %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{?}], replicated={"a"}>] out_shardings=[<@mesh, [{?}], replicated={"a"}>] manual_axes={"a"} (%arg1: tensor<8xf32>) {
```

测试注释：

```
// This test verifies that the manual axes are cleaned up before adding data
// flow edges.
```

**读法**：`"a"` 没切任何维度 → 移进 `replicated`；且这一步**先于**插边完成。

> 三个情形合起来说明一条规则：**manual 轴必须在所有分片里显式出现
> （切维或进 `replicated`）** —— 与 L1-07 的不变量一致，导入流水线负责补齐。

---

## 四、函数调用：`dedup-functions-fully=true`

三个用例覆盖调用图的不同形态（注释里的 `test: non_flat` 是关键词）：

### 单次调用

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
func.func private @foo(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) -> tensor<8xf32> {
  return %arg0 : tensor<8xf32>
}
```

### 两次调用同一个函数

```mlir
  %0 = call @foo(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
  %1 = call @foo(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
```

### 链式调用：`main` → `foo` → `bar`

```mlir
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
```

```mlir
  %0 = call @foo(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
  %1 = call @bar(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}
```

```mlir
// CHECK-LABEL: func private @foo(
// CHECK-SAME:  %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) -> tensor<8xf32> {
func.func private @foo(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) -> tensor<8xf32> {
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(
  // CHECK-NEXT: return
  %0 = call @bar(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
```

**读法**：`dedup-functions-fully=true` 影响的是**函数去重/展平的激进程度**；
这三个用例锁定了「不同调用图形态下函数**保持原样**」的行为
（注意 CHECK 里 `@foo` / `@bar` 都还在，且参数上的分片标注被保留）。

> 函数调用的完整处理在 L3-06（`-sdy-import-func-calls`）展开。
