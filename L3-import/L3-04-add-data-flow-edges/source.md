<!-- sdy-coverage
transforms/import/test/add_data_flow_edges.mlir
-->

# L3-04 · add-data-flow-edges — 源 IR

覆盖：`shardy/dialect/sdy/transforms/import/test/add_data_flow_edges.mlir`（253 行 / 16 个用例）。

```mlir
// RUN: sdy_opt %s -sdy-add-data-flow-edges -split-input-file | FileCheck %s
```

**与 L2-08 的关系**：L2-08 讲的是数据流边在**传播中**的行为；
这一课讲这些边**是怎么被插进来的**。

---

## 一、★ 核心规则：**每个结果一条边**

```mlir
func.func @case(%arg0: tensor<i32>, %arg1: tensor<8xi64>, %arg2: tensor<8xi64>,
                %arg3: tensor<8xi64>, %arg4: tensor<8xi64>)
    -> (tensor<8xi64>, tensor<8xi64>) {
```

```mlir
  %0:2 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1, %arg2 : tensor<8xi64>, tensor<8xi64>
  }, {
    stablehlo.return %arg3, %arg4 : tensor<8xi64>, tensor<8xi64>
  }) : (tensor<i32>) -> (tensor<8xi64>, tensor<8xi64>)
  %1 = stablehlo.add %0#1, %0#1 : tensor<8xi64>
  return %0#0, %1 : tensor<8xi64>, tensor<8xi64>
}
```

```mlir
  // CHECK-NEXT: %[[CASE:.*]]:2 = "stablehlo.case"(%arg0)
  // CHECK:      %[[EDGE_1:.*]] = sdy.data_flow_edge %[[CASE]]#0 : tensor<8xi64>
  // CHECK-NEXT: %[[EDGE_2:.*]] = sdy.data_flow_edge %[[CASE]]#1 : tensor<8xi64>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[EDGE_2]], %[[EDGE_2]]
  // CHECK-NEXT: return %[[EDGE_1]], %[[ADD]]
```

**读法**：
- `case` 有 **2 个结果** → 插入 **2 条边**（`%[[EDGE_1]]`、`%[[EDGE_2]]`）。
- 原来用 `%0#0` / `%0#1` 的地方，**全部改用边**。
- **边成为唯一的消费者**：区域算子的结果只被边使用。

**为什么**：区域算子本身不能携带分片（L2-08），分片必须写在边上。
让边独占结果，保证每个结果恰好对应一条边 —— 结构清晰、不会歧义。

---

## 二、结果**没被使用**也要插边

```mlir
func.func @while_unused_result(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
```

```mlir
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare  LT, %iterArg_2, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %4 = stablehlo.add %iterArg_2, %1 : tensor<i32>
    %5 = stablehlo.add %iterArg, %iterArg : tensor<32x96xf32>
    stablehlo.return %5, %4 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}
```

```mlir
  // CHECK:      %[[WHILE:.*]]:2 = stablehlo.while
  // CHECK:      %[[EDGE_1:.*]] = sdy.data_flow_edge %[[WHILE]]#0
  // CHECK-NEXT: %[[EDGE_2:.*]] = sdy.data_flow_edge %[[WHILE]]#1
  // CHECK-NEXT: return %[[EDGE_1]]
```

**读法**：`while` 有 2 个结果，但只有 `#0` 被 `return` 使用（`#1` 是循环计数器，没人用）。
→ **仍然插 2 条边**（`%[[EDGE_2]]` 是"死"的，但仍被创建）。

**结论**：判据是「结果的**个数**」，不是「有几个被使用」。
保持"每个结果一条边"的不变量，避免结构因使用情况而变形。

---

## 三、区域内也要插边：**块参数**

```mlir
func.func @named_computation_multiple_inputs_outputs(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
```

```mlir
  %0:2 = sdy.named_computation<"my_func">(%arg0, %arg1) (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>)  {
    sdy.return %arg2, %arg3 : tensor<8x2xi32>, tensor<4x2xi32>
  } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  return %0#0, %0#1 : tensor<8x2xi32>, tensor<4x2xi32>
}
```

```mlir
  // CHECK-NEXT: %[[NC:.*]]:2 = sdy.named_computation<"my_func">(%arg0, %arg1) (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>)  {
  // CHECK-NEXT:   %[[EDGE_1:.*]] = sdy.data_flow_edge %arg2 : tensor<8x2xi32>
  // CHECK-NEXT:   %[[EDGE_2:.*]] = sdy.data_flow_edge %arg3 : tensor<4x2xi32>
  // CHECK-NEXT:   return %[[EDGE_1]], %[[EDGE_2]] : tensor<8x2xi32>, tensor<4x2xi32>
  // CHECK-NEXT: } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  // CHECK-NEXT: %[[EDGE_3:.*]] = sdy.data_flow_edge %[[NC]]#0 : tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE_4:.*]] = sdy.data_flow_edge %[[NC]]#1 : tensor<4x2xi32>
  // CHECK-NEXT: return %[[EDGE_3]], %[[EDGE_4]] : tensor<8x2xi32>, tensor<4x2xi32>
```

**读法**（两处都插）：
- **区域内**：每个**块参数**一条边（`%[[EDGE_1]]` 对应 `%arg2`，`%[[EDGE_2]]` 对应 `%arg3`），
  且 `sdy.return` 改为返回边。
- **区域外**：每个**结果**一条边（`%[[EDGE_3]]`、`%[[EDGE_4]]`）。

**为什么两边都要**：块参数是"区域内的入口"，结果是"区域外的出口"。
分片信息需要在**边界的两侧**都能被表达 —— 这正是 L2-08 讲的"边是双向通道"的结构基础。

---

## 四、两条**跳过**规则

### 跳过 token（non-shaped 类型）

```mlir
func.func @named_computation_skip_tokens(%arg0: tensor<8x2xi32>, %arg1: !stablehlo.token) -> (tensor<8x2xi32>, !stablehlo.token) {
```

```mlir
  %0:2 = sdy.named_computation<"foo">(%arg0, %arg1) (%arg2: tensor<8x2xi32>, %arg3: !stablehlo.token) {
    sdy.return %arg2, %arg3 : tensor<8x2xi32>, !stablehlo.token
  } : (tensor<8x2xi32>, !stablehlo.token) -> (tensor<8x2xi32>, !stablehlo.token)
  return %0#0, %0#1 : tensor<8x2xi32>, !stablehlo.token
}
```

```mlir
  // CHECK-NEXT: %[[NC:.*]]:2 = sdy.named_computation<"foo">(%arg0, %arg1) (%arg2: tensor<8x2xi32>, %arg3: !stablehlo.token) {
  // CHECK-NEXT:   %[[EDGE_1:.*]] = sdy.data_flow_edge %arg2 : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[EDGE_1]], %arg3 : tensor<8x2xi32>, !stablehlo.token
  // CHECK-NEXT: } : (tensor<8x2xi32>, !stablehlo.token) -> (tensor<8x2xi32>, !stablehlo.token)
  // CHECK-NEXT: %[[EDGE_2:.*]] = sdy.data_flow_edge %[[NC]]#0 : tensor<8x2xi32>
  // CHECK-NEXT: return %[[EDGE_2]], %[[NC]]#1 : tensor<8x2xi32>, !stablehlo.token
```

**读法**：
- `%arg2`（tensor）**有**边；`%arg3`（**token**）**没有**边。
- 区域外同理：`%[[NC]]#0` 有边，`%[[NC]]#1`（token）直接返回。
- 注意 `sdy.return %[[EDGE_1]], %arg3` —— token 原样返回，不走边。

**理由**：token 是 non-shaped 类型，分片对它没有意义（L1-02 / L2-08 一致）。

### 跳过动态形状

```mlir
func.func @optimization_barrier_dynamic_shaped_tensor_skipped(%arg0: tensor<32x96xf32>, %arg1: tensor<?x?xf32>)
    -> (tensor<32x96xf32>, tensor<?x?xf32>) {
```

```mlir
  %0:2 = stablehlo.optimization_barrier %arg0, %arg1 : tensor<32x96xf32>, tensor<?x?xf32>
  return %0#0, %0#1 : tensor<32x96xf32>, tensor<?x?xf32>
}
```

```mlir
  // CHECK-NEXT: %[[OPT_BARRIER:.*]]:2 = stablehlo.optimization_barrier %arg0, %arg1
  // CHECK:      %[[EDGE_1:.*]] = sdy.data_flow_edge %[[OPT_BARRIER]]#0
  // CHECK-NEXT: return %[[EDGE_1]], %[[OPT_BARRIER]]#1
```

**读法**：`#0` 是静态形状 `tensor<32x96xf32>` → **有**边；
`#1` 是动态形状 `tensor<?x?xf32>` → **没有**边，直接返回。

**理由**：L1-08 讲过数据流边**要求静态形状**（`expected sdy.data_flow_edge to have a static-shaped result`）。
动态形状无法确定分片，插边没有意义。

---

## 五、`manual_computation` 的边**带分片**

```mlir
  %0:2 = sdy.manual_computation(%arg0, %arg1)
      in_shardings=[<@mesh, [{"a", ?}, {?}], replicated={"b"}>, <@mesh, [{"b", ?}, {?}]>]
      out_shardings=[<@mesh, [{"a", ?}, {?}], replicated={"b"}>, <@mesh, [{"b", ?}, {?}]>]
      manual_axes={"b"}  (%arg2: tensor<8x2xi32>, %arg3: tensor<2x2xi32>) {
    sdy.return %arg2, %arg3 : tensor<8x2xi32>, tensor<2x2xi32>
  } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  return %0#0, %0#1 : tensor<8x2xi32>, tensor<4x2xi32>
```

```mlir
  // CHECK-NEXT: %[[MC:.*]]:2 = sdy.manual_computation(%arg0, %arg1)
  // CHECK-NEXT:   %[[EDGE_1:.*]] = sdy.data_flow_edge %arg2 sharding=<@mesh, [{"a", ?}, {?}]> : tensor<8x2xi32>
  // CHECK-NEXT:   %[[EDGE_2:.*]] = sdy.data_flow_edge %arg3 sharding=<@mesh, [{?}, {?}]> : tensor<2x2xi32>
  // CHECK-NEXT:   sdy.return %[[EDGE_1]], %[[EDGE_2]] : tensor<8x2xi32>, tensor<2x2xi32>
  // CHECK-NEXT: } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  // CHECK-NEXT: %[[EDGE_3:.*]] = sdy.data_flow_edge %[[MC]]#0 sharding=<@mesh, [{"a", ?}, {?}], replicated={"b"}> : tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE_4:.*]] = sdy.data_flow_edge %[[MC]]#1 sharding=<@mesh, [{"b", ?}, {?}]> : tensor<4x2xi32>
  // CHECK-NEXT: return %[[EDGE_3]], %[[EDGE_4]] : tensor<8x2xi32>, tensor<4x2xi32>
```

**读法**（本课信息量最大的一处）：
- 区域内 `%arg2` 的边带 `[{"a", ?}, {?}]` —— 来自 `in_shardings[0]`，
  **注意 `replicated={"b"}` 没有带进来**（区域内的值不再有 manual 轴的复制标注）。
- 区域内 `%arg3` 的边带 `[{?}, {?}]` —— 来自 `in_shardings[1]` 的**局部形状**版本
  （注意 `%arg3` 的类型是 `tensor<2x2xi32>`，是切过 manual 轴后的局部形状）。
- 区域外 `%[[MC]]#0` 的边带 `[{"a", ?}, {?}], replicated={"b"}` —— 完整来自 `out_shardings[0]`。
- 区域外 `%[[MC]]#1` 的边带 `[{"b", ?}, {?}]` —— 来自 `out_shardings[1]`。

**结论**：`manual_computation` 的边会**携带 in/out_shardings 的分片信息** ——
因为它没有别的途径表达"输入输出该怎么切"（区域算子本身不带分片属性）。

---

## 六、用例族谱（16 个）

| 族 | 用例 | 说明 |
|---|---|---|
| **case** | `case`、`case_existing_sharding`、`case_token_result_skipped` | 每个结果一条边 |
| **barrier** | `optimization_barrier`、`optimization_barrier_dynamic_shaped_tensor_skipped` | 动态形状跳过 |
| **while** | `while_unused_result` | 未使用的结果也插边 |
| **named_computation** | `named_computation_multiple_inputs_outputs`、`_with_shardings`、`_ops_inside_and_outside`、`_unused_result`、`_skip_tokens` | 内外两侧都插 |
| **manual_computation** | `manual_computation_skip_tokens`、`_multiple_inputs_outputs`、`_user_priority` | 边带分片 |
| **综合** | `main` | |
