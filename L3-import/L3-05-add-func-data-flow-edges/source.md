<!-- sdy-coverage
transforms/import/test/add_func_data_flow_edges.mlir
-->

# L3-05 · add-func-data-flow-edges — 源 IR

覆盖：`shardy/dialect/sdy/transforms/import/test/add_func_data_flow_edges.mlir`
（471 行 / 17 个用例）。

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-add-func-data-flow-edges | FileCheck %s
```

**与 L3-04 的关系**：L3-04 插的是**算子级**边（区域算子的结果/块参数）；
这一课插的是**函数级**边（`sdy.func_data_flow_edge`）。

**与 L1-08 的关系**：L1-08 讲了 `func_data_flow_edge` 的**语法与校验**（两条约束）；
这一课讲它**怎么被插进来**。

---

## 一、★ 两个插入位置

### 位置 ①：函数**参数**（函数体内）

```mlir
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
```

```mlir
  %0 = stablehlo.negate %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT:                stablehlo.negate %[[EDGE]]
```

**读法**：函数参数 `%arg0` 后面插一条 `func_data_flow_edge`，
函数体里的使用者（`negate`）**改用边**。

### 位置 ②：**调用结果**（调用点）

```mlir
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
```

```mlir
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%[[ABS]])
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %[[CALL]]
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]]
```

**读法**：`call` 的结果 `%[[CALL]]` 后面插一条边，下游 `abs` **改用边**。

**两个位置合起来**就是 L1-08 说的「桥接调用点的实参与函数体的形参」：

```
调用点的实参  ←→  函数体的形参
   调用结果    ←→  函数的 return
```

---

## 二、多结果：每个结果一条边

```mlir
func.func @bar(%arg0: tensor<8xf32>) ->(tensor<8xf32>, tensor<8xf32>) {
```

```mlir
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  %1 = stablehlo.abs %arg0: tensor<8xf32>
  return %0, %1 : tensor<8xf32>, tensor<8xf32>
}
```

```mlir
// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) ->(tensor<8xf32>, tensor<8xf32>) {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT:                stablehlo.negate %[[EDGE]]
```

```mlir
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
```

```mlir
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1:2 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  %2 = stablehlo.add %1#0, %1#1 : tensor<8xf32>
  %3 = stablehlo.abs %1#1 : tensor<8xf32>
  %4 = stablehlo.multiply %2, %3 : tensor<8xf32>
  return %4 : tensor<8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[CALL:.*]]:2 = call @bar(%[[ABS]])
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.func_data_flow_edge %[[CALL]]#0
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.func_data_flow_edge %[[CALL]]#1
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[EDGE0]], %[[EDGE1]]
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE1]]
```

**读法**：`call` 有 2 个结果 → 插 **2 条边**；下游全部改用边。
注意 `%[[EDGE1]]` 被 `add` 与 `abs` **共用**（多使用者共享一条边）——
与 L3-04 的"边独占结果"是同一个不变量。

---

## 三、链式调用图

`main` → `bar` → `foo`：

```mlir
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
```

```mlir
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @foo(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}
```

```mlir
// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %[[EDGE0]]
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%[[ABS0]])
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.func_data_flow_edge %[[CALL]]
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[EDGE1]]
```

**读法**：`@bar` 里**同时**有两种边：
- `%[[EDGE0]]` —— 参数边（来自 `%arg0`）
- `%[[EDGE1]]` —— 调用结果边（来自 `call @foo`）

两者**互相独立**，各自服务自己那一段数据流。

---

## 四、token 被跳过（两个用例，注释写明了理由）

### 函数参数是 token

```mlir
// test: token_typed_func_argument_skipped
// Tokens are not static-shaped types, so no func_data_flow_edge should be
// created for them. Only the tensor argument gets an edge op.
// CHECK-LABEL: @bar(%arg0: !stablehlo.token, %arg1: tensor<8xf32>)
func.func @bar(%arg0: !stablehlo.token, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %arg1
  // CHECK-NEXT:                stablehlo.negate %[[EDGE]]
  // CHECK-NOT:  sdy.func_data_flow_edge %arg0
  %0 = stablehlo.negate %arg1 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %arg1
  // CHECK-NEXT:                stablehlo.negate %[[EDGE]]
  // CHECK-NOT:  sdy.func_data_flow_edge %arg0
```

**读法**：`%arg1`（tensor）**有**边；`%arg0`（**token**）**没有**边。
`CHECK-NOT` 精确断言了这一点。

### 调用结果是 token

```mlir
// test: token_typed_call_result_skipped
// Tokens are not static-shaped types, so no func_data_flow_edge should be
// created for them. Only the tensor result gets an edge op.
// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> (!stablehlo.token, tensor<8xf32>) {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT: %[[TOK:.*]] = stablehlo.create_token
  // CHECK-NEXT: %[[NEG:.*]] = stablehlo.negate %[[EDGE]]
  %tok = stablehlo.create_token : !stablehlo.token
  %0 = stablehlo.negate %arg0 : tensor<8xf32>
  return %tok, %0 : !stablehlo.token, tensor<8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[CALL:.*]]:2 = call @bar(%arg0)
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %[[CALL]]#1
  // CHECK-NOT:  sdy.func_data_flow_edge %[[CALL]]#0
  // CHECK-NEXT: stablehlo.abs %[[EDGE]]
```

**读法**：`%[[CALL]]#1`（tensor）**有**边；`%[[CALL]]#0`（**token**）**没有**。

**理由**（测试注释原文）：

```
// Tokens are not static-shaped types, so no func_data_flow_edge should be
// created for them.
```

这与 L3-04 的 token 跳过规则**完全一致**（那里跳过的是算子级边）。

---

## 五、其余用例速览

| 用例 | 看点 |
|---|---|
| `simple_call_graph_on_func_with_single_argument` | 最简形态 |
| `simple_call_graph_on_func_multiple_users_on_func_result` | 多使用者共用一条边 |
| `simple_call_graph_on_func_with_multiple_results` | 每个结果一条边 |
| `simple_call_graph_on_func_with_sharded_argument` | 带分片标注的参数 |
| `multiple_calls_on_same_func` | 同一函数被调多次（每次调用各自插边） |
| `simple_call_graph_on_func_with_multiple_argument` | 多参数 |
| `..._with_multiple_argument_same_operand` | 同一值传给两个参数 |
| `simple_chain_call_graph` | `main → bar → foo` |
| `simple_non_flat_call_graph` | 非扁平调用图 |
| `simple_non_flat_call_graph_one_after_the_other` | 顺序调用两个函数 |
| `call_on_same_func_twice_input_of_one_is_output_of_the_other` | 前一次的输出是后一次的输入 |
| `simple_call_graph_argument_is_input_to_call` | 参数直接作为调用实参 |
| `simple_call_graph_result_is_the_output_of_call` | 结果直接作为返回值 |
| `simple_call_graph_entry_contains_call_only`（×2） | 函数体只有一个调用 |
| `token_typed_func_argument_skipped` | token 参数跳过 |
| `token_typed_call_result_skipped` | token 结果跳过 |

---

## 六、三种边的对照

| 边 | 出现在 | 桥接什么 | 讲在哪 |
|---|---|---|---|
| `sdy.data_flow_edge` | 区域算子旁 | 区域算子的结果 ↔ 下游 | L1-08 / L2-08 / L3-04 |
| `sdy.func_data_flow_edge` | 函数参数 / 调用结果 | 调用实参 ↔ 函数形参 | **本课** |
| `sdy.named_computation` | 函数调用点 | 整个函数体（内联后） | L1-08 / L3-06 |
