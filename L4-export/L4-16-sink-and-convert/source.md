<!-- sdy-coverage
transforms/export/test/sink_data_flow_edges.mlir
transforms/export/test/sink_func_data_flow_edges.mlir
transforms/export/test/sharding_constraint_to_reshard.mlir
-->

# L4-16 · sink-and-convert — 源 IR

**共同点**：三个 pass 都在**消除中间算子** ——
把"标注型"的 op 转成"直接写在值上"或"等价的具体算子"。

| pass | 文件 | 行数 | 用例数 |
|---|---|---|---|
| `-sdy-sink-data-flow-edges` | `sink_data_flow_edges.mlir` | 251 | 13 |
| `-sdy-sink-func-data-flow-edges` | `sink_func_data_flow_edges.mlir` | 198 | 15 |
| `-sdy-sharding-constraint-to-reshard` | `sharding_constraint_to_reshard.mlir` | 19 | 2 |

网格：

```mlir
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>
```

---

## 一、★ 边的下沉：`sink_data_flow_edges`

### 背景：什么是数据流边

**L1-08** 讲过：区域算子（`while` / `case`）本身**不带分片属性** ——
分片写在 `sdy.data_flow_edge` 上。

```mlir
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>
```

```mlir
func.func @data_flow_edge_on_block_arg(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
```

```mlir
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.constant dense<1> : tensor<i32>
  %2 = stablehlo.constant dense<32> : tensor<i32>
```

```mlir
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare  LT, %iterArg_2, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %4 = sdy.data_flow_edge %iterArg sharding=<@mesh, [{"a"}, {}]> : tensor<32x96xf32>
    %5 = stablehlo.add %iterArg_2, %1 : tensor<i32>
    %6 = stablehlo.add %4, %4 : tensor<32x96xf32>
    stablehlo.return %6, %5 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}
```

**读法**：
- `sdy.data_flow_edge %iterArg sharding=<@mesh, [{"a"}, {}]>` ——
  这条边声明"`%iterArg` 这个**循环变量**的分片是 `[{"a"}, {}]`"。
- `%6 = stablehlo.add %4, %4` —— 后续算子用的是**边的结果** `%4`。

### 输出：边被删除

```mlir
  // CHECK:      %[[WHILE:.*]]:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %[[C0]])
  // CHECK:      } do {
  // CHECK-NEXT:   %[[ADD_1:.*]] = stablehlo.add %iterArg_2, %[[C1]]
  // CHECK-NEXT:   %[[ADD_2:.*]] = stablehlo.add %iterArg, %iterArg
  // CHECK-NEXT:   stablehlo.return %[[ADD_2]], %[[ADD_1]]
  // CHECK-NEXT: }
  // CHECK-NOT:  sdy.sharding
  // CHECK-NEXT: return %[[WHILE]]#0
```

**读法**（两处变化）：
1. `%[[ADD_2]] = stablehlo.add %iterArg, %iterArg` ——
   原来是 `add %4, %4`（用边的结果），现在**直接用 `%iterArg`**。
2. `CHECK-NOT: sdy.sharding` —— **边 op 被删了**。

**分片去哪了**：下沉到了 **`%iterArg` 这个值本身**上 ——
即 `while` 的操作数（`%arg0`）承载分片。

### 边作用在**算子结果**上时

```mlir
func.func @data_flow_edge_on_op_result(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
```

```mlir
  // CHECK:      %[[WHILE:.*]]:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %[[C0]])
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>, <@mesh, []>]>}
```

**读法**：这一次分片**落在了 `while` op 上**（`CHECK-SAME` 紧跟 while 行）——
因为边作用的 `%3#0` 是 `while` 的**结果**。

> **下沉的目标**：边 op 的**输入值**（`%iterArg` 或 `%3#0`）。
> 分片从"边的属性"变成"**那个值所在算子上的分片**"。

### 13 个用例覆盖的边界

| 用例 | 场景 |
|---|---|
| `data_flow_edge_on_block_arg` | 边在 **block argument** 上 |
| `data_flow_edge_on_op_result` | 边在**算子结果**上 |
| `no_shardings` | 边**没有** sharding |
| `some_edges_have_sharding` | **部分**边有 sharding |
| `all_edges_have_sharding` | **全部**边有 sharding |
| `missing_edge` | **缺**某条边 |
| `sharding_overrided` | 分片被**覆盖** |
| `edge_missing_sharding` | 边缺 sharding |

**注意 `missing_edge`**：`while` 有多个 iterArg，但只写了**部分**边 ——
说明边的数量与 iterArg 数量可以**不匹配**。

**测试开头的 TODO 注释**：

```
// TODO(tomnatan): once ops like while are allowed to have shardings with
// different meshes, add a test that verifies that the first mesh name is used
// for missing shardings.
```

**读法**：目前 `while` 上的分片**必须用同一个 mesh** ——
"缺 sharding 时用第一个 mesh 名"这个规则**还没有测试覆盖**（上游 TODO）。
这是**已知的测试空缺**，值得在课件里点出。

---

## 二、★ 函数级边的下沉：`sink_func_data_flow_edges`

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-sink-func-data-flow-edges | FileCheck %s
```

### 函数内部的边

```mlir
// CHECK-LABEL: func private @bar(%arg0: tensor<8xf32>)
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
```

```mlir
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xf32>
  %1 = stablehlo.negate %0: tensor<8xf32>
  return %1 : tensor<8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %arg0
  // CHECK-NEXT: return %[[NEGATE]]
```

**读法**：`negate %0` → **`negate %arg0`** —— 边被删，直接用原值。

### 调用点上的边

```mlir
// CHECK-LABEL: func @simple_call_graph_on_func_with_single_argument(%arg0: tensor<8xf32>)
func.func @simple_call_graph_on_func_with_single_argument(%arg0: tensor<8xf32>) -> tensor<8xf32> {
```

```mlir
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = sdy.func_data_flow_edge %1 : tensor<8xf32>
  %3 = stablehlo.abs %2 : tensor<8xf32>
  return %3 : tensor<8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%[[ABS0]])
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[CALL]]
  // CHECK-NEXT: return %[[ABS1]]
```

**读法**：`abs %2` → **`abs %[[CALL]]`** —— 边被删，直接用 `call` 的结果。

### 与 `data_flow_edge` 的区别

| | `sdy.data_flow_edge` | `sdy.func_data_flow_edge` |
|---|---|---|
| 作用域 | **区域算子内部**（while/case 的 block） | **函数调用**（参数与结果） |
| 覆盖课 | L1-08（边）、L3-04（插入） | L3-05（插入）、L3-10（搬分片） |
| 本课 pass | `-sdy-sink-data-flow-edges` | `-sdy-sink-func-data-flow-edges` |

**共同点**：都是"**桥接**"op —— 把分片从一处"传递"到另一处。
传播结束后，桥接的使命完成 → **下沉并删除**。

**15 个用例**覆盖各种调用图形态（单参数 / 多参数 / 嵌套调用等）。

---

## 三、`sharding_constraint_to_reshard`：约束变搬运

```mlir
// RUN: sdy_opt %s -sdy-sharding-constraint-to-reshard | FileCheck %s
```

```mlir
func.func @sharding_constraint_to_reshard(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
```

```mlir
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {?}]> {foo} :  tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
```

```mlir
  // CHECK: %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {?}]> {foo} : tensor<8x8xf32>
```

**读法**（**只改了 op 名**）：
- `sdy.sharding_constraint` → **`sdy.reshard`**
- 分片参数 `<@mesh, [{"a"}, {?}]>` **完全不变**
- 附加属性 `{foo}` **也保留**

### token 的特殊情形

```mlir
// Verify that ShardingConstraintOp on a token converts to a 0-dimensional
// ReshardOp.
// CHECK-LABEL: func @token_sharding_constraint_to_reshard
func.func @token_sharding_constraint_to_reshard(%arg0: !stablehlo.token) -> !stablehlo.token {
```

```mlir
  %0 = sdy.sharding_constraint %arg0 <@mesh, []> : !stablehlo.token
  return %0 : !stablehlo.token
}
```

```mlir
  // CHECK: %0 = sdy.reshard %arg0 <@mesh, []> : !stablehlo.token
```

**读法**：token 是**无维度**的值 → 转换成 **0 维 reshard**（`<@mesh, []>`）。

### 为什么需要这个转换

**L1-10** 讲过 `sdy.sharding_constraint` 的生命周期：
- **传播期**：它是**用户意图** —— 传播**消费**它（读它的分片，然后删除或保留）。
- **传播后**：如果它还在（比如悬空的约束，L1-09 讲过），
  它就变成一条**真正的搬运** → 需要转换成 `sdy.reshard`。

**为什么不能直接用 `reshard`**：
`sharding_constraint` 的语义是"**我希望这里是这个分片**"（**声明性**），
而 `reshard` 是"**请把它搬成这样**"（**命令性**）。
前者允许传播器**自由选择**是否满足（可能通过改变上游来满足），
后者是**明确的搬运指令**。

**传播结束后**，"我希望"已经没有意义了 —— 要么已经满足（那这条就是冗余的），
要么没满足（那必须真的搬）→ 所以统一转成 `reshard`。

**与 L4-10 的关系**：L4-10 消除**冗余**的 permute；本 pass 把**约束**转成**搬运**。
两者都在处理"传播后遗留的 op"。

---

## 四、三个 pass 的共同点

| pass | 消除什么 | 分片去哪 |
|---|---|---|
| `sink_data_flow_edges` | `sdy.data_flow_edge` | 落到边的**输入值**上 |
| `sink_func_data_flow_edges` | `sdy.func_data_flow_edge` | 落到边的**输入值**上 |
| `sharding_constraint_to_reshard` | `sdy.sharding_constraint` | **转成** `sdy.reshard` |

**一句话总结**：
> **传播结束后，所有"标注型"的 op 都要消失** ——
> 边被下沉（分片直接写在值上），约束被转成真正的搬运。
>
> 它们都是**传播期的辅助机制**，导出时不再需要。
