<!-- sdy-coverage
transforms/propagation/test/basic_propagation_data_flow_edges.mlir
transforms/propagation/test/basic_propagation_token.mlir
-->

# L2-08 · data-flow-edges — 源 IR

本课覆盖两个文件：

| 文件 | 行数 | 作用 |
|---|---|---|
| `transforms/propagation/test/basic_propagation_data_flow_edges.mlir` | 855 | 32 个用例：`case` / `while` / `optimization_barrier` / `manual_computation` |
| `transforms/propagation/test/basic_propagation_token.mlir` | 28 | token 类型在传播中被跳过 |

RUN 行分别：

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-basic-propagate 2>&1 | FileCheck %s
```

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-add-data-flow-edges -sdy-basic-propagate -sdy-sink-data-flow-edges 2>&1 | FileCheck %s
```

网格：`sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>`

---

## 一、核心模式：分片标在 `data_flow_edge` 上，**不在**区域算子上

以 `stablehlo.case` 为例。`%arg1` 带分片 `[{"a"}]`，且是第 0 个分支的返回值：

```mlir
  %0 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1 : tensor<4xi64>
  }, {
    stablehlo.return %arg2 : tensor<4xi64>
  // CHECK: })
  // CHECK-NOT: sdy.sharding
  }) : (tensor<i32>) -> tensor<4xi64>
```

```mlir
  %1 = sdy.data_flow_edge %0 : tensor<4xi64>
```

```mlir
  // CHECK-NEXT: %[[CASE:.*]] = "stablehlo.case"
```

```mlir
  // CHECK: })
  // CHECK-NOT: sdy.sharding
```

```mlir
  // CHECK-NEXT: sdy.data_flow_edge %[[CASE]] sharding=<@mesh_a_2_b_2, [{"a", ?}]>
```

**读法**：
- **`case` 算子上没有分片属性**（`CHECK-NOT: sdy.sharding`）。
- 分片出现在**后面那条 `sdy.data_flow_edge`** 上：`[{"a", ?}]`。
- 也就是说：区域算子被视为"数据流算子"，分片通过**边**来表达。

---

## 二、`while`：分片可以从前向后传

函数结果被锁成 `[{"a"}, {"b"}]`：

```mlir
func.func @while_func_return(%arg0: tensor<32x96xf32>) -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>}) {
```

```mlir
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %6 = stablehlo.compare  LT, %iterArg_2, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %6 : tensor<i1>
  } do {
    %6 = stablehlo.add %iterArg_2, %1 : tensor<i32>
    // CHECK: stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>} : tensor<32x96xf32>
    %7 = stablehlo.add %iterArg, %iterArg : tensor<32x96xf32>
    stablehlo.return %7, %6 : tensor<32x96xf32>, tensor<i32>
  }
```

```mlir
  %4 = sdy.data_flow_edge %3#0 : tensor<32x96xf32>
```

```mlir
  // CHECK: %[[WHILE:.*]]:2 = stablehlo.while
  // CHECK-NOT:  sdy.sharding
```

```mlir
    // CHECK: stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>} : tensor<32x96xf32>
```

```mlir
  // CHECK: sdy.data_flow_edge %[[WHILE]]#0 sharding=<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>
```

**读法**：函数结果的要求（`[{"a"},{"b"}]`）**反向**穿过 `while`，落到了体内的 `add` 上
（`[{"a", ?}, {"b", ?}]`），再回传到 `data_flow_edge`。

---

## 三、反向也行：从循环体**向外**传播

```mlir
    %7 = stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>]>} : tensor<32x96xf32>
```

```mlir
    // CHECK: stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>]>}
```

**读法**：这次是**体内算子显式带了闭维分片** `[{"a"}, {"b"}]`，
传播把它**反向**推给块参数 `%iterArg`，进而影响循环的输入与结果。

**结论**：数据流边是**双向**通道 —— 与前向/反向的数据流传播一致。

---

## 四、多结果：每条边**互相独立**

`stablehlo.optimization_barrier` 有 2 个结果：

```mlir
  %1:2 = stablehlo.optimization_barrier %0, %arg1 : tensor<32x96xf32>, tensor<32x96xf32>
```

```mlir
  %2 = sdy.data_flow_edge %1#0 : tensor<32x96xf32>
```

```mlir
  %3 = sdy.data_flow_edge %1#1 : tensor<32x96xf32>
```

```mlir
  // CHECK-NEXT: %[[OPT_BARRIER:.*]]:2 = stablehlo.optimization_barrier
  // CHECK-NOT:  sdy.sharding
```

```mlir
  // CHECK-NEXT: sdy.data_flow_edge %[[OPT_BARRIER]]#0 sharding=<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>
```

```mlir
  // CHECK-NEXT: sdy.data_flow_edge %[[OPT_BARRIER]]#1 sharding=<@mesh_a_2_b_2, [{"b", ?}, {?}]>
```

**读法**：两个结果的边拿到了**不同**的分片：
- `#0` → `[{"a", ?}, {"b", ?}]`
- `#1` → `[{"b", ?}, {?}]`

它们分别来自各自的下游要求，**互不干扰**。

---

## 五、`manual_computation`：体内也能有边

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_a_2_b_2_c_2, [{"b", ?}, {?}]>] out_shardings=[<@mesh_a_2_b_2_c_2, [{"b", ?}, {?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    %2 = sdy.data_flow_edge %arg1 sharding=<@mesh_a_2_b_2_c_2, [{?}, {?}]> : tensor<16x32xf32>
    %3 = stablehlo.add %2, %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>
    %4 = stablehlo.custom_call @sdy_testonly(%3) : (tensor<16x32xf32>) -> tensor<16x32xf32>
```

```mlir
  // CHECK-NEXT:   %[[EDGE_1:.*]] = sdy.data_flow_edge %arg1 sharding=<@mesh_a_2_b_2_c_2, [{"a", ?}, {"c", ?}]> : tensor<16x32xf32>
```

**读法**：体内的 `data_flow_edge` 从初始的 `[{?}, {?}]` 被更新为 `[{"a", ?}, {"c", ?}]` ——
`"a"` 是**自由轴**（由传播处理），`"c"` 来自 `%arg0` 的分片。

---

## 六、token：**直接跳过**

```mlir
func.func @func_token_arg_skipped(%arg0: !stablehlo.token) -> !stablehlo.token {
```

```mlir
  return %arg0 : !stablehlo.token
}
```

```mlir
// CHECK-LABEL: func @func_token_arg_skipped(
// CHECK-SAME:      %arg0: !stablehlo.token) -> !stablehlo.token
```

```mlir
  // CHECK-NEXT: return %arg0 : !stablehlo.token
```

**读法**：token 是 **non-shaped** 类型（L1-02），分片对它没有意义。
传播**完全跳过** token —— 不报错，也不添加任何属性。

跨函数的例子（`call` 的 token 结果同样被跳过）：

```mlir
func.func @main(%arg0: !stablehlo.token, %arg1: tensor<4xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}]>}) -> (!stablehlo.token, tensor<4xi64>) {
```

```mlir
  %0:2 = call @callee(%arg0, %arg1) : (!stablehlo.token, tensor<4xi64>) -> (!stablehlo.token, tensor<4xi64>)
```

```mlir
  %1 = sdy.sharding_constraint %0#1 <@mesh_a_2_b_2, [{"a"}]> : tensor<4xi64>
```

```mlir
  return %0#0, %1 : !stablehlo.token, tensor<4xi64>
}
```

```mlir
// CHECK-LABEL: func @main(
// CHECK-SAME:      %arg0: !stablehlo.token,
// CHECK-SAME:      %arg1: tensor<4xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}]>})
```

```mlir
  // CHECK: return %0#0, %1 : !stablehlo.token, tensor<4xi64>
```

**读法**：`%arg0`（token 参数）上**没有**任何分片属性；只有 `%arg1`（tensor）有。
同一次调用的两个结果也各自处理：token 被跳过，tensor 正常传播。
