<!-- sdy-coverage
ir/test/edge_sharding_parse_print.mlir
ir/test/edge_sharding_verification.mlir
-->

# L1-04 · edge-sharding — 边分片与传播调试信息

本课覆盖两个测试文件：

| 文件 | 行数 | 作用 |
|---|---|---|
| `ir/test/edge_sharding_parse_print.mlir` | 14 | `sdy.propagation_edges` 属性的解析与打印 |
| `ir/test/edge_sharding_verification.mlir` | 164 | 15 类校验用例（含 1 个"允许越界"的反例） |

---

## 一、`sdy.propagation_edges` 的语法

一条边分片记录"某个分片是**在第几步、从哪个值、沿哪条轴、传播到哪些值**"。

```mlir
%0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-2 = [{"y" = operand-1 -> [operand-0, result-0]}]}, {step-12345 = [{"x" = result-0 -> [operand-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"y"}, {"x" }]>]>} : tensor<16x8xf32>
```

打印时的换行形式（CHECK 行）：

```mlir
  // CHECK: %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[
```

```mlir
  // CHECK:   {step-2 = [{"y" = operand-1 -> [operand-0, result-0]}]},
```

```mlir
  // CHECK:   {step-12345 = [{"x" = result-0 -> [operand-0]}]}]>,
```

```mlir
  // CHECK:   sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"y"}, {"x"}]>]>
```

```mlir
  // CHECK: } : tensor<16x8xf32>
```

---

## 二、step 索引的约束

网格：`sdy.mesh @mesh = <["c"=8, "d"=8, "e"=8]>`

### 1. step 索引不可重复

```mlir
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-1 = [{"c":(1)4 = operand-0 -> [result-0]}, {"e" = operand-0 -> [result-0]}]},{step-1 = [{"d" = operand-1 -> [result-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"c":(1)4, ?}]>]>} : tensor<8x8xf32>
```

### 2. step 索引不可为负

```mlir
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step--3 = [{"c":(1)4 = operand-0 -> [result-0]}, {"e" = operand-0 -> [result-0]}]},{step-1 = [{"d" = operand-1 -> [result-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"c":(1)4, ?}]>]>} : tensor<8x8xf32>
```

---

## 三、source 与 target 的约束

### 1. source 不能等于 target（自环）

```mlir
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-1 = [{"a" = operand-0 -> [operand-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a":(1)4, ?}]>]>} : tensor<8x8xf32>
```

### 2. 同一 step、同一轴下 target 不可重复

```mlir
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-123 = [{"x" = result-0 -> [operand-0, operand-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"x":(1)4, ?}]>]>} : tensor<8x8xf32>
```

### 3. operand 索引必须在范围内

`stablehlo.add` 有两个操作数，索引必须落在 `[0, 2)`：

```mlir
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-22 = [{"z" = result-0 -> [operand-3]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"z", ?}]>]>} : tensor<8x8xf32>
```

### 4. result 索引必须在范围内

`stablehlo.add` 只有 1 个结果，索引必须落在 `[0, 1)`：

```mlir
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-22 = [{"z" = operand-1-> [operand-0,result-1]}]}]>} : tensor<8x8xf32>
```

### 5. 轴必须来自某个网格

```mlir
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-93 = [{"z" = operand-1-> [operand-0,result-1]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"z":(1)4, ?}]>]>} : tensor<8x8xf32>
```

---

## 四、必须同时带 sharding

### 1. 只有 edges、没有 sharding

```mlir
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-93 = [{"z" = operand-1-> [operand-0,result-0]}]}]>} : tensor<8x8xf32>
```

---

## 五、属性类型必须正确

### 1. `sdy.propagation_edges` 必须是 PropagationEdgesAttr

```mlir
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = 64} : tensor<8x8xf32>
```

### 2. `sdy.result_propagation_edges` 必须是 ArrayAttr

```mlir
    %0 = stablehlo.add %arg0, %arg0 {sdy.result_propagation_edges = 64} : tensor<8x8xf32>
```

### 3. 数组元素类型也要对

```mlir
    %0 = stablehlo.add %arg0, %arg0 {sdy.result_propagation_edges = [1,2,3]} : tensor<8x8xf32>
```

### 4. `sdy.block_arg_propagation_edges` 同理

```mlir
    %0 = stablehlo.add %arg0, %arg0 {sdy.block_arg_propagation_edges = 64} : tensor<8x8xf32>
```

### 5. block_arg 版本的数组元素类型

```mlir
    %0 = stablehlo.add %arg0, %arg0 {sdy.block_arg_propagation_edges = [0,2,4]} : tensor<8x8xf32>
```

### 6. 数据流算子上同样要求类型正确

```mlir
    %1 = sdy.data_flow_edge %0#0  {sdy.propagation_edges = 64} : tensor<32x96xf32>
```

---

## 六、反例：`sdy.func_data_flow_edge` 允许越界 operand

这一段**没有** `expected-error` —— 它验证的是"函数级数据流边上，越界的 operand 索引是允许的"。

网格：`sdy.mesh @mesh = <["z"=4]>`

```mlir
    %0 = sdy.func_data_flow_edge %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-22 = [{"z" = result-0 -> [operand-3]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"z", ?}]>]>} : tensor<8x8xf32>
```

---

## 附：本课用到的网格声明

```mlir
sdy.mesh @mesh1 = <["x"=2, "y"=2]>
```

```mlir
sdy.mesh @mesh = <["c"=8, "d"=8, "e"=8]>
```

```mlir
sdy.mesh @mesh = <["z"=4]>
```
