<!-- sdy-coverage
ir/test/named_computation_parse_print.mlir
ir/test/named_computation_verification.mlir
ir/test/data_flow_edge_verification.mlir
ir/test/func_data_flow_edge_verification.mlir
-->

# L1-08 · named-computation-and-dataflow — 源 IR

本课覆盖四个文件，共 238 行：

| 文件 | 行数 | 作用 |
|---|---|---|
| `ir/test/named_computation_parse_print.mlir` | 49 | 4 个合法用例：单入单出、多入多出、带 in/out 分片、token |
| `ir/test/named_computation_verification.mlir` | 89 | 9 条校验错误 |
| `ir/test/data_flow_edge_verification.mlir` | 42 | 4 条校验错误 |
| `ir/test/func_data_flow_edge_verification.mlir` | 58 | 2 条校验错误 |

---

## 一、`sdy.named_computation` 的基本形态

网格：`sdy.mesh @mesh = <["a"=2, "b"=2]>`

### 1. 单入单出

```mlir
  %0 = sdy.named_computation<"foo">(%arg0) (%arg1: tensor<8x2xi32>) {
    sdy.return %arg1 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
```

### 2. 多入多出

```mlir
  %0:2 = sdy.named_computation<"named_computation">(%arg0, %arg1) (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>) {
    sdy.return %arg2, %arg3 : tensor<8x2xi32>, tensor<4x2xi32>
  } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
```

### 3. 带 in/out 分片（输入输出可以不同）

```mlir
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"b"}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>] (%arg1: tensor<8x2xi32>) {
    sdy.return %arg1 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
```

### 4. token 类型：分片必须是 rank 0

```mlir
  %0:2 = sdy.named_computation<"foo">(%arg0, %arg1) in_shardings=[<@mesh, [{"b"}, {}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, []>] (%arg2: tensor<8x2xi32>, %arg3: !stablehlo.token) {
    sdy.return %arg2, %arg3 : tensor<8x2xi32>, !stablehlo.token
  } : (tensor<8x2xi32>, !stablehlo.token) -> (tensor<8x2xi32>, !stablehlo.token)
```

---

## 二、`named_computation` 的校验错误

### 1. 块参数类型必须与操作数类型一致

```mlir
  %0 = sdy.named_computation<"bar">(%arg0) (%arg1: tensor<4x2xi32>) {
```

### 2. 块参数个数必须等于操作数个数

```mlir
  %0 = sdy.named_computation<"bar">(%arg0) (%arg1: tensor<8x2xi32>, %arg2: tensor<8x2xi32>) {
```

### 3. 返回值必须能推出结果类型

```mlir
  %0 = sdy.named_computation<"bar">(%arg0) (%arg1: tensor<8x2xi32>) {
```

### 4. in/out 分片项数必须匹配

```mlir
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"b"}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, [{"a"}, {}]>] (%arg1: tensor<8x2xi32>) {
```

```mlir
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"b"}, {}]>, <@mesh, [{"a"}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>] (%arg1: tensor<8x2xi32>) {
```

### 5. 分片本身必须合法（rank / 网格）

```mlir
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"b"}, {}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>] (%arg1: tensor<8x2xi32>) {
```

```mlir
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@unknown_mesh, [{"b"}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>] (%arg1: tensor<8x2xi32>) {
```

---

## 三、`sdy.data_flow_edge`

网格：`sdy.mesh @mesh = <["a"=2]>`

### 1. 分片校验与普通张量分片相同

```mlir
  %0 = sdy.data_flow_edge %arg0 sharding=<@mesh, [{}, {"a"}]> : tensor<8xf32>
```

### 2. 结果必须是静态形状

```mlir
  %0 = sdy.data_flow_edge %arg0 : tensor<?x?xf32>
```

### 3. 输入只能有一个使用者

```mlir
  %0 = sdy.data_flow_edge %arg0 : tensor<32x96xf32>
```

### 4. 输入不能由 SDY 方言的算子定义

```mlir
  %1 = sdy.data_flow_edge %0 : tensor<32x96xf32>
```

（其前一行是 `sdy.sharding_constraint`，见下方完整片段）

```mlir
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {}]> : tensor<32x96xf32>
  // expected-error @+1 {{expected input of sdy.data_flow_edge to not be defined by an SdyDialect op}}
  %1 = sdy.data_flow_edge %0 : tensor<32x96xf32>
```

---

## 四、`sdy.func_data_flow_edge`

### 1. 结果必须是静态形状

```mlir
  %0 = sdy.func_data_flow_edge %arg0 : tensor<?xf32>
```

### 2. 操作数只能有一个使用者

```mlir
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xf32>
```
