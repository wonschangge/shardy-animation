<!-- sdy-coverage
ir/test/sharding_rule_parse_print.mlir
ir/test/sharding_rule_parsing_failure.mlir
ir/test/sharding_rule_verification.mlir
-->

# L1-05 · op-sharding-rule — 源 IR

本课覆盖三个测试文件：

| 文件 | 行数 | 作用 |
|---|---|---|
| `ir/test/sharding_rule_parse_print.mlir` | 69 | 8 个合法规则：rank-0、复合因子、>17 维符号、因子四分类、`custom` |
| `ir/test/sharding_rule_parsing_failure.mlir` | 223 | 47 条解析错误（长行用 CHECK 分段引用） |
| `ir/test/sharding_rule_verification.mlir` | 151 | 19 条语义校验错误 |

---

## 一、基本形态

### 1. rank-0：三个空列表

```mlir
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([], [])->([])>} : tensor<f32>
```

### 2. 最简的非平凡规则

```mlir
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=16, j=32}, custom>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
```

---

## 二、复合因子：reshape 的核心

把 `2x4` reshape 成 `8`：两个因子 `i`、`j` 合并成一个维度，写作 `[ij]`（**没有空格**）。

```mlir
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([ij]) {i=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>
```

---

## 三、超过 17 维：符号命名规则

`i`–`z` 只有 18 个符号，超过后用 `z_N` 续接。声明里逐个写 `z_1`、`z_2`，而映射里把相邻的多个拼在一起（如 `zz_1`、`z_8z_9z_10`）。

第 19 维（倒数第 2 维）的写法：

```mlir
  // CHECK:      #sdy.op_sharding_rule<([i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, z_1, z_2])
  // CHECK-SAME: ->([i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, zz_1, z_2])
  // CHECK-SAME: {i=2, j=2, k=2, l=2, m=2, n=2, o=2, p=2, q=2, r=2, s=2, t=2, u=2, v=2, w=2, x=2, y=2, z=2, z_1=2, z_2=2}>}
```

最后一维把三个符号拼在一起（`z_8z_9z_10`）：

```mlir
  // CHECK:      #sdy.op_sharding_rule<([i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8, z_9, z_10])
  // CHECK-SAME: ->([i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8z_9z_10])
  // CHECK-SAME: {i=2, j=2, k=2, l=2, m=2, n=2, o=2, p=2, q=2, r=2, s=2, t=2, u=2, v=2, w=2, x=2, y=2, z=2, z_1=2, z_2=2, z_3=2, z_4=2, z_5=2, z_6=2, z_7=2, z_8=2, z_9=2, z_10=2}>} :
```

---

## 四、因子四分类

一条规则可以标注哪些因子是特殊的。四类可同时出现：

```mlir
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k, l])->([i, k, l]) {i=2, j=3, k=5, l=7} reduction={j} need_replication={i, l} permutation={k}, custom>} : (tensor<2x3x5x7xf32>) -> tensor<2x11x7xf32>
```

再加上 `blocked_propagation`（注意这一类**不带** `custom`）：

```mlir
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k, l])->([i, k, l]) {i=2, j=3, k=5, l=7} reduction={j} need_replication={i, l} permutation={k} blocked_propagation={l}>} : (tensor<2x3x5x7xf32>) -> tensor<2x11x7xf32>
```

---

## 五、`custom` 标记与大因子

`custom` 表示这条规则由用户提供（不会被传播删除）。

```mlir
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=16, j=32}, custom>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
```

因子大小不受 32 位限制：

```mlir
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i])->([i]) {i=8589934592}, custom>} : (tensor<8589934592xf32>) -> tensor<8589934592xf32>
```

---

## 六、解析错误（47 条，节选）

### 1. `custom` 标记写错

```mlir
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=16, j=32},>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
```

```mlir
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=16, j=32}, custom_rule>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
```

### 2. 复合因子必须写成 `[ij]`，不能有空格

```mlir
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i j]) {i=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>
```

### 3. 因子大小必须按 iota 顺序书写

```mlir
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([ij]) {j=4, i=2}>} : (tensor<2x4xf32>) -> tensor<8xf32>
```

```mlir
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, k])->([ik]) {i=2, k=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>
```

### 4. 符号必须是 `i`–`z`（或 `z_N`）

```mlir
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([a, j])->([j]) {a=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>
```

```mlir
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([z_, j])->([i]) {z_=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>
```

```mlir
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i_1, j])->([i]) {i_1=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>
```

### 5. `z_` 后必须是正整数、无前导零、不溢出

```mlir
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([z_0z_, j])->([i]) {z_0=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>
```

```mlir
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([z_-1, j])->([i]) {i=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>
```

### 6. 空的操作数/结果列表也要写 `[]`

```mlir
  %0 = stablehlo.custom_call @foo() {sdy.sharding_rule = #sdy.op_sharding_rule<()->([i, j]) {i=2, j=8}, custom>} : () -> tensor<2x8xf32>
```

### 7. 特殊因子集合的写法

```mlir
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k, l])->([i, k, l]) {i=2, j=3, k=5, l=7} reduction: {j}>} : (tensor<2x3x5x7xf32>) -> tensor<2x5x7xf32>
```

```mlir
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k, l])->([i, k, l]) {i=2, j=3, k=5, l=7} reduce={j}>} : (tensor<2x3x5x7xf32>) -> tensor<2x5x7xf32>
```

### 8. reduction 因子不能出现在结果映射里

```mlir
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k, l])->([i, k, l]) {i=2, j=3, k=5, l=7} reduction={k}>} : (tensor<2x3x5x7xf32>) -> tensor<2x5x7xf32>
```

---

## 七、语义校验错误（19 条，节选）

### 1. 属性类型必须正确

```mlir
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = 1 : i64} : (tensor<8xf32>) -> tensor<8xf32>
```

### 2. 只支持有静态形状的 ranked tensor

```mlir
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=2, j=4}>} : tensor<*xf32>
```

```mlir
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=2, j=4}>} : tensor<?x?xf32>
```

### 3. 映射个数必须与操作数/结果个数一致

```mlir
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=2, j=4}>} : tensor<2x4xf32>
```

```mlir
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j], [i, j]) {i=2, j=4}>} : tensor<2x4xf32>
```

### 4. 每个映射的 rank 必须与张量一致

```mlir
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i])->([i, j]) {i=2, j=4}>} : tensor<2x4xf32>
```

```mlir
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i]) {i=2, j=4}>} : tensor<2x4xf32>
```

### 5. 复合因子中不允许 size=1 的因子

```mlir
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, jk], [i, j])->([i, j]) {i=2, j=8, k=1}>} : tensor<2x8xf32>
```

### 6. 维度映射不能为空，也不能重复用同一个因子

```mlir
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i,], [i,])->([i, j]) {i=2, j=8}>} : tensor<2x8xf32>
```

```mlir
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([ij, k])->([iik]) {i=2, j=2, k=2}>} : (tensor<4x2xf32>) -> tensor<8xf32>
```

```mlir
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, i])->([ij]) {i=2, j=2}>} : (tensor<2x2xf32>) -> tensor<4xf32>
```

### 7. 因子索引越界 / 未被使用

```mlir
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, u], [i, j])->([i, j]) {i=2, j=8}>} : tensor<2x8xf32>
```

```mlir
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=2, j=8, k=2}>} : tensor<2x8xf32>
```

### 8. 特殊因子集合：必须有序、唯一、在范围内、互斥

```mlir
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, k]) {i=2, j=4, k=8} need_replication={k, i}>} : (tensor<2x4x8xf32>) -> tensor<2x8xf32>
```

```mlir
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, k]) {i=2, j=4, k=8} need_replication={i, i}>} : (tensor<2x4x8xf32>) -> tensor<2x8xf32>
```

```mlir
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, k]) {i=2, j=4, k=8} need_replication={z}>} : (tensor<2x4x8xf32>) -> tensor<2x8xf32>
```

```mlir
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, k]) {i=2, j=4, k=8} blocked_propagation={k, i}>} : (tensor<2x4x8xf32>) -> tensor<2x8xf32>
```
