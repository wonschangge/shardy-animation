<!-- sdy-coverage
transforms/propagation/test/op_sharding_rule_registry.mlir
-->

# L2-10 · op-sharding-rule-registry — 源 IR

覆盖：`shardy/dialect/sdy/transforms/propagation/test/op_sharding_rule_registry.mlir`
（**1213 行 / 125 个用例**，本计划第二大文件）。

```mlir
// RUN: sdy_opt %s -sdy-populate-op-sharding-rules -verify-diagnostics 2>&1 | FileCheck %s
```

**注意 RUN 行**：`-sdy-populate-op-sharding-rules` 这个 pass 的作用是
**把注册表里为每个算子推导出的规则打印出来**，所以每个用例的 CHECK 行就是**生成的规则**。

> 本课的核心价值：L1-05 讲了规则的**语法**，这一课看**真实的注册表**为各类算子推导出了什么。

---

## 一、逐元素 / 标量：维度一一对应

```mlir
func.func @pointwise_op(%arg0: tensor<2x1x4xf32>) -> tensor<2x1x4xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg0: tensor<2x1x4xf32>
  return %0 : tensor<2x1x4xf32>
}
```

```mlir
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, j, k])->([i, j, k]) {i=2, j=1, k=4}>
```

最简形态：每个因子在操作数与结果里**位置相同**，没有任何特殊标注。
（`j=1` 也不特殊 —— 大小为 1 的维度照常参与。）

---

## 二、广播：结果**多出**因子

```mlir
func.func @broadcast_in_dim(%arg0: tensor<2x13x1xf32>) -> tensor<2x64x13x1xf32> {
```

```mlir
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 3] : (tensor<2x13x1xf32>) -> tensor<2x64x13x1xf32>
  return %0 :  tensor<2x64x13x1xf32>
}
```

```mlir
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, k, l])->([i, j, k, l]) {i=2, j=64, k=13, l=1}>
```

**读法**：输入只有 `(i, k, l)`，结果多了一个 `j=64`。
`j` 是**广播出来的维度**，在输入里**不存在** —— 所以输入侧的分片规则里没有它。

---

## 三、★ `reduction` 标注：收缩 / 归约因子

### `dot_general`

```mlir
func.func @dot_general_no_batching_dims(%arg0: tensor<8x32xf32>, %arg1: tensor<32x16xf32>) -> tensor<8x16xf32> {
```

```mlir
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
```

```mlir
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32} reduction={k}>
```

**读法**：`k` 是**收缩因子**（两个操作数都含它，结果里没有）→ 标 `reduction={k}`。
这告诉传播：**沿 k 切会产生部分和**（需要 all-reduce）。

### `reduce`

```mlir
func.func @reduce_single_result(%arg0: tensor<2x64x13xf32>) -> tensor<2x13xf32> {
```

```mlir
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [])->([i, k]) {i=2, j=64, k=13} reduction={j}>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  return %1 : tensor<2x13xf32>
}
```

```mlir
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [])->([i, k]) {i=2, j=64, k=13} reduction={j}>
```

**读法**：`j=64` 被 reduce 掉了 → `reduction={j}`。初始化常量是 rank 0，所以是 `[]`。

---

## 四、`permutation` 标注：尺寸会变的因子

### 卷积：`reduction` + `permutation` 同时出现

```mlir
func.func @conv_simple(%arg0 : tensor<2x224x224x192xf32>, %arg1 : tensor<3x3x192x64xf32>) -> tensor<2x112x112x64xf32> {
```

```mlir
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, jk, lm, n], [k, m, n, o])->([i, j, l, o]) {i=2, j=112, k=2, l=112, m=2, n=192, o=64} reduction={k, m, n} permutation={j, l}>
```

**读法**（这是最复杂的一条，值得逐段拆）：
- 输入用**复合因子** `jk` 与 `lm` —— 空间维 `224` 拆成"输出 112 × 窗口 2"。
- `reduction={k, m, n}`：窗口维 `k`、`m` 与输入通道 `n` 都是**归约**因子。
- `permutation={j, l}`：`j`/`l` 是 112，与输入侧的 224 **大小不同**（步长 2），
  所以标 `permutation` —— 它表示"这个因子在输出侧的尺寸与输入侧不成比例"。

### 转置：**不需要** `permutation`

```mlir
func.func @transpose(%arg0: tensor<256x32x64x100xf32>) -> tensor<100x32x256x64xf32> {
```

```mlir
  %0 = stablehlo.transpose %arg0, dims = [3, 1, 0, 2] : (tensor<256x32x64x100xf32>) -> tensor<100x32x256x64xf32>
  return %0 : tensor<100x32x256x64xf32>
}
```

```mlir
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([k, j, l, i])->([i, j, k, l]) {i=100, j=32, k=256, l=64}>
```

**读法**：转置只是**重排维度**，因子本身大小不变。
所以规则里输入侧写成 `([k, j, l, i])`（按输入的实际维序），**无需**任何标注。

### `pad`：**需要** `permutation`

```mlir
func.func @pad(%arg0: tensor<28x28x16xf32>, %arg1: tensor<f32>) -> tensor<30x26x16xf32> {
```

```mlir
  %0 = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0] : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  return %0 : tensor<30x26x16xf32>
}
```

```mlir
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [])->([i, j, k]) {i=28, j=28, k=16} permutation={i, j}>
```

**读法**：`pad` 改变了维度大小（28 → 30、28 → 26），且不是整数倍关系
→ `i`、`j` 标 `permutation`。`k` 没变，不标。

> **`permutation` 的判据**：因子的尺寸在操作数侧与结果侧**不成整数倍**。

---

## 五、复合因子：合并维度

```mlir
func.func @reshape_merge_dim(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
```

```mlir
  %0 = stablehlo.reshape %arg0 : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
```

```mlir
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([ij]) {i=2, j=4}>
```

**读法**：`2x4 → 8`，结果侧写成**复合因子** `[ij]`。
这正是 L2-01 讲的"reshape 沿因子传播"的规则来源。

---

## 六、`gather`：四类标注**同时出现**

```mlir
func.func @gather(%arg0: tensor<3x4x2x5xf32>, %arg1: tensor<2x3x3xi64>) -> tensor<2x3x2x2x1xf32> {
```

```mlir
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([o, k, l, m], [i, j, p])->([i, j, k, l, n]) {i=2, j=3, k=4, l=2, m=5, n=1, o=3, p=3} reduction={m, o} need_replication={k, n, p} blocked_propagation={k}>
```

**读法**（本课最复杂的一条）：
- `reduction={m, o}`：`o` 是被 `collapsed_slice_dims` 折叠掉的维，`m` 是切片维。
- `need_replication={k, n, p}`：这几个因子要求**复制**才能正确分片
  （`p` 是 index 维，`k`/`n` 与 slice 语义相关）。
- `blocked_propagation={k}`：`k` 上**不做传播** —— 这是 L1-05 讲的
  **正交标注**（与四类因子分类并列，不是第五类）。

---

## 七、`custom_call`：内置注册表 vs 用户自定义

### 内置注册（无需用户标注）

```mlir
func.func @custom_call_compact_wy_helper(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
```

```mlir
  %0 = stablehlo.custom_call @CompactWyHelper(%arg0) : (tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}
```

```mlir
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=128, j=128}>
```

注册表里为**已知的 custom_call**（`CompactWyHelper`、`X64Combine`、`TopK`、`PartialReduce`、
`Eigh`、`Qr`、`ApproxTopK` 等）内置了规则，用户不必自己写。

### 用户自定义：带 `custom` 标记

```mlir
func.func @unregisterd_custom_call_with_existing_rule(%arg0: tensor<4x2xf32>) -> tensor<2x4xf32> {
```

```mlir
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([j, i]) {i=4, j=2}, custom>} : (tensor<4x2xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
```

```mlir
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([j, i]) {i=4, j=2}, custom>
```

**读法**：`@foo` 未注册，但用户**自己写了规则**。注意末尾的 **`, custom`** 标记 ——
它表示"这条规则来自用户，不是注册表推导的"。转置语义 `([i,j])->([j,i])` 也被正确表达。

---

## 八、用例族谱（125 个）

| 族 | 代表用例 | 数量 |
|---|---|---|
| 逐元素 / 标量 | `pointwise_op`、`scalar_op`、`select`、`clamp`、`sort` | ~12 |
| 广播 | `broadcast_in_dim`（6 个变体） | 6 |
| 点积 | `dot_vector_vector`、`dot_matrix_matrix`、`dot_general_*` | 7 |
| 规约 | `reduce_*`（4）、`reduce_window`（3）、`reduce_scatter` | 8 |
| 卷积 | `conv_*`（7 个变体：窗口/分组/特征） | 7 |
| 形状变换 | `reshape_*`（16）、`slice`、`dynamic_slice`、`reverse` | ~22 |
| 索引类 | `gather_*`（5）、`scatter_*`（6）、`select_and_scatter` | 12 |
| 三角/分解 | `cholesky`、`triangular_solve_*`（4）、`fft_*`（8） | ~14 |
| `custom_call` | `custom_call_*`（22 个内置）+ 未注册用例 | 23 |
| 其它 | `concat_*`（10）、`pad`、`all_gather`、`all_to_all`、`batch_norm`、`rng_bit_generator`、`bitcast_convert_*` | ~14 |
