<!-- sdy-coverage
transforms/propagation/test/basic_propagation_keep_sharding_rules.mlir
-->

# L2-03 · keep-sharding-rules — 源 IR

覆盖：`shardy/dialect/sdy/transforms/propagation/test/basic_propagation_keep_sharding_rules.mlir`（17 行 / 1 个用例）。

```mlir
// RUN: sdy_opt %s -sdy-basic-propagate='keep-sharding-rules=true' 2>&1 | FileCheck %s
```

同样是**同一个 pass**，只多了一个选项 `keep-sharding-rules=true`。

---

## 唯一的用例：已有规则与新建规则都保留

网格 `@mesh = <["a"=2, "b"=2]>`：

```mlir
sdy.mesh @mesh = <["a"=2, "b"=2]>
```

```mlir
func.func @existing_and_created_rules_remain(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
                  %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.add %0, %0 {sdy.sharding_rule = #sdy.op_sharding_rule<([ij, k], [ij, k])->([ij, k]) {i=4, j=2, k=8}>} : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}
```

期望（CHECK 行）：

```mlir
// CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1
// CHECK-SAME:     {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>,
// CHECK-SAME:      sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=8}>}
```

```mlir
// CHECK-NEXT: stablehlo.add %[[ADD]], %[[ADD]]
// CHECK-SAME:     {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>,
// CHECK-SAME:      sdy.sharding_rule = #sdy.op_sharding_rule<([ij, k], [ij, k])->([ij, k]) {i=4, j=2, k=8}>}
```

---

## 两个值得注意的点

### 1. 规则分两类，都保留

- **第一个 `add`** 原本**没有**规则 —— 传播为它**推导**出一条
  `([i, j], [i, j])->([i, j]) {i=8, j=8}`（逐元素算子：维度一一对应）。
- **第二个 `add`** 原本**已有**规则 `([ij, k], [ij, k])->([ij, k]) {i=4, j=2, k=8}`
  —— 它被**原样保留**（注意这里有复合因子 `ij`，是用户自定义规则的典型形态）。

默认情况下，传播用完规则就会把它们删掉（避免污染 IR）。加上
`keep-sharding-rules=true` 后，**两类都留在 IR 里**。

### 2. 为什么需要这个选项

规则是传播的**输入依据**。当传播结果不符合预期时，第一个要回答的问题是：

> 传播当时用的是哪条规则？

把规则留下来，就能直接看到"它以为这个算子该怎么切"，从而判断是规则错了、
还是传播策略需要调整。这是 L2-12「传播调试」的基础工具之一。

> 注意 `%[[ADD:.*]]` 是 **FileCheck 变量**（捕获一个 SSA 名并复用），不是字面 IR。
