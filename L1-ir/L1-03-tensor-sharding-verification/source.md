<!-- sdy-coverage
ir/test/tensor_sharding_verification.mlir
-->

# L1-03 · tensor-sharding-verification — 源 IR

本课覆盖 **L1 最大的单个测试文件**：`ir/test/tensor_sharding_verification.mlir`（556 行）。

它用 `-verify-diagnostics` 逐条验证分片属性的**语义不变量**。
与 L1-02 的解析错误不同，这里的写法**语法完全正确**，但语义非法。

```mlir
// RUN: sdy_opt %s -split-input-file -verify-diagnostics
```

下面按不变量分类，每条均**逐字**摘自该文件。

---

## 一、网格与轴引用必须存在

### 1. 轴不在网格里（算子结果）

网格是 `@mesh = <["a"=2]>`，却引用了 `"c"`：

```mlir
%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"c"}]>]>} : tensor<8x8xf32>
```

### 2. 轴不在网格里（函数参数）

```mlir
func.func @func_arg_failure(%arg0: tensor<8x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"c"}]>},
```

### 3. 轴不在网格里（函数结果）

```mlir
func.func @func_result_failure(%arg0: tensor<8x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"a"}]>},
```

### 4. 网格符号不存在

```mlir
%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@other_mesh, [{}, {"a"}]>]>} : tensor<8x8xf32>
```

### 5. 网格符号不存在（函数参数）

```mlir
func.func @func_arg_unknown_mesh(%arg0: tensor<8x8xf32> {sdy.sharding=#sdy.sharding<@other_mesh, [{}, {"a"}]>},
```

---

## 二、每个轴最多出现一次

### 1. 维度分片与复制轴重复

`"a"` 既切了第 1 维，又出现在 `replicated` 里：

```mlir
%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a"}], replicated={"a"}>]>} : tensor<8x8xf32>
```

### 2. 子轴也是同一个轴，同样不能重复

```mlir
%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(2)2}], replicated={"a":(2)2}>]>} : tensor<8x8xf32>
```

### 3. 同一轴不能既用完整轴又用子轴

```mlir
%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a"}], replicated={"a":(2)2}>]>} : tensor<8x8xf32>
```

### 4. 子轴不能重叠

网格为 `@mesh = <["a"=8, "b"=2]>`，`"a":(2)4` 覆盖下标 2..5，`"a":(1)4` 覆盖 0..3，二者重叠：

```mlir
%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"a":(2)4}, {"b":(2)2}], replicated={"a":(1)4}>]>} : tensor<8x8xf32>
```

---

## 三、子轴参数必须合法

网格为 `@mesh = <["a"=8, "b"=2]>`（除第 3 条例外）。

### 1. pre-size 必须 ≥ 1

```mlir
%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(-1)2}]>]>} : tensor<8x8xf32>
```

### 2. 子轴大小必须 > 1

```mlir
%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(2)1}]>]>} : tensor<8x8xf32>
```

### 3. 子轴大小不能等于整轴大小

```mlir
%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(1)8}]>]>} : tensor<8x8xf32>
```

### 4. 下一个 pre-size 必须整除整轴大小（字段宽度写不下）

网格是 `@mesh = <["a"=2,"b"=4]>`，子轴 `"a":(3)2` 的 next pre-size 为 6，不整除 8？此处实际报错针对的是 `@mesh = <["a"=8, "b"=2]>` 的 `"a":(3)2`：

```mlir
%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(3)2}]>]>} : tensor<6x6xf32>
```

### 5. next pre-size 超出整轴范围

`"a":(4)4` 的 next pre-size 为 16，大于整轴大小 8：

```mlir
%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(4)4}]>]>} : tensor<8x8xf32>
```

### 6. 相邻子轴必须尽可能大（不可合并）

`"a":(2)2` 与 `"a":(4)4` 相邻，可合并为 `"a":(2)8`… 实际应写成更大的子轴：

```mlir
%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(2)2, "a":(4)4}]>]>} : tensor<8x8xf32>
```

---

## 四、复制轴与未归约轴必须按网格顺序排列

网格为 `@mesh = <["c"=2, "a"=2, "b"=2]>`（注意声明顺序是 c、a、b）。

### 1. 复制轴顺序错

```mlir
%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {}], replicated={"a", "b", "c"}>]>} : tensor<8x8xf32>
```

### 2. 未归约轴顺序错

```mlir
%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"a", "b", "c"}>]>} : tensor<8x8xf32>
```

---

## 五、维分片必须与张量 rank 一致

### 1. 给了 2 个维分片，张量是 rank 1

```mlir
%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"b"}], replicated={"a"}>]>} : tensor<8xf32>
```

### 2. 给了 1 个维分片，张量是 rank 2

```mlir
func.func @dynamic_shaped_tensor_rank_mismatch(%arg0: tensor<?x?xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> tensor<?x?xf32> {
```

---

## 六、空的闭维不能带优先级

空的闭维 `{}` 表示"这一维不切"，给它优先级没有意义。

```mlir
%0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}p3]>]>} : tensor<8x8xf32>
```

```mlir
func.func @dynamic_shaped_tensor_empty_closed_priority(%arg0: tensor<?x?xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}p1, {}]>}) -> tensor<?x?xf32> {
```

---

## 七、非张量 / 非 rank / 非 shaped 的限制

### 1. token 不能有非零 rank

```mlir
func.func @token_sharding_rank_non_zero(%arg0: !stablehlo.token {sdy.sharding=#sdy.sharding<@mesh, [{}]>}) -> !stablehlo.token {
```

### 2. token 不能有复制轴

```mlir
func.func @token_sharding_with_replicated_axes(%arg0: !stablehlo.token {sdy.sharding=#sdy.sharding<@mesh, [], replicated={"a"}>}) -> !stablehlo.token {
```

### 3. token 不能有未归约轴

```mlir
func.func @token_sharding_with_unreduced_axes(%arg0: !stablehlo.token {sdy.sharding=#sdy.sharding<@mesh, [], unreduced={"a"}>}) -> !stablehlo.token {
```

### 4. 无 rank 张量不能有分片

```mlir
func.func @unranked_tensor_with_sharding(%arg0: tensor<*xf32> {sdy.sharding=#sdy.sharding<@mesh, []>}) -> tensor<*xf32> {
```

---

## 八、属性类型与数量必须匹配

### 1. 算子上必须用 `sharding_per_value`，不能用 `sharding`

```mlir
%0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding<@mesh, [{}, {"a"}]>} : tensor<8x8xf32>
```

### 2. 函数参数上必须用 `sharding`，不能用 `sharding_per_value`

```mlir
func.func @func_arg_with_tensor_sharding_per_value_attr(
```

### 3. 分片数量必须等于值的数量

3 个分片 vs 2 个结果：

```mlir
%1:2 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %0) across dimensions = [1]
```

### 4. tuple 只支持大小为 1

```mlir
%0 = stablehlo.custom_call @sdy_testonly(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x8xf32>) -> tuple<tensor<8x8xf32>, tensor<8x8xf32>>
```

---

## 九、maximal-sharding 网格只能配 rank 0 且无复制/未归约

网格是 `@maximal_mesh = <[], device_ids=[0]>`。

### 1. 非 rank 0

```mlir
%0 = stablehlo.custom_call @sdy_testonly(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh, [{}, {}]>]>} : (tensor<8x8xf32>) -> tuple<tensor<8x8xf32>>
```

### 2. 非空维分片

```mlir
stablehlo.custom_call @foo(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh, [{}]>]>} : (tensor<8x8xf32>) -> ()
```

---

## 十、上下文约束：轴已被父级 manual_computation 绑定

`manual_axes={"a"}` 已经把轴 `"a"` 绑定在手动计算上，区域内不能再拿它切维度：

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a",?}, {?}]>] out_shardings=[<@mesh, [{"a",?}, {?}]>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) {
    %0 = stablehlo.add %arg1, %arg1 {sdy.sharding=#sdy.sharding_per_value<[ <@mesh, [{"a"}, {}]>]>} : tensor<8x32xf32>
    sdy.return %0 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
```

---

## 附：本课用到的网格声明

```mlir
sdy.mesh @mesh = <["a"=2]>
```

```mlir
sdy.mesh @mesh = <["a"=2, "b"=2]>
```

```mlir
sdy.mesh @mesh = <["c"=2, "a"=2, "b"=2]>
```

```mlir
sdy.mesh @mesh = <["a"=8, "b"=2]>
```

```mlir
sdy.mesh @maximal_mesh = <[], device_ids=[0]>
```
