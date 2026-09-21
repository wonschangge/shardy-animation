<!-- sdy-coverage
ir/test/tensor_sharding_parse_print.mlir
ir/test/tensor_sharding_parsing_failure.mlir
-->

# L1-02 · tensor-sharding-syntax — 源 IR

本课覆盖两个测试文件：

| 文件 | 行数 | 作用 |
|---|---|---|
| `ir/test/tensor_sharding_parse_print.mlir` | 235 | 合法写法：两种属性形式、开闭维、replicated/unreduced、内联 mesh、子轴、优先级、特殊类型 |
| `ir/test/tensor_sharding_parsing_failure.mlir` | 177 | 非法写法：15 类解析错误 |

以下每条均**逐字**摘自上述文件。

---

## 一、两种属性形式

### 1. 挂在算子结果上：`sdy.sharding_per_value`

一个列表，每个元素对应一个 operand/result。空列表 `[]` 表示"没有任何值的分片信息"：

```mlir
return {sdy.sharding = #sdy.sharding_per_value<[]>} %arg0 : tensor<8x8xf32>
```

单值形式：

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {}]>]>} : tensor<8x8xf32>
```

多值形式（`optimization_barrier` 有两个结果）：

```mlir
  %0:2 = stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=2, "b"=4]>, [{"a"}, {}]>, <@foo, [{"a"}, {}]>]>} %arg0, %arg1 : tensor<8x8xf32>, tensor<8x8xf32>
```

### 2. 挂在函数参数/结果上：`sdy.sharding`

```mlir
func.func @sharding_with_priority(%arg0 : tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@foo, [{"a"}p0, {"b"}]>}, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
```

### 3. rank-0 张量：维分片列表为空，但仍可显式复制

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [], replicated={"b"}>]>} : tensor<f32>
```

---

## 二、复制轴与未归约轴

### 1. 无复制轴（默认空）

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {}]>]>} : tensor<8x8xf32>
```

### 2. 显式复制

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], replicated={"a", "b"}>]>} : tensor<8x8xf32>
```

### 3. 未归约轴（默认 sum）

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], unreduced={"a", "b"}>]>} : tensor<8x8xf32>
```

### 4. 未归约轴 + max 归约

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], unreduced=max{"a"}>]>} : tensor<8x8xf32>
```

### 5. 复制轴与未归约轴共存

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], replicated={"a"}, unreduced={"b"}>]>} : tensor<8x8xf32>
```

### 6. 打印顺序规范化：无论输入怎么写，输出都是 replicated 在前

输入写成 `unreduced` 在前：

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], unreduced={"b"}, replicated={"a"}>]>} : tensor<8x8xf32>
```

期望打印（CHECK 行）：

```mlir
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@foo, [{}, {}], replicated={"a"}, unreduced={"b"}>]>
```

---

## 三、内联网格 vs 符号引用

### 1. 内联网格

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<mesh<["x"=2, "y"=2]>, [{"x"}, {}]>]>} : tensor<8x8xf32>
```

### 2. 内联网格非 iota 设备顺序（保留）

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<mesh<["x"=2], device_ids=[1, 0]>, [{"x"}, {}]>]>} : tensor<8x8xf32>
```

### 3. 内联网格 iota 设备顺序（打印时省略）

输入：

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<mesh<["x"=2], device_ids=[0, 1]>, [{"x"}, {}]>]>} : tensor<8x8xf32>
```

期望打印：

```mlir
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<mesh<["x"=2]>, [{"x"}, {}]>]>
```

---

## 四、开维与闭维

### 1. 两维都开

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a", ?}, {?}]>]>} : tensor<8x8xf32>
```

### 2. 两维都闭

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {}]>]>} : tensor<8x8xf32>
```

### 3. 一开一闭

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a", ?}, {}]>]>} : tensor<8x8xf32>
```

### 4. 开维上有多个轴

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a", "b", ?}, {}]>]>} : tensor<16x8xf32>
```

---

## 五、子轴

### 1. 维度分片中使用子轴

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {"b":(2)2}]>]>} : tensor<8x8xf32>
```

### 2. 显式复制子轴

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {}], replicated={"b":(2)2}>]>} : tensor<8x8xf32>
```

### 3. 未归约子轴

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {}], unreduced={"b":(2)2}>]>} : tensor<8x8xf32>
```

### 4. 完整轴夹在两个子轴之间（合法，因为不连续）

网格为 `@bar = <["a"=4, "b"=2]>`：

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@bar, [{"a":(1)2, "b", "a":(2)2}, {}]>]>} : tensor<8x8xf32>
```

### 5. 逆序的连续子轴（解析合法，后续校验会报错）

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@bar, [{"a":(2)2, "a":(1)2}, {}]>]>} : tensor<8x8xf32>
```

---

## 六、优先级

### 1. 函数参数上的优先级

```mlir
func.func @sharding_with_priority(%arg0 : tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@foo, [{"a"}p0, {"b"}]>}, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
```

### 2. 算子结果上的优先级

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {"b"}p1]>]>} : tensor<8x8xf32>
```

### 3. 多个维度各有优先级

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}p0, {"b"}p1]>]>} : tensor<8x8xf32>
```

---

## 七、特殊类型

### 1. 动态形状

```mlir
func.func @dynamic_shaped_tensor_with_sharding(%arg0: tensor<?x?xf32> {sdy.sharding = #sdy.sharding<@foo, [{}, {"a"}]>}) -> tensor<?x?xf32> {
```

### 2. tuple 类型

```mlir
  %0 = stablehlo.custom_call @sdy_testonly(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {}]>]>} : (tensor<8x8xf32>) -> tuple<tensor<8x8xf32>>
```

### 3. 无结果的算子：可对"每个操作数"给分片，也可整体复制

maximal-sharding 网格（单设备）：

```mlir
  stablehlo.custom_call @foo(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh, []>]>} : (tensor<8x8xf32>) -> ()
```

普通网格上的全复制：

```mlir
  stablehlo.custom_call @foo(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@foo, []>]>} : (tensor<8x8xf32>) -> ()
```

---

## 八、15 类解析错误

### 1. 优先级不是数字

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {"b"}phigh]>]>} : tensor<8x8xf32>
```

### 2. 优先级整数溢出

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {"b"}p9999999999999999999]>]>} : tensor<8x8xf32>
```

### 3. 优先级有前导零

```mlir
func.func @priority_with_leading_zeros(%arg0 : tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@foo, [{"a"}p01, {"b"}]>}, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
```

### 4. `replicated` 后漏了 `=`

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], replicated{"a", "b"}>]>} : tensor<8x8xf32>
```

### 5. `unreduced` 用了方括号

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], unreduced=["a", "b"]>]>} : tensor<8x8xf32>
```

### 6. `unreduced` 集合里混入了 `[]`

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], unreduced={"a", "b"[]}>]>} : tensor<8x8xf32>
```

### 7. 轴名没加引号

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], replicated={"a", b}>]>} : tensor<8x8xf32>
```

### 8. `unreduced` 里轴名没加引号

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], unreduced={"a", b}>]>} : tensor<8x8xf32>
```

### 9. 两个关键字连写

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], replicated unreduced={"a", "b"}>]>} : tensor<8x8xf32>
```

### 10. 逗号后没有内容

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}],>]>} : tensor<8x8xf32>
```

### 11. 未知字段名

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], unknown={"a"}>]>} : tensor<8x8xf32>
```

### 12. 逗号后直接结束

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], replicated={"a"}, >]>} : tensor<8x8xf32>
```

### 13. replicated 与 unreduced 后都多逗号

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], replicated={"a"}, unreduced={"b"}, >]>} : tensor<8x8xf32>
```

### 14. 重复写 replicated

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], replicated={"a"}, replicated={"b"}>]>} : tensor<8x8xf32>
```

### 15. 重复写 unreduced

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], replicated={"a"}, unreduced={"b"}, unreduced={"b"}>]>} : tensor<8x8xf32>
```

---

## 附：本课用到的网格声明

```mlir
sdy.mesh @foo = <["a"=2, "c"=2, "b"=4, "d"=2]>
```

```mlir
sdy.mesh @bar = <["a"=4, "b"=2]>
```

```mlir
sdy.mesh @maximal_mesh = <[], device_ids=[0]>
```
